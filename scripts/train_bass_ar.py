# scripts/train_bass_ar.py
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

IGNORE_INDEX = -100


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class BassARConfig:
    feat_dim: int
    vocab_size: int
    max_steps: int = 128

    # needed for decoding back to (degree, register, rhythm)
    n_degree: int = 7
    n_register: int = 3
    n_rhythm: int = 5

    d_model: int = 192
    n_heads: int = 4
    n_layers: int = 6
    dropout: float = 0.1


class BassARTransformer(nn.Module):
    """
    Autoregressive LM:
      input  = prev tokens (teacher forcing) + conditioning per-step features
      output = next token logits
    """
    def __init__(self, cfg: BassARConfig):
        super().__init__()
        self.cfg = cfg

        # +1 for START token
        self.tok_emb = nn.Embedding(cfg.vocab_size + 1, cfg.d_model)
        self.cond_proj = nn.Linear(cfg.feat_dim, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_steps, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.ln = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x_tok: torch.Tensor, x_cond: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x_tok: (B,T) in [0..vocab_size] where vocab_size is START token
        # x_cond: (B,T,F)
        B, T = x_tok.shape
        device = x_tok.device

        h = self.tok_emb(x_tok) + self.cond_proj(x_cond)
        idx = torch.arange(T, device=device)
        h = h + self.pos(idx)[None, :, :]

        causal = self._causal_mask(T, device=device)
        pad_mask = None
        if attn_mask is not None:
            pad_mask = ~attn_mask  # True where padding

        h = self.enc(h, mask=causal, src_key_padding_mask=pad_mask)
        h = self.ln(h)
        return self.lm_head(h)  # (B,T,vocab)


@torch.no_grad()
def masked_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    mask = (y != IGNORE_INDEX)
    denom = int(mask.sum().item())
    if denom == 0:
        return 0.0
    pred = logits.argmax(dim=-1)
    correct = (pred == y) & mask
    return float(correct.sum().item() / denom)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/ml/bass_steps.npz")
    ap.add_argument("--out", type=str, default="data/ml/bass_ar_model.pt")

    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.1)

    ap.add_argument("--d_model", type=int, default=192)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_steps", type=int, default=0, help="0 => use NPZ seq_len")

    args = ap.parse_args()
    set_seed(int(args.seed))

    d = np.load(args.data, allow_pickle=True)

    if "y_token" not in d.files:
        raise RuntimeError("NPZ missing y_token. Re-run preprocess_bass.py (updated version).")

    X = torch.tensor(d["X"], dtype=torch.float32)            # (N,T,F)
    y = torch.tensor(d["y_token"], dtype=torch.int64)        # (N,T)
    attn = torch.tensor(d["attn_mask"], dtype=torch.bool)    # (N,T)
    label = torch.tensor(d["label_mask"], dtype=torch.bool)  # (N,T)

    N, T, Fdim = X.shape

    n_degree = int(d["n_degree"][0]) if "n_degree" in d.files else 7
    n_register = int(d["n_register"][0]) if "n_register" in d.files else 3
    n_rhythm = int(d["n_rhythm"][0]) if "n_rhythm" in d.files else 5
    vocab_size = int(d["vocab_size"][0]) if "vocab_size" in d.files else (n_degree * n_register * n_rhythm)

    if int(args.max_steps) <= 0:
        args.max_steps = int(T)

    # Mask out unknown-chord steps for loss (and attention, to keep context clean)
    attn_eff = attn & label
    y = y.clone()
    y[~attn_eff] = IGNORE_INDEX

    START = vocab_size  # special token id used only for input embedding

    # Teacher forcing shift-right: x_tok[t] = y[t-1], x_tok[0] = START
    x_tok = torch.full_like(y, fill_value=START)
    x_tok[:, 1:] = y[:, :-1].clone()

    # If previous token was IGNORE, don't feed it; use START instead
    x_tok[(x_tok == IGNORE_INDEX)] = START
    x_tok[~attn_eff] = START  # padding/invalid positions

    # train/val split
    idx = torch.randperm(N)
    n_val = int(float(args.val_frac) * N)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    tr_ds = TensorDataset(X[tr_idx], x_tok[tr_idx], y[tr_idx], attn_eff[tr_idx])
    va_ds = TensorDataset(X[val_idx], x_tok[val_idx], y[val_idx], attn_eff[val_idx])

    tr_dl = DataLoader(tr_ds, batch_size=int(args.batch), shuffle=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=int(args.batch), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = BassARConfig(
        feat_dim=int(Fdim),
        vocab_size=int(vocab_size),
        max_steps=int(args.max_steps),
        n_degree=int(n_degree),
        n_register=int(n_register),
        n_rhythm=int(n_rhythm),
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        n_layers=int(args.n_layers),
        dropout=float(args.dropout),
    )
    model = BassARTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.wd))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[data] X={tuple(X.shape)} seq_len={T} feat_dim={Fdim}")
    print(f"[lm] vocab_size={vocab_size} (deg={n_degree} reg={n_register} rhy={n_rhythm}) START={START}")
    print(f"[model] d_model={cfg.d_model} heads={cfg.n_heads} layers={cfg.n_layers} max_steps={cfg.max_steps}")

    best_val = 1e18

    for epoch in range(1, int(args.epochs) + 1):
        # ---- train ----
        model.train()
        tr_loss = 0.0
        tr_acc = 0.0
        n_batches = 0

        for xb, xtb, yb, attb in tr_dl:
            xb = xb.to(device)
            xtb = xtb.to(device)
            yb = yb.to(device)
            attb = attb.to(device)

            logits = model(xtb, xb, attn_mask=attb)  # (B,T,V)

            loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size),
                yb.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_loss += float(loss.item())
            tr_acc += masked_accuracy(logits, yb)
            n_batches += 1

        tr_loss /= max(1, n_batches)
        tr_acc /= max(1, n_batches)

        # ---- val ----
        model.eval()
        va_loss = 0.0
        va_acc = 0.0
        n_batches = 0

        for xb, xtb, yb, attb in va_dl:
            xb = xb.to(device)
            xtb = xtb.to(device)
            yb = yb.to(device)
            attb = attb.to(device)

            logits = model(xtb, xb, attn_mask=attb)

            loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size),
                yb.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )

            va_loss += float(loss.item())
            va_acc += masked_accuracy(logits, yb)
            n_batches += 1

        va_loss /= max(1, n_batches)
        va_acc /= max(1, n_batches)

        print(f"epoch {epoch:02d}  train loss={tr_loss:.4f} acc={tr_acc:.3f} | val loss={va_loss:.4f} acc={va_acc:.3f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(
                {
                    "cfg": cfg.__dict__,
                    "state": model.state_dict(),
                    "meta": {
                        "step_beats": float(d["step_beats"][0]) if "step_beats" in d.files else 2.0,
                        "include_key": bool(d["include_key"][0]) if "include_key" in d.files else False,
                    },
                },
                out_path,
            )
            print(f"  saved best -> {out_path}")

    print("done.")


if __name__ == "__main__":
    main()
