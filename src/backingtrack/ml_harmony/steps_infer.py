from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..harmony_baseline import ChordEvent
from ..key_detect import estimate_key
from ..types import BarGrid, Note
from .model import ChordModelConfig, ChordTransformer

# -----------------------------------------------------------------------------
# Chord vocabulary
# -----------------------------------------------------------------------------
# IMPORTANT: This ordering MUST match the ordering used when you built
# data/ml/chords_steps.npz.
#
# In the POP909 preprocessor I provided, the vocab is:
#   0 = N (no chord)
#   1..12   = maj roots C..B
#   13..24  = min roots C..B
#   25..36  = 7   (dominant) roots C..B
#   37..48  = maj7 roots C..B
#   49..60  = min7 roots C..B
#   61..72  = dim roots C..B
#   73..84  = sus2 roots C..B
#   85..96  = sus4 roots C..B

QUALITIES: Tuple[str, ...] = (
    "maj",
    "min",
    "7",
    "maj7",
    "min7",
    "dim",
    "sus2",
    "sus4",
)

IGNORE_INDEX = -100


@dataclass(frozen=True)
class ChordSampleConfig:
    """Controls ML chord sampling."""

    step_beats: float = 2.0
    include_key: bool = True

    # Sampling
    stochastic: bool = False
    temperature: float = 1.0
    top_k: int = 12
    repeat_penalty: float = 1.2

    # If stochastic=False, you can optionally smooth with a small change penalty
    # (larger => fewer chord changes).
    change_penalty: float = 0.15

    # How to handle class 0 (N / no-chord)
    fill_no_chord: str = "hold"  # "hold" | "drop"

    seed: Optional[int] = None


def load_chord_model(
    model_path: str | Path,
    device: Optional[str] = None,
) -> Tuple[ChordTransformer, ChordModelConfig, torch.device]:
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Chord model not found: {p}")

    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(p), map_location=dev)

    cfg = ChordModelConfig(**ckpt["cfg"])
    model = ChordTransformer(cfg).to(dev)
    model.load_state_dict(ckpt["state"], strict=True)
    model.eval()
    return model, cfg, dev


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _key_features(melody_notes: Sequence[Note]) -> np.ndarray:
    k = estimate_key(melody_notes)
    tonic = int(k.tonic_pc) % 12
    tonic_oh = np.zeros(12, dtype=np.float32)
    tonic_oh[tonic] = 1.0
    mode_oh = np.zeros(2, dtype=np.float32)
    mode_oh[0 if k.mode == "major" else 1] = 1.0
    return np.concatenate([tonic_oh, mode_oh], axis=0)


def _extract_step_features(
    melody: Sequence[Note],
    step_start: float,
    step_end: float,
    step_len: float,
    key_feat: Optional[np.ndarray],
) -> np.ndarray:
    """Matches the feature format used in preprocessing."""

    hist = np.zeros(12, dtype=np.float32)
    tot = 0.0
    pitch_num = 0.0

    for n in melody:
        ov = _overlap(float(n.start), float(n.end), step_start, step_end)
        if ov <= 0:
            continue
        tot += ov
        hist[int(n.pitch) % 12] += float(ov)
        pitch_num += float(ov) * float(n.pitch)

    if hist.sum() > 1e-9:
        hist = hist / (hist.sum() + 1e-9)

    mean_pitch = (pitch_num / max(1e-9, tot)) if tot > 0 else 60.0
    mean_pitch_norm = float(np.clip(mean_pitch / 127.0, 0.0, 1.0))
    activity = float(np.clip(tot / max(1e-9, step_len), 0.0, 1.0))

    parts = [
        hist,
        np.array([mean_pitch_norm, activity], dtype=np.float32),
    ]
    if key_feat is not None:
        parts.append(key_feat.astype(np.float32))

    return np.concatenate(parts, axis=0).astype(np.float32)


def _id_to_event(cid: int, *, start: float, end: float, grid: BarGrid) -> Optional[ChordEvent]:
    cid = int(cid)
    if cid == 0:
        return None

    cid0 = cid - 1
    q_idx = cid0 // 12
    root_pc = cid0 % 12
    q = QUALITIES[q_idx]

    # Convert our quality token to (quality, extensions)
    if q == "maj":
        quality, exts = "maj", ()
    elif q == "min":
        quality, exts = "min", ()
    elif q == "7":
        quality, exts = "maj", (10,)  # dominant 7
    elif q == "maj7":
        quality, exts = "maj", (11,)
    elif q == "min7":
        quality, exts = "min", (10,)
    elif q == "dim":
        quality, exts = "dim", ()
    elif q == "sus2":
        quality, exts = "sus2", ()
    elif q == "sus4":
        quality, exts = "sus4", ()
    else:
        # unknown token (shouldn't happen)
        quality, exts = "maj", ()

    bar_index = int(grid.bar_index_at(start))
    return ChordEvent(
        root_pc=int(root_pc),
        quality=str(quality),
        extensions=tuple(int(x) for x in exts),
        bar_index=bar_index,
        start=float(start),
        end=float(end),
    )


def _top_k_sample(logits: np.ndarray, k: int, rng: np.random.Generator) -> int:
    """Sample from logits with optional top-k truncation."""
    if k is not None and int(k) > 0 and int(k) < logits.shape[0]:
        idx = np.argpartition(logits, -int(k))[-int(k):]
        sub = logits[idx]
        sub = sub - np.max(sub)
        p = np.exp(sub)
        p = p / (p.sum() + 1e-9)
        return int(rng.choice(idx, p=p))

    logits = logits - np.max(logits)
    p = np.exp(logits)
    p = p / (p.sum() + 1e-9)
    return int(rng.choice(np.arange(logits.shape[0]), p=p))


def generate_chords_ml_steps(
    *,
    melody_notes: Sequence[Note],
    grid: BarGrid,
    duration_seconds: float,
    model_path: str | Path,
    cfg: Optional[ChordSampleConfig] = None,
    device: Optional[str] = None,
) -> List[ChordEvent]:
    """Generate a chord timeline from melody using a step-based Transformer model."""

    scfg = cfg or ChordSampleConfig()

    model, mcfg, dev = load_chord_model(model_path, device=device)

    if float(scfg.step_beats) <= 0:
        raise ValueError("step_beats must be > 0")

    step_len = float(scfg.step_beats) * float(grid.seconds_per_beat)
    duration_seconds = float(max(0.0, duration_seconds))
    n_steps = int(np.ceil(max(1e-6, duration_seconds) / step_len))

    key_feat = _key_features(melody_notes) if scfg.include_key else None

    feats: List[np.ndarray] = []
    for i in range(n_steps):
        t0 = float(grid.start_time) + i * step_len
        t1 = t0 + step_len
        feats.append(_extract_step_features(melody_notes, t0, t1, step_len, key_feat))

    X = np.stack(feats, axis=0).astype(np.float32)  # (T,F)

    # Run the model. If the song is longer than the model's max position length
    # (mcfg.max_bars), we chunk it. This avoids positional-embedding overflow.
    max_len = int(getattr(mcfg, "max_bars", X.shape[0]))
    if max_len <= 0:
        max_len = X.shape[0]

    logits_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, X.shape[0], max_len):
            e = min(X.shape[0], s + max_len)
            xb = torch.tensor(X[None, s:e, :], dtype=torch.float32, device=dev)
            lc = model(xb)  # (1,L,C)
            logits_chunks.append(lc.squeeze(0).detach().cpu().numpy().astype(np.float32))

    logits_np = np.concatenate(logits_chunks, axis=0)  # (T,C)

    # Sanity: match class count
    if int(logits_np.shape[1]) != int(mcfg.n_classes):
        raise ValueError(
            f"Model output classes={logits_np.shape[1]} but cfg.n_classes={mcfg.n_classes}. "
            "Did you pass the right model?"
        )

    # Temperature
    temp = float(scfg.temperature)
    if temp <= 1e-6:
        temp = 1.0

    rng = np.random.default_rng(None if scfg.seed is None else int(scfg.seed))

    # Decode per step
    ids: List[int] = []

    if scfg.stochastic:
        prev: Optional[int] = None
        for t in range(n_steps):
            l = logits_np[t] / temp
            if prev is not None and float(scfg.repeat_penalty) > 0:
                l = l.copy()
                l[int(prev)] -= float(scfg.repeat_penalty)
            cid = _top_k_sample(l, int(scfg.top_k), rng)
            ids.append(int(cid))
            prev = int(cid)
    else:
        # Deterministic: argmax + optional smoothing (simple change penalty DP)
        base = logits_np / temp
        if float(scfg.change_penalty) <= 0:
            ids = [int(np.argmax(base[t])) for t in range(n_steps)]
        else:
            C = base.shape[1]
            dp = np.full((n_steps, C), -1e18, dtype=np.float32)
            back = np.zeros((n_steps, C), dtype=np.int32)

            dp[0, :] = base[0, :]
            for t in range(1, n_steps):
                # best previous per current
                prev_best = dp[t - 1, :]
                for c in range(C):
                    trans = prev_best - float(scfg.change_penalty)
                    trans[c] = prev_best[c]  # no penalty for staying
                    j = int(np.argmax(trans))
                    dp[t, c] = base[t, c] + trans[j]
                    back[t, c] = j

            last = int(np.argmax(dp[-1, :]))
            path = [last]
            for t in range(n_steps - 1, 0, -1):
                last = int(back[t, last])
                path.append(last)
            ids = list(reversed(path))

    # Handle no-chord
    if scfg.fill_no_chord not in ("hold", "drop"):
        raise ValueError("fill_no_chord must be 'hold' or 'drop'")

    if scfg.fill_no_chord == "hold":
        # replace N with previous non-N (or next non-N, or maj tonic fallback)
        # tonic fallback: choose the most likely non-N chord from first step
        fallback = None
        for cid in ids:
            if cid != 0:
                fallback = cid
                break
        if fallback is None:
            # if model predicted all N, pick C:maj id=1
            fallback = 1

        prev = fallback
        for i in range(len(ids)):
            if ids[i] == 0:
                ids[i] = int(prev)
            else:
                prev = int(ids[i])

    # Merge into events
    events: List[ChordEvent] = []
    if not ids:
        return events

    song_start = float(grid.start_time)
    song_end = song_start + duration_seconds

    cur = ids[0]
    cur_start = song_start
    for i in range(1, n_steps + 1):
        boundary = song_start + i * step_len
        if boundary > song_end:
            boundary = song_end
        if i == n_steps or ids[i] != cur:
            if cur != 0 or scfg.fill_no_chord == "hold":
                ev = _id_to_event(cur, start=cur_start, end=boundary, grid=grid)
                if ev is not None:
                    events.append(ev)
            if i < n_steps:
                cur = ids[i]
                cur_start = boundary

    # Optionally drop N events (if fill_no_chord == drop)
    if scfg.fill_no_chord == "drop":
        events = [e for e in events if e is not None]

    return events
