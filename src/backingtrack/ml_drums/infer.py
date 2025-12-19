# src/backingtrack/ml_drums/infer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..types import BarGrid, Note, TimeSignature
from .data import VOICE_MAP, N_VOICES
from .model import DrumModelConfig, DrumTransformer


# Inverse map: voice_index -> GM drum pitch
INV_VOICE_MAP: Dict[int, int] = {v: k for k, v in VOICE_MAP.items()}

# Voice indices (None-safe)
KICK_V = VOICE_MAP.get(36)
SNARE_V = VOICE_MAP.get(38)
CLOSED_HH_V = VOICE_MAP.get(42)
OPEN_HH_V = VOICE_MAP.get(46)
RIDE_V = VOICE_MAP.get(51)
CRASH_V = VOICE_MAP.get(49)
LOW_TOM_V = VOICE_MAP.get(45)
MID_TOM_V = VOICE_MAP.get(47)
HIGH_TOM_V = VOICE_MAP.get(50)

HAT_VOICES = [v for v in (CLOSED_HH_V, OPEN_HH_V, RIDE_V) if v is not None]
TOM_VOICES = [v for v in (LOW_TOM_V, MID_TOM_V, HIGH_TOM_V) if v is not None]


@dataclass(frozen=True)
class SampleConfig:
    # --- Sampling ---
    bars: int = 16
    temperature: float = 1.0
    threshold: float = 0.45
    stochastic: bool = True
    seed: Optional[int] = None

    # --- Groove constraints / “make it musical” ---
    lock_hats: bool = True
    hat_step: int = 2             # 2 => 8ths, 1 => 16ths
    hat_voice: str = "closed"     # "closed" | "ride"
    hat_vel_onbeat: float = 0.55
    hat_vel_offbeat: float = 0.42

    force_backbeat: bool = True
    keep_ghost_snares: bool = True
    ghost_snare_steps: Tuple[int, int] = (3, 11)
    ghost_snare_max_vel: float = 0.28

    max_nonhat_hits_per_step: int = 2

    enable_ml_fills: bool = False
    fill_every_bars: int = 8

    allow_open_hat: bool = True
    open_hat_step: int = 14
    open_hat_prob: float = 0.30

    restrict_crashes: bool = True
    crash_every_bars: int = 8

    reuse_groove: bool = True
    groove_mutation: float = 0.12

    # --- Velocity shaping ---
    vel_floor: int = 18
    vel_ceiling: int = 120
    intensity: float = 0.75


def load_model(model_path: str | Path, device: Optional[str] = None) -> Tuple[DrumTransformer, DrumModelConfig, torch.device]:
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")

    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(p), map_location=dev)

    cfg = DrumModelConfig(**ckpt["cfg"])
    model = DrumTransformer(cfg).to(dev)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, cfg, dev


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _ensure_int_in_range(x: int, lo: int, hi: int) -> int:
    x = int(x)
    return lo if x < lo else hi if x > hi else x


def _apply_bar_constraints(
    hits: np.ndarray,
    vels: np.ndarray,
    *,
    bar_index: int,
    scfg: SampleConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Post-process one bar (16,V) with musical constraints so it sounds like an actual groove.
    """
    H = hits.copy()
    V = vels.copy()
    Vn = H.shape[1]

    # --- hats: lock to a clean pattern ---
    if scfg.lock_hats and HAT_VOICES:
        for hv in HAT_VOICES:
            H[:, hv] = 0.0
            V[:, hv] = 0.0

        hat_voice_idx = CLOSED_HH_V if scfg.hat_voice == "closed" else (RIDE_V if RIDE_V is not None else CLOSED_HH_V)
        if hat_voice_idx is None:
            hat_voice_idx = HAT_VOICES[0]

        step = _ensure_int_in_range(scfg.hat_step, 1, 16)
        for s in range(0, 16, step):
            H[s, hat_voice_idx] = 1.0
            onbeat = (s % 4) == 0
            V[s, hat_voice_idx] = float(scfg.hat_vel_onbeat if onbeat else scfg.hat_vel_offbeat)

        # optional open-hat accent at phrase ends
        if scfg.allow_open_hat and OPEN_HH_V is not None and step >= 2:
            phrase = max(1, int(scfg.fill_every_bars))
            is_phrase_end = ((bar_index + 1) % phrase) == 0
            if is_phrase_end and rng.random() < float(scfg.open_hat_prob):
                s = _ensure_int_in_range(scfg.open_hat_step, 0, 15)
                if H[s, hat_voice_idx] > 0.5:
                    H[s, hat_voice_idx] = 0.0
                    V[s, hat_voice_idx] = 0.0
                H[s, OPEN_HH_V] = 1.0
                V[s, OPEN_HH_V] = max(float(V[s, OPEN_HH_V]), 0.55)

    # --- backbeat + snare cleanup ---
    if scfg.force_backbeat and SNARE_V is not None:
        for s in (4, 12):
            H[s, SNARE_V] = 1.0
            V[s, SNARE_V] = max(float(V[s, SNARE_V]), 0.75)

        allowed = {4, 12}
        if scfg.keep_ghost_snares:
            allowed |= set(int(x) for x in scfg.ghost_snare_steps)

        for s in range(16):
            if s in allowed:
                if s not in (4, 12):
                    V[s, SNARE_V] = min(float(V[s, SNARE_V]), float(scfg.ghost_snare_max_vel))
                continue
            H[s, SNARE_V] = 0.0
            V[s, SNARE_V] = 0.0

        if scfg.keep_ghost_snares:
            for s in scfg.ghost_snare_steps:
                s = int(s)
                if 0 <= s <= 15 and rng.random() < 0.25:
                    H[s, SNARE_V] = 1.0
                    V[s, SNARE_V] = max(float(V[s, SNARE_V]), 0.14)

    # --- crashes: only bar starts every N bars ---
    if scfg.restrict_crashes and CRASH_V is not None:
        H[:, CRASH_V] = 0.0
        V[:, CRASH_V] = 0.0
        every = max(1, int(scfg.crash_every_bars))
        if (bar_index % every) == 0:
            H[0, CRASH_V] = 1.0
            V[0, CRASH_V] = 0.75

    # --- toms: remove from grooves (fills injected later) ---
    if TOM_VOICES and not scfg.enable_ml_fills:
        for tv in TOM_VOICES:
            H[:, tv] = 0.0
            V[:, tv] = 0.0

    # --- stack limiter (excluding hats) ---
    nonhat = [i for i in range(Vn) if i not in HAT_VOICES]
    max_hits = max(1, int(scfg.max_nonhat_hits_per_step))

    for s in range(16):
        on = [i for i in nonhat if H[s, i] > 0.5]
        if len(on) <= max_hits:
            continue

        keep: List[int] = []
        if SNARE_V is not None and H[s, SNARE_V] > 0.5:
            keep.append(SNARE_V)
        if KICK_V is not None and H[s, KICK_V] > 0.5 and KICK_V not in keep and len(keep) < max_hits:
            keep.append(KICK_V)

        remaining = [i for i in on if i not in keep]
        remaining.sort(key=lambda i: float(V[s, i]), reverse=True)
        for i in remaining:
            if len(keep) >= max_hits:
                break
            keep.append(i)

        for i in on:
            if i not in keep:
                H[s, i] = 0.0
                V[s, i] = 0.0

    return H.astype(np.float32), _clamp01(V.astype(np.float32))


def _mutate_groove_bar(
    base_hits: np.ndarray,
    base_vels: np.ndarray,
    *,
    bar_index: int,
    scfg: SampleConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Small drummer-like variations: mostly DROPS + occasional kick syncopation.
    """
    H = base_hits.copy()
    V = base_vels.copy()

    rate = float(scfg.groove_mutation)
    if rate <= 0:
        return H, V

    protected: set[tuple[int, int]] = set()
    if SNARE_V is not None:
        protected.add((4, SNARE_V))
        protected.add((12, SNARE_V))
    if KICK_V is not None:
        protected.add((0, KICK_V))

    for s in range(16):
        for v in range(H.shape[1]):
            if (s, v) in protected:
                continue
            if rng.random() > rate:
                continue

            if H[s, v] > 0.5:
                H[s, v] = 0.0
                V[s, v] = 0.0
            else:
                if KICK_V is not None and v == KICK_V and s in (6, 7, 14) and rng.random() < 0.35:
                    H[s, v] = 1.0
                    V[s, v] = max(float(V[s, v]), 0.55)

    hit_mask = (H > 0.5).astype(np.float32)
    V = V + (rng.normal(0.0, 0.04, size=V.shape).astype(np.float32) * hit_mask)
    V = _clamp01(V)

    return _apply_bar_constraints(H, V, bar_index=bar_index, scfg=scfg, rng=rng)


def sample_one_bar(
    model: DrumTransformer,
    cfg: DrumModelConfig,
    device: torch.device,
    scfg: SampleConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(scfg.seed)

    Vn = cfg.n_voices
    assert Vn == N_VOICES, f"Model voices={Vn}, but code expects N_VOICES={N_VOICES}. Check VOICE_MAP."

    hits = np.zeros((cfg.steps, Vn), dtype=np.float32)
    vels = np.zeros((cfg.steps, Vn), dtype=np.float32)

    for t in range(cfg.steps):
        x_hits = np.zeros((cfg.steps, Vn), dtype=np.float32)
        x_vels = np.zeros((cfg.steps, Vn), dtype=np.float32)
        if t > 0:
            x_hits[1 : t + 1] = hits[0:t]
            x_vels[1 : t + 1] = vels[0:t]

        x = np.concatenate([x_hits, x_vels], axis=1)
        xt = torch.tensor(x[None, :, :], dtype=torch.float32, device=device)

        with torch.no_grad():
            hit_logits, vel_pred = model(xt)

        logits_t = hit_logits[0, t, :] / max(1e-6, float(scfg.temperature))
        probs_t = _sigmoid(logits_t).clamp(0.0, 1.0).cpu().numpy()
        vels_t = vel_pred[0, t, :].clamp(0.0, 1.0).cpu().numpy()

        if scfg.stochastic:
            step_hits = (rng.random(Vn) < probs_t).astype(np.float32)
        else:
            step_hits = (probs_t >= float(scfg.threshold)).astype(np.float32)

        step_vels = (vels_t * step_hits).astype(np.float32)

        hits[t, :] = step_hits
        vels[t, :] = step_vels

    return _apply_bar_constraints(hits, vels, bar_index=0, scfg=scfg, rng=rng)


def bars_to_notes(
    hits: np.ndarray,
    vels: np.ndarray,
    grid: BarGrid,
    *,
    start_bar: int = 0,
    bar_count: int = 1,
    scfg: SampleConfig = SampleConfig(),
) -> List[Note]:
    out: List[Note] = []
    step_len = float(grid.bar_duration) / 16.0

    def voice_dur(pitch: int) -> float:
        if pitch == 36:
            return 0.10
        if pitch == 38:
            return 0.10
        if pitch == 42:
            return 0.06
        if pitch == 46:
            return 0.12
        if pitch == 51:
            return 0.06
        if pitch == 49:
            return 0.14
        if pitch in (45, 47, 50):
            return 0.10
        return 0.10

    Vn = hits.shape[2] if hits.ndim == 3 else hits.shape[1]

    for b in range(bar_count):
        bar_start = float(grid.time_at(start_bar + b, 0.0))
        H = hits[b] if hits.ndim == 3 else hits
        VEL = vels[b] if vels.ndim == 3 else vels

        for step in range(16):
            t0 = bar_start + step * step_len

            for vidx in range(Vn):
                if H[step, vidx] <= 0.5:
                    continue

                pitch = INV_VOICE_MAP.get(vidx)
                if pitch is None:
                    continue

                dur = min(voice_dur(int(pitch)), step_len * 0.90)
                t1 = t0 + max(0.02, float(dur))

                raw = float(VEL[step, vidx]) * 127.0 * float(scfg.intensity)
                vel = int(max(scfg.vel_floor, min(scfg.vel_ceiling, raw)))

                out.append(Note(pitch=int(pitch), start=float(t0), end=float(t1), velocity=int(vel)))

    return out


def generate_ml_drums(
    model_path: str | Path,
    grid: BarGrid,
    scfg: SampleConfig = SampleConfig(),
) -> List[Note]:
    ts: TimeSignature = grid.time_signature
    if not (ts.numerator == 4 and ts.denominator == 4):
        raise ValueError("ML drums currently expects 4/4 (trained on 4/4 bars).")

    model, cfg, dev = load_model(model_path)

    rng = np.random.default_rng(scfg.seed)
    base_seed = scfg.seed if scfg.seed is not None else int(rng.integers(0, 1_000_000))

    all_hits: List[np.ndarray] = []
    all_vels: List[np.ndarray] = []

    if scfg.reuse_groove:
        anchor_cfg = SampleConfig(**{**scfg.__dict__, "seed": base_seed})
        anchor_hits, anchor_vels = sample_one_bar(model, cfg, dev, anchor_cfg)

        bar_rng = np.random.default_rng(base_seed + 1337)
        for b in range(int(scfg.bars)):
            if b == 0:
                h, v = anchor_hits, anchor_vels
            else:
                h, v = _mutate_groove_bar(anchor_hits, anchor_vels, bar_index=b, scfg=scfg, rng=bar_rng)

            h, v = _apply_bar_constraints(h, v, bar_index=b, scfg=scfg, rng=bar_rng)
            all_hits.append(h)
            all_vels.append(v)
    else:
        for b in range(int(scfg.bars)):
            bar_seed = base_seed + b * 997
            bar_cfg = SampleConfig(**{**scfg.__dict__, "seed": bar_seed})
            h, v = sample_one_bar(model, cfg, dev, bar_cfg)

            bar_rng = np.random.default_rng(bar_seed + 1337)
            h, v = _apply_bar_constraints(h, v, bar_index=b, scfg=scfg, rng=bar_rng)
            all_hits.append(h)
            all_vels.append(v)

    hits = np.stack(all_hits, axis=0)
    vels = np.stack(all_vels, axis=0)

    return bars_to_notes(hits, vels, grid, start_bar=0, bar_count=int(scfg.bars), scfg=scfg)
