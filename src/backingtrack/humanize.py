# src/backingtrack/humanize.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import random

from .types import BarGrid, Note
from .arrange import Arrangement, TrackName


@dataclass(frozen=True)
class HumanizeConfig:
    """
    Humanization controls.

    timing_jitter_ms:
      Random time shift (uniform) applied to note start/end.

    velocity_jitter:
      Random velocity shift (uniform integer) applied to note velocity.

    swing:
      0.0 = straight
      1.0 = heavy swing (off-beat 8ths delayed towards triplet feel)

    seed:
      If set, results are deterministic.
    """
    timing_jitter_ms: float = 15.0
    velocity_jitter: int = 8
    swing: float = 0.0
    seed: Optional[int] = None

    # Per-track timing multipliers (drums usually need less jitter)
    drums_timing_mult: float = 0.45
    bass_timing_mult: float = 0.75
    pad_timing_mult: float = 0.55

    # Per-track velocity multipliers
    drums_vel_mult: float = 0.60
    bass_vel_mult: float = 0.85
    pad_vel_mult: float = 0.55


def _swing_delay_seconds(grid: BarGrid, swing: float) -> float:
    """
    For 8th-note swing:
      straight offbeat is at 0.5 beat
      triplet offbeat is at 2/3 beat
      delta = (2/3 - 1/2) = 1/6 beat
    """
    if swing <= 0:
        return 0.0
    return float(swing) * (grid.seconds_per_beat / 6.0)


def _is_offbeat_8th(grid: BarGrid, t: float, tol_beats: float = 0.08) -> bool:
    """
    Returns True if time t is close to an offbeat 8th (…0.5, 1.5, 2.5 beats…).
    """
    beats = (t - grid.start_time) / grid.seconds_per_beat
    frac = beats % 1.0
    return abs(frac - 0.5) <= tol_beats


def humanize_notes(
    notes: Sequence[Note],
    grid: BarGrid,
    *,
    timing_jitter_ms: float,
    velocity_jitter: int,
    swing: float,
    apply_swing: bool,
    timing_mult: float,
    vel_mult: float,
    rng: random.Random,
) -> List[Note]:
    """
    Return a new list of Notes with humanized timing/velocity.
    Keeps each note's duration constant.
    """
    out: List[Note] = []
    jitter_sec = (timing_jitter_ms / 1000.0) * float(timing_mult)
    swing_delay = _swing_delay_seconds(grid, swing) if apply_swing else 0.0

    for n in notes:
        dur = n.end - n.start

        dt = 0.0
        if jitter_sec > 0:
            dt += rng.uniform(-jitter_sec, jitter_sec)

        if swing_delay > 0 and _is_offbeat_8th(grid, n.start):
            dt += swing_delay

        new_start = max(grid.start_time, n.start + dt)
        new_end = new_start + dur

        dv = 0
        if velocity_jitter > 0:
            spread = int(round(velocity_jitter * float(vel_mult)))
            if spread > 0:
                dv = rng.randint(-spread, spread)

        new_vel = max(1, min(127, int(n.velocity) + dv))

        # Safety: ensure end > start
        if new_end <= new_start:
            new_end = new_start + max(0.03, dur)

        out.append(Note(pitch=int(n.pitch), start=float(new_start), end=float(new_end), velocity=int(new_vel)))

    return out


def humanize_arrangement(
    arrangement: Arrangement,
    grid: BarGrid,
    config: HumanizeConfig = HumanizeConfig(),
) -> Arrangement:
    """
    Humanize bass/pad/drums tracks with slightly different amounts per track.
    """
    rng = random.Random(config.seed)

    tracks: Dict[TrackName, List[Note]] = {"bass": [], "pad": [], "drums": []}

    if arrangement.tracks.get("bass"):
        tracks["bass"] = humanize_notes(
            arrangement.tracks["bass"],
            grid,
            timing_jitter_ms=config.timing_jitter_ms,
            velocity_jitter=config.velocity_jitter,
            swing=config.swing,
            apply_swing=True,
            timing_mult=config.bass_timing_mult,
            vel_mult=config.bass_vel_mult,
            rng=rng,
        )

    if arrangement.tracks.get("pad"):
        tracks["pad"] = humanize_notes(
            arrangement.tracks["pad"],
            grid,
            timing_jitter_ms=config.timing_jitter_ms,
            velocity_jitter=config.velocity_jitter,
            swing=config.swing,
            apply_swing=False,  # pads usually sound better without swing
            timing_mult=config.pad_timing_mult,
            vel_mult=config.pad_vel_mult,
            rng=rng,
        )

    if arrangement.tracks.get("drums"):
        tracks["drums"] = humanize_notes(
            arrangement.tracks["drums"],
            grid,
            timing_jitter_ms=config.timing_jitter_ms,
            velocity_jitter=config.velocity_jitter,
            swing=config.swing,
            apply_swing=True,
            timing_mult=config.drums_timing_mult,
            vel_mult=config.drums_vel_mult,
            rng=rng,
        )

    return Arrangement(tracks=tracks)
