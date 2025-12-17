# src/backingtrack/harmony_baseline.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .types import BarGrid, KeyEstimate, Note, Mode
from .moods import MoodPreset


# ----------------------------
# Data model (local for now)
# ----------------------------

@dataclass(frozen=True)
class ChordEvent:
    """
    A chord spanning a bar (or multiple bars).

    root_pc: 0..11 pitch class (C=0, C#=1, ..., B=11)
    quality: 'maj', 'min', 'dim', 'aug', 'sus2', 'sus4'
    extensions: semitone intervals above the root (e.g., (10,) for b7, (11,) for maj7, (14,) for add9)
    bar_index: which bar (0-based) this chord starts on
    start/end: absolute seconds (aligned to BarGrid)
    """
    root_pc: int
    quality: str
    extensions: Tuple[int, ...]
    bar_index: int
    start: float
    end: float

    def pitch_classes(self) -> Tuple[int, ...]:
        triads: Dict[str, Tuple[int, int, int]] = {
            "maj": (0, 4, 7),
            "min": (0, 3, 7),
            "dim": (0, 3, 6),
            "aug": (0, 4, 8),
            "sus2": (0, 2, 7),
            "sus4": (0, 5, 7),
        }
        if self.quality not in triads:
            raise ValueError(f"Unknown chord quality: {self.quality}")

        pcs = [ (self.root_pc + iv) % 12 for iv in triads[self.quality] ]
        pcs += [ (self.root_pc + iv) % 12 for iv in self.extensions ]

        out: List[int] = []
        for pc in pcs:
            if pc not in out:
                out.append(pc)
        return tuple(out)


# ----------------------------
# Theory helpers
# ----------------------------

_MAJOR_SCALE = np.array([0, 2, 4, 5, 7, 9, 11], dtype=np.int32)
_MINOR_SCALE = np.array([0, 2, 3, 5, 7, 8, 10], dtype=np.int32)  # natural minor (aeolian)

# Diatonic triad qualities by scale degree (1..7)
_MAJOR_QUALITIES = {1: "maj", 2: "min", 3: "min", 4: "maj", 5: "maj", 6: "min", 7: "dim"}
_MINOR_QUALITIES = {1: "min", 2: "dim", 3: "maj", 4: "min", 5: "min", 6: "maj", 7: "maj"}  # natural minor


def _scale_offsets(mode: Mode) -> np.ndarray:
    return _MAJOR_SCALE if mode == "major" else _MINOR_SCALE


def _diatonic_quality(mode: Mode, degree: int) -> str:
    if degree < 1 or degree > 7:
        raise ValueError("degree must be 1..7")
    return (_MAJOR_QUALITIES if mode == "major" else _MINOR_QUALITIES)[degree]


def _degree_to_root_pc(tonic_pc: int, mode: Mode, degree: int) -> int:
    """Map (tonic, mode, degree 1..7) -> root pitch class 0..11."""
    offsets = _scale_offsets(mode)
    return int((tonic_pc + int(offsets[degree - 1])) % 12)


def _add_color_extensions(
    degree: int,
    quality: str,
    mode: Mode,
    mood: MoodPreset,
) -> Tuple[int, ...]:
    """
    Add tasteful extensions based on mood flags.
    Keep this conservative for v1 so it still sounds “normal”.
    """
    exts: List[int] = []

    # Sevenths
    if mood.allow_sevenths:
        # Dominant 7 on V in major often sounds good
        if mode == "major" and degree == 5 and quality == "maj":
            exts.append(10)  # b7
        # Maj7 on I/IV can sound bright/dreamy
        elif quality == "maj" and degree in (1, 4):
            exts.append(11)  # maj7
        # Minor7 on i/ii/vi etc.
        elif quality == "min":
            exts.append(10)  # b7

    # add9 (14 semitones) for dreamy/happy/sad color
    if mood.allow_add9 and quality in ("maj", "min"):
        # don’t overdo it; only on tonic/subdominant-ish degrees
        if degree in (1, 4, 6, 3):
            exts.append(14)

    # sus chords: instead of triad quality swap
    # We'll handle sus by overriding quality in templates (below) more explicitly.
    # So here we don't add sus extensions.

    # Dedup while preserving order
    out: List[int] = []
    for iv in exts:
        if iv not in out:
            out.append(iv)
    return tuple(out)


# ----------------------------
# Templates per mood
# ----------------------------
# Each template step is: (degree, optional_quality_override)
# If override is None -> use diatonic quality for the key/mode.
TemplateStep = Tuple[int, Optional[str]]

_TEMPLATES: Dict[str, Dict[Mode, List[List[TemplateStep]]]] = {
    "happy": {
        "major": [
            [(1, None), (5, None), (6, None), (4, None)],  # I V vi IV
            [(1, None), (6, None), (4, None), (5, None)],  # I vi IV V
            [(6, None), (4, None), (1, None), (5, None)],  # vi IV I V
            [(1, None), (4, None), (5, None), (4, None)],  # I IV V IV
        ],
        "minor": [
            # If user picks happy but key is minor, we can still use brighter minor progressions
            [(6, None), (7, None), (1, None), (7, None)],  # VI VII i VII
            [(1, None), (7, None), (6, None), (7, None)],  # i VII VI VII
        ],
    },
    "sad": {
        "minor": [
            [(1, None), (6, None), (3, None), (7, None)],  # i VI III VII
            [(1, None), (4, None), (7, None), (3, None)],  # i iv VII III
            [(1, None), (7, None), (6, None), (4, None)],  # i VII VI iv
        ],
        "major": [
            # Sad in major: lean into vi / ii / IV
            [(6, None), (4, None), (1, None), (5, None)],  # vi IV I V
            [(6, None), (2, None), (4, None), (1, None)],  # vi ii IV I
        ],
    },
    "dreamy": {
        "major": [
            [(1, None), (4, "sus2"), (6, None), (5, None)],  # I  IVsus2  vi  V
            [(1, None), (5, "sus4"), (6, None), (4, None)],  # I  Vsus4   vi  IV
            [(4, None), (1, None), (6, None), (5, None)],    # IV I vi V
        ],
        "minor": [
            [(1, None), (6, None), (7, None), (3, None)],    # i VI VII III
            [(1, None), (4, None), (6, None), (7, None)],    # i iv VI VII
        ],
    },
    "tense": {
        "minor": [
            [(1, None), (2, None), (5, None), (1, None)],    # i ii° v i  (natural minor v)
            [(1, None), (4, None), (2, None), (5, None)],    # i iv ii° v
        ],
        "major": [
            [(1, None), (2, None), (5, None), (1, None)],    # I ii V I
            [(1, None), (4, None), (2, None), (5, None)],    # I IV ii V
        ],
    },
    "neutral": {
        "major": [
            [(1, None), (4, None), (5, None), (1, None)],    # I IV V I
            [(1, None), (5, None), (4, None), (1, None)],    # I V IV I
        ],
        "minor": [
            [(1, None), (6, None), (7, None), (1, None)],    # i VI VII i
            [(1, None), (4, None), (7, None), (1, None)],    # i iv VII i
        ],
    },
}


# ----------------------------
# Melody-based scoring
# ----------------------------

def _bar_pitch_classes(melody: Sequence[Note], grid: BarGrid, num_bars: int) -> List[List[int]]:
    """
    For each bar, collect pitch classes from melody notes that overlap that bar.
    """
    bars: List[List[int]] = [[] for _ in range(num_bars)]
    if not melody:
        return bars

    for n in melody:
        start_bar = grid.bar_index_at(n.start)
        end_bar = grid.bar_index_at(max(n.start, n.end - 1e-6))
        for b in range(max(0, start_bar), min(num_bars, end_bar + 1)):
            pc = int(n.pitch) % 12
            if pc not in bars[b]:
                bars[b].append(pc)
    return bars


def _score_template(
    template: List[TemplateStep],
    key: KeyEstimate,
    mood: MoodPreset,
    bar_melody_pcs: List[List[int]],
) -> float:
    """
    Higher score = chords contain melody notes more often.
    """
    total = 0.0
    num_bars = len(bar_melody_pcs)
    for b in range(num_bars):
        degree, q_override = template[b % len(template)]
        quality = q_override or _diatonic_quality(key.mode, degree)
        root_pc = _degree_to_root_pc(key.tonic_pc, key.mode, degree)
        exts = _add_color_extensions(degree, quality, key.mode, mood)
        chord_pcs = set(ChordEvent(root_pc, quality, exts, b, 0.0, 0.0).pitch_classes())

        pcs = bar_melody_pcs[b]
        if not pcs:
            # Neutral score if melody silent in this bar
            total += 0.2
            continue

        # reward contained notes, penalize clashes
        hits = sum(1 for pc in pcs if pc in chord_pcs)
        misses = sum(1 for pc in pcs if pc not in chord_pcs)

        total += 1.0 * hits - 0.6 * misses
    return total


def _choose_template(
    key: KeyEstimate,
    mood: MoodPreset,
    melody: Optional[Sequence[Note]],
    grid: BarGrid,
    num_bars: int,
) -> List[TemplateStep]:
    mood_name = mood.name if mood.name in _TEMPLATES else "neutral"
    pool = _TEMPLATES[mood_name].get(key.mode) or _TEMPLATES["neutral"][key.mode]

    if not melody:
        # no melody info -> just pick the first template (stable)
        return pool[0]

    bar_pcs = _bar_pitch_classes(melody, grid, num_bars)

    best = pool[0]
    best_score = -1e18
    for tmpl in pool:
        s = _score_template(tmpl, key, mood, bar_pcs)
        if s > best_score:
            best_score = s
            best = tmpl
    return best


# ----------------------------
# Public API
# ----------------------------

def infer_num_bars(duration_seconds: float, grid: BarGrid) -> int:
    """
    Convert duration (seconds) -> number of bars, rounding up.
    """
    if duration_seconds <= 0:
        return 4
    bars = int(np.ceil(duration_seconds / max(1e-9, grid.bar_duration)))
    return max(1, bars)


def generate_chords(
    key: KeyEstimate,
    grid: BarGrid,
    duration_seconds: float,
    mood: MoodPreset,
    melody_notes: Optional[Sequence[Note]] = None,
    bars_per_chord: int = 1,
) -> List[ChordEvent]:
    """
    Generate a baseline chord progression for the whole track.

    - One chord per bar by default (bars_per_chord=1)
    - Selects a template based on mood + key mode
    - If melody_notes provided, picks the best-fitting template by scoring

    Returns: list[ChordEvent] with start/end in seconds.
    """
    if bars_per_chord < 1:
        raise ValueError("bars_per_chord must be >= 1")

    num_bars = infer_num_bars(duration_seconds, grid)
    template = _choose_template(key, mood, melody_notes, grid, num_bars)

    chords: List[ChordEvent] = []
    b = 0
    while b < num_bars:
        step = template[(b // bars_per_chord) % len(template)]
        degree, q_override = step
        quality = q_override or _diatonic_quality(key.mode, degree)
        root_pc = _degree_to_root_pc(key.tonic_pc, key.mode, degree)
        exts = _add_color_extensions(degree, quality, key.mode, mood)

        start = float(grid.time_at(b, 0.0))
        end = float(grid.time_at(min(num_bars, b + bars_per_chord), 0.0))

        chords.append(
            ChordEvent(
                root_pc=root_pc,
                quality=quality,
                extensions=exts,
                bar_index=b,
                start=start,
                end=end,
            )
        )
        b += bars_per_chord

    return chords
