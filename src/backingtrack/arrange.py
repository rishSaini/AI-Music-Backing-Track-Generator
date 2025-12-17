# src/backingtrack/arrange.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from .types import BarGrid, Note, TimeSignature
from .moods import MoodPreset
from .harmony_baseline import ChordEvent


TrackName = Literal["bass", "pad", "drums"]


@dataclass(frozen=True)
class Arrangement:
    """
    Container for generated backing-track notes.
    Each track is a list[Note] in absolute seconds.
    """
    tracks: Dict[TrackName, List[Note]]


# --- GM drum note numbers (General MIDI percussion) ---
KICK = 36
SNARE = 38
CLOSED_HH = 42
OPEN_HH = 46


def arrange_backing(
    chords: Sequence[ChordEvent],
    grid: BarGrid,
    mood: MoodPreset,
    *,
    make_bass: bool = True,
    make_pad: bool = True,
    make_drums: bool = True,
    bass_octave: int = 2,   # 2 => around MIDI 36..47
    pad_octave: int = 4,    # 4 => around MIDI ~60
) -> Arrangement:
    """
    Generate backing tracks from chords + mood.
    Returns Note events (not MIDI yet). render.py will convert these into tracks.
    """
    tracks: Dict[TrackName, List[Note]] = {"bass": [], "pad": [], "drums": []}

    if make_bass:
        tracks["bass"] = _make_bass(chords, grid, mood, bass_octave=bass_octave)

    if make_pad:
        tracks["pad"] = _make_pad(chords, grid, mood, pad_octave=pad_octave)

    if make_drums:
        tracks["drums"] = _make_drums(chords, grid, mood)

    return Arrangement(tracks=tracks)


# -------------------------
# Bass
# -------------------------

def _make_bass(
    chords: Sequence[ChordEvent],
    grid: BarGrid,
    mood: MoodPreset,
    *,
    bass_octave: int,
) -> List[Note]:
    """
    Simple bass patterns based on mood.rhythm_density:
      low: whole-bar root
      mid: half-notes (beat 1 and beat 3 in 4/4)
      high: quarter-notes (root each beat)
    """
    out: List[Note] = []

    base_pitch = 12 * bass_octave  # octave 2 => 24, but we’ll target around +12 later
    # Use a typical bass “center” near E2 (40) if octave=2
    center = base_pitch + 16  # ~40 when bass_octave=2

    density = float(mood.rhythm_density)

    for ch in chords:
        root = _nearest_midi_for_pc(ch.root_pc, around=center, lo=28, hi=52)

        # Decide subdivision pattern
        ts = grid.time_signature
        beats = int(ts.numerator)

        if density <= 0.40:
            # whole-bar sustain (or multi-bar sustain)
            out.append(Note(pitch=root, start=ch.start, end=ch.end, velocity=95))
            continue

        if density <= 0.70:
            # two hits per bar (best in 4/4; fallback to 1 + middle)
            if beats >= 4:
                hit_beats = [0.0, 2.0]  # beat 1 and 3
            else:
                hit_beats = [0.0, beats / 2.0]
        else:
            # quarter-note roots
            hit_beats = [float(b) for b in range(beats)]

        for b in hit_beats:
            t0 = max(ch.start, grid.time_at(grid.bar_index_at(ch.start), b))
            t1 = min(ch.end, t0 + grid.seconds_per_beat * 0.95)
            if t1 > t0:
                out.append(Note(pitch=root, start=t0, end=t1, velocity=95))

    return out


# -------------------------
# Pad
# -------------------------

_TRIAD_INTERVALS: dict[str, Tuple[int, int, int]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
}


def _make_pad(
    chords: Sequence[ChordEvent],
    grid: BarGrid,
    mood: MoodPreset,
    *,
    pad_octave: int,
) -> List[Note]:
    """
    Sustained block-chord pad with basic voice-leading:
    choose inversions/voicings that minimize movement from previous chord.
    """
    out: List[Note] = []

    shift = int(round(float(mood.brightness) * 8))
    target = 12 * pad_octave + 12 + shift  # around C5-ish

    max_notes = 3 if mood.rhythm_density < 0.55 else 4

    prev_voicing: Optional[List[int]] = None

    for ch in chords:
        pcs = _ordered_chord_pcs(ch)[:max_notes]

        candidates = _voicing_candidates(pcs, around=target, lo=48, hi=92)
        if not candidates:
            continue

        if prev_voicing is None:
            voicing = candidates[0]
        else:
            voicing = min(candidates, key=lambda v: _voicing_cost(v, prev_voicing))

        prev_voicing = voicing

        vel = 60 if mood.rhythm_density < 0.55 else 68
        start = ch.start
        end = max(start + 0.08, ch.end - 0.01)

        for p in voicing:
            out.append(Note(pitch=p, start=start, end=end, velocity=vel))

    return out


def _ordered_chord_pcs(ch: ChordEvent) -> List[int]:
    """
    Return chord pitch-classes in a musically sensible order:
    root, third/second/fourth, fifth, then extensions.
    """
    if ch.quality not in _TRIAD_INTERVALS:
        # fallback: just use whatever pitch_classes() gives
        return list(ch.pitch_classes())

    triad = _TRIAD_INTERVALS[ch.quality]
    pcs = [ (ch.root_pc + iv) % 12 for iv in triad ]
    pcs += [ (ch.root_pc + iv) % 12 for iv in ch.extensions ]

    # dedup, preserve order
    out: List[int] = []
    for pc in pcs:
        if pc not in out:
            out.append(pc)
    return out


def _voicing_from_pcs(pcs: Sequence[int], *, around: int, max_notes: int) -> List[int]:
    """
    Convert pitch-classes to MIDI pitches near 'around', creating a clean ascending voicing.
    """
    pcs = list(pcs)[:max_notes]

    # First pass: map each pc near around
    pitches = [_nearest_midi_for_pc(pc, around=around, lo=36, hi=96) for pc in pcs]
    pitches.sort()

    # Ensure strictly ascending (avoid collisions like same pitch)
    fixed: List[int] = []
    for p in pitches:
        if not fixed:
            fixed.append(p)
            continue
        while p <= fixed[-1]:
            p += 12
        if p <= 127:
            fixed.append(p)

    # Clamp and return
    return [min(127, max(0, p)) for p in fixed]


def _nearest_midi_for_pc(pc: int, *, around: int, lo: int, hi: int) -> int:
    """
    Find a MIDI pitch with pitch-class pc that is closest to 'around', clamped to [lo, hi].
    """
    pc = int(pc) % 12
    around = int(around)

    # Candidate pitches: around +/- a few octaves
    candidates: List[int] = []
    base = around - ((around % 12) - pc)
    for k in range(-5, 6):
        candidates.append(base + 12 * k)

    # Choose closest within bounds (or closest overall then clamp)
    best = min(candidates, key=lambda p: abs(p - around))
    best = max(lo, min(hi, best))
    return best


# -------------------------
# Drums
# -------------------------

def _make_drums(
    chords: Sequence[ChordEvent],
    grid: BarGrid,
    mood: MoodPreset,
) -> List[Note]:
    """
    Very simple groove using GM drum pitches.
    Assumes BarGrid beat == time_signature numerator unit.
    """
    out: List[Note] = []
    ts: TimeSignature = grid.time_signature
    beats = int(ts.numerator)
    spb = float(grid.seconds_per_beat)

    density = float(mood.rhythm_density)

    # hi-hat subdivision: 8ths for normal, 16ths when dense
    hat_step = 0.5 if density < 0.75 else 0.25  # in beats

    for ch in chords:
        bar = grid.bar_index_at(ch.start)
        bar_start = grid.time_at(bar, 0.0)
        bar_end = min(ch.end, grid.time_at(bar + 1, 0.0))  # only first bar span for patterning

        # KICK / SNARE patterns depend on meter
        if beats == 4:
            # Kick: 1 and 3, Snare: 2 and 4
            kick_beats = [0.0, 2.0] if density >= 0.35 else [0.0]
            snare_beats = [1.0, 3.0]
        elif beats == 3:
            # Waltz-ish: kick on 1, snare on 2
            kick_beats = [0.0]
            snare_beats = [1.0] if density >= 0.4 else []
        elif beats == 6:
            # 6/8 feel: kick on 1, snare on 4 (counting eighths)
            kick_beats = [0.0]
            snare_beats = [3.0] if density >= 0.4 else []
        else:
            # generic: kick on 1, snare mid-bar
            kick_beats = [0.0]
            snare_beats = [beats / 2.0] if beats > 1 else []

        # Add kick
        for b in kick_beats:
            t0 = bar_start + b * spb
            t1 = min(bar_end, t0 + 0.10)
            if t1 > ch.start and t0 < ch.end:
                out.append(Note(pitch=KICK, start=max(ch.start, t0), end=min(ch.end, t1), velocity=105))

        # Add snare
        for b in snare_beats:
            t0 = bar_start + b * spb
            t1 = min(bar_end, t0 + 0.10)
            if t1 > ch.start and t0 < ch.end:
                out.append(Note(pitch=SNARE, start=max(ch.start, t0), end=min(ch.end, t1), velocity=98))

        # Hi-hat across the bar
        # (Skip if super sparse)
        if density >= 0.25:
            hb = 0.0
            while hb < beats - 1e-9:
                t0 = bar_start + hb * spb
                t1 = min(bar_end, t0 + 0.06)
                if t1 > ch.start and t0 < ch.end:
                    out.append(Note(pitch=CLOSED_HH, start=max(ch.start, t0), end=min(ch.end, t1), velocity=70))
                hb += hat_step

            # Optional open hat on last beat for more energy
            if density >= 0.75 and beats >= 4:
                t0 = bar_start + (beats - 0.5) * spb
                t1 = min(bar_end, t0 + 0.12)
                if t1 > ch.start and t0 < ch.end:
                    out.append(Note(pitch=OPEN_HH, start=max(ch.start, t0), end=min(ch.end, t1), velocity=80))

    return out

def _voicing_candidates(pcs: Sequence[int], *, around: int, lo: int, hi: int) -> List[List[int]]:
    """
    Generate a few inversion-like voicings near 'around'.
    """
    pcs = list(pcs)
    if not pcs:
        return []

    candidates: List[List[int]] = []
    n = len(pcs)

    for inv in range(n):
        rot = pcs[inv:] + pcs[:inv]

        # Build ascending pitches
        p0 = _nearest_midi_for_pc(rot[0], around=around, lo=lo, hi=hi)
        pitches = [p0]

        for pc in rot[1:]:
            p = _nearest_midi_for_pc(pc, around=pitches[-1] + 7, lo=lo, hi=hi)
            while p <= pitches[-1] + 2:
                p += 12
            pitches.append(p)

        # If overall center is far from target, shift by octaves
        center = sum(pitches) / len(pitches)
        while center < around - 6 and max(pitches) + 12 <= hi:
            pitches = [p + 12 for p in pitches]
            center = sum(pitches) / len(pitches)
        while center > around + 6 and min(pitches) - 12 >= lo:
            pitches = [p - 12 for p in pitches]
            center = sum(pitches) / len(pitches)

        # Clamp and ensure sorted
        pitches = [max(lo, min(hi, p)) for p in pitches]
        pitches.sort()

        # Remove duplicates by pushing up octaves if needed
        fixed: List[int] = []
        for p in pitches:
            if not fixed:
                fixed.append(p)
                continue
            while p <= fixed[-1]:
                p += 12
            if p <= hi:
                fixed.append(p)

        if len(fixed) >= 3:
            candidates.append(fixed)

    # Dedup by tuple representation
    uniq: List[List[int]] = []
    seen = set()
    for c in candidates:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            uniq.append(c)
    return uniq


def _voicing_cost(curr: Sequence[int], prev: Sequence[int]) -> float:
    """
    Smaller cost = smoother movement.
    """
    m = min(len(curr), len(prev))
    cost = sum(abs(curr[i] - prev[i]) for i in range(m))

    # Penalize big jumps in chord "center"
    c_center = sum(curr) / len(curr)
    p_center = sum(prev) / len(prev)
    cost += 0.35 * abs(c_center - p_center)

    # Penalize extreme spread (too wide sounds thin, too tight can clash)
    spread = max(curr) - min(curr)
    if spread > 19:  # > ~a 10th
        cost += (spread - 19) * 0.25

    return cost
