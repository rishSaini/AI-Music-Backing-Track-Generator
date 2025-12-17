# src/backingtrack/types.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Tuple

# --------
# Type aliases 
# --------
Seconds = float          # time in seconds (PrettyMIDI uses seconds)
BPM = float              # tempo in beats per minute
MidiPitch = int          # MIDI note number 0..127
Velocity = int           # MIDI velocity 0..127
PitchClass = int         # 0..11 (C=0, C#=1, ..., B=11)

Mode = Literal["major", "minor"]
ChordQuality = Literal["maj", "min", "dim", "aug", "sus2", "sus4"]


def _clamp_int(x: int, lo: int, hi: int, name: str) -> int:
    """Clamp an int into [lo, hi] and raise error if it’s wildly out of range."""
    if x < lo or x > hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {x}")
    return x


def _mod12(x: int) -> int:
    """Normalize any integer into a pitch class 0..11."""
    return x % 12


# --------
# Core musical primitives
# --------
@dataclass(frozen=True)
class Note:
    """
    A single note event in time.

    pitch: MIDI note number (0..127)
    start/end: seconds
    velocity: 0..127
    """
    pitch: MidiPitch
    start: Seconds
    end: Seconds
    velocity: Velocity = 100

    def __post_init__(self) -> None:
        # dataclass(frozen=True) means fields are immutable;
        # validation still works in __post_init__.
        _clamp_int(int(self.pitch), 0, 127, "pitch")
        _clamp_int(int(self.velocity), 0, 127, "velocity")
        if self.end <= self.start:
            raise ValueError(f"end must be > start, got start={self.start}, end={self.end}")

    @property
    def duration(self) -> Seconds:
        """Convenience: how long the note lasts."""
        return self.end - self.start

    @property
    def pitch_class(self) -> PitchClass:
        """Pitch class 0..11 (ignores octave)."""
        return _mod12(self.pitch)


# Intervals (in semitones) for common triad qualities, relative to the root.
_TRIAD_INTERVALS: dict[ChordQuality, Tuple[int, int, int]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
}


@dataclass(frozen=True)
class Chord:
    """
    A chord label over some span of time (often one bar).

    root_pc: pitch class 0..11
    quality: 'maj', 'min', etc.
    extensions: extra chord tones as semitone intervals above the root
               e.g. (10,) for a dominant 7th (b7), (11,) for maj7, (14,) for add9
    start/end: optional seconds (filled in once we align to a grid)
    bar_index: optional bar index (0-based) when you’re working in bars instead of seconds
    """
    root_pc: PitchClass
    quality: ChordQuality
    extensions: Tuple[int, ...] = ()
    start: Optional[Seconds] = None
    end: Optional[Seconds] = None
    bar_index: Optional[int] = None

    def __post_init__(self) -> None:
        _clamp_int(int(self.root_pc), 0, 11, "root_pc")
        if self.start is not None and self.end is not None and self.end <= self.start:
            raise ValueError(f"Chord end must be > start, got start={self.start}, end={self.end}")
        if self.bar_index is not None and self.bar_index < 0:
            raise ValueError(f"bar_index must be >= 0, got {self.bar_index}")
        # extensions are intervals; they can be 0..(say) 24 safely for v1
        for iv in self.extensions:
            if iv < 0 or iv > 24:
                raise ValueError(f"extension intervals should be in [0, 24], got {iv}")

    def pitch_classes(self) -> Tuple[PitchClass, ...]:
        """
        Return the chord tones as pitch classes (0..11).
        Example: Cmaj -> (0,4,7)
        """
        base = _TRIAD_INTERVALS[self.quality]
        pcs = [_mod12(self.root_pc + iv) for iv in base]
        pcs += [_mod12(self.root_pc + iv) for iv in self.extensions]
        # keep stable ordering, remove duplicates
        out: list[int] = []
        for pc in pcs:
            if pc not in out:
                out.append(pc)
        return tuple(out)


@dataclass(frozen=True)
class KeyEstimate:
    """
    Output of key detection.

    tonic_pc: pitch class of tonic (0..11)
    mode: 'major' or 'minor'
    confidence: 0..1 (the algorithm defines what it means; v1 can be heuristic)
    """
    tonic_pc: PitchClass
    mode: Mode
    confidence: float = 0.0

    def __post_init__(self) -> None:
        _clamp_int(int(self.tonic_pc), 0, 11, "tonic_pc")
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")


@dataclass(frozen=True)
class TimeSignature:
    """
    Simple constant time signature.

    numerator/denominator: e.g. 4/4, 3/4, 6/8
    """
    numerator: int = 4
    denominator: int = 4

    def __post_init__(self) -> None:
        if self.numerator <= 0:
            raise ValueError(f"numerator must be > 0, got {self.numerator}")
        if self.denominator not in (1, 2, 4, 8, 16, 32):
            raise ValueError(f"denominator should be a power-of-two like 4 or 8, got {self.denominator}")


@dataclass(frozen=True)
class BarGrid:
    """
    A constant-tempo timing grid that lets you convert between:
    - (bar index, beat within bar) <-> absolute time in seconds

    Assumptions for v1:
    - one global tempo
    - one global time signature
    - bar 0 starts at start_time (usually 0.0)
    """
    tempo_bpm: BPM
    time_signature: TimeSignature = TimeSignature()
    start_time: Seconds = 0.0

    def __post_init__(self) -> None:
        if self.tempo_bpm <= 0:
            raise ValueError(f"tempo_bpm must be > 0, got {self.tempo_bpm}")

    @property
    def seconds_per_beat(self) -> Seconds:
        """At 120 BPM, seconds_per_beat = 0.5."""
        return 60.0 / self.tempo_bpm

    @property
    def beats_per_bar(self) -> float:
        """
        In 4/4, beats_per_bar=4.
        In 6/8, we still treat the 'beat' as the denominator unit (eighth note),
        so beats_per_bar=6.
        """
        return float(self.time_signature.numerator)

    @property
    def bar_duration(self) -> Seconds:
        """How long one bar lasts in seconds."""
        return self.beats_per_bar * self.seconds_per_beat

    def time_at(self, bar_index: int, beat_in_bar: float = 0.0) -> Seconds:
        """
        Convert (bar, beat) -> seconds.
        beat_in_bar can be fractional (e.g. 1.5 beats into the bar).
        """
        if bar_index < 0:
            raise ValueError("bar_index must be >= 0")
        if beat_in_bar < 0:
            raise ValueError("beat_in_bar must be >= 0")
        return self.start_time + bar_index * self.bar_duration + beat_in_bar * self.seconds_per_beat

    def bar_index_at(self, t: Seconds) -> int:
        """Convert time (seconds) -> bar index (0-based)."""
        if t < self.start_time:
            return 0
        return int((t - self.start_time) // self.bar_duration)

    def quantize_time_to_beat(self, t: Seconds) -> Seconds:
        """
        Snap a time to the nearest beat on this grid (useful for clean MIDI output).
        """
        if t < self.start_time:
            return self.start_time
        beats_from_start = (t - self.start_time) / self.seconds_per_beat
        nearest_beat = round(beats_from_start)
        return self.start_time + nearest_beat * self.seconds_per_beat


@dataclass(frozen=True)
class MidiInfo:
    """
    Metadata we’ll extract from the input MIDI and pass around.

    duration: total song length in seconds
    tempo_bpm: a single chosen tempo (v1: first tempo or a default)
    time_signature: single chosen time signature (v1: first TS or default 4/4)
    """
    duration: Seconds
    tempo_bpm: BPM = 120.0
    time_signature: TimeSignature = TimeSignature()

    def __post_init__(self) -> None:
        if self.duration < 0:
            raise ValueError(f"duration must be >= 0, got {self.duration}")
        if self.tempo_bpm <= 0:
            raise ValueError(f"tempo_bpm must be > 0, got {self.tempo_bpm}")


# Optional: role labels for tracks we generate
TrackRole = Literal["melody", "chords", "bass", "pad", "drums"]
