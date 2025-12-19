# src/backingtrack/midi_io.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pretty_midi

from .types import BarGrid, MidiInfo, TimeSignature


DEFAULT_TEMPO_BPM = 120.0
DEFAULT_TIME_SIGNATURE = TimeSignature(4, 4)


@dataclass(frozen=True)
class MelodySelection:
    """
    What we chose as the melody track (instrument) from the MIDI.
    Keeping this structured makes debugging + CLI output easier.
    """
    instrument_index: int
    instrument_name: str
    is_drum: bool


def load_midi(path: str | Path) -> pretty_midi.PrettyMIDI:
    """
    Load a MIDI file from disk and return a PrettyMIDI object.

    Why this exists:
    - central place to validate path / extension
    - central place to control parsing behavior later if needed
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"MIDI file not found: {p}")
    if p.suffix.lower() not in {".mid", ".midi"}:
        raise ValueError(f"Expected a .mid or .midi file, got: {p.name}")
    
    #TODO: More security: instead of just checking the suffix of the file, check file bytes to validate as MIDI.
    #TODO: Limit MIDI file size upload,

    # PrettyMIDI parses into instruments, notes, tempo changes, time sig changes, etc.
    return pretty_midi.PrettyMIDI(str(p))


def extract_midi_info(pm: pretty_midi.PrettyMIDI) -> MidiInfo:
    """
    Pull out simple global metadata from the MIDI for v1:
    - duration (end time)
    - a single chosen tempo (BPM)
    - a single chosen time signature

    NOTE: real MIDIs can have tempo/time-signature changes.
    For v1 we pick the earliest / first and treat it as constant.
    """
    duration = float(pm.get_end_time())

    # Tempo changes: returns (change_times_seconds, tempi_bpm)
    tempo_times, tempi = pm.get_tempo_changes()
    if len(tempi) > 0:
        tempo_bpm = float(tempi[0])  # earliest tempo
    else:
        tempo_bpm = DEFAULT_TEMPO_BPM

    # Time signature changes: list of pretty_midi.TimeSignature objects
    if pm.time_signature_changes:
        # choose the earliest time signature event
        ts0 = min(pm.time_signature_changes, key=lambda ts: ts.time)
        time_signature = TimeSignature(int(ts0.numerator), int(ts0.denominator))
    else:
        time_signature = DEFAULT_TIME_SIGNATURE

    return MidiInfo(duration=duration, tempo_bpm=tempo_bpm, time_signature=time_signature)


def build_bar_grid(info: MidiInfo, start_time: float = 0.0) -> BarGrid:
    """
    Create a constant-tempo BarGrid from extracted MidiInfo.
    This grid lets the rest of your code talk in bars/beats instead of seconds.
    """
    return BarGrid(
        tempo_bpm=float(info.tempo_bpm),
        time_signature=info.time_signature,
        start_time=float(start_time),
    )


def pick_melody_instrument(
    pm: pretty_midi.PrettyMIDI,
    instrument_index: Optional[int] = None,
) -> Tuple[pretty_midi.Instrument, MelodySelection]:
    """
    Choose which instrument track is "the melody".

    If instrument_index is provided: pick that exact instrument.
    Otherwise: score tracks and pick the most "lead-like" one.

    Key improvement:
    - prefer instruments that span a large portion of the song (coverage),
      so we don't accidentally pick a 3-second intro hook track.
    """
    instruments = pm.instruments
    if not instruments:
        raise ValueError("No instruments found in MIDI.")

    # If user explicitly specifies an index, just use it (with validation).
    if instrument_index is not None:
        if instrument_index < 0 or instrument_index >= len(instruments):
            raise IndexError(
                f"instrument_index out of range: {instrument_index} "
                f"(valid 0..{len(instruments)-1})"
            )
        inst = instruments[instrument_index]
        sel = MelodySelection(
            instrument_index=instrument_index,
            instrument_name=inst.name or f"Instrument {instrument_index}",
            is_drum=bool(inst.is_drum),
        )
        return inst, sel

    # Otherwise score candidates.
    best_idx = None
    best_score = -1e18

    song_end = float(pm.get_end_time()) or 1.0

    for idx, inst in enumerate(instruments):
        if inst.is_drum:
            continue
        if not inst.notes:
            continue

        pitches = np.array([n.pitch for n in inst.notes], dtype=np.float32)
        note_count = len(inst.notes)

        inst_start = float(min(n.start for n in inst.notes))
        inst_end = float(max(n.end for n in inst.notes))
        span = max(1e-6, inst_end - inst_start)

        # How much of the song this instrument covers (0..1)
        coverage = span / max(1e-6, song_end)
        # How close it gets to the end (0..1)
        end_ratio = inst_end / max(1e-6, song_end)

        # Pitch stats (melody tends to be higher)
        median_pitch = float(np.median(pitches))
        p90_pitch = float(np.percentile(pitches, 90))

        # Track-name hints
        name = (inst.name or "").strip().lower()
        penalty = 0.0
        bonus = 0.0

        # Penalize typical accompaniment names
        # NOTE: removed "guitar" because guitar is often the lead.
        for bad in ("bass", "pad", "chord", "harmony", "accomp", "comp", "rhythm"):
            if bad in name:
                penalty += 20.0

        # Reward typical melody names
        for good in ("melody", "lead", "vocal", "voice", "solo", "theme"):
            if good in name:
                bonus += 25.0

        # Strongly discourage tiny intro/FX tracks
        short_pen = 0.0
        if coverage < 0.20:
            short_pen = 120.0 * (0.20 - coverage) / 0.20  # up to -120

        score = (
            1.0 * median_pitch
            + 0.7 * p90_pitch
            + 5.0 * np.log1p(note_count)
            + 110.0 * coverage
            + 25.0 * end_ratio
            + bonus
            - penalty
            - short_pen
        )

        if score > best_score:
            best_score = score
            best_idx = idx

    # Fallback: if everything was drums/empty, pick the first with notes.
    if best_idx is None:
        for idx, inst in enumerate(instruments):
            if inst.notes:
                best_idx = idx
                break

    if best_idx is None:
        raise ValueError("MIDI contains no notes in any instrument track.")

    inst = instruments[best_idx]
    sel = MelodySelection(
        instrument_index=best_idx,
        instrument_name=inst.name or f"Instrument {best_idx}",
        is_drum=bool(inst.is_drum),
    )
    return inst, sel


def load_and_prepare(
    path: str | Path,
    melody_instrument_index: Optional[int] = None,
) -> Tuple[pretty_midi.PrettyMIDI, MidiInfo, BarGrid, pretty_midi.Instrument, MelodySelection]:
    """
    Convenience function for the rest of the app:
    - load MIDI
    - extract global info
    - build bar grid
    - choose melody instrument

    Returns everything the next steps (melody extraction, key detect, harmony) need.
    """
    pm = load_midi(path)
    info = extract_midi_info(pm)
    grid = build_bar_grid(info)

    melody_inst, selection = pick_melody_instrument(pm, instrument_index=melody_instrument_index)
    return pm, info, grid, melody_inst, selection
