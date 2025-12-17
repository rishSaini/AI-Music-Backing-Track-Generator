# src/backingtrack/melody.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, List

import pretty_midi

from .types import BarGrid, Note


MelodyStrategy = Literal["highest_pitch", "loudest"]


@dataclass(frozen=True)
class MelodyConfig:
    """
    Controls how we extract a single "melody line" from an instrument track.

    min_note_duration:
        Notes shorter than this (seconds) are dropped (often MIDI noise / ornaments).

    overlap_tolerance:
        If a new note starts within this many seconds of the previous note ending,
        we treat it as contiguous (helps avoid tiny overlaps or gaps from sloppy MIDI).

    merge_gap:
        If two consecutive melody notes have the same pitch and the gap between them
        is <= merge_gap, we merge them into one longer note.

    strategy:
        When there is polyphony/overlap, decide which note "wins":
          - highest_pitch: lead lines usually sit on top
          - loudest: useful if melody is emphasized by velocity

    quantize_to_beat:
        If True and a BarGrid is provided, snap note start/end times to the nearest beat.
        This can make generated backing tracks align more cleanly.
    """
    min_note_duration: float = 0.05
    overlap_tolerance: float = 0.02
    merge_gap: float = 0.03
    strategy: MelodyStrategy = "highest_pitch"
    quantize_to_beat: bool = False


def extract_melody_notes(
    instrument: pretty_midi.Instrument,
    grid: Optional[BarGrid] = None,
    config: MelodyConfig = MelodyConfig(),
) -> List[Note]:
    """
    Extract a monophonic melody line (list of Note) from a PrettyMIDI Instrument.

    Typical usage:
      melody_inst, _ = pick_melody_instrument(pm)
      melody_notes = extract_melody_notes(melody_inst, grid=grid)

    Steps:
    1) Filter out very short notes
    2) Sort notes by time
    3) Resolve overlaps into a single line (monophonic "topline")
    4) Merge repeated same-pitch notes with tiny gaps
    5) Optional quantization to beat grid
    """
    if not instrument.notes:
        return []

    # 1) Filter out tiny notes (often spurious)
    raw = [
        n for n in instrument.notes
        if (n.end - n.start) >= config.min_note_duration
    ]
    if not raw:
        return []

    # 2) Sort notes.
    #    For notes with the same start time, sort by “winner” criteria so the best candidate is first.
    def _note_rank(n: pretty_midi.Note) -> float:
        if config.strategy == "highest_pitch":
            return float(n.pitch)
        return float(n.velocity)  # "loudest"

    raw.sort(key=lambda n: (n.start, -_note_rank(n), -n.pitch, -(n.end - n.start)))

    # 3) Build monophonic line by walking notes in order and resolving overlaps.
    melody: List[pretty_midi.Note] = []

    for n in raw:
        if not melody:
            melody.append(n)
            continue

        prev = melody[-1]

        # If this note clearly starts after the previous ends (with a small tolerance), accept it.
        if n.start >= prev.end - config.overlap_tolerance:
            melody.append(n)
            continue

        # Otherwise, there is overlap/polyphony:
        # Decide whether to replace the previous note with this one, or ignore this one.
        prev_score = _note_rank(prev)
        curr_score = _note_rank(n)

        if curr_score > prev_score:
            # New note "wins". Trim previous note to end at the new note's start.
            trimmed_prev_end = max(prev.start + 1e-4, n.start)
            prev_trimmed = pretty_midi.Note(
                velocity=prev.velocity,
                pitch=prev.pitch,
                start=prev.start,
                end=trimmed_prev_end,
            )
            melody[-1] = prev_trimmed
            melody.append(n)
        else:
            # New note loses; likely harmony/accompaniment. Skip it.
            continue

    # 4) Merge consecutive same-pitch notes separated by a tiny gap.
    merged: List[pretty_midi.Note] = []
    for n in melody:
        if not merged:
            merged.append(n)
            continue

        prev = merged[-1]
        gap = n.start - prev.end

        if n.pitch == prev.pitch and gap <= config.merge_gap:
            # Merge: extend previous note
            merged[-1] = pretty_midi.Note(
                velocity=max(prev.velocity, n.velocity),
                pitch=prev.pitch,
                start=prev.start,
                end=max(prev.end, n.end),
            )
        else:
            merged.append(n)

    # 5) Convert to our internal Note type (+ optional quantization)
    out: List[Note] = []
    for n in merged:
        start = float(n.start)
        end = float(n.end)

        if config.quantize_to_beat and grid is not None:
            start = float(grid.quantize_time_to_beat(start))
            end = float(grid.quantize_time_to_beat(end))

            # If quantization collapses the note, nudge end slightly
            if end <= start:
                end = start + 0.05

        # Final safety filter
        if (end - start) < config.min_note_duration:
            continue

        out.append(Note(pitch=int(n.pitch), start=start, end=end, velocity=int(n.velocity)))

    return out
