from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pretty_midi

# Voices we model
VOICE_MAP: Dict[int, int] = {
    36: 0,  # Kick
    38: 1,  # Snare
    42: 2,  # Closed HH
    46: 3,  # Open HH
    51: 4,  # Ride
    49: 5,  # Crash
    45: 6,  # Low Tom
    47: 7,  # Mid Tom
    50: 8,  # High Tom
}
N_VOICES = 9


@dataclass(frozen=True)
class DrumBar:
    hits: np.ndarray  # (16, V) float32 in {0,1}
    vels: np.ndarray  # (16, V) float32 in [0,1]
    tempo_bpm: float


def _get_constant_tempo(pm: pretty_midi.PrettyMIDI) -> float:
    times, tempi = pm.get_tempo_changes()
    return float(tempi[0]) if len(tempi) else 120.0


def _is_4_4(pm: pretty_midi.PrettyMIDI) -> bool:
    if not pm.time_signature_changes:
        return True
    ts = pm.time_signature_changes[0]
    return ts.numerator == 4 and ts.denominator == 4


def extract_drum_bars_4_4_16th(path: Path) -> List[DrumBar]:
    """
    Extract 4/4 bars quantized to 16th steps from a drum track.
    """
    pm = pretty_midi.PrettyMIDI(str(path))
    if not _is_4_4(pm):
        return []

    tempo = _get_constant_tempo(pm)
    spb = 60.0 / tempo
    bar_len = 4.0 * spb
    step_len = bar_len / 16.0

    drum_inst = next((inst for inst in pm.instruments if inst.is_drum), None)
    if drum_inst is None or not drum_inst.notes:
        return []

    last_end = max(n.end for n in drum_inst.notes)
    n_bars = int(last_end // bar_len)
    if n_bars <= 0:
        return []

    bars: List[DrumBar] = []
    # Pre-bucket notes by bar index for speed
    notes_by_bar: List[List[pretty_midi.Note]] = [[] for _ in range(n_bars)]
    for n in drum_inst.notes:
        b = int(n.start // bar_len)
        if 0 <= b < n_bars:
            notes_by_bar[b].append(n)

    for b in range(n_bars):
        hits = np.zeros((16, N_VOICES), dtype=np.float32)
        vels = np.zeros((16, N_VOICES), dtype=np.float32)
        bar_start = b * bar_len

        for n in notes_by_bar[b]:
            vidx = VOICE_MAP.get(int(n.pitch))
            if vidx is None:
                continue
            step = int(round((n.start - bar_start) / step_len))
            step = max(0, min(15, step))
            hits[step, vidx] = 1.0
            vels[step, vidx] = max(vels[step, vidx], float(n.velocity) / 127.0)

        # Skip empty bars
        if hits.sum() > 0:
            bars.append(DrumBar(hits=hits, vels=vels, tempo_bpm=tempo))

    return bars
