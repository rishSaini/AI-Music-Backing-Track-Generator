# scripts/preprocess_bass.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pretty_midi

from backingtrack.midi_io import load_midi, extract_midi_info, build_bar_grid, pick_melody_instrument
from backingtrack.melody import MelodyConfig, extract_melody_notes
from backingtrack.types import Note
from backingtrack.key_detect import estimate_key


try:
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(x, **kwargs):  # type: ignore
        return x


IGNORE_INDEX = -100

# Feature vocab for chord qualities used in conditioning
QUAL_VOCAB = ["N", "maj", "min", "7", "maj7", "min7", "dim", "sus2", "sus4"]
QUAL_TO_I: Dict[str, int] = {q: i for i, q in enumerate(QUAL_VOCAB)}

PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE2PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


# -----------------------------
# POP909 chord annotations
# -----------------------------
@dataclass(frozen=True)
class ChordSeg:
    start: float
    end: float
    root_pc: Optional[int]  # None for N
    quality: str            # one of QUAL_VOCAB
    known: bool             # if we could map it


def _parse_root_pc(s: str) -> Optional[int]:
    s = s.strip()
    if not s or s.upper() == "N":
        return None
    letter = s[0].upper()
    if letter not in NOTE2PC:
        return None
    pc = NOTE2PC[letter]
    if len(s) >= 2:
        if s[1] == "#":
            pc = (pc + 1) % 12
        elif s[1].lower() == "b":
            pc = (pc - 1) % 12
    return int(pc)


def _normalize_quality(q: str) -> Optional[str]:
    q = (q or "").strip().lower()
    if q in ("", "maj", "major"):
        return "maj"
    if q in ("min", "minor", "m"):
        return "min"

    # common pop spellings
    if q in ("7", "dom7", "dominant7", "9", "11", "13"):
        return "7"
    if q in ("maj7", "major7", "maj9", "maj11", "maj13"):
        return "maj7"
    if q in ("min7", "m7", "minor7", "min9", "min11", "min13"):
        return "min7"
    if "sus2" in q:
        return "sus2"
    if "sus4" in q or q == "sus":
        return "sus4"
    if "dim" in q or "hdim" in q:
        return "dim"

    return None


def load_pop909_chords(song_dir: Path) -> List[ChordSeg]:
    cand = [song_dir / "chord_midi.txt", song_dir / "chord_audio.txt"]
    path = next((p for p in cand if p.exists()), None)
    if path is None:
        return []

    segs: List[ChordSeg] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            s = float(parts[0]); e = float(parts[1])
        except ValueError:
            continue
        if e <= s:
            continue
        name = parts[2].strip()

        if name.upper() == "N":
            segs.append(ChordSeg(s, e, None, "N", True))
            continue

        if ":" in name:
            root_s, qual_s = name.split(":", 1)
        else:
            root_s, qual_s = name, "maj"

        root_pc = _parse_root_pc(root_s)
        qual = _normalize_quality(qual_s)

        known = (root_pc is not None) and (qual is not None) and (qual in QUAL_TO_I)
        if not known:
            segs.append(ChordSeg(s, e, root_pc, qual or "N", False))
        else:
            segs.append(ChordSeg(s, e, int(root_pc), str(qual), True))

    segs.sort(key=lambda x: x.start)
    return segs


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def chord_at_time(segs: List[ChordSeg], t: float) -> ChordSeg:
    for s in segs:
        if s.start <= t < s.end:
            return s
    return ChordSeg(t, t + 1e-3, None, "N", True)


def chord_pitch_classes(root_pc: Optional[int], quality: str) -> Tuple[int, ...]:
    if root_pc is None or quality == "N":
        return tuple()

    q = quality
    if q not in QUAL_TO_I:
        q = "maj"

    if q == "maj":
        ivs = (0, 4, 7)
    elif q == "min":
        ivs = (0, 3, 7)
    elif q == "dim":
        ivs = (0, 3, 6)
    elif q == "sus2":
        ivs = (0, 2, 7)
    elif q == "sus4":
        ivs = (0, 5, 7)
    elif q == "7":
        ivs = (0, 4, 7, 10)
    elif q == "maj7":
        ivs = (0, 4, 7, 11)
    elif q == "min7":
        ivs = (0, 3, 7, 10)
    else:
        ivs = (0, 4, 7)

    pcs = [int((root_pc + iv) % 12) for iv in ivs]
    out: List[int] = []
    for pc in pcs:
        if pc not in out:
            out.append(pc)
    return tuple(out)


def chord_third_fifth_seventh(root_pc: int, quality: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    quality = quality if quality in QUAL_TO_I else "maj"
    if quality in ("maj", "7", "maj7", "sus2", "sus4"):
        third = (root_pc + 4) % 12
    elif quality in ("min", "min7", "dim"):
        third = (root_pc + 3) % 12
    else:
        third = (root_pc + 4) % 12

    if quality == "dim":
        fifth = (root_pc + 6) % 12
    else:
        fifth = (root_pc + 7) % 12

    seventh = None
    if quality == "7" or quality == "min7":
        seventh = (root_pc + 10) % 12
    elif quality == "maj7":
        seventh = (root_pc + 11) % 12

    return third, fifth, seventh


# -----------------------------
# Track picking: bass instrument (tighter + less false positives)
# -----------------------------
def _polyphony_ratio(notes: List[Note]) -> float:
    if len(notes) < 2:
        return 0.0
    events: List[Tuple[float, int]] = []
    for n in notes:
        events.append((float(n.start), +1))
        events.append((float(n.end), -1))
    events.sort(key=lambda x: (x[0], -x[1]))

    active = 0
    last_t = events[0][0]
    poly_time = 0.0
    total_time = 0.0

    for t, delta in events:
        dt = float(t - last_t)
        if dt > 0:
            total_time += dt
            if active >= 2:
                poly_time += dt
        active += int(delta)
        last_t = float(t)

    if total_time <= 1e-9:
        return 0.0
    return float(poly_time / total_time)


def _low_range_frac(notes: List[Note], lo: int = 36, hi: int = 60) -> float:
    if not notes:
        return 0.0
    pitches = np.array([n.pitch for n in notes], dtype=np.int32)
    return float(np.mean((pitches >= lo) & (pitches <= hi)))


def _median_pitch(notes: List[Note]) -> float:
    if not notes:
        return 127.0
    pitches = np.array([n.pitch for n in notes], dtype=np.float32)
    return float(np.median(pitches))


def _make_monophonic_lowest(notes: List[Note]) -> List[Note]:
    """
    Robust monophonic bass enforcement:
    For every time interval, keep ONLY the lowest-pitch active note.
    Implemented with a sweep-line + heap (O(N log N)), avoids invalid Note segments.
    """
    import heapq
    import math

    if not notes:
        return []

    # Filter junk / invalid
    cleaned: List[Note] = []
    for n in notes:
        s = float(n.start)
        e = float(n.end)
        if not (math.isfinite(s) and math.isfinite(e)):
            continue
        if e <= s:
            continue
        cleaned.append(Note(int(n.pitch), s, e, int(n.velocity)))

    if not cleaned:
        return []

    # Events: (time, kind, id, note)
    # kind: 0=end, 1=start  (end processed before start at same time)
    events: List[tuple] = []
    for i, n in enumerate(cleaned):
        events.append((float(n.start), 1, i, n))
        events.append((float(n.end), 0, i, n))
    events.sort(key=lambda x: (x[0], x[1]))

    active: dict[int, Note] = {}
    heap: List[tuple] = []  # (pitch, id)

    def _prune(t: float) -> None:
        # Remove heap tops that are no longer active at time t
        while heap:
            pitch, idx = heap[0]
            n = active.get(idx)
            if n is None:
                heapq.heappop(heap)
                continue
            # If note ended at or before current time, it should not be active
            if float(n.end) <= t:
                active.pop(idx, None)
                heapq.heappop(heap)
                continue
            # valid
            break

    out: List[Note] = []
    eps = 1e-9

    prev_t = float(events[0][0])
    k = 0
    while k < len(events):
        t = float(events[k][0])

        # Emit segment for [prev_t, t) using currently active lowest note
        _prune(prev_t)
        if t > prev_t + eps and heap:
            pitch, idx = heap[0]
            n = active.get(idx)
            if n is not None:
                seg_start = prev_t
                seg_end = t
                if seg_end > seg_start + eps:
                    out.append(Note(int(n.pitch), float(seg_start), float(seg_end), int(n.velocity)))

        # Process all events at time t (end before start due to sorting)
        while k < len(events) and float(events[k][0]) == t:
            _time, kind, idx, n = events[k]
            if kind == 0:
                active.pop(idx, None)
            else:
                # start
                active[idx] = n
                heapq.heappush(heap, (int(n.pitch), idx))
            k += 1

        prev_t = t

    # Merge adjacent segments with same pitch & velocity
    if not out:
        return []

    merged: List[Note] = [out[0]]
    for seg in out[1:]:
        last = merged[-1]
        if (
            seg.pitch == last.pitch
            and seg.velocity == last.velocity
            and abs(float(seg.start) - float(last.end)) < 1e-6
        ):
            merged[-1] = Note(last.pitch, float(last.start), float(seg.end), last.velocity)
        else:
            merged.append(seg)

    return merged


def pick_bass_instrument(pm: pretty_midi.PrettyMIDI) -> Optional[pretty_midi.Instrument]:
    cands: List[pretty_midi.Instrument] = [i for i in pm.instruments if (not i.is_drum) and i.notes]
    if not cands:
        return None

    song_end = float(pm.get_end_time()) or 1.0

    # Precompute a rough "lowest track" bonus
    med_by_inst: Dict[int, float] = {}
    for idx, inst in enumerate(cands):
        pitches = np.array([n.pitch for n in inst.notes], dtype=np.float32)
        med_by_inst[idx] = float(np.median(pitches)) if len(pitches) else 127.0
    lowest_med = min(med_by_inst.values()) if med_by_inst else 127.0

    def score(idx_inst: Tuple[int, pretty_midi.Instrument]) -> float:
        idx, inst = idx_inst
        notes = [Note(int(n.pitch), float(n.start), float(n.end), int(n.velocity)) for n in inst.notes if n.end > n.start]
        notes.sort(key=lambda x: (x.start, x.pitch))
        notes_mono = _make_monophonic_lowest(notes)

        pitches = np.array([n.pitch for n in notes_mono], dtype=np.float32) if notes_mono else np.array([127.0], dtype=np.float32)
        med = float(np.median(pitches))
        p10 = float(np.percentile(pitches, 10))

        span = float(max(n.end for n in notes_mono) - min(n.start for n in notes_mono)) if notes_mono else 0.0
        coverage = span / max(1e-6, song_end)

        poly = _polyphony_ratio(notes)  # measure original polyphony, not mono-ed
        low_frac = _low_range_frac(notes_mono, 36, 60)

        name = (inst.name or "").lower()
        program = int(getattr(inst, "program", 0))

        bonus = 0.0
        if "bass" in name:
            bonus += 80.0
        if "melody" in name or "lead" in name or "vocal" in name:
            bonus -= 80.0

        # GM bass programs usually 32-39
        if 32 <= program <= 39:
            bonus += 60.0

        # lowest-median track bonus
        if abs(med_by_inst.get(idx, 127.0) - lowest_med) < 1e-6:
            bonus += 30.0

        # penalties
        bonus -= 200.0 * max(0.0, poly - 0.10)          # heavy penalty if polyphonic
        bonus += 120.0 * np.clip(low_frac, 0.0, 1.0)    # prefer lots of notes in 36-60
        bonus -= 3.0 * max(0.0, med - 55.0)             # gently push med pitch down

        note_count = len(notes_mono)

        # Lower pitches + decent coverage wins
        return (-2.0 * med) + (-0.5 * p10) + (80.0 * coverage) + (6.0 * np.log1p(note_count)) + bonus

    scored = [(idx, inst) for idx, inst in enumerate(cands)]
    scored.sort(key=score, reverse=True)
    return scored[0][1] if scored else None


def instrument_notes(inst: pretty_midi.Instrument) -> List[Note]:
    out: List[Note] = []
    for n in inst.notes:
        if n.end <= n.start:
            continue
        out.append(Note(int(n.pitch), float(n.start), float(n.end), int(n.velocity)))
    out.sort(key=lambda x: (x.start, x.pitch))
    return out


# -----------------------------
# Feature extraction
# -----------------------------
def key_features(melody: List[Note]) -> np.ndarray:
    k = estimate_key(melody)
    tonic = int(k.tonic_pc) % 12
    tonic_oh = np.zeros(12, dtype=np.float32)
    tonic_oh[tonic] = 1.0
    mode_oh = np.zeros(2, dtype=np.float32)
    mode_oh[0 if k.mode == "major" else 1] = 1.0
    return np.concatenate([tonic_oh, mode_oh], axis=0)


def melody_step_features(melody: List[Note], t0: float, t1: float, step_len: float) -> np.ndarray:
    hist = np.zeros(12, dtype=np.float32)
    tot = 0.0
    pitch_num = 0.0

    for n in melody:
        ov = _overlap(n.start, n.end, t0, t1)
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

    return np.concatenate([hist, np.array([mean_pitch_norm, activity], dtype=np.float32)], axis=0).astype(np.float32)


def chord_step_features(ch: ChordSeg) -> np.ndarray:
    root_oh = np.zeros(12, dtype=np.float32)
    if ch.root_pc is not None:
        root_oh[int(ch.root_pc) % 12] = 1.0

    qual_oh = np.zeros(len(QUAL_VOCAB), dtype=np.float32)
    q = ch.quality if ch.quality in QUAL_TO_I else "N"
    qual_oh[QUAL_TO_I[q]] = 1.0

    return np.concatenate([root_oh, qual_oh], axis=0).astype(np.float32)


def step_in_bar_onehot(t0: float, grid_spb: float, step_beats: float) -> np.ndarray:
    bins = 2 if step_beats >= 2.0 else 4
    oh = np.zeros(bins, dtype=np.float32)
    beats = (t0 / max(1e-9, grid_spb)) % 4.0
    idx = int(np.floor(beats / (4.0 / bins))) % bins
    oh[idx] = 1.0
    return oh


# -----------------------------
# Labels: degree / register / rhythm
# -----------------------------
DEG_VOCAB = ["REST", "ROOT", "THIRD", "FIFTH", "SEVENTH", "CHORD_TONE", "NONCHORD"]
DEG_TO_I = {s: i for i, s in enumerate(DEG_VOCAB)}

RHY_VOCAB = ["REST", "SUSTAIN", "HIT_ON", "HIT_OFF", "MULTI"]
RHY_TO_I = {s: i for i, s in enumerate(RHY_VOCAB)}

REG_VOCAB = ["LOW", "MID", "HIGH"]


def bass_step_notes(bass: List[Note], t0: float, t1: float) -> Tuple[List[Note], List[Note]]:
    overlapping: List[Note] = []
    onsets: List[Note] = []
    for n in bass:
        if n.end <= t0 or n.start >= t1:
            continue
        overlapping.append(n)
        if t0 <= n.start < t1:
            onsets.append(n)
    onsets.sort(key=lambda x: x.start)
    overlapping.sort(key=lambda x: (x.start, -(x.end - x.start), x.pitch))
    return overlapping, onsets


def pick_representative_bass_note(overlapping: List[Note], onsets: List[Note]) -> Optional[Note]:
    # enforce monophonic representative: lowest onset, else lowest overlap
    if onsets:
        return min(onsets, key=lambda n: (n.start, n.pitch))
    if overlapping:
        return min(overlapping, key=lambda n: n.pitch)
    return None


def classify_register(pitch: int) -> int:
    if pitch < 44:
        return 0
    if pitch < 56:
        return 1
    return 2


def classify_rhythm(t0: float, t1: float, step_len: float, overlapping: List[Note], onsets: List[Note]) -> int:
    if not overlapping:
        return RHY_TO_I["REST"]
    if len(onsets) >= 2:
        return RHY_TO_I["MULTI"]
    if len(onsets) == 0:
        return RHY_TO_I["SUSTAIN"]
    onset_t = onsets[0].start
    if onset_t < (t0 + 0.5 * step_len):
        return RHY_TO_I["HIT_ON"]
    return RHY_TO_I["HIT_OFF"]


def classify_degree(rep: Note, chord: ChordSeg) -> int:
    if chord.root_pc is None or chord.quality == "N":
        return DEG_TO_I["NONCHORD"]

    root = int(chord.root_pc) % 12
    pc = int(rep.pitch) % 12

    third, fifth, seventh = chord_third_fifth_seventh(root, chord.quality)
    pcs = set(chord_pitch_classes(root, chord.quality))

    if pc == root:
        return DEG_TO_I["ROOT"]
    if third is not None and pc == third:
        return DEG_TO_I["THIRD"]
    if fifth is not None and pc == fifth:
        return DEG_TO_I["FIFTH"]
    if seventh is not None and pc == seventh:
        return DEG_TO_I["SEVENTH"]
    if pc in pcs:
        return DEG_TO_I["CHORD_TONE"]
    return DEG_TO_I["NONCHORD"]


def pack_token(deg: int, reg: int, rhy: int, n_deg: int, n_reg: int, n_rhy: int) -> int:
    if not (0 <= deg < n_deg and 0 <= reg < n_reg and 0 <= rhy < n_rhy):
        return IGNORE_INDEX
    return int((deg * n_reg + reg) * n_rhy + rhy)


# -----------------------------
# Main preprocessing
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop909_root", type=str, required=True, help="Path to POP909 root containing 001/002/... folders")
    ap.add_argument("--out", type=str, default="data/ml/bass_steps.npz")
    ap.add_argument("--only_4_4", action="store_true")
    ap.add_argument("--include_key", action="store_true")
    ap.add_argument("--seq_len", type=int, default=128, help="Sequence length in STEPS")
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--step_beats", type=float, default=2.0, help="2.0 => half-bar in 4/4")

    # NEW: filter windows with too many unknown chord labels
    ap.add_argument("--min_label_frac", type=float, default=0.85, help="Skip windows with label_mask fraction below this")

    args = ap.parse_args()

    root = Path(args.pop909_root)
    if not root.exists():
        raise FileNotFoundError(f"POP909 root not found: {root}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    melody_cfg = MelodyConfig(quantize_to_beat=False)

    X_windows: List[np.ndarray] = []
    y_deg_windows: List[np.ndarray] = []
    y_reg_windows: List[np.ndarray] = []
    y_rhy_windows: List[np.ndarray] = []
    y_tok_windows: List[np.ndarray] = []
    attn_windows: List[np.ndarray] = []
    label_windows: List[np.ndarray] = []

    song_dirs = [p for p in root.iterdir() if p.is_dir()]
    song_dirs.sort(key=lambda p: p.name)

    n_degree = len(DEG_VOCAB)
    n_register = len(REG_VOCAB)
    n_rhythm = len(RHY_VOCAB)
    vocab_size = n_degree * n_register * n_rhythm

    for sd in tqdm(song_dirs, desc="POP909 songs"):
        midi_path = sd / f"{sd.name}.mid"
        if not midi_path.exists():
            continue

        segs = load_pop909_chords(sd)
        if not segs:
            continue

        try:
            pm = load_midi(midi_path)
            info = extract_midi_info(pm)
        except Exception:
            continue

        if args.only_4_4:
            if not (info.time_signature.numerator == 4 and info.time_signature.denominator == 4):
                continue

        grid = build_bar_grid(info)
        step_len = float(args.step_beats) * float(grid.seconds_per_beat)
        if step_len <= 0:
            continue

        # Melody instrument: prefer a track named melody
        mel_inst = None
        for inst in pm.instruments:
            if inst.is_drum:
                continue
            if "melody" in (inst.name or "").lower():
                mel_inst = inst
                break
        if mel_inst is None:
            mel_inst, _ = pick_melody_instrument(pm)

        melody = extract_melody_notes(mel_inst, grid=None, config=melody_cfg)
        if not melody:
            continue

        bass_inst = pick_bass_instrument(pm)
        if bass_inst is None:
            continue
        bass_notes = instrument_notes(bass_inst)
        if not bass_notes:
            continue
        bass_notes = _make_monophonic_lowest(bass_notes)
        if not bass_notes:
            continue

        # extra sanity: require the selected bass to be plausibly low
        if _median_pitch(bass_notes) > 65 and _low_range_frac(bass_notes, 36, 60) < 0.35:
            continue

        kfeat = key_features(melody).astype(np.float32) if args.include_key else None

        dur = float(info.duration)
        n_steps = int(np.ceil(max(1e-6, dur) / step_len))

        feats: List[np.ndarray] = []
        y_deg: List[int] = []
        y_reg: List[int] = []
        y_rhy: List[int] = []
        y_tok: List[int] = []
        label_mask: List[bool] = []

        for i in range(n_steps):
            t0 = i * step_len
            t1 = t0 + step_len

            ch = chord_at_time(segs, t0 + 1e-4)
            ch_feat = chord_step_features(ch)
            mel_feat = melody_step_features(melody, t0, t1, step_len)
            pos_feat = step_in_bar_onehot(t0, grid.seconds_per_beat, float(args.step_beats))

            parts = [mel_feat, ch_feat, pos_feat]
            if kfeat is not None:
                parts.insert(1, kfeat)

            x = np.concatenate(parts, axis=0).astype(np.float32)
            feats.append(x)

            overlapping, onsets = bass_step_notes(bass_notes, t0, t1)

            ok = bool(ch.known)
            label_mask.append(ok)

            if not ok:
                y_deg.append(IGNORE_INDEX)
                y_reg.append(IGNORE_INDEX)
                y_rhy.append(IGNORE_INDEX)
                y_tok.append(IGNORE_INDEX)
                continue

            if not overlapping:
                deg = DEG_TO_I["REST"]
                reg = 0
                rhy = RHY_TO_I["REST"]
                y_deg.append(deg)
                y_reg.append(reg)
                y_rhy.append(rhy)
                y_tok.append(pack_token(deg, reg, rhy, n_degree, n_register, n_rhythm))
                continue

            rep = pick_representative_bass_note(overlapping, onsets)
            assert rep is not None

            deg = classify_degree(rep, ch)
            reg = classify_register(int(rep.pitch))
            rhy = classify_rhythm(t0, t1, step_len, overlapping, onsets)

            y_deg.append(deg)
            y_reg.append(reg)
            y_rhy.append(rhy)
            y_tok.append(pack_token(deg, reg, rhy, n_degree, n_register, n_rhythm))

        X = np.stack(feats, axis=0).astype(np.float32)          # (T,F)
        yd = np.array(y_deg, dtype=np.int64)                    # (T,)
        yr = np.array(y_reg, dtype=np.int64)
        yh = np.array(y_rhy, dtype=np.int64)
        yt = np.array(y_tok, dtype=np.int64)
        lm = np.array(label_mask, dtype=np.bool_)

        seq_len = int(args.seq_len)
        stride = int(args.stride)

        for start in range(0, n_steps, stride):
            end = start + seq_len
            L = min(end, n_steps) - start
            if L <= 0:
                continue

            sl = slice(start, start + L)

            xw = np.zeros((seq_len, X.shape[1]), dtype=np.float32)
            ydw = np.full((seq_len,), IGNORE_INDEX, dtype=np.int64)
            yrw = np.full((seq_len,), IGNORE_INDEX, dtype=np.int64)
            yhw = np.full((seq_len,), IGNORE_INDEX, dtype=np.int64)
            ytw = np.full((seq_len,), IGNORE_INDEX, dtype=np.int64)
            attn = np.zeros((seq_len,), dtype=np.bool_)
            lmw = np.zeros((seq_len,), dtype=np.bool_)

            xw[:L] = X[sl]
            ydw[:L] = yd[sl]
            yrw[:L] = yr[sl]
            yhw[:L] = yh[sl]
            ytw[:L] = yt[sl]
            attn[:L] = True
            lmw[:L] = lm[sl]

            # skip windows with too many unknown chord labels
            frac = float(lmw[:L].mean()) if L > 0 else 0.0
            if frac < float(args.min_label_frac):
                continue

            X_windows.append(xw)
            y_deg_windows.append(ydw)
            y_reg_windows.append(yrw)
            y_rhy_windows.append(yhw)
            y_tok_windows.append(ytw)
            attn_windows.append(attn)
            label_windows.append(lmw)

    if not X_windows:
        raise RuntimeError("No windows created. Check POP909 path and that chord_midi.txt exists in song folders.")

    X_out = np.stack(X_windows, axis=0)
    y_deg_out = np.stack(y_deg_windows, axis=0)
    y_reg_out = np.stack(y_reg_windows, axis=0)
    y_rhy_out = np.stack(y_rhy_windows, axis=0)
    y_tok_out = np.stack(y_tok_windows, axis=0)
    attn_out = np.stack(attn_windows, axis=0)
    label_out = np.stack(label_windows, axis=0)

    np.savez_compressed(
        out_path,
        X=X_out,
        y_degree=y_deg_out,
        y_register=y_reg_out,
        y_rhythm=y_rhy_out,
        y_token=y_tok_out,              # NEW: combined token for autoregressive LM
        attn_mask=attn_out,
        label_mask=label_out,
        feat_dim=np.array([X_out.shape[-1]], dtype=np.int64),
        seq_len=np.array([X_out.shape[1]], dtype=np.int64),
        step_beats=np.array([float(args.step_beats)], dtype=np.float32),
        include_key=np.array([bool(args.include_key)], dtype=np.bool_),
        degree_vocab=np.array(DEG_VOCAB, dtype=object),
        rhythm_vocab=np.array(RHY_VOCAB, dtype=object),
        register_vocab=np.array(REG_VOCAB, dtype=object),
        qual_vocab=np.array(QUAL_VOCAB, dtype=object),
        n_degree=np.array([len(DEG_VOCAB)], dtype=np.int64),
        n_rhythm=np.array([len(RHY_VOCAB)], dtype=np.int64),
        n_register=np.array([len(REG_VOCAB)], dtype=np.int64),
        vocab_size=np.array([vocab_size], dtype=np.int64),  # NEW
    )

    print(f"Saved: {out_path}")
    print(f"Windows: {X_out.shape[0]} | seq_len={X_out.shape[1]} | feat_dim={X_out.shape[2]}")
    print(f"Classes: degree={len(DEG_VOCAB)} register={len(REG_VOCAB)} rhythm={len(RHY_VOCAB)}")
    print(f"LM vocab_size={vocab_size}")
    print(f"step_beats={float(args.step_beats)} include_key={bool(args.include_key)} min_label_frac={float(args.min_label_frac)}")


if __name__ == "__main__":
    main()
