from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pretty_midi
from tqdm import tqdm

from backingtrack.key_detect import estimate_key
from backingtrack.melody import MelodyConfig, extract_melody_notes
from backingtrack.midi_io import build_bar_grid, extract_midi_info, load_midi, pick_melody_instrument
from backingtrack.types import Note

IGNORE_INDEX = -100  # convenient for torch CrossEntropyLoss(ignore_index=-100)

# ----------------------------
# Chord vocabulary (v1)
# ----------------------------
# 24 classes: root 0..11 x quality {maj, min}
QUALITY_TO_IDX = {"maj": 0, "min": 1}
IDX_TO_QUALITY = {0: "maj", 1: "min"}

PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def chord_id(root_pc: int, quality: str) -> int:
    return int(root_pc % 12) + 12 * int(QUALITY_TO_IDX[quality])


def chord_name(cid: int) -> str:
    root = int(cid % 12)
    q = IDX_TO_QUALITY[int(cid // 12)]
    return f"{PC_NAMES[root]}:{q}"


def triad_pcs(root_pc: int, quality: str) -> Tuple[int, int, int]:
    r = int(root_pc) % 12
    if quality == "maj":
        return (r, (r + 4) % 12, (r + 7) % 12)
    if quality == "min":
        return (r, (r + 3) % 12, (r + 7) % 12)
    raise ValueError(f"Unknown quality: {quality}")


# ----------------------------
# Helpers
# ----------------------------
def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Overlap length between [a0,a1] and [b0,b1]."""
    lo = max(a0, b0)
    hi = min(a1, b1)
    return max(0.0, hi - lo)


def _is_constant_4_4(pm: pretty_midi.PrettyMIDI) -> bool:
    # allow missing ts changes -> treat as 4/4
    if not pm.time_signature_changes:
        return True
    ts0 = min(pm.time_signature_changes, key=lambda ts: ts.time)
    return int(ts0.numerator) == 4 and int(ts0.denominator) == 4


def _pm_note_to_note(n: pretty_midi.Note) -> Note:
    return Note(pitch=int(n.pitch), start=float(n.start), end=float(n.end), velocity=int(n.velocity))


def _bucket_notes_by_bar(
    notes: List[pretty_midi.Note],
    *,
    bar_len: float,
    n_bars: int,
) -> List[List[pretty_midi.Note]]:
    """
    Put each note into every bar it overlaps (supports sustained notes crossing bars).
    """
    buckets: List[List[pretty_midi.Note]] = [[] for _ in range(n_bars)]
    eps = 1e-6
    for n in notes:
        if n.end <= n.start:
            continue
        b0 = int(max(0.0, n.start) // bar_len)
        b1 = int(max(0.0, (n.end - eps)) // bar_len)
        if b0 >= n_bars:
            continue
        b1 = min(b1, n_bars - 1)
        for b in range(max(0, b0), b1 + 1):
            buckets[b].append(n)
    return buckets


def _pc_hist_for_bar(
    bar_notes: List[pretty_midi.Note],
    *,
    bar_start: float,
    bar_end: float,
    velocity_weight: bool,
) -> np.ndarray:
    """
    12-dim pitch class histogram weighted by overlap duration (and optional velocity).
    """
    hist = np.zeros(12, dtype=np.float32)
    for n in bar_notes:
        ov = _overlap(float(n.start), float(n.end), bar_start, bar_end)
        if ov <= 0.0:
            continue
        w = float(ov)
        if velocity_weight:
            w *= (0.5 + 0.5 * (float(n.velocity) / 127.0))
        hist[int(n.pitch) % 12] += w
    return hist


def _melody_features_for_bar(
    mel_bar_notes: List[pretty_midi.Note],
    *,
    bar_start: float,
    bar_end: float,
    bar_len: float,
) -> Tuple[np.ndarray, float, float]:
    """
    Returns:
      - melody_pc_hist (12) normalized
      - mean_pitch_norm in [0,1] (duration-weighted)
      - activity_ratio in [0,1] = total overlap duration / bar_len
    """
    hist = np.zeros(12, dtype=np.float32)
    tot = 0.0
    pitch_num = 0.0

    for n in mel_bar_notes:
        ov = _overlap(float(n.start), float(n.end), bar_start, bar_end)
        if ov <= 0.0:
            continue
        tot += ov
        hist[int(n.pitch) % 12] += float(ov)
        pitch_num += float(ov) * float(n.pitch)

    if hist.sum() > 1e-9:
        hist = hist / (hist.sum() + 1e-9)

    mean_pitch = (pitch_num / max(1e-9, tot)) if tot > 0 else 60.0
    mean_pitch_norm = float(np.clip(mean_pitch / 127.0, 0.0, 1.0))
    activity_ratio = float(np.clip(tot / max(1e-9, bar_len), 0.0, 1.0))

    return hist, mean_pitch_norm, activity_ratio


def _best_triad_label(harm_hist: np.ndarray, *, off_penalty: float) -> Tuple[int, float]:
    """
    Pick best (maj/min) triad label from harmony pitch-class histogram.
    Returns (chord_id, confidence_gap).
    """
    total = float(harm_hist.sum())
    if total <= 1e-9:
        return chord_id(0, "maj"), 0.0

    best = (-1e18, 0)    # (score, chord_id)
    second = (-1e18, 0)

    for root in range(12):
        for quality in ("maj", "min"):
            pcs = triad_pcs(root, quality)
            on = float(harm_hist[pcs[0]] + harm_hist[pcs[1]] + harm_hist[pcs[2]])
            off = float(total - on)
            score = on - float(off_penalty) * off

            cid = chord_id(root, quality)
            if score > best[0]:
                second = best
                best = (score, cid)
            elif score > second[0]:
                second = (score, cid)

    gap = float(best[0] - second[0])
    return int(best[1]), gap


def _key_features(melody_notes: List[Note]) -> np.ndarray:
    """
    14 dims: tonic onehot(12) + mode onehot(2)
    """
    k = estimate_key(melody_notes)
    tonic = int(k.tonic_pc) % 12
    tonic_oh = np.zeros(12, dtype=np.float32)
    tonic_oh[tonic] = 1.0
    mode_oh = np.zeros(2, dtype=np.float32)
    mode_oh[0 if k.mode == "major" else 1] = 1.0
    return np.concatenate([tonic_oh, mode_oh], axis=0)


@dataclass
class SongBars:
    X: np.ndarray          # (B, F)
    y: np.ndarray          # (B,)
    attn_mask: np.ndarray  # (B,) 1 for real bars
    label_mask: np.ndarray # (B,) 1 where chord label is considered reliable
    chords_debug: List[str]


def extract_song_bars(
    midi_path: Path,
    *,
    only_4_4: bool,
    melody_index: Optional[int],
    min_harmony_weight: float,
    off_penalty: float,
    velocity_weight: bool,
    include_key: bool,
    melody_cfg: MelodyConfig,
) -> Optional[SongBars]:
    pm = load_midi(midi_path)
    if only_4_4 and (not _is_constant_4_4(pm)):
        return None

    info = extract_midi_info(pm)
    grid = build_bar_grid(info)
    bar_len = float(grid.bar_duration)

    dur = float(pm.get_end_time())
    if dur <= 0.0:
        return None
    n_bars = int(max(1, grid.bar_index_at(dur - 1e-6) + 1))

    mel_inst, _ = pick_melody_instrument(pm, instrument_index=melody_index)

    # Extract monophonic melody line
    mel_line = extract_melody_notes(mel_inst, grid=None, config=melody_cfg)
    mel_line_pm = [
        pretty_midi.Note(velocity=int(n.velocity), pitch=int(n.pitch), start=float(n.start), end=float(n.end))
        for n in mel_line
    ]

    # Harmony pool: all non-drum, non-melody instruments
    harm_notes: List[pretty_midi.Note] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        if inst is mel_inst:
            continue
        if inst.notes:
            harm_notes.extend(inst.notes)

    if len(harm_notes) == 0:
        return None

    mel_by_bar = _bucket_notes_by_bar(mel_line_pm, bar_len=bar_len, n_bars=n_bars)
    harm_by_bar = _bucket_notes_by_bar(harm_notes, bar_len=bar_len, n_bars=n_bars)

    key_feat = None
    if include_key:
        key_feat = _key_features([_pm_note_to_note(n) for n in mel_line_pm])

    feats: List[np.ndarray] = []
    labels: List[int] = []
    attn: List[int] = []
    lmask: List[int] = []
    chords_dbg: List[str] = []

    for b in range(n_bars):
        bs = b * bar_len
        be = bs + bar_len

        mel_hist, mean_pitch_norm, activity = _melody_features_for_bar(
            mel_by_bar[b], bar_start=bs, bar_end=be, bar_len=bar_len
        )

        x_parts = [
            mel_hist.astype(np.float32),
            np.array([mean_pitch_norm, activity], dtype=np.float32),
        ]
        if key_feat is not None:
            x_parts.append(key_feat.astype(np.float32))
        x = np.concatenate(x_parts, axis=0).astype(np.float32)

        harm_hist = _pc_hist_for_bar(
            harm_by_bar[b], bar_start=bs, bar_end=be, velocity_weight=velocity_weight
        )
        harm_weight = float(harm_hist.sum())
        cid, gap = _best_triad_label(harm_hist, off_penalty=off_penalty)

        reliable = 1 if harm_weight >= float(min_harmony_weight) else 0

        feats.append(x)
        labels.append(int(cid))
        attn.append(1)
        lmask.append(reliable)
        chords_dbg.append(f"{chord_name(cid)} (harm_w={harm_weight:.2f}, gap={gap:.2f})")

    X = np.stack(feats, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    attn_mask = np.asarray(attn, dtype=np.uint8)
    label_mask = np.asarray(lmask, dtype=np.uint8)

    return SongBars(X=X, y=y, attn_mask=attn_mask, label_mask=label_mask, chords_debug=chords_dbg)


def chunk_song(
    song: SongBars,
    *,
    seq_len: int,
    stride: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[Dict]]:
    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ams: List[np.ndarray] = []
    lms: List[np.ndarray] = []
    metas: List[Dict] = []

    B, F = song.X.shape
    for start in range(0, B, stride):
        end = min(B, start + seq_len)
        n = end - start

        Xc = np.zeros((seq_len, F), dtype=np.float32)
        yc = np.full((seq_len,), IGNORE_INDEX, dtype=np.int64)
        am = np.zeros((seq_len,), dtype=np.uint8)
        lm = np.zeros((seq_len,), dtype=np.uint8)

        Xc[:n] = song.X[start:end]
        yc[:n] = song.y[start:end]
        am[:n] = song.attn_mask[start:end]
        lm[:n] = song.label_mask[start:end]

        Xs.append(Xc)
        ys.append(yc)
        ams.append(am)
        lms.append(lm)
        metas.append({"start_bar": int(start), "n_bars": int(n)})

        if end >= B:
            break

    return Xs, ys, ams, lms, metas


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess MIDIs into chord-sequence training data (v1 triads).")
    ap.add_argument("--midi_dir", type=str, required=True, help="Folder containing .mid/.midi files")
    ap.add_argument("--out", type=str, default="data/ml/chords_seq.npz", help="Output .npz path")

    ap.add_argument("--only_4_4", action="store_true", help="Only keep 4/4 MIDIs (recommended)")
    ap.add_argument("--melody_index", type=int, default=None, help="Force melody instrument index (else auto-pick)")

    ap.add_argument("--seq_len", type=int, default=64, help="Bars per training sequence")
    ap.add_argument("--stride", type=int, default=64, help="Stride between sequences")

    ap.add_argument("--min_harmony_weight", type=float, default=0.35,
                    help="Min harmony evidence per bar (seconds-weighted) to trust chord label")
    ap.add_argument("--off_penalty", type=float, default=0.50,
                    help="Penalty weight for non-chord pitch classes in label scoring")
    ap.add_argument("--velocity_weight", action="store_true", help="Weight histograms by velocity (mildly)")

    ap.add_argument("--include_key", action="store_true",
                    help="Include global key features (tonic+mode onehot) in X")

    ap.add_argument("--max_files", type=int, default=None, help="Limit number of MIDIs processed")
    ap.add_argument("--meta_jsonl", type=str, default=None,
                    help="Optional JSONL metadata output (file + chunk start bar)")
    ap.add_argument("--debug_print", type=int, default=0,
                    help="Print chord labels for the first N processed songs")

    ap.add_argument("--skip_versions", action="store_true",
                    help="Skip POP909 'versions' files to avoid duplicates (recommended)")

    args = ap.parse_args()

    midi_dir = Path(args.midi_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = list(midi_dir.rglob("*.mid")) + list(midi_dir.rglob("*.midi"))
    files.sort()

    if args.skip_versions:
        files = [f for f in files if "versions" not in str(f).lower()]

    if args.max_files is not None:
        files = files[: int(args.max_files)]

    melody_cfg = MelodyConfig(
        min_note_duration=0.05,
        overlap_tolerance=0.02,
        merge_gap=0.03,
        strategy="highest_pitch",
        quantize_to_beat=False,
    )

    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_attn: List[np.ndarray] = []
    all_lmask: List[np.ndarray] = []
    meta_rows: List[Dict] = []

    kept_songs = 0
    skipped = 0

    for f in tqdm(files, desc="Preprocessing harmony"):
        try:
            song = extract_song_bars(
                f,
                only_4_4=bool(args.only_4_4),
                melody_index=args.melody_index,
                min_harmony_weight=float(args.min_harmony_weight),
                off_penalty=float(args.off_penalty),
                velocity_weight=bool(args.velocity_weight),
                include_key=bool(args.include_key),
                melody_cfg=melody_cfg,
            )
            if song is None:
                skipped += 1
                continue

            Xs, ys, ams, lms, metas = chunk_song(song, seq_len=int(args.seq_len), stride=int(args.stride))

            for Xc, yc, am, lm, m in zip(Xs, ys, ams, lms, metas):
                all_X.append(Xc)
                all_y.append(yc)
                all_attn.append(am)
                all_lmask.append(lm)
                meta_rows.append({"file": str(f), **m})

            kept_songs += 1

            if args.debug_print and kept_songs <= int(args.debug_print):
                print(f"\n=== {f} ===")
                for i, s in enumerate(song.chords_debug[:16]):
                    print(f"bar {i:02d}: {s}")

        except Exception:
            skipped += 1
            continue

    if not all_X:
        raise RuntimeError("No training sequences were created. Check your dataset and flags.")

    X = np.stack(all_X, axis=0).astype(np.float32)          # (N, seq_len, F)
    y = np.stack(all_y, axis=0).astype(np.int64)            # (N, seq_len)
    attn_mask = np.stack(all_attn, axis=0).astype(np.uint8) # (N, seq_len)
    label_mask = np.stack(all_lmask, axis=0).astype(np.uint8)# (N, seq_len)

    np.savez_compressed(out_path, X=X, y=y, attn_mask=attn_mask, label_mask=label_mask)

    print(f"\nSaved: {out_path}")
    print(f"Sequences: {len(X)} | seq_len={X.shape[1]} | feat_dim={X.shape[2]}")
    print(f"Songs kept: {kept_songs} | skipped: {skipped}")
    print("y vocab: 24 classes (root 0..11 x {maj,min}), padding uses IGNORE_INDEX=-100")

    if args.meta_jsonl:
        meta_p = Path(args.meta_jsonl)
        meta_p.parent.mkdir(parents=True, exist_ok=True)
        with meta_p.open("w", encoding="utf-8") as w:
            for row in meta_rows:
                w.write(json.dumps(row) + "\n")
        print(f"Metadata JSONL: {meta_p}")


if __name__ == "__main__":
    main()
