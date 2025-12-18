from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from backingtrack.ml_drums.data import extract_drum_bars_4_4_16th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi_dir", type=str, required=True, help="Folder containing .mid files (GMD)")
    ap.add_argument("--out", type=str, default="data/ml/drum_bars.npz")
    args = ap.parse_args()

    midi_dir = Path(args.midi_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_hits = []
    all_vels = []

    files = list(midi_dir.rglob("*.mid")) + list(midi_dir.rglob("*.midi"))
    for f in tqdm(files, desc="Extracting bars"):
        try:
            bars = extract_drum_bars_4_4_16th(f)
        except Exception:
            continue
        for b in bars:
            all_hits.append(b.hits)
            all_vels.append(b.vels)

    X = np.stack(all_hits, axis=0)  # (N, 16, V)
    V = np.stack(all_vels, axis=0)  # (N, 16, V)

    np.savez_compressed(out_path, hits=X, vels=V)
    print(f"Saved {len(X)} bars to {out_path}")

if __name__ == "__main__":
    main()
