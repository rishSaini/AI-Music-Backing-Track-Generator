# ChordCraft - AI Music Backing Track Generator

Generate a full backing track from an input MIDI: **chords + bass + pad + drums**, with both **rule-based** and **ML/AI** options.  
Built with **Python + Streamlit + pretty_midi + PyTorch**.

> **Important:** AI drums currently only support **4/4** time. If the uploaded MIDI is **not 4/4**, the app automatically falls back to **rule-based** drums.

---

## What it does

1. Upload a MIDI file  
2. The app analyzes musical metadata (tempo, time signature, key, etc.)  
3. Choose generation styles (rules vs AI/ML where available)  
4. Generate backing parts:
   - **Chords**
   - **Drums**
   - **Bass**
   - **Pad**
5. Export:
   - **Generated MIDI**
   - **WAV preview** (optional, via FluidSynth + SoundFont)

---

## Features

### Generation
- **Chords**
  - **RULES**: fast + deterministic
  - **AI**: model-based chord prediction (when enabled)
- **Drums**
  - **RULES**: works for any time signature
  - **AI**: groove generation (4/4 only; auto-fallback otherwise)
- **Arrangement**
  - Bass + pad layers aligned to the harmonic structure

### Output
- Download generated MIDI
- Optional WAV rendering for quick listening inside the app

---

## Project structure

```text
.
├─ app.py                      # Streamlit UI entrypoint
├─ cli.py                      # CLI entrypoint (batch / scripting)
├─ requirements.txt            # Python dependencies (recommended for deployment)
├─ packages.txt                # System deps for Streamlit Cloud (e.g., fluidsynth)
├─ pyproject.toml              # Project metadata (setuptools / src layout)
├─ src/
│  └─ backingtrack/            # Core pipeline + generation code
├─ data/
│  └─ ml/
│     └─ chord_model_new.pt    # Default chord model checkpoint (if used)
└─ soundfonts/
   └─ GeneralUser-GS.sf2       # Default SoundFont for WAV rendering
