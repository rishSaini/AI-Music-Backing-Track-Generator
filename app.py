# app.py
from __future__ import annotations

import base64
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import pretty_midi
import streamlit as st
import streamlit.components.v1 as components

from backingtrack.arrange import arrange_backing
from backingtrack.harmony_baseline import generate_chords
from backingtrack.humanize import HumanizeConfig, humanize_arrangement
from backingtrack.key_detect import estimate_key, key_to_string
from backingtrack.melody import MelodyConfig, extract_melody_notes
from backingtrack.midi_io import load_and_prepare
from backingtrack.ml_harmony.steps_infer import ChordSampleConfig, generate_chords_ml_steps
from backingtrack.moods import apply_mood_to_key, get_mood, list_moods
from backingtrack.render import RenderConfig, write_midi

# IMPORTANT: set_page_config must be the first Streamlit call
st.set_page_config(page_title="ChordCraft", page_icon="🎸", layout="centered")

PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def chord_label(root_pc: int, quality: str, extensions: tuple[int, ...]) -> str:
    name = f"{PC_NAMES[root_pc % 12]}{'' if quality == 'maj' else quality}"
    if 10 in extensions:
        name += "7"
    elif 11 in extensions:
        name += "maj7"
    if 14 in extensions:
        name += "add9"
    return name


# ----------------------------
# Mode labels (UI only)
# ----------------------------
def _is_ai_harmony(mode: str) -> bool:
    return str(mode).startswith("ml")


def _is_ai_drums(mode: str) -> bool:
    return str(mode).startswith("ml")


def _harmony_label(mode: str) -> str:
    return "AI-generated" if _is_ai_harmony(mode) else "Rules-based"


def _drums_label(mode: str) -> str:
    return "AI-generated" if _is_ai_drums(mode) else "Rules-based"


# ----------------------------
# Repo path helpers
# ----------------------------
def _find_repo_file(rel_path: str, *, max_parents: int = 6) -> str:
    """
    Try to locate a repo-relative file robustly in local/dev + deployed envs.
    Searches:
      - CWD
      - directory containing this file
      - up to `max_parents` parent directories of this file
    Returns empty string if not found.
    """
    rel = Path(rel_path)

    candidates: list[Path] = []
    try:
        here = Path(__file__).resolve().parent
        candidates.append(Path.cwd())
        candidates.append(here)
        candidates.extend(list(here.parents)[: max(0, int(max_parents))])
    except Exception:
        candidates.append(Path.cwd())

    for base in candidates:
        p = base / rel
        if p.exists():
            try:
                return str(p.resolve())
            except Exception:
                return str(p)
    return ""


# ----------------------------
# Fixed defaults (no UI paths)
# ----------------------------
DEFAULT_CHORD_MODEL_PATH = _find_repo_file("data/ml/chord_model_new.pt") or "data/ml/chord_model_new.pt"


def _guess_soundfont_path() -> str:
    """
    Best-effort default SoundFont path for FluidSynth rendering.
    Override via env var: CHORDCRAFT_SOUNDFONT
    """
    env = os.environ.get("CHORDCRAFT_SOUNDFONT", "").strip()
    if env and Path(env).exists():
        return str(Path(env).resolve())

    candidates = [
        "soundfonts/GeneralUser-GS.sf2",
        "soundfonts/GeneralUser_GS.sf2",
        "data/soundfonts/GeneralUser-GS.sf2",
        "data/soundfonts/GeneralUser_GS.sf2",
        "assets/GeneralUser-GS.sf2",
        "assets/GeneralUser_GS.sf2",
        "GeneralUser-GS.sf2",
        "GeneralUser_GS.sf2",
    ]
    for c in candidates:
        found = _find_repo_file(c)
        if found:
            return found
        if Path(c).exists():
            return str(Path(c).resolve())
    return ""


DEFAULT_SOUNDFONT_PATH = _guess_soundfont_path()


# ----------------------------
# Audio preview: MIDI -> WAV with FluidSynth
# ----------------------------
def _fluidsynth_cmd() -> Optional[str]:
    return shutil.which("fluidsynth")


def render_midi_to_wav_bytes(
    midi_path: Path,
    sf2_path: Path,
    *,
    sample_rate: int = 44100,
    gain: float = 0.8,
) -> bytes:
    """
    Offline render (NO speaker playback) using FluidSynth fast renderer.
    """
    if not sf2_path.exists():
        raise FileNotFoundError(f"SoundFont not found: {sf2_path}")

    cmd = _fluidsynth_cmd()
    if not cmd:
        raise RuntimeError("FluidSynth not found on PATH. Install fluidsynth to enable WAV preview.")

    tmp_wav = Path(tempfile.mkstemp(suffix=".wav")[1])

    # options FIRST, then soundfont, then midifile
    args = [
        cmd,
        "-ni",
        "-T",
        "wav",
        "-F",
        str(tmp_wav),
        "-r",
        str(sample_rate),
        "-g",
        str(gain),
        str(sf2_path),
        str(midi_path),
    ]

    try:
        subprocess.run(args, check=True, capture_output=True, text=True)
        return tmp_wav.read_bytes()
    finally:
        try:
            tmp_wav.unlink(missing_ok=True)
        except Exception:
            pass


def _render_midi_to_wav_bytes(
    midi_bytes: bytes,
    *,
    soundfont_path: str,
    sample_rate: int = 44100,
    gain: float = 0.8,
) -> bytes:
    sf2 = Path(soundfont_path)
    if not sf2.exists():
        raise FileNotFoundError(f"SoundFont not found: {soundfont_path}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        midi_path = td / "preview.mid"
        midi_path.write_bytes(midi_bytes)
        return render_midi_to_wav_bytes(midi_path, sf2, sample_rate=sample_rate, gain=gain)


@st.cache_data(show_spinner=False)
def render_midi_to_wav_cached(
    midi_bytes: bytes,
    soundfont_path: str,
    soundfont_mtime: float,
    sample_rate: int = 44100,
) -> bytes:
    _ = soundfont_mtime  # included in cache key
    return _render_midi_to_wav_bytes(midi_bytes, soundfont_path=soundfont_path, sample_rate=sample_rate)


def wav_player(wav_bytes: bytes) -> None:
    b64 = base64.b64encode(wav_bytes).decode("utf-8")
    html = f"""
    <audio controls preload="none" style="width:100%; height:40px;">
      <source src="data:audio/wav;base64,{b64}" type="audio/wav" />
      Your browser does not support the audio element.
    </audio>
    """
    components.html(html, height=70)


# ----------------------------
# GM Instrument presets
# ----------------------------
PAD_PRESETS = [
    ("Electric Piano 1", 4),
    ("Electric Piano 2", 5),
    ("Acoustic Grand Piano", 0),
    ("Acoustic Guitar (nylon)", 24),
    ("Acoustic Guitar (steel)", 25),
    ("Electric Guitar (clean)", 27),
    ("Strings Ensemble 1", 48),
    ("Choir Aahs", 52),
    ("Brass Section", 61),
    ("Synth Pad 2 (warm)", 89),
    ("Synth Pad 1 (new age)", 88),
]

BASS_PRESETS = [
    ("Electric Bass (finger)", 33),
    ("Electric Bass (pick)", 34),
    ("Acoustic Bass", 32),
    ("Synth Bass 1", 38),
    ("Synth Bass 2", 39),
]


def _preset_index(presets: list[tuple[str, int]], program: int) -> int:
    for i, (_, p) in enumerate(presets):
        if int(p) == int(program):
            return i
    return 0


# ----------------------------
# Auto-pick helpers
# ----------------------------
def _median_pitch(inst: pretty_midi.Instrument) -> float:
    pitches = sorted(n.pitch for n in inst.notes)
    if not pitches:
        return 0.0
    m = len(pitches)
    return float(pitches[m // 2]) if (m % 2 == 1) else 0.5 * (pitches[m // 2 - 1] + pitches[m // 2])


def _auto_pick_with_intro(
    pm: pretty_midi.PrettyMIDI,
    info_or_sel,
    melody_inst: Optional[pretty_midi.Instrument] = None,
    sel=None,
    max_intro: int = 2,
) -> tuple[list[pretty_midi.Instrument], list[int]]:
    """
    Backwards compatible with your earlier app.py signatures.
    Returns:
      (melody_source_insts, picked_intro_idxs)
    """
    info = None

    if sel is None and melody_inst is None and hasattr(info_or_sel, "instrument_index"):
        sel = info_or_sel
    elif sel is None and melody_inst is not None and hasattr(melody_inst, "instrument_index") and not hasattr(
        melody_inst, "notes"
    ):
        info = info_or_sel
        sel = melody_inst
        melody_inst = None
    else:
        info = info_or_sel

    if sel is None:
        raise ValueError("_auto_pick_with_intro: could not determine `sel` (melody selection).")

    base_idxs = list(getattr(sel, "instrument_indices", None) or [int(getattr(sel, "instrument_index", 0))])

    valid_base: list[int] = []
    for i in base_idxs:
        i = int(i)
        if 0 <= i < len(pm.instruments) and (not pm.instruments[i].is_drum) and pm.instruments[i].notes:
            if i not in valid_base:
                valid_base.append(i)

    if not valid_base:
        for i, inst in enumerate(pm.instruments):
            if (not inst.is_drum) and inst.notes:
                valid_base = [i]
                break

    if melody_inst is None:
        melody_inst = pm.instruments[valid_base[0]]

    song_end = float(getattr(info, "duration", 0.0) or 0.0)
    if song_end <= 1e-6:
        song_end = float(pm.get_end_time())

    all_base_pitches: list[int] = []
    for i in valid_base:
        all_base_pitches.extend([int(n.pitch) for n in pm.instruments[i].notes])
    if all_base_pitches:
        all_base_pitches.sort()
        m = len(all_base_pitches)
        main_med = float(all_base_pitches[m // 2]) if (m % 2 == 1) else 0.5 * (
            all_base_pitches[m // 2 - 1] + all_base_pitches[m // 2]
        )
    else:
        main_med = _median_pitch(melody_inst)

    base_set = set(valid_base)
    intro_candidates: list[tuple[int, float, int]] = []

    for idx, inst in enumerate(pm.instruments):
        if idx in base_set:
            continue
        if inst.is_drum or not inst.notes:
            continue

        start = float(min(n.start for n in inst.notes))
        end = float(max(n.end for n in inst.notes))
        span = max(1e-6, end - start)
        coverage = span / max(1e-6, song_end)

        med = _median_pitch(inst)
        note_count = len(inst.notes)

        if (
            start < 2.0
            and end < 0.25 * song_end
            and coverage < 0.25
            and note_count >= 6
            and med > (main_med + 6)
        ):
            intro_candidates.append((idx, med, note_count))

    intro_candidates.sort(key=lambda x: (-x[1], -x[2]))
    picked_intro_idxs = [idx for (idx, _, _) in intro_candidates[: max(0, int(max_intro))]]

    used_indices: list[int] = []
    for i in picked_intro_idxs + valid_base:
        if i not in used_indices:
            used_indices.append(i)

    melody_source_insts = [pm.instruments[i] for i in used_indices]
    return melody_source_insts, picked_intro_idxs


# ----------------------------
# Auto settings
# ----------------------------
@st.cache_data(show_spinner=False)
def recommend_settings(midi_bytes: bytes) -> Dict[str, Any]:
    """
    Heuristic 'auto' settings.
    Project preference:
      - Chords default to Rules-based
      - Drums default to AI-generated
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
            f.write(midi_bytes)
            tmp_path = Path(f.name)

        pm, info, grid, melody_inst, sel = load_and_prepare(tmp_path, melody_instrument_index=None)
        melody_source_insts, _ = _auto_pick_with_intro(pm, info, melody_inst, sel)

        analysis_inst = pretty_midi.Instrument(program=int(melody_source_insts[0].program), is_drum=False, name="Analysis")
        analysis_inst.notes = [n for inst in melody_source_insts for n in inst.notes]
        analysis_inst.notes.sort(key=lambda n: (n.start, n.pitch))

        melody_notes = extract_melody_notes(analysis_inst, grid=grid, config=MelodyConfig(quantize_to_beat=False))

        bpm = float(getattr(info, "tempo_bpm", 120.0) or 120.0)
        dur = float(getattr(info, "duration", 0.0) or 0.0)
        dur = max(dur, 1e-6)
        n_notes = len(melody_notes)
        notes_per_sec = n_notes / dur

        fast = bpm >= 150.0
        slow = bpm <= 85.0
        dense = notes_per_sec >= 3.0 or n_notes >= 450
        sparse = notes_per_sec <= 1.2 and n_notes <= 140

        harmony_mode = "baseline (rules)"
        bars_per_chord = 2 if slow else 1

        chord_step_beats = 4.0 if slow else 2.0
        chord_include_key = True
        chord_stochastic = bool(sparse and not fast)

        chord_temp = 1.0
        chord_top_k = 12
        if sparse:
            chord_temp = 1.15
            chord_top_k = 18
        elif dense:
            chord_temp = 0.95
            chord_top_k = 10

        chord_repeat_penalty = 1.2
        chord_change_penalty = 0.15
        if sparse:
            chord_repeat_penalty = 1.25
            chord_change_penalty = 0.10
        elif dense:
            chord_repeat_penalty = 1.15
            chord_change_penalty = 0.22

        drums_mode = "ml"
        ml_temp = 1.05 if (fast or sparse) else 1.00

        quantize_melody = bool(dense and bpm >= 110.0)

        humanize = True
        jitter_ms = 10.0 if fast else (18.0 if slow else 15.0)
        vel_jitter = 6 if fast else (10 if slow else 8)
        swing = 0.08 if fast else (0.18 if slow else 0.12)

        return {
            "harmony_mode": harmony_mode,
            "bars_per_chord": int(bars_per_chord),
            "chord_model_path": DEFAULT_CHORD_MODEL_PATH,
            "chord_step_beats": float(chord_step_beats),
            "chord_include_key": bool(chord_include_key),
            "chord_stochastic": bool(chord_stochastic),
            "chord_temp": float(chord_temp),
            "chord_top_k": int(chord_top_k),
            "chord_repeat_penalty": float(chord_repeat_penalty),
            "chord_change_penalty": float(chord_change_penalty),
            "quantize_melody": bool(quantize_melody),
            "drums_mode": str(drums_mode),
            "ml_temp": float(ml_temp),
            "humanize": bool(humanize),
            "jitter_ms": float(jitter_ms),
            "vel_jitter": int(vel_jitter),
            "swing": float(swing),
            "auto_sections": True,
        }
    finally:
        if tmp_path:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def _apply_auto_settings(reco: Dict[str, Any], *, file_sig: str, force: bool = False) -> None:
    last_sig = st.session_state.get("_auto_sig")
    if force or (last_sig != file_sig):
        st.session_state["_auto_sig"] = file_sig
        for k, v in reco.items():
            st.session_state[k] = v


# ----------------------------
# UI: defaults + reset
# ----------------------------
st.session_state.setdefault("uploader_key", 0)

DEFAULTS: Dict[str, Any] = {
    "mood_name": "neutral",
    "auto_settings": True,
    "harmony_mode": "baseline (rules)",
    "bars_per_chord": 1,
    "chord_model_path": DEFAULT_CHORD_MODEL_PATH,  # hidden from UI
    "chord_step_beats": 2.0,
    "chord_include_key": True,
    "chord_stochastic": False,
    "chord_temp": 1.0,
    "chord_top_k": 12,
    "chord_repeat_penalty": 1.2,
    "chord_change_penalty": 0.15,
    "quantize_melody": False,
    "drums_mode": "ml",
    "ml_temp": 1.00,
    "humanize": True,
    "jitter_ms": 15.0,
    "vel_jitter": 8,
    "swing": 0.15,
    "use_seed": False,
    "seed_value": 0,
    "auto_sections": True,
    "make_bass": True,
    "make_pad": True,
    "make_drums": True,
    "melody_volume": 1.0,
    "backing_volume": 1.0,
    "pad_program": 4,
    "bass_program": 33,
    "pad_custom_on": False,
    "bass_custom_on": False,
    "auto_render_audio": True,
    "melody_auto_pick": True,
    "melody_tracks_picked": [],
    "_melody_track_indices": None,
    "_melody_pick_sig": None,
    "_uploaded_midi_bytes": None,
    "_uploaded_midi_name": None,
    "_uploaded_midi_sig": None,
    "_generated_midi_bytes": None,
    "_generated_meta": None,
    "_generated_audio_wav": None,
    "_generated_audio_err": None,
    "_generated_audio_sig": None,
    "nav_page": "Controls",
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def _reset_app_state() -> None:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

    # Force the uploader to clear WITHOUT touching its widget key in session_state
    st.session_state["uploader_key"] = int(st.session_state.get("uploader_key", 0)) + 1

    # Also clear auto signature so auto settings re-apply on next upload
    st.session_state.pop("_auto_sig", None)

    st.rerun()


# ----------------------------
# Styling (hero + pills + steps + sticky header/footer)
# ----------------------------
st.markdown(
    """
    <style>
      .hero {
        padding: 14px 16px;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(69, 144, 255, 0.18), rgba(255, 0, 153, 0.10));
        border: 1px solid rgba(49, 51, 63, 0.15);
        margin-bottom: 10px;
        backdrop-filter: blur(10px);
      }
      .hero h1 { margin: 0; font-size: 3.0rem; line-height: 1.05; letter-spacing: 0.2px; }
      .hero p { margin: 8px 0 0 0; opacity: 0.82; font-size: 1.02rem; }
      .muted { opacity: 0.75; }
      .sp { height: 10px; }

      .pills { display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.25rem 0 0.35rem 0; }
      .pill { display: inline-block; padding: 0.20rem 0.60rem; border-radius: 999px; font-size: 0.85rem;
              border: 1px solid rgba(49, 51, 63, 0.20); background: rgba(49, 51, 63, 0.06); }
      .step { font-weight: 850; font-size: 1.05rem; margin-top: 0.65rem; margin-bottom: 0.15rem; }
      code { font-size: 0.95em; }

      /* Sticky hero */
      #cc-hero-wrap {
        position: sticky;
        top: 0.25rem;
        z-index: 999;
        padding-top: 0.25rem;
        padding-bottom: 0.25rem;
        background: rgba(0,0,0,0);
      }

      /* Output panel docked at bottom (ChatGPT-ish) */
      #cc-output-panel {
        position: fixed;
        bottom: 0;
        right: 0;
        left: 0;
        margin-left: 21rem; /* default Streamlit sidebar width when expanded */
        width: calc(100% - 21rem);
        max-height: 42vh;
        overflow: auto;
        padding: 12px 14px 16px 14px;
        border-top: 1px solid rgba(255,255,255,0.10);
        background: rgba(15, 16, 20, 0.92);
        backdrop-filter: blur(10px);
        box-shadow: 0 -10px 30px rgba(0,0,0,0.30);
        z-index: 1000;
      }

      /* Give the main page room so the fixed output doesn't cover content */
      div[data-testid="stAppViewContainer"] .block-container {
        padding-bottom: 46vh;
      }

      @media (max-width: 900px) {
        #cc-output-panel {
          margin-left: 0;
          width: 100%;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _render_settings_pills() -> None:
    chords_label = _harmony_label(st.session_state["harmony_mode"])
    drums_label = _drums_label(st.session_state["drums_mode"])
    structure_label = "Auto sections" if bool(st.session_state.get("auto_sections", True)) else "No structure"

    tracks = []
    if bool(st.session_state.get("make_bass", True)):
        tracks.append("Bass")
    if bool(st.session_state.get("make_pad", True)):
        tracks.append("Chords")
    if bool(st.session_state.get("make_drums", True)):
        tracks.append("Drums")
    tracks_label = " + ".join(tracks) if tracks else "Melody only"

    tags = [
        f"Mood: {st.session_state['mood_name']}",
        f"Chords: {chords_label}",
        f"Drums: {drums_label}",
        f"Structure: {structure_label}",
        f"Tracks: {tracks_label}",
        f"Humanize: {'On' if bool(st.session_state.get('humanize', True)) else 'Off'}",
        f"Quantize: {'On' if bool(st.session_state.get('quantize_melody', False)) else 'Off'}",
    ]
    html = "<div class='pills'>" + "".join([f"<span class='pill'>{t}</span>" for t in tags]) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ----------------------------
# Sidebar: navigation + actions
# ----------------------------
with st.sidebar:
    st.markdown("### Navigation")
    nav = st.radio(
        label="Go to",
        options=["Controls", "Advanced", "Help"],
        key="nav_page",
        label_visibility="collapsed",
        help="Switch between the main control surface, advanced knobs, and detailed help.",
    )

    st.divider()
    st.markdown("### Quick actions")
    if st.button("🔄 Reset to defaults", use_container_width=True, help="Resets all settings and clears the uploaded file/output."):
        _reset_app_state()


# ----------------------------
# Hero (sticky at top)
# ----------------------------
st.markdown(
    """
    <div id="cc-hero-wrap">
      <div class="hero">
        <h1>ChordCraft</h1>
        <p>Upload a MIDI melody → generate bass/chords/drums → download a new multi-track MIDI </p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------
# Pipeline (unchanged logic)
# ----------------------------
def run_pipeline(
    midi_bytes: bytes,
    *,
    mood_name: str,
    harmony_mode: str,
    chord_model_path: str,
    chord_step_beats: float,
    chord_include_key: bool,
    chord_stochastic: bool,
    chord_temp: float,
    chord_top_k: int,
    chord_repeat_penalty: float,
    chord_change_penalty: float,
    bars_per_chord: int,
    quantize_melody: bool,
    make_bass: bool,
    make_pad: bool,
    make_drums: bool,
    melody_track_indices: Optional[list[int]],
    seed: Optional[int],
    structure_mode: str,
    drums_mode: str,
    ml_temp: float,
    humanize: bool,
    jitter_ms: float,
    vel_jitter: int,
    swing: float,
    pad_program: int,
    bass_program: int,
    melody_volume: float,
    backing_volume: float,
) -> tuple[Path, dict]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
        f.write(midi_bytes)
        in_path = Path(f.name)

    try:
        first_idx = melody_track_indices[0] if melody_track_indices else None
        pm, info, grid, melody_inst, sel = load_and_prepare(in_path, melody_instrument_index=first_idx)

        picked_intro_idxs: list[int] = []
        used_melody_indices: list[int] = []

        if melody_track_indices:
            valid: list[int] = []
            for i in melody_track_indices:
                if 0 <= i < len(pm.instruments) and not pm.instruments[i].is_drum:
                    valid.append(i)
            if not valid:
                raise RuntimeError("No valid (non-drum) melody instruments selected.")
            melody_source_insts = [pm.instruments[i] for i in valid]
            used_melody_indices = valid
        else:
            base_idxs = getattr(sel, "instrument_indices", None)
            if not base_idxs:
                base_idxs = [getattr(sel, "instrument_index", 0)]

            base_valid: list[int] = []
            for i in base_idxs:
                try:
                    ii = int(i)
                except Exception:
                    continue
                if 0 <= ii < len(pm.instruments) and not pm.instruments[ii].is_drum:
                    base_valid.append(ii)

            if not base_valid:
                fb = int(getattr(sel, "instrument_index", 0))
                if 0 <= fb < len(pm.instruments) and not pm.instruments[fb].is_drum:
                    base_valid = [fb]

            _, picked_intro_idxs = _auto_pick_with_intro(pm, info, melody_inst, sel)
            picked_intro_idxs = [i for i in picked_intro_idxs if i not in base_valid]

            used_melody_indices = []
            for i in picked_intro_idxs + base_valid:
                if i not in used_melody_indices:
                    used_melody_indices.append(i)

            melody_source_insts = [pm.instruments[i] for i in used_melody_indices]

        analysis_inst = pretty_midi.Instrument(program=int(melody_source_insts[0].program), is_drum=False, name="Analysis")
        analysis_inst.notes = [n for inst in melody_source_insts for n in inst.notes]
        analysis_inst.notes.sort(key=lambda n: (n.start, n.pitch))

        melody_notes = extract_melody_notes(
            analysis_inst,
            grid=grid,
            config=MelodyConfig(quantize_to_beat=quantize_melody),
        )
        if not melody_notes:
            raise RuntimeError("No melody notes extracted. Try selecting a different melody track (or multiple tracks).")

        mood = get_mood(mood_name)
        raw_key = estimate_key(melody_notes)
        key = apply_mood_to_key(raw_key, mood)

        if str(harmony_mode).startswith("ml"):
            model_path = str(chord_model_path or DEFAULT_CHORD_MODEL_PATH)
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Chord model not found: {model_path}")

            chords = generate_chords_ml_steps(
                melody_notes=melody_notes,
                grid=grid,
                duration_seconds=float(info.duration),
                model_path=model_path,
                cfg=ChordSampleConfig(
                    step_beats=float(chord_step_beats),
                    include_key=bool(chord_include_key),
                    stochastic=bool(chord_stochastic),
                    temperature=float(chord_temp),
                    top_k=int(chord_top_k),
                    repeat_penalty=float(chord_repeat_penalty),
                    change_penalty=float(chord_change_penalty),
                    seed=seed,
                ),
            )
        else:
            chords = generate_chords(
                key=key,
                grid=grid,
                duration_seconds=info.duration,
                mood=mood,
                melody_notes=melody_notes,
                bars_per_chord=bars_per_chord,
            )

        arrangement = arrange_backing(
            chords=chords,
            grid=grid,
            mood=mood,
            make_bass=bool(make_bass),
            make_pad=make_pad,
            make_drums=make_drums,
            seed=seed,
            structure_mode=structure_mode,
            drums_mode=drums_mode,
            ml_drums_model_path="data/ml/drum_model.pt",
            ml_drums_temperature=ml_temp,
        )

        if humanize:
            arrangement = humanize_arrangement(
                arrangement,
                grid,
                HumanizeConfig(
                    timing_jitter_ms=jitter_ms,
                    velocity_jitter=vel_jitter,
                    swing=swing,
                    seed=seed,
                ),
            )

        out_path = Path(tempfile.mkstemp(suffix=".mid")[1])

        render_cfg = RenderConfig(
            melody_program=int(melody_source_insts[0].program),
            bass_program=int(bass_program),
            pad_program=int(pad_program),
            melody_vel_scale=float(melody_volume),
            backing_vel_scale=float(backing_volume),
        )

        write_midi(
            out_path,
            [],
            arrangement,
            info,
            config=render_cfg,
            melody_source_insts=melody_source_insts,
        )

        meta = {
            "info": info,
            "selection": sel,
            "selected_melody_indices": melody_track_indices,
            "used_melody_indices": used_melody_indices,
            "auto_intro_indices": picked_intro_idxs,
            "used_melody_track_names": [inst.name or "(unnamed)" for inst in melody_source_insts],
            "melody_note_count": len(melody_notes),
            "raw_key": raw_key,
            "key": key,
            "mood": mood,
            "harmony_mode": harmony_mode,
            "chord_model_path": str(chord_model_path),
            "chord_step_beats": float(chord_step_beats),
            "chord_stochastic": bool(chord_stochastic),
            "chord_temperature": float(chord_temp),
            "chord_top_k": int(chord_top_k),
            "chord_repeat_penalty": float(chord_repeat_penalty),
            "chord_change_penalty": float(chord_change_penalty),
            "arrangement_counts": {k: len(v) for k, v in arrangement.tracks.items()},
            "instrument_list": [
                {"idx": i, "name": (inst.name or f"Instrument {i}"), "is_drum": inst.is_drum, "notes": len(inst.notes)}
                for i, inst in enumerate(pm.instruments)
            ],
            "pad_program": int(pad_program),
            "bass_program": int(bass_program),
            "melody_program": int(melody_source_insts[0].program) if melody_source_insts else None,
            "melody_volume": float(melody_volume),
            "backing_volume": float(backing_volume),
            "chords": chords,
        }
        return out_path, meta
    finally:
        try:
            in_path.unlink(missing_ok=True)
        except Exception:
            pass


# ----------------------------
# Pages (rendered based on sidebar nav)
# ----------------------------
generate_btn = False

if nav == "Controls":
    st.markdown("<div class='step'>1) Upload a MIDI</div>", unsafe_allow_html=True)

    uploader_key = int(st.session_state.get("uploader_key", 0))
    uploaded = st.file_uploader(
        "MIDI file (.mid / .midi)",
        type=["mid", "midi"],
        key=f"uploaded_midi_{uploader_key}",
        help="Upload a MIDI that contains your lead melody (and optionally other tracks).",
    )

    if uploaded is not None:
        midi_bytes = uploaded.getvalue()
        sig = hashlib.sha1(midi_bytes).hexdigest()
        st.session_state["_uploaded_midi_bytes"] = midi_bytes
        st.session_state["_uploaded_midi_name"] = getattr(uploaded, "name", "input.mid")
        st.session_state["_uploaded_midi_sig"] = sig

    st.markdown("<div class='step'>2) Choose vibe</div>", unsafe_allow_html=True)

    moods = list_moods()
    if moods and st.session_state.get("mood_name") not in moods:
        st.session_state["mood_name"] = moods[0]
    st.selectbox(
        "Mood",
        moods,
        key="mood_name",
        help="Sets the overall vibe. This influences key bias + how the backing arrangement is shaped.",
    )

    st.toggle(
        "Auto-tune settings",
        key="auto_settings",
        help="Automatically picks sensible defaults for this MIDI (chords/drums/humanize/quantize/structure).",
    )

    uploaded_bytes: Optional[bytes] = st.session_state.get("_uploaded_midi_bytes")
    uploaded_sig: Optional[str] = st.session_state.get("_uploaded_midi_sig")

    if uploaded_bytes is not None and st.session_state["auto_settings"]:
        reco = recommend_settings(uploaded_bytes)
        st.caption(
            f"Auto: chords={_harmony_label(reco['harmony_mode'])} · drums={_drums_label(reco['drums_mode'])} · "
            f"quantize={'on' if reco['quantize_melody'] else 'off'}"
        )
        reapply = st.button(
            "Re-apply auto",
            use_container_width=True,
            help="Forces the auto-settings to re-run for the current file.",
        )
        _apply_auto_settings(reco, file_sig=str(uploaded_sig or ""), force=bool(reapply))

    st.markdown("**Style**")
    s1, s2 = st.columns(2)
    with s1:
        # Internal values stay the same; display text becomes "Rules-based" / "AI-generated"
        options_harmony = ["baseline (rules)", "ml (transformer)"]
        if st.session_state.get("harmony_mode") not in options_harmony:
            st.session_state["harmony_mode"] = options_harmony[0]
        st.selectbox(
            "Chord style",
            options_harmony,
            key="harmony_mode",
            format_func=lambda v: "Rules-based" if not str(v).startswith("ml") else "AI-generated",
            help="Rules-based = more predictable/safe. AI-generated = more creative/variable, but can be unstable on some MIDI files.",
        )
        if _is_ai_harmony(str(st.session_state.get("harmony_mode", ""))):
            st.caption(
                "🧪 **AI-generated chords are experimental.** For AI-specific knobs (temperature, top-k, penalties), open **Advanced** in the sidebar."
            )

    with s2:
        options_drums = ["rules", "ml"]
        if st.session_state.get("drums_mode") not in options_drums:
            st.session_state["drums_mode"] = options_drums[1]
        st.selectbox(
            "Drum style",
            options_drums,
            key="drums_mode",
            format_func=lambda v: "Rules-based" if str(v) == "rules" else "AI-generated",
            help="Rules-based = pattern-based. AI-generated = groove model that can sound more human, but may vary more between runs.",
        )

    st.toggle(
        "Auto structure (intro/verse/chorus/outro)",
        key="auto_sections",
        help="If on, the generator will vary patterns over time (e.g., sparser intro, fuller chorus).",
    )

    st.markdown("**Backing tracks**")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.toggle("Bass", key="make_bass", help="Generate a bassline that supports the chords and groove.")
    with b2:
        st.toggle("Chords", key="make_pad", help="Generate a chord/pad track (the harmonic backing).")
    with b3:
        st.toggle("Drums", key="make_drums", help="Generate a drum groove track.")

    st.markdown("**Mix**")
    m1, m2 = st.columns(2)
    with m1:
        st.slider(
            "Melody level",
            min_value=0.0,
            max_value=2.0,
            step=0.05,
            key="melody_volume",
            help="Scales the velocity (loudness) of your original melody track(s) in the output MIDI.",
        )
    with m2:
        st.slider(
            "Backing level",
            min_value=0.0,
            max_value=2.0,
            step=0.05,
            key="backing_volume",
            help="Scales the velocity (loudness) of generated backing tracks (bass/chords/drums).",
        )

    melody_track_indices: Optional[list[int]] = None

    if uploaded_bytes is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
            f.write(uploaded_bytes)
            tmp_path = Path(f.name)

        try:
            pm_preview, info_preview, grid_preview, melody_inst_preview, sel_preview = load_and_prepare(
                tmp_path, melody_instrument_index=None
            )
            _, intro_preview = _auto_pick_with_intro(pm_preview, info_preview, melody_inst_preview, sel_preview)

            with st.expander("Melody track selection", expanded=False):
                st.caption(
                    "If your melody cuts out (or is split across multiple tracks), turn **Auto-pick** off and select **all** lead tracks."
                )
                use_auto = st.toggle(
                    "Auto-pick melody track",
                    key="melody_auto_pick",
                    help="On: ChordCraft chooses a likely melody track automatically (and may include short intro tracks). "
                    "Off: You manually pick which tracks contain the melody.",
                )

                options: list[str] = []
                default_label: Optional[str] = None
                for i, inst in enumerate(pm_preview.instruments):
                    nm = inst.name or f"Instrument {i}"
                    tag = "DRUMS" if inst.is_drum else "INST"
                    label = f"{i}: {nm}  ·  {tag}  ·  notes={len(inst.notes)}"
                    options.append(label)
                    if i == sel_preview.instrument_index:
                        default_label = label

                if use_auto:
                    melody_track_indices = None
                    if intro_preview:
                        st.caption(
                            f"Auto-picked main: idx={sel_preview.instrument_index}, name='{sel_preview.instrument_name}' "
                            f"(+ intro tracks: {intro_preview})"
                        )
                    else:
                        st.caption(f"Auto-picked: idx={sel_preview.instrument_index}, name='{sel_preview.instrument_name}'")
                else:
                    preview_sig = str(uploaded_sig or "")
                    if st.session_state.get("_melody_pick_sig") != preview_sig:
                        st.session_state["_melody_pick_sig"] = preview_sig
                        st.session_state["melody_tracks_picked"] = [default_label] if default_label else []

                    picked = st.multiselect(
                        "Choose melody instrument(s) (pick ALL tracks that contain the lead)",
                        options=options,
                        key="melody_tracks_picked",
                        help="If your lead is split across tracks (common in exported MIDI), select them all.",
                    )
                    melody_track_indices = [int(x.split(":")[0].strip()) for x in picked] if picked else None

                with st.expander("Show instrument list"):
                    st.json(
                        [
                            {
                                "idx": i,
                                "name": (inst.name or f"Instrument {i}"),
                                "is_drum": inst.is_drum,
                                "notes": len(inst.notes),
                            }
                            for i, inst in enumerate(pm_preview.instruments)
                        ]
                    )
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    st.session_state["_melody_track_indices"] = melody_track_indices

    st.markdown("<div class='step'>3) Generate</div>", unsafe_allow_html=True)
    _render_settings_pills()
    st.caption("Tip: If the melody is wrong, fix **Melody track selection** before generating again.")
    generate_btn = st.button(
        "✨ Generate backing track",
        use_container_width=True,
        disabled=(uploaded_bytes is None),
        help="Runs the pipeline and produces a new multi-track MIDI. Output appears in the docked panel at the bottom.",
    )

elif nav == "Advanced":
    st.markdown("### Advanced controls")
    st.caption("Deeper knobs. Leave these alone at first — but they’re here when you want more control.")

    st.markdown("**Harmony (chords)**")
    harmony_mode = st.session_state["harmony_mode"]

    if str(harmony_mode).startswith("ml"):
        step_opts = [1.0, 2.0, 4.0]
        if float(st.session_state.get("chord_step_beats", 2.0)) not in step_opts:
            st.session_state["chord_step_beats"] = 2.0
        st.selectbox(
            "Chord step size (beats)",
            step_opts,
            key="chord_step_beats",
            help="How often the chord model is allowed to change chords. Smaller = more changes, larger = slower progression.",
        )
        st.toggle(
            "Include key features (recommended)",
            key="chord_include_key",
            help="Gives the model the estimated key as context. Usually improves stability and reduces weird chords.",
        )
        st.toggle(
            "Stochastic chords (more variety)",
            key="chord_stochastic",
            help="On: more randomness/variety. Off: more deterministic output (often more consistent).",
        )
        st.slider(
            "Chord temperature",
            min_value=0.7,
            max_value=1.6,
            step=0.01,
            key="chord_temp",
            help="Higher temperature = riskier/more diverse chords. Lower temperature = safer/more repetitive.",
        )
        st.slider(
            "Chord variety limit (top-k) — 0 = off",
            min_value=0,
            max_value=40,
            step=1,
            key="chord_top_k",
            help="Limits sampling to the top-k most likely chord candidates each step. Smaller = safer. 0 disables the cap.",
        )
        st.slider(
            "Chord repetition penalty",
            min_value=0.0,
            max_value=3.0,
            step=0.05,
            key="chord_repeat_penalty",
            help="Discourages repeating the same chord too often. Too high can cause jumpy progressions.",
        )
        st.slider(
            "Chord smoothness (change penalty)",
            min_value=0.0,
            max_value=0.6,
            step=0.01,
            key="chord_change_penalty",
            disabled=bool(st.session_state.get("chord_stochastic")),
            help="Encourages smoother chord changes (when stochastic is OFF). Higher = fewer abrupt changes.",
        )
        st.slider(
            "Bars per chord (Rules-based only)",
            min_value=1,
            max_value=4,
            step=1,
            key="bars_per_chord",
            disabled=True,
            help="Only used by the Rules-based chord engine (not AI-generated chords).",
        )
    else:
        st.slider(
            "Bars per chord",
            min_value=1,
            max_value=4,
            step=1,
            key="bars_per_chord",
            help="Rules-based chord engine: how long each chord lasts. Higher = slower progression.",
        )

    st.markdown("**Melody preprocessing**")
    st.toggle(
        "Quantize melody to beat grid",
        key="quantize_melody",
        help="Snaps the melody analysis to the beat grid. Helps messy MIDI, but can hurt expressive timing.",
    )

    st.divider()
    st.markdown("**Drums**")
    st.slider(
        "AI drum temperature",
        min_value=0.8,
        max_value=1.4,
        step=0.01,
        key="ml_temp",
        disabled=(st.session_state["drums_mode"] != "ml"),
        help="Controls randomness for AI-generated drums. Higher = more fills/variation; lower = steadier groove.",
    )

    st.divider()
    st.markdown("**Instruments (MIDI / General MIDI programs)**")
    st.caption("This changes the MIDI program numbers in the output. Your DAW/synth decides how they sound.")

    pad_idx = _preset_index(PAD_PRESETS, int(st.session_state["pad_program"]))
    pad_preset = st.selectbox(
        "Chords instrument (Pad track)",
        PAD_PRESETS,
        index=pad_idx,
        format_func=lambda x: f"{x[0]} ({x[1]})",
        key="pad_preset",
        help="Selects the General MIDI instrument program for the chords track.",
    )
    st.toggle(
        "Custom pad program # (0-127)",
        key="pad_custom_on",
        help="Use an exact General MIDI program number instead of a preset.",
    )
    if st.session_state["pad_custom_on"]:
        st.number_input(
            "Pad program",
            min_value=0,
            max_value=127,
            step=1,
            key="pad_program",
            help="General MIDI program number (0–127) for the chords/pad track.",
        )
    else:
        st.session_state["pad_program"] = int(pad_preset[1])

    bass_idx = _preset_index(BASS_PRESETS, int(st.session_state["bass_program"]))
    bass_preset = st.selectbox(
        "Bass instrument (Bass track)",
        BASS_PRESETS,
        index=bass_idx,
        format_func=lambda x: f"{x[0]} ({x[1]})",
        key="bass_preset",
        help="Selects the General MIDI instrument program for the bass track.",
    )
    st.toggle(
        "Custom bass program # (0-127)",
        key="bass_custom_on",
        help="Use an exact General MIDI program number instead of a preset.",
    )
    if st.session_state["bass_custom_on"]:
        st.number_input(
            "Bass program",
            min_value=0,
            max_value=127,
            step=1,
            key="bass_program",
            help="General MIDI program number (0–127) for the bass track.",
        )
    else:
        st.session_state["bass_program"] = int(bass_preset[1])

    st.divider()
    st.markdown("**Humanize**")
    st.toggle(
        "Humanize timing/velocity",
        key="humanize",
        help="Adds subtle timing + velocity variation so the backing feels less robotic.",
    )
    st.slider(
        "Timing jitter (ms)",
        min_value=0.0,
        max_value=35.0,
        step=1.0,
        key="jitter_ms",
        disabled=not bool(st.session_state["humanize"]),
        help="Randomly shifts note timings (in milliseconds). Too high can sound sloppy.",
    )
    st.slider(
        "Velocity jitter",
        min_value=0,
        max_value=20,
        step=1,
        key="vel_jitter",
        disabled=not bool(st.session_state["humanize"]),
        help="Randomly varies note velocities (loudness). Higher = more dynamic but less consistent.",
    )
    st.slider(
        "Swing (0..1)",
        min_value=0.0,
        max_value=0.6,
        step=0.01,
        key="swing",
        disabled=not bool(st.session_state["humanize"]),
        help="Applies swing feel. 0 = straight. Higher = more swing.",
    )

    st.divider()
    st.markdown("**Seed**")
    st.number_input(
        "Seed value",
        min_value=0,
        step=1,
        key="seed_value",
        help="A fixed seed makes AI-generated outputs more repeatable (same settings + same seed ⇒ similar results).",
    )
    st.toggle(
        "Use seed",
        key="use_seed",
        help="Turn on to make stochastic parts (AI-generated chords/drums + humanize) more repeatable.",
    )

    st.divider()
    st.markdown("**🎧 Audio preview (WAV)**")
    st.toggle(
        "Auto-render audio preview after generation",
        key="auto_render_audio",
        help="If enabled, ChordCraft tries to render a WAV preview automatically after generation (requires FluidSynth + a SoundFont).",
    )
    if not DEFAULT_SOUNDFONT_PATH:
        st.caption("SoundFont not found in repo (expected: soundfonts/GeneralUser-GS.sf2). WAV preview will be disabled.")
    if not _fluidsynth_cmd():
        st.caption("FluidSynth not found on PATH. WAV preview will be disabled.")

    st.divider()
    st.markdown("### Generate")
    _render_settings_pills()

    uploaded_bytes = st.session_state.get("_uploaded_midi_bytes")
    if uploaded_bytes is None:
        st.caption("Upload a MIDI in **Controls** first to enable generation.")

    generate_btn = st.button(
        "✨ Generate backing track",
        use_container_width=True,
        disabled=(uploaded_bytes is None),
        help="Runs the pipeline and updates the output panel at the bottom.",
    )


else:
    st.markdown("### Help & details")
    st.markdown(
        """
**What ChordCraft does**
- Takes a MIDI file, **finds the melody**, estimates the key, and generates **chords + bass + drums**.
- Exports a new multi-track MIDI you can drag into any DAW (Ableton, Logic, FL, etc.).
- Optional WAV preview (MIDI → audio) is done locally with FluidSynth + a SoundFont.

**Typical workflow**
1) Upload a MIDI → 2) pick a Mood → 3) Generate → 4) download MIDI (and optional WAV preview).
- If the melody is wrong or cuts out, open **Melody track selection** and manually pick the track(s).

**Rules-based vs AI-generated**
- **Rules-based**: safer, more predictable, fewer odd chords.
- **AI-generated**: more variety and sometimes more “musical”, but can be inconsistent on messy MIDI.
  - When using AI-generated chords, open **Advanced** to tune temperature/top-k/penalties.

**Audio preview (WAV)**
- Requires `fluidsynth` on PATH and a SoundFont (`.sf2`) available in the repo.
- If preview fails, you can still download the MIDI and audition it in a DAW.

**Tips for best results**
- Clean monophonic lead lines work best (single melody track, not dense stacked chords).
- If your MIDI has multiple lead layers, select all of them as melody sources.
- If things sound too rigid: enable **Humanize** + add a bit of swing.
- If things sound too chaotic with AI-generated: lower temperature and/or top-k in Advanced.

**Troubleshooting**
- “No melody notes extracted”: your selected melody track probably has no notes or is a drum track.
- Melody feels incomplete: often the lead is split across tracks → select multiple tracks.
- Drums too busy: lower AI drum temperature.
- Chords too weird: switch to Rules-based or reduce AI chord temperature/top-k.

**Privacy**
- ChordCraft runs locally in your Streamlit app. Uploaded MIDI stays in memory during the session (unless you deploy it somewhere and log/store files yourself).
"""
    )


# ============================
# GENERATE (only runs when Controls page button is pressed)
# ============================
if generate_btn:
    uploaded_bytes: Optional[bytes] = st.session_state.get("_uploaded_midi_bytes")
    if uploaded_bytes is None:
        st.error("No MIDI uploaded. Go to Controls and upload a file first.")
    else:
        seed: Optional[int] = int(st.session_state["seed_value"]) if bool(st.session_state["use_seed"]) else None
        structure_mode = "auto" if bool(st.session_state["auto_sections"]) else "none"
        melody_track_indices = st.session_state.get("_melody_track_indices")

        try:
            with st.spinner("Generating backing track..."):
                out_path, meta = run_pipeline(
                    midi_bytes=uploaded_bytes,
                    mood_name=st.session_state["mood_name"],
                    harmony_mode=st.session_state["harmony_mode"],
                    chord_model_path=DEFAULT_CHORD_MODEL_PATH,
                    chord_step_beats=float(st.session_state["chord_step_beats"]),
                    chord_include_key=bool(st.session_state["chord_include_key"]),
                    chord_stochastic=bool(st.session_state["chord_stochastic"]),
                    chord_temp=float(st.session_state["chord_temp"]),
                    chord_top_k=int(st.session_state["chord_top_k"]),
                    chord_repeat_penalty=float(st.session_state["chord_repeat_penalty"]),
                    chord_change_penalty=float(st.session_state["chord_change_penalty"]),
                    bars_per_chord=int(st.session_state["bars_per_chord"]),
                    quantize_melody=bool(st.session_state["quantize_melody"]),
                    make_bass=bool(st.session_state["make_bass"]),
                    make_pad=bool(st.session_state["make_pad"]),
                    make_drums=bool(st.session_state["make_drums"]),
                    melody_track_indices=melody_track_indices,
                    seed=seed,
                    structure_mode=structure_mode,
                    drums_mode=st.session_state["drums_mode"],
                    ml_temp=float(st.session_state["ml_temp"]),
                    humanize=bool(st.session_state["humanize"]),
                    jitter_ms=float(st.session_state["jitter_ms"]),
                    vel_jitter=int(st.session_state["vel_jitter"]),
                    swing=float(st.session_state["swing"]),
                    pad_program=int(st.session_state["pad_program"]),
                    bass_program=int(st.session_state["bass_program"]),
                    melody_volume=float(st.session_state["melody_volume"]),
                    backing_volume=float(st.session_state["backing_volume"]),
                )

            st.session_state["_generated_midi_bytes"] = out_path.read_bytes()
            st.session_state["_generated_meta"] = meta

            gen_sig = hashlib.sha1(st.session_state["_generated_midi_bytes"]).hexdigest()
            st.session_state["_generated_audio_sig"] = gen_sig
            st.session_state["_generated_audio_wav"] = None
            st.session_state["_generated_audio_err"] = None

            if bool(st.session_state.get("auto_render_audio", True)):
                sf2_path = DEFAULT_SOUNDFONT_PATH
                cmd = _fluidsynth_cmd()
                if cmd and sf2_path and Path(sf2_path).exists():
                    try:
                        with st.spinner("🎧 Rendering audio preview..."):
                            sf2_mtime = os.path.getmtime(sf2_path)
                            wav_bytes = render_midi_to_wav_cached(
                                st.session_state["_generated_midi_bytes"],
                                sf2_path,
                                sf2_mtime,
                                sample_rate=44100,
                            )
                        st.session_state["_generated_audio_wav"] = wav_bytes
                    except Exception as _e:
                        st.session_state["_generated_audio_err"] = str(_e)

        except Exception as e:
            st.error(f"Generation failed: {e}")


# ============================
# OUTPUT (Always visible: docked bottom panel)
# ============================

st.subheader("Output")
gen_bytes = st.session_state.get("_generated_midi_bytes")
meta = st.session_state.get("_generated_meta")

if not gen_bytes or not meta:
    st.markdown('<p class="muted">No output yet. Generate a backing track to see results here.</p>', unsafe_allow_html=True)
else:
    info = meta["info"]
    sel = meta["selection"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Tempo", f"{info.tempo_bpm:.1f} BPM")
    c2.metric("Time Sig", f"{info.time_signature.numerator}/{info.time_signature.denominator}")
    c3.metric("Length", f"{info.duration:.1f} s")

    st.markdown(f"**Detected key:** {key_to_string(meta['raw_key'])}")
    if meta["key"] != meta["raw_key"]:
        st.markdown(f"**After mood '{meta['mood'].name}' bias:** {key_to_string(meta['key'])}")

    chords = meta["chords"]
    preview = " · ".join(chord_label(c.root_pc, c.quality, c.extensions) for c in chords[:12])
    st.markdown("**Chord preview:**")
    st.code(preview if preview else "(none)")

    with st.expander("Details", expanded=False):
        st.markdown(f"**Pad program:** `{meta['pad_program']}`  ·  **Bass program:** `{meta['bass_program']}`")

        if meta["selected_melody_indices"]:
            st.markdown(f"**Melody tracks (manual):** {meta['selected_melody_indices']}")
        else:
            if meta["auto_intro_indices"]:
                st.markdown(
                    f"**Melody tracks (auto):** main idx={sel.instrument_index} · `{sel.instrument_name}` "
                    f"(+ intro: {meta['auto_intro_indices']})"
                )
            else:
                st.markdown(f"**Melody track (auto):** idx={sel.instrument_index} · `{sel.instrument_name}`")

        st.markdown(f"**Used melody indices:** {meta['used_melody_indices']}")
        st.markdown(f"**Melody notes extracted (for analysis):** `{meta['melody_note_count']}`")
        st.markdown("**Backing note counts:**")
        st.json(meta["arrangement_counts"])

    st.markdown("**🎧 Preview (audio)**")
    current_sig = hashlib.sha1(gen_bytes).hexdigest()
    if st.session_state.get("_generated_audio_sig") != current_sig:
        st.session_state["_generated_audio_sig"] = current_sig
        st.session_state["_generated_audio_wav"] = None
        st.session_state["_generated_audio_err"] = None

    sf2_path = DEFAULT_SOUNDFONT_PATH
    cmd = _fluidsynth_cmd()

    if not cmd:
        st.info("Install FluidSynth (so `fluidsynth` is on your PATH) to enable in-app audio preview.")
    elif not sf2_path:
        st.info("SoundFont not found in repo (expected: `soundfonts/GeneralUser-GS.sf2`). WAV preview is disabled.")
    elif not Path(sf2_path).exists():
        st.info(f"SoundFont file not found: `{sf2_path}`. WAV preview is disabled.")
    else:
        err = st.session_state.get("_generated_audio_err")
        wav_bytes = st.session_state.get("_generated_audio_wav")

        render_clicked = False
        if not wav_bytes:
            if err:
                st.warning(f"Audio preview failed: {err}")
            render_clicked = st.button("🎧 Render audio preview", use_container_width=True)

        if render_clicked and not wav_bytes:
            try:
                with st.spinner("Rendering audio preview..."):
                    sf2_mtime = os.path.getmtime(sf2_path)
                    wav_bytes = render_midi_to_wav_cached(gen_bytes, sf2_path, sf2_mtime, sample_rate=44100)
                st.session_state["_generated_audio_wav"] = wav_bytes
                st.session_state["_generated_audio_err"] = None
            except Exception as e:
                st.session_state["_generated_audio_err"] = str(e)
                st.warning(f"Audio preview failed: {e}")

        wav_bytes = st.session_state.get("_generated_audio_wav")
        if wav_bytes:
            wav_player(wav_bytes)

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    label="⬇️ Download WAV preview",
                    data=wav_bytes,
                    file_name="chordcraft_preview.wav",
                    mime="audio/wav",
                    use_container_width=True,
                )
            with dl2:
                st.download_button(
                    label="⬇️ Download generated MIDI",
                    data=gen_bytes,
                    file_name="backing_track.mid",
                    mime="audio/midi",
                    use_container_width=True,
                )
        else:
            st.download_button(
                label="⬇️ Download generated MIDI",
                data=gen_bytes,
                file_name="backing_track.mid",
                mime="audio/midi",
                use_container_width=True,
            )

