from backingtrack.midi_io import load_and_prepare
from backingtrack.melody import extract_melody_notes

pm, info, grid, melody_inst, sel = load_and_prepare("data/raw/85263.mid")
mel = extract_melody_notes(melody_inst, grid=grid)

print("picked:", sel)
print("melody notes:", len(mel))
print("first 5:", mel[:5])
