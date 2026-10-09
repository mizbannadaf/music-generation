# Music Generation with AI (LSTM)

Learns patterns from MIDI files and composes new music.

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
# 1) Train. Uses music21's built-in Bach corpus (no download needed)
python music_gen.py train --corpus bach --epochs 30

#    ...or train on your own MIDI files (classical, jazz, etc.)
python music_gen.py train --data_dir ./midi_files --epochs 40

# 2) Generate new music -> generated.mid
python music_gen.py generate --length 300 --temperature 0.9

# 3) Optional WAV export (needs FluidSynth + a .sf2 soundfont)
python music_gen.py generate --wav --soundfont path/to/font.sf2
```

## Where to get MIDI data
- Classical Piano MIDI dataset (Kaggle), MAESTRO, Lakh MIDI, or jazz MIDI from the Jazz MIDI Archive.
- Put the `.mid` files in a folder and use `--data_dir`.
- Tip: stick to ONE genre/composer per training run for more coherent output.

## How it works
| Step | What happens |
|------|--------------|
| Collect | Load `.mid` files with `music21` |
| Preprocess | `chordify()` merges simultaneous notes. Each event becomes a token like `60.64.67_1.0` (MIDI pitches + duration) |
| Model | Embedding -> 2-layer LSTM -> Linear, predicting the next token from the previous 50 |
| Train | Cross-entropy loss, Adam, gradient clipping. Best checkpoint saved to `music_model.pt` |
| Generate | Seed with 50 real tokens, sample with temperature, write MIDI via `music21` |

## Tuning
- More epochs and more data give better results; a GPU (e.g., Google Colab) speeds training a lot.
- `--temperature` 0.7 = more repetitive and safe, 1.2 = more adventurous.
- Increase `--seq_len` for longer musical memory.

## Ideas to extend
Add a GAN or Transformer, add velocity/rest tokens, or condition on key/style.
