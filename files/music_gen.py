"""
Music Generation with AI  (LSTM, PyTorch + music21)

Pipeline
  1. Collect MIDI    -> your own folder of .mid files OR music21's built-in corpus (Bach etc.)
  2. Preprocess      -> each note/chord becomes a token like "60_0.5" or "60.64.67_1.0"
                        (MIDI pitch number(s) + quantized duration)
  3. Model           -> Embedding -> 2-layer LSTM -> Linear (next-token prediction)
  4. Train           -> sliding windows of SEQ_LEN tokens predict the next token
  5. Generate        -> sample new tokens with temperature, write MIDI (+ optional WAV)

Usage
  python music_gen.py train    --corpus bach --epochs 30
  python music_gen.py train    --data_dir ./midi_files --epochs 40
  python music_gen.py generate --length 300 --temperature 0.9 --out generated.mid
"""
import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from music21 import chord, converter, corpus, note, stream, tempo

DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]  # allowed note lengths (quarter notes)
CKPT = "music_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------
# 1 + 2. DATA COLLECTION AND PREPROCESSING
# ----------------------------------------------------------------------------
def quantize(d):
    return min(DURATIONS, key=lambda x: abs(x - float(d)))


def load_scores(data_dir=None, corpus_composer="bach", max_files=100):
    """Yield music21 scores from a MIDI folder, or from music21's built-in corpus."""
    if data_dir:
        files = glob.glob(os.path.join(data_dir, "**", "*.mid*"), recursive=True)
        random.shuffle(files)
        for f in files[:max_files]:
            try:
                yield converter.parse(f)
            except Exception as e:
                print(f"  skipped {f}: {e}")
    else:
        paths = corpus.getComposer(corpus_composer)[:max_files]
        for p in paths:
            try:
                yield corpus.parse(p)
            except Exception as e:
                print(f"  skipped {p}: {e}")


def score_to_tokens(score):
    """Turn a score into a list of tokens. Simultaneous notes are merged into chords
    (chordify), so polyphonic music becomes a single sequence."""
    tokens = []
    for el in score.chordify().flatten().notes:
        pitches = ".".join(str(p.midi) for p in el.pitches)
        tokens.append(f"{pitches}_{quantize(el.duration.quarterLength)}")
    return tokens


def build_dataset(data_dir, composer, max_files, seq_len):
    all_tokens = []
    for i, score in enumerate(load_scores(data_dir, composer, max_files), 1):
        toks = score_to_tokens(score)
        all_tokens.extend(toks)
        print(f"  parsed file {i}: {len(toks)} tokens")
    if len(all_tokens) <= seq_len + 1:
        raise SystemExit("Not enough music found. Check --data_dir or use --corpus.")

    vocab = sorted(set(all_tokens))
    tok2id = {t: i for i, t in enumerate(vocab)}
    ids = np.array([tok2id[t] for t in all_tokens], dtype=np.int64)

    n = len(ids) - seq_len
    X = np.stack([ids[i:i + seq_len] for i in range(n)])
    y = ids[seq_len:seq_len + n]
    print(f"Total tokens: {len(ids)} | vocabulary: {len(vocab)} | training samples: {n}")
    return X, y, vocab, ids


# ----------------------------------------------------------------------------
# 3. MODEL
# ----------------------------------------------------------------------------
class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, embed=128, hidden=256, layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed)
        self.lstm = nn.LSTM(embed, hidden, layers, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        out, _ = self.lstm(self.embed(x))
        return self.fc(self.drop(out[:, -1, :]))  # predict next token from last step


# ----------------------------------------------------------------------------
# 4. TRAINING
# ----------------------------------------------------------------------------
def train(args):
    print("Collecting and preprocessing MIDI data...")
    X, y, vocab, ids = build_dataset(args.data_dir, args.corpus, args.max_files, args.seq_len)

    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                        batch_size=args.batch_size, shuffle=True)
    model = MusicLSTM(len(vocab)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    history, best = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(xb)
        avg = total / len(X)
        history.append(avg)
        print(f"Epoch {epoch:3d}/{args.epochs}  loss = {avg:.4f}")
        if avg < best:
            best = avg
            torch.save({"model": model.state_dict(), "vocab": vocab,
                        "seeds": ids[:20000], "seq_len": args.seq_len}, CKPT)

    print(f"Training done. Best model saved to {CKPT}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.plot(history); plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title("Training loss"); plt.savefig("loss_curve.png", dpi=120)
        print("Saved loss_curve.png")
    except Exception:
        pass


# ----------------------------------------------------------------------------
# 5. GENERATION -> MIDI -> AUDIO
# ----------------------------------------------------------------------------
@torch.no_grad()
def generate_tokens(model, vocab, seed_ids, seq_len, length, temperature):
    model.eval()
    window = list(seed_ids)
    out = []
    for _ in range(length):
        x = torch.tensor([window[-seq_len:]], dtype=torch.long, device=DEVICE)
        logits = model(x)[0] / max(temperature, 1e-3)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        nxt = int(np.random.choice(len(vocab), p=probs))
        window.append(nxt)
        out.append(vocab[nxt])
    return out


def tokens_to_midi(tokens, path, bpm=100):
    s = stream.Stream()
    s.insert(0, tempo.MetronomeMark(number=bpm))
    offset = 0.0
    for tok in tokens:
        pitch_part, dur = tok.rsplit("_", 1)
        dur = float(dur)
        pitches = [int(p) for p in pitch_part.split(".")]
        if len(pitches) == 1:
            el = note.Note(pitches[0])
        else:
            el = chord.Chord(pitches)
        el.quarterLength = dur
        s.insert(offset, el)
        offset += dur
    s.write("midi", fp=path)
    print(f"Saved MIDI: {path}")


def midi_to_audio(midi_path, wav_path, soundfont=None):
    """Optional. Needs FluidSynth installed + a .sf2 soundfont."""
    try:
        from midi2audio import FluidSynth
        FluidSynth(soundfont) if soundfont else FluidSynth()
        (FluidSynth(soundfont) if soundfont else FluidSynth()).midi_to_audio(midi_path, wav_path)
        print(f"Saved audio: {wav_path}")
    except Exception as e:
        print(f"Audio conversion skipped ({e}). Open the .mid in VLC/MuseScore, "
              "or install FluidSynth to export WAV.")


def generate(args):
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    vocab, seq_len = ckpt["vocab"], ckpt["seq_len"]
    model = MusicLSTM(len(vocab)).to(DEVICE)
    model.load_state_dict(ckpt["model"])

    seeds = ckpt["seeds"]
    start = random.randint(0, len(seeds) - seq_len - 1)
    seed_ids = seeds[start:start + seq_len].tolist()

    tokens = generate_tokens(model, vocab, seed_ids, seq_len, args.length, args.temperature)
    tokens_to_midi(tokens, args.out, args.bpm)
    if args.wav:
        midi_to_audio(args.out, args.out.replace(".mid", ".wav"), args.soundfont)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LSTM music generator")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data_dir", default=None, help="folder with .mid files")
    t.add_argument("--corpus", default="bach", help="music21 corpus composer (if no --data_dir)")
    t.add_argument("--max_files", type=int, default=100)
    t.add_argument("--seq_len", type=int, default=50)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch_size", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3)

    g = sub.add_parser("generate")
    g.add_argument("--length", type=int, default=300, help="number of notes/chords")
    g.add_argument("--temperature", type=float, default=0.9, help="lower=safer, higher=wilder")
    g.add_argument("--bpm", type=int, default=100)
    g.add_argument("--out", default="generated.mid")
    g.add_argument("--wav", action="store_true", help="also export WAV via FluidSynth")
    g.add_argument("--soundfont", default=None)

    a = p.parse_args()
    train(a) if a.cmd == "train" else generate(a)
