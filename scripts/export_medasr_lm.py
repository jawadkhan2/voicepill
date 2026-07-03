"""Convert MedASR's lm_6.arpa.xz to VoicePill's medasr-lm.bin.

The LM is a 6-gram KenLM over SentencePiece tokens where each ARPA "word" is a
model token's piece with '▁' replaced by '#' (see the reference notebook's
LasrCtcBeamSearchDecoder), plus <s> / </s> / <unk>. We re-key every word to the
model's own token id (0..511) so the Rust decoder needs no string table:
  token id (as emitted by CTC)      -> same id
  <s> -> 512, </s> -> 513, <unk> -> 514
N-grams containing words that map to no reachable token are dropped (the
decoder can never query them).

Binary layout (little-endian), see medasr.rs LmModel for the reader:
  0   magic  b"VPL1"
  4   u32    order (6)
  8   u32    bos id (512)
  12  u32    eos id (513)
  16  u32    unk id (514)
  20  u32    reserved (0)
  24  u64[6] n-gram count per order
  72  for each order o = 1..6:
        u64[count]  keys, sorted ascending; key = sum(id << 10*(o-1-i))
        f32[count]  log10 probabilities
        f32[count]  log10 backoff weights (omitted for o == 6)
        pad to 8-byte alignment
"""

import json
import lzma
import struct
import sys
import time

import numpy as np

ARPA = sys.argv[1]
TOKENIZER_JSON = sys.argv[2]
OUT = sys.argv[3]

ORDER = 6
BOS, EOS, UNK = 512, 513, 514

# --- token piece -> model id map ---------------------------------------------
tok = json.load(open(TOKENIZER_JSON, encoding="utf-8"))
piece_to_id = {}
vocab = tok["model"]["vocab"]
if isinstance(vocab, list):  # Unigram: [[piece, score], ...] with index = id
    for pid, entry in enumerate(vocab):
        piece_to_id[entry[0]] = pid
else:
    piece_to_id = {p: i for p, i in vocab.items()}
for added in tok.get("added_tokens", []):
    piece_to_id.setdefault(added["content"], added["id"])

assert not any("#" in p for p in piece_to_id), "literal '#' piece breaks the ▁->#Mapping"

def word_id(w: str):
    if w == "<s>":
        return BOS
    if w == "</s>":
        return EOS
    if w == "<unk>":
        return UNK
    pid = piece_to_id.get(w.replace("#", "▁"))
    if pid is None:
        pid = piece_to_id.get(w)
    # ids >= 512 are unreachable by the 512-way CTC head -> unmappable
    return pid if pid is not None and pid < 512 else None

# --- parse ARPA ---------------------------------------------------------------
t0 = time.time()
counts = {}
f = lzma.open(ARPA, "rt", encoding="utf-8")
for line in f:
    line = line.strip()
    if line.startswith("ngram "):
        o, n = line[6:].split("=")
        counts[int(o)] = int(n)
    elif line.startswith("\\1-grams"):
        break
assert sorted(counts) == list(range(1, ORDER + 1)), counts
print("counts:", counts)

keys = {o: np.empty(counts[o], dtype=np.uint64) for o in counts}
probs = {o: np.empty(counts[o], dtype=np.float32) for o in counts}
backs = {o: np.zeros(counts[o], dtype=np.float32) for o in counts}
n_in = {o: 0 for o in counts}
dropped = 0

order = 1
for line in f:
    line = line.strip()
    if not line:
        continue
    if line.startswith("\\"):
        if line == "\\end\\":
            break
        order = int(line[1 : line.index("-")])
        print(f"  parsing {order}-grams... ({time.time()-t0:.0f}s)")
        continue
    parts = line.split("\t")
    prob = float(parts[0])
    words = parts[1].split(" ")
    key = 0
    ok = True
    for w in words:
        wid = word_id(w)
        if wid is None:
            ok = False
            break
        key = (key << 10) | wid
    if not ok:
        dropped += 1
        continue
    i = n_in[order]
    keys[order][i] = key
    probs[order][i] = prob
    if len(parts) > 2:
        backs[order][i] = float(parts[2])
    n_in[order] = i + 1
f.close()
print(f"parsed in {time.time()-t0:.0f}s, dropped {dropped} n-grams with unmappable words")

# --- sort + write -------------------------------------------------------------
out = open(OUT, "wb")
out.write(b"VPL1")
out.write(struct.pack("<IIIII", ORDER, BOS, EOS, UNK, 0))
for o in range(1, ORDER + 1):
    out.write(struct.pack("<Q", n_in[o]))
for o in range(1, ORDER + 1):
    n = n_in[o]
    k = keys[o][:n]
    idx = np.argsort(k, kind="stable")
    assert len(np.unique(k)) == n, f"duplicate {o}-gram keys"
    out.write(k[idx].tobytes())
    out.write(probs[o][:n][idx].tobytes())
    if o < ORDER:
        out.write(backs[o][:n][idx].tobytes())
    if out.tell() % 8:
        out.write(b"\0" * (8 - out.tell() % 8))
out.close()
print(f"wrote {OUT} ({time.time()-t0:.0f}s total)")
