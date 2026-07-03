# MedASR Medical Dictation Backend — Integration

Status: **implemented + verified headless (Phases 0–5); remaining: live dictation test + asset upload (both user)**. Personal-use app, so the gated HAI-DEF license is fine.

## What MedASR actually is (corrected from first research)

Google MedASR is **not** Wav2Vec2-Conformer as first assumed. Ground truth from the model files:

- Custom **`lasr_ctc`** architecture (Google LASR, JAX-origin), encoder `lasr_encoder` — Conformer-CTC, 105M params, English-only.
- **Requires `transformers >= 5.0`** to load (custom `Lasr*` classes; 5.10.2 works). Python **3.12** — 3.14 segfaults on load.
- Frontend = **128-bin log-mel** (n_fft 512, win 400 symmetric-Hann, hop 160, kaldi mel scale 125–7500 Hz, DC bin dropped, `log(max(·,1e-5))`, **no mean/var norm**). Not raw waveform.
- Tokenizer = **SentencePiece Unigram**, model output vocab **512**, blank/pad id **0**.
- Decode (from `LasrTokenizer._decode` + `LasrForCTC.generate`): per-frame `argmax` → **collapse consecutive dup ids** → **drop id 0** → SP detokenize (`▁`→space) → maps spoken punctuation.
- Emits **spoken punctuation + section headers** as text: `{period}` `{comma}` `{colon}` `{new paragraph}` `[FINDINGS]` `[IMPRESSION]` `</s>`. The `{…}` command set is **open-ended and undocumented** (commands are assembled from ordinary SP pieces — the vocab has no command tokens, the model card lists none, and Google's own `LasrTokenizer` decode leaves them literal). Live use also produced `{next line}`. VoicePill's `post_process` therefore uses a generic `{command}` scanner with a Dragon-style vocabulary (punctuation, parens/quotes/brackets, hyphen/dash/slash, next|new line/paragraph, tab, …) with per-command spacing rules; **unknown commands pass through verbatim** so nothing dictated is silently dropped — extend `command()` in `medasr.rs` when a new one shows up.
- WER: greedy **6.6%** / +6-gram KenLM beam-8 **4.6%** on radiology dictation vs Whisper large-v3 **25.3%**. We ship **greedy** (no LM) — still ~4× better than Whisper.
- **Speed: ~0.57 s on CPU for 44 s audio (~77× realtime).** So we run **ONNX on CPU only** — no second CUDA consumer, no `ort`/whisper CUDA coexistence risk.

## Phase 0 — asset prep (DONE, on dev machine)

Built in a **uv venv (Python 3.12)** with `transformers==5.10.2`, `torch 2.12 CPU`, `optimum-onnx`, `onnxruntime`, `onnxscript`.

- Exported a **self-contained `medasr_wave.onnx`**: input = raw f32 waveform `[1, N]`, output = logits `[1, frames, 512]`. **The log-mel frontend is baked into the ONNX graph** (unfold/rfft/mel/log as torch ops), so Rust feeds the same `&[f32]` buffer the audio coordinator already produces — **zero mel reimplementation in Rust**.
- Dropped the attention mask (VoicePill transcribes one buffer, batch 1, no padding → all frames valid).
- Verified ONNX vs PyTorch: **argmax agreement 1.0**, and the Rust-intended decode path (onnx→argmax→groupby→drop-0→`tokenizers` decode) reproduces Google's official `generate()`+`batch_decode()` output **byte-for-byte**.
- **int8 dynamic quant rejected** — too lossy for medical (86% frame agreement; dropped "54", garbled "main PA"). Ship **fp32**.

**Assets** (staged at `%APPDATA%/com.voicepill.app/models/` for local testing):
- `medasr.onnx` — 405 MB, self-contained fp32 (was `medasr_wave_fp32.onnx`, external data merged in).
- `medasr-tokenizer.json` — 52 KB (HF `tokenizer.json`).

**TODO (user):** upload both to a GitHub release tagged **`medasr-v1`** as `medasr.onnx` + `medasr-tokenizer.json` (see `MEDASR_BASE` in `models.rs`). 405 MB upload; publishing is the user's call.

Export scripts live in the session scratchpad (`export_wave.py`, `validate_decode.py`, `quantize.py`, `ipv4.py`).
Network note: this machine's **IPv6 route to huggingface.co is dead**; the scripts import `ipv4.py` to force IPv4.

## Phase 1 — Rust ONNX backend (DONE) — `src-tauri/src/medasr.rs`

- `MedAsrBackend` — lazy-loads an `ort` `Session` (CPU EP) + `tokenizers::Tokenizer`, cached like `WhisperContext`.
- `transcribe(&[f32])`: `Value::from_array([1,N])` → run → `try_extract_tensor::<f32>` → `greedy_ctc` (argmax + collapse + drop blank 0) → `tokenizer.decode` → `post_process` (spoken-punctuation → real chars, whitespace cleanup).
- Deps: `ort = "=2.0.0-rc.10"` (features `std`, `download-binaries`; **not** `ndarray` — clashes with ndarray 0.16), `tokenizers = "0.21"` (default-features off + `onig`).
- Unit tests for `post_process` + `greedy_ctc`; `#[ignore]`d `medasr_end_to_end` runs the real onnx.

## Phase 2 — catalog + downloader (DONE) — `src-tauri/src/models.rs`

- `Engine { Whisper, MedAsr }`; `ModelEntry.engine`; `engine_of(id)` for routing.
- `files_for(entry)` → per-engine `(filename, url)` list. Whisper = `ggml-{id}.bin` from HF; MedASR = `{id}.onnx` + `{id}-tokenizer.json` from `MEDASR_BASE` release.
- Catalog entry: id `medasr`, "MedASR (Medical · English)", 405 MB.
- `download` loops all files, emits one `model:done` after all land; `list`/`delete` handle multi-file; `ModelInfo` gains `engine`.

## Phase 3 — routing + UI (DONE)

- `transcribe.rs`: `Transcriber` owns both the Whisper slot and a `MedAsrBackend`; `is_loaded`/`ensure_loaded`/`transcribe` route by `engine_of(id)`. Audio coordinator unchanged.
- `lib.rs`: `mod medasr;`.
- `src/settings.ts`: `ModelInfo.engine`; "Medical · English" badge on the MedASR row; **language dropdown disabled** when MedASR is the active model (English-only).

`cargo check` is green.

## Phase 4 — verification + packaging

1. **Headless end-to-end (DONE)**: `medasr_end_to_end` (real ONNX + Google's radiology
   `test_audio.wav` from the HF cache) passes in ~1.7 s. Also fixed a real
   `post_process` bug it flushed out: a space survived after an inserted newline
   (`{new paragraph} impression` → `"\n\n impression"`); cleanup now drops spaces
   hugging newlines.
2. **Packaging (RESOLVED — nothing to bundle)**: `ort` download-binaries on
   x86_64-pc-windows-msvc links onnxruntime **statically**
   (`cargo:rustc-link-lib=static=onnxruntime`, pyke `dfbin` cache) — there is **no
   onnxruntime.dll** to ship. It dynamic-links `DirectML.dll`, which is inbox in
   Windows 10 1903+ (System32), so the NSIS installer needs no changes.
3. **UI polish (DONE)**: added the missing `.model-tag` badge + `.row.disabled` CSS
   (settings.ts referenced both but styles.css had neither).
4. **Real-app dictation test (user)**: `tauri dev`, select MedASR, dictate, confirm
   paste. Everything up to the mic is covered headless.
5. **Host assets (user)**: upload `medasr.onnx` + `medasr-tokenizer.json` (staged in
   `%APPDATA%/com.voicepill.app/models/`) to a GitHub release tagged `medasr-v1`.
6. Optional: fp16 export (~200 MB, verify argmax) if 405 MB download is too big.

## Phase 5 — KenLM beam search + input normalization (DONE)

Closes the gap to the advertised 4.6% WER (greedy alone is 6.6%). Google ships a
**6-gram KenLM over SentencePiece tokens** in the HF repo (`lm_6.kenlm` /
`lm_6.arpa.xz`, 33.5M n-grams); the reference notebook decodes with pyctcdecode
(`beam_width=8`, defaults alpha 0.5 / beta 1.5 / unk −10, each token treated as
one LM "word": `▁` + piece with inner `▁`→`#`).

- **No KenLM C++ binding.** `scripts/export_medasr_lm.py` converts the ARPA to
  `medasr-lm.bin` (448 MB): per order, sorted `u64` keys (token ids packed 10
  bits each, re-keyed to the model's own ids; `<s>`=512 `</s>`=513 `<unk>`=514)
  + f32 log10 prob/backoff arrays. 8 296 n-grams with unmappable words (ids
  ≥ 512, unreachable by the CTC head) dropped.
- `src-tauri/src/medasr_lm.rs`: `LmModel` mmaps the file (memmap2, instant
  load, OS-paged) with binary-search lookups + Katz backoff (`BaseScore`
  semantics), and `beam_search` runs CTC prefix beam search with pyctcdecode's
  scoring. One deliberate difference: LM scores apply at token emission rather
  than at next-word start — final hypothesis scores identical, only mid-search
  ranking can differ by one word of lookahead.
- `medasr.rs`: LM is **optional on disk** — missing/corrupt `medasr-lm.bin`
  logs and falls back to greedy. Also **peak-normalizes** input to 0.95 (gain
  capped at 30×) before inference: the LASR frontend has no mean/var norm, so a
  quiet mic costs accuracy.
- Verified: unit tests for backoff math + beam; `lm_real_file_spot_checks`
  (`--ignored`) confirms exported values match `lm_6.arpa` byte-for-byte;
  `medasr_end_to_end` passes with the LM active (radiology clip decodes
  correctly, same wall time ~1.9 s incl. model load).
- `models.rs`: catalog entry now 853 MB, downloads `medasr-lm.bin` as third
  file from the `medasr-v1` release.

**TODO (user):** upload `medasr-lm.bin` (staged in
`%APPDATA%/com.voicepill.app/models/`) to the `medasr-v1` GitHub release next to
the onnx + tokenizer.

Remaining model-level limits no decoder can fix: radiology-domain tuning
(general speech is worse than the headline WER) and known-bad date
transcription (training-data anonymization stripped dates).

## Risks / watch-items

- fp32 is 405 MB — larger than a Whisper small, smaller than medium/large. Acceptable.
- No no-speech prob from CTC — rely on the existing upstream RMS/silence gate (see memory `transcribe-emi-and-silence-gate`), which runs before `transcribe`.
- Download progress bar restarts near-instantly for the tiny tokenizer file after the 405 MB onnx (per-file percentages); cosmetic only.

## Sources
- https://huggingface.co/google/medasr (gated)
- transformers `models/lasr/{modeling,tokenization,feature_extraction}_lasr.py`
