//! NVIDIA Parakeet TDT 0.6B v2 (English) on ONNX Runtime's CUDA provider.
//!
//! Parakeet is a FastConformer encoder + Token-and-Duration Transducer (TDT)
//! decoder. Compared to Whisper large-v3 it has a lower English WER on the Open
//! ASR Leaderboard (~6.05% vs ~7.4%), is a fraction of the size (0.6B params),
//! emits punctuation + capitalisation natively and does not hallucinate text
//! into silence the way Whisper does.
//!
//! The model is three ONNX graphs, all run on the GPU:
//! - `preprocessor.onnx`: waveform → 128-bin log-mel features (NeMo featurizer,
//!   per-feature normalised). Rewritten from onnx-asr's `nemo128.onnx` so the
//!   STFT is a Conv against a windowed DFT basis — ONNX Runtime has no CUDA STFT
//!   kernel, so the stock graph would run it on the CPU.
//! - `encoder.onnx`: features → `[1, 1024, T/8]` acoustic frames.
//! - `decoder_joint.onnx`: one TDT step — (frame, last token, LSTM state) →
//!   1025 token logits (1024 SentencePiece tokens + blank) followed by 5
//!   duration logits (skip 0..=4 frames).
//!
//! Greedy TDT decoding mirrors onnx-asr / NeMo: at frame `t`, take the best
//! token; a non-blank token is emitted and advances the prediction network;
//! the best duration says how many frames to jump. A blank with duration 0
//! still advances one frame, and at most `MAX_TOKENS_PER_STEP` tokens are
//! emitted on one frame.

use std::path::PathBuf;
use std::time::Instant;

#[cfg(windows)]
use ort::ep::{self, cuda::ConvAlgorithmSearch};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

use crate::ort_env;

const PREPROCESSOR: &str = "preprocessor.onnx";
const ENCODER: &str = "encoder.onnx";
const DECODER_JOINT: &str = "decoder_joint.onnx";
const VOCAB: &str = "vocab.txt";

/// Files that make up a Parakeet model directory (`<models>/<id>/…`).
pub const MODEL_FILES: &[&str] = &[PREPROCESSOR, ENCODER, DECODER_JOINT, VOCAB];

/// Encoder output width.
const ENC_DIM: usize = 1024;
/// Prediction-network LSTM: 2 layers × hidden 640.
const STATE_LAYERS: usize = 2;
const STATE_DIM: usize = 640;
/// Same cap NeMo / onnx-asr use for symbols emitted on a single frame.
const MAX_TOKENS_PER_STEP: usize = 10;
const SAMPLE_RATE: usize = 16_000;
/// Every dictation up to this length runs at this one input shape.
const FIXED_SECONDS: usize = 30;

/// Audio is zero-padded to a fixed input length before inference (the real
/// length is passed alongside, so the padding is masked out). ONNX Runtime +
/// cuDNN re-plan the encoder whenever its input shape differs from the
/// previous call — measured ~55 ms per switch on an RTX 5080, more than the
/// encoder itself (~15 ms for 30 s). So every dictation up to 30 s runs at the
/// one 30 s shape; longer ones round up to 10 s steps.
fn bucket_len(samples: usize) -> usize {
    let secs = samples.div_ceil(SAMPLE_RATE).max(1);
    let padded = if secs <= FIXED_SECONDS {
        FIXED_SECONDS
    } else {
        secs.div_ceil(10) * 10
    };
    padded * SAMPLE_RATE
}

pub struct ParakeetBackend {
    models_dir: PathBuf,
    loaded: Option<Loaded>,
}

struct Loaded {
    id: String,
    pre: Session,
    enc: Session,
    dec: Session,
    /// Token id → text, with SentencePiece's `▁` already turned into a space.
    vocab: Vec<String>,
    blank: usize,
}

impl ParakeetBackend {
    pub fn new(models_dir: PathBuf) -> Self {
        ParakeetBackend {
            models_dir,
            loaded: None,
        }
    }

    fn dir(&self, id: &str) -> PathBuf {
        self.models_dir.join(id)
    }

    pub fn is_loaded(&self, id: &str) -> bool {
        self.loaded.as_ref().map(|l| l.id == id).unwrap_or(false)
    }

    /// Release the sessions (and their VRAM), e.g. when switching to Whisper.
    pub fn unload(&mut self) {
        self.loaded = None;
    }

    /// Load all three graphs onto the GPU (reloading if `id` changed) and run
    /// one warm-up pass so the first real dictation doesn't pay for CUDA/cuDNN
    /// initialisation and kernel loading.
    pub fn ensure_loaded(&mut self, id: &str) -> Result<(), String> {
        if self.is_loaded(id) {
            return Ok(());
        }
        let dir = self.dir(id);
        for f in MODEL_FILES {
            if !dir.join(f).exists() {
                return Err(format!("model '{id}' is not downloaded ({f} missing)"));
            }
        }
        // Free the old model's VRAM before allocating the new one.
        self.loaded = None;
        ort_env::add_cuda_runtime_to_path(&self.models_dir.join(ort_env::CUDA_RUNTIME_DIR));

        let t0 = Instant::now();
        let pre = gpu_session(&dir.join(PREPROCESSOR))?;
        let enc = gpu_session(&dir.join(ENCODER))?;
        let dec = gpu_session(&dir.join(DECODER_JOINT))?;
        let (vocab, blank) = load_vocab(&dir.join(VOCAB))?;
        let mut loaded = Loaded {
            id: id.to_string(),
            pre,
            enc,
            dec,
            vocab,
            blank,
        };
        let load_ms = t0.elapsed().as_millis();

        // Warm-up at the fixed shape: the first run pays for CUDA/cuDNN
        // initialisation and kernel loading. Low-level noise, not zeros, so
        // the feature normaliser doesn't divide 0 by 0.
        let t1 = Instant::now();
        let warm: Vec<f32> = (0..SAMPLE_RATE)
            .map(|i| ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 1e-3)
            .collect();
        loaded.run(&warm)?;
        crate::vlog!(
            "[parakeet] loaded '{id}' on CUDA in {load_ms}ms (warm-up {}ms)",
            t1.elapsed().as_millis()
        );
        self.loaded = Some(loaded);
        Ok(())
    }

    /// Transcribe a 16 kHz mono f32 buffer.
    pub fn transcribe(&mut self, id: &str, audio: &[f32]) -> Result<String, String> {
        self.ensure_loaded(id)?;
        let loaded = self.loaded.as_mut().expect("ensure_loaded set the model");
        // The encoder subsamples 8×; anything under ~0.1 s has no frames.
        if audio.len() < 1_600 {
            return Ok(String::new());
        }
        let t0 = Instant::now();
        let (text, stats) = loaded.run(audio)?;
        crate::vlog!(
            "[parakeet] {:.2}s audio in {}ms (features+encoder {}ms, decode {}ms / {} steps)",
            audio.len() as f32 / 16_000.0,
            t0.elapsed().as_millis(),
            stats.encode_ms,
            stats.decode_ms,
            stats.steps
        );
        Ok(text)
    }
}

/// A session pinned to the CUDA execution provider. `error_on_failure` turns a
/// missing/broken CUDA or cuDNN install into a load error instead of letting
/// ONNX Runtime quietly run the whole model on the CPU.
#[cfg(not(windows))]
fn gpu_session(_path: &std::path::Path) -> Result<Session, String> {
    Err("Parakeet needs an NVIDIA GPU (CUDA); use a Whisper model on this platform".into())
}

#[cfg(windows)]
fn gpu_session(path: &std::path::Path) -> Result<Session, String> {
    let env = ort_env::env()?;
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("?");
    Session::builder(env)
        .map_err(|e| format!("ort session builder: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("ort opt level: {e}"))?
        .with_execution_providers([ep::CUDA::default()
            .with_device_id(0)
            // Dictations vary in length and every new input shape would
            // otherwise trigger cuDNN's exhaustive conv benchmark.
            .with_conv_algorithm_search(ConvAlgorithmSearch::Heuristic)
            .with_tf32(true)
            .build()
            .error_on_failure()])
        .map_err(|e| format!("CUDA provider unavailable for {name}: {e}"))?
        .commit_from_file(path)
        .map_err(|e| format!("failed to load {name} on CUDA: {e}"))
}

/// `vocab.txt` lines are `<token> <id>`; the blank is `<blk>`.
fn load_vocab(path: &std::path::Path) -> Result<(Vec<String>, usize), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read vocab: {e}"))?;
    parse_vocab(&text)
}

fn parse_vocab(text: &str) -> Result<(Vec<String>, usize), String> {
    let mut entries: Vec<(usize, String)> = Vec::new();
    for line in text.lines().filter(|l| !l.is_empty()) {
        let (tok, id) = line
            .rsplit_once(' ')
            .ok_or_else(|| format!("bad vocab line: {line:?}"))?;
        let id: usize = id.parse().map_err(|_| format!("bad vocab id: {line:?}"))?;
        entries.push((id, tok.replace('\u{2581}', " ")));
    }
    let mut vocab = vec![String::new(); entries.len()];
    for (id, tok) in entries {
        *vocab
            .get_mut(id)
            .ok_or_else(|| format!("vocab id {id} out of range"))? = tok;
    }
    let blank = vocab
        .iter()
        .position(|t| t == "<blk>")
        .ok_or("vocab has no <blk> token")?;
    Ok((vocab, blank))
}

/// Where the time went in one transcription (for the log).
struct RunStats {
    encode_ms: u128,
    decode_ms: u128,
    steps: usize,
}

impl Loaded {
    fn run(&mut self, audio: &[f32]) -> Result<(String, RunStats), String> {
        let t0 = Instant::now();
        // ---- features (GPU) ------------------------------------------------
        let padded = bucket_len(audio.len());
        let mut wave = Vec::with_capacity(padded);
        wave.extend_from_slice(audio);
        wave.resize(padded, 0.0);
        let wave = Tensor::from_array(([1usize, padded], wave))
            .map_err(|e| format!("waveform tensor: {e}"))?;
        let wave_len = Tensor::from_array(([1usize], vec![audio.len() as i64]))
            .map_err(|e| format!("waveform length tensor: {e}"))?;
        let pre_out = self
            .pre
            .run(ort::inputs!["waveforms" => wave, "waveforms_lens" => wave_len])
            .map_err(|e| format!("preprocessor failed: {e}"))?;
        let (fshape, feats) = pre_out["features"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("extract features: {e}"))?;
        let fshape: Vec<usize> = fshape.iter().map(|&d| d as usize).collect();
        let feats = feats.to_vec();
        let (_, flen) = pre_out["features_lens"]
            .try_extract_tensor::<i64>()
            .map_err(|e| format!("extract feature lengths: {e}"))?;
        let flen = flen[0];
        drop(pre_out);

        // ---- encoder (GPU) -------------------------------------------------
        let feats = Tensor::from_array((fshape, feats)).map_err(|e| format!("features: {e}"))?;
        let flen = Tensor::from_array(([1usize], vec![flen])).map_err(|e| format!("len: {e}"))?;
        let enc_out = self
            .enc
            .run(ort::inputs!["audio_signal" => feats, "length" => flen])
            .map_err(|e| format!("encoder failed: {e}"))?;
        let (eshape, enc) = enc_out["outputs"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("extract encoder output: {e}"))?;
        // [1, 1024, T] → frame-major [T][1024] so each decode step reads one
        // contiguous slice.
        let (dim, frames) = (eshape[1] as usize, eshape[2] as usize);
        if dim != ENC_DIM {
            return Err(format!("unexpected encoder width {dim}"));
        }
        let mut by_frame = vec![0f32; frames * dim];
        for d in 0..dim {
            for t in 0..frames {
                by_frame[t * dim + d] = enc[d * frames + t];
            }
        }
        let (_, elen) = enc_out["encoded_lengths"]
            .try_extract_tensor::<i64>()
            .map_err(|e| format!("extract encoder lengths: {e}"))?;
        let n_frames = (elen[0].max(0) as usize).min(frames);
        drop(enc_out);

        // ---- greedy TDT decode (GPU per step) ------------------------------
        let encode_ms = t0.elapsed().as_millis();
        let t1 = Instant::now();
        let (tokens, steps) = self.decode(&by_frame, n_frames)?;
        let stats = RunStats {
            encode_ms,
            decode_ms: t1.elapsed().as_millis(),
            steps,
        };
        Ok((detokenize(&self.vocab, &tokens), stats))
    }

    /// Returns the emitted token ids and the number of decoder calls.
    fn decode(&mut self, frames: &[f32], n_frames: usize) -> Result<(Vec<usize>, usize), String> {
        let vocab_size = self.vocab.len();
        let state_len = STATE_LAYERS * STATE_DIM;
        let mut state1 = vec![0f32; state_len];
        let mut state2 = vec![0f32; state_len];
        let mut tokens: Vec<usize> = Vec::new();
        let mut last = self.blank;

        let mut t = 0usize;
        let mut emitted = 0usize;
        let mut steps = 0usize;
        while t < n_frames {
            steps += 1;
            let frame = frames[t * ENC_DIM..(t + 1) * ENC_DIM].to_vec();
            let inputs = ort::inputs![
                "encoder_outputs" => Tensor::from_array(([1usize, ENC_DIM, 1], frame)).map_err(|e| e.to_string())?,
                "targets" => Tensor::from_array(([1usize, 1], vec![last as i32])).map_err(|e| e.to_string())?,
                "target_length" => Tensor::from_array(([1usize], vec![1i32])).map_err(|e| e.to_string())?,
                "input_states_1" => Tensor::from_array(([STATE_LAYERS, 1, STATE_DIM], state1.clone())).map_err(|e| e.to_string())?,
                "input_states_2" => Tensor::from_array(([STATE_LAYERS, 1, STATE_DIM], state2.clone())).map_err(|e| e.to_string())?,
            ];
            let out = self
                .dec
                .run(inputs)
                .map_err(|e| format!("decoder failed: {e}"))?;
            let (_, logits) = out["outputs"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("extract logits: {e}"))?;
            if logits.len() <= vocab_size {
                return Err(format!("decoder output too small ({})", logits.len()));
            }
            let token = argmax(&logits[..vocab_size]);
            let step = argmax(&logits[vocab_size..]);

            if token != self.blank {
                // Only an emitted token advances the prediction network.
                let (_, s1) = out["output_states_1"]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("extract state: {e}"))?;
                let (_, s2) = out["output_states_2"]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("extract state: {e}"))?;
                state1.copy_from_slice(s1);
                state2.copy_from_slice(s2);
                tokens.push(token);
                last = token;
                emitted += 1;
            }
            if step > 0 {
                t += step;
                emitted = 0;
            } else if token == self.blank || emitted == MAX_TOKENS_PER_STEP {
                t += 1;
                emitted = 0;
            }
        }
        Ok((tokens, steps))
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// Join SentencePiece pieces (`▁` already mapped to a space) and clean spacing
/// exactly like onnx-asr's `\A\s|\s\B|(\s)\b` rule: drop a leading space and
/// any space not followed by a word character (so "word ." → "word.").
fn detokenize(vocab: &[String], ids: &[usize]) -> String {
    let joined: String = ids.iter().map(|&i| vocab[i].as_str()).collect();
    let chars: Vec<char> = joined.chars().collect();
    let mut out = String::with_capacity(joined.len());
    for (i, &c) in chars.iter().enumerate() {
        if c.is_whitespace() {
            let next_is_word = chars
                .get(i + 1)
                .map(|n| n.is_alphanumeric() || *n == '_')
                .unwrap_or(false);
            if i == 0 || !next_is_word {
                continue;
            }
            out.push(' ');
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_parses_and_maps_word_boundaries() {
        let (v, blank) = parse_vocab("<unk> 0\n\u{2581}the 1\ns 2\n, 3\n<blk> 4\n").unwrap();
        assert_eq!(v[1], " the");
        assert_eq!(blank, 4);
    }

    #[test]
    fn dictations_share_one_input_shape() {
        let s = |x: f32| bucket_len((x * 16_000.0) as usize) / 16_000;
        assert_eq!(s(0.4), 30);
        assert_eq!(s(12.0), 30);
        assert_eq!(s(30.0), 30);
        assert_eq!(s(30.5), 40);
        assert_eq!(s(46.1), 50);
    }

    #[test]
    fn detokenize_cleans_spaces() {
        let v: Vec<String> = [" Hello", ",", " wor", "ld", " .", " it", "'s"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(detokenize(&v, &[0, 1, 2, 3, 4]), "Hello, world.");
        assert_eq!(detokenize(&v, &[5, 6]), "it's");
    }

    /// GPU end-to-end (needs the model in `%APPDATA%\com.voicepill.app\models`
    /// and a 16 kHz mono PCM16 WAV):
    /// `VOICEPILL_TEST_WAV=clip.wav cargo test --release --lib parakeet_end_to_end -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn parakeet_end_to_end() {
        let wav = std::env::var("VOICEPILL_TEST_WAV").expect("set VOICEPILL_TEST_WAV");
        let id = std::env::var("VOICEPILL_TEST_MODEL").unwrap_or_else(|_| "parakeet-tdt-0.6b-v2".into());
        let bytes = std::fs::read(&wav).unwrap();
        let data_at = bytes.windows(4).position(|w| w == b"data").unwrap() + 8;
        let audio: Vec<f32> = bytes[data_at..]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect();
        let dir = PathBuf::from(std::env::var("APPDATA").unwrap())
            .join("com.voicepill.app")
            .join("models");
        let mut p = ParakeetBackend::new(dir);
        let t0 = Instant::now();
        p.ensure_loaded(&id).unwrap();
        println!("load+warm-up {} ms", t0.elapsed().as_millis());
        for run in 0..3 {
            let t0 = Instant::now();
            let text = p.transcribe(&id, &audio).unwrap();
            println!("run {run} ({} ms): {text}", t0.elapsed().as_millis());
            assert!(!text.is_empty());
        }
        // Real dictations never repeat a length: time a spread of fresh ones.
        for secs in [2.3f32, 4.7, 6.1, 8.9, 11.3, 13.7, 17.2, 21.9] {
            let n = ((secs * 16_000.0) as usize).min(audio.len());
            let t0 = Instant::now();
            let text = p.transcribe(&id, &audio[..n]).unwrap();
            println!("{secs:>5.1}s fresh length: {} ms ({} chars)", t0.elapsed().as_millis(), text.len());
        }
    }
}
