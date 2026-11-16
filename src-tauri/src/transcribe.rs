//! Whisper transcription via `whisper-rs` (whisper.cpp bindings).
//!
//! Built with a native GPU backend where available: CUDA on Windows/NVIDIA and
//! Metal on macOS/Apple Silicon. The model is loaded lazily and cached:
//! reloading a multi-GB model per utterance would be far too slow, so we keep
//! the `WhisperContext` — and its decode `WhisperState` — alive and only reload
//! when the selected model id changes.

use std::path::PathBuf;

use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

use crate::medasr::MedAsrBackend;
use crate::models::{engine_of, Engine};
use crate::parakeet::ParakeetBackend;

/// Front door for transcription. Routes each model id to its engine: Parakeet
/// (ONNX Runtime, CUDA), Whisper (ggml/whisper.cpp, GPU where available) or
/// MedASR (ONNX/CPU). The audio coordinator talks only to this type and never
/// has to know which engine runs. Dictation is English-only.
pub struct Transcriber {
    models_dir: PathBuf,
    /// Whisper: the loaded model. `None` until the first load.
    loaded: Option<LoadedWhisper>,
    /// MedASR backend (ONNX). Owns its own lazy-loaded session.
    medasr: MedAsrBackend,
    /// Parakeet backend (ONNX Runtime CUDA). Owns its own sessions.
    parakeet: ParakeetBackend,
}

/// A loaded Whisper model plus its reusable decode state.
struct LoadedWhisper {
    id: String,
    ctx: WhisperContext,
    /// `None` only transiently (taken during a call) or after a failed call.
    state: Option<WhisperState>,
}

impl Transcriber {
    pub fn new(models_dir: PathBuf) -> Self {
        Transcriber {
            medasr: MedAsrBackend::new(models_dir.clone()),
            parakeet: ParakeetBackend::new(models_dir.clone()),
            models_dir,
            loaded: None,
        }
    }

    fn model_path(&self, id: &str) -> PathBuf {
        self.models_dir.join(format!("ggml-{id}.bin"))
    }

    /// Whether the model for `id` is already resident on its engine.
    pub fn is_loaded(&self, id: &str) -> bool {
        match engine_of(id) {
            Engine::MedAsr => self.medasr.is_loaded(id),
            Engine::Parakeet => self.parakeet.is_loaded(id),
            Engine::Whisper => self.loaded.as_ref().map(|l| l.id == id).unwrap_or(false),
        }
    }

    /// Ensure the model for `id` is loaded (reloading if the model changed).
    /// `pub` so the audio coordinator can preload a model on demand (when the
    /// user clicks "Use") rather than lazily on first transcription.
    pub fn ensure_loaded(&mut self, id: &str) -> Result<(), String> {
        match engine_of(id) {
            Engine::MedAsr => return self.medasr.ensure_loaded(id),
            Engine::Parakeet => {
                // One GPU model at a time: drop Whisper's VRAM first.
                self.loaded = None;
                return self.parakeet.ensure_loaded(id);
            }
            Engine::Whisper => {}
        }

        let already = self.loaded.as_ref().map(|l| l.id == id).unwrap_or(false);
        if already {
            return Ok(());
        }

        let path = self.model_path(id);
        if !path.exists() {
            return Err(format!("model '{id}' is not downloaded"));
        }
        let path_str = path.to_str().ok_or("model path is not valid UTF-8")?;

        // Free the previous model (including a Parakeet model) *before* loading
        // the next one so two models never sit in VRAM at once (large-v3 + its
        // state is ~4 GB; doubling that on a busy 16 GB card pushes WDDM into
        // paging to system RAM, which makes inference crawl as if it were
        // running on the CPU).
        self.loaded = None;
        self.parakeet.unload();

        let use_gpu = should_use_gpu();
        let mut params = WhisperContextParameters::default();
        params.use_gpu(use_gpu);
        // Flash attention: smaller KV caches and faster decode on CUDA/Metal.
        // whisper.cpp's own default is on; whisper-rs defaults it off.
        params.flash_attn(use_gpu);
        let ctx = WhisperContext::new_with_params(path_str, params)
            .map_err(|e| format!("failed to load model '{id}': {e}"))?;
        // Allocate the decode state (KV caches + compute buffers) once, up
        // front, and reuse it for every utterance. Creating it per call used to
        // re-allocate hundreds of MB of VRAM on every dictation — slow, and on a
        // nearly-full card exactly the allocation that fails or spills.
        let state = ctx
            .create_state()
            .map_err(|e| format!("failed to init state for '{id}': {e}"))?;
        crate::vlog!("[transcribe] loaded whisper '{id}' (gpu={use_gpu}, flash_attn={use_gpu})");

        self.loaded = Some(LoadedWhisper {
            id: id.to_string(),
            ctx,
            state: Some(state),
        });
        Ok(())
    }

    /// Transcribe a 16 kHz mono f32 buffer to English text, with spoken
    /// formatting commands ("next line") already applied.
    pub fn transcribe(&mut self, model_id: &str, audio: &[f32]) -> Result<String, String> {
        match engine_of(model_id) {
            // MedASR expands its own `{command}` tokens.
            Engine::MedAsr => self.medasr.transcribe(model_id, audio),
            Engine::Parakeet => {
                self.ensure_loaded(model_id)?;
                let text = self.parakeet.transcribe(model_id, audio)?;
                Ok(crate::voice_commands::apply(&text))
            }
            Engine::Whisper => {
                let text = self.transcribe_whisper(model_id, audio)?;
                Ok(crate::voice_commands::apply(&text))
            }
        }
    }

    fn transcribe_whisper(&mut self, model_id: &str, audio: &[f32]) -> Result<String, String> {
        self.ensure_loaded(model_id)?;
        let loaded = self.loaded.as_mut().expect("ensure_loaded set the model");

        // Reuse the cached state. If a previous run failed mid-inference the
        // state was dropped; build a fresh one.
        let mut state = match loaded.state.take() {
            Some(s) => s,
            None => loaded.ctx.create_state().map_err(|e| e.to_string())?,
        };

        match run_full(&mut state, audio) {
            Ok(text) => {
                loaded.state = Some(state);
                Ok(text)
            }
            Err(first) => {
                // A failed whisper_full can leave the state unusable (e.g. a
                // transient CUDA allocation failure while another app holds most
                // of the VRAM). Retry once on a fresh state before giving up so
                // a one-off hiccup doesn't cost the user their sentence.
                crate::vlog!("[transcribe] {first}; retrying on a fresh state");
                drop(state);
                let mut fresh = loaded.ctx.create_state().map_err(|e| e.to_string())?;
                let text = run_full(&mut fresh, audio)?;
                loaded.state = Some(fresh);
                Ok(text)
            }
        }
    }
}

/// Run one full decode on `state` and collect the text.
fn run_full(state: &mut WhisperState, audio: &[f32]) -> Result<String, String> {
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    // English only: skips language detection (faster, and no more short clips
    // misdetected as another language).
    params.set_language(Some("en"));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    // Anti-hallucination: start greedy at temperature 0, suppress leading
    // blanks and non-speech tokens like "[music]" / "(laughs)" that Whisper
    // emits for noise. whisper.cpp's temperature fallback stays on — it only
    // kicks in when a window decodes badly.
    params.set_temperature(0.0);
    params.set_suppress_blank(true);
    params.set_suppress_nst(true);
    // Silence detection is whisper.cpp's own OpenAI-style rule: a 30 s window
    // is discarded only when no_speech_prob > 0.6 AND the decode's average
    // log-prob < -1.0 (the model is unsure there is speech *and* unsure of the
    // words). Pinned explicitly so a crate bump can't silently change it.
    params.set_no_speech_thold(0.6);
    params.set_logprob_thold(-1.0);

    let t0 = std::time::Instant::now();
    state
        .full(params, audio)
        .map_err(|e| format!("transcription failed: {e}"))?;
    let ms = t0.elapsed().as_millis();

    // We deliberately do NOT drop segments on `no_speech_probability()` alone.
    // That value is per 30 s *window* (every segment in the window shares it)
    // and smaller models report >= 0.6 on perfectly clear speech, so the old
    // `no_speech >= 0.6 → skip` filter silently threw away whole sentences —
    // and on long dictations whole 30 s chunks, leaving only the tail. The only
    // thing still filtered here is Whisper's stock silence hallucinations
    // ("Thank you.", "you", ...) and only when the model also thinks the window
    // is probably not speech.
    let mut text = String::new();
    let mut n_segments = 0;
    let mut dropped = 0;
    for segment in state.as_iter() {
        n_segments += 1;
        let Ok(s) = segment.to_str_lossy() else { continue };
        if segment.no_speech_probability() >= 0.6 && is_stock_hallucination(&s) {
            dropped += 1;
            continue;
        }
        text.push_str(&s);
    }
    crate::vlog!(
        "[transcribe] {:.2}s audio in {ms}ms, segments={n_segments}, dropped={dropped}",
        audio.len() as f32 / 16_000.0
    );
    Ok(text.trim().to_string())
}

/// Whisper's well-known outputs for silence / room noise (learned from video
/// subtitles). Matched on the whole segment, case- and punctuation-insensitive.
fn is_stock_hallucination(segment: &str) -> bool {
    let norm: String = segment
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .to_lowercase();
    let norm = norm.split_whitespace().collect::<Vec<_>>().join(" ");
    matches!(
        norm.as_str(),
        "" | "you"
            | "thank you"
            | "thanks"
            | "thank you so much"
            | "thank you very much"
            | "thanks for watching"
            | "thank you for watching"
            | "bye"
    )
}

fn should_use_gpu() -> bool {
    cfg!(any(windows, target_os = "macos"))
}

#[cfg(test)]
mod tests {
    use super::is_stock_hallucination;

    #[test]
    fn stock_hallucinations_match_loosely() {
        assert!(is_stock_hallucination(" Thank you."));
        assert!(is_stock_hallucination("you"));
        assert!(is_stock_hallucination(" Thanks for watching!"));
    }

    /// Headless GPU end-to-end check (needs a downloaded model + a 16 kHz mono
    /// PCM16 WAV):
    /// `VOICEPILL_TEST_WAV=clip.wav VOICEPILL_TEST_MODEL=tiny cargo test --release
    ///  --lib whisper_end_to_end -- --ignored --nocapture`
    /// Prints per-segment no-speech probabilities (what the old filter keyed on)
    /// and the final text, then transcribes again to exercise state reuse.
    #[test]
    #[ignore]
    fn whisper_end_to_end() {
        let wav = std::env::var("VOICEPILL_TEST_WAV").expect("set VOICEPILL_TEST_WAV");
        let model = std::env::var("VOICEPILL_TEST_MODEL").unwrap_or_else(|_| "tiny".into());
        let bytes = std::fs::read(&wav).unwrap();
        let data_at = bytes.windows(4).position(|w| w == b"data").unwrap() + 8;
        let audio: Vec<f32> = bytes[data_at..]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / i16::MAX as f32)
            .collect();
        let dir = std::path::PathBuf::from(std::env::var("APPDATA").unwrap())
            .join("com.voicepill.app")
            .join("models");
        let mut t = super::Transcriber::new(dir);
        t.ensure_loaded(&model).unwrap();

        // Old behaviour, for comparison: what `no_speech >= 0.6 → skip` kept.
        {
            let loaded = t.loaded.as_mut().unwrap();
            let state = loaded.state.as_mut().unwrap();
            let _ = super::run_full(state, &audio).unwrap();
            let mut old = String::new();
            for seg in state.as_iter() {
                let s = seg.to_str_lossy().unwrap().to_string();
                println!("segment nsp={:.3} {:?}", seg.no_speech_probability(), s);
                if seg.no_speech_probability() < 0.6 {
                    old.push_str(&s);
                }
            }
            println!("OLD FILTER: {:?}", old.trim());
        }
        for run in 0..2 {
            let t0 = std::time::Instant::now();
            let text = t.transcribe(&model, &audio).unwrap();
            println!("NEW run {run} ({} ms): {text:?}", t0.elapsed().as_millis());
            assert!(!text.is_empty());
        }
    }

    #[test]
    fn real_speech_is_never_a_hallucination() {
        assert!(!is_stock_hallucination(" Thank you for the review, ship it."));
        assert!(!is_stock_hallucination(" to a deep dive code review."));
    }
}
