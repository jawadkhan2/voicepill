//! Whisper transcription via `whisper-rs` (whisper.cpp bindings).
//!
//! Built with the `cuda` feature so the RTX 5080 does the work. The model is
//! loaded lazily and cached: reloading a multi-GB model per utterance would be
//! far too slow, so we keep the `WhisperContext` alive and only reload when the
//! selected model id changes.

use std::path::PathBuf;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub struct Transcriber {
    models_dir: PathBuf,
    /// (model id, loaded context). `None` until the first transcription.
    loaded: Option<(String, WhisperContext)>,
}

impl Transcriber {
    pub fn new(models_dir: PathBuf) -> Self {
        Transcriber {
            models_dir,
            loaded: None,
        }
    }

    fn model_path(&self, id: &str) -> PathBuf {
        self.models_dir.join(format!("ggml-{id}.bin"))
    }

    /// Whether the context for `id` is already resident.
    pub fn is_loaded(&self, id: &str) -> bool {
        self.loaded.as_ref().map(|(m, _)| m == id).unwrap_or(false)
    }

    /// Ensure the context for `id` is loaded (reloading if the model changed).
    /// `pub` so the audio coordinator can preload a model into VRAM on demand
    /// (when the user clicks "Use") rather than lazily on first transcription.
    pub fn ensure_loaded(&mut self, id: &str) -> Result<(), String> {
        let already = self.loaded.as_ref().map(|(m, _)| m == id).unwrap_or(false);
        if already {
            return Ok(());
        }

        let path = self.model_path(id);
        if !path.exists() {
            return Err(format!("model '{id}' is not downloaded"));
        }
        let path_str = path.to_str().ok_or("model path is not valid UTF-8")?;

        let mut params = WhisperContextParameters::default();
        params.use_gpu(true);
        let ctx = WhisperContext::new_with_params(path_str, params)
            .map_err(|e| format!("failed to load model '{id}': {e}"))?;

        self.loaded = Some((id.to_string(), ctx));
        Ok(())
    }

    /// Transcribe a 16 kHz mono f32 buffer to text.
    pub fn transcribe(
        &mut self,
        model_id: &str,
        language: &str,
        audio: &[f32],
    ) -> Result<String, String> {
        self.ensure_loaded(model_id)?;
        let ctx = &self.loaded.as_ref().unwrap().1;

        let mut state = ctx.create_state().map_err(|e| e.to_string())?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        if language != "auto" {
            params.set_language(Some(language));
        }
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        // Anti-hallucination: greedy at temperature 0 (no random fallback that
        // invents text), suppress leading blanks and non-speech tokens like
        // "[music]" / "(laughs)" that Whisper emits for noise.
        params.set_temperature(0.0);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);

        state
            .full(params, audio)
            .map_err(|e| format!("transcription failed: {e}"))?;

        // Drop segments the model itself judges to be silence. On a near-silent
        // clip Whisper still emits a confident-looking segment ("you", "Thank
        // you", ...) but with a high no-speech probability — this filters those
        // out while leaving genuine quiet/whispered speech (low prob) intact.
        const NO_SPEECH_THRESHOLD: f32 = 0.6;
        let mut text = String::new();
        for segment in state.as_iter() {
            if segment.no_speech_probability() >= NO_SPEECH_THRESHOLD {
                continue;
            }
            if let Ok(s) = segment.to_str_lossy() {
                text.push_str(&s);
            }
        }
        Ok(text.trim().to_string())
    }
}
