//! Microphone capture and resampling.
//!
//! Whisper wants **16 kHz mono f32**. Mics give us whatever the device's default
//! config is (commonly 44.1/48 kHz, 1–2 channels, f32/i16/u16). So we:
//!   1. Open the selected (or default) input device with `cpal`.
//!   2. In the audio callback, convert to f32 and downmix to mono into a buffer.
//!   3. On stop, resample that mono buffer to 16 kHz with `rubato`.
//!
//! `cpal::Stream` is `!Send` on Windows, so the stream is built, held, and dropped
//! entirely on the coordinator thread (`run`). The rdev hook thread only sends
//! lightweight `SessionEvent`s over a channel — it must never block.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use tauri::{AppHandle, Emitter, Manager};

use crate::history::TranscriptHistoryState;
use crate::settings::SettingsState;
use crate::transcribe::Transcriber;

/// Whisper's required input sample rate.
const TARGET_RATE: u32 = 16_000;

/// Speech gate thresholds (see the gate in `run`). Tuned to reject silence and
/// accidental quick toggles while still passing whispers/quiet speech. If real
/// quiet speech ever gets dropped, lower `SPEECH_RMS_FLOOR`; if silence still
/// slips through, raise it.
const MIN_SPEECH_SECONDS: f32 = 0.3;
const SPEECH_RMS_FLOOR: f32 = 0.0025; // ≈ -52 dBFS

/// Sent from the input hook to the audio coordinator thread.
pub enum SessionEvent {
    Start,
    Stop,
    /// Preload a model into VRAM (user clicked "Use"). Handled on this thread so
    /// the `!Send` CUDA context stays single-threaded; reports progress via
    /// `model:loading` / `model:loaded` / `model:load-error` events.
    Load(String),
}

/// The capture stream plus the buffer its callback fills. Kept alive for the
/// app's lifetime and merely paused between recordings, so the OS never sees the
/// mic endpoint deactivate (closing it makes some USB devices play a sound).
struct Active {
    /// Held only to keep the stream alive — dropping it stops capture. We never
    /// call methods on it (the `capturing` flag gates recording instead).
    #[allow(dead_code)]
    stream: cpal::Stream,
    buffer: Arc<Mutex<Vec<f32>>>,
    in_rate: u32,
    /// The device name we built this stream for, so we can detect when the user
    /// switches input devices in settings and rebuild.
    requested_device: Option<String>,
    /// The physical/default device that CPAL actually opened. For a missing
    /// saved device we fall back to the current system default and track it here.
    actual_device_name: Option<String>,
    using_fallback: bool,
    /// Set by CPAL's stream error callback when the endpoint disappears or the
    /// stream otherwise becomes unusable. Checked on the next recording start.
    stream_failed: Arc<AtomicBool>,
}

/// List input device names for the settings picker.
pub fn list_input_devices() -> Vec<String> {
    let host = cpal::default_host();
    match host.input_devices() {
        Ok(devs) => devs.filter_map(|d| d.name().ok()).collect(),
        Err(_) => Vec::new(),
    }
}

/// Coordinator loop: owns the cpal stream and turns Start/Stop into a captured,
/// resampled 16 kHz mono buffer. Runs on its own thread for the lifetime of the app.
pub fn run(app: AppHandle, rx: Receiver<SessionEvent>) {
    let mut active: Option<Active> = None;
    // The capture stream runs continuously once opened; this flag tells its
    // callback whether to actually record into the buffer. Stopping/starting the
    // stream per session makes some USB mics play a sound, so we never do that.
    let capturing = Arc::new(AtomicBool::new(false));

    // Lazily-loaded Whisper context lives here for the app's lifetime.
    let models_dir = app
        .path()
        .app_data_dir()
        .map(|d| d.join("models"))
        .unwrap_or_else(|_| std::path::PathBuf::from("models"));
    let mut transcriber = Transcriber::new(models_dir);

    while let Ok(event) = rx.recv() {
        match event {
            SessionEvent::Load(id) => {
                if transcriber.is_loaded(&id) {
                    // Already resident — report current VRAM so the UI updates.
                    let _ = app.emit(
                        "model:loaded",
                        serde_json::json!({ "id": id, "ms": 0, "gpu": crate::gpu::info() }),
                    );
                    continue;
                }
                let _ = app.emit("model:loading", serde_json::json!({ "id": id }));
                let t0 = std::time::Instant::now();
                match transcriber.ensure_loaded(&id) {
                    Ok(()) => {
                        let ms = t0.elapsed().as_millis() as u64;
                        println!("[transcribe] loaded '{id}' into VRAM in {ms}ms");
                        let _ = app.emit(
                            "model:loaded",
                            serde_json::json!({ "id": id, "ms": ms, "gpu": crate::gpu::info() }),
                        );
                    }
                    Err(e) => {
                        eprintln!("[transcribe] load '{id}' failed: {e}");
                        let _ = app.emit(
                            "model:load-error",
                            serde_json::json!({ "id": id, "error": e }),
                        );
                    }
                }
            }
            SessionEvent::Start => {
                let device = app
                    .state::<SettingsState>()
                    .0
                    .lock()
                    .unwrap()
                    .audio_device
                    .clone();
                // Build the stream once, or rebuild only if the user picked a
                // different input device. The stream then runs for the app's
                // lifetime — we never stop it, so the mic never makes a sound.
                let need_new = match &active {
                    Some(a) => stream_needs_rebuild(a, &device),
                    None => true,
                };
                if need_new {
                    // Drop a dead/stale stream before opening the replacement so
                    // the OS can release the old endpoint cleanly.
                    capturing.store(false, Ordering::SeqCst);
                    active = None;
                    match start_stream(device, capturing.clone()) {
                        Ok(a) => active = Some(a),
                        Err(e) => {
                            eprintln!("[audio] failed to start capture: {e}");
                            let _ = app.emit("pill:state", "error");
                            continue;
                        }
                    }
                }
                if let Some(a) = active.as_ref() {
                    a.buffer.lock().unwrap().clear();
                    // Open the gate: the running callback now records into buffer.
                    capturing.store(true, Ordering::SeqCst);
                }
            }
            SessionEvent::Stop => {
                let Some(a) = active.as_ref() else { continue };
                // Close the gate so the still-running callback discards audio.
                // The stream keeps playing, so the device never sees capture stop.
                capturing.store(false, Ordering::SeqCst);
                let in_rate = a.in_rate;
                let mono = {
                    let mut b = a.buffer.lock().unwrap();
                    let snapshot = b.clone();
                    b.clear();
                    snapshot
                };

                let dur = if in_rate > 0 {
                    mono.len() as f32 / in_rate as f32
                } else {
                    0.0
                };
                let samples_16k = match resample_to_16k(&mono, in_rate) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("[audio] resample failed: {e}");
                        Vec::new()
                    }
                };
                println!(
                    "[audio] captured {dur:.2}s @ {in_rate}Hz -> {} samples @16k",
                    samples_16k.len()
                );

                // Speech gate: Whisper invents words ("you", "Thank you", ...)
                // when handed silence or a sliver of audio, which is what made a
                // quick toggle paste random text. Reject clips that are too short
                // or too quiet to plausibly contain speech. The RMS floor is set
                // low on purpose so whispers and quiet speech still pass; the
                // model's own per-segment no-speech check (in transcribe) is the
                // second line of defense for borderline cases.
                if !samples_16k.is_empty() {
                    let secs = samples_16k.len() as f32 / TARGET_RATE as f32;
                    let rms = (samples_16k.iter().map(|s| s * s).sum::<f32>()
                        / samples_16k.len() as f32)
                        .sqrt();
                    if secs < MIN_SPEECH_SECONDS || rms < SPEECH_RMS_FLOOR {
                        println!("[audio] skipped (no speech): {secs:.2}s rms={rms:.4}");
                        let _ = app.emit("pill:state", "idle");
                        continue;
                    }
                }

                // Pull the active model + language + paste behavior.
                let (model, language, restore_clipboard) = {
                    let s = app.state::<SettingsState>();
                    let g = s.0.lock().unwrap();
                    (g.model.clone(), g.language.clone(), g.restore_clipboard)
                };

                let Some(model_id) = model else {
                    // No model selected yet — nudge the user to pick one.
                    let _ = app.emit("transcribe:no-model", ());
                    let _ = app.emit("pill:state", "idle");
                    continue;
                };
                if samples_16k.is_empty() {
                    let _ = app.emit("pill:state", "idle");
                    continue;
                }

                let _ = app.emit("pill:state", "busy");
                match transcriber.transcribe(&model_id, &language, &samples_16k) {
                    Ok(text) => {
                        println!("[transcribe] {text:?}");
                        // Whisper output often has a leading space and no trailing
                        // one. Normalize: trim, then add a single trailing space so
                        // back-to-back dictations don't run together ("a.b" -> "a. b").
                        let text = text.trim();
                        if !text.is_empty() {
                            let saved = {
                                let h = app.state::<TranscriptHistoryState>();
                                let mut history = h.0.lock().unwrap();
                                crate::history::add(&app, &mut history, text.to_string())
                            };
                            match saved {
                                Ok(entry) => {
                                    let _ = app.emit("transcript:new", &entry);
                                    let _ = app.emit_to("main", "transcript:new", &entry);
                                }
                                Err(e) => eprintln!("[transcript] failed to save history: {e}"),
                            }
                            let to_paste = format!("{text} ");
                            if let Err(e) = crate::inject::paste_text(&to_paste, restore_clipboard)
                            {
                                eprintln!("[inject] {e}");
                            }
                        }
                        let _ = app.emit("pill:state", "idle");
                    }
                    Err(e) => {
                        eprintln!("[transcribe] {e}");
                        let _ = app.emit("transcribe:error", e);
                        let _ = app.emit("pill:state", "error");
                    }
                }
            }
        }
    }
}

/// Open `device_name` (or the system default) and start a stream that records
/// into a mono buffer whenever `capturing` is set. The stream runs continuously;
/// the flag — not stopping the stream — is what gates a recording.
fn start_stream(device_name: Option<String>, capturing: Arc<AtomicBool>) -> Result<Active, String> {
    // Remember what was requested so the coordinator can detect device changes.
    let requested = device_name.clone();
    let host = cpal::default_host();
    let (device, using_fallback) = match device_name {
        Some(name) => {
            let selected = host
                .input_devices()
                .ok()
                .and_then(|mut devs| devs.find(|d| d.name().map(|n| n == name).unwrap_or(false)));
            match selected {
                Some(device) => (device, false),
                None => (
                    host.default_input_device().ok_or_else(|| {
                        format!(
                            "selected input device '{name}' is unavailable and no default input device exists"
                        )
                    })?,
                    true,
                ),
            }
        }
        None => (
            host.default_input_device()
                .ok_or("no default input device")?,
            false,
        ),
    };
    let actual_device_name = device.name().ok();
    if using_fallback {
        eprintln!(
            "[audio] requested input {:?} unavailable; using default input {:?}",
            requested, actual_device_name
        );
    }

    let config = device.default_input_config().map_err(|e| e.to_string())?;
    let in_rate = config.sample_rate().0;
    let channels = config.channels() as usize;
    let sample_format = config.sample_format();

    let buffer = Arc::new(Mutex::new(Vec::<f32>::new()));
    let buf = buffer.clone();
    let stream_failed = Arc::new(AtomicBool::new(false));
    let failed = stream_failed.clone();
    let err_fn = move |e| {
        eprintln!("[audio] stream error: {e}");
        failed.store(true, Ordering::SeqCst);
    };
    let cfg: cpal::StreamConfig = config.into();

    let stream = match sample_format {
        cpal::SampleFormat::F32 => {
            let cap = capturing.clone();
            device.build_input_stream(
                &cfg,
                move |data: &[f32], _: &_| {
                    if !cap.load(Ordering::Relaxed) {
                        return;
                    }
                    push_mono(&buf, data, channels);
                },
                err_fn,
                None,
            )
        }
        cpal::SampleFormat::I16 => {
            let cap = capturing.clone();
            device.build_input_stream(
                &cfg,
                move |data: &[i16], _: &_| {
                    if !cap.load(Ordering::Relaxed) {
                        return;
                    }
                    let f: Vec<f32> = data.iter().map(|s| *s as f32 / i16::MAX as f32).collect();
                    push_mono(&buf, &f, channels);
                },
                err_fn,
                None,
            )
        }
        cpal::SampleFormat::U16 => {
            let cap = capturing.clone();
            device.build_input_stream(
                &cfg,
                move |data: &[u16], _: &_| {
                    if !cap.load(Ordering::Relaxed) {
                        return;
                    }
                    let f: Vec<f32> = data
                        .iter()
                        .map(|s| (*s as f32 / u16::MAX as f32) * 2.0 - 1.0)
                        .collect();
                    push_mono(&buf, &f, channels);
                },
                err_fn,
                None,
            )
        }
        other => return Err(format!("unsupported sample format: {other:?}")),
    }
    .map_err(|e| e.to_string())?;

    stream.play().map_err(|e| e.to_string())?;
    Ok(Active {
        stream,
        buffer,
        in_rate,
        requested_device: requested,
        actual_device_name,
        using_fallback,
        stream_failed,
    })
}

fn stream_needs_rebuild(active: &Active, requested_device: &Option<String>) -> bool {
    let available = list_input_devices();
    let current_default = default_input_device_name();
    stream_needs_rebuild_for_names(
        &active.requested_device,
        &active.actual_device_name,
        active.using_fallback,
        active.stream_failed.load(Ordering::SeqCst),
        requested_device,
        &available,
        current_default.as_deref(),
    )
}

fn default_input_device_name() -> Option<String> {
    cpal::default_host()
        .default_input_device()
        .and_then(|d| d.name().ok())
}

fn stream_needs_rebuild_for_names(
    active_requested: &Option<String>,
    active_actual: &Option<String>,
    active_using_fallback: bool,
    stream_failed: bool,
    requested_device: &Option<String>,
    available: &[String],
    current_default: Option<&str>,
) -> bool {
    if active_requested != requested_device || stream_failed {
        return true;
    }

    if let Some(actual) = active_actual {
        if !available.iter().any(|name| name == actual) {
            return true;
        }
    }

    match requested_device {
        Some(requested) => {
            let requested_available = available.iter().any(|name| name == requested);
            if requested_available {
                active_using_fallback || active_actual.as_deref() != Some(requested.as_str())
            } else if active_using_fallback {
                match (active_actual.as_deref(), current_default) {
                    (Some(actual), Some(default)) => actual != default,
                    (None, Some(_)) => true,
                    (_, None) => true,
                }
            } else {
                true
            }
        }
        None => match (active_actual.as_deref(), current_default) {
            (Some(actual), Some(default)) => actual != default,
            (Some(_), None) => true,
            _ => false,
        },
    }
}

/// Append `data` (interleaved, `channels`-wide) to `buf`, downmixed to mono.
fn push_mono(buf: &Arc<Mutex<Vec<f32>>>, data: &[f32], channels: usize) {
    let mut b = buf.lock().unwrap();
    if channels <= 1 {
        b.extend_from_slice(data);
    } else {
        for frame in data.chunks(channels) {
            let sum: f32 = frame.iter().sum();
            b.push(sum / frame.len() as f32);
        }
    }
}

/// Resample mono f32 from `in_rate` to 16 kHz. Pass-through when already 16 kHz.
fn resample_to_16k(input: &[f32], in_rate: u32) -> Result<Vec<f32>, String> {
    if in_rate == TARGET_RATE || input.is_empty() {
        return Ok(input.to_vec());
    }

    let chunk = 1024usize;
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f32>::new(
        TARGET_RATE as f64 / in_rate as f64,
        2.0,
        params,
        chunk,
        1, // mono
    )
    .map_err(|e| e.to_string())?;

    let mut out = Vec::with_capacity(input.len() * TARGET_RATE as usize / in_rate as usize + chunk);
    let mut pos = 0;
    while pos < input.len() {
        let end = (pos + chunk).min(input.len());
        let mut frame = input[pos..end].to_vec();
        frame.resize(chunk, 0.0); // zero-pad the final short chunk
        let resampled = resampler
            .process(&[frame], None)
            .map_err(|e| e.to_string())?;
        out.extend_from_slice(&resampled[0]);
        pos += chunk;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opt(value: Option<&str>) -> Option<String> {
        value.map(|v| v.to_string())
    }

    fn names(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| (*v).to_string()).collect()
    }

    fn needs_rebuild(
        active_requested: Option<&str>,
        active_actual: Option<&str>,
        active_using_fallback: bool,
        stream_failed: bool,
        requested_device: Option<&str>,
        available: &[&str],
        current_default: Option<&str>,
    ) -> bool {
        stream_needs_rebuild_for_names(
            &opt(active_requested),
            &opt(active_actual),
            active_using_fallback,
            stream_failed,
            &opt(requested_device),
            &names(available),
            current_default,
        )
    }

    #[test]
    fn resample_48k_to_16k_length() {
        let input = vec![0.0f32; 48_000]; // 1 second @ 48 kHz
        let out = resample_to_16k(&input, 48_000).unwrap();
        // Expect ~16k samples; allow slack for chunk zero-padding at the tail.
        assert!(
            (out.len() as i32 - 16_000).abs() < 2_000,
            "expected ~16000, got {}",
            out.len()
        );
    }

    #[test]
    fn resample_passthrough_at_16k() {
        let input = vec![0.1f32; 1_000];
        let out = resample_to_16k(&input, 16_000).unwrap();
        assert_eq!(out.len(), 1_000);
    }

    #[test]
    fn resample_empty_is_empty() {
        let out = resample_to_16k(&[], 48_000).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn stream_stays_active_when_default_device_is_unchanged() {
        assert!(!needs_rebuild(
            None,
            Some("Desk Mic"),
            false,
            false,
            None,
            &["Desk Mic"],
            Some("Desk Mic"),
        ));
    }

    #[test]
    fn stream_rebuilds_after_cpal_reports_error() {
        assert!(needs_rebuild(
            None,
            Some("Desk Mic"),
            false,
            true,
            None,
            &["Desk Mic"],
            Some("Desk Mic"),
        ));
    }

    #[test]
    fn stream_rebuilds_when_actual_device_disappears() {
        assert!(needs_rebuild(
            None,
            Some("Old Mic"),
            false,
            false,
            None,
            &["New Mic"],
            Some("New Mic"),
        ));
    }

    #[test]
    fn stream_rebuilds_when_system_default_changes() {
        assert!(needs_rebuild(
            None,
            Some("Old Mic"),
            false,
            false,
            None,
            &["Old Mic", "New Mic"],
            Some("New Mic"),
        ));
    }

    #[test]
    fn fallback_stream_tracks_default_until_selected_device_returns() {
        assert!(!needs_rebuild(
            Some("Travel Mic"),
            Some("Laptop Mic"),
            true,
            false,
            Some("Travel Mic"),
            &["Laptop Mic"],
            Some("Laptop Mic"),
        ));

        assert!(needs_rebuild(
            Some("Travel Mic"),
            Some("Laptop Mic"),
            true,
            false,
            Some("Travel Mic"),
            &["Travel Mic", "Laptop Mic"],
            Some("Laptop Mic"),
        ));

        assert!(needs_rebuild(
            Some("Travel Mic"),
            Some("Laptop Mic"),
            true,
            false,
            Some("Travel Mic"),
            &["Laptop Mic", "Dock Mic"],
            Some("Dock Mic"),
        ));
    }
}
