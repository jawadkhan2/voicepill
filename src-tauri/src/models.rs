//! Model catalog + downloader.
//!
//! Two engines are offered. Whisper (ggml) models are pulled from the official
//! `ggerganov/whisper.cpp` repo on Hugging Face and stored as `ggml-<id>.bin`.
//! MedASR (a Google LASR Conformer-CTC model, exported to ONNX) is pulled from a
//! VoicePill GitHub release and stored as `<id>.onnx` + `<id>-tokenizer.json`.
//! Downloads run on a background thread and stream progress to the UI via
//! `model:progress` / `model:done` / `model:error` events.

use std::io::Read;
use std::path::PathBuf;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};

/// Which inference backend a model runs on.
#[derive(Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Engine {
    Whisper,
    MedAsr,
}

/// One entry in the built-in model catalog.
struct ModelEntry {
    id: &'static str,
    label: &'static str,
    size_mb: u32,
    engine: Engine,
}

/// Models offered in the picker, smallest → largest (Whisper), then specialty.
const CATALOG: &[ModelEntry] = &[
    ModelEntry {
        id: "tiny",
        label: "Tiny",
        size_mb: 75,
        engine: Engine::Whisper,
    },
    ModelEntry {
        id: "base",
        label: "Base",
        size_mb: 142,
        engine: Engine::Whisper,
    },
    ModelEntry {
        id: "small",
        label: "Small",
        size_mb: 466,
        engine: Engine::Whisper,
    },
    ModelEntry {
        id: "medium",
        label: "Medium",
        size_mb: 1536,
        engine: Engine::Whisper,
    },
    ModelEntry {
        id: "large-v3-turbo",
        label: "Large v3 Turbo",
        size_mb: 1620,
        engine: Engine::Whisper,
    },
    ModelEntry {
        id: "large-v3",
        label: "Large v3",
        size_mb: 3094,
        engine: Engine::Whisper,
    },
    ModelEntry {
        id: "medasr",
        label: "MedASR (Medical · English)",
        size_mb: 853,
        engine: Engine::MedAsr,
    },
];

const HF_BASE: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";
/// GitHub release hosting the MedASR ONNX + tokenizer assets.
const MEDASR_BASE: &str = "https://github.com/jawadkhan2/voicepill/releases/download/medasr-v1";

fn entry(id: &str) -> Option<&'static ModelEntry> {
    CATALOG.iter().find(|m| m.id == id)
}

/// The inference engine for a model id (defaults to Whisper for unknown ids).
pub fn engine_of(id: &str) -> Engine {
    entry(id).map(|m| m.engine).unwrap_or(Engine::Whisper)
}

/// Files that make up a model on disk, as `(filename, download url)` pairs.
/// Whisper is a single ggml blob; MedASR is an ONNX graph plus its tokenizer.
fn files_for(m: &ModelEntry) -> Vec<(String, String)> {
    match m.engine {
        Engine::Whisper => vec![(
            format!("ggml-{}.bin", m.id),
            format!("{HF_BASE}/ggml-{}.bin", m.id),
        )],
        Engine::MedAsr => vec![
            (
                format!("{}.onnx", m.id),
                format!("{MEDASR_BASE}/medasr.onnx"),
            ),
            (
                format!("{}-tokenizer.json", m.id),
                format!("{MEDASR_BASE}/medasr-tokenizer.json"),
            ),
            // 6-gram LM for beam-search decoding (~2% absolute WER win). The
            // backend degrades to greedy decode if this file is missing.
            (
                format!("{}-lm.bin", m.id),
                format!("{MEDASR_BASE}/medasr-lm.bin"),
            ),
        ],
    }
}

/// Serialized to the UI: catalog entry + whether the file is on disk.
#[derive(Serialize)]
pub struct ModelInfo {
    id: String,
    label: String,
    size_mb: u32,
    engine: Engine,
    downloaded: bool,
}

fn models_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join("models");
    Ok(dir)
}

fn is_known(id: &str) -> bool {
    CATALOG.iter().any(|m| m.id == id)
}

pub fn validate_id(id: &str) -> Result<(), String> {
    if is_known(id) {
        Ok(())
    } else {
        Err(format!("unknown model: {id}"))
    }
}

/// Whether every file that makes up `id` is present on disk.
fn is_downloaded(app: &AppHandle, m: &ModelEntry) -> bool {
    let dir = match models_dir(app) {
        Ok(d) => d,
        Err(_) => return false,
    };
    files_for(m).iter().all(|(name, _)| dir.join(name).exists())
}

/// List the catalog with on-disk status for each model.
pub fn list(app: &AppHandle) -> Vec<ModelInfo> {
    CATALOG
        .iter()
        .map(|m| ModelInfo {
            id: m.id.to_string(),
            label: m.label.to_string(),
            size_mb: m.size_mb,
            engine: m.engine,
            downloaded: is_downloaded(app, m),
        })
        .collect()
}

/// Delete all files for a downloaded model.
pub fn delete(app: &AppHandle, id: &str) -> Result<(), String> {
    validate_id(id)?;
    let m = entry(id).ok_or_else(|| format!("unknown model: {id}"))?;
    let dir = models_dir(app)?;
    for (name, _) in files_for(m) {
        let path = dir.join(&name);
        if path.exists() {
            std::fs::remove_file(&path).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

/// Begin downloading `id` on a background thread. Returns immediately; progress
/// is reported via events. Re-downloads overwrite via a temp file + rename.
pub fn download(app: &AppHandle, id: String) -> Result<(), String> {
    validate_id(&id)?;
    let dir = models_dir(app)?;
    let m = entry(&id).ok_or_else(|| format!("unknown model: {id}"))?;
    let files = files_for(m);
    let app = app.clone();

    std::thread::spawn(move || {
        // Download every file that makes up the model (Whisper: one blob; MedASR:
        // ONNX + tokenizer). A single `model:done` fires once all files land.
        for (name, url) in &files {
            let dest = dir.join(name);
            if let Err(e) = run_download(&app, &id, url, &dir, &dest) {
                eprintln!("[models] download '{id}' ({name}) failed: {e}");
                let _ = app.emit("model:error", serde_json::json!({ "id": id, "error": e }));
                return;
            }
        }
        let _ = app.emit("model:done", serde_json::json!({ "id": id }));
    });
    Ok(())
}

/// Number of times to retry a download before giving up.
const MAX_ATTEMPTS: u32 = 4;

fn run_download(
    app: &AppHandle,
    id: &str,
    url: &str,
    dir: &PathBuf,
    dest: &PathBuf,
) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;

    // One client, reused across retries. A User-Agent + connect timeout make HF's
    // CDNs behave; without retries a single reset connection (common when several
    // downloads start at once) would fail the whole download instantly.
    let client = reqwest::blocking::Client::builder()
        .user_agent(concat!("VoicePill/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| e.to_string())?;

    let tmp = dest.with_extension("part");
    let mut last_err = String::new();

    for attempt in 1..=MAX_ATTEMPTS {
        match attempt_download(app, id, url, &client, &tmp) {
            Ok(_received) => {
                std::fs::rename(&tmp, dest).map_err(|e| e.to_string())?;
                // Per-file success is silent; `download` emits one `model:done`
                // once every file for the model has landed.
                return Ok(());
            }
            Err(e) => {
                last_err = e;
                eprintln!("[models] '{id}' attempt {attempt}/{MAX_ATTEMPTS} failed: {last_err}");
                let _ = std::fs::remove_file(&tmp); // start the next attempt clean
                if attempt < MAX_ATTEMPTS {
                    std::thread::sleep(std::time::Duration::from_millis(500 * attempt as u64));
                }
            }
        }
    }
    Err(format!("after {MAX_ATTEMPTS} attempts: {last_err}"))
}

/// A single download attempt: stream the body into `tmp`, returning bytes written.
fn attempt_download(
    app: &AppHandle,
    id: &str,
    url: &str,
    client: &reqwest::blocking::Client,
    tmp: &PathBuf,
) -> Result<u64, String> {
    let mut resp = client.get(url).send().map_err(err_chain)?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let total = resp.content_length().unwrap_or(0);

    let mut file = std::fs::File::create(tmp).map_err(|e| e.to_string())?;
    let mut buf = [0u8; 64 * 1024];
    let mut received: u64 = 0;
    let mut last_emit = 0u64;
    loop {
        let n = resp.read(&mut buf).map_err(err_chain)?;
        if n == 0 {
            break;
        }
        std::io::Write::write_all(&mut file, &buf[..n]).map_err(|e| e.to_string())?;
        received += n as u64;
        // Throttle progress events to ~every 1 MB.
        if received - last_emit >= 1_048_576 {
            last_emit = received;
            let _ = app.emit(
                "model:progress",
                serde_json::json!({ "id": id, "received": received, "total": total }),
            );
        }
    }
    std::io::Write::flush(&mut file).map_err(|e| e.to_string())?;
    // Guard against a body that closed early (proxy/CDN reset on the multi-GB
    // blobs): a short read that surfaces as clean EOF would otherwise rename a
    // truncated file into place and report the model as fully downloaded.
    if total > 0 && received != total {
        return Err(format!(
            "incomplete download: {received}/{total} bytes"
        ));
    }
    Ok(received)
}

/// Flatten an error's `source()` chain into one string so the real cause
/// (e.g. "connection reset", "timed out") survives, not just "error sending request".
fn err_chain<E: std::error::Error>(e: E) -> String {
    let mut msg = e.to_string();
    let mut src = e.source();
    while let Some(inner) = src {
        msg.push_str(": ");
        msg.push_str(&inner.to_string());
        src = inner.source();
    }
    msg
}
