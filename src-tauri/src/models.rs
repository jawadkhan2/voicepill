//! Model catalog + downloader.
//!
//! Three engines are offered. Whisper (ggml) models are pulled from the official
//! `ggerganov/whisper.cpp` repo on Hugging Face and stored as `ggml-<id>.bin`.
//! MedASR (a Google LASR Conformer-CTC model, exported to ONNX) is pulled from a
//! VoicePill GitHub release and stored as `<id>.onnx` + `<id>-tokenizer.json`.
//! Parakeet (NVIDIA FastConformer-TDT, ONNX, run on CUDA) is pulled from a
//! VoicePill GitHub release into `<id>/`, together with the cuDNN DLLs its CUDA
//! provider needs (`cuda-runtime/`, shared by every Parakeet model).
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
    Parakeet,
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
        id: PARAKEET_V2,
        label: "Parakeet TDT 0.6B v2",
        size_mb: PARAKEET_SIZE_MB,
        engine: Engine::Parakeet,
    },
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
/// GitHub release hosting the Parakeet ONNX graphs + the cuDNN runtime.
const PARAKEET_BASE: &str = "https://github.com/jawadkhan2/voicepill/releases/download/parakeet-v1";

pub const PARAKEET_V2: &str = "parakeet-tdt-0.6b-v2";
/// Model graphs (~1.2 GB) + cuDNN runtime (~0.7 GB).
const PARAKEET_SIZE_MB: u32 = 1900;

/// cuDNN 9.28 (CUDA 13 build) DLLs the CUDA execution provider loads. cuDNN
/// pulls its sub-libraries in lazily per operation (the LSTM needs `adv`, the
/// conv front end needs `engines_tensor_ir`, ...), so the full set ships rather
/// than a pruned one that fails on some op later. Shared by every Parakeet
/// model, so deleting a model leaves them in place.
const CUDA_RUNTIME_FILES: &[&str] = &[
    "cudnn64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_engines_tensor_ir64_9.dll",
    "cudnn_ext64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_ops64_9.dll",
];

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
        Engine::Parakeet => {
            let mut files: Vec<(String, String)> = crate::parakeet::MODEL_FILES
                .iter()
                .map(|f| (format!("{}/{f}", m.id), format!("{PARAKEET_BASE}/{}-{f}", m.id)))
                .collect();
            files.extend(CUDA_RUNTIME_FILES.iter().map(|f| {
                (
                    format!("{}/{f}", crate::ort_env::CUDA_RUNTIME_DIR),
                    format!("{PARAKEET_BASE}/{f}"),
                )
            }));
            files
        }
    }
}

/// Files shared between models (the CUDA runtime), which `delete` keeps.
fn is_shared_file(name: &str) -> bool {
    name.starts_with(&format!("{}/", crate::ort_env::CUDA_RUNTIME_DIR))
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
        if is_shared_file(&name) {
            continue;
        }
        let path = dir.join(&name);
        if path.exists() {
            std::fs::remove_file(&path).map_err(|e| e.to_string())?;
        }
        // Also drop any half-finished download so it doesn't sit on disk.
        let _ = std::fs::remove_file(path.with_extension("part"));
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
            // Already on disk (e.g. the shared CUDA runtime from another
            // Parakeet model, or a file finished before a cancelled download).
            if dest.exists() {
                continue;
            }
            if let Err(e) = run_download(&app, &id, url, &dir, &dest) {
                crate::vlog!("[models] download '{id}' ({name}) failed: {e}");
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
    // Multi-file models live in a subfolder (`parakeet-…/`, `cuda-runtime/`).
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }

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
                crate::vlog!("[models] '{id}' attempt {attempt}/{MAX_ATTEMPTS} failed: {last_err}");
                // Keep the .part: the next attempt (or a later click on
                // Download) resumes from where this one stopped.
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
    // Resume a previous partial download if there is one (HF and GitHub
    // release assets both honour Range). A 200 reply means the server ignored
    // the range, so start over.
    let have = std::fs::metadata(tmp).map(|m| m.len()).unwrap_or(0);
    let mut req = client.get(url);
    if have > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={have}-"));
    }
    let mut resp = req.send().map_err(err_chain)?;
    if resp.status() == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
        // Stale/oversized .part — discard and fetch fresh next attempt.
        let _ = std::fs::remove_file(tmp);
        return Err("partial download was invalid; restarting".into());
    }
    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let resuming = have > 0 && resp.status() == reqwest::StatusCode::PARTIAL_CONTENT;
    let offset = if resuming { have } else { 0 };
    let total = resp.content_length().map(|n| n + offset).unwrap_or(0);

    let mut file = if resuming {
        std::fs::OpenOptions::new()
            .append(true)
            .open(tmp)
            .map_err(|e| e.to_string())?
    } else {
        std::fs::File::create(tmp).map_err(|e| e.to_string())?
    };
    if resuming {
        crate::vlog!("[models] '{id}' resuming at {have} bytes");
    }
    let mut buf = [0u8; 64 * 1024];
    let mut received: u64 = offset;
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
