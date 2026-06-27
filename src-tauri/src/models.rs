//! Whisper model catalog + downloader.
//!
//! ggml models are pulled on demand from the official `ggerganov/whisper.cpp`
//! repo on Hugging Face and stored as `ggml-<id>.bin` in the app data dir.
//! Downloads run on a background thread and stream progress to the UI via
//! `model:progress` / `model:done` / `model:error` events.

use std::io::Read;
use std::path::PathBuf;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};

/// One entry in the built-in model catalog.
struct ModelEntry {
    id: &'static str,
    label: &'static str,
    size_mb: u32,
}

/// Models offered in the picker, smallest → largest.
const CATALOG: &[ModelEntry] = &[
    ModelEntry {
        id: "tiny",
        label: "Tiny",
        size_mb: 75,
    },
    ModelEntry {
        id: "base",
        label: "Base",
        size_mb: 142,
    },
    ModelEntry {
        id: "small",
        label: "Small",
        size_mb: 466,
    },
    ModelEntry {
        id: "medium",
        label: "Medium",
        size_mb: 1536,
    },
    ModelEntry {
        id: "large-v3-turbo",
        label: "Large v3 Turbo",
        size_mb: 1620,
    },
    ModelEntry {
        id: "large-v3",
        label: "Large v3",
        size_mb: 3094,
    },
];

const HF_BASE: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

/// Serialized to the UI: catalog entry + whether the file is on disk.
#[derive(Serialize)]
pub struct ModelInfo {
    id: String,
    label: String,
    size_mb: u32,
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

fn model_file(app: &AppHandle, id: &str) -> Result<PathBuf, String> {
    Ok(models_dir(app)?.join(format!("ggml-{id}.bin")))
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

/// List the catalog with on-disk status for each model.
pub fn list(app: &AppHandle) -> Vec<ModelInfo> {
    CATALOG
        .iter()
        .map(|m| ModelInfo {
            id: m.id.to_string(),
            label: m.label.to_string(),
            size_mb: m.size_mb,
            downloaded: model_file(app, m.id).map(|p| p.exists()).unwrap_or(false),
        })
        .collect()
}

/// Delete a downloaded model file.
pub fn delete(app: &AppHandle, id: &str) -> Result<(), String> {
    validate_id(id)?;
    let path = model_file(app, id)?;
    if path.exists() {
        std::fs::remove_file(&path).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Begin downloading `id` on a background thread. Returns immediately; progress
/// is reported via events. Re-downloads overwrite via a temp file + rename.
pub fn download(app: &AppHandle, id: String) -> Result<(), String> {
    validate_id(&id)?;
    let dir = models_dir(app)?;
    let dest = model_file(app, &id)?;
    let url = format!("{HF_BASE}/ggml-{id}.bin");
    let app = app.clone();

    std::thread::spawn(move || {
        if let Err(e) = run_download(&app, &id, &url, &dir, &dest) {
            eprintln!("[models] download '{id}' failed: {e}");
            let _ = app.emit("model:error", serde_json::json!({ "id": id, "error": e }));
        }
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
            Ok(received) => {
                std::fs::rename(&tmp, dest).map_err(|e| e.to_string())?;
                let _ = app.emit(
                    "model:done",
                    serde_json::json!({ "id": id, "received": received }),
                );
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
