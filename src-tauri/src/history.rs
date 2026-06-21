//! Persistent transcription history.
//!
//! Kept separate from user settings so settings UI round-trips cannot overwrite
//! or stale out transcript history. The list is tiny: newest-first, capped at 10.

use std::path::PathBuf;
use std::sync::Mutex;

use tauri::{AppHandle, Manager};

use crate::settings::TranscriptEntry;

pub struct TranscriptHistoryState(pub Mutex<Vec<TranscriptEntry>>);

fn history_path(app: &AppHandle) -> PathBuf {
    app.path()
        .app_config_dir()
        .expect("app config dir unavailable")
        .join("transcripts.json")
}

pub fn load(app: &AppHandle, fallback: &[TranscriptEntry]) -> Vec<TranscriptEntry> {
    let path = history_path(app);
    match std::fs::read_to_string(&path) {
        Ok(txt) => serde_json::from_str(&txt).unwrap_or_else(|_| fallback.to_vec()),
        Err(_) => fallback.to_vec(),
    }
}

fn save(app: &AppHandle, history: &[TranscriptEntry]) -> Result<(), String> {
    let path = history_path(app);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let txt = serde_json::to_string_pretty(history).map_err(|e| e.to_string())?;
    std::fs::write(&path, txt).map_err(|e| e.to_string())
}

pub fn add(
    app: &AppHandle,
    history: &mut Vec<TranscriptEntry>,
    text: String,
) -> Result<TranscriptEntry, String> {
    let created_at_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_millis() as u64;
    let entry = TranscriptEntry {
        id: created_at_ms.to_string(),
        text,
        created_at_ms,
    };
    history.insert(0, entry.clone());
    history.truncate(10);
    save(app, history)?;
    Ok(entry)
}
