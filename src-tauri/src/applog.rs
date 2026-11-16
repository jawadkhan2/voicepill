//! Tiny persistent log file.
//!
//! Release builds are a GUI-subsystem exe with no console, so every
//! `println!`/`eprintln!` diagnostic in the pipeline used to vanish — when a
//! dictation got dropped there was no way to tell whether the speech gate, the
//! model, or the paste step ate it. Everything now goes through [`line`], which
//! appends a timestamped line to `<app_log_dir>/voicepill.log` (and still echoes
//! to stderr for `tauri dev`).
//!
//! whisper.cpp / ggml logs are routed here too (via whisper-rs's `log_backend`)
//! so the file records which compute backend the model actually loaded on
//! (e.g. `ggml_cuda_init: found 1 CUDA devices` / `using CUDA0 backend`).
//!
//! The file is rotated once at startup when it exceeds `MAX_BYTES` (the old one
//! is kept as `voicepill.log.old`), so it never grows without bound.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use tauri::{AppHandle, Manager};

const MAX_BYTES: u64 = 2 * 1024 * 1024;

static FILE: OnceLock<Mutex<File>> = OnceLock::new();

/// Open (and rotate) the log file, and route whisper.cpp logs into it. Safe to
/// call once at startup; before it runs, [`line`] just echoes to stderr.
pub fn init(app: &AppHandle) {
    let Some(path) = log_path(app) else { return };
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if std::fs::metadata(&path).map(|m| m.len() > MAX_BYTES).unwrap_or(false) {
        let _ = std::fs::rename(&path, path.with_extension("log.old"));
    }
    if let Ok(f) = OpenOptions::new().create(true).append(true).open(&path) {
        let _ = FILE.set(Mutex::new(f));
    }

    // whisper.cpp → `log` crate → here. Only whisper_rs targets are kept so the
    // rest of the dependency tree's `log` output doesn't flood the file.
    if log::set_logger(&WHISPER_LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Info);
    }
    whisper_rs::install_logging_hooks();

    line(&format!(
        "---- VoicePill {} starting ----",
        env!("CARGO_PKG_VERSION")
    ));
}

fn log_path(app: &AppHandle) -> Option<PathBuf> {
    app.path()
        .app_log_dir()
        .ok()
        .map(|d| d.join("voicepill.log"))
}

/// Append one timestamped line to the log (and stderr).
pub fn line(msg: &str) {
    eprintln!("{msg}");
    let Some(f) = FILE.get() else { return };
    let ts = timestamp();
    if let Ok(mut f) = f.lock() {
        let _ = writeln!(f, "{ts} {msg}");
    }
}

/// `format!`-style front end for [`line`].
#[macro_export]
macro_rules! vlog {
    ($($arg:tt)*) => { $crate::applog::line(&format!($($arg)*)) };
}

/// Local-time-free UTC `HH:MM:SS.mmm` plus the date — no chrono dependency.
fn timestamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let ms = now.subsec_millis();
    let days = secs / 86_400;
    let (y, m, d) = civil_from_days(days as i64);
    let rem = secs % 86_400;
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{ms:03}Z",
        rem / 3600,
        (rem / 60) % 60,
        rem % 60
    )
}

/// Days since 1970-01-01 → (year, month, day). Howard Hinnant's algorithm.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

struct WhisperLogger;
static WHISPER_LOGGER: WhisperLogger = WhisperLogger;

impl log::Log for WhisperLogger {
    fn enabled(&self, meta: &log::Metadata) -> bool {
        meta.target().starts_with("whisper_rs") && meta.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            let msg = record.args().to_string();
            let msg = msg.trim();
            if !msg.is_empty() {
                line(&format!("[whisper.cpp] {msg}"));
            }
        }
    }
    fn flush(&self) {}
}

#[cfg(test)]
mod tests {
    use super::civil_from_days;

    #[test]
    fn civil_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_723), (2024, 1, 1));
        assert_eq!(civil_from_days(20_773), (2026, 11, 16));
    }
}
