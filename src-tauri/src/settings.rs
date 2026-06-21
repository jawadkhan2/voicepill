//! Persistent user settings.
//!
//! Stored as plain JSON in the OS app-config dir (`%APPDATA%\com.voicepill.app\settings.json`
//! on Windows). We roll our own tiny load/save instead of `tauri-plugin-store` so the
//! Rust side owns the schema and the webview just round-trips the struct over IPC.

use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};

/// How the trigger button behaves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TriggerMode {
    /// Record only while the button is held down.
    PushToTalk,
    /// First press starts recording, second press stops.
    Toggle,
}

impl Default for TriggerMode {
    fn default() -> Self {
        TriggerMode::PushToTalk
    }
}

/// Whether the bound trigger is a mouse button or a keyboard key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TriggerKind {
    Mouse,
    Key,
}

/// UI color theme. `System` follows the OS light/dark preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Theme {
    Light,
    Dark,
    System,
}

impl Default for Theme {
    fn default() -> Self {
        Theme::System
    }
}

/// A bound trigger: stable `code` for matching, friendly `label` for the UI.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TriggerBinding {
    pub kind: TriggerKind,
    /// Stable identifier used for event matching (e.g. `"Unknown(1)"`, `"KeyF8"`).
    pub code: String,
    /// Human-readable name shown in settings (e.g. `"Mouse 4"`, `"F8"`).
    pub label: String,
}

/// A single completed transcription shown in the history window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptEntry {
    /// Millisecond timestamp string; stable enough for frontend keys.
    pub id: String,
    pub text: String,
    pub created_at_ms: u64,
}

/// The full settings document persisted to disk and shared with the webview.
///
/// Every field is `#[serde(default)]` so older/partial files still load, and new
/// fields don't break existing installs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// The bound trigger button. `None` until the user completes first-run binding.
    #[serde(default)]
    pub trigger: Option<TriggerBinding>,
    #[serde(default)]
    pub trigger_mode: TriggerMode,
    /// Selected Whisper model id (filename stem), e.g. `"ggml-base.en"`.
    #[serde(default)]
    pub model: Option<String>,
    /// Spoken language hint; `"auto"` lets Whisper detect.
    #[serde(default = "default_language")]
    pub language: String,
    /// Selected audio input device name; `None` = system default.
    #[serde(default)]
    pub audio_device: Option<String>,
    #[serde(default)]
    pub autostart: bool,
    /// Restore the user's previous clipboard contents after pasting.
    #[serde(default = "default_true")]
    pub restore_clipboard: bool,
    // ---- Appearance ----
    /// Light / dark / follow-system color theme.
    #[serde(default)]
    pub theme: Theme,
    /// Accent color as a CSS hex string (drives buttons, active states, mic dot).
    #[serde(default = "default_accent")]
    pub accent: String,
    /// Floating pill scale multiplier (1.0 = default size).
    #[serde(default = "default_pill_scale")]
    pub pill_scale: f32,
    /// Floating pill opacity (0.0–1.0).
    #[serde(default = "default_pill_opacity")]
    pub pill_opacity: f32,
    /// When true the pill hides its text label and shows just the icon.
    #[serde(default)]
    pub pill_compact: bool,
    /// Set once the first-run bind flow has completed.
    #[serde(default)]
    pub onboarded: bool,
    /// Last on-screen pill position in physical pixels `[x, y]`; `None` = use the
    /// configured default. Persisted so the pill reappears where the user left it.
    #[serde(default)]
    pub pill_pos: Option<[f64; 2]>,
    /// Most recent completed transcriptions, newest first.
    #[serde(default)]
    pub transcript_history: Vec<TranscriptEntry>,
}

fn default_language() -> String {
    "auto".to_string()
}

fn default_true() -> bool {
    true
}

fn default_accent() -> String {
    "#6c5ce7".to_string()
}

fn default_pill_scale() -> f32 {
    1.0
}

fn default_pill_opacity() -> f32 {
    1.0
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            trigger: None,
            trigger_mode: TriggerMode::default(),
            model: None,
            language: default_language(),
            audio_device: None,
            autostart: false,
            restore_clipboard: true,
            theme: Theme::default(),
            accent: default_accent(),
            pill_scale: default_pill_scale(),
            pill_opacity: default_pill_opacity(),
            pill_compact: false,
            onboarded: false,
            pill_pos: None,
            transcript_history: Vec::new(),
        }
    }
}

/// Tauri-managed state wrapper around the live settings.
pub struct SettingsState(pub Mutex<Settings>);

fn settings_path(app: &AppHandle) -> PathBuf {
    app.path()
        .app_config_dir()
        .expect("app config dir unavailable")
        .join("settings.json")
}

/// Load settings from disk, falling back to defaults if missing/corrupt.
pub fn load(app: &AppHandle) -> Settings {
    let path = settings_path(app);
    match std::fs::read_to_string(&path) {
        Ok(txt) => serde_json::from_str(&txt).unwrap_or_default(),
        Err(_) => Settings::default(),
    }
}

/// Persist settings to disk (creating the config dir if needed).
pub fn save(app: &AppHandle, settings: &Settings) -> Result<(), String> {
    let path = settings_path(app);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let txt = serde_json::to_string_pretty(settings).map_err(|e| e.to_string())?;
    std::fs::write(&path, txt).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_round_trip() {
        let s = Settings {
            trigger: Some(TriggerBinding {
                kind: TriggerKind::Mouse,
                code: "Unknown(1)".into(),
                label: "Mouse 4".into(),
            }),
            trigger_mode: TriggerMode::Toggle,
            model: Some("ggml-base.en".into()),
            language: "en".into(),
            audio_device: None,
            autostart: true,
            restore_clipboard: false,
            theme: Theme::Light,
            accent: "#ff0000".into(),
            pill_scale: 1.2,
            pill_opacity: 0.8,
            pill_compact: true,
            onboarded: true,
            pill_pos: Some([120.0, 240.0]),
            transcript_history: vec![TranscriptEntry {
                id: "1".into(),
                text: "hello".into(),
                created_at_ms: 1,
            }],
        };
        let txt = serde_json::to_string(&s).unwrap();
        let back: Settings = serde_json::from_str(&txt).unwrap();
        assert_eq!(back.trigger, s.trigger);
        assert_eq!(back.trigger_mode, s.trigger_mode);
        assert_eq!(back.model, s.model);
        assert_eq!(back.restore_clipboard, s.restore_clipboard);
        assert_eq!(back.theme, s.theme);
        assert_eq!(back.accent, s.accent);
        assert_eq!(back.pill_compact, s.pill_compact);
        assert_eq!(back.onboarded, s.onboarded);
        assert_eq!(back.transcript_history, s.transcript_history);
    }

    #[test]
    fn partial_json_uses_defaults() {
        let back: Settings = serde_json::from_str("{}").unwrap();
        assert!(back.trigger.is_none());
        assert_eq!(back.trigger_mode, TriggerMode::PushToTalk);
        assert_eq!(back.language, "auto");
        assert!(back.restore_clipboard);
        assert_eq!(back.theme, Theme::System);
        assert_eq!(back.accent, "#6c5ce7");
        assert_eq!(back.pill_scale, 1.0);
        assert!(!back.pill_compact);
        assert!(!back.onboarded);
        assert!(back.transcript_history.is_empty());
    }
}
