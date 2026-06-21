//! The `#[tauri::command]` surface the webview calls over IPC.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use tauri::{AppHandle, Emitter, Manager, State, WebviewWindow};
use tauri_plugin_autostart::ManagerExt;

use crate::history::TranscriptHistoryState;
use crate::input_hook::HookState;
use crate::settings::{self, Settings, SettingsState, TranscriptEntry};

fn show_window(app: &AppHandle, label: &str) {
    if let Some(win) = app.get_webview_window(label) {
        let _ = win.show();
        let _ = win.unminimize();
        let _ = win.set_focus();
    }
}

/// Return the current settings document.
#[tauri::command]
pub fn get_settings(state: State<'_, SettingsState>) -> Settings {
    state.0.lock().unwrap().clone()
}

/// Persist a full settings document, then sync the live hook + autostart entry.
#[tauri::command]
pub fn update_settings(
    app: AppHandle,
    settings: Settings,
    state: State<'_, SettingsState>,
    hook: State<'_, Arc<HookState>>,
) -> Result<Settings, String> {
    settings::save(&app, &settings)?;

    // Keep the input hook's view of the binding/mode current.
    *hook.binding.lock().unwrap() = settings.trigger.clone();
    *hook.mode.lock().unwrap() = settings.trigger_mode;

    // Toggle the Windows autostart registry entry.
    let autolaunch = app.autolaunch();
    let _ = if settings.autostart {
        autolaunch.enable()
    } else {
        autolaunch.disable()
    };

    *state.0.lock().unwrap() = settings.clone();

    // Let other windows (the pill) re-apply appearance — theme, accent, pill
    // size/opacity/compact — without polling.
    let _ = app.emit("settings:changed", &settings);

    Ok(settings)
}

/// Arm bind-capture: the next button/key press becomes the new trigger.
#[tauri::command]
pub fn start_bind_capture(hook: State<'_, Arc<HookState>>) {
    hook.capturing.store(true, Ordering::SeqCst);
}

/// Disarm bind-capture without changing the binding.
#[tauri::command]
pub fn cancel_bind_capture(hook: State<'_, Arc<HookState>>) {
    hook.capturing.store(false, Ordering::SeqCst);
}

/// List available microphone input device names for the settings picker.
#[tauri::command]
pub fn list_audio_devices() -> Vec<String> {
    crate::audio::list_input_devices()
}

/// List the Whisper model catalog with on-disk status.
#[tauri::command]
pub fn list_models(app: AppHandle) -> Vec<crate::models::ModelInfo> {
    crate::models::list(&app)
}

/// Start downloading a model (progress streamed via `model:*` events).
#[tauri::command]
pub fn download_model(app: AppHandle, id: String) -> Result<(), String> {
    crate::models::download(&app, id)
}

/// Delete a downloaded model file.
#[tauri::command]
pub fn delete_model(app: AppHandle, id: String) -> Result<(), String> {
    crate::models::delete(&app, &id)
}

/// Select a model: persist the choice, then ask the audio coordinator to load
/// it into VRAM (progress streamed via `model:loading` / `model:loaded`).
/// Returns the updated settings so the UI's copy stays canonical.
#[tauri::command]
pub fn select_model(
    app: AppHandle,
    id: String,
    state: State<'_, SettingsState>,
    hook: State<'_, Arc<HookState>>,
) -> Result<Settings, String> {
    let updated = {
        let mut g = state.0.lock().unwrap();
        g.model = Some(id.clone());
        settings::save(&app, &g)?;
        g.clone()
    };
    // Kick off the GPU load on the coordinator thread (non-blocking here).
    let _ = hook.session_tx.send(crate::audio::SessionEvent::Load(id));
    Ok(updated)
}

/// Current GPU / VRAM stats for the settings panel (polled by the UI).
#[tauri::command]
pub fn gpu_info() -> crate::gpu::GpuInfo {
    crate::gpu::info()
}

/// Return the most recent completed transcriptions, newest first.
#[tauri::command]
pub fn get_transcripts(history: State<'_, TranscriptHistoryState>) -> Vec<TranscriptEntry> {
    history.0.lock().unwrap().clone()
}

/// Put a transcript on the clipboard so the user can paste it manually.
#[tauri::command]
pub fn copy_transcript(text: String) -> Result<(), String> {
    let mut clipboard = arboard::Clipboard::new().map_err(|e| e.to_string())?;
    clipboard
        .set_text(text)
        .map_err(|e| format!("clipboard set failed: {e}"))
}

/// Hide history and paste a transcript into the field Windows restores focus to.
#[tauri::command]
pub fn paste_transcript(
    text: String,
    window: WebviewWindow,
    state: State<'_, SettingsState>,
    last_window: State<'_, Arc<crate::focus::LastExternalWindow>>,
) -> Result<(), String> {
    let restore_clipboard = state.0.lock().unwrap().restore_clipboard;
    let _ = window.hide();
    std::thread::sleep(Duration::from_millis(80));
    let _ = crate::focus::focus_last_external(last_window.inner());
    std::thread::sleep(Duration::from_millis(120));
    crate::inject::paste_text(&format!("{text} "), restore_clipboard)
}

/// Show the main window and switch it to the settings view.
#[tauri::command]
pub fn open_settings_window(app: AppHandle) {
    show_window(&app, "main");
    let _ = app.emit_to("main", "nav:settings", ());
}

/// Show the main window on the transcription view and reload its list.
#[tauri::command]
pub fn open_history_window(app: AppHandle) {
    show_window(&app, "main");
    let _ = app.emit_to("main", "nav:transcripts", ());
    let _ = app.emit_to("main", "history:refresh", ());
}

/// Pill click: open history if hidden, close it if already visible.
#[tauri::command]
pub fn toggle_history_window(app: AppHandle) {
    if let Some(win) = app.get_webview_window("main") {
        if win.is_visible().unwrap_or(false) {
            let _ = win.hide();
            return;
        }
    }
    open_history_window(app);
}

/// Persist the pill's on-screen position (debounced by the frontend on move).
#[tauri::command]
pub fn save_pill_pos(
    app: AppHandle,
    x: f64,
    y: f64,
    state: State<'_, SettingsState>,
) -> Result<(), String> {
    let mut g = state.0.lock().unwrap();
    g.pill_pos = Some([x, y]);
    settings::save(&app, &g)
}
