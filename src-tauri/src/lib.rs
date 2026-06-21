mod audio;
mod commands;
mod focus;
mod gpu;
#[cfg(windows)]
mod hidpp;
mod history;
mod inject;
mod input_hook;
mod models;
mod settings;
mod transcribe;

use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};

use tauri::{
    menu::{MenuBuilder, MenuItemBuilder},
    tray::TrayIconBuilder,
    Emitter, Manager,
};
use tauri_plugin_autostart::MacosLauncher;

use focus::LastExternalWindow;
use history::TranscriptHistoryState;
use input_hook::HookState;
use settings::SettingsState;

/// Show (or focus) a window by label.
fn show_window(app: &tauri::AppHandle, label: &str) {
    if let Some(win) = app.get_webview_window(label) {
        let _ = win.show();
        let _ = win.unminimize();
        let _ = win.set_focus();
    }
}

fn toggle_window(app: &tauri::AppHandle, label: &str) {
    if let Some(win) = app.get_webview_window(label) {
        if win.is_visible().unwrap_or(false) {
            let _ = win.hide();
        } else {
            let _ = win.show();
            let _ = win.set_focus();
        }
    }
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_autostart::init(
            MacosLauncher::LaunchAgent,
            Some(vec!["--minimized"]),
        ))
        // OTA updates (checks GitHub Releases) + relaunch after install.
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .invoke_handler(tauri::generate_handler![
            commands::get_settings,
            commands::update_settings,
            commands::start_bind_capture,
            commands::cancel_bind_capture,
            commands::list_audio_devices,
            commands::list_models,
            commands::download_model,
            commands::delete_model,
            commands::select_model,
            commands::gpu_info,
            commands::get_transcripts,
            commands::copy_transcript,
            commands::paste_transcript,
            commands::open_settings_window,
            commands::open_history_window,
            commands::toggle_history_window,
            commands::save_pill_pos,
        ])
        .setup(|app| {
            let handle = app.handle().clone();
            let last_external_window = LastExternalWindow::new();
            focus::spawn(last_external_window.clone());
            app.manage(last_external_window);

            // ---- Audio coordinator thread (owns the !Send cpal stream) ------
            let (session_tx, session_rx) = channel::<audio::SessionEvent>();
            let audio_handle = handle.clone();
            std::thread::spawn(move || audio::run(audio_handle, session_rx));

            // ---- Load settings + wire shared state ---------------------------
            let loaded = settings::load(&handle);
            let loaded_history = history::load(&handle, &loaded.transcript_history);
            let needs_onboarding = loaded.trigger.is_none();

            // Manage settings state immediately so the concurrently-booting pill
            // webview's `get_settings` call can't lose a race against setup and
            // fall back to the default accent on startup.
            app.manage(SettingsState(Mutex::new(loaded.clone())));

            // Restore the pill to where the user last left it.
            if let Some([x, y]) = loaded.pill_pos {
                if let Some(w) = app.get_webview_window("pill") {
                    let _ = w.set_position(tauri::PhysicalPosition::new(x, y));
                }
            }

            // Warm the selected model into VRAM at startup so the first
            // transcription isn't stalled by a multi-GB load. The pill shows a
            // "Loading model…" state while this runs.
            if let Some(model) = loaded.model.clone() {
                let _ = session_tx.send(audio::SessionEvent::Load(model));
            }

            let hook = Arc::new(HookState::new(
                loaded.trigger.clone(),
                loaded.trigger_mode,
                session_tx,
            ));
            app.manage(TranscriptHistoryState(Mutex::new(loaded_history)));
            app.manage(hook.clone());

            // ---- Global input hook (rdev) on its own thread -----------------
            input_hook::spawn(handle.clone(), hook.clone());

            // ---- Logitech HID++ listener on its own thread ------------------
            // The MX Master 3 gesture button emits no HID report on a bare mouse;
            // this actively diverts it over HID++ so its presses reach the same
            // trigger state machine. No-op if such a device isn't present.
            #[cfg(windows)]
            hidpp::spawn(handle.clone(), hook);

            // First run (nothing bound yet): open the panel so the user can set
            // a trigger and download a model. main.ts opens to the settings view
            // by itself when the user hasn't onboarded yet.
            if needs_onboarding {
                show_window(&handle, "main");
            }

            // ---- System tray -------------------------------------------------
            let settings_item = MenuItemBuilder::with_id("settings", "Settings").build(app)?;
            let toggle_item =
                MenuItemBuilder::with_id("toggle_pill", "Show / Hide Pill").build(app)?;
            let quit_item = MenuItemBuilder::with_id("quit", "Quit").build(app)?;

            let menu = MenuBuilder::new(app)
                .items(&[&settings_item, &toggle_item])
                .separator()
                .items(&[&quit_item])
                .build()?;

            TrayIconBuilder::with_id("tray")
                .icon(app.default_window_icon().unwrap().clone())
                .tooltip("VoicePill")
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(|app, event| match event.id().as_ref() {
                    "settings" => {
                        show_window(app, "main");
                        let _ = app.emit_to("main", "nav:settings", ());
                    }
                    "toggle_pill" => toggle_window(app, "pill"),
                    "quit" => app.exit(0),
                    _ => {}
                })
                .build(app)?;

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running VoicePill");
}
