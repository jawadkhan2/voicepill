//! Global input hook.
//!
//! Tauri's global-shortcut plugin can't see mouse side-buttons (Mouse4/Mouse5),
//! so we run an `rdev` listener on a dedicated thread. It serves two jobs:
//!
//! 1. **Bind capture** — when `capturing` is set, the next press is recorded as the
//!    new trigger and emitted to the UI (`bind:captured`). Escape cancels.
//! 2. **Trigger dispatch** — once a binding exists, press/release of the bound
//!    button drives the recording state machine and emits `pill:state` +
//!    `trigger:start` / `trigger:stop`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

use rdev::{listen, Button, EventType, Key};
use tauri::{AppHandle, Emitter, Manager};

use crate::audio::SessionEvent;
use crate::settings::{SettingsState, TriggerBinding, TriggerKind, TriggerMode};

/// Shared, thread-safe state for the input hook.
pub struct HookState {
    /// The currently bound trigger, kept in sync with persisted settings.
    pub binding: Mutex<Option<TriggerBinding>>,
    pub mode: Mutex<TriggerMode>,
    /// When true, the next press is consumed as a new binding instead of firing.
    pub capturing: AtomicBool,
    /// True while we are recording (push-to-talk held, or toggle "on").
    pub recording: AtomicBool,
    /// Debounces OS key auto-repeat: true between a real press and its release.
    pub physically_down: AtomicBool,
    /// Drives the audio coordinator thread. Sending must stay non-blocking so the
    /// rdev callback never stalls (a slow global hook degrades system input).
    pub session_tx: Sender<SessionEvent>,
}

impl HookState {
    pub fn new(
        binding: Option<TriggerBinding>,
        mode: TriggerMode,
        session_tx: Sender<SessionEvent>,
    ) -> Self {
        HookState {
            binding: Mutex::new(binding),
            mode: Mutex::new(mode),
            capturing: AtomicBool::new(false),
            recording: AtomicBool::new(false),
            physically_down: AtomicBool::new(false),
            session_tx,
        }
    }
}

/// Spawn the rdev listener on a background thread. `rdev::listen` blocks and runs
/// its own OS message loop, so it must own the thread.
pub fn spawn(app: AppHandle, state: Arc<HookState>) {
    std::thread::spawn(move || {
        let callback = move |event: rdev::Event| handle_event(&app, &state, event);
        if let Err(err) = listen(callback) {
            eprintln!("[input_hook] rdev listen failed: {err:?}");
        }
    });
}

fn handle_event(app: &AppHandle, state: &Arc<HookState>, event: rdev::Event) {
    let (kind, code, label, pressed) = match event.event_type {
        EventType::ButtonPress(b) => (TriggerKind::Mouse, button_code(b), button_label(b), true),
        EventType::ButtonRelease(b) => (TriggerKind::Mouse, button_code(b), button_label(b), false),
        EventType::KeyPress(k) => (TriggerKind::Key, key_code(k), key_label(k), true),
        EventType::KeyRelease(k) => (TriggerKind::Key, key_code(k), key_label(k), false),
        _ => return,
    };
    dispatch(app, state, kind, code, label, pressed);
}

/// The shared bind-capture + trigger state machine, fed by any input source
/// (the `rdev` hook and the Windows Raw Input listener) once an event has been
/// decoded to a `(kind, code, label, pressed)` tuple. Must stay non-blocking so
/// the calling hook thread never stalls.
pub(crate) fn dispatch(
    app: &AppHandle,
    state: &Arc<HookState>,
    kind: TriggerKind,
    code: String,
    label: String,
    pressed: bool,
) {
    // ---- Bind-capture mode ------------------------------------------------
    if state.capturing.load(Ordering::SeqCst) {
        if !pressed {
            return; // bind on the press, ignore the matching release
        }
        // Escape aborts the capture without changing the binding.
        if kind == TriggerKind::Key && code == "Escape" {
            state.capturing.store(false, Ordering::SeqCst);
            let _ = app.emit("bind:cancelled", ());
            return;
        }
        let binding = TriggerBinding { kind, code, label };
        *state.binding.lock().unwrap() = Some(binding.clone());
        state.capturing.store(false, Ordering::SeqCst);
        let _ = app.emit("bind:captured", binding);
        return;
    }

    // ---- Trigger dispatch -------------------------------------------------
    let is_trigger = {
        let guard = state.binding.lock().unwrap();
        guard
            .as_ref()
            .map(|b| b.kind == kind && b.code == code)
            .unwrap_or(false)
    };
    if !is_trigger {
        return;
    }

    let mode = *state.mode.lock().unwrap();
    match mode {
        TriggerMode::PushToTalk => {
            if pressed {
                // Ignore auto-repeat presses while already held.
                if state.physically_down.swap(true, Ordering::SeqCst) {
                    return;
                }
                start_recording(app, state);
            } else {
                state.physically_down.store(false, Ordering::SeqCst);
                stop_recording(app, state);
            }
        }
        TriggerMode::Toggle => {
            if pressed {
                if state.physically_down.swap(true, Ordering::SeqCst) {
                    return; // auto-repeat: don't flip repeatedly
                }
                if state.recording.load(Ordering::SeqCst) {
                    stop_recording(app, state);
                } else {
                    start_recording(app, state);
                }
            } else {
                state.physically_down.store(false, Ordering::SeqCst);
            }
        }
    }
}

fn start_recording(app: &AppHandle, state: &Arc<HookState>) {
    // No model selected — don't start the listening animation. Nudge the user
    // to pick one instead. Without this the pill shows "listening" through a
    // whole recording on a fresh install, then silently drops it at transcribe.
    let has_model = app
        .state::<SettingsState>()
        .0
        .lock()
        .unwrap()
        .model
        .is_some();
    if !has_model {
        let _ = app.emit("transcribe:no-model", ());
        return;
    }

    if state.recording.swap(true, Ordering::SeqCst) {
        return; // already recording
    }
    let _ = state.session_tx.send(SessionEvent::Start);
    let _ = app.emit("trigger:start", ());
    let _ = app.emit("pill:state", "listening");
}

fn stop_recording(app: &AppHandle, state: &Arc<HookState>) {
    if !state.recording.swap(false, Ordering::SeqCst) {
        return; // wasn't recording
    }
    let _ = state.session_tx.send(SessionEvent::Stop);
    let _ = app.emit("trigger:stop", ());
    let _ = app.emit("pill:state", "idle");
}

// ---- rdev <-> our string codes/labels ------------------------------------

fn button_code(b: Button) -> String {
    match b {
        Button::Left => "Left".into(),
        Button::Right => "Right".into(),
        Button::Middle => "Middle".into(),
        Button::Unknown(n) => format!("Unknown({n})"),
    }
}

fn button_label(b: Button) -> String {
    match b {
        Button::Left => "Mouse Left".into(),
        Button::Right => "Mouse Right".into(),
        Button::Middle => "Mouse Middle".into(),
        // Windows side buttons (XBUTTON1/2) arrive as Unknown(1)/Unknown(2).
        Button::Unknown(1) => "Mouse 4".into(),
        Button::Unknown(2) => "Mouse 5".into(),
        Button::Unknown(n) => format!("Mouse btn {n}"),
    }
}

fn key_code(k: Key) -> String {
    format!("{k:?}")
}

fn key_label(k: Key) -> String {
    // Debug names are already reasonable (e.g. "F8", "ControlLeft"); strip the
    // common "Key" prefix on letter/number keys for a cleaner look.
    let s = format!("{k:?}");
    s.strip_prefix("Key").map(str::to_string).unwrap_or(s)
}
