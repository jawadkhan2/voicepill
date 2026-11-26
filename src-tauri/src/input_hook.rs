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
//!
//! On Windows, mouse buttons come from [`crate::mouse_hook`] instead (a
//! `WH_MOUSE_LL` hook that can *swallow* the bound button so no other app sees
//! it); rdev then handles the keyboard only.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rdev::{listen, Button, EventType, Key};
use tauri::{AppHandle, Emitter, Manager};

use crate::audio::{Capture, SessionEvent};
use crate::settings::{SettingsState, TriggerBinding, TriggerKind, TriggerMode};

/// Shared, thread-safe state for the input hook.
pub struct HookState {
    /// The currently bound trigger, kept in sync with persisted settings.
    pub binding: Mutex<Option<TriggerBinding>>,
    pub mode: Mutex<TriggerMode>,
    /// When true, the next press is consumed as a new binding instead of firing.
    pub capturing: AtomicBool,
    /// True while we are recording (push-to-talk held, or toggle "on").
    pub recording: Arc<AtomicBool>,
    /// Debounces OS key auto-repeat: true between a real press and its release.
    pub physically_down: AtomicBool,
    /// Drives the audio coordinator thread. Sending must stay non-blocking so the
    /// rdev callback never stalls (a slow global hook degrades system input).
    pub session_tx: Sender<SessionEvent>,
    /// Capture gate + buffer, opened/closed right here on the input thread so
    /// recording starts the instant the trigger goes down — even while the
    /// coordinator is still busy transcribing the previous sentence.
    pub capture: Arc<Capture>,
    /// Tap-or-hold mode: hold time (ms) before a press counts as a hold.
    pub hold_ms: AtomicU32,
    /// Tap-or-hold mode: where the current press is in tap-vs-hold detection.
    tap: Mutex<TapState>,
    /// Bumped on every tap-or-hold press so a stale hold timer can tell it
    /// belongs to an earlier press and do nothing.
    press_seq: AtomicU64,
}

/// Tap-or-hold press tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TapState {
    /// Trigger is up.
    Idle,
    /// Down, but not yet held past the threshold. Audio is already being
    /// captured (silently) so the first words aren't lost if it becomes a hold.
    Pending { seq: u64 },
    /// Held past the threshold: this press is a dictation.
    Holding,
}

/// Replays the bound button's original action for a tap (e.g. a synthetic
/// XBUTTON2 click = browser Forward). Run on its own thread, never inside the
/// low-level hook callback.
pub(crate) type Replay = fn();

impl HookState {
    pub fn new(
        binding: Option<TriggerBinding>,
        mode: TriggerMode,
        hold_ms: u32,
        session_tx: Sender<SessionEvent>,
        capture: Arc<Capture>,
    ) -> Self {
        HookState {
            binding: Mutex::new(binding),
            mode: Mutex::new(mode),
            capturing: AtomicBool::new(false),
            recording: Arc::new(AtomicBool::new(false)),
            physically_down: AtomicBool::new(false),
            session_tx,
            capture,
            hold_ms: AtomicU32::new(hold_ms),
            tap: Mutex::new(TapState::Idle),
            press_seq: AtomicU64::new(0),
        }
    }
}

/// Spawn the rdev listener on a background thread. `rdev::listen` blocks and runs
/// its own OS message loop, so it must own the thread.
pub fn spawn(app: AppHandle, state: Arc<HookState>) {
    std::thread::spawn(move || {
        let callback = move |event: rdev::Event| handle_event(&app, &state, event);
        if let Err(err) = listen(callback) {
            crate::vlog!("[input_hook] rdev listen failed: {err:?}");
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
    // The suppressing mouse hook owns mouse buttons once it's installed;
    // feeding them from rdev too would double-dispatch every click.
    #[cfg(windows)]
    if kind == TriggerKind::Mouse && crate::mouse_hook::is_active() {
        return;
    }
    dispatch(app, state, kind, code, label, pressed, None);
}

/// The shared bind-capture + trigger state machine, fed by any input source
/// (the `rdev` hook and the Windows Raw Input listener) once an event has been
/// decoded to a `(kind, code, label, pressed)` tuple. Must stay non-blocking so
/// the calling hook thread never stalls.
///
/// `replay` re-creates the button's normal action when tap-or-hold mode sees a
/// tap. Only a source that swallowed the press passes one; for the others
/// (rdev keys, HID++) the OS already delivered the original input.
pub(crate) fn dispatch(
    app: &AppHandle,
    state: &Arc<HookState>,
    kind: TriggerKind,
    code: String,
    label: String,
    pressed: bool,
    replay: Option<Replay>,
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
        TriggerMode::TapHold => {
            if pressed {
                if state.physically_down.swap(true, Ordering::SeqCst) {
                    return; // auto-repeat
                }
                tap_hold_press(app, state);
            } else {
                state.physically_down.store(false, Ordering::SeqCst);
                tap_hold_release(app, state, replay);
            }
        }
    }
}

/// Tap-or-hold press: start capturing silently and arm the hold timer. Nothing
/// is shown yet, so if this turns out to be a tap the user never sees the pill.
fn tap_hold_press(app: &AppHandle, state: &Arc<HookState>) {
    let seq = state.press_seq.fetch_add(1, Ordering::SeqCst) + 1;
    *state.tap.lock().unwrap() = TapState::Pending { seq };
    state.capture.begin();

    let hold = Duration::from_millis(state.hold_ms.load(Ordering::SeqCst) as u64);
    let (app, state) = (app.clone(), state.clone());
    // Short-lived timer thread: the hook callback itself must return fast.
    std::thread::spawn(move || {
        std::thread::sleep(hold);
        tap_hold_promote(&app, &state, seq);
    });
}

/// Hold timer fired: if the same press is still down, it's a dictation.
fn tap_hold_promote(app: &AppHandle, state: &Arc<HookState>, seq: u64) {
    {
        let mut tap = state.tap.lock().unwrap();
        if *tap != (TapState::Pending { seq }) {
            return; // released already (a tap), or a newer press owns the state
        }
        *tap = TapState::Holding;
    }
    let has_model = app.state::<SettingsState>().0.lock().unwrap().model.is_some();
    if !has_model {
        state.capture.abort();
        let _ = app.emit("transcribe:no-model", ());
        return;
    }
    if state.recording.swap(true, Ordering::SeqCst) {
        return;
    }
    // The capture gate has been open since the press; keep that audio.
    let _ = state.session_tx.send(SessionEvent::Start {
        recording: state.recording.clone(),
    });
    let _ = app.emit("trigger:start", ());
    let _ = app.emit("pill:state", "listening");
}

/// Tap-or-hold release: a tap drops the silent capture and replays the
/// button's own action; a hold stops and transcribes like push-to-talk.
fn tap_hold_release(app: &AppHandle, state: &Arc<HookState>, replay: Option<Replay>) {
    let prev = std::mem::replace(&mut *state.tap.lock().unwrap(), TapState::Idle);
    match prev {
        TapState::Pending { .. } => {
            state.capture.abort();
            if let Some(replay) = replay {
                std::thread::spawn(replay);
            }
        }
        TapState::Holding => stop_recording(app, state),
        TapState::Idle => {}
    }
}

fn start_recording(app: &AppHandle, state: &Arc<HookState>) {
    // No model selected — don't start the listening animation. Nudge the user
    // to pick one instead. Without this the pill shows "listening" through a
    // whole recording on a fresh install, then silently drops it at transcribe.
    let has_model = app.state::<SettingsState>().0.lock().unwrap().model.is_some();
    if !has_model {
        let _ = app.emit("transcribe:no-model", ());
        return;
    }

    // NOTE: no microphone enumeration here. This runs on the low-level input
    // hook thread, and WASAPI device enumeration can take long enough that
    // Windows' LowLevelHooksTimeout silently unhooks us (the trigger then just
    // stops working). The coordinator validates the device on `Start` instead
    // and reports problems through `audio:error`.
    if state.recording.swap(true, Ordering::SeqCst) {
        return; // already recording
    }
    state.capture.begin();
    let _ = state.session_tx.send(SessionEvent::Start {
        recording: state.recording.clone(),
    });
    let _ = app.emit("trigger:start", ());
    let _ = app.emit("pill:state", "listening");
}

fn stop_recording(app: &AppHandle, state: &Arc<HookState>) {
    if !state.recording.swap(false, Ordering::SeqCst) {
        return; // wasn't recording
    }
    let clip = state.capture.end();
    let _ = state.session_tx.send(SessionEvent::Stop(clip));
    let _ = app.emit("trigger:stop", ());
    let _ = app.emit("pill:state", "idle");
}

// ---- rdev <-> our string codes/labels ------------------------------------

pub(crate) fn button_code(b: Button) -> String {
    match b {
        Button::Left => "Left".into(),
        Button::Right => "Right".into(),
        Button::Middle => "Middle".into(),
        Button::Unknown(n) => format!("Unknown({n})"),
    }
}

pub(crate) fn button_label(b: Button) -> String {
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
