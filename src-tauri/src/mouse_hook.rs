//! Suppressing low-level mouse hook (Windows).
//!
//! `rdev::listen` can only *observe* input, so a mouse side-button bound as the
//! trigger still reached every app (Mouse 5 = browser Forward on every press).
//! This module installs its own `WH_MOUSE_LL` hook that feeds all mouse
//! buttons into [`crate::input_hook::dispatch`] and **swallows** the bound one,
//! so no other application sees it.
//!
//! Tap-or-hold mode builds on that: the press is always swallowed, and if it
//! turns out to be a quick tap we replay a synthetic click of the same button
//! (tagged with [`REPLAY_TAG`] so this hook lets it through untouched).
//!
//! Limits: apps that read Raw Input / DirectInput directly (some games) may
//! still see the physical press — a low-level hook can't hide it from them.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use rdev::Button;
use tauri::AppHandle;
use windows::Win32::Foundation::{LPARAM, LRESULT, WPARAM};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    SendInput, INPUT, INPUT_0, INPUT_MOUSE, MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP,
    MOUSEEVENTF_XDOWN, MOUSEEVENTF_XUP, MOUSEINPUT, MOUSE_EVENT_FLAGS,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CallNextHookEx, GetMessageW, SetWindowsHookExW, UnhookWindowsHookEx, HC_ACTION, MSG,
    MSLLHOOKSTRUCT, WH_MOUSE_LL, WM_LBUTTONDOWN, WM_LBUTTONUP, WM_MBUTTONDOWN, WM_MBUTTONUP,
    WM_RBUTTONDOWN, WM_RBUTTONUP, WM_XBUTTONDOWN, WM_XBUTTONUP,
};

use crate::input_hook::{button_code, button_label, dispatch, HookState, Replay};
use crate::settings::TriggerKind;

/// `dwExtraInfo` marker on the clicks we synthesize ("VPLL"), so the hook
/// passes them straight through instead of treating them as a new press.
const REPLAY_TAG: usize = 0x5650_4C4C;

/// Set once the hook is installed. Until then (or if installing failed) rdev
/// keeps delivering mouse buttons, unsuppressed.
static ACTIVE: AtomicBool = AtomicBool::new(false);

static CTX: OnceLock<(AppHandle, Arc<HookState>)> = OnceLock::new();

/// Whether we swallowed the down event of Middle / XBUTTON1 / XBUTTON2, so the
/// matching up event is swallowed too. Deciding per-event could split a click
/// (e.g. the binding changes mid-press) and leave an app with a stuck button.
static SWALLOWED_DOWN: [AtomicBool; 3] = [
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
];

pub fn is_active() -> bool {
    ACTIVE.load(Ordering::SeqCst)
}

/// Install the hook on a dedicated thread with its own message loop (a
/// low-level hook is called on the installing thread's message queue).
pub fn spawn(app: AppHandle, state: Arc<HookState>) {
    if CTX.set((app, state)).is_err() {
        return; // already running
    }
    std::thread::spawn(|| unsafe {
        let hook = match SetWindowsHookExW(WH_MOUSE_LL, Some(hook_proc), None, 0) {
            Ok(h) => h,
            Err(e) => {
                crate::vlog!("[mouse_hook] SetWindowsHookExW failed: {e}; falling back to rdev");
                return;
            }
        };
        ACTIVE.store(true, Ordering::SeqCst);
        crate::vlog!("[mouse_hook] installed");
        let mut msg = MSG::default();
        while GetMessageW(&mut msg, None, 0, 0).as_bool() {}
        ACTIVE.store(false, Ordering::SeqCst);
        let _ = UnhookWindowsHookEx(hook);
    });
}

/// Map a hook message to `(button, pressed, swallow-slot)`.
fn decode(msg: u32, info: &MSLLHOOKSTRUCT) -> Option<(Button, bool, Option<usize>)> {
    let xbtn = (info.mouseData >> 16) & 0xFFFF;
    Some(match msg {
        WM_LBUTTONDOWN => (Button::Left, true, None),
        WM_LBUTTONUP => (Button::Left, false, None),
        WM_RBUTTONDOWN => (Button::Right, true, None),
        WM_RBUTTONUP => (Button::Right, false, None),
        WM_MBUTTONDOWN => (Button::Middle, true, Some(0)),
        WM_MBUTTONUP => (Button::Middle, false, Some(0)),
        // Same codes rdev produced (Unknown(1)/Unknown(2)), so existing
        // bindings keep working.
        WM_XBUTTONDOWN | WM_XBUTTONUP if xbtn == 1 || xbtn == 2 => (
            Button::Unknown(xbtn as u8),
            msg == WM_XBUTTONDOWN,
            Some(xbtn as usize),
        ),
        _ => return None,
    })
}

unsafe extern "system" fn hook_proc(ncode: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    if ncode == HC_ACTION as i32 {
        let info = &*(lparam.0 as *const MSLLHOOKSTRUCT);
        if info.dwExtraInfo != REPLAY_TAG {
            if let (Some((button, pressed, slot)), Some((app, state))) =
                (decode(wparam.0 as u32, info), CTX.get())
            {
                let code = button_code(button);
                let swallow = match slot {
                    // Left/Right are never swallowed, whatever is bound.
                    None => false,
                    Some(i) if pressed => {
                        let s = should_swallow(state, &code);
                        SWALLOWED_DOWN[i].store(s, Ordering::SeqCst);
                        s
                    }
                    Some(i) => SWALLOWED_DOWN[i].swap(false, Ordering::SeqCst),
                };
                let replay: Option<Replay> = match (swallow, slot) {
                    (true, Some(0)) => Some(replay_middle),
                    (true, Some(1)) => Some(replay_x1),
                    (true, Some(2)) => Some(replay_x2),
                    _ => None,
                };
                dispatch(app, state, TriggerKind::Mouse, code, button_label(button), pressed, replay);
                if swallow {
                    return LRESULT(1);
                }
            }
        }
    }
    CallNextHookEx(None, ncode, wparam, lparam)
}

/// Swallow the bound mouse button — except while binding a new trigger, so the
/// press the user makes to bind it still behaves normally.
fn should_swallow(state: &HookState, code: &str) -> bool {
    if state.capturing.load(Ordering::SeqCst) {
        return false;
    }
    state
        .binding
        .lock()
        .unwrap()
        .as_ref()
        .map(|b| b.kind == TriggerKind::Mouse && b.code == code)
        .unwrap_or(false)
}

fn send_click(data: u32, down: MOUSE_EVENT_FLAGS, up: MOUSE_EVENT_FLAGS) {
    let mk = |flags| INPUT {
        r#type: INPUT_MOUSE,
        Anonymous: INPUT_0 {
            mi: MOUSEINPUT {
                dx: 0,
                dy: 0,
                mouseData: data,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: REPLAY_TAG,
            },
        },
    };
    let inputs = [mk(down), mk(up)];
    let sent = unsafe { SendInput(&inputs, std::mem::size_of::<INPUT>() as i32) };
    if sent != inputs.len() as u32 {
        crate::vlog!("[mouse_hook] replay SendInput sent {sent}/{}", inputs.len());
    }
}

fn replay_middle() {
    send_click(0, MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP);
}

fn replay_x1() {
    send_click(1, MOUSEEVENTF_XDOWN, MOUSEEVENTF_XUP);
}

fn replay_x2() {
    send_click(2, MOUSEEVENTF_XDOWN, MOUSEEVENTF_XUP);
}
