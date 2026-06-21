//! Tracks the last foreground window outside this process.
//!
//! The history window takes focus when the user clicks it, so "paste into the
//! previous field" cannot rely on Windows naturally restoring focus after we
//! hide the window. Instead we keep the last foreground HWND whose process is
//! not VoicePill, then explicitly focus it before synthesizing Ctrl+V.

use std::sync::{Arc, Mutex};
use std::time::Duration;

pub struct LastExternalWindow(pub Mutex<Option<isize>>);

impl LastExternalWindow {
    pub fn new() -> Arc<Self> {
        Arc::new(Self(Mutex::new(None)))
    }
}

#[cfg(windows)]
pub fn spawn(state: Arc<LastExternalWindow>) {
    std::thread::spawn(move || loop {
        if let Some(hwnd) = current_external_hwnd() {
            *state.0.lock().unwrap() = Some(hwnd);
        }
        std::thread::sleep(Duration::from_millis(100));
    });
}

#[cfg(not(windows))]
pub fn spawn(_state: Arc<LastExternalWindow>) {}

#[cfg(windows)]
pub fn focus_last_external(state: &Arc<LastExternalWindow>) -> bool {
    let Some(raw) = *state.0.lock().unwrap() else {
        return false;
    };
    let hwnd = windows::Win32::Foundation::HWND(raw as *mut core::ffi::c_void);
    unsafe { windows::Win32::UI::WindowsAndMessaging::SetForegroundWindow(hwnd).as_bool() }
}

#[cfg(not(windows))]
pub fn focus_last_external(_state: &Arc<LastExternalWindow>) -> bool {
    false
}

#[cfg(windows)]
fn current_external_hwnd() -> Option<isize> {
    use windows::Win32::Foundation::HWND;
    use windows::Win32::System::Threading::GetCurrentProcessId;
    use windows::Win32::UI::WindowsAndMessaging::{
        GetForegroundWindow, GetWindowThreadProcessId, IsWindowVisible,
    };

    unsafe {
        let hwnd: HWND = GetForegroundWindow();
        if hwnd.0.is_null() || !IsWindowVisible(hwnd).as_bool() {
            return None;
        }

        let mut pid = 0u32;
        GetWindowThreadProcessId(hwnd, Some(&mut pid));
        if pid == 0 || pid == GetCurrentProcessId() {
            return None;
        }

        Some(hwnd.0 as isize)
    }
}
