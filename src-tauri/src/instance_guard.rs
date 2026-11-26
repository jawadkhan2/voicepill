//! Hard single-instance guard, checked in `main` before Tauri starts.
//!
//! `tauri-plugin-single-instance` alone is not enough. Its Windows check is
//! "mutex already exists AND the first instance's hidden message window is
//! found"; if the mutex exists but `FindWindowW` comes back empty (two launches
//! racing — autostart + a click, a relaunch — or a first instance that can't be
//! opened because it runs at another integrity level, where `CreateMutexW`
//! fails with ACCESS_DENIED instead of ALREADY_EXISTS), the second process just
//! keeps going as a full VoicePill. Two VoicePills both catch the trigger, both
//! hold a model in VRAM and race the clipboard, and sentences get dropped.
//!
//! This guard owns its own named mutex for the whole process lifetime. A second
//! launch never gets past `main`: it hands its args to the running instance
//! through the plugin's message window (so a manual launch still brings the
//! panel up) and exits. Ownership is released by the OS when the owner dies,
//! even on a crash (WAIT_ABANDONED counts as acquired), so a relaunch waits for
//! the old process to go away instead of being refused.

#[cfg(windows)]
mod imp {
    use std::time::{Duration, Instant};

    use windows::core::w;
    use windows::Win32::Foundation::{
        GetLastError, ERROR_ACCESS_DENIED, HANDLE, LPARAM, WAIT_ABANDONED, WAIT_OBJECT_0, WPARAM,
    };
    use windows::Win32::System::DataExchange::COPYDATASTRUCT;
    use windows::Win32::System::Threading::{CreateMutexW, WaitForSingleObject};
    use windows::Win32::UI::WindowsAndMessaging::{
        AllowSetForegroundWindow, FindWindowW, GetWindowThreadProcessId, SendMessageTimeoutW,
        SMTO_ABORTIFHUNG, WM_COPYDATA,
    };

    /// How long a launch waits for a running instance to either exit (relaunch)
    /// or show its message window (two launches racing) before giving up.
    const WAIT: Duration = Duration::from_secs(8);
    const POLL: Duration = Duration::from_millis(100);
    const RELAUNCH_GRACE: Duration = Duration::from_millis(1500);

    /// Must match tauri-plugin-single-instance's protocol (windows.rs):
    /// class/window names derive from the bundle identifier, payload is
    /// "cwd|arg0|arg1…\0" tagged with this dwData.
    const PLUGIN_COPYDATA_TAG: usize = 1542;

    pub fn acquire_or_exit() {
        // Session-local name, distinct from the plugin's own `-sim` mutex.
        let mutex = unsafe { CreateMutexW(None, false, w!("Local\\com.voicepill.app.instance-guard")) };
        let last = unsafe { GetLastError() };

        let handle: Option<HANDLE> = match mutex {
            Ok(h) => Some(h),
            // The running instance's mutex exists but we may not open it (other
            // integrity level). It's still a running instance.
            Err(_) if last == ERROR_ACCESS_DENIED => None,
            // Couldn't create a mutex at all — don't block startup on the guard.
            Err(_) => return,
        };

        let start = Instant::now();
        loop {
            if let Some(h) = handle {
                let r = unsafe { WaitForSingleObject(h, POLL.as_millis() as u32) };
                if r == WAIT_OBJECT_0 || r == WAIT_ABANDONED {
                    // We own it now (first launch, or an old instance finished
                    // exiting during a relaunch/update). Leak the handle:
                    // ownership must last for the whole process and the OS
                    // releases it at exit.
                    return;
                }
            } else {
                std::thread::sleep(POLL);
            }

            // Another instance is alive. Give it a moment in case it is the old
            // process of a relaunch, still on its way out (its message window
            // lives until the very end), then hand off once its window exists.
            if start.elapsed() >= RELAUNCH_GRACE && hand_off_to_running_instance() {
                std::process::exit(0);
            }
            if start.elapsed() >= WAIT {
                // Alive but no message window (hung or still booting). Never run
                // a second copy alongside it.
                std::process::exit(0);
            }
        }
    }

    fn hand_off_to_running_instance() -> bool {
        unsafe {
            let hwnd = match FindWindowW(
                w!("com.voicepill.app-sic"),
                w!("com.voicepill.app-siw"),
            ) {
                Ok(h) if !h.is_invalid() => h,
                _ => return false,
            };
            let mut pid = 0u32;
            GetWindowThreadProcessId(hwnd, Some(&mut pid as *mut u32));
            if pid != 0 {
                let _ = AllowSetForegroundWindow(pid);
            }
            let cwd = std::env::current_dir().unwrap_or_default();
            let args = std::env::args().collect::<Vec<_>>().join("|");
            let data = format!("{}|{}\0", cwd.to_string_lossy(), args);
            let cds = COPYDATASTRUCT {
                dwData: PLUGIN_COPYDATA_TAG,
                cbData: data.len() as u32,
                lpData: data.as_ptr() as *mut _,
            };
            let _ = SendMessageTimeoutW(
                hwnd,
                WM_COPYDATA,
                WPARAM(0),
                LPARAM(&cds as *const _ as isize),
                SMTO_ABORTIFHUNG,
                2000,
                None,
            );
            true
        }
    }
}

#[cfg(windows)]
pub use imp::acquire_or_exit;

#[cfg(not(windows))]
pub fn acquire_or_exit() {}
