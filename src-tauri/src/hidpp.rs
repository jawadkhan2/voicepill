//! Logitech HID++ 2.0 client — makes the MX Master 3 **gesture button** usable as
//! a trigger.
//!
//! ## Why this exists
//! The thumb gesture button is *not* one of the five buttons the low-level mouse
//! hook (`WH_MOUSE_LL` / `rdev`) can carry, and — confirmed by diagnostics — the
//! firmware emits **no HID report at all** for it on a bare mouse with no
//! Logitech software. Windows Raw Input therefore can't see it either: there is
//! no event to observe.
//!
//! The button only becomes visible once a host explicitly **diverts** it over
//! Logitech's HID++ vendor protocol (the same thing Options+/Solaar do). This
//! module speaks just enough HID++ 2.0 to do that:
//!
//! 1. Find the device's HID++ "long message" collection (usage page `0xFF43`,
//!    usage `0x0202`) and open it for read/write.
//! 2. Resolve the feature index of `REPROG_CONTROLS_V4` (`0x1B04`) via the root
//!    feature.
//! 3. `setCidReporting` to **temporarily divert** the gesture control
//!    (CID `0x00C3`) to software.
//! 4. Read `divertedButtonsEvent` notifications; diff the pressed-control set to
//!    derive press/release edges; feed them into [`crate::input_hook::dispatch`]
//!    so the existing bind-capture + toggle/PTT machinery handles the rest.
//!
//! ## Protocol notes (HID++ 2.0, Bluetooth direct connection)
//! Reports are 20-byte "long" reports, report id `0x11`, laid out as
//! `[0x11, deviceIndex, featureIndex, (function<<4)|softwareId, params…]`. Over a
//! direct Bluetooth link the device index is `0xFF`. We use a non-zero software
//! id (`0x05`) so our command *responses* (low nibble `0x05`) are distinguishable
//! from spontaneous *events* (low nibble `0x00`).
//!
//! `setCidReporting` (function 3) takes the control id (big-endian) and a flags
//! byte where each setting is paired with a validity bit: `divert=0x01`,
//! `dvalid=0x02`. So `0x03` = "I am setting divert, and divert = on".
//!
//! `divertedButtonsEvent` (event 0) carries up to four currently-pressed control
//! ids as big-endian `u16`s at params 0..8; all-zero means everything released.
//! It is a *state snapshot*, so we diff against the previous set for edges.
//!
//! ## Reconnect
//! A temporary divert is reset when the device sleeps / reconnects, so the whole
//! open→handshake→divert→listen sequence runs inside a reconnect loop: if the
//! device isn't present yet, or the link drops, we retry and re-divert.

#![cfg(windows)]

use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use tauri::AppHandle;

use windows::core::PCWSTR;
use windows::Win32::Devices::HumanInterfaceDevice::{
    HidD_FreePreparsedData, HidD_GetPreparsedData, HidP_GetCaps, HIDP_CAPS, PHIDP_PREPARSED_DATA,
};
use windows::Win32::Foundation::{CloseHandle, ERROR_IO_PENDING, HANDLE, WAIT_OBJECT_0};
use windows::Win32::Storage::FileSystem::{
    CreateFileW, ReadFile, WriteFile, FILE_FLAG_OVERLAPPED, FILE_SHARE_READ, FILE_SHARE_WRITE,
    OPEN_EXISTING,
};
use windows::Win32::System::Threading::{CreateEventW, ResetEvent, WaitForSingleObject};
use windows::Win32::System::IO::{CancelIoEx, GetOverlappedResult, OVERLAPPED};
use windows::Win32::UI::Input::{
    GetRawInputDeviceInfoW, GetRawInputDeviceList, RAWINPUTDEVICELIST, RIDI_DEVICEINFO,
    RIDI_DEVICENAME, RID_DEVICE_INFO,
};

use crate::input_hook::HookState;
use crate::settings::TriggerKind;

// ---- HID++ protocol constants --------------------------------------------

/// Raw Input device-type discriminant for HID devices (winuser.h `RIM_TYPEHID`).
const RIM_TYPE_HID: u32 = 2;

/// The Logitech HID++ "long message" top-level collection. Commands and events
/// both ride 20-byte reports here.
const HIDPP_USAGE_PAGE: u16 = 0xFF43;
const HIDPP_USAGE: u16 = 0x0202;

const REPORT_LONG: u8 = 0x11; // long-report id
const DEV_INDEX_BT: u8 = 0xFF; // device index for a direct Bluetooth link
const SW_ID: u8 = 0x05; // our software id (must be non-zero)

const ROOT_FEATURE_IDX: u8 = 0x00; // the root feature is always index 0
const ROOT_FN_GET_FEATURE: u8 = 0x00; // IRoot.getFeature(featureId) -> index

const FEAT_REPROG_CONTROLS_V4: u16 = 0x1B04;
const REPROG_FN_GET_COUNT: u8 = 0x00;
const REPROG_FN_GET_CID_INFO: u8 = 0x01;
const REPROG_FN_SET_CID_REPORTING: u8 = 0x03;

/// MX Master 3 thumb gesture button control id ("Virtual Gesture Button"). If a
/// future device reports it under a different id, the diagnostic log
/// (`VOICEPILL_HIDPP_DEBUG=1`, control enumeration) reveals the real value.
const GESTURE_CID: u16 = 0x00C3;

// setCidReporting flags: each setting is paired with a "valid"/update bit.
const FLAG_DIVERT: u8 = 0x01;
const FLAG_DVALID: u8 = 0x02;

/// `getCidInfo` flags byte: bit 5 marks a control as divertable.
const CID_FLAG_DIVERTABLE: u8 = 0x20;

const DEFAULT_REPORT_LEN: usize = 20;
const CALL_TIMEOUT_MS: u32 = 1000;

/// Spawn the HID++ listener thread. Returns immediately; the thread lives for the
/// duration of the process, reconnecting as needed.
pub fn spawn(app: AppHandle, state: Arc<HookState>) {
    std::thread::spawn(move || run(app, state));
}

fn run(app: AppHandle, state: Arc<HookState>) {
    let debug = std::env::var_os("VOICEPILL_HIDPP_DEBUG").is_some();
    if debug {
        eprintln!("[hidpp] starting; looking for the MX Master 3 HID++ channel…");
    }
    // Reconnect loop: a temporary divert is lost on sleep/reconnect, so we
    // re-establish the whole session whenever it ends.
    loop {
        match try_session(&app, &state, debug) {
            Ok(()) => {
                if debug {
                    eprintln!("[hidpp] session ended cleanly; retrying…");
                }
            }
            Err(e) => {
                if debug {
                    eprintln!("[hidpp] session error: {e}; retrying in 3s…");
                }
            }
        }
        thread::sleep(Duration::from_secs(3));
    }
}

/// Find the device, open it, divert the gesture control, then read events until
/// the link drops (returns `Err`, triggering a reconnect).
fn try_session(app: &AppHandle, state: &Arc<HookState>, debug: bool) -> Result<(), String> {
    let paths = unsafe { find_hidpp_paths() };
    if paths.is_empty() {
        return Err("no HID++ collection (0xFF43/0x0202) found".into());
    }

    // Try each candidate path until one answers HID++.
    let mut dev = None;
    let mut reprog_idx = 0u8;
    for path in &paths {
        let Some(d) = (unsafe { Device::open(path, debug) }) else {
            continue;
        };
        match unsafe { d.get_feature(FEAT_REPROG_CONTROLS_V4) } {
            Some(idx) if idx != 0 => {
                reprog_idx = idx;
                dev = Some(d);
                break;
            }
            _ => {
                if debug {
                    eprintln!("[hidpp] a candidate device did not expose REPROG_CONTROLS_V4");
                }
            }
        }
    }
    let dev = dev.ok_or_else(|| "found HID++ device(s) but none answered 0x1B04".to_string())?;

    if debug {
        eprintln!("[hidpp] REPROG_CONTROLS_V4 is feature index {reprog_idx}");
        unsafe { dev.enumerate_controls(reprog_idx) };
    }

    // Temporarily divert the gesture button so its presses reach software.
    let params = [
        (GESTURE_CID >> 8) as u8,
        (GESTURE_CID & 0xFF) as u8,
        FLAG_DVALID | FLAG_DIVERT,
        0x00, // remap hi (0 = no remap)
        0x00, // remap lo
    ];
    match unsafe { dev.call(reprog_idx, REPROG_FN_SET_CID_REPORTING, &params) } {
        Some(_) => {
            if debug {
                eprintln!(
                    "[hidpp] diverted gesture control 0x{GESTURE_CID:04x}; listening for presses"
                );
            }
        }
        None => {
            return Err(format!(
                "setCidReporting(divert) for CID 0x{GESTURE_CID:04x} got no acknowledgement"
            ));
        }
    }

    // ---- Event loop ------------------------------------------------------
    // divertedButtonsEvent is a snapshot of currently-pressed control ids; diff
    // it against the previous snapshot to derive press/release edges.
    let mut prev: HashSet<u16> = HashSet::new();
    loop {
        // u32::MAX = wait indefinitely; a `None` here means the handle died
        // (device slept/disconnected) → end the session to reconnect.
        let report = unsafe { dev.read(u32::MAX) }
            .ok_or_else(|| "read failed (device disconnected?)".to_string())?;
        if report.len() < 12 || report[0] != REPORT_LONG {
            continue;
        }
        // Only the diverted-buttons event on our feature: feature index matches,
        // low nibble (software id) is 0 and high nibble (event id) is 0.
        if report[2] != reprog_idx || report[3] != 0x00 {
            if debug && report[2] == reprog_idx {
                eprintln!("[hidpp] feature event byte3=0x{:02x} (ignored)", report[3]);
            }
            continue;
        }

        let mut cur: HashSet<u16> = HashSet::new();
        for k in 0..4 {
            let cid = ((report[4 + k * 2] as u16) << 8) | report[5 + k * 2] as u16;
            if cid != 0 {
                cur.insert(cid);
            }
        }

        for &cid in cur.difference(&prev) {
            emit(app, state, debug, cid, true);
        }
        for &cid in prev.difference(&cur) {
            emit(app, state, debug, cid, false);
        }
        prev = cur;
    }
}

/// Forward a control-id edge into the shared trigger state machine.
fn emit(app: &AppHandle, state: &Arc<HookState>, debug: bool, cid: u16, pressed: bool) {
    if debug {
        eprintln!(
            "[hidpp] control 0x{cid:04x} {}",
            if pressed { "DOWN" } else { "UP" }
        );
    }
    let (code, label) = encode(cid);
    crate::input_hook::dispatch(app, state, TriggerKind::Mouse, code, label, pressed);
}

/// Stable matching `code` + friendly `label` for a diverted HID++ control.
///
/// `code` is persisted and matched against the active binding, so it must stay
/// stable across runs.
fn encode(cid: u16) -> (String, String) {
    let code = format!("HIDPP(cid:0x{cid:04x})");
    let label = match cid {
        GESTURE_CID => "Gesture Button".to_string(),
        _ => format!("Mouse Control 0x{cid:04x}"),
    };
    (code, label)
}

// ---- Device handle + HID++ I/O -------------------------------------------

/// An opened HID++ device interface with an event for overlapped I/O.
struct Device {
    handle: HANDLE,
    event: HANDLE,
    in_len: usize,
    out_len: usize,
    debug: bool,
}

impl Device {
    /// Open a HID device path for overlapped read/write and size its reports.
    unsafe fn open(path: &[u16], debug: bool) -> Option<Device> {
        // GENERIC_READ | GENERIC_WRITE as raw bits (stable across crate versions).
        const ACCESS: u32 = 0x8000_0000 | 0x4000_0000;
        let handle = CreateFileW(
            PCWSTR(path.as_ptr()),
            ACCESS,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            None,
            OPEN_EXISTING,
            FILE_FLAG_OVERLAPPED,
            None,
        )
        .ok()?;

        let (in_len, out_len) =
            report_lengths(handle).unwrap_or((DEFAULT_REPORT_LEN, DEFAULT_REPORT_LEN));

        // Manual-reset event, reset before each operation.
        let event = match CreateEventW(None, true, false, PCWSTR::null()) {
            Ok(e) => e,
            Err(_) => {
                let _ = CloseHandle(handle);
                return None;
            }
        };

        Some(Device {
            handle,
            event,
            in_len,
            out_len,
            debug,
        })
    }

    /// Send a HID++ long-report request (no response awaited).
    unsafe fn request(&self, feature_idx: u8, func: u8, params: &[u8]) -> bool {
        let mut req = vec![0u8; self.out_len.max(DEFAULT_REPORT_LEN)];
        req[0] = REPORT_LONG;
        req[1] = DEV_INDEX_BT;
        req[2] = feature_idx;
        req[3] = (func << 4) | SW_ID;
        let n = params.len().min(req.len().saturating_sub(4));
        req[4..4 + n].copy_from_slice(&params[..n]);
        self.write(&req)
    }

    /// Send a request and wait for its matching response, skipping unrelated
    /// reports (e.g. spontaneous events). Returns the 20-byte response.
    unsafe fn call(&self, feature_idx: u8, func: u8, params: &[u8]) -> Option<[u8; 20]> {
        if !self.request(feature_idx, func, params) {
            return None;
        }
        let want_b3 = (func << 4) | SW_ID;
        let deadline = Instant::now() + Duration::from_millis(CALL_TIMEOUT_MS as u64);
        while Instant::now() < deadline {
            let remaining = (deadline.saturating_duration_since(Instant::now())).as_millis() as u32;
            let resp = self.read(remaining.max(1))?;
            if resp.len() < 7 || resp[0] != REPORT_LONG {
                continue;
            }
            // HID++ 2.0 error: feature index 0xFF, echoing the offending
            // feature index (resp[3]) and function|sw byte (resp[4]).
            if resp[2] == 0xFF && resp[3] == feature_idx && resp[4] == want_b3 {
                if self.debug {
                    eprintln!(
                        "[hidpp] error reply: feat=0x{feature_idx:02x} func=0x{func:02x} \
                         code=0x{:02x}",
                        resp[5]
                    );
                }
                return None;
            }
            if resp[2] == feature_idx && resp[3] == want_b3 {
                let mut out = [0u8; 20];
                let n = resp.len().min(20);
                out[..n].copy_from_slice(&resp[..n]);
                return Some(out);
            }
            // Unrelated report (an event arriving mid-handshake): keep waiting.
        }
        None
    }

    /// Resolve a feature id to its per-device feature index via the root feature.
    unsafe fn get_feature(&self, feature_id: u16) -> Option<u8> {
        let params = [(feature_id >> 8) as u8, (feature_id & 0xFF) as u8];
        let resp = self.call(ROOT_FEATURE_IDX, ROOT_FN_GET_FEATURE, &params)?;
        Some(resp[4]) // 0 = unsupported
    }

    /// Log every reprogrammable control the device exposes (diagnostic only).
    /// This is the ground-truth fallback if the gesture button's control id ever
    /// differs from [`GESTURE_CID`].
    unsafe fn enumerate_controls(&self, reprog_idx: u8) {
        let Some(resp) = self.call(reprog_idx, REPROG_FN_GET_COUNT, &[]) else {
            eprintln!("[hidpp] getControlCount failed");
            return;
        };
        let count = resp[4];
        eprintln!("[hidpp] device reports {count} reprogrammable control(s):");
        for i in 0..count {
            let Some(info) = self.call(reprog_idx, REPROG_FN_GET_CID_INFO, &[i]) else {
                continue;
            };
            let cid = ((info[4] as u16) << 8) | info[5] as u16;
            let tid = ((info[6] as u16) << 8) | info[7] as u16;
            let flags = info[8];
            let divertable = flags & CID_FLAG_DIVERTABLE != 0;
            eprintln!(
                "[hidpp]   cid=0x{cid:04x} task=0x{tid:04x} flags=0x{flags:02x} \
                 divertable={divertable}"
            );
        }
    }

    /// Overlapped read of a single report. `timeout_ms` of `u32::MAX` waits
    /// indefinitely. Returns the report bytes, or `None` on timeout/error.
    unsafe fn read(&self, timeout_ms: u32) -> Option<Vec<u8>> {
        let mut buf = vec![0u8; self.in_len.max(DEFAULT_REPORT_LEN)];
        let mut ov: OVERLAPPED = std::mem::zeroed();
        ov.hEvent = self.event;
        let _ = ResetEvent(self.event);

        match ReadFile(self.handle, Some(buf.as_mut_slice()), None, Some(&mut ov)) {
            Ok(()) => {} // completed synchronously
            Err(e) if e.code() == ERROR_IO_PENDING.to_hresult() => {
                if WaitForSingleObject(self.event, timeout_ms) != WAIT_OBJECT_0 {
                    let _ = CancelIoEx(self.handle, Some(&ov));
                    let _ = WaitForSingleObject(self.event, 100);
                    return None;
                }
            }
            Err(_) => return None,
        }

        let mut transferred: u32 = 0;
        if GetOverlappedResult(self.handle, &ov, &mut transferred, false).is_err() {
            return None;
        }
        buf.truncate(transferred as usize);
        Some(buf)
    }

    /// Overlapped write of one output report (padded/truncated to `out_len`).
    unsafe fn write(&self, payload: &[u8]) -> bool {
        let mut buf = vec![0u8; self.out_len.max(DEFAULT_REPORT_LEN)];
        let n = payload.len().min(buf.len());
        buf[..n].copy_from_slice(&payload[..n]);

        let mut ov: OVERLAPPED = std::mem::zeroed();
        ov.hEvent = self.event;
        let _ = ResetEvent(self.event);

        match WriteFile(self.handle, Some(buf.as_slice()), None, Some(&mut ov)) {
            Ok(()) => {}
            Err(e) if e.code() == ERROR_IO_PENDING.to_hresult() => {
                if WaitForSingleObject(self.event, CALL_TIMEOUT_MS) != WAIT_OBJECT_0 {
                    let _ = CancelIoEx(self.handle, Some(&ov));
                    let _ = WaitForSingleObject(self.event, 100);
                    return false;
                }
            }
            Err(_) => return false,
        }

        let mut transferred: u32 = 0;
        GetOverlappedResult(self.handle, &ov, &mut transferred, false).is_ok()
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            let _ = CloseHandle(self.event);
            let _ = CloseHandle(self.handle);
        }
    }
}

/// Query a HID device's input/output report byte lengths (report id included).
unsafe fn report_lengths(handle: HANDLE) -> Option<(usize, usize)> {
    let mut pp = PHIDP_PREPARSED_DATA(0);
    if !HidD_GetPreparsedData(handle, &mut pp) {
        return None;
    }
    let mut caps: HIDP_CAPS = std::mem::zeroed();
    let st = HidP_GetCaps(pp, &mut caps);
    let _ = HidD_FreePreparsedData(pp);
    if st.0 < 0 {
        return None;
    }
    Some((
        caps.InputReportByteLength as usize,
        caps.OutputReportByteLength as usize,
    ))
}

/// Enumerate the device-interface paths of every HID++ long-message collection
/// (`0xFF43`/`0x0202`) present. There may be more than one candidate (e.g. a
/// receiver plus a Bluetooth link), so callers try each in turn.
unsafe fn find_hidpp_paths() -> Vec<Vec<u16>> {
    let mut out = Vec::new();
    let cb = std::mem::size_of::<RAWINPUTDEVICELIST>() as u32;

    let mut count: u32 = 0;
    if GetRawInputDeviceList(None, &mut count, cb) == u32::MAX || count == 0 {
        return out;
    }
    let mut list: Vec<RAWINPUTDEVICELIST> =
        (0..count as usize).map(|_| std::mem::zeroed()).collect();
    let got = GetRawInputDeviceList(Some(list.as_mut_ptr()), &mut count, cb);
    if got == u32::MAX {
        return out;
    }

    for dev in list.iter().take(got as usize) {
        if dev.dwType.0 != RIM_TYPE_HID {
            continue;
        }
        let mut info: RID_DEVICE_INFO = std::mem::zeroed();
        info.cbSize = std::mem::size_of::<RID_DEVICE_INFO>() as u32;
        let mut sz = info.cbSize;
        let r = GetRawInputDeviceInfoW(
            Some(dev.hDevice),
            RIDI_DEVICEINFO,
            Some(&mut info as *mut _ as *mut c_void),
            &mut sz,
        );
        if r == u32::MAX || r == 0 {
            continue;
        }
        let hid = info.Anonymous.hid;
        if hid.usUsagePage != HIDPP_USAGE_PAGE || hid.usUsage != HIDPP_USAGE {
            continue;
        }
        if let Some(name) = device_name(dev.hDevice) {
            out.push(name);
        }
    }
    out
}

/// Fetch a Raw Input device's interface path (a null-terminated wide string
/// usable directly with `CreateFileW`).
unsafe fn device_name(hdevice: HANDLE) -> Option<Vec<u16>> {
    // First call: size in characters.
    let mut sz: u32 = 0;
    GetRawInputDeviceInfoW(Some(hdevice), RIDI_DEVICENAME, None, &mut sz);
    if sz == 0 {
        return None;
    }
    let mut buf = vec![0u16; sz as usize + 1];
    let got = GetRawInputDeviceInfoW(
        Some(hdevice),
        RIDI_DEVICENAME,
        Some(buf.as_mut_ptr() as *mut c_void),
        &mut sz,
    );
    if got == u32::MAX {
        return None;
    }
    if !buf.contains(&0) {
        buf.push(0);
    }
    Some(buf)
}
