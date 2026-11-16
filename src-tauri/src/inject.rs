//! Inject transcribed text into whatever field currently has focus.
//!
//! Per the user's choice we paste rather than type: stash the current clipboard,
//! set our text, synthesize the platform paste shortcut, then (optionally)
//! restore the old clipboard. Pasting is instant regardless of length and
//! preserves Unicode, where synthesizing each keystroke would be slow and
//! locale-dependent.

use std::path::PathBuf;
use std::time::Duration;

use arboard::{Clipboard, ImageData};
use enigo::{
    Direction::{Click, Press, Release},
    Enigo, Key, Keyboard, Settings,
};

/// A snapshot of the clipboard taken before we overwrite it, so the prior
/// contents can be restored after the paste lands. Prefer richer formats first
/// so file lists, images, and HTML are not flattened to plain text.
enum Snapshot {
    Text(String),
    Html {
        html: String,
        alt_text: Option<String>,
    },
    Image(ImageData<'static>),
    FileList(Vec<PathBuf>),
    Empty,
    Unsupported,
}

fn capture(clipboard: &mut Clipboard) -> Snapshot {
    if let Ok(files) = clipboard.get().file_list() {
        if !files.is_empty() {
            return Snapshot::FileList(files);
        }
    }
    if let Ok(img) = clipboard.get_image() {
        return Snapshot::Image(img);
    }
    let html = clipboard.get().html().ok();
    let text = clipboard.get_text().ok();
    if let Some(html) = html {
        return Snapshot::Html {
            html,
            alt_text: text,
        };
    }
    if let Some(text) = text {
        return Snapshot::Text(text);
    }
    unknown_snapshot()
}

fn restore(clipboard: &mut Clipboard, snap: Snapshot) {
    match snap {
        Snapshot::Text(t) => {
            let _ = clipboard.set_text(t);
        }
        Snapshot::Html { html, alt_text } => {
            let _ = clipboard.set().html(html, alt_text);
        }
        Snapshot::Image(img) => {
            let _ = clipboard.set_image(img);
        }
        Snapshot::FileList(files) => {
            let _ = clipboard.set().file_list(&files);
        }
        Snapshot::Empty => {
            let _ = clipboard.clear();
        }
        Snapshot::Unsupported => {}
    }
}

#[cfg(windows)]
fn unknown_snapshot() -> Snapshot {
    match clipboard_has_data() {
        Some(true) | None => Snapshot::Unsupported,
        Some(false) => Snapshot::Empty,
    }
}

#[cfg(not(windows))]
fn unknown_snapshot() -> Snapshot {
    Snapshot::Empty
}

#[cfg(windows)]
fn clipboard_has_data() -> Option<bool> {
    use windows::Win32::System::DataExchange::{
        CloseClipboard, CountClipboardFormats, OpenClipboard,
    };

    unsafe {
        if OpenClipboard(None).is_err() {
            return None;
        }
        let count = CountClipboardFormats();
        let _ = CloseClipboard();
        Some(count > 0)
    }
}

/// Paste `text` into the focused field. When `restore_clipboard` is set, the
/// user's previous supported clipboard contents are put back after the
/// paste lands.
///
/// Never silently drops the text: if the clipboard is held by another app
/// (clipboard managers, browsers and Office briefly lock it all the time) we
/// retry, and if the current clipboard contents can't be snapshotted for
/// restore we type the text directly instead of refusing to paste.
pub fn paste_text(text: &str, restore_clipboard: bool) -> Result<(), String> {
    if text.is_empty() {
        return Ok(());
    }

    let mut clipboard = retry(|| Clipboard::new().map_err(|e| e.to_string()))
        .map_err(|e| format!("clipboard unavailable: {e}"))?;
    let previous = if restore_clipboard {
        // `Unsupported` is usually a transient "clipboard is open in another
        // process" rather than a truly exotic format, so re-check a few times.
        let mut snap = capture(&mut clipboard);
        for _ in 0..RETRIES {
            if !matches!(snap, Snapshot::Unsupported) {
                break;
            }
            std::thread::sleep(RETRY_DELAY);
            snap = capture(&mut clipboard);
        }
        if matches!(snap, Snapshot::Unsupported) {
            // Overwriting would destroy data we can't put back; refusing would
            // lose the dictation. Type it instead — the clipboard is untouched.
            crate::vlog!("[inject] clipboard not restorable; typing text instead of pasting");
            return type_text(text);
        }
        Some(snap)
    } else {
        None
    };

    retry(|| clipboard.set_text(text).map_err(|e| e.to_string()))
        .map_err(|e| format!("clipboard set failed: {e}"))?;
    // Give the OS a moment to register the new clipboard owner before pasting.
    std::thread::sleep(Duration::from_millis(40));

    let mut enigo = Enigo::new(&Settings::default()).map_err(|e| e.to_string())?;
    let modifier = paste_modifier();
    enigo.key(modifier, Press).map_err(|e| e.to_string())?;
    let click = enigo.key(Key::Unicode('v'), Click).map_err(|e| e.to_string());
    // Always release the modifier, even if the click failed, so Ctrl is never
    // left logically held down.
    enigo.key(modifier, Release).map_err(|e| e.to_string())?;
    click?;

    if let Some(prev) = previous {
        // Wait for the target app to actually read the clipboard before we
        // overwrite it again, otherwise it pastes the *restored* contents.
        // Electron apps and terminals under load can take well over 100 ms.
        std::thread::sleep(RESTORE_DELAY);
        restore(&mut clipboard, prev);
    }

    Ok(())
}

const RETRIES: usize = 5;
const RETRY_DELAY: Duration = Duration::from_millis(30);
const RESTORE_DELAY: Duration = Duration::from_millis(250);

/// Run a clipboard operation, retrying briefly while another process holds
/// the clipboard open.
fn retry<T>(mut op: impl FnMut() -> Result<T, String>) -> Result<T, String> {
    let mut last = String::new();
    for attempt in 0..=RETRIES {
        match op() {
            Ok(v) => return Ok(v),
            Err(e) => last = e,
        }
        if attempt < RETRIES {
            std::thread::sleep(RETRY_DELAY);
        }
    }
    Err(last)
}

/// Fallback delivery: synthesize the text as Unicode keystrokes.
fn type_text(text: &str) -> Result<(), String> {
    let mut enigo = Enigo::new(&Settings::default()).map_err(|e| e.to_string())?;
    enigo.text(text).map_err(|e| e.to_string())
}

#[cfg(target_os = "macos")]
fn paste_modifier() -> Key {
    Key::Meta
}

#[cfg(not(target_os = "macos"))]
fn paste_modifier() -> Key {
    Key::Control
}
