//! Inject transcribed text into whatever field currently has focus.
//!
//! Per the user's choice we paste rather than type: stash the current clipboard,
//! set our text, synthesize the platform paste shortcut, then (optionally)
//! restore the old clipboard. Pasting is instant regardless of length and
//! preserves Unicode, where synthesizing each keystroke would be slow and
//! locale-dependent.

use std::time::Duration;

use arboard::{Clipboard, ImageData};
use enigo::{
    Direction::{Click, Press, Release},
    Enigo, Key, Keyboard, Settings,
};

/// A snapshot of the clipboard taken before we overwrite it, so the prior
/// contents can be restored after the paste lands. arboard exposes text and
/// images; whichever the clipboard held is captured (text wins if both exist).
enum Snapshot {
    Text(String),
    Image(ImageData<'static>),
    Empty,
}

fn capture(clipboard: &mut Clipboard) -> Snapshot {
    if let Ok(text) = clipboard.get_text() {
        return Snapshot::Text(text);
    }
    if let Ok(img) = clipboard.get_image() {
        return Snapshot::Image(img);
    }
    Snapshot::Empty
}

fn restore(clipboard: &mut Clipboard, snap: Snapshot) {
    match snap {
        Snapshot::Text(t) => {
            let _ = clipboard.set_text(t);
        }
        Snapshot::Image(img) => {
            let _ = clipboard.set_image(img);
        }
        // Nothing recognizable was there (or it was empty); clear our text so we
        // don't leave the transcript lingering on the clipboard.
        Snapshot::Empty => {
            let _ = clipboard.clear();
        }
    }
}

/// Paste `text` into the focused field. When `restore_clipboard` is set, the
/// user's previous clipboard contents (text or image) are put back after the
/// paste lands.
pub fn paste_text(text: &str, restore_clipboard: bool) -> Result<(), String> {
    if text.is_empty() {
        return Ok(());
    }

    let mut clipboard = Clipboard::new().map_err(|e| e.to_string())?;
    let previous = if restore_clipboard {
        Some(capture(&mut clipboard))
    } else {
        None
    };

    clipboard
        .set_text(text)
        .map_err(|e| format!("clipboard set failed: {e}"))?;
    // Give the OS a moment to register the new clipboard owner before pasting.
    std::thread::sleep(Duration::from_millis(40));

    let mut enigo = Enigo::new(&Settings::default()).map_err(|e| e.to_string())?;
    let modifier = paste_modifier();
    enigo.key(modifier, Press).map_err(|e| e.to_string())?;
    enigo
        .key(Key::Unicode('v'), Click)
        .map_err(|e| e.to_string())?;
    enigo.key(modifier, Release).map_err(|e| e.to_string())?;

    if let Some(prev) = previous {
        // Wait for the target app to actually read the clipboard before we
        // overwrite it again, otherwise it may paste the restored contents.
        std::thread::sleep(Duration::from_millis(120));
        restore(&mut clipboard, prev);
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn paste_modifier() -> Key {
    Key::Meta
}

#[cfg(not(target_os = "macos"))]
fn paste_modifier() -> Key {
    Key::Control
}
