# Core

- Desktop app is Tauri v2 multi-window: floating pill (`pill.html`/`src/pill.ts`) plus settings (`settings.html`/`src/settings.ts`).
- Rust backend lives under `src-tauri/src`; `lib.rs` wires Tauri commands, windows, tray, global hooks, audio coordinator thread, and startup model warmup.
- Audio flow: `input_hook` sends `audio::SessionEvent`; `audio.rs` owns cpal stream and calls `transcribe.rs`; frontend state is updated through Tauri events.
- Persistent user data is owned by Rust settings schema in `src-tauri/src/settings.rs`; frontend round-trips via commands in `commands.rs`.
- Read `mem:tech_stack` for framework/build details, `mem:conventions` for local implementation patterns, and `mem:task_completion` for verification.