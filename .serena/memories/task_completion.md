# Task Completion

- Run `rtk npm run build` for frontend TypeScript/Vite validation.
- Run `rtk proxy powershell -NoProfile -Command cargo test --manifest-path src-tauri/Cargo.toml` when Rust behavior/schema changed.
- For full desktop validation, run `rtk npm run tauri -- dev` and manually exercise windows, tray, trigger capture, and transcription flow.
- If Serena memories were changed, the user can run `serena memories check` from the project root.