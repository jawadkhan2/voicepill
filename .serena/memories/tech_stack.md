# Tech Stack

- Frontend: TypeScript, Vite multi-page build, strict TS (`noUnusedLocals`, `noUnusedParameters`, `strict`).
- Desktop shell: Tauri v2 with `@tauri-apps/api` and Rust command/event IPC.
- Backend: Rust 2021. Key crates: `tauri`, `tauri-plugin-autostart`, `cpal`, `rubato`, `whisper-rs` with CUDA, `rdev`, `arboard`, `enigo`, `nvml-wrapper`, Windows HID APIs.
- Package manager: npm (`package-lock.json` present). Tauri config uses Vite dev URL `127.0.0.1:1420`.
- Windows-first project; some HID++ code is `cfg(windows)`.