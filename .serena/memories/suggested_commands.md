# Suggested Commands

- `rtk npm run build` - TypeScript/Vite production build.
- `rtk npm run dev` - Vite dev server on port 1420.
- `rtk npm run tauri -- dev` - run full Tauri app in development.
- `rtk proxy powershell -NoProfile -Command cargo test --manifest-path src-tauri/Cargo.toml` - Rust tests from repo root.
- `rtk proxy powershell -NoProfile -Command cargo build --manifest-path src-tauri/Cargo.toml` - Rust compile check from repo root.
- `rtk proxy powershell -NoProfile -Command Get-ChildItem <path> -Recurse` - Windows recursive listing when `rtk ls` is unavailable.