# Conventions

- Keep frontend windows as separate HTML entrypoints and TS modules; add new windows to both `src-tauri/tauri.conf.json` and `vite.config.ts` rollup inputs.
- Shared appearance behavior belongs in `src/appearance.ts`; windows apply theme/accent on boot and listen to `settings:changed` when needed.
- Rust settings schema fields use `#[serde(default)]` so older settings files survive upgrades; add defaults and update tests when adding persistent fields.
- Tauri commands live in `commands.rs` and must be registered in `lib.rs` `invoke_handler`.
- Backend state changes that affect UI are usually broadcast with `app.emit(...)`; frontend listens via `@tauri-apps/api/event`.
- Existing UI style uses compact sections/cards, CSS variables from `styles.css`, and imperative DOM rendering rather than a frontend framework.