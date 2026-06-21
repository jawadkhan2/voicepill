import { defineConfig } from "vite";

// Tauri expects a fixed port and a multi-page build (one HTML per window).
export default defineConfig({
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: "127.0.0.1",
    // Cargo writes into src-tauri/target while building; never let Vite's
    // file watcher follow it (causes EBUSY crashes on Windows).
    watch: {
      ignored: ["**/src-tauri/**"],
    },
  },
  // Tauri watches src-tauri itself; don't let Vite reload on those changes.
  envPrefix: ["VITE_", "TAURI_"],
  build: {
    target: "es2022",
    rollupOptions: {
      input: {
        pill: "pill.html",
        main: "main.html",
      },
    },
  },
});
