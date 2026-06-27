// The single panel window. Hosts two stacked views — the transcription
// history and the settings panel — and animates between them when the user
// clicks the gear. Owns the things that are shared across both views: the
// window's close behaviour, appearance (theme/accent), and navigation.

import { getCurrentWindow } from "@tauri-apps/api/window";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { applyAppearance, watchSystemTheme, type Appearance } from "./appearance";
import { initHistory } from "./history";
import { cancelPendingCapture, initSettings } from "./settings";
import { checkForUpdate } from "./update";
import { toast } from "./toast";

type View = "transcripts" | "settings";

const FALLBACK_APPEARANCE: Appearance = {
  theme: "system",
  accent: "#6c5ce7",
  pill_scale: 1,
  pill_opacity: 1,
  pill_compact: false,
};

const SUBTITLE: Record<View, string> = {
  transcripts: "Recent transcriptions",
  settings: "Settings",
};

const shell = document.getElementById("app")!;
const navToggle = document.getElementById("nav-toggle") as HTMLButtonElement;
const sub = document.getElementById("app-sub")!;

const win = getCurrentWindow();

// Hide (instead of destroy) on close, so the window re-shows instantly.
win.onCloseRequested(async (event) => {
  event.preventDefault();
  cancelPendingCapture();
  await win.hide();
});

// ---- Views ---------------------------------------------------------------
const historyView = initHistory();
initSettings();

let view: View = "transcripts";

function setView(next: View) {
  if (next === view) return;
  if (view === "settings" && next !== "settings") cancelPendingCapture();
  view = next;
  shell.dataset.view = next; // CSS slides/crossfades the two views
  sub.textContent = SUBTITLE[next];
  const label = next === "transcripts" ? "Open settings" : "Back to transcriptions";
  navToggle.title = label;
  navToggle.setAttribute("aria-label", label);
  if (next === "transcripts") void historyView.refresh();
}

navToggle.onclick = () => setView(view === "transcripts" ? "settings" : "transcripts");

// ---- Custom window chrome (borderless: decorations are drawn by us) -------
const winMin = document.getElementById("win-min") as HTMLButtonElement;
const winMax = document.getElementById("win-max") as HTMLButtonElement;
const winClose = document.getElementById("win-close") as HTMLButtonElement;

winMin.onclick = () => void win.minimize();
winMax.onclick = () => void win.toggleMaximize();
// Match the title-bar close to the OS close: hide (the app lives in the tray).
winClose.onclick = () => {
  cancelPendingCapture();
  void win.hide();
};

/** Reflect the maximized state in the chrome (square corners + restore icon). */
async function syncMaximized() {
  try {
    shell.classList.toggle("maximized", await win.isMaximized());
  } catch {
    /* permission/platform issue — leave chrome as-is */
  }
}
void win.onResized(() => void syncMaximized());
void syncMaximized();

// Edge/corner grips drive native resize on the borderless window.
for (const grip of document.querySelectorAll<HTMLElement>(".grip")) {
  grip.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    // dataset.dir is one of the ResizeDirection string values (e.g. "SouthEast").
    void win.startResizeDragging(grip.dataset.dir as never);
  });
}

// Refresh the list when the window regains focus while it's showing.
void win.onFocusChanged(({ payload: focused }) => {
  if (focused && view === "transcripts") void historyView.refresh();
});

// Backend-driven navigation (pill click, tray menu, no-model prompt).
void win.listen("nav:settings", () => setView("settings"));
void win.listen("nav:transcripts", () => setView("transcripts"));

// ---- Appearance ----------------------------------------------------------
let appearance: Appearance = FALLBACK_APPEARANCE;

void listen<Appearance>("settings:changed", (event) => {
  appearance = event.payload;
  applyAppearance(appearance, false);
});

watchSystemTheme(() => appearance, false);

void (async () => {
  try {
    const loaded = await invoke<Appearance & { onboarded: boolean }>("get_settings");
    appearance = loaded;
    // First run: drop the user straight into settings to bind a key + model.
    if (!loaded.onboarded) setView("settings");
  } catch (e) {
    console.error("failed to load settings", e);
  }
  applyAppearance(appearance, false);
})();

// Silent OTA check on launch: if a newer version is on GitHub Releases, nudge
// the user with a toast. Installing is opt-in via Settings → Updates.
void (async () => {
  try {
    const update = await checkForUpdate();
    if (update) toast(`VoicePill ${update.version} is available — see Settings → Updates`, 5000);
  } catch {
    /* offline or endpoint unreachable — stay quiet on startup */
  }
})();
