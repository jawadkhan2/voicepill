import { getCurrentWindow } from "@tauri-apps/api/window";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { applyAppearance, watchSystemTheme, type Appearance } from "./appearance";

type PillState = "idle" | "listening" | "busy" | "error";

interface AudioError {
  kind: string;
  message: string;
  detail: string;
  sticky: boolean;
}

const pill = document.getElementById("pill") as HTMLDivElement;
const label = document.getElementById("pill-label") as HTMLDivElement;
const DEFAULT_TITLE = pill.title;

const STATE_LABEL: Record<PillState, string> = {
  idle: "Ready",
  listening: "Listening…",
  busy: "Transcribing…",
  error: "Error",
};

const NO_MIC_ERROR: AudioError = {
  kind: "no_input_device",
  message: "No microphone detected",
  detail: "Plug in or enable a microphone, then try again.",
  sticky: true,
};

let currentState: PillState = "idle";
let stickyAudioError: AudioError | null = null;
let errorReset: number | undefined;
let lastKnownHasDevices: boolean | null = null;

function applyState(state: PillState, text = STATE_LABEL[state], title = DEFAULT_TITLE) {
  currentState = state;
  pill.classList.remove("state-idle", "state-listening", "state-busy", "state-error");
  pill.classList.add(`state-${state}`);
  label.textContent = text;
  pill.title = title;
  pill.setAttribute("aria-label", title);
}

function audioErrorLabel(error: AudioError): string {
  if (error.kind === "no_input_device") return "No mic";
  if (error.kind === "selected_input_unavailable") return "Mic missing";
  if (error.kind === "input_stream_lost") return "Mic lost";
  if (error.kind === "model_load_failed") return "Model load failed";
  if (error.kind === "no_model") return "Pick a model";
  return "Mic error";
}

function audioErrorTitle(error: AudioError): string {
  return error.detail ? `${error.message}. ${error.detail}` : error.message;
}

function applyAudioError(error: AudioError) {
  applyState("error", audioErrorLabel(error), audioErrorTitle(error));
}

function setState(state: PillState) {
  if (state === "listening") stickyAudioError = null;
  if (state === "idle" && stickyAudioError) {
    applyAudioError(stickyAudioError);
    return;
  }
  if (state === "error" && stickyAudioError) {
    applyAudioError(stickyAudioError);
    return;
  }
  applyState(state);
}

function showAudioError(error: AudioError) {
  window.clearTimeout(errorReset);
  if (error.sticky) {
    stickyAudioError = error;
    applyAudioError(error);
    return;
  }
  const previousSticky = stickyAudioError;
  stickyAudioError = null;
  applyAudioError(error);
  errorReset = window.setTimeout(() => {
    if (previousSticky && lastKnownHasDevices === false) stickyAudioError = previousSticky;
    setState("idle");
  }, 2400);
}

async function openSettings() {
  await invoke("open_settings_window");
}

async function toggleHistory() {
  await invoke("toggle_history_window");
}

// Distinguish a click (open history) from a drag (move the pill).
// On mousedown we remember the position; if the pointer moves past a small
// threshold we hand off to the OS drag loop, otherwise mouseup = click.
let down: { x: number; y: number } | null = null;

pill.addEventListener("mousedown", (e) => {
  if (e.button !== 0) return;
  down = { x: e.clientX, y: e.clientY };
});

pill.addEventListener("mousemove", (e) => {
  if (!down) return;
  if (Math.abs(e.clientX - down.x) > 4 || Math.abs(e.clientY - down.y) > 4) {
    down = null;
    void getCurrentWindow().startDragging();
  }
});

window.addEventListener("mouseup", () => {
  if (down) {
    down = null;
    void toggleHistory();
  }
});

// Persist the pill's position after the user drags it (debounced).
let posSave: number | undefined;
void getCurrentWindow().onMoved(({ payload }) => {
  window.clearTimeout(posSave);
  posSave = window.setTimeout(() => {
    void invoke("save_pill_pos", { x: payload.x, y: payload.y });
  }, 400);
});

// Backend drives the pill state through this event.
void listen<PillState>("pill:state", (event) => {
  const state = event.payload;
  setState(state);
  // The error state is momentary — flash, then fall back to idle.
  window.clearTimeout(errorReset);
  if (state === "error" && !stickyAudioError) {
    errorReset = window.setTimeout(() => setState("idle"), 1800);
  }
});

void listen<AudioError>("audio:error", (event) => showAudioError(event.payload));

// Model is being loaded into VRAM (boot warm-up or a "Use" click). Show a
// spinner on the pill so the user knows the GPU is busy getting ready.
void listen("model:loading", () => {
  window.clearTimeout(errorReset);
  setState("busy");
  label.textContent = "Loading model…";
});
void listen("model:loaded", () => setState("idle"));
void listen<{ id: string; error: string }>("model:load-error", () => {
  showAudioError({
    kind: "model_load_failed",
    message: "Model load failed",
    detail: "Open settings to retry or choose another model.",
    sticky: false,
  });
});

// Triggered with no model selected: briefly tell the user and open settings.
void listen("transcribe:no-model", () => {
  showAudioError({
    kind: "no_model",
    message: "Pick a model",
    detail: "Open settings and choose a transcription model.",
    sticky: false,
  });
  void openSettings();
});

setState("idle");

async function refreshAudioAvailability() {
  try {
    const devices = await invoke<string[]>("list_audio_devices");
    const hasDevices = devices.length > 0;
    if (lastKnownHasDevices === hasDevices) return;
    lastKnownHasDevices = hasDevices;
    if (!hasDevices) {
      showAudioError(NO_MIC_ERROR);
      return;
    }
    if (stickyAudioError) {
      stickyAudioError = null;
      window.clearTimeout(errorReset);
      if (currentState === "error") setState("idle");
    }
  } catch {
    /* Keep the last known pill state if device enumeration fails. */
  }
}

void refreshAudioAvailability();
window.setInterval(() => void refreshAudioAvailability(), 2500);

// ---- Appearance (theme / accent / pill size + opacity + compact) ---------
// The pill reads its look from settings on boot, follows the OS theme when set
// to "system", and re-applies live when the settings window changes anything.
const FALLBACK_APPEARANCE: Appearance = {
  theme: "system",
  accent: "#6c5ce7",
  pill_scale: 1,
  pill_opacity: 1,
  pill_compact: false,
};
let appearance: Appearance = FALLBACK_APPEARANCE;

watchSystemTheme(() => appearance, true);

void listen<Appearance>("settings:changed", (e) => {
  appearance = e.payload;
  applyAppearance(appearance, true);
});
void listen<Appearance>("appearance:preview", (e) => {
  appearance = e.payload;
  applyAppearance(appearance, true);
});

void (async () => {
  // The pill webview boots concurrently with the Rust `setup` hook, so a fast
  // first paint can fire `get_settings` before `SettingsState` is managed —
  // the command then rejects and we'd be stuck on the fallback (purple) accent
  // until the next live update. Retry a few times so the saved accent is
  // applied on initial startup, not just after the user changes it.
  for (let attempt = 0; attempt < 10; attempt++) {
    try {
      appearance = await invoke<Appearance>("get_settings");
      break;
    } catch (e) {
      if (attempt === 9) console.error("failed to load appearance", e);
      else await new Promise((r) => setTimeout(r, 100));
    }
  }
  applyAppearance(appearance, true);
})();
