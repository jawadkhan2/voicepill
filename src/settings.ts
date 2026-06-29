import { invoke } from "@tauri-apps/api/core";
import { listen, emit } from "@tauri-apps/api/event";
import { getVersion } from "@tauri-apps/api/app";
import { applyAppearance, type Appearance, type Theme } from "./appearance";
import { toast } from "./toast";
import { checkForUpdate, installAndRelaunch, type Update } from "./update";

// Line icons keyed by section title, shown left of each card heading.
const SECTION_ICONS: Record<string, string> = {
  Trigger:
    '<svg viewBox="0 0 24 24"><rect x="3" y="6" width="18" height="12" rx="2"/><path d="M7 10h.01M11 10h.01M15 10h.01M8 14h8"/></svg>',
  Transcription: '<svg viewBox="0 0 24 24"><path d="M4 7h16M4 12h12M4 17h8"/></svg>',
  GPU: '<svg viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="1"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3"/></svg>',
  Audio: '<svg viewBox="0 0 24 24"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/></svg>',
  Appearance:
    '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4 12H2M22 12h-2M5 5l1.5 1.5M17.5 17.5L19 19M19 5l-1.5 1.5M6.5 17.5L5 19"/></svg>',
  General:
    '<svg viewBox="0 0 24 24"><path d="M4 6h16M4 12h16M4 18h16"/><circle cx="9" cy="6" r="2"/><circle cx="15" cy="12" r="2"/><circle cx="8" cy="18" r="2"/></svg>',
  Updates:
    '<svg viewBox="0 0 24 24"><path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 3v6h-6"/></svg>',
};

// ---- Types mirroring the Rust `Settings` struct --------------------------
type TriggerKind = "mouse" | "key";
type TriggerMode = "push_to_talk" | "toggle";

interface TriggerBinding {
  kind: TriggerKind;
  code: string;
  label: string;
}

interface Settings {
  trigger: TriggerBinding | null;
  trigger_mode: TriggerMode;
  model: string | null;
  language: string;
  audio_device: string | null;
  autostart: boolean;
  restore_clipboard: boolean;
  theme: Theme;
  accent: string;
  pill_scale: number;
  pill_opacity: number;
  pill_compact: boolean;
  onboarded: boolean;
}

interface ModelInfo {
  id: string;
  label: string;
  size_mb: number;
  downloaded: boolean;
}

interface GpuInfo {
  available: boolean;
  name: string;
  driver: string;
  vram_total_mb: number;
  vram_used_mb: number;
  vram_process_mb: number;
}

interface AudioError {
  kind: string;
  message: string;
  detail: string;
  sticky: boolean;
}

const LANGUAGES: Array<[string, string]> = [
  ["auto", "Auto-detect"],
  ["en", "English"],
  ["es", "Spanish"],
  ["fr", "French"],
  ["de", "German"],
  ["it", "Italian"],
  ["pt", "Portuguese"],
  ["nl", "Dutch"],
  ["ru", "Russian"],
  ["zh", "Chinese"],
  ["ja", "Japanese"],
  ["ko", "Korean"],
  ["hi", "Hindi"],
  ["ar", "Arabic"],
];

let settings: Settings;
let capturing = false;
let audioDevices: string[] = [];
let audioError: AudioError | null = null;
let audioRefreshTimer: number | undefined;
let models: ModelInfo[] = [];
// modelId -> percent (0-100) while a download is in flight.
const downloading = new Map<string, number>();
// modelId -> last download error message (cleared on retry/success).
const downloadErrors = new Map<string, string>();
// modelIds currently being loaded into VRAM (after a "Use" click or at boot).
const loadingModels = new Set<string>();
// modelId -> short "Loaded in 1.8s" note shown after a successful GPU load.
const loadStatus = new Map<string, string>();
// modelId -> GPU load error message.
const loadErrors = new Map<string, string>();
// Latest GPU snapshot, polled while settings is open. null = unknown/no GPU.
let gpu: GpuInfo | null = null;

const root = document.getElementById("settings-root")!;

export async function initSettings() {
  settings = await invoke<Settings>("get_settings");
  await refreshAudioDevices(false);
  await refreshModels();
  render();
  if (audioRefreshTimer === undefined) {
    audioRefreshTimer = window.setInterval(() => void refreshAudioDevices(), 2500);
  }
  // Poll GPU/VRAM stats while settings is open (updates in place, no re-render).
  void pollGpu();
  window.setInterval(() => void pollGpu(), 1500);
}

async function refreshModels() {
  try {
    models = await invoke<ModelInfo[]>("list_models");
  } catch {
    models = [];
  }
}

function sameStringList(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

async function refreshAudioDevices(renderOnChange = true) {
  let next: string[];
  try {
    next = await invoke<string[]>("list_audio_devices");
  } catch {
    next = [];
  }
  if (sameStringList(audioDevices, next)) return;
  audioDevices = next;
  if (audioDevices.length > 0 && audioError?.sticky) audioError = null;
  if (renderOnChange) render();
}

/** Human label for a model id (falls back to the id if not yet loaded). */
function modelLabel(id: string): string {
  return models.find((m) => m.id === id)?.label ?? id;
}

// ---- Model download progress --------------------------------------------
// Progress fires many times a second; updating the bar in place (instead of a
// full re-render) keeps the rest of the panel — including other rows' buttons —
// from being torn down and rebuilt mid-click.
listen<{ id: string; received: number; total: number }>("model:progress", (e) => {
  const { id, received, total } = e.payload;
  const pct = total > 0 ? Math.round((received / total) * 100) : 0;
  downloading.set(id, pct);
  const fill = document.querySelector<HTMLElement>(`.progress-fill[data-id="${id}"]`);
  const lbl = document.querySelector<HTMLElement>(`.progress-label[data-id="${id}"]`);
  if (fill && lbl) {
    fill.style.width = `${pct}%`;
    lbl.textContent = `${pct}%`;
  } else {
    render(); // bar not built yet (first tick) — build it once
  }
});

listen<{ id: string }>("model:done", async (e) => {
  downloading.delete(e.payload.id);
  await refreshModels();
  toast(`${modelLabel(e.payload.id)} downloaded`);
  // Auto-select the first model the user downloads (also loads it into VRAM).
  if (!settings.model) {
    await selectModel(e.payload.id);
    return;
  }
  render();
});

listen<{ id: string; error: string }>("model:error", (e) => {
  downloading.delete(e.payload.id);
  downloadErrors.set(e.payload.id, e.payload.error);
  console.error("model download failed", e.payload.error);
  toast("Download failed");
  render();
});

// ---- Model GPU load (VRAM) ----------------------------------------------
listen<{ id: string }>("model:loading", (e) => {
  loadingModels.add(e.payload.id);
  loadErrors.delete(e.payload.id);
  render();
});

listen<{ id: string; ms: number; gpu: GpuInfo }>("model:loaded", (e) => {
  loadingModels.delete(e.payload.id);
  loadErrors.delete(e.payload.id);
  loadStatus.set(
    e.payload.id,
    e.payload.ms > 0 ? `Loaded in ${(e.payload.ms / 1000).toFixed(1)}s` : "Loaded on GPU",
  );
  if (e.payload.gpu) gpu = e.payload.gpu;
  toast(`${modelLabel(e.payload.id)} ready`);
  render();
});

listen<{ id: string; error: string }>("model:load-error", (e) => {
  loadingModels.delete(e.payload.id);
  loadErrors.set(e.payload.id, e.payload.error);
  console.error("model load failed", e.payload.error);
  toast("Model load failed");
  render();
});

listen<AudioError>("audio:error", (e) => {
  audioError = e.payload;
  toast(e.payload.message);
  void refreshAudioDevices(false);
  render();
});

async function downloadModel(id: string) {
  downloadErrors.delete(id);
  downloading.set(id, 0);
  render();
  await invoke("download_model", { id });
}

async function deleteModel(id: string) {
  await invoke("delete_model", { id });
  if (settings.model === id) {
    settings.model = null;
    await persist();
  }
  await refreshModels();
  render();
}

async function selectModel(id: string) {
  // Optimistically reflect the choice + loading state; the backend persists and
  // streams model:loading/loaded as it pulls the model into VRAM.
  loadErrors.delete(id);
  loadStatus.delete(id);
  loadingModels.add(id);
  settings.model = id;
  render();
  settings = await invoke<Settings>("select_model", { id });
  render();
}

/** Persist the current `settings` object and refresh from the canonical copy. */
async function persist() {
  settings = await invoke<Settings>("update_settings", { settings });
}

// ---- Bind capture --------------------------------------------------------
listen<TriggerBinding>("bind:captured", async (e) => {
  capturing = false;
  settings.trigger = e.payload;
  if (!settings.onboarded) settings.onboarded = true;
  await persist();
  render();
});

listen("bind:cancelled", () => {
  capturing = false;
  render();
});

async function startCapture() {
  capturing = true;
  render();
  await invoke("start_bind_capture");
}

async function cancelCapture() {
  capturing = false;
  await invoke("cancel_bind_capture");
  render();
}

export function cancelPendingCapture() {
  if (capturing) void cancelCapture();
}

// ---- Rendering -----------------------------------------------------------
function render() {
  root.innerHTML = "";
  root.appendChild(triggerSection());
  root.appendChild(transcriptionSection());
  root.appendChild(gpuSection());
  root.appendChild(audioSection());
  root.appendChild(appearanceSection());
  root.appendChild(generalSection());
  root.appendChild(updatesSection());
  fillGpu(); // populate the freshly-built GPU section from the latest snapshot
  void fillVersion(); // async: stamp the current app version once known
}

function section(title: string): HTMLElement {
  const s = document.createElement("section");
  s.className = "card";
  const h = document.createElement("h2");
  h.className = "card-title";
  const icon = SECTION_ICONS[title];
  if (icon) h.innerHTML = icon;
  const label = document.createElement("span");
  label.textContent = title;
  h.appendChild(label);
  s.appendChild(h);
  return s;
}

function row(label: string, control: HTMLElement, hint?: string): HTMLElement {
  const r = document.createElement("div");
  r.className = "row";
  const left = document.createElement("div");
  left.className = "row-label";
  left.textContent = label;
  if (hint) {
    const h = document.createElement("span");
    h.className = "row-hint";
    h.textContent = hint;
    left.appendChild(h);
  }
  r.appendChild(left);
  r.appendChild(control);
  return r;
}

function notice(kind: "error" | "warning", title: string, detail: string): HTMLElement {
  const n = document.createElement("div");
  n.className = `notice ${kind}`;
  const h = document.createElement("div");
  h.className = "notice-title";
  h.textContent = title;
  const p = document.createElement("div");
  p.className = "notice-detail";
  p.textContent = detail;
  n.appendChild(h);
  n.appendChild(p);
  return n;
}

function triggerSection(): HTMLElement {
  const s = section("Trigger");

  // Current binding + rebind button
  const bindWrap = document.createElement("div");
  bindWrap.className = "bind-wrap";

  const badge = document.createElement("span");
  badge.className = "key-badge";
  badge.textContent = capturing
    ? "Press any button…"
    : settings.trigger?.label ?? "Not bound";
  if (capturing) badge.classList.add("capturing");
  if (!settings.trigger && !capturing) badge.classList.add("unbound");
  bindWrap.appendChild(badge);

  const btn = document.createElement("button");
  btn.className = "btn";
  if (capturing) {
    btn.textContent = "Cancel";
    btn.onclick = cancelCapture;
  } else {
    btn.textContent = settings.trigger ? "Rebind" : "Bind";
    btn.onclick = startCapture;
  }
  bindWrap.appendChild(btn);

  s.appendChild(
    row(
      "Trigger button",
      bindWrap,
      capturing ? "Press a mouse button or key — Esc to cancel" : "Mouse side-buttons supported",
    ),
  );

  // Mode toggle
  const modeWrap = document.createElement("div");
  modeWrap.className = "seg";
  for (const [val, lbl] of [
    ["push_to_talk", "Push to talk"],
    ["toggle", "Toggle"],
  ] as Array<[TriggerMode, string]>) {
    const opt = document.createElement("button");
    opt.className = "seg-opt";
    opt.textContent = lbl;
    if (settings.trigger_mode === val) opt.classList.add("active");
    opt.onclick = async () => {
      if (settings.trigger_mode === val) return;
      settings.trigger_mode = val;
      await persist();
      render();
    };
    modeWrap.appendChild(opt);
  }
  s.appendChild(
    row(
      "Mode",
      modeWrap,
      settings.trigger_mode === "push_to_talk"
        ? "Hold to record, release to transcribe"
        : "Click to start, click again to stop",
    ),
  );

  return s;
}

function transcriptionSection(): HTMLElement {
  const s = section("Transcription");

  const list = document.createElement("div");
  list.className = "model-list";
  for (const m of models) {
    list.appendChild(modelRow(m));
  }
  s.appendChild(list);

  const lang = document.createElement("select");
  lang.className = "select";
  for (const [val, lbl] of LANGUAGES) {
    const o = document.createElement("option");
    o.value = val;
    o.textContent = lbl;
    if (settings.language === val) o.selected = true;
    lang.appendChild(o);
  }
  lang.onchange = async () => {
    settings.language = lang.value;
    await persist();
  };
  s.appendChild(row("Language", lang));

  return s;
}

function modelRow(m: ModelInfo): HTMLElement {
  const r = document.createElement("div");
  r.className = "model-row";
  const active = settings.model === m.id;
  if (active) r.classList.add("active");

  const info = document.createElement("div");
  info.className = "model-info";
  const name = document.createElement("span");
  name.className = "model-name";
  name.textContent = m.label;
  const size = document.createElement("span");
  size.className = "model-size";
  size.textContent = m.size_mb >= 1024
    ? `${(m.size_mb / 1024).toFixed(1)} GB`
    : `${m.size_mb} MB`;
  info.appendChild(name);
  info.appendChild(size);
  // Small note under the size: GPU load result for the active model, or an error.
  const loadErr = loadErrors.get(m.id);
  const note = loadErr ? "Load failed" : active ? loadStatus.get(m.id) : undefined;
  if (note) {
    const n = document.createElement("span");
    n.className = loadErr ? "model-note error" : "model-note";
    n.textContent = note;
    if (loadErr) n.title = loadErr;
    info.appendChild(n);
  }
  r.appendChild(info);

  const actions = document.createElement("div");
  actions.className = "model-actions";

  const pct = downloading.get(m.id);
  if (pct !== undefined) {
    const bar = document.createElement("div");
    bar.className = "progress";
    const fill = document.createElement("div");
    fill.className = "progress-fill";
    fill.style.width = `${pct}%`;
    fill.dataset.id = m.id;
    bar.appendChild(fill);
    const lbl = document.createElement("span");
    lbl.className = "progress-label";
    lbl.textContent = `${pct}%`;
    lbl.dataset.id = m.id;
    actions.appendChild(bar);
    actions.appendChild(lbl);
  } else if (loadingModels.has(m.id)) {
    const loading = document.createElement("span");
    loading.className = "model-loading";
    const spin = document.createElement("span");
    spin.className = "mini-spinner";
    loading.appendChild(spin);
    loading.appendChild(document.createTextNode("Loading…"));
    actions.appendChild(loading);
  } else if (m.downloaded) {
    const sel = document.createElement("button");
    sel.className = active ? "btn btn-ghost" : "btn";
    sel.textContent = active ? "Active" : "Use";
    sel.disabled = active;
    sel.onclick = () => selectModel(m.id);
    actions.appendChild(sel);

    const del = document.createElement("button");
    del.className = "btn btn-ghost";
    del.textContent = "Delete";
    del.onclick = () => deleteModel(m.id);
    actions.appendChild(del);
  } else {
    const failed = downloadErrors.get(m.id);
    if (failed) {
      const err = document.createElement("span");
      err.className = "model-error";
      err.textContent = "Download failed";
      err.title = failed;
      actions.appendChild(err);
    }
    const dl = document.createElement("button");
    dl.className = "btn";
    dl.textContent = failed ? "Retry" : "Download";
    dl.onclick = () => downloadModel(m.id);
    actions.appendChild(dl);
  }

  r.appendChild(actions);
  return r;
}

// ---- GPU panel -----------------------------------------------------------
function gpuSection(): HTMLElement {
  const s = section("GPU");

  const dev = document.createElement("div");
  dev.className = "row";
  const devLabel = document.createElement("div");
  devLabel.className = "row-label";
  devLabel.appendChild(document.createTextNode("Device"));
  const hint = document.createElement("span");
  hint.className = "row-hint";
  hint.id = "gpu-name";
  hint.textContent = "Detecting…";
  devLabel.appendChild(hint);
  const drv = document.createElement("span");
  drv.className = "muted-inline";
  drv.id = "gpu-driver";
  dev.appendChild(devLabel);
  dev.appendChild(drv);
  s.appendChild(dev);

  const vram = document.createElement("div");
  vram.className = "vram";
  vram.innerHTML = `
    <div class="vram-head">
      <span>VRAM</span>
      <span class="progress-label" id="vram-text"></span>
    </div>
    <div class="vram-bar"><div class="vram-fill" id="vram-fill"></div></div>
    <div class="vram-sub" id="vram-sub"></div>`;
  s.appendChild(vram);
  return s;
}

/** Push the latest `gpu` snapshot into the GPU section's elements (no re-render). */
function fillGpu() {
  const nameEl = document.getElementById("gpu-name");
  if (!nameEl) return; // section not mounted
  const drv = document.getElementById("gpu-driver")!;
  const text = document.getElementById("vram-text")!;
  const fill = document.getElementById("vram-fill") as HTMLElement;
  const sub = document.getElementById("vram-sub")!;

  if (!gpu || !gpu.available) {
    nameEl.textContent = "No NVIDIA GPU detected";
    drv.textContent = "";
    text.textContent = "";
    fill.style.width = "0%";
    sub.textContent = "";
    return;
  }
  nameEl.textContent = gpu.name || "GPU";
  drv.textContent = gpu.driver ? `Driver ${gpu.driver}` : "";
  const pct = gpu.vram_total_mb > 0
    ? Math.round((gpu.vram_used_mb / gpu.vram_total_mb) * 100)
    : 0;
  text.textContent = `${fmtMb(gpu.vram_used_mb)} / ${fmtMb(gpu.vram_total_mb)}`;
  fill.style.width = `${pct}%`;
  sub.textContent = gpu.vram_process_mb > 0
    ? `VoicePill is using ${fmtMb(gpu.vram_process_mb)} of VRAM`
    : "";
}

function fmtMb(mb: number): string {
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb} MB`;
}

async function pollGpu() {
  try {
    gpu = await invoke<GpuInfo>("gpu_info");
  } catch {
    gpu = null;
  }
  fillGpu();
}

function audioSection(): HTMLElement {
  const s = section("Audio");
  const selectedMissing = Boolean(
    settings.audio_device && !audioDevices.includes(settings.audio_device),
  );

  if (audioDevices.length === 0) {
    s.appendChild(
      notice(
        "error",
        "No microphone detected",
        "Plug in or enable a microphone. VoicePill will update when one appears.",
      ),
    );
  } else if (selectedMissing && settings.audio_device) {
    s.appendChild(
      notice(
        "warning",
        "Selected microphone unavailable",
        `"${settings.audio_device}" is not connected. Choose another input or use system default.`,
      ),
    );
  } else if (audioError && !audioError.sticky) {
    s.appendChild(notice("error", audioError.message, audioError.detail));
  }

  const dev = document.createElement("select");
  dev.className = "select";

  const def = document.createElement("option");
  def.value = "";
  def.textContent = "System default";
  if (!settings.audio_device) def.selected = true;
  dev.appendChild(def);

  if (selectedMissing && settings.audio_device) {
    const missing = document.createElement("option");
    missing.value = settings.audio_device;
    missing.textContent = `${settings.audio_device} (not found)`;
    missing.selected = true;
    dev.appendChild(missing);
  }

  for (const name of audioDevices) {
    const o = document.createElement("option");
    o.value = name;
    o.textContent = name;
    if (settings.audio_device === name) o.selected = true;
    dev.appendChild(o);
  }
  dev.onchange = async () => {
    settings.audio_device = dev.value === "" ? null : dev.value;
    await persist();
  };

  s.appendChild(
    row(
      "Input device",
      dev,
      audioDevices.length
        ? selectedMissing
          ? "Selected input is not connected"
          : undefined
        : "No input devices detected",
    ),
  );
  return s;
}

/** The appearance subset of `settings`, for live-preview events to the pill. */
function appearanceOf(s: Settings): Appearance {
  return {
    theme: s.theme,
    accent: s.accent,
    pill_scale: s.pill_scale,
    pill_opacity: s.pill_opacity,
    pill_compact: s.pill_compact,
  };
}

// Live preview: update this window's own theme/accent immediately, and push the
// full appearance to the pill window so size/opacity/compact preview in real
// time. Persisting (which fires the authoritative `settings:changed`) is done
// separately, only when the user commits a change.
function previewAppearance(): void {
  applyAppearance(settings, false);
  void emit("appearance:preview", appearanceOf(settings));
}

function appearanceSection(): HTMLElement {
  const s = section("Appearance");

  // Theme — light / dark / system
  const seg = document.createElement("div");
  seg.className = "seg";
  const themes: Array<[Theme, string]> = [
    ["light", "Light"],
    ["dark", "Dark"],
    ["system", "System"],
  ];
  for (const [value, label] of themes) {
    const opt = document.createElement("button");
    opt.className = "seg-opt";
    opt.textContent = label;
    if (settings.theme === value) opt.classList.add("active");
    opt.onclick = async () => {
      if (settings.theme === value) return;
      settings.theme = value;
      previewAppearance();
      await persist();
      render();
    };
    seg.appendChild(opt);
  }
  s.appendChild(row("Theme", seg, "Light, dark, or follow your system"));

  // Pill size
  s.appendChild(
    row(
      "Pill size",
      slider(settings.pill_scale, 0.8, 1.4, 0.05, (v) => `${Math.round(v * 100)}%`, (v) => {
        settings.pill_scale = v;
        previewAppearance();
      }),
    ),
  );

  // Pill opacity
  s.appendChild(
    row(
      "Pill opacity",
      slider(settings.pill_opacity, 0.4, 1.0, 0.05, (v) => `${Math.round(v * 100)}%`, (v) => {
        settings.pill_opacity = v;
        previewAppearance();
      }),
    ),
  );

  // Compact pill
  s.appendChild(
    row(
      "Compact pill",
      toggleSwitch(settings.pill_compact, async (v) => {
        settings.pill_compact = v;
        previewAppearance();
        await persist();
      }),
      "Hide the text label — show just the icon",
    ),
  );

  return s;
}

/**
 * A range slider with a live value readout. `onInput` fires while dragging;
 * the change is persisted once on release (`change`).
 */
function slider(
  value: number,
  min: number,
  max: number,
  step: number,
  fmt: (v: number) => string,
  onInput: (v: number) => void,
): HTMLElement {
  const wrap = document.createElement("div");
  wrap.className = "slider-wrap";
  const input = document.createElement("input");
  input.type = "range";
  input.className = "slider";
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = String(value);
  const val = document.createElement("span");
  val.className = "slider-val";
  val.textContent = fmt(value);
  input.oninput = () => {
    const v = parseFloat(input.value);
    val.textContent = fmt(v);
    onInput(v);
  };
  input.onchange = () => void persist();
  wrap.appendChild(input);
  wrap.appendChild(val);
  return wrap;
}

function generalSection(): HTMLElement {
  const s = section("General");

  s.appendChild(
    row(
      "Start on boot",
      toggleSwitch(settings.autostart, async (v) => {
        const previous = settings.autostart;
        settings.autostart = v;
        try {
          await persist();
        } catch (e) {
          settings.autostart = previous;
          toast(`Start on boot failed: ${e}`);
          render();
        }
      }),
    ),
  );

  s.appendChild(
    row(
      "Restore clipboard",
      toggleSwitch(settings.restore_clipboard, async (v) => {
        settings.restore_clipboard = v;
        await persist();
      }),
      "Put your previous clipboard back after pasting",
    ),
  );

  return s;
}

// ---- Updates -------------------------------------------------------------
// Build once (sync); the version label and check results fill in async, the
// same build-then-populate pattern the GPU section uses.
let checkingUpdate = false;

function updatesSection(): HTMLElement {
  const s = section("Updates");

  const verRow = document.createElement("div");
  verRow.className = "row";
  const verLabel = document.createElement("div");
  verLabel.className = "row-label";
  verLabel.appendChild(document.createTextNode("Current version"));
  const verHint = document.createElement("span");
  verHint.className = "row-hint";
  verHint.id = "app-version";
  verHint.textContent = "…";
  verLabel.appendChild(verHint);

  const btn = document.createElement("button");
  btn.className = "btn";
  btn.id = "update-check-btn";
  btn.textContent = checkingUpdate ? "Checking…" : "Check for updates";
  btn.disabled = checkingUpdate;
  btn.onclick = () => void runUpdateCheck();

  verRow.appendChild(verLabel);
  verRow.appendChild(btn);
  s.appendChild(verRow);

  // Status line below the row — holds "Up to date", an error, or the
  // available-update prompt with an Install button + progress.
  const status = document.createElement("div");
  status.className = "row-hint";
  status.id = "update-status";
  status.style.marginTop = "8px";
  s.appendChild(status);

  return s;
}

async function fillVersion() {
  const el = document.getElementById("app-version");
  if (!el) return;
  try {
    el.textContent = `v${await getVersion()}`;
  } catch {
    el.textContent = "";
  }
}

async function runUpdateCheck() {
  if (checkingUpdate) return;
  checkingUpdate = true;
  const btn = document.getElementById("update-check-btn") as HTMLButtonElement | null;
  const status = document.getElementById("update-status");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "Checking…";
  }
  if (status) status.textContent = "";

  try {
    const update = await checkForUpdate();
    if (!update) {
      if (status) status.textContent = "You're on the latest version.";
    } else if (status) {
      renderAvailableUpdate(status, update);
    }
  } catch (e) {
    if (status) status.textContent = `Update check failed: ${e}`;
  } finally {
    checkingUpdate = false;
    if (btn) {
      btn.disabled = false;
      btn.textContent = "Check for updates";
    }
  }
}

function renderAvailableUpdate(status: HTMLElement, update: Update) {
  status.innerHTML = "";
  const msg = document.createElement("span");
  msg.textContent = `Version ${update.version} is available. `;
  const install = document.createElement("button");
  install.className = "btn";
  install.textContent = "Install & restart";
  install.onclick = async () => {
    install.disabled = true;
    install.textContent = "Downloading… 0%";
    try {
      await installAndRelaunch(update, (p) => {
        install.textContent =
          p === null ? "Downloading…" : `Downloading… ${Math.round(p * 100)}%`;
      });
      // installAndRelaunch relaunches on success, so we don't get here.
    } catch (e) {
      install.disabled = false;
      install.textContent = "Install & restart";
      status.appendChild(document.createTextNode(` — failed: ${e}`));
    }
  };
  status.appendChild(msg);
  status.appendChild(install);
}

function toggleSwitch(on: boolean, onChange: (v: boolean) => void): HTMLElement {
  const label = document.createElement("label");
  label.className = "switch";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = on;
  input.onchange = () => onChange(input.checked);
  const track = document.createElement("span");
  track.className = "switch-track";
  label.appendChild(input);
  label.appendChild(track);
  return label;
}
