// Shared appearance logic used by both the pill and settings windows.
//
// Theme + accent apply to any window; pill size/opacity/compact only affect the
// floating pill (and resize its borderless window to fit the new dimensions).

import { getCurrentWindow } from "@tauri-apps/api/window";
import { LogicalSize } from "@tauri-apps/api/dpi";

export type Theme = "light" | "dark" | "system";

/** The appearance subset of the Rust `Settings` struct. */
export interface Appearance {
  theme: Theme;
  accent: string;
  pill_scale: number;
  pill_opacity: number;
  pill_compact: boolean;
}

const root = document.documentElement;

/** Resolve `system` to a concrete light/dark using the OS preference. */
function resolveTheme(theme: Theme): "light" | "dark" {
  if (theme === "system") {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  return theme;
}

/** Mix a #rrggbb color toward white by `amt` (0–1) to derive a lighter shade. */
function lighten(hex: string, amt: number): string {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return hex;
  const n = parseInt(m[1], 16);
  const r = (n >> 16) & 0xff;
  const g = (n >> 8) & 0xff;
  const b = n & 0xff;
  const mix = (c: number) => Math.round(c + (255 - c) * amt);
  const to2 = (c: number) => c.toString(16).padStart(2, "0");
  return `#${to2(mix(r))}${to2(mix(g))}${to2(mix(b))}`;
}

/**
 * Apply appearance settings to the current window. Pass `isPill` so only the
 * pill window touches pill-specific styling and resizes itself.
 */
export function applyAppearance(a: Appearance, isPill: boolean): void {
  // Theme + accent affect every window.
  root.dataset.theme = resolveTheme(a.theme);
  root.style.setProperty("--vp-accent", a.accent);
  root.style.setProperty("--vp-accent-2", lighten(a.accent, 0.4));
  // Listening waveform/indicator follows the accent instead of a fixed green.
  root.style.setProperty("--vp-listening", a.accent);

  if (!isPill) return;

  const pill = document.getElementById("pill");
  if (!pill) return;
  pill.classList.toggle("compact", a.pill_compact);
  root.style.setProperty("--pill-opacity", String(a.pill_opacity));
  resizePillTo(pill, a.pill_scale);
}

/** Size the borderless pill window to fit the pill at the given scale. */
function resizePillTo(pill: HTMLElement, scale: number): void {
  // Padding around the pill so its drop shadow isn't clipped by the window edge.
  const SHADOW_PAD = 28;
  // Measure the unscaled layout box, then scale + size the window analytically
  // (transform doesn't change layout, so the CSS scale and our math agree).
  root.style.setProperty("--pill-scale", "1");
  const base = pill.getBoundingClientRect();
  root.style.setProperty("--pill-scale", String(scale));
  const w = Math.ceil(base.width * scale) + SHADOW_PAD;
  const h = Math.ceil(base.height * scale) + SHADOW_PAD;
  void getCurrentWindow().setSize(new LogicalSize(w, h));
}

/**
 * Re-apply appearance whenever the OS light/dark preference changes, but only
 * while the theme is set to `system`. Returns nothing; lives for the page.
 */
export function watchSystemTheme(getAppearance: () => Appearance, isPill: boolean): void {
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    const a = getAppearance();
    if (a.theme === "system") applyAppearance(a, isPill);
  });
}
