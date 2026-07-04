// Lightweight toast notifications. A single fixed host at the bottom of the
// window stacks transient messages; each fades in, lingers, then fades out.

let host: HTMLElement | null = null;

export type ToastKind = "info" | "ok" | "err";

function ensureHost(): HTMLElement {
  if (host) return host;
  host = document.createElement("div");
  host.className = "toast-host";
  document.body.appendChild(host);
  return host;
}

const ICONS: Record<Exclude<ToastKind, "info">, string> = {
  ok: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 12l6 6L20 6"/></svg>',
  err: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>',
};

/** Show a transient toast. `ms` is the on-screen time before it fades out.
 *  `kind` picks the leading glyph: neutral dot, success check, or error cross. */
export function toast(message: string, ms = 1600, kind: ToastKind = "info"): void {
  const h = ensureHost();

  const el = document.createElement("div");
  el.className = kind === "info" ? "toast" : `toast ${kind}`;
  if (kind === "info") {
    const dot = document.createElement("span");
    dot.className = "dot";
    el.appendChild(dot);
  } else {
    el.innerHTML = ICONS[kind];
  }
  el.appendChild(document.createTextNode(message));
  h.appendChild(el);

  // Next frame: trigger the fade-in transition.
  requestAnimationFrame(() => el.classList.add("show"));

  window.setTimeout(() => {
    el.classList.remove("show");
    window.setTimeout(() => el.remove(), 220); // after fade-out transition
  }, ms);
}
