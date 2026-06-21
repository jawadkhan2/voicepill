// Lightweight toast notifications. A single fixed host at the bottom of the
// window stacks transient messages; each fades in, lingers, then fades out.

let host: HTMLElement | null = null;

function ensureHost(): HTMLElement {
  if (host) return host;
  host = document.createElement("div");
  host.className = "toast-host";
  document.body.appendChild(host);
  return host;
}

/** Show a transient toast. `ms` is the on-screen time before it fades out. */
export function toast(message: string, ms = 1600): void {
  const h = ensureHost();

  const el = document.createElement("div");
  el.className = "toast";
  const dot = document.createElement("span");
  dot.className = "dot";
  el.appendChild(dot);
  el.appendChild(document.createTextNode(message));
  h.appendChild(el);

  // Next frame: trigger the fade-in transition.
  requestAnimationFrame(() => el.classList.add("show"));

  window.setTimeout(() => {
    el.classList.remove("show");
    window.setTimeout(() => el.remove(), 220); // after fade-out transition
  }, ms);
}
