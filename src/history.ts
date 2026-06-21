// The transcription-history view. Renders the last few transcripts into
// #history-root and keeps itself current via window events. Window lifecycle
// and appearance are owned by main.ts; this module only renders the list.

import { invoke } from "@tauri-apps/api/core";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { toast } from "./toast";

interface TranscriptEntry {
  id: string;
  text: string;
  created_at_ms: number;
}

const root = document.getElementById("history-root")!;
const win = getCurrentWindow();

let transcripts: TranscriptEntry[] = [];
let copiedId: string | null = null;
let copiedReset: number | undefined;

/** Mount the history view; returns a `refresh` the host can call (e.g. on focus). */
export function initHistory(): { refresh: (force?: boolean) => Promise<void> } {
  void win.listen<TranscriptEntry>("transcript:new", (event) => {
    transcripts = [event.payload, ...transcripts.filter((t) => t.id !== event.payload.id)].slice(0, 10);
    render();
  });

  void win.listen("history:refresh", () => {
    void refreshTranscripts();
  });

  renderSkeleton();
  void refreshTranscripts(true);

  return { refresh: refreshTranscripts };
}

/** Shimmering placeholders shown until the first fetch resolves. */
function renderSkeleton() {
  root.innerHTML = "";
  for (let i = 0; i < 3; i++) {
    const card = document.createElement("div");
    card.className = "skel";
    card.innerHTML = '<div class="skel-line"></div><div class="skel-line"></div><div class="skel-line"></div>';
    root.appendChild(card);
  }
}

async function refreshTranscripts(force = false) {
  const next = await invoke<TranscriptEntry[]>("get_transcripts");
  if (force || !sameTranscripts(transcripts, next)) {
    transcripts = next;
    render();
  }
}

function sameTranscripts(a: TranscriptEntry[], b: TranscriptEntry[]): boolean {
  if (a.length !== b.length) return false;
  return a.every((entry, i) => {
    const other = b[i];
    return (
      other !== undefined &&
      entry.id === other.id &&
      entry.text === other.text &&
      entry.created_at_ms === other.created_at_ms
    );
  });
}

function render() {
  root.innerHTML = "";

  if (transcripts.length === 0) {
    const empty = document.createElement("section");
    empty.className = "history-empty";
    empty.innerHTML = `
      <div class="empty-ring"><span class="mic-dot"></span></div>
      <h4>No transcriptions yet</h4>
      <p>Hold your trigger key and speak — your text lands here.</p>`;
    root.appendChild(empty);
    return;
  }

  // Insert a day separator whenever the calendar day changes down the list.
  let lastDay = "";
  for (const entry of transcripts) {
    const day = dayKey(entry.created_at_ms);
    if (day !== lastDay) {
      lastDay = day;
      const sep = document.createElement("div");
      sep.className = "day-sep";
      sep.textContent = dayLabel(entry.created_at_ms);
      root.appendChild(sep);
    }
    root.appendChild(transcriptRow(entry));
  }
}

function transcriptRow(entry: TranscriptEntry): HTMLElement {
  const row = document.createElement("article");
  row.className = "history-item";

  const text = document.createElement("p");
  text.className = "history-text";
  text.textContent = entry.text;
  row.appendChild(text);

  const actions = document.createElement("div");
  actions.className = "history-actions";

  const time = document.createElement("span");
  time.className = "history-time";
  time.textContent = relativeTime(entry.created_at_ms);
  actions.appendChild(time);

  const copy = document.createElement("button");
  copy.className = "btn btn-copy";
  copy.type = "button";
  if (copiedId === entry.id) {
    copy.classList.add("copied");
    copy.innerHTML = '<svg class="check-svg" viewBox="0 0 24 24"><path d="M5 13l4 4L19 7"/></svg>';
  } else {
    copy.textContent = "Copy";
  }
  copy.onclick = async () => {
    await invoke("copy_transcript", { text: entry.text });
    copiedId = entry.id;
    toast("Copied to clipboard");
    window.clearTimeout(copiedReset);
    copiedReset = window.setTimeout(() => {
      copiedId = null;
      render();
    }, 1400);
    render();
  };

  const paste = document.createElement("button");
  paste.className = "btn btn-ghost";
  paste.type = "button";
  paste.textContent = "Paste";
  paste.onclick = async () => {
    await invoke("paste_transcript", { text: entry.text });
    toast("Pasted");
  };

  const buttons = document.createElement("div");
  buttons.className = "history-buttons";
  buttons.appendChild(copy);
  buttons.appendChild(paste);
  actions.appendChild(buttons);

  row.appendChild(actions);
  return row;
}

function clockTime(ms: number): string {
  return new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(ms));
}

/** "Just now" / "4m ago" / "3:42 PM" / "Yesterday, 3:42 PM" / "Jun 18". */
function relativeTime(ms: number): string {
  const diff = Date.now() - ms;
  if (diff < 45_000) return "Just now";
  if (diff < 3_600_000) return `${Math.round(diff / 60_000)}m ago`;

  const key = dayKey(ms);
  if (key === dayKey(Date.now())) return clockTime(ms);
  if (key === dayKey(Date.now() - 86_400_000)) return `Yesterday, ${clockTime(ms)}`;
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(new Date(ms));
}

/** Stable per-calendar-day key for grouping. */
function dayKey(ms: number): string {
  return new Date(ms).toDateString();
}

/** Section heading for a day group: Today / Yesterday / "Jun 18". */
function dayLabel(ms: number): string {
  const key = dayKey(ms);
  if (key === dayKey(Date.now())) return "Today";
  if (key === dayKey(Date.now() - 86_400_000)) return "Yesterday";
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", year: "numeric" }).format(new Date(ms));
}
