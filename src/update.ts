// Over-the-air updates. Thin wrapper over the Tauri updater + process plugins:
// `check()` hits the signed `latest.json` on GitHub Releases, and an available
// `Update` can stream-download, install (signature-verified by the backend),
// and relaunch into the new version.

import { check, type Update, type DownloadEvent } from "@tauri-apps/plugin-updater";
import { relaunch } from "@tauri-apps/plugin-process";

export type { Update };

/**
 * Ask the endpoint whether a newer version exists.
 * Returns the `Update` handle if one is available, otherwise `null`.
 * Throws on network/parse errors so callers can decide whether to surface them.
 */
export async function checkForUpdate(): Promise<Update | null> {
  const update = await check();
  // `check()` resolves to null when up to date; some versions return an object
  // with `available === false`. Normalise both to null.
  if (!update || !update.available) return null;
  return update;
}

/** Download progress as a 0..1 fraction, or null while the total size is unknown. */
export type Progress = number | null;

/**
 * Download + install an update, reporting progress, then relaunch the app.
 * Does not return on success (the process is replaced by the relaunch).
 */
export async function installAndRelaunch(
  update: Update,
  onProgress?: (p: Progress) => void,
): Promise<void> {
  let downloaded = 0;
  let total = 0;

  await update.downloadAndInstall((event: DownloadEvent) => {
    switch (event.event) {
      case "Started":
        total = event.data.contentLength ?? 0;
        onProgress?.(total > 0 ? 0 : null);
        break;
      case "Progress":
        downloaded += event.data.chunkLength;
        onProgress?.(total > 0 ? Math.min(downloaded / total, 1) : null);
        break;
      case "Finished":
        onProgress?.(1);
        break;
    }
  });

  await relaunch();
}
