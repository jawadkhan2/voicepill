#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script must be run on macOS."
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${VOICEPILL_MAC_TARGET:-aarch64-apple-darwin}"
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}"

cd "$ROOT"

if ! rustup target list --installed | grep -qx "$TARGET"; then
  rustup target add "$TARGET"
fi

(
  cd src-tauri
  cargo clean -p whisper-rs-sys --target "$TARGET"
)

npm run tauri -- build \
  --target "$TARGET" \
  --bundles app,dmg \
  --config '{"bundle":{"createUpdaterArtifacts":false}}' \
  --ci

APP="src-tauri/target/$TARGET/release/bundle/macos/VoicePill.app"
if [[ -d "$APP" ]]; then
  codesign --verify --deep --strict --verbose=2 "$APP"
fi
