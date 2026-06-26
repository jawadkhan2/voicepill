<#
.SYNOPSIS
  Build a signed VoicePill installer locally and publish it as a GitHub Release
  so the in-app updater can pick it up.

.DESCRIPTION
  whisper-rs's CUDA backend can't be built on GitHub-hosted CI (needs the local
  CUDA 13.3 + RTX 5080 toolchain), so releases are cut from this machine. This
  script optionally bumps the version across the three manifests, builds the NSIS
  installer with the updater signature, assembles latest.json, and uploads both
  as assets on a GitHub release tagged v<version>.

  The in-app endpoint resolves
    https://github.com/jawadkhan2/voicepill/releases/latest/download/latest.json
  so whichever release is "latest" on GitHub is what clients update to.

.PARAMETER Version
  New semver (e.g. 0.1.1). If omitted, the current version in tauri.conf.json is
  reused (rebuild/republish the same version).

.PARAMETER Notes
  Release notes shown in the GitHub release and returned by the updater.

.PARAMETER Draft
  Create the GitHub release as a draft (won't become "latest" until published).

.PREREQUISITES
  - Updater key password: env var VOICEPILL_SIGN_PW, OR a file at
    $HOME/.tauri/voicepill_updater.pw (env var wins if both present).
  - Updater private key at $HOME/.tauri/voicepill_updater.key.
  - gh CLI authenticated; CUDA build env wired via src-tauri/.cargo/config.toml.
#>
[CmdletBinding()]
param(
  [string]$Version,
  [string]$Notes = "",
  [switch]$Draft
)

$ErrorActionPreference = "Stop"
$repo = "jawadkhan2/voicepill"

# Resolve paths relative to this script (scripts/ lives at the project root).
$root      = Split-Path -Parent $PSScriptRoot
$confPath  = Join-Path $root "src-tauri/tauri.conf.json"
$cargoPath = Join-Path $root "src-tauri/Cargo.toml"
$pkgPath   = Join-Path $root "package.json"
$lockPath  = Join-Path $root "package-lock.json"
$keyPath   = Join-Path $HOME ".tauri/voicepill_updater.key"
$pwPath    = Join-Path $HOME ".tauri/voicepill_updater.pw"

# ---- Preconditions --------------------------------------------------------
# Password: env var wins; else fall back to the on-disk .pw file.
$signPw = $env:VOICEPILL_SIGN_PW
if (-not $signPw -and (Test-Path $pwPath)) {
  $signPw = (Get-Content $pwPath -Raw).Trim()
}
if (-not $signPw) {
  throw "No key password: set `$env:VOICEPILL_SIGN_PW or create $pwPath"
}
if (-not (Test-Path $keyPath)) {
  throw "Updater private key not found at $keyPath"
}

# ---- Version bump (optional) ----------------------------------------------
$conf = Get-Content $confPath -Raw | ConvertFrom-Json
if (-not $Version) { $Version = $conf.version }
Write-Host "Publishing VoicePill v$Version" -ForegroundColor Cyan

if ($conf.version -ne $Version) {
  Write-Host "Bumping version $($conf.version) -> $Version" -ForegroundColor Yellow

  # tauri.conf.json — rewrite just the top-level "version" field.
  (Get-Content $confPath -Raw) `
    -replace '("version"\s*:\s*")[^"]+(")', "`${1}$Version`${2}" |
    Set-Content $confPath -NoNewline
  Add-Content $confPath ""  # keep trailing newline

  # Cargo.toml — the package version is the first `version = "..."` line.
  $cargo = Get-Content $cargoPath
  $done = $false
  $cargo = $cargo | ForEach-Object {
    if (-not $done -and $_ -match '^\s*version\s*=\s*".*"') {
      $done = $true
      "version = `"$Version`""
    } else { $_ }
  }
  Set-Content $cargoPath $cargo

  # package.json
  (Get-Content $pkgPath -Raw) `
    -replace '("version"\s*:\s*")[^"]+(")', "`${1}$Version`${2}" |
    Set-Content $pkgPath -NoNewline
  Add-Content $pkgPath ""  # keep trailing newline

  if (Test-Path $lockPath) {
    $lock = Get-Content $lockPath -Raw | ConvertFrom-Json
    $lock.version = $Version
    if ($lock.packages -and $lock.packages.PSObject.Properties.Name -contains "") {
      $lock.packages.PSObject.Properties[""].Value.version = $Version
    }
    ($lock | ConvertTo-Json -Depth 100) | Set-Content $lockPath
  }
}

# ---- Build (signed) -------------------------------------------------------
$env:TAURI_SIGNING_PRIVATE_KEY = $keyPath
$env:TAURI_SIGNING_PRIVATE_KEY_PASSWORD = $signPw

Write-Host "Building signed installer..." -ForegroundColor Cyan
Push-Location $root
try {
  Write-Host "Refreshing native Whisper/CUDA build cache..." -ForegroundColor Cyan
  Push-Location (Join-Path $root "src-tauri")
  try {
    cargo clean -p whisper-rs-sys
    if ($LASTEXITCODE -ne 0) { throw "cargo clean failed (exit $LASTEXITCODE)" }
  } finally {
    Pop-Location
  }

  npm run tauri build
  if ($LASTEXITCODE -ne 0) { throw "tauri build failed (exit $LASTEXITCODE)" }
} finally {
  Pop-Location
}

# ---- Locate artifacts -----------------------------------------------------
$nsisDir   = Join-Path $root "src-tauri/target/release/bundle/nsis"
$installer = Get-ChildItem $nsisDir -Filter "*_${Version}_x64-setup.exe" | Select-Object -First 1
if (-not $installer) { throw "Installer not found in $nsisDir for version $Version" }
$sigFile = "$($installer.FullName).sig"
if (-not (Test-Path $sigFile)) { throw "Signature not found: $sigFile (is createUpdaterArtifacts enabled?)" }

$signature = (Get-Content $sigFile -Raw).Trim()
$assetUrl  = "https://github.com/$repo/releases/download/v$Version/$($installer.Name)"

# ---- latest.json ----------------------------------------------------------
$manifest = [ordered]@{
  version   = $Version
  notes     = $Notes
  pub_date  = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
  platforms = [ordered]@{
    "windows-x86_64" = [ordered]@{
      signature = $signature
      url       = $assetUrl
    }
  }
}
$manifestPath = Join-Path $nsisDir "latest.json"
($manifest | ConvertTo-Json -Depth 8) | Set-Content $manifestPath -Encoding utf8
Write-Host "Wrote $manifestPath" -ForegroundColor Green

# ---- Publish to GitHub ----------------------------------------------------
$tag = "v$Version"
$draftFlag = if ($Draft) { @("--draft") } else { @() }

# If the release/tag already exists, upload (clobber) assets; else create it.
$releaseExists = $false
try {
  gh release view $tag --repo $repo *> $null
  $releaseExists = ($LASTEXITCODE -eq 0)
} catch {
  $releaseExists = $false
}

if ($releaseExists) {
  Write-Host "Release $tag exists - uploading assets (clobber)..." -ForegroundColor Yellow
  gh release upload $tag $installer.FullName $manifestPath --repo $repo --clobber
} else {
  Write-Host "Creating release $tag..." -ForegroundColor Cyan
  gh release create $tag $installer.FullName $manifestPath `
    --repo $repo --title "VoicePill $tag" --notes $Notes @draftFlag
}
if ($LASTEXITCODE -ne 0) { throw "gh release step failed (exit $LASTEXITCODE)" }

Write-Host "`nPublished VoicePill $tag" -ForegroundColor Green
Write-Host "Updater endpoint: https://github.com/$repo/releases/latest/download/latest.json"
