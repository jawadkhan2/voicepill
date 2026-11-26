//! Build guard: refuses to compile anything in this repo unless the user
//! explicitly allows it for that one command (VOICEPILL_ALLOW_BUILD=1).
//! Rebuilding whisper.cpp/CUDA additionally needs VOICEPILL_ALLOW_CUDA_BUILD=1.
//!
//! Why: compiling this project builds whisper.cpp with nvcc for several GPU
//! architectures — many minutes at 100% CPU on every core. Background tools
//! (rust-analyzer started by an editor or an MCP server, agents running
//! `cargo check`) kicked that off on their own and overheated the machine.
//!
//! The same binary is installed under two names (see .cargo/config.toml):
//! - `build-guard.exe` is cargo's `build.rustc-wrapper`: blocks every rustc
//!   call (build, check, test, clippy). rust-analyzer can override this one
//!   via the RUSTC_WRAPPER env var, hence the second layer:
//! - `cmake-guard.exe` is set as `CMAKE` in cargo's `[env]`, which the cmake
//!   crate uses instead of `cmake`. whisper.cpp/CUDA only ever builds through
//!   CMake, so the expensive part stays blocked no matter who runs cargo.
//!
//! Build (once, no cargo needed), from the repo root:
//!   rustc -O tools/build-guard/main.rs -o tools/build-guard/build-guard.exe
//!   copy tools/build-guard/build-guard.exe tools/build-guard/cmake-guard.exe
//! If an exe is missing, cargo can't run it and the build fails — still blocked.

use std::ffi::OsString;
use std::process::{exit, Command};

fn main() {
    let mut args: Vec<OsString> = std::env::args_os().collect();
    let argv0 = args.remove(0);
    let as_cmake = std::path::Path::new(&argv0)
        .file_stem()
        .map(|s| s.to_string_lossy().to_ascii_lowercase().contains("cmake"))
        .unwrap_or(false);

    // CMake = whisper.cpp's CUDA build, the part that pins the CPU for minutes.
    // It needs its own opt-in on top of VOICEPILL_ALLOW_BUILD, so a normal
    // build can reuse the cached whisper.cpp but never rebuild it.
    let allow = |var: &str| std::env::var(var).as_deref() == Ok("1");
    let allowed = allow("VOICEPILL_ALLOW_BUILD") && (!as_cmake || allow("VOICEPILL_ALLOW_CUDA_BUILD"));
    if !allowed {
        let what = if as_cmake { "CMake (whisper.cpp/CUDA)" } else { "rustc" };
        eprintln!();
        eprintln!("==============================================================");
        eprintln!(" VoicePill: compilation is DISABLED (tools/build-guard).");
        eprintln!(" Blocked: {what}.");
        eprintln!(" Building this project compiles whisper.cpp with CUDA and");
        eprintln!(" pins the CPU for minutes, so nothing may build it");
        eprintln!(" automatically. To build on purpose, set it for one command:");
        eprintln!("   PowerShell: $env:VOICEPILL_ALLOW_BUILD=1; npm run tauri build");
        eprintln!("   then: Remove-Item Env:VOICEPILL_ALLOW_BUILD");
        eprintln!(" Rebuilding whisper.cpp/CUDA additionally needs");
        eprintln!("   VOICEPILL_ALLOW_CUDA_BUILD=1 (minutes of 100% CPU).");
        eprintln!("==============================================================");
        exit(1);
    }

    // Allowed: run the real tool. As rustc-wrapper, cargo passes the rustc
    // path first; as CMAKE, forward everything to `cmake` on PATH.
    let (program, rest) = if as_cmake {
        (OsString::from("cmake"), args)
    } else {
        if args.is_empty() {
            eprintln!("build-guard: expected the rustc path as the first argument");
            exit(2);
        }
        let rustc = args.remove(0);
        (rustc, args)
    };

    match Command::new(&program).args(rest).status() {
        Ok(status) => exit(status.code().unwrap_or(1)),
        Err(e) => {
            eprintln!("build-guard: failed to run {}: {e}", program.to_string_lossy());
            exit(1);
        }
    }
}
