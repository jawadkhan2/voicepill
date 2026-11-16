//! Shared ONNX Runtime environment + CUDA runtime discovery.
//!
//! One `ort` environment serves every ONNX engine in the process (Parakeet on
//! CUDA, MedASR on CPU). ONNX Runtime's core is linked statically; its CUDA
//! execution provider is a DLL that ships next to the exe
//! (`onnxruntime_providers_cuda.dll` + `onnxruntime_providers_shared.dll`).
//! That provider imports cuDNN 9, which is too big to put in every app update,
//! so it is downloaded once alongside the Parakeet model into
//! `<models>/cuda-runtime/` and that folder is put on the DLL search path before
//! the first CUDA session is created. cuBLAS / cudart come from the CUDA 13
//! toolkit on PATH — the same place whisper.cpp's CUDA build already loads them.

use std::path::Path;
use std::sync::OnceLock;

use ort::environment::Environment;

static ENV: OnceLock<Result<Environment, String>> = OnceLock::new();

/// The process-wide ONNX Runtime environment.
pub fn env() -> Result<&'static Environment, String> {
    ENV.get_or_init(|| {
        ort::init()
            .with_name("voicepill")
            .build()
            .map_err(|e| format!("onnxruntime init failed: {e}"))
    })
    .as_ref()
    .map_err(|e| e.clone())
}

/// Folder (inside the models dir) holding the downloaded cuDNN DLLs.
pub const CUDA_RUNTIME_DIR: &str = "cuda-runtime";

/// Make `dir` (the downloaded cuDNN DLLs) loadable by the CUDA provider.
///
/// cuDNN 9 is split into a small front DLL that `LoadLibrary`s its sub-libraries
/// (`cudnn_engines_*`, `cudnn_cnn64_9`, …) by name at run time, so registering
/// one directory with `AddDllDirectory` isn't enough — those nested loads use
/// the default search order. Prepending the folder to this process's `PATH`
/// covers both. Idempotent.
pub fn add_cuda_runtime_to_path(dir: &Path) {
    let Some(dir_str) = dir.to_str() else { return };
    let current = std::env::var_os("PATH").unwrap_or_default();
    let already = std::env::split_paths(&current).any(|p| p == dir);
    if already {
        return;
    }
    let mut paths = vec![dir.to_path_buf()];
    paths.extend(std::env::split_paths(&current));
    if let Ok(joined) = std::env::join_paths(paths) {
        // Set once from the transcription thread, before any CUDA session
        // exists and before cuDNN's lazy sub-library loads.
        std::env::set_var("PATH", joined);
        crate::vlog!("[ort] cuda runtime dir on PATH: {dir_str}");
    }
}
