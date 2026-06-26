//! Live GPU / VRAM stats.
//!
//! On Windows this uses NVIDIA NVML.
//! `nvml.dll` ships with the NVIDIA driver, so `Nvml::init()` loads it at runtime
//! — no CUDA toolkit needed just to *read* stats. We use this purely for UI
//! feedback: confirm the CUDA device exists and show how much VRAM is in use
//! (system-wide and, where the driver exposes it, by our own process). Seeing
//! VRAM jump when a model loads is the clearest proof the GPU is actually doing
//! the work.

use serde::Serialize;

/// NVML handle, initialized once. `None` if the lib/driver is unavailable (e.g.
/// no NVIDIA GPU) — the UI just hides the panel in that case.
#[cfg(windows)]
use std::sync::OnceLock;

#[cfg(windows)]
use nvml_wrapper::enums::device::UsedGpuMemory;
#[cfg(windows)]
use nvml_wrapper::Nvml;

#[cfg(windows)]
static NVML: OnceLock<Option<Nvml>> = OnceLock::new();

#[cfg(windows)]
fn nvml() -> Option<&'static Nvml> {
    NVML.get_or_init(|| match Nvml::init() {
        Ok(n) => Some(n),
        Err(e) => {
            eprintln!("[gpu] NVML init failed (no GPU stats): {e}");
            None
        }
    })
    .as_ref()
}

/// GPU snapshot for the settings UI. All memory figures are in mebibytes.
#[derive(Serialize, Clone, Default)]
pub struct GpuInfo {
    /// False when no NVIDIA GPU / NVML is present — UI hides the panel.
    pub available: bool,
    pub name: String,
    pub driver: String,
    pub vram_total_mb: u64,
    pub vram_used_mb: u64,
    /// VRAM attributed to *this* process; 0 if the driver doesn't expose it
    /// (common under Windows WDDM) or nothing is loaded yet.
    pub vram_process_mb: u64,
}

/// Read current GPU stats for device 0. Cheap enough to poll once a second.
#[cfg(windows)]
pub fn info() -> GpuInfo {
    let Some(nvml) = nvml() else {
        return GpuInfo::default();
    };
    let Ok(device) = nvml.device_by_index(0) else {
        return GpuInfo::default();
    };

    let name = device.name().unwrap_or_default();
    let driver = nvml.sys_driver_version().unwrap_or_default();
    let (total, used) = device
        .memory_info()
        .map(|m| (m.total, m.used))
        .unwrap_or((0, 0));

    // Per-process usage: sum the compute-process entries matching our PID.
    let pid = std::process::id();
    let process_bytes = device
        .running_compute_processes()
        .map(|procs| {
            procs
                .iter()
                .filter(|p| p.pid == pid)
                .map(|p| match p.used_gpu_memory {
                    UsedGpuMemory::Used(bytes) => bytes,
                    UsedGpuMemory::Unavailable => 0,
                })
                .sum::<u64>()
        })
        .unwrap_or(0);

    GpuInfo {
        available: true,
        name,
        driver,
        vram_total_mb: total / 1_048_576,
        vram_used_mb: used / 1_048_576,
        vram_process_mb: process_bytes / 1_048_576,
    }
}

/// Apple Silicon uses Metal/unified memory through whisper.cpp, but this UI
/// panel is currently NVIDIA/NVML-specific. Hide it on macOS for now.
#[cfg(not(windows))]
pub fn info() -> GpuInfo {
    GpuInfo::default()
}
