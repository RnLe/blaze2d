//! Browser adapter for the shared scientific contract, using the f64 FFT backend.
pub type WasmBackend = blaze2d_backend_cpu::CpuBackend<f64>;

#[cfg(feature="bindings")]
mod bindings;
