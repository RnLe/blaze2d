//! WASM-oriented backend shims.

use mpb2d-core::backend::{SpectralBackend, SpectralBuffer};
use mpb2d-core::grid::Grid2D;
use num_complex::Complex64;

#[cfg(feature = "bindings")]
use wasm_bindgen::prelude::*;

#[cfg_attr(feature = "bindings", wasm_bindgen)]
pub struct WasmBackend;

pub struct WasmField {
    grid: Grid2D,
    data: Vec<Complex64>,
}

impl SpectralBuffer for WasmField {
    fn len(&self) -> usize {
        self.data.len()
    }
}

impl SpectralBackend for WasmBackend {
    type Buffer = WasmField;

    fn alloc_field(&self, grid: Grid2D) -> Self::Buffer {
        WasmField {
            grid,
            data: vec![Complex64::default(); grid.len()],
        }
    }

    fn forward_fft_2d(&self, _buffer: &mut Self::Buffer) {}

    fn inverse_fft_2d(&self, _buffer: &mut Self::Buffer) {}

    fn scale(&self, _alpha: Complex64, _buffer: &mut Self::Buffer) {}
}
