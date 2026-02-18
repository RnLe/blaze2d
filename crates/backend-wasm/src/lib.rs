//! WASM-oriented backend shims.

use mpb2d_core::backend::{SpectralBackend, SpectralBuffer};
use mpb2d_core::grid::Grid2D;
use num_complex::Complex64;

#[cfg(feature = "bindings")]
use wasm_bindgen::prelude::*;

#[derive(Clone)]
#[cfg_attr(feature = "bindings", wasm_bindgen)]
pub struct WasmBackend;

#[derive(Clone)]
pub struct WasmField {
    grid: Grid2D,
    data: Vec<Complex64>,
}

impl SpectralBuffer for WasmField {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn grid(&self) -> Grid2D {
        self.grid
    }

    fn as_slice(&self) -> &[Complex64] {
        &self.data
    }

    fn as_mut_slice(&mut self) -> &mut [Complex64] {
        &mut self.data
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

    fn scale(&self, alpha: Complex64, buffer: &mut Self::Buffer) {
        for value in &mut buffer.data {
            *value *= alpha;
        }
    }

    fn axpy(&self, alpha: Complex64, x: &Self::Buffer, y: &mut Self::Buffer) {
        for (dst, src) in y.data.iter_mut().zip(&x.data) {
            *dst += alpha * src;
        }
    }

    fn dot(&self, x: &Self::Buffer, y: &Self::Buffer) -> Complex64 {
        x.data.iter().zip(&y.data).map(|(a, b)| a.conj() * b).sum()
    }
}
