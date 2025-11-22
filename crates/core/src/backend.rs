//! Backend traits for spectral operations.

use num_complex::Complex64;

use crate::grid::Grid2D;

pub trait SpectralBuffer {
    fn len(&self) -> usize;
}

pub trait SpectralBackend {
    type Buffer: SpectralBuffer;

    fn alloc_field(&self, grid: Grid2D) -> Self::Buffer;
    fn forward_fft_2d(&self, buffer: &mut Self::Buffer);
    fn inverse_fft_2d(&self, buffer: &mut Self::Buffer);
    fn scale(&self, alpha: Complex64, buffer: &mut Self::Buffer);
}
