//! Backend traits for spectral operations.

use num_complex::Complex64;

use crate::{field::Field2D, grid::Grid2D};

pub trait SpectralBuffer {
    fn len(&self) -> usize;
    fn grid(&self) -> Grid2D;
    fn as_slice(&self) -> &[Complex64];
    fn as_mut_slice(&mut self) -> &mut [Complex64];
}

impl SpectralBuffer for Field2D {
    fn len(&self) -> usize {
        self.len()
    }

    fn grid(&self) -> Grid2D {
        self.grid()
    }

    fn as_slice(&self) -> &[Complex64] {
        self.as_slice()
    }

    fn as_mut_slice(&mut self) -> &mut [Complex64] {
        self.as_mut_slice()
    }
}

pub trait SpectralBackend {
    type Buffer: SpectralBuffer + Clone;

    fn alloc_field(&self, grid: Grid2D) -> Self::Buffer;
    fn forward_fft_2d(&self, buffer: &mut Self::Buffer);
    fn inverse_fft_2d(&self, buffer: &mut Self::Buffer);
    fn scale(&self, alpha: Complex64, buffer: &mut Self::Buffer);
    fn axpy(&self, alpha: Complex64, x: &Self::Buffer, y: &mut Self::Buffer);
    fn dot(&self, x: &Self::Buffer, y: &Self::Buffer) -> Complex64;
}
