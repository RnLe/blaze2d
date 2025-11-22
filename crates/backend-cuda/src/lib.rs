//! CUDA backend placeholder using cudarc when enabled.

use mpb2d_core::backend::{SpectralBackend, SpectralBuffer};
use mpb2d_core::grid::Grid2D;
use num_complex::Complex64;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;

pub struct CudaBackend {
    #[cfg(feature = "cuda")]
    device: CudaDevice,
}

impl CudaBackend {
    pub fn new() -> Self {
        #[cfg(feature = "cuda")]
        {
            let device = CudaDevice::new(0).expect("CUDA device 0");
            return Self { device };
        }
        #[cfg(not(feature = "cuda"))]
        {
            Self {}
        }
    }
}

#[derive(Clone)]
pub struct CudaField {
    grid: Grid2D,
    data: Vec<Complex64>,
}

impl SpectralBuffer for CudaField {
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

impl SpectralBackend for CudaBackend {
    type Buffer = CudaField;

    fn alloc_field(&self, grid: Grid2D) -> Self::Buffer {
        CudaField {
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
