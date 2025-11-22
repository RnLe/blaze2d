//! CUDA backend placeholder using cudarc when enabled.

use mpb2d-core::backend::{SpectralBackend, SpectralBuffer};
use mpb2d-core::grid::Grid2D;
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

pub struct CudaField {
    grid: Grid2D,
    data: Vec<Complex64>,
}

impl SpectralBuffer for CudaField {
    fn len(&self) -> usize {
        self.data.len()
    }
}

impl SpectralBackend for CudaBackend {
    type Buffer = CudaField;

    fn alloc_field(&self, grid: Grid2D) -> Self::Buffer {
        let _ = grid;
        CudaField {
            grid,
            data: Vec::new(),
        }
    }

    fn forward_fft_2d(&self, _buffer: &mut Self::Buffer) {}

    fn inverse_fft_2d(&self, _buffer: &mut Self::Buffer) {}

    fn scale(&self, _alpha: Complex64, _buffer: &mut Self::Buffer) {}
}
