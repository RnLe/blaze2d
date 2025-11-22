//! CPU spectral backend built on rustfft (scaffolding only for now).

use std::sync::Arc;

use mpb2d-core::backend::{SpectralBackend, SpectralBuffer};
use mpb2d-core::grid::Grid2D;
use num_complex::Complex64;
use rustfft::FftPlanner;

pub struct CpuBackend {
    planner: Arc<FftPlanner<f64>>,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            planner: Arc::new(FftPlanner::new()),
        }
    }
}

pub struct CpuField {
    pub grid: Grid2D,
    pub data: Vec<Complex64>,
}

impl SpectralBuffer for CpuField {
    fn len(&self) -> usize {
        self.data.len()
    }
}

impl SpectralBackend for CpuBackend {
    type Buffer = CpuField;

    fn alloc_field(&self, grid: Grid2D) -> Self::Buffer {
        CpuField {
            grid,
            data: vec![Complex64::default(); grid.len()],
        }
    }

    fn forward_fft_2d(&self, buffer: &mut Self::Buffer) {
        let _planner: &Arc<FftPlanner<f64>> = &self.planner;
        let _grid = buffer.grid;
        // TODO: Phase 1 will implement actual 2D FFT using rustfft plans.
    }

    fn inverse_fft_2d(&self, buffer: &mut Self::Buffer) {
        let _planner: &Arc<FftPlanner<f64>> = &self.planner;
        let _grid = buffer.grid;
        // TODO: Phase 1 will implement actual inverse FFT.
    }

    fn scale(&self, alpha: Complex64, buffer: &mut Self::Buffer) {
        for value in &mut buffer.data {
            *value *= alpha;
        }
    }
}
