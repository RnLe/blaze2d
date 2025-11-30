//! CPU spectral backend built on rustfft.

use std::sync::Arc;

use mpb2d_core::backend::SpectralBackend;
use mpb2d_core::field::Field2D;
use mpb2d_core::grid::Grid2D;
use num_complex::Complex64;
use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};
use rustfft::{Fft, FftPlanner};

const DEFAULT_PARALLEL_THRESHOLD: usize = 4096;

#[derive(Clone)]
pub struct CpuBackend {
    parallel_fft: bool,
    parallel_min_points: usize,
    parallel_pool: Option<Arc<ThreadPool>>,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            parallel_fft: false,
            parallel_min_points: DEFAULT_PARALLEL_THRESHOLD,
            parallel_pool: None,
        }
    }

    pub fn new_parallel() -> Self {
        Self::new()
            .with_parallel_fft(true)
            .with_parallel_threads(num_cpus::get())
            .with_parallel_threshold(DEFAULT_PARALLEL_THRESHOLD)
    }

    pub fn with_parallel_fft(mut self, enabled: bool) -> Self {
        self.parallel_fft = enabled;
        self
    }

    pub fn with_parallel_threshold(mut self, min_points: usize) -> Self {
        self.parallel_min_points = min_points.max(1);
        self
    }

    pub fn with_parallel_threads(mut self, threads: usize) -> Self {
        if threads == 0 {
            self.parallel_pool = None;
            return self;
        }
        self.parallel_pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .ok()
            .map(Arc::new);
        self
    }

    fn fft_2d(&self, buffer: &mut Field2D, direction: FftDirection) {
        let grid = buffer.grid();
        let nx = grid.nx;
        let ny = grid.ny;
        assert!(nx > 0 && ny > 0, "grid must be non-zero length");

        let mut planner = FftPlanner::<f64>::new();
        let row_fft = match direction {
            FftDirection::Forward => planner.plan_fft_forward(nx),
            FftDirection::Inverse => planner.plan_fft_inverse(nx),
        };
        let col_fft = match direction {
            FftDirection::Forward => planner.plan_fft_forward(ny),
            FftDirection::Inverse => planner.plan_fft_inverse(ny),
        };

        let data = buffer.as_mut_slice();
        let use_parallel = self.parallel_fft && grid.len() >= self.parallel_min_points;
        if use_parallel {
            let mut transposed = vec![Complex64::default(); data.len()];
            execute_parallel_fft(
                self.parallel_pool.as_deref(),
                data,
                &mut transposed,
                nx,
                ny,
                row_fft.clone(),
                col_fft.clone(),
            );
        } else {
            process_rows_serial(data, nx, &row_fft);
            process_columns_serial(data, nx, ny, &col_fft);
        }

        if matches!(direction, FftDirection::Inverse) {
            let scale = 1.0 / (nx * ny) as f64;
            if use_parallel {
                data.par_iter_mut().for_each(|value| *value *= scale);
            } else {
                for value in data.iter_mut() {
                    *value *= scale;
                }
            }
        }
    }
}

enum FftDirection {
    Forward,
    Inverse,
}

fn process_rows_serial(data: &mut [Complex64], nx: usize, fft: &Arc<dyn Fft<f64>>) {
    for row in data.chunks_mut(nx) {
        fft.process(row);
    }
}

fn process_rows_parallel(data: &mut [Complex64], nx: usize, fft: Arc<dyn Fft<f64>>) {
    data.par_chunks_mut(nx).for_each(|row| {
        fft.process(row);
    });
}

fn process_columns_serial(data: &mut [Complex64], nx: usize, ny: usize, fft: &Arc<dyn Fft<f64>>) {
    let mut scratch = vec![Complex64::default(); ny];
    for ix in 0..nx {
        for iy in 0..ny {
            scratch[iy] = data[iy * nx + ix];
        }
        fft.process(&mut scratch);
        for iy in 0..ny {
            data[iy * nx + ix] = scratch[iy];
        }
    }
}

fn transpose_into(src: &[Complex64], dst: &mut [Complex64], nx: usize, ny: usize) {
    assert_eq!(src.len(), dst.len());
    for iy in 0..ny {
        for ix in 0..nx {
            let src_idx = iy * nx + ix;
            let dst_idx = ix * ny + iy;
            dst[dst_idx] = src[src_idx];
        }
    }
}

fn execute_parallel_fft(
    pool: Option<&ThreadPool>,
    data: &mut [Complex64],
    transposed: &mut [Complex64],
    nx: usize,
    ny: usize,
    row_fft: Arc<dyn Fft<f64>>,
    col_fft: Arc<dyn Fft<f64>>,
) {
    let job = || {
        process_rows_parallel(data, nx, row_fft.clone());
        transpose_into(data, transposed, nx, ny);
        process_rows_parallel(transposed, ny, col_fft);
        transpose_into(transposed, data, ny, nx);
    };
    if let Some(pool) = pool {
        pool.install(job);
    } else {
        job();
    }
}

impl SpectralBackend for CpuBackend {
    type Buffer = Field2D;

    fn alloc_field(&self, grid: Grid2D) -> Self::Buffer {
        Field2D::zeros(grid)
    }

    fn forward_fft_2d(&self, buffer: &mut Self::Buffer) {
        self.fft_2d(buffer, FftDirection::Forward);
    }

    fn inverse_fft_2d(&self, buffer: &mut Self::Buffer) {
        self.fft_2d(buffer, FftDirection::Inverse);
    }

    fn scale(&self, alpha: Complex64, buffer: &mut Self::Buffer) {
        for value in buffer.as_mut_slice() {
            *value *= alpha;
        }
    }

    fn axpy(&self, alpha: Complex64, x: &Self::Buffer, y: &mut Self::Buffer) {
        for (dst, src) in y.as_mut_slice().iter_mut().zip(x.as_slice()) {
            *dst += alpha * src;
        }
    }

    fn dot(&self, x: &Self::Buffer, y: &Self::Buffer) -> Complex64 {
        x.as_slice()
            .iter()
            .zip(y.as_slice())
            .map(|(a, b)| a.conj() * b)
            .sum()
    }
}

#[cfg(test)]
mod _tests_lib;
