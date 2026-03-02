//! CPU spectral backend built on rustfft.

use std::sync::{Arc, Mutex};

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
    plan_cache: Arc<Mutex<FftPlanner<f64>>>,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            parallel_fft: false,
            parallel_min_points: DEFAULT_PARALLEL_THRESHOLD,
            parallel_pool: None,
            plan_cache: Arc::new(Mutex::new(FftPlanner::new())),
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

        let (row_fft, col_fft) = {
            let mut planner = self
                .plan_cache
                .lock()
                .expect("fft plan cache mutex poisoned");
            let row = match direction {
                FftDirection::Forward => planner.plan_fft_forward(nx),
                FftDirection::Inverse => planner.plan_fft_inverse(nx),
            };
            let col = match direction {
                FftDirection::Forward => planner.plan_fft_forward(ny),
                FftDirection::Inverse => planner.plan_fft_inverse(ny),
            };
            (row, col)
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

#[test]
fn test_fft_normalization() {
    use crate::CpuBackend;
    use mpb2d_core::backend::SpectralBackend;
    use mpb2d_core::field::Field2D;
    use mpb2d_core::grid::Grid2D;
    use std::f64::consts::PI;

    let backend = CpuBackend::new();
    let grid = Grid2D {
        nx: 8,
        ny: 8,
        lx: 1.0,
        ly: 1.0,
    };

    // Create a plane wave with G = (2π, 0)
    let mut data = vec![Complex64::default(); 64];
    for iy in 0..8 {
        for ix in 0..8 {
            let x = ix as f64 / 8.0;
            let arg = 2.0 * PI * x;
            data[iy * 8 + ix] = Complex64::new(arg.cos(), arg.sin());
        }
    }
    let mut field = Field2D::from_vec(grid, data);

    backend.forward_fft_2d(&mut field);

    // After FFT, should have a peak at frequency (1, 0)
    // The value should be N^2 = 64 (for 2D FFT without normalization)
    let peak = field.as_slice()[0 * 8 + 1]; // freq_x=1, freq_y=0
    eprintln!("FFT peak at (1,0): {:?}", peak);
    eprintln!("Peak magnitude: {}", peak.norm());

    // Check k-vector for this frequency
    let g = 2.0 * PI * 1.0 / 1.0; // 2π * freq / L
    eprintln!("Expected |G|^2 = {}", g * g);
}

#[test]
fn test_tm_operator_eigenvalue() {
    use crate::CpuBackend;
    use mpb2d_core::backend::SpectralBackend;
    use mpb2d_core::dielectric::{Dielectric2D, DielectricOptions};
    use mpb2d_core::field::Field2D;
    use mpb2d_core::geometry::Geometry2D;
    use mpb2d_core::grid::Grid2D;
    use mpb2d_core::lattice::Lattice2D;
    use mpb2d_core::operators::{LinearOperator, ThetaOperator};
    use mpb2d_core::polarization::Polarization;
    use std::f64::consts::PI;

    let backend = CpuBackend::new();
    let grid = Grid2D {
        nx: 8,
        ny: 8,
        lx: 1.0,
        ly: 1.0,
    };

    // Uniform dielectric eps = 1
    let lattice = Lattice2D::square(1.0);
    let geom = Geometry2D {
        lattice,
        eps_bg: 1.0,
        atoms: Vec::new(),
    };
    let dielectric = Dielectric2D::from_geometry(&geom, grid, &DielectricOptions::default());

    // Create TM operator at Γ point (k = 0)
    let bloch = [0.0, 0.0];
    let mut operator = ThetaOperator::new(backend.clone(), dielectric, Polarization::TM, bloch);

    // Create a plane wave with G = (2π, 0)
    let mut input_data = vec![Complex64::default(); 64];
    for iy in 0..8 {
        for ix in 0..8 {
            let x = ix as f64 / 8.0;
            let arg = 2.0 * PI * x;
            input_data[iy * 8 + ix] = Complex64::new(arg.cos(), arg.sin());
        }
    }
    let input = Field2D::from_vec(grid, input_data);

    // Apply operator
    let mut output = operator.alloc_field();
    operator.apply(&input, &mut output);

    // Compute Rayleigh quotient: <x, Ax> / <x, Bx>
    // For eps=1, Bx = x, so <x, Bx> = <x, x>
    let x_dot_ax = backend.dot(&input, &output);
    let x_dot_x = backend.dot(&input, &input);
    let rayleigh = x_dot_ax.re / x_dot_x.re;

    eprintln!("<x, Ax> = {:?}", x_dot_ax);
    eprintln!("<x, x> = {:?}", x_dot_x);
    eprintln!("Rayleigh quotient = {}", rayleigh);

    // Expected: |G|^2 = (2π)^2 ≈ 39.48
    let expected = (2.0 * PI) * (2.0 * PI);
    eprintln!("Expected |G|^2 = {}", expected);

    assert!(
        (rayleigh - expected).abs() < 1.0,
        "Rayleigh quotient {} should be close to {}",
        rayleigh,
        expected
    );
}

#[test]
fn test_te_operator_eigenvalue() {
    use crate::CpuBackend;
    use mpb2d_core::backend::SpectralBackend;
    use mpb2d_core::dielectric::{Dielectric2D, DielectricOptions};
    use mpb2d_core::field::Field2D;
    use mpb2d_core::geometry::Geometry2D;
    use mpb2d_core::grid::Grid2D;
    use mpb2d_core::lattice::Lattice2D;
    use mpb2d_core::operators::{LinearOperator, ThetaOperator};
    use mpb2d_core::polarization::Polarization;
    use std::f64::consts::PI;

    let backend = CpuBackend::new();
    let grid = Grid2D {
        nx: 8,
        ny: 8,
        lx: 1.0,
        ly: 1.0,
    };

    // Uniform dielectric eps = 1
    let lattice = Lattice2D::square(1.0);
    let geom = Geometry2D {
        lattice,
        eps_bg: 1.0,
        atoms: Vec::new(),
    };
    let dielectric = Dielectric2D::from_geometry(&geom, grid, &DielectricOptions::default());

    // Create TE operator at Γ point (k = 0)
    let bloch = [0.0, 0.0];
    let mut operator = ThetaOperator::new(backend.clone(), dielectric, Polarization::TE, bloch);

    // Create a plane wave with G = (2π, 0)
    let mut input_data = vec![Complex64::default(); 64];
    for iy in 0..8 {
        for ix in 0..8 {
            let x = ix as f64 / 8.0;
            let arg = 2.0 * PI * x;
            input_data[iy * 8 + ix] = Complex64::new(arg.cos(), arg.sin());
        }
    }
    let input = Field2D::from_vec(grid, input_data);

    // Apply operator
    let mut output = operator.alloc_field();
    operator.apply(&input, &mut output);

    // Compute Rayleigh quotient: <x, Ax> / <x, Bx>
    // For TM with eps=1 (so inv_eps=1), Bx = x, so <x, Bx> = <x, x>
    let x_dot_ax = backend.dot(&input, &output);
    let x_dot_x = backend.dot(&input, &input);
    let rayleigh = x_dot_ax.re / x_dot_x.re;

    eprintln!("TM: <x, Ax> = {:?}", x_dot_ax);
    eprintln!("TM: <x, x> = {:?}", x_dot_x);
    eprintln!("TM: Rayleigh quotient = {}", rayleigh);

    // Expected: |G|^2 = (2π)^2 ≈ 39.48
    let expected = (2.0 * PI) * (2.0 * PI);
    eprintln!("Expected |G|^2 = {}", expected);

    assert!(
        (rayleigh - expected).abs() < 1.0,
        "TM Rayleigh quotient {} should be close to {}",
        rayleigh,
        expected
    );
}
