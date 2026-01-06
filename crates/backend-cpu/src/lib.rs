//! CPU spectral backend built on rustfft.
//!
//! # Mixed Precision Support
//!
//! When compiled with `--features mixed-precision`:
//! - Fields are stored as Complex<f32> for 2x memory bandwidth
//! - FFTs are performed in f32 (rustfft supports both)
//! - Dot products accumulate in f64 to prevent catastrophic cancellation
//! - All eigenvalues and Rayleigh-Ritz operations remain in f64

use std::sync::{Arc, Mutex};

use blaze2d_core::backend::SpectralBackend;
use blaze2d_core::field::{Field2D, FieldScalar, FieldReal};
use blaze2d_core::grid::Grid2D;
use num_complex::Complex64;
use rustfft::Fft;

#[cfg(not(feature = "mixed-precision"))]
use rustfft::FftPlanner;

#[cfg(feature = "mixed-precision")]
use num_complex::Complex32;

#[cfg(feature = "mixed-precision")]
use rustfft::FftPlanner as FftPlanner32;

#[cfg(feature = "parallel")]
use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};

#[cfg(feature = "parallel")]
const DEFAULT_PARALLEL_THRESHOLD: usize = 4096;

// Type aliases for FFT precision - must match FieldScalar
#[cfg(not(feature = "mixed-precision"))]
type FftPlannerType = FftPlanner<f64>;

#[cfg(feature = "mixed-precision")]
type FftPlannerType = FftPlanner32<f32>;

#[derive(Clone)]
pub struct CpuBackend {
    #[cfg(feature = "parallel")]
    parallel_fft: bool,
    #[cfg(feature = "parallel")]
    parallel_min_points: usize,
    #[cfg(feature = "parallel")]
    parallel_pool: Option<Arc<ThreadPool>>,
    plan_cache: Arc<Mutex<FftPlannerType>>,
    scratch_cache: Arc<Mutex<Vec<FieldScalar>>>,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "parallel")]
            parallel_fft: false,
            #[cfg(feature = "parallel")]
            parallel_min_points: DEFAULT_PARALLEL_THRESHOLD,
            #[cfg(feature = "parallel")]
            parallel_pool: None,
            plan_cache: Arc::new(Mutex::new(FftPlannerType::new())),
            scratch_cache: Arc::new(Mutex::new(Vec::new())),
        }
    }

    #[cfg(feature = "parallel")]
    pub fn new_parallel() -> Self {
        Self::new()
            .with_parallel_fft(true)
            .with_parallel_threads(num_cpus::get())
            .with_parallel_threshold(DEFAULT_PARALLEL_THRESHOLD)
    }

    #[cfg(feature = "parallel")]
    pub fn with_parallel_fft(mut self, enabled: bool) -> Self {
        self.parallel_fft = enabled;
        self
    }

    #[cfg(feature = "parallel")]
    pub fn with_parallel_threshold(mut self, min_points: usize) -> Self {
        self.parallel_min_points = min_points.max(1);
        self
    }

    #[cfg(feature = "parallel")]
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
        blaze2d_core::profiler::start_timer(match direction {
            FftDirection::Forward => "fft_2d::forward",
            FftDirection::Inverse => "fft_2d::inverse",
        });
        
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

        // Prepare scratch buffer for column transpose
        let required_len = ny;

        let mut scratch_guard = self.scratch_cache.lock().unwrap();
        if scratch_guard.len() < required_len {
            scratch_guard.resize(required_len, FieldScalar::default());
        }
        let scratch = &mut *scratch_guard;

        let data = buffer.as_mut_slice();
        
        #[cfg(feature = "parallel")]
        let use_parallel = self.parallel_fft && grid.len() >= self.parallel_min_points;
        
        #[cfg(not(feature = "parallel"))]
        let use_parallel = false;
        
        if use_parallel {
            #[cfg(feature = "parallel")]
            {
                let mut transposed = vec![FieldScalar::default(); data.len()];
                execute_parallel_fft(
                    self.parallel_pool.as_deref(),
                    data,
                    &mut transposed,
                    nx,
                    ny,
                    row_fft.clone(),
                    col_fft.clone(),
                );
            }
        } else {
            // Serial execution
            process_rows_serial(data, nx, &row_fft);
            process_columns_serial_with_scratch_buffer(data, nx, ny, &col_fft, scratch);
        }

        if matches!(direction, FftDirection::Inverse) {
            let scale = 1.0 / (nx * ny) as FieldReal;
            #[cfg(feature = "parallel")]
            if use_parallel {
                data.par_iter_mut().for_each(|value| *value *= scale);
            } else {
                for value in data.iter_mut() {
                    *value *= scale;
                }
            }
            #[cfg(not(feature = "parallel"))]
            for value in data.iter_mut() {
                *value *= scale;
            }
        }
        
        blaze2d_core::profiler::stop_timer(match direction {
            FftDirection::Forward => "fft_2d::forward",
            FftDirection::Inverse => "fft_2d::inverse",
        });
    }

    /// Batched 2D FFT: process all rows across all buffers first, then all columns.
    ///
    /// This improves cache utilization for FFT plans and reduces mutex contention
    /// by acquiring the plan once and reusing it across all buffers.
    fn batch_fft_2d(&self, buffers: &mut [Field2D], direction: FftDirection) {
        if buffers.is_empty() {
            return;
        }

        blaze2d_core::profiler::start_timer(match direction {
            FftDirection::Forward => "batch_fft_2d::forward",
            FftDirection::Inverse => "batch_fft_2d::inverse",
        });

        // All buffers must have the same grid (same nx, ny)
        let grid = buffers[0].grid();
        let nx = grid.nx;
        let ny = grid.ny;
        assert!(nx > 0 && ny > 0, "grid must be non-zero length");

        #[cfg(debug_assertions)]
        for buf in buffers.iter() {
            let g = buf.grid();
            assert_eq!(g.nx, nx, "all buffers must have same nx");
            assert_eq!(g.ny, ny, "all buffers must have same ny");
        }

        // Acquire FFT plans once for the entire batch
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

        #[cfg(feature = "parallel")]
        let use_parallel = self.parallel_fft && (grid.len() * buffers.len()) >= self.parallel_min_points;

        #[cfg(not(feature = "parallel"))]
        let use_parallel = false;

        if use_parallel {
            #[cfg(feature = "parallel")]
            batch_fft_2d_parallel(
                buffers,
                nx,
                ny,
                row_fft.clone(),
                col_fft.clone(),
                self.parallel_pool.as_deref(),
            );
        } else {
            // Process all rows across all buffers first
            for buffer in buffers.iter_mut() {
                process_rows_serial(buffer.as_mut_slice(), nx, &row_fft);
            }

            // Process all columns across all buffers
            // Reuse a single scratch buffer for all column transforms
            let mut scratch = vec![FieldScalar::default(); ny];
            for buffer in buffers.iter_mut() {
                process_columns_serial_with_scratch_buffer(buffer.as_mut_slice(), nx, ny, &col_fft, &mut scratch);
            }
        }

        // Apply inverse normalization if needed
        if matches!(direction, FftDirection::Inverse) {
            let scale = 1.0 / (nx * ny) as FieldReal;
            for buffer in buffers.iter_mut() {
                let data = buffer.as_mut_slice();
                #[cfg(feature = "parallel")]
                if use_parallel {
                    data.par_iter_mut().for_each(|value| *value *= scale);
                } else {
                    for value in data.iter_mut() {
                        *value *= scale;
                    }
                }
                #[cfg(not(feature = "parallel"))]
                for value in data.iter_mut() {
                    *value *= scale;
                }
            }
        }

        blaze2d_core::profiler::stop_timer(match direction {
            FftDirection::Forward => "batch_fft_2d::forward",
            FftDirection::Inverse => "batch_fft_2d::inverse",
        });
    }
}

enum FftDirection {
    Forward,
    Inverse,
}

fn process_rows_serial(data: &mut [FieldScalar], nx: usize, fft: &Arc<dyn Fft<FieldReal>>) {
    for row in data.chunks_mut(nx) {
        fft.process(row);
    }
}

#[cfg(feature = "parallel")]
fn process_rows_parallel(data: &mut [FieldScalar], nx: usize, fft: Arc<dyn Fft<FieldReal>>) {
    data.par_chunks_mut(nx).for_each(|row| {
        fft.process(row);
    });
}

#[allow(dead_code)]
fn process_columns_serial(data: &mut [FieldScalar], nx: usize, ny: usize, fft: &Arc<dyn Fft<FieldReal>>) {
    let mut scratch = vec![FieldScalar::default(); ny];
    process_columns_serial_with_scratch_buffer(data, nx, ny, fft, &mut scratch);
}

/// Process columns with an externally provided scratch buffer (used as transpose buffer).
fn process_columns_serial_with_scratch_buffer(
    data: &mut [FieldScalar],
    nx: usize,
    ny: usize,
    fft: &Arc<dyn Fft<FieldReal>>,
    scratch: &mut [FieldScalar],
) {
    debug_assert!(scratch.len() >= ny);
    
    // We only use the scratch for transposing. 
    // We let the FFT manage its own internal scratch (allocating if necessary) to avoid panics.
    let transpose_buf = &mut scratch[..ny];
    
    for ix in 0..nx {
        for iy in 0..ny {
            transpose_buf[iy] = data[iy * nx + ix];
        }
        fft.process(transpose_buf);
        for iy in 0..ny {
            data[iy * nx + ix] = transpose_buf[iy];
        }
    }
}

#[cfg(feature = "parallel")]
fn transpose_into(src: &[FieldScalar], dst: &mut [FieldScalar], nx: usize, ny: usize) {
    assert_eq!(src.len(), dst.len());
    for iy in 0..ny {
        for ix in 0..nx {
            let src_idx = iy * nx + ix;
            let dst_idx = ix * ny + iy;
            dst[dst_idx] = src[src_idx];
        }
    }
}

#[cfg(feature = "parallel")]
fn execute_parallel_fft(
    pool: Option<&ThreadPool>,
    data: &mut [FieldScalar],
    transposed: &mut [FieldScalar],
    nx: usize,
    ny: usize,
    row_fft: Arc<dyn Fft<FieldReal>>,
    col_fft: Arc<dyn Fft<FieldReal>>,
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

/// Batched parallel FFT for multiple buffers.
///
/// Processes all rows across all buffers in parallel, then all columns.
/// Uses transpose-based column processing for better parallelism.
#[cfg(feature = "parallel")]
fn batch_fft_2d_parallel(
    buffers: &mut [Field2D],
    nx: usize,
    ny: usize,
    row_fft: Arc<dyn Fft<FieldReal>>,
    col_fft: Arc<dyn Fft<FieldReal>>,
    pool: Option<&ThreadPool>,
) {
    let mut job = || {
        // Phase 1: Process all rows across all buffers in parallel
        // Collect all row data pointers for parallel iteration
        let total_rows = buffers.len() * ny;
        let mut row_slices: Vec<&mut [FieldScalar]> = Vec::with_capacity(total_rows);
        
        for buffer in buffers.iter_mut() {
            for row in buffer.as_mut_slice().chunks_mut(nx) {
                // SAFETY: We're collecting mutable references to non-overlapping rows.
                // This is sound because chunks_mut returns non-overlapping slices.
                row_slices.push(row);
            }
        }
        
        let row_fft_clone = row_fft.clone();
        row_slices.par_iter_mut().for_each(|row| {
            row_fft_clone.process(*row);
        });
        
        // Phase 2: Process columns using transpose trick for each buffer
        // This is done per-buffer since we need a transposed buffer for each
        for buffer in buffers.iter_mut() {
            let data = buffer.as_mut_slice();
            let mut transposed = vec![FieldScalar::default(); data.len()];
            transpose_into(data, &mut transposed, nx, ny);
            
            // Process transposed rows (which are original columns) in parallel
            transposed.par_chunks_mut(ny).for_each(|row| {
                col_fft.process(row);
            });
            
            // Transpose back
            transpose_into(&transposed, data, ny, nx);
        }
    };
    
    if let Some(pool) = pool {
        pool.install(&mut job);
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
        // Convert f64 alpha to storage precision
        #[cfg(not(feature = "mixed-precision"))]
        {
            for value in buffer.as_mut_slice() {
                *value *= alpha;
            }
        }
        #[cfg(feature = "mixed-precision")]
        {
            let alpha32 = Complex32::new(alpha.re as f32, alpha.im as f32);
            for value in buffer.as_mut_slice() {
                *value *= alpha32;
            }
        }
    }

    fn axpy(&self, alpha: Complex64, x: &Self::Buffer, y: &mut Self::Buffer) {
        // Convert f64 alpha to storage precision
        #[cfg(not(feature = "mixed-precision"))]
        {
            for (dst, src) in y.as_mut_slice().iter_mut().zip(x.as_slice()) {
                *dst += alpha * src;
            }
        }
        #[cfg(feature = "mixed-precision")]
        {
            let alpha32 = Complex32::new(alpha.re as f32, alpha.im as f32);
            for (dst, src) in y.as_mut_slice().iter_mut().zip(x.as_slice()) {
                *dst += alpha32 * src;
            }
        }
    }

    /// Conjugate dot product with f64 accumulation.
    ///
    /// **CRITICAL**: Even in mixed-precision mode (f32 storage), we accumulate
    /// in f64 to prevent catastrophic cancellation during orthogonalization.
    /// The CPU-to-register conversion cost is negligible compared to the
    /// numerical stability benefit.
    fn dot(&self, x: &Self::Buffer, y: &Self::Buffer) -> Complex64 {
        #[cfg(not(feature = "mixed-precision"))]
        {
            x.as_slice()
                .iter()
                .zip(y.as_slice())
                .map(|(a, b)| a.conj() * b)
                .sum()
        }
        #[cfg(feature = "mixed-precision")]
        {
            // Accumulate in f64 even though storage is f32
            x.as_slice()
                .iter()
                .zip(y.as_slice())
                .map(|(a, b)| {
                    let a64 = Complex64::new(a.re as f64, a.im as f64);
                    let b64 = Complex64::new(b.re as f64, b.im as f64);
                    a64.conj() * b64
                })
                .fold(Complex64::new(0.0, 0.0), |acc, x| acc + x)
        }
    }

    fn batch_forward_fft_2d(&self, buffers: &mut [Self::Buffer]) {
        self.batch_fft_2d(buffers, FftDirection::Forward);
    }

    fn batch_inverse_fft_2d(&self, buffers: &mut [Self::Buffer]) {
        self.batch_fft_2d(buffers, FftDirection::Inverse);
    }
}

#[cfg(test)]
mod _tests_lib;

#[test]
fn test_fft_normalization() {
    use crate::CpuBackend;
    use blaze2d_core::backend::SpectralBackend;
    use blaze2d_core::field::{Field2D, FieldScalar};
    use blaze2d_core::grid::Grid2D;
    use std::f64::consts::PI;

    let backend = CpuBackend::new();
    let grid = Grid2D {
        nx: 8,
        ny: 8,
        lx: 1.0,
        ly: 1.0,
    };

    // Create a plane wave with G = (2π, 0) - using f64 for computation, then convert
    let data: Vec<FieldScalar> = (0..64)
        .map(|i| {
            let ix = i % 8;
            let x = ix as f64 / 8.0;
            let arg = 2.0 * PI * x;
            #[cfg(not(feature = "mixed-precision"))]
            { Complex64::new(arg.cos(), arg.sin()) }
            #[cfg(feature = "mixed-precision")]
            { Complex32::new(arg.cos() as f32, arg.sin() as f32) }
        })
        .collect();
    let mut field = Field2D::from_vec(grid, data);

    backend.forward_fft_2d(&mut field);

    // After FFT, should have a peak at frequency (1, 0)
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
    use blaze2d_core::backend::SpectralBackend;
    use blaze2d_core::dielectric::{Dielectric2D, DielectricOptions};
    use blaze2d_core::field::{Field2D, FieldScalar};
    use blaze2d_core::geometry::Geometry2D;
    use blaze2d_core::grid::Grid2D;
    use blaze2d_core::lattice::Lattice2D;
    use blaze2d_core::operators::{LinearOperator, ThetaOperator};
    use blaze2d_core::polarization::Polarization;
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
    let input_data: Vec<FieldScalar> = (0..64)
        .map(|i| {
            let ix = i % 8;
            let x = ix as f64 / 8.0;
            let arg = 2.0 * PI * x;
            #[cfg(not(feature = "mixed-precision"))]
            { Complex64::new(arg.cos(), arg.sin()) }
            #[cfg(feature = "mixed-precision")]
            { Complex32::new(arg.cos() as f32, arg.sin() as f32) }
        })
        .collect();
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
    use blaze2d_core::backend::SpectralBackend;
    use blaze2d_core::dielectric::{Dielectric2D, DielectricOptions};
    use blaze2d_core::field::{Field2D, FieldScalar};
    use blaze2d_core::geometry::Geometry2D;
    use blaze2d_core::grid::Grid2D;
    use blaze2d_core::lattice::Lattice2D;
    use blaze2d_core::operators::{LinearOperator, ThetaOperator};
    use blaze2d_core::polarization::Polarization;
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
    let input_data: Vec<FieldScalar> = (0..64)
        .map(|i| {
            let ix = i % 8;
            let x = ix as f64 / 8.0;
            let arg = 2.0 * PI * x;
            #[cfg(not(feature = "mixed-precision"))]
            { Complex64::new(arg.cos(), arg.sin()) }
            #[cfg(feature = "mixed-precision")]
            { Complex32::new(arg.cos() as f32, arg.sin() as f32) }
        })
        .collect();
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
