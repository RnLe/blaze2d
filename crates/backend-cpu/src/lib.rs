//! CPU spectral backend built on rustfft.
//!
//! # Runtime-selectable precision
//!
//! [`CpuBackend<R>`] is generic over the storage precision `R: Real + FftNum`.
//! The crate ships both `CpuBackend<f32>` and `CpuBackend<f64>` monomorphisations
//! so a single binary can serve both via runtime dispatch. The default type
//! parameter is `f64` so unannotated `CpuBackend` keeps its previous meaning.
//!
//! Both monomorphisations honour the f32-storage / f64-accumulation invariant:
//! storage and FFTs run at `R` precision, dot products accumulate in `Complex<f64>`,
//! and `scale` / `axpy` accept f64-precision coefficients (downcast on the fly).

use std::sync::{Arc, Mutex};

use blaze2d_core::backend::SpectralBackend;
use blaze2d_core::field::{Field2D, Real, cscalar};
use blaze2d_core::grid::Grid2D;
use num_complex::{Complex, Complex64};
use rustfft::{Fft, FftNum, FftPlanner};

#[cfg(feature = "parallel")]
use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};

#[cfg(feature = "parallel")]
const DEFAULT_PARALLEL_THRESHOLD: usize = 4096;

#[derive(Clone)]
pub struct CpuBackend<R: Real + FftNum = f64> {
    #[cfg(feature = "parallel")]
    parallel_fft: bool,
    #[cfg(feature = "parallel")]
    parallel_min_points: usize,
    #[cfg(feature = "parallel")]
    parallel_pool: Option<Arc<ThreadPool>>,
    plan_cache: Arc<Mutex<FftPlanner<R>>>,
    scratch_cache: Arc<Mutex<Vec<Complex<R>>>>,
}

impl<R: Real + FftNum> Default for CpuBackend<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Real + FftNum> CpuBackend<R> {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "parallel")]
            parallel_fft: false,
            #[cfg(feature = "parallel")]
            parallel_min_points: DEFAULT_PARALLEL_THRESHOLD,
            #[cfg(feature = "parallel")]
            parallel_pool: None,
            plan_cache: Arc::new(Mutex::new(FftPlanner::<R>::new())),
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

    fn fft_2d(&self, buffer: &mut Field2D<R>, direction: FftDirection) {
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

        let required_len = ny;
        let zero = Complex::<R>::new(R::zero(), R::zero());

        let mut scratch_guard = self.scratch_cache.lock().unwrap();
        if scratch_guard.len() < required_len {
            scratch_guard.resize(required_len, zero);
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
                let mut transposed = vec![zero; data.len()];
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
            process_rows_serial(data, nx, &row_fft);
            process_columns_serial_with_scratch_buffer(data, nx, ny, &col_fft, scratch);
        }

        if matches!(direction, FftDirection::Inverse) {
            let scale = R::from_accum(1.0 / (nx * ny) as f64);
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

    fn batch_fft_2d(&self, buffers: &mut [Field2D<R>], direction: FftDirection) {
        if buffers.is_empty() {
            return;
        }

        blaze2d_core::profiler::start_timer(match direction {
            FftDirection::Forward => "batch_fft_2d::forward",
            FftDirection::Inverse => "batch_fft_2d::inverse",
        });

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

        let zero = Complex::<R>::new(R::zero(), R::zero());

        #[cfg(feature = "parallel")]
        let use_parallel =
            self.parallel_fft && (grid.len() * buffers.len()) >= self.parallel_min_points;

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
            for buffer in buffers.iter_mut() {
                process_rows_serial(buffer.as_mut_slice(), nx, &row_fft);
            }
            let mut scratch = vec![zero; ny];
            for buffer in buffers.iter_mut() {
                process_columns_serial_with_scratch_buffer(
                    buffer.as_mut_slice(),
                    nx,
                    ny,
                    &col_fft,
                    &mut scratch,
                );
            }
        }

        if matches!(direction, FftDirection::Inverse) {
            let scale = R::from_accum(1.0 / (nx * ny) as f64);
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

fn process_rows_serial<R: FftNum>(data: &mut [Complex<R>], nx: usize, fft: &Arc<dyn Fft<R>>) {
    for row in data.chunks_mut(nx) {
        fft.process(row);
    }
}

#[cfg(feature = "parallel")]
fn process_rows_parallel<R: FftNum>(data: &mut [Complex<R>], nx: usize, fft: Arc<dyn Fft<R>>) {
    data.par_chunks_mut(nx).for_each(|row| {
        fft.process(row);
    });
}

fn process_columns_serial_with_scratch_buffer<R: FftNum>(
    data: &mut [Complex<R>],
    nx: usize,
    ny: usize,
    fft: &Arc<dyn Fft<R>>,
    scratch: &mut [Complex<R>],
) {
    debug_assert!(scratch.len() >= ny);

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
fn transpose_into<R: Real>(src: &[Complex<R>], dst: &mut [Complex<R>], nx: usize, ny: usize) {
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
fn execute_parallel_fft<R: Real + FftNum>(
    pool: Option<&ThreadPool>,
    data: &mut [Complex<R>],
    transposed: &mut [Complex<R>],
    nx: usize,
    ny: usize,
    row_fft: Arc<dyn Fft<R>>,
    col_fft: Arc<dyn Fft<R>>,
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

#[cfg(feature = "parallel")]
fn batch_fft_2d_parallel<R: Real + FftNum>(
    buffers: &mut [Field2D<R>],
    nx: usize,
    ny: usize,
    row_fft: Arc<dyn Fft<R>>,
    col_fft: Arc<dyn Fft<R>>,
    pool: Option<&ThreadPool>,
) {
    let zero = Complex::<R>::new(R::zero(), R::zero());
    let mut job = || {
        let total_rows = buffers.len() * ny;
        let mut row_slices: Vec<&mut [Complex<R>]> = Vec::with_capacity(total_rows);

        for buffer in buffers.iter_mut() {
            for row in buffer.as_mut_slice().chunks_mut(nx) {
                row_slices.push(row);
            }
        }

        let row_fft_clone = row_fft.clone();
        row_slices.par_iter_mut().for_each(|row| {
            row_fft_clone.process(*row);
        });

        for buffer in buffers.iter_mut() {
            let data = buffer.as_mut_slice();
            let mut transposed = vec![zero; data.len()];
            transpose_into(data, &mut transposed, nx, ny);

            transposed.par_chunks_mut(ny).for_each(|row| {
                col_fft.process(row);
            });

            transpose_into(&transposed, data, ny, nx);
        }
    };

    if let Some(pool) = pool {
        pool.install(&mut job);
    } else {
        job();
    }
}

impl<R: Real + FftNum> SpectralBackend for CpuBackend<R> {
    type Real = R;
    type Buffer = Field2D<R>;

    fn alloc_field(&self, grid: Grid2D) -> Self::Buffer {
        Field2D::<R>::zeros(grid)
    }

    fn forward_fft_2d(&self, buffer: &mut Self::Buffer) {
        self.fft_2d(buffer, FftDirection::Forward);
    }

    fn inverse_fft_2d(&self, buffer: &mut Self::Buffer) {
        self.fft_2d(buffer, FftDirection::Inverse);
    }

    fn scale(&self, alpha: Complex64, buffer: &mut Self::Buffer) {
        let alpha_r = cscalar::<R>(alpha.re, alpha.im);
        for value in buffer.as_mut_slice() {
            *value *= alpha_r;
        }
    }

    fn axpy(&self, alpha: Complex64, x: &Self::Buffer, y: &mut Self::Buffer) {
        let alpha_r = cscalar::<R>(alpha.re, alpha.im);
        for (dst, src) in y.as_mut_slice().iter_mut().zip(x.as_slice()) {
            *dst += alpha_r * src;
        }
    }

    /// Conjugate dot product, accumulated in `Complex<f64>` regardless of `R`.
    ///
    /// This is the f32-storage / f64-accumulation invariant: storage may be
    /// `Complex<f32>` for bandwidth, but accumulation is always in `Complex<f64>`
    /// to prevent catastrophic cancellation during orthogonalisation.
    fn dot(&self, x: &Self::Buffer, y: &Self::Buffer) -> Complex64 {
        x.as_slice()
            .iter()
            .zip(y.as_slice())
            .map(|(a, b)| {
                let a64 = Complex64::new(a.re.to_accum(), a.im.to_accum());
                let b64 = Complex64::new(b.re.to_accum(), b.im.to_accum());
                a64.conj() * b64
            })
            .sum()
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
    use blaze2d_core::field::Field2D;
    use blaze2d_core::grid::Grid2D;
    use std::f64::consts::PI;

    let backend = CpuBackend::<f64>::new();
    let grid = Grid2D::new(8, 8, 1.0, 1.0);

    let data: Vec<Complex64> = (0..64)
        .map(|i| {
            let ix = i % 8;
            let x = ix as f64 / 8.0;
            let arg = 2.0 * PI * x;
            Complex64::new(arg.cos(), arg.sin())
        })
        .collect();
    let mut field = Field2D::from_vec(grid, data);

    backend.forward_fft_2d(&mut field);

    let peak = field.as_slice()[1];
    eprintln!("FFT peak at (1,0): {:?}", peak);
    eprintln!("Peak magnitude: {}", peak.norm());

    let g = 2.0 * PI;
    eprintln!("Expected |G|^2 = {}", g * g);
}

#[test]
fn test_tm_operator_eigenvalue() {
    use crate::CpuBackend;
    use blaze2d_core::backend::SpectralBackend;
    use blaze2d_core::dielectric::{Dielectric2D, DielectricOptions};
    use blaze2d_core::field::Field2D;
    use blaze2d_core::geometry::Geometry2D;
    use blaze2d_core::grid::Grid2D;
    use blaze2d_core::lattice::Lattice2D;
    use blaze2d_core::operators::{LinearOperator, ThetaOperator};
    use blaze2d_core::polarization::Polarization;
    use std::f64::consts::PI;

    let backend = CpuBackend::<f64>::new();
    let grid = Grid2D::new(8, 8, 1.0, 1.0);

    let lattice = Lattice2D::square(1.0);
    let geom = Geometry2D {
        lattice,
        eps_bg: 1.0,
        atoms: Vec::new(),
    };
    let dielectric = Dielectric2D::from_geometry(&geom, grid, &DielectricOptions::default());

    let bloch = [0.0, 0.0];
    let mut operator = ThetaOperator::new(backend.clone(), dielectric, Polarization::TM, bloch);

    let input_data: Vec<Complex64> = (0..64)
        .map(|i| {
            let ix = i % 8;
            let x = ix as f64 / 8.0;
            let arg = 2.0 * PI * x;
            Complex64::new(arg.cos(), arg.sin())
        })
        .collect();
    let input = Field2D::from_vec(grid, input_data);

    let mut output = operator.alloc_field();
    operator.apply(&input, &mut output);

    let x_dot_ax = backend.dot(&input, &output);
    let x_dot_x = backend.dot(&input, &input);
    let rayleigh = x_dot_ax.re / x_dot_x.re;

    eprintln!("<x, Ax> = {:?}", x_dot_ax);
    eprintln!("<x, x> = {:?}", x_dot_x);
    eprintln!("Rayleigh quotient = {}", rayleigh);

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
    use blaze2d_core::field::Field2D;
    use blaze2d_core::geometry::Geometry2D;
    use blaze2d_core::grid::Grid2D;
    use blaze2d_core::lattice::Lattice2D;
    use blaze2d_core::operators::{LinearOperator, ThetaOperator};
    use blaze2d_core::polarization::Polarization;
    use std::f64::consts::PI;

    let backend = CpuBackend::<f64>::new();
    let grid = Grid2D::new(8, 8, 1.0, 1.0);

    let lattice = Lattice2D::square(1.0);
    let geom = Geometry2D {
        lattice,
        eps_bg: 1.0,
        atoms: Vec::new(),
    };
    let dielectric = Dielectric2D::from_geometry(&geom, grid, &DielectricOptions::default());

    let bloch = [0.0, 0.0];
    let mut operator = ThetaOperator::new(backend.clone(), dielectric, Polarization::TE, bloch);

    let input_data: Vec<Complex64> = (0..64)
        .map(|i| {
            let ix = i % 8;
            let x = ix as f64 / 8.0;
            let arg = 2.0 * PI * x;
            Complex64::new(arg.cos(), arg.sin())
        })
        .collect();
    let input = Field2D::from_vec(grid, input_data);

    let mut output = operator.alloc_field();
    operator.apply(&input, &mut output);

    let x_dot_ax = backend.dot(&input, &output);
    let x_dot_x = backend.dot(&input, &input);
    let rayleigh = x_dot_ax.re / x_dot_x.re;

    eprintln!("TM: <x, Ax> = {:?}", x_dot_ax);
    eprintln!("TM: <x, x> = {:?}", x_dot_x);
    eprintln!("TM: Rayleigh quotient = {}", rayleigh);

    let expected = (2.0 * PI) * (2.0 * PI);
    eprintln!("Expected |G|^2 = {}", expected);

    assert!(
        (rayleigh - expected).abs() < 1.0,
        "TM Rayleigh quotient {} should be close to {}",
        rayleigh,
        expected
    );
}
