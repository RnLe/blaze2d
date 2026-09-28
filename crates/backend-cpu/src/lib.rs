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
    workspace: Arc<Mutex<FftWorkspace<R>>>,
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
            workspace: Arc::new(Mutex::new(FftWorkspace::default())),
        }
    }

    /// Share immutable plans with an independent worker. Ordinary clones share
    /// a serial workspace; concurrent workers must use this constructor.
    pub fn fork_worker(&self) -> Self {
        let mut worker = self.clone();
        worker.workspace = Arc::new(Mutex::new(FftWorkspace::default()));
        worker
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

        let mut workspace = self.workspace.lock().expect("FFT workspace poisoned");
        workspace.prepare(&self.plan_cache, nx, ny);
        let (row_fft, col_fft) = workspace.plans(direction);
        let data = buffer.as_mut_slice();

        #[cfg(feature = "parallel")]
        let use_parallel = self.parallel_fft && grid.len() >= self.parallel_min_points;

        #[cfg(not(feature = "parallel"))]
        let use_parallel = false;

        if use_parallel {
            #[cfg(feature = "parallel")]
            {
                let zero = Complex::<R>::new(R::zero(), R::zero());
                workspace.transposed.resize(data.len(), zero);
                execute_parallel_fft(
                    self.parallel_pool.as_deref(),
                    data,
                    &mut workspace.transposed,
                    nx,
                    ny,
                    row_fft.clone(),
                    col_fft.clone(),
                );
            }
        } else {
            workspace.serial(data, nx, ny, &row_fft, &col_fft);
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

        let mut workspace = self.workspace.lock().expect("FFT workspace poisoned");
        workspace.prepare(&self.plan_cache, nx, ny);
        let (row_fft, col_fft) = workspace.plans(direction);

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
                workspace.serial(buffer.as_mut_slice(), nx, ny, &row_fft, &col_fft);
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

#[derive(Clone, Copy)]
enum FftDirection {
    Forward,
    Inverse,
}

struct FftWorkspace<R: Real + FftNum> {
    shape: Option<(usize, usize)>,
    plans: Vec<Arc<dyn Fft<R>>>,
    column: Vec<Complex<R>>,
    algorithm: Vec<Complex<R>>,
    #[cfg(feature = "parallel")]
    transposed: Vec<Complex<R>>,
}

impl<R: Real + FftNum> Default for FftWorkspace<R> {
    fn default() -> Self {
        Self { shape: None, plans: Vec::new(), column: Vec::new(), algorithm: Vec::new(),
            #[cfg(feature = "parallel")] transposed: Vec::new() }
    }
}

impl<R: Real + FftNum> FftWorkspace<R> {
    fn prepare(&mut self, planner: &Mutex<FftPlanner<R>>, nx: usize, ny: usize) {
        if self.shape == Some((nx,ny)) { return; }
        blaze2d_core::profile_scope!("fft.planning");
        let mut planner = planner.lock().expect("FFT planner poisoned");
        self.plans = vec![planner.plan_fft_forward(nx), planner.plan_fft_forward(ny),
            planner.plan_fft_inverse(nx), planner.plan_fft_inverse(ny)];
        let required = self.plans.iter().map(|p|p.get_inplace_scratch_len()).max().unwrap_or(0);
        let zero = Complex::new(R::zero(), R::zero());
        self.algorithm.resize(required, zero);
        self.column.resize(ny, zero);
        self.shape = Some((nx,ny));
        blaze2d_core::profiler::count("fft.plan_sets", 1);
        blaze2d_core::profiler::count("fft.workspace.requested_bytes", (required+ny)*std::mem::size_of::<Complex<R>>());
    }

    fn plans(&self, direction: FftDirection) -> (Arc<dyn Fft<R>>, Arc<dyn Fft<R>>) {
        let offset = match direction { FftDirection::Forward=>0, FftDirection::Inverse=>2 };
        (self.plans[offset].clone(), self.plans[offset+1].clone())
    }

    fn serial(&mut self, data: &mut [Complex<R>], nx: usize, ny: usize,
        row: &Arc<dyn Fft<R>>, col: &Arc<dyn Fft<R>>) {
        // RustFFT accepts a contiguous batch of rows in one scratch-taking call.
        row.process_with_scratch(data, &mut self.algorithm);
        process_columns(data, nx, ny, col, &mut self.column, &mut self.algorithm);
        blaze2d_core::profiler::count("fft.1d_transforms", nx+ny);
        blaze2d_core::profiler::count("fft.1d_dispatches", 1+nx);
        blaze2d_core::profiler::count("fft.column_copy_bytes", 2*data.len()*std::mem::size_of::<Complex<R>>());
    }
}

#[cfg(feature = "parallel")]
fn process_rows_parallel<R: FftNum>(data: &mut [Complex<R>], nx: usize, fft: Arc<dyn Fft<R>>) {
    let zero = Complex::new(R::zero(), R::zero());
    data.par_chunks_mut(nx).for_each_init(||vec![zero;fft.get_inplace_scratch_len()], |scratch,row| {
        fft.process_with_scratch(row, scratch);
    });
}

fn process_columns<R: FftNum>(data: &mut [Complex<R>], nx: usize, ny: usize,
    fft: &Arc<dyn Fft<R>>, column: &mut [Complex<R>], algorithm: &mut [Complex<R>]) {
    let column = &mut column[..ny];
    for ix in 0..nx {
        for iy in 0..ny { column[iy] = data[iy*nx+ix]; }
        fft.process_with_scratch(column, algorithm);
        for iy in 0..ny { data[iy*nx+ix] = column[iy]; }
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
    let required = row_fft.get_inplace_scratch_len().max(col_fft.get_inplace_scratch_len());
    let mut job = || {
        buffers.par_iter_mut().for_each_init(||(vec![zero;ny],vec![zero;required]), |(column,algorithm),buffer| {
            row_fft.process_with_scratch(buffer.as_mut_slice(), algorithm);
            process_columns(buffer.as_mut_slice(), nx, ny, &col_fft, column, algorithm);
        });
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
        blaze2d_core::profiler::count("fft.forward.vectors", 1);
        blaze2d_core::profiler::count("fft.forward.elements", buffer.len());
        self.fft_2d(buffer, FftDirection::Forward);
    }

    fn inverse_fft_2d(&self, buffer: &mut Self::Buffer) {
        blaze2d_core::profiler::count("fft.inverse.vectors", 1);
        blaze2d_core::profiler::count("fft.inverse.elements", buffer.len());
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
        blaze2d_core::profiler::count("fft.forward.vectors", buffers.len());
        blaze2d_core::profiler::count("fft.forward.elements", buffers.iter().map(|b| b.len()).sum());
        self.batch_fft_2d(buffers, FftDirection::Forward);
    }

    fn batch_inverse_fft_2d(&self, buffers: &mut [Self::Buffer]) {
        blaze2d_core::profiler::count("fft.inverse.vectors", buffers.len());
        blaze2d_core::profiler::count("fft.inverse.elements", buffers.iter().map(|b| b.len()).sum());
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

#[test]
fn private_workspaces_handle_prime_rectangular_batches_in_both_precisions() {
    fn check<R: Real + FftNum>(tolerance: f64, backend: CpuBackend<R>) {
        let worker=backend.fork_worker();
        assert!(Arc::ptr_eq(&backend.plan_cache,&worker.plan_cache));
        assert!(!Arc::ptr_eq(&backend.workspace,&worker.workspace));
        for (nx,ny) in [(67,71),(8,5),(67,71)] {
            let grid=Grid2D::new(nx,ny,1.,1.);
            let mut x=Field2D::<R>::zeros(grid);
            for j in 0..ny {for i in 0..nx {
                let z=Complex64::from_polar(1.,std::f64::consts::TAU*(3.*i as f64/nx as f64+2.*j as f64/ny as f64));
                x.as_mut_slice()[grid.idx(i,j)]=cscalar(z.re,z.im);
            }}
            let original=x.clone();
            let mut batch=vec![x.clone(),x.clone()];
            backend.forward_fft_2d(&mut x);
            worker.batch_forward_fft_2d(&mut batch);
            let peak=x.as_slice()[grid.idx(3,2)];
            assert!((peak.re.to_accum()/grid.len() as f64-1.).abs()<tolerance);
            for field in &batch {for (a,b) in field.as_slice().iter().zip(x.as_slice()) {
                assert!((a.re.to_accum()-b.re.to_accum()).hypot(a.im.to_accum()-b.im.to_accum())<tolerance*grid.len() as f64);
            }}
            worker.batch_inverse_fft_2d(&mut batch);
            for field in &batch {for (a,b) in field.as_slice().iter().zip(original.as_slice()) {
                assert!((a.re.to_accum()-b.re.to_accum()).hypot(a.im.to_accum()-b.im.to_accum())<tolerance);
            }}
        }
    }
    check::<f64>(1e-11, CpuBackend::new());
    check::<f32>(2e-6, CpuBackend::new());
    #[cfg(feature="parallel")]
    {
        check::<f64>(1e-11, CpuBackend::new().with_parallel_fft(true).with_parallel_threads(2).with_parallel_threshold(1));
        check::<f32>(2e-6, CpuBackend::new().with_parallel_fft(true).with_parallel_threads(2).with_parallel_threshold(1));
    }
}
