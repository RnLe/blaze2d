#![cfg(test)]

use std::f64::consts::PI;

use num_complex::Complex64;

use super::backend::SpectralBackend;
use super::bandstructure::{BandStructureJob, Verbosity, run};
#[cfg(test)]
use super::bandstructure::{RunDebugProbe, run_with_debug};
use super::eigensolver::{EigenOptions, PreconditionerKind};
use super::field::Field2D;
use super::geometry::Geometry2D;
use super::grid::Grid2D;
use super::lattice::Lattice2D;
use super::polarization::Polarization;

#[derive(Clone)]
struct DeterministicBackend;

impl SpectralBackend for DeterministicBackend {
    type Buffer = Field2D;

    fn alloc_field(&self, grid: Grid2D) -> Self::Buffer {
        Field2D::zeros(grid)
    }

    fn forward_fft_2d(&self, buffer: &mut Self::Buffer) {
        discrete_fft(buffer, false);
    }

    fn inverse_fft_2d(&self, buffer: &mut Self::Buffer) {
        discrete_fft(buffer, true);
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

fn discrete_fft(buffer: &mut Field2D, inverse: bool) {
    let grid = buffer.grid();
    let nx = grid.nx;
    let ny = grid.ny;
    let data = buffer.as_mut_slice();
    let mut output = vec![Complex64::default(); data.len()];
    let norm = if inverse { 1.0 / (nx * ny) as f64 } else { 1.0 };
    for ky in 0..ny {
        for kx in 0..nx {
            let mut sum = Complex64::default();
            for y in 0..ny {
                for x in 0..nx {
                    let idx = y * nx + x;
                    let phase = if inverse {
                        2.0 * PI * ((kx * x) as f64 / nx as f64 + (ky * y) as f64 / ny as f64)
                    } else {
                        -2.0 * PI * ((kx * x) as f64 / nx as f64 + (ky * y) as f64 / ny as f64)
                    };
                    sum += data[idx] * Complex64::from_polar(1.0, phase);
                }
            }
            output[ky * nx + kx] = sum * norm;
        }
    }
    data.copy_from_slice(&output);
}

fn uniform_geometry() -> Geometry2D {
    Geometry2D {
        lattice: Lattice2D::square(1.0),
        eps_bg: 8.0,
        atoms: Vec::new(),
    }
}

fn small_job(k_path: Vec<[f64; 2]>) -> BandStructureJob {
    BandStructureJob {
        geom: uniform_geometry(),
        grid: Grid2D::new(2, 2, 1.0, 1.0),
        pol: Polarization::TE,
        k_path,
        eigensolver: EigenOptions {
            n_bands: 1,
            max_iter: 16,
            tol: 1e-8,
            ..Default::default()
        },
    }
}

#[test]
fn run_collects_bands_and_distances_with_deterministic_backend() {
    let k_path = vec![[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]];
    let backend = DeterministicBackend;
    let job = small_job(k_path.clone());
    let result = run(backend, &job, Verbosity::Quiet);
    assert_eq!(result.k_path, job.k_path);
    assert_eq!(result.bands.len(), job.k_path.len());
    assert_eq!(result.distances.len(), job.k_path.len());
    assert!(
        result
            .bands
            .iter()
            .all(|band| band.len() == job.eigensolver.n_bands)
    );
    assert!(
        result
            .bands
            .iter()
            .all(|band| band.iter().all(|omega| omega.is_finite() && *omega >= 0.0))
    );
    assert!((result.distances[1] - 0.5).abs() < 1e-12);
    assert!((result.distances[2] - 1.0).abs() < 1e-12);
}

#[test]
fn run_handles_empty_k_path() {
    let backend = DeterministicBackend;
    let job = small_job(Vec::new());
    let result = run(backend, &job, Verbosity::Quiet);
    assert!(result.k_path.is_empty());
    assert!(result.bands.is_empty());
    assert!(result.distances.is_empty());
}

#[test]
fn verbose_run_matches_quiet_output() {
    let backend = DeterministicBackend;
    let k_path = vec![[0.0, 0.0], [0.25, 0.0]];
    let job = small_job(k_path);
    let quiet = run(backend.clone(), &job, Verbosity::Quiet);
    let verbose = run(backend, &job, Verbosity::Verbose);
    assert_eq!(quiet.k_path, verbose.k_path);
    assert_eq!(quiet.distances, verbose.distances);
    assert_eq!(quiet.bands, verbose.bands);
}

#[test]
fn run_engages_warm_start_and_theta_cache() {
    let backend = DeterministicBackend;
    let mut job = small_job(vec![[0.0, 0.0], [0.0, 0.0], [0.25, 0.25]]);
    job.eigensolver.preconditioner = PreconditionerKind::FourierDiagonal;
    job.eigensolver.deflation.enabled = true;
    job.eigensolver.deflation.max_vectors = 2;
    job.eigensolver.warm_start.enabled = true;
    job.eigensolver.warm_start.max_vectors = 2;
    job.eigensolver.gamma.enabled = true;
    let mut probe = RunDebugProbe::default();
    let result = run_with_debug(backend, &job, Verbosity::Quiet, &mut probe);
    assert_eq!(result.k_path.len(), job.k_path.len());
    assert_eq!(result.bands.len(), job.k_path.len());
    assert_eq!(probe.theta_instances, job.k_path.len());
    assert_eq!(probe.warm_start_hits, job.k_path.len() - 1);
}
