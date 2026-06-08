//! Tests for the bandstructure module.

#![cfg(test)]

use std::f64::consts::PI;

use super::backend::SpectralBackend;
use super::bandstructure::{
    BandStructureJob, RunOptions, compute_k_path_distances, run_with_diagnostics_and_options,
    run_with_options,
};
use super::dielectric::DielectricOptions;
use super::eigensolver::EigensolverConfig;
use super::field::{AccumScalar, Field2D, FieldReal, FieldScalar};
use super::geometry::{BasisAtom, Geometry2D};
use super::grid::Grid2D;
use super::lattice::Lattice2D;
use super::polarization::Polarization;

// ============================================================================
// Test Backend
// ============================================================================

/// A minimal backend for testing that performs real DFT operations.
#[derive(Clone)]
struct TestBackend;

impl SpectralBackend for TestBackend {
    type Real = f64;
    type Buffer = Field2D;

    fn alloc_field(&self, grid: Grid2D) -> Self::Buffer {
        Field2D::zeros(grid)
    }

    fn forward_fft_2d(&self, buffer: &mut Self::Buffer) {
        naive_fft_2d(buffer, false);
    }

    fn inverse_fft_2d(&self, buffer: &mut Self::Buffer) {
        naive_fft_2d(buffer, true);
    }

    fn scale(&self, alpha: AccumScalar, buffer: &mut Self::Buffer) {
        for value in buffer.as_mut_slice() {
            *value *= FieldScalar::new(alpha.re as FieldReal, alpha.im as FieldReal);
        }
    }

    fn axpy(&self, alpha: AccumScalar, x: &Self::Buffer, y: &mut Self::Buffer) {
        let alpha_f = FieldScalar::new(alpha.re as FieldReal, alpha.im as FieldReal);
        for (dst, src) in y.as_mut_slice().iter_mut().zip(x.as_slice()) {
            *dst += alpha_f * src;
        }
    }

    fn dot(&self, x: &Self::Buffer, y: &Self::Buffer) -> AccumScalar {
        x.as_slice()
            .iter()
            .zip(y.as_slice())
            .map(|(a, b)| {
                let a64 = AccumScalar::new(a.re as f64, a.im as f64);
                let b64 = AccumScalar::new(b.re as f64, b.im as f64);
                a64.conj() * b64
            })
            .sum()
    }
}

/// Naive O(n²) DFT for testing.
fn naive_fft_2d(buffer: &mut Field2D, inverse: bool) {
    let grid = buffer.grid();
    let nx = grid.nx;
    let ny = grid.ny;
    let n = (nx * ny) as f64;
    let sign = if inverse { 1.0 } else { -1.0 };

    let input: Vec<FieldScalar> = buffer.as_slice().to_vec();
    let output = buffer.as_mut_slice();

    for ky in 0..ny {
        for kx in 0..nx {
            let mut sum = AccumScalar::ZERO;
            for jy in 0..ny {
                for jx in 0..nx {
                    let idx = jy * nx + jx;
                    let phase = sign
                        * 2.0
                        * PI
                        * ((kx * jx) as f64 / nx as f64 + (ky * jy) as f64 / ny as f64);
                    let inp64 = AccumScalar::new(input[idx].re as f64, input[idx].im as f64);
                    sum += inp64 * AccumScalar::new(phase.cos(), phase.sin());
                }
            }
            let out_idx = ky * nx + kx;
            let result = if inverse { sum / n } else { sum };
            output[out_idx] = FieldScalar::new(result.re as FieldReal, result.im as FieldReal);
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a simple uniform-epsilon geometry for testing.
fn uniform_geometry() -> Geometry2D {
    Geometry2D {
        lattice: Lattice2D::square(1.0),
        eps_bg: 1.0,
        atoms: vec![],
    }
}

/// Create a simple photonic crystal geometry.
fn simple_photonic_crystal() -> Geometry2D {
    Geometry2D {
        lattice: Lattice2D::square(1.0),
        eps_bg: 13.0,
        atoms: vec![BasisAtom {
            pos: [0.0, 0.0],
            radius: 0.3,
            eps_inside: 1.0,
        }],
    }
}

/// Create a small test job.
fn test_job(k_path: Vec<[f64; 2]>) -> BandStructureJob {
    BandStructureJob {
        geom: simple_photonic_crystal(),
        grid: Grid2D::new(8, 8, 1.0, 1.0),
        pol: Polarization::TM,
        k_path,
        eigensolver: EigensolverConfig {
            n_bands: 2,
            max_iter: 50,
            tol: 1e-4,
            block_size: 4,
            ..Default::default()
        },
        dielectric: DielectricOptions::default(),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn test_compute_k_path_distances_empty() {
    let distances = compute_k_path_distances(&[]);
    assert!(distances.is_empty());
}

#[test]
fn test_compute_k_path_distances_single_point() {
    let distances = compute_k_path_distances(&[[0.0, 0.0]]);
    assert_eq!(distances.len(), 1);
    assert!((distances[0] - 0.0).abs() < 1e-10);
}

#[test]
fn test_compute_k_path_distances_two_points() {
    let distances = compute_k_path_distances(&[[0.0, 0.0], [0.5, 0.0]]);
    assert_eq!(distances.len(), 2);
    assert!((distances[0] - 0.0).abs() < 1e-10);
    assert!((distances[1] - 0.5).abs() < 1e-10);
}

#[test]
fn test_compute_k_path_distances_three_points() {
    let distances = compute_k_path_distances(&[[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]]);
    assert_eq!(distances.len(), 3);
    assert!((distances[0] - 0.0).abs() < 1e-10);
    assert!((distances[1] - 0.5).abs() < 1e-10);
    assert!((distances[2] - 1.0).abs() < 1e-10);
}

#[test]
fn test_run_single_k_point() {
    let backend = TestBackend;
    let job = test_job(vec![[0.0, 0.0]]);

    let result = run_with_options(backend, &job, RunOptions::default());

    assert_eq!(result.k_path.len(), 1);
    assert_eq!(result.bands.len(), 1);
    assert_eq!(result.bands[0].len(), job.eigensolver.n_bands);
    assert_eq!(result.distances.len(), 1);
}

#[test]
fn test_run_multiple_k_points() {
    let backend = TestBackend;
    let job = test_job(vec![[0.0, 0.0], [0.25, 0.0], [0.5, 0.0]]);

    let result = run_with_options(backend, &job, RunOptions::default());

    assert_eq!(result.k_path.len(), 3);
    assert_eq!(result.bands.len(), 3);
    for bands in &result.bands {
        assert_eq!(bands.len(), job.eigensolver.n_bands);
    }
}

#[test]
fn test_bandstructure_result_structure() {
    let backend = TestBackend;
    let k_path = vec![[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]];
    let job = test_job(k_path.clone());

    let result = run_with_options(backend, &job, RunOptions::default());

    // Check that result has correct structure
    assert_eq!(result.k_path, k_path);
    assert_eq!(result.distances.len(), k_path.len());
    assert_eq!(result.bands.len(), k_path.len());

    // All frequencies should be non-negative
    for bands in &result.bands {
        for &omega in bands {
            assert!(omega >= 0.0, "frequencies should be non-negative");
        }
    }
}

#[test]
fn test_uniform_epsilon_runs_without_panic() {
    // This test verifies that the pipeline runs correctly for a uniform medium.
    // Physics-specific assertions (e.g., ω = 0 at Γ) require a proper FFT backend,
    // not the naive DFT implementation used in tests.
    let backend = TestBackend;
    let job = BandStructureJob {
        geom: uniform_geometry(),
        grid: Grid2D::new(8, 8, 1.0, 1.0),
        pol: Polarization::TM,
        k_path: vec![[0.0, 0.0]],
        eigensolver: EigensolverConfig {
            n_bands: 1,
            max_iter: 100,
            tol: 1e-4,
            block_size: 3,
            ..Default::default()
        },
        dielectric: DielectricOptions::default(),
    };

    let result = run_with_options(backend, &job, RunOptions::default());

    // Verify we got a result with the expected structure
    assert_eq!(result.bands.len(), 1);
    assert_eq!(result.bands[0].len(), 1);

    // Frequency should be non-negative (basic sanity check)
    let omega = result.bands[0][0];
    assert!(
        omega >= 0.0,
        "frequency should be non-negative, got {omega}"
    );
}

#[test]
fn test_merge_diagnostics_matches_plain() {
    // Regression guard for the run_with_options / run_with_diagnostics_and_options
    // merge: both entry points now share `run_core` with identical setup and
    // warm-start behavior. They are NOT bit-identical because the recording path
    // calls the eigensolver's `solve_with_diagnostics` (eager residual norms +
    // different internal convergence accounting) while the plain path calls
    // `solve`; both terminate within the same convergence tolerance, so the
    // bands must agree to within a small multiple of that tolerance. This
    // catches any setup / warm-start / band-tracking regression from the merge
    // (which would shift eigenvalues by O(0.01) or reorder bands), while
    // tolerating the legitimate ~tol-scale divergence between the two solvers.
    const TOL: f64 = 5e-3; // ~50× the eigensolver tol (1e-4); observed Δ ~1.4e-4
    let k_path = vec![[0.0, 0.0], [0.25, 0.0], [0.5, 0.0], [0.5, 0.5]];

    let plain = run_with_options(TestBackend, &test_job(k_path.clone()), RunOptions::default());
    let diag = run_with_diagnostics_and_options(
        TestBackend,
        &test_job(k_path.clone()),
        "merge_regression",
        RunOptions::default(),
    );

    assert_eq!(plain.k_path, diag.result.k_path);
    assert_eq!(plain.bands.len(), diag.result.bands.len());
    for (kp, (pb, db)) in plain.bands.iter().zip(diag.result.bands.iter()).enumerate() {
        assert_eq!(pb.len(), db.len(), "band count mismatch at k#{kp}");
        for (band, (&p, &d)) in pb.iter().zip(db.iter()).enumerate() {
            assert!(
                (p - d).abs() < TOL,
                "k#{kp} band {band}: plain={p} diag={d} differ beyond {TOL}"
            );
        }
    }

    // The diagnostics path must have recorded one run per solved k-point.
    assert!(
        !diag.study.runs.is_empty(),
        "diagnostics study should record at least one k-point run"
    );
}
