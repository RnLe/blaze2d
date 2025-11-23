#![cfg(test)]

use crate::CpuBackend;
use mpb2d_core::backend::SpectralBackend;
use mpb2d_core::dielectric::{Dielectric2D, DielectricOptions};
use mpb2d_core::eigensolver::{
    EigenOptions, GammaContext, PowerIterationOptions, power_iteration, solve_lowest_eigenpairs,
};
use mpb2d_core::field::Field2D;
use mpb2d_core::geometry::Geometry2D;
use mpb2d_core::grid::Grid2D;
use mpb2d_core::lattice::Lattice2D;
use mpb2d_core::operator::{ThetaOperator, ToyLaplacian};
use mpb2d_core::polarization::Polarization;
use mpb2d_core::reference::load_reference_dataset;
use num_complex::Complex64;
use std::f64::consts::PI;
use std::path::PathBuf;

fn dedup_sorted(values: &[f64]) -> Vec<f64> {
    let mut uniq: Vec<f64> = Vec::new();
    for &val in values {
        if uniq
            .last()
            .map(|last| (val - *last).abs() > 1e-9)
            .unwrap_or(true)
        {
            uniq.push(val);
        }
    }
    uniq
}

fn reference_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("python/reference-data")
}

fn validate_uniform_reference(mpb_file: &str, polarization: Polarization) {
    let reference_dir = reference_dir();
    let mpb_path = reference_dir.join(mpb_file);
    let fallback_path = reference_dir.join("square_tm_uniform.json");
    if !mpb_path.exists() && !fallback_path.exists() {
        eprintln!(
            "skipping {mpb_file} regression test (no reference datasets under {:?})",
            reference_dir
        );
        return;
    }
    let reference = load_reference_dataset(&mpb_path)
        .or_else(|_| load_reference_dataset(&fallback_path))
        .unwrap_or_else(|_| panic!("failed to load reference dataset {mpb_file}"));
    assert_eq!(reference.bands.len(), reference.k_path.len());
    let lattice = Lattice2D::square(1.0);
    let geom = Geometry2D {
        lattice,
        eps_bg: 12.0,
        atoms: Vec::new(),
    };
    let grid = Grid2D::new(48, 48, 1.0, 1.0);
    let dielectric = Dielectric2D::from_geometry(&geom, grid, &DielectricOptions::default());
    let reference_unique = reference
        .bands
        .get(0)
        .map(|row| dedup_sorted(row))
        .unwrap_or_default();
    let eigen_opts = EigenOptions {
        n_bands: reference_unique.len() + 4,
        max_iter: 128,
        tol: 1e-8,
        ..Default::default()
    };
    let sample_indices: Vec<usize> = if reference.k_nodes.is_empty() {
        (0..reference.k_path.len()).step_by(8).collect()
    } else {
        reference.k_nodes.iter().map(|node| node.index).collect()
    };
    for &idx in &sample_indices {
        let kp = &reference.k_path[idx];
        let bloch = [2.0 * PI * kp.kx, 2.0 * PI * kp.ky];
        let bloch_norm = (bloch[0] * bloch[0] + bloch[1] * bloch[1]).sqrt();
        let gamma_context = GammaContext::new(eigen_opts.gamma.should_deflate(bloch_norm));
        let mut theta =
            ThetaOperator::new(CpuBackend::new(), dielectric.clone(), polarization, bloch);
        let result = solve_lowest_eigenpairs(
            &mut theta,
            &eigen_opts,
            None,
            gamma_context,
            None,
            None,
            None,
        );
        let expected = dedup_sorted(&reference.bands[idx]);
        let scaled_omegas: Vec<f64> = result
            .omegas
            .iter()
            .map(|&omega| omega / (2.0 * PI))
            .collect();
        let actual = dedup_sorted(&scaled_omegas);
        assert!(
            expected.len() <= actual.len(),
            "insufficient eigenpairs: have {}, need {}",
            actual.len(),
            expected.len()
        );
        let mut used = vec![false; actual.len()];
        for (band_idx, &target) in expected.iter().enumerate() {
            if target < 1e-6 {
                if gamma_context.is_gamma && result.gamma_deflated {
                    continue;
                }
                if let Some((found_idx, _omega)) = actual
                    .iter()
                    .enumerate()
                    .find(|&(i, omega)| !used[i] && *omega < 1e-3)
                {
                    used[found_idx] = true;
                    continue;
                }
                panic!("k#{idx} band {band_idx} missing near-zero mode");
            }
            let mut best: Option<(usize, f64, f64)> = None;
            for (cand_idx, &omega) in actual.iter().enumerate() {
                if used[cand_idx] {
                    continue;
                }
                let rel_err = ((omega - target) / target).abs();
                if best
                    .as_ref()
                    .map(|(_, _, best_rel)| rel_err < *best_rel)
                    .unwrap_or(true)
                {
                    best = Some((cand_idx, omega, rel_err));
                }
            }
            match best {
                Some((cand_idx, omega, rel_err)) => {
                    used[cand_idx] = true;
                    assert!(
                        rel_err < 5e-2,
                        "k#{idx} band {band_idx} mismatch: got {omega}, target {target}, rel {rel_err}"
                    );
                }
                None => panic!("k#{idx} band {band_idx} missing candidate"),
            }
        }
    }
}

#[test]
fn fft_roundtrip_recovers_signal() {
    let backend = CpuBackend::new();
    let grid = Grid2D::new(4, 4, 1.0, 1.0);
    let mut field = Field2D::zeros(grid);
    for (idx, value) in field.as_mut_slice().iter_mut().enumerate() {
        *value = Complex64::new(idx as f64, -(idx as f64));
    }
    let original = field.clone();
    backend.forward_fft_2d(&mut field);
    backend.inverse_fft_2d(&mut field);
    for (rec, expect) in field.as_slice().iter().zip(original.as_slice()) {
        let diff = (*rec - *expect).norm();
        assert!(diff < 1e-9, "FFT roundtrip diverged: diff={diff}");
    }
}

#[test]
fn fft_roundtrip_preserves_energy_norm() {
    let backend = CpuBackend::new();
    let grid = Grid2D::new(6, 2, 1.0, 1.0);
    let mut field = Field2D::zeros(grid);
    for (idx, value) in field.as_mut_slice().iter_mut().enumerate() {
        *value = Complex64::new((idx as f64).sin(), (idx as f64).cos());
    }
    let before = field.as_slice().iter().map(|v| v.norm_sqr()).sum::<f64>();
    backend.forward_fft_2d(&mut field);
    backend.inverse_fft_2d(&mut field);
    let after = field.as_slice().iter().map(|v| v.norm_sqr()).sum::<f64>();
    assert!(
        (before - after).abs() < 1e-9,
        "energy drifted by {}",
        after - before
    );
}

#[test]
fn axpy_and_dot_behave() {
    let backend = CpuBackend::new();
    let grid = Grid2D::new(2, 2, 1.0, 1.0);
    let mut x = Field2D::zeros(grid);
    let mut y = Field2D::zeros(grid);
    for (i, value) in x.as_mut_slice().iter_mut().enumerate() {
        *value = Complex64::new(i as f64 + 1.0, 0.0);
    }
    backend.axpy(Complex64::new(2.0, 0.0), &x, &mut y);
    let expected: Vec<Complex64> = x
        .as_slice()
        .iter()
        .map(|v| Complex64::new(2.0, 0.0) * v)
        .collect();
    for (actual, expect) in y.as_slice().iter().zip(expected.iter()) {
        assert!((*actual - *expect).norm() < 1e-9);
    }
    let dot = backend.dot(&x, &x);
    assert!((dot - Complex64::new(30.0, 0.0)).norm() < 1e-9);
}

#[test]
fn toy_laplacian_matches_plane_wave_eigenvalue() {
    let backend = CpuBackend::new();
    let grid = Grid2D::new(8, 8, 1.0, 1.0);
    let mut laplacian = ToyLaplacian::new(backend, grid);
    let mut vec = laplacian.alloc_field();
    let nx = grid.nx;
    let ny = grid.ny;
    for iy in 0..ny {
        for ix in 0..nx {
            let idx = iy * nx + ix;
            let x = ix as f64 / nx as f64;
            vec.as_mut_slice()[idx] = Complex64::from_polar(1.0, 2.0 * PI * x);
        }
    }
    let opts = PowerIterationOptions {
        max_iter: 32,
        tol: 1e-10,
    };
    let eig = power_iteration(&mut laplacian, &mut vec, &opts);
    let expected = (2.0 * PI).powi(2);
    assert!(
        (eig - expected).abs() < 1e-6,
        "expected {expected}, got {eig}"
    );
}

#[test]
fn gamma_deflation_removes_constant_mode() {
    let backend = CpuBackend::new();
    let grid = Grid2D::new(8, 8, 1.0, 1.0);
    let mut laplacian = ToyLaplacian::new(backend.clone(), grid);
    let opts = EigenOptions {
        n_bands: 2,
        max_iter: 120,
        tol: 1e-10,
        ..Default::default()
    };
    let baseline = solve_lowest_eigenpairs(
        &mut laplacian,
        &opts,
        None,
        GammaContext::default(),
        None,
        None,
        None,
    );
    assert!(
        baseline.omegas[0] < 5e-5,
        "should capture zero band without deflation (got {:.3e})",
        baseline.omegas[0]
    );

    let mut laplacian = ToyLaplacian::new(backend, grid);
    let deflated = solve_lowest_eigenpairs(
        &mut laplacian,
        &opts,
        None,
        GammaContext::new(true),
        None,
        None,
        None,
    );
    assert!(
        deflated.gamma_deflated,
        "deflation flag should be propagated"
    );
    assert!(deflated.omegas[0] > 1e-3, "constant band must be removed");
    let expected = 2.0 * PI;
    assert!(
        (deflated.omegas[0] - expected).abs() < 1e-2,
        "expected {expected}, got {}",
        deflated.omegas[0]
    );
}

#[test]
fn tm_operator_matches_uniform_medium_limit() {
    let backend = CpuBackend::new();
    let lattice = Lattice2D::square(1.0);
    let geom = Geometry2D {
        lattice,
        eps_bg: 12.0,
        atoms: Vec::new(),
    };
    let grid = Grid2D::new(8, 8, 1.0, 1.0);
    let dielectric = Dielectric2D::from_geometry(&geom, grid, &DielectricOptions::default());
    let mut theta = ThetaOperator::new(backend, dielectric, Polarization::TM, [0.0, 0.0]);
    let mut vec = theta.alloc_field();
    let nx = grid.nx;
    let ny = grid.ny;
    for iy in 0..ny {
        for ix in 0..nx {
            let idx = iy * nx + ix;
            let x = ix as f64 / nx as f64;
            vec.as_mut_slice()[idx] = Complex64::from_polar(1.0, 2.0 * PI * x);
        }
    }
    let opts = PowerIterationOptions {
        max_iter: 40,
        tol: 1e-10,
    };
    let eig = power_iteration(&mut theta, &mut vec, &opts);
    let expected = (2.0 * PI).powi(2) / geom.eps_bg;
    assert!(
        (eig - expected).abs() < 1e-5,
        "expected {expected}, got {eig}"
    );
}

#[test]
fn tm_operator_tracks_uniform_reference_data() {
    validate_uniform_reference("square_tm_uniform_mpb.json", Polarization::TM);
}

#[test]
fn te_operator_tracks_uniform_reference_data() {
    validate_uniform_reference("square_te_uniform_mpb.json", Polarization::TE);
}
