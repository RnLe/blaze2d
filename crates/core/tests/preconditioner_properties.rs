use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::Dielectric2D,
    eigensolver::{Eigensolver, EigensolverConfig, HistoryRule, StoppingRule, refine::certify_without_refinement, solve_hermitian_eigen},
    field::Field2D,
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    operators::{LinearOperator, ThetaOperator},
    polarization::Polarization,
    preconditioners::{OperatorPreconditioner, PreconditionerKind, PreconditionerSpec, ShiftRule, TransverseWeight},
    backend::SpectralBackend,
};
use num_complex::Complex64 as C;

type B = CpuBackend<f64>;

fn operator(lattice: Lattice2D, n: usize, pol: Polarization, k: [f64; 2], eps: (f64, f64)) -> ThetaOperator<B> {
    let grid = Grid2D::new(n, n, 1.0, 1.0);
    let dielectric = Dielectric2D::from_geometry(
        &Geometry2D { lattice, eps_bg: eps.1, atoms: vec![BasisAtom { pos: [0.0, 0.0], radius: 0.3, eps_inside: eps.0 }] },
        grid,
        &Default::default(),
    );
    ThetaOperator::new(B::new(), dielectric, pol, lattice.reciprocal().fractional_to_cartesian(k))
}

fn probe(op: &ThetaOperator<B>) -> Field2D {
    let mut v = op.alloc_field();
    for (i, z) in v.as_mut_slice().iter_mut().enumerate() {
        *z = C::new((i as f64 * 0.723).sin(), (i as f64 * 1.31).cos());
    }
    v
}

fn dense(op: &ThetaOperator<B>, p: &mut dyn OperatorPreconditioner<B>) -> (Vec<C>, usize) {
    let n = op.grid().len();
    let mut matrix = vec![C::new(0.0, 0.0); n * n];
    for j in 0..n {
        let mut v = op.alloc_field();
        v.as_mut_slice()[j] = C::new(1.0, 0.0);
        p.apply(op.backend(), &mut v);
        for (i, z) in v.as_slice().iter().enumerate() {
            matrix[i + n * j] = *z;
        }
    }
    (matrix, n)
}

#[test]
fn transverse_projection_is_hermitian_positive_semidefinite() {
    let spec = PreconditionerSpec { kind: PreconditionerKind::Transverse, ..Default::default() };
    for lattice in [Lattice2D::square(1.0), Lattice2D::hexagonal(1.0)] {
        for k in [[0.0, 0.0], [0.13, 0.21], [0.5, 0.0]] {
            let op = operator(lattice, 10, Polarization::TE, k, (1.0, 13.0));
            let (mut p, _) = op.build_preconditioner(&spec, None);
            let (matrix, n) = dense(&op, p.as_deref_mut().unwrap());
            let scale = matrix.iter().map(|z| z.norm()).fold(0.0, f64::max);
            for i in 0..n {
                for j in 0..n {
                    assert!((matrix[i + n * j] - matrix[j + n * i].conj()).norm() <= 1e-12 * scale);
                }
            }
            let spectrum = solve_hermitian_eigen(&matrix, n);
            let lowest = spectrum.eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
            assert!(lowest >= -1e-12 * scale, "lowest eigenvalue {lowest} at k={k:?}");
        }
    }
}

#[test]
fn factory_reproduces_legacy_builders() {
    let previous = [0.0, 0.31, 0.47, 0.52, 0.9];
    for pol in [Polarization::TM, Polarization::TE] {
        for k in [[0.0, 0.0], [0.13, 0.21]] {
            let op = operator(Lattice2D::square(1.0), 12, pol, k, (8.9, 1.0));
            let input = probe(&op);
            let legacy: Vec<Box<dyn OperatorPreconditioner<B>>> = match pol {
                Polarization::TM => vec![
                    Box::new(op.build_homogeneous_preconditioner_adaptive()),
                    Box::new(op.build_homogeneous_preconditioner_band_window(&previous, Some(0.5), Some(0.5))),
                ],
                Polarization::TE => vec![
                    Box::new(op.build_transverse_projection_preconditioner_adaptive()),
                    Box::new(op.build_transverse_projection_preconditioner_band_window(&previous, Some(0.5), Some(0.5))),
                ],
            };
            let specs = [PreconditionerSpec::legacy(), PreconditionerSpec { shift: ShiftRule::BandWindow, ..PreconditionerSpec::legacy() }];
            for (mut old, spec) in legacy.into_iter().zip(specs) {
                let (new, _) = op.build_preconditioner(&spec, Some(&previous));
                let mut new = new.unwrap();
                let (mut a, mut b) = (input.clone(), input.clone());
                old.apply(op.backend(), &mut a);
                new.apply(op.backend(), &mut b);
                assert_eq!(a.as_slice(), b.as_slice());
            }
        }
    }
}

#[test]
fn residual_stopping_certifies_fresh_residuals() {
    let tol = 1e-8;
    for pol in [Polarization::TM, Polarization::TE] {
        for history in [HistoryRule::Ritz, HistoryRule::Conjugate, HistoryRule::None] {
            for k in [[0.0, 0.0], [0.13, 0.21]] {
                let mut op = operator(Lattice2D::square(1.0), 16, pol, k, (8.9, 1.0));
                let (mut p, _): (Option<Box<dyn OperatorPreconditioner<B>>>, _) =
                    op.build_preconditioner(&PreconditionerSpec::default(), None);
                let config = EigensolverConfig {
                    n_bands: 6, max_iter: 400, stopping: StoppingRule::Residual, residual_tol: tol, history,
                    trace: true, ..Default::default()
                };
                let mut solver = Eigensolver::new(&mut op, config,
                    p.as_mut().map(|p| &mut **p as &mut dyn OperatorPreconditioner<B>), None);
                let result = solver.solve();
                let fields = solver.all_eigenvectors();
                drop(solver);
                assert!(result.converged, "{pol:?} {history:?} {k:?}");
                assert_eq!(result.trace.len(), result.iterations);
                let certificate = certify_without_refinement(&mut op, &result.eigenvalues, &fields);
                for (i, (&eta, &value)) in certificate.residuals.iter().zip(&result.eigenvalues).enumerate() {
                    if value == 0.0 {
                        continue;
                    }
                    assert!(eta <= 1.5 * tol, "{pol:?} {history:?} {k:?} band {i}: {eta:e}");
                }
            }
        }
    }
}

fn dense_operator(op: &mut ThetaOperator<B>, mass: bool) -> Vec<C> {
    let n = op.grid().len();
    let mut matrix = vec![C::new(0.0, 0.0); n * n];
    for j in 0..n {
        let mut v = op.alloc_field();
        v.as_mut_slice()[j] = C::new(1.0, 0.0);
        let mut out = op.alloc_field();
        if mass { op.apply_mass(&v, &mut out) } else { op.apply(&v, &mut out) }
        for (i, z) in out.as_slice().iter().enumerate() {
            matrix[i + n * j] = *z;
        }
    }
    matrix
}

fn assert_hermitian_psd(matrix: &[C], n: usize, what: &str) {
    let scale = matrix.iter().map(|z| z.norm()).fold(0.0, f64::max);
    for i in 0..n {
        for j in 0..n {
            assert!((matrix[i + n * j] - matrix[j + n * i].conj()).norm() <= 1e-12 * scale, "{what}: not Hermitian");
        }
    }
    let lowest = solve_hermitian_eigen(matrix, n).eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(lowest >= -1e-11 * scale, "{what}: lowest eigenvalue {lowest}");
}

#[test]
fn plane_wave_galerkin_equals_the_dense_restriction_and_the_correction_is_positive() {
    for lattice in [Lattice2D::square(1.0), Lattice2D::hexagonal(1.0)] {
        for pol in [Polarization::TE, Polarization::TM] {
            for k in [[0.0, 0.0], [0.13, 0.21], [0.5, 0.0]] {
                let mut op = operator(lattice, 10, pol, k, (1.0, 13.0));
                let n = op.grid().len();
                for sigma in [0.0, 0.7] {
                    let (indices, galerkin) = op.plane_wave_galerkin(9, sigma, 0.0);
                    let m = indices.len();
                    assert!(m >= 9 && m < n);
                    // Orthonormal plane waves in the solver's spatial ordering.
                    let columns: Vec<Field2D> = indices.iter().map(|&i| {
                        let mut z = op.alloc_field();
                        z.as_mut_slice()[i] = C::new((n as f64).sqrt(), 0.0);
                        op.backend().inverse_fft_2d(&mut z);
                        z
                    }).collect();
                    let (a, b) = (dense_operator(&mut op, false), dense_operator(&mut op, true));
                    let scale = galerkin.iter().map(|z| z.norm()).fold(0.0, f64::max);
                    for (r, zr) in columns.iter().enumerate() {
                        for (c, zc) in columns.iter().enumerate() {
                            let mut value = C::new(0.0, 0.0);
                            for i in 0..n {
                                let mut kz = C::new(0.0, 0.0);
                                for j in 0..n {
                                    kz += (a[i + n * j] + b[i + n * j] * sigma) * zc.as_slice()[j];
                                }
                                value += zr.as_slice()[i].conj() * kz;
                            }
                            assert!((value - galerkin[r * m + c]).norm() <= 1e-11 * scale,
                                "{pol:?} k={k:?} sigma={sigma}: entry ({r},{c}) {value} vs {}", galerkin[r * m + c]);
                        }
                    }
                }
                for spec in [
                    PreconditionerSpec { coarse_modes: 9, shift: ShiftRule::Continuous, shift_fraction: 0.0, ..Default::default() },
                    PreconditionerSpec { coarse_modes: 9, transverse_weight: TransverseWeight::Tensor, ..Default::default() },
                ] {
                    let (mut p, info) = op.build_preconditioner(&spec, None);
                    assert!(info.coarse_modes >= 9);
                    let (matrix, n) = dense(&op, p.as_deref_mut().unwrap());
                    assert_hermitian_psd(&matrix, n, &format!("two-level {pol:?} k={k:?}"));
                }
            }
        }
    }
}

#[test]
fn continuous_shift_vanishes_at_gamma_and_tensor_projection_stays_positive() {
    let spec = PreconditionerSpec { shift: ShiftRule::Continuous, shift_fraction: 0.5, ..Default::default() };
    for pol in [Polarization::TE, Polarization::TM] {
        let shifts: Vec<f64> = [[0.0, 0.0], [1e-4, 0.0], [1e-2, 0.0]].iter()
            .map(|&k| operator(Lattice2D::square(1.0), 10, pol, k, (1.0, 13.0)).build_preconditioner(&spec, None).1.shift).collect();
        assert_eq!(shifts[0], 0.0);
        assert!(shifts[1] < 1e-6 && shifts[1] < shifts[2], "{pol:?} {shifts:?}");
        let legacy = operator(Lattice2D::square(1.0), 10, pol, [0.0, 0.0], (1.0, 13.0))
            .build_preconditioner(&PreconditionerSpec::legacy(), None).1.shift;
        assert!(legacy > 1.0, "the legacy rule jumps at Gamma");
    }
    let tensor = PreconditionerSpec { kind: PreconditionerKind::Transverse, transverse_weight: TransverseWeight::Tensor, ..Default::default() };
    for k in [[0.0, 0.0], [0.13, 0.21]] {
        let op = operator(Lattice2D::hexagonal(1.0), 10, Polarization::TE, k, (1.0, 13.0));
        let (mut p, _) = op.build_preconditioner(&tensor, None);
        let (matrix, n) = dense(&op, p.as_deref_mut().unwrap());
        assert_hermitian_psd(&matrix, n, "tensor transverse projection");
    }
}

#[test]
fn two_level_continuous_preconditioning_certifies_fresh_residuals() {
    let tol = 1e-8;
    let spec = PreconditionerSpec { shift: ShiftRule::Continuous, shift_fraction: 0.0, coarse_modes: 32, ..Default::default() };
    for pol in [Polarization::TM, Polarization::TE] {
        for k in [[0.0, 0.0], [0.13, 0.21]] {
            let mut op = operator(Lattice2D::square(1.0), 16, pol, k, (8.9, 1.0));
            let (mut p, info) = op.build_preconditioner(&spec, None);
            assert!(info.coarse_modes >= 32);
            let config = EigensolverConfig {
                n_bands: 8, max_iter: 400, stopping: StoppingRule::Residual, residual_tol: tol, ..Default::default()
            };
            let mut solver = Eigensolver::new(&mut op, config,
                p.as_mut().map(|p| &mut **p as &mut dyn OperatorPreconditioner<B>), None);
            let result = solver.solve();
            let fields = solver.all_eigenvectors();
            drop(solver);
            assert!(result.converged, "{pol:?} {k:?}");
            let certificate = certify_without_refinement(&mut op, &result.eigenvalues, &fields);
            for (&eta, &value) in certificate.residuals.iter().zip(&result.eigenvalues) {
                if value != 0.0 { assert!(eta <= 1.5 * tol, "{pol:?} {k:?}: {eta:e}"); }
            }
        }
    }
}
