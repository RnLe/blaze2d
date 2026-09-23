use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::Dielectric2D,
    eigensolver::EigensolverConfig,
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    operators::{LinearOperator, ThetaOperator},
    polarization::Polarization,
    symmetry::{
        galerkin::{
            GalerkinApplication as Application, GalerkinBudget, GalerkinOperator, dense_pencil,
            estimated_payload_bytes, solve_dense, solve_iterative,
        },
        support::FourierSupport,
    },
};
use num_complex::Complex64 as C;
use std::f64::consts::TAU;

fn rod(lattice: Lattice2D, grid: Grid2D) -> Dielectric2D {
    Dielectric2D::from_geometry(
        &Geometry2D {
            lattice,
            eps_bg: 1.0,
            atoms: vec![BasisAtom {
                pos: [0.0; 2],
                radius: 0.2,
                eps_inside: 8.9,
            }],
        },
        grid,
        &Default::default(),
    )
}
fn support(lattice: Lattice2D, grid: Grid2D, k: [f64; 2], radius: f64) -> FourierSupport {
    FourierSupport::metric_ball(grid, lattice, k, TAU * radius, Default::default()).unwrap()
}
fn config(n_bands: usize) -> EigensolverConfig {
    EigensolverConfig {
        n_bands,
        tol: 1e-14,
        max_iter: 500,
        ..Default::default()
    }
}
fn relative(a: &[C], b: &[C]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x - *y).norm_sqr())
        .sum::<f64>()
        .sqrt()
        / a.iter().map(C::norm_sqr).sum::<f64>().sqrt().max(1.0)
}

#[test]
fn direct_and_literal_projection_agree_for_scalar_and_tensor_maxwell_pencils() {
    for n in [12, 13] {
        let grid = Grid2D::new(n, n, 1.0, 1.0);
        for lattice in [Lattice2D::square(1.0), Lattice2D::hexagonal(1.0)] {
            for pol in [Polarization::TM, Polarization::TE] {
                let k = [2.0 / 3.0, 1.0 / 3.0];
                let space = support(lattice, grid, k, 2.6);
                let d = rod(lattice, grid);
                if pol == Polarization::TE {
                    assert!(d.inv_eps_tensors().is_some());
                }
                let mut full = ThetaOperator::new(
                    CpuBackend::<f64>::new(),
                    d,
                    pol,
                    lattice.reciprocal().fractional_to_cartesian(k),
                );
                let mut matrices = Vec::new();
                for application in [Application::SpatialReference, Application::DirectFourier] {
                    let mut projected =
                        GalerkinOperator::new(&mut full, &space, application, usize::MAX).unwrap();
                    matrices.push(dense_pencil(&mut projected, 1 << 24).unwrap());
                }
                assert!(
                    relative(&matrices[0].0, &matrices[1].0) < 2e-13,
                    "{n} {pol:?}"
                );
                assert!(relative(&matrices[0].1, &matrices[1].1) < 2e-13);
                let m = space.len();
                for j in 0..m {
                    for i in 0..m {
                        assert!(
                            (matrices[1].0[i + j * m] - matrices[1].0[j + i * m].conj()).norm()
                                < 1e-12
                        );
                        assert!(
                            (matrices[1].1[i + j * m] - matrices[1].1[j + i * m].conj()).norm()
                                < 1e-12
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn complete_homogeneous_spectra_match_analytic_plane_waves_including_folded_gamma() {
    let lattice = Lattice2D::hexagonal(1.0);
    let grid = Grid2D::new(13, 13, 1.0, 1.0);
    let reciprocal = lattice.reciprocal();
    let epsilon = 3.7;
    let d = Dielectric2D::from_sampled_epsilon(
        grid,
        reciprocal.b1,
        reciprocal.b2,
        vec![epsilon; grid.len()],
        None,
    );
    for k in [[0.0; 2], [2.0 / 3.0, 1.0 / 3.0], [1.0, 0.0]] {
        let space = support(lattice, grid, k, 2.1);
        let mut expected: Vec<_> = space
            .modes()
            .iter()
            .map(|g| {
                let q =
                    reciprocal.fractional_to_cartesian([g[0] as f64 + k[0], g[1] as f64 + k[1]]);
                (q[0] * q[0] + q[1] * q[1]) / epsilon
            })
            .collect();
        expected.sort_by(f64::total_cmp);
        for pol in [Polarization::TM, Polarization::TE] {
            let mut op = ThetaOperator::new(
                CpuBackend::<f64>::new(),
                d.clone(),
                pol,
                reciprocal.fractional_to_cartesian(k),
            );
            let c = config(space.len());
            let result = solve_dense(
                &mut op,
                &space,
                &c,
                Application::DirectFourier,
                Default::default(),
            )
            .unwrap();
            assert!(result.report.accepted, "{k:?} {pol:?}: {:?}", result.report);
            assert_eq!(result.fields.len(), space.len());
            assert_eq!(result.support_identity, space.identity());
            for (&a, &b) in expected.iter().zip(&result.eigenvalues) {
                assert!((a - b).abs() / a.abs().max(1.0) < 2e-12, "{a} {b}");
            }
            let full = result.report.full_rectangle.as_ref().unwrap();
            for (i, &a) in full.absolute_residuals.iter().enumerate() {
                assert!(a < 1e-10, "{k:?} {pol:?} {i} {a}");
            }
        }
    }
}

#[test]
fn lobpcg_matches_dense_new_pencil_and_keeps_full_rectangle_error_separate() {
    let grid = Grid2D::new(17, 17, 1.0, 1.0);
    for lattice in [Lattice2D::square(1.0), Lattice2D::hexagonal(1.0)] {
        for (k, pol) in [
            ([0.0; 2], Polarization::TM),
            ([0.0; 2], Polarization::TE),
            ([2.0 / 3.0, 1.0 / 3.0], Polarization::TM),
            ([2.0 / 3.0, 1.0 / 3.0], Polarization::TE),
        ] {
            let space = support(lattice, grid, k, 3.1);
            let mut op = ThetaOperator::new(
                CpuBackend::<f64>::new(),
                rod(lattice, grid),
                pol,
                lattice.reciprocal().fractional_to_cartesian(k),
            );
            let c = config(6);
            let reference = solve_dense(
                &mut op,
                &space,
                &c,
                Application::SpatialReference,
                Default::default(),
            )
            .unwrap();
            assert!(reference.report.accepted, "dense: {:?}", reference.report);
            let mut p = op.build_homogeneous_preconditioner_adaptive();
            let result = solve_iterative(
                &mut op,
                &space,
                &c,
                None,
                Some(&mut p),
                Application::DirectFourier,
                Default::default(),
            )
            .unwrap();
            assert!(result.report.accepted, "{k:?} {pol:?}: {:?}", result.report);
            for (&a, &b) in reference.eigenvalues.iter().zip(&result.eigenvalues) {
                assert!(
                    (a - b).abs() / a.abs().max(1.0) < 1e-8,
                    "{k:?} {pol:?}: {a} {b}"
                );
            }
            assert!(
                result
                    .report
                    .full_rectangle
                    .as_ref()
                    .unwrap()
                    .residuals
                    .iter()
                    .skip(1)
                    .any(|&x| x > 1e-3)
            );
            assert!(
                result
                    .report
                    .projected
                    .as_ref()
                    .unwrap()
                    .residuals
                    .iter()
                    .skip(1)
                    .all(|&x| x < 1e-5)
            );
            let warm = solve_iterative(
                &mut op,
                &space,
                &c,
                Some(&result.fields),
                Some(&mut p),
                Application::DirectFourier,
                Default::default(),
            )
            .unwrap();
            assert!(warm.report.accepted, "warm: {:?}", warm.report);
        }
    }
}

#[test]
fn folded_kernel_is_deflated_without_changing_physical_bloch_vector() {
    let lattice = Lattice2D::square(1.0);
    let grid = Grid2D::new(13, 13, 1.0, 1.0);
    let k = [1.0, 0.0];
    let q = lattice.reciprocal().fractional_to_cartesian(k);
    let space = support(lattice, grid, k, 2.8);
    for pol in [Polarization::TM, Polarization::TE] {
        let mut op = ThetaOperator::new(CpuBackend::<f64>::new(), rod(lattice, grid), pol, q);
        {
            let mut projected =
                GalerkinOperator::new(&mut op, &space, Application::DirectFourier, usize::MAX)
                    .unwrap();
            assert_eq!(projected.bloch(), q);
            assert!(projected.analytic_kernel_available());
            let mut x = projected.alloc_field();
            let mut ax = projected.alloc_field();
            projected.fill_gamma_kernel(&mut x);
            projected.apply(&x, &mut ax);
            assert!(ax.as_slice().iter().all(|z| z.norm() < 1e-12));
            assert_eq!(x.as_slice().iter().filter(|z| z.norm() > 0.0).count(), 1);
        }
        let mut p = op.build_homogeneous_preconditioner_adaptive();
        let result = solve_iterative(
            &mut op,
            &space,
            &config(5),
            None,
            Some(&mut p),
            Application::DirectFourier,
            Default::default(),
        )
        .unwrap();
        assert!(result.report.accepted, "{:?}", result.report);
        assert!(result.report.analytic_kernel);
        assert!(result.eigenvalues[0].abs() < 1e-10);
    }
}

#[test]
fn admission_and_failed_attempts_are_explicit() {
    let lattice = Lattice2D::square(1.0);
    let grid = Grid2D::new(13, 13, 1.0, 1.0);
    let k = [0.17, 0.09];
    let space = support(lattice, grid, k, 3.6);
    let mut op = ThetaOperator::new(
        CpuBackend::<f64>::new(),
        rod(lattice, grid),
        Polarization::TE,
        lattice.reciprocal().fractional_to_cartesian(k),
    );
    let c = config(5);
    let bytes = estimated_payload_bytes(&space, &c, false).unwrap();
    assert!(
        solve_iterative(
            &mut op,
            &space,
            &c,
            None,
            None,
            Application::DirectFourier,
            GalerkinBudget {
                max_estimated_payload_bytes: bytes - 1,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        solve_dense(
            &mut op,
            &space,
            &c,
            Application::DirectFourier,
            GalerkinBudget {
                max_dense_dimension: 1,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(GalerkinOperator::new(&mut op, &space, Application::DirectFourier, 1).is_err());
    let wrong = support(lattice, grid, [0.17, 0.08], 3.6);
    assert!(
        GalerkinOperator::new(&mut op, &wrong, Application::DirectFourier, usize::MAX).is_err()
    );
    assert!(
        solve_dense(
            &mut op,
            &space,
            &config(space.len() + 1),
            Application::DirectFourier,
            Default::default()
        )
        .is_err()
    );
    let mut early = c.clone();
    early.max_iter = 1;
    let failed = solve_iterative(
        &mut op,
        &space,
        &early,
        None,
        None,
        Application::DirectFourier,
        Default::default(),
    )
    .unwrap();
    assert!(!failed.report.accepted && !failed.report.solver_converged);
    assert!(failed.report.iterations > 0 && failed.report.rejection.is_some());
    assert_eq!(failed.fields.len(), c.n_bands);
    assert!(failed.report.projected.is_some());
    let mut tensor = vec![[0.5, 0.0, 0.0, 0.5]; grid.len()];
    tensor[0] = [0.5, 2.0, 2.0, 0.5];
    let reciprocal = lattice.reciprocal();
    let d = Dielectric2D::from_sampled_epsilon(
        grid,
        reciprocal.b1,
        reciprocal.b2,
        vec![2.0; grid.len()],
        Some(tensor),
    );
    let mut bad = ThetaOperator::new(CpuBackend::<f64>::new(), d, Polarization::TE, op.bloch());
    assert!(
        GalerkinOperator::new(&mut bad, &space, Application::DirectFourier, usize::MAX).is_err()
    );
}

#[test]
fn nested_supports_obey_the_generalized_variational_order_on_fixed_material() {
    let lattice = Lattice2D::hexagonal(1.0);
    let grid = Grid2D::new(15, 15, 1.0, 1.0);
    for pol in [Polarization::TM, Polarization::TE] {
        let mut op =
            ThetaOperator::new(CpuBackend::<f64>::new(), rod(lattice, grid), pol, [0.0; 2]);
        let mut previous: Option<Vec<f64>> = None;
        for radius in [1.4, 2.1, 3.1, 4.2] {
            let space = support(lattice, grid, [0.0; 2], radius);
            let result = solve_dense(
                &mut op,
                &space,
                &config(5),
                Application::DirectFourier,
                Default::default(),
            )
            .unwrap();
            assert!(result.report.accepted, "{:?}", result.report);
            if let Some(old) = previous {
                for (&a, &b) in old.iter().zip(&result.eigenvalues) {
                    assert!(b <= a + 1e-10 * a.abs().max(1.0), "{pol:?}: {a} → {b}");
                }
            }
            previous = Some(result.eigenvalues);
        }
    }
}
