use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::Dielectric2D,
    drivers::single_solve::{SingleSolveJob, solve, solve_with_warmstart},
    eigensolver::refine::certify_without_refinement,
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    operators::ThetaOperator,
    polarization::Polarization,
    symmetry::precision::{F32SeedConfig, prepare_f32_seed},
};

#[test]
fn f32_initialization_keeps_the_f64_physical_target_and_reports_its_accuracy_floor() {
    let lattice = Lattice2D::square(1.0);
    let grid = Grid2D::new(13, 13, 1.0, 1.0);
    let dielectric = Dielectric2D::from_geometry(
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
    );
    let job = SingleSolveJob::new(6)
        .with_tolerance(1e-14)
        .with_max_iterations(500);
    for pol in [Polarization::TM, Polarization::TE] {
        for k in [[0.0; 2], [0.13, 0.17]] {
            let mut target = ThetaOperator::new(
                CpuBackend::<f64>::new(),
                dielectric.clone(),
                pol,
                lattice.reciprocal().fractional_to_cartesian(k),
            );
            let mut preconditioner = target.build_homogeneous_preconditioner_adaptive();
            let reference = solve(&mut target, Some(&mut preconditioner), &job);
            for budget in [1, 64] {
                let seed = prepare_f32_seed(
                    &mut target,
                    CpuBackend::<f32>::new(),
                    &job,
                    F32SeedConfig {
                        max_iterations: budget,
                        ..Default::default()
                    },
                )
                .unwrap();
                assert_eq!(
                    seed.report.low_field_column_bytes * 2,
                    seed.report.target_field_column_bytes
                );
                assert!(seed.report.low_precision_iterations <= budget);
                if budget == 1 {
                    assert!(!seed.report.low_precision_converged);
                }
                assert!(
                    seed.report
                        .promoted_physical_residuals
                        .iter()
                        .all(|r| r.is_finite())
                );
                assert!(seed.report.target_seed.b_orthogonality_defect < 1e-8);
                let corrected = solve_with_warmstart(
                    &mut target,
                    Some(&mut preconditioner),
                    &job,
                    &seed.fields,
                );
                assert!(corrected.converged, "{pol:?} {k:?} budget={budget}");
                let cert = certify_without_refinement(
                    &mut target,
                    &corrected.eigenvalues,
                    &corrected.eigenvectors,
                );
                assert!(
                    cert.residuals
                        .iter()
                        .skip(usize::from(k == [0.0; 2]))
                        .all(|r| *r < 1e-5)
                );
                if k == [0.0; 2] {
                    assert!(cert.absolute_residuals[0] < 1e-7);
                }
                assert!(cert.b_orthogonality_defect < 1e-8);
                for (a, b) in reference.eigenvalues.iter().zip(&corrected.eigenvalues) {
                    assert!(
                        (a - b).abs() / a.abs().max(1.0) < 1e-7,
                        "{pol:?} {k:?}: {a} {b}"
                    );
                }
            }
            assert!(
                prepare_f32_seed(
                    &mut target,
                    CpuBackend::<f32>::new(),
                    &job,
                    F32SeedConfig {
                        eigenvalue_tolerance: f64::NAN,
                        ..Default::default()
                    }
                )
                .is_err()
            );
        }
    }
}
