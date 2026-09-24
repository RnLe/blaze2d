use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::{Dielectric2D, DielectricOptions},
    eigensolver::EigensolverConfig,
    field::Field2D,
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    operators::ThetaOperator,
    polarization::Polarization,
    symmetry::{
        affine::PlaneGroup,
        galerkin::{self, GalerkinApplication},
        support::FourierSupport,
        support_qualification::QualifiedSupportGroup,
        support_sectors::{self, SupportSectorOptions, SupportSectorPlan},
    },
};
use num_complex::Complex64 as C;
use std::f64::consts::TAU;

fn setup(
    name: PlaneGroup,
    k: [f64; 2],
    pol: Polarization,
    n: usize,
    radius: f64,
) -> (ThetaOperator<CpuBackend<f64>>, FourierSupport) {
    let hex = matches!(name, PlaneGroup::P3 | PlaneGroup::P6 | PlaneGroup::P6m);
    let lattice = if hex {
        Lattice2D::hexagonal(1.0)
    } else {
        Lattice2D::square(1.0)
    };
    let grid = Grid2D::new(n, n, 1.0, 1.0);
    let reciprocal = lattice.reciprocal();
    let d = if matches!(name, PlaneGroup::Pg | PlaneGroup::Pgg) {
        let eps = (0..grid.len())
            .map(|i| {
                let x = TAU * (i % n) as f64 / n as f64;
                let y = TAU * (i / n) as f64 / n as f64;
                3.0 + 0.6 * x.sin() * y.sin() + 0.3 * (2.0 * x).cos() * (2.0 * y).cos()
            })
            .collect();
        Dielectric2D::from_sampled_epsilon(grid, reciprocal.b1, reciprocal.b2, eps, None)
    } else {
        let atoms = if name == PlaneGroup::P4 {
            let mut pos = [0.23, 0.11];
            (0..4)
                .map(|_| {
                    let atom = BasisAtom {
                        pos,
                        radius: 0.08,
                        eps_inside: 8.9,
                    };
                    pos = [-pos[1], pos[0]];
                    atom
                })
                .collect()
        } else {
            vec![BasisAtom {
                pos: [0.0; 2],
                radius: 0.2,
                eps_inside: 8.9,
            }]
        };
        let mut options = DielectricOptions::default();
        if hex {
            options.smoothing.mesh_size = 1;
        }
        Dielectric2D::from_geometry(
            &Geometry2D {
                lattice,
                eps_bg: 1.0,
                atoms,
            },
            grid,
            &options,
        )
    };
    let full = ThetaOperator::new(
        CpuBackend::new(),
        d,
        pol,
        reciprocal.fractional_to_cartesian(k),
    );
    let support =
        FourierSupport::metric_ball(grid, lattice, k, TAU * radius, Default::default()).unwrap();
    (full, support)
}
fn config(n_bands: usize) -> EigensolverConfig {
    EigensolverConfig {
        n_bands,
        tol: 1e-14,
        max_iter: 1000,
        ..Default::default()
    }
}
fn agree(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (&x, &y) in a.iter().zip(b) {
        assert!(
            (x - y).abs() / x.abs().max(y.abs()).max(1.0) < 2e-8,
            "{x} != {y}"
        );
    }
}

#[test]
fn complete_reduced_spectra_match_the_same_nonuniform_galerkin_pencil() {
    for (name, k) in [
        (PlaneGroup::P3, [0.0; 2]),
        (PlaneGroup::P6m, [0.0; 2]),
        (PlaneGroup::P6m, [2.0 / 3.0, 1.0 / 3.0]),
        (PlaneGroup::P6m, [1.0, 0.0]),
        (PlaneGroup::P4, [0.0; 2]),
        (PlaneGroup::P4m, [0.5, 0.5]),
        (PlaneGroup::Pg, [0.5, 0.0]),
        (PlaneGroup::Pgg, [0.5, 0.0]),
    ] {
        for pol in [Polarization::TM, Polarization::TE] {
            let (mut full, space) = setup(name, k, pol, 16, 2.1);
            let cfg = config(space.len());
            let reference = galerkin::solve_dense(
                &mut full,
                &space,
                &cfg,
                GalerkinApplication::SpatialReference,
                Default::default(),
            )
            .unwrap();
            assert!(
                reference.report.accepted,
                "{name:?} {pol:?} {:?}",
                reference.report
            );
            for partners in [false, true] {
                let options = SupportSectorOptions {
                    complete_dense: true,
                    reconstruct_partners: partners,
                    ..Default::default()
                };
                let group = name.group().little_group(k, 1e-12).unwrap();
                let result =
                    support_sectors::solve(&mut full, &space, group, &cfg, None, None, options)
                        .unwrap();
                assert!(
                    result.report.accepted,
                    "{name:?} {k:?} {pol:?}: {:?}",
                    result.report
                );
                agree(&reference.eigenvalues, &result.spectrum.eigenvalues);
                assert_eq!(result.report.computed_with_multiplicity, space.len());
                assert_eq!(result.report.output_boundary, "complete finite spectrum");
                assert_eq!(result.spectrum.fields.len(), space.len());
                assert_eq!(result.report.output_labels.len(), space.len());
                assert_eq!(
                    result.report.kernel_owners,
                    usize::from(k == [0.0; 2] || k == [1.0, 0.0])
                );
                assert!(
                    result
                        .report
                        .attempts
                        .iter()
                        .all(|a| a.dense_complete && a.iterations == 0 && a.accepted)
                );
            }
        }
    }
}

#[test]
fn iterative_prefixes_guards_and_partners_match_dense_low_modes() {
    let mut growing_cases = 0;
    for (name, k, n) in [
        (PlaneGroup::P6m, [0.0; 2], 33),
        (PlaneGroup::P6m, [2.0 / 3.0, 1.0 / 3.0], 33),
        (PlaneGroup::P4, [0.0; 2], 33),
        (PlaneGroup::Pgg, [0.5, 0.0], 32),
    ] {
        for pol in [Polarization::TM, Polarization::TE] {
            let (mut full, space) = setup(name, k, pol, n, 7.0);
            let cfg = config(8);
            let reference = galerkin::solve_dense(
                &mut full,
                &space,
                &cfg,
                GalerkinApplication::DirectFourier,
                Default::default(),
            )
            .unwrap();
            assert!(reference.report.accepted);
            for adaptive in [false, true] {
                let group = name.group().little_group(k, 1e-12).unwrap();
                let mut p = full.build_homogeneous_preconditioner_adaptive();
                let options = SupportSectorOptions {
                    adaptive_prefixes: adaptive,
                    ..Default::default()
                };
                let result = support_sectors::solve(
                    &mut full,
                    &space,
                    group,
                    &cfg,
                    None,
                    Some(&mut p),
                    options,
                )
                .unwrap();
                assert!(
                    result.report.accepted,
                    "{name:?} {k:?} {pol:?} adaptive={adaptive}: {:?}",
                    result.report
                );
                agree(&reference.eigenvalues, &result.spectrum.eigenvalues);
                assert!(
                    result
                        .report
                        .attempts
                        .iter()
                        .any(|a| !a.dense_complete && a.iterations > 0)
                );
                assert_eq!(
                    result.spectrum.report.iterations,
                    result
                        .report
                        .attempts
                        .iter()
                        .map(|a| a.iterations)
                        .sum::<usize>()
                );
                assert!(result.report.attempts.iter().all(|a| a.max_leakage < 1e-10));
                if adaptive && result.report.allocation_rounds > 1 {
                    growing_cases += 1;
                }
            }
        }
    }
    assert!(
        growing_cases > 0,
        "fixture must exercise adaptive growth, not only initial allocation"
    );
}

#[test]
fn warm_fields_are_rank_selected_and_a_retained_plan_rejects_changed_inputs() {
    let name = PlaneGroup::P6m;
    let k = [0.0; 2];
    let (mut full, space) = setup(name, k, Polarization::TM, 33, 7.0);
    let cfg = config(8);
    let reference = galerkin::solve_dense(
        &mut full,
        &space,
        &cfg,
        GalerkinApplication::DirectFourier,
        Default::default(),
    )
    .unwrap();
    let group = QualifiedSupportGroup::qualify(&mut full, &space, name.group(), Default::default())
        .unwrap();
    let plan = SupportSectorPlan::new(group, &full, &space, true, 1 << 29).unwrap();
    let mut warm = reference.fields.clone();
    warm.push(warm[0].clone());
    warm.push(warm[1].clone());
    let mut p = full.build_homogeneous_preconditioner_adaptive();
    let result = support_sectors::solve_with_plan(
        &mut full,
        &space,
        &plan,
        &cfg,
        Some(&warm),
        Some(&mut p),
        Default::default(),
    )
    .unwrap();
    assert!(result.report.accepted, "{:?}", result.report);
    agree(&reference.eigenvalues, &result.spectrum.eigenvalues);
    for (field, label) in result
        .spectrum
        .coefficients
        .iter()
        .zip(&result.report.output_labels)
    {
        let row = label.row.unwrap();
        let weights = plan
            .group()
            .representation()
            .matrix_unit_weights(label.irrep, row, row);
        let mut projected = vec![C::ZERO; space.len()];
        let mut scratch = projected.clone();
        for (action, &weight) in plan.group().actions().iter().zip(&weights) {
            action.apply(field.as_slice(), &mut scratch);
            for (a, &b) in projected.iter_mut().zip(&scratch) {
                *a += weight * b;
            }
        }
        let scale = field.as_slice().iter().map(C::norm_sqr).sum::<f64>();
        let error = projected
            .iter()
            .zip(field.as_slice())
            .map(|(a, b)| (*a - *b).norm_sqr())
            .sum::<f64>();
        assert!(
            (error / scale).sqrt() < 1e-10,
            "final certification must preserve physical row labels"
        );
    }
    assert!(
        result
            .report
            .attempts
            .iter()
            .any(|a| a.seed_rank > 0 && a.seed_rank < warm.len())
    );
    let bad_warm = vec![Field2D::zeros(Grid2D::new(5, 5, 1.0, 1.0))];
    assert!(
        support_sectors::solve_with_plan(
            &mut full,
            &space,
            &plan,
            &cfg,
            Some(&bad_warm),
            None,
            Default::default()
        )
        .is_err()
    );
    let changed = FourierSupport::metric_ball(
        space.grid(),
        space.lattice(),
        k,
        TAU * 6.0,
        Default::default(),
    )
    .unwrap();
    assert!(
        support_sectors::solve_with_plan(
            &mut full,
            &changed,
            &plan,
            &cfg,
            None,
            None,
            Default::default()
        )
        .is_err()
    );
}

#[test]
fn wide_dependent_seeds_are_budgeted_before_rank_extraction() {
    let name = PlaneGroup::P6m;
    let k = [0.0; 2];
    let (mut full, space) = setup(name, k, Polarization::TM, 33, 7.0);
    let qualified =
        QualifiedSupportGroup::qualify(&mut full, &space, name.group(), Default::default())
            .unwrap();
    let plan = SupportSectorPlan::new(qualified, &full, &space, true, 1 << 29).unwrap();
    let cfg = EigensolverConfig {
        block_size: space.len(),
        ..config(8)
    };
    let mut p = full.build_homogeneous_preconditioner_adaptive();
    let cold = support_sectors::solve_with_plan(
        &mut full,
        &space,
        &plan,
        &cfg,
        None,
        Some(&mut p),
        Default::default(),
    )
    .unwrap();
    assert!(cold.report.accepted, "{:?}", cold.report);
    let warm = vec![cold.spectrum.fields[0].clone(); space.len()];
    let mut options = SupportSectorOptions::default();
    options.budget.max_estimated_payload_bytes = cold.report.estimated_payload_bytes;
    let attempt = support_sectors::solve_with_plan(
        &mut full,
        &space,
        &plan,
        &cfg,
        Some(&warm),
        Some(&mut p),
        options,
    );
    assert!(matches!(attempt,Err(message) if message.contains("payload estimate")));
}

#[test]
fn failed_iterations_and_limited_growth_retain_attempted_work() {
    let name = PlaneGroup::P6m;
    let k = [0.0; 2];
    let (mut full, space) = setup(name, k, Polarization::TE, 33, 7.0);
    let cfg = EigensolverConfig {
        max_iter: 1,
        tol: 1e-30,
        ..config(8)
    };
    let mut p = full.build_homogeneous_preconditioner_adaptive();
    let result = support_sectors::solve(
        &mut full,
        &space,
        name.group(),
        &cfg,
        None,
        Some(&mut p),
        Default::default(),
    )
    .unwrap();
    assert!(!result.report.accepted && !result.spectrum.report.accepted);
    assert!(
        result
            .report
            .attempts
            .iter()
            .any(|a| !a.solver_converged && a.iterations > 0)
    );
    assert_eq!(
        result.spectrum.report.iterations,
        result
            .report
            .attempts
            .iter()
            .map(|a| a.iterations)
            .sum::<usize>()
    );
    assert!(!result.spectrum.coefficients.is_empty());
    let mut options = SupportSectorOptions::default();
    options.budget.max_estimated_payload_bytes = 1;
    assert!(
        support_sectors::solve(
            &mut full,
            &space,
            name.group(),
            &config(8),
            None,
            None,
            options
        )
        .is_err()
    );
    // With no second allocation round, the completed first round must remain
    // visible even if its converged tails cannot establish requested coverage.
    let options = SupportSectorOptions {
        max_allocation_rounds: 1,
        ..Default::default()
    };
    let result = support_sectors::solve(
        &mut full,
        &space,
        name.group(),
        &config(16),
        None,
        Some(&mut p),
        options,
    )
    .unwrap();
    assert!(
        !result.report.accepted,
        "fixture must require prefix growth"
    );
    assert!(
        result
            .report
            .rejection
            .as_ref()
            .unwrap()
            .contains("round limit")
    );
    assert!(result.report.attempts.iter().all(|a| a.accepted));
    assert!(result.spectrum.report.iterations > 0);
}
