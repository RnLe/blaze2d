use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::Dielectric2D,
    drivers::{
        bandstructure::{BandStructureJob, KPointResult, RunOptions},
        support_bands::{self, MetricBallOptions},
    },
    eigensolver::{EigensolverConfig, iteration_total, refine::certify_without_refinement},
    field::Field2D,
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    operators::ThetaOperator,
    polarization::Polarization,
    symmetry::{
        affine::PlaneGroup,
        fourier::Arithmetic,
        galerkin::{self, GalerkinApplication, GalerkinOperator},
        reduced::SectorStorage,
        reuse::{KReuse, SymmetryOptions},
        sectors::SectorPolicy,
        support::FourierSupport,
        support_qualification::QualifiedSupportGroup,
    },
};
use num_complex::Complex64 as C;
use std::f64::consts::TAU;

fn job(pol: Polarization, raw: bool) -> BandStructureJob {
    let mut dielectric = blaze2d_core::dielectric::DielectricOptions::default();
    if raw {
        dielectric.smoothing.mesh_size = 1;
    }
    BandStructureJob {
        geom: Geometry2D {
            lattice: Lattice2D::hexagonal(1.0),
            eps_bg: 1.0,
            atoms: vec![BasisAtom {
                pos: [0.0; 2],
                radius: 0.2,
                eps_inside: 8.9,
            }],
        },
        grid: Grid2D::new(17, 17, 1.0, 1.0),
        pol,
        k_path: vec![[0.0; 2], [0.17, 0.09], [2.0 / 3.0, 1.0 / 3.0], [1.0, 0.0]],
        eigensolver: EigensolverConfig {
            n_bands: 5,
            tol: 1e-14,
            max_iter: 1000,
            ..Default::default()
        },
        dielectric,
    }
}
fn selected() -> MetricBallOptions {
    MetricBallOptions {
        angular_cutoff: TAU * 3.1,
        support_budget: Default::default(),
        budget: Default::default(),
        max_allocation_rounds: 32,
    }
}
fn options() -> RunOptions {
    RunOptions {
        reuse_gamma: false,
        retain_eigenvectors: true,
        symmetry: SymmetryOptions {
            arithmetic: Arithmetic::FourierComplex,
            sectors: SectorPolicy::AdaptiveGroup,
            plane_group: Some(PlaneGroup::P6m),
            sector_storage: SectorStorage::Orbit,
            reconstruct_partners: true,
            sector_guard_budget: Some(8),
            ..Default::default()
        },
        ..Default::default()
    }
}
fn operator(j: &BandStructureJob, k: [f64; 2]) -> ThetaOperator<CpuBackend<f64>> {
    ThetaOperator::new(
        CpuBackend::new(),
        Dielectric2D::from_geometry(&j.geom, j.grid, &j.dielectric),
        j.pol,
        j.geom.lattice.reciprocal().fractional_to_cartesian(k),
    )
}

#[test]
fn tracked_path_retains_support_certificates_labels_and_folded_kernel() {
    let mut swaps = 0;
    for pol in [Polarization::TM, Polarization::TE] {
        let j = job(pol, true);
        let mut points = Vec::new();
        let result = support_bands::run_with_k_streaming(
            CpuBackend::<f64>::new(),
            &j,
            options(),
            selected(),
            |p| points.push(p),
        )
        .unwrap();
        assert_eq!(result.k_path, j.k_path);
        assert_eq!(points.len(), j.k_path.len());
        for point in &points {
            assert!(point.converged);
            let report = point.discretization.as_ref().unwrap();
            let k = point.k_point;
            let space = FourierSupport::metric_ball(
                j.grid,
                j.geom.lattice,
                k,
                selected().angular_cutoff,
                Default::default(),
            )
            .unwrap();
            assert_eq!(report.support.as_ref().unwrap(), &space.identity());
            let mut full = operator(&j, k);
            let reference = galerkin::solve_dense(
                &mut full,
                &space,
                &j.eigensolver,
                GalerkinApplication::SpatialReference,
                Default::default(),
            )
            .unwrap();
            assert!(reference.report.accepted);
            let values: Vec<_> = point.omegas.iter().map(|x| x * x).collect();
            let mut sorted = values.clone();
            sorted.sort_by(f64::total_cmp);
            assert!(
                sorted
                    .iter()
                    .zip(&reference.eigenvalues)
                    .all(|(a, b)| (a - b).abs() / b.abs().max(1.0) < 1e-8)
            );
            let mut projected = GalerkinOperator::new(
                &mut full,
                &space,
                GalerkinApplication::DirectFourier,
                usize::MAX,
            )
            .unwrap();
            let conversion = CpuBackend::<f64>::new();
            let mut scratch = Field2D::zeros(j.grid);
            let coefficients: Vec<_> = point
                .eigenvectors
                .as_ref()
                .unwrap()
                .iter()
                .map(|field| {
                    let mut c = Field2D::zeros(space.coordinate_grid());
                    space.restrict(&conversion, field, &mut scratch, c.as_mut_slice());
                    c
                })
                .collect();
            let certificate = certify_without_refinement(&mut projected, &values, &coefficients);
            for (i, &r) in point.residuals.iter().enumerate() {
                if values[i].abs() > 1e-7 {
                    assert!((r - certificate.residuals[i]).abs() < 1e-8);
                }
                assert!(
                    (point.absolute_residuals[i] - certificate.absolute_residuals[i]).abs() < 1e-8
                );
            }
            assert!(certificate.b_orthogonality_defect < 1e-8);
            assert_eq!(
                report
                    .selected
                    .as_ref()
                    .unwrap()
                    .projected
                    .as_ref()
                    .unwrap()
                    .residuals,
                point.residuals
            );
            assert_eq!(
                point.iterations,
                point
                    .symmetry
                    .solver_attempts
                    .iter()
                    .map(|a| a.iterations)
                    .sum::<usize>()
            );
            if report
                .returned_permutation
                .iter()
                .enumerate()
                .any(|(i, &p)| i != p)
            {
                swaps += 1;
            }
            drop(projected);
            if let Some(labels) = &report.returned_labels {
                let q = QualifiedSupportGroup::qualify(
                    &mut full,
                    &space,
                    PlaneGroup::P6m.group().little_group(k, 1e-12).unwrap(),
                    Default::default(),
                )
                .unwrap();
                for (label, field) in labels.iter().zip(&coefficients) {
                    let weights = label.row.map_or_else(
                        || q.representation().isotypic_weights(label.irrep),
                        |row| {
                            q.representation()
                                .matrix_unit_weights(label.irrep, row, row)
                        },
                    );
                    let mut output = vec![C::ZERO; space.len()];
                    let mut scratch = output.clone();
                    for (action, w) in q.actions().iter().zip(weights) {
                        action.apply(field.as_slice(), &mut scratch);
                        for (dst, &v) in output.iter_mut().zip(&scratch) {
                            *dst += w * v;
                        }
                    }
                    let error: f64 = output
                        .iter()
                        .zip(field.as_slice())
                        .map(|(a, b)| (a - b).norm_sqr())
                        .sum();
                    let scale: f64 = field.as_slice().iter().map(C::norm_sqr).sum();
                    assert!(
                        (error / scale).sqrt() < 1e-8,
                        "returned tracking label must describe the returned field"
                    );
                }
            }
        }
        assert!(
            points.last().unwrap().is_gamma,
            "integer-reciprocal Gamma retains its actual kernel"
        );
        assert!(
            points[1]
                .discretization
                .as_ref()
                .unwrap()
                .returned_labels
                .is_none()
        );
    }
    assert!(swaps > 0, "fixture must exercise tracking permutations");
}

#[test]
fn rejected_material_falls_back_on_the_same_projected_pencil() {
    for pol in [Polarization::TM, Polarization::TE] {
        let mut j = job(pol, false);
        j.k_path = vec![[0.0; 2]];
        let mut points = Vec::new();
        support_bands::run_with_k_streaming(
            CpuBackend::<f64>::new(),
            &j,
            options(),
            selected(),
            |p| points.push(p),
        )
        .unwrap();
        let p = &points[0];
        let r = p.discretization.as_ref().unwrap();
        assert!(!r.qualification.as_ref().unwrap().authorized);
        assert!(
            r.qualification
                .as_ref()
                .unwrap()
                .rejection
                .as_ref()
                .unwrap()
                .contains("material")
        );
        assert!(r.returned_labels.is_none());
        assert_eq!(p.symmetry.solver_attempts.len(), 2);
        assert!(!p.symmetry.solver_attempts[0].retained && p.symmetry.solver_attempts[1].retained);
        let mut control = options();
        control.symmetry = SymmetryOptions {
            arithmetic: Arithmetic::FourierComplex,
            ..Default::default()
        };
        let mut reference = Vec::new();
        support_bands::run_with_k_streaming(
            CpuBackend::<f64>::new(),
            &j,
            control,
            selected(),
            |p| reference.push(p),
        )
        .unwrap();
        assert_eq!(p.omegas, reference[0].omegas);
        assert_eq!(p.residuals, reference[0].residuals);
        assert_eq!(
            r.support,
            reference[0].discretization.as_ref().unwrap().support
        );
        assert!(
            r.selected
                .as_ref()
                .unwrap()
                .full_rectangle
                .as_ref()
                .unwrap()
                .residuals
                .iter()
                .skip(1)
                .any(|&v| v > 1e-3)
        );
    }
}

#[test]
fn failed_attempts_and_partial_paths_remain_visible() {
    let mut j = job(Polarization::TE, true);
    j.grid = Grid2D::new(33, 33, 1.0, 1.0);
    j.eigensolver.n_bands = 8;
    j.eigensolver.max_iter = 1;
    j.k_path = vec![[0.0; 2], [0.17, 0.09]];
    let mut selected = selected();
    selected.angular_cutoff = TAU * 7.0;
    let before = iteration_total();
    let mut points: Vec<KPointResult> = Vec::new();
    let failure = support_bands::run_with_k_streaming(
        CpuBackend::<f64>::new(),
        &j,
        options(),
        selected,
        |p| points.push(p),
    )
    .unwrap_err();
    assert_eq!(failure.k_index, Some(0));
    assert!(!points[0].converged);
    assert_eq!(failure.partial_result.bands.len(), 1);
    assert_eq!(failure.symmetry.solver_attempts.len(), 2);
    assert!(
        failure
            .discretization
            .unwrap()
            .sector_attempt
            .unwrap()
            .attempts
            .iter()
            .any(|a| !a.accepted)
    );
    assert_eq!(
        failure.symmetry.attempted_iterations,
        iteration_total() - before
    );
    assert_eq!(
        failure.symmetry.attempted_iterations,
        failure
            .symmetry
            .solver_attempts
            .iter()
            .map(|a| a.iterations)
            .sum::<usize>()
    );
    assert!(failure.symmetry.attempted_iterations > failure.symmetry.retained_iterations);
}

#[test]
fn growth_limit_fallback_keeps_failed_work_and_returns_the_unreduced_support() {
    let mut j = job(Polarization::TE, true);
    j.grid = Grid2D::new(33, 33, 1.0, 1.0);
    j.k_path = vec![[0.0; 2]];
    j.eigensolver.n_bands = 8;
    let mut settings = selected();
    settings.angular_cutoff = TAU * 7.0;
    settings.max_allocation_rounds = 1;
    let mut points = Vec::new();
    support_bands::run_with_k_streaming(CpuBackend::<f64>::new(), &j, options(), settings, |p| {
        points.push(p)
    })
    .unwrap();
    let point = &points[0];
    let report = point.discretization.as_ref().unwrap();
    assert!(report.qualification.as_ref().unwrap().authorized);
    assert!(!report.sector_attempt.as_ref().unwrap().accepted);
    assert!(
        report
            .sector_attempt
            .as_ref()
            .unwrap()
            .rejection
            .as_ref()
            .unwrap()
            .contains("allocation round limit reached")
    );
    assert!(point.converged && report.returned_labels.is_none());
    assert!(point.iterations > point.symmetry.retained_iterations);
    assert!(
        point.symmetry.solver_attempts[0].iterations > 0
            && !point.symmetry.solver_attempts[0].retained
    );
    assert_eq!(report.selected.as_ref().unwrap().method, "lobpcg");
    let mut control = options();
    control.symmetry = SymmetryOptions {
        arithmetic: Arithmetic::FourierComplex,
        ..Default::default()
    };
    let mut reference = Vec::new();
    support_bands::run_with_k_streaming(CpuBackend::<f64>::new(), &j, control, settings, |p| {
        reference.push(p)
    })
    .unwrap();
    assert_eq!(point.omegas, reference[0].omegas);
    assert_eq!(point.residuals, reference[0].residuals);
}

#[test]
fn invalid_precision_budget_and_combinations_fail_before_solver_work() {
    assert!(support_bands::driver_payload_bytes(usize::MAX, usize::MAX, usize::MAX).is_err());
    assert!(
        galerkin::estimated_payload_bytes_for_dimensions(
            usize::MAX,
            usize::MAX,
            &EigensolverConfig {
                n_bands: usize::MAX,
                block_size: usize::MAX,
                ..Default::default()
            },
            true
        )
        .is_err()
    );
    let j = job(Polarization::TM, true);
    let before = iteration_total();
    let err = support_bands::run_with_k_streaming(
        CpuBackend::<f32>::new(),
        &j,
        options(),
        selected(),
        |_| panic!("no point should run"),
    )
    .unwrap_err();
    assert!(err.reason.contains("f64"));
    assert!(err.partial_result.bands.is_empty());
    let mut low = selected();
    low.budget.max_estimated_payload_bytes = 1;
    assert!(
        support_bands::run_with_k_streaming(
            CpuBackend::<f64>::new(),
            &j,
            options(),
            low,
            |_| panic!()
        )
        .unwrap_err()
        .reason
        .contains("payload")
    );
    let mut incompatible = options();
    incompatible.symmetry.k_traversal = blaze2d_core::symmetry::traversal::KTraversal::Nearest;
    assert!(
        support_bands::run_with_k_streaming(
            CpuBackend::<f64>::new(),
            &j,
            incompatible,
            selected(),
            |_| panic!()
        )
        .unwrap_err()
        .reason
        .contains("reuse")
    );
    assert_eq!(iteration_total(), before);
}

#[test]
fn projected_pt_preserves_triangular_te_tm_pencils_and_real_storage() {
    for pol in [Polarization::TE, Polarization::TM] {
        let mut j = job(pol, false);
        j.dielectric.smoothing.method = blaze2d_core::dielectric::SmoothingMethod::Voronoi;
        let o = RunOptions { retain_eigenvectors: true, reuse_gamma: false,
            symmetry: SymmetryOptions { arithmetic: Arithmetic::PtRealTe, ..Default::default() }, ..Default::default() };
        support_bands::run_with_k_streaming(CpuBackend::<f64>::new(), &j, o, selected(), |point| {
            let coordinates = point.symmetry.coordinates.as_ref().expect("qualified PT must execute");
            let space = FourierSupport::metric_ball(j.grid, j.geom.lattice, point.k_point,
                selected().angular_cutoff, Default::default()).unwrap();
            assert_eq!(coordinates.bytes_per_column, space.len().div_ceil(2) * 16);
            let mut full = operator(&j, point.k_point);
            let reference = galerkin::solve_dense(&mut full, &space, &j.eigensolver,
                GalerkinApplication::SpatialReference, Default::default()).unwrap();
            let mut actual = point.omegas.clone(); actual.sort_by(f64::total_cmp);
            for (a,b) in actual.iter().zip(&reference.eigenvalues) {
                assert!((a - b.max(0.0).sqrt()).abs() < 2e-7, "pol={pol:?} k={:?} got={:?} expected={:?}", point.k_point, actual, reference.eigenvalues.iter().map(|v| v.max(0.0).sqrt()).collect::<Vec<_>>());
            }
        }).unwrap();
    }
}

#[test]
fn projected_pt_star_reuse_coarse_and_geometry_chains_match_dense_spectra() {
    use blaze2d_core::symmetry::{affine::AffineOp, continuation::BandContinuation, coarse::CoarseStartOptions};
    for pol in [Polarization::TE, Polarization::TM] {
        let mut j = job(pol, false);
        j.dielectric.smoothing.method = blaze2d_core::dielectric::SmoothingMethod::Voronoi;
        let k = [0.17,0.09];
        let rotation = AffineOp::new([[0,-1],[1,1]], [blaze2d_core::symmetry::affine::Rational::ZERO;2]).unwrap();
        j.k_path = vec![[0.0;2], k, rotation.map_wavevector(k), [-k[0],-k[1]], [0.0;2]];
        let o = RunOptions { reuse_gamma: false, retain_eigenvectors: true,
            symmetry: SymmetryOptions { arithmetic: Arithmetic::PtRealTe, k_reuse: KReuse::TimeReversal,
                plane_group: Some(PlaneGroup::P6m), max_cache_bytes: 1<<20,
                coarse_start: Some(CoarseStartOptions { resolution: [9,9], max_seed_field_bytes: 1<<20 }),
                ..Default::default() }, ..Default::default() };
        let mut continuation = BandContinuation::new(1<<20);
        let mut seeds = 0; let mut reuses = 0; let mut coarse = 0;
        for (index,radius) in [0.18,0.20,0.22].into_iter().enumerate() {
            continuation.job_index = index; j.geom.atoms[0].radius = radius;
            support_bands::run_with_continuation(CpuBackend::<f64>::new(),&j,o.clone(),selected(),Some(&mut continuation),|point| {
                assert!(point.converged);
                assert_eq!(point.symmetry.attempted_iterations, point.symmetry.solver_attempts.iter().map(|a|a.iterations).sum::<usize>());
                seeds += usize::from(point.symmetry.continuation.as_ref().and_then(|r|r.seed.as_ref()).is_some());
                reuses += usize::from(point.symmetry.source_index.is_some());
                coarse += usize::from(point.symmetry.coarse_start.as_ref().is_some_and(|r|r.used_as_seed));
                let space = FourierSupport::metric_ball(j.grid,j.geom.lattice,point.k_point,selected().angular_cutoff,Default::default()).unwrap();
                let reference = galerkin::solve_dense(&mut operator(&j,point.k_point),&space,&j.eigensolver,
                    GalerkinApplication::SpatialReference,Default::default()).unwrap();
                let mut actual = point.omegas.clone(); actual.sort_by(f64::total_cmp);
                for (a,b) in actual.iter().zip(&reference.eigenvalues) {
                    assert!((a-b.max(0.0).sqrt()).abs() < 2e-7,"{pol:?} k={:?}",point.k_point);
                }
            }).unwrap();
            assert!(continuation.cache.payload_bytes() <= 1<<20);
        }
        assert_eq!(coarse,1); assert_eq!(seeds,4); assert_eq!(reuses,9);
    }
}
