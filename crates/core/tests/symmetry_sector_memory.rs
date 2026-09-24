use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::Dielectric2D,
    drivers::bandstructure::{self, BandStructureJob, RunOptions},
    eigensolver::{EigensolverConfig, iteration_total},
    field::Field2D,
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    operators::ThetaOperator,
    polarization::Polarization,
    symmetry::{
        affine::PlaneGroup,
        reduced::{ReducedKernel, SectorStorage},
        reuse::SymmetryOptions,
        sectors::{self, SectorPlan, SectorPolicy, memory},
    },
};
use num_complex::Complex64 as C;

fn job(n: usize, bands: usize, homogeneous: bool) -> BandStructureJob {
    BandStructureJob {
        geom: Geometry2D {
            lattice: Lattice2D::square(1.0),
            eps_bg: 1.0,
            atoms: if homogeneous {
                vec![]
            } else {
                vec![BasisAtom {
                    pos: [0.0; 2],
                    radius: 0.24,
                    eps_inside: 8.9,
                }]
            },
        },
        grid: Grid2D::new(n, n, 1.0, 1.0),
        pol: Polarization::TM,
        k_path: vec![[0.0; 2]],
        eigensolver: EigensolverConfig {
            n_bands: bands,
            tol: 1e-14,
            max_iter: 500,
            ..Default::default()
        },
        dielectric: Default::default(),
    }
}
fn operator(j: &BandStructureJob) -> ThetaOperator<CpuBackend<f64>> {
    ThetaOperator::new(
        CpuBackend::new(),
        Dielectric2D::from_geometry(&j.geom, j.grid, &j.dielectric),
        j.pol,
        j.geom
            .lattice
            .reciprocal()
            .fractional_to_cartesian(j.k_path[0]),
    )
}
fn options() -> SymmetryOptions {
    SymmetryOptions {
        sectors: SectorPolicy::AdaptiveGroup,
        plane_group: Some(PlaneGroup::Pm),
        ..Default::default()
    }
}
fn solve(
    j: &BandStructureJob,
    s: &SymmetryOptions,
    warm: Option<&[Field2D]>,
) -> Result<sectors::SectorResult, String> {
    let mut op = operator(j);
    let mut p = op.build_homogeneous_preconditioner_adaptive();
    sectors::solve_conservative(
        &mut op,
        &j.geom.lattice,
        j.k_path[0],
        PlaneGroup::Pm.group(),
        &j.eigensolver,
        warm,
        s,
        Some(&mut p),
    )
}

#[test]
fn setup_dense_and_original_seed_width_are_admitted_before_solver_work() {
    let j = job(13, 4, false);
    let mut s = options();
    s.max_sector_workspace_bytes = 1;
    let before = iteration_total();
    let error = SectorPlan::qualify(
        &mut operator(&j),
        &j.geom.lattice,
        j.k_path[0],
        PlaneGroup::Pm.group(),
        &s,
    )
    .err()
    .unwrap();
    assert!(error.contains("qualification/basis setup"));
    assert_eq!(iteration_total(), before);

    s.max_sector_workspace_bytes = 1 << 20;
    let reference = solve(&j, &s, None).unwrap();
    assert!(
        reference
            .report
            .workspace
            .as_ref()
            .unwrap()
            .peak_estimated_payload_bytes
            <= s.max_sector_workspace_bytes
    );
    // All 64 inputs are identical. Their original width, not their final rank,
    // determines the temporary projection fields and rank-extraction Gram.
    let warm = vec![Field2D::from_vec(j.grid, vec![C::ONE; j.grid.len()]); 64];
    let before = iteration_total();
    let error = solve(&j, &s, Some(&warm)).err().unwrap();
    assert!(error.contains("sector 0 prefix"), "{error}");
    assert_eq!(iteration_total(), before);

    let small = job(5, 25, true);
    s.max_sector_workspace_bytes = 70_000;
    let before = iteration_total();
    let error = solve(&small, &s, None).err().unwrap();
    assert!(error.contains("prefix"), "{error}");
    assert_eq!(iteration_total(), before);
}

#[test]
fn growth_budget_rejection_preserves_attempted_work_and_falls_back() {
    let j = job(9, 3, true);
    let mut s = options();
    let reference = solve(&j, &s, None).unwrap();
    let memory = reference.report.workspace.as_ref().unwrap();
    assert!(reference.report.allocation_rounds > 1);
    assert!(memory.peak_estimated_payload_bytes > memory.first_round_peak_estimated_payload_bytes);
    s.max_sector_workspace_bytes = memory.first_round_peak_estimated_payload_bytes;
    let before = iteration_total();
    let error = solve(&j, &s, None).err().unwrap();
    assert!(error.contains("allocation round 2"), "{error}");
    assert!(iteration_total() > before);
    let run = |symmetry| {
        let mut points = vec![];
        bandstructure::run_with_k_streaming(
            CpuBackend::<f64>::new(),
            &j,
            RunOptions {
                symmetry,
                reuse_gamma: false,
                retain_eigenvectors: true,
                disable_band_tracking: true,
                ..Default::default()
            },
            |p| points.push(p),
        );
        points.remove(0)
    };
    let fallback = run(s);
    let control = run(SymmetryOptions::default());
    assert!(fallback.converged && fallback.symmetry.sectors.is_none());
    assert!(
        fallback.symmetry.solver_attempts[0].iterations > 0
            && !fallback.symmetry.solver_attempts[0].retained
    );
    assert!(fallback.iterations > control.iterations);
    assert_eq!(
        fallback.iterations,
        fallback
            .symmetry
            .solver_attempts
            .iter()
            .map(|a| a.iterations)
            .sum::<usize>()
    );
    assert_eq!(fallback.omegas, control.omegas);
    assert_eq!(fallback.residuals, control.residuals);
    for (a, b) in fallback
        .eigenvectors
        .unwrap()
        .iter()
        .zip(control.eigenvectors.unwrap())
    {
        assert_eq!(a.as_slice(), b.as_slice());
    }
}

#[test]
fn accepted_admission_caps_preserve_fields_and_finite_reservations_cover_the_report() {
    for storage in [SectorStorage::FullGrid, SectorStorage::Orbit] {
        let mut j = job(9, 6, false);
        j.pol = Polarization::TE;
        let mut s = options();
        s.sector_storage = storage;
        let original = solve(&j, &s, None).unwrap();
        let m = original.report.workspace.as_ref().unwrap();
        let reservation = memory::reservation(
            9,
            9,
            2,
            6,
            storage,
            ReducedKernel::FullGrid,
            s.max_sector_workspace_bytes,
        )
        .unwrap();
        assert!(m.peak_estimated_payload_bytes <= reservation);
        s.max_sector_workspace_bytes = m.peak_estimated_payload_bytes;
        let bounded = solve(&j, &s, None).unwrap();
        assert_eq!(original.spectrum.eigenvalues, bounded.spectrum.eigenvalues);
        assert_eq!(original.iterations, bounded.iterations);
        for (a, b) in original.spectrum.fields.iter().zip(bounded.spectrum.fields) {
            assert_eq!(a.as_slice(), b.as_slice());
        }
    }
    assert!(memory::worker_payload(usize::MAX, usize::MAX, usize::MAX).is_err());
    assert!(memory::setup_payload(usize::MAX, 12, SectorStorage::Orbit).is_err());
    assert!(memory::spectrum_payload(usize::MAX, usize::MAX).is_err());
    // A bounded reservation must not require representing the hypothetical
    // exhausted N² workspace; ordinary wasm32 grids can overflow that bound.
    assert_eq!(
        memory::reservation(
            usize::MAX / 1024,
            1,
            2,
            8,
            SectorStorage::FullGrid,
            ReducedKernel::FullGrid,
            1 << 20
        )
        .unwrap(),
        1 << 20
    );
    assert!(
        memory::reservation(
            usize::MAX,
            2,
            2,
            8,
            SectorStorage::FullGrid,
            ReducedKernel::FullGrid,
            1 << 20
        )
        .is_err()
    );
    assert!(memory::mirror_payload(1, usize::MAX, ReducedKernel::MirrorTmTe).is_err());
    assert!(
        memory::mirror_payload(5, 501, ReducedKernel::MirrorTmTe).unwrap()
            > memory::mirror_payload(501, 5, ReducedKernel::MirrorTmTe).unwrap()
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn owned_workers_honor_the_common_cap_before_starting() {
    use sectors::concurrency::{self, ExecutionBudget};
    let j = job(13, 4, false);
    let mut s = options();
    s.max_sector_workspace_bytes = 1;
    let before = iteration_total();
    let error = concurrency::solve_owned(
        &mut operator(&j),
        &j.geom.lattice,
        j.k_path[0],
        PlaneGroup::Pm.group(),
        &j.eigensolver,
        None,
        &s,
        ExecutionBudget {
            workers: 2,
            logical_cpu_budget: 2,
            ..Default::default()
        },
        CpuBackend::<f64>::new,
    )
    .err()
    .unwrap();
    assert_eq!(error.scheduling.effective_workspace_limit, 1);
    assert!(error.scheduling.attempts.is_empty());
    assert_eq!(iteration_total(), before);
}
