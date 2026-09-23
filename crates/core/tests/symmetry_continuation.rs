use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::Dielectric2D,
    drivers::single_solve::{SingleSolveJob, solve, solve_with_warmstart},
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    operators::ThetaOperator,
    polarization::Polarization,
    symmetry::{
        continuation::{GeometryContinuationCache, SeedHistory},
        reuse::CachedSpectrum,
    },
};

fn operator(eps: f64, pol: Polarization, k: [f64; 2]) -> ThetaOperator<CpuBackend<f64>> {
    let lattice = Lattice2D::square(1.0);
    let grid = Grid2D::new(9, 9, 1.0, 1.0);
    let dielectric = Dielectric2D::from_geometry(
        &Geometry2D {
            lattice,
            eps_bg: 1.0,
            atoms: vec![BasisAtom {
                pos: [0.0; 2],
                radius: 0.2,
                eps_inside: eps,
            }],
        },
        grid,
        &Default::default(),
    );
    ThetaOperator::new(
        CpuBackend::<f64>::new(),
        dielectric,
        pol,
        lattice.reciprocal().fractional_to_cartesian(k),
    )
}

fn source(op: &mut ThetaOperator<CpuBackend<f64>>, k: [f64; 2]) -> CachedSpectrum {
    let mut p = op.build_homogeneous_preconditioner_adaptive();
    let result = solve(
        op,
        Some(&mut p),
        &SingleSolveJob::new(4)
            .with_tolerance(1e-14)
            .with_max_iterations(500),
    );
    assert!(result.converged);
    CachedSpectrum {
        source_index: 0,
        k,
        eigenvalues: result.eigenvalues,
        fields: result.eigenvectors,
        residuals: Vec::new(),
        absolute_residuals: Vec::new(),
        orthogonality_defect: 0.0,
    }
}

#[test]
fn geometry_union_is_only_a_target_metric_seed_and_keeps_the_outer_problem() {
    let k = [0.13, 0.17];
    for pol in [Polarization::TM, Polarization::TE] {
        let mut cache = GeometryContinuationCache::new(1024 * 1024);
        for (i, eps) in [4.0, 5.0].into_iter().enumerate() {
            let mut op = operator(eps, pol, k);
            let entry = source(&mut op, k);
            assert!(
                cache
                    .insert(
                        &mut op,
                        format!("geometry{i}"),
                        vec![eps],
                        entry,
                        &Default::default()
                    )
                    .unwrap()
                    .stored
            );
        }
        let mut target = operator(6.0, pol, k);
        let prepared = cache
            .prepare(&mut target, SeedHistory::Recent { snapshots: 2 }, 4, 1e-9)
            .unwrap()
            .unwrap();
        assert_eq!(prepared.fields.len(), 4);
        assert_eq!(prepared.report.sources.len(), 2);
        assert_eq!(prepared.report.rank.source_columns, 8);
        assert!(prepared.report.rank.retained_rank > 4);
        assert!(prepared.report.rank.b_orthogonality_defect < 1e-8);
        let reference = source(&mut target, k);
        let mut p = target.build_homogeneous_preconditioner_adaptive();
        let result = solve_with_warmstart(
            &mut target,
            Some(&mut p),
            &SingleSolveJob::new(4)
                .with_tolerance(1e-14)
                .with_max_iterations(500),
            &prepared.fields,
        );
        assert!(result.converged && result.final_residuals.iter().all(|r| *r < 1e-5));
        for (a, b) in reference.eigenvalues.iter().zip(result.eigenvalues) {
            assert!((a - b).abs() / a.abs().max(1.0) < 1e-7);
        }
        let mut other_k = operator(6.0, pol, [0.21, 0.17]);
        assert!(
            cache
                .prepare(&mut other_k, SeedHistory::Latest, 4, 1e-9)
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn redundant_snapshots_compress_by_rank_and_eviction_respects_the_budget() {
    let k = [0.13, 0.17];
    let mut op = operator(5.0, Polarization::TM, k);
    let entry = source(&mut op, k);
    let mut cache = GeometryContinuationCache::new(1024 * 1024);
    let first = cache
        .insert(
            &mut op,
            "a".into(),
            vec![5.0],
            entry.clone(),
            &Default::default(),
        )
        .unwrap();
    let mut rephased = entry.clone();
    for field in &mut rephased.fields {
        for value in field.as_mut_slice() {
            *value *= num_complex::Complex64::new(0.0, 1.0);
        }
    }
    cache
        .insert(
            &mut op,
            "b".into(),
            vec![5.0],
            rephased,
            &Default::default(),
        )
        .unwrap();
    let seed = cache
        .prepare(&mut op, SeedHistory::Recent { snapshots: 2 }, 4, 1e-9)
        .unwrap()
        .unwrap();
    assert_eq!(seed.report.rank.retained_rank, 4);
    assert_eq!(seed.report.rank.source_columns, 8);
    let mut bounded = GeometryContinuationCache::new(first.entry_payload_bytes);
    for id in ["a", "b"] {
        let report = bounded
            .insert(
                &mut op,
                id.into(),
                vec![5.0],
                entry.clone(),
                &Default::default(),
            )
            .unwrap();
        assert!(report.stored);
    }
    assert_eq!(bounded.len(), 1);
    assert!(bounded.payload_bytes() <= first.entry_payload_bytes);
    assert_eq!(
        bounded
            .prepare(&mut op, SeedHistory::Recent { snapshots: 2 }, 4, 1e-9)
            .unwrap()
            .unwrap()
            .report
            .sources,
        vec!["b"]
    );
    let mut zero = GeometryContinuationCache::new(0);
    assert!(
        !zero
            .insert(
                &mut op,
                "a".into(),
                vec![5.0],
                entry.clone(),
                &Default::default()
            )
            .unwrap()
            .stored
    );
    assert!(zero.is_empty());
    let mut broken = entry;
    broken.fields[0].as_mut_slice()[0] = num_complex::Complex64::new(f64::NAN, 0.0);
    assert!(
        cache
            .insert(
                &mut op,
                "invalid".into(),
                vec![5.0],
                broken,
                &Default::default()
            )
            .is_err()
    );
}
