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
        continuation::{GeometryContinuationCache, GeometryEviction, SeedHistory},
        parameter::ParameterMetric,
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

#[test]
fn eviction_controls_preserve_exact_identity_and_target_metric_seeds() {
    let k = [0.13, 0.17];
    let metric = ParameterMetric::new(vec![1.0]).unwrap();
    let mut ops: Vec<_> = [4.0, 6.0, 8.0]
        .into_iter()
        .map(|p| operator(p, Polarization::TM, k))
        .collect();
    let entries: Vec<_> = ops.iter_mut().map(|op| source(op, k)).collect();
    let mut probe = GeometryContinuationCache::new(usize::MAX);
    let width = probe
        .insert(
            &mut ops[0],
            "a".into(),
            vec![4.0],
            entries[0].clone(),
            &Default::default(),
        )
        .unwrap()
        .entry_payload_bytes;
    for (policy, expected_victim) in [
        (GeometryEviction::Fifo, "a"),
        (GeometryEviction::LeastRecentlyUsed, "b"),
        (
            GeometryEviction::FarthestFromIncoming {
                metric: metric.clone(),
            },
            "a",
        ),
    ] {
        let mut cache = GeometryContinuationCache::with_eviction(2 * width, policy);
        for i in 0..2 {
            cache
                .insert(
                    &mut ops[i],
                    ["a", "b"][i].into(),
                    vec![[4.0, 6.0][i]],
                    entries[i].clone(),
                    &Default::default(),
                )
                .unwrap();
        }
        let exact = cache
            .reuse_exact(&mut ops[0], k, 4, &Default::default())
            .unwrap()
            .unwrap();
        assert_eq!(exact.source_id, "a");
        assert_eq!(exact.spectrum.eigenvalues, entries[0].eigenvalues);
        assert!(exact.spectrum.passes(&Default::default()));
        assert!(
            cache
                .reuse_exact(&mut ops[2], k, 4, &Default::default())
                .unwrap()
                .is_none()
        );
        assert!(
            cache
                .reuse_exact(&mut ops[0], k, 3, &Default::default())
                .unwrap()
                .is_none()
        );
        let mut other_k = operator(4.0, Polarization::TM, [0.21, 0.17]);
        assert!(
            cache
                .reuse_exact(&mut other_k, [0.21, 0.17], 4, &Default::default())
                .unwrap()
                .is_none()
        );
        let insertion = cache
            .insert(
                &mut ops[2],
                "c".into(),
                vec![8.0],
                entries[2].clone(),
                &Default::default(),
            )
            .unwrap();
        assert_eq!(insertion.evicted_ids, [expected_victim]);
        assert!(cache.payload_bytes() <= 2 * width);
    }

    let mut cache =
        GeometryContinuationCache::with_eviction(2 * width, GeometryEviction::LeastRecentlyUsed);
    for i in 0..2 {
        cache
            .insert(
                &mut ops[i],
                ["a", "b"][i].into(),
                vec![[4.0, 6.0][i]],
                entries[i].clone(),
                &Default::default(),
            )
            .unwrap();
    }
    let mut target = operator(4.2, Polarization::TM, k);
    let seed = cache
        .prepare_tracked(
            &mut target,
            &[4.2],
            &metric,
            SeedHistory::Nearest { snapshots: 1 },
            4,
            1e-9,
        )
        .unwrap()
        .unwrap();
    assert_eq!(seed.report.sources, ["a"]);
    assert!(seed.report.rank.b_orthogonality_defect < 1e-8);
    let mut preconditioner = target.build_homogeneous_preconditioner_adaptive();
    let corrected = solve_with_warmstart(
        &mut target,
        Some(&mut preconditioner),
        &SingleSolveJob::new(4)
            .with_tolerance(1e-14)
            .with_max_iterations(500),
        &seed.fields,
    );
    assert!(corrected.converged && corrected.final_residuals.iter().all(|r| *r < 1e-5));
    // A tracked seed access refreshes LRU, but cannot convert changed material
    // into an exact hit. Equal parameter labels alone are insufficient.
    assert!(
        cache
            .reuse_exact(&mut target, k, 4, &Default::default())
            .unwrap()
            .is_none()
    );
    let insertion = cache
        .insert(
            &mut ops[2],
            "c".into(),
            vec![8.0],
            entries[2].clone(),
            &Default::default(),
        )
        .unwrap();
    assert_eq!(insertion.evicted_ids, ["b"]);
    let ids = cache
        .resident_ids()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let mut broken = entries[0].clone();
    broken.fields[0].as_mut_slice()[0].re = f64::NAN;
    assert!(
        cache
            .insert(
                &mut ops[0],
                "bad".into(),
                vec![4.0],
                broken,
                &Default::default()
            )
            .is_err()
    );
    assert_eq!(
        cache.resident_ids(),
        ids.iter().map(String::as_str).collect::<Vec<_>>()
    );
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
