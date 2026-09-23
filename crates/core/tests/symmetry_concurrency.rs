#![cfg(not(target_arch = "wasm32"))]
use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::Dielectric2D,
    eigensolver::{EigensolverConfig, iteration_total},
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    operators::ThetaOperator,
    polarization::Polarization,
    symmetry::{
        affine::PlaneGroup,
        reduced::SectorStorage,
        reuse::SymmetryOptions,
        sectors::{
            SectorPolicy,
            concurrency::{ExecutionBudget, solve_owned},
        },
    },
};

fn operator(n: usize, k: [f64; 2], polarization: Polarization) -> ThetaOperator<CpuBackend<f64>> {
    let lattice = Lattice2D::square(1.0);
    ThetaOperator::new(
        CpuBackend::new(),
        Dielectric2D::from_geometry(
            &Geometry2D {
                lattice,
                eps_bg: 1.0,
                atoms: vec![BasisAtom {
                    pos: [0.0; 2],
                    radius: 0.24,
                    eps_inside: 8.9,
                }],
            },
            Grid2D::new(n, n, 1.0, 1.0),
            &Default::default(),
        ),
        polarization,
        lattice.reciprocal().fractional_to_cartesian(k),
    )
}

#[test]
fn private_worker_scheduling_preserves_fields_and_adaptive_decisions() {
    for (family, n, k, storage, partners) in [
        (
            PlaneGroup::Pm,
            13,
            [0.17, 0.0],
            SectorStorage::FullGrid,
            false,
        ),
        (PlaneGroup::P4m, 9, [0.0; 2], SectorStorage::FullGrid, true),
        (PlaneGroup::P4m, 9, [0.0; 2], SectorStorage::Orbit, true),
    ] {
        for polarization in [Polarization::TM, Polarization::TE] {
            let config = EigensolverConfig {
                n_bands: 6,
                tol: 1e-14,
                max_iter: 500,
                ..Default::default()
            };
            let options = SymmetryOptions {
                sectors: SectorPolicy::AdaptiveGroup,
                sector_guard_budget: Some(6),
                sector_storage: storage,
                reconstruct_partners: partners,
                ..Default::default()
            };
            let run = |workers, cap| {
                let mut op = operator(n, k, polarization);
                let before = iteration_total();
                let output = solve_owned(
                    &mut op,
                    &Lattice2D::square(1.0),
                    k,
                    family.group(),
                    &config,
                    None,
                    &options,
                    ExecutionBudget {
                        workers,
                        logical_cpu_budget: 2,
                        max_workspace_bytes: cap,
                    },
                    CpuBackend::<f64>::new,
                )
                .unwrap_or_else(|e| panic!("{family:?} {polarization:?} {storage:?}: {e}"));
                assert_eq!(
                    iteration_total().wrapping_sub(before),
                    output.scheduling.attempted_iterations
                );
                assert_eq!(
                    output.result.iterations,
                    output.scheduling.attempted_iterations
                );
                assert!(output.result.spectrum.passes(&options));
                assert!(output.scheduling.peak_estimated_payload_bytes <= cap);
                output
            };
            let serial = run(1, 512 * 1024 * 1024);
            let parallel = run(2, 512 * 1024 * 1024);
            assert_eq!(
                serial.result.spectrum.eigenvalues,
                parallel.result.spectrum.eigenvalues
            );
            for (a, b) in serial
                .result
                .spectrum
                .fields
                .iter()
                .zip(&parallel.result.spectrum.fields)
            {
                assert_eq!(
                    a.as_slice(),
                    b.as_slice(),
                    "worker scheduling changed a physical field"
                );
            }
            assert_eq!(serial.result.iterations, parallel.result.iterations);
            assert_eq!(
                serial.result.report.allocation_rounds,
                parallel.result.report.allocation_rounds
            );
            assert_eq!(
                serde_json::to_value(&serial.result.report.sectors).unwrap(),
                serde_json::to_value(&parallel.result.report.sectors).unwrap()
            );
            let shape = |r: &blaze2d_core::symmetry::sectors::concurrency::ConcurrentResult| {
                r.scheduling
                    .attempts
                    .iter()
                    .map(|a| {
                        (
                            a.sector_index,
                            a.allocation_round,
                            a.prefix,
                            a.guards,
                            a.attempted_iterations,
                        )
                    })
                    .collect::<Vec<_>>()
            };
            assert_eq!(shape(&serial), shape(&parallel));
            assert!(parallel.scheduling.maximum_live_workers <= 2);
            assert_eq!(
                parallel.scheduling.effective_worker_limit,
                2.min(parallel.scheduling.available_logical_cpus)
            );
            // The same numerical payload cap as the serial run may force
            // smaller batches; it must never change the selected prefixes.
            let bounded = run(2, serial.scheduling.peak_estimated_payload_bytes);
            assert_eq!(
                serial.result.spectrum.eigenvalues,
                bounded.result.spectrum.eigenvalues
            );
            assert_eq!(shape(&serial), shape(&bounded));
        }
    }
}

#[test]
fn rejected_worker_work_is_counted_and_budgets_reject_before_launch() {
    let k = [0.17, 0.0];
    let lattice = Lattice2D::square(1.0);
    let mut op = operator(13, k, Polarization::TE);
    let config = EigensolverConfig {
        n_bands: 4,
        max_iter: 1,
        tol: 1e-14,
        ..Default::default()
    };
    let options = SymmetryOptions::default();
    let before = iteration_total();
    let error = match solve_owned(
        &mut op,
        &lattice,
        k,
        PlaneGroup::Pm.group(),
        &config,
        None,
        &options,
        ExecutionBudget {
            workers: 2,
            logical_cpu_budget: 2,
            ..Default::default()
        },
        CpuBackend::<f64>::new,
    ) {
        Ok(_) => panic!("one iteration should not produce a certified sector spectrum"),
        Err(e) => e,
    };
    assert!(error.scheduling.attempted_iterations > 0);
    assert_eq!(
        error.scheduling.attempted_iterations,
        iteration_total().wrapping_sub(before)
    );
    assert_eq!(
        error.scheduling.attempted_iterations,
        error
            .scheduling
            .attempts
            .iter()
            .map(|a| a.attempted_iterations)
            .sum::<usize>()
    );
    assert!(
        error
            .scheduling
            .attempts
            .iter()
            .all(|a| a.rejection.is_some())
    );
    assert_eq!(
        error.scheduling.attempts.len(),
        error.scheduling.effective_worker_limit.min(2)
    );
    for budget in [
        ExecutionBudget {
            workers: 0,
            ..Default::default()
        },
        ExecutionBudget {
            logical_cpu_budget: 0,
            ..Default::default()
        },
        ExecutionBudget {
            max_workspace_bytes: 1,
            ..Default::default()
        },
    ] {
        let before = iteration_total();
        let error = match solve_owned(
            &mut op,
            &lattice,
            k,
            PlaneGroup::Pm.group(),
            &config,
            None,
            &options,
            budget,
            CpuBackend::<f64>::new,
        ) {
            Ok(_) => panic!("invalid resource budget accepted"),
            Err(e) => e,
        };
        assert!(error.scheduling.attempts.is_empty());
        assert_eq!(error.scheduling.attempted_iterations, 0);
        assert_eq!(iteration_total(), before);
    }
}
