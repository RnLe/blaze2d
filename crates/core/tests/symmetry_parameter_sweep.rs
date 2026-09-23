use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    drivers::single_solve::SingleSolveJob,
    eigensolver::iteration_total,
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    polarization::Polarization,
    symmetry::{
        continuation::{GeometryEviction, SeedHistory},
        parameter::{
            ParameterMetric, ParameterOrder,
            sweep::{ParameterJob, SweepPolicy, run},
        },
    },
};

fn jobs() -> Vec<ParameterJob> {
    [4.0, 6.0, 4.0, 5.0]
        .into_iter()
        .enumerate()
        .map(|(i, eps)| ParameterJob {
            id: format!("j{i}"),
            parameters: vec![eps],
            geometry: Geometry2D {
                lattice: Lattice2D::square(1.0),
                eps_bg: 1.0,
                atoms: vec![BasisAtom {
                    pos: [0.0; 2],
                    radius: 0.2,
                    eps_inside: eps,
                }],
            },
        })
        .collect()
}

#[test]
fn scheduling_and_eviction_keep_the_fixed_physical_jobs_and_fields() {
    let jobs = jobs();
    let grid = Grid2D::new(9, 9, 1.0, 1.0);
    let metric = ParameterMetric::new(vec![1.0]).unwrap();
    let config = SingleSolveJob::new(4)
        .with_tolerance(1e-14)
        .with_max_iterations(500);
    for pol in [Polarization::TM, Polarization::TE] {
        let mut reference = None;
        for (order, eviction, history, budget) in [
            (
                ParameterOrder::Given,
                GeometryEviction::Fifo,
                SeedHistory::Latest,
                0,
            ),
            (
                ParameterOrder::Given,
                GeometryEviction::Fifo,
                SeedHistory::Latest,
                14000,
            ),
            (
                ParameterOrder::Nearest,
                GeometryEviction::Fifo,
                SeedHistory::Latest,
                14000,
            ),
            (
                ParameterOrder::Given,
                GeometryEviction::LeastRecentlyUsed,
                SeedHistory::Latest,
                14000,
            ),
            (
                ParameterOrder::Given,
                GeometryEviction::FarthestFromIncoming {
                    metric: metric.clone(),
                },
                SeedHistory::Latest,
                14000,
            ),
            (
                ParameterOrder::Given,
                GeometryEviction::Fifo,
                SeedHistory::Nearest { snapshots: 1 },
                14000,
            ),
        ] {
            let before = iteration_total();
            let result = run(
                CpuBackend::<f64>::new,
                &jobs,
                grid,
                [0.13, 0.17],
                pol,
                &Default::default(),
                &config,
                &SweepPolicy {
                    order,
                    metric: metric.clone(),
                    eviction,
                    history,
                    exact_reuse: true,
                    max_cache_payload_bytes: budget,
                    max_output_field_bytes: 4 * 4 * grid.len() * 16,
                },
            )
            .unwrap();
            assert!(
                result.report.rejection.is_none(),
                "{:?}",
                result.report.rejection
            );
            assert_eq!(result.report.completed_jobs, jobs.len());
            assert_eq!(
                result.report.iterations,
                iteration_total().wrapping_sub(before)
            );
            assert!(result.report.peak_cache_payload_bytes <= budget);
            assert_eq!(
                result.report.retained_output_field_bytes,
                4 * 4 * grid.len() * 16
            );
            if order == ParameterOrder::Nearest {
                assert!(result.report.exact_cache_hits > 0);
            }
            let values: Vec<Vec<f64>> = result
                .outputs
                .iter()
                .enumerate()
                .map(|(i, o)| {
                    assert_eq!(o.report.original_index, i);
                    assert_eq!(o.report.id, jobs[i].id);
                    assert_eq!(o.spectrum.source_index, i);
                    assert_eq!(o.spectrum.fields.len(), 4);
                    assert!(o.spectrum.passes(&Default::default()));
                    o.spectrum.eigenvalues.clone()
                })
                .collect();
            if let Some(reference) = &reference {
                let reference: &Vec<Vec<f64>> = reference;
                for (a, b) in reference.iter().flatten().zip(values.iter().flatten()) {
                    assert!((a - b).abs() / a.abs().max(1.0) < 1e-7);
                }
            } else {
                reference = Some(values);
            }
        }
    }
}

#[test]
fn failed_sweeps_retain_work_and_output_budgets_reject_before_solving() {
    let jobs = jobs();
    let grid = Grid2D::new(9, 9, 1.0, 1.0);
    let mut policy = SweepPolicy {
        order: ParameterOrder::Nearest,
        metric: ParameterMetric::new(vec![1.0]).unwrap(),
        eviction: GeometryEviction::Fifo,
        history: SeedHistory::Latest,
        exact_reuse: true,
        max_cache_payload_bytes: 14000,
        max_output_field_bytes: 100000,
    };
    let config = SingleSolveJob::new(4)
        .with_tolerance(1e-14)
        .with_max_iterations(1);
    let before = iteration_total();
    let result = run(
        CpuBackend::<f64>::new,
        &jobs,
        grid,
        [0.13, 0.17],
        Polarization::TE,
        &Default::default(),
        &config,
        &policy,
    )
    .unwrap();
    assert!(result.report.rejection.is_some());
    assert_eq!(result.report.failed_job_index, Some(0));
    assert_eq!(result.report.completed_jobs, 0);
    assert_eq!(result.outputs.len(), 1);
    assert_eq!(
        result.report.iterations,
        iteration_total().wrapping_sub(before)
    );
    assert!(result.report.iterations > 0);
    policy.max_output_field_bytes = 1;
    let before = iteration_total();
    assert!(
        run(
            CpuBackend::<f64>::new,
            &jobs,
            grid,
            [0.13, 0.17],
            Polarization::TE,
            &Default::default(),
            &config,
            &policy
        )
        .is_err()
    );
    assert_eq!(before, iteration_total());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn parallel_chains_share_the_total_budget_and_count_failed_work() {
    use blaze2d_core::symmetry::parameter::sweep::concurrency::{
        Sharding, SweepExecutionBudget, run as run_parallel,
    };
    let jobs = jobs();
    let grid = Grid2D::new(9, 9, 1.0, 1.0);
    let policy = SweepPolicy {
        order: ParameterOrder::Nearest,
        metric: ParameterMetric::new(vec![1.0]).unwrap(),
        eviction: GeometryEviction::Fifo,
        history: SeedHistory::Latest,
        exact_reuse: true,
        max_cache_payload_bytes: 28000,
        max_output_field_bytes: 100000,
    };
    for polarization in [Polarization::TM, Polarization::TE] {
        let config = SingleSolveJob::new(4)
            .with_tolerance(1e-14)
            .with_max_iterations(500);
        let reference = run(
            CpuBackend::<f64>::new,
            &jobs,
            grid,
            [0.13, 0.17],
            polarization,
            &Default::default(),
            &config,
            &policy,
        )
        .unwrap();
        for (workers, sharding) in [
            (1, Sharding::RoundRobin),
            (2, Sharding::RoundRobin),
            (2, Sharding::Contiguous),
        ] {
            let before = iteration_total();
            let parallel = run_parallel(
                CpuBackend::<f64>::new,
                &jobs,
                grid,
                [0.13, 0.17],
                polarization,
                &Default::default(),
                &config,
                &policy,
                SweepExecutionBudget {
                    workers,
                    logical_cpu_budget: 2,
                    max_total_payload_bytes: 8 * 1024 * 1024,
                },
                sharding,
            )
            .unwrap();
            let result = parallel.result;
            assert!(
                result.report.rejection.is_none(),
                "{:?}",
                result.report.rejection
            );
            assert_eq!(
                result.report.iterations,
                iteration_total().wrapping_sub(before)
            );
            assert_eq!(
                result.report.iterations,
                parallel
                    .execution
                    .chains
                    .iter()
                    .map(|w| w.iterations)
                    .sum::<usize>()
            );
            assert_eq!(result.report.completed_jobs, jobs.len());
            assert_eq!(
                parallel
                    .execution
                    .chains
                    .iter()
                    .map(|w| w.cache_budget_bytes)
                    .sum::<usize>(),
                policy.max_cache_payload_bytes
            );
            assert!(result.report.peak_cache_payload_bytes <= policy.max_cache_payload_bytes);
            assert_eq!(
                result.report.retained_output_field_bytes,
                reference.report.retained_output_field_bytes
            );
            for (index, (a, b)) in reference.outputs.iter().zip(&result.outputs).enumerate() {
                assert_eq!(b.report.original_index, index);
                assert_eq!(b.report.id, jobs[index].id);
                assert_eq!(b.spectrum.source_index, index);
                assert!(b.spectrum.passes(&Default::default()));
                for (x, y) in a.spectrum.eigenvalues.iter().zip(&b.spectrum.eigenvalues) {
                    assert!((x - y).abs() / x.abs().max(1.0) < 1e-7);
                }
                if workers == 1 {
                    assert_eq!(a.spectrum.eigenvalues, b.spectrum.eigenvalues);
                    for (x, y) in a.spectrum.fields.iter().zip(&b.spectrum.fields) {
                        assert_eq!(x.as_slice(), y.as_slice());
                    }
                }
            }
        }
    }
    let config = SingleSolveJob::new(4)
        .with_tolerance(1e-14)
        .with_max_iterations(1);
    let before = iteration_total();
    let failed = run_parallel(
        CpuBackend::<f64>::new,
        &jobs,
        grid,
        [0.13, 0.17],
        Polarization::TE,
        &Default::default(),
        &config,
        &policy,
        SweepExecutionBudget {
            workers: 2,
            logical_cpu_budget: 2,
            max_total_payload_bytes: 8 * 1024 * 1024,
        },
        Sharding::RoundRobin,
    )
    .unwrap();
    assert!(failed.result.report.rejection.is_some());
    assert_eq!(
        failed.result.report.iterations,
        iteration_total().wrapping_sub(before)
    );
    assert_eq!(
        failed.result.outputs.len(),
        failed.execution.effective_workers
    );
    assert!(
        failed
            .execution
            .chains
            .iter()
            .all(|w| w.iterations > 0 && w.rejection.is_some())
    );
    let before = iteration_total();
    assert!(
        run_parallel(
            CpuBackend::<f64>::new,
            &jobs,
            grid,
            [0.13, 0.17],
            Polarization::TE,
            &Default::default(),
            &config,
            &policy,
            SweepExecutionBudget {
                workers: 2,
                logical_cpu_budget: 2,
                max_total_payload_bytes: 1
            },
            Sharding::Contiguous
        )
        .is_err()
    );
    assert_eq!(before, iteration_total());
}
