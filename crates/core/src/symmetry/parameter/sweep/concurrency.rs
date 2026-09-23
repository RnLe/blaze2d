//! Parallel parameter chains with a fixed aggregate cache/output allowance.
//! Each worker receives only its assigned jobs and cache share; no output field
//! is borrowed as free history from another chain. Changing chains can increase
//! cold starts and numerical work even when total elapsed time improves.
use super::{
    ParameterJob, ParameterOutput, ParameterSweepResult, SweepPolicy, SweepReport, prepare,
};
use crate::{
    backend::SpectralBackend,
    dielectric::DielectricOptions,
    drivers::single_solve::SingleSolveJob,
    eigensolver::{EigensolverConfig, absorb_worker_iterations, iteration_total},
    grid::Grid2D,
    polarization::Polarization,
    symmetry::{continuation::SeedHistory, parameter::ParameterOrder},
    timing::Timer,
};
use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Sharding {
    Contiguous,
    RoundRobin,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct SweepExecutionBudget {
    pub workers: usize,
    pub logical_cpu_budget: usize,
    /// Conservative numerical payload model, not a hard process-RSS limit.
    pub max_total_payload_bytes: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChainReport {
    pub original_indices: Vec<usize>,
    pub cache_budget_bytes: usize,
    pub output_budget_bytes: usize,
    pub seconds: f64,
    pub iterations: usize,
    pub peak_cache_payload_bytes: usize,
    pub cold_starts: usize,
    pub exact_cache_hits: usize,
    pub seed_cache_hits: usize,
    pub completed_jobs: usize,
    pub rejection: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ParallelReport {
    pub budget: SweepExecutionBudget,
    pub sharding: Sharding,
    pub available_logical_cpus: usize,
    pub effective_workers: usize,
    pub estimated_aggregate_payload_bytes: usize,
    pub summed_chain_seconds: f64,
    pub longest_chain_seconds: f64,
    pub chains: Vec<ChainReport>,
}

pub struct ParallelSweepResult {
    pub result: ParameterSweepResult,
    pub execution: ParallelReport,
}

#[allow(clippy::too_many_arguments)]
pub fn run<B, F>(
    factory: F,
    jobs: &[ParameterJob],
    grid: Grid2D,
    k: [f64; 2],
    polarization: Polarization,
    dielectric_options: &DielectricOptions,
    config: &SingleSolveJob,
    policy: &SweepPolicy,
    budget: SweepExecutionBudget,
    sharding: Sharding,
) -> Result<ParallelSweepResult, String>
where
    B: SpectralBackend + Clone,
    F: Fn() -> B + Sync,
{
    crate::profile_scope!("parameter_sweep.parallel");
    crate::init_faer_sequential();
    let timer = Timer::start();
    let (route, reserved) = prepare::<B>(jobs, grid, k, config, policy)?;
    let available = std::thread::available_parallelism().map_or(1, usize::from);
    let mut workers = budget
        .workers
        .min(budget.logical_cpu_budget)
        .min(available)
        .min(jobs.len().max(1));
    if workers == 0 || budget.max_total_payload_bytes == 0 {
        return Err("invalid parameter worker budget".into());
    }
    let estimate = |workers: usize| -> Result<usize, String> {
        let columns = EigensolverConfig {
            n_bands: config.n_bands,
            block_size: config.block_size,
            ..Default::default()
        }
        .effective_block_size()
        .min(grid.len());
        let snapshots = match policy.history {
            SeedHistory::Latest => 1,
            SeedHistory::Recent { snapshots } | SeedHistory::Nearest { snapshots } => snapshots,
        };
        let per_field = grid
            .len()
            .checked_mul(16)
            .ok_or("worker payload overflow")?;
        let resident_columns = policy.max_cache_payload_bytes.div_ceil(workers) / per_field;
        let source_columns = snapshots
            .saturating_mul(config.n_bands)
            .min(resident_columns);
        let vectors = columns
            .checked_mul(24)
            .and_then(|n| source_columns.checked_mul(2)?.checked_add(n))
            .and_then(|n| n.checked_add(32))
            .ok_or("worker payload overflow")?;
        let dense_columns = columns.max(source_columns);
        let per_worker = per_field
            .checked_mul(vectors)
            .and_then(|n| {
                dense_columns
                    .checked_mul(dense_columns)?
                    .checked_mul(1024)?
                    .checked_add(n)
            })
            .ok_or("worker payload overflow")?;
        per_worker
            .checked_mul(workers)
            .and_then(|n| n.checked_add(policy.max_cache_payload_bytes))
            .and_then(|n| n.checked_add(reserved))
            .ok_or_else(|| "aggregate sweep payload overflow".into())
    };
    while workers > 1 && estimate(workers)? > budget.max_total_payload_bytes {
        workers -= 1;
    }
    let payload = estimate(workers)?;
    if payload > budget.max_total_payload_bytes {
        return Err("parameter chains exceed estimated aggregate payload budget".into());
    }
    let mut assignments = vec![Vec::new(); workers];
    match sharding {
        Sharding::RoundRobin => {
            for (i, &index) in route.order.iter().enumerate() {
                assignments[i % workers].push(index);
            }
        }
        Sharding::Contiguous => {
            let mut offset = 0;
            for (i, group) in assignments.iter_mut().enumerate() {
                let count = jobs.len() / workers + usize::from(i < jobs.len() % workers);
                group.extend_from_slice(&route.order[offset..offset + count]);
                offset += count;
            }
        }
    }
    let execute = |worker: usize| {
        let start = Timer::start();
        let before = iteration_total();
        let assigned: Vec<_> = assignments[worker]
            .iter()
            .map(|&i| jobs[i].clone())
            .collect();
        let mut inner = policy.clone();
        inner.order = ParameterOrder::Given;
        inner.max_cache_payload_bytes = policy.max_cache_payload_bytes / workers
            + usize::from(worker < policy.max_cache_payload_bytes % workers);
        inner.max_output_field_bytes = assigned.len() * grid.len() * config.n_bands * 16;
        let result = super::run(
            &factory,
            &assigned,
            grid,
            k,
            polarization,
            dielectric_options,
            config,
            &inner,
        );
        (
            result,
            start.elapsed_secs(),
            iteration_total().wrapping_sub(before),
            inner,
        )
    };
    let attempts: Vec<_> = if workers == 1 {
        vec![execute(0)]
    } else {
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|i| {
                    let execute = &execute;
                    scope.spawn(move || execute(i))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|p| std::panic::resume_unwind(p)))
                .collect()
        })
    };
    let mut report = SweepReport {
        route,
        policy: policy.clone(),
        requested_jobs: jobs.len(),
        completed_jobs: 0,
        solved_jobs: 0,
        exact_cache_hits: 0,
        seed_cache_hits: 0,
        cold_starts: 0,
        evicted_entries: 0,
        iterations: 0,
        peak_cache_payload_bytes: 0,
        reserved_output_field_bytes: reserved,
        retained_output_field_bytes: 0,
        summed_job_seconds: 0.0,
        seconds: 0.0,
        failed_job_index: None,
        rejection: None,
    };
    let mut outputs: Vec<ParameterOutput> = Vec::new();
    let mut execution = ParallelReport {
        budget,
        sharding,
        available_logical_cpus: available,
        effective_workers: workers,
        estimated_aggregate_payload_bytes: payload,
        summed_chain_seconds: 0.0,
        longest_chain_seconds: 0.0,
        chains: Vec::new(),
    };
    for (worker, (attempt, seconds, iterations, inner)) in attempts.into_iter().enumerate() {
        if workers > 1 {
            absorb_worker_iterations(iterations);
        }
        report.iterations += iterations;
        execution.summed_chain_seconds += seconds;
        execution.longest_chain_seconds = execution.longest_chain_seconds.max(seconds);
        let map = &assignments[worker];
        let mut chain = ChainReport {
            original_indices: map.clone(),
            cache_budget_bytes: inner.max_cache_payload_bytes,
            output_budget_bytes: inner.max_output_field_bytes,
            seconds,
            iterations,
            peak_cache_payload_bytes: 0,
            cold_starts: 0,
            exact_cache_hits: 0,
            seed_cache_hits: 0,
            completed_jobs: 0,
            rejection: None,
        };
        match attempt {
            Ok(mut part) => {
                let r = &part.report;
                chain.peak_cache_payload_bytes = r.peak_cache_payload_bytes;
                chain.cold_starts = r.cold_starts;
                chain.exact_cache_hits = r.exact_cache_hits;
                chain.seed_cache_hits = r.seed_cache_hits;
                chain.completed_jobs = r.completed_jobs;
                chain.rejection = r.rejection.clone();
                report.completed_jobs += r.completed_jobs;
                report.solved_jobs += r.solved_jobs;
                report.exact_cache_hits += r.exact_cache_hits;
                report.seed_cache_hits += r.seed_cache_hits;
                report.cold_starts += r.cold_starts;
                report.evicted_entries += r.evicted_entries;
                // Sum of worker high-water payloads is a conservative bound on
                // simultaneous resident cache; it is not a sampled RSS trace.
                report.peak_cache_payload_bytes += r.peak_cache_payload_bytes;
                report.retained_output_field_bytes += r.retained_output_field_bytes;
                report.summed_job_seconds += r.summed_job_seconds;
                if report.rejection.is_none() && r.rejection.is_some() {
                    report.rejection = r.rejection.clone();
                    report.failed_job_index = r.failed_job_index.map(|i| map[i]);
                }
                for out in &mut part.outputs {
                    let original = map[out.report.original_index];
                    out.report.original_index = original;
                    out.spectrum.source_index = original;
                    out.report.previous_execution_index =
                        out.report.previous_execution_index.map(|i| map[i]);
                    // Job positions are local to their chain. The chain index
                    // map in execution reports defines the complete schedule.
                }
                outputs.extend(part.outputs);
            }
            Err(reason) => {
                chain.rejection = Some(reason.clone());
                if report.rejection.is_none() {
                    report.rejection = Some(reason);
                    report.failed_job_index = map.first().copied();
                }
            }
        }
        execution.chains.push(chain);
    }
    outputs.sort_by_key(|o| o.report.original_index);
    report.seconds = timer.elapsed_secs();
    Ok(ParallelSweepResult {
        result: ParameterSweepResult { outputs, report },
        execution,
    })
}
