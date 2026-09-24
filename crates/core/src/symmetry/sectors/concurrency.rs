//! Bounded independent-sector scheduling with unchanged round decisions.
//! Workers create private backends/operators/preconditioners from a factory.
//! A shallow backend clone may share FFT scratch and is not a private worker.
use super::{
    SectorPlan, SectorReport, SectorResult, allocate_guards, finish_sectors, grow_prefixes,
    initial_prefixes, is_adaptive, memory, solve_prefix,
};
use crate::{
    backend::SpectralBackend,
    eigensolver::{EigensolverConfig, absorb_worker_iterations, iteration_total},
    field::Field2D,
    lattice::Lattice2D,
    operators::{LinearOperator, ThetaOperator},
    symmetry::{
        affine::AffineGroup,
        reduced::{OrbitBasis, SectorStorage},
        reuse::{CachedSpectrum, SymmetryOptions},
    },
};
use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ExecutionBudget {
    pub workers: usize,
    pub logical_cpu_budget: usize,
    /// Conservative numerical payload estimate, including retained results and
    /// simultaneous worker workspaces. Backend/allocator RSS is not a hard cap.
    pub max_workspace_bytes: usize,
}
impl Default for ExecutionBudget {
    fn default() -> Self {
        Self {
            workers: 1,
            logical_cpu_budget: 1,
            max_workspace_bytes: 512 * 1024 * 1024,
        }
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct WorkerAttemptReport {
    pub sector_index: usize,
    pub allocation_round: usize,
    pub prefix: usize,
    pub guards: usize,
    pub attempted_iterations: usize,
    pub worker_seconds: f64,
    pub estimated_worker_payload_bytes: usize,
    pub rejection: Option<String>,
}
#[derive(Debug, Clone, Serialize)]
pub struct ConcurrencyReport {
    pub budget: ExecutionBudget,
    pub effective_workspace_limit: usize,
    pub payload_model: &'static str,
    pub available_logical_cpus: usize,
    pub effective_worker_limit: usize,
    pub maximum_live_workers: usize,
    pub batches: usize,
    pub peak_estimated_payload_bytes: usize,
    pub peak_retained_spectrum_payload_bytes: usize,
    pub attempted_iterations: usize,
    pub summed_worker_seconds: f64,
    pub attempts: Vec<WorkerAttemptReport>,
}
pub struct ConcurrentResult {
    pub result: SectorResult,
    pub scheduling: ConcurrencyReport,
}
#[derive(Debug, Clone, Serialize, thiserror::Error)]
#[error("{reason}")]
pub struct ConcurrencyError {
    pub reason: String,
    pub scheduling: ConcurrencyReport,
}
struct Attempt {
    result: Result<(CachedSpectrum, SectorReport), String>,
    iterations: usize,
    seconds: f64,
}

/// Serial-owned (`workers=1`) and concurrent execution use identical private
/// backend lifetimes and numerical inputs. The factory must create independent
/// scratch, e.g. `CpuBackend::new`; `CpuBackend::clone` shares an FFT scratch lock.
///
/// Adaptive prefixes and guards are decided serially between complete rounds.
/// Completed batches merge in sector order. All started ordinary failures are
/// joined and counted before returning an error. Worker panics propagate.
#[allow(clippy::too_many_arguments)]
pub fn solve_owned<B, F>(
    operator: &mut ThetaOperator<B>,
    lattice: &Lattice2D,
    k: [f64; 2],
    group: AffineGroup,
    config: &EigensolverConfig,
    warm: Option<&[Field2D]>,
    options: &SymmetryOptions,
    budget: ExecutionBudget,
    factory: F,
) -> Result<ConcurrentResult, ConcurrencyError>
where
    B: SpectralBackend + Clone,
    F: Fn() -> B + Sync,
{
    crate::profile_scope!("symmetry.concurrent_sectors");
    crate::init_faer_sequential();
    let available = std::thread::available_parallelism().map_or(1, usize::from);
    let mut report = ConcurrencyReport {
        effective_workspace_limit: budget
            .max_workspace_bytes
            .min(options.max_sector_workspace_bytes),
        payload_model: memory::MODEL,
        budget,
        available_logical_cpus: available,
        effective_worker_limit: budget.workers.min(budget.logical_cpu_budget).min(available),
        maximum_live_workers: 0,
        batches: 0,
        peak_estimated_payload_bytes: 0,
        peak_retained_spectrum_payload_bytes: 0,
        attempted_iterations: 0,
        summed_worker_seconds: 0.0,
        attempts: Vec::new(),
    };
    let attempt = run_owned(
        operator,
        lattice,
        k,
        group,
        config,
        warm,
        options,
        &factory,
        &mut report,
    );
    match attempt {
        Ok(result) => Ok(ConcurrentResult {
            result,
            scheduling: report,
        }),
        Err(reason) => Err(ConcurrencyError {
            reason,
            scheduling: report,
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_owned<B, F>(
    operator: &mut ThetaOperator<B>,
    lattice: &Lattice2D,
    k: [f64; 2],
    group: AffineGroup,
    config: &EigensolverConfig,
    warm: Option<&[Field2D]>,
    options: &SymmetryOptions,
    factory: &F,
    report: &mut ConcurrencyReport,
) -> Result<SectorResult, String>
where
    B: SpectralBackend + Clone,
    F: Fn() -> B + Sync,
{
    let grid = operator.grid();
    if report.effective_worker_limit == 0
        || report.effective_workspace_limit == 0
        || config.n_bands == 0
        || config.n_bands > grid.len()
        || config.max_iter == 0
        || !config.tol.is_finite()
        || config.tol <= 0.0
        || options.correction_projection_interval == 0
    {
        return Err("invalid concurrent sector configuration or resource budget".into());
    }
    if options.correction_projection_interval != 1
        && options.sector_storage != SectorStorage::FullGrid
    {
        return Err("correction projection cadence applies only to full-grid sectors".into());
    }
    if (options.arithmetic != crate::symmetry::fourier::Arithmetic::SpatialComplex && options.sector_storage!=SectorStorage::FullGrid)
        || options.share_sector_workspace
        || (options.sector_storage == SectorStorage::FullGrid
            && options.sector_kernel != crate::symmetry::reduced::ReducedKernel::FullGrid)
    {
        return Err(
            "owned workers require private buffers; Fourier arithmetic requires full-grid sectors".into(),
        );
    }
    if warm.is_some_and(|fields| {
        fields.iter().any(|f| {
            let g = f.grid();
            (g.nx, g.ny, g.centered, g.lx.to_bits(), g.ly.to_bits())
                != (
                    grid.nx,
                    grid.ny,
                    grid.centered,
                    grid.lx.to_bits(),
                    grid.ly.to_bits(),
                )
                || f.as_slice()
                    .iter()
                    .any(|z| !z.re.is_finite() || !z.im.is_finite())
        })
    }) {
        return Err("invalid concurrent sector warm fields".into());
    }
    // Bound table/basis setup before constructing it. This is a payload model,
    // not an assertion about the allocator's process high-water mark.
    let group_order = group.operations.len();
    let qualification_bound = memory::sum(&[
        memory::setup_payload(grid.len(), group_order, options.sector_storage)?,
        memory::coordinate_setup_payload(grid.len(),super::uses_coordinates(options))?,
    ])?;
    report.peak_estimated_payload_bytes = qualification_bound;
    if qualification_bound > report.effective_workspace_limit {
        return Err("qualification/basis setup exceeds concurrent payload budget".into());
    }
    let plan = SectorPlan::qualify(operator, lattice, k, group, options)?;
    let bases = if options.sector_storage == SectorStorage::Orbit {
        plan.sectors
            .iter()
            .map(|s| {
                if s.rank == 0 {
                    Ok(None)
                } else {
                    OrbitBasis::from_projector(&plan.actions, &s.weights, s.rank).map(Some)
                }
            })
            .collect::<Result<Vec<_>, String>>()?
    } else {
        (0..plan.sectors.len()).map(|_| None).collect()
    };
    let fixed_payload = memory::sum(&[
        qualification_bound,
        memory::spectrum_payload(grid.len(), warm.map_or(0, <[Field2D]>::len))?,
    ])?;
    let kernel_payload = memory::mirror_payload(grid.nx, grid.ny, options.sector_kernel)?;
    if fixed_payload > report.effective_workspace_limit {
        return Err(
            "shared qualification/result workspace exceeds concurrent payload budget".into(),
        );
    }
    let adaptive = is_adaptive(options);
    let mut prefixes = initial_prefixes(&plan, config.n_bands, adaptive);
    let mut slots: Vec<Option<(CachedSpectrum, SectorReport)>> =
        (0..plan.sectors.len()).map(|_| None).collect();
    let mut rounds = 0;
    let mut allocations = 0;
    loop {
        rounds += 1;
        let guards = allocate_guards(&prefixes, options.sector_guard_budget, grid.len());
        let ready: Vec<_> = plan
            .sectors
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                (s.rank > 0
                    && slots[i]
                        .as_ref()
                        .is_none_or(|(_, r)| r.prefix != prefixes[i]))
                .then_some(i)
            })
            .collect();
        let mut cursor = 0;
        while cursor < ready.len() {
            let retained = slots
                .iter()
                .flatten()
                .try_fold(0usize, |n, (e, _)| n.checked_add(e.bytes()))
                .ok_or("payload overflow")?;
            report.peak_retained_spectrum_payload_bytes =
                report.peak_retained_spectrum_payload_bytes.max(retained);
            let shared = fixed_payload
                .checked_add(retained)
                .ok_or("payload overflow")?;
            let mut batch = Vec::new();
            let mut payload = shared;
            while cursor < ready.len() && batch.len() < report.effective_worker_limit {
                let i = ready[cursor];
                let seed = slots[i].as_ref().map(|(e, _)| e.fields.as_slice()).or(warm);
                let dense = plan.sectors[i].rank <= prefixes[i].saturating_add(2);
                let columns = if dense {
                    plan.sectors[i].rank
                } else {
                    prefixes[i]
                        .saturating_add(guards[i])
                        .min(plan.sectors[i].rank - usize::from(plan.sectors[i].owns_kernel))
                };
                let estimate = memory::sum(&[
                    memory::worker_payload(grid.len(), columns, seed.map_or(0, <[Field2D]>::len))?,
                    memory::coordinate_worker_payload(grid.len(),seed.map_or(0,<[Field2D]>::len),super::uses_coordinates(options))?,
                    kernel_payload,
                ])?;
                let combined = payload.checked_add(estimate).ok_or("payload overflow")?;
                if combined > report.effective_workspace_limit {
                    if batch.is_empty() {
                        return Err(format!(
                            "sector {i} needs estimated payload {combined}, above budget {}",
                            report.effective_workspace_limit
                        ));
                    }
                    break;
                }
                batch.push((i, estimate));
                payload = combined;
                cursor += 1;
            }
            report.batches += 1;
            report.maximum_live_workers = report.maximum_live_workers.max(batch.len());
            report.peak_estimated_payload_bytes = report.peak_estimated_payload_bytes.max(payload);
            let material = operator.dielectric();
            let polarization = operator.polarization();
            let bloch = operator.bloch();
            let run = |i: usize| {
                let start = std::time::Instant::now();
                let before = iteration_total();
                let result = {
                    crate::profile_scope!("symmetry.sector_worker");
                    crate::profiler::count("sector.worker_jobs", 1);
                    let mut child =
                        ThetaOperator::new(factory(), material.clone(), polarization, bloch);
                    let mut preconditioner = child.build_homogeneous_preconditioner_adaptive();
                    let output = child.alloc_field();
                    let scratch = child.alloc_field();
                    crate::profiler::count("sector.projection_workspace_allocations", 2);
                    let seed = slots[i].as_ref().map(|(e, _)| e.fields.as_slice()).or(warm);
                    solve_prefix(
                        &mut child,
                        &plan,
                        i,
                        config,
                        prefixes[i],
                        guards[i],
                        rounds,
                        k,
                        seed,
                        options,
                        bases[i].as_ref(),
                        Some(&mut preconditioner),
                        output,
                        scratch,
                    )
                    .map(|(e, r, _)| (e, r))
                };
                Attempt {
                    result,
                    iterations: iteration_total().wrapping_sub(before),
                    seconds: start.elapsed().as_secs_f64(),
                }
            };
            let parallel = batch.len() > 1;
            let attempts: Vec<_> = if parallel {
                std::thread::scope(|scope| {
                    let handles: Vec<_> = batch
                        .iter()
                        .map(|&(i, _)| {
                            let run = &run;
                            scope.spawn(move || run(i))
                        })
                        .collect();
                    handles
                        .into_iter()
                        .map(|h| {
                            h.join()
                                .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
                        })
                        .collect()
                })
            } else {
                vec![run(batch[0].0)]
            };
            let mut rejection = None;
            for ((i, estimate), attempt) in batch.into_iter().zip(attempts) {
                if parallel {
                    absorb_worker_iterations(attempt.iterations);
                }
                report.attempted_iterations += attempt.iterations;
                report.summed_worker_seconds += attempt.seconds;
                allocations += 2;
                report.attempts.push(WorkerAttemptReport {
                    sector_index: i,
                    allocation_round: rounds,
                    prefix: prefixes[i],
                    guards: guards[i],
                    attempted_iterations: attempt.iterations,
                    worker_seconds: attempt.seconds,
                    estimated_worker_payload_bytes: estimate,
                    rejection: attempt.result.as_ref().err().cloned(),
                });
                match attempt.result {
                    Ok((entry, mut current)) => {
                        if let Some((_, previous)) = &slots[i] {
                            current.iterations += previous.iterations;
                            current.correction_projection_rounds +=
                                previous.correction_projection_rounds;
                            current.correction_projection_skipped_rounds +=
                                previous.correction_projection_skipped_rounds;
                        }
                        slots[i] = Some((entry, current));
                    }
                    Err(reason) => {
                        if rejection.is_none() {
                            rejection = Some(reason);
                        }
                    }
                }
            }
            if let Some(reason) = rejection {
                return Err(reason);
            }
        }
        if !grow_prefixes(&slots, &mut prefixes, config.n_bands, adaptive) {
            break;
        }
    }
    let retained = slots.iter().flatten().map(|(e, _)| e.bytes()).sum();
    report.peak_retained_spectrum_payload_bytes =
        report.peak_retained_spectrum_payload_bytes.max(retained);
    // Full partner reconstruction and fresh certification remain parent work.
    // Reserve copies of retained fields and output before invoking the common merge.
    let reconstructed = slots
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.as_ref().map(|(e, _)| (i, e)))
        .try_fold(0usize, |n, (i, e)| {
            e.bytes()
                .checked_mul(plan.sectors[i].partners)?
                .checked_add(n)
        })
        .ok_or("partner output payload overflow")?;
    let merge_payload =
        memory::merge_payload(fixed_payload, grid.len(), config.n_bands, reconstructed)?;
    report.peak_estimated_payload_bytes = report.peak_estimated_payload_bytes.max(merge_payload);
    if merge_payload > report.effective_workspace_limit {
        return Err("sector output reconstruction exceeds concurrent payload budget".into());
    }
    finish_sectors(
        operator,
        &plan,
        slots,
        k,
        config.n_bands,
        adaptive,
        options,
        report.attempted_iterations,
        rounds,
        allocations,
        report.maximum_live_workers * grid.len() * 32,
    )
}
