//! Complete serial parameter sweeps under fixed cache/output payload budgets.
//! Every job returns physical fields in the original requested order. Output
//! retention never supplies extra free seed history outside the bounded cache.
use super::{ParameterMetric, ParameterOrder, ParameterRoute};
use crate::{
    backend::SpectralBackend,
    dielectric::{Dielectric2D, DielectricOptions},
    drivers::single_solve::{SingleSolveJob, solve, solve_with_warmstart},
    eigensolver::{iteration_total, refine::certify_without_refinement},
    geometry::Geometry2D,
    grid::Grid2D,
    operators::ThetaOperator,
    polarization::Polarization,
    symmetry::{
        continuation::{
            ContinuationReport, GeometryContinuationCache, GeometryEviction, InsertionReport,
            SeedHistory,
        },
        reuse::{CachedSpectrum, SymmetryOptions},
    },
    timing::Timer,
};
use serde::Serialize;
use std::collections::BTreeSet;

#[cfg(not(target_arch = "wasm32"))]
pub mod concurrency;

#[derive(Debug, Clone)]
pub struct ParameterJob {
    pub id: String,
    pub parameters: Vec<f64>,
    pub geometry: Geometry2D,
}

#[derive(Debug, Clone, Serialize)]
pub struct SweepPolicy {
    pub order: ParameterOrder,
    pub metric: ParameterMetric,
    pub eviction: GeometryEviction,
    pub history: SeedHistory,
    pub exact_reuse: bool,
    pub max_cache_payload_bytes: usize,
    pub max_output_field_bytes: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct JobReport {
    pub original_index: usize,
    pub execution_position: usize,
    pub id: String,
    pub parameters: Vec<f64>,
    pub previous_execution_index: Option<usize>,
    pub iterations: usize,
    pub seconds: f64,
    pub predictor_seconds: f64,
    pub exact_source: Option<String>,
    pub predictor: Option<ContinuationReport>,
    pub insertion: Option<InsertionReport>,
    pub cache_payload_bytes: usize,
    pub converged: bool,
    pub certified: bool,
    /// Coordinate distance is a continuation diagnostic, not an error bound.
    pub nearest_completed_parameter_distance: Option<f64>,
    pub selected_parameter_distance: Option<f64>,
}

pub struct ParameterOutput {
    pub spectrum: CachedSpectrum,
    pub report: JobReport,
}

#[derive(Debug, Clone, Serialize)]
pub struct SweepReport {
    pub route: ParameterRoute,
    pub policy: SweepPolicy,
    pub requested_jobs: usize,
    pub completed_jobs: usize,
    pub solved_jobs: usize,
    pub exact_cache_hits: usize,
    pub seed_cache_hits: usize,
    pub cold_starts: usize,
    pub evicted_entries: usize,
    pub iterations: usize,
    pub peak_cache_payload_bytes: usize,
    pub reserved_output_field_bytes: usize,
    pub retained_output_field_bytes: usize,
    /// Sum of per-job elapsed time. Jobs form a serial dependency chain here.
    pub summed_job_seconds: f64,
    pub seconds: f64,
    pub failed_job_index: Option<usize>,
    pub rejection: Option<String>,
}

pub struct ParameterSweepResult {
    /// Original requested order; partial failures retain all available fields.
    pub outputs: Vec<ParameterOutput>,
    pub report: SweepReport,
}

/// The reference scheduler is serial. Its complete elapsed time includes route
/// construction, every geometry/operator, source solve, cache certificate,
/// target preparation, correction and output reordering. The same policy can
/// run on native or portable f64 backends; this does not create native threads.
///
/// Invalid configuration rejects before any solve. A numerical failure returns
/// its available fields and prior work with `report.rejection`, and never seeds
/// later jobs. No partial run is marked as the complete requested sweep.
#[allow(clippy::too_many_arguments)]
pub fn run<B, F>(
    mut factory: F,
    jobs: &[ParameterJob],
    grid: Grid2D,
    k: [f64; 2],
    polarization: Polarization,
    dielectric_options: &DielectricOptions,
    config: &SingleSolveJob,
    policy: &SweepPolicy,
) -> Result<ParameterSweepResult, String>
where
    B: SpectralBackend + Clone,
    F: FnMut() -> B,
{
    crate::profile_scope!("parameter_sweep");
    let timer = Timer::start();
    let first_iteration = iteration_total();
    let (route, reserved) = prepare::<B>(jobs, grid, k, config, policy)?;
    let mut cache = GeometryContinuationCache::with_eviction(
        policy.max_cache_payload_bytes,
        policy.eviction.clone(),
    );
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
    let gates = SymmetryOptions::default();
    let mut outputs: Vec<ParameterOutput> = Vec::new();
    for (position, &index) in report.route.order.iter().enumerate() {
        let start = Timer::start();
        let before = iteration_total();
        let job = &jobs[index];
        let mut op = ThetaOperator::new(
            factory(),
            Dielectric2D::from_geometry(&job.geometry, grid, dielectric_options),
            polarization,
            job.geometry.lattice.reciprocal().fractional_to_cartesian(k),
        );
        let mut point = JobReport {
            original_index: index,
            execution_position: position,
            id: job.id.clone(),
            parameters: job.parameters.clone(),
            previous_execution_index: position.checked_sub(1).map(|i| report.route.order[i]),
            iterations: 0,
            seconds: 0.0,
            predictor_seconds: 0.0,
            exact_source: None,
            predictor: None,
            insertion: None,
            cache_payload_bytes: cache.payload_bytes(),
            converged: false,
            certified: false,
            nearest_completed_parameter_distance: None,
            selected_parameter_distance: None,
        };
        // Metadata-only comparison to previous jobs. Retained output fields
        // are deliberately inaccessible to the predictor outside the cache.
        let same_lattice = |a: &Geometry2D, b: &Geometry2D| {
            a.lattice.a1.map(f64::to_bits) == b.lattice.a1.map(f64::to_bits)
                && a.lattice.a2.map(f64::to_bits) == b.lattice.a2.map(f64::to_bits)
        };
        let attempt = (|| -> Result<CachedSpectrum, String> {
            point.nearest_completed_parameter_distance = outputs
                .iter()
                .filter(|o| {
                    o.report.certified
                        && same_lattice(&jobs[o.report.original_index].geometry, &job.geometry)
                })
                .map(|o| {
                    policy
                        .metric
                        .distance(&job.parameters, &o.report.parameters)
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .min_by(f64::total_cmp);
            if policy.exact_reuse {
                if let Some(reused) = cache.reuse_exact(&mut op, k, config.n_bands, &gates)? {
                    report.exact_cache_hits += 1;
                    point.exact_source = Some(reused.source_id);
                    point.converged = true;
                    return Ok(reused.spectrum);
                }
            }
            let predictor_start = Timer::start();
            let seed = cache.prepare_tracked(
                &mut op,
                &job.parameters,
                &policy.metric,
                policy.history,
                config.n_bands,
                gates.relative_rank_tolerance,
            )?;
            point.predictor_seconds = predictor_start.elapsed_secs();
            let mut preconditioner = op.build_homogeneous_preconditioner_adaptive();
            let solved = if let Some(seed) = &seed {
                report.seed_cache_hits += 1;
                point.selected_parameter_distance = seed
                    .report
                    .source_parameters
                    .iter()
                    .map(|p| policy.metric.distance(p, &job.parameters))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .min_by(f64::total_cmp);
                solve_with_warmstart(&mut op, Some(&mut preconditioner), config, &seed.fields)
            } else {
                report.cold_starts += 1;
                solve(&mut op, Some(&mut preconditioner), config)
            };
            report.solved_jobs += 1;
            point.converged = solved.converged;
            point.predictor = seed.map(|s| s.report);
            Ok(CachedSpectrum {
                source_index: index,
                k,
                eigenvalues: solved.eigenvalues,
                fields: solved.eigenvectors,
                residuals: Vec::new(),
                absolute_residuals: Vec::new(),
                orthogonality_defect: f64::INFINITY,
            })
        })();
        match attempt {
            Ok(mut spectrum) => {
                spectrum.source_index = index;
                let cert =
                    certify_without_refinement(&mut op, &spectrum.eigenvalues, &spectrum.fields);
                spectrum.residuals = cert.residuals;
                spectrum.absolute_residuals = cert.absolute_residuals;
                spectrum.orthogonality_defect = cert.b_orthogonality_defect;
                point.certified = point.converged
                    && spectrum.fields.len() == config.n_bands
                    && spectrum.passes(&gates);
                if point.certified && point.exact_source.is_none() {
                    match cache.insert(
                        &mut op,
                        job.id.clone(),
                        job.parameters.clone(),
                        spectrum.clone(),
                        &gates,
                    ) {
                        Ok(insertion) => {
                            report.evicted_entries += insertion.evicted_entries;
                            point.insertion = Some(insertion);
                        }
                        Err(reason) => {
                            report.rejection = Some(reason);
                        }
                    }
                } else if !point.certified {
                    report.rejection = Some(format!(
                        "parameter job {index} failed convergence or fresh physical certification"
                    ));
                }
                point.iterations = iteration_total().wrapping_sub(before);
                point.seconds = start.elapsed_secs();
                point.cache_payload_bytes = cache.payload_bytes();
                report.summed_job_seconds += point.seconds;
                report.peak_cache_payload_bytes =
                    report.peak_cache_payload_bytes.max(cache.payload_bytes());
                report.completed_jobs += usize::from(point.certified);
                report.retained_output_field_bytes +=
                    spectrum.fields.iter().map(|f| f.len() * 16).sum::<usize>();
                outputs.push(ParameterOutput {
                    spectrum,
                    report: point,
                });
            }
            Err(reason) => {
                report.rejection = Some(reason);
                report.summed_job_seconds += start.elapsed_secs();
            }
        }
        if report.rejection.is_some() {
            report.failed_job_index = Some(index);
            break;
        }
    }
    if report.rejection.is_none() {
        outputs = report.route.restore(outputs)?;
    } else {
        outputs.sort_by_key(|o| o.report.original_index);
    }
    report.iterations = iteration_total().wrapping_sub(first_iteration);
    // Cache release belongs to the completed sweep; returned physical fields
    // remain retained independently under the output budget.
    drop(cache);
    report.seconds = timer.elapsed_secs();
    Ok(ParameterSweepResult { outputs, report })
}

fn prepare<B: SpectralBackend>(
    jobs: &[ParameterJob],
    grid: Grid2D,
    k: [f64; 2],
    config: &SingleSolveJob,
    policy: &SweepPolicy,
) -> Result<(ParameterRoute, usize), String> {
    let rows = grid
        .nx
        .checked_mul(grid.ny)
        .ok_or("parameter grid size overflow")?;
    if std::mem::size_of::<B::Real>() != 8
        || rows == 0
        || !grid.lx.is_finite()
        || !grid.ly.is_finite()
        || grid.lx <= 0.0
        || grid.ly <= 0.0
        || config.n_bands == 0
        || config.n_bands > rows
        || config.max_iterations == 0
        || !config.tolerance.is_finite()
        || config.tolerance <= 0.0
        || config.record_diagnostics
        || config.estimate_condition_number
        || k.iter().any(|v| !v.is_finite())
    {
        return Err("parameter sweeps require valid f64 dimensions/solver controls and no diagnostic-study request".into());
    }
    let field_bytes = rows
        .checked_mul(config.n_bands)
        .and_then(|x| x.checked_mul(16))
        .ok_or("output field size overflow")?;
    let reserved = field_bytes
        .checked_mul(jobs.len())
        .ok_or("parameter output payload overflow")?;
    if reserved > policy.max_output_field_bytes {
        return Err("parameter outputs exceed field payload budget".into());
    }
    let mut ids = BTreeSet::new();
    for job in jobs {
        if job.id.is_empty() || !ids.insert(job.id.as_str()) {
            return Err("parameter job IDs must be nonempty and unique".into());
        }
        policy.metric.validate(&job.parameters)?;
        if let GeometryEviction::FarthestFromIncoming { metric } = &policy.eviction {
            metric.validate(&job.parameters)?;
        }
        let g = &job.geometry;
        let determinant = g.lattice.a1[0] * g.lattice.a2[1] - g.lattice.a1[1] * g.lattice.a2[0];
        if !determinant.is_finite()
            || determinant == 0.0
            || !g.eps_bg.is_finite()
            || g.eps_bg <= 0.0
            || g.atoms.iter().any(|a| {
                !a.radius.is_finite()
                    || a.radius < 0.0
                    || !a.eps_inside.is_finite()
                    || a.eps_inside <= 0.0
                    || a.pos.iter().any(|p| !p.is_finite())
            })
        {
            return Err("invalid physical parameter geometry".into());
        }
    }
    if matches!(
        policy.history,
        SeedHistory::Recent { snapshots: 0 } | SeedHistory::Nearest { snapshots: 0 }
    ) {
        return Err("parameter seed history must be positive".into());
    }
    let parameters: Vec<_> = jobs.iter().map(|j| j.parameters.clone()).collect();
    let route = ParameterRoute::new(&parameters, &policy.metric, policy.order)?;
    Ok((route, reserved))
}
