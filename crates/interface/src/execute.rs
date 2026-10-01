//! Backend-independent execution of a fully planned scientific job.

use blaze2d_core::{backend::SpectralBackend, drivers::{bandstructure, operator_data}, timing::Timer};
use serde_json::json;
use crate::{Array, Diagnostic, Event, JobFailure, PlannedJob, Precision, ResultRecord, Task};

/// Persistent state owned by one serial job chain, including browser jobs.
/// Only certified spectra enter the bounded cache; backend scratch is local.
pub struct ExecutionContext {
    continuation: blaze2d_core::symmetry::continuation::BandContinuation,
    max_continuation_bytes: usize,
}

impl ExecutionContext {
    pub fn new(max_continuation_bytes: usize) -> Self {
        Self { continuation: blaze2d_core::symmetry::continuation::BandContinuation::new(max_continuation_bytes), max_continuation_bytes }
    }
    pub fn clear(&mut self) { self.continuation.cache.clear(); }
}

pub fn execute_in_context<B: SpectralBackend + Clone>(backend: B, backend_name: &str, job: &PlannedJob,
    context: &mut ExecutionContext, on_event: impl FnMut(Event)) -> Result<ResultRecord, JobFailure>
{
    let result = execute_impl(backend, backend_name, job, None, None, Some(&mut *context), on_event);
    if result.is_err() { context.clear(); }
    result
}

pub fn execute<B: SpectralBackend + Clone>(backend: B, backend_name: &str, job: &PlannedJob,
    on_event: impl FnMut(Event)) -> Result<ResultRecord, JobFailure>
{
    execute_with_fields(backend, backend_name, job, None, None, on_event)
}

/// Native reference and warm-start blocks use the full solved band window.
pub fn execute_with_fields<B: SpectralBackend + Clone>(backend: B, backend_name: &str, job: &PlannedJob,
    reference: Option<&[blaze2d_core::field::Field2D]>, warmstart: Option<&[blaze2d_core::field::Field2D]>,
    on_event: impl FnMut(Event)) -> Result<ResultRecord, JobFailure>
{
    let budget = job.resolved.config.bands.as_ref().map_or(0, |b| b.symmetry.max_continuation_bytes);
    let mut context = ExecutionContext::new(budget);
    execute_impl(backend, backend_name, job, reference, warmstart, Some(&mut context), on_event)
}

fn execute_impl<B: SpectralBackend + Clone>(backend: B, backend_name: &str, job: &PlannedJob,
    reference: Option<&[blaze2d_core::field::Field2D]>, warmstart: Option<&[blaze2d_core::field::Field2D]>,
    context: Option<&mut ExecutionContext>,
    mut on_event: impl FnMut(Event)) -> Result<ResultRecord, JobFailure>
{
    let fail = |path: &str, message: &str| JobFailure {job_index:job.index,
        diagnostic:Diagnostic::new("external_input",path,message),partial_result:None};
    let op = job.resolved.config.operators.as_ref();
    if job.resolved.config.dielectric.source == crate::DielectricSource::External {
        return Err(fail("dielectric.source", "Supply the declared dielectric array through the native Python interface"));
    }
    if op.is_some_and(|o|o.reference.is_some()) != reference.is_some() {
        return Err(fail("operators.reference", "Declare reference = { source = 'external' } and supply the reference block through Python"));
    }
    if (reference.is_some() || warmstart.is_some()) && (op.is_none() || op.is_some_and(|o|o.k_stencil.is_some())) {
        return Err(fail("operators", "External fields apply to point extraction only"));
    }
    for (name, fields) in [("reference_eigenvectors",reference),("warmstart_eigenvectors",warmstart)] {
        if let Some(fields) = fields {
            let [nx,ny] = job.resolved.resolution;
            if fields.len() != job.resolved.solved_bands || fields.iter().any(|f|
                f.grid().nx != nx || f.grid().ny != ny || f.as_slice().iter().any(|c|!c.re.is_finite() || !c.im.is_finite())) {
                return Err(fail(name,"Expected a finite block with shape (solved_band, ny, nx)"));
            }
        }
    }
    let actual = if std::mem::size_of::<B::Real>() == 4 { Precision::F32 } else { Precision::F64 };
    if actual != job.resolved.config.eigensolver.precision {
        return Err(JobFailure { job_index: job.index, diagnostic: Diagnostic::new("precision", "eigensolver.precision", "Backend storage precision does not match the calculation"), partial_result: None });
    }
    on_event(Event::JobStart { job_index: job.index, job:Box::new(job.clone()) });
    let start = Timer::start();
    let mut out = ResultRecord::new(job, backend_name);
    if reference.is_some() || warmstart.is_some() {
        out.metadata["external_fields"] = json!({
            "reference_sha256":reference.map(|v|crate::external::fingerprint(v.iter().flat_map(|f|f.as_slice().iter().flat_map(|c|[c.re,c.im])))),
            "warmstart_sha256":warmstart.map(|v|crate::external::fingerprint(v.iter().flat_map(|f|f.as_slice().iter().flat_map(|c|[c.re,c.im])))),
            "band_indices":(0..job.resolved.solved_bands).collect::<Vec<_>>()});
    }
    match job.resolved.config.task {
        Task::Bands => {
            let b = job.resolved.config.bands.as_ref().unwrap();
            let [nx, ny] = job.resolved.resolution;
            let n = b.count;
            let nk = job.resolved.k_points_fractional.len();
            let mut iterations = Vec::new();
            let mut converged = Vec::new();
            let mut residuals = Vec::new();
            let mut absolute_residuals = Vec::new();
            let mut orthogonality = Vec::new();
            let mut fields = Vec::new();
            let mut symmetry_reports = Vec::new();
            let mut discretization_reports = Vec::new();
            let mut point_seconds = Vec::new();
            let mut point_work = Vec::new();
            let mut point_preconditioners = Vec::new();
            let mut collect = |point: bandstructure::KPointResult| {
                    point_seconds.push(point.elapsed_seconds);
                    point_work.push(point.work);
                    point_preconditioners.push(point.preconditioner);
                    on_event(Event::Progress { job_index: job.index, sample_index: point.k_index,
                        completed: point.k_index + 1, total: nk, iterations: point.iterations, converged: point.converged,
                        band_point:Some(crate::BandPoint {frequencies:point.omegas.iter().map(|w|w/std::f64::consts::TAU).collect(),
                            k_point:point.k_point,distance:job.resolved.distances[point.k_index]}) });
                    iterations.push(point.iterations); converged.push(point.converged);
                    symmetry_reports.push(point.symmetry);
                    discretization_reports.push(point.discretization);
                    orthogonality.push(point.b_orthogonality_defect);
                    residuals.extend(point.residuals.into_iter().take(n));
                    absolute_residuals.extend(point.absolute_residuals.into_iter().take(n));
                    if let Some(v) = point.eigenvectors { fields.extend(v.into_iter().take(n).flat_map(|v| v.as_slice().to_vec())); }
                };
            let (effective_symmetry, policy_decision) = crate::policy::resolve(job, backend_name);
            let mut effective_bands = b.clone();
            effective_bands.symmetry = effective_symmetry.clone();
            let mut options=effective_bands.run_options(job.resolved.config.results.eigenvectors);
            options.preconditioner=job.preconditioner();
            let selected=b.discretization.metric_options();
            let continuation = context.filter(|_| effective_symmetry.continuation != crate::ContinuationPolicy::Off
                && b.symmetry.max_continuation_bytes > 0 && actual == Precision::F64)
                .map(|c| {
                    c.continuation.cache.set_payload_limit(c.max_continuation_bytes.min(b.symmetry.max_continuation_bytes));
                    c.continuation.job_index = job.index;
                    c.continuation.first_point_only = effective_symmetry.continuation == crate::ContinuationPolicy::FirstPoint;
                    &mut c.continuation
                });
            let (result,failure)=if let Some(discretization)=selected {
                match blaze2d_core::drivers::support_bands::run_with_continuation(backend,&job.bands_job(),options,discretization,continuation,&mut collect) {
                    Ok(result)=>(result,None),
                    Err(failure)=>(failure.partial_result.clone(),Some(failure)),
                }
            } else {(bandstructure::run_with_continuation(backend,&job.bands_job(),options,continuation,&mut collect),None)};
            let requested_points=nk;
            let nk=result.bands.len();
            out.arrays.insert("frequencies".into(), Array::real(&[nk,n], &["k_point","band"],
                result.bands.into_iter().flatten().map(|w| w / std::f64::consts::TAU)));
            out.arrays.insert("k_points".into(), Array::real(&[nk,2], &["k_point","direction"], job.resolved.k_points_fractional.iter().take(nk).flatten().copied()));
            out.arrays.insert("k_points_cartesian".into(), Array::real(&[nk,2], &["k_point","direction"], job.resolved.k_points_cartesian.iter().take(nk).flatten().copied()));
            out.arrays.insert("distances".into(), Array::real(&[nk], &["k_point"], job.resolved.distances.iter().take(nk).copied()));
            out.arrays.insert("residuals".into(), Array::real(&[nk,n], &["k_point","band"], residuals));
            out.arrays.insert("absolute_residuals".into(), Array::real(&[nk,n], &["k_point","band"], absolute_residuals));
            if job.resolved.config.results.eigenvectors {
                out.arrays.insert("eigenvectors".into(), Array::complex(&[nk,n,ny,nx], &["k_point","band","y","x"], fields));
            }
            out.metadata["iterations"] = json!(iterations);
            out.metadata["symmetry"] = json!({"requested":b.symmetry,"points":symmetry_reports,
                "coverage":"unrestricted source spectrum; qualified finite-operator covariance and fresh transformed residuals"});
            if let Some(decision) = policy_decision {
                out.metadata["symmetry"]["automatic"] = json!(decision);
                out.metadata["symmetry"]["resolved"] = json!(effective_symmetry);
            }
            out.metadata["converged"] = json!(converged.iter().all(|c| *c));
            out.metadata["converged_per_point"] = json!(converged);
            out.metadata["labels"] = json!(job.resolved.k_labels.iter().zip(&job.resolved.k_label_indices).filter(|(_,i)|**i<nk).map(|(label,_)|label).collect::<Vec<_>>());
            out.metadata["label_indices"] = json!(job.resolved.k_label_indices.iter().filter(|&&i|i<nk).collect::<Vec<_>>());
            out.metadata["band_indices"] = json!((0..n).collect::<Vec<_>>());
            let e = &job.resolved.config.eigensolver;
            out.metadata["stopping_criterion"] = json!(match e.stopping {
                crate::Stopping::EigenvalueChange => "relative_eigenvalue_change",
                crate::Stopping::Residual => "relative_residual",
            });
            let mut total_work = blaze2d_core::work::Work::default();
            for w in &point_work { total_work.add(*w); }
            out.metadata["point_seconds"] = json!(point_seconds);
            out.metadata["work"] = json!(total_work);
            out.metadata["point_work"] = json!(point_work);
            out.metadata["eigensolver"] = json!({"stopping": e.stopping, "tolerance": e.tolerance, "residual_tolerance": e.residual_tolerance,
                "history": e.history, "block_size": job.eigensolver().effective_block_size()});
            out.metadata["preconditioner"] = json!({"requested": e.preconditioner, "points": point_preconditioners});
            out.metadata["certification"] = json!({"source":"fresh_operator_application",
                "residual_convention":"norm(Au-lambda*Bu)/(norm(Au)+abs(lambda)*norm(Bu))",
                "absolute_residual_convention":"norm(Au-lambda*Bu)/norm(u)",
                "absolute_residual_unit":"inverse_reference_length_squared",
                "max_absolute_residual":out.arrays["absolute_residuals"].data.iter().copied().fold(0.0_f64, f64::max),
                "max_residual":out.arrays["residuals"].data.iter().copied().fold(0.0_f64, f64::max),
                "b_orthogonality_defect":orthogonality.iter().copied().fold(0.0_f64, f64::max),
                "b_orthogonality_per_point":orthogonality});
            if selected.is_some() {
                out.metadata["discretization"]=json!({"requested":b.discretization,"points":discretization_reports,
                    "requested_points":requested_points,"completed_points":nk,
                    "pencil":"S_adjoint_A_S_and_S_adjoint_B_S",
                    "full_rectangle_residuals":"discretization diagnostics only; not the projected solve acceptance gate",
                    "payload_limit_scope":"modeled driver and solve numerical payload; excludes caller-retained output, FFT plans, allocator and library scratch"});
                out.metadata["symmetry"]["coverage"]=json!("qualified sectors or unrestricted solve on the explicitly selected metric Fourier support; per-point reports state empirical or complete coverage");
                out.metadata["certification"]["pencil"]=json!("selected_metric_fourier_support");
                out.metadata["certification"]["source"]=json!("fresh_projected_operator_application");
            }
            out.metadata["gauge"] = json!(if b.tracking {"tracked_path"} else {"independent_sorted_eigenvalues"});
            out.metadata["quantities"] = json!({"requested":["frequencies"],
                "computed":["frequencies","eigenvectors","residuals","absolute_residuals"], "retained":out.arrays.keys().collect::<Vec<_>>(), "unavailable":{}});
            if let Some(failure)=failure {
                out.metadata["converged"]=json!(false);
                out.metadata["elapsed_seconds"]=json!(start.elapsed_secs());
                out.metadata["failed_point"]=json!({"k_index":failure.k_index,"reason":failure.reason,
                    "symmetry":failure.symmetry,"discretization":failure.discretization});
                crate::accuracy::assess(&mut out, b.accuracy.as_ref());
                crate::summary::record(&mut out);
                return Err(JobFailure {job_index:job.index,diagnostic:Diagnostic::new("metric_support","bands.discretization",failure.reason),partial_result:Some(Box::new(out))});
            }
        }
        Task::Operators => {
            let o = job.resolved.config.operators.as_ref().unwrap();
            let lower = job.operators_job();
            if let Some(s) = &o.k_stencil {
                let mut execution_order = Vec::new();
                let stencil = operator_data::run_k_stencil_with_progress(backend, &lower, s.points_per_axis, s.half_width,
                    |sample_index, completed, r| { execution_order.push(sample_index); on_event(Event::Progress { job_index: job.index, sample_index,
                        completed, total: s.points_per_axis * s.points_per_axis,
                        iterations: r.ingredients.n_iterations, converged: r.ingredients.converged, band_point:None }); });
                out.samples.push(crate::operator_result::operator_sample(job, 0, stencil.center));
                for (i, point) in stencil.neighbors.into_iter().enumerate() {
                    out.samples.push(crate::operator_result::operator_sample(job, i+1, point));
                }
                out.samples[0].metadata["reference_sample_index"] = json!(null);
                for pair in execution_order.windows(2) {
                    out.samples[pair[1]].metadata["reference_sample_index"] = json!(pair[0]);
                }
                out.metadata["stencil_execution_order"] = json!(execution_order);
                out.metadata["stencil_order"] = json!("center, then increasing x offset, then increasing y offset; center skipped");
                out.metadata["converged"] = json!(out.samples.iter().all(|s| s.metadata["converged"] == true));
            } else {
                let retained_reference = reference.map(|r|&r[o.band_lo..o.band_lo + o.retained_bands]);
                let r = if let Some(warm) = warmstart {
                    operator_data::run_with_warmstart(backend, &lower, warm, retained_reference)
                } else { operator_data::run_with_reference(backend, &lower, retained_reference) };
                on_event(Event::Progress { job_index: job.index, sample_index: 0, completed: 1, total: 1,
                    iterations: r.ingredients.n_iterations, converged: r.ingredients.converged, band_point:None });
                let sample = crate::operator_result::operator_sample(job, 0, r);
                out.arrays = sample.arrays;
                out.metadata.as_object_mut().unwrap().extend(sample.metadata.as_object().unwrap().clone());
            }
        }
    }
    out.metadata["elapsed_seconds"] = json!(start.elapsed_secs());
    let gate = out.metadata.get("residual_gate_violation").filter(|v| !v.is_null())
        .or_else(|| out.samples.iter().find_map(|s| s.metadata.get("residual_gate_violation").filter(|v| !v.is_null())));
    if let Some(gate) = gate {
        let message = format!("Certified residual gate failed: band {} has residual {}", gate[0], gate[1]);
        return Err(JobFailure { job_index: job.index, diagnostic: Diagnostic::new("residual_gate", "operators.fail_on_residual", message), partial_result: Some(Box::new(out)) });
    }
    if let Err(diagnostic) = out.validate_arrays() {
        out.metadata["accuracy"] = json!({"status":"invalid_output", "accepted":false});
        crate::summary::record(&mut out);
        return Err(JobFailure { job_index: job.index, diagnostic, partial_result: Some(Box::new(out)) });
    }
    if let Some(bands) = &job.resolved.config.bands {
        let diagnostic = crate::accuracy::assess(&mut out, bands.accuracy.as_ref());
        crate::summary::record(&mut out);
        if let Some(diagnostic) = diagnostic {
            return Err(JobFailure { job_index: job.index, diagnostic, partial_result: Some(Box::new(out)) });
        }
    }
    Ok(out)
}
