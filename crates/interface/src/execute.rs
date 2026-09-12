//! Backend-independent execution of a fully planned scientific job.

use blaze2d_core::{backend::SpectralBackend, drivers::{bandstructure, operator_data}, timing::Timer};
use serde_json::json;
use crate::{Array, Diagnostic, Event, JobFailure, PlannedJob, Precision, ResultRecord, Task};

pub fn execute<B: SpectralBackend + Clone>(backend: B, backend_name: &str, job: &PlannedJob,
    on_event: impl FnMut(Event)) -> Result<ResultRecord, JobFailure>
{
    execute_with_fields(backend, backend_name, job, None, None, on_event)
}

/// Native reference and warm-start blocks use the full solved band window.
pub fn execute_with_fields<B: SpectralBackend + Clone>(backend: B, backend_name: &str, job: &PlannedJob,
    reference: Option<&[blaze2d_core::field::Field2D]>, warmstart: Option<&[blaze2d_core::field::Field2D]>,
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
            let mut orthogonality = Vec::new();
            let mut fields = Vec::new();
            let result = bandstructure::run_with_k_streaming(backend, &job.bands_job(),
                bandstructure::RunOptions { reuse_gamma: false, disable_band_tracking: !b.tracking,
                    retain_eigenvectors: job.resolved.config.results.eigenvectors, ..Default::default() }, |point| {
                    on_event(Event::Progress { job_index: job.index, sample_index: point.k_index,
                        completed: point.k_index + 1, total: nk, iterations: point.iterations, converged: point.converged,
                        band_point:Some(crate::BandPoint {frequencies:point.omegas.iter().map(|w|w/std::f64::consts::TAU).collect(),
                            k_point:point.k_point,distance:job.resolved.distances[point.k_index]}) });
                    iterations.push(point.iterations); converged.push(point.converged);
                    orthogonality.push(point.b_orthogonality_defect);
                    residuals.extend(point.residuals.into_iter().take(n));
                    if let Some(v) = point.eigenvectors { fields.extend(v.into_iter().take(n).flat_map(|v| v.as_slice().to_vec())); }
                });
            out.arrays.insert("frequencies".into(), Array::real(&[nk,n], &["k_point","band"],
                result.bands.into_iter().flatten().map(|w| w / std::f64::consts::TAU)));
            out.arrays.insert("k_points".into(), Array::real(&[nk,2], &["k_point","direction"], job.resolved.k_points_fractional.iter().flatten().copied()));
            out.arrays.insert("k_points_cartesian".into(), Array::real(&[nk,2], &["k_point","direction"], job.resolved.k_points_cartesian.iter().flatten().copied()));
            out.arrays.insert("distances".into(), Array::real(&[nk], &["k_point"], job.resolved.distances.iter().copied()));
            out.arrays.insert("residuals".into(), Array::real(&[nk,n], &["k_point","band"], residuals));
            if job.resolved.config.results.eigenvectors {
                out.arrays.insert("eigenvectors".into(), Array::complex(&[nk,n,ny,nx], &["k_point","band","y","x"], fields));
            }
            out.metadata["iterations"] = json!(iterations);
            out.metadata["converged"] = json!(converged.iter().all(|c| *c));
            out.metadata["converged_per_point"] = json!(converged);
            out.metadata["labels"] = json!(job.resolved.k_labels);
            out.metadata["label_indices"] = json!(job.resolved.k_label_indices);
            out.metadata["band_indices"] = json!((0..n).collect::<Vec<_>>());
            out.metadata["stopping_criterion"] = json!("relative_eigenvalue_change");
            out.metadata["certification"] = json!({"source":"fresh_operator_application",
                "residual_convention":"norm(Au-lambda*Bu)/(norm(Au)+abs(lambda)*norm(Bu))",
                "max_residual":out.arrays["residuals"].data.iter().copied().fold(0.0_f64, f64::max),
                "b_orthogonality_defect":orthogonality.iter().copied().fold(0.0_f64, f64::max),
                "b_orthogonality_per_point":orthogonality});
            out.metadata["gauge"] = json!(if b.tracking {"tracked_path"} else {"independent_sorted_eigenvalues"});
            out.metadata["quantities"] = json!({"requested":["frequencies"],
                "computed":["frequencies","eigenvectors","residuals"], "retained":out.arrays.keys().collect::<Vec<_>>(), "unavailable":{}});
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
        return Err(JobFailure { job_index: job.index, diagnostic, partial_result: Some(Box::new(out)) });
    }
    Ok(out)
}
