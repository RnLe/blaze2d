//! Backend-independent execution of a fully planned scientific job.

use blaze2d_core::{backend::SpectralBackend, drivers::{bandstructure, operator_data}, timing::Timer};
use serde_json::json;
use crate::{Array, Diagnostic, Event, JobFailure, PlannedJob, Precision, ResultRecord, Task};

pub fn execute<B: SpectralBackend + Clone>(backend: B, backend_name: &str, job: &PlannedJob,
    mut on_event: impl FnMut(Event)) -> Result<ResultRecord, JobFailure>
{
    let actual = if std::mem::size_of::<B::Real>() == 4 { Precision::F32 } else { Precision::F64 };
    if actual != job.resolved.config.eigensolver.precision {
        return Err(JobFailure { job_index: job.index, diagnostic: Diagnostic::new("precision", "eigensolver.precision", "Backend storage precision does not match the calculation"), partial_result: None });
    }
    on_event(Event::JobStart { job_index: job.index });
    let start = Timer::start();
    let mut out = ResultRecord::new(job, backend_name);
    match job.resolved.config.task {
        Task::Bands => {
            let b = job.resolved.config.bands.as_ref().unwrap();
            let [nx, ny] = job.resolved.resolution;
            let n = b.count;
            let nk = job.resolved.k_points_fractional.len();
            let mut iterations = Vec::new();
            let mut converged = Vec::new();
            let mut residuals = Vec::new();
            let mut fields = Vec::new();
            let result = bandstructure::run_with_k_streaming(backend, &job.bands_job(),
                bandstructure::RunOptions { reuse_gamma: false, disable_band_tracking: !b.tracking,
                    retain_eigenvectors: job.resolved.config.results.eigenvectors, ..Default::default() }, |point| {
                    on_event(Event::Progress { job_index: job.index, sample_index: point.k_index,
                        completed: point.k_index + 1, total: nk, iterations: point.iterations, converged: point.converged });
                    iterations.push(point.iterations); converged.push(point.converged);
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
            out.metadata["certification"] = json!({"source":"solver_reported", "b_orthogonality_defect":null});
            out.metadata["gauge"] = json!(if b.tracking {"tracked_path"} else {"independent_sorted_eigenvalues"});
            out.metadata["quantities"] = json!({"requested":["frequencies"],
                "computed":["frequencies","eigenvectors","residuals"], "retained":out.arrays.keys().collect::<Vec<_>>(), "unavailable":{}});
        }
        Task::Operators => {
            let o = job.resolved.config.operators.as_ref().unwrap();
            let lower = job.operators_job();
            if let Some(s) = &o.k_stencil {
                let stencil = operator_data::run_k_stencil_with_progress(backend, &lower, s.points_per_axis, s.half_width,
                    |sample_index, completed, r| on_event(Event::Progress { job_index: job.index, sample_index,
                        completed, total: s.points_per_axis * s.points_per_axis,
                        iterations: r.ingredients.n_iterations, converged: r.ingredients.converged }));
                out.samples.push(crate::operator_result::operator_sample(job, 0, stencil.center));
                for (i, point) in stencil.neighbors.into_iter().enumerate() {
                    out.samples.push(crate::operator_result::operator_sample(job, i+1, point));
                }
                out.metadata["stencil_order"] = json!("center, then increasing x offset, then increasing y offset; center skipped");
                out.metadata["converged"] = json!(out.samples.iter().all(|s| s.metadata["converged"] == true));
            } else {
                let r = operator_data::run(backend, &lower);
                on_event(Event::Progress { job_index: job.index, sample_index: 0, completed: 1, total: 1,
                    iterations: r.ingredients.n_iterations, converged: r.ingredients.converged });
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
