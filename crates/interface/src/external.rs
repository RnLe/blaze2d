//! Native research inputs still lower through the common scientific settings.
use blaze2d_core::{backend::SpectralBackend, dielectric::Dielectric2D,
    drivers::single_solve::{self, SingleSolveJob}, operators::ThetaOperator};
use sha2::{Digest, Sha256};
use serde_json::json;
use crate::{Array, Diagnostic, DielectricSource, InterfaceResult, PlannedJob, Precision, ResultRecord, Task};

pub fn fingerprint(data: impl IntoIterator<Item=f64>) -> String {
    let mut hash = Sha256::new();
    for value in data { hash.update(value.to_le_bytes()); }
    format!("{:x}",hash.finalize())
}

pub fn execute_sampled<B: SpectralBackend>(backend: B, backend_name: &str, job: &PlannedJob,
    epsilon: Vec<f64>, tensors: Option<Vec<[f64;4]>>) -> InterfaceResult<ResultRecord>
{
    let fail = |message| Diagnostic::new("external_input", "dielectric.source", message);
    if job.resolved.config.task != Task::Bands || job.resolved.config.dielectric.source != DielectricSource::External || job.resolved.k_points_cartesian.len() != 1 {
        return Err(fail("Declare dielectric.source = 'external' with one explicitly sampled band point"));
    }
    let actual = if std::mem::size_of::<B::Real>() == 4 { Precision::F32 } else { Precision::F64 };
    if actual != job.resolved.config.eigensolver.precision { return Err(fail("Backend precision does not match the calculation")); }
    let [nx,ny] = job.resolved.resolution;
    let n = job.resolved.solved_bands;
    if epsilon.len() != nx*ny || epsilon.iter().any(|v| !v.is_finite() || *v<=0.0) {
        return Err(fail("Epsilon must be a finite positive array with shape (ny, nx)"));
    }
    if let Some(tensors) = &tensors {
        if tensors.len() != nx*ny || tensors.iter().any(|t| t.iter().any(|v|!v.is_finite()) ||
            t[0]<=0.0 || t[3]<=0.0 || (t[1]-t[2]).abs()>1e-12*t[0].max(t[3]) || t[0]*t[3]-t[1]*t[2]<=0.0) {
            return Err(fail("Inverse-epsilon tensors must be finite symmetric positive definite 2x2 matrices on the grid"));
        }
    }
    let mut out = ResultRecord::new(job,backend_name);
    out.metadata["external_dielectric"] = json!({"epsilon_sha256":fingerprint(epsilon.iter().copied()),
        "inverse_tensor_sha256":tensors.as_ref().map(|t|fingerprint(t.iter().flatten().copied())),
        "smoothing_applied":false});
    let reciprocal = job.geometry().lattice.reciprocal();
    let dielectric = Dielectric2D::from_sampled_epsilon(job.grid(),reciprocal.b1,reciprocal.b2,epsilon.clone(),tensors.clone());
    let e = job.eigensolver();
    let settings = SingleSolveJob {n_bands:n,tolerance:e.tol,max_iterations:e.max_iter,block_size:e.block_size,..Default::default()};
    let mut theta = ThetaOperator::new(backend,dielectric,job.polarization(),job.resolved.k_points_cartesian[0]);
    let mut preconditioner = theta.build_homogeneous_preconditioner_adaptive();
    let result = single_solve::solve(&mut theta,Some(&mut preconditioner),&settings);
    out.arrays.insert("frequencies".into(),Array::real(&[1,n],&["k_point","band"],result.eigenvalues.iter().map(|v|v.max(0.0).sqrt()/std::f64::consts::TAU)));
    out.arrays.insert("eigenvalues".into(),Array::real(&[1,n],&["k_point","band"],result.eigenvalues));
    out.arrays.insert("residuals".into(),Array::real(&[1,n],&["k_point","band"],result.final_residuals));
    out.arrays.insert("k_points".into(),Array::real(&[1,2],&["k_point","direction"],job.resolved.k_points_fractional[0]));
    out.arrays.insert("k_points_cartesian".into(),Array::real(&[1,2],&["k_point","direction"],job.resolved.k_points_cartesian[0]));
    out.arrays.insert("distances".into(),Array::real(&[1],&["k_point"],[0.0]));
    out.arrays.insert("inputs.epsilon".into(),Array::real(&[ny,nx],&["y","x"],epsilon));
    if let Some(t) = tensors { out.arrays.insert("inputs.inverse_epsilon_tensors".into(),Array::real(&[ny,nx,2,2],&["y","x","direction","direction"],t.into_iter().flatten())); }
    if job.resolved.config.results.eigenvectors {
        out.arrays.insert("eigenvectors".into(),Array::complex(&[1,n,ny,nx],&["k_point","band","y","x"],result.eigenvectors.into_iter().take(n).flat_map(|f|f.as_slice().to_vec())));
    }
    out.metadata["iterations"] = json!([result.iterations]);
    out.metadata["converged"] = json!(result.converged);
    out.metadata["converged_per_point"] = json!([result.converged]);
    out.metadata["elapsed_seconds"] = json!(result.elapsed_seconds);
    out.metadata["certification"] = json!({"source":"fresh_rayleigh_ritz","b_orthogonality_defect":result.b_orthogonality_defect});
    out.metadata["band_indices"] = json!((0..n).collect::<Vec<_>>());
    out.metadata["gauge"] = json!("independent_sorted_eigenvalues");
    out.metadata["quantities"] = json!({"requested":["frequencies"],"computed":["frequencies","eigenvalues","eigenvectors","residuals"],
        "retained":out.arrays.keys().collect::<Vec<_>>(),"unavailable":{}});
    out.validate_arrays()?;
    Ok(out)
}
