use blaze2d_core::drivers::operator_data::OperatorDataDriverResult;
use serde_json::json;
use crate::{Array, Arrays, PlannedJob, Quantity, SampleRecord};

/// Preserve the extractor's actual tensor basis and complete numerical dataset.
pub fn operator_sample(job: &PlannedJob, sample_index: usize, result: OperatorDataDriverResult) -> SampleRecord {
    let d = result.ingredients;
    let n = d.eigenvalues.len();
    let r = d.n_retained;
    let [nx, ny] = d.grid_dims;
    let retained: Vec<usize> = (d.band_lo..d.band_lo + r).collect();
    let remote: Vec<usize> = (0..n).filter(|i| !retained.contains(i)).collect();
    let nr = remote.len();
    let keep_fields = job.resolved.config.results.eigenvectors;
    let requested = &job.resolved.config.operators.as_ref().unwrap().quantities;
    let max_residual = d.residuals.iter().copied().fold(0.0,f64::max);
    let mut arrays = Arrays::new();
    macro_rules! real { ($name:expr, $shape:expr, $dims:expr, $data:expr) => {
        arrays.insert($name.into(), Array::real($shape, $dims, $data));
    } }
    macro_rules! complex { ($name:expr, $shape:expr, $dims:expr, $data:expr) => {
        arrays.insert($name.into(), Array::complex($shape, $dims, $data));
    } }
    macro_rules! matrix { ($field:ident) => {
        if let Some(data) = d.$field { complex!(stringify!($field), &[r,r], &["retained_band","retained_band"], data); }
    } }
    macro_rules! direction { ($field:ident, $cols:expr, $colname:expr) => {
        if let Some(data) = d.$field { complex!(stringify!($field), &[2,r,$cols], &["direction","retained_band",$colname], data.into_iter().flatten()); }
    } }
    macro_rules! tensor { ($field:ident) => {
        if let Some(data) = d.$field { complex!(stringify!($field), &[2,2,r,r], &["direction","direction","retained_band","retained_band"], data.into_iter().flatten().flatten()); }
    } }
    real!("eigenvalues", &[n], &["solved_band"], d.eigenvalues);
    real!("residuals", &[n], &["solved_band"], d.residuals);
    if keep_fields {
        complex!("eigenvectors", &[n,ny,nx], &["solved_band","y","x"], d.eigenvectors.into_iter().flatten());
    }
    complex!("velocity_matrices", &[2,r,n], &["direction","retained_band","solved_band"], d.velocity_matrices.into_iter().flatten());
    complex!("w_matrices", &[2,2,r,r], &["direction","direction","retained_band","retained_band"], d.w_matrices.into_iter().flatten().flatten());
    // The core uses W as internal storage when correction is disabled. It is
    // already exported as w_matrices and must not be named a corrected mass.
    if requested.contains(&Quantity::MassTensor) {
        complex!("mass_tensor_inv", &[2,2,r,r], &["direction","direction","retained_band","retained_band"], d.mass_tensor_inv.into_iter().flatten().flatten());
    }
    direction!(r_derivative_matrices, n, "solved_band");
    direction!(metric_derivative_matrices, n, "solved_band");
    direction!(berry_connection_matrices, r, "retained_band");
    matrix!(born_huang);
    tensor!(born_huang_tensor);
    matrix!(slow_coefficient_potential);
    tensor!(slow_coefficient_tensor);
    matrix!(xi_scalar_first_order);
    direction!(kappa_matrices, r, "retained_band");
    matrix!(weighted_leakage_scalar);
    if let Some(data) = d.lowdin_t_matrices {
        complex!("lowdin_t_matrices", &[2,nr,r], &["direction","remote_band","retained_band"], data.into_iter().flatten());
    }
    if let Some(data) = d.lowdin_r_matrix {
        complex!("lowdin_r_matrix", &[nr,r], &["remote_band","retained_band"], data);
    }
    matrix!(overlap_matrix);
    let mut computed: Vec<String> = arrays.keys().cloned().collect();
    if !keep_fields { computed.push("eigenvectors".into()); }
    if let Some(t) = d.exact_tm {
        macro_rules! tm_dir { ($field:ident, $size:expr, $dim:expr) => {
            complex!(concat!("exact_tm.",stringify!($field)), &[2,$size,$size], &["direction",$dim,$dim], t.$field.into_iter().flatten());
        } }
        macro_rules! tm_mat { ($field:ident, $size:expr, $dim:expr) => {
            complex!(concat!("exact_tm.",stringify!($field)), &[$size,$size], &[$dim,$dim], t.$field);
        } }
        macro_rules! tm_grid { ($field:ident) => {
            real!(concat!("exact_tm.",stringify!($field)), &[2,ny,nx], &["direction","y","x"], t.$field.into_iter().flatten());
        } }
        if keep_fields {
            complex!("exact_tm.hermitized_eigenvectors", &[n,ny,nx], &["solved_band","y","x"], t.hermitized_eigenvectors.into_iter().flatten());
        } else { computed.push("exact_tm.hermitized_eigenvectors".into()); }
        tm_dir!(velocity_matrices, n, "solved_band");
        tm_dir!(local_r_derivative_matrices, n, "solved_band");
        tm_dir!(local_r_second_derivative_matrices, n, "solved_band");
        tm_mat!(first_order_remainder, n, "solved_band");
        tm_mat!(direct_metric, r, "retained_band");
        tm_dir!(direct_b_matrices, r, "retained_band");
        tm_mat!(direct_gamma2, r, "retained_band");
        complex!("exact_tm.mass_tensor_inv", &[2,2,r,r], &["direction","direction","retained_band","retained_band"], t.mass_tensor_inv.into_iter().flatten().flatten());
        tm_grid!(epsilon_r_derivatives);
        tm_grid!(epsilon_r_second_derivatives);
        tm_grid!(rho_r_derivatives);
        tm_grid!(rho_r_second_derivatives);
    }
    computed.extend(arrays.keys().cloned());
    computed.sort(); computed.dedup();
    let unavailable = if requested.contains(&Quantity::Overlap) && !arrays.contains_key("overlap_matrix") {
        json!({"overlap_matrix": "Center has no parent reference"})
    } else { json!({}) };
    SampleRecord { sample_index, metadata: json!({
        "k_point": d.k0, "k_basis": "cartesian_angular", "registry": d.registry,
        "solved_band_indices": (0..n).collect::<Vec<_>>(), "retained_band_indices": retained,
        "remote_band_indices": remote, "iterations": d.n_iterations, "converged": d.converged,
        "certification": {"source": "fresh_rayleigh_ritz", "max_residual":max_residual,
            "b_orthogonality_defect": d.b_orthogonality_defect},
        "timings": {"solve_seconds": result.solve_time_seconds, "extraction_seconds": result.extract_time_seconds},
        "residual_gate_violation": result.residual_gate_violation,
        "quantities": {"requested": requested, "computed": computed,
            "retained": arrays.keys().collect::<Vec<_>>(), "unavailable": unavailable},
    }), arrays }
}
