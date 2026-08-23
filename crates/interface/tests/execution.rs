use blaze2d_backend_cpu::CpuBackend;
use blaze2d_interface::*;
use serde_json::json;

fn operator_config(polarization: &str, quantities: &[&str]) -> Config {
    let mut value = serde_json::to_value(Config::default()).unwrap();
    value["task"] = json!("operators");
    value["polarization"] = json!(polarization);
    value.as_object_mut().unwrap().remove("bands");
    value["grid"] = json!({"resolution":[12,16]});
    value["operators"] = json!({"band_lo":2,"retained_bands":2,"remote_bands":1,
        "k_point":{"value":[0.19,0.13],"basis":"reciprocal_fractional"},
        "quantities":quantities,
        "registry":{"object":"rod","points":[[0.9,-0.1]],"fd_step":0.001}});
    Config::from_value(value).unwrap()
}

fn solve(c: Config) -> Result<ResultRecord, JobFailure> {
    let plan = Plan::new(c, Platform::Native).unwrap();
    execute(CpuBackend::<f64>::new(), "cpu", &plan.job(0).unwrap(), |_| {})
}

#[test]
fn lowering_preserves_lengths_registry_and_dielectric() {
    let mut c = operator_config("TE", &["velocity"]);
    c.geometry.lattice.a = Some(2.5);
    c.geometry.objects[0].center = vec![0.3,0.05];
    c.dielectric.smoothing = Smoothing::Subgrid;
    c.dielectric.mesh_size = 5;
    let plan = Plan::new(c, Platform::Native).unwrap();
    let job = plan.job(0).unwrap();
    let lower = job.operators_job();
    assert_eq!([lower.grid.nx, lower.grid.ny], [12,16]);
    assert_eq!(lower.n_total_bands(), 5);
    assert!((lower.geom.atoms[0].radius_cartesian(&lower.geom.lattice)-0.2).abs()<1e-15);
    assert!((lower.geom.atoms[0].pos[0]-0.2).abs()<1e-15);
    assert!((lower.geom.atoms[0].pos[1]-0.95).abs()<1e-15);
    assert_eq!(lower.dielectric.smoothing.mesh_size, 5);
    assert_eq!(plan.job(0).unwrap().geometry().atoms[0].pos, lower.geom.atoms[0].pos);
}

#[test]
fn complete_te_dataset_has_actual_remote_dimensions() {
    let out = solve(operator_config("TE", &["velocity","mass_tensor","r_derivatives","born_huang","slow_coefficient"])).unwrap();
    assert_eq!(out.arrays["eigenvalues"].shape, [5]);
    assert_eq!(out.arrays["velocity_matrices"].shape, [2,2,5]);
    assert_eq!(out.arrays["lowdin_t_matrices"].shape, [2,3,2]);
    assert_eq!(out.arrays["mass_tensor_inv"].shape, [2,2,2,2]);
    assert_eq!(out.metadata["remote_band_indices"], json!([0,1,4]));
    assert!(!out.arrays.contains_key("eigenvectors"));
    assert_eq!(out.arrays["residuals"].data.len(), 5);
    assert!(out.metadata["certification"]["b_orthogonality_defect"].as_f64().unwrap().is_finite());
    let restored: ResultRecord = serde_json::from_str(&out.to_json().unwrap()).unwrap();
    assert_eq!(restored.arrays, out.arrays);
}

#[test]
fn exact_tm_and_eigenvector_layout_are_complete() {
    let mut c = operator_config("TM", &["exact_tm"]);
    c.results.eigenvectors = true;
    let out = solve(c).unwrap();
    assert_eq!(out.arrays["eigenvectors"].shape, [5,16,12]);
    assert_eq!(out.arrays["eigenvectors"].dtype, DType::Complex128);
    assert_eq!(out.arrays["exact_tm.velocity_matrices"].shape, [2,5,5]);
    assert_eq!(out.arrays["exact_tm.epsilon_r_derivatives"].shape, [2,16,12]);
    assert!(!out.arrays.contains_key("mass_tensor_inv"));
    assert!(out.arrays.contains_key("w_matrices"));
}

#[test]
fn residual_gate_returns_failed_record_with_certification() {
    let mut c = operator_config("TM", &["velocity"]);
    c.operators.as_mut().unwrap().fail_on_residual = Some(1e-30);
    let failure = solve(c).unwrap_err();
    assert_eq!(failure.diagnostic.code, "residual_gate");
    assert!(failure.partial_result.unwrap().arrays.contains_key("residuals"));
}

#[test]
fn singleton_and_compound_stencil_preserve_order_and_progress() {
    for size in [1,3] {
        let mut c = operator_config("TE", &["velocity"]);
        let o = c.operators.as_mut().unwrap();
        o.k_stencil = Some(KStencil { points_per_axis: size, half_width: 0.01 });
        if size > 1 { o.quantities.push(Quantity::Overlap); }
        let plan = Plan::new(c, Platform::Native).unwrap();
        let mut order = Vec::new();
        let out = execute(CpuBackend::<f64>::new(), "cpu", &plan.job(0).unwrap(), |e| {
            if let Event::Progress {sample_index,..} = e { order.push(sample_index); }
        }).unwrap();
        assert_eq!(order.len(), size*size);
        assert_eq!(order[0], 0);
        assert_eq!(out.samples.len(), size*size);
        assert!(out.samples.iter().enumerate().all(|(i,s)| s.sample_index == i));
        if size > 1 {
            assert_ne!(order, (0..9).collect::<Vec<_>>());
            assert!(out.samples[1..].iter().all(|s| s.arrays.contains_key("overlap_matrix")));
            assert!(!out.samples[0].arrays.contains_key("overlap_matrix"));
        }
    }
}

#[test]
fn band_results_report_quality_and_recompute_closing_gamma() {
    let mut c = Config::default();
    c.grid.resolution = Resolution::Axes(vec![12,16]);
    let b = c.bands.as_mut().unwrap();
    b.count = 3;
    b.path = KPath { points: Some(vec![vec![0.,0.],vec![0.2,0.13],vec![0.,0.]]),
        intervals_per_segment: None, ..Default::default() };
    c.results.eigenvectors = true;
    let out = solve(c).unwrap();
    assert_eq!(out.arrays["frequencies"].shape, [3,3]);
    assert_eq!(out.arrays["residuals"].shape, [3,3]);
    assert_eq!(out.arrays["eigenvectors"].shape, [3,3,16,12]);
    assert_eq!(out.arrays["k_points"].data[4..], [0.,0.]);
    assert!(out.metadata["iterations"][2].as_u64().unwrap() > 0);
}

#[test]
fn array_export_rejects_invalid_shape_and_nonfinite_data() {
    assert!(Array::real(&[2], &["band"], [1.0]).validate("test").is_err());
    assert!(Array::real(&[1], &["band"], [f64::NAN]).validate("test").is_err());
}
