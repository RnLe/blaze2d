use blaze2d_interface::*;
use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::field::Field2D;
use num_complex::Complex64;
use serde_json::json;

fn config() -> Config {
    let mut c = serde_json::to_value(Config::default()).unwrap();
    c["task"]=json!("operators"); c.as_object_mut().unwrap().remove("bands");
    c["grid"]=json!({"resolution":[12,16]});
    c["results"]=json!({"eigenvectors":true});
    c["operators"]=json!({"band_lo":2,"retained_bands":2,"remote_bands":1,
        "k_point":{"value":[0.19,0.13],"basis":"reciprocal_fractional"},"quantities":["velocity"]});
    Config::from_value(c).unwrap()
}

#[test]
fn external_overlap_selects_the_retained_reference_window() {
    let c=config(); let p=Plan::new(c.clone(),Platform::Native).unwrap(); let job=p.job(0).unwrap();
    let initial=execute(CpuBackend::<f64>::new(),"cpu",&job,|_|{}).unwrap();
    let data=&initial.arrays["eigenvectors"].data;
    let fields:Vec<_>=data.chunks_exact(2*12*16).map(|row|Field2D::from_f64_vec(job.grid(),
        row.chunks_exact(2).map(|p|Complex64::new(p[0],p[1])).collect())).collect();
    let mut c=c;
    let op=c.operators.as_mut().unwrap();op.quantities.push(Quantity::Overlap);op.reference=Some(Reference::External);
    assert!(Plan::new(c.clone(),Platform::Browser).is_err());
    let job=Plan::new(c,Platform::Native).unwrap().job(0).unwrap();
    assert!(execute(CpuBackend::<f64>::new(),"cpu",&job,|_|{}).is_err());
    let result=execute_with_fields(CpuBackend::<f64>::new(),"cpu",&job,Some(&fields),None,|_|{}).unwrap();
    let s=&result.arrays["overlap_matrix"].data;
    assert!((s[0].hypot(s[1])-1.0).abs()<1e-8);
    assert!(s[2].hypot(s[3])<1e-8);
    assert!(s[4].hypot(s[5])<1e-8);
    assert!((s[6].hypot(s[7])-1.0).abs()<1e-8);
    assert_eq!(result.metadata["external_fields"]["reference_sha256"].as_str().unwrap().len(),64);
}

#[test]
fn a_small_stencil_has_nearly_unit_overlaps_with_nonzero_band_offset() {
    let mut c=config();let op=c.operators.as_mut().unwrap();op.quantities.push(Quantity::Overlap);
    op.k_stencil=Some(KStencil {points_per_axis:3,half_width:1e-7});
    let job=Plan::new(c,Platform::Native).unwrap().job(0).unwrap();
    let result=execute(CpuBackend::<f64>::new(),"cpu",&job,|_|{}).unwrap();
    for sample in &result.samples[1..] {
        let s=&sample.arrays["overlap_matrix"].data;
        assert!((s[0].hypot(s[1])-1.0).abs()<1e-6);
        assert!((s[6].hypot(s[7])-1.0).abs()<1e-6);
    }
}

#[test]
fn external_homogeneous_dielectric_preserves_rectangular_dimensions_and_certification() {
    let mut c=Config::default();c.geometry.objects.clear();c.grid.resolution=Resolution::Axes(vec![12,16]);
    c.bands.as_mut().unwrap().count=3;
    c.bands.as_mut().unwrap().path=KPath {points:Some(vec![vec![0.19,0.13]]),intervals_per_segment:None,..Default::default()};
    c.dielectric.source=DielectricSource::External;c.results.eigenvectors=true;
    assert!(Plan::new(c.clone(),Platform::Browser).is_err());
    let job=Plan::new(c,Platform::Native).unwrap().job(0).unwrap();
    let result=external::execute_sampled(CpuBackend::<f64>::new(),"cpu",&job,vec![4.0;12*16],None).unwrap();
    let k=job.resolved.k_points_cartesian[0];
    assert!((result.arrays["eigenvalues"].data[0]-(k[0]*k[0]+k[1]*k[1])/4.0).abs()<1e-8);
    assert_eq!(result.arrays["eigenvectors"].shape,[1,3,16,12]);
    assert_eq!(result.arrays["inputs.epsilon"].shape,[16,12]);
    assert!(external::execute_sampled(CpuBackend::<f64>::new(),"cpu",&job,vec![-1.0;12*16],None).is_err());
}
