//! Fresh f64 auditing of the public native-f32 geometry and path workflow.
use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{dielectric::Dielectric2D, eigensolver::refine::certify_without_refinement,
    field::Field2D, operators::ThetaOperator};
use blaze2d_interface::*;
use blaze2d_runner::{collect, Options};
use num_complex::Complex64;
use serde_json::json;

fn main() {
    let n: usize = std::env::args().nth(1).map_or(24, |s| s.parse().unwrap());
    let accuracy = BandAccuracy {max_relative_residual:5e-4,max_absolute_residual:2e-3,max_orthogonality_defect:2e-5};
    let mut records=Vec::new();
    let mut passed=true;
    for lattice in [LatticeKind::Square,LatticeKind::Triangular] {
      for pol in [Polarization::TM,Polarization::TE] {
        let mut c=Config::default();
        c.geometry.lattice.kind=lattice;c.polarization=pol;c.grid.resolution=Resolution::Uniform(n);
        c.dielectric.smoothing=Smoothing::Voronoi;c.results.eigenvectors=true;
        c.eigensolver.tolerance=Some(1e-14);c.eigensolver.max_iterations=Some(500);
        let b=c.bands.as_mut().unwrap();b.count=4;b.tracking=true;b.accuracy=Some(BandAccuracy::default());
        b.path.points=Some(vec![vec![0.,0.],vec![0.19,0.13],vec![0.2,0.14],vec![0.5,0.],vec![0.,0.]]);
        b.path.intervals_per_segment=None;
        c.sweeps=vec![Sweep {name:"radius".into(),target:"geometry.objects.rod.radius".into(),
            values:Some(vec![json!(0.191),json!(0.193),json!(0.195)]),linspace:None}];
        let reference=collect(Plan::new(c.clone(),Platform::Native).unwrap(),Options {threads:1,..Default::default()}).unwrap();
        assert!(reference.errors.is_empty(),"f64 reference failed: {:?}",reference.errors);
        c.eigensolver.precision=Precision::F32;c.eigensolver.tolerance=Some(1e-9);
        c.bands.as_mut().unwrap().accuracy=Some(accuracy.clone());
        for method in [None,Some(GeometryMethod::Copy),Some(GeometryMethod::Secant),Some(GeometryMethod::History),Some(GeometryMethod::Residual),Some(GeometryMethod::Response)] {
            c.bands.as_mut().unwrap().warm_start=method.map(|method| GeometryWarmStart {
                method, scope:GeometryScope::Path,guards:2,innovation_columns:if method==GeometryMethod::History {4}else{0},
                ..Default::default()});
            let plan=Plan::new(c.clone(),Platform::Native).unwrap();
            let study=collect(plan.clone(),Options {threads:1,error_policy:blaze2d_runner::ErrorPolicy::Continue,..Default::default()}).unwrap();
            let mut audits=Vec::new();
            let mut valid=study.errors.is_empty();
            for result in &study.results {
                let job=plan.job(result.job_index).unwrap();let core=job.bands_job();
                let material=Dielectric2D::from_geometry(&core.geom,core.grid,&core.dielectric);
                let reference=&reference.results[result.job_index];
                let mut points=Vec::new();
                for (ki,k) in job.resolved.k_points_cartesian.iter().enumerate() {
                    let mut theta=ThetaOperator::new(CpuBackend::<f64>::new(),material.clone(),core.pol,*k);
                    let frequencies=&result.arrays["frequencies"].data[ki*4..(ki+1)*4];
                    let values:Vec<_>=frequencies.iter().map(|f|(f*std::f64::consts::TAU).powi(2)).collect();
                    let fields:Vec<_>=result.arrays["eigenvectors"].data[ki*4*n*n*2..(ki+1)*4*n*n*2]
                        .chunks_exact(n*n*2).map(|x| Field2D::from_vec(core.grid,x.chunks_exact(2)
                            .map(|z|Complex64::new(z[0],z[1])).collect())).collect();
                    let fresh=certify_without_refinement(&mut theta,&values,&fields);
                    let mut a=frequencies.to_vec();let mut b=reference.arrays["frequencies"].data[ki*4..(ki+1)*4].to_vec();
                    a.sort_by(f64::total_cmp);b.sort_by(f64::total_cmp);
                    let frequency_error=a.iter().zip(b).map(|(a,b)|(a-b).abs()).fold(0.,f64::max);
                    let pass=fresh.b_orthogonality_defect<=accuracy.max_orthogonality_defect
                        && fresh.residuals.iter().zip(&fresh.absolute_residuals).all(|(&r,&a)|
                            r<=accuracy.max_relative_residual || a<=accuracy.max_absolute_residual)
                        && frequency_error<2e-5;
                    valid &= pass;
                    points.push(json!({"k_index":ki,"pass":pass,"f64_residuals":fresh.residuals,
                        "f64_absolute_residuals":fresh.absolute_residuals,"f64_orthogonality":fresh.b_orthogonality_defect,
                        "max_frequency_difference":frequency_error}));
                }
                audits.push(json!({"job":result.job_index,"points":points,"warm_start":result.metadata["geometry_warm_start"],
                    "iterations":result.metadata["iterations"]}));
            }
            passed &= valid;
            eprintln!("{lattice:?} {pol:?} {method:?}: pass={valid}, completed={}, failed={}",study.results.len(),study.errors.len());
            records.push(json!({"config":c,"method":method,"pass":valid,"audits":audits,"errors":study.errors}));
        }
      }
    }
    println!("{}",json!({"scope":"f32 storage audited by fresh f64 operator actions; accuracy only, not timing",
        "resolution":n,"accuracy":accuracy,"pass":passed,"records":records}));
    if !passed {std::process::exit(1);}
}
