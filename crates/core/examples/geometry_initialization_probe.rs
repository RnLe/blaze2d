//! Diagnostic export/import probe; elapsed time includes observations and is not a timing benchmark.
use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    backend::SpectralBackend,
    dielectric::Dielectric2D,
    drivers::single_solve::{SingleSolveJob, solve},
    eigensolver::{Eigensolver, EigensolverConfig, refine::certify_without_refinement},
    field::Field2D,
    geometry::{BasisAtom, Geometry2D},
    grid::Grid2D,
    lattice::Lattice2D,
    operators::{LinearOperator, ThetaOperator},
    polarization::Polarization,
};
use num_complex::Complex64 as C;
use serde_json::{Value, json};

fn operator<B: SpectralBackend>(backend: B, n: usize, pol: Polarization, radius: f64, k: [f64; 2]) -> ThetaOperator<B> {
    let geometry = Geometry2D {
        lattice: Lattice2D::square(1.0), eps_bg: 1.0,
        atoms: vec![BasisAtom { pos: [0.0; 2], radius, eps_inside: 8.9 }],
    };
    ThetaOperator::new(backend, Dielectric2D::from_geometry(&geometry,
        Grid2D::new(n,n,1.0,1.0), &Default::default()), pol, k)
}

fn metrics(fields: &[Field2D], mass: &[f64], gamma: bool) -> Value {
    let mut defect = 0.0_f64;
    let mut norms = Vec::new();
    let mut leakage = 0.0_f64;
    for (i, x) in fields.iter().enumerate() {
        let norm: f64 = x.as_slice().iter().zip(mass).map(|(v,m)| v.norm_sqr()*m).sum();
        norms.push(norm);
        let mean: C = x.as_slice().iter().zip(mass).map(|(v,m)| v*m).sum();
        if gamma { leakage = leakage.max(mean.norm()/mass.iter().sum::<f64>().sqrt()); }
        for (j,y) in fields.iter().enumerate() {
            let gram: C = x.as_slice().iter().zip(y.as_slice()).zip(mass).map(|((a,b),m)| a.conj()*b*m).sum();
            defect = defect.max((gram-C::new(if i==j {1.0} else {0.0},0.0)).norm());
        }
    }
    json!({"observed_columns":fields.len(), "mass_norm_squared":norms,
        "gram_defect":defect,"kernel_leakage":gamma.then_some(leakage)})
}

fn run<B: SpectralBackend + Clone>(backend: B, precision: &str, n: usize, bands: usize) -> Vec<Value> {
    let mut records = Vec::new();
    for pol in [Polarization::TM, Polarization::TE] {
        let mut source_operator=operator(CpuBackend::<f64>::new(),n,pol,0.2,[0.0;2]);
        let mut preconditioner=source_operator.build_homogeneous_preconditioner_adaptive();
        let source=solve(&mut source_operator,Some(&mut preconditioner),
            &SingleSolveJob::new(bands).with_tolerance(1e-14).with_max_iterations(600));
        for (name, radius, k, count) in [
            ("public_identical",0.2,[0.0;2],bands),
            ("positive_identical",0.2,[0.0;2],bands),
            ("public_radius",0.201,[0.0;2],bands),
            ("positive_radius",0.201,[0.0;2],bands),
            ("constant",0.2,[0.0;2],bands),
            ("scaled_constant",0.2,[0.0;2],bands),
            ("rounded_constant",0.2,[0.0;2],bands),
            ("kernel_only_request",0.2,[0.0;2],1),
            ("near_gamma",0.2,[1e-13,0.0],bands),
            ("nonzero",0.2,[1e-4,0.0],bands),
        ] {
            let warm = if name.contains("constant") {
                let mut field=source.eigenvectors[0].clone();
                for (i,z) in field.as_mut_slice().iter_mut().enumerate() {
                    *z=C::new(if name=="scaled_constant" {1e-16} else {1.0},0.0);
                    if name=="rounded_constant" { z.re += if precision=="f32" {1e-8} else {1e-16} * (i%3) as f64; }
                }
                vec![field]
            } else if name.starts_with("positive") { source.eigenvectors[1..].to_vec() }
            else { source.eigenvectors[..count].to_vec() };
            let mut target=operator(backend.clone(),n,pol,radius,k);
            let mass=if pol==Polarization::TM {target.dielectric().eps().to_vec()} else {vec![1.;n*n]};
            let kernel_available=target.analytic_kernel_available();
            let mut pc=target.build_homogeneous_preconditioner_adaptive();
            let config=EigensolverConfig { n_bands:count, tol:if precision=="f32" {1e-6} else {1e-14}, max_iter:600, ..Default::default() };
            let mut solver=Eigensolver::new(&mut target,config,Some(&mut pc),Some(&warm));
            solver.initialize();
            let initial=metrics(&solver.eigenvectors(),&mass,k==[0.0;2]);
            let init_report=solver.initialization_report().unwrap();
            let initialization=json!({"warm_accepts":init_report.warm_start_hits,
                "warm_rejections":init_report.rejected_warm_vectors,
                "random_columns":init_report.random_vectors,"rank_tolerance":init_report.rank_tolerance});
            let active=solver.block_size();
            let locked=solver.locked_count();
            let result=solver.solve();
            let fields=solver.all_eigenvectors();
            drop(solver);
            let certificate=certify_without_refinement(&mut target,&result.eigenvalues,&fields);
            let mut audit=operator(CpuBackend::<f64>::new(),n,pol,radius,k);
            let audit=certify_without_refinement(&mut audit,&result.eigenvalues,&fields);
            records.push(json!({"case":name,"grid":[n,n],"polarization":format!("{pol:?}"),"precision":precision,
                "bands":count,"radius":radius,"k_cartesian":k,"source_converged":source.converged,
                "source_residuals":source.final_residuals,"analytic_kernel_available":kernel_available,
                "initialization":initialization,"active_columns":active,"locked_columns":locked,"initial":initial,
                "iterations":result.iterations,"converged":result.converged,"eigenvalues":result.eigenvalues,
                "residuals":certificate.residuals,"absolute_residuals":certificate.absolute_residuals,
                "orthogonality":certificate.b_orthogonality_defect,"audit_f64_residuals":audit.residuals,
                "audit_f64_absolute_residuals":audit.absolute_residuals,"audit_f64_orthogonality":audit.b_orthogonality_defect}));
        }
    }
    records
}

fn main() {
    let args:Vec<_>=std::env::args().collect();
    let n=args.get(1).map(|s|s.parse().unwrap()).unwrap_or(13);
    let bands=args.get(2).map(|s|s.parse().unwrap()).unwrap_or(8);
    let mut records=run(CpuBackend::<f64>::new(),"f64",n,bands);
    records.extend(run(CpuBackend::<f32>::new(),"f32",n,bands));
    println!("{}",json!({"scope":"Initialization correctness probe, including f64 auditing of stored f32 fields; no runtime speedup claim",
        "records":records,"counters":blaze2d_core::profiler::get_counters(),
        "profile":serde_json::from_str::<Value>(&blaze2d_core::profiler::get_profile_json()).unwrap()}));
}
