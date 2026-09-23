//! Export once, then replay the identical physical seed for paired experiments.
use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{dielectric::Dielectric2D,geometry::{Geometry2D,BasisAtom},grid::Grid2D,lattice::Lattice2D,operators::ThetaOperator,polarization::Polarization,
    drivers::single_solve::{solve,solve_with_warmstart,SingleSolveJob},eigensolver::refine::certify_without_refinement,
    symmetry::{cache::{SpectralSnapshot,SnapshotIdentity},reuse::CachedSpectrum}};
use serde_json::json;

fn main() {
    let args:Vec<_>=std::env::args().collect();let mode=args.get(1).map(String::as_str).unwrap_or("copy");let file=args.get(2).expect("snapshot path");
    let n:usize=args.get(3).map(|s|s.parse().unwrap()).unwrap_or(24);let bands:usize=args.get(4).map(|s|s.parse().unwrap()).unwrap_or(8);
    let pol=if args.get(5).is_some_and(|s|s=="TE"){Polarization::TE}else{Polarization::TM};
    let lattice=Lattice2D::square(1.0);let grid=Grid2D::new(n,n,1.0,1.0);let source_k=[0.17,0.07];let target_k=[0.19,0.11];
    let backend=CpuBackend::<f64>::new();let d=Dielectric2D::from_geometry(&Geometry2D{lattice,eps_bg:1.0,atoms:vec![BasisAtom {pos:[0.0;2],radius:0.2,eps_inside:8.9}]},grid,&Default::default());
    let mut source=ThetaOperator::new(backend.clone(),d.clone(),pol,lattice.reciprocal().fractional_to_cartesian(source_k));
    let job=SingleSolveJob::new(bands).with_tolerance(1e-12).with_max_iterations(500);
    if mode=="export" {
        let mut precond=source.build_homogeneous_preconditioner_adaptive();let result=solve(&mut source,Some(&mut precond),&job);assert!(result.converged);
        let cert=certify_without_refinement(&mut source,&result.eigenvalues,&result.eigenvectors);
        let entry=CachedSpectrum {source_index:0,k:source_k,eigenvalues:result.eigenvalues,fields:result.eigenvectors,residuals:cert.residuals,absolute_residuals:cert.absolute_residuals,orthogonality_defect:cert.b_orthogonality_defect};
        let snapshot=SpectralSnapshot::capture(&entry,SnapshotIdentity::from_operator(&source),option_env!("BLAZE_BENCH_REVISION").unwrap_or("recorded by harness"));
        let start=std::time::Instant::now();let bytes=serde_json::to_vec(&snapshot).unwrap();let encode_seconds=start.elapsed().as_secs_f64();std::fs::write(file,&bytes).unwrap();
        println!("{}",json!({"source_solve_seconds":result.elapsed_seconds,"source_iterations":result.iterations,"snapshot_bytes":bytes.len(),"encode_seconds":encode_seconds}));return;
    }
    assert!(mode=="copy" || mode=="rank");
    let start=std::time::Instant::now();
    let bytes=std::fs::read(file).unwrap();let snapshot:SpectralSnapshot=serde_json::from_slice(&bytes).unwrap();let decoded=snapshot.restore_checked(&mut source,&Default::default()).unwrap();
    let replay_seconds=start.elapsed().as_secs_f64();
    let mut target=ThetaOperator::new(backend,d,pol,lattice.reciprocal().fractional_to_cartesian(target_k));let mut precond=target.build_homogeneous_preconditioner_adaptive();
    let prepared=if mode=="rank" {Some(blaze2d_core::symmetry::seed::rank_aware_seed(&mut target,&decoded.fields,1e-9).unwrap())} else {None};
    let input=prepared.as_ref().map(|s|s.fields.as_slice()).unwrap_or(&decoded.fields);
    let result=solve_with_warmstart(&mut target,Some(&mut precond),&job,input);
    println!("{}",json!({"mode":mode,"grid":[n,n],"bands":bands,"polarization":format!("{pol:?}"),"preconditioner":"homogeneous_fourier_diagonal","seed":prepared.map(|s|s.report),"source_k":source_k,"target_k":target_k,"replay_seconds":replay_seconds,"seconds":start.elapsed().as_secs_f64(),"iterations":result.iterations,"converged":result.converged,"eigenvalues":result.eigenvalues,"residuals":result.final_residuals,"orthogonality":result.b_orthogonality_defect,"counters":blaze2d_core::profiler::get_counters()}));
}
