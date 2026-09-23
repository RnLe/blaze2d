//! Fixed-input operator experiments, independent of eigensolver trajectories.
use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    backend::SpectralBackend, dielectric::Dielectric2D, geometry::{BasisAtom,Geometry2D},
    grid::Grid2D,lattice::Lattice2D,operators::{LinearOperator,ThetaOperator},polarization::Polarization,
    symmetry::{action::BlochActionPlan,affine::PlaneGroup,qualification::qualify,
        representation::RepresentationPlan,reduced::OrbitBasis,mirror_kernel::{MirrorYKernel,MirrorYTeKernel}},
};
use num_complex::Complex64 as C;
use serde_json::json;

fn main() {
    let args:Vec<_>=std::env::args().collect();
    let n:usize=args.get(1).map(|s|s.parse().unwrap()).unwrap_or(31);
    let pol=match args.get(2).map(String::as_str).unwrap_or("TM") {"TM"=>Polarization::TM,"TE"=>Polarization::TE,_=>panic!("expected TM or TE")};
    let mode=args.get(3).map(String::as_str).unwrap_or("full");assert!(matches!(mode,"full"|"reduced"));
    let repetitions:usize=args.get(4).map(|s|s.parse().unwrap()).unwrap_or(1000);assert!(repetitions>0);
    let sector:usize=args.get(5).map(|s|s.parse().unwrap()).unwrap_or(0);assert!(sector<2);
    let lattice=Lattice2D::square(1.0);let grid=Grid2D::new(n,n,1.0,1.0);let k=[0.17,0.0];
    let backend=CpuBackend::<f64>::new();
    let d=Dielectric2D::from_geometry(&Geometry2D {lattice,eps_bg:1.0,
        atoms:vec![BasisAtom {pos:[0.0;2],radius:0.2,eps_inside:8.9}]},grid,&Default::default());
    let mut op=ThetaOperator::new(backend.clone(),d.clone(),pol,lattice.reciprocal().fractional_to_cartesian(k));
    let mut other=ThetaOperator::new(backend.clone(),d,pol,op.bloch());
    let representation=RepresentationPlan::ordinary(PlaneGroup::Pm.group(),k).unwrap();
    let actions:Vec<_>=representation.group.operations.iter().map(|&g|
        qualify(BlochActionPlan::new(g,k,k,grid,pol,1e-12).unwrap(),&lattice,&mut op,&mut other,1e-10).unwrap()).collect();
    let weights=representation.isotypic_weights(sector);
    let rank=actions.iter().zip(&weights).map(|(a,w)|a.plan().spatial_trace()*w).sum::<C>().re.round() as usize;
    let basis=OrbitBasis::from_projector(&actions,&weights,rank).unwrap();
    let mut tm=if pol==Polarization::TM {Some(MirrorYKernel::new(&op,&basis).unwrap())} else {None};
    let mut te=if pol==Polarization::TE {Some(MirrorYTeKernel::new(&op,&basis).unwrap())} else {None};
    let mut input=backend.alloc_field(basis.coordinate_grid());
    for (i,z) in input.as_mut_slice().iter_mut().enumerate() {*z=C::new((i as f64*0.723).sin(),(i as f64*1.31).cos());}
    let mut output=input.clone();let mut spatial=op.alloc_field();let mut applied=op.alloc_field();let mut reference=input.clone();
    basis.expand(&input,&mut spatial);op.apply(&spatial,&mut applied);basis.restrict(&applied,&mut reference);
    let mut apply=|out:&mut blaze2d_core::field::Field2D| {
        if mode=="full" {basis.expand(std::hint::black_box(&input),&mut spatial);op.apply(&spatial,&mut applied);basis.restrict(&applied,out);}
        else if let Some(kernel)=&mut tm {kernel.apply(&backend,std::hint::black_box(&input),out);}
        else {te.as_mut().unwrap().apply(&backend,std::hint::black_box(&input),out);}
        std::hint::black_box(out);
    };
    for _ in 0..20 {apply(&mut output);}
    blaze2d_core::profiler::reset();
    let start=std::time::Instant::now();for _ in 0..repetitions {apply(&mut output);}let seconds=start.elapsed().as_secs_f64();
    let error=output.as_slice().iter().zip(reference.as_slice()).map(|(x,y)|(x-y).norm_sqr()).sum::<f64>().sqrt();
    let norm=reference.as_slice().iter().map(C::norm_sqr).sum::<f64>().sqrt();let relative_error=error/norm.max(1e-300);
    println!("{}",json!({"mode":mode,"grid":[n,n],"polarization":format!("{pol:?}"),"sector":sector,"stored_rows":rank,
        "repetitions":repetitions,"seconds":seconds,"relative_error":relative_error,"finite":output.as_slice().iter().all(|z|z.re.is_finite()&&z.im.is_finite()),
        "counters":blaze2d_core::profiler::get_counters(),"profile":serde_json::from_str::<serde_json::Value>(&blaze2d_core::profiler::get_profile_json()).unwrap()}));
}
