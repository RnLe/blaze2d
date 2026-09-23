use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    backend::SpectralBackend, dielectric::Dielectric2D, field::Field2D,
    geometry::{BasisAtom,Geometry2D}, grid::Grid2D, lattice::Lattice2D,
    operators::{LinearOperator,ThetaOperator}, polarization::Polarization,
    symmetry::{action::BlochActionPlan,affine::PlaneGroup,qualification::qualify,
        representation::RepresentationPlan,reduced::OrbitBasis,mirror_kernel::{MirrorYKernel,MirrorYTeKernel}},
};
use num_complex::Complex64 as C;

#[test]
fn reduced_tm_mirror_kernel_matches_the_same_finite_pencil() {
    let lattice=Lattice2D::square(1.0);
    for (nx,ny,k) in [(8,7,[0.0;2]),(13,9,[0.17,0.0]),(12,13,[0.5,0.0]),(12,8,[0.17,-0.5])] {
        let grid=Grid2D::new(nx,ny,1.0,1.0);
        let d=Dielectric2D::from_geometry(&Geometry2D {lattice,eps_bg:1.0,
            atoms:vec![BasisAtom {pos:[0.13,0.0],radius:0.21,eps_inside:8.9}]},grid,&Default::default());
        let backend=CpuBackend::<f64>::new();
        let mut op=ThetaOperator::new(backend.clone(),d.clone(),Polarization::TM,lattice.reciprocal().fractional_to_cartesian(k));
        let mut other=ThetaOperator::new(backend.clone(),d,Polarization::TM,op.bloch());
        let representation=RepresentationPlan::ordinary(PlaneGroup::Pm.group(),k).unwrap();
        let actions:Vec<_>=representation.group.operations.iter().map(|&g|
            qualify(BlochActionPlan::new(g,k,k,grid,Polarization::TM,1e-12).unwrap(),&lattice,&mut op,&mut other,1e-10).unwrap()).collect();
        for irrep in 0..representation.irreps.len() {
            let weights=representation.isotypic_weights(irrep);
            let rank=actions.iter().zip(&weights).map(|(a,w)|a.plan().spatial_trace()*w).sum::<C>().re.round() as usize;
            let basis=OrbitBasis::from_projector(&actions,&weights,rank).unwrap();
            if ny%2==0 {assert!(MirrorYKernel::new(&op,&basis).is_err());continue;}
            let mut kernel=MirrorYKernel::new(&op,&basis).unwrap();
            let mut input=backend.alloc_field(basis.coordinate_grid());
            for (i,z) in input.as_mut_slice().iter_mut().enumerate() {*z=C::new((i as f64*0.723).sin(),(i as f64*1.31).cos());}
            let mut spatial=op.alloc_field();basis.expand(&input,&mut spatial);
            let mut applied=op.alloc_field();let mut reference=Field2D::zeros(basis.coordinate_grid());let mut reduced=reference.clone();
            op.apply(&spatial,&mut applied);basis.restrict(&applied,&mut reference);
            #[cfg(feature="profiling")]
            let before=fft_elements();
            kernel.apply(&backend,&input,&mut reduced);
            #[cfg(feature="profiling")]
            assert_eq!(fft_elements()-before,2*rank as u64);
            assert_close(reference.as_slice(),reduced.as_slice());
            op.apply_mass(&spatial,&mut applied);basis.restrict(&applied,&mut reference);
            kernel.apply_mass(&input,&mut reduced);assert_close(reference.as_slice(),reduced.as_slice());
        }
    }
}

#[test]
fn reduced_tm_mirror_prefixes_preserve_the_physical_spectrum() {
    use blaze2d_core::{eigensolver::EigensolverConfig,symmetry::{reuse::SymmetryOptions,
        reduced::{SectorStorage,ReducedKernel},sectors::{SectorPolicy,solve_conservative}}};
    let lattice=Lattice2D::square(1.0);let grid=Grid2D::new(13,13,1.0,1.0);
    let d=Dielectric2D::from_geometry(&Geometry2D {lattice,eps_bg:1.0,
        atoms:vec![BasisAtom {pos:[0.0;2],radius:0.2,eps_inside:8.9}]},grid,&Default::default());
    let mut warm=None;
    for k in [[0.0;2],[0.17,0.0],[0.5,0.0]] {
        let mut op=ThetaOperator::new(CpuBackend::<f64>::new(),d.clone(),Polarization::TM,lattice.reciprocal().fractional_to_cartesian(k));
        let mut p=op.build_homogeneous_preconditioner_adaptive();
        let config=EigensolverConfig {n_bands:8,tol:1e-14,max_iter:500,..Default::default()};
        let mut options=SymmetryOptions {sectors:SectorPolicy::AdaptiveGroup,sector_storage:SectorStorage::Orbit,sector_guard_budget:Some(2),..Default::default()};
        let reference=solve_conservative(&mut op,&lattice,k,PlaneGroup::Pm.group(),&config,warm.as_deref(),&options,Some(&mut p)).unwrap();
        options.sector_kernel=ReducedKernel::MirrorTm;
        let candidate=solve_conservative(&mut op,&lattice,k,PlaneGroup::Pm.group(),&config,warm.as_deref(),&options,Some(&mut p)).unwrap();
        for report in &candidate.report.sectors {
            if !report.dense_complete {
                let kernel=report.reduced_kernel.as_ref().unwrap();assert_eq!(kernel.applied,ReducedKernel::MirrorTm);
                assert_eq!(kernel.tangent_fft_length,grid.nx);assert_eq!(kernel.normal_dense_dimension*grid.nx,report.dimension);
            }
        }
        for (&x,&y) in reference.spectrum.eigenvalues.iter().zip(&candidate.spectrum.eigenvalues) {assert!((x-y).abs()/x.abs().max(1.0)<1e-8);}
        warm=Some(candidate.spectrum.fields);
    }
}

#[test]
fn reduced_te_mirror_kernel_preserves_interface_and_off_diagonal_tensor_terms() {
    let lattice=Lattice2D::square(1.0);
    for (nx,ny,k) in [(8,7,[0.0;2]),(13,9,[0.17,0.0]),(12,13,[0.5,0.0])] {
        let grid=Grid2D::new(nx,ny,1.0,1.0);
        for analytic_tensor in [false,true] {
            let d=if analytic_tensor {
                let tensor=(0..grid.len()).map(|index| {
                    let x=std::f64::consts::TAU*(index%nx) as f64/nx as f64;
                    let y=std::f64::consts::TAU*(index/nx) as f64/ny as f64;
                    let xy=0.07*y.sin()*(1.0+0.2*x.cos());
                    [0.6+0.1*y.cos(),xy,xy,0.8+0.1*x.cos()]
                }).collect();
                let r=lattice.reciprocal();
                Dielectric2D::from_sampled_epsilon(grid,r.b1,r.b2,vec![2.0;grid.len()],Some(tensor))
            } else {
                Dielectric2D::from_geometry(&Geometry2D {lattice,eps_bg:1.0,
                    atoms:vec![BasisAtom {pos:[0.13,0.0],radius:0.21,eps_inside:8.9}]},grid,&Default::default())
            };
            let backend=CpuBackend::<f64>::new();
            let mut op=ThetaOperator::new(backend.clone(),d.clone(),Polarization::TE,lattice.reciprocal().fractional_to_cartesian(k));
            let mut other=ThetaOperator::new(backend.clone(),d,Polarization::TE,op.bloch());
            let representation=RepresentationPlan::ordinary(PlaneGroup::Pm.group(),k).unwrap();
            let actions:Vec<_>=representation.group.operations.iter().map(|&g|
                qualify(BlochActionPlan::new(g,k,k,grid,Polarization::TE,1e-12).unwrap(),&lattice,&mut op,&mut other,1e-10).unwrap()).collect();
            for irrep in 0..representation.irreps.len() {
                let weights=representation.isotypic_weights(irrep);
                let rank=actions.iter().zip(&weights).map(|(a,w)|a.plan().spatial_trace()*w).sum::<C>().re.round() as usize;
                let basis=OrbitBasis::from_projector(&actions,&weights,rank).unwrap();
                let mut kernel=MirrorYTeKernel::new(&op,&basis).unwrap();
                let mut input=backend.alloc_field(basis.coordinate_grid());
                for (i,z) in input.as_mut_slice().iter_mut().enumerate() {*z=C::new((i as f64*0.723).sin(),(i as f64*1.31).cos());}
                let mut spatial=op.alloc_field();basis.expand(&input,&mut spatial);
                let mut applied=op.alloc_field();let mut reference=Field2D::zeros(basis.coordinate_grid());let mut reduced=reference.clone();
                op.apply(&spatial,&mut applied);basis.restrict(&applied,&mut reference);
                #[cfg(feature="profiling")]
                let before=fft_elements();
                kernel.apply(&backend,&input,&mut reduced);
                #[cfg(feature="profiling")]
                assert_eq!(fft_elements()-before,4*rank as u64);
                assert_close(reference.as_slice(),reduced.as_slice());
                op.apply_mass(&spatial,&mut applied);basis.restrict(&applied,&mut reference);
                kernel.apply_mass(&input,&mut reduced);assert_close(reference.as_slice(),reduced.as_slice());
            }
        }
    }
}

#[test]
fn reduced_te_mirror_prefixes_reconstruct_certified_fields() {
    use blaze2d_core::{eigensolver::EigensolverConfig,symmetry::{reuse::SymmetryOptions,
        reduced::{SectorStorage,ReducedKernel},sectors::{SectorPolicy,solve_conservative}}};
    let lattice=Lattice2D::square(1.0);let grid=Grid2D::new(13,13,1.0,1.0);
    let d=Dielectric2D::from_geometry(&Geometry2D {lattice,eps_bg:1.0,
        atoms:vec![BasisAtom {pos:[0.0;2],radius:0.2,eps_inside:8.9}]},grid,&Default::default());
    let mut warm=None;
    for k in [[0.0;2],[0.17,0.0],[0.5,0.0]] {
        let mut op=ThetaOperator::new(CpuBackend::<f64>::new(),d.clone(),Polarization::TE,lattice.reciprocal().fractional_to_cartesian(k));
        let mut p=op.build_transverse_projection_preconditioner_adaptive();
        let config=EigensolverConfig {n_bands:8,tol:1e-14,max_iter:500,..Default::default()};
        let mut options=SymmetryOptions {sectors:SectorPolicy::AdaptiveGroup,sector_storage:SectorStorage::Orbit,sector_guard_budget:Some(2),..Default::default()};
        let reference=solve_conservative(&mut op,&lattice,k,PlaneGroup::Pm.group(),&config,warm.as_deref(),&options,Some(&mut p)).unwrap();
        options.sector_kernel=ReducedKernel::MirrorTmTe;
        let candidate=solve_conservative(&mut op,&lattice,k,PlaneGroup::Pm.group(),&config,warm.as_deref(),&options,Some(&mut p)).unwrap();
        assert_eq!(candidate.report.kernel_owners,usize::from(k==[0.0;2]));
        for report in &candidate.report.sectors {if !report.dense_complete {assert_eq!(report.reduced_kernel.as_ref().unwrap().applied,ReducedKernel::MirrorTmTe);}}
        for (&x,&y) in reference.spectrum.eigenvalues.iter().zip(&candidate.spectrum.eigenvalues) {assert!((x-y).abs()/x.abs().max(1.0)<1e-8);}
        warm=Some(candidate.spectrum.fields);
    }
}

fn assert_close(a:&[C],b:&[C]) {
    let error=a.iter().zip(b).map(|(x,y)|(x-y).norm_sqr()).sum::<f64>().sqrt();
    let norm=a.iter().map(C::norm_sqr).sum::<f64>().sqrt();assert!(error<2e-11*norm.max(1.0),"{error:e} / {norm:e}");
}
#[cfg(feature="profiling")]
fn fft_elements()->u64 {let c=blaze2d_core::profiler::get_counters();c.get("fft.forward.elements").copied().unwrap_or(0)+c.get("fft.inverse.elements").copied().unwrap_or(0)}
