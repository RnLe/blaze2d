use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{dielectric::{Dielectric2D,DielectricOptions},geometry::{Geometry2D,BasisAtom},grid::Grid2D,lattice::Lattice2D,operators::ThetaOperator,polarization::Polarization,
    symmetry::{affine::{AffineOp,PlaneGroup,Rational},action::BlochActionPlan,qualification::{qualify,RejectionReason}}};

fn dielectric(lattice:Lattice2D,grid:Grid2D,center:[f64;2])->Dielectric2D {
    Dielectric2D::from_geometry(&Geometry2D{lattice,eps_bg:1.0,atoms:vec![BasisAtom {pos:center,radius:0.2,eps_inside:8.9}]},grid,&DielectricOptions::default())
}

#[test]
fn actual_smoothed_square_maxwell_pencils_qualify_for_both_polarizations() {
    let lattice=Lattice2D::square(1.0);let grid=Grid2D::new(9,9,1.0,1.0);let d=dielectric(lattice,grid,[0.0;2]);let backend=CpuBackend::<f64>::new();let k=[0.13,0.21];
    for pol in [Polarization::TM,Polarization::TE] {for op in PlaneGroup::P4m.group().operations {
        let target=op.map_wavevector(k);let plan=BlochActionPlan::new(op,k,target,grid,pol,1e-12).unwrap();
        let mut a=ThetaOperator::new(backend.clone(),d.clone(),pol,lattice.reciprocal().fractional_to_cartesian(k));let mut b=ThetaOperator::new(backend.clone(),d.clone(),pol,lattice.reciprocal().fractional_to_cartesian(target));
        let qualified=qualify(plan,&lattice,&mut a,&mut b,1e-10).unwrap();
        assert!(qualified.report().operator_defect<1e-12);assert!(qualified.report().mass_defect<1e-12);
    }}
}

#[test]
fn rejects_broken_motif_hex_index_flip_and_tensor_covariance() {
    let grid=Grid2D::new(9,9,1.0,1.0);let backend=CpuBackend::<f64>::new();
    for (lattice,center,pol,tensor) in [(Lattice2D::square(1.0),[0.17,0.07],Polarization::TM,false),(Lattice2D::hexagonal(1.0),[0.0;2],Polarization::TM,false),(Lattice2D::square(1.0),[0.0;2],Polarization::TE,true)] {
        let mut d=dielectric(lattice,grid,center);
        let op=if tensor {PlaneGroup::P4.group().operations[1]}else{AffineOp::new([[-1,0],[0,1]],[Rational::ZERO;2]).unwrap()};
        if tensor {let r=lattice.reciprocal();d=Dielectric2D::from_sampled_epsilon(grid,r.b1,r.b2,vec![2.0;grid.len()],Some(vec![[0.5,0.0,0.0,1.0];grid.len()]));}
        let plan=BlochActionPlan::new(op,[0.0;2],[0.0;2],grid,pol,1e-12).unwrap();
        let mut a=ThetaOperator::new(backend.clone(),d.clone(),pol,[0.0;2]);let mut b=ThetaOperator::new(backend.clone(),d,pol,[0.0;2]);
        let report=qualify(plan,&lattice,&mut a,&mut b,1e-10).unwrap_err();assert!(!report.authorized);
        assert!(report.rejection_reasons.iter().any(|r|matches!(r,RejectionReason::MaterialMismatch|RejectionReason::MetricMismatch|RejectionReason::TensorCovarianceFailure)));
    }
}

#[test]
fn actual_even_grid_pt_covariance_is_supported_without_aliasing() {
    let lattice=Lattice2D::square(1.0);let grid=Grid2D::new(8,6,1.0,1.0);let d=dielectric(lattice,grid,[0.0;2]);let backend=CpuBackend::<f64>::new();let k=[0.13,0.21];
    let op=PlaneGroup::P2.group().operations[1].with_conjugation(true);
    for pol in [Polarization::TM,Polarization::TE] {
        let plan=BlochActionPlan::new(op,k,k,grid,pol,1e-12).unwrap();let bloch=lattice.reciprocal().fractional_to_cartesian(k);
        let mut a=ThetaOperator::new(backend.clone(),d.clone(),pol,bloch);let mut b=ThetaOperator::new(backend.clone(),d.clone(),pol,bloch);
        qualify(plan,&lattice,&mut a,&mut b,1e-10).unwrap();
    }
}

#[test]
fn nonfinite_operator_probe_cannot_authorize_reconstruction() {
    let grid=Grid2D::new(5,5,1.0,1.0);let lattice=Lattice2D::square(1.0);let r=lattice.reciprocal();
    let d=Dielectric2D::from_sampled_epsilon(grid,r.b1,r.b2,vec![2.0;grid.len()],Some(vec![[1e308,0.0,0.0,1e308];grid.len()]));
    let backend=CpuBackend::<f64>::new();let mut a=ThetaOperator::new(backend.clone(),d.clone(),Polarization::TE,[0.0;2]);let mut b=ThetaOperator::new(backend,d,Polarization::TE,[0.0;2]);
    let plan=BlochActionPlan::new(AffineOp::IDENTITY,[0.0;2],[0.0;2],grid,Polarization::TE,1e-12).unwrap();
    assert!(qualify(plan,&lattice,&mut a,&mut b,1e-10).is_err());
}
