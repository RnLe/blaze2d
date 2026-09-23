use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{dielectric::Dielectric2D,grid::Grid2D,geometry::Geometry2D,lattice::Lattice2D,operators::ThetaOperator,polarization::Polarization,field::Field2D,
    symmetry::{cache::{SpectralSnapshot,SnapshotIdentity,SnapshotError},reuse::CachedSpectrum}};

#[test]
fn cache_replay_is_lossless_and_rejects_an_incompatible_problem() {
    let grid=Grid2D::new(5,3,1.0,1.0);let lattice=Lattice2D::square(1.0);
    let d=Dielectric2D::from_geometry(&Geometry2D{lattice,eps_bg:2.0,atoms:vec![]},grid,&Default::default());
    let mut operator=ThetaOperator::new(CpuBackend::<f64>::new(),d,Polarization::TM,[0.1,0.2]);let id=SnapshotIdentity::from_operator(&operator);
    let mut field=Field2D::zeros(grid);for (i,z) in field.as_mut_slice().iter_mut().enumerate() {*z=num_complex::Complex64::new((i as f64*1.71).sin(),-0.0);}
    let original=CachedSpectrum {source_index:3,k:[0.1/std::f64::consts::TAU,0.2/std::f64::consts::TAU],eigenvalues:vec![1.2345678901234567],fields:vec![field],residuals:vec![1e-10],absolute_residuals:vec![3e-10],orthogonality_defect:1e-15};
    let snapshot=SpectralSnapshot::capture(&original,id.clone(),"fixture");let json=serde_json::to_string(&snapshot).unwrap();let parsed:SpectralSnapshot=serde_json::from_str(&json).unwrap();
    let replay=parsed.restore(&id).unwrap();assert_eq!(replay.eigenvalues[0].to_bits(),original.eigenvalues[0].to_bits());
    for (a,b) in replay.fields[0].as_slice().iter().zip(original.fields[0].as_slice()) {assert_eq!(a.re.to_bits(),b.re.to_bits());assert_eq!(a.im.to_bits(),b.im.to_bits());}
    assert_eq!(replay.residuals,original.residuals);let mut wrong=id.clone();wrong.operator_bits[9]^=1;assert_eq!(snapshot.clone().restore(&wrong).unwrap_err(),SnapshotError::Incompatible);
    assert_eq!(snapshot.clone().restore_checked(&mut operator,&Default::default()).unwrap_err(),SnapshotError::FailedCertification);
    let mut corrupt=snapshot.clone();corrupt.field_bits[0][0][0]=f64::NAN.to_bits();assert_eq!(corrupt.restore(&id).unwrap_err(),SnapshotError::InvalidData);
    let mut corrupt=snapshot;corrupt.cluster_ranges=vec![[0,0]];assert_eq!(corrupt.restore(&id).unwrap_err(),SnapshotError::InvalidData);
}
