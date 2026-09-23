use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{dielectric::Dielectric2D,grid::Grid2D,geometry::Geometry2D,lattice::Lattice2D,operators::ThetaOperator,polarization::Polarization,field::Field2D,
    symmetry::seed::rank_aware_seed};
use num_complex::Complex64 as C;

#[test]
fn two_nonzero_projected_columns_can_have_only_one_target_mass_direction() {
    let grid=Grid2D::new(5,3,1.0,1.0);let d=Dielectric2D::from_geometry(&Geometry2D{lattice:Lattice2D::square(1.0),eps_bg:3.0,atoms:vec![]},grid,&Default::default());
    let mut operator=ThetaOperator::new(CpuBackend::<f64>::new(),d,Polarization::TM,[0.2,0.3]);
    let mut even=Field2D::zeros(grid);even.as_mut_slice()[0]=C::new(1.0/2.0_f64.sqrt(),0.0);
    // P_even (e+o)/sqrt(2) = P_even (e-o)/sqrt(2).
    let seed=rank_aware_seed(&mut operator,&[even.clone(),even],1e-9).unwrap();
    assert_eq!(seed.report.source_columns,2);assert_eq!(seed.report.retained_rank,1);assert_eq!(seed.fields.len(),1);assert!(seed.report.b_orthogonality_defect<1e-12);
    assert!((seed.fields[0].as_slice().iter().map(|z|3.0*z.norm_sqr()).sum::<f64>()-1.0).abs()<1e-12);
    assert_eq!(seed.report.singular_values.len(),2);assert!(seed.report.singular_values[0]<1e-12);
}

#[test]
fn target_ritz_recovers_plane_waves_from_a_mixed_seed_span() {
    let grid=Grid2D::new(5,3,1.0,1.0);let lattice=Lattice2D::square(1.0);let d=Dielectric2D::from_geometry(&Geometry2D{lattice,eps_bg:2.0,atoms:vec![]},grid,&Default::default());
    let k=[0.2,0.3];let mut operator=ThetaOperator::new(CpuBackend::<f64>::new(),d,Polarization::TM,k);
    let mut a=Field2D::zeros(grid);let mut b=a.clone();
    for j in 0..grid.ny {for i in 0..grid.nx {let z=C::from_polar(1.0,std::f64::consts::TAU*i as f64/grid.nx as f64);a.as_mut_slice()[grid.idx(i,j)]=C::ONE+z;b.as_mut_slice()[grid.idx(i,j)]=C::ONE-z;}}
    let seed=rank_aware_seed(&mut operator,&[a,b],1e-9).unwrap();assert_eq!(seed.fields.len(),2);
    let expected=[(k[0]*k[0]+k[1]*k[1])/2.0,((k[0]+std::f64::consts::TAU).powi(2)+k[1]*k[1])/2.0];
    for (a,b) in seed.report.initial_ritz_values.iter().zip(expected) {assert!((a-b).abs()<1e-12);}
    assert!(seed.report.target_residuals.iter().all(|&x|x<1e-12));
    let zero=rank_aware_seed(&mut operator,&[Field2D::zeros(grid)],1e-9).unwrap();assert_eq!(zero.report.retained_rank,0);
}

#[test]
fn rank_transfer_preserves_the_natural_path_spectrum_and_fields() {
    use blaze2d_core::{drivers::bandstructure::{BandStructureJob,RunOptions,run_with_k_streaming},eigensolver::EigensolverConfig,symmetry::{seed::SeedTransfer,reuse::SymmetryOptions}};
    for pol in [Polarization::TM,Polarization::TE] {
        let job=BandStructureJob {geom:Geometry2D{lattice:Lattice2D::square(1.0),eps_bg:1.0,atoms:vec![blaze2d_core::geometry::BasisAtom{pos:[0.0;2],radius:0.2,eps_inside:8.9}]},grid:Grid2D::new(13,13,1.0,1.0),pol,k_path:vec![[0.0,0.0],[0.17,0.07],[0.19,0.11]],eigensolver:EigensolverConfig{n_bands:5,tol:1e-12,max_iter:500,..Default::default()},dielectric:Default::default()};
        let run=|seed_transfer|{let mut points=Vec::new();run_with_k_streaming(CpuBackend::<f64>::new(),&job,RunOptions{symmetry:SymmetryOptions{seed_transfer,..Default::default()},reuse_gamma:false,disable_band_tracking:true,retain_eigenvectors:true,..Default::default()},|p|points.push(p));points};
        let full=run(SeedTransfer::Copy);let ranked=run(SeedTransfer::RankAware);
        assert!(ranked[0].symmetry.seed.is_none());assert!(ranked[1..].iter().all(|p|p.symmetry.seed.as_ref().unwrap().retained_rank==5));
        for (a,b) in full.iter().zip(&ranked) {assert!(b.converged);assert_eq!(b.eigenvectors.as_ref().unwrap().len(),5);assert!(b.b_orthogonality_defect<1e-8);for (&x,&y) in a.omegas.iter().zip(&b.omegas) {assert!((x-y).abs()<1e-6,"{pol:?}: {x} {y}");}}
    }
}
