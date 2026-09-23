use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{dielectric::Dielectric2D,geometry::{Geometry2D,BasisAtom},grid::Grid2D,lattice::Lattice2D,operators::ThetaOperator,polarization::Polarization,eigensolver::EigensolverConfig,
    drivers::single_solve::{solve,SingleSolveJob},symmetry::fourier::solve_fourier};

#[test]
fn unitary_fourier_coordinates_preserve_physical_fields_and_spectra() {
    let lattice=Lattice2D::square(1.0);
    for n in [12,13] {for pol in [Polarization::TM,Polarization::TE] {
        let grid=Grid2D::new(n,n,1.0,1.0);let d=Dielectric2D::from_geometry(&Geometry2D{lattice,eps_bg:1.0,atoms:vec![BasisAtom{pos:[0.0;2],radius:0.2,eps_inside:8.9}]},grid,&Default::default());
        let mut warm=None;
        for k in [[0.0,0.0],[0.17,0.11],[0.5,0.0]] {
            let mut op=ThetaOperator::new(CpuBackend::<f64>::new(),d.clone(),pol,lattice.reciprocal().fractional_to_cartesian(k));let mut p=op.build_homogeneous_preconditioner_adaptive();
            let reference=solve(&mut op,Some(&mut p),&SingleSolveJob::new(6).with_tolerance(1e-14).with_max_iterations(500));
            let config=EigensolverConfig{n_bands:6,tol:1e-14,max_iter:500,..Default::default()};
            let candidate=solve_fourier(&mut op,&lattice,k,&config,warm.as_deref(),&Default::default(),Some(&mut p)).unwrap_or_else(|e|panic!("{n} {pol:?} {k:?}: {e}"));
            assert_eq!(candidate.report.bytes_per_column,grid.len()*16);assert_eq!(candidate.spectrum.fields.len(),6);
            for (a,b) in reference.eigenvalues.iter().zip(&candidate.spectrum.eigenvalues) {assert!((a-b).abs()/a.abs().max(1.0)<1e-7,"{n} {pol:?} {k:?}: {a} {b}");}
            warm=Some(candidate.spectrum.fields);
        }
    }}
}

#[test]
fn packed_pt_coordinates_reproduce_all_modes_with_unchanged_scalar_precision() {
    use blaze2d_core::symmetry::{fourier::Arithmetic,reuse::SymmetryOptions,sectors::solve_conservative,affine::PlaneGroup};
    let lattice=Lattice2D::square(1.0);
    for ny in [4,5] {for pol in [Polarization::TM,Polarization::TE] {
        let grid=Grid2D::new(5,ny,1.0,1.0);let d=Dielectric2D::from_geometry(&Geometry2D{lattice,eps_bg:1.0,atoms:vec![BasisAtom{pos:[0.0;2],radius:0.2,eps_inside:8.9}]},grid,&Default::default());
        for k in [[0.0;2],[0.17,0.11],[0.5,0.0]] {
            let mut op=ThetaOperator::new(CpuBackend::<f64>::new(),d.clone(),pol,lattice.reciprocal().fractional_to_cartesian(k));let mut precond=op.build_homogeneous_preconditioner_adaptive();
            let config=EigensolverConfig{n_bands:grid.len(),tol:1e-13,max_iter:500,..Default::default()};
            let reference=solve_conservative(&mut op,&lattice,k,PlaneGroup::P1.group(),&config,None,&Default::default(),None).unwrap();
            for arithmetic in [Arithmetic::PtComplex,Arithmetic::PtRealStorage,Arithmetic::PtRealVectors,Arithmetic::PtRealGram,Arithmetic::PtRealDense,Arithmetic::PtRealTm,Arithmetic::PtRealTe,Arithmetic::PtRealReferenceVectors] {
                let options=SymmetryOptions{arithmetic,..Default::default()};
                let candidate=solve_fourier(&mut op,&lattice,k,&config,None,&options,Some(&mut precond)).unwrap_or_else(|e|panic!("{grid:?} {pol:?} {k:?} {arithmetic:?}: {e}"));
                assert_eq!(candidate.spectrum.fields.len(),grid.len());assert!(candidate.report.maximum_real_subspace_leakage<1e-10);
                assert_eq!(candidate.report.bytes_per_column,if arithmetic!=Arithmetic::PtComplex {grid.len().div_ceil(2)*16} else {grid.len()*16});
                for (x,y) in reference.spectrum.eigenvalues.iter().zip(candidate.spectrum.eigenvalues) {assert!((x-y).abs()/x.abs().max(1.0)<1e-8,"{pol:?}: {x} {y}");}
            }
        }
    }}
}

#[test]
fn shifted_pt_center_requires_its_gauge_and_broken_origin_is_rejected() {
    use blaze2d_core::symmetry::{fourier::Arithmetic,reuse::SymmetryOptions,affine::{AffineOp,Rational}};
    let lattice=Lattice2D::square(1.0);let grid=Grid2D::new(12,12,1.0,1.0);let k=[0.13,0.21];
    for pol in [Polarization::TM,Polarization::TE] {
        let d=Dielectric2D::from_geometry(&Geometry2D{lattice,eps_bg:1.0,atoms:vec![BasisAtom{pos:[0.25,0.0],radius:0.2,eps_inside:8.9}]},grid,&Default::default());
        let mut op=ThetaOperator::new(CpuBackend::<f64>::new(),d,pol,lattice.reciprocal().fractional_to_cartesian(k));let mut p=op.build_homogeneous_preconditioner_adaptive();
        let config=EigensolverConfig{n_bands:5,tol:1e-14,max_iter:500,..Default::default()};
        let mut options=SymmetryOptions{arithmetic:Arithmetic::PtRealTe,..Default::default()};
        assert!(solve_fourier(&mut op,&lattice,k,&config,None,&options,Some(&mut p)).is_err());
        options.pt_operation=Some(AffineOp::new([[-1,0],[0,-1]],[Rational::HALF,Rational::ZERO]).unwrap().with_conjugation(true));
        let packed=solve_fourier(&mut op,&lattice,k,&config,None,&options,Some(&mut p)).unwrap();
        let full=solve(&mut op,Some(&mut p),&SingleSolveJob::new(5).with_tolerance(1e-14).with_max_iterations(500));
        for (a,b) in full.eigenvalues.iter().zip(packed.spectrum.eigenvalues) {assert!((a-b).abs()/a.abs().max(1.0)<1e-7);}
    }
}
