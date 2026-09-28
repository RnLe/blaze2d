use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    backend::SpectralBackend, dielectric::Dielectric2D,
    eigensolver::{Eigensolver, EigensolverConfig, InitialAcceptance},
    field::Field2D, geometry::{BasisAtom, Geometry2D}, grid::Grid2D,
    lattice::Lattice2D, operators::{LinearOperator, ThetaOperator}, polarization::Polarization,
};
use num_complex::Complex64 as C;

fn operator<B: SpectralBackend>(backend: B, pol: Polarization, k: [f64;2], heterogeneous: bool) -> ThetaOperator<B> {
    let geom = Geometry2D {lattice: Lattice2D::square(1.), eps_bg: 2.,
        atoms: if heterogeneous {vec![BasisAtom {pos:[0.;2],radius:0.201,eps_inside:8.9}]} else {vec![]}};
    ThetaOperator::new(backend, Dielectric2D::from_geometry(&geom,Grid2D::new(13,13,1.,1.),&Default::default()),pol,k)
}

fn projected_rank<B: SpectralBackend + Clone>(backend: B, tolerance: f64) {
    for pol in [Polarization::TM,Polarization::TE] {
        let mut op=operator(backend.clone(),pol,[0.;2],true);
        let grid=op.grid();
        let mass=if pol==Polarization::TM {op.dielectric().eps().to_vec()} else {vec![1.;grid.len()]};
        let mut constant=Field2D::zeros(grid);constant.as_mut_slice().fill(C::ONE);
        let mut wave=constant.clone();
        for j in 0..grid.ny {for i in 0..grid.nx {wave.as_mut_slice()[grid.idx(i,j)]+=C::from_polar(1.,std::f64::consts::TAU*i as f64/grid.nx as f64);}}
        let invalid=Field2D::zeros(Grid2D::new(13,13,2.,1.));
        let mut nonfinite=constant.clone();nonfinite.as_mut_slice()[0]=C::new(f64::NAN,0.);
        let mut scaled=constant.clone();for z in scaled.as_mut_slice(){*z*=1e-16;}
        let seeds=[constant,scaled,invalid,nonfinite,wave.clone(),wave];
        let mut solver=Eigensolver::new(&mut op,EigensolverConfig{n_bands:5,..Default::default()},None,Some(&seeds));
        solver.initialize();
        let report=solver.initialization_report().unwrap();
        assert_eq!(report.warm_start_hits,1,"{pol:?}: {report:?}");
        assert_eq!(report.rejected_warm_vectors,5);
        let fields=solver.eigenvectors();
        for (i,x) in fields.iter().enumerate() {
            let leak:C=x.as_slice().iter().zip(&mass).map(|(x,m)|x*m).sum();
            assert!(leak.norm()/mass.iter().sum::<f64>().sqrt()<tolerance,"kernel leakage {leak}");
            for (j,y) in fields.iter().enumerate() {
                let gram:C=x.as_slice().iter().zip(y.as_slice()).zip(&mass).map(|((x,y),m)|x.conj()*y*m).sum();
                assert!((gram-C::new(if i==j {1.} else {0.},0.)).norm()<tolerance,"Gram {i} {j}: {gram}");
            }
        }
    }
}
#[test]
fn target_metric_rank_repair_respects_storage_precision() {
    projected_rank(CpuBackend::<f64>::new(),1e-11);
    projected_rank(CpuBackend::<f32>::new(),2e-6);
}

#[test]
fn exact_gamma_is_analytic_but_near_gamma_is_not() {
    for pol in [Polarization::TM,Polarization::TE] {
        for entry in 0..3 {
            let mut op=operator(CpuBackend::<f64>::new(),pol,[0.;2],true);
            let mut solver=Eigensolver::new(&mut op,EigensolverConfig{n_bands:1,..Default::default()},None,None);
            let result=match entry {0=>solver.solve(),1=>solver.solve_with_progress(|_|{}),_=>solver.solve_with_diagnostics("analytic").result};
            assert_eq!(result.iterations,0);assert!(result.converged);
            assert_eq!(result.eigenvalues,vec![0.]);assert_eq!(solver.block_size(),0);
            assert_eq!(solver.all_eigenvectors().len(),1);
        }
        let mut op=operator(CpuBackend::<f64>::new(),pol,[1e-13,0.],true);
        assert!(!op.analytic_kernel_available());
        let mut solver=Eigensolver::new(&mut op,EigensolverConfig{n_bands:1,..Default::default()},None,None);
        solver.initialize();assert_eq!(solver.locked_count(),0);
    }
}

#[test]
fn qualified_mixed_lowest_span_uses_shared_zero_correction_gate() {
    // The homogeneous plane-wave spectrum independently establishes coverage.
    // An internal degeneracy is mixed deliberately; target Ritz must recover it.
    for entry in 0..3 {
        let mut op=operator(CpuBackend::<f64>::new(),Polarization::TM,[0.;2],false);
        let grid=op.grid();let mut waves=Vec::new();
        for (gx,gy) in [(0.,0.),(1.,0.),(-1.,0.),(0.,1.),(0.,-1.)] {
            let mut field=Field2D::zeros(grid);
            for j in 0..grid.ny {for i in 0..grid.nx {field.as_mut_slice()[grid.idx(i,j)]=C::from_polar(1.,std::f64::consts::TAU*(gx*i as f64+gy*j as f64)/13.);}}
            waves.push(field);
        }
        let a=waves[1].clone();let b=waves[2].clone();
        for i in 0..grid.len(){waves[1].as_mut_slice()[i]=a.as_slice()[i]+b.as_slice()[i];waves[2].as_mut_slice()[i]=a.as_slice()[i]-b.as_slice()[i];}
        let mut solver=Eigensolver::new(&mut op,EigensolverConfig{n_bands:5,tol:1e-14,..Default::default()},None,Some(&waves));
        solver.set_initial_acceptance(InitialAcceptance{certified_prefix:5,relative_residual:1e-10,absolute_residual:1e-9,b_orthogonality:1e-10});
        let result=match entry {0=>solver.solve(),1=>solver.solve_with_progress(|_|{}),_=>solver.solve_with_diagnostics("mixed").result};
        assert!(result.converged);assert_eq!(result.iterations,0);
        for &lambda in &result.eigenvalues[1..]{assert!((lambda-std::f64::consts::TAU.powi(2)/2.).abs()<1e-10);}
    }
}
