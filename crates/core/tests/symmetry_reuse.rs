use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{drivers::bandstructure::{BandStructureJob,RunOptions,run_with_k_streaming,KPointResult},geometry::{Geometry2D,BasisAtom},grid::Grid2D,lattice::Lattice2D,polarization::Polarization,eigensolver::EigensolverConfig,symmetry::reuse::{KReuse,SymmetryOptions}};

fn run(policy:KReuse,bytes:usize,tracking:bool)->Vec<KPointResult> {
    let job=BandStructureJob {geom:Geometry2D{lattice:Lattice2D::square(1.0),eps_bg:1.0,atoms:vec![BasisAtom{pos:[0.0;2],radius:0.2,eps_inside:8.9}]},grid:Grid2D::new(12,12,1.0,1.0),pol:Polarization::TM,
        k_path:vec![[0.13,0.21],[0.27,0.09],[0.13,0.21],[0.27,0.09]],eigensolver:EigensolverConfig{n_bands:4,tol:1e-12,max_iter:500,..Default::default()},dielectric:Default::default()};
    let mut points=Vec::new();run_with_k_streaming(CpuBackend::<f64>::new(),&job,RunOptions{reuse_gamma:false,retain_eigenvectors:true,disable_band_tracking:!tracking,symmetry:SymmetryOptions{k_reuse:policy,max_cache_bytes:bytes,..Default::default()},..Default::default()},|p|points.push(p));points
}

#[test]
fn exact_reuse_retains_fields_and_tracking_and_counts_only_executed_iterations() {
    for tracking in [false,true] {
        let baseline=run(KReuse::Off,0,tracking);let candidate=run(KReuse::Identical,1<<20,tracking);
        assert!(candidate.iter().all(|p|p.converged));assert_eq!(candidate[2].symmetry.source_index,Some(0));assert_eq!(candidate[3].symmetry.source_index,Some(1));assert_eq!(candidate[2].iterations,0);assert_eq!(candidate[3].iterations,0);
        for (a,b) in baseline.iter().zip(&candidate) {let mut x=a.omegas.clone();let mut y=b.omegas.clone();x.sort_by(f64::total_cmp);y.sort_by(f64::total_cmp);for (x,y) in x.iter().zip(y) {assert!((x-y).abs()<1e-6);}assert_eq!(b.eigenvectors.as_ref().unwrap().len(),4);assert!(b.residuals.iter().all(|&r|r<1e-5));}
    }
}
#[test]
fn zero_cache_budget_preserves_unrestricted_solves() {let points=run(KReuse::Identical,0,false);assert!(points.iter().all(|p|p.iterations>0 && p.symmetry.source_index.is_none() && p.symmetry.cache_bytes==0));}

#[test]
fn fallback_reports_count_discarded_iterations_without_profiling() {
    use blaze2d_core::symmetry::{fourier::Arithmetic,sectors::SectorPolicy};
    for sectors in [false,true] {
        let job=BandStructureJob {geom:Geometry2D {lattice:Lattice2D::square(1.0),eps_bg:1.0,
            atoms:vec![BasisAtom {pos:[0.0;2],radius:0.2,eps_inside:8.9}]},
            grid:Grid2D::new(13,13,1.0,1.0),pol:Polarization::TM,k_path:vec![[0.17,0.0]],
            eigensolver:EigensolverConfig {n_bands:4,tol:1e-14,max_iter:500,..Default::default()},dielectric:Default::default()};
        let options=RunOptions {reuse_gamma:false,symmetry:SymmetryOptions {
            arithmetic:if sectors {Arithmetic::SpatialComplex} else {Arithmetic::PtRealTe},
            sectors:if sectors {SectorPolicy::ConservativeMirror} else {SectorPolicy::Off},
            // Deliberately fail the optimization's final quality gate after
            // performing its solve. The unrestricted result is still returned.
            max_relative_residual:0.0,..Default::default()},..Default::default()};
        let mut points=Vec::new();
        run_with_k_streaming(CpuBackend::<f64>::new(),&job,options,|p|points.push(p));
        let point=&points[0];let report=&point.symmetry;
        assert!(point.converged);
        assert!(report.solver_attempts.iter().any(|a|!a.retained && a.iterations>0));
        assert_eq!(report.solver_attempts.last().unwrap().method,"unrestricted");
        assert_eq!(report.solver_attempts.iter().map(|a|a.iterations).sum::<usize>(),point.iterations);
        assert_eq!(report.attempted_iterations,point.iterations);
        assert!(point.iterations>report.retained_iterations);
    }
}

fn star(policy:KReuse,n:usize,center:[f64;2],pol:Polarization)->Vec<KPointResult> {
    let job=BandStructureJob {geom:Geometry2D{lattice:Lattice2D::square(1.0),eps_bg:1.0,atoms:vec![BasisAtom{pos:center,radius:0.2,eps_inside:8.9}]},grid:Grid2D::new(n,n,1.0,1.0),pol,
        k_path:vec![[0.13,0.21],[-0.21,0.13],[-0.13,-0.21],[0.21,-0.13],[0.13,-0.21],[0.21,0.13],[-0.13,0.21],[-0.21,-0.13]],eigensolver:EigensolverConfig{n_bands:4,tol:1e-12,max_iter:500,..Default::default()},dielectric:Default::default()};
    let mut points=Vec::new();run_with_k_streaming(CpuBackend::<f64>::new(),&job,RunOptions{reuse_gamma:false,retain_eigenvectors:true,disable_band_tracking:true,symmetry:SymmetryOptions{k_reuse:policy,..Default::default()},..Default::default()},|p|points.push(p));points
}
#[test]
fn qualified_odd_grid_stars_reconstruct_complete_requested_spectra() {
    for pol in [Polarization::TM,Polarization::TE] {
        let full=star(KReuse::Off,13,[0.0;2],pol);let reduced=star(KReuse::Unitary,13,[0.0;2],pol);
        assert_eq!(reduced.iter().filter(|p|p.iterations>0).count(),1);
        for (a,b) in full.iter().zip(&reduced) {assert!(b.converged);assert!(b.residuals.iter().all(|&r|r<1e-5));assert!(b.b_orthogonality_defect<1e-8);for (&x,&y) in a.omegas.iter().zip(&b.omegas) {assert!((x-y).abs()<1e-6,"{pol:?}: {x} vs {y}");}}
    }
}
#[test]
fn even_grid_aliases_and_broken_material_fall_back() {
    let even=star(KReuse::Unitary,12,[0.0;2],Polarization::TE);
    assert_eq!(even.iter().filter(|p|p.iterations>0).count(),4);
    assert!(even.iter().any(|p|p.symmetry.rejected.iter().any(|r|r.contains("Fourier support"))));
    let broken=star(KReuse::Unitary,13,[0.13,0.07],Polarization::TM);
    assert!(broken.iter().all(|p|p.iterations>0 && p.symmetry.source_index.is_none()));
    assert!(broken.iter().any(|p|p.symmetry.rejected.iter().any(|r|r.contains("MaterialMismatch"))));
}

#[test]
fn time_reversal_removes_only_additional_supported_orbits() {
    for pol in [Polarization::TM,Polarization::TE] {
        let full=star(KReuse::Off,13,[0.13,0.07],pol);let reciprocal=star(KReuse::TimeReversal,13,[0.13,0.07],pol);
        assert_eq!(reciprocal.iter().filter(|p|p.iterations>0).count(),4);
        for (a,b) in full.iter().zip(&reciprocal) {for (&x,&y) in a.omegas.iter().zip(&b.omegas) {assert!((x-y).abs()<1e-6);}}
        let symmetric=star(KReuse::TimeReversal,13,[0.0;2],pol);assert_eq!(symmetric.iter().filter(|p|p.iterations>0).count(),1);
        let even=star(KReuse::TimeReversal,12,[0.13,0.07],pol);assert_eq!(even.iter().filter(|p|p.iterations>0).count(),8);
    }
}
