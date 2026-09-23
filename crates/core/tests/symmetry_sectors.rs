use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{dielectric::Dielectric2D,geometry::{Geometry2D,BasisAtom},grid::Grid2D,lattice::Lattice2D,operators::ThetaOperator,polarization::Polarization,eigensolver::EigensolverConfig,
    symmetry::{affine::{PlaneGroup,AffineGroup},reuse::SymmetryOptions,sectors::solve_conservative}};

#[test]
fn serial_projection_workspace_reuse_preserves_fields_and_isolates_jobs() {
    let lattice=Lattice2D::square(1.0);
    let grid=Grid2D::new(9,9,1.0,1.0);
    for pol in [Polarization::TM,Polarization::TE] {
        let d=Dielectric2D::from_geometry(&Geometry2D{lattice,eps_bg:1.0,atoms:vec![BasisAtom{pos:[0.0;2],radius:0.24,eps_inside:8.9}]},grid,&Default::default());
        let mut op=ThetaOperator::new(CpuBackend::<f64>::new(),d,pol,[0.0;2]);
        let mut p=op.build_homogeneous_preconditioner_adaptive();
        let config=EigensolverConfig{n_bands:5,tol:1e-14,max_iter:500,..Default::default()};
        let mut options=SymmetryOptions{reconstruct_partners:true,..Default::default()};
        let reference=solve_conservative(&mut op,&lattice,[0.0;2],PlaneGroup::P4m.group(),&config,None,&options,Some(&mut p)).unwrap();
        options.share_sector_workspace=true;
        let pooled=solve_conservative(&mut op,&lattice,[0.0;2],PlaneGroup::P4m.group(),&config,None,&options,Some(&mut p)).unwrap();
        assert_eq!(reference.iterations,pooled.iterations);
        assert_eq!(reference.spectrum.eigenvalues,pooled.spectrum.eigenvalues);
        for (a,b) in reference.spectrum.fields.iter().zip(&pooled.spectrum.fields) {
            assert_eq!(a.as_slice(),b.as_slice());
        }
        assert_eq!(pooled.report.projection_workspace_allocations,2);
        assert!(reference.report.projection_workspace_allocations>2);
        assert_eq!(reference.report.projection_workspace_payload_bytes,pooled.report.projection_workspace_payload_bytes);
    }
}

#[test]
fn correction_projection_cadence_keeps_quality_gates_when_te_drifts() {
    let lattice=Lattice2D::square(1.0);
    let grid=Grid2D::new(13,13,1.0,1.0);
    let k=[0.17,0.0];
    for pol in [Polarization::TM,Polarization::TE] {
        let d=Dielectric2D::from_geometry(&Geometry2D{lattice,eps_bg:1.0,atoms:vec![BasisAtom{pos:[0.0;2],radius:0.24,eps_inside:12.0}]},grid,&Default::default());
        let mut op=ThetaOperator::new(CpuBackend::<f64>::new(),d,pol,lattice.reciprocal().fractional_to_cartesian(k));
        let mut p=op.build_homogeneous_preconditioner_adaptive();
        let config=EigensolverConfig{n_bands:8,tol:1e-14,max_iter:500,..Default::default()};
        let mut options=SymmetryOptions{share_sector_workspace:true,..Default::default()};
        let reference=solve_conservative(&mut op,&lattice,k,PlaneGroup::Pm.group(),&config,None,&options,Some(&mut p)).unwrap();
        for interval in [2,4] {
            options.correction_projection_interval=interval;
            let attempt=solve_conservative(&mut op,&lattice,k,PlaneGroup::Pm.group(),&config,None,&options,Some(&mut p));
            // TE can amplify roundoff contamination between projections into a
            // different sector. It must return a valid prefix or reject it.
            let result=match attempt {
                Ok(r)=>r,
                Err(e)=>{
                    assert_eq!(pol,Polarization::TE,"TM cadence failed: {e}");
                    assert!(e.contains("failed fresh quality gates") || e.contains("leaked"),"{e}");
                    continue;
                }
            };
            for (a,b) in reference.spectrum.eigenvalues.iter().zip(&result.spectrum.eigenvalues) {
                assert!((a-b).abs()/a.abs().max(1.0)<1e-7);
            }
            assert!(result.spectrum.passes(&options));
            for sector in result.report.sectors {
                let total=sector.correction_projection_rounds+sector.correction_projection_skipped_rounds;
                assert!(total>10);
                assert_eq!(sector.correction_projection_rounds,total.div_ceil(interval));
                assert!(sector.leakage<1e-8);
            }
        }
        options.correction_projection_interval=0;
        assert!(solve_conservative(&mut op,&lattice,k,PlaneGroup::Pm.group(),&config,None,&options,Some(&mut p)).is_err());
    }
}

#[test]
fn mirror_and_glide_sectors_cover_the_entire_actual_maxwell_pencil() {
    for glide in [false,true] {for pol in [Polarization::TM,Polarization::TE] {
        let (grid,k,group,atoms)=if glide {(Grid2D::new(6,5,1.0,1.0),[0.5,0.0],PlaneGroup::Pg.group(),vec![BasisAtom{pos:[0.0,0.2],radius:0.16,eps_inside:5.0},BasisAtom{pos:[0.5,-0.2],radius:0.16,eps_inside:5.0}])}
        else {(Grid2D::new(5,5,1.0,1.0),[0.0;2],PlaneGroup::Pm.group(),vec![BasisAtom{pos:[0.0;2],radius:0.23,eps_inside:8.9}])};
        let lattice=Lattice2D::square(1.0);let d=Dielectric2D::from_geometry(&Geometry2D{lattice,eps_bg:1.0,atoms},grid,&Default::default());
        let mut op=ThetaOperator::new(CpuBackend::<f64>::new(),d,pol,lattice.reciprocal().fractional_to_cartesian(k));
        let config=EigensolverConfig{n_bands:grid.len(),tol:1e-14,max_iter:500,..Default::default()};
        let full=solve_conservative(&mut op,&lattice,k,AffineGroup::from_generators(&[],1).unwrap(),&config,None,&Default::default(),None).unwrap();
        let sectors=solve_conservative(&mut op,&lattice,k,group,&config,None,&Default::default(),None).unwrap();
        assert_eq!(sectors.report.sum_sector_dimensions,grid.len());assert_eq!(sectors.spectrum.fields.len(),grid.len());assert_eq!(sectors.report.kernel_owners,usize::from(!glide));
        assert_eq!(sectors.spectrum.eigenvalues.iter().filter(|&&x|x==0.0).count(),usize::from(!glide));
        for (x,y) in full.spectrum.eigenvalues.iter().zip(sectors.spectrum.eigenvalues) {assert!((x-y).abs()/x.abs().max(1.0)<1e-10,"{pol:?} glide={glide}: {x} {y}");}
    }}
}

#[test]
fn iterative_mirror_prefixes_match_unrestricted_results_and_keep_one_kernel() {
    use blaze2d_core::drivers::single_solve::{SingleSolveJob,solve};
    for pol in [Polarization::TM,Polarization::TE] {for k in [[0.0,0.0],[0.17,0.0]] {
        let lattice=Lattice2D::square(1.0);let grid=Grid2D::new(13,13,1.0,1.0);
        let d=Dielectric2D::from_geometry(&Geometry2D{lattice,eps_bg:1.0,atoms:vec![BasisAtom{pos:[0.0;2],radius:0.2,eps_inside:8.9}]},grid,&Default::default());
        let mut op=ThetaOperator::new(CpuBackend::<f64>::new(),d,pol,lattice.reciprocal().fractional_to_cartesian(k));
        let mut p=op.build_homogeneous_preconditioner_adaptive();let full=solve(&mut op,Some(&mut p),&SingleSolveJob::new(8).with_tolerance(1e-14).with_max_iterations(500));
        let config=EigensolverConfig{n_bands:8,tol:1e-14,max_iter:500,..Default::default()};
        let result=solve_conservative(&mut op,&lattice,k,PlaneGroup::Pm.group(),&config,None,&SymmetryOptions::default(),Some(&mut p)).unwrap();
        assert_eq!(result.report.kernel_owners,usize::from(k==[0.0;2]));assert_eq!(result.report.computed,16);
        for (x,y) in full.eigenvalues.iter().zip(result.spectrum.eigenvalues) {assert!((x-y).abs()/x.abs().max(1.0)<1e-7,"{pol:?} {k:?}: {x} {y}");}
        assert!(result.report.sectors.iter().all(|s|s.leakage<1e-8));
    }}
}

#[test]
fn adaptive_counts_match_conservative_prefixes_across_cluster_edges() {
    use blaze2d_core::symmetry::sectors::SectorPolicy;
    let lattice=Lattice2D::square(1.0);let grid=Grid2D::new(9,9,1.0,1.0);
    for homogeneous in [false,true] {for pol in [Polarization::TM,Polarization::TE] {
        let atoms=if homogeneous {vec![]}else{vec![BasisAtom{pos:[0.0;2],radius:0.35,eps_inside:8.9}]};
        let d=Dielectric2D::from_geometry(&Geometry2D{lattice,eps_bg:1.0,atoms},grid,&Default::default());
        let mut op=ThetaOperator::new(CpuBackend::<f64>::new(),d,pol,[0.0;2]);let mut p=op.build_homogeneous_preconditioner_adaptive();
        for count in [3,8,12] {
            let config=EigensolverConfig{n_bands:count,tol:1e-14,max_iter:500,..Default::default()};
            let reference=solve_conservative(&mut op,&lattice,[0.0;2],PlaneGroup::Pm.group(),&config,None,&Default::default(),Some(&mut p)).unwrap();
            for sector_guard_budget in [None,Some(2)] {
            let options=SymmetryOptions{sectors:SectorPolicy::AdaptiveMirror,sector_guard_budget,..Default::default()};
            let adaptive=solve_conservative(&mut op,&lattice,[0.0;2],PlaneGroup::Pm.group(),&config,None,&options,Some(&mut p)).unwrap_or_else(|e|panic!("homogeneous={homogeneous}, pol={pol:?}, count={count}: {e}"));
            for (x,y) in reference.spectrum.eigenvalues.iter().zip(&adaptive.spectrum.eigenvalues) {assert!((x-y).abs()/x.abs().max(1.0)<1e-7,"{pol:?} q={count}: {x} {y}");}
            let cutoff=*adaptive.spectrum.eigenvalues.last().unwrap();
            for (tail,sector) in adaptive.report.tail_eigenvalues.iter().zip(&adaptive.report.sectors) {assert!(sector.prefix==sector.dimension || tail.unwrap()>cutoff+1e-7*cutoff.abs().max(1.0));}
            }
        }
    }}
}

#[test]
fn row_intertwiners_reconstruct_multiplicity_and_complete_group_spectra() {
    for group_name in [PlaneGroup::P4,PlaneGroup::P4m,PlaneGroup::Pgg] {for pol in [Polarization::TM,Polarization::TE] {
        let nonsymmorphic=group_name==PlaneGroup::Pgg;
        let n=if nonsymmorphic {6}else{5};let grid=Grid2D::new(n,n,1.0,1.0);let lattice=Lattice2D::square(1.0);
        let k=if nonsymmorphic {[-0.5,-0.5]}else{[0.0;2]};
        let atoms=if nonsymmorphic {vec![]}else{vec![BasisAtom{pos:[0.0;2],radius:0.23,eps_inside:8.9}]};
        let d=Dielectric2D::from_geometry(&Geometry2D{lattice,eps_bg:1.0,atoms},grid,&Default::default());
        let mut op=ThetaOperator::new(CpuBackend::<f64>::new(),d,pol,lattice.reciprocal().fractional_to_cartesian(k));
        let config=EigensolverConfig{n_bands:grid.len(),tol:1e-14,max_iter:500,..Default::default()};
        let independent=solve_conservative(&mut op,&lattice,k,group_name.group(),&config,None,&Default::default(),None).unwrap();
        let options=SymmetryOptions{reconstruct_partners:true,..Default::default()};
        let partners=solve_conservative(&mut op,&lattice,k,group_name.group(),&config,None,&options,None).unwrap();
        assert_eq!(partners.report.sum_sector_dimensions,grid.len());assert_eq!(partners.spectrum.fields.len(),grid.len());assert!(partners.spectrum.orthogonality_defect<1e-8);
        for (x,y) in independent.spectrum.eigenvalues.iter().zip(&partners.spectrum.eigenvalues) {assert!((x-y).abs()/x.abs().max(1.0)<1e-9,"{group_name:?} {pol:?}: {x} {y}");}
        if group_name==PlaneGroup::P4m {assert!(partners.report.sectors.iter().any(|s|s.partner_rows==2));assert!(partners.report.sectors.iter().map(|s|s.prefix).sum::<usize>()<grid.len());}
    }}
}
