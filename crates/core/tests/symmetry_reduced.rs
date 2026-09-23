use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{
    dielectric::Dielectric2D, eigensolver::EigensolverConfig, field::Field2D,
    geometry::{BasisAtom, Geometry2D}, grid::Grid2D, lattice::Lattice2D,
    operators::{LinearOperator, ThetaOperator}, polarization::Polarization,
    symmetry::{
        action::BlochActionPlan, affine::PlaneGroup, qualification::qualify,
        reduced::{OrbitBasis, SectorStorage}, representation::RepresentationPlan,
        reuse::SymmetryOptions, sectors::{SectorPolicy, solve_conservative},
    },
};
use num_complex::Complex64 as C;

#[test]
fn sparse_orbit_bases_reproduce_ordinary_and_projective_row_projectors() {
    let lattice = Lattice2D::square(1.0);
    for name in [PlaneGroup::Pm, PlaneGroup::P4, PlaneGroup::P4m, PlaneGroup::Pg, PlaneGroup::Pgg] {
        let (grid,k,atoms) = match name {
            PlaneGroup::Pg => (Grid2D::new(6,7,1.0,1.0),[0.5,0.0],vec![
                BasisAtom {pos:[0.0,0.2],radius:0.16,eps_inside:5.0},
                BasisAtom {pos:[0.5,-0.2],radius:0.16,eps_inside:5.0},
            ]),
            PlaneGroup::Pgg => (Grid2D::new(6,6,1.0,1.0),[-0.5,-0.5],vec![]),
            _ => (Grid2D::new(7,7,1.0,1.0),[0.0;2],vec![BasisAtom {pos:[0.0;2],radius:0.2,eps_inside:8.9}]),
        };
        for pol in [Polarization::TM, Polarization::TE] {
            let d = Dielectric2D::from_geometry(&Geometry2D { lattice,eps_bg:1.0,atoms:atoms.clone() },grid,&Default::default());
            let mut op = ThetaOperator::new(CpuBackend::<f64>::new(),d.clone(),pol,lattice.reciprocal().fractional_to_cartesian(k));
            let mut other = ThetaOperator::new(CpuBackend::<f64>::new(),d,pol,op.bloch());
            let representation = RepresentationPlan::ordinary(name.group(),k)
                .or_else(|_| RepresentationPlan::projective(name.group(),k)).unwrap();
            let actions: Vec<_> = representation.group.operations.iter().map(|&g| {
                qualify(BlochActionPlan::new(g,k,k,grid,pol,1e-12).unwrap(),&lattice,&mut op,&mut other,1e-10).unwrap()
            }).collect();
            let mut input = Field2D::zeros(grid);
            for (i,z) in input.as_mut_slice().iter_mut().enumerate() { *z=C::new((i as f64*0.723).sin(),(i as f64*1.31).cos()); }
            for (irrep,d) in representation.irreps.iter().enumerate() {
                for row in 0..d.dimension {
                    let weights=representation.matrix_unit_weights(irrep,row,row);
                    let rank=actions.iter().zip(&weights).map(|(a,w)|a.plan().spatial_trace()*w).sum::<C>().re.round() as usize;
                    let basis=OrbitBasis::from_projector(&actions,&weights,rank).unwrap();
                    assert_eq!(basis.rank(),rank);
                    let mut projected=Field2D::zeros(grid);let mut scratch=Field2D::zeros(grid);
                    representation.apply_weights(&actions,&weights,&input,&mut projected,&mut scratch).unwrap();
                    let mut reduced=Field2D::zeros(basis.coordinate_grid());
                    basis.restrict(&input,&mut reduced);basis.expand(&reduced,&mut scratch);
                    assert_close(projected.as_slice(),scratch.as_slice());
                    for j in 0..rank {
                        reduced.as_mut_slice().fill(C::ZERO);reduced.as_mut_slice()[j]=C::ONE;
                        basis.expand(&reduced,&mut scratch);
                        let mut restored=Field2D::zeros(basis.coordinate_grid());basis.restrict(&scratch,&mut restored);
                        assert_close(reduced.as_slice(),restored.as_slice());
                    }
                    let bad:Vec<_>=weights.iter().map(|z|z*1.2).collect();
                    if rank>0 {assert!(OrbitBasis::from_projector(&actions,&bad,rank).is_err());}
                }
            }
        }
    }
}

#[test]
fn reduced_iterative_sectors_preserve_warm_starts_kernel_and_partner_multiplicity() {
    let lattice=Lattice2D::square(1.0);let grid=Grid2D::new(13,13,1.0,1.0);
    let d=Dielectric2D::from_geometry(&Geometry2D {lattice,eps_bg:1.0,
        atoms:vec![BasisAtom {pos:[0.0;2],radius:0.2,eps_inside:8.9}]},grid,&Default::default());
    for pol in [Polarization::TM,Polarization::TE] {
        let mut warm=None;
        for (name,k,partners) in [(PlaneGroup::Pm,[0.0;2],false),(PlaneGroup::Pm,[0.17,0.0],false),(PlaneGroup::P4m,[0.0;2],true)] {
            let mut op=ThetaOperator::new(CpuBackend::<f64>::new(),d.clone(),pol,lattice.reciprocal().fractional_to_cartesian(k));
            let mut p=op.build_homogeneous_preconditioner_adaptive();
            let config=EigensolverConfig {n_bands:8,tol:1e-14,max_iter:500,..Default::default()};
            let mut options=SymmetryOptions {sectors:SectorPolicy::AdaptiveGroup,sector_guard_budget:Some(2),reconstruct_partners:partners,..Default::default()};
            let reference=solve_conservative(&mut op,&lattice,k,name.group(),&config,warm.as_deref(),&options,Some(&mut p)).unwrap();
            options.sector_storage=SectorStorage::Orbit;
            let reduced=solve_conservative(&mut op,&lattice,k,name.group(),&config,warm.as_deref(),&options,Some(&mut p))
                .unwrap_or_else(|e|panic!("{pol:?} {name:?} {k:?}: {e}"));
            assert_eq!(reduced.report.kernel_owners,usize::from(k==[0.0;2]));
            assert!(reduced.report.sectors.iter().any(|s|!s.dense_complete && s.stored_rows<grid.len()));
            for s in &reduced.report.sectors {assert_eq!(s.vector_column_bytes,s.stored_rows*16);assert!(s.basis_payload_bytes>0);}
            for (&x,&y) in reference.spectrum.eigenvalues.iter().zip(&reduced.spectrum.eigenvalues) {
                assert!((x-y).abs()/x.abs().max(1.0)<1e-8,"{pol:?} {name:?}: {x} {y}");
            }
            warm=Some(reduced.spectrum.fields);
        }
    }
}

fn assert_close(a:&[C],b:&[C]) {
    let error=a.iter().zip(b).map(|(x,y)|(x-y).norm_sqr()).sum::<f64>().sqrt();
    let norm=a.iter().map(C::norm_sqr).sum::<f64>().sqrt();
    assert!(error<1e-9*norm.max(1.0),"{error:e} / {norm:e}");
}
