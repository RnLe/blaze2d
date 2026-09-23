//! Complete finite-spectrum checks; residuals alone would not detect lost modes.
use num_complex::Complex64 as C;
use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::{dielectric::{Dielectric2D,DielectricOptions},geometry::{Geometry2D,BasisAtom},grid::Grid2D,lattice::Lattice2D,field::Field2D,operators::{ThetaOperator,LinearOperator},polarization::Polarization,
    eigensolver::dense::solve_hermitian_eigen,
    symmetry::{affine::PlaneGroup,action::BlochActionPlan,qualification::qualify,representation::RepresentationPlan}};

fn dot(a:&[C],b:&[C])->C {a.iter().zip(b).map(|(x,y)|x.conj()*y).sum()}
fn apply_matrix(a:&[C],x:&[C],n:usize)->Vec<C> {(0..n).map(|i|(0..n).map(|j|a[i+n*j]*x[j]).sum()).collect()}

#[test]
fn square_irreps_reconstruct_the_complete_sampled_maxwell_spectrum() {
    let lattice=Lattice2D::square(1.0);let grid=Grid2D::new(5,5,1.0,1.0);let n=grid.len();let backend=CpuBackend::<f64>::new();
    let dielectric=Dielectric2D::from_geometry(&Geometry2D {lattice,eps_bg:1.0,atoms:vec![BasisAtom {pos:[0.0;2],radius:0.23,eps_inside:8.9}]},grid,&DielectricOptions::default());
    for pol in [Polarization::TM,Polarization::TE] {
        let representation=RepresentationPlan::projective(PlaneGroup::P4m.group(),[0.0;2]).unwrap();
        let mut operator=ThetaOperator::new(backend.clone(),dielectric.clone(),pol,[0.0;2]);
        let mut other=ThetaOperator::new(backend.clone(),dielectric.clone(),pol,[0.0;2]);
        let mut actions=Vec::new();for &op in &representation.group.operations {actions.push(qualify(BlochActionPlan::new(op,[0.0;2],[0.0;2],grid,pol,1e-12).unwrap(),&lattice,&mut operator,&mut other,1e-10).unwrap());}
        let sqrt_b:Vec<_>=dielectric.eps().iter().map(|x|if pol==Polarization::TM{x.sqrt()}else{1.0}).collect();
        let mut matrix=vec![C::ZERO;n*n];let mut output=Field2D::zeros(grid);
        for j in 0..n {let mut input=Field2D::zeros(grid);input.as_mut_slice()[j]=C::new(1.0/sqrt_b[j],0.0);operator.apply(&input,&mut output);for i in 0..n {matrix[i+n*j]=output.as_slice()[i]/sqrt_b[i];}}
        let full=solve_hermitian_eigen(&matrix,n);let mut union=Vec::new();let mut ranks=Vec::new();let mut kernel_owners=0;
        for irrep in 0..representation.irreps.len() {
            let weights=representation.isotypic_weights(irrep);let mut basis:Vec<Vec<C>>=Vec::new();let mut scratch=Field2D::zeros(grid);
            for j in 0..n {
                let mut input=Field2D::zeros(grid);input.as_mut_slice()[j]=C::ONE;
                representation.apply_weights(&actions,&weights,&input,&mut output,&mut scratch).unwrap();
                let mut v=output.as_slice().to_vec();for _ in 0..2 {for q in &basis {let overlap=dot(q,&v);for (x,y) in v.iter_mut().zip(q) {*x-=overlap*y;}}}
                let norm=dot(&v,&v).re.max(0.0).sqrt();if norm>1e-9 {for x in &mut v {*x/=norm;}basis.push(v);}
            }
            let rank=basis.len();ranks.push(rank);let mut reduced=vec![C::ZERO;rank*rank];
            for j in 0..rank {let y=apply_matrix(&matrix,&basis[j],n);for i in 0..rank {reduced[i+rank*j]=dot(&basis[i],&y);}}
            let eig=solve_hermitian_eigen(&reduced,rank);kernel_owners+=eig.eigenvalues.iter().filter(|x|x.abs()<1e-9).count();union.extend(eig.eigenvalues);
        }
        union.sort_by(f64::total_cmp);assert_eq!(union.len(),n);assert_eq!(ranks.iter().sum::<usize>(),n);assert_eq!(kernel_owners,1,"one physical zero mode per polarization");
        for (a,b) in union.iter().zip(full.eigenvalues) {assert!((a-b).abs()<1e-9,"{pol:?}: {a} vs {b}");}
    }
}
