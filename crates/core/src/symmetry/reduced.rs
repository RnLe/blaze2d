//! Sparse orbit coordinates for an already qualified finite projector.
//!
//! Persistent eigensolver columns have the projector's rank. A/B and the
//! preconditioner still expand to the original grid and use full-size FFTs.
use num_complex::Complex64 as C;
use serde::Serialize;
use std::collections::BTreeMap;
use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    eigensolver::{Eigensolver, EigensolverConfig},
    field::{Field2D, Real, cscalar}, grid::Grid2D,
    operators::{LinearOperator, ThetaOperator}, preconditioners::OperatorPreconditioner,
};
use super::qualification::QualifiedAction;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all="snake_case")]
pub enum SectorStorage { #[default] FullGrid, Orbit }

/// Euclidean-orthonormal columns S, supported on disjoint group orbits.
/// The reduced pencil is S† A S, S† B S; B is not assumed to be identity.
#[derive(Debug, Clone)]
pub struct OrbitBasis { grid: Grid2D, columns: Vec<Vec<(usize,C)>> }
impl OrbitBasis {
    pub fn from_projector(actions:&[QualifiedAction], weights:&[C], expected_rank:usize) -> Result<Self,String> {
        crate::profile_scope!("symmetry.orbit_basis");
        let first=actions.first().ok_or("empty action set")?;
        let grid=first.plan().grid();let n=grid.len();
        if actions.len()!=weights.len() || weights.iter().any(|z|!z.re.is_finite()||!z.im.is_finite()) || actions.iter().any(|a|a.plan().operation().conjugates() || !same_grid(a.plan().grid(),grid)) {return Err("incompatible orbit projector".into());}
        let mut parent:Vec<_>=(0..n).collect();
        fn root(parent:&mut[usize],mut i:usize)->usize {
            while parent[i]!=i {parent[i]=parent[parent[i]];i=parent[i];}i
        }
        for a in actions {for (i,&j) in a.plan().spatial_permutation().iter().enumerate() {
            let x=root(&mut parent,i);let y=root(&mut parent,j);parent[y]=x;
        }}
        let mut orbits:BTreeMap<usize,Vec<usize>>=BTreeMap::new();
        for i in 0..n {orbits.entry(root(&mut parent,i)).or_default().push(i);}
        let mut columns=Vec::new();
        for orbit in orbits.into_values() {
            let d=orbit.len();
            if d>actions.len() {return Err("action list does not close a finite orbit".into());}
            let mut projector=vec![C::ZERO;d*d];
            for (a,&weight) in actions.iter().zip(weights) {for (i,&row) in orbit.iter().enumerate() {
                let j=orbit.binary_search(&a.plan().spatial_permutation()[row]).map_err(|_|"incomplete orbit")?;
                projector[i+d*j]+=weight*a.plan().spatial_phases()[row];
            }}
            for i in 0..d {for j in 0..d {
                let square:C=(0..d).map(|r|projector[i+d*r]*projector[r+d*j]).sum();
                if (projector[i+d*j]-projector[j+d*i].conj()).norm()>1e-9 || (square-projector[i+d*j]).norm()>1e-9 {return Err("orbit weights do not define an orthogonal projector".into());}
            }}
            let mut local:Vec<Vec<C>>=Vec::new();
            for j in 0..d {
                let mut v=projector[j*d..(j+1)*d].to_vec();
                for _ in 0..2 {for q in &local {let dot:C=q.iter().zip(&v).map(|(a,b)|a.conj()*b).sum();for (x,&y) in v.iter_mut().zip(q) {*x-=dot*y;}}}
                let norm=v.iter().map(C::norm_sqr).sum::<f64>().sqrt();if norm<=1e-10 {continue;}
                for z in &mut v {*z/=norm;}
                columns.push(orbit.iter().copied().zip(v.iter().copied()).filter(|(_,z)|z.norm_sqr()>0.0).collect());
                local.push(v);
            }
        }
        if columns.len()!=expected_rank {return Err(format!("orbit rank {} differs from character rank {expected_rank}",columns.len()));}
        Ok(Self{grid,columns})
    }
    pub fn rank(&self)->usize {self.columns.len()}
    pub fn grid(&self)->Grid2D {self.grid}
    pub fn coordinate_grid(&self)->Grid2D {Grid2D::new(self.rank(),1,1.0,1.0)}
    /// Index and coefficient payload only; excludes Vec/allocator overhead.
    pub fn payload_bytes(&self)->usize {self.columns.iter().map(|c|c.len()*(std::mem::size_of::<usize>()+std::mem::size_of::<C>())).sum()}
    pub fn expand<F:SpectralBuffer,G:SpectralBuffer>(&self,input:&F,output:&mut G) {
        assert_eq!(input.len(),self.rank());assert_eq!(output.len(),self.grid.len());
        crate::profiler::count("symmetry.orbit.expand_terms",self.columns.iter().map(Vec::len).sum());
        output.as_mut_slice().fill(cscalar::<G::Real>(0.0,0.0));
        for (&a,column) in input.as_slice().iter().zip(&self.columns) {
            let a=C::new(a.re.to_accum(),a.im.to_accum());
            for &(i,b) in column {let z=a*b;output.as_mut_slice()[i]+=cscalar::<G::Real>(z.re,z.im);}
        }
    }
    pub fn restrict<F:SpectralBuffer,G:SpectralBuffer>(&self,input:&F,output:&mut G) {
        assert_eq!(input.len(),self.grid.len());assert_eq!(output.len(),self.rank());
        crate::profiler::count("symmetry.orbit.restrict_terms",self.columns.iter().map(Vec::len).sum());
        for (dst,column) in output.as_mut_slice().iter_mut().zip(&self.columns) {
            let z:C=column.iter().map(|&(i,b)| {let a=input.as_slice()[i];b.conj()*C::new(a.re.to_accum(),a.im.to_accum())}).sum();
            *dst=cscalar::<G::Real>(z.re,z.im);
        }
    }
}

struct ReducedOperator<'a,B:SpectralBackend> {
    full:&'a mut ThetaOperator<B>,basis:&'a OrbitBasis,input:B::Buffer,output:B::Buffer,owns_kernel:bool,
}
impl<B:SpectralBackend> LinearOperator<B> for ReducedOperator<'_,B> {
    fn apply(&mut self,x:&B::Buffer,y:&mut B::Buffer) {self.basis.expand(x,&mut self.input);self.full.apply(&self.input,&mut self.output);self.basis.restrict(&self.output,y);}
    fn apply_mass(&mut self,x:&B::Buffer,y:&mut B::Buffer) {self.basis.expand(x,&mut self.input);self.full.apply_mass(&self.input,&mut self.output);self.basis.restrict(&self.output,y);}
    fn alloc_field(&self)->B::Buffer {self.full.backend().alloc_field(self.grid())}
    fn backend(&self)->&B {self.full.backend()}
    fn backend_mut(&mut self)->&mut B {self.full.backend_mut()}
    fn grid(&self)->Grid2D {self.basis.coordinate_grid()}
    fn bloch(&self)->[f64;2] {self.full.bloch()}
    fn subspace_dimension(&self)->usize {self.basis.rank()}
    fn owns_gamma_kernel(&self)->bool {self.owns_kernel}
    fn fill_gamma_kernel(&self,v:&mut B::Buffer) {
        let mut constant=self.full.alloc_field();constant.as_mut_slice().fill(cscalar::<B::Real>(1.0,0.0));self.basis.restrict(&constant,v);
    }
    fn seed_initial_vector(&mut self,v:&mut B::Buffer,phase:f64) {
        crate::eigensolver::initialization::seed_random_vector(self.input.as_mut_slice(),phase);self.basis.restrict(&self.input,v);
    }
}
struct ReducedPreconditioner<'a,B:SpectralBackend> {inner:&'a mut dyn OperatorPreconditioner<B>,basis:&'a OrbitBasis,scratch:B::Buffer}
impl<B:SpectralBackend> OperatorPreconditioner<B> for ReducedPreconditioner<'_,B> {
    fn apply(&mut self,backend:&B,v:&mut B::Buffer) {self.basis.expand(v,&mut self.scratch);self.inner.apply(backend,&mut self.scratch);self.basis.restrict(&self.scratch,v);}
}

pub(crate) fn solve_reduced<B:SpectralBackend>(operator:&mut ThetaOperator<B>,basis:&OrbitBasis,owns_kernel:bool,config:EigensolverConfig,warm:Option<&[Field2D]>,preconditioner:Option<&mut dyn OperatorPreconditioner<B>>)->Result<(Vec<f64>,Vec<Field2D>,usize),String> {
    crate::profile_scope!("symmetry.reduced_solve");
    if basis.rank()==0 || !same_grid(basis.grid(),operator.grid()) {return Err("incompatible reduced operator".into());}
    let warm=warm.map(|fields|fields.iter().map(|field| {let mut small=Field2D::zeros(basis.coordinate_grid());basis.restrict(field,&mut small);small}).collect::<Vec<_>>());
    let input=operator.alloc_field();let output=operator.alloc_field();let mut scratch=operator.alloc_field();
    let mut preconditioner=preconditioner.map(|inner|ReducedPreconditioner{inner,basis,scratch:operator.alloc_field()});
    let mut reduced=ReducedOperator{full:operator,basis,input,output,owns_kernel};
    let mut solver=Eigensolver::new(&mut reduced,config,preconditioner.as_mut().map(|p|p as &mut dyn OperatorPreconditioner<B>),warm.as_deref());
    let result=solver.solve();let coordinates=solver.all_eigenvectors();drop(solver);
    if !result.converged {return Err(format!("reduced sector failed after {} iterations",result.iterations));}
    let mut fields=Vec::new();
    for small in coordinates {basis.expand(&small,&mut scratch);fields.push(Field2D::from_vec(basis.grid(),scratch.as_slice().iter().map(|z|C::new(z.re.to_accum(),z.im.to_accum())).collect()));}
    Ok((result.eigenvalues,fields,result.iterations))
}

fn same_grid(a:Grid2D,b:Grid2D)->bool { (a.nx,a.ny,a.lx.to_bits(),a.ly.to_bits(),a.centered)==(b.nx,b.ny,b.lx.to_bits(),b.ly.to_bits(),b.centered) }
