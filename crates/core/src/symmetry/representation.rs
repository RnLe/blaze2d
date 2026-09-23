//! Ordinary and projective little-group representations and matrix units.
use num_complex::Complex64 as C;
use crate::{backend::SpectralBuffer, field::cscalar};
use super::{affine::AffineGroup,qualification::QualifiedAction};

#[derive(Debug,Clone,thiserror::Error)]
pub enum RepresentationError {
    #[error("the supplied group is not a unitary little group at this k")]
    NotUnitaryLittleGroup,
    #[error("the little group has a nontrivial multiplier")]
    ProjectiveMultiplier,
    #[error("representation validation failed")]
    InvalidRepresentation,
    #[error("qualified actions do not match this representation")]
    IncompatibleActions,
}

#[derive(Debug,Clone)]
pub struct Irrep { pub label:String, pub dimension:usize, pub matrices:Vec<Vec<C>> }
impl Irrep {
    pub fn character(&self,g:usize)->C {(0..self.dimension).map(|i|self.matrices[g][i+self.dimension*i]).sum()}
}
#[derive(Debug,Clone)]
pub struct RepresentationPlan { pub group:AffineGroup, pub k:[f64;2], pub irreps:Vec<Irrep>, pub multiplier:Vec<Vec<C>>, pub maximum_law_defect:f64 }

fn multiplier(group:&AffineGroup,k:[f64;2])->Result<Vec<Vec<C>>,RepresentationError> {
    if group.operations.iter().any(|g|g.conjugates()) || group.little_group(k,1e-12).map_err(|_|RepresentationError::NotUnitaryLittleGroup)?.order()!=group.order() {return Err(RepresentationError::NotUnitaryLittleGroup);}
    Ok(group.translation_carry.iter().map(|row|row.iter().map(|c|C::from_polar(1.0,-std::f64::consts::TAU*(k[0]*c[0] as f64+k[1]*c[1] as f64))).collect()).collect())
}
fn multiply(a:&[C],b:&[C],n:usize)->Vec<C> {(0..n*n).map(|p| {let (i,j)=(p%n,p/n);(0..n).map(|k|a[i+n*k]*b[k+n*j]).sum()}).collect()}

impl RepresentationPlan {
    /// Decompose the small twisted regular representation. Each candidate is
    /// checked for unitarity, irreducibility, mutual orthogonality, dimension
    /// sum and the actual multiplier law. Accidental clustering is retried
    /// with an independent deterministic commutant; failure is explicit.
    pub fn projective(group:AffineGroup,k:[f64;2])->Result<Self,RepresentationError> {
        let omega=multiplier(&group,k)?;let h=group.order();
        crate::init_faer_sequential();
        for attempt in 0..4 {
            let mut seed=20260923_u64+attempt*7919;
            let mut random=|| {seed=seed.wrapping_add(0x9e3779b97f4a7c15);let mut z=seed;z=(z^(z>>30)).wrapping_mul(0xbf58476d1ce4e5b9);z=(z^(z>>27)).wrapping_mul(0x94d049bb133111eb);((z^(z>>31))>>11) as f64/(1u64<<53) as f64-0.5};
            let mut h0=vec![C::ZERO;h*h];for i in 0..h {for j in i..h {let v=C::new(random(),if i==j {0.0}else{random()});h0[i+h*j]=v;h0[j+h*i]=v.conj();}}
            let mut commutant=vec![C::ZERO;h*h];
            for g in 0..h {for i in 0..h {for j in 0..h {
                let row=group.product_index[g][i];let col=group.product_index[g][j];
                commutant[row+h*col]+=omega[g][i]*h0[i+h*j]*omega[g][j].conj()/h as f64;
            }}}
            let eigen=crate::eigensolver::dense::solve_hermitian_eigen(&commutant,h);
            let mut unique:Vec<Irrep>=Vec::new();let mut start=0;let mut invalid=false;
            while start<h {
                let mut end=start+1;
                while end<h && (eigen.eigenvalues[end]-eigen.eigenvalues[start]).abs()<=1e-8*eigen.eigenvalues[start].abs().max(1.0) {end+=1;}
                let d=end-start;let mut matrices=Vec::new();
                for g in 0..h {let mut matrix=vec![C::ZERO;d*d];for i in 0..d {for j in 0..d {
                    matrix[i+d*j]=(0..h).map(|q|eigen.eigenvectors[group.product_index[g][q]+h*(start+i)].conj()*omega[g][q]*eigen.eigenvectors[q+h*(start+j)]).sum();
                }}matrices.push(matrix);}
                let candidate=Irrep{label:format!("projective:{}",unique.len()),dimension:d,matrices};
                let norm=(0..h).map(|g|candidate.character(g).norm_sqr()).sum::<f64>()/h as f64;
                if (norm-1.0).abs()>1e-8 {invalid=true;break;}
                let mut equivalent=false;
                for irrep in &unique {let overlap=(0..h).map(|g|irrep.character(g).conj()*candidate.character(g)).sum::<C>()/h as f64;
                    if (overlap-C::ONE).norm()<1e-8 {equivalent=true;} else if overlap.norm()>1e-8 {invalid=true;}
                }
                if !equivalent {unique.push(candidate);}
                start=end;
            }
            if !invalid {if let Ok(plan)=Self::validated(group.clone(),k,unique,omega.clone()) {return Ok(plan);}}
        }
        Err(RepresentationError::InvalidRepresentation)
    }
    /// Analytic cyclic/dihedral irreps. Only valid in a gauge with unit multiplier.
    pub fn ordinary(group:AffineGroup,k:[f64;2])->Result<Self,RepresentationError> {
        let omega=multiplier(&group,k)?;if omega.iter().flatten().any(|&v|(v-C::ONE).norm()>1e-12) {return Err(RepresentationError::ProjectiveMultiplier);}
        let h=group.order();let rotations:Vec<_>=group.operations.iter().enumerate().filter_map(|(i,g)|(g.determinant()==1).then_some(i)).collect();let n=rotations.len();
        let order=|r:usize| {let mut p=0;for j in 1..=h {p=group.product_index[p][r];if p==0{return j;}}h+1};
        let r=*rotations.iter().find(|&&r|order(r)==n).ok_or(RepresentationError::InvalidRepresentation)?;
        let mirror=group.operations.iter().position(|g|g.determinant()==-1);
        let mut powers=vec![0];for j in 1..n {powers.push(group.product_index[powers[j-1]][r]);}
        let mut words=vec![(0,false);h];for j in 0..n {words[powers[j]]=(j,false);if let Some(m)=mirror {words[group.product_index[powers[j]][m]]=(j,true);}}
        let mut irreps=Vec::new();
        if mirror.is_none() {
            for ell in 0..n {irreps.push(Irrep {label:format!("C{n}:{ell}"),dimension:1,matrices:words.iter().map(|&(j,_)|vec![C::from_polar(1.0,std::f64::consts::TAU*(ell*j) as f64/n as f64)]).collect()});}
        } else {
            for rsign in if n%2==0 {vec![1.0,-1.0]} else {vec![1.0]} {for msign in [1.0,-1.0] {
                irreps.push(Irrep {label:format!("D{n}:R{rsign:+}M{msign:+}"),dimension:1,matrices:words.iter().map(|&(j,m)|vec![C::new(if j%2==0 {1.0}else{rsign}*if m {msign}else{1.0},0.0)]).collect()});
            }}
            for ell in 1..=(n.saturating_sub(1)/2) {
                irreps.push(Irrep {label:format!("D{n}:E{ell}"),dimension:2,matrices:words.iter().map(|&(j,m)|{let p=C::from_polar(1.0,std::f64::consts::TAU*(ell*j) as f64/n as f64);if m {vec![C::ZERO,p.conj(),p,C::ZERO]} else {vec![p,C::ZERO,C::ZERO,p.conj()]}}).collect()});
            }
        }
        Self::validated(group,k,irreps,omega)
    }
    fn validated(group:AffineGroup,k:[f64;2],irreps:Vec<Irrep>,multiplier:Vec<Vec<C>>)->Result<Self,RepresentationError> {
        let h=group.order();if irreps.iter().map(|d|d.dimension*d.dimension).sum::<usize>()!=h {return Err(RepresentationError::InvalidRepresentation);}
        let mut defect:f64=0.0;
        for (a,d) in irreps.iter().enumerate() {
            let n=d.dimension;
            for g in 0..h {for v in 0..h {let prod=multiply(&d.matrices[g],&d.matrices[v],n);let expected=&d.matrices[group.product_index[g][v]];for (x,y) in prod.iter().zip(expected) {defect=defect.max((*x-multiplier[g][v]*y).norm());}}}
            for (b,e) in irreps.iter().enumerate() {let dot:C=(0..h).map(|g|d.character(g).conj()*e.character(g)).sum::<C>()/h as f64;defect=defect.max((dot-C::new(if a==b {1.0}else{0.0},0.0)).norm());}
            for matrix in &d.matrices {for i in 0..n {for j in 0..n {let dot:C=(0..n).map(|r|matrix[r+n*i].conj()*matrix[r+n*j]).sum();defect=defect.max((dot-C::new(if i==j {1.0}else{0.0},0.0)).norm());}}}
        }
        if !defect.is_finite() || defect>1e-9 {return Err(RepresentationError::InvalidRepresentation);}
        Ok(Self{group,k,irreps,multiplier,maximum_law_defect:defect})
    }
    /// Coefficients for P^alpha_ij. With i=j this projects one irrep row;
    /// with i≠j it reconstructs the partner row. No eigenvalues are deduplicated.
    pub fn matrix_unit_weights(&self,irrep:usize,i:usize,j:usize)->Vec<C> {
        let d=&self.irreps[irrep];assert!(i<d.dimension && j<d.dimension);
        d.matrices.iter().map(|g|g[i+d.dimension*j].conj()*(d.dimension as f64/self.group.order() as f64)).collect()
    }
    pub fn isotypic_weights(&self,irrep:usize)->Vec<C> {let d=&self.irreps[irrep];(0..self.group.order()).map(|g|d.character(g).conj()*(d.dimension as f64/self.group.order() as f64)).collect()}
    pub fn check_actions(&self,actions:&[QualifiedAction])->Result<(),RepresentationError> {
        if actions.len()!=self.group.order() || actions.iter().zip(&self.group.operations).any(|(a,g)|a.plan().operation()!=*g || a.plan().source()!=self.k || a.plan().destination()!=self.k) {return Err(RepresentationError::IncompatibleActions);}Ok(())
    }
    pub fn apply_weights<F:SpectralBuffer>(&self,actions:&[QualifiedAction],weights:&[C],input:&F,output:&mut F,scratch:&mut F)->Result<(),RepresentationError> {
        self.check_actions(actions)?;if weights.len()!=actions.len() {return Err(RepresentationError::IncompatibleActions);}
        output.as_mut_slice().fill(cscalar::<F::Real>(0.0,0.0));
        for (action,&weight) in actions.iter().zip(weights) {action.plan().apply_spatial(input,scratch);let w=cscalar::<F::Real>(weight.re,weight.im);for (dst,&v) in output.as_mut_slice().iter_mut().zip(scratch.as_slice()) {*dst+=w*v;}}
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;use super::super::affine::PlaneGroup;
    #[test]
    fn all_ordinary_point_families_obey_schur_and_matrix_unit_algebra() {
        for name in PlaneGroup::ALL {
            let p=RepresentationPlan::ordinary(name.group(),[0.0;2]).unwrap();let h=p.group.order();
            // Every matrix unit acts as E_ij in its own irrep and zero elsewhere.
            for (a,irrep) in p.irreps.iter().enumerate() {for i in 0..irrep.dimension {for j in 0..irrep.dimension {
                let weights=p.matrix_unit_weights(a,i,j);
                for (b,other) in p.irreps.iter().enumerate() {for row in 0..other.dimension {for col in 0..other.dimension {
                    let v:C=(0..h).map(|g|weights[g]*other.matrices[g][row+other.dimension*col]).sum();
                    let expected=if a==b && row==i && col==j {1.0}else{0.0};assert!((v-C::new(expected,0.0)).norm()<1e-12,"{name:?}");
                }}}
            }}}
        }
    }
    #[test]
    fn ordinary_tables_refuse_boundary_glides() {assert!(matches!(RepresentationPlan::ordinary(PlaneGroup::Pg.group(),[0.5,0.0]),Err(RepresentationError::ProjectiveMultiplier)));}
    #[test]
    fn projective_laws_for_all_51_little_group_fixtures() {
        for name in PlaneGroup::ALL {for k in [[0.0,0.0],[0.5,0.0],[0.21,0.13]] {
            let little=name.group().little_group(k,1e-12).unwrap();
            let plan=RepresentationPlan::projective(little,k).unwrap();assert!(plan.maximum_law_defect<1e-10,"{name:?} {k:?}");
        }}
        let glide=RepresentationPlan::projective(PlaneGroup::Pg.group(),[0.5,0.0]).unwrap();
        assert_eq!(glide.irreps.len(),2);for d in glide.irreps {assert_eq!(d.dimension,1);let v=d.matrices[1][0];assert!(v.re.abs()<1e-12);assert!((v.im.abs()-1.0).abs()<1e-12);}
    }
}
