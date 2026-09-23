//! Lossless spectral-cache snapshots for frozen-input continuation experiments.
use serde::{Serialize,Deserialize};
use crate::{backend::SpectralBackend,field::Field2D,grid::Grid2D,operators::{LinearOperator,ThetaOperator},polarization::Polarization};
use super::reuse::CachedSpectrum;

#[derive(Debug,Clone,PartialEq,Eq,Serialize,Deserialize)]
pub struct SnapshotIdentity {
    pub nx:usize,
    pub ny:usize,
    pub storage_bits:usize,
    pub centered:bool,
    /// Exact bitwise operator description, compared directly rather than relying
    /// on a hash collision assumption. Includes both scalar and tensor material.
    pub operator_bits:Vec<u64>,
}
impl SnapshotIdentity {
    pub fn from_operator<B:SpectralBackend>(operator:&ThetaOperator<B>)->Self {
        let grid=operator.grid();let d=operator.dielectric();
        let mut bits=vec![grid.lx.to_bits(),grid.ly.to_bits(),if operator.polarization()==Polarization::TM{0}else{1}];
        bits.extend(operator.bloch().map(f64::to_bits));bits.extend(d.reciprocal_b1().map(f64::to_bits));bits.extend(d.reciprocal_b2().map(f64::to_bits));
        bits.extend(d.eps().iter().map(|x|x.to_bits()));bits.extend(d.inv_eps().iter().map(|x|x.to_bits()));
        bits.push(u64::from(d.inv_eps_tensors().is_some()));
        if let Some(t)=d.inv_eps_tensors() {bits.extend(t.iter().flatten().map(|x|x.to_bits()));}
        Self{nx:grid.nx,ny:grid.ny,centered:grid.centered,storage_bits:std::mem::size_of::<B::Real>()*8,operator_bits:bits}
    }
}
#[derive(Debug,Clone,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpectralSnapshot {
    pub schema:String,
    pub representation:String,
    pub identity:SnapshotIdentity,
    pub source_revision:String,
    pub source_index:usize,
    pub k_bits:[u64;2],
    pub eigenvalue_bits:Vec<u64>,
    pub field_bits:Vec<Vec<[u64;2]>>,
    pub residual_bits:Vec<u64>,
    pub absolute_residual_bits:Vec<u64>,
    pub orthogonality_bits:u64,
    pub kernel_ids:Vec<Option<String>>,
    pub cluster_ranges:Vec<[usize;2]>,
    pub guard_columns:usize,
    pub coverage:String,
}
#[derive(Debug,Clone,PartialEq,Eq,thiserror::Error)]
pub enum SnapshotError {
    #[error("cache schema, representation, or operator identity is incompatible")]
    Incompatible,
    #[error("cache dimensions, finite values, or multiplicities are invalid")]
    InvalidData,
    #[error("fresh certification of replayed fields failed")]
    FailedCertification,
}
impl SpectralSnapshot {
    pub fn capture(entry:&CachedSpectrum,identity:SnapshotIdentity,revision:&str)->Self {
        let mut clusters=Vec::new();let mut start=0;
        while start<entry.eigenvalues.len() {let mut end=start+1;while end<entry.eigenvalues.len() && (entry.eigenvalues[end]-entry.eigenvalues[end-1]).abs()<=1e-7*entry.eigenvalues[end].abs().max(1.0) {end+=1;}clusters.push([start,end]);start=end;}
        Self {schema:"blaze2d/spectral-cache/1".into(),representation:"spatial_periodic_complex128".into(),identity,source_revision:revision.into(),source_index:entry.source_index,k_bits:entry.k.map(f64::to_bits),
            eigenvalue_bits:entry.eigenvalues.iter().map(|x|x.to_bits()).collect(),field_bits:entry.fields.iter().map(|f|f.as_slice().iter().map(|z|[z.re.to_bits(),z.im.to_bits()]).collect()).collect(),
            residual_bits:entry.residuals.iter().map(|x|x.to_bits()).collect(),absolute_residual_bits:entry.absolute_residuals.iter().map(|x|x.to_bits()).collect(),orthogonality_bits:entry.orthogonality_defect.to_bits(),
            kernel_ids:entry.eigenvalues.iter().map(|&x|(x==0.0 && entry.k==[0.0;2]).then(||"scalar_gamma".into())).collect(),cluster_ranges:clusters,guard_columns:0,coverage:"unrestricted_prefix; cluster boundary completeness not certified".into()}
    }
    /// Decode a seed block exactly. Stored residual metadata is provenance, not
    /// authorization for exact solve reuse; use `restore_checked` for that.
    pub fn restore(self,expected:&SnapshotIdentity)->Result<CachedSpectrum,SnapshotError> {
        if self.schema!="blaze2d/spectral-cache/1" || self.representation!="spatial_periodic_complex128" || &self.identity!=expected {return Err(SnapshotError::Incompatible);}
        let size=expected.nx.checked_mul(expected.ny).ok_or(SnapshotError::InvalidData)?;let n=self.eigenvalue_bits.len();
        if size==0 || n==0 || self.field_bits.len()!=n || self.residual_bits.len()!=n || self.absolute_residual_bits.len()!=n || self.kernel_ids.len()!=n || expected.operator_bits.len()<9 ||
            self.field_bits.iter().any(|f|f.len()!=size) || self.kernel_ids.iter().filter(|v|v.is_some()).count()>1 || self.guard_columns>n {return Err(SnapshotError::InvalidData);}
        let mut next=0;for &[start,end] in &self.cluster_ranges {if start!=next || end<=start || end>n {return Err(SnapshotError::InvalidData);}next=end;}if next!=n {return Err(SnapshotError::InvalidData);}
        let all=self.k_bits.iter().chain(&self.eigenvalue_bits).chain(&self.residual_bits).chain(&self.absolute_residual_bits).chain(std::iter::once(&self.orthogonality_bits)).chain(self.field_bits.iter().flat_map(|f|f.iter().flatten()));
        if all.into_iter().any(|&v|!f64::from_bits(v).is_finite()) {return Err(SnapshotError::InvalidData);}
        let k=self.k_bits.map(f64::from_bits);
        for axis in 0..2 {
            let cart=k[0]*f64::from_bits(expected.operator_bits[5+axis])+k[1]*f64::from_bits(expected.operator_bits[7+axis]);
            let expected_cart=f64::from_bits(expected.operator_bits[3+axis]);
            if !cart.is_finite() || (cart-expected_cart).abs()>8.0*f64::EPSILON*expected_cart.abs().max(1.0) {return Err(SnapshotError::Incompatible);}
        }
        for i in 0..n {
            let is_kernel=self.k_bits.map(f64::from_bits)==[0.0;2] && f64::from_bits(self.eigenvalue_bits[i])==0.0;
            if self.kernel_ids[i].as_deref()!=if is_kernel {Some("scalar_gamma")}else{None} {return Err(SnapshotError::InvalidData);}
        }
        let mut grid=Grid2D::new(expected.nx,expected.ny,f64::from_bits(expected.operator_bits[0]),f64::from_bits(expected.operator_bits[1]));grid.centered=expected.centered;
        Ok(CachedSpectrum {source_index:self.source_index,k:self.k_bits.map(f64::from_bits),eigenvalues:self.eigenvalue_bits.into_iter().map(f64::from_bits).collect(),
            fields:self.field_bits.into_iter().map(|f|Field2D::from_vec(grid,f.into_iter().map(|z|num_complex::Complex64::new(f64::from_bits(z[0]),f64::from_bits(z[1]))).collect())).collect(),
            residuals:self.residual_bits.into_iter().map(f64::from_bits).collect(),absolute_residuals:self.absolute_residual_bits.into_iter().map(f64::from_bits).collect(),orthogonality_defect:f64::from_bits(self.orthogonality_bits)})
    }
    pub fn restore_checked<B:SpectralBackend>(self,operator:&mut ThetaOperator<B>,options:&super::reuse::SymmetryOptions)->Result<CachedSpectrum,SnapshotError> {
        let mut entry=self.restore(&SnapshotIdentity::from_operator(operator))?;
        let certificate=crate::eigensolver::refine::certify_without_refinement(operator,&entry.eigenvalues,&entry.fields);
        entry.residuals=certificate.residuals;entry.absolute_residuals=certificate.absolute_residuals;entry.orthogonality_defect=certificate.b_orthogonality_defect;
        if !entry.passes(options) {return Err(SnapshotError::FailedCertification);}Ok(entry)
    }
}
