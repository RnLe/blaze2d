//! Target-metric rank extraction and fresh Rayleigh–Ritz seed preparation.
use num_complex::Complex64 as C;
use serde::{Serialize,Deserialize};
use crate::{backend::{SpectralBackend,SpectralBuffer},field::{Field2D,Real,cscalar},operators::LinearOperator,eigensolver::{dense::solve_hermitian_eigen,refine::certify_without_refinement}};

#[derive(Debug,Clone,thiserror::Error)]
pub enum SeedError {
    #[error("rank-aware transfer currently requires f64 field storage")]
    PrecisionUnsupported,
    #[error("seed dimensions, rank tolerance, or numerical values are invalid")]
    InvalidInput,
    #[error("target mass Gram matrix is not positive semidefinite")]
    InvalidMetric,
    #[error("rank-selected seed failed target mass orthogonality")]
    OrthogonalityFailure,
}
#[derive(Debug,Clone,Serialize)]
pub struct SeedReport {
    pub source_columns:usize,
    pub retained_rank:usize,
    pub singular_values:Vec<f64>,
    pub gram_eigenvalue_threshold:f64,
    pub discarded_gram_energy:f64,
    pub initial_ritz_values:Vec<f64>,
    pub target_residuals:Vec<f64>,
    pub target_absolute_residuals:Vec<f64>,
    pub b_orthogonality_defect:f64,
}
pub struct RankAwareSeed {pub fields:Vec<Field2D>,pub report:SeedReport}

#[derive(Debug,Clone,Copy,Default,PartialEq,Eq,Serialize,Deserialize)]
#[serde(rename_all="snake_case")]
pub enum SeedTransfer {#[default] Copy, RankAware}

/// Input columns may already have been projected into a qualified target sector.
/// This prepares independent initial directions only; rank is not a certificate
/// that the globally lowest requested modes are present.
pub fn rank_aware_seed<O,B>(operator:&mut O,input:&[Field2D],relative_rank_tolerance:f64)->Result<RankAwareSeed,SeedError>
where O:LinearOperator<B>,B:SpectralBackend {
    crate::profile_scope!("initialization.rank_aware");
    if std::mem::size_of::<B::Real>()!=8 {return Err(SeedError::PrecisionUnsupported);}
    if !relative_rank_tolerance.is_finite() || relative_rank_tolerance<=0.0 || relative_rank_tolerance>=1.0 || input.iter().any(|f|f.grid().nx!=operator.grid().nx || f.grid().ny!=operator.grid().ny || f.as_slice().iter().any(|z|!z.re.is_finite() || !z.im.is_finite())) {return Err(SeedError::InvalidInput);}
    crate::init_faer_sequential();let p=input.len();
    let mut columns=Vec::with_capacity(p);let mut mass=Vec::with_capacity(p);
    for field in input {let mut v=operator.alloc_field();for (dst,src) in v.as_mut_slice().iter_mut().zip(field.as_slice()) {*dst=cscalar::<B::Real>(src.re,src.im);}let mut b=operator.alloc_field();operator.apply_mass(&v,&mut b);columns.push(v);mass.push(b);}
    let mut gram=vec![C::ZERO;p*p];
    for i in 0..p {for j in i..p {let a=operator.backend().dot(&columns[i],&mass[j]);let b=operator.backend().dot(&columns[j],&mass[i]).conj();let g=0.5*(a+b);gram[i+p*j]=g;gram[j+p*i]=g.conj();}}
    if gram.iter().any(|z|!z.re.is_finite() || !z.im.is_finite()) {return Err(SeedError::InvalidMetric);}
    let spectrum=solve_hermitian_eigen(&gram,p);let largest=spectrum.eigenvalues.last().copied().unwrap_or(0.0).max(0.0);
    // Gram formation squares the condition number. State the precision floor
    // explicitly instead of claiming to resolve arbitrarily small singular values.
    let threshold=(relative_rank_tolerance.powi(2)).max(8.0*f64::EPSILON*p as f64)*largest;
    if spectrum.eigenvalues.iter().any(|&v|!v.is_finite() || v < -threshold.max(f64::MIN_POSITIVE)) {return Err(SeedError::InvalidMetric);}
    let retained:Vec<_>=(0..p).filter(|&i|spectrum.eigenvalues[i]>threshold).collect();let rank=retained.len();
    crate::profiler::count("seed.source_columns",p);crate::profiler::count("seed.retained_rank",rank);crate::profiler::count("seed.dropped_columns",p-rank);
    let mut y:Vec<B::Buffer>=Vec::with_capacity(rank);let mut by:Vec<B::Buffer>=Vec::with_capacity(rank);
    for &col in &retained {
        let mut v=operator.alloc_field();let factor=spectrum.eigenvalues[col].sqrt().recip();
        for j in 0..p {operator.backend().axpy(spectrum.eigenvectors[j+p*col]*factor,&columns[j],&mut v);}
        let mut b=operator.alloc_field();operator.apply_mass(&v,&mut b);
        // Reorthogonalize in the actual target metric, including the smaller
        // retained directions. This avoids trusting a marginal Gram inverse.
        for _ in 0..2 {for (q,bq) in y.iter().zip(&by) {let overlap=operator.backend().dot(q,&b);operator.backend().axpy(-overlap,q,&mut v);operator.backend().axpy(-overlap,bq,&mut b);}}
        operator.apply_mass(&v,&mut b);let norm=operator.backend().dot(&v,&b).re;
        if !norm.is_finite() || norm<=1e-12 {return Err(SeedError::OrthogonalityFailure);}
        operator.backend().scale(C::new(norm.sqrt().recip(),0.0),&mut v);operator.backend().scale(C::new(norm.sqrt().recip(),0.0),&mut b);y.push(v);by.push(b);
    }
    let mut h=vec![C::ZERO;rank*rank];let mut a=operator.alloc_field();
    for j in 0..rank {operator.apply(&y[j],&mut a);for i in 0..rank {h[i+rank*j]=operator.backend().dot(&y[i],&a);}}
    for i in 0..rank {for j in i..rank {let z=0.5*(h[i+rank*j]+h[j+rank*i].conj());h[i+rank*j]=z;h[j+rank*i]=z.conj();}}
    if h.iter().any(|z|!z.re.is_finite() || !z.im.is_finite()) {return Err(SeedError::InvalidInput);}
    let ritz=solve_hermitian_eigen(&h,rank);let mut fields=Vec::with_capacity(rank);
    for j in 0..rank {let mut v=operator.alloc_field();for i in 0..rank {operator.backend().axpy(ritz.eigenvectors[i+rank*j],&y[i],&mut v);}fields.push(Field2D::from_vec(operator.grid(),v.as_slice().iter().map(|z|C::new(z.re.to_accum(),z.im.to_accum())).collect()));}
    let cert=certify_without_refinement(operator,&ritz.eigenvalues,&fields);
    if !cert.b_orthogonality_defect.is_finite() || cert.b_orthogonality_defect>1e-8 {return Err(SeedError::OrthogonalityFailure);}
    let report=SeedReport{source_columns:p,retained_rank:rank,singular_values:spectrum.eigenvalues.iter().map(|v|v.max(0.0).sqrt()).collect(),gram_eigenvalue_threshold:threshold,
        discarded_gram_energy:spectrum.eigenvalues.iter().filter(|&&v|v<=threshold).map(|v|v.max(0.0)).sum(),initial_ritz_values:ritz.eigenvalues,target_residuals:cert.residuals,target_absolute_residuals:cert.absolute_residuals,b_orthogonality_defect:cert.b_orthogonality_defect};
    Ok(RankAwareSeed{fields,report})
}
