//! Bounded, invocation-local spectral reuse. Cache keys compare both k
//! coordinates directly; material, grid, precision and output requests are
//! fixed by the owning band run and are never shared across different jobs.
use std::collections::VecDeque;
use serde::{Serialize,Deserialize};
use crate::field::Field2D;
use super::affine::{AffineOp,PlaneGroup,Rational};
use crate::{backend::SpectralBackend,dielectric::Dielectric2D,lattice::Lattice2D,operators::ThetaOperator};

#[derive(Debug,Clone,Copy,Default,PartialEq,Eq,Serialize,Deserialize)]
#[serde(rename_all="snake_case")]
pub enum KReuse { #[default] Off, Identical, Unitary, TimeReversal }

#[derive(Debug,Clone)]
pub struct SymmetryOptions {
    pub k_reuse:KReuse,
    pub k_traversal:super::traversal::KTraversal,
    /// Additional physical-field payload retained to restore requested output
    /// order and tracking after a changed solve route. Separate from the cache.
    pub max_traversal_field_bytes:usize,
    pub max_cache_bytes:usize,
    /// An explicit primitive setting, still subject to all numerical checks.
    /// None enumerates metric candidates with zero translation and entries ±2.
    pub plane_group:Option<PlaneGroup>,
    pub seed_transfer:super::seed::SeedTransfer,
    pub sectors:super::sectors::SectorPolicy,
    pub sector_guard_budget:Option<usize>,
    /// Rectangular sector plan, prefix and reconstruction payload admission.
    /// FFT plans, allocator/container overhead and library scratch are extra.
    pub max_sector_workspace_bytes:usize,
    pub sector_storage:super::reduced::SectorStorage,
    pub sector_kernel:super::reduced::ReducedKernel,
    pub reconstruct_partners:bool,
    /// Reuse the two full-grid projection buffers between serial sectors.
    /// The Maxwell operator and its FFT scratch are shared in either mode.
    pub share_sector_workspace:bool,
    /// Project the first correction batch and then every n-th batch. Seeds and
    /// final leakage/physical residual checks are unconditional. Full-grid only.
    pub correction_projection_interval:usize,
    pub arithmetic:super::fourier::Arithmetic,
    pub pt_operation:Option<AffineOp>,
    pub relative_rank_tolerance:f64,
    pub qualification_tolerance:f64,
    pub max_relative_residual:f64,
    pub max_zero_mode_absolute_residual:f64,
    pub max_orthogonality_defect:f64,
}
impl Default for SymmetryOptions {
    fn default()->Self {Self {k_reuse:KReuse::Off,k_traversal:super::traversal::KTraversal::Given,max_traversal_field_bytes:64*1024*1024,max_cache_bytes:64*1024*1024,plane_group:None,seed_transfer:super::seed::SeedTransfer::Copy,sectors:super::sectors::SectorPolicy::Off,sector_guard_budget:None,max_sector_workspace_bytes:super::sectors::memory::DEFAULT_LIMIT_BYTES,sector_storage:super::reduced::SectorStorage::FullGrid,sector_kernel:super::reduced::ReducedKernel::FullGrid,reconstruct_partners:false,share_sector_workspace:false,correction_projection_interval:1,arithmetic:super::fourier::Arithmetic::SpatialComplex,pt_operation:None,relative_rank_tolerance:1e-9,qualification_tolerance:1e-10,max_relative_residual:1e-5,max_zero_mode_absolute_residual:1e-7,max_orthogonality_defect:1e-8}}
}
#[derive(Debug,Clone,Serialize)]
pub struct SolveAttempt {
    pub method: &'static str,
    pub iterations: usize,
    /// Whether this attempt supplies the output; convergence and residual
    /// quality are separate fields on the enclosing point result.
    pub retained: bool,
    pub rejection: Option<String>,
}
#[derive(Debug,Clone,Default,Serialize)]
pub struct ReuseReport {
    pub requested:KReuse,
    pub traversal:Option<super::traversal::TraversalReport>,
    pub retained_iterations:usize,
    pub attempted_iterations:usize,
    pub solver_attempts:Vec<SolveAttempt>,
    pub source_index:Option<usize>,
    pub operation:Option<AffineOp>,
    pub cache_bytes:usize,
    pub cache_entries:usize,
    pub candidate_operation_count:usize,
    pub qualification_attempts:usize,
    pub qualification:Option<super::qualification::QualificationReport>,
    pub seed:Option<super::seed::SeedReport>,
    pub sectors:Option<super::sectors::CoverageReport>,
    pub coordinates:Option<super::fourier::CoordinateReport>,
    pub rejected:Vec<String>,
}
#[derive(Debug,Clone)]
pub struct CachedSpectrum {
    pub source_index:usize,
    pub k:[f64;2],
    pub eigenvalues:Vec<f64>,
    pub fields:Vec<Field2D>,
    pub residuals:Vec<f64>,
    pub absolute_residuals:Vec<f64>,
    pub orthogonality_defect:f64,
}
impl CachedSpectrum {
    pub fn bytes(&self)->usize {self.fields.iter().map(|f|f.len()*std::mem::size_of::<num_complex::Complex64>()).sum::<usize>()+(self.eigenvalues.len()+self.residuals.len()+self.absolute_residuals.len())*std::mem::size_of::<f64>()}
    pub fn passes(&self,options:&SymmetryOptions)->bool {
        let n=self.eigenvalues.len();
        n>0 && self.fields.len()==n && self.residuals.len()==n && self.absolute_residuals.len()==n &&
        self.fields.iter().all(|field|field.as_slice().iter().all(|z|z.re.is_finite() && z.im.is_finite())) &&
        self.orthogonality_defect.is_finite() && self.orthogonality_defect<=options.max_orthogonality_defect &&
        self.eigenvalues.iter().enumerate().all(|(i,&lambda)| lambda.is_finite() &&
            if lambda==0.0 && self.k==[0.0;2] {self.absolute_residuals[i].is_finite() && self.absolute_residuals[i]<=options.max_zero_mode_absolute_residual}
            else {self.residuals[i].is_finite() && self.residuals[i]<=options.max_relative_residual})
    }
}
pub struct SpectralCache { entries:VecDeque<CachedSpectrum>, bytes:usize, limit:usize }
impl SpectralCache {
    pub fn new(limit:usize)->Self {Self{entries:VecDeque::new(),bytes:0,limit}}
    pub fn exact(&self,k:[f64;2])->Option<CachedSpectrum> {self.entries.iter().find(|e|e.k==k).cloned()}
    pub fn insert(&mut self,entry:CachedSpectrum) {
        let bytes=entry.bytes();if bytes>self.limit || self.entries.iter().any(|e|e.k==entry.k) {return;}
        while bytes>self.limit-self.bytes {if let Some(old)=self.entries.pop_front() {self.bytes-=old.bytes();}else{break;}}
        self.bytes+=bytes;self.entries.push_back(entry);
    }
    pub fn bytes(&self)->usize {self.bytes}
    pub fn len(&self)->usize {self.entries.len()}
    /// Reconstruct only after finite support, material, operator, mass and fresh
    /// transformed residual checks. Failure leaves the ordinary solver in charge.
    pub fn related<B:SpectralBackend+Clone>(&self,target:&mut ThetaOperator<B>,lattice:&Lattice2D,dielectric:&Dielectric2D,k:[f64;2],operations:&[AffineOp],options:&SymmetryOptions,report:&mut ReuseReport)->Option<CachedSpectrum> {
        crate::profile_scope!("symmetry.reconstruct");
        report.candidate_operation_count=operations.len();
        for entry in &self.entries {for &op in operations {
            if op==AffineOp::IDENTITY {continue;}
            let raw=op.map_wavevector(entry.k);
            // Only floating arithmetic roundoff is tolerated; do not snap a
            // nearby requested point onto a symmetry orbit.
            let coordinate_tolerance=8.0*f64::EPSILON;
            if (0..2).any(|i|{let d=raw[i]-k[i];(d-d.round()).abs()>coordinate_tolerance*raw[i].abs().max(k[i].abs()).max(1.0)}) {continue;}
            let plan=match super::action::BlochActionPlan::new(op,entry.k,k,target.grid(),target.polarization(),coordinate_tolerance) {
                Ok(p)=>p,Err(e)=>{report.rejected.push(e.to_string());continue;}
            };
            let mut source=ThetaOperator::new(target.backend().clone(),dielectric.clone(),target.polarization(),lattice.reciprocal().fractional_to_cartesian(entry.k));
            report.qualification_attempts+=1;
            let qualified=match super::qualification::qualify(plan,lattice,&mut source,target,options.qualification_tolerance) {
                Ok(q)=>q,Err(q)=>{report.rejected.push(format!("{:?}",q.rejection_reasons));continue;}
            };
            let mut fields=Vec::with_capacity(entry.fields.len());
            for input in &entry.fields {let mut output=Field2D::zeros(input.grid());qualified.plan().apply_spatial(input,&mut output);fields.push(output);}
            let certificate=crate::eigensolver::refine::certify_without_refinement(target,&entry.eigenvalues,&fields);
            let result=CachedSpectrum {source_index:entry.source_index,k,eigenvalues:entry.eigenvalues.clone(),fields,residuals:certificate.residuals,absolute_residuals:certificate.absolute_residuals,orthogonality_defect:certificate.b_orthogonality_defect};
            if !result.passes(options) {report.rejected.push("transformed spectrum failed fresh quality gates".into());continue;}
            report.source_index=Some(entry.source_index);report.operation=Some(op);report.qualification=Some(qualified.report().clone());
            return Some(result);
        }}None
    }
}

/// Geometry candidates only. A lattice label or this list authorizes no solve.
pub fn candidate_operations(lattice:&Lattice2D,requested:Option<PlaneGroup>)->Vec<AffineOp> {
    if let Some(group)=requested {return group.group().operations;}
    let mut candidates=Vec::new();
    for a in -2..=2 {for b in -2..=2 {for c in -2..=2 {for d in -2..=2 {
        if let Ok(op)=AffineOp::new([[a,b],[c,d]],[Rational::ZERO;2]) {if op.metric_defect(lattice)<1e-12 {candidates.push(op);}}
    }}}}candidates
}

#[cfg(test)]
mod tests {
    use super::*;use crate::grid::Grid2D;
    fn entry(k:[f64;2])->CachedSpectrum {CachedSpectrum {source_index:0,k,eigenvalues:vec![1.0],fields:vec![Field2D::zeros(Grid2D::new(2,2,1.0,1.0))],residuals:vec![1e-8],absolute_residuals:vec![1e-8],orthogonality_defect:0.0}}
    #[test]
    fn exact_keys_and_bounded_eviction() {
        let a=entry([0.13,0.2]);let size=a.bytes();let mut cache=SpectralCache::new(size);cache.insert(a);
        assert!(cache.exact([0.13+1e-14,0.2]).is_none());cache.insert(entry([0.3,0.2]));assert!(cache.exact([0.13,0.2]).is_none());assert_eq!(cache.bytes(),size);assert_eq!(cache.len(),1);
        let mut zero=SpectralCache::new(0);zero.insert(entry([0.0;2]));assert_eq!(zero.len(),0);
    }
    #[test]
    fn refuses_failed_quality_and_keeps_zero_mode_absolute_branch() {
        let mut e=entry([0.0;2]);e.eigenvalues[0]=0.0;e.residuals[0]=1.0;assert!(e.passes(&Default::default()));
        e.absolute_residuals[0]=1.0;assert!(!e.passes(&Default::default()));e.k=[0.13,0.2];e.residuals[0]=f64::NAN;assert!(!e.passes(&Default::default()));
    }
}
