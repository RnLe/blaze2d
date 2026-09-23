//! Qualified full-grid sector solves. Spatial operator work remains full size.
use num_complex::Complex64 as C;
use serde::Serialize;
use crate::{backend::{SpectralBackend,SpectralBuffer},field::{Field2D,Real,cscalar},grid::Grid2D,lattice::Lattice2D,operators::{LinearOperator,ThetaOperator},eigensolver::{Eigensolver,EigensolverConfig,dense::solve_hermitian_eigen,refine::{rayleigh_ritz_certify,certify_without_refinement}}};
use super::{affine::AffineGroup,action::BlochActionPlan,qualification::{QualifiedAction,qualify},representation::RepresentationPlan,reuse::{CachedSpectrum,SymmetryOptions}};

#[cfg(not(target_arch = "wasm32"))]
pub mod concurrency;

#[derive(Debug,Clone,Copy,Default,PartialEq,Eq,Serialize)]
#[serde(rename_all="snake_case")]
pub enum SectorPolicy {#[default] Off, ConservativeMirror, AdaptiveMirror, ConservativeGroup, AdaptiveGroup}

pub struct SectorPlan {
    representation:RepresentationPlan,
    actions:Vec<QualifiedAction>,
    sectors:Vec<Sector>,
}
struct Sector {label:String,irrep:usize,partners:usize,weights:Vec<C>,rank:usize,owns_kernel:bool}

#[derive(Debug,Clone,Serialize)]
pub struct SectorReport {
    pub label:String,
    pub dimension:usize,
    pub prefix:usize,
    pub working_columns:usize,
    pub stored_rows:usize,
    pub vector_column_bytes:usize,
    pub basis_payload_bytes:usize,
    pub reduced_kernel:Option<super::reduced::KernelReport>,
    pub partner_rows:usize,
    pub owns_gamma_kernel:bool,
    pub iterations:usize,
    pub dense_complete:bool,
    pub seed_rank:usize,
    pub leakage:f64,
    pub correction_projection_interval:usize,
    pub correction_projection_rounds:usize,
    pub correction_projection_skipped_rounds:usize,
}
#[derive(Debug,Clone,Serialize)]
pub struct CoverageReport {
    pub evidence:String,
    pub finite_dimension:usize,
    pub sum_sector_dimensions:usize,
    pub requested:usize,
    pub computed:usize,
    pub kernel_owners:usize,
    pub qualification_bytes:usize,
    pub projection_workspace_allocations:usize,
    pub projection_workspace_payload_bytes:usize,
    pub allocation_rounds:usize,
    pub tail_eigenvalues:Vec<Option<f64>>,
    pub sectors:Vec<SectorReport>,
}
pub struct SectorResult {pub spectrum:CachedSpectrum,pub iterations:usize,pub report:CoverageReport}

impl SectorPlan {
    pub fn qualify<B:SpectralBackend+Clone>(operator:&mut ThetaOperator<B>,lattice:&Lattice2D,k:[f64;2],group:AffineGroup,options:&SymmetryOptions)->Result<Self,String> {
        if std::mem::size_of::<B::Real>()!=8 {return Err("sector solves currently require f64".into());}
        let representation=RepresentationPlan::ordinary(group.clone(),k).or_else(|_|RepresentationPlan::projective(group,k)).map_err(|e|e.to_string())?;
        let mut other=ThetaOperator::new(operator.backend().clone(),operator.dielectric().clone(),operator.polarization(),operator.bloch());
        let mut actions=Vec::new();
        for &g in &representation.group.operations {
            let plan=BlochActionPlan::new(g,k,k,operator.grid(),operator.polarization(),1e-12).map_err(|e|e.to_string())?;
            actions.push(qualify(plan,lattice,operator,&mut other,options.qualification_tolerance).map_err(|r|format!("sector qualification: {:?}",r.rejection_reasons))?);
        }
        let mut sectors=Vec::new();let grid=operator.grid();let gamma=k==[0.0;2];
        let mut constant=operator.alloc_field();constant.as_mut_slice().fill(cscalar::<B::Real>(1.0,0.0));
        let mut out=operator.alloc_field();let mut scratch=operator.alloc_field();
        for i in 0..representation.irreps.len() {
            let partners=if options.reconstruct_partners {representation.irreps[i].dimension} else {1};
            let weights=if options.reconstruct_partners {representation.matrix_unit_weights(i,0,0)} else {representation.isotypic_weights(i)};
            let trace:C=actions.iter().zip(&weights).map(|(a,w)|a.plan().spatial_trace()*w).sum();
            if trace.im.abs()>1e-7 || (trace.re-trace.re.round()).abs()>1e-7 || trace.re < -1e-7 {return Err("nonintegral finite sector rank".into());}
            let rank=trace.re.round() as usize;
            representation.apply_weights(&actions,&weights,&constant,&mut out,&mut scratch).map_err(|e|e.to_string())?;
            let constant_part=operator.backend().dot(&out,&out).re/grid.len() as f64;
            if gamma && constant_part>1e-9 && (constant_part-1.0).abs()>1e-9 {return Err("ambiguous Gamma kernel ownership".into());}
            sectors.push(Sector{label:representation.irreps[i].label.clone(),irrep:i,partners,weights,rank,owns_kernel:gamma && constant_part>0.5});
        }
        if sectors.iter().map(|s|s.rank*s.partners).sum::<usize>()!=grid.len() || (gamma && sectors.iter().filter(|s|s.owns_kernel).count()!=1) {return Err("sector dimensions or kernel ownership do not cover the finite space".into());}
        Ok(Self{representation,actions,sectors})
    }
    pub fn ranks(&self)->Vec<usize> {self.sectors.iter().map(|s|s.rank).collect()}
    pub(crate) fn project<F:SpectralBuffer>(&self,sector:usize,input:&F,output:&mut F,scratch:&mut F) {
        crate::profile_scope!("symmetry.project");
        self.representation.apply_weights(&self.actions,&self.sectors[sector].weights,input,output,scratch).expect("plan actions were checked at construction");
    }
}

struct ConstrainedOperator<'a,B:SpectralBackend> {
    full:&'a mut ThetaOperator<B>,plan:&'a SectorPlan,sector:usize,output:B::Buffer,scratch:B::Buffer,
    correction_projection_interval:usize,correction_rounds:usize,projected_rounds:usize,
}
impl<B:SpectralBackend> LinearOperator<B> for ConstrainedOperator<'_,B> {
    fn apply(&mut self,x:&B::Buffer,y:&mut B::Buffer) {self.full.apply(x,y)}
    fn apply_mass(&mut self,x:&B::Buffer,y:&mut B::Buffer) {self.full.apply_mass(x,y)}
    fn alloc_field(&self)->B::Buffer {self.full.alloc_field()}
    fn backend(&self)->&B {self.full.backend()}
    fn backend_mut(&mut self)->&mut B {self.full.backend_mut()}
    fn grid(&self)->Grid2D {self.full.grid()}
    fn bloch(&self)->[f64;2] {self.full.bloch()}
    fn owns_gamma_kernel(&self)->bool {self.plan.sectors[self.sector].owns_kernel}
    fn subspace_dimension(&self)->usize {self.plan.sectors[self.sector].rank}
    fn constrain(&mut self,x:&mut B::Buffer) {
        self.plan.project(self.sector,x,&mut self.output,&mut self.scratch);
        x.as_mut_slice().copy_from_slice(self.output.as_slice());
    }
    fn constrain_corrections(&mut self,vectors:&mut[B::Buffer]) {
        if vectors.is_empty() {return;}
        if self.correction_rounds%self.correction_projection_interval==0 {
            for vector in vectors {self.constrain(vector);}
            self.projected_rounds+=1;
            crate::profiler::count("symmetry.correction_projection_rounds",1);
        } else {
            crate::profiler::count("symmetry.skipped_correction_projection_rounds",1);
        }
        self.correction_rounds+=1;
    }
}

fn promote<B:SpectralBackend>(grid:Grid2D,v:&B::Buffer)->Field2D {Field2D::from_vec(grid,v.as_slice().iter().map(|z|C::new(z.re.to_accum(),z.im.to_accum())).collect())}

/// Full finite-dimensional reference for a small or exhausted sector. Basis is
/// B-orthonormal and A is always applied by the original Maxwell operator.
fn dense_sector<B:SpectralBackend>(op:&mut ConstrainedOperator<'_,B>)->Result<(Vec<f64>,Vec<Field2D>),String> {
    crate::profile_scope!("symmetry.dense_complete");
    let rank=op.subspace_dimension();let n=op.grid().len();
    if n.checked_mul(rank).and_then(|x|x.checked_mul(48)).is_none_or(|x|x>64*1024*1024) {return Err("complete sector basis exceeds 64 MiB workspace budget".into());}
    let mut q:Vec<B::Buffer>=Vec::new();let mut bq:Vec<B::Buffer>=Vec::new();
    for j in 0..n {
        let mut v=op.alloc_field();v.as_mut_slice()[j]=cscalar::<B::Real>(1.0,0.0);op.constrain(&mut v);
        let mut b=op.alloc_field();op.apply_mass(&v,&mut b);
        for _ in 0..2 {for (x,bx) in q.iter().zip(&bq) {let dot=op.backend().dot(x,&b);op.backend().axpy(-dot,x,&mut v);op.backend().axpy(-dot,bx,&mut b);}}
        let norm=op.backend().dot(&v,&b).re.max(0.0).sqrt();if norm<=1e-10 {continue;}
        op.backend().scale(C::new(1.0/norm,0.0),&mut v);op.backend().scale(C::new(1.0/norm,0.0),&mut b);q.push(v);bq.push(b);
        if q.len()==rank {break;}
    }
    if q.len()!=rank {return Err(format!("sector basis rank {} differs from character trace {rank}",q.len()));}
    let mut h=vec![C::ZERO;rank*rank];let mut a=op.alloc_field();
    for j in 0..rank {op.apply(&q[j],&mut a);for i in 0..rank {h[i+rank*j]=op.backend().dot(&q[i],&a);}}
    let eig=solve_hermitian_eigen(&h,rank);let mut fields=Vec::new();
    for j in 0..rank {let mut v=op.alloc_field();for i in 0..rank {op.backend().axpy(eig.eigenvectors[i+rank*j],&q[i],&mut v);}fields.push(promote::<B>(op.grid(),&v));}
    let mut values=eig.eigenvalues;
    if op.owns_gamma_kernel() && values.first().is_some_and(|v|v.abs()<1e-8) {values[0]=0.0;}
    Ok((values,fields))
}

/// Conservative coverage: request at least q lowest values from every nonempty
/// sector (or exhaust it). Iterative prefix ordering remains empirical; only a
/// full dense solve is a finite-spectrum certificate.
pub fn solve_conservative<B:SpectralBackend+Clone>(operator:&mut ThetaOperator<B>,lattice:&Lattice2D,k:[f64;2],group:AffineGroup,config:&EigensolverConfig,warm:Option<&[Field2D]>,options:&SymmetryOptions,mut preconditioner:Option<&mut dyn crate::preconditioners::OperatorPreconditioner<B>>)->Result<SectorResult,String> {
    crate::profile_scope!("symmetry.sectors");
    if options.correction_projection_interval==0 {return Err("correction projection interval must be positive".into());}
    if options.correction_projection_interval!=1 && options.sector_storage!=super::reduced::SectorStorage::FullGrid {
        return Err("correction projection cadence applies only to full-grid sectors".into());
    }
    let plan=SectorPlan::qualify(operator,lattice,k,group,options)?;
    let bases=if options.sector_storage==super::reduced::SectorStorage::Orbit {
        plan.sectors.iter().map(|s|if s.rank==0 {Ok(None)} else {super::reduced::OrbitBasis::from_projector(&plan.actions,&s.weights,s.rank).map(Some)}).collect::<Result<Vec<_>,String>>()?
    } else {(0..plan.sectors.len()).map(|_|None).collect()};
    let adaptive=is_adaptive(options);
    let mut prefixes=initial_prefixes(&plan,config.n_bands,adaptive);
    let mut slots:Vec<Option<(CachedSpectrum,SectorReport)>>=(0..plan.sectors.len()).map(|_|None).collect();
    let mut iterations=0;let mut rounds=0;
    // Invocation-local ownership prevents cross-job aliasing. Both modes use
    // exactly the same operator and buffers; only their allocation lifetime
    // differs. This pool does not include persistent eigensolver vectors.
    let mut projection_workspace:Option<(B::Buffer,B::Buffer)>=None;
    let mut projection_workspace_allocations=0;
    let projection_workspace_payload_bytes=2*operator.grid().len()*std::mem::size_of::<num_complex::Complex<B::Real>>();
    loop {
    rounds+=1;
    let guards=allocate_guards(&prefixes,options.sector_guard_budget,operator.grid().len());
    for (i,sector) in plan.sectors.iter().enumerate() {
        if sector.rank==0 {continue;}
        let prefix=prefixes[i];
        if slots[i].as_ref().is_some_and(|(_,r)|r.prefix==prefix) {continue;}
        let previous_steps=slots[i].as_ref().map_or(0,|(_,r)|r.iterations);
        let previous_projected=slots[i].as_ref().map_or(0,|(_,r)|r.correction_projection_rounds);
        let previous_skipped=slots[i].as_ref().map_or(0,|(_,r)|r.correction_projection_skipped_rounds);
        let warm=slots[i].as_ref().map(|(e,_)|e.fields.as_slice()).or(warm);
        let (output,scratch)=projection_workspace.take().unwrap_or_else(|| {
            projection_workspace_allocations+=2;
            crate::profiler::count("sector.projection_workspace_allocations",2);
            (operator.alloc_field(),operator.alloc_field())
        });
        let (entry,mut report,(output,scratch))=solve_prefix(operator,&plan,i,config,prefix,guards[i],rounds,k,warm,options,bases[i].as_ref(),preconditioner.as_mut().map(|p|&mut **p as &mut dyn crate::preconditioners::OperatorPreconditioner<B>),output,scratch)?;
        iterations+=report.iterations;
        report.iterations+=previous_steps;
        report.correction_projection_rounds+=previous_projected;
        report.correction_projection_skipped_rounds+=previous_skipped;
        slots[i]=Some((entry,report));
        if options.share_sector_workspace {
            projection_workspace=Some((output,scratch));
        }
    }
    let grow=grow_prefixes(&slots,&mut prefixes,config.n_bands,adaptive);
    if !grow {break;}
    }
    finish_sectors(operator,&plan,slots,k,config.n_bands,adaptive,options,iterations,rounds,
        projection_workspace_allocations,projection_workspace_payload_bytes)
}

/// One unchanged prefix solve, shared by serial and worker-owned schedules.
#[allow(clippy::too_many_arguments)]
fn solve_prefix<B:SpectralBackend+Clone>(
    operator:&mut ThetaOperator<B>, plan:&SectorPlan, i:usize,
    config:&EigensolverConfig, prefix:usize, guard_columns:usize, rounds:usize, k:[f64;2],
    warm:Option<&[Field2D]>, options:&SymmetryOptions,
    basis:Option<&super::reduced::OrbitBasis>,
    mut preconditioner:Option<&mut dyn crate::preconditioners::OperatorPreconditioner<B>>,
    output:B::Buffer, scratch:B::Buffer,
)->Result<(CachedSpectrum,SectorReport,(B::Buffer,B::Buffer)),String> {
    let sector=&plan.sectors[i];
        let mut op=ConstrainedOperator{full:operator,plan,sector:i,output,scratch,correction_projection_interval:options.correction_projection_interval,correction_rounds:0,projected_rounds:0};
        let mut seed_rank=0;
        let projected_seed=if let Some(warm)=warm {
            let mut projected=Vec::new();
            for field in warm {let mut v=op.alloc_field();for (dst,z) in v.as_mut_slice().iter_mut().zip(field.as_slice()) {*dst=cscalar::<B::Real>(z.re,z.im);}op.constrain(&mut v);projected.push(promote::<B>(op.grid(),&v));}
            let seed=super::seed::rank_aware_seed(&mut op,&projected,options.relative_rank_tolerance).map_err(|e|e.to_string())?;seed_rank=seed.fields.len();Some(seed.fields)
        } else {None};
        let dense=sector.rank<=prefix+2;
        let mut reduced_kernel=None;
        let (mut values,mut fields,steps)=if dense {let (a,b)=dense_sector(&mut op)?;(a,b,0)} else {
            let mut cfg=config.clone();cfg.n_bands=prefix;cfg.block_size=prefix.saturating_add(guard_columns).min(sector.rank-usize::from(sector.owns_kernel));
            if let Some(basis)=basis {
                let result=super::reduced::solve_reduced(op.full,basis,sector.owns_kernel,cfg,projected_seed.as_deref(),preconditioner.as_mut().map(|p| &mut **p as &mut dyn crate::preconditioners::OperatorPreconditioner<B>),options.sector_kernel)?;
                reduced_kernel=Some(result.kernel);(result.eigenvalues,result.fields,result.iterations)
            } else {
            let mut solver=Eigensolver::new(&mut op,cfg,preconditioner.as_mut().map(|p| &mut **p as &mut dyn crate::preconditioners::OperatorPreconditioner<B>),projected_seed.as_deref());
            let result=solver.solve();let fields=solver.all_eigenvectors();
            if !result.converged {return Err(format!("sector {} failed to converge after {} iterations (prefix {prefix}, round {rounds}, seed rank {seed_rank})",sector.label,result.iterations));}
            (result.eigenvalues,fields,result.iterations)
            }
        };
        values.truncate(prefix);fields.truncate(prefix);
        // Refine within this sector, never across different physical labels.
        let _=rayleigh_ritz_certify(&mut op,&mut values,&mut fields);
        if sector.owns_kernel && values.first().is_some_and(|v|v.abs()<1e-8) {values[0]=0.0;}
        let cert=certify_without_refinement(&mut op,&values,&fields);
        let entry=CachedSpectrum{source_index:0,k,eigenvalues:values,fields,residuals:cert.residuals,absolute_residuals:cert.absolute_residuals,orthogonality_defect:cert.b_orthogonality_defect};
        let mut leakage:f64=0.0;
        for field in &entry.fields {let mut v=op.alloc_field();for (dst,z) in v.as_mut_slice().iter_mut().zip(field.as_slice()) {*dst=cscalar::<B::Real>(z.re,z.im);}op.constrain(&mut v);let error:f64=v.as_slice().iter().zip(field.as_slice()).map(|(a,b)|(C::new(a.re.to_accum(),a.im.to_accum())-b).norm_sqr()).sum();let scale:f64=field.as_slice().iter().map(|z|z.norm_sqr()).sum();leakage=leakage.max((error/scale).sqrt());}
        if !entry.passes(options) || entry.fields.len()!=prefix {return Err(format!("sector {} failed fresh quality gates: residuals={:?}, orthogonality={:e}, leakage={leakage:e}",sector.label,entry.residuals,entry.orthogonality_defect));}
        if !leakage.is_finite() || leakage>1e-8 {return Err(format!("sector {} leaked by {leakage:e}",sector.label));}
        let stored_rows=if !dense && basis.is_some() {sector.rank} else {op.grid().len()};
        let report=SectorReport{label:sector.label.clone(),dimension:sector.rank,prefix,stored_rows,reduced_kernel,vector_column_bytes:stored_rows*std::mem::size_of::<C>(),basis_payload_bytes:basis.map_or(0,|b|b.payload_bytes()),partner_rows:sector.partners,working_columns:if dense {sector.rank} else {prefix.saturating_add(guard_columns).min(sector.rank-usize::from(sector.owns_kernel))},owns_gamma_kernel:sector.owns_kernel,iterations:steps,dense_complete:dense,seed_rank,leakage,correction_projection_interval:options.correction_projection_interval,correction_projection_rounds:op.projected_rounds,correction_projection_skipped_rounds:op.correction_rounds-op.projected_rounds};

    Ok((entry,report,(op.output,op.scratch)))
}

fn is_adaptive(options:&SymmetryOptions)->bool {matches!(options.sectors,SectorPolicy::AdaptiveMirror|SectorPolicy::AdaptiveGroup)}
fn initial_prefixes(plan:&SectorPlan,requested:usize,adaptive:bool)->Vec<usize> {
    let active_sectors=plan.sectors.iter().filter(|s|s.rank>0).count();
    let initial=if adaptive {requested.div_ceil(active_sectors)+2} else {requested};
    let prefixes:Vec<_>=plan.sectors.iter().map(|s|initial.div_ceil(s.partners).min(s.rank)).collect();
    prefixes
}
fn allocate_guards(prefixes:&[usize],budget:Option<usize>,finite_dimension:usize)->Vec<usize> {
    let mut guards=vec![2;prefixes.len()];
    if let Some(budget)=budget {
        let budget=budget.min(finite_dimension);
        guards.fill(0);
        let total=prefixes.iter().sum::<usize>().max(1);
        let mut remainder=Vec::new();let mut used=0;
        for i in 0..guards.len() {if prefixes[i]>0 {guards[i]=budget.saturating_mul(prefixes[i])/total;used+=guards[i];remainder.push((budget.saturating_mul(prefixes[i])%total,i));}}
        remainder.sort_by(|a,b|b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        for &(_,i) in remainder.iter().take(budget.saturating_sub(used)) {guards[i]+=1;}
    }
    guards
}
fn grow_prefixes(slots:&[Option<(CachedSpectrum,SectorReport)>],prefixes:&mut[usize],requested:usize,adaptive:bool)->bool {
    let mut values:Vec<_>=slots.iter().flatten().flat_map(|(e,r)|e.eigenvalues.iter().flat_map(move |&x|std::iter::repeat_n(x,r.partner_rows))).collect();values.sort_by(f64::total_cmp);
    let cutoff=values.get(requested.saturating_sub(1)).copied().unwrap_or(f64::INFINITY);
    let mut grow=false;
    if adaptive {for (i,slot) in slots.iter().enumerate() {if let Some((entry,report))=slot {
        let tail=*entry.eigenvalues.last().unwrap();
        // A converged tail beyond the cut establishes empirical separation of
        // retained prefixes. It is not an inertia count or a rigorous bound on
        // an eigenvalue missed by the underlying iterative method.
        if report.prefix<report.dimension && tail<=cutoff+1e-7*cutoff.abs().max(1.0) {
            prefixes[i]=(report.prefix+report.prefix.div_ceil(2).max(2)).min(report.dimension);grow=true;
        }
    }}}
    grow
}
#[allow(clippy::too_many_arguments)]
fn finish_sectors<B:SpectralBackend+Clone>(operator:&mut ThetaOperator<B>,plan:&SectorPlan,
    slots:Vec<Option<(CachedSpectrum,SectorReport)>>,k:[f64;2],requested:usize,adaptive:bool,options:&SymmetryOptions,
    iterations:usize,rounds:usize,projection_workspace_allocations:usize,projection_workspace_payload_bytes:usize,
)->Result<SectorResult,String> {
    let mut reports=Vec::new();let mut all:Vec<(f64,Field2D)>=Vec::new();let mut tails=Vec::new();
    for (i,slot) in slots.into_iter().enumerate() {if let Some((entry,report))=slot {
        tails.push(entry.eigenvalues.last().copied());reports.push(report);
        for row in 1..plan.sectors[i].partners {
            crate::profile_scope!("symmetry.partners");
            let weights=plan.representation.matrix_unit_weights(plan.sectors[i].irrep,row,0);
            for (&value,field) in entry.eigenvalues.iter().zip(&entry.fields) {
                let mut output=Field2D::zeros(operator.grid());let mut scratch=Field2D::zeros(operator.grid());
                plan.representation.apply_weights(&plan.actions,&weights,field,&mut output,&mut scratch).map_err(|e|e.to_string())?;
                all.push((value,output));
            }
        }
        all.extend(entry.eigenvalues.into_iter().zip(entry.fields));
    }}
    let all_dense=reports.iter().all(|r|r.dense_complete);
    let computed=all.len();all.sort_by(|a,b|a.0.total_cmp(&b.0));all.truncate(requested);
    if all.len()!=requested {return Err("not enough finite modes for the requested output".into());}
    let (eigenvalues,fields):(Vec<_>,Vec<_>)=all.into_iter().unzip();let cert=certify_without_refinement(operator,&eigenvalues,&fields);
    let spectrum=CachedSpectrum{source_index:0,k,eigenvalues,fields,residuals:cert.residuals,absolute_residuals:cert.absolute_residuals,orthogonality_defect:cert.b_orthogonality_defect};
    if !spectrum.passes(options) {return Err("merged sector output failed original-pencil certification".into());}
    let report=CoverageReport{projection_workspace_allocations,projection_workspace_payload_bytes,evidence:if all_dense {"complete finite dense spectra"} else if adaptive {"adaptive prefixes with converged tails beyond the requested cut; empirical coverage, not inertia"} else {"conservative sector prefixes; iterative lowest-prefix coverage validated empirically, not by inertia"}.into(),finite_dimension:operator.grid().len(),sum_sector_dimensions:plan.sectors.iter().map(|s|s.rank*s.partners).sum(),requested:requested,computed,kernel_owners:plan.sectors.iter().filter(|s|s.owns_kernel).count(),allocation_rounds:rounds,tail_eigenvalues:tails,qualification_bytes:plan.actions.iter().map(|a|a.plan().storage_bytes()).sum(),sectors:reports};
    Ok(SectorResult{spectrum,iterations,report})
}
