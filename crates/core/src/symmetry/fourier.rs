//! Unitary Fourier-coordinate representation of the existing eigensolver.
use std::{sync::{Arc,atomic::{AtomicUsize,Ordering}},rc::Rc,cell::Cell};
use num_complex::Complex64 as C;
use serde::Serialize;
use crate::{backend::{SpectralBackend,SpectralBuffer},field::{Field2D,Real,cscalar},grid::Grid2D,lattice::Lattice2D,operators::{LinearOperator,ThetaOperator},preconditioners::OperatorPreconditioner,eigensolver::{Eigensolver,EigensolverConfig,refine::certify_without_refinement}};
use super::reuse::{CachedSpectrum,SymmetryOptions};

pub(crate) mod sector;
pub(crate) mod projected;

#[derive(Debug,Clone,Copy,Default,PartialEq,Eq,Serialize)]
#[serde(rename_all="snake_case")]
pub enum Arithmetic {#[default] SpatialComplex, FourierComplex, PtComplex, PtRealStorage, PtRealVectors, PtRealGram, PtRealDense, PtRealTm, PtRealTe, PtRealReferenceVectors}
#[derive(Debug,Clone,Serialize)]
pub struct CoordinateReport {
    pub representation:Arithmetic,
    pub physical_coefficients:usize,
    pub bytes_per_column:usize,
    pub peak_solver_vector_bytes:usize,
    pub conversion_scratch_bytes:usize,
    pub maximum_real_subspace_leakage:f64,
    /// Relative correction applied before the final sector-local Ritz rotation.
    #[serde(skip_serializing_if="Option::is_none")]
    pub final_sector_projection_defect:Option<f64>,
    pub qualification:Option<super::qualification::QualificationReport>,
}
pub struct CoordinateResult {pub spectrum:CachedSpectrum,pub iterations:usize,pub report:CoordinateReport}

/// Immutable qualification and gauge, shared by all prefixes at one k point.
pub(crate) struct CoordinateSetup {
    arithmetic: Arithmetic,
    gauge: Arc<Vec<C>>,
    qualification: Option<super::qualification::QualificationReport>,
}
impl CoordinateSetup {
    pub(crate) fn complex(n:usize)->Self {
        Self{arithmetic:Arithmetic::FourierComplex,gauge:Arc::new(vec![C::ONE;n]),qualification:None}
    }
    pub(crate) fn real(&self)->bool { !matches!(self.arithmetic,Arithmetic::SpatialComplex|Arithmetic::FourierComplex) }
    pub(crate) fn qualify<B:SpectralBackend+Clone>(operator:&mut ThetaOperator<B>,lattice:&Lattice2D,k:[f64;2],options:&SymmetryOptions)->Result<Self,String> {
        if std::mem::size_of::<B::Real>()!=8 {return Err("Fourier-coordinate experiments currently require f64".into());}
        let mut setup=Self::complex(operator.grid().len());
        if matches!(options.arithmetic,Arithmetic::SpatialComplex|Arithmetic::FourierComplex) {return Ok(setup);}
        let pt=options.pt_operation.unwrap_or_else(||super::affine::PlaneGroup::P2.group().operations[1].with_conjugation(true));
        if !pt.conjugates() || pt.matrix()!=[[-1,0],[0,-1]] {return Err("real scalar packing requires antiunitary inversion".into());}
        let plan=super::action::BlochActionPlan::new(pt,k,k,operator.grid(),operator.polarization(),1e-12).map_err(|e|e.to_string())?;
        if plan.fourier_permutation().iter().enumerate().any(|(i,&j)|i!=j) {return Err("PT action is not diagonal in this Fourier representation".into());}
        let gauge=plan.fourier_phases().iter().map(|z|C::from_polar(1.0,0.5*z.arg())).collect();
        let mut other=ThetaOperator::new(operator.backend().clone(),operator.dielectric().clone(),operator.polarization(),operator.bloch());
        let qualified=super::qualification::qualify(plan,lattice,operator,&mut other,options.qualification_tolerance).map_err(|r|format!("PT qualification failed: {:?}",r.rejection_reasons))?;
        setup.arithmetic=options.arithmetic;setup.gauge=Arc::new(gauge);setup.qualification=Some(qualified.report().clone());
        Ok(setup)
    }
}

#[derive(Default)]
struct AllocationLedger {live:AtomicUsize,peak:AtomicUsize}
impl AllocationLedger {
    fn add(&self,n:usize) {let live=self.live.fetch_add(n,Ordering::Relaxed)+n;self.peak.fetch_max(live,Ordering::Relaxed);}
}
struct CoordinateBuffer {field:Field2D,ledger:Arc<AllocationLedger>}
impl Clone for CoordinateBuffer {fn clone(&self)->Self {self.ledger.add(self.field.len()*16);Self{field:self.field.clone(),ledger:self.ledger.clone()}}}
impl Drop for CoordinateBuffer {fn drop(&mut self) {self.ledger.live.fetch_sub(self.field.len()*16,Ordering::Relaxed);}}
impl SpectralBuffer for CoordinateBuffer {
    type Real=f64;
    fn len(&self)->usize {self.field.len()}
    fn grid(&self)->Grid2D {self.field.grid()}
    fn as_slice(&self)->&[C] {self.field.as_slice()}
    fn as_mut_slice(&mut self)->&mut[C] {self.field.as_mut_slice()}
}
#[derive(Clone)]
struct CoordinateBackend {ledger:Arc<AllocationLedger>,nreal:usize,real_space:bool,packed:bool,real_vectors:bool,real_gram:bool,real_dense:bool}
impl SpectralBackend for CoordinateBackend {
    type Real=f64;type Buffer=CoordinateBuffer;
    fn alloc_field(&self,grid:Grid2D)->CoordinateBuffer {self.ledger.add(grid.len()*16);CoordinateBuffer{field:Field2D::zeros(grid),ledger:self.ledger.clone()}}
    fn forward_fft_2d(&self,_:&mut CoordinateBuffer) {unreachable!("Fourier coordinates use the physical FFT backend")}
    fn inverse_fft_2d(&self,_:&mut CoordinateBuffer) {unreachable!("Fourier coordinates use the physical FFT backend")}
    fn real_coefficients(&self)->bool {self.real_space}
    fn prefer_backend_linear_combinations(&self)->bool {self.real_vectors}
    fn gram_rows(&self,buffer:&CoordinateBuffer)->usize {if self.real_space {self.nreal}else{buffer.len()}}
    fn dense_eigen(&self,matrix:&[C],n:usize)->crate::eigensolver::dense::DenseEigenResult {
        if self.real_dense {crate::eigensolver::dense::solve_real_symmetric_eigen(matrix,n)} else if self.real_space {crate::eigensolver::dense::solve_real_via_complex(matrix,n)} else {crate::eigensolver::dense::solve_hermitian_eigen(matrix,n)}
    }
    fn scale(&self,a:C,v:&mut CoordinateBuffer) {if self.real_space {assert!(a.im.abs()<1e-10*a.norm().max(1.0));for x in v.as_mut_slice() {*x*=a.re;}} else {for x in v.as_mut_slice() {*x*=a;}}}
    fn axpy(&self,a:C,x:&CoordinateBuffer,y:&mut CoordinateBuffer) {if self.real_space {assert!(a.im.abs()<1e-10*a.norm().max(1.0));for (dst,src) in y.as_mut_slice().iter_mut().zip(x.as_slice()) {*dst+=a.re*src;}} else {for (dst,src) in y.as_mut_slice().iter_mut().zip(x.as_slice()) {*dst+=a*src;}}}
    fn dot(&self,x:&CoordinateBuffer,y:&CoordinateBuffer)->C {
        if self.real_space {C::new((0..self.nreal).map(|i|scalar(x.as_slice(),i,self.packed)*scalar(y.as_slice(),i,self.packed)).sum(),0.0)}
        else {x.as_slice().iter().zip(y.as_slice()).map(|(a,b)|a.conj()*b).sum()}
    }
    fn linear_combinations(&self,q:&[CoordinateBuffer],coefficients:&[C],m:usize)->Vec<CoordinateBuffer> {
        let r=q.len();if r==0 || m==0 {return vec![];}
        assert_eq!(coefficients.len(),r*m);
        if !self.real_vectors {
            let mut outputs=Vec::new();for j in 0..m {let mut out=self.alloc_field(q[0].grid());for i in 0..r {self.axpy(coefficients[i+r*j],&q[i],&mut out);}outputs.push(out);}return outputs;
        }
        assert!(coefficients.iter().all(|z|z.im==0.0));
        crate::profile_scope!("real_vector.linear_combinations");
        crate::profiler::count("real_vector.multiply_terms",self.nreal*r*m);
        #[cfg(feature="native-linalg")]
        {
            let a=faer::Mat::<f64>::from_fn(self.nreal,r,|i,j|scalar(q[j].as_slice(),i,self.packed));
            let b=faer::Mat::<f64>::from_fn(r,m,|i,j|coefficients[i+r*j].re);let result=a*b;
            (0..m).map(|j|{let mut out=self.alloc_field(q[0].grid());for i in 0..self.nreal {set_scalar(out.as_mut_slice(),i,result[(i,j)],self.packed);}out}).collect()
        }
        #[cfg(feature="wasm-linalg")]
        {(0..m).map(|j|{let mut out=self.alloc_field(q[0].grid());for i in 0..self.nreal {let value=(0..r).map(|p|scalar(q[p].as_slice(),i,self.packed)*coefficients[p+r*j].re).sum();set_scalar(out.as_mut_slice(),i,value,self.packed);}out}).collect()}
    }
    fn gram_matrix(&self,x:&[CoordinateBuffer],y:&[CoordinateBuffer])->Vec<C> {
        // The storage experiment expands real coordinates into the same complex
        // dense Gram kernel. Real GEMM is a separately measured change.
        let (p,q)=(x.len(),y.len());
        if !self.real_space {return (0..p*q).map(|i|self.dot(&x[i/q],&y[i%q])).collect();}
        #[cfg(feature="native-linalg")]
        {
            if self.real_gram {
                crate::profile_scope!("real_gram");crate::profiler::count("real_gram.multiply_terms",self.nreal*p*q);
                let a=faer::Mat::<f64>::from_fn(self.nreal,p,|i,j|scalar(x[j].as_slice(),i,self.packed));
                let b=faer::Mat::<f64>::from_fn(self.nreal,q,|i,j|scalar(y[j].as_slice(),i,self.packed));
                let product=a.transpose()*b;return (0..p*q).map(|i|C::new(product[(i/q,i%q)],0.0)).collect();
            }
            let a=faer::Mat::<faer::c64>::from_fn(self.nreal,p,|i,j|C::new(scalar(x[j].as_slice(),i,self.packed),0.0));
            let b=faer::Mat::<faer::c64>::from_fn(self.nreal,q,|i,j|C::new(scalar(y[j].as_slice(),i,self.packed),0.0));
            let product=a.adjoint()*b;
            (0..p*q).map(|i|C::new(product[(i/q,i%q)].re,0.0)).collect()
        }
        #[cfg(feature="wasm-linalg")]
        {(0..p*q).map(|i|self.dot(&x[i/q],&y[i%q])).collect()}
    }
}

// Each complex storage lane can hold two independent real coordinates. No
// unsafe casts and no interpretation as a complex vector space are involved.
fn scalar(v:&[C],i:usize,packed:bool)->f64 {if packed {if i%2==0 {v[i/2].re}else{v[i/2].im}} else {v[i].re}}
fn set_scalar(v:&mut[C],i:usize,value:f64,packed:bool) {if packed {if i%2==0 {v[i/2].re=value}else{v[i/2].im=value}} else {v[i]=C::new(value,0.0);}}
struct Conversion<B:SpectralBackend+Clone> {backend:B,grid:Grid2D,gauge:Arc<Vec<C>>,real_space:bool,packed:bool,leakage:Rc<Cell<f64>>}
impl<B:SpectralBackend+Clone> Clone for Conversion<B> {fn clone(&self)->Self {Self{backend:self.backend.clone(),grid:self.grid,gauge:self.gauge.clone(),real_space:self.real_space,packed:self.packed,leakage:self.leakage.clone()}}}
impl<B:SpectralBackend+Clone> Conversion<B> {
    fn coordinate_grid(&self)->Grid2D {if self.packed {Grid2D::new(self.grid.len().div_ceil(2),1,1.0,1.0)} else {self.grid}}
    fn unpack_fourier(&self,coordinates:&[C],out:&mut B::Buffer) {
        for (i,dst) in out.as_mut_slice().iter_mut().enumerate() {let value=if self.real_space {C::new(scalar(coordinates,i,self.packed),0.0)} else {coordinates[i]};let z=self.gauge[i]*value;*dst=cscalar::<B::Real>(z.re,z.im);}
    }
    fn spatial(&self,coordinates:&[C],out:&mut B::Buffer) {
        self.unpack_fourier(coordinates,out);
        self.backend.inverse_fft_2d(out);self.backend.scale(C::new((self.grid.len() as f64).sqrt(),0.0),out);
    }
    fn fourier(&self,spatial:&mut B::Buffer,out:&mut[C],check_real:bool) {
        self.backend.forward_fft_2d(spatial);self.pack_fourier(spatial,out,(self.grid.len() as f64).sqrt().recip(),check_real);
    }
    fn pack_fourier(&self,spatial:&B::Buffer,out:&mut[C],factor:f64,check_real:bool) {
        let mut imaginary=0.0;let mut norm=0.0;out.fill(C::ZERO);
        for (i,src) in spatial.as_slice().iter().enumerate() {let z=C::new(src.re.to_accum()*factor,src.im.to_accum()*factor)*self.gauge[i].conj();
            if self.real_space {set_scalar(out,i,z.re,self.packed);imaginary+=z.im*z.im;norm+=z.norm_sqr();} else {out[i]=z;}
        }
        if self.real_space && check_real {let defect=(imaginary/norm.max(f64::MIN_POSITIVE)).sqrt();self.leakage.set(if defect.is_finite() {self.leakage.get().max(defect)}else{f64::INFINITY});}
    }
    fn promote(&self,buffer:&B::Buffer)->Field2D {Field2D::from_vec(self.grid,buffer.as_slice().iter().map(|z|C::new(z.re.to_accum(),z.im.to_accum())).collect())}
}
struct FourierOperator<'a,B:SpectralBackend+Clone> {
    full:&'a mut ThetaOperator<B>,backend:CoordinateBackend,conversion:Conversion<B>,input:B::Buffer,output:B::Buffer,direct_tm:bool,direct_te:bool,
    sector:Option<sector::SectorProjection<'a>>,correction_rounds:usize,projected_rounds:usize,
}
impl<B:SpectralBackend+Clone> LinearOperator<CoordinateBackend> for FourierOperator<'_,B> {
    fn apply(&mut self,x:&CoordinateBuffer,y:&mut CoordinateBuffer) {
        if self.direct_tm {
            crate::profile_scope!("apply_tm_fourier_diagonal");
            crate::profiler::count("operator_A.columns",1);
            crate::profiler::count("operator_A.elements",self.backend.nreal);
            y.as_mut_slice().fill(C::ZERO);
            for (i,&weight) in self.full.k_plus_g_squares().iter().enumerate() {
                set_scalar(y.as_mut_slice(),i,scalar(x.as_slice(),i,self.backend.packed)*weight,self.backend.packed);
            }
        } else if self.direct_te {
            self.conversion.unpack_fourier(x.as_slice(),&mut self.input);
            self.full.apply_fourier(&self.input,&mut self.output);
            self.conversion.pack_fourier(&self.output,y.as_mut_slice(),1.0,true);
        } else {
            self.conversion.spatial(x.as_slice(),&mut self.input);self.full.apply(&self.input,&mut self.output);self.conversion.fourier(&mut self.output,y.as_mut_slice(),true);
        }
    }
    fn apply_mass(&mut self,x:&CoordinateBuffer,y:&mut CoordinateBuffer) {
        if self.direct_te {
            crate::profiler::count("operator_B.columns",1);crate::profiler::count("operator_B.elements",self.backend.nreal);
            y.as_mut_slice().copy_from_slice(x.as_slice());
        } else {self.conversion.spatial(x.as_slice(),&mut self.input);self.full.apply_mass(&self.input,&mut self.output);self.conversion.fourier(&mut self.output,y.as_mut_slice(),true);}
    }
    fn alloc_field(&self)->CoordinateBuffer {self.backend.alloc_field(self.conversion.coordinate_grid())}
    fn backend(&self)->&CoordinateBackend {&self.backend}
    fn backend_mut(&mut self)->&mut CoordinateBackend {&mut self.backend}
    fn grid(&self)->Grid2D {self.conversion.coordinate_grid()}
    fn subspace_dimension(&self)->usize {self.sector.as_ref().map_or(self.full.grid().len(),|s|s.dimension)}
    fn owns_gamma_kernel(&self)->bool {self.sector.as_ref().is_none_or(|s|s.owns_gamma_kernel)}
    fn constrain(&mut self,v:&mut CoordinateBuffer) {
        if let Some(sector)=&self.sector {
            self.conversion.unpack_fourier(v.as_slice(),&mut self.input);
            sector.project(&self.input,&mut self.output);
            if self.conversion.real_space {
                // An exactly annihilated axis can leave pure-imaginary roundoff.
                // Projector leakage is relative to its input, not that near-zero
                // output. The complete gauged projector was checked beforehand.
                let norm=self.input.as_slice().iter().map(|z|z.re.to_accum().powi(2)+z.im.to_accum().powi(2)).sum::<f64>();
                let imaginary=self.output.as_slice().iter().zip(self.conversion.gauge.iter()).map(|(z,g)| {
                    let z=C::new(z.re.to_accum(),z.im.to_accum())*g.conj();z.im*z.im
                }).sum::<f64>();
                let defect=(imaginary/norm.max(f64::MIN_POSITIVE)).sqrt();
                self.conversion.leakage.set(if defect.is_finite() {self.conversion.leakage.get().max(defect)} else {f64::INFINITY});
            }
            self.conversion.pack_fourier(&self.output,v.as_mut_slice(),1.0,false);
        }
    }
    fn constrain_corrections(&mut self,vectors:&mut[CoordinateBuffer]) {
        let Some(sector)=&self.sector else {return;};
        if vectors.is_empty() {return;}
        if self.correction_rounds%sector.correction_projection_interval==0 {
            for v in vectors {self.constrain(v);}
            self.projected_rounds+=1;
            crate::profiler::count("symmetry.correction_projection_rounds",1);
        } else {crate::profiler::count("symmetry.skipped_correction_projection_rounds",1);}
        self.correction_rounds+=1;
    }
    fn bloch(&self)->[f64;2] {self.full.bloch()}
    fn fill_gamma_kernel(&self,v:&mut CoordinateBuffer) {v.as_mut_slice().fill(C::ZERO);v.as_mut_slice()[0]=C::ONE;}
    fn seed_initial_vector(&mut self,v:&mut CoordinateBuffer,phase:f64) {
        crate::eigensolver::initialization::seed_random_vector(self.input.as_mut_slice(),phase);
        if self.conversion.real_space {
            v.as_mut_slice().fill(C::ZERO);
            for (i,z) in self.input.as_slice().iter().enumerate() {set_scalar(v.as_mut_slice(),i,z.re.to_accum(),self.conversion.packed);}
        } else {self.conversion.fourier(&mut self.input,v.as_mut_slice(),false);}
    }
}
struct FourierPreconditioner<'a,B:SpectralBackend+Clone> {inner:&'a mut dyn OperatorPreconditioner<B>,conversion:Conversion<B>,scratch:B::Buffer,direct_diagonal:bool,direct_fourier:bool}
impl<B:SpectralBackend+Clone> OperatorPreconditioner<CoordinateBackend> for FourierPreconditioner<'_,B> {
    fn apply(&mut self,_:&CoordinateBackend,v:&mut CoordinateBuffer) {
        if self.direct_diagonal {
            if let Some(diagonal)=self.inner.fourier_diagonal() {
                crate::profile_scope!("preconditioner_fourier_diagonal");
                assert_eq!(diagonal.len(),self.conversion.grid.len());
                for (i,&weight) in diagonal.iter().enumerate() {
                    let value=scalar(v.as_slice(),i,self.conversion.packed)*weight;
                    set_scalar(v.as_mut_slice(),i,value,self.conversion.packed);
                }
                return;
            }
        }
        if self.direct_fourier {
            self.conversion.unpack_fourier(v.as_slice(),&mut self.scratch);
            if self.inner.apply_fourier(&self.conversion.backend,&mut self.scratch) {
                self.conversion.pack_fourier(&self.scratch,v.as_mut_slice(),1.0,true);return;
            }
        }
        self.conversion.spatial(v.as_slice(),&mut self.scratch);self.inner.apply(&self.conversion.backend,&mut self.scratch);self.conversion.fourier(&mut self.scratch,v.as_mut_slice(),true);
    }
}

pub fn solve_fourier<B:SpectralBackend+Clone>(operator:&mut ThetaOperator<B>,lattice:&Lattice2D,k:[f64;2],config:&EigensolverConfig,warm:Option<&[Field2D]>,options:&SymmetryOptions,preconditioner:Option<&mut dyn OperatorPreconditioner<B>>)->Result<CoordinateResult,String> {
    crate::profile_scope!("symmetry.fourier_coordinates");
    let setup=CoordinateSetup::qualify(operator,lattice,k,options)?;
    solve_prepared(operator,k,config,warm,options,preconditioner,&setup,None,false).map(|r|r.0)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_prepared<B:SpectralBackend+Clone>(operator:&mut ThetaOperator<B>,k:[f64;2],config:&EigensolverConfig,warm:Option<&[Field2D]>,options:&SymmetryOptions,preconditioner:Option<&mut dyn OperatorPreconditioner<B>>,setup:&CoordinateSetup,sector:Option<sector::SectorProjection<'_>>,complete:bool)->Result<(CoordinateResult,usize,usize),String> {
    let grid=operator.grid();let n=grid.len();
    let arithmetic=setup.arithmetic;let real_space=setup.real();let packed=real_space && arithmetic!=Arithmetic::PtComplex;
    let conversion=Conversion{backend:operator.backend().clone(),grid,gauge:setup.gauge.clone(),real_space,packed,leakage:Rc::new(Cell::new(0.0))};
    let backend=CoordinateBackend{ledger:Arc::default(),nreal:n,real_space,packed,real_vectors:matches!(arithmetic,Arithmetic::PtRealVectors|Arithmetic::PtRealGram|Arithmetic::PtRealDense|Arithmetic::PtRealTm|Arithmetic::PtRealTe),real_gram:matches!(arithmetic,Arithmetic::PtRealGram|Arithmetic::PtRealDense|Arithmetic::PtRealTm|Arithmetic::PtRealTe|Arithmetic::PtRealReferenceVectors),real_dense:matches!(arithmetic,Arithmetic::PtRealDense|Arithmetic::PtRealTm|Arithmetic::PtRealTe|Arithmetic::PtRealReferenceVectors)};
    let mut scratch=operator.alloc_field();
    if warm.is_some_and(|fields|fields.iter().any(|f|f.len()!=n || f.grid().nx!=grid.nx || f.grid().ny!=grid.ny)) {return Err("warm-start shape differs from physical Fourier grid".into());}
    let warm=warm.map(|fields| {
        let mut converted=Vec::new();
        for field in fields {
            for (dst,src) in scratch.as_mut_slice().iter_mut().zip(field.as_slice()) {*dst=cscalar::<B::Real>(src.re,src.im);}
            let mut out=Field2D::zeros(conversion.coordinate_grid());conversion.fourier(&mut scratch,out.as_mut_slice(),false);converted.push(out);
            if real_space {
                // Both real and imaginary parts can carry independent members
                // of a degenerate eigenspace. Initialization handles their rank.
                let mut imaginary=Field2D::zeros(conversion.coordinate_grid());let factor=(n as f64).sqrt().recip();
                for (i,z) in scratch.as_slice().iter().enumerate() {let value=C::new(z.re.to_accum(),z.im.to_accum())*factor*conversion.gauge[i].conj();set_scalar(imaginary.as_mut_slice(),i,value.im,packed);}
                converted.push(imaginary);
            }
        }
        converted
    });
    let direct_tm=matches!(arithmetic,Arithmetic::PtRealTm|Arithmetic::PtRealTe|Arithmetic::PtRealReferenceVectors) && operator.polarization()==crate::polarization::Polarization::TM;
    let direct_te=matches!(arithmetic,Arithmetic::PtRealTe|Arithmetic::PtRealReferenceVectors) && operator.polarization()==crate::polarization::Polarization::TE;
    let mut preconditioner=preconditioner.map(|p|FourierPreconditioner{inner:p,conversion:conversion.clone(),scratch:operator.alloc_field(),direct_diagonal:direct_tm||direct_te,direct_fourier:direct_te});
    let input=operator.alloc_field();let output=operator.alloc_field();
    let mut op=FourierOperator{full:operator,backend:backend.clone(),conversion:conversion.clone(),input,output,direct_tm,direct_te,sector,correction_rounds:0,projected_rounds:0};
    let (mut eigenvalues,mut coordinates,iterations,converged)=if complete {
        let (values,fields)=sector::dense_complete(&mut op)?;(values,fields,0,true)
    } else {
        let mut solver=Eigensolver::new(&mut op,config.clone(),preconditioner.as_mut().map(|p|p as &mut dyn OperatorPreconditioner<CoordinateBackend>),warm.as_deref());
        let result=solver.solve();let coordinates=solver.all_eigenvectors();
        (result.eigenvalues,coordinates,result.iterations,result.converged)
    };
    eigenvalues.truncate(config.n_bands);
    coordinates.truncate(config.n_bands);
    let final_sector_projection_defect=if op.sector.is_some() {
        // Near-dependent warm directions can amplify projector roundoff during
        // normalization. Restore the qualified subspace before the final Ritz
        // rotation; fresh physical residual and orthogonality gates still apply.
        let mut maximum:f64=0.0;
        for field in &mut coordinates {
            let mut v=op.alloc_field();v.as_mut_slice().copy_from_slice(field.as_slice());
            let norm=op.backend.dot(&v,&v).re.max(f64::MIN_POSITIVE);
            op.constrain(&mut v);
            let mut difference=v.clone();
            for (dst,src) in difference.as_mut_slice().iter_mut().zip(field.as_slice()) {*dst-=src;}
            let defect=(op.backend.dot(&difference,&difference).re.max(0.0)/norm).sqrt();
            if !defect.is_finite() {return Err("nonfinite final sector projection defect".into());}
            maximum=maximum.max(defect);
            field.as_mut_slice().copy_from_slice(v.as_slice());
        }
        // Refine in the selected real/complex algebra. A later complex rotation
        // in physical coordinates could destroy the exported PT gauge.
        crate::eigensolver::refine::rayleigh_ritz_certify(&mut op,&mut eigenvalues,&mut coordinates);
        if op.owns_gamma_kernel() && k==[0.0;2] && eigenvalues.first().is_some_and(|v|v.abs()<1e-8) {eigenvalues[0]=0.0;}
        Some(maximum)
    } else {None};
    let mut fields=Vec::new();for v in coordinates {conversion.spatial(v.as_slice(),&mut scratch);fields.push(conversion.promote(&scratch));}
    fields.truncate(config.n_bands);
    let cert=certify_without_refinement(op.full,&eigenvalues,&fields);
    let spectrum=CachedSpectrum{source_index:0,k,eigenvalues,fields,residuals:cert.residuals,absolute_residuals:cert.absolute_residuals,orthogonality_defect:cert.b_orthogonality_defect};
    if !conversion.leakage.get().is_finite() || conversion.leakage.get()>options.qualification_tolerance {return Err(format!("PT real subspace leakage: {:e}",conversion.leakage.get()));}
    if !converged || !spectrum.passes(options) || spectrum.fields.len()!=config.n_bands {return Err("Fourier-coordinate solve failed fresh physical quality gates".into());}
    let report=CoordinateReport{representation:if real_space {arithmetic} else {Arithmetic::FourierComplex},physical_coefficients:grid.len(),bytes_per_column:conversion.coordinate_grid().len()*16,peak_solver_vector_bytes:backend.ledger.peak.load(Ordering::Relaxed),maximum_real_subspace_leakage:conversion.leakage.get(),final_sector_projection_defect,qualification:setup.qualification.clone(),conversion_scratch_bytes:grid.len()*16*(3+usize::from(preconditioner.is_some()))};
    Ok((CoordinateResult{spectrum,iterations,report},op.projected_rounds,op.correction_rounds-op.projected_rounds))
}
