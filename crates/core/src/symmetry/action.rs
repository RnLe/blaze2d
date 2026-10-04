//! Bloch pullbacks with explicit reciprocal folding and finite FFT support.
use num_complex::Complex64;
use serde::Serialize;
use crate::{backend::SpectralBuffer, field::cscalar, grid::Grid2D, polarization::Polarization};
use super::affine::{AffineOp, GroupError, Rational};

#[derive(Debug, Clone, PartialEq, thiserror::Error, Serialize)]
pub enum ActionError {
    #[error("invalid grid, wavevector, or tolerance")]
    InvalidInput,
    #[error("source and destination are not related by this Bloch action")]
    DifferentFiber,
    #[error("Fourier support is not closed: {index:?} maps to {mapped:?}")]
    FourierSupportNotClosed { index: [i64;2], mapped: [i64;2] },
    #[error("affine action does not permute the sampling nodes")]
    SamplingNotClosed,
    #[error("centered sampling offsets are not supported by this action plan")]
    UnsupportedSamplingOrigin,
    #[error("exact coordinate arithmetic overflow")]
    ArithmeticOverflow,
}
impl From<GroupError> for ActionError { fn from(_:GroupError)->Self {Self::ArithmeticOverflow} }

/// The order and Nyquist convention used by ThetaOperator, without aliasing.
pub fn fft_indices(n: usize) -> Vec<i64> { (0..n).map(|i|if i<=n/2 {i as i64} else {i as i64-n as i64}).collect() }

/// Permutations already derived, by operation and grid size: they do not depend on the
/// wavevector, and exact rational arithmetic per node dominated per-k setup.
static PERMUTATIONS: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<(AffineOp,usize,usize),std::sync::Arc<Vec<usize>>>>> =
    std::sync::LazyLock::new(Default::default);
const CACHED_PERMUTATIONS: usize = 64;

/// Exact pullback of sampling nodes, independent of any Fourier support.
/// Callers must budget its index array. This does not authorize derivatives.
pub(super) fn sampled_node_permutation(operation: AffineOp, grid: Grid2D) -> Result<Vec<usize>,ActionError> {
    if grid.centered {return Err(ActionError::UnsupportedSamplingOrigin);}
    if grid.nx==0 || grid.ny==0 || grid.nx>i32::MAX as usize || grid.ny>i32::MAX as usize || grid.nx.checked_mul(grid.ny).is_none() {return Err(ActionError::InvalidInput);}
    let key=(operation,grid.nx,grid.ny);
    if let Some(cached)=PERMUTATIONS.lock().unwrap().get(&key) {return Ok(cached.as_ref().clone());}
    let result=derive_node_permutation(operation,grid)?;
    let mut cache=PERMUTATIONS.lock().unwrap();
    if cache.len()<CACHED_PERMUTATIONS {cache.insert(key,std::sync::Arc::new(result.clone()));}
    Ok(result)
}

fn derive_node_permutation(operation: AffineOp, grid: Grid2D) -> Result<Vec<usize>,ActionError> {
    let wr=operation.reciprocal_matrix();let winv=[[wr[0][0],wr[1][0]],[wr[0][1],wr[1][1]]];
    let shifts=operation.translation();let mut result=vec![0;grid.len()];
    for j in 0..grid.ny {for i in 0..grid.nx {
        let xi=[Rational::new(i as i64,grid.nx as i64)?,Rational::new(j as i64,grid.ny as i64)?];
        let delta=[xi[0].checked_add(shifts[0].checked_scale(-1)?)?,xi[1].checked_add(shifts[1].checked_scale(-1)?)?];
        let mut src=[0usize;2];
        for d in 0..2 {
            let a=i32::try_from(winv[d][0]).map_err(|_|ActionError::ArithmeticOverflow)?;
            let b=i32::try_from(winv[d][1]).map_err(|_|ActionError::ArithmeticOverflow)?;
            let v=delta[0].checked_scale(a)?.checked_add(delta[1].checked_scale(b)?)?;
            let size=[grid.nx,grid.ny][d] as i128;let scaled=v.numerator() as i128*size;let denominator=v.denominator() as i128;
            if scaled%denominator!=0 {return Err(ActionError::SamplingNotClosed);}
            src[d]=(scaled/denominator).rem_euclid(size) as usize;
        }
        result[grid.idx(i,j)]=grid.idx(src[0],src[1]);
    }}
    Ok(result)
}

#[derive(Debug, Clone)]
pub struct BlochActionPlan {
    operation: AffineOp,
    source: [f64;2],
    destination: [f64;2],
    grid: Grid2D,
    polarization: Polarization,
    folding: [i64;2],
    fourier_destination: Vec<usize>,
    fourier_phase: Vec<Complex64>,
    spatial_source: Vec<usize>,
    spatial_phase: Vec<Complex64>,
}

impl BlochActionPlan {
    /// Reject any lost Fourier mode, including an aliased Nyquist mode. A valid
    /// sampled permutation alone is insufficient to authorize derivatives.
    pub fn new(operation: AffineOp, source: [f64;2], destination: [f64;2], grid: Grid2D, polarization: Polarization, tolerance: f64) -> Result<Self,ActionError> {
        if grid.centered {return Err(ActionError::UnsupportedSamplingOrigin);}
        if grid.nx==0 || grid.ny==0 || grid.nx>i32::MAX as usize || grid.ny>i32::MAX as usize ||
            !source.iter().chain(&destination).all(|x|x.is_finite() && x.abs()<1e6) ||
            !tolerance.is_finite() || !(0.0..1e-3).contains(&tolerance) {return Err(ActionError::InvalidInput);}
        let raw=operation.map_wavevector(source);let mut folding=[0;2];
        for i in 0..2 { let shift=raw[i]-destination[i]; if (shift-shift.round()).abs()>tolerance {return Err(ActionError::DifferentFiber);} folding[i]=shift.round() as i64; }
        let nx=fft_indices(grid.nx); let ny=fft_indices(grid.ny);
        // Position of FFT index v on an axis of n nodes (inverse of fft_indices): v itself or
        // v + n, for hi − n < v ≤ hi with hi = ⌊n/2⌋; anything else is outside the support.
        let position=|v:i64,n:usize|->Option<usize> {let (n,hi)=(n as i64,(n/2) as i64);(v<=hi && v>hi-n).then_some((if v>=0 {v} else {v+n}) as usize)};
        let wr=operation.reciprocal_matrix();let sign: i64=if operation.conjugates(){-1}else{1};
        // Physical Ez is polar; Hz is axial and odd under time reversal.
        let character=match polarization {Polarization::TM=>1.0,Polarization::TE=>(operation.determinant() as i64*sign) as f64};
        let tau=operation.translation().map(|t|t.value());
        let mut fourier_destination=vec![0;grid.len()];let mut fourier_phase=vec![Complex64::ZERO;grid.len()];
        for (j,&y) in ny.iter().enumerate() { for (i,&x) in nx.iter().enumerate() {
            let index=[x,y];let mapped=[sign*(wr[0][0]*x+wr[0][1]*y)+folding[0],sign*(wr[1][0]*x+wr[1][1]*y)+folding[1]];
            let pos=grid.idx(i,j);
            fourier_destination[pos]=match (position(mapped[0],grid.nx),position(mapped[1],grid.ny)) {
                (Some(a),Some(b))=>grid.idx(a,b),
                _=>return Err(ActionError::FourierSupportNotClosed{index,mapped}),
            };
            let q=operation.map_wavevector([x as f64+source[0],y as f64+source[1]]);
            fourier_phase[pos]=Complex64::from_polar(character,-std::f64::consts::TAU*(q[0]*tau[0]+q[1]*tau[1]));
        } }
        let spatial_source=sampled_node_permutation(operation,grid)?;let mut spatial_phase=vec![Complex64::ZERO;grid.len()];
        for j in 0..grid.ny {for i in 0..grid.nx {
            // Node fractions i/nx, j/ny: one correctly rounded division, as Rational::value.
            let xi=[i as f64/grid.nx as f64,j as f64/grid.ny as f64];
            let pos=grid.idx(i,j);
            let phase=-raw[0]*tau[0]-raw[1]*tau[1]+folding[0] as f64*xi[0]+folding[1] as f64*xi[1];
            spatial_phase[pos]=Complex64::from_polar(character,std::f64::consts::TAU*phase);
        } }
        Ok(Self {operation,source,destination,grid,polarization,folding,fourier_destination,fourier_phase,spatial_source,spatial_phase})
    }
    pub fn operation(&self)->AffineOp {self.operation}
    pub fn source(&self)->[f64;2] {self.source}
    pub fn destination(&self)->[f64;2] {self.destination}
    pub fn grid(&self)->Grid2D {self.grid}
    pub fn polarization(&self)->Polarization {self.polarization}
    pub fn folding(&self)->[i64;2] {self.folding}
    pub fn spatial_permutation(&self)->&[usize] {&self.spatial_source}
    pub fn spatial_phases(&self)->&[Complex64] {&self.spatial_phase}
    pub fn fourier_phases(&self)->&[Complex64] {&self.fourier_phase}
    pub fn fourier_permutation(&self)->&[usize] {&self.fourier_destination}
    /// Trace of a unitary sampled action, including fixed-node Bloch phases.
    pub fn spatial_trace(&self)->Complex64 {
        assert!(!self.operation.conjugates());
        self.spatial_source.iter().enumerate().filter_map(|(i,&j)|(i==j).then_some(self.spatial_phase[i])).sum()
    }
    pub fn storage_bytes(&self)->usize { self.grid.len()*(2*std::mem::size_of::<usize>()+2*std::mem::size_of::<Complex64>()) }
    pub fn apply_spatial<F:SpectralBuffer>(&self,input:&F,output:&mut F) {
        assert_eq!((input.grid().nx,input.grid().ny),(self.grid.nx,self.grid.ny));
        assert_eq!((output.grid().nx,output.grid().ny),(self.grid.nx,self.grid.ny));
        crate::profiler::count("symmetry.transform.elements",self.grid.len());
        for (j,dst) in output.as_mut_slice().iter_mut().enumerate() {
            let v=input.as_slice()[self.spatial_source[j]];let v=if self.operation.conjugates(){v.conj()}else{v};
            let p=self.spatial_phase[j];*dst=v*cscalar::<F::Real>(p.re,p.im);
        }
    }
    pub fn apply_fourier<F:SpectralBuffer>(&self,input:&F,output:&mut F) {
        assert_eq!(input.len(),self.grid.len());assert_eq!(output.len(),self.grid.len());
        for (i,&j) in self.fourier_destination.iter().enumerate() {
            let v=input.as_slice()[i];let v=if self.operation.conjugates(){v.conj()}else{v};let p=self.fourier_phase[i];
            output.as_mut_slice()[j]=v*cscalar::<F::Real>(p.re,p.im);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*; use super::super::affine::PlaneGroup;
    #[test]
    fn refuses_hexagonal_box_and_wrong_boundary_aliases() {
        let rotation=PlaneGroup::P6.group().operations[1];
        assert!(matches!(BlochActionPlan::new(rotation,[0.0;2],[0.0;2],Grid2D::new(7,7,1.0,1.0),Polarization::TM,1e-12),Err(ActionError::FourierSupportNotClosed{..})));
        let mx=AffineOp::new([[-1,0],[0,1]],[Rational::ZERO;2]).unwrap();
        assert!(BlochActionPlan::new(mx,[0.5,0.0],[0.5,0.0],Grid2D::new(8,8,1.0,1.0),Polarization::TM,1e-12).is_err());
        let plan=BlochActionPlan::new(mx,[-0.5,0.0],[-0.5,0.0],Grid2D::new(8,8,1.0,1.0),Polarization::TM,1e-12).unwrap();
        assert_eq!(plan.folding(),[1,0]);
    }
    #[test]
    fn pt_preserves_even_support_but_time_reversal_alone_does_not() {
        let inversion=PlaneGroup::P2.group().operations[1].with_conjugation(true);let grid=Grid2D::new(8,6,1.0,1.0);let k=[0.23,0.11];
        assert!(BlochActionPlan::new(inversion,k,k,grid,Polarization::TM,1e-12).is_ok());
        assert!(BlochActionPlan::new(AffineOp::IDENTITY.with_conjugation(true),k,[-k[0],-k[1]],grid,Polarization::TM,1e-12).is_err());
    }
}
