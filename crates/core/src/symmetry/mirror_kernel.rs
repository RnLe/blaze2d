//! Reduced TM reference kernel for an origin mirror normal to the second axis.
//!
//! Tangential FFTs operate on the retained rows only. Normal differentiation
//! uses a reduced dense parity matrix, so this is not an asymptotically fast
//! cosine/sine transform. The dense cost is measured separately.
use num_complex::Complex64 as C;
use crate::{backend::{SpectralBackend,SpectralBuffer},field::{Real,cscalar},grid::Grid2D,operators::ThetaOperator,polarization::Polarization};
use super::reduced::OrbitBasis;

pub struct MirrorYKernel<B:SpectralBackend> {
    nx:usize,ny:usize,start:usize,parity:f64,rows:usize,mapping:Vec<(usize,C)>,normal_laplacian:Vec<f64>,tangent_symbol:Vec<f64>,mass:Vec<f64>,
    input:Vec<C>,output:Vec<C>,row:B::Buffer,
}
impl<B:SpectralBackend> MirrorYKernel<B> {
    pub fn new(operator:&ThetaOperator<B>,basis:&OrbitBasis)->Result<Self,String> {
        if operator.polarization()!=Polarization::TM {return Err("the reduced TM mirror kernel requires TM polarization".into());}
        Self::new_layout(operator,basis)
    }
    fn new_layout(operator:&ThetaOperator<B>,basis:&OrbitBasis)->Result<Self,String> {
        let grid=operator.grid();
        if (basis.grid().nx,basis.grid().ny,basis.grid().centered)!=(grid.nx,grid.ny,grid.centered) {return Err("basis grid differs from operator".into());}
        if grid.centered || grid.ny%2==0 || grid.ny<3 {return Err("the reduced mirror kernel requires an odd node-centered normal axis".into());}
        let mut parity=None;
        for column in basis.columns() {
            if column.len()==1 {
                if column[0].0/grid.nx!=0 || (column[0].1.norm()-1.0).abs()>1e-10 {return Err("basis is not an origin mirror range".into());}
                if parity==Some(-1.0) {return Err("mixed mirror parities".into());}parity=Some(1.0);
            } else if column.len()==2 {
                let (a,x)=column[0];let (b,y)=column[1];
                if a%grid.nx!=b%grid.nx || a/grid.nx+b/grid.nx!=grid.ny || (x.norm()-std::f64::consts::FRAC_1_SQRT_2).abs()>1e-10 || (y.norm()-x.norm()).abs()>1e-10 {return Err("basis is not an origin mirror range".into());}
                let ratio=y/x;let sign=if (ratio-C::ONE).norm()<1e-10 {1.0} else if (ratio+C::ONE).norm()<1e-10 {-1.0} else {return Err("Bloch-folded and nonsymmorphic mirror kernels are not supported".into());};
                if parity.is_some_and(|p|p!=sign) {return Err("mixed mirror parities".into());}parity=Some(sign);
            } else {return Err("kernel requires rank-one mirror orbits".into());}
        }
        let parity=parity.ok_or("empty mirror basis")?;let start=usize::from(parity<0.0);let rows=grid.ny/2+1-start;
        if basis.rank()!=grid.nx*rows {return Err("incomplete mirror range".into());}
        let mut mapping=Vec::new();let mut seen=vec![false;basis.rank()];let mut mass=vec![0.0;basis.rank()];
        for column in basis.columns() {
            let &(index,coefficient)=column.iter().min_by_key(|(i,_)|i/grid.nx).unwrap();
            let normal=index/grid.nx;
            if normal<start || normal>grid.ny/2 {return Err("mirror orbit has no canonical positive node".into());}
            let canonical=(normal-start)*grid.nx+index%grid.nx;
            if seen[canonical] {return Err("duplicate mirror orbit".into());}seen[canonical]=true;
            let phase=coefficient*if normal==0 {1.0}else{std::f64::consts::SQRT_2};
            mapping.push((canonical,phase));
            mass[canonical]=column.iter().map(|&(i,z)|z.norm_sqr()*operator.dielectric().eps()[i]).sum();
        }
        let symbol=operator.k_plus_g_squares();let scale=symbol.iter().copied().fold(1.0,f64::max);
        let tangent_symbol=symbol[..grid.nx].to_vec();
        let normal_symbol:Vec<_>=(0..grid.ny).map(|j|symbol[grid.idx(0,j)]-symbol[0]).collect();
        for j in 0..grid.ny {for i in 0..grid.nx {
            if (symbol[grid.idx(i,j)]-tangent_symbol[i]-normal_symbol[j]).abs()>1e-12*scale {return Err("nonseparable reciprocal metric".into());}
        }}
        for j in 1..grid.ny {if (normal_symbol[j]-normal_symbol[grid.ny-j]).abs()>1e-12*scale {return Err("nonzero Bloch component normal to the mirror".into());}}
        let convolution:Vec<_>=(0..grid.ny).map(|delta|normal_symbol.iter().enumerate().map(|(j,&weight)|weight*(std::f64::consts::TAU*j as f64*delta as f64/grid.ny as f64).cos()).sum::<f64>()/grid.ny as f64).collect();
        let orbit=|j:usize|if j==0 {vec![(0,1.0)]} else {vec![(j,std::f64::consts::FRAC_1_SQRT_2),(grid.ny-j,parity*std::f64::consts::FRAC_1_SQRT_2)]};
        let mut normal_laplacian=vec![0.0;rows*rows];
        for i in 0..rows {for j in 0..rows {
            for (a,x) in orbit(i+start) {for (b,y) in orbit(j+start) {normal_laplacian[i*rows+j]+=x*y*convolution[(a+grid.ny-b)%grid.ny];}}
        }}
        let row=operator.backend().alloc_field(Grid2D::new(grid.nx,1,grid.lx,grid.ly));
        Ok(Self{nx:grid.nx,ny:grid.ny,start,parity,rows,mapping,normal_laplacian,tangent_symbol,mass,input:vec![C::ZERO;basis.rank()],output:vec![C::ZERO;basis.rank()],row})
    }
    pub fn normal_rows(&self)->usize {self.rows}
    pub fn payload_bytes(&self)->usize {self.mapping.len()*(std::mem::size_of::<usize>()+16)+(self.normal_laplacian.len()+self.tangent_symbol.len()+self.mass.len())*8+(self.input.len()+self.output.len())*16+self.row.len()*std::mem::size_of::<num_complex::Complex<B::Real>>()}
    pub fn apply(&mut self,backend:&B,x:&B::Buffer,y:&mut B::Buffer) {
        assert_eq!(x.len(),self.mapping.len());assert_eq!(y.len(),self.mapping.len());
        crate::profile_scope!("apply_tm_mirror_reduced");
        crate::profiler::count("operator_A.columns",1);crate::profiler::count("operator_A.elements",self.input.len());
        crate::profiler::count("mirror.normal_multiply_terms",self.rows*self.rows*self.nx);
        for (z,&(i,p)) in x.as_slice().iter().zip(&self.mapping) {self.input[i]=p*C::new(z.re.to_accum(),z.im.to_accum());}
        for r in 0..self.rows {
            for (i,z) in self.row.as_mut_slice().iter_mut().enumerate() {let a=self.input[r*self.nx+i];*z=cscalar::<B::Real>(a.re,a.im);}
            backend.forward_fft_2d(&mut self.row);
            for (z,&weight) in self.row.as_mut_slice().iter_mut().zip(&self.tangent_symbol) {*z*=B::Real::from_accum(weight);}
            backend.inverse_fft_2d(&mut self.row);
            for (i,z) in self.row.as_slice().iter().enumerate() {self.output[r*self.nx+i]=C::new(z.re.to_accum(),z.im.to_accum());}
        }
        for r in 0..self.rows {for j in 0..self.rows {let a=self.normal_laplacian[r*self.rows+j];for i in 0..self.nx {self.output[r*self.nx+i]+=a*self.input[j*self.nx+i];}}}
        for (z,&(i,p)) in y.as_mut_slice().iter_mut().zip(&self.mapping) {let a=p.conj()*self.output[i];*z=cscalar::<B::Real>(a.re,a.im);}
    }
    pub fn apply_mass(&self,x:&B::Buffer,y:&mut B::Buffer) {
        assert_eq!(x.len(),self.mapping.len());assert_eq!(y.len(),self.mapping.len());
        crate::profiler::count("operator_B.columns",1);crate::profiler::count("operator_B.elements",self.mapping.len());
        for ((z,&a),&(i,_)) in y.as_mut_slice().iter_mut().zip(x.as_slice()).zip(&self.mapping) {*z=a*B::Real::from_accum(self.mass[i]);}
    }
}

/// Tensor TE specialization on the same qualified mirror range. The normal
/// derivative maps between opposite parity spaces; its transpose returns the
/// flux to the original space. Tangential derivatives use retained-row FFTs.
pub struct MirrorYTeKernel<B:SpectralBackend> {
    layout:MirrorYKernel<B>,opposite_start:usize,opposite_rows:usize,normal_derivative:Vec<f64>,
    tangent_derivative:Vec<f64>,tensors:Vec<[f64;4]>,gradient_x:Vec<C>,gradient_y:Vec<C>,
}
impl<B:SpectralBackend> MirrorYTeKernel<B> {
    pub fn new(operator:&ThetaOperator<B>,basis:&OrbitBasis)->Result<Self,String> {
        if operator.polarization()!=Polarization::TE {return Err("the reduced TE mirror kernel requires TE polarization".into());}
        let mut layout=MirrorYKernel::new_layout(operator,basis)?;
        let grid=operator.grid();let (kx,ky)=operator.k_plus_g_components();
        let scale=kx.iter().chain(ky).map(|x|x.abs()).fold(1.0,f64::max);
        for j in 0..grid.ny {for i in 0..grid.nx {
            if (kx[grid.idx(i,j)]-kx[i]).abs()>1e-12*scale || (ky[grid.idx(i,j)]-ky[grid.idx(0,j)]).abs()>1e-12*scale {return Err("TE mirror kernel currently requires Cartesian-aligned reciprocal axes".into());}
        }}
        if ky[0].abs()>1e-12*scale {return Err("nonzero normal Bloch component".into());}
        let opposite_start=1-layout.start;let opposite_rows=grid.ny/2+1-opposite_start;
        let tangent_derivative=kx[..grid.nx].to_vec();
        let convolution:Vec<_>=(0..grid.ny).map(|delta|
            -(0..grid.ny).map(|j|ky[grid.idx(0,j)]*(std::f64::consts::TAU*j as f64*delta as f64/grid.ny as f64).sin()).sum::<f64>()/grid.ny as f64).collect();
        let orbit=|j:usize,p:f64|if j==0 {vec![(0,1.0)]} else {vec![(j,std::f64::consts::FRAC_1_SQRT_2),(grid.ny-j,p*std::f64::consts::FRAC_1_SQRT_2)]};
        let mut normal_derivative=vec![0.0;opposite_rows*layout.rows];
        for i in 0..opposite_rows {for j in 0..layout.rows {
            for (a,x) in orbit(i+opposite_start,-layout.parity) {for (b,y) in orbit(j+layout.start,layout.parity) {
                normal_derivative[i*layout.rows+j]+=x*y*convolution[(a+grid.ny-b)%grid.ny];
            }}
        }}
        let tensor=|index:usize|operator.dielectric().inv_eps_tensors().map(|t|t[index]).unwrap_or_else(||{let v=operator.dielectric().inv_eps()[index];[v,0.0,0.0,v]});
        let mut tensors=vec![[0.0;4];grid.nx*(grid.ny/2+1)];
        for j in 0..=grid.ny/2 {for i in 0..grid.nx {
            let a=tensor(grid.idx(i,j));let b=tensor(grid.idx(i,(grid.ny-j)%grid.ny));
            tensors[j*grid.nx+i]=[0.5*(a[0]+b[0]),0.5*(a[1]-b[1]),0.5*(a[2]-b[2]),0.5*(a[3]+b[3])];
        }}
        // TE uses the first derivative between parities, not this TM table.
        layout.normal_laplacian=Vec::new();layout.mass=Vec::new();
        let gradient_x=vec![C::ZERO;layout.rows*grid.nx];let gradient_y=vec![C::ZERO;opposite_rows*grid.nx];
        Ok(Self{layout,opposite_start,opposite_rows,normal_derivative,tangent_derivative,tensors,gradient_x,gradient_y})
    }
    pub fn normal_rows(&self)->usize {self.layout.rows}
    pub fn opposite_rows(&self)->usize {self.opposite_rows}
    pub fn payload_bytes(&self)->usize {self.layout.payload_bytes()+(self.normal_derivative.len()+self.tangent_derivative.len())*8+self.tensors.len()*32+(self.gradient_x.len()+self.gradient_y.len())*16}
    pub fn apply(&mut self,backend:&B,x:&B::Buffer,y:&mut B::Buffer) {
        assert_eq!(x.len(),self.layout.mapping.len());assert_eq!(y.len(),self.layout.mapping.len());
        crate::profile_scope!("apply_te_mirror_reduced");
        let nx=self.layout.nx;let rows=self.layout.rows;
        crate::profiler::count("operator_A.columns",1);crate::profiler::count("operator_A.elements",self.layout.input.len());
        crate::profiler::count("mirror.normal_multiply_terms",2*self.opposite_rows*rows*nx);
        for (z,&(i,p)) in x.as_slice().iter().zip(&self.layout.mapping) {self.layout.input[i]=p*C::new(z.re.to_accum(),z.im.to_accum());}
        derivative_rows(backend,&mut self.layout.row,nx,rows,&self.layout.input,&mut self.gradient_x,&self.tangent_derivative,false);
        self.gradient_y.fill(C::ZERO);
        for r in 0..self.opposite_rows {for j in 0..rows {let a=self.normal_derivative[r*rows+j];for i in 0..nx {self.gradient_y[r*nx+i]+=a*self.layout.input[j*nx+i];}}}
        for normal in 0..=self.layout.ny/2 {for i in 0..nx {
            let p=normal.checked_sub(self.layout.start).map(|r|r*nx+i);
            let q=normal.checked_sub(self.opposite_start).map(|r|r*nx+i);
            let gx=p.map_or(C::ZERO,|j|self.gradient_x[j]);let gy=q.map_or(C::ZERO,|j|self.gradient_y[j]);
            let a=self.tensors[normal*nx+i];
            if let Some(j)=p {self.gradient_x[j]=a[0]*gx+a[1]*gy;}
            if let Some(j)=q {self.gradient_y[j]=a[2]*gx+a[3]*gy;}
        }}
        derivative_rows(backend,&mut self.layout.row,nx,rows,&self.gradient_x,&mut self.layout.output,&self.tangent_derivative,true);
        for r in 0..rows {for j in 0..self.opposite_rows {let a=self.normal_derivative[j*rows+r];for i in 0..nx {self.layout.output[r*nx+i]+=a*self.gradient_y[j*nx+i];}}}
        for (z,&(i,p)) in y.as_mut_slice().iter_mut().zip(&self.layout.mapping) {let a=p.conj()*self.layout.output[i];*z=cscalar::<B::Real>(a.re,a.im);}
    }
    pub fn apply_mass(&self,x:&B::Buffer,y:&mut B::Buffer) {
        assert_eq!(x.len(),self.layout.mapping.len());assert_eq!(y.len(),self.layout.mapping.len());
        crate::profiler::count("operator_B.columns",1);crate::profiler::count("operator_B.elements",self.layout.mapping.len());
        y.as_mut_slice().copy_from_slice(x.as_slice());
    }
}
fn derivative_rows<B:SpectralBackend>(backend:&B,row:&mut B::Buffer,nx:usize,rows:usize,input:&[C],output:&mut[C],symbol:&[f64],adjoint:bool) {
    for r in 0..rows {
        for (i,z) in row.as_mut_slice().iter_mut().enumerate() {let a=input[r*nx+i];*z=cscalar::<B::Real>(a.re,a.im);}
        backend.forward_fft_2d(row);
        for (z,&weight) in row.as_mut_slice().iter_mut().zip(symbol) {*z*=cscalar::<B::Real>(0.0,if adjoint {-weight}else{weight});}
        backend.inverse_fft_2d(row);
        for (i,z) in row.as_slice().iter().enumerate() {output[r*nx+i]=C::new(z.re.to_accum(),z.im.to_accum());}
    }
}
