//! CUDA kernels for photonic crystal eigensolver operations.
//!
//! These kernels eliminate host synchronization in the LOBPCG iteration by
//! performing pointwise operations entirely on the GPU.

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

/// CUDA kernel source code for photonic crystal operations.
///
/// All kernels operate on complex numbers stored as f64 pairs: [re, im, re, im, ...]
/// This matches the memory layout of Complex64.
#[cfg(feature = "cuda")]
const KERNEL_SOURCE: &str = r#"
extern "C" {

// Complex number helpers (stored as 2 consecutive f64s)
// For a complex array 'arr' of n elements, arr[2*i] = real, arr[2*i+1] = imag

/// Multiply field by i*k factor (gradient in Fourier space)
/// gx[i] *= i * kx[i], gy[i] *= i * ky[i]
/// i * (a + bi) = -b + ai
__global__ void gradient_factors_kernel(
    double* __restrict__ gx,      // [re, im, re, im, ...] length 2*n
    double* __restrict__ gy,      // [re, im, re, im, ...] length 2*n
    const double* __restrict__ kx, // real, length n
    const double* __restrict__ ky, // real, length n
    size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // gx[i] *= i * kx[i]
        double re_x = gx[2*i];
        double im_x = gx[2*i + 1];
        double factor_x = kx[i];
        // (re + im*i) * (i*factor) = (re + im*i) * (0 + factor*i)
        // = -im*factor + re*factor*i
        gx[2*i] = -im_x * factor_x;
        gx[2*i + 1] = re_x * factor_x;
        
        // gy[i] *= i * ky[i]
        double re_y = gy[2*i];
        double im_y = gy[2*i + 1];
        double factor_y = ky[i];
        gy[2*i] = -im_y * factor_y;
        gy[2*i + 1] = re_y * factor_y;
    }
}

/// Multiply field by scalar dielectric: f[i] *= eps[i] (real scalar)
/// Used for: TM mass operator, preconditioner
__global__ void scalar_multiply_kernel(
    double* __restrict__ field,   // [re, im, re, im, ...] length 2*n
    const double* __restrict__ eps, // real, length n
    size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double factor = eps[i];
        field[2*i] *= factor;
        field[2*i + 1] *= factor;
    }
}

/// Apply inverse epsilon (scalar version): gx[i] *= inv_eps[i], gy[i] *= inv_eps[i]
/// Used for: TE operator without averaging
__global__ void inv_eps_scalar_kernel(
    double* __restrict__ gx,         // [re, im, ...] length 2*n
    double* __restrict__ gy,         // [re, im, ...] length 2*n
    const double* __restrict__ inv_eps, // real, length n
    size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double factor = inv_eps[i];
        gx[2*i] *= factor;
        gx[2*i + 1] *= factor;
        gy[2*i] *= factor;
        gy[2*i + 1] *= factor;
    }
}

/// Apply inverse epsilon tensor (2x2 real tensor per point):
/// [out_x]   [t[0] t[1]] [gx]
/// [out_y] = [t[2] t[3]] [gy]
/// Used for: TE operator with subpixel averaging
__global__ void inv_eps_tensor_kernel(
    double* __restrict__ gx,            // [re, im, ...] length 2*n
    double* __restrict__ gy,            // [re, im, ...] length 2*n
    const double* __restrict__ tensors, // [t0, t1, t2, t3, t0, t1, ...] length 4*n
    size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double t0 = tensors[4*i];
        double t1 = tensors[4*i + 1];
        double t2 = tensors[4*i + 2];
        double t3 = tensors[4*i + 3];
        
        double gx_re = gx[2*i];
        double gx_im = gx[2*i + 1];
        double gy_re = gy[2*i];
        double gy_im = gy[2*i + 1];
        
        // out_x = t0*gx + t1*gy
        gx[2*i] = t0 * gx_re + t1 * gy_re;
        gx[2*i + 1] = t0 * gx_im + t1 * gy_im;
        
        // out_y = t2*gx + t3*gy
        gy[2*i] = t2 * gx_re + t3 * gy_re;
        gy[2*i + 1] = t2 * gx_im + t3 * gy_im;
    }
}

/// Assemble divergence: out[i] = -(i*kx[i]*gx[i] + i*ky[i]*gy[i])
/// Used for: TE operator final step
__global__ void divergence_kernel(
    const double* __restrict__ gx,     // [re, im, ...] length 2*n
    const double* __restrict__ gy,     // [re, im, ...] length 2*n
    double* __restrict__ out,          // [re, im, ...] length 2*n
    const double* __restrict__ kx,     // real, length n
    const double* __restrict__ ky,     // real, length n
    size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // i*k * (a + bi) = i*k*a + i*i*k*b = -k*b + i*k*a
        double kx_val = kx[i];
        double ky_val = ky[i];
        
        double gx_re = gx[2*i];
        double gx_im = gx[2*i + 1];
        double gy_re = gy[2*i];
        double gy_im = gy[2*i + 1];
        
        // i*kx*gx = (-kx*gx_im, kx*gx_re)
        // i*ky*gy = (-ky*gy_im, ky*gy_re)
        double div_re = -kx_val * gx_im - ky_val * gy_im;
        double div_im = kx_val * gx_re + ky_val * gy_re;
        
        // out = -div
        out[2*i] = -div_re;
        out[2*i + 1] = -div_im;
    }
}

/// Multiply by k²: field[i] *= k_sq[i]
/// Used for: TM operator (Laplacian in Fourier space)
__global__ void k_squared_kernel(
    double* __restrict__ field,        // [re, im, ...] length 2*n
    const double* __restrict__ k_sq,   // real, length n
    size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double factor = k_sq[i];
        field[2*i] *= factor;
        field[2*i + 1] *= factor;
    }
}

/// Diagonal preconditioner: field[i] *= diag[i] (complex)
/// Used for: Fourier diagonal preconditioner
__global__ void diagonal_complex_kernel(
    double* __restrict__ field,        // [re, im, ...] length 2*n
    const double* __restrict__ diag,   // [re, im, ...] length 2*n (complex)
    size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double f_re = field[2*i];
        double f_im = field[2*i + 1];
        double d_re = diag[2*i];
        double d_im = diag[2*i + 1];
        
        // (f_re + f_im*i) * (d_re + d_im*i)
        // = f_re*d_re - f_im*d_im + (f_re*d_im + f_im*d_re)*i
        field[2*i] = f_re * d_re - f_im * d_im;
        field[2*i + 1] = f_re * d_im + f_im * d_re;
    }
}

/// Invert gradient for transverse projection: out_x = -i*kx/(kx²+ky²) * f
///                                            out_y = -i*ky/(kx²+ky²) * f
/// For k=0, output is 0 (handled by setting inv_k_sq[0] = 0)
__global__ void invert_gradient_kernel(
    const double* __restrict__ f,       // [re, im, ...] length 2*n
    double* __restrict__ out_x,         // [re, im, ...] length 2*n
    double* __restrict__ out_y,         // [re, im, ...] length 2*n
    const double* __restrict__ kx,      // real, length n
    const double* __restrict__ ky,      // real, length n
    const double* __restrict__ inv_k_sq, // 1/(kx²+ky²), length n (0 at origin)
    size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double f_re = f[2*i];
        double f_im = f[2*i + 1];
        double kx_val = kx[i];
        double ky_val = ky[i];
        double ik_sq = inv_k_sq[i];
        
        // -i*k * inv_k_sq * f
        // = -i * (kx or ky) * inv_k_sq * (f_re + f_im*i)
        // = -(kx*inv) * i * (f_re + f_im*i)
        // = -(kx*inv) * (i*f_re - f_im)
        // = -(kx*inv) * (-f_im + i*f_re)
        // = (kx*inv)*f_im - i*(kx*inv)*f_re
        
        double scale_x = kx_val * ik_sq;
        double scale_y = ky_val * ik_sq;
        
        out_x[2*i] = scale_x * f_im;
        out_x[2*i + 1] = -scale_x * f_re;
        
        out_y[2*i] = scale_y * f_im;
        out_y[2*i + 1] = -scale_y * f_re;
    }
}

/// Copy buffer: dst = src
__global__ void copy_kernel(
    double* __restrict__ dst,
    const double* __restrict__ src,
    size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[2*i] = src[2*i];
        dst[2*i + 1] = src[2*i + 1];
    }
}

} // extern "C"
"#;

/// Compiled CUDA kernels for operator operations.
#[cfg(feature = "cuda")]
pub struct OperatorKernels {
    module: Arc<CudaModule>,
    gradient_factors: CudaFunction,
    scalar_multiply: CudaFunction,
    inv_eps_scalar: CudaFunction,
    inv_eps_tensor: CudaFunction,
    divergence: CudaFunction,
    k_squared: CudaFunction,
    diagonal_complex: CudaFunction,
    invert_gradient: CudaFunction,
    copy: CudaFunction,
}

#[cfg(feature = "cuda")]
impl OperatorKernels {
    /// Compile and load the operator kernels.
    ///
    /// This should be called once at startup. Kernel compilation is cached by the driver.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, Box<dyn std::error::Error>> {
        log::info!("Compiling CUDA operator kernels...");
        
        let ptx = cudarc::nvrtc::compile_ptx(KERNEL_SOURCE)?;
        let module = ctx.load_module(ptx)?;
        
        let gradient_factors = module.load_function("gradient_factors_kernel")?;
        let scalar_multiply = module.load_function("scalar_multiply_kernel")?;
        let inv_eps_scalar = module.load_function("inv_eps_scalar_kernel")?;
        let inv_eps_tensor = module.load_function("inv_eps_tensor_kernel")?;
        let divergence = module.load_function("divergence_kernel")?;
        let k_squared = module.load_function("k_squared_kernel")?;
        let diagonal_complex = module.load_function("diagonal_complex_kernel")?;
        let invert_gradient = module.load_function("invert_gradient_kernel")?;
        let copy = module.load_function("copy_kernel")?;
        
        log::info!("CUDA operator kernels compiled successfully");
        
        Ok(Self {
            module,
            gradient_factors,
            scalar_multiply,
            inv_eps_scalar,
            inv_eps_tensor,
            divergence,
            k_squared,
            diagonal_complex,
            invert_gradient,
            copy,
        })
    }

    /// Get optimal launch configuration for n elements.
    fn launch_config(n: usize) -> LaunchConfig {
        const BLOCK_SIZE: u32 = 256;
        let grid_size = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Apply gradient factors: gx *= i*kx, gy *= i*ky
    ///
    /// # Safety
    /// All device pointers must be valid and of correct size (n complex elements = 2n f64).
    /// kx, ky must have n f64 elements.
    pub unsafe fn apply_gradient_factors(
        &self,
        stream: &CudaStream,
        gx: &mut CudaSlice<f64>,
        gy: &mut CudaSlice<f64>,
        kx: &CudaSlice<f64>,
        ky: &CudaSlice<f64>,
        n: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = Self::launch_config(n);
        stream
            .launch_builder(&self.gradient_factors)
            .arg(gx)
            .arg(gy)
            .arg(kx)
            .arg(ky)
            .arg(&n)
            .launch(cfg)?;
        Ok(())
    }

    /// Multiply field by scalar array: field *= scalar
    ///
    /// # Safety
    /// field must have n complex elements (2n f64), scalar must have n f64 elements.
    pub unsafe fn apply_scalar_multiply(
        &self,
        stream: &CudaStream,
        field: &mut CudaSlice<f64>,
        scalar: &CudaSlice<f64>,
        n: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = Self::launch_config(n);
        stream
            .launch_builder(&self.scalar_multiply)
            .arg(field)
            .arg(scalar)
            .arg(&n)
            .launch(cfg)?;
        Ok(())
    }

    /// Apply inverse epsilon (scalar): gx *= inv_eps, gy *= inv_eps
    ///
    /// # Safety
    /// gx, gy must have n complex elements (2n f64), inv_eps must have n f64 elements.
    pub unsafe fn apply_inv_eps_scalar(
        &self,
        stream: &CudaStream,
        gx: &mut CudaSlice<f64>,
        gy: &mut CudaSlice<f64>,
        inv_eps: &CudaSlice<f64>,
        n: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = Self::launch_config(n);
        stream
            .launch_builder(&self.inv_eps_scalar)
            .arg(gx)
            .arg(gy)
            .arg(inv_eps)
            .arg(&n)
            .launch(cfg)?;
        Ok(())
    }

    /// Apply inverse epsilon tensor (2x2 per point)
    ///
    /// # Safety
    /// gx, gy must have n complex elements. tensors must have 4n f64 elements.
    pub unsafe fn apply_inv_eps_tensor(
        &self,
        stream: &CudaStream,
        gx: &mut CudaSlice<f64>,
        gy: &mut CudaSlice<f64>,
        tensors: &CudaSlice<f64>,
        n: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = Self::launch_config(n);
        stream
            .launch_builder(&self.inv_eps_tensor)
            .arg(gx)
            .arg(gy)
            .arg(tensors)
            .arg(&n)
            .launch(cfg)?;
        Ok(())
    }

    /// Assemble divergence: out = -(i*kx*gx + i*ky*gy)
    ///
    /// # Safety
    /// gx, gy, out must have n complex elements. kx, ky must have n f64 elements.
    pub unsafe fn apply_divergence(
        &self,
        stream: &CudaStream,
        gx: &CudaSlice<f64>,
        gy: &CudaSlice<f64>,
        out: &mut CudaSlice<f64>,
        kx: &CudaSlice<f64>,
        ky: &CudaSlice<f64>,
        n: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = Self::launch_config(n);
        stream
            .launch_builder(&self.divergence)
            .arg(gx)
            .arg(gy)
            .arg(out)
            .arg(kx)
            .arg(ky)
            .arg(&n)
            .launch(cfg)?;
        Ok(())
    }

    /// Multiply by k²: field *= k_sq
    ///
    /// # Safety
    /// field must have n complex elements. k_sq must have n f64 elements.
    pub unsafe fn apply_k_squared(
        &self,
        stream: &CudaStream,
        field: &mut CudaSlice<f64>,
        k_sq: &CudaSlice<f64>,
        n: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = Self::launch_config(n);
        stream
            .launch_builder(&self.k_squared)
            .arg(field)
            .arg(k_sq)
            .arg(&n)
            .launch(cfg)?;
        Ok(())
    }

    /// Apply complex diagonal: field *= diag
    ///
    /// # Safety
    /// field and diag must have n complex elements (2n f64 each).
    pub unsafe fn apply_diagonal_complex(
        &self,
        stream: &CudaStream,
        field: &mut CudaSlice<f64>,
        diag: &CudaSlice<f64>,
        n: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = Self::launch_config(n);
        stream
            .launch_builder(&self.diagonal_complex)
            .arg(field)
            .arg(diag)
            .arg(&n)
            .launch(cfg)?;
        Ok(())
    }

    /// Invert gradient for transverse projection
    ///
    /// # Safety
    /// f, out_x, out_y must have n complex elements. kx, ky, inv_k_sq must have n f64.
    pub unsafe fn apply_invert_gradient(
        &self,
        stream: &CudaStream,
        f: &CudaSlice<f64>,
        out_x: &mut CudaSlice<f64>,
        out_y: &mut CudaSlice<f64>,
        kx: &CudaSlice<f64>,
        ky: &CudaSlice<f64>,
        inv_k_sq: &CudaSlice<f64>,
        n: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = Self::launch_config(n);
        stream
            .launch_builder(&self.invert_gradient)
            .arg(f)
            .arg(out_x)
            .arg(out_y)
            .arg(kx)
            .arg(ky)
            .arg(inv_k_sq)
            .arg(&n)
            .launch(cfg)?;
        Ok(())
    }

    /// Copy buffer: dst = src
    ///
    /// # Safety
    /// dst and src must have n complex elements (2n f64).
    pub unsafe fn copy(
        &self,
        stream: &CudaStream,
        dst: &mut CudaSlice<f64>,
        src: &CudaSlice<f64>,
        n: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = Self::launch_config(n);
        stream
            .launch_builder(&self.copy)
            .arg(dst)
            .arg(src)
            .arg(&n)
            .launch(cfg)?;
        Ok(())
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use num_complex::Complex64;

    fn complex_to_f64(data: &[Complex64]) -> Vec<f64> {
        data.iter().flat_map(|c| [c.re, c.im]).collect()
    }

    fn f64_to_complex(data: &[f64]) -> Vec<Complex64> {
        data.chunks(2)
            .map(|c| Complex64::new(c[0], c[1]))
            .collect()
    }

    #[test]
    fn test_gradient_factors_kernel() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        let stream = ctx.default_stream();
        let kernels = OperatorKernels::new(&ctx).expect("Compile kernels");
        
        // Test data: gx = [1+2i, 3+4i], gy = [5+6i, 7+8i]
        let gx_host = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
        let gy_host = vec![Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)];
        let kx_host = vec![0.5, 1.5];
        let ky_host = vec![0.25, 0.75];
        
        let mut gx_dev = stream.clone_htod(&complex_to_f64(&gx_host)).unwrap();
        let mut gy_dev = stream.clone_htod(&complex_to_f64(&gy_host)).unwrap();
        let kx_dev = stream.clone_htod(&kx_host).unwrap();
        let ky_dev = stream.clone_htod(&ky_host).unwrap();
        
        unsafe {
            kernels.apply_gradient_factors(&stream, &mut gx_dev, &mut gy_dev, &kx_dev, &ky_dev, 2).unwrap();
        }
        
        let gx_result: Vec<f64> = stream.clone_dtoh(&gx_dev).unwrap();
        let gy_result: Vec<f64> = stream.clone_dtoh(&gy_dev).unwrap();
        
        let gx_complex = f64_to_complex(&gx_result);
        let gy_complex = f64_to_complex(&gy_result);
        
        // Check: gx[i] *= i*kx[i]
        // (1+2i) * (0 + 0.5i) = 0.5i + i² = -1 + 0.5i = ... 
        // Actually: (a+bi) * (i*k) = -b*k + a*k*i
        // (1+2i) * (0.5i) = -2*0.5 + 1*0.5*i = -1 + 0.5i
        let expected_gx0 = Complex64::new(0.0, 0.5) * gx_host[0];
        let expected_gx1 = Complex64::new(0.0, 1.5) * gx_host[1];
        
        assert!((gx_complex[0] - expected_gx0).norm() < 1e-10, "gx[0]: {:?} vs {:?}", gx_complex[0], expected_gx0);
        assert!((gx_complex[1] - expected_gx1).norm() < 1e-10, "gx[1]: {:?} vs {:?}", gx_complex[1], expected_gx1);
        
        let expected_gy0 = Complex64::new(0.0, 0.25) * gy_host[0];
        let expected_gy1 = Complex64::new(0.0, 0.75) * gy_host[1];
        
        assert!((gy_complex[0] - expected_gy0).norm() < 1e-10, "gy[0]: {:?} vs {:?}", gy_complex[0], expected_gy0);
        assert!((gy_complex[1] - expected_gy1).norm() < 1e-10, "gy[1]: {:?} vs {:?}", gy_complex[1], expected_gy1);
        
        println!("✓ gradient_factors_kernel works!");
    }

    #[test]
    fn test_k_squared_kernel() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        let stream = ctx.default_stream();
        let kernels = OperatorKernels::new(&ctx).expect("Compile kernels");
        
        let field_host = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
        let k_sq_host = vec![0.5, 2.0];
        
        let mut field_dev = stream.clone_htod(&complex_to_f64(&field_host)).unwrap();
        let k_sq_dev = stream.clone_htod(&k_sq_host).unwrap();
        
        unsafe {
            kernels.apply_k_squared(&stream, &mut field_dev, &k_sq_dev, 2).unwrap();
        }
        
        let result: Vec<f64> = stream.clone_dtoh(&field_dev).unwrap();
        let result_complex = f64_to_complex(&result);
        
        // field[i] *= k_sq[i]
        assert!((result_complex[0] - field_host[0] * 0.5).norm() < 1e-10);
        assert!((result_complex[1] - field_host[1] * 2.0).norm() < 1e-10);
        
        println!("✓ k_squared_kernel works!");
    }

    #[test]
    fn test_divergence_kernel() {
        let ctx = CudaContext::new(0).expect("CUDA context");
        let stream = ctx.default_stream();
        let kernels = OperatorKernels::new(&ctx).expect("Compile kernels");
        
        let gx_host = vec![Complex64::new(1.0, 0.0)];
        let gy_host = vec![Complex64::new(0.0, 1.0)];
        let kx_host = vec![1.0];
        let ky_host = vec![1.0];
        
        let gx_dev = stream.clone_htod(&complex_to_f64(&gx_host)).unwrap();
        let gy_dev = stream.clone_htod(&complex_to_f64(&gy_host)).unwrap();
        let kx_dev = stream.clone_htod(&kx_host).unwrap();
        let ky_dev = stream.clone_htod(&ky_host).unwrap();
        let mut out_dev = stream.alloc_zeros::<f64>(2).unwrap();
        
        unsafe {
            kernels.apply_divergence(&stream, &gx_dev, &gy_dev, &mut out_dev, &kx_dev, &ky_dev, 1).unwrap();
        }
        
        let result: Vec<f64> = stream.clone_dtoh(&out_dev).unwrap();
        let result_complex = f64_to_complex(&result);
        
        // out = -(i*kx*gx + i*ky*gy)
        // i*1*(1+0i) + i*1*(0+i) = i + i*i = i - 1 = -1 + i
        // out = -(-1 + i) = 1 - i
        let expected = Complex64::new(1.0, -1.0);
        
        assert!((result_complex[0] - expected).norm() < 1e-10, "div: {:?} vs {:?}", result_complex[0], expected);
        
        println!("✓ divergence_kernel works!");
    }
}
