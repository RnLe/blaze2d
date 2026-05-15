//! CUDA backend using cudarc when enabled.
//!
//! This backend provides GPU-accelerated spectral operations using CUDA.
//! When the `cuda` feature is disabled, this provides a stub implementation.
//!
//! # Memory Model
//!
//! The `CudaField` type maintains data on both host and device:
//! - GPU operations work on device memory directly (stored as f64 pairs)
//! - `as_slice()`/`as_mut_slice()` provide access to host memory (with sync)
//! - Explicit `sync_to_host()`/`sync_to_device()` for bulk transfers
//!
//! # Data Layout
//!
//! Complex numbers are stored as contiguous `[re, im, re, im, ...]` f64 arrays
//! on the GPU, which matches the memory layout of `Complex64` and allows
//! direct reinterpret casts.

use blaze2d_core::backend::{SpectralBackend, SpectralBuffer};
use blaze2d_core::field::FieldScalar;
use blaze2d_core::grid::Grid2D;
use num_complex::Complex64;

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};

#[cfg(feature = "cuda")]
use cudarc::cublas::CudaBlas;

/// CUDA backend for GPU-accelerated spectral operations.
///
/// When CUDA is available, this uses the GPU for FFT and linear algebra.
/// Falls back to a no-op stub when CUDA is not available.
pub struct CudaBackend {
    #[cfg(feature = "cuda")]
    ctx: Arc<CudaContext>,
    #[cfg(feature = "cuda")]
    stream: Arc<CudaStream>,
    #[cfg(feature = "cuda")]
    blas: Arc<CudaBlas>,
}

impl CudaBackend {
    /// Create a new CUDA backend on device 0.
    ///
    /// Returns `None` if CUDA is not available or device initialization fails.
    #[cfg(feature = "cuda")]
    pub fn try_new() -> Option<Self> {
        let ctx = CudaContext::new(0).ok()?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(Arc::clone(&stream)).ok()?;
        Some(Self { 
            ctx, 
            stream,
            blas: Arc::new(blas),
        })
    }

    /// Create a new CUDA backend, panicking if unavailable.
    pub fn new() -> Self {
        #[cfg(feature = "cuda")]
        {
            Self::try_new().expect("Failed to initialize CUDA device 0")
        }
        #[cfg(not(feature = "cuda"))]
        {
            Self {}
        }
    }

    /// Check if CUDA is available at runtime.
    #[cfg(feature = "cuda")]
    pub fn is_available() -> bool {
        CudaContext::new(0).is_ok()
    }

    #[cfg(not(feature = "cuda"))]
    pub fn is_available() -> bool {
        false
    }

    /// Get a reference to the CUDA stream.
    #[cfg(feature = "cuda")]
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl Default for CudaBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for CudaBackend {
    fn clone(&self) -> Self {
        #[cfg(feature = "cuda")]
        {
            Self {
                ctx: Arc::clone(&self.ctx),
                stream: Arc::clone(&self.stream),
                blas: Arc::clone(&self.blas),
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Self {}
        }
    }
}

// ============================================================================
// GPU Field Buffer (with CUDA feature)
// ============================================================================

#[cfg(feature = "cuda")]
mod gpu_field {
    use super::*;

    /// Reinterpret a slice of Complex64 as a slice of f64 pairs.
    /// This is safe because Complex64 is repr(C) with layout [re: f64, im: f64].
    fn complex_to_f64_slice(data: &[Complex64]) -> &[f64] {
        unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const f64,
                data.len() * 2,
            )
        }
    }

    use std::cell::{Cell, UnsafeCell};

    /// Reinterpret a mutable slice of Complex64 as a mutable slice of f64 pairs.
    fn complex_to_f64_slice_mut(data: &mut [Complex64]) -> &mut [f64] {
        unsafe {
            std::slice::from_raw_parts_mut(
                data.as_mut_ptr() as *mut f64,
                data.len() * 2,
            )
        }
    }

    /// Field buffer with GPU-resident data.
    ///
    /// This type maintains data on both host and device memory:
    /// - `device_data`: Primary storage on GPU (as f64 pairs: [re, im, re, im, ...])
    /// - `host_cache`: Cached host copy as Complex64 for `SpectralBuffer` trait
    /// - `host_dirty`: Flag indicating host cache needs refresh from device
    /// - `device_dirty`: Flag indicating device needs refresh from host
    ///
    /// # Synchronization
    ///
    /// The buffer uses lazy synchronization:
    /// - GPU operations set `host_dirty = true`
    /// - `as_slice()` syncs device→host if `host_dirty`
    /// - `as_mut_slice()` syncs device→host if `host_dirty`, sets `device_dirty`
    /// - GPU operations sync host→device if `device_dirty`
    ///
    /// # Interior Mutability
    ///
    /// Uses UnsafeCell/Cell for interior mutability to support the SpectralBuffer trait
    /// which requires `&self` for `as_slice()`. This is safe because:
    /// - Synchronization is "logically const" - doesn't change observable state
    /// - Single-threaded access is assumed (CudaField is !Sync)
    pub struct CudaField {
        grid: Grid2D,
        /// GPU-resident data as f64 pairs (primary storage for GPU operations)
        /// Layout: [re_0, im_0, re_1, im_1, ...] with length = 2 * grid.len()
        device_data: UnsafeCell<CudaSlice<f64>>,
        /// Host-side cache as Complex64 for SpectralBuffer trait compatibility
        host_cache: UnsafeCell<Vec<Complex64>>,
        /// Stream for synchronization
        stream: Arc<CudaStream>,
        /// True if host cache is stale (device has newer data)
        host_dirty: Cell<bool>,
        /// True if device is stale (host has newer data)  
        device_dirty: Cell<bool>,
    }

    // Note: CudaField is !Sync because of UnsafeCell usage.
    // It uses single-threaded interior mutability for the sync operations.

    impl CudaField {
        /// Create a new GPU field with zeroed data.
        pub fn zeros(stream: Arc<CudaStream>, grid: Grid2D) -> Self {
            let len = grid.len();
            // Allocate device memory for f64 pairs (2 * len elements)
            let device_data = stream
                .alloc_zeros::<f64>(len * 2)
                .expect("Failed to allocate GPU memory");
            let host_cache = vec![Complex64::default(); len];
            
            Self {
                grid,
                device_data: UnsafeCell::new(device_data),
                host_cache: UnsafeCell::new(host_cache),
                stream,
                host_dirty: Cell::new(false),
                device_dirty: Cell::new(false),
            }
        }

        /// Create a GPU field from host data.
        pub fn from_host(stream: Arc<CudaStream>, grid: Grid2D, data: Vec<Complex64>) -> Self {
            assert_eq!(data.len(), grid.len(), "Data length must match grid size");
            // Convert Complex64 slice to f64 slice and copy to device
            let f64_data = complex_to_f64_slice(&data);
            let device_data = stream
                .clone_htod(f64_data)
                .expect("Failed to copy data to GPU");
            
            Self {
                grid,
                device_data: UnsafeCell::new(device_data),
                host_cache: UnsafeCell::new(data),
                stream,
                host_dirty: Cell::new(false),
                device_dirty: Cell::new(false),
            }
        }

        /// Get a reference to the stream for GPU operations.
        pub fn stream(&self) -> &Arc<CudaStream> {
            &self.stream
        }

        /// Mark that the device data has been modified.
        /// Call this after GPU kernels that modify the data.
        pub fn mark_device_modified(&self) {
            self.host_dirty.set(true);
        }

        /// Ensure host cache is current (sync from device if dirty).
        /// 
        /// # Safety
        /// This uses UnsafeCell for interior mutability - callers must ensure
        /// no aliased mutable references exist.
        fn ensure_host_current(&self) {
            if self.host_dirty.get() {
                // Safety: Single-threaded access pattern, no aliasing possible
                let host_cache = unsafe { &mut *self.host_cache.get() };
                let device_data = unsafe { &*self.device_data.get() };
                let f64_slice = complex_to_f64_slice_mut(host_cache);
                self.stream
                    .memcpy_dtoh(device_data, f64_slice)
                    .expect("Failed to sync device to host");
                self.host_dirty.set(false);
            }
        }

        /// Ensure device data is current (sync from host if dirty).
        /// 
        /// # Safety
        /// This uses UnsafeCell for interior mutability - callers must ensure
        /// no aliased mutable references exist.
        fn ensure_device_current(&self) {
            if self.device_dirty.get() {
                // Safety: Single-threaded access pattern, no aliasing possible
                let host_cache = unsafe { &*self.host_cache.get() };
                let device_data = unsafe { &mut *self.device_data.get() };
                let f64_slice = complex_to_f64_slice(host_cache);
                self.stream
                    .memcpy_htod(f64_slice, device_data)
                    .expect("Failed to sync host to device");
                self.device_dirty.set(false);
            }
        }

        /// Force sync device → host.
        pub fn sync_to_host(&self) {
            // Safety: Single-threaded access pattern, no aliasing possible
            let host_cache = unsafe { &mut *self.host_cache.get() };
            let device_data = unsafe { &*self.device_data.get() };
            let f64_slice = complex_to_f64_slice_mut(host_cache);
            self.stream
                .memcpy_dtoh(device_data, f64_slice)
                .expect("Failed to sync device to host");
            self.host_dirty.set(false);
        }

        /// Force sync host → device.
        pub fn sync_to_device(&self) {
            // Safety: Single-threaded access pattern, no aliasing possible
            let host_cache = unsafe { &*self.host_cache.get() };
            let device_data = unsafe { &mut *self.device_data.get() };
            let f64_slice = complex_to_f64_slice(host_cache);
            self.stream
                .memcpy_htod(f64_slice, device_data)
                .expect("Failed to sync host to device");
            self.device_dirty.set(false);
        }

        /// Get mutable access to device data for GPU operations.
        /// This ensures device is current and marks host as dirty.
        pub fn with_device_data_mut<F, R>(&self, f: F) -> R
        where
            F: FnOnce(&mut CudaSlice<f64>) -> R,
        {
            self.ensure_device_current();
            // Safety: Single-threaded access pattern, no aliasing possible
            let device_data = unsafe { &mut *self.device_data.get() };
            let result = f(device_data);
            self.host_dirty.set(true);
            result
        }

        /// Get read-only access to device data for GPU operations.
        /// This ensures device is current.
        pub fn with_device_data<F, R>(&self, f: F) -> R
        where
            F: FnOnce(&CudaSlice<f64>) -> R,
        {
            self.ensure_device_current();
            // Safety: Single-threaded access pattern, no aliasing possible
            let device_data = unsafe { &*self.device_data.get() };
            f(device_data)
        }
    }

    impl Clone for CudaField {
        fn clone(&self) -> Self {
            // Ensure host is current for cloning
            self.ensure_host_current();
            // Safety: Single-threaded access pattern, no aliasing during clone
            let host_cache = unsafe { &*self.host_cache.get() };
            let f64_slice = complex_to_f64_slice(host_cache);
            let device_data = self.stream
                .clone_htod(f64_slice)
                .expect("Failed to clone GPU memory");
            
            Self {
                grid: self.grid,
                device_data: UnsafeCell::new(device_data),
                host_cache: UnsafeCell::new(host_cache.clone()),
                stream: Arc::clone(&self.stream),
                host_dirty: Cell::new(false),
                device_dirty: Cell::new(false),
            }
        }
    }

    impl SpectralBuffer for CudaField {
        fn len(&self) -> usize {
            self.grid.len()
        }

        fn grid(&self) -> Grid2D {
            self.grid
        }

        fn as_slice(&self) -> &[Complex64] {
            // Sync is "logically const" - doesn't change observable state
            self.ensure_host_current();
            // Safety: Single-threaded access pattern, no aliasing possible
            // The returned reference lives as long as self
            unsafe { &*self.host_cache.get() }
        }

        fn as_mut_slice(&mut self) -> &mut [Complex64] {
            self.ensure_host_current();
            self.device_dirty.set(true);
            // Safety: We have &mut self, so exclusive access is guaranteed
            unsafe { &mut *self.host_cache.get() }
        }
    }
}

#[cfg(feature = "cuda")]
pub use gpu_field::CudaField;

// ============================================================================
// Stub Field Buffer (without CUDA feature)
// ============================================================================

#[cfg(not(feature = "cuda"))]
mod stub_field {
    use super::*;

    /// Stub field buffer when CUDA is not available.
    #[derive(Clone)]
    pub struct CudaField {
        grid: Grid2D,
        data: Vec<FieldScalar>,
    }

    impl CudaField {
        pub fn zeros(grid: Grid2D) -> Self {
            Self {
                grid,
                data: vec![FieldScalar::default(); grid.len()],
            }
        }
    }

    impl SpectralBuffer for CudaField {
        fn len(&self) -> usize {
            self.data.len()
        }

        fn grid(&self) -> Grid2D {
            self.grid
        }

        fn as_slice(&self) -> &[FieldScalar] {
            &self.data
        }

        fn as_mut_slice(&mut self) -> &mut [FieldScalar] {
            &mut self.data
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub use stub_field::CudaField;

// ============================================================================
// SpectralBackend Implementation
// ============================================================================

#[cfg(feature = "cuda")]
impl SpectralBackend for CudaBackend {
    type Buffer = CudaField;

    fn alloc_field(&self, grid: Grid2D) -> Self::Buffer {
        CudaField::zeros(Arc::clone(&self.stream), grid)
    }

    fn forward_fft_2d(&self, buffer: &mut Self::Buffer) {
        // Use CPU FFT via rustfft - cuFFT has too much overhead for small grids
        use rustfft::FftPlanner;
        
        let grid = buffer.grid();
        let (nx, ny) = (grid.nx, grid.ny);
        
        // Get data on host side for FFT
        let data = buffer.as_mut_slice();
        let mut planner = FftPlanner::<f64>::new();
        let fft_x = planner.plan_fft_forward(nx);
        let fft_y = planner.plan_fft_forward(ny);
        
        // Row FFTs
        for row in 0..ny {
            let start = row * nx;
            let end = start + nx;
            fft_x.process(&mut data[start..end]);
        }
        
        // Column FFTs (need to gather/scatter)
        let mut col_buf = vec![Complex64::ZERO; ny];
        for col in 0..nx {
            for row in 0..ny {
                col_buf[row] = data[row * nx + col];
            }
            fft_y.process(&mut col_buf);
            for row in 0..ny {
                data[row * nx + col] = col_buf[row];
            }
        }
    }

    fn inverse_fft_2d(&self, buffer: &mut Self::Buffer) {
        // Use CPU FFT via rustfft
        use rustfft::FftPlanner;
        
        let grid = buffer.grid();
        let (nx, ny) = (grid.nx, grid.ny);
        let scale = 1.0 / (nx * ny) as f64;
        
        let data = buffer.as_mut_slice();
        let mut planner = FftPlanner::<f64>::new();
        let fft_x = planner.plan_fft_inverse(nx);
        let fft_y = planner.plan_fft_inverse(ny);
        
        // Row FFTs
        for row in 0..ny {
            let start = row * nx;
            let end = start + nx;
            fft_x.process(&mut data[start..end]);
        }
        
        // Column FFTs (need to gather/scatter)
        let mut col_buf = vec![Complex64::ZERO; ny];
        for col in 0..nx {
            for row in 0..ny {
                col_buf[row] = data[row * nx + col];
            }
            fft_y.process(&mut col_buf);
            for row in 0..ny {
                data[row * nx + col] = col_buf[row] * scale;
            }
        }
    }

    fn scale(&self, alpha: Complex64, buffer: &mut Self::Buffer) {
        use cudarc::cublas::sys::{cublasZscal_v2, cuDoubleComplex, cublasStatus_t};
        
        let n = buffer.len() as i32;
        let alpha_c = cuDoubleComplex { x: alpha.re, y: alpha.im };
        
        buffer.with_device_data_mut(|device_data| {
            // Reinterpret f64 pairs as cuDoubleComplex
            let x_ptr = device_data.device_ptr_mut(&self.stream).0 as *mut cuDoubleComplex;
            unsafe {
                let status = cublasZscal_v2(
                    *self.blas.handle(),
                    n,
                    &alpha_c,
                    x_ptr,
                    1,  // incx = 1 (contiguous)
                );
                assert_eq!(status, cublasStatus_t::CUBLAS_STATUS_SUCCESS, "cuBLAS Zscal failed");
            }
        });
        // Data was modified on device, mark host as dirty
        buffer.mark_device_modified();
    }

    fn axpy(&self, alpha: Complex64, x: &Self::Buffer, y: &mut Self::Buffer) {
        use cudarc::cublas::sys::{cublasZaxpy_v2, cuDoubleComplex, cublasStatus_t};
        
        let n = x.len() as i32;
        let alpha_c = cuDoubleComplex { x: alpha.re, y: alpha.im };
        
        x.with_device_data(|x_data| {
            y.with_device_data_mut(|y_data| {
                let x_ptr = x_data.device_ptr(&self.stream).0 as *const cuDoubleComplex;
                let y_ptr = y_data.device_ptr_mut(&self.stream).0 as *mut cuDoubleComplex;
                unsafe {
                    let status = cublasZaxpy_v2(
                        *self.blas.handle(),
                        n,
                        &alpha_c,
                        x_ptr,
                        1,  // incx = 1
                        y_ptr,
                        1,  // incy = 1
                    );
                    assert_eq!(status, cublasStatus_t::CUBLAS_STATUS_SUCCESS, "cuBLAS Zaxpy failed");
                }
            });
        });
        // y was modified on device
        y.mark_device_modified();
    }

    fn dot(&self, x: &Self::Buffer, y: &Self::Buffer) -> Complex64 {
        use cudarc::cublas::sys::{cublasZdotc_v2, cuDoubleComplex, cublasStatus_t};
        
        let n = x.len() as i32;
        let mut result = cuDoubleComplex { x: 0.0, y: 0.0 };
        
        x.with_device_data(|x_data| {
            y.with_device_data(|y_data| {
                let x_ptr = x_data.device_ptr(&self.stream).0 as *const cuDoubleComplex;
                let y_ptr = y_data.device_ptr(&self.stream).0 as *const cuDoubleComplex;
                unsafe {
                    let status = cublasZdotc_v2(
                        *self.blas.handle(),
                        n,
                        x_ptr,
                        1,  // incx = 1
                        y_ptr,
                        1,  // incy = 1
                        &mut result,
                    );
                    assert_eq!(status, cublasStatus_t::CUBLAS_STATUS_SUCCESS, "cuBLAS Zdotc failed");
                }
            });
        });
        
        Complex64::new(result.x, result.y)
    }

    fn gram_matrix(&self, x: &[Self::Buffer], y: &[Self::Buffer]) -> Vec<Complex64> {
        use cudarc::cublas::sys::{cublasZgemm_v2, cuDoubleComplex, cublasStatus_t, cublasOperation_t};
        
        let p = x.len();
        let q = y.len();
        
        if p == 0 || q == 0 {
            return vec![];
        }
        
        let n = x[0].len();
        
        // For GEMM: C = α * op(A) * op(B) + β * C
        // We want: G(p×q) = X^H(p×n) × Y(n×q)
        // 
        // cuBLAS uses column-major storage.
        // Our vectors are stored row-major in each buffer.
        //
        // Strategy: Build X as (n × p) column-major and Y as (n × q) column-major
        // Then G = X^H × Y where X^H is (p × n)
        //
        // GEMM call: C(m×n) = α * op(A)(m×k) × op(B)(k×n) + β * C
        // For us: G(p×q) = α * X^H(p×n) × Y(n×q)
        //   m = p, n = q, k = n_vec
        //   transa = CUBLAS_OP_C (conjugate transpose)
        //   transb = CUBLAS_OP_N (no transpose)
        //   A = X (n_vec × p), lda = n_vec
        //   B = Y (n_vec × q), ldb = n_vec
        //   C = G (p × q), ldc = p
        
        // Allocate device memory for matrices X, Y, and G
        let x_host: Vec<Complex64> = x.iter()
            .flat_map(|buf| buf.as_slice().iter().copied())
            .collect();
        let y_host: Vec<Complex64> = y.iter()
            .flat_map(|buf| buf.as_slice().iter().copied())
            .collect();
        
        // We need to transpose from "p vectors of length n" to "n × p column-major"
        // Current: x_host = [x0[0], x0[1], ..., x0[n-1], x1[0], ..., x1[n-1], ...]
        // Needed:  X_col  = [x0[0], x1[0], ..., xp-1[0], x0[1], x1[1], ...]
        // Actually, each xi is contiguous, so if we concatenate them we get:
        // [x0, x1, ..., xp-1] where each xi has n elements
        // This IS column-major for a (n × p) matrix! Each column is one vector.
        
        // Reinterpret as f64 for device copy
        let x_f64 = unsafe {
            std::slice::from_raw_parts(x_host.as_ptr() as *const f64, x_host.len() * 2)
        };
        let y_f64 = unsafe {
            std::slice::from_raw_parts(y_host.as_ptr() as *const f64, y_host.len() * 2)
        };
        
        // Copy to device
        let x_dev = self.stream.clone_htod(x_f64).expect("Failed to copy X to GPU");
        let y_dev = self.stream.clone_htod(y_f64).expect("Failed to copy Y to GPU");
        
        // Allocate output matrix G (p × q) on device
        let mut g_dev = self.stream.alloc_zeros::<f64>(p * q * 2).expect("Failed to allocate G on GPU");
        
        let alpha = cuDoubleComplex { x: 1.0, y: 0.0 };
        let beta = cuDoubleComplex { x: 0.0, y: 0.0 };
        
        let x_ptr = x_dev.device_ptr(&self.stream).0 as *const cuDoubleComplex;
        let y_ptr = y_dev.device_ptr(&self.stream).0 as *const cuDoubleComplex;
        let g_ptr = g_dev.device_ptr_mut(&self.stream).0 as *mut cuDoubleComplex;
        
        unsafe {
            let status = cublasZgemm_v2(
                *self.blas.handle(),
                cublasOperation_t::CUBLAS_OP_C,  // X^H (conjugate transpose)
                cublasOperation_t::CUBLAS_OP_N,  // Y (no transpose)
                p as i32,                         // m = rows of op(A) = p
                q as i32,                         // n = cols of op(B) = q
                n as i32,                         // k = cols of op(A) = rows of op(B) = n
                &alpha,
                x_ptr,
                n as i32,                         // lda = leading dimension of A = n
                y_ptr,
                n as i32,                         // ldb = leading dimension of B = n
                &beta,
                g_ptr,
                p as i32,                         // ldc = leading dimension of C = p
            );
            assert_eq!(status, cublasStatus_t::CUBLAS_STATUS_SUCCESS, "cuBLAS Zgemm failed");
        }
        
        // Copy result back to host
        let mut g_f64 = vec![0.0f64; p * q * 2];
        self.stream.memcpy_dtoh(&g_dev, &mut g_f64).expect("Failed to copy G from GPU");
        
        // Reinterpret as Complex64 (column-major order from GEMM)
        // GEMM output G is column-major (p × q), but we want row-major
        // Column-major G: G[i + j*p] = G_ij
        // Row-major result: result[i*q + j] = G_ij
        let g_complex = unsafe {
            std::slice::from_raw_parts(g_f64.as_ptr() as *const Complex64, p * q)
        };
        
        // Transpose from column-major to row-major
        let mut result = vec![Complex64::ZERO; p * q];
        for i in 0..p {
            for j in 0..q {
                result[i * q + j] = g_complex[i + j * p];
            }
        }
        
        result
    }

    fn linear_combinations(
        &self,
        q: &[Self::Buffer],
        coeffs: &[Complex64],
        num_outputs: usize,
    ) -> Vec<Self::Buffer> {
        use cudarc::cublas::sys::{cublasZgemm_v2, cuDoubleComplex, cublasStatus_t, cublasOperation_t};
        
        let r = q.len();
        let m = num_outputs;
        
        assert!(!q.is_empty(), "Input vectors cannot be empty");
        assert_eq!(coeffs.len(), r * m, "coeffs must have r×m elements");
        
        let n = q[0].len();
        let grid = q[0].grid();
        
        // For GEMM: out = α * op(A) * op(B) + β * out
        // We want: X(n×m) = Q(n×r) × C(r×m)
        //
        // cuBLAS uses column-major storage.
        // Q vectors are concatenated as (n × r) column-major (each column is one vector)
        // C coeffs are provided in column-major: coeffs[i + j*r] = C[i,j]
        //
        // GEMM call: out(m_g×n_g) = α * A(m_g×k_g) × B(k_g×n_g) + β * out
        // For us: X(n×m) = Q(n×r) × C(r×m)
        //   m_g = n (vector length)
        //   n_g = m (number of outputs)  
        //   k_g = r (number of inputs)
        //   transa = CUBLAS_OP_N (no transpose for Q)
        //   transb = CUBLAS_OP_N (no transpose for C - already column-major)
        //   A = Q (n × r), lda = n
        //   B = C (r × m), ldb = r
        //   out = X (n × m), ldc = n
        
        // Build Q matrix: concatenate all input vectors
        // This produces column-major (n × r) since each vector is contiguous
        let q_host: Vec<Complex64> = q.iter()
            .flat_map(|buf| buf.as_slice().iter().copied())
            .collect();
        
        // Coeffs are already column-major from the trait specification
        // coeffs[i + j*r] = C[i,j], which is exactly what cuBLAS expects
        
        // Reinterpret as f64 for device copy
        let q_f64 = unsafe {
            std::slice::from_raw_parts(q_host.as_ptr() as *const f64, q_host.len() * 2)
        };
        let c_f64 = unsafe {
            std::slice::from_raw_parts(coeffs.as_ptr() as *const f64, coeffs.len() * 2)
        };
        
        // Copy to device
        let q_dev = self.stream.clone_htod(q_f64).expect("Failed to copy Q to GPU");
        let c_dev = self.stream.clone_htod(c_f64).expect("Failed to copy C to GPU");
        
        // Allocate output matrix X (n × m) on device
        let mut x_dev = self.stream.alloc_zeros::<f64>(n * m * 2).expect("Failed to allocate X on GPU");
        
        let alpha = cuDoubleComplex { x: 1.0, y: 0.0 };
        let beta = cuDoubleComplex { x: 0.0, y: 0.0 };
        
        let q_ptr = q_dev.device_ptr(&self.stream).0 as *const cuDoubleComplex;
        let c_ptr = c_dev.device_ptr(&self.stream).0 as *const cuDoubleComplex;
        let x_ptr = x_dev.device_ptr_mut(&self.stream).0 as *mut cuDoubleComplex;
        
        unsafe {
            let status = cublasZgemm_v2(
                *self.blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,  // Q (no transpose)
                cublasOperation_t::CUBLAS_OP_N,  // C (no transpose, already column-major)
                n as i32,                         // m_g = rows of Q = n
                m as i32,                         // n_g = cols of C = m
                r as i32,                         // k_g = cols of Q = rows of C = r
                &alpha,
                q_ptr,
                n as i32,                         // lda = n
                c_ptr,
                r as i32,                         // ldb = r
                &beta,
                x_ptr,
                n as i32,                         // ldc = n
            );
            assert_eq!(status, cublasStatus_t::CUBLAS_STATUS_SUCCESS, "cuBLAS Zgemm failed");
        }
        
        // Copy result back to host
        let mut x_f64 = vec![0.0f64; n * m * 2];
        self.stream.memcpy_dtoh(&x_dev, &mut x_f64).expect("Failed to copy X from GPU");
        
        // Reinterpret as Complex64 (column-major from GEMM)
        let x_complex = unsafe {
            std::slice::from_raw_parts(x_f64.as_ptr() as *const Complex64, n * m)
        };
        
        // Create output buffers - each column of X is one output vector
        let mut outputs = Vec::with_capacity(m);
        for j in 0..m {
            // Column j: x_complex[j*n .. (j+1)*n]
            let col_data: Vec<Complex64> = x_complex[j * n..(j + 1) * n].to_vec();
            outputs.push(CudaField::from_host(Arc::clone(&self.stream), grid, col_data));
        }
        
        outputs
    }
}

#[cfg(not(feature = "cuda"))]
impl SpectralBackend for CudaBackend {
    type Buffer = CudaField;

    fn alloc_field(&self, grid: Grid2D) -> Self::Buffer {
        CudaField::zeros(grid)
    }

    fn forward_fft_2d(&self, _buffer: &mut Self::Buffer) {
        // No-op stub
    }

    fn inverse_fft_2d(&self, _buffer: &mut Self::Buffer) {
        // No-op stub
    }

    fn scale(&self, alpha: Complex64, buffer: &mut Self::Buffer) {
        for value in buffer.as_mut_slice() {
            let v64 = Complex64::new(value.re as f64, value.im as f64);
            let result = v64 * alpha;
            #[cfg(not(feature = "mixed-precision"))]
            {
                *value = result;
            }
            #[cfg(feature = "mixed-precision")]
            {
                *value = FieldScalar::new(result.re as f32, result.im as f32);
            }
        }
    }

    fn axpy(&self, alpha: Complex64, x: &Self::Buffer, y: &mut Self::Buffer) {
        for (dst, src) in y.as_mut_slice().iter_mut().zip(x.as_slice()) {
            let src64 = Complex64::new(src.re as f64, src.im as f64);
            let dst64 = Complex64::new(dst.re as f64, dst.im as f64);
            let result = dst64 + alpha * src64;
            #[cfg(not(feature = "mixed-precision"))]
            {
                *dst = result;
            }
            #[cfg(feature = "mixed-precision")]
            {
                *dst = FieldScalar::new(result.re as f32, result.im as f32);
            }
        }
    }

    fn dot(&self, x: &Self::Buffer, y: &Self::Buffer) -> Complex64 {
        x.as_slice()
            .iter()
            .zip(y.as_slice())
            .map(|(a, b)| {
                let a64 = Complex64::new(a.re as f64, a.im as f64);
                let b64 = Complex64::new(b.re as f64, b.im as f64);
                a64.conj() * b64
            })
            .sum()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_available() {
        assert!(CudaBackend::is_available(), "CUDA should be available");
        let backend = CudaBackend::try_new();
        assert!(backend.is_some(), "Should be able to create CUDA backend");
        println!("✓ CUDA backend initialized successfully!");
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_field_allocation() {
        let backend = CudaBackend::new();
        let grid = Grid2D::new(8, 8, 1.0, 1.0);
        
        let field = backend.alloc_field(grid);
        assert_eq!(field.len(), 64);
        
        // Check that data is zeroed
        for &val in field.as_slice() {
            assert_eq!(val, Complex64::default());
        }
        println!("✓ GPU field allocation works!");
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_field_roundtrip() {
        let backend = CudaBackend::new();
        let grid = Grid2D::new(4, 4, 1.0, 1.0);
        
        let mut field = backend.alloc_field(grid);
        
        // Write some data via host
        let data = field.as_mut_slice();
        for (i, val) in data.iter_mut().enumerate() {
            *val = Complex64::new(i as f64, -(i as f64));
        }
        
        // Sync to device
        field.sync_to_device();
        
        // Read back via host (should trigger sync)
        let result = field.as_slice();
        for (i, &val) in result.iter().enumerate() {
            assert_eq!(val, Complex64::new(i as f64, -(i as f64)));
        }
        println!("✓ GPU field roundtrip works!");
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_field_clone() {
        let backend = CudaBackend::new();
        let grid = Grid2D::new(4, 4, 1.0, 1.0);
        
        let mut field = backend.alloc_field(grid);
        
        // Write data
        for (i, val) in field.as_mut_slice().iter_mut().enumerate() {
            *val = Complex64::new(i as f64, 0.0);
        }
        field.sync_to_device();
        
        // Clone
        let field2 = field.clone();
        
        // Verify clone has same data
        for (i, &val) in field2.as_slice().iter().enumerate() {
            assert_eq!(val, Complex64::new(i as f64, 0.0));
        }
        
        // Modify original
        field.as_mut_slice()[0] = Complex64::new(999.0, 0.0);
        field.sync_to_device();
        
        // Clone should be unaffected
        assert_eq!(field2.as_slice()[0], Complex64::new(0.0, 0.0));
        println!("✓ GPU field clone works!");
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_spectral_backend_operations() {
        let backend = CudaBackend::new();
        let grid = Grid2D::new(4, 4, 1.0, 1.0);
        
        // Test scale
        let mut x = backend.alloc_field(grid);
        for val in x.as_mut_slice().iter_mut() {
            *val = Complex64::new(1.0, 0.0);
        }
        x.sync_to_device();
        
        backend.scale(Complex64::new(2.0, 0.0), &mut x);
        
        for &val in x.as_slice() {
            assert_eq!(val, Complex64::new(2.0, 0.0));
        }
        
        // Test axpy: y = alpha*x + y
        let mut y = backend.alloc_field(grid);
        for val in y.as_mut_slice().iter_mut() {
            *val = Complex64::new(1.0, 0.0);
        }
        y.sync_to_device();
        
        backend.axpy(Complex64::new(0.5, 0.0), &x, &mut y);
        
        // y = 0.5 * 2.0 + 1.0 = 2.0
        for &val in y.as_slice() {
            assert_eq!(val, Complex64::new(2.0, 0.0));
        }
        
        // Test dot: <x, y> = sum(conj(x) * y)
        let dot_result = backend.dot(&x, &y);
        // x = [2, 2, ..., 2] (16 elements), y = [2, 2, ..., 2]
        // dot = 16 * conj(2) * 2 = 16 * 4 = 64
        assert_eq!(dot_result, Complex64::new(64.0, 0.0));
        
        println!("✓ SpectralBackend operations work!");
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_complex_memory_layout() {
        // Verify that Complex64 has the expected memory layout
        let data = vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
        ];
        
        let f64_ptr = data.as_ptr() as *const f64;
        unsafe {
            assert_eq!(*f64_ptr.add(0), 1.0); // re of first
            assert_eq!(*f64_ptr.add(1), 2.0); // im of first
            assert_eq!(*f64_ptr.add(2), 3.0); // re of second
            assert_eq!(*f64_ptr.add(3), 4.0); // im of second
        }
        println!("✓ Complex64 memory layout is [re, im, re, im, ...]");
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_complex_dot_product() {
        let backend = CudaBackend::new();
        let grid = Grid2D::new(2, 2, 1.0, 1.0);
        
        let mut x = backend.alloc_field(grid);
        let mut y = backend.alloc_field(grid);
        
        // x = [1+i, 2+2i, 3+3i, 4+4i]
        x.as_mut_slice()[0] = Complex64::new(1.0, 1.0);
        x.as_mut_slice()[1] = Complex64::new(2.0, 2.0);
        x.as_mut_slice()[2] = Complex64::new(3.0, 3.0);
        x.as_mut_slice()[3] = Complex64::new(4.0, 4.0);
        x.sync_to_device();
        
        // y = [1, 1, 1, 1]
        for val in y.as_mut_slice().iter_mut() {
            *val = Complex64::new(1.0, 0.0);
        }
        y.sync_to_device();
        
        // dot = sum(conj(x_i) * y_i)
        // conj(1+i) * 1 + conj(2+2i) * 1 + conj(3+3i) * 1 + conj(4+4i) * 1
        // = (1-i) + (2-2i) + (3-3i) + (4-4i)
        // = 10 - 10i
        let dot_result = backend.dot(&x, &y);
        assert!((dot_result.re - 10.0).abs() < 1e-10);
        assert!((dot_result.im - (-10.0)).abs() < 1e-10);
        println!("✓ Complex dot product works correctly!");
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_complex_scale() {
        let backend = CudaBackend::new();
        let grid = Grid2D::new(2, 2, 1.0, 1.0);
        
        let mut x = backend.alloc_field(grid);
        x.as_mut_slice()[0] = Complex64::new(1.0, 2.0);
        x.as_mut_slice()[1] = Complex64::new(3.0, 4.0);
        x.as_mut_slice()[2] = Complex64::new(5.0, 6.0);
        x.as_mut_slice()[3] = Complex64::new(7.0, 8.0);
        x.sync_to_device();
        
        // Scale by i (0 + 1i)
        // (a+bi) * i = -b + ai
        backend.scale(Complex64::new(0.0, 1.0), &mut x);
        
        let result = x.as_slice();
        // (1+2i) * i = -2 + i
        assert!((result[0].re - (-2.0)).abs() < 1e-10);
        assert!((result[0].im - 1.0).abs() < 1e-10);
        // (3+4i) * i = -4 + 3i
        assert!((result[1].re - (-4.0)).abs() < 1e-10);
        assert!((result[1].im - 3.0).abs() < 1e-10);
        println!("✓ Complex scale works correctly!");
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gram_matrix() {
        let backend = CudaBackend::new();
        let grid = Grid2D::new(4, 4, 1.0, 1.0);  // 16 elements per vector
        
        // Create 3 vectors for X and 2 vectors for Y
        let mut x0 = backend.alloc_field(grid);
        let mut x1 = backend.alloc_field(grid);
        let mut x2 = backend.alloc_field(grid);
        
        let mut y0 = backend.alloc_field(grid);
        let mut y1 = backend.alloc_field(grid);
        
        // Fill with known values
        // x0 = [1, 0, 0, ..., 0] (unit vector)
        // x1 = [0, 1, 0, ..., 0]
        // x2 = [1, 1, 0, ..., 0] / sqrt(2)
        x0.as_mut_slice()[0] = Complex64::new(1.0, 0.0);
        x1.as_mut_slice()[1] = Complex64::new(1.0, 0.0);
        let s = 1.0 / 2.0_f64.sqrt();
        x2.as_mut_slice()[0] = Complex64::new(s, 0.0);
        x2.as_mut_slice()[1] = Complex64::new(s, 0.0);
        
        // y0 = [1, 0, 0, ...] 
        // y1 = [i, 0, 0, ...]
        y0.as_mut_slice()[0] = Complex64::new(1.0, 0.0);
        y1.as_mut_slice()[0] = Complex64::new(0.0, 1.0);
        
        x0.sync_to_device();
        x1.sync_to_device();
        x2.sync_to_device();
        y0.sync_to_device();
        y1.sync_to_device();
        
        let x_vecs: Vec<CudaField> = vec![x0, x1, x2];
        let y_vecs: Vec<CudaField> = vec![y0, y1];
        
        // Compute Gram matrix G_ij = <x_i, y_j>
        let gram = backend.gram_matrix(&x_vecs, &y_vecs);
        
        // Expected (3x2 matrix, row-major):
        // G[0,0] = <x0, y0> = conj(1)*1 = 1
        // G[0,1] = <x0, y1> = conj(1)*i = i
        // G[1,0] = <x1, y0> = 0
        // G[1,1] = <x1, y1> = 0
        // G[2,0] = <x2, y0> = conj(s)*1 = s
        // G[2,1] = <x2, y1> = conj(s)*i = s*i
        
        assert_eq!(gram.len(), 6);
        assert!((gram[0] - Complex64::new(1.0, 0.0)).norm() < 1e-10, "G[0,0] = {:?}", gram[0]);
        assert!((gram[1] - Complex64::new(0.0, 1.0)).norm() < 1e-10, "G[0,1] = {:?}", gram[1]);
        assert!((gram[2] - Complex64::new(0.0, 0.0)).norm() < 1e-10, "G[1,0] = {:?}", gram[2]);
        assert!((gram[3] - Complex64::new(0.0, 0.0)).norm() < 1e-10, "G[1,1] = {:?}", gram[3]);
        assert!((gram[4] - Complex64::new(s, 0.0)).norm() < 1e-10, "G[2,0] = {:?}", gram[4]);
        assert!((gram[5] - Complex64::new(0.0, s)).norm() < 1e-10, "G[2,1] = {:?}", gram[5]);
        
        println!("✓ Gram matrix (ZGEMM) works correctly!");
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gram_matrix_hermitian() {
        // Test that G = V^H * V is Hermitian for the same set of vectors
        let backend = CudaBackend::new();
        let grid = Grid2D::new(4, 4, 1.0, 1.0);
        
        let mut v0 = backend.alloc_field(grid);
        let mut v1 = backend.alloc_field(grid);
        let mut v2 = backend.alloc_field(grid);
        
        // Random-ish complex vectors
        v0.as_mut_slice()[0] = Complex64::new(1.0, 2.0);
        v0.as_mut_slice()[1] = Complex64::new(3.0, 4.0);
        v1.as_mut_slice()[0] = Complex64::new(5.0, 6.0);
        v1.as_mut_slice()[2] = Complex64::new(7.0, 8.0);
        v2.as_mut_slice()[1] = Complex64::new(9.0, 10.0);
        v2.as_mut_slice()[3] = Complex64::new(11.0, 12.0);
        
        v0.sync_to_device();
        v1.sync_to_device();
        v2.sync_to_device();
        
        let vecs: Vec<CudaField> = vec![v0, v1, v2];
        
        // Compute G = V^H * V (same vectors for X and Y)
        let gram = backend.gram_matrix(&vecs, &vecs);
        
        // G should be 3x3 Hermitian: G_ij = conj(G_ji)
        assert_eq!(gram.len(), 9);
        
        // Check Hermitian property
        for i in 0..3 {
            for j in 0..3 {
                let g_ij = gram[i * 3 + j];
                let g_ji = gram[j * 3 + i];
                assert!(
                    (g_ij - g_ji.conj()).norm() < 1e-10,
                    "G[{},{}] = {:?} != conj(G[{},{}]) = {:?}",
                    i, j, g_ij, j, i, g_ji.conj()
                );
            }
        }
        
        // Diagonal should be real and positive (squared norms)
        for i in 0..3 {
            let g_ii = gram[i * 3 + i];
            assert!(g_ii.im.abs() < 1e-10, "G[{},{}] should be real: {:?}", i, i, g_ii);
            assert!(g_ii.re >= 0.0, "G[{},{}] should be non-negative: {:?}", i, i, g_ii);
        }
        
        println!("✓ Gram matrix is Hermitian!");
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_linear_combinations() {
        let backend = CudaBackend::new();
        let grid = Grid2D::new(4, 4, 1.0, 1.0);  // 16 elements per vector
        
        // Create 2 input vectors (r=2) and compute 3 outputs (m=3)
        let mut q0 = backend.alloc_field(grid);
        let mut q1 = backend.alloc_field(grid);
        
        // q0 = [1, 0, 0, ..., 0] 
        // q1 = [0, 1, 0, ..., 0]
        q0.as_mut_slice()[0] = Complex64::new(1.0, 0.0);
        q1.as_mut_slice()[1] = Complex64::new(1.0, 0.0);
        q0.sync_to_device();
        q1.sync_to_device();
        
        let q_vecs: Vec<CudaField> = vec![q0, q1];
        
        // Coefficients (r=2, m=3), column-major: coeffs[i + j*r] = C[i,j]
        // C = [[1, 2, 3],   <- row 0 (for q0)
        //      [4, 5, 6]]   <- row 1 (for q1)
        // 
        // Column-major layout: [C[0,0], C[1,0], C[0,1], C[1,1], C[0,2], C[1,2]]
        //                    = [1, 4, 2, 5, 3, 6]
        //
        // Output[j] = sum_i C[i,j] * q[i]
        // out0 = 1*q0 + 4*q1 = [1, 4, 0, ...]
        // out1 = 2*q0 + 5*q1 = [2, 5, 0, ...]
        // out2 = 3*q0 + 6*q1 = [3, 6, 0, ...]
        let coeffs = vec![
            Complex64::new(1.0, 0.0), Complex64::new(4.0, 0.0),  // column 0
            Complex64::new(2.0, 0.0), Complex64::new(5.0, 0.0),  // column 1
            Complex64::new(3.0, 0.0), Complex64::new(6.0, 0.0),  // column 2
        ];
        
        let outputs = backend.linear_combinations(&q_vecs, &coeffs, 3);
        
        assert_eq!(outputs.len(), 3);
        
        // Check out0 = [1, 4, 0, ...]
        let out0 = outputs[0].as_slice();
        assert!((out0[0] - Complex64::new(1.0, 0.0)).norm() < 1e-10, "out0[0] = {:?}", out0[0]);
        assert!((out0[1] - Complex64::new(4.0, 0.0)).norm() < 1e-10, "out0[1] = {:?}", out0[1]);
        for i in 2..16 {
            assert!((out0[i] - Complex64::ZERO).norm() < 1e-10, "out0[{}] = {:?}", i, out0[i]);
        }
        
        // Check out1 = [2, 5, 0, ...]
        let out1 = outputs[1].as_slice();
        assert!((out1[0] - Complex64::new(2.0, 0.0)).norm() < 1e-10, "out1[0] = {:?}", out1[0]);
        assert!((out1[1] - Complex64::new(5.0, 0.0)).norm() < 1e-10, "out1[1] = {:?}", out1[1]);
        
        // Check out2 = [3, 6, 0, ...]
        let out2 = outputs[2].as_slice();
        assert!((out2[0] - Complex64::new(3.0, 0.0)).norm() < 1e-10, "out2[0] = {:?}", out2[0]);
        assert!((out2[1] - Complex64::new(6.0, 0.0)).norm() < 1e-10, "out2[1] = {:?}", out2[1]);
        
        println!("✓ Linear combinations (ZGEMM) works correctly!");
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_linear_combinations_complex() {
        let backend = CudaBackend::new();
        let grid = Grid2D::new(2, 2, 1.0, 1.0);  // 4 elements per vector
        
        // Create 2 input vectors with complex values
        let mut q0 = backend.alloc_field(grid);
        let mut q1 = backend.alloc_field(grid);
        
        // q0 = [1+i, 0, 0, 0]
        // q1 = [0, 2-i, 0, 0]
        q0.as_mut_slice()[0] = Complex64::new(1.0, 1.0);
        q1.as_mut_slice()[1] = Complex64::new(2.0, -1.0);
        q0.sync_to_device();
        q1.sync_to_device();
        
        let q_vecs: Vec<CudaField> = vec![q0, q1];
        
        // Coefficients with complex values (r=2, m=1)
        // out0 = (1+2i)*q0 + (3-i)*q1
        //      = (1+2i)*(1+i) + (3-i)*(2-i) at positions 0,1
        //      = (1+i+2i+2i²) + (6-3i-2i+i²) 
        //      = (1+3i-2) + (6-5i-1)
        //      = (-1+3i) + (5-5i)
        //      at pos 0: (-1+3i)
        //      at pos 1: (5-5i)
        let coeffs = vec![
            Complex64::new(1.0, 2.0),   // coeff for q0
            Complex64::new(3.0, -1.0),  // coeff for q1
        ];
        
        let outputs = backend.linear_combinations(&q_vecs, &coeffs, 1);
        
        assert_eq!(outputs.len(), 1);
        
        let out = outputs[0].as_slice();
        // (1+2i)*(1+i) = 1 + i + 2i + 2i² = 1 + 3i - 2 = -1 + 3i
        let expected0 = Complex64::new(1.0, 2.0) * Complex64::new(1.0, 1.0);
        // (3-i)*(2-i) = 6 - 3i - 2i + i² = 6 - 5i - 1 = 5 - 5i
        let expected1 = Complex64::new(3.0, -1.0) * Complex64::new(2.0, -1.0);
        
        assert!((out[0] - expected0).norm() < 1e-10, "out[0] = {:?}, expected {:?}", out[0], expected0);
        assert!((out[1] - expected1).norm() < 1e-10, "out[1] = {:?}, expected {:?}", out[1], expected1);
        
        println!("✓ Linear combinations with complex coefficients works!");
    }
}
