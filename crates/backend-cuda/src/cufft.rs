//! Safe Rust wrapper for cuFFT 2D complex double-precision transforms.
//!
//! This module provides a safe, RAII-based interface to cuFFT for performing
//! 2D Z2Z (complex double) FFT transforms on GPU data.
//!
//! # Example
//!
//! ```ignore
//! let plan = CufftPlan2d::new(64, 64)?;
//! plan.set_stream(stream)?;
//! plan.forward(data)?;  // In-place forward FFT
//! plan.inverse(data)?;  // In-place inverse FFT (remember to normalize!)
//! ```

use crate::cufft_sys::{
    cufftDestroy, cufftExecZ2Z, cufftHandle, cufftPlan2d, cufftSetStream,
    cufftType, cudaStream_t, CufftError, CUFFT_FORWARD, CUFFT_INVERSE,
    cufftDoubleComplex,
};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtrMut};
use std::sync::Arc;

/// A 2D cuFFT plan for complex double-precision (Z2Z) transforms.
///
/// This type manages the lifecycle of a cuFFT plan handle, automatically
/// destroying it when dropped. Plans should be reused for multiple transforms
/// of the same size for best performance.
///
/// # Thread Safety
///
/// cuFFT plans are thread-safe for execution but not for creation/destruction.
/// This type is `Send` but not `Sync`.
pub struct CufftPlan2d {
    handle: cufftHandle,
    nx: usize,
    ny: usize,
}

// CufftPlan2d can be sent between threads but shouldn't be shared
// (cuFFT documentation says plans can be executed from any thread,
// but creation/destruction should be serialized)
unsafe impl Send for CufftPlan2d {}

impl CufftPlan2d {
    /// Creates a new 2D FFT plan for complex double-precision transforms.
    ///
    /// # Parameters
    /// - `nx`: Number of rows (slowest changing dimension)
    /// - `ny`: Number of columns (fastest changing dimension, contiguous in memory)
    ///
    /// # Returns
    /// A new plan on success, or a `CufftError` if plan creation fails.
    ///
    /// # Notes
    /// - Plan creation involves JIT compilation and can be slow on first call
    /// - Plans are cached internally by CUDA, subsequent same-size plans are faster
    /// - The plan is configured for in-place transforms by default
    pub fn new(nx: usize, ny: usize) -> Result<Self, CufftError> {
        let mut handle: cufftHandle = 0;
        unsafe {
            let result = cufftPlan2d(
                &mut handle,
                nx as i32,
                ny as i32,
                cufftType::CUFFT_Z2Z,
            );
            result.to_result()?;
        }
        Ok(Self { handle, nx, ny })
    }

    /// Associates this plan with a CUDA stream.
    ///
    /// All FFT executions will use this stream, enabling overlap with other
    /// GPU operations. If not called, the default stream (0) is used.
    ///
    /// # Parameters
    /// - `stream`: The CUDA stream to associate with this plan
    ///
    /// # Returns
    /// `Ok(())` on success, or a `CufftError` if stream association fails.
    pub fn set_stream(&self, stream: &Arc<CudaStream>) -> Result<(), CufftError> {
        // Get the raw CUDA stream pointer from cudarc
        let stream_ptr = stream.cu_stream() as cudaStream_t;
        unsafe { 
            let result = cufftSetStream(self.handle, stream_ptr);
            result.to_result()
        }
    }

    /// Returns the grid dimensions (nx, ny) for this plan.
    #[allow(dead_code)]
    pub fn dims(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }

    /// Executes an in-place forward FFT on the given device data.
    ///
    /// # Parameters
    /// - `data`: Device buffer containing `nx * ny` complex doubles as `[re, im, ...]`
    /// - `stream`: The CUDA stream for obtaining mutable device pointer
    ///
    /// # Returns
    /// `Ok(())` on success, or a `CufftError` if execution fails.
    ///
    /// # Panics
    /// Panics if `data.len() != 2 * nx * ny` (f64 pairs for complex numbers).
    pub fn forward(&self, data: &mut CudaSlice<f64>, stream: &Arc<CudaStream>) -> Result<(), CufftError> {
        let expected_len = self.nx * self.ny * 2; // f64 pairs
        assert_eq!(
            data.len(),
            expected_len,
            "Data length {} doesn't match plan size {}×{} (expected {} f64s)",
            data.len(),
            self.nx,
            self.ny,
            expected_len
        );

        let ptr = data.device_ptr_mut(stream).0 as *mut cufftDoubleComplex;
        unsafe {
            let result = cufftExecZ2Z(
                self.handle,
                ptr,
                ptr, // in-place
                CUFFT_FORWARD,
            );
            result.to_result()
        }
    }

    /// Executes an in-place inverse FFT on the given device data.
    ///
    /// # Parameters
    /// - `data`: Device buffer containing `nx * ny` complex doubles as `[re, im, ...]`
    /// - `stream`: The CUDA stream for obtaining mutable device pointer
    ///
    /// # Returns
    /// `Ok(())` on success, or a `CufftError` if execution fails.
    ///
    /// # Note
    /// cuFFT inverse transforms are **unnormalized**. You must divide each
    /// element by `nx * ny` after this call to get the proper inverse.
    ///
    /// # Panics
    /// Panics if `data.len() != 2 * nx * ny` (f64 pairs for complex numbers).
    pub fn inverse(&self, data: &mut CudaSlice<f64>, stream: &Arc<CudaStream>) -> Result<(), CufftError> {
        let expected_len = self.nx * self.ny * 2; // f64 pairs
        assert_eq!(
            data.len(),
            expected_len,
            "Data length {} doesn't match plan size {}×{} (expected {} f64s)",
            data.len(),
            self.nx,
            self.ny,
            expected_len
        );

        let ptr = data.device_ptr_mut(stream).0 as *mut cufftDoubleComplex;
        unsafe {
            let result = cufftExecZ2Z(
                self.handle,
                ptr,
                ptr, // in-place
                CUFFT_INVERSE,
            );
            result.to_result()
        }
    }
}

impl Drop for CufftPlan2d {
    fn drop(&mut self) {
        // Ignore errors on destruction - not much we can do about them
        unsafe {
            let _ = cufftDestroy(self.handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_creation() {
        // This test requires CUDA to be available
        if cudarc::driver::CudaContext::new(0).is_err() {
            eprintln!("Skipping test_plan_creation: CUDA not available");
            return;
        }

        let plan = CufftPlan2d::new(64, 64);
        assert!(plan.is_ok(), "Failed to create cuFFT plan: {:?}", plan.err());
        
        let plan = plan.unwrap();
        assert_eq!(plan.dims(), (64, 64));
        println!("✓ cuFFT plan creation works!");
    }

    #[test]
    fn test_plan_various_sizes() {
        if cudarc::driver::CudaContext::new(0).is_err() {
            eprintln!("Skipping test_plan_various_sizes: CUDA not available");
            return;
        }

        // Test power-of-2 sizes
        for size in [8, 16, 32, 64, 128, 256] {
            let plan = CufftPlan2d::new(size, size);
            assert!(plan.is_ok(), "Failed for size {}: {:?}", size, plan.err());
        }

        // Test non-square sizes
        let plan = CufftPlan2d::new(64, 128);
        assert!(plan.is_ok(), "Failed for 64×128: {:?}", plan.err());

        // Test non-power-of-2 sizes (cuFFT supports these)
        let plan = CufftPlan2d::new(48, 48);
        assert!(plan.is_ok(), "Failed for 48×48: {:?}", plan.err());

        println!("✓ cuFFT plan creation works for various sizes!");
    }
}
