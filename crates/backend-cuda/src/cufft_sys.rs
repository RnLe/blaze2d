//! Raw FFI bindings to NVIDIA cuFFT library.
//!
//! This module provides low-level C bindings to the cuFFT functions needed
//! for 2D complex double-precision (Z2Z) transforms.
//!
//! cuFFT documentation: https://docs.nvidia.com/cuda/cufft/

#![allow(non_camel_case_types)]
#![allow(dead_code)]

use std::ffi::c_int;

/// cuFFT plan handle.
pub type cufftHandle = c_int;

/// cuFFT result/error codes.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cufftResult {
    CUFFT_SUCCESS = 0,
    CUFFT_INVALID_PLAN = 1,
    CUFFT_ALLOC_FAILED = 2,
    CUFFT_INVALID_TYPE = 3,
    CUFFT_INVALID_VALUE = 4,
    CUFFT_INTERNAL_ERROR = 5,
    CUFFT_EXEC_FAILED = 6,
    CUFFT_SETUP_FAILED = 7,
    CUFFT_INVALID_SIZE = 8,
    CUFFT_UNALIGNED_DATA = 9,
    CUFFT_INCOMPLETE_PARAMETER_LIST = 10,
    CUFFT_INVALID_DEVICE = 11,
    CUFFT_PARSE_ERROR = 12,
    CUFFT_NO_WORKSPACE = 13,
    CUFFT_NOT_IMPLEMENTED = 14,
    CUFFT_LICENSE_ERROR = 15,
    CUFFT_NOT_SUPPORTED = 16,
}

impl cufftResult {
    /// Returns true if the result indicates success.
    pub fn is_success(self) -> bool {
        self == cufftResult::CUFFT_SUCCESS
    }

    /// Convert to a Result type.
    pub fn to_result(self) -> Result<(), CufftError> {
        if self.is_success() {
            Ok(())
        } else {
            Err(CufftError(self))
        }
    }
}

/// cuFFT transform types.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cufftType {
    /// Real to Complex (interleaved) - single precision
    CUFFT_R2C = 0x2a,
    /// Complex (interleaved) to Real - single precision
    CUFFT_C2R = 0x2c,
    /// Complex to Complex (interleaved) - single precision
    CUFFT_C2C = 0x29,
    /// Double to Double-Complex (interleaved)
    CUFFT_D2Z = 0x6a,
    /// Double-Complex (interleaved) to Double
    CUFFT_Z2D = 0x6c,
    /// Double-Complex to Double-Complex (interleaved)
    CUFFT_Z2Z = 0x69,
}

/// Direction constants for cuFFT transforms.
pub const CUFFT_FORWARD: c_int = -1;
pub const CUFFT_INVERSE: c_int = 1;

/// Double-precision complex number (matches cuDoubleComplex).
/// 
/// This type has the same memory layout as `num_complex::Complex64`:
/// `[real: f64, imag: f64]` with C representation.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct cufftDoubleComplex {
    pub x: f64,  // real part
    pub y: f64,  // imaginary part
}

/// CUDA stream type (opaque pointer).
/// We use the raw pointer type for FFI compatibility.
pub type cudaStream_t = *mut std::ffi::c_void;

// Link against the cuFFT library.
// The actual library path is handled by build.rs or system configuration.
#[link(name = "cufft")]
unsafe extern "C" {
    /// Creates a 2D FFT plan.
    ///
    /// # Parameters
    /// - `plan`: Pointer to store the created plan handle
    /// - `nx`: Transform size in the x dimension (rows, slowest changing)
    /// - `ny`: Transform size in the y dimension (columns, fastest changing/contiguous)
    /// - `fft_type`: The transform data type (e.g., CUFFT_Z2Z)
    ///
    /// # Returns
    /// `CUFFT_SUCCESS` on success, or an error code.
    pub fn cufftPlan2d(
        plan: *mut cufftHandle,
        nx: c_int,
        ny: c_int,
        fft_type: cufftType,
    ) -> cufftResult;

    /// Executes a double-precision complex-to-complex transform.
    ///
    /// # Parameters
    /// - `plan`: The plan handle created by `cufftPlan2d`
    /// - `idata`: Pointer to input data (GPU memory)
    /// - `odata`: Pointer to output data (GPU memory, can be same as idata for in-place)
    /// - `direction`: `CUFFT_FORWARD` (-1) or `CUFFT_INVERSE` (1)
    ///
    /// # Returns
    /// `CUFFT_SUCCESS` on success, or an error code.
    ///
    /// # Note
    /// cuFFT is **unnormalized**: forward followed by inverse produces N*x, not x.
    /// The caller must apply 1/(nx*ny) normalization after inverse transform.
    pub fn cufftExecZ2Z(
        plan: cufftHandle,
        idata: *mut cufftDoubleComplex,
        odata: *mut cufftDoubleComplex,
        direction: c_int,
    ) -> cufftResult;

    /// Associates a CUDA stream with a cuFFT plan.
    ///
    /// All kernel launches for the plan will use this stream, enabling
    /// overlap with other operations.
    ///
    /// # Parameters
    /// - `plan`: The plan handle
    /// - `stream`: CUDA stream handle (0 for default stream)
    ///
    /// # Returns
    /// `CUFFT_SUCCESS` on success, or an error code.
    pub fn cufftSetStream(
        plan: cufftHandle,
        stream: cudaStream_t,
    ) -> cufftResult;

    /// Destroys a cuFFT plan and releases associated resources.
    ///
    /// # Parameters
    /// - `plan`: The plan handle to destroy
    ///
    /// # Returns
    /// `CUFFT_SUCCESS` on success, or an error code.
    pub fn cufftDestroy(plan: cufftHandle) -> cufftResult;
}

/// Error type for cuFFT operations.
#[derive(Debug, Clone, Copy)]
pub struct CufftError(pub cufftResult);

impl std::fmt::Display for CufftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cuFFT error: {:?}", self.0)
    }
}

impl std::error::Error for CufftError {}
