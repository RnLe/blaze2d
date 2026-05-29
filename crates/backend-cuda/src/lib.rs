//! CUDA backend — honest typed stub.
//!
//! Blaze2D's GPU path is not currently maintained. This crate keeps the
//! `CudaBackend` name reserved so downstream call sites that select a backend
//! at the type level still compile, but every method panics if you actually
//! invoke it. See `docs/roadmap.md` (§ "GPU revival") for the plan to bring
//! a real implementation back.
//!
//! The struct deliberately mirrors `blaze2d_backend_cpu::CpuBackend`'s
//! constructor surface so swapping backends only requires changing a type
//! alias once revival lands.

use blaze2d_core::backend::SpectralBackend;
use blaze2d_core::field::{Field2D, FieldScalar};
use blaze2d_core::grid::Grid2D;
use num_complex::Complex64;

const NOT_IMPLEMENTED: &str =
    "blaze2d-backend-cuda is a typed stub (see docs/roadmap.md §GPU revival)";

#[derive(Debug, Clone, Default)]
pub struct CudaBackend;

impl CudaBackend {
    pub fn new() -> Self {
        Self
    }
}

impl SpectralBackend for CudaBackend {
    type Real = f64;
    type Buffer = Field2D;

    fn alloc_field(&self, _grid: Grid2D) -> Self::Buffer {
        unimplemented!("{NOT_IMPLEMENTED}");
    }

    fn forward_fft_2d(&self, _buffer: &mut Self::Buffer) {
        unimplemented!("{NOT_IMPLEMENTED}");
    }

    fn inverse_fft_2d(&self, _buffer: &mut Self::Buffer) {
        unimplemented!("{NOT_IMPLEMENTED}");
    }

    fn scale(&self, _alpha: Complex64, _buffer: &mut Self::Buffer) {
        unimplemented!("{NOT_IMPLEMENTED}");
    }

    fn axpy(&self, _alpha: Complex64, _x: &Self::Buffer, _y: &mut Self::Buffer) {
        unimplemented!("{NOT_IMPLEMENTED}");
    }

    fn dot(&self, _x: &Self::Buffer, _y: &Self::Buffer) -> Complex64 {
        unimplemented!("{NOT_IMPLEMENTED}");
    }
}

// Silence unused-import warnings when downstream code touches the FieldScalar
// path via the trait without naming it explicitly.
#[allow(dead_code)]
fn _assert_buffer_shape(_: &dyn Fn() -> FieldScalar) {}
