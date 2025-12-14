//! Simple operator preconditioners shared by eigensolvers.

use crate::backend::{SpectralBackend, SpectralBuffer};

pub(crate) const FOURIER_DIAGONAL_SHIFT: f64 = 1e-3;

pub trait OperatorPreconditioner<B: SpectralBackend> {
    fn apply(&mut self, backend: &B, buffer: &mut B::Buffer);
}

#[derive(Debug, Clone)]
pub struct FourierDiagonalPreconditioner {
    inverse_diagonal: Vec<f64>,
}

impl FourierDiagonalPreconditioner {
    pub(crate) fn new(inverse_diagonal: Vec<f64>) -> Self {
        Self { inverse_diagonal }
    }
}

impl<B: SpectralBackend> OperatorPreconditioner<B> for FourierDiagonalPreconditioner {
    fn apply(&mut self, backend: &B, buffer: &mut B::Buffer) {
        backend.forward_fft_2d(buffer);
        for (value, scale) in buffer
            .as_mut_slice()
            .iter_mut()
            .zip(self.inverse_diagonal.iter())
        {
            *value *= *scale;
        }
        backend.inverse_fft_2d(buffer);
    }
}
