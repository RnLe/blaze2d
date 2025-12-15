//! Simple operator preconditioners shared by eigensolvers.

use crate::backend::{SpectralBackend, SpectralBuffer};

pub(crate) const FOURIER_DIAGONAL_SHIFT: f64 = 1e-3;

pub trait OperatorPreconditioner<B: SpectralBackend> {
    fn apply(&mut self, backend: &B, buffer: &mut B::Buffer);
}

#[derive(Debug, Clone)]
pub struct FourierDiagonalPreconditioner {
    inverse_diagonal: Vec<f64>,
    spatial_weights: Option<Vec<f64>>,
}

impl FourierDiagonalPreconditioner {
    pub(crate) fn new(inverse_diagonal: Vec<f64>) -> Self {
        Self {
            inverse_diagonal,
            spatial_weights: None,
        }
    }

    pub(crate) fn with_weights(inverse_diagonal: Vec<f64>, spatial_weights: Vec<f64>) -> Self {
        Self {
            inverse_diagonal,
            spatial_weights: Some(spatial_weights),
        }
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn inverse_diagonal(&self) -> &[f64] {
        &self.inverse_diagonal
    }

    #[cfg(test)]
    pub(crate) fn spatial_weights(&self) -> Option<&[f64]> {
        self.spatial_weights.as_deref()
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
        if let Some(weights) = &self.spatial_weights {
            for (value, weight) in buffer.as_mut_slice().iter_mut().zip(weights.iter()) {
                *value *= *weight;
            }
        }
    }
}
