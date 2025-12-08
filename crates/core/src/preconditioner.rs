//! Simple operator preconditioners shared by eigensolvers.

use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    dielectric::Dielectric2D,
};

pub trait OperatorPreconditioner<B: SpectralBackend> {
    fn apply(&mut self, backend: &B, buffer: &mut B::Buffer);
}

#[derive(Debug, Clone)]
pub struct RealSpaceJacobi {
    scales: Vec<f64>,
}

impl RealSpaceJacobi {
    pub fn from_dielectric(dielectric: &Dielectric2D) -> Self {
        Self {
            scales: dielectric.eps().to_vec(),
        }
    }
}

impl<B: SpectralBackend> OperatorPreconditioner<B> for RealSpaceJacobi {
    fn apply(&mut self, _backend: &B, buffer: &mut B::Buffer) {
        for (value, scale) in buffer.as_mut_slice().iter_mut().zip(&self.scales) {
            *value *= *scale;
        }
    }
}
