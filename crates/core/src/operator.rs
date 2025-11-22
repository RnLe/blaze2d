//! Placeholder operator scaffolding.

use num_complex::Complex64;

use crate::{backend::SpectralBackend, dielectric::Dielectric2D, polarization::Polarization};

pub struct ThetaOperator<B: SpectralBackend> {
    pub backend: B,
    pub dielectric: Dielectric2D,
    pub polarization: Polarization,
    pub kx: f64,
    pub ky: f64,
}

impl<B: SpectralBackend> ThetaOperator<B> {
    pub fn apply(&self, _input: &B::Buffer, _output: &mut B::Buffer) {
        // Implementation arrives in later phases.
    }

    pub fn eigenvalue_hint(&self) -> Complex64 {
        Complex64::new(self.kx * self.kx + self.ky * self.ky, 0.0)
    }
}
