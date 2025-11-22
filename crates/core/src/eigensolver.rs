//! Eigensolver placeholder (Lanczos/LOBPCG planned).

use num_complex::Complex64;

use crate::backend::SpectralBackend;

pub struct EigenOptions {
    pub n_bands: usize,
    pub max_iter: usize,
    pub tol: f64,
}

impl Default for EigenOptions {
    fn default() -> Self {
        Self {
            n_bands: 8,
            max_iter: 200,
            tol: 1e-8,
        }
    }
}

pub struct EigenResult {
    pub omegas: Vec<f64>,
}

pub fn solve_lowest_eigenpairs<B: SpectralBackend>(_backend: &B, _opts: &EigenOptions) -> EigenResult {
    EigenResult { omegas: Vec::new() }
}
