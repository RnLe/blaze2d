//! Maxwell operator for 2D photonic crystal band structure calculations.
//!
//! This module provides the `ThetaOperator` which implements the Maxwell curl-curl
//! operator for computing photonic band structures in 2D periodic dielectric structures.
//!
//! # Physical Background
//!
//! The operator solves Maxwell's equations for electromagnetic modes in periodic media:
//!
//! ```text
//! ∇ × (ε⁻¹ ∇ × H) = (ω/c)² H     (TE mode, H out of plane)
//! ∇ · (ε⁻¹ ∇ E_z) = (ω/c)² ε E_z  (TM mode, E out of plane)
//! ```
//!
//! With Bloch boundary conditions: H(r + R) = e^{ik·R} H(r).
//!
//! # Polarization Modes
//!
//! - **TM (Transverse Magnetic)**: E-field out of plane, H in-plane.
//!   Uses generalized eigenproblem A·x = λ·B·x where B = ε(r).
//!
//! - **TE (Transverse Electric)**: H-field out of plane, E in-plane.
//!   Uses standard eigenproblem A·x = λ·x (B = I).

use num_complex::{Complex, Complex64};

use crate::backend::{SpectralBackend, SpectralBuffer};
use crate::dielectric::{Dielectric2D, DielectricDerivative};
use crate::field::{Real, cscalar, czero};
use crate::grid::Grid2D;
use crate::operators::{K_PLUS_G_NEAR_ZERO_FLOOR, LinearOperator, TM_PRECONDITIONER_MASS_FRACTION};
use crate::polarization::Polarization;
use crate::preconditioners::{
    FourierDiagonalPreconditioner, SpectralStats, TransverseProjectionPreconditioner,
};

/// Promote a storage-precision complex slice to `Vec<Complex64>` for
/// snapshot data (always f64 for downstream precision).
#[inline]
fn to_complex64_vec<R: Real>(slice: &[Complex<R>]) -> Vec<Complex64> {
    slice
        .iter()
        .map(|c| Complex64::new(c.re.to_accum(), c.im.to_accum()))
        .collect()
}

// ============================================================================
// ThetaOperator - Maxwell Curl-Curl Operator
// ============================================================================

/// The Maxwell Θ operator for 2D photonic crystal band structure.
///
/// This operator implements:
/// - **TE mode**: Θ = -∇·(ε⁻¹∇) (standard eigenproblem, B = I)
/// - **TM mode**: Θ = -∇² (generalized eigenproblem, B = ε)
///
/// # Construction
///
/// The operator is constructed for a specific k-point (Bloch wavevector):
///
/// ```ignore
/// let theta = ThetaOperator::new(backend, dielectric, Polarization::TM, [0.0, 0.0]);
/// ```
///
/// # G-Vector Computation
///
/// G-vectors are computed using the reciprocal lattice basis:
/// G = n₁·b₁ + n₂·b₂, where b₁, b₂ are reciprocal lattice vectors.
/// This is essential for non-orthogonal lattices (hexagonal, oblique).
pub struct ThetaOperator<B: SpectralBackend> {
    backend: B,
    dielectric: Dielectric2D,
    polarization: Polarization,
    grid: Grid2D,
    bloch: [f64; 2],
    kx_shifted: Vec<f64>,
    ky_shifted: Vec<f64>,
    k_plus_g_x: Vec<f64>,
    k_plus_g_y: Vec<f64>,
    k_plus_g_sq: Vec<f64>,
    #[allow(dead_code)]
    k_plus_g_sq_min: f64,
    #[allow(dead_code)]
    k_plus_g_sq_min_raw: f64,
    #[allow(dead_code)]
    k_plus_g_floor_count: usize,
    scratch: B::Buffer,
    grad_x: B::Buffer,
    grad_y: B::Buffer,
    k_plus_g_was_clamped: Vec<bool>,
}

/// Snapshot of operator state for debugging and visualization.
#[derive(Debug, Clone)]
pub struct OperatorSnapshotData {
    pub grid: Grid2D,
    pub field_spatial: Vec<Complex64>,
    pub field_fourier: Vec<Complex64>,
    pub theta_spatial: Vec<Complex64>,
    pub theta_fourier: Vec<Complex64>,
    pub grad_x: Option<Vec<Complex64>>,
    pub grad_y: Option<Vec<Complex64>>,
    pub eps_grad_x: Option<Vec<Complex64>>,
    pub eps_grad_y: Option<Vec<Complex64>>,
}

impl OperatorSnapshotData {
    pub fn len(&self) -> usize {
        self.grid.len()
    }
}

// Physics notation: L, A, B = Maxwell operators; k = Bloch wavevector; R = atomic position vector.
// Snake-case is suppressed to preserve readability of derivative method names (dL_dk, d2L_dk2, dL_dR, etc.).
#[allow(non_snake_case)]
impl<B: SpectralBackend> ThetaOperator<B> {
    /// Create a new ThetaOperator for band structure calculation.
    ///
    /// This operator implements Maxwell's equations for 2D photonic crystals:
    /// - TM mode: Uses generalized eigenproblem A·x = λ·B·x where B = ε(r).
    /// - TE mode: B = I (standard eigenproblem).
    ///
    /// **Important:** The G-vectors are computed using the reciprocal lattice basis:
    ///   G = n1*b1 + n2*b2
    /// This is essential for non-orthogonal lattices (hexagonal, oblique).
    pub fn new(
        backend: B,
        dielectric: Dielectric2D,
        polarization: Polarization,
        bloch_k: [f64; 2],
    ) -> Self {
        let grid = dielectric.grid;

        // Get reciprocal lattice vectors from dielectric for proper G-vector computation
        let b1 = dielectric.reciprocal_b1();
        let b2 = dielectric.reciprocal_b2();

        // Build k+G tables using the reciprocal lattice basis
        let (
            k_plus_g_x,
            k_plus_g_y,
            k_plus_g_sq,
            k_plus_g_sq_min_raw,
            k_plus_g_sq_min,
            k_plus_g_floor_count,
            k_plus_g_was_clamped,
        ) = build_k_plus_g_tables_with_reciprocal(grid, b1, b2, bloch_k);

        // kx_shifted and ky_shifted derived from k_plus_g values
        let kx_shifted = k_plus_g_x.clone();
        let ky_shifted = k_plus_g_y.clone();

        let scratch = backend.alloc_field(grid);
        let grad_x = backend.alloc_field(grid);
        let grad_y = backend.alloc_field(grid);

        Self {
            backend,
            dielectric,
            polarization,
            grid,
            bloch: bloch_k,
            kx_shifted,
            ky_shifted,
            k_plus_g_x,
            k_plus_g_y,
            k_plus_g_sq,
            k_plus_g_sq_min,
            k_plus_g_sq_min_raw,
            k_plus_g_floor_count,
            scratch,
            grad_x,
            grad_y,
            k_plus_g_was_clamped,
        }
    }

    /// Returns whether this operator is at the Γ-point (k ≈ 0).
    ///
    /// At Γ, the G=0 mode has |k+G|² = 0 and must be deflated.
    /// Away from Γ, the G=0 mode has |k+G|² = |k|² > 0 and is a legitimate mode.
    pub fn is_gamma(&self) -> bool {
        const GAMMA_TOL: f64 = 1e-12;
        self.bloch[0].abs() < GAMMA_TOL && self.bloch[1].abs() < GAMMA_TOL
    }

    /// Allocate a new field buffer.
    pub fn alloc_field(&self) -> B::Buffer {
        self.backend.alloc_field(self.grid)
    }

    /// Get a reference to the backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Get a mutable reference to the backend.
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    /// Get the polarization mode.
    pub fn polarization(&self) -> Polarization {
        self.polarization
    }

    /// Get the grid.
    pub fn grid(&self) -> Grid2D {
        self.grid
    }

    /// Compute spectral statistics for the current k-point.
    pub fn spectral_stats(&self) -> SpectralStats {
        SpectralStats::compute(&self.k_plus_g_sq)
    }

    /// Build homogeneous preconditioner with k-dependent (adaptive) shift.
    pub fn build_homogeneous_preconditioner_adaptive(&self) -> FourierDiagonalPreconditioner {
        let stats = self.spectral_stats();
        let shift = stats.adaptive_shift();
        log::debug!(
            "preconditioner: adaptive shift σ(k)={:.2e} (α={:.1}, s_min={:.2e}, s_med={:.2e}, s_max={:.2e})",
            shift,
            crate::preconditioners::SHIFT_SMIN_FRACTION,
            stats.s_min,
            stats.s_median,
            stats.s_max
        );
        self.build_homogeneous_preconditioner_with_shift(shift)
    }

    /// Build homogeneous preconditioner with band-window-aware adaptive shift.
    ///
    /// This uses eigenvalue estimates from current LOBPCG iterations to compute
    /// a more targeted shift that focuses on the actual band window rather than
    /// the full geometric spectral range.
    ///
    /// # Arguments
    ///
    /// - `eigenvalues`: Current eigenvalue estimates from LOBPCG (λ = ω²).
    /// - `blend`: Blending factor β ∈ [0, 1]. β=1 uses only s_min, β=0 uses only band window.
    ///            Use `None` for the default value (0.5).
    /// - `band_scale`: Scaling factor c for band-window shift.
    ///            Use `None` for the default value (0.5).
    ///
    /// # Example
    ///
    /// ```ignore
    /// // After first few LOBPCG iterations, refine the preconditioner:
    /// let eigenvalues = solver.eigenvalues();
    /// let precond = theta.build_homogeneous_preconditioner_band_window(
    ///     eigenvalues,
    ///     Some(0.3),  // Favor band window (less s_min influence)
    ///     None,       // Use default band scale
    /// );
    /// ```
    pub fn build_homogeneous_preconditioner_band_window(
        &self,
        eigenvalues: &[f64],
        blend: Option<f64>,
        band_scale: Option<f64>,
    ) -> FourierDiagonalPreconditioner {
        let mut stats = self.spectral_stats();
        stats.set_band_window(eigenvalues);

        let blend = blend.unwrap_or(crate::preconditioners::DEFAULT_BAND_WINDOW_BLEND);
        let band_scale = band_scale.unwrap_or(crate::preconditioners::DEFAULT_BAND_WINDOW_SCALE);
        let shift = stats.adaptive_shift_blended(blend, band_scale);

        if let Some(ref window) = stats.band_window {
            log::debug!(
                "preconditioner: band-window shift σ(k)={:.2e} (blend={:.2}, scale={:.2}, λ=[{:.2e}..{:.2e}], s_min={:.2e})",
                shift,
                blend,
                band_scale,
                window.lambda_min,
                window.lambda_max,
                stats.s_min
            );
        } else {
            log::debug!(
                "preconditioner: band-window fallback (no valid eigenvalues) σ(k)={:.2e}",
                shift
            );
        }

        self.build_homogeneous_preconditioner_with_shift(shift)
    }

    /// Build homogeneous preconditioner with a specific shift value.
    fn build_homogeneous_preconditioner_with_shift(
        &self,
        shift: f64,
    ) -> FourierDiagonalPreconditioner {
        let near_zero_mask = if self.is_gamma() {
            Some(self.k_plus_g_was_clamped.as_slice())
        } else {
            None
        };

        match self.polarization {
            Polarization::TM => {
                let eps_eff = self.effective_tm_epsilon();
                let mass_floor = tm_preconditioner_mass_floor(eps_eff);
                let inverse_diagonal = build_inverse_diagonal(
                    &self.k_plus_g_sq,
                    shift,
                    eps_eff,
                    mass_floor,
                    near_zero_mask,
                );
                FourierDiagonalPreconditioner::new(inverse_diagonal)
            }
            Polarization::TE => {
                let eps_eff = self.effective_te_epsilon();
                let inverse_diagonal =
                    build_inverse_diagonal(&self.k_plus_g_sq, shift, eps_eff, 0.0, near_zero_mask);
                FourierDiagonalPreconditioner::new(inverse_diagonal)
            }
        }
    }

    /// Build the MPB-style transverse-projection preconditioner with adaptive shift.
    pub fn build_transverse_projection_preconditioner_adaptive(
        &self,
    ) -> TransverseProjectionPreconditioner<B> {
        let stats = self.spectral_stats();
        let shift = stats.adaptive_shift();
        log::debug!(
            "transverse-projection preconditioner: adaptive shift σ(k)={:.2e}",
            shift
        );
        self.build_transverse_projection_preconditioner_with_shift(shift)
    }

    /// Build the transverse-projection preconditioner with band-window-aware adaptive shift.
    ///
    /// Similar to `build_homogeneous_preconditioner_band_window`, but for the
    /// MPB-style transverse-projection preconditioner.
    pub fn build_transverse_projection_preconditioner_band_window(
        &self,
        eigenvalues: &[f64],
        blend: Option<f64>,
        band_scale: Option<f64>,
    ) -> TransverseProjectionPreconditioner<B> {
        let mut stats = self.spectral_stats();
        stats.set_band_window(eigenvalues);

        let blend = blend.unwrap_or(crate::preconditioners::DEFAULT_BAND_WINDOW_BLEND);
        let band_scale = band_scale.unwrap_or(crate::preconditioners::DEFAULT_BAND_WINDOW_SCALE);
        let shift = stats.adaptive_shift_blended(blend, band_scale);

        if let Some(ref window) = stats.band_window {
            log::debug!(
                "transverse-projection preconditioner: band-window shift σ(k)={:.2e} (λ_med={:.2e})",
                shift,
                window.lambda_median
            );
        }

        self.build_transverse_projection_preconditioner_with_shift(shift)
    }

    /// Build the transverse-projection preconditioner with a specific shift value.
    fn build_transverse_projection_preconditioner_with_shift(
        &self,
        shift: f64,
    ) -> TransverseProjectionPreconditioner<B> {
        TransverseProjectionPreconditioner::new(
            &self.backend,
            &self.dielectric,
            self.polarization,
            self.k_plus_g_x.clone(),
            self.k_plus_g_y.clone(),
            self.k_plus_g_sq.clone(),
            self.k_plus_g_was_clamped.clone(),
            shift,
        )
    }

    /// Estimate the condition number κ = λ_max / λ_min of the operator A.
    pub fn estimate_condition_number(&mut self, n_iters: usize) -> (f64, f64, f64) {
        let mut v = self.alloc_field();
        let mut av = self.alloc_field();

        // Initialize with pseudo-random vector
        for (i, val) in v.as_mut_slice().iter_mut().enumerate() {
            let re = (i as f64 * 0.618033988749895).sin();
            let im = (i as f64 * 0.414213562373095).cos();
            *val = cscalar::<B::Real>(re, im);
        }

        // Normalize (accumulate in f64)
        let norm: f64 = v
            .as_slice()
            .iter()
            .map(|c| c.re.to_accum().powi(2) + c.im.to_accum().powi(2))
            .sum::<f64>()
            .sqrt();
        if norm > 1e-15 {
            let norm_r = B::Real::from_accum(norm);
            for val in v.as_mut_slice().iter_mut() {
                *val /= norm_r;
            }
        }

        let mut lambda_max = 0.0;
        for _ in 0..n_iters {
            self.apply(&v, &mut av);
            // Accumulate Rayleigh quotient in f64
            let numerator: f64 = v
                .as_slice()
                .iter()
                .zip(av.as_slice().iter())
                .map(|(vi, avi)| {
                    let vi_re = vi.re.to_accum();
                    let vi_im = vi.im.to_accum();
                    let avi_re = avi.re.to_accum();
                    let avi_im = avi.im.to_accum();
                    vi_re * avi_re + vi_im * avi_im // Re(conj(vi) * avi)
                })
                .sum();
            lambda_max = numerator;

            let norm: f64 = av
                .as_slice()
                .iter()
                .map(|c| c.re.to_accum().powi(2) + c.im.to_accum().powi(2))
                .sum::<f64>()
                .sqrt();
            if norm > 1e-15 {
                let norm_r = B::Real::from_accum(norm);
                for val in av.as_mut_slice().iter_mut() {
                    *val /= norm_r;
                }
            }
            std::mem::swap(&mut v, &mut av);
        }

        let lambda_min = self.k_plus_g_sq_min;
        let kappa = if lambda_min > 1e-15 {
            lambda_max / lambda_min
        } else {
            f64::INFINITY
        };

        (lambda_max, lambda_min, kappa)
    }

    /// Estimate the condition number of the preconditioned operator M⁻¹A.
    pub fn estimate_preconditioned_condition_number(
        &mut self,
        precond: &mut dyn crate::preconditioners::OperatorPreconditioner<B>,
        n_iters: usize,
    ) -> (f64, f64, f64) {
        let mut v = self.alloc_field();
        let mut av = self.alloc_field();

        for (i, val) in v.as_mut_slice().iter_mut().enumerate() {
            let re = (i as f64 * 0.618033988749895).sin();
            let im = (i as f64 * 0.414213562373095).cos();
            *val = cscalar::<B::Real>(re, im);
        }

        let norm: f64 = v
            .as_slice()
            .iter()
            .map(|c| c.re.to_accum().powi(2) + c.im.to_accum().powi(2))
            .sum::<f64>()
            .sqrt();
        if norm > 1e-15 {
            let norm_r = B::Real::from_accum(norm);
            for val in v.as_mut_slice().iter_mut() {
                *val /= norm_r;
            }
        }

        let mut lambda_max = 0.0;
        for _ in 0..n_iters {
            self.apply(&v, &mut av);
            precond.apply(&self.backend, &mut av);

            let numerator: f64 = v
                .as_slice()
                .iter()
                .zip(av.as_slice().iter())
                .map(|(vi, avi)| {
                    let vi_re = vi.re.to_accum();
                    let vi_im = vi.im.to_accum();
                    let avi_re = avi.re.to_accum();
                    let avi_im = avi.im.to_accum();
                    vi_re * avi_re + vi_im * avi_im
                })
                .sum();
            lambda_max = numerator;

            let norm: f64 = av
                .as_slice()
                .iter()
                .map(|c| c.re.to_accum().powi(2) + c.im.to_accum().powi(2))
                .sum::<f64>()
                .sqrt();
            if norm > 1e-15 {
                let norm_r = B::Real::from_accum(norm);
                for val in av.as_mut_slice().iter_mut() {
                    *val /= norm_r;
                }
            }
            std::mem::swap(&mut v, &mut av);
        }

        let lambda_min_approx = 1.0;
        let kappa = if lambda_min_approx > 1e-15 {
            lambda_max / lambda_min_approx
        } else {
            f64::INFINITY
        };

        (lambda_max, lambda_min_approx, kappa)
    }

    /// Check self-adjointness of the operator.
    pub fn check_self_adjointness(&mut self) -> f64 {
        let mut x = self.alloc_field();
        let mut y = self.alloc_field();
        let mut ax = self.alloc_field();
        let mut ay = self.alloc_field();

        for (i, val) in x.as_mut_slice().iter_mut().enumerate() {
            let re = (i as f64 * 0.618033988749895).sin();
            let im = (i as f64 * 0.414213562373095).cos();
            *val = cscalar::<B::Real>(re, im);
        }
        for (i, val) in y.as_mut_slice().iter_mut().enumerate() {
            let re = (i as f64 * std::f64::consts::SQRT_2).cos();
            let im = (i as f64 * 3.0_f64.sqrt()).sin();
            *val = cscalar::<B::Real>(re, im);
        }

        self.apply(&x, &mut ax);
        self.apply(&y, &mut ay);

        // Compute inner products in f64 for accuracy
        let ax_y: Complex64 = ax
            .as_slice()
            .iter()
            .zip(y.as_slice().iter())
            .map(|(a, b)| {
                let a64 = Complex64::new(a.re.to_accum(), a.im.to_accum());
                let b64 = Complex64::new(b.re.to_accum(), b.im.to_accum());
                a64.conj() * b64
            })
            .sum();
        let x_ay: Complex64 = x
            .as_slice()
            .iter()
            .zip(ay.as_slice().iter())
            .map(|(a, b)| {
                let a64 = Complex64::new(a.re.to_accum(), a.im.to_accum());
                let b64 = Complex64::new(b.re.to_accum(), b.im.to_accum());
                a64.conj() * b64
            })
            .sum();

        let norm_ax: f64 = ax
            .as_slice()
            .iter()
            .map(|c| c.re.to_accum().powi(2) + c.im.to_accum().powi(2))
            .sum::<f64>()
            .sqrt();
        let norm_y: f64 = y
            .as_slice()
            .iter()
            .map(|c| c.re.to_accum().powi(2) + c.im.to_accum().powi(2))
            .sum::<f64>()
            .sqrt();

        let diff = (ax_y - x_ay).norm();
        let scale = norm_ax * norm_y;

        if scale > 1e-15 { diff / scale } else { 0.0 }
    }

    /// Get dielectric contrast ratio: ε_max / ε_min
    pub fn dielectric_contrast(&self) -> f64 {
        let eps = self.dielectric.eps();
        if eps.is_empty() {
            return 1.0;
        }
        let eps_min = eps.iter().cloned().fold(f64::INFINITY, f64::min);
        let eps_max = eps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if eps_min > 1e-15 {
            eps_max / eps_min
        } else {
            f64::INFINITY
        }
    }

    /// Get effective epsilon values for logging.
    pub fn effective_epsilons(&self) -> (f64, f64) {
        (self.effective_te_epsilon(), self.effective_tm_epsilon())
    }

    pub fn kx_shifted(&self) -> &[f64] {
        &self.kx_shifted
    }

    pub fn ky_shifted(&self) -> &[f64] {
        &self.ky_shifted
    }

    pub fn k_plus_g_squares(&self) -> &[f64] {
        &self.k_plus_g_sq
    }

    pub fn k_plus_g_components(&self) -> (&[f64], &[f64]) {
        (&self.k_plus_g_x, &self.k_plus_g_y)
    }

    pub fn k_plus_g_clamp_mask(&self) -> &[bool] {
        &self.k_plus_g_was_clamped
    }

    #[allow(dead_code)]
    pub(crate) fn k_plus_g_sq_min(&self) -> f64 {
        self.k_plus_g_sq_min
    }

    #[allow(dead_code)]
    pub(crate) fn k_plus_g_sq_min_raw(&self) -> f64 {
        self.k_plus_g_sq_min_raw
    }

    #[allow(dead_code)]
    pub(crate) fn k_plus_g_near_zero_count(&self) -> usize {
        self.k_plus_g_floor_count
    }

    /// Capture a snapshot of operator state for debugging.
    pub fn capture_snapshot(&mut self, input: &B::Buffer) -> OperatorSnapshotData {
        match self.polarization {
            Polarization::TE => self.capture_te_snapshot(input),
            Polarization::TM => self.capture_tm_snapshot(input),
        }
    }

    fn apply_te(&mut self, input: &B::Buffer, output: &mut B::Buffer) {
        crate::profiler::start_timer("apply_te");

        copy_buffer(&mut self.scratch, input);
        self.backend.forward_fft_2d(&mut self.scratch);

        compute_gradients_from_potential(
            self.scratch.as_slice(),
            self.grad_x.as_mut_slice(),
            self.grad_y.as_mut_slice(),
            &self.k_plus_g_x,
            &self.k_plus_g_y,
        );

        self.backend.inverse_fft_2d(&mut self.grad_x);
        self.backend.inverse_fft_2d(&mut self.grad_y);

        apply_inv_eps(
            self.grad_x.as_mut_slice(),
            self.grad_y.as_mut_slice(),
            &self.dielectric,
        );

        self.backend.forward_fft_2d(&mut self.grad_x);
        self.backend.forward_fft_2d(&mut self.grad_y);

        assemble_divergence(
            self.grad_x.as_slice(),
            self.grad_y.as_slice(),
            output.as_mut_slice(),
            &self.k_plus_g_x,
            &self.k_plus_g_y,
        );

        self.backend.inverse_fft_2d(output);

        crate::profiler::stop_timer("apply_te");
    }

    fn apply_tm(&mut self, input: &B::Buffer, output: &mut B::Buffer) {
        crate::profiler::start_timer("apply_tm");

        // Optimization: Work directly on output buffer to save a copy
        copy_buffer(output, input);
        self.backend.forward_fft_2d(output);
        let data = output.as_mut_slice();
        #[cfg(debug_assertions)]
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                let max_k_sq = self.k_plus_g_sq.iter().cloned().fold(0.0_f64, f64::max);
                let mut sorted: Vec<f64> = self.k_plus_g_sq.iter().cloned().collect();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                log::debug!(
                    "[operator] TM k_plus_g_sq first 8: {:?}",
                    &sorted[..sorted.len().min(8)]
                );
                log::debug!("[operator] TM k_plus_g_sq max={:.4}", max_k_sq);
            });
        }
        for (value, &k_sq) in data.iter_mut().zip(self.k_plus_g_sq.iter()) {
            *value *= B::Real::from_accum(k_sq);
        }
        self.backend.inverse_fft_2d(output);

        crate::profiler::stop_timer("apply_tm");
    }

    /// Batched TM operator: apply -∇² to multiple vectors with batched FFTs.
    ///
    /// TM flow per vector:
    /// 1. Forward FFT
    /// 2. Multiply by |k+G|²
    /// 3. Inverse FFT
    ///
    /// Batched: Forward FFT all → k² multiply all → Inverse FFT all
    fn batch_apply_tm(&mut self, inputs: &[B::Buffer], outputs: &mut [B::Buffer]) {
        crate::profiler::start_timer("batch_apply_tm");

        // Sequential per-vector application is fastest on CPU thanks to cache locality.
        for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
            self.apply_tm(input, output);
        }

        crate::profiler::stop_timer("batch_apply_tm");
    }

    /// Batched TE operator: apply -∇·(ε⁻¹∇) to multiple vectors
    fn batch_apply_te(&mut self, inputs: &[B::Buffer], outputs: &mut [B::Buffer]) {
        crate::profiler::start_timer("batch_apply_te");

        let n = inputs.len();
        if n == 0 {
            crate::profiler::stop_timer("batch_apply_te");
            return;
        }

        // Sequential per-vector application beats a batched FFT path on CPU
        // because the shared `grad_x`/`grad_y` scratch buffers keep the inner
        // loop hot in L2 cache instead of cycling through n gradient pairs.
        for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
            self.apply_te(input, output);
        }

        crate::profiler::stop_timer("batch_apply_te");
    }

    fn capture_te_snapshot(&mut self, input: &B::Buffer) -> OperatorSnapshotData {
        let field_spatial = to_complex64_vec(input.as_slice());
        copy_buffer(&mut self.scratch, input);
        self.backend.forward_fft_2d(&mut self.scratch);
        let field_fourier = to_complex64_vec(self.scratch.as_slice());

        compute_gradients_from_potential(
            self.scratch.as_slice(),
            self.grad_x.as_mut_slice(),
            self.grad_y.as_mut_slice(),
            &self.k_plus_g_x,
            &self.k_plus_g_y,
        );

        self.backend.inverse_fft_2d(&mut self.grad_x);
        self.backend.inverse_fft_2d(&mut self.grad_y);
        let grad_x = to_complex64_vec(self.grad_x.as_slice());
        let grad_y = to_complex64_vec(self.grad_y.as_slice());

        apply_inv_eps(
            self.grad_x.as_mut_slice(),
            self.grad_y.as_mut_slice(),
            &self.dielectric,
        );
        let eps_grad_x = to_complex64_vec(self.grad_x.as_slice());
        let eps_grad_y = to_complex64_vec(self.grad_y.as_slice());

        self.backend.forward_fft_2d(&mut self.grad_x);
        self.backend.forward_fft_2d(&mut self.grad_y);

        assemble_divergence(
            self.grad_x.as_slice(),
            self.grad_y.as_slice(),
            self.scratch.as_mut_slice(),
            &self.k_plus_g_x,
            &self.k_plus_g_y,
        );
        let theta_fourier = to_complex64_vec(self.scratch.as_slice());

        self.backend.inverse_fft_2d(&mut self.scratch);
        let theta_spatial = to_complex64_vec(self.scratch.as_slice());

        OperatorSnapshotData {
            grid: self.grid,
            field_spatial,
            field_fourier,
            theta_spatial,
            theta_fourier,
            grad_x: Some(grad_x),
            grad_y: Some(grad_y),
            eps_grad_x: Some(eps_grad_x),
            eps_grad_y: Some(eps_grad_y),
        }
    }

    fn capture_tm_snapshot(&mut self, input: &B::Buffer) -> OperatorSnapshotData {
        let field_spatial = to_complex64_vec(input.as_slice());
        copy_buffer(&mut self.scratch, input);
        self.backend.forward_fft_2d(&mut self.scratch);
        let field_fourier = to_complex64_vec(self.scratch.as_slice());

        compute_gradients_from_potential(
            self.scratch.as_slice(),
            self.grad_x.as_mut_slice(),
            self.grad_y.as_mut_slice(),
            &self.k_plus_g_x,
            &self.k_plus_g_y,
        );

        self.backend.inverse_fft_2d(&mut self.grad_x);
        self.backend.inverse_fft_2d(&mut self.grad_y);
        let grad_x = to_complex64_vec(self.grad_x.as_slice());
        let grad_y = to_complex64_vec(self.grad_y.as_slice());

        for (value, &k_sq) in self
            .scratch
            .as_mut_slice()
            .iter_mut()
            .zip(self.k_plus_g_sq.iter())
        {
            *value *= B::Real::from_accum(k_sq);
        }
        let theta_fourier = to_complex64_vec(self.scratch.as_slice());

        self.backend.inverse_fft_2d(&mut self.scratch);
        let theta_spatial = to_complex64_vec(self.scratch.as_slice());

        OperatorSnapshotData {
            grid: self.grid,
            field_spatial,
            field_fourier,
            theta_spatial,
            theta_fourier,
            grad_x: Some(grad_x),
            grad_y: Some(grad_y),
            eps_grad_x: None,
            eps_grad_y: None,
        }
    }

    fn effective_tm_epsilon(&self) -> f64 {
        arithmetic_mean(self.dielectric.eps()).unwrap_or(1.0)
    }

    fn effective_te_epsilon(&self) -> f64 {
        harmonic_mean(self.dielectric.inv_eps()).unwrap_or(1.0)
    }

    // ========================================================================
    // Operator derivative methods for envelope approximation
    // ========================================================================
    //
    // These methods apply partial derivatives of the Maxwell operator L₀ with
    // respect to the Bloch wavevector k and the registry coordinate R.
    //
    // Convention: eigenvalue λ = (ω/c)² in units of (2π/a)².
    //             k-vector components in units of 2π/a.
    //             R in fractional lattice coordinates.
    //
    // The derivative operators have the same units as L₀ itself (eigenvalue units).
    //
    // Reference: thesis §envelope_approximation, Eq. (velocity_matrix) ff.
    // ========================================================================

    /// Apply ∂L₀/∂kᵢ to `input`, writing the result to `output`.
    ///
    /// # TE mode
    ///
    /// L₀ = -D(k)·(ε⁻¹ D(k)) where D(k) = ∇ + ik.
    ///
    /// ∂L₀/∂kᵢ = -(∂Dᵢ/∂kᵢ)·(ε⁻¹ D u) - D·(ε⁻¹ (∂Dᵢ/∂kᵢ) u)
    ///
    /// Since ∂D(k)/∂kᵢ = i·eᵢ (unit vector), we get two terms:
    ///
    ///   Term A: -i · (ε⁻¹ D u)ᵢ        (extract i-th component after ε⁻¹·grad)
    ///   Term B: -D · (ε⁻¹ · i·eᵢ · u)  (divergence of ε⁻¹ column times i·u)
    ///
    /// Together: ∂L₀/∂kᵢ u = -i·(ε⁻¹ D u)ᵢ - D·(i·ε⁻¹_{·i}·u)
    ///
    /// # TM mode
    ///
    /// A = -|k+G|² (diagonal in Fourier space), B = ε(r).
    /// ∂A/∂kᵢ û(G) = -2(k+G)ᵢ · û(G)
    ///
    /// For the generalized eigenproblem, the velocity matrix element is:
    ///   v^(i)_mn = ⟨uₘ|∂A/∂kᵢ|uₙ⟩_B / ... but we expose ∂A/∂kᵢ directly.
    ///
    /// # Arguments
    ///
    /// * `direction` — 0 for x, 1 for y
    /// * `input` — eigenvector u in spatial/Fourier representation
    /// * `output` — result ∂L₀/∂kᵢ · u
    pub fn apply_dL_dk(&mut self, input: &B::Buffer, output: &mut B::Buffer, direction: usize) {
        assert!(direction < 2, "direction must be 0 or 1");
        match self.polarization {
            Polarization::TE => self.apply_dL_dk_te(input, output, direction),
            Polarization::TM => self.apply_dA_dk_tm(input, output, direction),
        }
    }

    /// Apply ∂²L₀/∂kᵢ∂kⱼ to `input`, writing the result to `output`.
    ///
    /// # TE mode
    ///
    /// ∂²L₀/∂kᵢ∂kⱼ u = 2 ε⁻¹_{ij}(r) u(r)
    ///
    /// This is a pointwise multiplication in spatial domain — no FFTs needed.
    /// The factor 2 comes from differentiating the bilinear form D·(ε⁻¹ D)
    /// twice w.r.t. k (each D contributes one factor of i·eᵢ).
    ///
    /// **Sign convention:** L₀ = -D·ε⁻¹D, and differentiating the explicit
    /// first-derivative terms yields ∂²L₀/∂kᵢ∂kⱼ = 2ε⁻¹_{ij}.
    ///
    /// # TM mode
    ///
    /// ∂²A/∂kᵢ∂kⱼ u = 2δᵢⱼ u
    ///
    /// # Arguments
    ///
    /// * `i`, `j` — directions (0=x, 1=y)
    pub fn apply_d2L_dk2(
        &mut self,
        input: &B::Buffer,
        output: &mut B::Buffer,
        i: usize,
        j: usize,
    ) {
        assert!(i < 2 && j < 2, "directions must be 0 or 1");
        match self.polarization {
            Polarization::TE => self.apply_d2L_dk2_te(input, output, i, j),
            Polarization::TM => self.apply_d2A_dk2_tm(input, output, i, j),
        }
    }

    /// Apply ∂L₀/∂Rⱼ to `input` using a precomputed dielectric derivative.
    ///
    /// # TE mode
    ///
    /// ∂L₀/∂Rⱼ = -D(k)·(∂ε⁻¹/∂Rⱼ)·D(k)
    ///
    /// Same structure as L₀ but with ∂ε⁻¹/∂Rⱼ replacing ε⁻¹.
    ///
    /// # TM mode
    ///
    /// A does not depend on R (it's just -|k+G|²).
    /// B = ε(r; R), so ∂B/∂Rⱼ = ∂ε/∂Rⱼ.
    ///
    /// For the generalized eigenproblem perturbation theory:
    ///   ⟨uₘ|∂L₀/∂Rⱼ|uₙ⟩ involves both ∂A/∂R and ∂B/∂R terms.
    ///   Since ∂A/∂R = 0, we apply ∂B/∂Rⱼ as pointwise multiply by ∂ε/∂Rⱼ.
    ///
    /// The caller (OperatorDataExtractor) is responsible for combining these correctly
    /// for the generalized eigenproblem perturbation formula.
    ///
    /// # Arguments
    ///
    /// * `diel_deriv` — Precomputed dielectric derivative for direction j
    pub fn apply_dL_dR(
        &mut self,
        input: &B::Buffer,
        output: &mut B::Buffer,
        diel_deriv: &DielectricDerivative,
    ) {
        match self.polarization {
            Polarization::TE => self.apply_dL_dR_te(input, output, diel_deriv),
            Polarization::TM => self.apply_dB_dR_tm(input, output, diel_deriv),
        }
    }

    /// TE helper: return the selected spatial component of
    /// ``(∂ε⁻¹/∂R_j) · D(k) u`` before the outer divergence is applied.
    ///
    /// The input and output are in the usual spatial field representation.
    /// This is used by the exact TE downfolding diagnostics, where the
    /// appendix-level remainder terms keep the explicit slow-coefficient block
    /// rather than only the compact scalar reduction.
    pub fn apply_dinv_eps_gradient_component_te(
        &mut self,
        input: &B::Buffer,
        output: &mut B::Buffer,
        diel_deriv: &DielectricDerivative,
        component: usize,
    ) {
        assert!(component < 2, "component must be 0 or 1");

        copy_buffer(&mut self.scratch, input);
        self.backend.forward_fft_2d(&mut self.scratch);

        compute_gradients_from_potential(
            self.scratch.as_slice(),
            self.grad_x.as_mut_slice(),
            self.grad_y.as_mut_slice(),
            &self.k_plus_g_x,
            &self.k_plus_g_y,
        );

        self.backend.inverse_fft_2d(&mut self.grad_x);
        self.backend.inverse_fft_2d(&mut self.grad_y);

        apply_dielectric_derivative(
            self.grad_x.as_mut_slice(),
            self.grad_y.as_mut_slice(),
            diel_deriv,
        );

        if component == 0 {
            copy_buffer(output, &self.grad_x);
        } else {
            copy_buffer(output, &self.grad_y);
        }
    }

    /// TM exact helper: apply the hermitized fast-derivative coefficient
    ///
    ///   O_i^(TM) = -2 ε^{-1} D_i - ∂_i ε^{-1}
    ///
    /// to a hermitized TM microscopic state χ = √ε u.
    ///
    /// This is the fast operator multiplying D_{R_i} in the exact Stage-1
    /// hermitized TM Hamiltonian. The input and output both live in the
    /// standard spatial χ-representation, so the projected matrix elements can
    /// be taken with the ordinary backend dot product.
    pub fn apply_tm_hermitized_fast_derivative(
        &mut self,
        input: &B::Buffer,
        output: &mut B::Buffer,
        direction: usize,
    ) {
        assert!(
            self.polarization == Polarization::TM,
            "TM hermitized fast-derivative helper requires TM polarization"
        );
        assert!(direction < 2, "direction must be 0 or 1");

        copy_buffer(&mut self.scratch, input);
        self.backend.forward_fft_2d(&mut self.scratch);

        compute_gradients_from_potential(
            self.scratch.as_slice(),
            self.grad_x.as_mut_slice(),
            self.grad_y.as_mut_slice(),
            &self.k_plus_g_x,
            &self.k_plus_g_y,
        );

        self.backend.inverse_fft_2d(&mut self.grad_x);
        self.backend.inverse_fft_2d(&mut self.grad_y);

        if direction == 0 {
            copy_buffer(output, &self.grad_x);
        } else {
            copy_buffer(output, &self.grad_y);
        }

        for (value, &inv_eps) in output
            .as_mut_slice()
            .iter_mut()
            .zip(self.dielectric.inv_eps().iter())
        {
            let scale = B::Real::from_accum(-2.0 * inv_eps);
            *value *= scale;
        }

        for (dst, &inv_eps) in self
            .scratch
            .as_mut_slice()
            .iter_mut()
            .zip(self.dielectric.inv_eps().iter())
        {
            *dst = cscalar::<B::Real>(inv_eps, 0.0);
        }

        self.backend.forward_fft_2d(&mut self.scratch);
        compute_gradients_from_potential(
            self.scratch.as_slice(),
            self.grad_x.as_mut_slice(),
            self.grad_y.as_mut_slice(),
            &self.k_plus_g_x,
            &self.k_plus_g_y,
        );
        self.backend.inverse_fft_2d(&mut self.grad_x);
        self.backend.inverse_fft_2d(&mut self.grad_y);

        let d_inv_eps = if direction == 0 {
            self.grad_x.as_slice()
        } else {
            self.grad_y.as_slice()
        };
        let input_slice = input.as_slice();

        for ((value, &d_inv_eps_value), &input_value) in output
            .as_mut_slice()
            .iter_mut()
            .zip(d_inv_eps.iter())
            .zip(input_slice.iter())
        {
            {
                *value -= d_inv_eps_value * input_value;
            }
        }
    }

    /// TM exact helper: apply the local hermitized operator derivative
    ///
    ///   ∂_(R_i) L^(0)_TM = (∂_(R_i) ρ) A ρ + ρ A (∂_(R_i) ρ),
    ///
    /// where A = -D_r(k_0)^2 and ρ = ε^(-1/2).
    pub fn apply_tm_hermitized_local_r_derivative(
        &mut self,
        input: &B::Buffer,
        output: &mut B::Buffer,
        rho_derivative: &[f64],
    ) {
        assert!(
            self.polarization == Polarization::TM,
            "TM hermitized local R-derivative helper requires TM polarization"
        );
        assert_eq!(rho_derivative.len(), self.grid.len(), "rho_derivative length must match grid");

        let rho: Vec<f64> = self.dielectric.eps().iter().map(|&eps| eps.powf(-0.5)).collect();

        let mut tmp = self.alloc_field();
        let mut applied = self.alloc_field();

        copy_buffer(&mut tmp, input);
        scale_by_real_field(tmp.as_mut_slice(), &rho);
        self.apply_tm(&tmp, &mut applied);
        copy_buffer(output, &applied);
        scale_by_real_field(output.as_mut_slice(), rho_derivative);

        copy_buffer(&mut tmp, input);
        scale_by_real_field(tmp.as_mut_slice(), rho_derivative);
        self.apply_tm(&tmp, &mut applied);
        scale_by_real_field(applied.as_mut_slice(), &rho);
        add_buffer_in_place(output.as_mut_slice(), applied.as_slice());
    }

    /// TM exact helper: apply the local hermitized second R-derivative
    ///
    ///   ∂_(R_i)^2 L^(0)_TM = (∂_(R_i)^2 ρ) A ρ + 2 (∂_(R_i) ρ) A (∂_(R_i) ρ) + ρ A (∂_(R_i)^2 ρ).
    pub fn apply_tm_hermitized_local_r_second_derivative(
        &mut self,
        input: &B::Buffer,
        output: &mut B::Buffer,
        rho_derivative: &[f64],
        rho_second_derivative: &[f64],
    ) {
        assert!(
            self.polarization == Polarization::TM,
            "TM hermitized local second R-derivative helper requires TM polarization"
        );
        assert_eq!(rho_derivative.len(), self.grid.len(), "rho_derivative length must match grid");
        assert_eq!(
            rho_second_derivative.len(),
            self.grid.len(),
            "rho_second_derivative length must match grid"
        );

        let rho: Vec<f64> = self.dielectric.eps().iter().map(|&eps| eps.powf(-0.5)).collect();

        let mut tmp = self.alloc_field();
        let mut applied = self.alloc_field();

        copy_buffer(&mut tmp, input);
        scale_by_real_field(tmp.as_mut_slice(), &rho);
        self.apply_tm(&tmp, &mut applied);
        copy_buffer(output, &applied);
        scale_by_real_field(output.as_mut_slice(), rho_second_derivative);

        copy_buffer(&mut tmp, input);
        scale_by_real_field(tmp.as_mut_slice(), rho_derivative);
        self.apply_tm(&tmp, &mut applied);
        scale_by_real_field(applied.as_mut_slice(), rho_derivative);
        scale_buffer_in_place(applied.as_mut_slice(), 2.0);
        add_buffer_in_place(output.as_mut_slice(), applied.as_slice());

        copy_buffer(&mut tmp, input);
        scale_by_real_field(tmp.as_mut_slice(), rho_second_derivative);
        self.apply_tm(&tmp, &mut applied);
        scale_by_real_field(applied.as_mut_slice(), &rho);
        add_buffer_in_place(output.as_mut_slice(), applied.as_slice());
    }

    /// TM exact helper: apply ρ D_(r_i) to an arbitrary spatial field.
    pub fn apply_tm_rho_covariant_gradient(
        &mut self,
        input: &B::Buffer,
        output: &mut B::Buffer,
        direction: usize,
    ) {
        assert!(
            self.polarization == Polarization::TM,
            "TM rho-gradient helper requires TM polarization"
        );
        assert!(direction < 2, "direction must be 0 or 1");

        let rho: Vec<f64> = self.dielectric.eps().iter().map(|&eps| eps.powf(-0.5)).collect();

        copy_buffer(&mut self.scratch, input);
        self.backend.forward_fft_2d(&mut self.scratch);

        compute_gradients_from_potential(
            self.scratch.as_slice(),
            self.grad_x.as_mut_slice(),
            self.grad_y.as_mut_slice(),
            &self.k_plus_g_x,
            &self.k_plus_g_y,
        );

        if direction == 0 {
            copy_buffer(output, &self.grad_x);
        } else {
            copy_buffer(output, &self.grad_y);
        }

        self.backend.inverse_fft_2d(output);
        scale_by_real_field(output.as_mut_slice(), &rho);
    }

    /// Expose the dielectric (needed by EA extractor for inner products).
    pub fn dielectric(&self) -> &Dielectric2D {
        &self.dielectric
    }

    // ---- TE derivative implementations ----

    /// TE: ∂L₀/∂kᵢ u = -i·(ε⁻¹ Du)ᵢ - D·(i·ε⁻¹_{·i} u)
    fn apply_dL_dk_te(&mut self, input: &B::Buffer, output: &mut B::Buffer, direction: usize) {
        // --- Term A: -i · (ε⁻¹ D u)ᵢ ---
        // Step 1: FFT input
        copy_buffer(&mut self.scratch, input);
        self.backend.forward_fft_2d(&mut self.scratch);

        // Step 2: Compute gradients Dᵢ u = i(k+G)ᵢ û
        compute_gradients_from_potential(
            self.scratch.as_slice(),
            self.grad_x.as_mut_slice(),
            self.grad_y.as_mut_slice(),
            &self.k_plus_g_x,
            &self.k_plus_g_y,
        );

        // Step 3: IFFT gradients to spatial domain
        self.backend.inverse_fft_2d(&mut self.grad_x);
        self.backend.inverse_fft_2d(&mut self.grad_y);

        // Step 4: Apply ε⁻¹ tensor to get (ε⁻¹ D u) in spatial domain
        apply_inv_eps(
            self.grad_x.as_mut_slice(),
            self.grad_y.as_mut_slice(),
            &self.dielectric,
        );
        // Now grad_x = (ε⁻¹ D u)_x, grad_y = (ε⁻¹ D u)_y  (spatial domain)

        // Step 5: Extract i-th component, multiply by -i
        // Term A = -i · (ε⁻¹ D u)ᵢ in spatial domain
        let term_a_spatial: Vec<Complex<B::Real>> = if direction == 0 {
            self.grad_x.as_slice().to_vec()
        } else {
            self.grad_y.as_slice().to_vec()
        };
        // Multiply by -i
        let term_a: Vec<Complex<B::Real>> = term_a_spatial
            .iter()
            .map(|&v| {
                // -i * v = -i * (a + bi) = b - ai
                Complex::new(v.im, -v.re)
            })
            .collect();

        // --- Term B: -D · (i · ε⁻¹_{·i} · u) ---
        // We need to compute the divergence of the vector field
        // f_α = i · ε⁻¹_{αi}(r) · u(r)  for α ∈ {x, y}
        //
        // Then Term B = -D · f = -Σ_α i(k+G)_α f̂_α  (negated divergence)
        //
        // Note: ε⁻¹_{αi} is the (α, i) component of the ε⁻¹ tensor.
        // For isotropic ε⁻¹: ε⁻¹_{αi} = δ_{αi}/ε, so f_α = i·δ_{αi}·u/ε.
        //   If i=x: f_x = i·u/ε, f_y = 0
        //   If i=y: f_x = 0,     f_y = i·u/ε

        // Build f_x, f_y in spatial domain
        {
            let input_slice = input.as_slice();
            let fx = self.grad_x.as_mut_slice();
            let fy = self.grad_y.as_mut_slice();

            if let Some(tensors) = self.dielectric.inv_eps_tensors() {
                // Anisotropic: f_α = i · ε⁻¹_{α,direction} · u
                for (idx, ((out_x, out_y), &u_val)) in
                    fx.iter_mut().zip(fy.iter_mut()).zip(input_slice.iter()).enumerate()
                {
                    let t = &tensors[idx];
                    let ex = B::Real::from_accum(t[direction]);
                    let ey = B::Real::from_accum(t[2 + direction]);
                    // i * eps_inv * u = i * c * (a+bi) = c*(-b + ai)
                    *out_x = Complex::new(-u_val.im * ex, u_val.re * ex);
                    *out_y = Complex::new(-u_val.im * ey, u_val.re * ey);
                }
            } else {
                // Isotropic: f_α = δ_{α,direction} · i · (1/ε) · u
                let inv_eps = self.dielectric.inv_eps();
                let zero = czero::<B::Real>();
                for (idx, ((out_x, out_y), &u_val)) in
                    fx.iter_mut().zip(fy.iter_mut()).zip(input_slice.iter()).enumerate()
                {
                    let inv = B::Real::from_accum(inv_eps[idx]);
                    let iu = Complex::new(-u_val.im * inv, u_val.re * inv);
                    if direction == 0 {
                        *out_x = iu;
                        *out_y = zero;
                    } else {
                        *out_x = zero;
                        *out_y = iu;
                    }
                }
            }
        }

        // FFT f_x, f_y to Fourier domain
        self.backend.forward_fft_2d(&mut self.grad_x);
        self.backend.forward_fft_2d(&mut self.grad_y);

        // Divergence: -D·f = -Σ_α i(k+G)_α f̂_α (negated)
        assemble_divergence(
            self.grad_x.as_slice(),
            self.grad_y.as_slice(),
            output.as_mut_slice(),
            &self.k_plus_g_x,
            &self.k_plus_g_y,
        );

        // IFFT Term B to spatial domain
        self.backend.inverse_fft_2d(output);

        // --- Combine: output = Term A + Term B ---
        let out = output.as_mut_slice();
        for (o, &a) in out.iter_mut().zip(term_a.iter()) {
            {
                *o += a;
            }
        }
    }

    /// TE: ∂²L₀/∂kᵢ∂kⱼ u = -2 ε⁻¹_{ij}(r) u(r)
    ///
    /// Pointwise multiplication — no FFTs needed.
    fn apply_d2L_dk2_te(
        &mut self,
        input: &B::Buffer,
        output: &mut B::Buffer,
        i: usize,
        j: usize,
    ) {
        let inp = input.as_slice();
        let out = output.as_mut_slice();

        if let Some(tensors) = self.dielectric.inv_eps_tensors() {
            // Anisotropic: 2 * ε⁻¹_{ij} * u
            // Tensor layout: [xx, xy, yx, yy] → index i*2+j
            let tensor_idx = i * 2 + j;
            for (idx, (o, &u_val)) in out.iter_mut().zip(inp.iter()).enumerate() {
                let coeff = B::Real::from_accum(2.0 * tensors[idx][tensor_idx]);
                *o = u_val * coeff;
            }
        } else {
            // Isotropic: ε⁻¹_{ij} = δ_{ij}/ε
            if i != j {
                // Off-diagonal: zero
                let zero = czero::<B::Real>();
                for o in out.iter_mut() {
                    *o = zero;
                }
            } else {
                // Diagonal: 2/ε
                let inv_eps = self.dielectric.inv_eps();
                for (idx, (o, &u_val)) in out.iter_mut().zip(inp.iter()).enumerate() {
                    let coeff = B::Real::from_accum(2.0 * inv_eps[idx]);
                    *o = u_val * coeff;
                }
            }
        }
    }

    /// TE: ∂L₀/∂Rⱼ u = -D(k) · (∂ε⁻¹/∂Rⱼ · D(k) u)
    ///
    /// Same structure as apply_te but with ∂ε⁻¹/∂Rⱼ instead of ε⁻¹.
    fn apply_dL_dR_te(
        &mut self,
        input: &B::Buffer,
        output: &mut B::Buffer,
        diel_deriv: &DielectricDerivative,
    ) {
        // Step 1: FFT input
        copy_buffer(&mut self.scratch, input);
        self.backend.forward_fft_2d(&mut self.scratch);

        // Step 2: Compute gradients D u = i(k+G) û
        compute_gradients_from_potential(
            self.scratch.as_slice(),
            self.grad_x.as_mut_slice(),
            self.grad_y.as_mut_slice(),
            &self.k_plus_g_x,
            &self.k_plus_g_y,
        );

        // Step 3: IFFT gradients to spatial domain
        self.backend.inverse_fft_2d(&mut self.grad_x);
        self.backend.inverse_fft_2d(&mut self.grad_y);

        // Step 4: Apply ∂ε⁻¹/∂Rⱼ (instead of ε⁻¹)
        apply_dielectric_derivative(
            self.grad_x.as_mut_slice(),
            self.grad_y.as_mut_slice(),
            diel_deriv,
        );

        // Step 5: FFT back to Fourier domain
        self.backend.forward_fft_2d(&mut self.grad_x);
        self.backend.forward_fft_2d(&mut self.grad_y);

        // Step 6: Assemble divergence (negated)
        assemble_divergence(
            self.grad_x.as_slice(),
            self.grad_y.as_slice(),
            output.as_mut_slice(),
            &self.k_plus_g_x,
            &self.k_plus_g_y,
        );

        // Step 7: IFFT
        self.backend.inverse_fft_2d(output);
    }

    // ---- TM derivative implementations ----

    /// TM: ∂A/∂kᵢ û(G) = 2(k+G)ᵢ û(G)
    ///
    /// Fourier-diagonal: FFT → pointwise multiply → IFFT
    fn apply_dA_dk_tm(&mut self, input: &B::Buffer, output: &mut B::Buffer, direction: usize) {
        copy_buffer(output, input);
        self.backend.forward_fft_2d(output);

        let data = output.as_mut_slice();
        let k_component = if direction == 0 {
            &self.k_plus_g_x
        } else {
            &self.k_plus_g_y
        };

        for (val, &k_i) in data.iter_mut().zip(k_component.iter()) {
            let coeff = B::Real::from_accum(2.0 * k_i);
            *val *= coeff;
        }

        self.backend.inverse_fft_2d(output);
    }

    /// TM: ∂²A/∂kᵢ∂kⱼ u = 2δᵢⱼ u
    fn apply_d2A_dk2_tm(
        &self,
        input: &B::Buffer,
        output: &mut B::Buffer,
        i: usize,
        j: usize,
    ) {
        if i == j {
            let inp = input.as_slice();
            let out = output.as_mut_slice();
            let two = B::Real::from_accum(2.0);
            for (o, &v) in out.iter_mut().zip(inp.iter()) {
                *o = v * two;
            }
        } else {
            // Off-diagonal: zero
            let zero = czero::<B::Real>();
            for o in output.as_mut_slice().iter_mut() {
                *o = zero;
            }
        }
    }

    /// TM: ∂B/∂Rⱼ u = (∂ε/∂Rⱼ)(r) · u(r)
    ///
    /// Pointwise multiplication by the derivative of the dielectric.
    fn apply_dB_dR_tm(
        &self,
        input: &B::Buffer,
        output: &mut B::Buffer,
        diel_deriv: &DielectricDerivative,
    ) {
        let inp = input.as_slice();
        let out = output.as_mut_slice();
        for (idx, (o, &v)) in out.iter_mut().zip(inp.iter()).enumerate() {
            let d_eps = B::Real::from_accum(diel_deriv.d_eps[idx]);
            *o = v * d_eps;
        }
    }
}

impl<B: SpectralBackend> LinearOperator<B> for ThetaOperator<B> {
    fn apply(&mut self, input: &B::Buffer, output: &mut B::Buffer) {
        match self.polarization {
            Polarization::TE => self.apply_te(input, output),
            Polarization::TM => self.apply_tm(input, output),
        }
    }

    fn apply_mass(&mut self, input: &B::Buffer, output: &mut B::Buffer) {
        match self.polarization {
            Polarization::TE => copy_buffer(output, input),
            Polarization::TM => {
                copy_buffer(output, input);
                apply_scalar_eps(output.as_mut_slice(), self.dielectric.eps());
            }
        }
    }

    fn batch_apply(&mut self, inputs: &[B::Buffer], outputs: &mut [B::Buffer]) {
        assert_eq!(
            inputs.len(),
            outputs.len(),
            "batch_apply: inputs and outputs must have same length"
        );
        match self.polarization {
            Polarization::TE => self.batch_apply_te(inputs, outputs),
            Polarization::TM => self.batch_apply_tm(inputs, outputs),
        }
    }

    fn alloc_field(&self) -> B::Buffer {
        self.backend.alloc_field(self.grid)
    }

    fn backend(&self) -> &B {
        &self.backend
    }

    fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    fn grid(&self) -> Grid2D {
        self.grid
    }

    fn bloch(&self) -> [f64; 2] {
        self.bloch
    }

    fn gamma_kernel_transform(&self) -> Option<&[f64]> {
        None
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn build_fft_indices(n: usize) -> Vec<isize> {
    (0..n)
        .map(|i| {
            if i <= n / 2 {
                i as isize
            } else {
                i as isize - n as isize
            }
        })
        .collect()
}

fn build_g_vectors_with_reciprocal_lattice(
    grid: Grid2D,
    b1: [f64; 2],
    b2: [f64; 2],
) -> (Vec<f64>, Vec<f64>) {
    let n1_indices = build_fft_indices(grid.nx);
    let n2_indices = build_fft_indices(grid.ny);
    let len = grid.len();
    let mut gx = vec![0.0; len];
    let mut gy = vec![0.0; len];

    for iy in 0..grid.ny {
        for ix in 0..grid.nx {
            let idx = grid.idx(ix, iy);
            let n1 = n1_indices[ix] as f64;
            let n2 = n2_indices[iy] as f64;
            gx[idx] = n1 * b1[0] + n2 * b2[0];
            gy[idx] = n1 * b1[1] + n2 * b2[1];
        }
    }

    (gx, gy)
}

fn build_k_plus_g_tables_with_reciprocal(
    grid: Grid2D,
    b1: [f64; 2],
    b2: [f64; 2],
    bloch: [f64; 2],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64, f64, usize, Vec<bool>) {
    let (gx_base, gy_base) = build_g_vectors_with_reciprocal_lattice(grid, b1, b2);
    let len = grid.len();

    #[cfg(debug_assertions)]
    {
        let mut unique_gx: Vec<f64> = gx_base.iter().cloned().collect();
        unique_gx.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique_gx.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
        log::debug!(
            "[k-vectors] unique G_x values (first 8): {:?}",
            &unique_gx[..unique_gx.len().min(8)]
        );
    }

    let mut k_plus_g_x = vec![0.0; len];
    let mut k_plus_g_y = vec![0.0; len];
    let mut squares = vec![0.0; len];
    let mut clamp_mask = vec![false; len];
    let mut raw_min = f64::INFINITY;
    let mut clamped_min = f64::INFINITY;
    let mut floor_count = 0usize;

    for idx in 0..len {
        let raw_kx = gx_base[idx] + bloch[0];
        let raw_ky = gy_base[idx] + bloch[1];
        let raw_sq = raw_kx * raw_kx + raw_ky * raw_ky;
        if raw_sq.is_finite() {
            raw_min = raw_min.min(raw_sq);
        }
        let (clamped_kx, clamped_ky) = clamp_gradient_components(raw_kx, raw_ky);
        let clamped_sq = clamped_kx * clamped_kx + clamped_ky * clamped_ky;
        clamped_min = clamped_min.min(clamped_sq);
        if raw_sq <= K_PLUS_G_NEAR_ZERO_FLOOR {
            floor_count += 1;
            clamp_mask[idx] = true;
        }
        k_plus_g_x[idx] = clamped_kx;
        k_plus_g_y[idx] = clamped_ky;
        squares[idx] = clamped_sq;
    }

    if raw_min == f64::INFINITY {
        raw_min = 0.0;
    }
    if clamped_min == f64::INFINITY {
        clamped_min = 0.0;
    }

    (
        k_plus_g_x,
        k_plus_g_y,
        squares,
        raw_min,
        clamped_min,
        floor_count,
        clamp_mask,
    )
}

fn clamp_gradient_components(kx: f64, ky: f64) -> (f64, f64) {
    if !kx.is_finite() || !ky.is_finite() {
        let magnitude = K_PLUS_G_NEAR_ZERO_FLOOR.sqrt();
        return (magnitude, 0.0);
    }

    let norm_sq = kx * kx + ky * ky;
    if norm_sq >= K_PLUS_G_NEAR_ZERO_FLOOR {
        (kx, ky)
    } else if norm_sq == 0.0 {
        let magnitude = K_PLUS_G_NEAR_ZERO_FLOOR.sqrt();
        (magnitude, 0.0)
    } else {
        let scale = (K_PLUS_G_NEAR_ZERO_FLOOR / norm_sq).sqrt();
        (kx * scale, ky * scale)
    }
}

fn tm_preconditioner_mass_floor(eps_eff: f64) -> f64 {
    if !eps_eff.is_finite() || eps_eff <= 0.0 {
        return 0.0;
    }
    eps_eff * TM_PRECONDITIONER_MASS_FRACTION
}

fn inverse_scale(k_sq: f64, shift: f64, eps_eff: f64, mass_floor: f64) -> f64 {
    if !k_sq.is_finite() || !eps_eff.is_finite() || eps_eff <= 0.0 {
        return 0.0;
    }

    let safe_mass = if mass_floor.is_finite() && mass_floor > 0.0 {
        mass_floor
    } else {
        0.0
    };

    let safe_k_sq = k_sq.max(0.0);
    let shift_scaled = shift * eps_eff.max(1e-12);
    let denominator = safe_k_sq + safe_mass + shift_scaled;

    eps_eff / denominator
}

fn build_inverse_diagonal(
    values: &[f64],
    shift: f64,
    eps_eff: f64,
    mass_floor: f64,
    near_zero_mask: Option<&[bool]>,
) -> Vec<f64> {
    let mut result: Vec<f64> = values
        .iter()
        .copied()
        .map(|k| inverse_scale(k, shift, eps_eff, mass_floor))
        .collect();

    if let Some(mask) = near_zero_mask {
        for (scale, &is_near_zero) in result.iter_mut().zip(mask.iter()) {
            if is_near_zero {
                *scale = 0.0;
            }
        }
    }

    result
}

fn copy_buffer<T: SpectralBuffer>(dst: &mut T, src: &T) {
    dst.as_mut_slice().copy_from_slice(src.as_slice());
}

fn compute_gradients_from_potential<R: Real>(
    potential: &[Complex<R>],
    grad_x: &mut [Complex<R>],
    grad_y: &mut [Complex<R>],
    k_plus_g_x: &[f64],
    k_plus_g_y: &[f64],
) {
    for ((((val, gx), gy), &kx), &ky) in potential
        .iter()
        .zip(grad_x.iter_mut())
        .zip(grad_y.iter_mut())
        .zip(k_plus_g_x.iter())
        .zip(k_plus_g_y.iter())
    {
        let factor_x = cscalar::<R>(0.0, kx);
        let factor_y = cscalar::<R>(0.0, ky);
        *gx = *val * factor_x;
        *gy = *val * factor_y;
    }
}

fn apply_inv_eps<R: Real>(
    grad_x: &mut [Complex<R>],
    grad_y: &mut [Complex<R>],
    dielectric: &Dielectric2D,
) {
    if let Some(tensors) = dielectric.inv_eps_tensors() {
        for ((gx, gy), tensor) in grad_x.iter_mut().zip(grad_y.iter_mut()).zip(tensors.iter()) {
            let orig_x = *gx;
            let orig_y = *gy;
            let t0 = R::from_accum(tensor[0]);
            let t1 = R::from_accum(tensor[1]);
            let t2 = R::from_accum(tensor[2]);
            let t3 = R::from_accum(tensor[3]);
            *gx = orig_x * t0 + orig_y * t1;
            *gy = orig_x * t2 + orig_y * t3;
        }
    } else {
        for ((gx, gy), &inv) in grad_x
            .iter_mut()
            .zip(grad_y.iter_mut())
            .zip(dielectric.inv_eps().iter())
        {
            let s = R::from_accum(inv);
            *gx *= s;
            *gy *= s;
        }
    }
}

fn apply_scalar_eps<R: Real>(field: &mut [Complex<R>], eps: &[f64]) {
    for (value, &eps_val) in field.iter_mut().zip(eps.iter()) {
        *value *= R::from_accum(eps_val);
    }
}

fn scale_by_real_field<R: Real>(field: &mut [Complex<R>], values: &[f64]) {
    for (value, &scale) in field.iter_mut().zip(values.iter()) {
        *value *= R::from_accum(scale);
    }
}

fn scale_buffer_in_place<R: Real>(field: &mut [Complex<R>], scale: f64) {
    let s = R::from_accum(scale);
    for value in field.iter_mut() {
        *value *= s;
    }
}

fn add_buffer_in_place<R: Real>(dst: &mut [Complex<R>], src: &[Complex<R>]) {
    for (lhs, rhs) in dst.iter_mut().zip(src.iter()) {
        *lhs += *rhs;
    }
}

fn assemble_divergence<R: Real>(
    grad_x: &[Complex<R>],
    grad_y: &[Complex<R>],
    output: &mut [Complex<R>],
    k_plus_g_x: &[f64],
    k_plus_g_y: &[f64],
) {
    for ((((out, &gx), &gy), &kx), &ky) in output
        .iter_mut()
        .zip(grad_x.iter())
        .zip(grad_y.iter())
        .zip(k_plus_g_x.iter())
        .zip(k_plus_g_y.iter())
    {
        let factor_x = cscalar::<R>(0.0, kx);
        let factor_y = cscalar::<R>(0.0, ky);
        let div = factor_x * gx + factor_y * gy;
        *out = -div;
    }
}

fn arithmetic_mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let sum: f64 = values.iter().copied().sum();
    Some(sum / values.len() as f64)
}

fn harmonic_mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let sum: f64 = values.iter().copied().sum();
    if sum <= 0.0 {
        return None;
    }
    Some(1.0 / (sum / values.len() as f64))
}

/// Apply ∂ε⁻¹/∂Rⱼ to gradient vectors in spatial domain.
///
/// Analogous to `apply_inv_eps` but uses the derivative of the inverse
/// dielectric tensor from a precomputed `DielectricDerivative`.
fn apply_dielectric_derivative<R: Real>(
    grad_x: &mut [Complex<R>],
    grad_y: &mut [Complex<R>],
    diel_deriv: &DielectricDerivative,
) {
    if let Some(ref d_tensors) = diel_deriv.d_inv_eps_tensors {
        // Anisotropic: multiply by ∂ε⁻¹/∂Rⱼ tensor (2×2 matrix per grid point)
        for ((gx, gy), tensor) in grad_x.iter_mut().zip(grad_y.iter_mut()).zip(d_tensors.iter()) {
            let orig_x = *gx;
            let orig_y = *gy;
            let t0 = R::from_accum(tensor[0]);
            let t1 = R::from_accum(tensor[1]);
            let t2 = R::from_accum(tensor[2]);
            let t3 = R::from_accum(tensor[3]);
            *gx = orig_x * t0 + orig_y * t1;
            *gy = orig_x * t2 + orig_y * t3;
        }
    } else {
        // Isotropic: multiply by scalar ∂(1/ε)/∂Rⱼ
        for ((gx, gy), &d_inv) in grad_x
            .iter_mut()
            .zip(grad_y.iter_mut())
            .zip(diel_deriv.d_inv_eps.iter())
        {
            let s = R::from_accum(d_inv);
            *gx *= s;
            *gy *= s;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_gradient_handles_zero_and_nan() {
        let magnitude = K_PLUS_G_NEAR_ZERO_FLOOR.sqrt();
        let (x_zero, y_zero) = clamp_gradient_components(0.0, 0.0);
        assert_eq!(x_zero, magnitude);
        assert_eq!(y_zero, 0.0);

        let (x_nan, y_nan) = clamp_gradient_components(f64::NAN, f64::NAN);
        assert_eq!(x_nan, magnitude);
        assert_eq!(y_nan, 0.0);
    }

    #[test]
    fn inverse_scale_sanitizes_non_finite_and_underflow() {
        assert_eq!(inverse_scale(f64::NAN, 1e-3, 1.0, 0.0), 0.0);
        assert_eq!(inverse_scale(1.0, 1e-3, f64::NAN, 0.0), 0.0);

        let tiny = K_PLUS_G_NEAR_ZERO_FLOOR / 10.0;
        let shift = 1e-3;
        let eps_eff = 1.0;
        let shift_scaled = shift * eps_eff;
        let expected = eps_eff / (tiny + shift_scaled);
        let actual = inverse_scale(tiny, shift, eps_eff, 0.0);
        assert!((actual - expected).abs() < 1e-12);
    }

    #[test]
    fn te_mass_floor_enters_denominator() {
        let eps_eff = 12.0;
        let mass_floor = tm_preconditioner_mass_floor(eps_eff);
        let tiny = K_PLUS_G_NEAR_ZERO_FLOOR / 10.0;
        let baseline = inverse_scale(tiny, 1e-3, eps_eff, 0.0);
        let mass_adjusted = inverse_scale(tiny, 1e-3, eps_eff, mass_floor);
        assert!(mass_floor > 0.0);
        assert!(mass_adjusted < baseline);
    }
}
