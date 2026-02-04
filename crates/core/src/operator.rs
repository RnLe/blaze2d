//! Operator implementations (toy Laplacian + physical Θ operator).

use num_complex::Complex64;

use std::{
    collections::HashMap,
    f64::consts::PI,
    sync::{Arc, Mutex, OnceLock},
};

use crate::{
    backend::{SpectralBackend, SpectralBuffer},
    dielectric::Dielectric2D,
    grid::Grid2D,
    polarization::Polarization,
    preconditioner::{
        FOURIER_DIAGONAL_SHIFT, FourierDiagonalPreconditioner, SpectralStats,
        TransverseProjectionPreconditioner,
    },
};

pub(crate) const K_PLUS_G_NEAR_ZERO_FLOOR: f64 = 1e-9;
pub(crate) const TE_PRECONDITIONER_MASS_FRACTION: f64 = 1e-2;
pub(crate) const STRUCTURED_WEIGHT_MIN: f64 = 1e-3;
pub(crate) const STRUCTURED_WEIGHT_MAX: f64 = 1e3;

pub trait LinearOperator<B: SpectralBackend> {
    fn apply(&mut self, input: &B::Buffer, output: &mut B::Buffer);
    fn apply_mass(&mut self, input: &B::Buffer, output: &mut B::Buffer);
    fn alloc_field(&self) -> B::Buffer;
    fn backend(&self) -> &B;
    fn backend_mut(&mut self) -> &mut B;
    fn grid(&self) -> Grid2D;
    /// Get the Bloch wavevector (k-point) for this operator.
    fn bloch(&self) -> [f64; 2];

    /// Get the transformation factor for Γ-point kernel basis.
    ///
    /// For operators using a similarity transform (like transformed TE where
    /// A' = ε^{-1/2} A ε^{-1/2}), the kernel basis must also be transformed:
    /// v₀ = ε^{1/2} u₀ instead of the naive constant mode u₀.
    ///
    /// Returns `Some(&[f64])` with the pointwise transformation factor (ε^{1/2}),
    /// or `None` if no transformation is needed (standard formulation).
    fn gamma_kernel_transform(&self) -> Option<&[f64]> {
        None // Default: no transformation needed
    }
}

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
    /// Whether to use the transformed TE formulation (B = I).
    /// When true, the TE operator becomes A' = ε^{-1/2} · A · ε^{-1/2}
    /// and the mass operator becomes identity.
    use_transformed_te: bool,
    /// Precomputed ε^{1/2}(r) for the transformation (only for TE with transformation).
    eps_sqrt: Option<Vec<f64>>,
    /// Precomputed ε^{-1/2}(r) for the transformation (only for TE with transformation).
    eps_inv_sqrt: Option<Vec<f64>>,
}

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

impl<B: SpectralBackend> ThetaOperator<B> {
    /// Create a new ThetaOperator with the generalized eigenproblem formulation.
    ///
    /// For TE mode, this uses A·x = λ·B·x where B = ε(r).
    /// For TM mode, B = I (already standard).
    pub fn new(
        backend: B,
        dielectric: Dielectric2D,
        polarization: Polarization,
        bloch_k: [f64; 2],
    ) -> Self {
        Self::new_internal(backend, dielectric, polarization, bloch_k, false)
    }

    /// Create a new ThetaOperator with the transformed TE formulation (B = I).
    ///
    /// For TE mode, this transforms the problem so that:
    /// - A' = ε^{-1/2} · A · ε^{-1/2}  (transformed stiffness)
    /// - B' = I (identity mass)
    ///
    /// This converts the generalized eigenproblem to a standard one,
    /// simplifying orthogonalization and Rayleigh quotients.
    ///
    /// For TM mode, this is equivalent to `new()` since B = I already.
    pub fn new_transformed(
        backend: B,
        dielectric: Dielectric2D,
        polarization: Polarization,
        bloch_k: [f64; 2],
    ) -> Self {
        Self::new_internal(backend, dielectric, polarization, bloch_k, true)
    }

    fn new_internal(
        backend: B,
        dielectric: Dielectric2D,
        polarization: Polarization,
        bloch_k: [f64; 2],
        use_transformed_te: bool,
    ) -> Self {
        let grid = dielectric.grid;
        let wave_tables = cached_wave_tables(grid);
        let kx_shifted = shift_k_vector(wave_tables.kx(), bloch_k[0]);
        let ky_shifted = shift_k_vector(wave_tables.ky(), bloch_k[1]);
        let (
            k_plus_g_x,
            k_plus_g_y,
            k_plus_g_sq,
            k_plus_g_sq_min_raw,
            k_plus_g_sq_min,
            k_plus_g_floor_count,
            k_plus_g_was_clamped,
        ) = build_shifted_k_plus_g_tables(&wave_tables, bloch_k);
        let scratch = backend.alloc_field(grid);
        let grad_x = backend.alloc_field(grid);
        let grad_y = backend.alloc_field(grid);

        // Precompute ε^{1/2} and ε^{-1/2} if using transformed TE
        let (eps_sqrt, eps_inv_sqrt) = if use_transformed_te && polarization == Polarization::TE {
            let eps = dielectric.eps();
            let sqrt: Vec<f64> = eps.iter().map(|&e| e.sqrt()).collect();
            let inv_sqrt: Vec<f64> = eps.iter().map(|&e| 1.0 / e.sqrt()).collect();
            (Some(sqrt), Some(inv_sqrt))
        } else {
            (None, None)
        };

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
            use_transformed_te,
            eps_sqrt,
            eps_inv_sqrt,
        }
    }

    /// Returns whether this operator uses the transformed TE formulation.
    pub fn is_transformed(&self) -> bool {
        self.use_transformed_te && self.polarization == Polarization::TE
    }

    /// Returns whether this operator is at the Γ-point (k ≈ 0).
    ///
    /// At Γ, the G=0 mode has |k+G|² = 0 and must be deflated.
    /// Away from Γ, the G=0 mode has |k+G|² = |k|² > 0 and is a legitimate mode.
    pub fn is_gamma(&self) -> bool {
        const GAMMA_TOL: f64 = 1e-12;
        self.bloch[0].abs() < GAMMA_TOL && self.bloch[1].abs() < GAMMA_TOL
    }

    /// Get ε^{1/2} values if using transformed TE mode.
    ///
    /// For transformed TE, the kernel of A' = ε^{-1/2} A ε^{-1/2} is v₀ = ε^{1/2} u₀,
    /// where u₀ is the constant mode. This accessor is needed for proper Γ deflation.
    pub fn eps_sqrt(&self) -> Option<&[f64]> {
        self.eps_sqrt.as_deref()
    }

    pub fn alloc_field(&self) -> B::Buffer {
        self.backend.alloc_field(self.grid)
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    /// Compute spectral statistics for the current k-point.
    ///
    /// Returns statistics about the |k+G|² spectrum, which can be used
    /// for k-dependent preconditioner regularization.
    pub fn spectral_stats(&self) -> SpectralStats {
        SpectralStats::compute(&self.k_plus_g_sq)
    }

    /// Build homogeneous preconditioner with legacy global shift.
    ///
    /// Uses a fixed σ = 1e-3 shift for regularization, which doesn't
    /// scale with the k-point's spectral properties.
    pub fn build_homogeneous_preconditioner(&self) -> FourierDiagonalPreconditioner {
        self.build_homogeneous_preconditioner_with_shift(FOURIER_DIAGONAL_SHIFT)
    }

    /// Build homogeneous preconditioner with k-dependent shift.
    ///
    /// Uses σ(k) = β * s_median(k), where s_median is the median of |k+G|².
    /// This ensures the shift scales with the local spectral range.
    pub fn build_homogeneous_preconditioner_adaptive(&self) -> FourierDiagonalPreconditioner {
        let stats = self.spectral_stats();
        let shift = stats.adaptive_shift();
        log::debug!(
            "preconditioner: adaptive shift σ(k)={:.2e} (α={:.1}, s_min={:.2e}, s_med={:.2e}, s_max={:.2e})",
            shift,
            crate::preconditioner::SHIFT_SMIN_FRACTION,
            stats.s_min,
            stats.s_median,
            stats.s_max
        );
        self.build_homogeneous_preconditioner_with_shift(shift)
    }

    /// Build homogeneous preconditioner with a specific shift value.
    ///
    /// # Kernel Compensation Strategy
    ///
    /// - **At Γ (k=0)**: Zero the G=0 mode in the preconditioner. This mode is
    ///   deflated from the eigenproblem anyway, so zeroing it prevents amplification.
    /// - **Away from Γ (k≠0)**: Use floor-based regularization only (no hard zeroing).
    ///   The G=0 mode now has |k+G|² = |k|² > 0 and is a legitimate physical mode.
    ///   Hard-zeroing it would destroy the long-wavelength components that dominate
    ///   the first bands near Γ, causing catastrophic iteration counts.
    fn build_homogeneous_preconditioner_with_shift(
        &self,
        shift: f64,
    ) -> FourierDiagonalPreconditioner {
        // Only zero near-zero modes at Γ-point where they're actually in the null space.
        // Away from Γ, these modes have |k+G|² = |k|² > 0 and must NOT be zeroed.
        let near_zero_mask = if self.is_gamma() {
            Some(self.k_plus_g_was_clamped.as_slice())
        } else {
            None // No kernel compensation away from Γ - use regularization floor only
        };

        match self.polarization {
            Polarization::TE => {
                // For transformed TE, use ε^{1/2} weights + kernel-compensation
                // This combines the best of both approaches
                if self.is_transformed() {
                    if let Some(eps_sqrt) = &self.eps_sqrt {
                        let inverse_diagonal = build_inverse_diagonal(
                            &self.k_plus_g_sq,
                            shift,
                            1.0, // No eps_eff scaling - weights handle it
                            0.0, // No mass floor needed
                            near_zero_mask, // Kernel-compensation: zero near-zero modes ONLY at Γ
                        );
                        return FourierDiagonalPreconditioner::with_weights(
                            inverse_diagonal,
                            eps_sqrt.clone(),
                        );
                    }
                }
                // Untransformed TE: generalized eigenproblem (A x = λ B x, B = ε)
                let eps_eff = self.effective_te_epsilon();
                let mass_floor = te_preconditioner_mass_floor(eps_eff);
                let inverse_diagonal = build_inverse_diagonal(
                    &self.k_plus_g_sq,
                    shift,
                    eps_eff,
                    mass_floor,
                    near_zero_mask, // Kernel-compensation ONLY at Γ
                );
                FourierDiagonalPreconditioner::new(inverse_diagonal)
            }
            Polarization::TM => {
                let eps_eff = self.effective_tm_epsilon();
                let inverse_diagonal =
                    build_inverse_diagonal(&self.k_plus_g_sq, shift, eps_eff, 0.0, near_zero_mask);
                FourierDiagonalPreconditioner::new(inverse_diagonal)
            }
        }
    }

    /// Build structured preconditioner with legacy global shift.
    ///
    /// Uses a fixed σ = 1e-3 shift for regularization.
    pub fn build_structured_preconditioner(&self) -> FourierDiagonalPreconditioner {
        self.build_structured_preconditioner_with_shift(FOURIER_DIAGONAL_SHIFT)
    }

    /// Build structured preconditioner with k-dependent shift.
    ///
    /// Uses σ(k) = β * s_median(k), where s_median is the median of |k+G|².
    pub fn build_structured_preconditioner_adaptive(&self) -> FourierDiagonalPreconditioner {
        let stats = self.spectral_stats();
        let shift = stats.adaptive_shift();
        log::debug!(
            "preconditioner: adaptive shift σ(k)={:.2e} (α={:.1}, s_min={:.2e}, s_med={:.2e}, s_max={:.2e})",
            shift,
            crate::preconditioner::SHIFT_SMIN_FRACTION,
            stats.s_min,
            stats.s_median,
            stats.s_max
        );
        self.build_structured_preconditioner_with_shift(shift)
    }

    /// Build structured preconditioner with a specific shift value.
    ///
    /// For transformed TE mode, uses ε^{1/2} as weights to approximate (A')^{-1}:
    ///   M^{-1} = ε^{1/2} · F^{-1} · D · F · ε^{1/2}
    /// where D = 1/(|k+G|² + σ²). This directly approximates the inverse of
    /// A' = ε^{-1/2} · (-∇²) · ε^{-1/2}.
    ///
    /// # Kernel Compensation Strategy
    ///
    /// Same as homogeneous preconditioner: only zero near-zero modes at Γ.
    fn build_structured_preconditioner_with_shift(
        &self,
        shift: f64,
    ) -> FourierDiagonalPreconditioner {
        // Only zero near-zero modes at Γ-point where they're actually in the null space.
        // Away from Γ, these modes have |k+G|² = |k|² > 0 and must NOT be zeroed.
        let near_zero_mask = if self.is_gamma() {
            Some(self.k_plus_g_was_clamped.as_slice())
        } else {
            None // No kernel compensation away from Γ - use regularization floor only
        };

        match self.polarization {
            Polarization::TE => {
                // For transformed TE, use ε^{1/2} directly as weights
                // This gives M^{-1} ≈ (A')^{-1} = ε^{1/2} · (-∇²)^{-1} · ε^{1/2}
                if self.is_transformed() {
                    if let Some(eps_sqrt) = &self.eps_sqrt {
                        // Use 1/(|k+G|² + σ²) without eps_eff scaling
                        // The ε^{1/2} weights handle the dielectric structure
                        let inverse_diagonal = build_inverse_diagonal(
                            &self.k_plus_g_sq,
                            shift,
                            1.0, // No eps_eff scaling - weights handle it
                            0.0, // No mass floor needed
                            near_zero_mask, // Kernel-compensation ONLY at Γ
                        );
                        return FourierDiagonalPreconditioner::with_weights(
                            inverse_diagonal,
                            eps_sqrt.clone(),
                        );
                    }
                }
                // Untransformed TE: generalized eigenproblem (A x = λ B x, B = ε)
                // Use ε/ε_eff weights for structured preconditioning
                let eps_eff = self.effective_te_epsilon();
                let mass_floor = te_preconditioner_mass_floor(eps_eff);
                let inverse_diagonal = build_inverse_diagonal(
                    &self.k_plus_g_sq,
                    shift,
                    eps_eff,
                    mass_floor,
                    near_zero_mask, // Kernel-compensation ONLY at Γ
                );
                let weights = build_structured_weights_te(&self.dielectric, eps_eff);
                FourierDiagonalPreconditioner::with_weights(inverse_diagonal, weights)
            }
            Polarization::TM => {
                let eps_eff = self.effective_tm_epsilon();
                let inverse_diagonal =
                    build_inverse_diagonal(&self.k_plus_g_sq, shift, eps_eff, 0.0, near_zero_mask);
                let weights = build_structured_weights_tm(&self.dielectric);
                FourierDiagonalPreconditioner::with_weights(inverse_diagonal, weights)
            }
        }
    }

    #[allow(dead_code)]
    pub fn build_fourier_diagonal_preconditioner(&self) -> FourierDiagonalPreconditioner {
        self.build_homogeneous_preconditioner()
    }

    /// Build the MPB-style transverse-projection preconditioner.
    ///
    /// This is the most effective preconditioner for TM mode, as it accounts for
    /// the spatial variation of ε(r) in the approximate inverse. For TE mode,
    /// it falls back to a Fourier-diagonal preconditioner since the operator is
    /// already diagonal in Fourier space.
    ///
    /// Uses the legacy global shift (σ = 1e-3) for regularization.
    ///
    /// # Performance
    ///
    /// - TM mode: 6 FFTs per application (vs 2 for Fourier-diagonal)
    /// - TE mode: 2 FFTs per application (same as Fourier-diagonal)
    ///
    /// The extra cost for TM is typically offset by the dramatic reduction in
    /// iteration count (often 5-10× fewer iterations).
    pub fn build_transverse_projection_preconditioner(
        &self,
    ) -> TransverseProjectionPreconditioner<B> {
        self.build_transverse_projection_preconditioner_with_shift(FOURIER_DIAGONAL_SHIFT)
    }

    /// Build the MPB-style transverse-projection preconditioner with adaptive shift.
    ///
    /// Uses σ(k) = α * s_min(k), where s_min is the smallest nonzero |k+G|².
    /// This ensures proper scaling at different k-points.
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
    ///
    /// Uses power iteration to estimate λ_max. For λ_min, we use the smallest
    /// non-zero |k+G|² as an approximation (exact for homogeneous dielectric).
    ///
    /// # Arguments
    /// * `n_iters` - Number of power iterations (default ~10-20 is usually sufficient)
    ///
    /// # Returns
    /// (λ_max_estimate, λ_min_estimate, κ_estimate)
    pub fn estimate_condition_number(&mut self, n_iters: usize) -> (f64, f64, f64) {
        // Power iteration for λ_max
        let mut v = self.alloc_field();
        let mut av = self.alloc_field();

        // Initialize with random-ish vector (use k+G values as pseudo-random)
        for (i, val) in v.as_mut_slice().iter_mut().enumerate() {
            *val = Complex64::new(
                (i as f64 * 0.618033988749895).sin(), // Golden ratio for pseudo-random
                (i as f64 * 0.414213562373095).cos(), // sqrt(2)-1
            );
        }

        // Normalize
        let norm: f64 = v
            .as_slice()
            .iter()
            .map(|c| c.norm_sqr())
            .sum::<f64>()
            .sqrt();
        if norm > 1e-15 {
            for val in v.as_mut_slice().iter_mut() {
                *val /= norm;
            }
        }

        let mut lambda_max = 0.0;
        for _ in 0..n_iters {
            // av = A * v
            self.apply(&v, &mut av);

            // Rayleigh quotient: λ ≈ v* A v / v* v
            let numerator: f64 = v
                .as_slice()
                .iter()
                .zip(av.as_slice().iter())
                .map(|(vi, avi)| (vi.conj() * avi).re)
                .sum();
            lambda_max = numerator; // v is normalized, so v*v = 1

            // Normalize av for next iteration
            let norm: f64 = av
                .as_slice()
                .iter()
                .map(|c| c.norm_sqr())
                .sum::<f64>()
                .sqrt();
            if norm > 1e-15 {
                for val in av.as_mut_slice().iter_mut() {
                    *val /= norm;
                }
            }
            std::mem::swap(&mut v, &mut av);
        }

        // For λ_min, use the smallest non-zero |k+G|²
        // This is exact for TM with uniform ε, and a reasonable lower bound otherwise
        let lambda_min = self.k_plus_g_sq_min;

        let kappa = if lambda_min > 1e-15 {
            lambda_max / lambda_min
        } else {
            f64::INFINITY
        };

        (lambda_max, lambda_min, kappa)
    }

    /// Estimate the condition number of the preconditioned operator M⁻¹A.
    ///
    /// For a well-chosen preconditioner, κ(M⁻¹A) should be much smaller than κ(A).
    ///
    /// # Arguments
    /// * `precond` - The preconditioner to evaluate
    /// * `n_iters` - Number of power iterations
    ///
    /// # Returns
    /// (λ_max_estimate, λ_min_estimate, κ_estimate) for M⁻¹A
    pub fn estimate_preconditioned_condition_number(
        &mut self,
        precond: &mut dyn crate::preconditioner::OperatorPreconditioner<B>,
        n_iters: usize,
    ) -> (f64, f64, f64) {
        // Power iteration for λ_max of M⁻¹A
        let mut v = self.alloc_field();
        let mut av = self.alloc_field();

        // Initialize with pseudo-random vector
        for (i, val) in v.as_mut_slice().iter_mut().enumerate() {
            *val = Complex64::new(
                (i as f64 * 0.618033988749895).sin(),
                (i as f64 * 0.414213562373095).cos(),
            );
        }

        // Normalize
        let norm: f64 = v
            .as_slice()
            .iter()
            .map(|c| c.norm_sqr())
            .sum::<f64>()
            .sqrt();
        if norm > 1e-15 {
            for val in v.as_mut_slice().iter_mut() {
                *val /= norm;
            }
        }

        let mut lambda_max = 0.0;
        for _ in 0..n_iters {
            // av = A * v
            self.apply(&v, &mut av);
            // av = M⁻¹ * av (in-place)
            precond.apply(&self.backend, &mut av);

            // Rayleigh quotient: λ ≈ v* M⁻¹A v / v* v
            let numerator: f64 = v
                .as_slice()
                .iter()
                .zip(av.as_slice().iter())
                .map(|(vi, avi)| (vi.conj() * avi).re)
                .sum();
            lambda_max = numerator;

            // Normalize for next iteration
            let norm: f64 = av
                .as_slice()
                .iter()
                .map(|c| c.norm_sqr())
                .sum::<f64>()
                .sqrt();
            if norm > 1e-15 {
                for val in av.as_mut_slice().iter_mut() {
                    *val /= norm;
                }
            }
            std::mem::swap(&mut v, &mut av);
        }

        // For λ_min of M⁻¹A, use inverse power iteration
        // Re-initialize
        for (i, val) in v.as_mut_slice().iter_mut().enumerate() {
            *val = Complex64::new(
                (i as f64 * 1.414213562373095).sin(),
                (i as f64 * 1.732050807568877).cos(),
            );
        }
        let norm: f64 = v
            .as_slice()
            .iter()
            .map(|c| c.norm_sqr())
            .sum::<f64>()
            .sqrt();
        if norm > 1e-15 {
            for val in v.as_mut_slice().iter_mut() {
                *val /= norm;
            }
        }

        // For the minimum eigenvalue, we'd need (M⁻¹A)⁻¹ = A⁻¹M, which is expensive.
        // Instead, estimate from the preconditioner spectrum: for a diagonal preconditioner
        // M⁻¹(q) ≈ ε_eff / (|k+G|² + σ²), the eigenvalues of M⁻¹A are approximately
        // λ ≈ |k+G|² * ε_eff / (|k+G|² + σ²), which ranges from near 0 (small |k+G|²)
        // to ε_eff (large |k+G|²).
        //
        // A practical estimate: λ_min ≈ s_min * ε_eff / (s_min + σ²) where s_min is
        // the smallest |k+G|². But this can be near 0, so we use 1.0 as a rough floor.
        let lambda_min_approx = 1.0; // Rough estimate - ideal preconditioner gives λ_min ≈ 1

        let kappa = if lambda_min_approx > 1e-15 {
            lambda_max / lambda_min_approx
        } else {
            f64::INFINITY
        };

        (lambda_max, lambda_min_approx, kappa)
    }

    /// Check self-adjointness of the operator: |⟨Ax, y⟩ - ⟨x, Ay⟩| / (‖Ax‖‖y‖)
    ///
    /// For a self-adjoint operator, this should be close to machine epsilon.
    /// Returns the relative deviation from self-adjointness.
    pub fn check_self_adjointness(&mut self) -> f64 {
        let mut x = self.alloc_field();
        let mut y = self.alloc_field();
        let mut ax = self.alloc_field();
        let mut ay = self.alloc_field();

        // Initialize with pseudo-random vectors
        for (i, val) in x.as_mut_slice().iter_mut().enumerate() {
            *val = Complex64::new(
                (i as f64 * 0.618033988749895).sin(),
                (i as f64 * 0.414213562373095).cos(),
            );
        }
        for (i, val) in y.as_mut_slice().iter_mut().enumerate() {
            *val = Complex64::new(
                (i as f64 * 1.414213562373095).cos(),
                (i as f64 * 1.732050807568877).sin(),
            );
        }

        // Compute Ax and Ay
        self.apply(&x, &mut ax);
        self.apply(&y, &mut ay);

        // Compute ⟨Ax, y⟩ and ⟨x, Ay⟩
        let ax_y: Complex64 = ax
            .as_slice()
            .iter()
            .zip(y.as_slice().iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        let x_ay: Complex64 = x
            .as_slice()
            .iter()
            .zip(ay.as_slice().iter())
            .map(|(a, b)| a.conj() * b)
            .sum();

        // Compute norms for normalization
        let norm_ax: f64 = ax.as_slice().iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        let norm_y: f64 = y.as_slice().iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();

        let diff = (ax_y - x_ay).norm();
        let scale = norm_ax * norm_y;

        if scale > 1e-15 {
            diff / scale
        } else {
            0.0
        }
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

    /// Get effective epsilon values for logging
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

    pub fn capture_snapshot(&mut self, input: &B::Buffer) -> OperatorSnapshotData {
        match self.polarization {
            Polarization::TM => self.capture_tm_snapshot(input),
            Polarization::TE => self.capture_te_snapshot(input),
        }
    }

    fn apply_tm(&mut self, input: &B::Buffer, output: &mut B::Buffer) {
        copy_buffer(&mut self.scratch, input);
        self.backend.forward_fft_2d(&mut self.scratch);

        copy_buffer(&mut self.grad_x, &self.scratch);
        copy_buffer(&mut self.grad_y, &self.scratch);

        apply_gradient_factors(
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
            self.scratch.as_mut_slice(),
            &self.k_plus_g_x,
            &self.k_plus_g_y,
        );

        self.backend.inverse_fft_2d(&mut self.scratch);
        copy_buffer(output, &self.scratch);
    }

    /// Apply the TE operator: A·x = |k+G|²·x (Laplacian in Fourier space)
    ///
    /// For transformed mode: A' = ε^{-1/2} · A · ε^{-1/2}
    fn apply_te(&mut self, input: &B::Buffer, output: &mut B::Buffer) {
        if self.use_transformed_te {
            // Transformed TE: A' = ε^{-1/2} · A · ε^{-1/2}
            // Step 1: multiply by ε^{-1/2} in real space
            let eps_inv_sqrt = self.eps_inv_sqrt.as_ref().unwrap();
            copy_buffer(&mut self.scratch, input);
            for (value, &inv_sqrt) in self
                .scratch
                .as_mut_slice()
                .iter_mut()
                .zip(eps_inv_sqrt.iter())
            {
                *value *= inv_sqrt;
            }

            // Step 2: apply Laplacian in Fourier space
            self.backend.forward_fft_2d(&mut self.scratch);
            for (value, &k_sq) in self
                .scratch
                .as_mut_slice()
                .iter_mut()
                .zip(self.k_plus_g_sq.iter())
            {
                *value *= k_sq;
            }
            self.backend.inverse_fft_2d(&mut self.scratch);

            // Step 3: multiply by ε^{-1/2} in real space
            for (value, &inv_sqrt) in self
                .scratch
                .as_mut_slice()
                .iter_mut()
                .zip(eps_inv_sqrt.iter())
            {
                *value *= inv_sqrt;
            }

            copy_buffer(output, &self.scratch);
        } else {
            // Standard TE: A·x = |k+G|²·x
            copy_buffer(&mut self.scratch, input);
            self.backend.forward_fft_2d(&mut self.scratch);
            let data = self.scratch.as_mut_slice();
            #[cfg(debug_assertions)]
            {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    let max_k_sq = self.k_plus_g_sq.iter().cloned().fold(0.0_f64, f64::max);
                    let mut sorted: Vec<f64> = self.k_plus_g_sq.iter().cloned().collect();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    eprintln!(
                        "[operator-debug] TE k_plus_g_sq first 8: {:?}",
                        &sorted[..sorted.len().min(8)]
                    );
                    eprintln!("[operator-debug] TE k_plus_g_sq max={:.4}", max_k_sq);
                });
            }
            for (value, &k_sq) in data.iter_mut().zip(self.k_plus_g_sq.iter()) {
                *value *= k_sq;
            }
            self.backend.inverse_fft_2d(&mut self.scratch);
            copy_buffer(output, &self.scratch);
        }
    }

    fn capture_tm_snapshot(&mut self, input: &B::Buffer) -> OperatorSnapshotData {
        let field_spatial = input.as_slice().to_vec();
        copy_buffer(&mut self.scratch, input);
        self.backend.forward_fft_2d(&mut self.scratch);
        let field_fourier = self.scratch.as_slice().to_vec();

        copy_buffer(&mut self.grad_x, &self.scratch);
        copy_buffer(&mut self.grad_y, &self.scratch);
        apply_gradient_factors(
            self.grad_x.as_mut_slice(),
            self.grad_y.as_mut_slice(),
            &self.k_plus_g_x,
            &self.k_plus_g_y,
        );

        self.backend.inverse_fft_2d(&mut self.grad_x);
        self.backend.inverse_fft_2d(&mut self.grad_y);
        let grad_x = self.grad_x.as_slice().to_vec();
        let grad_y = self.grad_y.as_slice().to_vec();

        apply_inv_eps(
            self.grad_x.as_mut_slice(),
            self.grad_y.as_mut_slice(),
            &self.dielectric,
        );
        let eps_grad_x = self.grad_x.as_slice().to_vec();
        let eps_grad_y = self.grad_y.as_slice().to_vec();

        self.backend.forward_fft_2d(&mut self.grad_x);
        self.backend.forward_fft_2d(&mut self.grad_y);

        assemble_divergence(
            self.grad_x.as_slice(),
            self.grad_y.as_slice(),
            self.scratch.as_mut_slice(),
            &self.k_plus_g_x,
            &self.k_plus_g_y,
        );
        let theta_fourier = self.scratch.as_slice().to_vec();

        self.backend.inverse_fft_2d(&mut self.scratch);
        let theta_spatial = self.scratch.as_slice().to_vec();

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

    fn capture_te_snapshot(&mut self, input: &B::Buffer) -> OperatorSnapshotData {
        let field_spatial = input.as_slice().to_vec();
        copy_buffer(&mut self.scratch, input);
        self.backend.forward_fft_2d(&mut self.scratch);
        let field_fourier = self.scratch.as_slice().to_vec();

        copy_buffer(&mut self.grad_x, &self.scratch);
        copy_buffer(&mut self.grad_y, &self.scratch);
        apply_gradient_factors(
            self.grad_x.as_mut_slice(),
            self.grad_y.as_mut_slice(),
            &self.k_plus_g_x,
            &self.k_plus_g_y,
        );

        self.backend.inverse_fft_2d(&mut self.grad_x);
        self.backend.inverse_fft_2d(&mut self.grad_y);
        let grad_x = self.grad_x.as_slice().to_vec();
        let grad_y = self.grad_y.as_slice().to_vec();

        for (value, &k_sq) in self
            .scratch
            .as_mut_slice()
            .iter_mut()
            .zip(self.k_plus_g_sq.iter())
        {
            *value *= k_sq;
        }
        let theta_fourier = self.scratch.as_slice().to_vec();

        self.backend.inverse_fft_2d(&mut self.scratch);
        let theta_spatial = self.scratch.as_slice().to_vec();

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
}

impl<B: SpectralBackend> ThetaOperator<B> {
    fn effective_te_epsilon(&self) -> f64 {
        arithmetic_mean(self.dielectric.eps()).unwrap_or(1.0)
    }

    fn effective_tm_epsilon(&self) -> f64 {
        harmonic_mean(self.dielectric.inv_eps()).unwrap_or(1.0)
    }
}

impl<B: SpectralBackend> LinearOperator<B> for ThetaOperator<B> {
    fn apply(&mut self, input: &B::Buffer, output: &mut B::Buffer) {
        match self.polarization {
            Polarization::TM => self.apply_tm(input, output),
            Polarization::TE => self.apply_te(input, output),
        }
    }

    fn apply_mass(&mut self, input: &B::Buffer, output: &mut B::Buffer) {
        match self.polarization {
            Polarization::TM => copy_buffer(output, input),
            Polarization::TE => {
                if self.is_transformed() {
                    // Transformed TE mode: B = I (standard eigenproblem)
                    copy_buffer(output, input);
                } else {
                    // Generalized eigenproblem: B = ε
                    copy_buffer(output, input);
                    apply_scalar_eps(output.as_mut_slice(), self.dielectric.eps());
                }
            }
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
        // For transformed TE, the kernel of A' = ε^{-1/2} A ε^{-1/2} is v₀ = ε^{1/2} u₀
        // where u₀ is the constant mode. We return ε^{1/2} as the transformation factor.
        if self.is_transformed() {
            self.eps_sqrt.as_deref()
        } else {
            None
        }
    }
}

pub struct ToyLaplacian<B: SpectralBackend> {
    backend: B,
    grid: Grid2D,
    kx: Vec<f64>,
    ky: Vec<f64>,
    scratch: B::Buffer,
}

impl<B: SpectralBackend> ToyLaplacian<B> {
    pub fn new(backend: B, grid: Grid2D) -> Self {
        assert!(
            grid.nx > 0 && grid.ny > 0,
            "grid must have non-zero dimensions"
        );
        assert!(
            grid.lx > 0.0 && grid.ly > 0.0,
            "grid lengths must be positive"
        );
        let kx = build_k_vector(grid.nx, grid.lx);
        let ky = build_k_vector(grid.ny, grid.ly);
        let scratch = backend.alloc_field(grid);
        Self {
            backend,
            grid,
            kx,
            ky,
            scratch,
        }
    }

    pub fn grid(&self) -> Grid2D {
        self.grid
    }

    pub fn alloc_field(&self) -> B::Buffer {
        self.backend.alloc_field(self.grid)
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }
}

impl<B: SpectralBackend> LinearOperator<B> for ToyLaplacian<B> {
    fn apply(&mut self, input: &B::Buffer, output: &mut B::Buffer) {
        copy_buffer(&mut self.scratch, input);
        self.backend.forward_fft_2d(&mut self.scratch);
        let data = self.scratch.as_mut_slice();
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        for iy in 0..ny {
            for ix in 0..nx {
                let idx = iy * nx + ix;
                let k2 = self.kx[ix] * self.kx[ix] + self.ky[iy] * self.ky[iy];
                data[idx] *= k2;
            }
        }
        self.backend.inverse_fft_2d(&mut self.scratch);
        copy_buffer(output, &self.scratch);
    }

    fn apply_mass(&mut self, input: &B::Buffer, output: &mut B::Buffer) {
        copy_buffer(output, input);
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
        [0.0, 0.0] // ToyLaplacian doesn't use Bloch wavevectors
    }
}

// ============================================================================
// Toy Diagonal SPD Operator for Eigensolver Testing
// ============================================================================

/// A simple diagonal SPD operator with known eigenvalues for testing the eigensolver.
///
/// This operator has eigenvalues λ_k = 1 + k for k = 0, 1, 2, ... (N-1) in Fourier space,
/// where each Fourier mode is an exact eigenvector. This allows us to verify that
/// the eigensolver converges to machine precision on a well-conditioned problem.
///
/// The operator is:
/// - **Self-adjoint**: A = A^* (diagonal in Fourier space with real entries)
/// - **Positive definite**: All eigenvalues λ_k ≥ 1 > 0
/// - **Well-conditioned**: κ = λ_max / λ_min = N / 1 = N (linear in grid size)
///
/// The mass operator is the identity (B = I), so this is a standard eigenproblem.
///
/// # Expected eigenvalues
/// For an N×N grid, the smallest eigenvalues are:
/// - λ_0 = 1 (constant mode at index 0)
/// - λ_1 = 2, λ_2 = 3, ... (subsequent modes)
///
/// The eigenvectors are the standard Fourier basis functions.
pub struct ToyDiagonalSPD<B: SpectralBackend> {
    backend: B,
    grid: Grid2D,
    /// Eigenvalues λ_k = 1 + k for each Fourier mode (sorted by magnitude)
    eigenvalues: Vec<f64>,
    scratch: B::Buffer,
}

impl<B: SpectralBackend> ToyDiagonalSPD<B> {
    /// Create a new ToyDiagonalSPD operator.
    ///
    /// The eigenvalues are λ_k = 1 + k for k = 0, 1, 2, ..., N-1.
    pub fn new(backend: B, grid: Grid2D) -> Self {
        assert!(
            grid.nx > 0 && grid.ny > 0,
            "grid must have non-zero dimensions"
        );

        let n = grid.len();
        // Eigenvalues: 1, 2, 3, ..., N
        let eigenvalues: Vec<f64> = (0..n).map(|k| 1.0 + k as f64).collect();
        let scratch = backend.alloc_field(grid);

        Self {
            backend,
            grid,
            eigenvalues,
            scratch,
        }
    }

    /// Get the exact eigenvalues (for verification).
    pub fn exact_eigenvalues(&self, n_bands: usize) -> Vec<f64> {
        self.eigenvalues[..n_bands.min(self.eigenvalues.len())].to_vec()
    }

    /// Get the condition number κ = λ_max / λ_min.
    pub fn condition_number(&self) -> f64 {
        let n = self.eigenvalues.len();
        if n == 0 {
            return 1.0;
        }
        self.eigenvalues[n - 1] / self.eigenvalues[0]
    }

    pub fn grid(&self) -> Grid2D {
        self.grid
    }

    pub fn alloc_field(&self) -> B::Buffer {
        self.backend.alloc_field(self.grid)
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }
}

impl<B: SpectralBackend> LinearOperator<B> for ToyDiagonalSPD<B> {
    /// Apply A: multiply each Fourier mode by its eigenvalue λ_k = 1 + k.
    fn apply(&mut self, input: &B::Buffer, output: &mut B::Buffer) {
        copy_buffer(&mut self.scratch, input);
        self.backend.forward_fft_2d(&mut self.scratch);

        // Multiply each Fourier mode by λ_k = 1 + k
        let data = self.scratch.as_mut_slice();
        for (k, value) in data.iter_mut().enumerate() {
            *value *= self.eigenvalues[k];
        }

        self.backend.inverse_fft_2d(&mut self.scratch);
        copy_buffer(output, &self.scratch);
    }

    /// Mass operator is identity: B = I.
    fn apply_mass(&mut self, input: &B::Buffer, output: &mut B::Buffer) {
        copy_buffer(output, input);
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
        [0.0, 0.0] // Not used for this toy problem
    }
}

fn build_k_vector(n: usize, length: f64) -> Vec<f64> {
    let two_pi = 2.0 * PI;
    (0..n)
        .map(|i| {
            let centered = if i <= n / 2 {
                i as isize
            } else {
                i as isize - n as isize
            };
            two_pi * centered as f64 / length
        })
        .collect()
}

fn build_shifted_k_plus_g_tables(
    base: &GridWaveTables,
    bloch: [f64; 2],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64, f64, usize, Vec<bool>) {
    let len = base.grid.len();
    let gx_base = base.gx();
    let gy_base = base.gy();
    #[cfg(debug_assertions)]
    {
        let mut unique_gx: Vec<f64> = gx_base.iter().cloned().collect();
        unique_gx.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique_gx.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
        eprintln!(
            "[k-debug] unique G_x values (first 8): {:?}",
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

fn te_preconditioner_mass_floor(eps_eff: f64) -> f64 {
    if !eps_eff.is_finite() || eps_eff <= 0.0 {
        return 0.0;
    }
    eps_eff * TE_PRECONDITIONER_MASS_FRACTION
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

    // For very small k_sq (near-DC modes), we should NOT amplify them.
    // The DC mode is in the null space of Laplacian operators.
    // Cap the maximum amplification to prevent preconditioner from
    // creating components orthogonal to the search direction.
    //
    // We use a floor that ensures inverse_scale <= 1/shift (reasonable cap).
    // For k_sq << shift, use k_sq + shift as the denominator instead of
    // clamping k_sq to a tiny floor.
    let safe_k_sq = k_sq.max(0.0); // Just ensure non-negative
    let shift_scaled = shift * eps_eff.max(1e-12);
    let denominator = safe_k_sq + safe_mass + shift_scaled;

    // The denominator is at least shift_scaled (since safe_k_sq >= 0, safe_mass >= 0)
    // This gives a maximum value of eps_eff / shift_scaled = 1 / shift
    eps_eff / denominator
}

fn copy_buffer<T: SpectralBuffer>(dst: &mut T, src: &T) {
    dst.as_mut_slice().copy_from_slice(src.as_slice());
}

fn apply_gradient_factors(
    grad_x: &mut [Complex64],
    grad_y: &mut [Complex64],
    k_plus_g_x: &[f64],
    k_plus_g_y: &[f64],
) {
    for (((gx, gy), &kx), &ky) in grad_x
        .iter_mut()
        .zip(grad_y.iter_mut())
        .zip(k_plus_g_x.iter())
        .zip(k_plus_g_y.iter())
    {
        let factor_x = Complex64::new(0.0, kx);
        let factor_y = Complex64::new(0.0, ky);
        *gx *= factor_x;
        *gy *= factor_y;
    }
}

fn apply_inv_eps(grad_x: &mut [Complex64], grad_y: &mut [Complex64], dielectric: &Dielectric2D) {
    if let Some(tensors) = dielectric.inv_eps_tensors() {
        for ((gx, gy), tensor) in grad_x.iter_mut().zip(grad_y.iter_mut()).zip(tensors.iter()) {
            let orig_x = *gx;
            let orig_y = *gy;
            let out_x = orig_x * tensor[0] + orig_y * tensor[1];
            let out_y = orig_x * tensor[2] + orig_y * tensor[3];
            *gx = out_x;
            *gy = out_y;
        }
    } else {
        for ((gx, gy), &inv) in grad_x
            .iter_mut()
            .zip(grad_y.iter_mut())
            .zip(dielectric.inv_eps().iter())
        {
            *gx *= inv;
            *gy *= inv;
        }
    }
}

fn apply_scalar_eps(field: &mut [Complex64], eps: &[f64]) {
    for (value, &eps_val) in field.iter_mut().zip(eps.iter()) {
        *value *= eps_val;
    }
}

fn assemble_divergence(
    grad_x: &[Complex64],
    grad_y: &[Complex64],
    output: &mut [Complex64],
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
        let factor_x = Complex64::new(0.0, kx);
        let factor_y = Complex64::new(0.0, ky);
        let div = factor_x * gx + factor_y * gy;
        *out = -div;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        K_PLUS_G_NEAR_ZERO_FLOOR, clamp_gradient_components, inverse_scale,
        te_preconditioner_mass_floor,
    };

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

        // For tiny k_sq, the denominator is dominated by shift_scaled = shift * eps_eff
        // With eps_eff = 1.0 and shift = 1e-3, denominator ≈ 1e-3
        // So inverse_scale ≈ eps_eff / 1e-3 = 1000 for the DC mode
        let tiny = K_PLUS_G_NEAR_ZERO_FLOOR / 10.0;
        let shift = 1e-3;
        let eps_eff = 1.0;
        let shift_scaled = shift * eps_eff;
        // denominator = tiny + shift_scaled ≈ shift_scaled for tiny << shift_scaled
        let expected = eps_eff / (tiny + shift_scaled);
        let actual = inverse_scale(tiny, shift, eps_eff, 0.0);
        assert!((actual - expected).abs() < 1e-12);
    }

    #[test]
    fn te_mass_floor_enters_denominator() {
        let eps_eff = 12.0;
        let mass_floor = te_preconditioner_mass_floor(eps_eff);
        let tiny = K_PLUS_G_NEAR_ZERO_FLOOR / 10.0;
        let baseline = inverse_scale(tiny, 1e-3, eps_eff, 0.0);
        let mass_adjusted = inverse_scale(tiny, 1e-3, eps_eff, mass_floor);
        assert!(mass_floor > 0.0);
        assert!(mass_adjusted < baseline);
    }
}

fn shift_k_vector(base: &[f64], shift: f64) -> Vec<f64> {
    base.iter().map(|&k| k + shift).collect()
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

/// Build the inverse diagonal scaling for the Fourier-space preconditioner.
///
/// # Arguments
/// * `values` - The |k+G|² values for each Fourier mode
/// * `shift` - Regularization shift σ² added to denominator
/// * `eps_eff` - Effective permittivity for scaling
/// * `mass_floor` - Additional mass term for TE mode stability
/// * `near_zero_mask` - Optional mask indicating which modes have |k+G|² ≈ 0
///
/// # Near-Zero Mode Handling
///
/// Modes with |k+G|² ≈ 0 are in the null space of Laplacian-type operators.
/// These must be zeroed in the preconditioner to avoid amplifying components
/// that should be handled by deflation. This only occurs at/near the Γ-point
/// (k ≈ 0) where the G=0 mode has |k+G|² = |k|² ≈ 0.
///
/// For k ≠ 0, the G=0 mode has |k+G|² = |k|² > 0 and should NOT be zeroed.
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

    // Zero out modes that are in/near the null space (|k+G|² ≈ 0).
    // This is determined by the clamp mask, which correctly identifies
    // near-zero modes at ANY k-point (not just Γ).
    //
    // At Γ (k=0): the G=0 mode has |k+G|² = 0 → zeroed (correct)
    // At k≠0: the G=0 mode has |k+G|² = |k|² > 0 → NOT zeroed (correct)
    if let Some(mask) = near_zero_mask {
        for (scale, &is_near_zero) in result.iter_mut().zip(mask.iter()) {
            if is_near_zero {
                *scale = 0.0;
            }
        }
    }

    result
}

#[derive(Debug)]
struct GridWaveTables {
    grid: Grid2D,
    kx: Vec<f64>,
    ky: Vec<f64>,
    gx: Vec<f64>,
    gy: Vec<f64>,
    g_sq: Vec<f64>,
}

impl GridWaveTables {
    fn new(grid: Grid2D) -> Self {
        let kx = build_k_vector(grid.nx, grid.lx);
        let ky = build_k_vector(grid.ny, grid.ly);
        let len = grid.len();
        let mut gx = vec![0.0; len];
        let mut gy = vec![0.0; len];
        for iy in 0..grid.ny {
            for ix in 0..grid.nx {
                let idx = grid.idx(ix, iy);
                gx[idx] = kx[ix];
                gy[idx] = ky[iy];
            }
        }
        let g_sq = gx
            .iter()
            .zip(gy.iter())
            .map(|(&gx_val, &gy_val)| gx_val * gx_val + gy_val * gy_val)
            .collect();
        Self {
            grid,
            kx,
            ky,
            gx,
            gy,
            g_sq,
        }
    }

    fn kx(&self) -> &[f64] {
        &self.kx
    }

    fn ky(&self) -> &[f64] {
        &self.ky
    }

    fn gx(&self) -> &[f64] {
        &self.gx
    }

    fn gy(&self) -> &[f64] {
        &self.gy
    }

    #[allow(dead_code)]
    fn g_sq(&self) -> &[f64] {
        &self.g_sq
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct GridKey {
    nx: usize,
    ny: usize,
    lx_bits: u64,
    ly_bits: u64,
}

impl From<Grid2D> for GridKey {
    fn from(grid: Grid2D) -> Self {
        Self {
            nx: grid.nx,
            ny: grid.ny,
            lx_bits: grid.lx.to_bits(),
            ly_bits: grid.ly.to_bits(),
        }
    }
}

static GRID_WAVE_CACHE: OnceLock<Mutex<HashMap<GridKey, Arc<GridWaveTables>>>> = OnceLock::new();

fn cached_wave_tables(grid: Grid2D) -> Arc<GridWaveTables> {
    let key = GridKey::from(grid);
    let cache = GRID_WAVE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap();
    if let Some(existing) = guard.get(&key) {
        return existing.clone();
    }
    let tables = Arc::new(GridWaveTables::new(grid));
    guard.insert(key, tables.clone());
    tables
}

fn build_structured_weights_te(dielectric: &Dielectric2D, eps_eff: f64) -> Vec<f64> {
    let eps = dielectric.eps();
    if eps.is_empty() || eps_eff <= 0.0 {
        return vec![1.0; dielectric.grid.len()];
    }
    eps.iter().map(|&val| clamp_weight(val / eps_eff)).collect()
}

fn build_structured_weights_tm(dielectric: &Dielectric2D) -> Vec<f64> {
    let inv_eps = dielectric.inv_eps();
    if inv_eps.is_empty() {
        return vec![1.0; dielectric.grid.len()];
    }
    let avg_inv = arithmetic_mean(inv_eps).unwrap_or(0.0);
    if avg_inv <= 0.0 {
        return vec![1.0; dielectric.grid.len()];
    }
    inv_eps
        .iter()
        .map(|&val| clamp_weight(val / avg_inv))
        .collect()
}

fn clamp_weight(value: f64) -> f64 {
    value.max(STRUCTURED_WEIGHT_MIN).min(STRUCTURED_WEIGHT_MAX)
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
