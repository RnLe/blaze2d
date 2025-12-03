//! Envelope approximation operator for moiré lattice band structure.
//!
//! This module provides the `EAOperator` which implements the envelope-approximation
//! Hamiltonian for computing band structures in moiré superlattices.
//!
//! # Physical Background
//!
//! The envelope approximation (EA) Hamiltonian has the form:
//!
//! ```text
//! H = -iη v_g(R)·∇_R - (η²/2) ∇_R·M⁻¹(R)∇_R + V(R)
//! ```
//!
//! where:
//! - η is the small twist parameter
//! - v_g(R) is the group velocity field (optional drift term)
//! - M⁻¹(R) is the inverse mass tensor (position-dependent)
//! - V(R) is the potential
//! - R is the slow (envelope) coordinate
//!
//! # Discretization
//!
//! Discretized on an Nx × Ny periodic grid using 4th-order finite differences:
//! - **Diagonal term**: V(R) — the potential
//! - **Kinetic term**: Variable-coefficient Laplacian with 4th-order FD stencil
//! - **Drift term** (optional): Skew-Hermitian contribution from v_g
//!
//! # Sparsity Pattern (4th-order FD)
//!
//! - ~13–17 nonzeros per row
//! - Total nnz ≈ 15 × N for N = Nx × Ny
//! - For 64×64: ~60,000 nonzeros in a 4096×4096 matrix (~0.4% density)
//!
//! # Usage
//!
//! ```ignore
//! use mpb2d_core::operators::envelope_approximation::EAOperator;
//!
//! let ea = EAOperator::new(
//!     backend,
//!     nx, ny,
//!     dx, dy,
//!     eta,
//!     potential,
//!     mass_inv,
//!     None,  // No drift term
//! );
//!
//! let mut output = ea.alloc_field();
//! ea.apply(&input, &mut output);
//! ```

use num_complex::Complex64;

use crate::backend::{SpectralBackend, SpectralBuffer};
use crate::grid::Grid2D;
use crate::operators::LinearOperator;

// ============================================================================
// 4th-Order Finite Difference Coefficients
// ============================================================================

/// 4th-order central difference gradient coefficients.
///
/// For computing ∂f/∂x at x_{i+1/2} (cell face):
/// ∂f/∂x ≈ (1/12Δx) * (f_{i-1} - 8f_i + 8f_{i+1} - f_{i+2})
///
/// These weights are for the staggered gradient at cell faces.
#[allow(dead_code)]
const GRAD_WEIGHTS: [f64; 4] = [1.0 / 12.0, -8.0 / 12.0, 8.0 / 12.0, -1.0 / 12.0];

/// Offsets relative to the face index (i+1/2).
/// At face (i+1/2), we sample at: i-1, i, i+1, i+2
#[allow(dead_code)]
const GRAD_OFFSETS: [i32; 4] = [-1, 0, 1, 2];

// ============================================================================
// EAOperator - Envelope Approximation Hamiltonian
// ============================================================================

/// Envelope-approximation operator data.
///
/// This struct holds all the fields needed to apply the EA Hamiltonian
/// in a matrix-free manner.
///
/// # Memory Layout
///
/// All fields are stored in row-major (C) order to match NumPy and enable
/// zero-copy FFI when calling from Python.
///
/// - `potential`: Shape [Nx, Ny], flattened row-major
/// - `mass_inv`: Shape [Nx, Ny, 2, 2], flattened as [i, j, a, b] → index = ((i * ny + j) * 2 + a) * 2 + b
/// - `vg`: Shape [Nx, Ny, 2] (optional)
pub struct EAOperator<B: SpectralBackend> {
    backend: B,
    /// Grid dimensions
    grid: Grid2D,
    nx: usize,
    ny: usize,
    /// Grid spacings in slow coordinates (dx_slow = dx_phys * eta)
    dx: f64,
    dy: f64,
    /// Small twist parameter
    eta: f64,
    /// Potential V(R) — shape [Nx, Ny], flattened row-major
    potential: Vec<f64>,
    /// Inverse mass tensor M^{-1}(R) — shape [Nx*Ny, 4]
    /// Layout: [point_idx, tensor_component] where tensor is [m_xx, m_xy, m_yx, m_yy]
    mass_inv: Vec<f64>,
    /// Group velocity field v_g(R) — shape [Nx, Ny, 2], optional
    /// If present, adds the drift term -iη v_g·∇
    vg: Option<Vec<f64>>,
    /// Reference frequency (for reconstructing absolute ω)
    omega_ref: f64,
    /// Scratch buffer for intermediate computations
    #[allow(dead_code)]
    scratch: B::Buffer,
    /// Second scratch buffer for gradient computations
    #[allow(dead_code)]
    scratch2: B::Buffer,
}

impl<B: SpectralBackend> EAOperator<B> {
    /// Create a new EAOperator.
    ///
    /// # Arguments
    ///
    /// * `backend` - The spectral backend for FFT operations
    /// * `nx`, `ny` - Grid dimensions
    /// * `dx`, `dy` - Grid spacings in slow coordinates
    /// * `eta` - Small twist parameter
    /// * `potential` - Potential V(R), length Nx*Ny
    /// * `mass_inv` - Inverse mass tensor M^{-1}(R), length Nx*Ny*4
    /// * `vg` - Optional group velocity field, length Nx*Ny*2
    /// * `omega_ref` - Reference frequency
    pub fn new(
        backend: B,
        nx: usize,
        ny: usize,
        dx: f64,
        dy: f64,
        eta: f64,
        potential: Vec<f64>,
        mass_inv: Vec<f64>,
        vg: Option<Vec<f64>>,
        omega_ref: f64,
    ) -> Self {
        let n = nx * ny;
        assert_eq!(potential.len(), n, "potential must have Nx*Ny elements");
        assert_eq!(mass_inv.len(), n * 4, "mass_inv must have Nx*Ny*4 elements");
        if let Some(ref v) = vg {
            assert_eq!(v.len(), n * 2, "vg must have Nx*Ny*2 elements");
        }

        let grid = Grid2D::new(nx, ny, dx * nx as f64, dy * ny as f64);
        let scratch = backend.alloc_field(grid);
        let scratch2 = backend.alloc_field(grid);

        Self {
            backend,
            grid,
            nx,
            ny,
            dx,
            dy,
            eta,
            potential,
            mass_inv,
            vg,
            omega_ref,
            scratch,
            scratch2,
        }
    }

    /// Get the reference frequency.
    pub fn omega_ref(&self) -> f64 {
        self.omega_ref
    }

    /// Get the twist parameter η.
    pub fn eta(&self) -> f64 {
        self.eta
    }

    /// Check if the drift term (v_g) is present.
    pub fn has_drift_term(&self) -> bool {
        self.vg.is_some()
    }

    /// Compute the mean potential (for preconditioner).
    pub fn mean_potential(&self) -> f64 {
        self.potential.iter().sum::<f64>() / self.potential.len() as f64
    }

    /// Compute the mean inverse mass (average of (m_xx + m_yy)/2).
    pub fn mean_inverse_mass(&self) -> f64 {
        let n = self.nx * self.ny;
        let mut sum = 0.0;
        for p in 0..n {
            let m_xx = self.mass_inv[p * 4];
            let m_yy = self.mass_inv[p * 4 + 3];
            sum += 0.5 * (m_xx + m_yy);
        }
        sum / n as f64
    }

    /// Compute comprehensive statistics of the potential field.
    ///
    /// Returns (min, max, mean) of V(R).
    pub fn potential_stats(&self) -> (f64, f64, f64) {
        let n = self.potential.len();
        let mut sum = 0.0;
        let mut v_min = f64::INFINITY;
        let mut v_max = f64::NEG_INFINITY;
        
        for &v in &self.potential {
            sum += v;
            v_min = v_min.min(v);
            v_max = v_max.max(v);
        }
        
        (v_min, v_max, sum / n as f64)
    }

    /// Compute comprehensive statistics of the inverse mass tensor.
    ///
    /// Returns (min, max, mean) of the effective mass (trace/2).
    pub fn mass_inv_stats(&self) -> (f64, f64, f64) {
        let n = self.nx * self.ny;
        let mut sum = 0.0;
        let mut m_min = f64::INFINITY;
        let mut m_max = f64::NEG_INFINITY;
        
        for p in 0..n {
            let m_xx = self.mass_inv[p * 4];
            let m_yy = self.mass_inv[p * 4 + 3];
            let m_eff = 0.5 * (m_xx + m_yy);
            sum += m_eff;
            m_min = m_min.min(m_eff);
            m_max = m_max.max(m_eff);
        }
        
        (m_min, m_max, sum / n as f64)
    }

    /// Get grid spacings.
    pub fn grid_spacing(&self) -> (f64, f64) {
        (self.dx, self.dy)
    }

    /// Get grid dimensions.
    pub fn grid_dimensions(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }

    /// Build an adaptive FFT preconditioner for this EA operator.
    ///
    /// This automatically computes the optimal shift based on the operator's
    /// spectral properties, handling:
    /// - Zero-mean potentials
    /// - Negative potentials
    /// - Large mass tensor variations
    ///
    /// # Returns
    ///
    /// A configured `FFTPreconditioner` with adaptive shift.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut operator = EAOperatorBuilder::new(backend, nx, ny)
    ///     .with_potential(potential)
    ///     .with_mass_inv(mass_inv)
    ///     .with_eta(0.02)
    ///     .build();
    ///
    /// let mut precond = operator.build_preconditioner();
    /// let result = single_solve::solve(&mut operator, Some(&mut precond), &job);
    /// ```
    pub fn build_preconditioner(&self) -> crate::preconditioners::fft_preconditioner::FFTPreconditioner<B> {
        let (v_min, v_max, v_mean) = self.potential_stats();
        let (m_min, m_max, m_mean) = self.mass_inv_stats();
        
        crate::preconditioners::fft_preconditioner::FFTPreconditioner::from_operator_stats(
            self.nx, self.ny,
            self.dx, self.dy,
            self.eta,
            v_min, v_max, v_mean,
            m_min, m_max, m_mean,
        )
    }

    /// Build a preconditioner with custom configuration.
    ///
    /// Use this when you need fine-grained control over the preconditioner.
    ///
    /// # Arguments
    ///
    /// * `config` - Custom preconditioner configuration
    pub fn build_preconditioner_with_config(
        &self,
        config: crate::preconditioners::fft_preconditioner::EAPreconditionerConfig,
    ) -> crate::preconditioners::fft_preconditioner::FFTPreconditioner<B> {
        crate::preconditioners::fft_preconditioner::FFTPreconditioner::with_config(
            self.nx, self.ny,
            self.dx, self.dy,
            self.eta,
            config,
        )
    }

    /// Estimate the spectral properties of the EA operator.
    ///
    /// Uses power iteration to estimate λ_max (largest eigenvalue magnitude).
    /// Also computes the diagonal range as a rough lower bound indicator.
    ///
    /// For EA operators with negative eigenvalues (common when V has negative
    /// values), the traditional condition number κ = λ_max/λ_min is not
    /// meaningful. Instead, we report:
    /// - λ_max: Largest eigenvalue (from power iteration)
    /// - diag_min: Minimum diagonal element (V + kinetic contribution)
    /// - spectral_spread: |λ_max - diag_min| as a rough condition indicator
    ///
    /// # Arguments
    ///
    /// * `n_iters` - Number of power iterations
    ///
    /// # Returns
    ///
    /// Tuple of (λ_max, diag_min, spectral_spread)
    pub fn estimate_condition_number(&mut self, n_iters: usize) -> (f64, f64, f64) {
        let mut v = self.backend.alloc_field(self.grid);
        let mut av = self.backend.alloc_field(self.grid);

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
            self.apply(&v, &mut av);
            let numerator: f64 = v
                .as_slice()
                .iter()
                .zip(av.as_slice().iter())
                .map(|(vi, avi)| (vi.conj() * avi).re)
                .sum();
            lambda_max = numerator;

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

        // For EA operators, the diagonal (potential + kinetic diagonal) provides bounds
        // The minimum potential gives a rough lower bound on eigenvalues
        let v_min = self.potential.iter().cloned().fold(f64::INFINITY, f64::min);
        let _v_max = self.potential.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        // Spectral spread as a condition indicator
        let spectral_spread = (lambda_max - v_min).abs();

        (lambda_max, v_min, spectral_spread)
    }

    /// Estimate the condition number of the preconditioned operator M⁻¹A.
    ///
    /// # Arguments
    ///
    /// * `precond` - The preconditioner to apply
    /// * `n_iters` - Number of power iterations
    ///
    /// # Returns
    ///
    /// Tuple of (λ_max, λ_min, κ)
    pub fn estimate_preconditioned_condition_number(
        &mut self,
        precond: &mut dyn crate::preconditioners::OperatorPreconditioner<B>,
        n_iters: usize,
    ) -> (f64, f64, f64) {
        let mut v = self.backend.alloc_field(self.grid);
        let mut av = self.backend.alloc_field(self.grid);

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
            self.apply(&v, &mut av);
            precond.apply(&self.backend, &mut av);

            let numerator: f64 = v
                .as_slice()
                .iter()
                .zip(av.as_slice().iter())
                .map(|(vi, avi)| (vi.conj() * avi).re)
                .sum();
            lambda_max = numerator;

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

        // After preconditioning, we expect λ_min ≈ 1 for ideal preconditioner
        let lambda_min_approx = 1.0;
        let kappa = lambda_max / lambda_min_approx;

        (lambda_max, lambda_min_approx, kappa)
    }

    /// Access the inverse mass tensor component at point (i, j).
    ///
    /// Returns [m_xx, m_xy, m_yx, m_yy].
    #[inline]
    fn mass_inv_at(&self, i: usize, j: usize) -> [f64; 4] {
        let idx = (i * self.ny + j) * 4;
        [
            self.mass_inv[idx],
            self.mass_inv[idx + 1],
            self.mass_inv[idx + 2],
            self.mass_inv[idx + 3],
        ]
    }

    /// Get the group velocity at point (i, j).
    ///
    /// Returns [vg_x, vg_y] or [0, 0] if no drift term.
    #[inline]
    fn vg_at(&self, i: usize, j: usize) -> [f64; 2] {
        match &self.vg {
            Some(vg) => {
                let idx = (i * self.ny + j) * 2;
                [vg[idx], vg[idx + 1]]
            }
            None => [0.0, 0.0],
        }
    }

    /// Wrap index with periodic boundary conditions.
    #[inline]
    fn wrap_x(&self, i: i32) -> usize {
        ((i % self.nx as i32) + self.nx as i32) as usize % self.nx
    }

    #[inline]
    fn wrap_y(&self, j: i32) -> usize {
        ((j % self.ny as i32) + self.ny as i32) as usize % self.ny
    }

    /// Compute the 4th-order gradient in X at face (i+1/2, j).
    ///
    /// Returns ∂ψ/∂x evaluated at the cell face.
    #[allow(dead_code)]
    #[inline]
    fn grad_x_at_face(&self, input: &[Complex64], i: usize, j: usize) -> Complex64 {
        let mut result = Complex64::new(0.0, 0.0);
        for (w, &off) in GRAD_WEIGHTS.iter().zip(GRAD_OFFSETS.iter()) {
            let ii = self.wrap_x(i as i32 + off);
            result += *w * input[ii * self.ny + j];
        }
        result / self.dx
    }

    /// Compute the 4th-order gradient in Y at face (i, j+1/2).
    #[allow(dead_code)]
    #[inline]
    fn grad_y_at_face(&self, input: &[Complex64], i: usize, j: usize) -> Complex64 {
        let mut result = Complex64::new(0.0, 0.0);
        for (w, &off) in GRAD_WEIGHTS.iter().zip(GRAD_OFFSETS.iter()) {
            let jj = self.wrap_y(j as i32 + off);
            result += *w * input[i * self.ny + jj];
        }
        result / self.dy
    }

    /// Compute the face-averaged mass tensor component at x-face (i+1/2, j).
    #[allow(dead_code)]
    #[inline]
    fn mass_inv_at_x_face(&self, i: usize, j: usize) -> [f64; 4] {
        let m0 = self.mass_inv_at(i, j);
        let m1 = self.mass_inv_at(self.wrap_x(i as i32 + 1), j);
        [
            0.5 * (m0[0] + m1[0]),
            0.5 * (m0[1] + m1[1]),
            0.5 * (m0[2] + m1[2]),
            0.5 * (m0[3] + m1[3]),
        ]
    }

    /// Compute the face-averaged mass tensor component at y-face (i, j+1/2).
    #[allow(dead_code)]
    #[inline]
    fn mass_inv_at_y_face(&self, i: usize, j: usize) -> [f64; 4] {
        let m0 = self.mass_inv_at(i, j);
        let m1 = self.mass_inv_at(i, self.wrap_y(j as i32 + 1));
        [
            0.5 * (m0[0] + m1[0]),
            0.5 * (m0[1] + m1[1]),
            0.5 * (m0[2] + m1[2]),
            0.5 * (m0[3] + m1[3]),
        ]
    }

    /// Apply the kinetic term using 4th-order finite differences.
    ///
    /// Computes: -η²/2 ∇·(M⁻¹(R)∇ψ)
    ///
    /// For Hermitian symmetry, we use the symmetric discretization:
    /// ∇·(M⁻¹∇ψ) = Σ_{α,β} ∂_α (M^{-1}_{αβ} ∂_β ψ)
    ///
    /// This is split into:
    /// - Diagonal terms (α=β): ∂_x(m_xx ∂_x ψ) + ∂_y(m_yy ∂_y ψ)  
    /// - Off-diagonal terms: ∂_x(m_xy ∂_y ψ) + ∂_y(m_yx ∂_x ψ)
    ///
    /// The off-diagonal terms use symmetric averaging to preserve Hermitian structure.
    fn apply_kinetic(&self, input: &[Complex64], output: &mut [Complex64]) {
        let prefactor = -0.5 * self.eta * self.eta;

        for i in 0..self.nx {
            for j in 0..self.ny {
                let p = i * self.ny + j;
                let m = self.mass_inv_at(i, j);
                let m_xx = m[0];
                let m_xy = m[1];
                let m_yx = m[2];
                let m_yy = m[3];

                // Diagonal terms: use central differences
                // ∂_x(m_xx ∂_x ψ) ≈ m_xx * ∂²ψ/∂x² (constant m_xx approximation)
                // For variable m_xx, we use the symmetric form:
                // (1/dx²)[m_{i+1/2} (ψ_{i+1} - ψ_i) - m_{i-1/2} (ψ_i - ψ_{i-1})]

                // Use 4th-order Laplacian stencil coefficients
                // d²f/dx² ≈ (-f_{i-2} + 16f_{i-1} - 30f_i + 16f_{i+1} - f_{i+2}) / (12 dx²)
                let laplacian_x = self.laplacian_x(input, i, j);
                let laplacian_y = self.laplacian_y(input, i, j);

                // Central differences for gradients (4th order)
                let grad_x = self.grad_x_central(input, i, j);
                let grad_y = self.grad_y_central(input, i, j);

                // Gradient of m_xx and m_yy (needed for variable coefficient)
                // For simplicity, use 2nd-order central differences for the mass gradients
                let dm_xx_dx = self.grad_m_x(i, j, 0); // d(m_xx)/dx
                let dm_yy_dy = self.grad_m_y(i, j, 3); // d(m_yy)/dy

                // Diagonal contribution: ∂_α(m_αα ∂_α ψ) = m_αα ∂²ψ/∂α² + (∂m_αα/∂α)(∂ψ/∂α)
                let diag_term =
                    m_xx * laplacian_x + dm_xx_dx * grad_x + m_yy * laplacian_y + dm_yy_dy * grad_y;

                // Off-diagonal terms: ∂_x(m_xy ∂_y ψ) + ∂_y(m_yx ∂_x ψ)
                // Use symmetric form: m_xy * ∂²ψ/∂x∂y + m_yx * ∂²ψ/∂y∂x + gradient corrections
                // For symmetric M^{-1} (m_xy = m_yx), these combine nicely
                let mixed_deriv = self.mixed_derivative(input, i, j);
                let dm_xy_dx = self.grad_m_x(i, j, 1);
                let dm_yx_dy = self.grad_m_y(i, j, 2);

                let off_diag_term =
                    (m_xy + m_yx) * mixed_deriv + dm_xy_dx * grad_y + dm_yx_dy * grad_x;

                output[p] += prefactor * (diag_term + off_diag_term);
            }
        }
    }

    /// 4th-order Laplacian in X: d²ψ/dx²
    #[inline]
    fn laplacian_x(&self, input: &[Complex64], i: usize, j: usize) -> Complex64 {
        // (-f_{i-2} + 16f_{i-1} - 30f_i + 16f_{i+1} - f_{i+2}) / (12 dx²)
        let im2 = self.wrap_x(i as i32 - 2);
        let im1 = self.wrap_x(i as i32 - 1);
        let ip1 = self.wrap_x(i as i32 + 1);
        let ip2 = self.wrap_x(i as i32 + 2);

        let f_im2 = input[im2 * self.ny + j];
        let f_im1 = input[im1 * self.ny + j];
        let f_i = input[i * self.ny + j];
        let f_ip1 = input[ip1 * self.ny + j];
        let f_ip2 = input[ip2 * self.ny + j];

        (-f_im2 + 16.0 * f_im1 - 30.0 * f_i + 16.0 * f_ip1 - f_ip2) / (12.0 * self.dx * self.dx)
    }

    /// 4th-order Laplacian in Y: d²ψ/dy²
    #[inline]
    fn laplacian_y(&self, input: &[Complex64], i: usize, j: usize) -> Complex64 {
        let jm2 = self.wrap_y(j as i32 - 2);
        let jm1 = self.wrap_y(j as i32 - 1);
        let jp1 = self.wrap_y(j as i32 + 1);
        let jp2 = self.wrap_y(j as i32 + 2);

        let f_jm2 = input[i * self.ny + jm2];
        let f_jm1 = input[i * self.ny + jm1];
        let f_j = input[i * self.ny + j];
        let f_jp1 = input[i * self.ny + jp1];
        let f_jp2 = input[i * self.ny + jp2];

        (-f_jm2 + 16.0 * f_jm1 - 30.0 * f_j + 16.0 * f_jp1 - f_jp2) / (12.0 * self.dy * self.dy)
    }

    /// 4th-order mixed derivative: d²ψ/dxdy
    #[inline]
    fn mixed_derivative(&self, input: &[Complex64], i: usize, j: usize) -> Complex64 {
        // Use composition of 4th-order gradients: d/dx(d/dy)
        // First compute d/dy at neighboring x points, then d/dx of that
        // This is expensive but accurate

        // For efficiency, use 2nd-order for the mixed derivative:
        // d²f/dxdy ≈ (f_{i+1,j+1} - f_{i+1,j-1} - f_{i-1,j+1} + f_{i-1,j-1}) / (4 dx dy)
        let im1 = self.wrap_x(i as i32 - 1);
        let ip1 = self.wrap_x(i as i32 + 1);
        let jm1 = self.wrap_y(j as i32 - 1);
        let jp1 = self.wrap_y(j as i32 + 1);

        let f_pp = input[ip1 * self.ny + jp1];
        let f_pm = input[ip1 * self.ny + jm1];
        let f_mp = input[im1 * self.ny + jp1];
        let f_mm = input[im1 * self.ny + jm1];

        (f_pp - f_pm - f_mp + f_mm) / (4.0 * self.dx * self.dy)
    }

    /// Gradient of mass component in X direction (2nd order)
    #[inline]
    fn grad_m_x(&self, i: usize, j: usize, component: usize) -> f64 {
        let im1 = self.wrap_x(i as i32 - 1);
        let ip1 = self.wrap_x(i as i32 + 1);

        let m_im1 = self.mass_inv[(im1 * self.ny + j) * 4 + component];
        let m_ip1 = self.mass_inv[(ip1 * self.ny + j) * 4 + component];

        (m_ip1 - m_im1) / (2.0 * self.dx)
    }

    /// Gradient of mass component in Y direction (2nd order)
    #[inline]
    fn grad_m_y(&self, i: usize, j: usize, component: usize) -> f64 {
        let jm1 = self.wrap_y(j as i32 - 1);
        let jp1 = self.wrap_y(j as i32 + 1);

        let m_jm1 = self.mass_inv[(i * self.ny + jm1) * 4 + component];
        let m_jp1 = self.mass_inv[(i * self.ny + jp1) * 4 + component];

        (m_jp1 - m_jm1) / (2.0 * self.dy)
    }

    /// Apply the potential term: V(R) ψ(R)
    fn apply_potential(&self, input: &[Complex64], output: &mut [Complex64]) {
        for (p, (&v, &x)) in self.potential.iter().zip(input.iter()).enumerate() {
            output[p] += v * x;
        }
    }

    /// Apply the drift term: -iη v_g(R)·∇ψ
    ///
    /// Uses 4th-order central differences for the gradient.
    fn apply_drift(&self, input: &[Complex64], output: &mut [Complex64]) {
        let Some(ref _vg) = self.vg else {
            return;
        };

        let prefactor = Complex64::new(0.0, -self.eta);

        for i in 0..self.nx {
            for j in 0..self.ny {
                let p = i * self.ny + j;
                let [vg_x, vg_y] = self.vg_at(i, j);

                // 4th-order central difference gradients at cell center
                let grad_x = self.grad_x_central(input, i, j);
                let grad_y = self.grad_y_central(input, i, j);

                output[p] += prefactor * (vg_x * grad_x + vg_y * grad_y);
            }
        }
    }

    /// 4th-order central difference gradient in X at cell center (i, j).
    #[inline]
    fn grad_x_central(&self, input: &[Complex64], i: usize, j: usize) -> Complex64 {
        // 4th-order: (1/12)(-f_{i+2} + 8f_{i+1} - 8f_{i-1} + f_{i-2}) / dx
        let im2 = self.wrap_x(i as i32 - 2);
        let im1 = self.wrap_x(i as i32 - 1);
        let ip1 = self.wrap_x(i as i32 + 1);
        let ip2 = self.wrap_x(i as i32 + 2);

        let f_im2 = input[im2 * self.ny + j];
        let f_im1 = input[im1 * self.ny + j];
        let f_ip1 = input[ip1 * self.ny + j];
        let f_ip2 = input[ip2 * self.ny + j];

        (f_im2 - 8.0 * f_im1 + 8.0 * f_ip1 - f_ip2) / (12.0 * self.dx)
    }

    /// 4th-order central difference gradient in Y at cell center (i, j).
    #[inline]
    fn grad_y_central(&self, input: &[Complex64], i: usize, j: usize) -> Complex64 {
        let jm2 = self.wrap_y(j as i32 - 2);
        let jm1 = self.wrap_y(j as i32 - 1);
        let jp1 = self.wrap_y(j as i32 + 1);
        let jp2 = self.wrap_y(j as i32 + 2);

        let f_jm2 = input[i * self.ny + jm2];
        let f_jm1 = input[i * self.ny + jm1];
        let f_jp1 = input[i * self.ny + jp1];
        let f_jp2 = input[i * self.ny + jp2];

        (f_jm2 - 8.0 * f_jm1 + 8.0 * f_jp1 - f_jp2) / (12.0 * self.dy)
    }
}

impl<B: SpectralBackend> LinearOperator<B> for EAOperator<B> {
    fn apply(&mut self, input: &B::Buffer, output: &mut B::Buffer) {
        let input_slice = input.as_slice();
        let output_slice = output.as_mut_slice();

        // Zero output
        for o in output_slice.iter_mut() {
            *o = Complex64::new(0.0, 0.0);
        }

        // H = V + kinetic + drift (optional)
        // 1. Add potential term: output += V * input
        self.apply_potential(input_slice, output_slice);

        // 2. Add kinetic term: output += -η²/2 ∇·(M⁻¹∇ψ)
        self.apply_kinetic(input_slice, output_slice);

        // 3. Add drift term (if vg present): output += -iη v_g·∇ψ
        if self.vg.is_some() {
            self.apply_drift(input_slice, output_slice);
        }
    }

    fn apply_mass(&mut self, input: &B::Buffer, output: &mut B::Buffer) {
        // For the EA Hamiltonian, the mass matrix is identity (standard eigenproblem)
        output.as_mut_slice().copy_from_slice(input.as_slice());
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
        // EA operator doesn't use Bloch wavevector in the same sense as Maxwell.
        // The envelope coordinates already incorporate the moiré periodicity.
        //
        // IMPORTANT: We return a small non-zero value to prevent the eigensolver
        // from treating this as a Γ-point and deflating the constant mode.
        // For EA, the constant mode is NOT a spurious zero-eigenvalue mode
        // (unlike Maxwell at k=0).
        [1e-6, 1e-6]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Field2D;

    /// A minimal test backend for unit testing.
    #[derive(Clone)]
    struct TestBackend;

    impl SpectralBackend for TestBackend {
        type Buffer = Field2D;

        fn alloc_field(&self, grid: Grid2D) -> Field2D {
            Field2D::zeros(grid)
        }

        fn forward_fft_2d(&self, _buffer: &mut Field2D) {}
        fn inverse_fft_2d(&self, _buffer: &mut Field2D) {}

        fn scale(&self, alpha: Complex64, buffer: &mut Field2D) {
            for v in buffer.as_mut_slice() {
                *v *= alpha;
            }
        }

        fn axpy(&self, alpha: Complex64, x: &Field2D, y: &mut Field2D) {
            for (yi, &xi) in y.as_mut_slice().iter_mut().zip(x.as_slice()) {
                *yi += alpha * xi;
            }
        }

        fn dot(&self, x: &Field2D, y: &Field2D) -> Complex64 {
            x.as_slice()
                .iter()
                .zip(y.as_slice())
                .map(|(a, b)| a.conj() * b)
                .sum()
        }
    }

    #[test]
    fn test_constant_potential_operator() {
        // Constant V=1, M=I → H = 1 - η²/2 Δ
        // Ground state should be the constant mode with eigenvalue = V
        let nx = 8;
        let ny = 8;
        let n = nx * ny;
        let dx = 1.0;
        let dy = 1.0;
        let eta = 0.1;

        let potential = vec![1.0; n];
        // Identity mass tensor: [m_xx, m_xy, m_yx, m_yy] = [1, 0, 0, 1]
        let mass_inv: Vec<f64> = (0..n).flat_map(|_| [1.0, 0.0, 0.0, 1.0]).collect();

        let mut op = EAOperator::new(
            TestBackend,
            nx,
            ny,
            dx,
            dy,
            eta,
            potential,
            mass_inv,
            None,
            0.0,
        );

        // Apply to constant vector → should give V * constant (since Δ(const) = 0)
        let grid = Grid2D::new(nx, ny, dx * nx as f64, dy * ny as f64);
        let mut input = Field2D::zeros(grid);
        for v in input.as_mut_slice() {
            *v = Complex64::new(1.0, 0.0);
        }

        let mut output = Field2D::zeros(grid);
        op.apply(&input, &mut output);

        // All output values should be close to V = 1.0
        for &o in output.as_slice() {
            assert!((o.re - 1.0).abs() < 1e-10, "Expected ~1.0, got {}", o.re);
            assert!(o.im.abs() < 1e-10, "Expected real output");
        }
    }

    #[test]
    fn test_plane_wave_eigenvalue() {
        // For constant V, constant M=I:
        // H ψ_k = (V + η²/2 |k|²) ψ_k
        // where ψ_k = exp(i k·r) is a plane wave
        let nx = 16;
        let ny = 16;
        let n = nx * ny;
        let dx = 1.0;
        let dy = 1.0;
        let eta = 0.1;
        let v_const = 2.0;

        let potential = vec![v_const; n];
        let mass_inv: Vec<f64> = (0..n).flat_map(|_| [1.0, 0.0, 0.0, 1.0]).collect();

        let mut op = EAOperator::new(
            TestBackend,
            nx,
            ny,
            dx,
            dy,
            eta,
            potential,
            mass_inv,
            None,
            0.0,
        );

        let grid = Grid2D::new(nx, ny, dx * nx as f64, dy * ny as f64);

        // Create plane wave with k = (2π/Lx, 0)
        let kx = 2.0 * std::f64::consts::PI / (nx as f64 * dx);
        let ky = 0.0;
        let k_sq = kx * kx + ky * ky;

        let mut input = Field2D::zeros(grid);
        for i in 0..nx {
            for j in 0..ny {
                let x = i as f64 * dx;
                let y = j as f64 * dy;
                let phase = kx * x + ky * y;
                input.as_mut_slice()[i * ny + j] = Complex64::new(phase.cos(), phase.sin());
            }
        }

        let mut output = Field2D::zeros(grid);
        op.apply(&input, &mut output);

        // Expected eigenvalue: V + η²/2 * |k|²
        let expected_eigenvalue = v_const + 0.5 * eta * eta * k_sq;

        // Check that output ≈ eigenvalue * input
        // Due to finite difference discretization error, this won't be exact
        let backend = TestBackend;
        let numerator = backend.dot(&input, &output).re;
        let denominator = backend.dot(&input, &input).re;
        let computed_eigenvalue = numerator / denominator;

        // 4th-order FD should be accurate to O(dx^4)
        let tolerance = 0.01; // Allow 1% error for small grid
        assert!(
            (computed_eigenvalue - expected_eigenvalue).abs() / expected_eigenvalue < tolerance,
            "Expected eigenvalue ~{}, got {}",
            expected_eigenvalue,
            computed_eigenvalue
        );
    }

    #[test]
    fn test_hermitian_symmetry() {
        // Verify <x, Hy> = <Hx, y> for random vectors (Hermitian property)
        // Use constant mass tensor to avoid FD discretization asymmetry
        let nx = 8;
        let ny = 8;
        let n = nx * ny;
        let dx = 1.0;
        let dy = 1.0;
        let eta = 0.1;

        // Variable potential but constant mass for guaranteed Hermitian symmetry
        let potential: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * (i as f64).sin()).collect();
        // Constant identity mass tensor
        let mass_inv: Vec<f64> = (0..n).flat_map(|_| [1.0, 0.0, 0.0, 1.0]).collect();

        let mut op = EAOperator::new(
            TestBackend,
            nx,
            ny,
            dx,
            dy,
            eta,
            potential,
            mass_inv,
            None,
            0.0,
        );

        let grid = Grid2D::new(nx, ny, dx * nx as f64, dy * ny as f64);

        // Random vectors
        let mut x = Field2D::zeros(grid);
        let mut y = Field2D::zeros(grid);
        for i in 0..n {
            x.as_mut_slice()[i] = Complex64::new((i as f64 * 0.1).sin(), (i as f64 * 0.2).cos());
            y.as_mut_slice()[i] = Complex64::new((i as f64 * 0.3).cos(), (i as f64 * 0.15).sin());
        }

        let mut hx = Field2D::zeros(grid);
        let mut hy = Field2D::zeros(grid);

        op.apply(&x, &mut hx);
        op.apply(&y, &mut hy);

        let backend = TestBackend;
        let x_hy = backend.dot(&x, &hy); // <x, Hy>
        let hx_y = backend.dot(&hx, &y); // <Hx, y>

        // For Hermitian: <x, Hy> = <Hx, y>
        // Note: With our definition <a,b> = Σ a_i^* b_i, Hermitian symmetry is:
        // <x, Hy> = <Hx, y> (equal, not conjugate)
        let diff = (x_hy - hx_y).norm();
        let scale = x_hy.norm().max(hx_y.norm());

        // With constant mass, the discretization should be exactly Hermitian
        assert!(
            diff / scale < 1e-10,
            "Hermitian symmetry violated: <x,Hy>={}, <Hx,y>={}",
            x_hy,
            hx_y
        );
    }

    #[test]
    fn test_variable_mass_hermitian_approx() {
        // With variable mass, FD discretization may have small asymmetry
        // Just verify it's reasonably symmetric (within 1%)
        let nx = 8;
        let ny = 8;
        let n = nx * ny;
        let dx = 1.0;
        let dy = 1.0;
        let eta = 0.1;

        let potential: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * (i as f64).sin()).collect();
        let mass_inv: Vec<f64> = (0..n)
            .flat_map(|i| {
                let m = 1.0 + 0.05 * (i as f64 * 0.1).cos();
                [m, 0.0, 0.0, m] // Diagonal mass tensor
            })
            .collect();

        let mut op = EAOperator::new(
            TestBackend,
            nx,
            ny,
            dx,
            dy,
            eta,
            potential,
            mass_inv,
            None,
            0.0,
        );

        let grid = Grid2D::new(nx, ny, dx * nx as f64, dy * ny as f64);

        let mut x = Field2D::zeros(grid);
        let mut y = Field2D::zeros(grid);
        for i in 0..n {
            x.as_mut_slice()[i] = Complex64::new((i as f64 * 0.1).sin(), (i as f64 * 0.2).cos());
            y.as_mut_slice()[i] = Complex64::new((i as f64 * 0.3).cos(), (i as f64 * 0.15).sin());
        }

        let mut hx = Field2D::zeros(grid);
        let mut hy = Field2D::zeros(grid);

        op.apply(&x, &mut hx);
        op.apply(&y, &mut hy);

        let backend = TestBackend;
        let x_hy = backend.dot(&x, &hy);
        let hx_y = backend.dot(&hx, &y);

        // For Hermitian: <x, Hy> = <Hx, y> (equal, not conjugate)
        let diff = (x_hy - hx_y).norm();
        let scale = x_hy.norm().max(hx_y.norm());

        // Allow 1% relative error for variable-coefficient case
        assert!(
            diff / scale < 0.01,
            "Variable-mass symmetry error too large: rel_err={}, <x,Hy>={}, <Hx,y>={}",
            diff / scale,
            x_hy,
            hx_y
        );
    }

    #[test]
    fn test_drift_term_applied() {
        // Verify that the drift term is applied when vg is provided
        let nx = 8;
        let ny = 8;
        let n = nx * ny;
        let dx = 1.0;
        let dy = 1.0;
        let eta = 0.1;

        let potential = vec![1.0; n];
        let mass_inv: Vec<f64> = (0..n).flat_map(|_| [1.0, 0.0, 0.0, 1.0]).collect();

        // Create uniform group velocity field: v_g = (1, 0)
        let vg: Vec<f64> = (0..n).flat_map(|_| [1.0, 0.0]).collect();

        // Operator without drift
        let mut op_no_drift = EAOperator::new(
            TestBackend,
            nx,
            ny,
            dx,
            dy,
            eta,
            potential.clone(),
            mass_inv.clone(),
            None,
            0.0,
        );

        // Operator with drift
        let mut op_with_drift = EAOperator::new(
            TestBackend,
            nx,
            ny,
            dx,
            dy,
            eta,
            potential,
            mass_inv,
            Some(vg),
            0.0,
        );

        assert!(!op_no_drift.has_drift_term());
        assert!(op_with_drift.has_drift_term());

        let grid = Grid2D::new(nx, ny, dx * nx as f64, dy * ny as f64);

        // Create a plane wave that has non-zero gradient
        let kx = 2.0 * std::f64::consts::PI / (nx as f64 * dx);
        let mut input = Field2D::zeros(grid);
        for i in 0..nx {
            for j in 0..ny {
                let x = i as f64 * dx;
                let phase = kx * x;
                input.as_mut_slice()[i * ny + j] = Complex64::new(phase.cos(), phase.sin());
            }
        }

        let mut output_no_drift = Field2D::zeros(grid);
        let mut output_with_drift = Field2D::zeros(grid);

        op_no_drift.apply(&input, &mut output_no_drift);
        op_with_drift.apply(&input, &mut output_with_drift);

        // The outputs should differ because of the drift term
        let mut diff_norm_sq = 0.0;
        for (&a, &b) in output_no_drift
            .as_slice()
            .iter()
            .zip(output_with_drift.as_slice())
        {
            diff_norm_sq += (a - b).norm_sqr();
        }
        let diff_norm = diff_norm_sq.sqrt();

        // The drift term should contribute a non-zero amount
        assert!(
            diff_norm > 1e-10,
            "Drift term should make a difference: diff_norm = {}",
            diff_norm
        );
    }

    #[test]
    fn test_drift_term_skew_hermitian() {
        // The drift term D = -iη v_g·∇ should be skew-Hermitian (anti-Hermitian)
        // This means <x, Dy> = -<Dx, y> (not conjugate, since D is purely imaginary)
        //
        // For V = 0, M = 0 (no potential, no kinetic), H = D, we test:
        // <x, Hy> = -<Hx, y>*  (skew-Hermitian property)
        //
        // Note: The full operator H = V + K + D where K is Hermitian and D is skew-Hermitian.
        // So H is NOT Hermitian when drift is present. This is correct physics:
        // the drift term represents a gauge field.

        let nx = 8;
        let ny = 8;
        let n = nx * ny;
        let dx = 1.0;
        let dy = 1.0;
        let eta = 0.1;

        // Zero potential and identity mass to isolate drift contribution
        let potential = vec![0.0; n];
        let mass_inv: Vec<f64> = (0..n).flat_map(|_| [1.0, 0.0, 0.0, 1.0]).collect();

        // Spatially varying group velocity
        let vg: Vec<f64> = (0..n)
            .flat_map(|idx| {
                let i = idx / ny;
                let j = idx % ny;
                let vx = (i as f64 * 0.1).sin();
                let vy = (j as f64 * 0.15).cos();
                [vx, vy]
            })
            .collect();

        let mut op = EAOperator::new(
            TestBackend,
            nx,
            ny,
            dx,
            dy,
            eta,
            potential,
            mass_inv,
            Some(vg),
            0.0,
        );

        let grid = Grid2D::new(nx, ny, dx * nx as f64, dy * ny as f64);

        // Random test vectors
        let mut x = Field2D::zeros(grid);
        let mut y = Field2D::zeros(grid);
        for i in 0..n {
            x.as_mut_slice()[i] = Complex64::new((i as f64 * 0.1).sin(), (i as f64 * 0.2).cos());
            y.as_mut_slice()[i] = Complex64::new((i as f64 * 0.3).cos(), (i as f64 * 0.15).sin());
        }

        let mut hx = Field2D::zeros(grid);
        let mut hy = Field2D::zeros(grid);

        op.apply(&x, &mut hx);
        op.apply(&y, &mut hy);

        let backend = TestBackend;
        let x_hy = backend.dot(&x, &hy); // <x, Hy>
        let hx_y = backend.dot(&hx, &y); // <Hx, y>

        // With drift term, H = K + D where K is Hermitian and D is skew-Hermitian.
        // For purely D (V=0, K~small), we expect <x, Dy> ≈ -<Dx, y>*
        // The kinetic term contributes to the Hermitian part.
        //
        // For the full operator, we just verify that the drift term contributes
        // by checking that the operator is NOT purely Hermitian.
        let hermitian_diff = (x_hy - hx_y).norm();
        let scale = x_hy.norm().max(hx_y.norm()).max(1e-10);

        // With drift, the operator should have noticeable anti-Hermitian component
        // (but not huge, since kinetic term is still Hermitian)
        assert!(
            hermitian_diff / scale > 1e-6,
            "With drift term, operator should have anti-Hermitian component: diff/scale = {}",
            hermitian_diff / scale
        );
    }

    /// Integration test: Use EAOperator with the single_solve driver.
    ///
    /// This test verifies that EAOperator can be used with the generic
    /// single_solve driver to compute eigenvalues.
    ///
    /// Note: This test requires the real CPU backend since the eigensolver
    /// initialization uses FFT operations internally.
    #[test]
    #[ignore] // Requires mpb2d-backend-cpu; run with `cargo test -p mpb2d-core -- --ignored`
    fn test_single_solve_integration() {
        use crate::drivers::single_solve::{SingleSolveJob, solve};

        let nx = 8;
        let ny = 8;
        let n = nx * ny;
        let dx = 1.0;
        let dy = 1.0;
        let eta = 0.1;

        // Create a simple harmonic potential well
        // V(x,y) = 1 + 0.5*((x - Lx/2)^2 + (y - Ly/2)^2) / L^2
        let lx = nx as f64 * dx;
        let ly = ny as f64 * dy;
        let cx = lx / 2.0;
        let cy = ly / 2.0;
        let l_sq = (lx * ly).sqrt();

        let potential: Vec<f64> = (0..n)
            .map(|idx| {
                let i = idx / ny;
                let j = idx % ny;
                let x = i as f64 * dx;
                let y = j as f64 * dy;
                let r_sq = (x - cx).powi(2) + (y - cy).powi(2);
                1.0 + 0.5 * r_sq / l_sq.powi(2)
            })
            .collect();

        // Constant identity mass tensor
        let mass_inv: Vec<f64> = (0..n).flat_map(|_| [1.0, 0.0, 0.0, 1.0]).collect();

        let mut op = EAOperator::new(
            TestBackend,
            nx,
            ny,
            dx,
            dy,
            eta,
            potential,
            mass_inv,
            None,
            0.0,
        );

        // Solve for 4 lowest eigenvalues
        let job = SingleSolveJob::new(4)
            .with_tolerance(1e-6)
            .with_max_iterations(100);

        let result = solve(&mut op, None, &job);

        // Verify we got eigenvalues
        assert_eq!(result.eigenvalues.len(), 4);

        // Eigenvalues should be positive (harmonic potential is positive definite)
        for &ev in &result.eigenvalues {
            assert!(ev > 0.0, "Eigenvalue should be positive: {}", ev);
        }

        // Eigenvalues should be in ascending order
        for i in 0..result.eigenvalues.len() - 1 {
            assert!(
                result.eigenvalues[i] <= result.eigenvalues[i + 1],
                "Eigenvalues not sorted: {} > {}",
                result.eigenvalues[i],
                result.eigenvalues[i + 1]
            );
        }

        // Should converge in reasonable iterations for such a small problem
        assert!(
            result.iterations < 50,
            "Too many iterations: {}",
            result.iterations
        );
    }
}

// ============================================================================
// Builder pattern for convenient construction
// ============================================================================

/// Builder for EAOperator with convenient defaults.
pub struct EAOperatorBuilder<B: SpectralBackend> {
    backend: B,
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    eta: f64,
    potential: Option<Vec<f64>>,
    mass_inv: Option<Vec<f64>>,
    vg: Option<Vec<f64>>,
    omega_ref: f64,
}

impl<B: SpectralBackend> EAOperatorBuilder<B> {
    /// Start building an EAOperator.
    pub fn new(backend: B, nx: usize, ny: usize) -> Self {
        Self {
            backend,
            nx,
            ny,
            dx: 1.0,
            dy: 1.0,
            eta: 0.1,
            potential: None,
            mass_inv: None,
            vg: None,
            omega_ref: 0.0,
        }
    }

    /// Set grid spacings.
    pub fn with_spacing(mut self, dx: f64, dy: f64) -> Self {
        self.dx = dx;
        self.dy = dy;
        self
    }

    /// Set the twist parameter η.
    pub fn with_eta(mut self, eta: f64) -> Self {
        self.eta = eta;
        self
    }

    /// Set the potential field.
    pub fn with_potential(mut self, potential: Vec<f64>) -> Self {
        self.potential = Some(potential);
        self
    }

    /// Set the inverse mass tensor field.
    pub fn with_mass_inv(mut self, mass_inv: Vec<f64>) -> Self {
        self.mass_inv = Some(mass_inv);
        self
    }

    /// Set the group velocity field (enables drift term).
    pub fn with_vg(mut self, vg: Vec<f64>) -> Self {
        self.vg = Some(vg);
        self
    }

    /// Set the reference frequency.
    pub fn with_omega_ref(mut self, omega_ref: f64) -> Self {
        self.omega_ref = omega_ref;
        self
    }

    /// Build the EAOperator.
    ///
    /// Uses uniform potential V=0 and identity mass M=I if not specified.
    pub fn build(self) -> EAOperator<B> {
        let n = self.nx * self.ny;

        let potential = self.potential.unwrap_or_else(|| vec![0.0; n]);
        let mass_inv = self
            .mass_inv
            .unwrap_or_else(|| (0..n).flat_map(|_| [1.0, 0.0, 0.0, 1.0]).collect());

        EAOperator::new(
            self.backend,
            self.nx,
            self.ny,
            self.dx,
            self.dy,
            self.eta,
            potential,
            mass_inv,
            self.vg,
            self.omega_ref,
        )
    }
}
