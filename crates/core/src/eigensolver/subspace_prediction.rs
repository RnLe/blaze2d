//! Subspace prediction for accelerated warm-start in band structure calculations.
//!
//! This module implements parallel transport of the eigenspace along k-paths
//! using eigenvector overlap analysis and subspace rotation. The technique is
//! mathematically equivalent to smooth gauge tracking used in:
//! - Wannier90 (maximally localized Wannier functions)
//! - Berry phase / Wilson-loop computations
//! - Smooth-gauge tracking for Bloch bands
//!
//! # Algorithm Overview
//!
//! Given eigenvector matrices at consecutive k-points X_{n-1} and X_n:
//!
//! 1. Compute the complex overlap matrix: O = X_{n-1}^† B X_n
//! 2. Extract the unitary rotation via polar decomposition: O = W Σ V^†, U = W V^†
//! 3. Align X_n to X_{n-1}'s gauge: X̃_n = X_n U^†
//! 4. (Optional) Extrapolate: X_{n+1}^pred = (1+α)X̃_n - α X_{n-1}
//! 5. B-orthonormalize the predicted vectors
//!
//! # Backend Selection
//!
//! - `native-linalg` feature: Uses faer for GEMM and SVD operations
//! - `wasm-linalg` feature: Uses nalgebra (WASM-compatible)
//!
//! # Usage
//!
//! ```ignore
//! let mut history = SubspaceHistory::new(n_bands);
//!
//! for (k_idx, k_point) in k_path.iter().enumerate() {
//!     // Get prediction for warm-start
//!     let prediction = history.predict(k_point, eps.as_deref());
//!     
//!     // Run eigensolver with predicted vectors
//!     let warm_start = prediction.vectors();
//!     let result = solver.solve_with_warm_start(warm_start);
//!     
//!     // Update history with converged eigenvectors
//!     history.update(k_point, &eigenvectors);
//! }
//! ```

#[cfg(feature = "native-linalg")]
use faer::Mat;

#[cfg(feature = "wasm-linalg")]
use nalgebra::{DMatrix, Complex as NaComplex};

use log::debug;
use num_complex::Complex64;

use crate::field::Field2D;

// ============================================================================
// Configuration Constants
// ============================================================================

/// Threshold for detecting degenerate/crossing regions.
/// If the smallest singular value of the overlap matrix falls below this,
/// extrapolation is disabled and we fall back to rotation-only.
const DEGENERACY_THRESHOLD: f64 = 0.1;

/// Threshold for detecting k-path corners (direction change).
/// If dot(dk_prev, dk_next) < 0, we're at a corner.
const CORNER_DOT_THRESHOLD: f64 = 0.0;

// ============================================================================
// Prediction Method Enum
// ============================================================================

/// The method used to generate the prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionMethod {
    /// No prediction available (first k-point or insufficient history).
    None,
    /// Simple copy of previous eigenvectors (second k-point).
    Copy,
    /// Rotation-only: X_pred = X_n U^† (Stage 1).
    Rotation,
    /// Linear extrapolation with rotation (Stage 2).
    Extrapolation,
}

// ============================================================================
// Prediction Result
// ============================================================================

/// Result of subspace prediction.
#[derive(Debug, Clone)]
pub struct PredictionResult {
    /// The predicted eigenvectors (B-orthonormalized).
    pub predicted_vectors: Vec<Field2D>,
    /// The method used for prediction.
    pub method_used: PredictionMethod,
    /// Smallest singular value of the overlap matrix (quality indicator).
    /// Lower values indicate potential degeneracies or band crossings.
    pub singular_value_min: f64,
}

impl PredictionResult {
    /// Create an empty prediction (no vectors available).
    pub fn none() -> Self {
        Self {
            predicted_vectors: Vec::new(),
            method_used: PredictionMethod::None,
            singular_value_min: 0.0,
        }
    }

    /// Get the predicted vectors as a slice (for warm-start).
    pub fn vectors(&self) -> Option<&[Field2D]> {
        if self.predicted_vectors.is_empty() {
            None
        } else {
            Some(&self.predicted_vectors)
        }
    }

    /// Check if a valid prediction is available.
    pub fn has_prediction(&self) -> bool {
        !self.predicted_vectors.is_empty() && self.method_used != PredictionMethod::None
    }
}

// ============================================================================
// Rotation Matrix Type (backend-agnostic)
// ============================================================================

/// Backend-agnostic rotation matrix storage.
/// This is stored as a flat column-major Vec<Complex64> to avoid generic types.
#[derive(Debug, Clone)]
pub struct RotationMatrix {
    data: Vec<Complex64>,
    dim: usize,
}

impl RotationMatrix {
    /// Create a new zero-filled rotation matrix of given dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            data: vec![Complex64::ZERO; dim * dim],
            dim,
        }
    }

    /// Get element at (i, j).
    pub fn get(&self, i: usize, j: usize) -> Complex64 {
        self.data[i + j * self.dim]
    }

    /// Set element at (i, j).
    pub fn set(&mut self, i: usize, j: usize, val: Complex64) {
        self.data[i + j * self.dim] = val;
    }

    /// Get the dimension of the matrix.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the number of columns (same as dim for square matrix).
    pub fn ncols(&self) -> usize {
        self.dim
    }
}

// ============================================================================
// Subspace History
// ============================================================================

/// Stores eigenvector history for subspace prediction.
///
/// Maintains the last two sets of eigenvectors and their k-points to enable
/// rotation tracking and linear extrapolation.
#[derive(Debug, Clone)]
pub struct SubspaceHistory {
    /// Number of bands being tracked.
    n_bands: usize,
    /// Eigenvectors from k_{n-1} (older).
    prev_eigenvectors: Option<Vec<Field2D>>,
    /// Eigenvectors from k_n (current).
    curr_eigenvectors: Option<Vec<Field2D>>,
    /// k-point at n-1.
    prev_k: Option<[f64; 2]>,
    /// k-point at n.
    curr_k: Option<[f64; 2]>,
    /// Cached rotation matrix U from the last (n-1) -> n transition.
    cached_rotation: Option<RotationMatrix>,
    /// Cached smallest singular value from the last overlap computation.
    cached_sigma_min: f64,
    /// Whether extrapolation is enabled (Stage 2).
    extrapolation_enabled: bool,
}

impl SubspaceHistory {
    /// Create a new subspace history tracker.
    ///
    /// # Arguments
    /// * `n_bands` - Number of bands to track
    pub fn new(n_bands: usize) -> Self {
        Self {
            n_bands,
            prev_eigenvectors: None,
            curr_eigenvectors: None,
            prev_k: None,
            curr_k: None,
            cached_rotation: None,
            cached_sigma_min: 1.0,
            extrapolation_enabled: false, // Start with Stage 1 (rotation-only)
        }
    }

    /// Enable linear extrapolation (Stage 2).
    pub fn enable_extrapolation(&mut self) {
        self.extrapolation_enabled = true;
    }

    /// Disable linear extrapolation (back to Stage 1).
    pub fn disable_extrapolation(&mut self) {
        self.extrapolation_enabled = false;
    }

    /// Check if extrapolation is currently enabled.
    pub fn is_extrapolation_enabled(&self) -> bool {
        self.extrapolation_enabled
    }

    /// Get the number of k-points in history (0, 1, or 2).
    pub fn history_depth(&self) -> usize {
        match (&self.prev_eigenvectors, &self.curr_eigenvectors) {
            (Some(_), Some(_)) => 2,
            (None, Some(_)) => 1,
            _ => 0,
        }
    }

    /// Clear all history (e.g., when starting a new k-path).
    pub fn clear(&mut self) {
        self.prev_eigenvectors = None;
        self.curr_eigenvectors = None;
        self.prev_k = None;
        self.curr_k = None;
        self.cached_rotation = None;
        self.cached_sigma_min = 1.0;
    }

    /// Update the history with new eigenvectors after a successful solve.
    ///
    /// This shifts the history window: curr -> prev, new -> curr.
    ///
    /// # Arguments
    /// * `k_point` - The k-point (fractional coordinates) for these eigenvectors
    /// * `eigenvectors` - The converged eigenvectors (should be B-orthonormal)
    pub fn update(&mut self, k_point: [f64; 2], eigenvectors: &[Field2D]) {
        // Shift history window
        self.prev_eigenvectors = self.curr_eigenvectors.take();
        self.prev_k = self.curr_k.take();

        // Store new eigenvectors (clone them)
        let n = eigenvectors.len().min(self.n_bands);
        self.curr_eigenvectors = Some(eigenvectors.iter().take(n).cloned().collect());
        self.curr_k = Some(k_point);

        // Clear cached rotation (will be recomputed on next predict)
        self.cached_rotation = None;
    }

    /// Predict eigenvectors for the next k-point.
    ///
    /// # Arguments
    /// * `k_next` - The next k-point (fractional coordinates)
    /// * `eps` - Optional dielectric function for B-weighting (TM mode)
    ///
    /// # Returns
    /// A `PredictionResult` containing the predicted vectors and metadata.
    pub fn predict(&mut self, k_next: [f64; 2], eps: Option<&[f64]>) -> PredictionResult {
        match self.history_depth() {
            0 => {
                // No history: return empty prediction (will use random init)
                debug!("[subspace_pred] No history available, using random init");
                PredictionResult::none()
            }
            1 => {
                // Only one previous k-point: simple copy
                let curr = self.curr_eigenvectors.as_ref().unwrap();
                debug!(
                    "[subspace_pred] Single history point, copying {} vectors",
                    curr.len()
                );
                PredictionResult {
                    predicted_vectors: curr.clone(),
                    method_used: PredictionMethod::Copy,
                    singular_value_min: 1.0,
                }
            }
            2 => {
                // Two history points: can do rotation (and optionally extrapolation)
                self.predict_with_rotation(k_next, eps)
            }
            _ => unreachable!(),
        }
    }

    /// Internal: predict using rotation (and optionally extrapolation).
    fn predict_with_rotation(&mut self, k_next: [f64; 2], eps: Option<&[f64]>) -> PredictionResult {
        let prev = self.prev_eigenvectors.as_ref().unwrap();
        let curr = self.curr_eigenvectors.as_ref().unwrap();
        let k_prev = self.prev_k.unwrap();
        let k_curr = self.curr_k.unwrap();

        // Compute complex overlap matrix O = X_{n-1}^† B X_n
        let overlap = compute_complex_overlap_matrix(prev, curr, eps);

        // Extract rotation via polar decomposition
        let (rotation, sigma_min) = polar_decomposition(&overlap);
        self.cached_rotation = Some(rotation.clone());
        self.cached_sigma_min = sigma_min;

        // Check for degenerate region
        let is_degenerate = sigma_min < DEGENERACY_THRESHOLD;
        if is_degenerate {
            debug!(
                "[subspace_pred] Degenerate region detected (σ_min={:.4}), using rotation-only",
                sigma_min
            );
        }

        // Check for corner (direction change)
        let dk_prev = [k_curr[0] - k_prev[0], k_curr[1] - k_prev[1]];
        let dk_next = [k_next[0] - k_curr[0], k_next[1] - k_curr[1]];
        let dot_product = dk_prev[0] * dk_next[0] + dk_prev[1] * dk_next[1];
        let is_corner = dot_product < CORNER_DOT_THRESHOLD;
        if is_corner {
            debug!(
                "[subspace_pred] Corner detected (dot={:.4}), using rotation-only",
                dot_product
            );
        }

        // Decide prediction strategy
        let use_extrapolation = self.extrapolation_enabled && !is_degenerate && !is_corner;

        let (predicted, method) = if use_extrapolation {
            // Stage 2: Linear extrapolation
            // α = |dk_next| / |dk_prev|
            let dk_prev_norm = (dk_prev[0] * dk_prev[0] + dk_prev[1] * dk_prev[1]).sqrt();
            let dk_next_norm = (dk_next[0] * dk_next[0] + dk_next[1] * dk_next[1]).sqrt();
            let alpha = if dk_prev_norm > 1e-12 {
                dk_next_norm / dk_prev_norm
            } else {
                1.0
            };

            debug!(
                "[subspace_pred] Extrapolating with α={:.3}, σ_min={:.4}",
                alpha, sigma_min
            );

            let predicted = extrapolate_with_rotation(prev, curr, &rotation, alpha);
            (predicted, PredictionMethod::Extrapolation)
        } else {
            // Stage 1: Rotation-only
            debug!(
                "[subspace_pred] Rotation-only prediction, σ_min={:.4}",
                sigma_min
            );

            let predicted = apply_rotation(curr, &rotation);
            (predicted, PredictionMethod::Rotation)
        };

        // B-orthonormalize the predicted vectors
        let orthonormalized = orthonormalize_fields(&predicted, eps);

        PredictionResult {
            predicted_vectors: orthonormalized,
            method_used: method,
            singular_value_min: sigma_min,
        }
    }
}

// ============================================================================
// Complex Overlap Matrix Computation
// ============================================================================

/// Overlap matrix storage (backend-agnostic).
#[derive(Debug, Clone)]
pub struct OverlapMatrix {
    data: Vec<Complex64>,
    dim: usize,
}

impl OverlapMatrix {
    /// Create a new zero-filled overlap matrix of given dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            data: vec![Complex64::ZERO; dim * dim],
            dim,
        }
    }

    /// Get element at (i, j).
    pub fn get(&self, i: usize, j: usize) -> Complex64 {
        self.data[i + j * self.dim]
    }

    /// Set element at (i, j).
    pub fn set(&mut self, i: usize, j: usize, val: Complex64) {
        self.data[i + j * self.dim] = val;
    }

    /// Get the dimension of the matrix.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Compute the complex B-weighted overlap matrix using GEMM.
///
/// O = X_prev^† B X_curr where O[i,j] = ⟨prev[i] | curr[j]⟩_B
///
/// This uses matrix multiplication: O = (B^{1/2} X_prev)^† (B^{1/2} X_curr)
/// For diagonal B (TM mode with ε), we apply B^{1/2} element-wise.
/// For identity B (TE mode), this is simply X_prev^† X_curr.
///
/// # Arguments
/// * `prev` - Eigenvectors from k_{n-1}
/// * `curr` - Eigenvectors from k_n
/// * `eps` - Optional dielectric for B-weighting (TM mode). None = identity (TE mode).
///
/// # Returns
/// An m×m overlap matrix where m = min(prev.len(), curr.len())
pub fn compute_complex_overlap_matrix(
    prev: &[Field2D],
    curr: &[Field2D],
    eps: Option<&[f64]>,
) -> OverlapMatrix {
    let m = prev.len().min(curr.len());
    if m == 0 {
        return OverlapMatrix::new(0);
    }

    let n = prev[0].len(); // Grid points

    #[cfg(feature = "native-linalg")]
    {
        compute_overlap_faer(prev, curr, eps, m, n)
    }

    #[cfg(feature = "wasm-linalg")]
    {
        compute_overlap_nalgebra(prev, curr, eps, m, n)
    }
}

#[cfg(feature = "native-linalg")]
fn compute_overlap_faer(
    prev: &[Field2D],
    curr: &[Field2D],
    eps: Option<&[f64]>,
    m: usize,
    n: usize,
) -> OverlapMatrix {
    let overlap_mat = if let Some(epsilon) = eps {
        // TM mode: apply √ε weighting
        let sqrt_eps: Vec<f64> = epsilon.iter().map(|&e| e.sqrt()).collect();

        let prev_weighted = Mat::<faer::c64>::from_fn(n, m, |row, col| {
            let c = prev[col].as_slice()[row];
            let w = sqrt_eps[row];
            faer::c64::new(c.re * w, c.im * w)
        });

        let curr_weighted = Mat::<faer::c64>::from_fn(n, m, |row, col| {
            let c = curr[col].as_slice()[row];
            let w = sqrt_eps[row];
            faer::c64::new(c.re * w, c.im * w)
        });

        prev_weighted.adjoint() * &curr_weighted
    } else {
        // TE mode: identity B
        let prev_mat = Mat::<faer::c64>::from_fn(n, m, |row, col| {
            let c = prev[col].as_slice()[row];
            faer::c64::new(c.re, c.im)
        });

        let curr_mat = Mat::<faer::c64>::from_fn(n, m, |row, col| {
            let c = curr[col].as_slice()[row];
            faer::c64::new(c.re, c.im)
        });

        prev_mat.adjoint() * &curr_mat
    };

    // Convert to OverlapMatrix
    let mut result = OverlapMatrix::new(m);
    for i in 0..m {
        for j in 0..m {
            let c = overlap_mat.get(i, j);
            result.set(i, j, Complex64::new(c.re, c.im));
        }
    }
    result
}

#[cfg(feature = "wasm-linalg")]
fn compute_overlap_nalgebra(
    prev: &[Field2D],
    curr: &[Field2D],
    eps: Option<&[f64]>,
    m: usize,
    n: usize,
) -> OverlapMatrix {
    let overlap_mat = if let Some(epsilon) = eps {
        // TM mode: apply √ε weighting
        let sqrt_eps: Vec<f64> = epsilon.iter().map(|&e| e.sqrt()).collect();

        let prev_weighted = DMatrix::<NaComplex<f64>>::from_fn(n, m, |row, col| {
            let c = prev[col].as_slice()[row];
            let w = sqrt_eps[row];
            NaComplex::new(c.re * w, c.im * w)
        });

        let curr_weighted = DMatrix::<NaComplex<f64>>::from_fn(n, m, |row, col| {
            let c = curr[col].as_slice()[row];
            let w = sqrt_eps[row];
            NaComplex::new(c.re * w, c.im * w)
        });

        prev_weighted.adjoint() * &curr_weighted
    } else {
        // TE mode: identity B
        let prev_mat = DMatrix::<NaComplex<f64>>::from_fn(n, m, |row, col| {
            let c = prev[col].as_slice()[row];
            NaComplex::new(c.re, c.im)
        });

        let curr_mat = DMatrix::<NaComplex<f64>>::from_fn(n, m, |row, col| {
            let c = curr[col].as_slice()[row];
            NaComplex::new(c.re, c.im)
        });

        prev_mat.adjoint() * &curr_mat
    };

    // Convert to OverlapMatrix
    let mut result = OverlapMatrix::new(m);
    for i in 0..m {
        for j in 0..m {
            let c = overlap_mat[(i, j)];
            result.set(i, j, Complex64::new(c.re, c.im));
        }
    }
    result
}

/// Standard inner product: ⟨x, y⟩ = Σ x^* · y
fn inner_product(x: &[Complex64], y: &[Complex64]) -> Complex64 {
    x.iter().zip(y.iter()).map(|(xi, yi)| xi.conj() * yi).sum()
}

/// B-weighted inner product: ⟨x, y⟩_B = Σ x^* · ε · y
fn b_inner_product(x: &[Complex64], y: &[Complex64], eps: &[f64]) -> Complex64 {
    x.iter()
        .zip(y.iter())
        .zip(eps.iter())
        .map(|((xi, yi), &e)| xi.conj() * yi * e)
        .sum()
}

// ============================================================================
// Polar Decomposition
// ============================================================================

/// Extract the unitary rotation from an overlap matrix via polar decomposition.
///
/// Given O, compute O = W Σ V^† (SVD), then U = W V^†.
///
/// # Arguments
/// * `overlap` - The m×m complex overlap matrix
///
/// # Returns
/// A tuple (U, σ_min) where:
/// * U is the m×m unitary rotation matrix
/// * σ_min is the smallest singular value (quality indicator)
pub fn polar_decomposition(overlap: &OverlapMatrix) -> (RotationMatrix, f64) {
    let m = overlap.dim();
    if m == 0 {
        return (RotationMatrix::new(0), 0.0);
    }

    #[cfg(feature = "native-linalg")]
    {
        polar_decomposition_faer(overlap, m)
    }

    #[cfg(feature = "wasm-linalg")]
    {
        polar_decomposition_nalgebra(overlap, m)
    }
}

#[cfg(feature = "native-linalg")]
fn polar_decomposition_faer(overlap: &OverlapMatrix, m: usize) -> (RotationMatrix, f64) {
    // Convert to faer matrix
    let overlap_faer = Mat::<faer::c64>::from_fn(m, m, |i, j| {
        let c = overlap.get(i, j);
        faer::c64::new(c.re, c.im)
    });

    // Compute SVD: O = W Σ V^†
    let svd = overlap_faer
        .svd()
        .expect("SVD should succeed for overlap matrix");
    let u_mat = svd.U();
    let v_mat = svd.V();
    let s_diag = svd.S().column_vector();

    // Find smallest singular value
    let sigma_min = (0..m)
        .map(|i| s_diag.get(i).re)
        .fold(f64::INFINITY, f64::min);

    // Compute U = W V^†
    let rotation_faer = u_mat * v_mat.adjoint();

    // Convert to RotationMatrix
    let mut rotation = RotationMatrix::new(m);
    for i in 0..m {
        for j in 0..m {
            let c = rotation_faer.get(i, j);
            rotation.set(i, j, Complex64::new(c.re, c.im));
        }
    }

    (rotation, sigma_min)
}

#[cfg(feature = "wasm-linalg")]
fn polar_decomposition_nalgebra(overlap: &OverlapMatrix, m: usize) -> (RotationMatrix, f64) {
    // Convert to nalgebra matrix
    let overlap_na = DMatrix::<NaComplex<f64>>::from_fn(m, m, |i, j| {
        let c = overlap.get(i, j);
        NaComplex::new(c.re, c.im)
    });

    // Compute SVD: O = U Σ V^†
    let svd = overlap_na.svd(true, true);
    let u_mat = svd.u.as_ref().expect("SVD should return U");
    let v_mat = svd.v_t.as_ref().expect("SVD should return V^T");

    // Find smallest singular value
    let sigma_min = svd.singular_values.iter().cloned().fold(f64::INFINITY, f64::min);

    // Compute rotation = U V^T (note: nalgebra stores V^T not V)
    let rotation_na = u_mat * v_mat;

    // Convert to RotationMatrix
    let mut rotation = RotationMatrix::new(m);
    for i in 0..m {
        for j in 0..m {
            let c = rotation_na[(i, j)];
            rotation.set(i, j, Complex64::new(c.re, c.im));
        }
    }

    (rotation, sigma_min)
}

/// Extended polar decomposition returning all singular values.
///
/// This variant is used for degenerate block detection, where we need to
/// identify clusters of bands with similar (small) singular values.
///
/// # Arguments
/// * `overlap` - The m×m complex overlap matrix
///
/// # Returns
/// A tuple (U, σ_all, σ_min) where:
/// * U is the m×m unitary rotation matrix
/// * σ_all is a vector of all singular values (sorted descending)
/// * σ_min is the smallest singular value (convenience)
pub fn polar_decomposition_with_singular_values(
    overlap: &OverlapMatrix,
) -> (RotationMatrix, Vec<f64>, f64) {
    let m = overlap.dim();
    if m == 0 {
        return (RotationMatrix::new(0), Vec::new(), 0.0);
    }

    #[cfg(feature = "native-linalg")]
    {
        polar_decomposition_with_sv_faer(overlap, m)
    }

    #[cfg(feature = "wasm-linalg")]
    {
        polar_decomposition_with_sv_nalgebra(overlap, m)
    }
}

#[cfg(feature = "native-linalg")]
fn polar_decomposition_with_sv_faer(overlap: &OverlapMatrix, m: usize) -> (RotationMatrix, Vec<f64>, f64) {
    let overlap_faer = Mat::<faer::c64>::from_fn(m, m, |i, j| {
        let c = overlap.get(i, j);
        faer::c64::new(c.re, c.im)
    });

    let svd = overlap_faer
        .svd()
        .expect("SVD should succeed for overlap matrix");
    let u_mat = svd.U();
    let v_mat = svd.V();
    let s_diag = svd.S().column_vector();

    // Extract all singular values (sorted descending)
    let mut sigma_all: Vec<f64> = (0..m).map(|i| s_diag.get(i).re).collect();
    sigma_all.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let sigma_min = sigma_all.last().copied().unwrap_or(0.0);

    let rotation_faer = u_mat * v_mat.adjoint();

    let mut rotation = RotationMatrix::new(m);
    for i in 0..m {
        for j in 0..m {
            let c = rotation_faer.get(i, j);
            rotation.set(i, j, Complex64::new(c.re, c.im));
        }
    }

    (rotation, sigma_all, sigma_min)
}

#[cfg(feature = "wasm-linalg")]
fn polar_decomposition_with_sv_nalgebra(overlap: &OverlapMatrix, m: usize) -> (RotationMatrix, Vec<f64>, f64) {
    let overlap_na = DMatrix::<NaComplex<f64>>::from_fn(m, m, |i, j| {
        let c = overlap.get(i, j);
        NaComplex::new(c.re, c.im)
    });

    let svd = overlap_na.svd(true, true);
    let u_mat = svd.u.as_ref().expect("SVD should return U");
    let v_mat = svd.v_t.as_ref().expect("SVD should return V^T");

    // Extract singular values (sorted descending)
    let mut sigma_all: Vec<f64> = svd.singular_values.iter().cloned().collect();
    sigma_all.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let sigma_min = sigma_all.last().copied().unwrap_or(0.0);

    let rotation_na = u_mat * v_mat;

    let mut rotation = RotationMatrix::new(m);
    for i in 0..m {
        for j in 0..m {
            let c = rotation_na[(i, j)];
            rotation.set(i, j, Complex64::new(c.re, c.im));
        }
    }

    (rotation, sigma_all, sigma_min)
}

// ============================================================================
// Rotation Application
// ============================================================================

/// Apply the rotation to align eigenvectors: X̃ = X U^†
///
/// Uses GEMM for efficient matrix multiplication.
///
/// # Arguments
/// * `vectors` - The eigenvectors to rotate (X_n)
/// * `rotation` - The m×m unitary rotation matrix U
///
/// # Returns
/// The rotated eigenvectors X̃ = X U^†
pub fn apply_rotation(vectors: &[Field2D], rotation: &RotationMatrix) -> Vec<Field2D> {
    let m = vectors.len();
    if m == 0 {
        return Vec::new();
    }

    let grid = vectors[0].grid();
    let n = vectors[0].len(); // Grid points

    #[cfg(feature = "native-linalg")]
    {
        apply_rotation_faer(vectors, rotation, grid, m, n)
    }

    #[cfg(feature = "wasm-linalg")]
    {
        apply_rotation_nalgebra(vectors, rotation, grid, m, n)
    }
}

#[cfg(feature = "native-linalg")]
fn apply_rotation_faer(
    vectors: &[Field2D],
    rotation: &RotationMatrix,
    grid: crate::grid::Grid2D,
    m: usize,
    n: usize,
) -> Vec<Field2D> {
    // Build X matrix (n × m) from vectors in faer format
    let x_mat = Mat::<faer::c64>::from_fn(n, m, |row, col| {
        let c = vectors[col].as_slice()[row];
        faer::c64::new(c.re, c.im)
    });

    // Build rotation^† in faer format
    let rotation_adj = Mat::<faer::c64>::from_fn(m, m, |i, j| {
        // adjoint: swap indices and conjugate
        let c = rotation.get(j, i);
        faer::c64::new(c.re, -c.im)
    });

    // Compute X̃ = X * U^† using faer GEMM
    let x_tilde = &x_mat * &rotation_adj;

    // Convert back to Vec<Field2D>
    let mut result: Vec<Field2D> = Vec::with_capacity(m);
    for j in 0..m {
        let output_data: Vec<Complex64> = (0..n)
            .map(|k| {
                let c = x_tilde.get(k, j);
                Complex64::new(c.re, c.im)
            })
            .collect();
        result.push(Field2D::from_vec(grid, output_data));
    }

    result
}

#[cfg(feature = "wasm-linalg")]
fn apply_rotation_nalgebra(
    vectors: &[Field2D],
    rotation: &RotationMatrix,
    grid: crate::grid::Grid2D,
    m: usize,
    n: usize,
) -> Vec<Field2D> {
    // Build X matrix (n × m) from vectors
    let x_mat = DMatrix::<NaComplex<f64>>::from_fn(n, m, |row, col| {
        let c = vectors[col].as_slice()[row];
        NaComplex::new(c.re, c.im)
    });

    // Build rotation^† (adjoint = transpose + conjugate)
    let rotation_adj = DMatrix::<NaComplex<f64>>::from_fn(m, m, |i, j| {
        let c = rotation.get(j, i);
        NaComplex::new(c.re, -c.im)
    });

    // Compute X̃ = X * U^†
    let x_tilde = &x_mat * &rotation_adj;

    // Convert back to Vec<Field2D>
    let mut result: Vec<Field2D> = Vec::with_capacity(m);
    for j in 0..m {
        let output_data: Vec<Complex64> = (0..n)
            .map(|k| {
                let c = x_tilde[(k, j)];
                Complex64::new(c.re, c.im)
            })
            .collect();
        result.push(Field2D::from_vec(grid, output_data));
    }

    result
}

// ============================================================================
// Linear Extrapolation with Rotation
// ============================================================================

/// Linear extrapolation with rotation alignment using fused GEMM.
///
/// Computes X_pred = (1+α)(X_curr U^†) - α X_prev in a single efficient pass:
/// 1. Form transformation T = (1+α) U^† (m×m matrix)
/// 2. X_pred = X_curr * T - α * X_prev
///
/// This fuses the rotation and extrapolation to avoid intermediate allocations.
///
/// # Arguments
/// * `prev` - Eigenvectors at k_{n-1}
/// * `curr` - Eigenvectors at k_n
/// * `rotation` - The unitary rotation U from (n-1)->n overlap
/// * `alpha` - Step ratio |dk_next| / |dk_prev|
///
/// # Returns
/// The extrapolated eigenvectors (not yet orthonormalized)
pub fn extrapolate_with_rotation(
    prev: &[Field2D],
    curr: &[Field2D],
    rotation: &RotationMatrix,
    alpha: f64,
) -> Vec<Field2D> {
    let m = prev.len().min(curr.len());
    if m == 0 {
        return Vec::new();
    }

    let grid = prev[0].grid();
    let n = prev[0].len();

    // Coefficients
    let coeff_aligned = 1.0 + alpha;
    let coeff_prev = -alpha;

    #[cfg(feature = "native-linalg")]
    {
        extrapolate_faer(prev, curr, rotation, coeff_aligned, coeff_prev, grid, m, n)
    }

    #[cfg(feature = "wasm-linalg")]
    {
        extrapolate_nalgebra(prev, curr, rotation, coeff_aligned, coeff_prev, grid, m, n)
    }
}

#[cfg(feature = "native-linalg")]
fn extrapolate_faer(
    prev: &[Field2D],
    curr: &[Field2D],
    rotation: &RotationMatrix,
    coeff_aligned: f64,
    coeff_prev: f64,
    grid: crate::grid::Grid2D,
    m: usize,
    n: usize,
) -> Vec<Field2D> {
    // Build X_curr matrix (n × m)
    let x_curr = Mat::<faer::c64>::from_fn(n, m, |row, col| {
        let c = curr[col].as_slice()[row];
        faer::c64::new(c.re, c.im)
    });

    // Compute scaled transformation: T = (1+α) U^†
    let rotation_adj = Mat::<faer::c64>::from_fn(m, m, |i, j| {
        let c = rotation.get(j, i);
        faer::c64::new(c.re * coeff_aligned, -c.im * coeff_aligned)
    });

    // X_aligned = X_curr * T = (1+α) X_curr U^† using GEMM
    let x_aligned = &x_curr * &rotation_adj;

    // Build result: X_pred = X_aligned + coeff_prev * X_prev
    let mut result: Vec<Field2D> = Vec::with_capacity(m);
    let coeff_p = Complex64::new(coeff_prev, 0.0);

    for j in 0..m {
        let prev_slice = prev[j].as_slice();
        let output_data: Vec<Complex64> = (0..n)
            .map(|k| {
                let aligned = x_aligned.get(k, j);
                Complex64::new(aligned.re, aligned.im) + coeff_p * prev_slice[k]
            })
            .collect();
        result.push(Field2D::from_vec(grid, output_data));
    }

    result
}

#[cfg(feature = "wasm-linalg")]
fn extrapolate_nalgebra(
    prev: &[Field2D],
    curr: &[Field2D],
    rotation: &RotationMatrix,
    coeff_aligned: f64,
    coeff_prev: f64,
    grid: crate::grid::Grid2D,
    m: usize,
    n: usize,
) -> Vec<Field2D> {
    // Build X_curr matrix (n × m)
    let x_curr = DMatrix::<NaComplex<f64>>::from_fn(n, m, |row, col| {
        let c = curr[col].as_slice()[row];
        NaComplex::new(c.re, c.im)
    });

    // Compute scaled transformation: T = (1+α) U^†
    let rotation_adj = DMatrix::<NaComplex<f64>>::from_fn(m, m, |i, j| {
        let c = rotation.get(j, i);
        NaComplex::new(c.re * coeff_aligned, -c.im * coeff_aligned)
    });

    // X_aligned = X_curr * T
    let x_aligned = &x_curr * &rotation_adj;

    // Build result: X_pred = X_aligned + coeff_prev * X_prev
    let mut result: Vec<Field2D> = Vec::with_capacity(m);
    let coeff_p = Complex64::new(coeff_prev, 0.0);

    for j in 0..m {
        let prev_slice = prev[j].as_slice();
        let output_data: Vec<Complex64> = (0..n)
            .map(|k| {
                let aligned = x_aligned[(k, j)];
                Complex64::new(aligned.re, aligned.im) + coeff_p * prev_slice[k]
            })
            .collect();
        result.push(Field2D::from_vec(grid, output_data));
    }

    result
}

// ============================================================================
// B-Orthonormalization
// ============================================================================

/// B-orthonormalize a set of field vectors using modified Gram-Schmidt.
///
/// This is a simple implementation for predicted vectors. For production use,
/// the eigensolver's SVQB is more robust, but this suffices for warm-start.
///
/// # Arguments
/// * `vectors` - The vectors to orthonormalize
/// * `eps` - Optional dielectric for B-weighting (TM mode)
///
/// # Returns
/// The B-orthonormalized vectors
pub fn orthonormalize_fields(vectors: &[Field2D], eps: Option<&[f64]>) -> Vec<Field2D> {
    let m = vectors.len();
    if m == 0 {
        return Vec::new();
    }

    let grid = vectors[0].grid();
    let n = vectors[0].len();

    // Clone input vectors (we'll modify them)
    let mut result: Vec<Vec<Complex64>> = vectors.iter().map(|v| v.as_slice().to_vec()).collect();

    // Modified Gram-Schmidt
    for i in 0..m {
        // Normalize vector i
        let norm = b_norm(&result[i], eps);
        if norm > 1e-14 {
            let scale = 1.0 / norm;
            for val in result[i].iter_mut() {
                *val *= scale;
            }
        }

        // Project out from remaining vectors
        for j in (i + 1)..m {
            // coeff = <v_i, v_j>_B
            let coeff = if let Some(epsilon) = eps {
                b_inner_product(&result[i], &result[j], epsilon)
            } else {
                inner_product(&result[i], &result[j])
            };

            // v_j = v_j - coeff * v_i
            // Clone v_i to avoid borrow conflict
            let vi: Vec<Complex64> = result[i].clone();
            for k in 0..n {
                result[j][k] -= coeff * vi[k];
            }
        }
    }

    // Convert back to Field2D
    result
        .into_iter()
        .map(|data| Field2D::from_vec(grid, data))
        .collect()
}

/// Compute B-norm: ||x||_B = sqrt(⟨x, x⟩_B)
fn b_norm(x: &[Complex64], eps: Option<&[f64]>) -> f64 {
    let norm_sq = if let Some(epsilon) = eps {
        x.iter()
            .zip(epsilon.iter())
            .map(|(xi, &e)| (xi.conj() * xi * e).re)
            .sum::<f64>()
    } else {
        x.iter().map(|xi| (xi.conj() * xi).re).sum::<f64>()
    };
    norm_sq.max(0.0).sqrt()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid2D;

    fn make_test_field(grid: Grid2D, values: Vec<Complex64>) -> Field2D {
        Field2D::from_vec(grid, values)
    }

    #[test]
    fn test_inner_product() {
        let x = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
        let y = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];

        let result = inner_product(&x, &y);
        // <x, y> = 1*1 + (-i)*i = 1 + 1 = 2
        assert!((result.re - 2.0).abs() < 1e-10);
        assert!(result.im.abs() < 1e-10);
    }

    #[test]
    fn test_b_inner_product() {
        let x = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
        let y = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
        let eps = vec![2.0, 3.0];

        let result = b_inner_product(&x, &y, &eps);
        // <x, y>_B = 1*2*1 + 1*3*1 = 5
        assert!((result.re - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_overlap_identity() {
        // If prev == curr (orthonormal), overlap should be ~identity
        let grid = Grid2D::new(2, 2, 1.0, 1.0);

        let v1 = make_test_field(
            grid,
            vec![
                Complex64::new(0.5, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(0.5, 0.0),
            ],
        );
        let v2 = make_test_field(
            grid,
            vec![
                Complex64::new(0.5, 0.0),
                Complex64::new(-0.5, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(-0.5, 0.0),
            ],
        );

        let prev = vec![v1.clone(), v2.clone()];
        let curr = vec![v1, v2];

        let overlap = compute_complex_overlap_matrix(&prev, &curr, None);

        // Should be close to identity
        assert!((overlap.get(0, 0).re - 1.0).abs() < 1e-10);
        assert!((overlap.get(1, 1).re - 1.0).abs() < 1e-10);
        assert!(overlap.get(0, 1).re.abs() < 1e-10);
        assert!(overlap.get(1, 0).re.abs() < 1e-10);
    }

    #[test]
    fn test_polar_decomposition_identity() {
        // For identity overlap, rotation should be identity
        let mut identity = OverlapMatrix::new(2);
        identity.set(0, 0, Complex64::new(1.0, 0.0));
        identity.set(1, 1, Complex64::new(1.0, 0.0));

        let (rotation, sigma_min) = polar_decomposition(&identity);

        // Rotation should be identity
        assert!((rotation.get(0, 0).re - 1.0).abs() < 1e-10);
        assert!((rotation.get(1, 1).re - 1.0).abs() < 1e-10);
        assert!(rotation.get(0, 1).re.abs() < 1e-10);
        assert!(rotation.get(1, 0).re.abs() < 1e-10);

        // All singular values are 1
        assert!((sigma_min - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_is_unitary() {
        // Create a non-trivial overlap matrix
        let mut overlap = OverlapMatrix::new(2);
        overlap.set(0, 0, Complex64::new(0.9, 0.1));
        overlap.set(0, 1, Complex64::new(0.1, 0.05));
        overlap.set(1, 0, Complex64::new(0.1, -0.05));
        overlap.set(1, 1, Complex64::new(0.95, 0.0));

        let (rotation, _) = polar_decomposition(&overlap);

        // Check U U^† = I by computing manually
        for i in 0..2 {
            for j in 0..2 {
                // (U U^†)[i,j] = sum_k U[i,k] * conj(U[j,k])
                let mut sum = Complex64::ZERO;
                for k in 0..2 {
                    sum += rotation.get(i, k) * rotation.get(j, k).conj();
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum.re - expected).abs() < 1e-10,
                    "U U^† [{},{}] = {:?}, expected {}",
                    i,
                    j,
                    sum,
                    expected
                );
                assert!(
                    sum.im.abs() < 1e-10,
                    "U U^† [{},{}] has imaginary part {:?}",
                    i,
                    j,
                    sum.im
                );
            }
        }
    }

    #[test]
    fn test_subspace_history_depth() {
        let mut history = SubspaceHistory::new(2);
        assert_eq!(history.history_depth(), 0);

        let grid = Grid2D::new(2, 2, 1.0, 1.0);
        let v = Field2D::zeros(grid);

        history.update([0.0, 0.0], &[v.clone(), v.clone()]);
        assert_eq!(history.history_depth(), 1);

        history.update([0.1, 0.0], &[v.clone(), v.clone()]);
        assert_eq!(history.history_depth(), 2);

        // Further updates keep depth at 2
        history.update([0.2, 0.0], &[v.clone(), v]);
        assert_eq!(history.history_depth(), 2);
    }

    #[test]
    fn test_predict_no_history() {
        let mut history = SubspaceHistory::new(2);
        let result = history.predict([0.0, 0.0], None);

        assert_eq!(result.method_used, PredictionMethod::None);
        assert!(!result.has_prediction());
    }

    #[test]
    fn test_predict_single_history() {
        let mut history = SubspaceHistory::new(2);
        let grid = Grid2D::new(2, 2, 1.0, 1.0);
        let v = Field2D::zeros(grid);

        history.update([0.0, 0.0], &[v.clone(), v]);
        let result = history.predict([0.1, 0.0], None);

        assert_eq!(result.method_used, PredictionMethod::Copy);
        assert!(result.has_prediction());
        assert_eq!(result.predicted_vectors.len(), 2);
    }

    #[test]
    fn test_extrapolation_enabled() {
        let mut history = SubspaceHistory::new(2);
        assert!(!history.is_extrapolation_enabled());

        history.enable_extrapolation();
        assert!(history.is_extrapolation_enabled());

        history.disable_extrapolation();
        assert!(!history.is_extrapolation_enabled());
    }

    #[test]
    fn test_predict_rotation_only_with_two_history() {
        // With extrapolation disabled, should use Rotation method
        let mut history = SubspaceHistory::new(2);
        let grid = Grid2D::new(2, 2, 1.0, 1.0);

        // Create simple orthonormal vectors
        let v1 = make_test_field(
            grid,
            vec![
                Complex64::new(0.5, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(0.5, 0.0),
            ],
        );
        let v2 = make_test_field(
            grid,
            vec![
                Complex64::new(0.5, 0.0),
                Complex64::new(-0.5, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(-0.5, 0.0),
            ],
        );

        history.update([0.0, 0.0], &[v1.clone(), v2.clone()]);
        history.update([0.1, 0.0], &[v1, v2]);

        // Extrapolation disabled by default
        let result = history.predict([0.2, 0.0], None);

        assert_eq!(result.method_used, PredictionMethod::Rotation);
        assert!(result.has_prediction());
        assert_eq!(result.predicted_vectors.len(), 2);
        assert!(result.singular_value_min > 0.9); // Should be ~1.0 for identity overlap
    }

    #[test]
    fn test_predict_extrapolation_with_two_history() {
        // With extrapolation enabled, should use Extrapolation method
        let mut history = SubspaceHistory::new(2);
        history.enable_extrapolation();

        let grid = Grid2D::new(2, 2, 1.0, 1.0);

        // Create simple orthonormal vectors
        let v1 = make_test_field(
            grid,
            vec![
                Complex64::new(0.5, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(0.5, 0.0),
            ],
        );
        let v2 = make_test_field(
            grid,
            vec![
                Complex64::new(0.5, 0.0),
                Complex64::new(-0.5, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(-0.5, 0.0),
            ],
        );

        history.update([0.0, 0.0], &[v1.clone(), v2.clone()]);
        history.update([0.1, 0.0], &[v1, v2]);

        // Extrapolation enabled
        let result = history.predict([0.2, 0.0], None);

        assert_eq!(result.method_used, PredictionMethod::Extrapolation);
        assert!(result.has_prediction());
        assert_eq!(result.predicted_vectors.len(), 2);
    }

    #[test]
    fn test_extrapolation_disabled_at_corner() {
        // At a corner (direction change), should fall back to Rotation
        let mut history = SubspaceHistory::new(2);
        history.enable_extrapolation();

        let grid = Grid2D::new(2, 2, 1.0, 1.0);
        let v1 = make_test_field(
            grid,
            vec![
                Complex64::new(0.5, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(0.5, 0.0),
            ],
        );
        let v2 = make_test_field(
            grid,
            vec![
                Complex64::new(0.5, 0.0),
                Complex64::new(-0.5, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(-0.5, 0.0),
            ],
        );

        // Path: (0,0) -> (0.5,0) -> (0.4,0)
        // Direction changes from +x to -x (dot product < 0)
        history.update([0.0, 0.0], &[v1.clone(), v2.clone()]);
        history.update([0.5, 0.0], &[v1, v2]);

        // Next point goes backward (corner with negative dot product)
        let result = history.predict([0.4, 0.0], None);

        // Should fall back to rotation-only at corner
        assert_eq!(result.method_used, PredictionMethod::Rotation);
    }

    #[test]
    fn test_extrapolate_with_rotation_formula() {
        // Test the extrapolation formula: X_pred = (1+α)X̃_n - α X_{n-1}
        let grid = Grid2D::new(2, 2, 1.0, 1.0);

        // Create vectors where prev and curr differ
        let prev = vec![make_test_field(
            grid,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )];
        let curr = vec![make_test_field(
            grid,
            vec![
                Complex64::new(2.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )];

        // Identity rotation (aligned already)
        let mut rotation = RotationMatrix::new(1);
        rotation.set(0, 0, Complex64::new(1.0, 0.0));

        // alpha = 1.0 (uniform steps)
        // X_pred = (1+1)*curr - 1*prev = 2*curr - prev = 2*[2,0,0,0] - [1,0,0,0] = [3,0,0,0]
        let result = extrapolate_with_rotation(&prev, &curr, &rotation, 1.0);

        assert_eq!(result.len(), 1);
        let predicted = result[0].as_slice();
        assert!((predicted[0].re - 3.0).abs() < 1e-10);
        assert!(predicted[0].im.abs() < 1e-10);
    }

    #[test]
    fn test_extrapolate_with_alpha_half() {
        // Test with α = 0.5 (half step)
        let grid = Grid2D::new(2, 2, 1.0, 1.0);

        let prev = vec![make_test_field(
            grid,
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )];
        let curr = vec![make_test_field(
            grid,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )];

        let mut rotation = RotationMatrix::new(1);
        rotation.set(0, 0, Complex64::new(1.0, 0.0));

        // alpha = 0.5
        // X_pred = (1+0.5)*curr - 0.5*prev = 1.5*[1,0,0,0] - 0.5*[0,0,0,0] = [1.5,0,0,0]
        let result = extrapolate_with_rotation(&prev, &curr, &rotation, 0.5);

        let predicted = result[0].as_slice();
        assert!((predicted[0].re - 1.5).abs() < 1e-10);
    }
}
