//! Deflation subspace for the LOBPCG eigensolver.
//!
//! This module implements deflation (also called "locking") for the LOBPCG algorithm.
//! Deflation removes converged eigenvectors from the active search space, which:
//!
//! 1. **Prevents wobbling**: Converged bands stay fixed and don't get perturbed
//! 2. **Speeds convergence**: The solver focuses on remaining unconverged bands
//! 3. **Reduces work**: Smaller active block means fewer operations per iteration
//!
//! # Mathematical Background
//!
//! Given a deflation subspace Y ∈ ℂ^{n×p} with Y^* B Y = I (B-orthonormal),
//! the B-orthogonal projector onto the complement of range(Y) is:
//!
//! ```text
//! P_Y = I - Y Y^* B
//! ```
//!
//! For any vector v, the projection removes the Y-component:
//! ```text
//! P_Y v = v - Y(Y^* B v)
//! ```
//!
//! This ensures that P_Y v is B-orthogonal to all columns of Y:
//! ```text
//! Y^* B (P_Y v) = Y^* B v - Y^* B Y (Y^* B v) = Y^* B v - I (Y^* B v) = 0
//! ```
//!
//! # Usage in LOBPCG
//!
//! The deflation subspace Y contains:
//! - **Locked eigenvectors**: Bands that have converged below tolerance
//! - **Γ constant mode**: The zero-eigenvalue mode at k=0 (handled separately)
//!
//! During each iteration:
//! 1. After computing residuals R, apply P_Y to remove deflated components
//! 2. After preconditioning P = M^{-1} R, apply P_Y again
//! 3. When building Z = [X, P, W], apply P_Y before SVQB
//!
//! This ensures the search subspace never contains components along locked directions.

use log::debug;
use num_complex::Complex64;

use super::normalization::{SvqbConfig, b_norm, svqb_orthonormalize};
use crate::backend::SpectralBackend;

// ============================================================================
// Deflation Subspace
// ============================================================================

/// A deflation subspace containing locked (converged) eigenvectors.
///
/// The deflation subspace Y stores B-orthonormal vectors that have been
/// "locked" and removed from the active LOBPCG iteration. All search
/// directions are projected to be B-orthogonal to Y.
///
/// # Type Parameters
/// - `B`: The spectral backend type
#[derive(Debug)]
pub struct DeflationSubspace<B: SpectralBackend> {
    /// The locked vectors Y (B-orthonormal).
    vectors: Vec<B::Buffer>,
    /// Precomputed B*Y for efficient projection.
    mass_vectors: Vec<B::Buffer>,
    /// Eigenvalues of the locked vectors.
    eigenvalues: Vec<f64>,
    /// Original band indices (for reporting).
    band_indices: Vec<usize>,
}

impl<B: SpectralBackend> DeflationSubspace<B> {
    /// Create an empty deflation subspace.
    pub fn new() -> Self {
        Self {
            vectors: Vec::new(),
            mass_vectors: Vec::new(),
            eigenvalues: Vec::new(),
            band_indices: Vec::new(),
        }
    }

    /// Get the number of locked vectors.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if the deflation subspace is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Get the locked eigenvalues.
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// Get the locked band indices.
    pub fn band_indices(&self) -> &[usize] {
        &self.band_indices
    }

    /// Get a reference to the locked vectors.
    pub fn vectors(&self) -> &[B::Buffer] {
        &self.vectors
    }

    /// Get a reference to the mass vectors (B*Y).
    pub fn mass_vectors(&self) -> &[B::Buffer] {
        &self.mass_vectors
    }

    /// Add a single vector to the deflation subspace.
    ///
    /// The vector must already be B-normalized. This function:
    /// 1. Projects out components along existing Y vectors
    /// 2. Re-normalizes if the remainder is non-negligible
    /// 3. Adds to Y if linearly independent
    ///
    /// # Arguments
    /// * `backend` - The spectral backend
    /// * `vector` - The vector to add (will be cloned)
    /// * `mass_vector` - B * vector (will be cloned)
    /// * `eigenvalue` - The eigenvalue of this vector
    /// * `band_index` - The original band index
    ///
    /// # Returns
    /// `true` if the vector was added, `false` if it was linearly dependent.
    pub fn add_vector(
        &mut self,
        backend: &B,
        vector: &B::Buffer,
        mass_vector: &B::Buffer,
        eigenvalue: f64,
        band_index: usize,
    ) -> bool {
        // Clone and project out existing Y components
        let mut v = vector.clone();
        let mut bv = mass_vector.clone();

        self.project_single(backend, &mut v, &mut bv);

        // Check if remainder is non-negligible
        let norm = b_norm(backend, &v, &bv);
        if norm < 1e-10 {
            // Vector is linearly dependent with existing Y
            return false;
        }

        // Normalize
        let scale = Complex64::new(1.0 / norm, 0.0);
        backend.scale(scale, &mut v);
        backend.scale(scale, &mut bv);

        // Add to deflation subspace
        self.vectors.push(v);
        self.mass_vectors.push(bv);
        self.eigenvalues.push(eigenvalue);
        self.band_indices.push(band_index);

        true
    }

    /// Add multiple vectors to the deflation subspace using SVQB.
    ///
    /// This is more numerically stable than adding vectors one-by-one
    /// when multiple bands converge simultaneously.
    ///
    /// # Arguments
    /// * `backend` - The spectral backend
    /// * `vectors` - The vectors to add
    /// * `mass_vectors` - B * vectors
    /// * `eigenvalues` - The eigenvalues
    /// * `band_indices` - The original band indices
    ///
    /// # Returns
    /// The number of vectors successfully added.
    pub fn add_vectors_batch<F>(
        &mut self,
        backend: &B,
        vectors: Vec<B::Buffer>,
        mass_vectors: Vec<B::Buffer>,
        eigenvalues: &[f64],
        band_indices: &[usize],
        mut apply_mass: F,
    ) -> usize
    where
        F: FnMut(&B::Buffer) -> B::Buffer,
    {
        if vectors.is_empty() {
            return 0;
        }

        // Combine existing Y with new vectors
        let n_existing = self.vectors.len();
        let n_new = vectors.len();
        let n_total = n_existing + n_new;

        let mut combined: Vec<B::Buffer> = Vec::with_capacity(n_total);
        let mut combined_mass: Vec<B::Buffer> = Vec::with_capacity(n_total);

        // Add existing vectors first
        for v in &self.vectors {
            combined.push(v.clone());
        }
        for bv in &self.mass_vectors {
            combined_mass.push(bv.clone());
        }

        // Add new vectors
        for v in vectors {
            combined.push(v);
        }
        for bv in mass_vectors {
            combined_mass.push(bv);
        }

        // Apply SVQB to the combined set
        let config = SvqbConfig::default();
        let result = svqb_orthonormalize(backend, &mut combined, &mut combined_mass, &config);

        // The first n_existing should remain (they were already orthonormal)
        // Any beyond that are new additions
        let n_kept = result.output_rank;
        let n_added = n_kept.saturating_sub(n_existing);

        // If SVQB dropped some existing vectors, we have a problem (shouldn't happen)
        if n_kept < n_existing {
            debug!(
                "[deflation] Warning: SVQB dropped {} existing locked vectors!",
                n_existing - n_kept
            );
        }

        // Truncate to kept vectors
        combined.truncate(n_kept);
        combined_mass.truncate(n_kept);

        // Recompute mass vectors for all (SVQB may have modified them)
        for i in 0..n_kept {
            combined_mass[i] = apply_mass(&combined[i]);
        }

        // Update stored vectors
        self.vectors = combined;
        self.mass_vectors = combined_mass;

        // Add eigenvalues and band indices for new vectors
        for i in 0..n_added.min(eigenvalues.len()) {
            self.eigenvalues.push(eigenvalues[i]);
            self.band_indices.push(band_indices[i]);
        }

        n_added
    }

    /// Project a single vector to be B-orthogonal to the deflation subspace.
    ///
    /// Computes: v ← P_Y v = v - Y(Y^* B v)
    ///
    /// # Arguments
    /// * `backend` - The spectral backend
    /// * `vector` - The vector to project (modified in-place)
    /// * `mass_vector` - B * vector (modified in-place)
    pub fn project_single(&self, backend: &B, vector: &mut B::Buffer, mass_vector: &mut B::Buffer) {
        if self.is_empty() {
            return;
        }

        // B-orthogonal projection: v' = v - Σ_i <y_i, v>_B * y_i
        // where <y, v>_B = y^H B v is the B-inner product.
        //
        // backend.dot(y, Bv) computes y^H (Bv) = <y, v>_B directly.
        // No conjugation needed - we use the coefficient as-is.
        //
        // Proof that this works:
        // <y, v'>_B = <y, v - <y,v>_B y>_B = <y,v>_B - <y,v>_B * <y,y>_B = <y,v>_B - <y,v>_B = 0
        // (assuming <y,y>_B = 1, i.e., y is B-normalized)
        for (y, by) in self.vectors.iter().zip(self.mass_vectors.iter()) {
            let c = backend.dot(y, mass_vector); // c = <y, v>_B = y^H (Bv)
            backend.axpy(-c, y, vector);
            backend.axpy(-c, by, mass_vector);
        }
    }

    /// Project a block of vectors to be B-orthogonal to the deflation subspace.
    ///
    /// This is more efficient than projecting one-by-one for large blocks.
    ///
    /// # Arguments
    /// * `backend` - The spectral backend
    /// * `vectors` - The vectors to project (modified in-place)
    /// * `mass_vectors` - B * vectors (modified in-place)
    pub fn project_block(
        &self,
        backend: &B,
        vectors: &mut [B::Buffer],
        mass_vectors: &mut [B::Buffer],
    ) {
        if self.is_empty() {
            return;
        }

        // For each vector in the block, project out all Y components
        for (v, bv) in vectors.iter_mut().zip(mass_vectors.iter_mut()) {
            self.project_single(backend, v, bv);
        }
    }

    /// Project a block of vectors (without precomputed mass vectors).
    ///
    /// This is a convenience method when you don't have B*V precomputed.
    /// It computes the coefficients using the stored B*Y vectors.
    ///
    /// # Arguments
    /// * `backend` - The spectral backend
    /// * `vectors` - The vectors to project (modified in-place)
    pub fn project_block_no_mass(&self, backend: &B, vectors: &mut [B::Buffer]) {
        if self.is_empty() {
            return;
        }

        // For each vector in the block, project out all Y components
        // We use: c_i = <y_i, v>_B = (B y_i)^* v (equivalent formulation)
        for v in vectors.iter_mut() {
            for (y, by) in self.vectors.iter().zip(self.mass_vectors.iter()) {
                // c = <y, v>_B = (By)^* v = conj(v^* By)
                let c = backend.dot(v, by).conj();
                backend.axpy(-c, y, v);
            }
        }
    }

    /// Clear all locked vectors.
    pub fn clear(&mut self) {
        self.vectors.clear();
        self.mass_vectors.clear();
        self.eigenvalues.clear();
        self.band_indices.clear();
    }
}

impl<B: SpectralBackend> Default for DeflationSubspace<B> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Locking Logic
// ============================================================================

/// Result of checking which bands should be locked.
#[derive(Debug, Clone)]
pub struct LockingResult {
    /// Indices of bands to lock (in the current active block).
    pub bands_to_lock: Vec<usize>,
    /// Indices of bands to keep active.
    pub bands_to_keep: Vec<usize>,
}

impl LockingResult {
    /// Check if any bands should be locked.
    pub fn has_locks(&self) -> bool {
        !self.bands_to_lock.is_empty()
    }
}

/// Determine which bands should be locked based on eigenvalue changes.
///
/// A band is locked when its relative eigenvalue change drops below the
/// convergence tolerance, indicating it has stabilized.
///
/// # Arguments
/// * `relative_eigenvalue_changes` - Relative eigenvalue change |Δλ|/|λ| for each active band
/// * `tol` - Convergence tolerance (same as EigensolverConfig.tol)
///
/// # Returns
/// A `LockingResult` indicating which bands to lock and which to keep.
pub fn check_for_locking(relative_eigenvalue_changes: &[f64], tol: f64) -> LockingResult {
    let mut bands_to_lock = Vec::new();
    let mut bands_to_keep = Vec::new();

    for (i, &change) in relative_eigenvalue_changes.iter().enumerate() {
        if change < tol {
            bands_to_lock.push(i);
        } else {
            bands_to_keep.push(i);
        }
    }

    LockingResult {
        bands_to_lock,
        bands_to_keep,
    }
}
