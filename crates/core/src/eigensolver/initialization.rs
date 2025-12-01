//! Block initialization for the LOBPCG eigensolver.
//!
//! This module handles the creation of the initial block of vectors X_0
//! for the LOBPCG iteration. The vectors are:
//! 1. Optionally seeded from warm-start vectors (from previous k-points)
//! 2. Filled with random vectors if needed
//! 3. Orthonormalized with respect to the mass matrix B
//!
//! # Γ-Point Constant Mode
//!
//! At k=0 (the Γ point), there is always a spurious zero-eigenvalue mode
//! corresponding to the constant field u₀ = 1 everywhere. This mode satisfies
//! (∇ + ik) × u₀ = 0 when k=0, giving λ = 0.
//!
//! To prevent convergence issues, we pre-compute this mode and add it to
//! the deflation subspace during initialization. The B-normalized constant
//! mode is:
//!
//! ```text
//! y₀ = u₀ / √(u₀^* B u₀)
//! ```
//!
//! where u₀ has value 1 for all Fourier components.

use log::{debug, warn};
use num_complex::Complex64;

use crate::backend::{SpectralBackend, SpectralBuffer};
use crate::field::Field2D;
use crate::operators::LinearOperator;

use super::normalization::{normalize_to_unit_b_norm_with_tol, project_out};

// ============================================================================
// Block Entry
// ============================================================================

/// A single block entry containing a vector, its mass-weighted version, and operator application.
///
/// This represents one column of the block matrices in LOBPCG:
/// - `vector`: The eigenvector approximation x_i
/// - `mass`: B * x_i (precomputed for efficiency in B-inner products)
/// - `applied`: A * x_i (precomputed for Rayleigh quotient and residual computation)
#[derive(Clone)]
pub struct BlockEntry<B: SpectralBackend> {
    pub vector: B::Buffer,
    pub mass: B::Buffer,
    pub applied: B::Buffer,
}

impl<B: SpectralBackend> BlockEntry<B> {
    /// Compute the Rayleigh quotient λ = <x, Ax> / <x, Bx> for this entry.
    pub fn rayleigh_quotient(&self, backend: &B) -> f64 {
        let x_ax = backend.dot(&self.vector, &self.applied).re;
        let x_bx = backend.dot(&self.vector, &self.mass).re;
        if x_bx.abs() > 1e-15 { x_ax / x_bx } else { 0.0 }
    }

    /// Compute the B-norm ||x||_B = sqrt(<x, Bx>).
    pub fn b_norm(&self, backend: &B) -> f64 {
        backend.dot(&self.vector, &self.mass).re.max(0.0).sqrt()
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for block initialization.
#[derive(Debug, Clone)]
pub struct InitializationConfig {
    /// Target number of vectors in the block.
    pub block_size: usize,
    /// Maximum random initialization attempts before giving up.
    pub max_random_attempts: usize,
    /// Tolerance for considering a vector as zero (and rejecting it).
    pub zero_tolerance: f64,
}

impl Default for InitializationConfig {
    fn default() -> Self {
        Self {
            block_size: 10,
            max_random_attempts: 64,
            zero_tolerance: 1e-12,
        }
    }
}

// ============================================================================
// Result
// ============================================================================

/// Result of block initialization.
#[derive(Debug, Clone)]
pub struct InitializationResult {
    /// Number of vectors successfully initialized from warm-start.
    pub warm_start_hits: usize,
    /// Number of vectors initialized randomly.
    pub random_vectors: usize,
    /// Total vectors in the block (may be less than requested if initialization failed).
    pub total_vectors: usize,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Seed a buffer with pseudo-random values for initialization.
///
/// Uses a simple xorshift-based PRNG seeded by the phase value.
/// The resulting values are real-valued in the range [-1, 1].
pub fn seed_random_vector(data: &mut [Complex64], phase: f64) {
    let mut state = phase.to_bits().wrapping_mul(0x9E37_79B9_7F4A_7C15);
    if state == 0 {
        state = 0xDEAD_BEEF_CAFE_BABE;
    }
    for value in data.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let real = ((state >> 12) as f64) / ((1u64 << 52) as f64) * 2.0 - 1.0;
        *value = Complex64::new(real, 0.0);
    }
}

/// Check if a Bloch wavevector is at the Γ-point (k ≈ 0).
///
/// Uses a small tolerance to account for floating-point representation.
///
/// # Arguments
/// * `bloch` - The Bloch wavevector [kx, ky]
/// * `tol` - Tolerance for considering k ≈ 0 (default: 1e-12)
pub fn is_gamma_point(bloch: [f64; 2], tol: f64) -> bool {
    bloch[0].abs() < tol && bloch[1].abs() < tol
}

/// Tolerance for Γ-point detection.
pub const GAMMA_TOLERANCE: f64 = 1e-12;

/// Create the B-normalized kernel mode for Γ-point deflation.
///
/// At k=0, the constant field u₀ = 1 (everywhere) is an eigenvector of
/// the curl-curl operator with eigenvalue λ = 0. This function creates
/// the B-normalized version:
///
/// ```text
/// y₀ = u₀ / √(u₀^* B u₀)
/// ```
///
/// **For transformed TE mode**, the operator is A' = ε^{-1/2} A ε^{-1/2},
/// and the kernel is v₀ = ε^{1/2} u₀, not u₀. The function automatically
/// detects this via `operator.gamma_kernel_transform()` and applies the
/// correct transformation.
///
/// # Arguments
/// * `operator` - The linear operator (provides B and allocation)
///
/// # Returns
/// A tuple of (y₀, B·y₀, norm) where:
/// - y₀ is the B-normalized kernel vector
/// - B·y₀ is the mass-weighted version
/// - norm is the original B-norm (before normalization)
///
/// Also verifies that A·y₀ ≈ 0, which should hold for the kernel mode at Γ.
pub fn create_gamma_mode<O, B>(operator: &mut O) -> (B::Buffer, B::Buffer, f64)
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    // Allocate and fill with constant 1.0
    let mut u0 = operator.alloc_field();
    u0.as_mut_slice().fill(Complex64::new(1.0, 0.0));

    // Check if the operator uses a transformed basis (e.g., transformed TE)
    // If so, apply the transformation factor: v₀ = transform · u₀
    if let Some(transform) = operator.gamma_kernel_transform() {
        debug!(
            "[gamma] Applying kernel transformation for transformed operator (e.g., ε^{{1/2}} for TE)"
        );
        for (value, &factor) in u0.as_mut_slice().iter_mut().zip(transform.iter()) {
            *value *= factor;
        }
    }

    // Compute B·u₀
    let mut bu0 = operator.alloc_field();
    operator.apply_mass(&u0, &mut bu0);

    // Compute B-norm: ||u₀||_B = √(u₀^* B u₀)
    let norm_sq = operator.backend().dot(&u0, &bu0).re;
    let norm = if norm_sq > 0.0 { norm_sq.sqrt() } else { 1.0 };

    // Normalize: y₀ = u₀ / ||u₀||_B
    let scale = Complex64::new(1.0 / norm, 0.0);
    operator.backend().scale(scale, &mut u0);
    operator.backend().scale(scale, &mut bu0);

    // Verify that A·y₀ ≈ 0 (kernel mode should be a true eigenvector with λ=0)
    let mut au0 = operator.alloc_field();
    operator.apply(&u0, &mut au0);
    let a_norm_sq = operator.backend().dot(&au0, &au0).re;
    let a_norm = if a_norm_sq > 0.0 {
        a_norm_sq.sqrt()
    } else {
        0.0
    };

    debug!(
        "[gamma] Created Γ-point kernel mode: ||u₀||_B = {:.6e}, ||A·y₀|| = {:.6e}",
        norm, a_norm
    );

    // Warn if A·y₀ is not close to zero (indicates discretization issue or wrong kernel)
    // Threshold of 1e-8 accounts for numerical precision in double precision
    if a_norm > 1e-8 {
        warn!(
            "[gamma] WARNING: A·y₀ not zero! ||A·y₀|| = {:.6e} (expected ~0). \
             This may indicate a discretization issue or incorrect kernel basis at the Γ-point.",
            a_norm
        );
    }

    (u0, bu0, norm)
}

// ============================================================================
// Main Initialization Function
// ============================================================================

/// Initialize a block of vectors for the LOBPCG iteration.
///
/// This function:
/// 1. First tries to use warm-start vectors if provided
/// 2. Fills remaining slots with random vectors
/// 3. B-orthonormalizes all vectors
///
/// The returned vectors satisfy:
/// - ||x_i||_B = 1 (unit B-norm)
/// - <x_i, x_j>_B ≈ 0 for i ≠ j (B-orthogonal)
///
/// # Arguments
/// * `operator` - The linear operator (provides A, B, and allocation)
/// * `config` - Initialization configuration
/// * `warm_start` - Optional warm-start vectors from previous k-point
///
/// # Returns
/// A tuple of (block_entries, initialization_result)
pub fn initialize_block<O, B>(
    operator: &mut O,
    config: &InitializationConfig,
    warm_start: Option<&[Field2D]>,
) -> (Vec<BlockEntry<B>>, InitializationResult)
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    let mut entries: Vec<BlockEntry<B>> = Vec::with_capacity(config.block_size);
    let mut warm_hits = 0usize;

    // Phase 1: Try to use warm-start vectors
    if let Some(seeds) = warm_start {
        for field in seeds.iter().take(config.block_size) {
            if let Some(entry) =
                build_entry_from_field(operator, field, &entries, config.zero_tolerance)
            {
                entries.push(entry);
                warm_hits += 1;
            }
            if entries.len() >= config.block_size {
                break;
            }
        }
    }

    // Phase 2: Fill remaining slots with random vectors
    let mut attempts = 0usize;
    let mut phase = 1.0f64;

    while entries.len() < config.block_size && attempts < config.max_random_attempts {
        let mut vector = operator.alloc_field();
        seed_random_vector(vector.as_mut_slice(), phase);
        phase += 1.0;

        if let Some(entry) =
            build_entry_from_buffer(operator, vector, &entries, config.zero_tolerance)
        {
            entries.push(entry);
        } else {
            attempts += 1;
        }
    }

    let result = InitializationResult {
        warm_start_hits: warm_hits,
        random_vectors: entries.len().saturating_sub(warm_hits),
        total_vectors: entries.len(),
    };

    (entries, result)
}

/// Build a block entry from a Field2D (e.g., warm-start vector).
fn build_entry_from_field<O, B>(
    operator: &mut O,
    field: &Field2D,
    existing: &[BlockEntry<B>],
    zero_tol: f64,
) -> Option<BlockEntry<B>>
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    let mut vector = operator.alloc_field();
    vector.as_mut_slice().copy_from_slice(field.as_slice());
    build_entry_from_buffer(operator, vector, existing, zero_tol)
}

/// Build a block entry from a raw buffer, orthogonalizing against existing entries.
fn build_entry_from_buffer<O, B>(
    operator: &mut O,
    mut vector: B::Buffer,
    existing: &[BlockEntry<B>],
    zero_tol: f64,
) -> Option<BlockEntry<B>>
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    // Compute mass vector
    let mut mass = operator.alloc_field();
    operator.apply_mass(&vector, &mut mass);

    // Project out components along existing basis vectors (B-orthogonalization)
    for entry in existing {
        project_out(
            operator.backend(),
            &mut vector,
            &mut mass,
            &entry.vector,
            &entry.mass,
        );
    }

    // Normalize to unit B-norm (returns 0.0 if norm is below tolerance)
    let norm =
        normalize_to_unit_b_norm_with_tol(operator.backend(), &mut vector, &mut mass, zero_tol);
    if norm == 0.0 {
        return None;
    }

    // Apply the operator to get A*x
    let mut applied = operator.alloc_field();
    operator.apply(&vector, &mut applied);

    Some(BlockEntry {
        vector,
        mass,
        applied,
    })
}
