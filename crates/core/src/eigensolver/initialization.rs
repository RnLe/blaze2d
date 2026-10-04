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
use num_complex::{Complex, Complex64};

use crate::backend::{SpectralBackend, SpectralBuffer};
use crate::field::{Field2D, Real, cscalar};
use crate::operators::LinearOperator;

use super::normalization::{b_norm, normalize_to_unit_b_norm_with_tol, project_out};

// ============================================================================
// Block Entry
// ============================================================================

/// A single block entry containing a vector, its mass-weighted version, and operator application.
///
/// This represents one column of the block matrices in LOBPCG:
/// - `vector`: The eigenvector approximation x_i
/// - `mass`: B * x_i (precomputed for efficiency in B-inner products); `None` when B is
///   the identity, so the vector is its own image and is not copied
/// - `applied`: A * x_i (precomputed for Rayleigh quotient and residual computation)
#[derive(Clone)]
pub struct BlockEntry<B: SpectralBackend> {
    pub vector: B::Buffer,
    pub mass: Option<B::Buffer>,
    pub applied: B::Buffer,
}

impl<B: SpectralBackend> BlockEntry<B> {
    /// B·x (the vector itself when B is the identity).
    pub fn mass(&self) -> &B::Buffer {
        self.mass.as_ref().unwrap_or(&self.vector)
    }

    /// Compute the Rayleigh quotient λ = <x, Ax> / <x, Bx> for this entry.
    pub fn rayleigh_quotient(&self, backend: &B) -> f64 {
        let x_ax = backend.dot(&self.vector, &self.applied).re;
        let x_bx = backend.dot(&self.vector, self.mass()).re;
        if x_bx.abs() > 1e-15 { x_ax / x_bx } else { 0.0 }
    }

    /// Compute the B-norm ||x||_B = sqrt(<x, Bx>).
    pub fn b_norm(&self, backend: &B) -> f64 {
        backend.dot(&self.vector, self.mass()).re.max(0.0).sqrt()
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
pub fn seed_random_vector<R: Real>(data: &mut [Complex<R>], phase: f64) {
    let mut state = phase.to_bits().wrapping_mul(0x9E37_79B9_7F4A_7C15);
    if state == 0 {
        state = 0xDEAD_BEEF_CAFE_BABE;
    }
    for value in data.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let real = ((state >> 12) as f64) / ((1u64 << 52) as f64) * 2.0 - 1.0;
        *value = cscalar::<R>(real, 0.0);
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
    operator.fill_gamma_kernel(&mut u0);

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
    initialize_block_with_deflation(operator,config,warm_start,None)
}

/// Project known locked modes before rank decisions, so a warm-started kernel
/// cannot occupy a slot that becomes zero only after block construction.
pub(crate) fn initialize_block_with_deflation<O,B>(operator:&mut O,config:&InitializationConfig,warm_start:Option<&[Field2D]>,deflation:Option<&super::deflation::DeflationSubspace<B>>)->(Vec<BlockEntry<B>>,InitializationResult)
where O:LinearOperator<B>, B:SpectralBackend {
    // Accepted (x, B·x) pairs; A·x is applied to the finished block in one batch.
    let mut accepted: Vec<(B::Buffer, B::Buffer)> = Vec::with_capacity(config.block_size);
    let mut warm_hits = 0usize;

    // Phase 1: Try to use warm-start vectors. The seeds that can fill the block get
    // their B·x from one batched application; later seeds are only needed after rejections.
    if let Some(seeds) = warm_start {
        let batch = seeds.len().min(config.block_size);
        let mut vectors: Vec<B::Buffer> = seeds[..batch].iter().map(|field| field_to_buffer(operator, field)).collect();
        // Constraints apply to both warm starts and random rank repair.
        for vector in &mut vectors {
            operator.constrain(vector);
        }
        let mut masses: Vec<B::Buffer> = (0..batch).map(|_| operator.alloc_output()).collect();
        operator.batch_apply_mass(&vectors, &mut masses);
        let mut batched = vectors.into_iter().zip(masses);
        for field in seeds {
            if accepted.len() >= config.block_size {
                break;
            }
            crate::profiler::count("initialization.warm_attempts", 1);
            let (vector, mass) = match batched.next() {
                Some(pair) => pair,
                None => {
                    let mut vector = field_to_buffer(operator, field);
                    operator.constrain(&mut vector);
                    let mut mass = operator.alloc_output();
                    operator.apply_mass(&vector, &mut mass);
                    (vector, mass)
                }
            };
            if let Some(pair) = orthonormalize_entry(operator, vector, mass, &accepted, config.zero_tolerance, deflation) {
                accepted.push(pair);
                warm_hits += 1;
            }
        }
    }

    // Phase 2: Fill remaining slots with random vectors
    let mut attempts = 0usize;
    let mut phase = 1.0f64;

    while accepted.len() < config.block_size && attempts < config.max_random_attempts {
        crate::profiler::count("initialization.random_attempts", 1);
        let mut vector = operator.alloc_field();
        operator.seed_initial_vector(&mut vector, phase);
        phase += 1.0;
        operator.constrain(&mut vector);
        let mut mass = operator.alloc_output();
        operator.apply_mass(&vector, &mut mass);

        if let Some(pair) = orthonormalize_entry(operator, vector, mass, &accepted, config.zero_tolerance, deflation) {
            accepted.push(pair);
        } else {
            attempts += 1;
        }
    }

    let (vectors, masses): (Vec<B::Buffer>, Vec<B::Buffer>) = accepted.into_iter().unzip();
    let identity = operator.mass_is_identity();
    let mut applied: Vec<B::Buffer> = (0..vectors.len()).map(|_| operator.alloc_output()).collect();
    operator.batch_apply(&vectors, &mut applied);
    let entries: Vec<BlockEntry<B>> = vectors.into_iter().zip(masses).zip(applied)
        .map(|((vector, mass), applied)| BlockEntry { vector, mass: (!identity).then_some(mass), applied })
        .collect();

    let result = InitializationResult {
        warm_start_hits: warm_hits,
        random_vectors: entries.len().saturating_sub(warm_hits),
        total_vectors: entries.len(),
    };
    crate::profiler::count("initialization.warm_accepts", warm_hits);
    crate::profiler::count("initialization.random_accepts", result.random_vectors);
    crate::profiler::count("initialization.random_rejections", attempts);

    (entries, result)
}

/// An f64 Field2D warm-start vector at the backend's storage precision (`B::Real`).
fn field_to_buffer<O, B>(operator: &O, field: &Field2D) -> B::Buffer
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    let mut vector = operator.alloc_field();
    let dst = vector.as_mut_slice();
    for (d, &s) in dst.iter_mut().zip(field.as_slice().iter()) {
        *d = Complex::new(
            <B::Real as Real>::from_accum(s.re),
            <B::Real as Real>::from_accum(s.im),
        );
    }
    vector
}

/// B-orthonormalize a constrained vector and its B-image against the deflated modes and
/// the accepted pairs; `None` when nothing above `zero_tol` remains.
fn orthonormalize_entry<O, B>(
    operator: &mut O,
    mut vector: B::Buffer,
    mut mass: B::Buffer,
    existing: &[(B::Buffer, B::Buffer)],
    zero_tol: f64,
    deflation:Option<&super::deflation::DeflationSubspace<B>>,
) -> Option<(B::Buffer, B::Buffer)>
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    let backend = operator.backend();
    let before = b_norm(backend, &vector, &mass);
    if let Some(locked)=deflation {locked.project_single(backend,&mut vector,&mut mass);}

    // Project out components along existing basis vectors (B-orthogonalization)
    for (basis, basis_mass) in existing {
        project_out(backend, &mut vector, &mut mass, basis, basis_mass);
    }

    // Normalize to unit B-norm (returns 0.0 if norm is below tolerance)
    let norm = normalize_to_unit_b_norm_with_tol(backend, &mut vector, &mut mass, zero_tol);
    if norm == 0.0 {
        return None;
    }
    // Projection can cancel most of a nearly dependent seed; the incrementally updated
    // B·x then carries rounding error amplified by the lost norm, so it is recomputed.
    if norm < 0.5 * before {
        operator.apply_mass(&vector, &mut mass);
        if normalize_to_unit_b_norm_with_tol(operator.backend(), &mut vector, &mut mass, zero_tol) == 0.0 {
            return None;
        }
    }
    Some((vector, mass))
}
