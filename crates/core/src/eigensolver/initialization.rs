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
//! At k=0 (the Γ point), the physical zero-eigenvalue mode is known analytically
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
//! where u₀ has value 1 at every spatial grid point.

use log::{debug, warn};
use num_complex::{Complex, Complex64};

use crate::backend::{SpectralBackend, SpectralBuffer};
use crate::field::{Field2D, Real, cscalar};
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
    /// Relative B-norm threshold; storage roundoff sets an additional lower bound.
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
    /// Rejected warm columns, including invalid shape, nonfinite values and lost rank.
    pub rejected_warm_vectors: usize,
    /// Relative norm cutoff used in rank decisions.
    pub rank_tolerance: f64,
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
    initialize_input(operator, config, SeedInput::Public(warm_start.unwrap_or(&[]).iter()), deflation)
}

enum SeedInput<'a, B: SpectralBackend> {
    Public(std::slice::Iter<'a, Field2D>),
    Native(std::vec::IntoIter<B::Buffer>, Option<std::vec::IntoIter<B::Buffer>>),
}

pub(crate) fn initialize_native_block<O: LinearOperator<B>, B: SpectralBackend>(
    operator: &mut O, config: &InitializationConfig, seeds: Vec<B::Buffer>, actions: Option<Vec<B::Buffer>>,
    deflation: Option<&super::deflation::DeflationSubspace<B>>,
) -> (Vec<BlockEntry<B>>, InitializationResult) {
    initialize_input(operator, config, SeedInput::Native(seeds.into_iter(),actions.map(Vec::into_iter)), deflation)
}

fn initialize_input<O: LinearOperator<B>, B: SpectralBackend>(
    operator: &mut O, config: &InitializationConfig, mut seeds: SeedInput<'_, B>,
    deflation: Option<&super::deflation::DeflationSubspace<B>>,
) -> (Vec<BlockEntry<B>>, InitializationResult) {
    let mut entries: Vec<BlockEntry<B>> = Vec::with_capacity(config.block_size);
    let mut warm_hits = 0usize;
    let mut warm_attempts = 0usize;
    let rank_tolerance = config.zero_tolerance.max(32.0 * <B::Real as num_traits::Float>::epsilon().to_accum());

    while entries.len() < config.block_size {
        let entry = match &mut seeds {
            SeedInput::Public(fields) => {
                let Some(field) = fields.next() else { break; };
                build_entry_from_field(operator, field, &entries, rank_tolerance, deflation)
            }
            SeedInput::Native(fields, actions) => {
                let Some(vector) = fields.next() else { break; };
                let action = actions.as_mut().and_then(Iterator::next);
                let a = vector.grid(); let b = operator.grid();
                if (a.nx,a.ny,a.lx,a.ly,a.centered) != (b.nx,b.ny,b.lx,b.ly,b.centered)
                    || vector.len() != a.len() {
                    crate::profiler::count("initialization.invalid_fields", 1);
                    None
                } else {
                    crate::profiler::count("initialization.native_imports", 1);
                    build_entry_with_action(operator, vector, action, &entries, rank_tolerance, deflation)
                }
            }
        };
        warm_attempts += 1;
        crate::profiler::count("initialization.warm_attempts", 1);
        if let Some(entry) = entry { entries.push(entry); warm_hits += 1; }
    }

    // Phase 2: Fill remaining slots with random vectors
    let mut attempts = 0usize;
    let mut phase = 1.0f64;

    while entries.len() < config.block_size && attempts < config.max_random_attempts {
        crate::profiler::count("initialization.random_attempts", 1);
        let mut vector = operator.alloc_field();
        operator.seed_initial_vector(&mut vector, phase);
        phase += 1.0;

        if let Some(entry) =
            build_entry_from_buffer(operator, vector, &entries, rank_tolerance, deflation)
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
        rejected_warm_vectors: warm_attempts - warm_hits,
        rank_tolerance,
    };
    crate::profiler::count("initialization.warm_accepts", warm_hits);
    crate::profiler::count("initialization.random_accepts", result.random_vectors);
    crate::profiler::count("initialization.random_rejections", attempts);

    (entries, result)
}

/// Build a block entry from an f64 Field2D warm-start vector, demoting to
/// the backend's storage precision (`B::Real`) on the way in.
fn build_entry_from_field<O, B>(
    operator: &mut O,
    field: &Field2D,
    existing: &[BlockEntry<B>],
    zero_tol: f64,
    deflation:Option<&super::deflation::DeflationSubspace<B>>,
) -> Option<BlockEntry<B>>
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    let source_grid = field.grid();
    let target_grid = operator.grid();
    if (source_grid.nx, source_grid.ny, source_grid.lx, source_grid.ly, source_grid.centered)
        != (target_grid.nx, target_grid.ny, target_grid.lx, target_grid.ly, target_grid.centered)
        || field.as_slice().iter().any(|z| !z.re.is_finite() || !z.im.is_finite())
    {
        crate::profiler::count("initialization.invalid_fields", 1);
        return None;
    }
    let mut vector = operator.alloc_field();
    crate::profiler::count("initialization.import_source_bytes", field.len()*std::mem::size_of::<Complex64>());
    crate::profiler::count("initialization.import_destination_bytes", field.len()*std::mem::size_of::<Complex<B::Real>>());
    let dst = vector.as_mut_slice();
    for (d, &s) in dst.iter_mut().zip(field.as_slice().iter()) {
        *d = Complex::new(
            <B::Real as Real>::from_accum(s.re),
            <B::Real as Real>::from_accum(s.im),
        );
    }
    build_entry_from_buffer(operator, vector, existing, zero_tol, deflation)
}

/// Build a block entry from a raw buffer, orthogonalizing against existing entries.
fn build_entry_from_buffer<O, B>(
    operator: &mut O,
    vector: B::Buffer,
    existing: &[BlockEntry<B>],
    zero_tol: f64,
    deflation:Option<&super::deflation::DeflationSubspace<B>>,
) -> Option<BlockEntry<B>>
where
    O: LinearOperator<B>,
    B: SpectralBackend,
{
    build_entry_with_action(operator, vector, None, existing, zero_tol, deflation)
}

fn build_entry_with_action<O: LinearOperator<B>, B: SpectralBackend>(
    operator: &mut O, mut vector: B::Buffer, mut action: Option<B::Buffer>,
    existing: &[BlockEntry<B>], zero_tol: f64,
    deflation: Option<&super::deflation::DeflationSubspace<B>>,
) -> Option<BlockEntry<B>> {
    // Only the analytic zero eigenspace may be projected without applying A to
    // its basis. Other locked modes force fresh preparation of the input.
    if deflation.is_some_and(|d|d.eigenvalues().iter().any(|&l|l!=0.)) {action=None;}
    if action.as_ref().is_some_and(|a|a.len()!=vector.len()
        || a.as_slice().iter().any(|z|!z.re.to_accum().is_finite() || !z.im.to_accum().is_finite())) {action=None;}
    // Compare rank after constraints with the incoming norm. In particular,
    // cancellation residue of a projected kernel is not a new direction.
    if vector.as_slice().iter().any(|z| !z.re.to_accum().is_finite() || !z.im.to_accum().is_finite()) {
        return None;
    }
    let mut mass = operator.alloc_field();
    operator.apply_mass(&vector, &mut mass);
    let incoming_norm = operator.backend().dot(&vector, &mass).re.sqrt();
    if !incoming_norm.is_finite() || incoming_norm == 0.0 { return None; }
    operator.constrain(&mut vector);
    operator.apply_mass(&vector, &mut mass);
    for _ in 0..2 {
        if let Some(locked) = deflation {
            locked.project_single(operator.backend(), &mut vector, &mut mass);
        }
        for entry in existing {
            if let Some(applied)=&mut action {
                let coefficient=operator.backend().dot(&entry.vector,&mass);
                operator.backend().axpy(-coefficient,&entry.applied,applied);
            }
            project_out(operator.backend(), &mut vector, &mut mass, &entry.vector, &entry.mass);
        }
        // In f32, independently updated Bx can retain cancellation error even
        // when x is nearly zero. Rank and the next pass use the stored x.
        operator.apply_mass(&vector, &mut mass);
    }
    let norm = normalize_to_unit_b_norm_with_tol(
        operator.backend(), &mut vector, &mut mass, zero_tol * incoming_norm);
    if norm == 0.0 { return None; }
    operator.apply_mass(&vector, &mut mass);

    let applied = if let Some(mut applied)=action {
        operator.backend().scale(Complex64::new(1.0/norm,0.0),&mut applied);
        crate::profiler::count("geometry.initial_actions_reused",1);
        applied
    } else {
        let mut applied=operator.alloc_field(); operator.apply(&vector,&mut applied); applied
    };

    Some(BlockEntry {
        vector,
        mass,
        applied,
    })
}

/// Initial target Rayleigh–Ritz in native storage. Rank repair has already made
/// the Gram matrix nonsingular; whitening accounts for its remaining roundoff.
pub(crate) fn ritz_block<B: SpectralBackend>(
    backend: &B, entries: Vec<BlockEntry<B>>,
) -> (Vec<BlockEntry<B>>, Vec<f64>) {
    use super::refine::{hermitize_in_place, matmul_col_major};
    crate::profile_scope!("initialization.ritz");
    let n = entries.len();
    if n == 0 { return (entries, Vec::new()); }
    let mut s = vec![Complex64::ZERO; n * n];
    let mut h = s.clone();
    for j in 0..n { for i in 0..n {
        s[i+j*n] = backend.dot(&entries[i].vector, &entries[j].mass);
        h[i+j*n] = backend.dot(&entries[i].vector, &entries[j].applied);
    }}
    hermitize_in_place(&mut s, n);
    hermitize_in_place(&mut h, n);
    let eig = backend.dense_eigen(&s, n);
    let mut w = vec![Complex64::ZERO; n*n];
    for k in 0..n {
        assert!(eig.eigenvalues[k] > 0.0, "initial block lost target-metric rank");
        let scale = eig.eigenvalues[k].sqrt().recip();
        for j in 0..n { for i in 0..n {
            w[i+j*n] += eig.eigenvectors[i+k*n] * eig.eigenvectors[j+k*n].conj() * scale;
        }}
    }
    let mut projected = matmul_col_major(&w, &matmul_col_major(&h, &w, n), n);
    hermitize_in_place(&mut projected, n);
    let eig = backend.dense_eigen(&projected, n);
    let coefficients = matmul_col_major(&w, &eig.eigenvectors, n);
    let (mut x, mut bx, mut ax) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for entry in entries { x.push(entry.vector); bx.push(entry.mass); ax.push(entry.applied); }
    let x = backend.linear_combinations(&x, &coefficients, n);
    let bx = backend.linear_combinations(&bx, &coefficients, n);
    let ax = backend.linear_combinations(&ax, &coefficients, n);
    (x.into_iter().zip(bx).zip(ax).map(|((vector,mass),applied)| BlockEntry {vector,mass,applied}).collect(), eig.eigenvalues)
}
