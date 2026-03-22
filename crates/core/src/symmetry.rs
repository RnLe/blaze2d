//! Symmetry utilities and k-path generation for photonic band structure calculations.
//!
//! # Module Status: Archived Symmetry Projection Code
//!
//! This module contains two distinct functionalities:
//!
//! ## Active: K-Path Generation
//!
//! The k-path utilities (`standard_path`, `PathType`, etc.) are actively used by
//! the CLI and band structure runner to generate k-point paths along high-symmetry
//! directions in the Brillouin zone. These remain essential for band structure
//! calculations.
//!
//! **Note**: The new `brillouin` module provides improved path generation with
//! support for rectangular lattices. This module's path generation is maintained
//! for backward compatibility.
//!
//! ## Archived: Symmetry Projectors (Not Used)
//!
//! The symmetry projector code (`SymmetryProjector`, `SymmetrySector`, `Parity`,
//! `enumerate_sectors`, etc.) is **kept for reference but NOT actively used** in
//! the eigensolver. The symmetry projection approach was removed because:
//!
//! 1. **No Performance Win**: In practice, applying symmetry projections at every
//!    LOBPCG iteration adds overhead without proportional convergence speedup.
//!    The matrix-free operator application is already efficient, and the projector
//!    operations (FFT transforms to apply mirror symmetry) are not free.
//!
//! 2. **Complexity vs Benefit**: Multi-sector scheduling (running LOBPCG once per
//!    irrep, then merging results) requires significant bookkeeping and does not
//!    provide meaningful advantages for typical 2D photonic crystal calculations.
//!
//! 3. **Warm-Start Compatibility**: The simple k-path approach with warm-starting
//!    from previous k-points works well without symmetry constraints, and LOBPCG
//!    naturally finds the lowest eigenvalues regardless of their symmetry class.
//!
//! The code is preserved here as documentation of the approach and for potential
//! future use if symmetry-based eigenvalue filtering becomes necessary.
//!
//! # Original Design (for reference)
//!
//! The symmetry projector approach was designed to:
//!
//! 1. **Block-diagonalization**: The Hilbert space splits into orthogonal
//!    invariant subspaces, reducing the effective problem size.
//!
//! 2. **Degeneracy removal**: Near-degenerate eigenvalues from different
//!    irreps are separated, preventing mode mixing and oscillations.
//!
//! 3. **Physical classification**: Modes are classified by their symmetry
//!    (even/odd parity under reflections), matching physical expectations.
//!
//! # Multi-Sector Approach
//!
//! The proper way to use symmetry is to run LOBPCG once per symmetry sector
//! (irreducible representation), then merge all eigenpairs:
//!
//! 1. **Identify active reflections** for the k-point's little group:
//!    - At Γ (k=0): both Rₓ and Rᵧ mirrors → 4 sectors: (+,+), (+,−), (−,+), (−,−)
//!    - On Γ-X (k_y=0): only Rᵧ mirror → 2 sectors: (+), (−)
//!    - At generic k: no mirrors → 1 sector (full space)
//!
//! 2. **Run LOBPCG per sector**: Each run uses a `SymmetryProjector` that
//!    enforces the sector's parity constraints.
//!
//! 3. **Merge results**: Union all eigenpairs, sort by eigenvalue, take top N.
//!
//! This approach:
//! - Gives complete, correct bands comparable to MPB
//! - Each sector converges faster (smaller subspace, no mixing)
//! - Sectors can be parallelized (marked with PARALLELIZATION comments)
//!
//! # Plane-Wave Basis Representation
//!
//! Fields are expanded in plane waves:
//! ```text
//! u(r) = Σ_G c_G exp(i(k+G)·r)
//! ```
//!
//! For a mirror symmetry R_y: y → -y, the action on coefficients at k_y = 0 is:
//! ```text
//! (R_y c)_G = c_{R_y G}  where R_y(G_x, G_y) = (G_x, -G_y)
//! ```
//!
//! The parity projectors are:
//! - Even: c^(+)_G = (c_G + c_{RG}) / 2
//! - Odd:  c^(-)_G = (c_G - c_{RG}) / 2
//!
//! # Little Group Selection
//!
//! Symmetry projectors should only be applied when k lies in a symmetry-invariant
//! subspace (the "little group" G_k). For a mirror:
//! - Mirror y→-y applies when k_y ≈ 0 (on Γ-X line for square lattice)
//! - Mirror x→-x applies when k_x ≈ 0 (on Γ-Y line)
//! - Both mirrors: applicable at Γ, X, M points
//!
//! At generic k-points, the little group is trivial and no symmetry applies.
//!
//! ---
//!
//! **Note**: The CLI flags `--symmetry` and `--multi-sector` were removed.
//! The eigensolver no longer applies symmetry projections during iteration.
//! Only the k-path generation utilities below are actively used.

use log::debug;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::{
    backend::SpectralBuffer,
    brillouin::{BrillouinPath, generate_path},
    grid::Grid2D,
    lattice::{Lattice2D, LatticeClass},
};

// ============================================================================
// K-Path Types (unchanged from original)
// ============================================================================

/// Type of high-symmetry path through the Brillouin zone.
#[derive(Debug, Clone)]
pub enum PathType {
    /// Square lattice: Γ → X → M → Γ
    Square,
    /// Hexagonal lattice: Γ → M → K → Γ
    Hexagonal,
    /// Custom path specified as explicit k-points.
    Custom(Vec<[f64; 2]>),
}

impl PathType {
    /// Convert to the new BrillouinPath type.
    pub fn to_brillouin_path(&self) -> BrillouinPath {
        match self {
            PathType::Square => BrillouinPath::Square,
            PathType::Hexagonal => BrillouinPath::Hexagonal,
            PathType::Custom(points) => BrillouinPath::Custom(points.clone()),
        }
    }
}

/// Generate a standard k-path for the given lattice type.
///
/// The path starts at Γ (k=0), which is optimal for LOBPCG convergence:
/// - Γ gets fresh random initialization (no warm-start poison)
/// - Γ deflation removes the spurious constant mode
/// - Converged Γ eigenvectors provide valid warm-starts for subsequent k-points
///
/// **Note**: For rectangular lattices, use `brillouin::generate_path` with
/// `BrillouinPath::Rectangular` instead.
pub fn standard_path(
    lattice: &Lattice2D,
    path: PathType,
    segments_per_leg: usize,
) -> Vec<[f64; 2]> {
    let _ = lattice;
    match path {
        PathType::Custom(seq) => seq,
        PathType::Square => generate_path(&BrillouinPath::Square, segments_per_leg),
        PathType::Hexagonal => generate_path(&BrillouinPath::Hexagonal, segments_per_leg),
    }
}

/// Standard k-path for square lattice: Γ → X → M → Γ
///
/// High-symmetry points (fractional coordinates):
/// - Γ = (0, 0) - zone center
/// - X = (1/2, 0) - zone face center  
/// - M = (1/2, 1/2) - zone corner
pub const SQUARE_GXMG: [[f64; 2]; 4] = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.0]];

/// Standard k-path for hexagonal/triangular lattice: Γ → M → K → Γ
///
/// Uses the 60° lattice convention: a₁ = [a, 0], a₂ = [a/2, a√3/2]
///
/// High-symmetry points (fractional coordinates):
/// - Γ = (0, 0) - zone center
/// - M = (1/2, 0) - zone edge midpoint
/// - K = (1/3, 1/3) - zone corner (Dirac point)
pub const HEX_GMK: [[f64; 2]; 4] = [[0.0, 0.0], [0.5, 0.0], [1.0 / 3.0, 1.0 / 3.0], [0.0, 0.0]];

/// Standard k-path for rectangular lattice: Γ → X → S → Y → Γ
///
/// High-symmetry points (fractional coordinates):
/// - Γ = (0, 0) - zone center
/// - X = (1/2, 0) - face center along kₓ
/// - S = (1/2, 1/2) - zone corner
/// - Y = (0, 1/2) - face center along kᵧ
pub const RECT_GXSYG: [[f64; 2]; 5] = [
    [0.0, 0.0], // Γ
    [0.5, 0.0], // X
    [0.5, 0.5], // S
    [0.0, 0.5], // Y
    [0.0, 0.0], // Γ
];

// ============================================================================
// Symmetry Types
// ============================================================================

/// Reflection axis for mirror symmetry.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReflectionAxis {
    /// Mirror about x-axis: (x, y) → (x, -y), i.e., y-reflection
    X,
    /// Mirror about y-axis: (x, y) → (-x, y), i.e., x-reflection
    Y,
}

/// Parity under reflection (eigenvalue of mirror operator).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Parity {
    /// Even parity: f(r) = f(R·r), eigenvalue +1
    Even,
    /// Odd parity: f(r) = -f(R·r), eigenvalue -1
    Odd,
}

impl Default for Parity {
    fn default() -> Self {
        Parity::Even
    }
}

/// A single reflection constraint (axis + parity).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReflectionConstraint {
    /// Which axis to reflect about.
    pub axis: ReflectionAxis,
    /// Required parity under this reflection.
    pub parity: Parity,
}

// NOTE: Both TE (H_z) and TM (E_z) are true scalar fields under point group
// operations. Under a mirror reflection R: r → R·r, both fields transform as:
//   H_z(r) → H_z(R·r)    (no sign change)
//   E_z(r) → E_z(R·r)    (no sign change)
//
// Therefore, the parity constraint applies identically to both polarizations.
// There is no pseudo-vector sign flip needed for either case.
// The `Polarization` parameter was removed from symmetry functions since
// it has no effect on the symmetry constraints.

// ============================================================================
// Multi-Sector Symmetry Types
// ============================================================================

/// A symmetry sector (irreducible representation) defined by parity under each active mirror.
///
/// For a k-point with N active mirrors, there are 2^N sectors. Each sector is defined
/// by the parity eigenvalue (+1 or -1) under each mirror operation.
///
/// # Examples
///
/// - At Γ with 2 mirrors (Rₓ, Rᵧ): 4 sectors `(+,+), (+,-), (-,+), (-,-)`
/// - On Γ-X with 1 mirror (Rᵧ): 2 sectors `(+), (-)`
/// - At generic k: 1 sector (full space, no constraints)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymmetrySector {
    /// The reflection constraints that define this sector.
    /// Each constraint specifies a mirror axis and the required parity.
    pub constraints: Vec<ReflectionConstraint>,
    /// Human-readable label for this sector (e.g., "(+,+)" or "(−)").
    pub label: String,
}

impl SymmetrySector {
    /// Create a new sector with the given constraints.
    pub fn new(constraints: Vec<ReflectionConstraint>) -> Self {
        let label = Self::make_label(&constraints);
        Self { constraints, label }
    }

    /// Create the trivial sector (no symmetry constraints, full space).
    pub fn trivial() -> Self {
        Self {
            constraints: Vec::new(),
            label: "(full)".to_string(),
        }
    }

    /// Generate a human-readable label from constraints.
    fn make_label(constraints: &[ReflectionConstraint]) -> String {
        if constraints.is_empty() {
            return "(full)".to_string();
        }
        let parts: Vec<&str> = constraints
            .iter()
            .map(|c| match c.parity {
                Parity::Even => "+",
                Parity::Odd => "−",
            })
            .collect();
        format!("({})", parts.join(","))
    }

    /// Check if this is the trivial sector (no constraints).
    pub fn is_trivial(&self) -> bool {
        self.constraints.is_empty()
    }

    /// Get the number of active mirrors in this sector.
    pub fn n_mirrors(&self) -> usize {
        self.constraints.len()
    }
}

/// Schedule of all sectors to solve for a given k-point.
///
/// This struct contains all the symmetry sectors that must be solved
/// and merged to get the complete set of bands at a k-point.
#[derive(Debug, Clone)]
pub struct SectorSchedule {
    /// The k-point (fractional coordinates).
    pub k_frac: [f64; 2],
    /// All sectors to solve at this k-point.
    pub sectors: Vec<SymmetrySector>,
    /// Active mirror axes at this k-point (the little group mirrors).
    pub active_mirrors: Vec<ReflectionAxis>,
}

impl SectorSchedule {
    /// Create a schedule with just the trivial sector (no symmetry).
    pub fn trivial(k_frac: [f64; 2]) -> Self {
        Self {
            k_frac,
            sectors: vec![SymmetrySector::trivial()],
            active_mirrors: Vec::new(),
        }
    }

    /// Get the number of sectors to solve.
    pub fn n_sectors(&self) -> usize {
        self.sectors.len()
    }

    /// Check if this is a trivial schedule (single full-space sector).
    pub fn is_trivial(&self) -> bool {
        self.sectors.len() == 1 && self.sectors[0].is_trivial()
    }

    /// Estimate the relative cost compared to a single full-space solve.
    ///
    /// With N mirrors and 2^N sectors, each sector has ~1/2^N of the DOFs.
    /// Matvec cost is O(n log n), so total cost is approximately:
    ///   2^N * (N/2^N) * log(N/2^N) ≈ N * (log N - N)
    /// which is slightly cheaper than a single solve on full N.
    pub fn estimated_relative_cost(&self) -> f64 {
        let n_sectors = self.n_sectors() as f64;
        if n_sectors <= 1.0 {
            return 1.0;
        }
        // Each sector is ~1/n_sectors of the full space
        // Cost per sector: (1/n_sectors) * log(1/n_sectors)
        // Total: n_sectors * (1/n_sectors) * log(N/n_sectors)
        //      = log(N) - log(n_sectors)
        // For simplicity, return approximate ratio
        1.0 - (n_sectors.log2() / 10.0).min(0.5)
    }
}

/// Eigenpair from a single sector solve.
///
/// This associates an eigenvalue with its sector origin, which is useful
/// for merging results and understanding band character.
#[derive(Debug, Clone)]
pub struct SectorEigenpair {
    /// The eigenvalue (ω² in physical units).
    pub eigenvalue: f64,
    /// Index of the sector this eigenpair came from.
    pub sector_idx: usize,
    /// Band index within the sector (0-indexed).
    pub band_idx_in_sector: usize,
}

/// Result of solving all sectors at a k-point.
///
/// Contains the merged eigenpairs sorted by eigenvalue.
#[derive(Debug, Clone)]
pub struct MultiSectorResult {
    /// All eigenpairs from all sectors, sorted by eigenvalue.
    pub eigenpairs: Vec<SectorEigenpair>,
    /// The sector schedule that was used.
    pub schedule: SectorSchedule,
    /// Number of eigenpairs per sector (before merging).
    pub bands_per_sector: Vec<usize>,
}

impl MultiSectorResult {
    /// Get the N lowest eigenvalues (merged across all sectors).
    pub fn lowest_eigenvalues(&self, n: usize) -> Vec<f64> {
        self.eigenpairs
            .iter()
            .take(n)
            .map(|ep| ep.eigenvalue)
            .collect()
    }

    /// Get the total number of eigenpairs found.
    pub fn total_eigenpairs(&self) -> usize {
        self.eigenpairs.len()
    }
}

// ============================================================================
// Sector Enumeration
// ============================================================================

/// Enumerate all symmetry sectors for a k-point based on its little group.
///
/// This is the core function for multi-sector scheduling. Given a k-point
/// and lattice class, it determines:
/// 1. Which mirror symmetries are active (the little group)
/// 2. All 2^N combinations of parities (the sectors/irreps)
///
/// # Arguments
/// - `k_frac`: Bloch wavevector in fractional coordinates [kx, ky]
/// - `lattice_class`: Type of lattice (Square, Hexagonal, Oblique)
/// - `tolerance`: Tolerance for detecting symmetry lines (default: 1e-6)
///
/// # Returns
/// A `SectorSchedule` containing all sectors to solve.
pub fn enumerate_sectors(
    k_frac: [f64; 2],
    lattice_class: LatticeClass,
    tolerance: f64,
) -> SectorSchedule {
    let active_mirrors = detect_active_mirrors(k_frac, lattice_class, tolerance);

    if active_mirrors.is_empty() {
        // Generic k-point: no symmetry, single full-space sector
        debug!(
            "[symmetry] k=({:.4},{:.4}): generic k-point, 1 sector (full space)",
            k_frac[0], k_frac[1]
        );
        return SectorSchedule::trivial(k_frac);
    }

    // Enumerate all 2^N parity combinations
    let n_mirrors = active_mirrors.len();
    let n_sectors = 1usize << n_mirrors; // 2^N
    let mut sectors = Vec::with_capacity(n_sectors);

    for sector_idx in 0..n_sectors {
        let mut constraints = Vec::with_capacity(n_mirrors);
        for (mirror_idx, &axis) in active_mirrors.iter().enumerate() {
            // Use bit pattern to assign parities
            let parity = if (sector_idx >> mirror_idx) & 1 == 0 {
                Parity::Even
            } else {
                Parity::Odd
            };
            constraints.push(ReflectionConstraint { axis, parity });
        }
        sectors.push(SymmetrySector::new(constraints));
    }

    debug!(
        "[symmetry] k=({:.4},{:.4}): {} active mirror(s) → {} sectors: {:?}",
        k_frac[0],
        k_frac[1],
        n_mirrors,
        n_sectors,
        sectors.iter().map(|s| s.label.clone()).collect::<Vec<_>>()
    );

    SectorSchedule {
        k_frac,
        sectors,
        active_mirrors,
    }
}

/// Detect which mirror axes are active at a given k-point.
///
/// A mirror is active if the k-point lies on the mirror plane:
/// - Mirror y→-y (ReflectionAxis::X) is active when k_y ≈ 0
/// - Mirror x→-x (ReflectionAxis::Y) is active when k_x ≈ 0
fn detect_active_mirrors(
    k_frac: [f64; 2],
    lattice_class: LatticeClass,
    tolerance: f64,
) -> Vec<ReflectionAxis> {
    let mut mirrors = Vec::new();
    let k_x = k_frac[0];
    let k_y = k_frac[1];

    match lattice_class {
        LatticeClass::Square | LatticeClass::Rectangular => {
            // Mirror y→-y applies when k_y ≈ 0 (on Γ-X line)
            if k_y.abs() < tolerance {
                mirrors.push(ReflectionAxis::X);
            }
            // Mirror x→-x applies when k_x ≈ 0 (on Γ-Y line)
            if k_x.abs() < tolerance {
                mirrors.push(ReflectionAxis::Y);
            }
            // Note: Diagonal mirrors on Γ-M (k_x = k_y) not yet implemented
        }
        LatticeClass::Triangular => {
            // Hexagonal lattice: similar primary mirrors
            if k_y.abs() < tolerance {
                mirrors.push(ReflectionAxis::X);
            }
            if k_x.abs() < tolerance {
                mirrors.push(ReflectionAxis::Y);
            }
        }
        LatticeClass::Oblique => {
            // No mirror symmetry
        }
    }

    mirrors
}

/// Convenience function: build a SymmetryProjector for a specific sector.
///
/// This creates a projector that enforces all the parity constraints
/// of the given sector.
pub fn projector_for_sector(grid: Grid2D, sector: &SymmetrySector) -> Option<SymmetryProjector> {
    if sector.is_trivial() {
        return None;
    }
    SymmetryProjector::new(grid, sector.constraints.clone())
}

// ============================================================================
// Symmetry Configuration
// ============================================================================

/// Main symmetry configuration for eigensolver.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SymmetryConfig {
    /// Whether symmetry projection is enabled.
    /// When false, no symmetry constraints are applied (all modes computed).
    pub enabled: bool,

    /// Parity to use for all applicable reflections.
    /// This sets a single parity for all mirror symmetries.
    pub parity: Parity,

    /// Tolerance for determining if k lies on a symmetry-invariant subspace.
    /// If |k_component| < tolerance, the corresponding mirror applies.
    pub bloch_tolerance: f64,

    /// Explicitly specified reflections (overrides auto-detection if non-empty).
    #[serde(default)]
    pub reflections: Vec<ReflectionConstraint>,
}

impl Default for SymmetryConfig {
    fn default() -> Self {
        Self {
            enabled: true, // Symmetry enabled by default
            parity: Parity::Even,
            bloch_tolerance: 1e-6,
            reflections: Vec::new(),
        }
    }
}

impl SymmetryConfig {
    /// Create a config with symmetry disabled.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Create a config with even parity.
    pub fn even() -> Self {
        Self {
            enabled: true,
            parity: Parity::Even,
            ..Default::default()
        }
    }

    /// Create a config with odd parity.
    pub fn odd() -> Self {
        Self {
            enabled: true,
            parity: Parity::Odd,
            ..Default::default()
        }
    }

    /// Builder: set parity.
    pub fn with_parity(mut self, parity: Parity) -> Self {
        self.parity = parity;
        self
    }

    /// Builder: disable symmetry.
    pub fn without_symmetry(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Builder: enable symmetry.
    pub fn with_symmetry(mut self) -> Self {
        self.enabled = true;
        self
    }
}

// ============================================================================
// G-Index Mirror Partner Table
// ============================================================================

/// Precomputed mirror partner indices for fast projection.
///
/// For a grid with dimensions (nx, ny), each G-vector index has a partner
/// under each mirror operation. This table is computed once per grid and
/// reused for all projections.
#[derive(Debug, Clone)]
pub struct MirrorPartnerTable {
    /// partner_y[idx] = index of G' = (G_x, -G_y) in the FFT layout.
    /// Used for mirror about x-axis (y → -y).
    partner_y: Vec<usize>,
    /// partner_x[idx] = index of G' = (-G_x, G_y) in the FFT layout.
    /// Used for mirror about y-axis (x → -x).
    partner_x: Vec<usize>,
}

impl MirrorPartnerTable {
    /// Build the mirror partner table for the given grid.
    ///
    /// The FFT layout uses standard frequency ordering:
    /// - For dimension n: indices 0, 1, ..., n/2-1, -n/2, ..., -1
    /// - The mirror of frequency f is -f (mod n)
    pub fn new(grid: Grid2D) -> Self {
        let n = grid.len();
        let mut partner_y = vec![0usize; n];
        let mut partner_x = vec![0usize; n];

        for iy in 0..grid.ny {
            for ix in 0..grid.nx {
                let idx = grid.idx(ix, iy);

                // Mirror y-component: iy → (ny - iy) mod ny
                // This corresponds to G_y → -G_y in FFT frequency space
                let mirror_iy = if iy == 0 { 0 } else { grid.ny - iy };
                partner_y[idx] = grid.idx(ix, mirror_iy);

                // Mirror x-component: ix → (nx - ix) mod nx
                // This corresponds to G_x → -G_x in FFT frequency space
                let mirror_ix = if ix == 0 { 0 } else { grid.nx - ix };
                partner_x[idx] = grid.idx(mirror_ix, iy);
            }
        }

        Self {
            partner_y,
            partner_x,
        }
    }

    /// Get the mirror partner index under y-reflection (y → -y).
    #[inline]
    pub fn partner_under_y_mirror(&self, idx: usize) -> usize {
        self.partner_y[idx]
    }

    /// Get the mirror partner index under x-reflection (x → -x).
    #[inline]
    pub fn partner_under_x_mirror(&self, idx: usize) -> usize {
        self.partner_x[idx]
    }

    /// Check if an index is on the y=0 axis (G_y = 0).
    /// These indices are self-partners under y-reflection.
    #[inline]
    pub fn is_on_y_axis(&self, idx: usize) -> bool {
        self.partner_y[idx] == idx
    }

    /// Check if an index is on the x=0 axis (G_x = 0).
    /// These indices are self-partners under x-reflection.
    #[inline]
    pub fn is_on_x_axis(&self, idx: usize) -> bool {
        self.partner_x[idx] == idx
    }
}

// ============================================================================
// The Symmetry Projector
// ============================================================================

/// Symmetry projector that enforces parity constraints in G-space.
///
/// This projector works directly on plane-wave coefficients {c_G} to enforce
/// definite parity under reflection symmetries. It should be applied to:
/// - Initial guess vectors
/// - Residuals after computation
/// - Preconditioned residuals
/// - Updated Ritz vectors and history directions
///
/// # Thread Safety
///
/// The projector is read-only during application and can be shared across
/// multiple vectors. It does not allocate during `apply()`.
#[derive(Debug, Clone)]
pub struct SymmetryProjector {
    /// Mirror partner lookup table.
    table: MirrorPartnerTable,
    /// Active reflection constraints for this k-point.
    reflections: Vec<ReflectionConstraint>,
}

impl SymmetryProjector {
    /// Create a projector with the given constraints.
    ///
    /// Returns `None` if no reflections are active (trivial projector).
    pub fn new(grid: Grid2D, reflections: Vec<ReflectionConstraint>) -> Option<Self> {
        if reflections.is_empty() {
            return None;
        }
        Some(Self {
            table: MirrorPartnerTable::new(grid),
            reflections,
        })
    }

    /// Create a projector for a specific k-point based on symmetry config.
    ///
    /// Determines which mirror symmetries apply based on k-point position:
    /// - Mirror y→-y applies when k_y ≈ 0 (on Γ-X line for square)
    /// - Mirror x→-x applies when k_x ≈ 0 (on Γ-Y line for square)
    ///
    /// # Arguments
    /// - `grid`: The computational grid
    /// - `bloch_frac`: Bloch wavevector in fractional coordinates [k_x, k_y]
    /// - `config`: Symmetry configuration
    /// - `lattice_class`: Type of lattice (Square, Hexagonal, etc.)
    pub fn for_k_point(
        grid: Grid2D,
        bloch_frac: [f64; 2],
        config: &SymmetryConfig,
        lattice_class: LatticeClass,
    ) -> Option<Self> {
        if !config.enabled {
            return None;
        }

        // Use explicit reflections if provided
        if !config.reflections.is_empty() {
            return Self::new(grid, config.reflections.clone());
        }

        // Auto-detect applicable reflections based on k-point and lattice
        let reflections = detect_applicable_reflections(
            bloch_frac,
            config.parity,
            config.bloch_tolerance,
            lattice_class,
        );

        if reflections.is_empty() {
            debug!(
                "[symmetry] k=({:.4},{:.4}): no applicable reflections (generic k)",
                bloch_frac[0], bloch_frac[1]
            );
            return None;
        }

        debug!(
            "[symmetry] k=({:.4},{:.4}): {} reflection(s) active ({:?})",
            bloch_frac[0],
            bloch_frac[1],
            reflections.len(),
            reflections
                .iter()
                .map(|r| format!("{:?}={:?}", r.axis, r.parity))
                .collect::<Vec<_>>()
        );

        Self::new(grid, reflections)
    }

    /// Check if this projector has any active constraints.
    pub fn is_empty(&self) -> bool {
        self.reflections.is_empty()
    }

    /// Get the number of active reflection constraints.
    pub fn len(&self) -> usize {
        self.reflections.len()
    }

    /// Get the active reflections.
    pub fn reflections(&self) -> &[ReflectionConstraint] {
        &self.reflections
    }

    /// Apply the symmetry projection to a single buffer (in-place).
    ///
    /// This projects the buffer onto the subspace with definite parity
    /// under all active reflection symmetries. The projection is applied
    /// sequentially for each reflection.
    ///
    /// # Performance
    ///
    /// - No allocations (works in-place with a single temporary value)
    /// - O(n) where n = grid size, for each reflection
    /// - Uses visited flags to avoid double-processing pairs
    pub fn apply<B: SpectralBuffer>(&self, buffer: &mut B) {
        for reflection in &self.reflections {
            self.apply_single_reflection(buffer, reflection);
        }
    }

    /// Apply the symmetry projection to multiple buffers.
    pub fn apply_block<B: SpectralBuffer>(&self, buffers: &mut [B]) {
        for buffer in buffers {
            self.apply(buffer);
        }
    }

    /// Apply a single reflection projection.
    fn apply_single_reflection<B: SpectralBuffer>(
        &self,
        buffer: &mut B,
        reflection: &ReflectionConstraint,
    ) {
        let n = buffer.len();
        let data = buffer.as_mut_slice();

        // We need to track which indices we've already processed
        // to avoid applying the projection twice to the same pair.
        // Using a bit vector would be cleaner but this avoids allocation.

        for idx in 0..n {
            // Get the mirror partner
            let partner_idx = match reflection.axis {
                ReflectionAxis::X => self.table.partner_under_y_mirror(idx),
                ReflectionAxis::Y => self.table.partner_under_x_mirror(idx),
            };

            // Only process each pair once (when idx <= partner)
            // This also handles self-partner cases (idx == partner)
            if idx > partner_idx {
                continue;
            }

            let c_i = data[idx];
            let c_j = data[partner_idx];

            if idx == partner_idx {
                // Self-partner (on the mirror axis):
                // - Even parity: keep as-is (already symmetric)
                // - Odd parity: must be zero (antisymmetric requires c = -c)
                match reflection.parity {
                    Parity::Even => { /* keep as-is */ }
                    Parity::Odd => {
                        data[idx] = Complex64::new(0.0, 0.0);
                    }
                }
            } else {
                // Regular pair: project onto symmetric/antisymmetric combination
                match reflection.parity {
                    Parity::Even => {
                        let even = (c_i + c_j) * 0.5;
                        data[idx] = even;
                        data[partner_idx] = even;
                    }
                    Parity::Odd => {
                        let odd = (c_i - c_j) * 0.5;
                        data[idx] = odd;
                        data[partner_idx] = -odd;
                    }
                }
            }
        }
    }
}

// ============================================================================
// Little Group Detection
// ============================================================================

/// Detect which reflection symmetries apply at a given k-point.
///
/// The "little group" G_k of a k-point consists of all point group operations
/// that leave k invariant (up to a reciprocal lattice vector). For mirrors:
///
/// - Mirror y→-y (ReflectionAxis::X) applies when k_y = 0
/// - Mirror x→-x (ReflectionAxis::Y) applies when k_x = 0
///
/// This function checks if each component is within tolerance of 0.
fn detect_applicable_reflections(
    bloch_frac: [f64; 2],
    parity: Parity,
    tolerance: f64,
    lattice_class: LatticeClass,
) -> Vec<ReflectionConstraint> {
    let mut reflections = Vec::new();

    let k_x = bloch_frac[0];
    let k_y = bloch_frac[1];

    match lattice_class {
        LatticeClass::Square | LatticeClass::Rectangular => {
            // Square/rectangular lattice has C4v (square) or C2v (rectangular) symmetry
            // Mirror y→-y applies when k_y ≈ 0 (on Γ-X line)
            if k_y.abs() < tolerance {
                reflections.push(ReflectionConstraint {
                    axis: ReflectionAxis::X, // y → -y
                    parity,
                });
            }
            // Mirror x→-x applies when k_x ≈ 0 (on Γ-Y line)
            if k_x.abs() < tolerance {
                reflections.push(ReflectionConstraint {
                    axis: ReflectionAxis::Y, // x → -x
                    parity,
                });
            }
            // Additional: on Γ-M diagonal (k_x = k_y), diagonal mirrors apply
            // For now we skip these as they're more complex
        }
        LatticeClass::Triangular => {
            // Triangular/hexagonal lattice has C6v symmetry
            // Similar logic for primary mirror axes
            // The mirror planes are along primitive reciprocal lattice directions
            // For simplicity, use the same x/y mirrors at Γ
            if k_y.abs() < tolerance {
                reflections.push(ReflectionConstraint {
                    axis: ReflectionAxis::X,
                    parity,
                });
            }
            if k_x.abs() < tolerance {
                reflections.push(ReflectionConstraint {
                    axis: ReflectionAxis::Y,
                    parity,
                });
            }
        }
        LatticeClass::Oblique => {
            // Oblique lattice has no mirror symmetry
            // Return empty (no applicable reflections)
        }
    }

    reflections
}

/// Check if a k-point is at a high-symmetry point where multiple mirrors apply.
pub fn is_high_symmetry_point(bloch_frac: [f64; 2], tolerance: f64) -> bool {
    let k_x = bloch_frac[0];
    let k_y = bloch_frac[1];

    // Γ point: k = (0, 0)
    if k_x.abs() < tolerance && k_y.abs() < tolerance {
        return true;
    }

    // X point: k = (0.5, 0) in square lattice
    if (k_x - 0.5).abs() < tolerance && k_y.abs() < tolerance {
        return true;
    }

    // M point: k = (0.5, 0.5) in square lattice
    if (k_x - 0.5).abs() < tolerance && (k_y - 0.5).abs() < tolerance {
        return true;
    }

    false
}

/// Check if a k-point lies on a high-symmetry line.
pub fn is_on_symmetry_line(bloch_frac: [f64; 2], tolerance: f64) -> bool {
    let k_x = bloch_frac[0];
    let k_y = bloch_frac[1];

    // On Γ-X line (k_y = 0)
    if k_y.abs() < tolerance {
        return true;
    }

    // On Γ-Y line (k_x = 0)
    if k_x.abs() < tolerance {
        return true;
    }

    // On Γ-M diagonal (k_x = k_y)
    if (k_x - k_y).abs() < tolerance {
        return true;
    }

    false
}

// ============================================================================
// Legacy Compatibility (SymmetryOptions type)
// ============================================================================

/// Legacy symmetry options type for backward compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SymmetryOptions {
    /// Whether symmetry is enabled.
    #[serde(default = "default_symmetry_enabled")]
    pub enabled: bool,
    /// Explicit reflections (if any).
    pub reflections: Vec<ReflectionConstraint>,
    /// Auto-detection settings.
    pub auto: Option<AutoSymmetry>,
    /// Resolved auto-reflections (internal).
    #[serde(skip)]
    pub(crate) auto_reflections: Vec<ReflectionConstraint>,
}

fn default_symmetry_enabled() -> bool {
    true // Enabled by default (changed from false)
}

impl Default for SymmetryOptions {
    fn default() -> Self {
        Self {
            enabled: true, // Enabled by default
            reflections: Vec::new(),
            auto: Some(AutoSymmetry::default()),
            auto_reflections: Vec::new(),
        }
    }
}

/// Auto-detection settings for symmetry.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AutoSymmetry {
    /// Parity to use for auto-detected reflections.
    #[serde(default)]
    pub parity: Parity,
    /// Tolerance for k-component checks.
    #[serde(default = "default_bloch_tolerance")]
    pub bloch_tolerance: f64,
}

fn default_bloch_tolerance() -> f64 {
    1e-6
}

impl Default for AutoSymmetry {
    fn default() -> Self {
        Self {
            parity: Parity::Even,
            bloch_tolerance: default_bloch_tolerance(),
        }
    }
}

impl SymmetryOptions {
    /// Create disabled symmetry options.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            reflections: Vec::new(),
            auto: None,
            auto_reflections: Vec::new(),
        }
    }

    /// Disable symmetry.
    pub fn disable(&mut self) {
        self.enabled = false;
        self.reflections.clear();
        self.auto = None;
        self.auto_reflections.clear();
    }

    /// Convert to SymmetryConfig.
    pub fn to_config(&self) -> SymmetryConfig {
        SymmetryConfig {
            enabled: self.enabled,
            parity: self.auto.as_ref().map(|a| a.parity).unwrap_or_default(),
            bloch_tolerance: self
                .auto
                .as_ref()
                .map(|a| a.bloch_tolerance)
                .unwrap_or(1e-6),
            reflections: self.reflections.clone(),
        }
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Get default reflections for a lattice type.
pub fn reflections_for_lattice(lattice: &Lattice2D, parity: Parity) -> Vec<ReflectionConstraint> {
    match lattice.classify() {
        LatticeClass::Square | LatticeClass::Rectangular => vec![
            ReflectionConstraint {
                axis: ReflectionAxis::X,
                parity,
            },
            ReflectionConstraint {
                axis: ReflectionAxis::Y,
                parity,
            },
        ],
        LatticeClass::Triangular => vec![
            ReflectionConstraint {
                axis: ReflectionAxis::X,
                parity,
            },
            ReflectionConstraint {
                axis: ReflectionAxis::Y,
                parity,
            },
        ],
        LatticeClass::Oblique => Vec::new(),
    }
}
