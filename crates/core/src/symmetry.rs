//! K-path generation for photonic band structure calculations.
//!
//! This module provides utilities to generate k-point paths along high-symmetry
//! directions in the Brillouin zone for common 2D Bravais lattices.
//!
//! The (formerly co-located) symmetry projector / multi-sector scheduling code
//! is archived in `archive/symmetry_projectors/` — it was removed in the
//! Stage 6 sanitization pass after consistently failing to produce a
//! convergence win on the problems Blaze2D targets. See
//! [docs/state_report.md](../../../docs/state_report.md) for context.
//!
//! For new code, prefer the [`crate::brillouin`] module, which supports
//! rectangular lattices as well as square and hexagonal. This module is
//! retained because callers (CLI, bulk-driver, examples) still import
//! `PathType` / `standard_path`.

use crate::{
    brillouin::{BrillouinPath, generate_path},
    lattice::Lattice2D,
};

// ============================================================================
// K-Path Types
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
        PathType::Custom(seq) => generate_path(&BrillouinPath::Custom(seq), segments_per_leg),
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
