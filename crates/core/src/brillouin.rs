//! High-symmetry paths through the Brillouin zone.
//!
//! This module defines standard high-symmetry paths for different 2D Bravais
//! lattice types. These paths are used for band structure calculations to
//! sample the Brillouin zone along directions that reveal band gaps and
//! important physical features.
//!
//! # Lattice Convention
//!
//! All lattice types use the following conventions for primitive vectors:
//!
//! - **Square**: a₁ = [a, 0], a₂ = [0, a]
//! - **Rectangular**: a₁ = [a, 0], a₂ = [0, b]
//! - **Triangular/Hexagonal (60° convention)**: a₁ = [a, 0], a₂ = [a/2, a√3/2]
//! - **Oblique**: a₁ = [a, 0], a₂ = [b·cos(α), b·sin(α)]
//!
//! The triangular/hexagonal lattice uses the standard 60° crystallographic
//! convention matching Setyawan-Curtarolo, SeeK-path, and MPB's Python interface.
//!
//! # High-Symmetry Points (in fractional/reduced coordinates)
//!
//! Each lattice type has specific high-symmetry points in the Brillouin zone:
//!
//! ## Square Lattice
//! - Γ = (0, 0) - Zone center
//! - X = (1/2, 0) - Zone face center
//! - M = (1/2, 1/2) - Zone corner
//!
//! ## Rectangular Lattice  
//! - Γ = (0, 0) - Zone center
//! - X = (1/2, 0) - Face center along kₓ
//! - S = (1/2, 1/2) - Zone corner
//! - Y = (0, 1/2) - Face center along kᵧ
//!
//! ## Triangular/Hexagonal Lattice (60° convention)
//! - Γ = (0, 0) - Zone center
//! - M = (1/2, 0) - Zone edge midpoint
//! - K = (1/3, 1/3) - Zone corner (Dirac point in graphene)
//!
//! ## Oblique Lattice
//! - No standard path (must specify explicitly)

use serde::{Deserialize, Serialize};

use crate::bravais::{BravaisLattice, LatticeType};

// ============================================================================
// High-Symmetry Points
// ============================================================================

/// A high-symmetry point in the Brillouin zone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HighSymmetryPoint {
    /// Name/label of the point (e.g., "Γ", "X", "M", "K").
    pub name: &'static str,
    /// Fractional k-space coordinates [k₁, k₂].
    pub k_frac: [f64; 2],
}

impl HighSymmetryPoint {
    /// Create a new high-symmetry point.
    pub const fn new(name: &'static str, k_frac: [f64; 2]) -> Self {
        Self { name, k_frac }
    }
}

// Common high-symmetry points
/// Zone center Γ = (0, 0)
pub const GAMMA: HighSymmetryPoint = HighSymmetryPoint::new("Γ", [0.0, 0.0]);

// Square lattice points
/// X = (0.5, 0) - face center for square lattice
pub const X_SQUARE: HighSymmetryPoint = HighSymmetryPoint::new("X", [0.5, 0.0]);
/// M = (0.5, 0.5) - corner for square lattice
pub const M_SQUARE: HighSymmetryPoint = HighSymmetryPoint::new("M", [0.5, 0.5]);

// Rectangular lattice points
/// X = (0.5, 0) - face center along kₓ for rectangular lattice
pub const X_RECT: HighSymmetryPoint = HighSymmetryPoint::new("X", [0.5, 0.0]);
/// S = (0.5, 0.5) - corner for rectangular lattice
pub const S_RECT: HighSymmetryPoint = HighSymmetryPoint::new("S", [0.5, 0.5]);
/// Y = (0, 0.5) - face center along kᵧ for rectangular lattice
pub const Y_RECT: HighSymmetryPoint = HighSymmetryPoint::new("Y", [0.0, 0.5]);

// Triangular/hexagonal lattice points
/// M = (0.5, 0) - edge midpoint for triangular lattice
pub const M_HEX: HighSymmetryPoint = HighSymmetryPoint::new("M", [0.5, 0.0]);
/// K = (1/3, 1/3) - corner for triangular lattice
pub const K_HEX: HighSymmetryPoint = HighSymmetryPoint::new("K", [1.0 / 3.0, 1.0 / 3.0]);

// ============================================================================
// Path Type
// ============================================================================

/// Type of high-symmetry path through the Brillouin zone.
///
/// Each variant corresponds to a standard path for a specific lattice type,
/// or allows custom paths for more complex cases.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BrillouinPath {
    /// Square lattice path: Γ → X → M → Γ
    ///
    /// This path samples:
    /// - Γ-X: Along the zone boundary, looking for band gaps
    /// - X-M: Corner of the irreducible Brillouin zone
    /// - M-Γ: Diagonal through the zone
    Square,

    /// Rectangular lattice path: Γ → X → S → Y → Γ
    ///
    /// This path visits all four edges of the rectangular Brillouin zone,
    /// which is necessary because the zone is not symmetric under 90° rotation.
    Rectangular,

    /// Triangular/hexagonal lattice path: Γ → M → K → Γ
    ///
    /// This path samples the irreducible Brillouin zone of the hexagonal lattice.
    /// The K point is particularly important as it can host Dirac points.
    Triangular,

    /// Alias for Triangular - uses the same path.
    #[serde(alias = "hex")]
    Hexagonal,

    /// Custom path specified as a sequence of k-points.
    ///
    /// Use this for oblique lattices or when you need non-standard paths.
    Custom(Vec<[f64; 2]>),
}

impl BrillouinPath {
    /// Get the high-symmetry points defining this path.
    ///
    /// Returns the corner points of the path (not the densified version).
    pub fn high_symmetry_points(&self) -> Vec<HighSymmetryPoint> {
        match self {
            BrillouinPath::Square => vec![GAMMA, X_SQUARE, M_SQUARE, GAMMA],
            BrillouinPath::Rectangular => vec![GAMMA, X_RECT, S_RECT, Y_RECT, GAMMA],
            BrillouinPath::Triangular | BrillouinPath::Hexagonal => {
                vec![GAMMA, M_HEX, K_HEX, GAMMA]
            }
            BrillouinPath::Custom(points) => points
                .iter()
                .map(|&k| HighSymmetryPoint::new("?", k))
                .collect(),
        }
    }

    /// Get the raw k-point coordinates for this path (not densified).
    pub fn raw_k_points(&self) -> Vec<[f64; 2]> {
        self.high_symmetry_points()
            .into_iter()
            .map(|p| p.k_frac)
            .collect()
    }

    /// Check if this path is compatible with the given lattice type.
    pub fn is_compatible_with(&self, lattice_type: LatticeType) -> bool {
        match (self, lattice_type) {
            (BrillouinPath::Square, LatticeType::Square) => true,
            (BrillouinPath::Rectangular, LatticeType::Rectangular) => true,
            (BrillouinPath::Triangular | BrillouinPath::Hexagonal, LatticeType::Triangular) => true,
            (BrillouinPath::Custom(_), _) => true, // Custom always allowed
            _ => false,
        }
    }

    /// Get the recommended path for a given lattice type.
    ///
    /// Returns `None` for oblique lattices which have no standard path.
    pub fn for_lattice_type(lattice_type: LatticeType) -> Option<Self> {
        match lattice_type {
            LatticeType::Square => Some(BrillouinPath::Square),
            LatticeType::Rectangular => Some(BrillouinPath::Rectangular),
            LatticeType::Triangular => Some(BrillouinPath::Triangular),
            LatticeType::Oblique => None,
        }
    }

    /// Get a human-readable name for this path type.
    pub fn name(&self) -> &'static str {
        match self {
            BrillouinPath::Square => "Γ-X-M-Γ (square)",
            BrillouinPath::Rectangular => "Γ-X-S-Y-Γ (rectangular)",
            BrillouinPath::Triangular => "Γ-M-K-Γ (triangular)",
            BrillouinPath::Hexagonal => "Γ-M-K-Γ (hexagonal)",
            BrillouinPath::Custom(_) => "(custom)",
        }
    }

    /// Get the number of legs in this path.
    ///
    /// A leg is a segment between two high-symmetry points.
    pub fn num_legs(&self) -> usize {
        let points = self.high_symmetry_points();
        if points.len() < 2 {
            0
        } else {
            points.len() - 1
        }
    }
}

// ============================================================================
// Path Generation
// ============================================================================

/// Densify a path by interpolating between high-symmetry points.
///
/// This generates a smooth path through the Brillouin zone suitable for
/// band structure calculations.
///
/// # Arguments
/// - `path`: The high-symmetry path type
/// - `segments_per_leg`: Number of k-points per segment between high-symmetry points
///
/// # Returns
/// A vector of k-points in fractional coordinates.
pub fn generate_path(path: &BrillouinPath, segments_per_leg: usize) -> Vec<[f64; 2]> {
    let nodes = path.raw_k_points();
    densify_path(&nodes, segments_per_leg)
}

/// Generate the standard path for a lattice.
///
/// Convenience function that selects the appropriate path type and densifies it.
///
/// # Returns
/// `None` if the lattice is oblique (no standard path exists).
pub fn standard_path_for_lattice(
    lattice: &BravaisLattice,
    segments_per_leg: usize,
) -> Option<Vec<[f64; 2]>> {
    BrillouinPath::for_lattice_type(lattice.lattice_type())
        .map(|path| generate_path(&path, segments_per_leg))
}

/// Densify a k-path by linear interpolation.
fn densify_path(nodes: &[[f64; 2]], segments_per_leg: usize) -> Vec<[f64; 2]> {
    if nodes.len() <= 1 {
        return nodes.to_vec();
    }
    let segments = segments_per_leg.max(1);
    let mut path = Vec::with_capacity(nodes.len() * segments);
    path.push(nodes[0]);

    for window in nodes.windows(2) {
        let start = window[0];
        let end = window[1];
        for step in 1..=segments {
            let t = step as f64 / segments as f64;
            let point = [
                (1.0 - t) * start[0] + t * end[0],
                (1.0 - t) * start[1] + t * end[1],
            ];
            // Avoid duplicate points at corners
            if path.last().map(|last| last != &point).unwrap_or(true) {
                path.push(point);
            }
        }
    }
    path
}

/// Calculate the cumulative distance along a k-path in Cartesian coordinates.
///
/// This is useful for plotting band structures with proper spacing.
pub fn path_distances(k_path: &[[f64; 2]], lattice: &BravaisLattice) -> Vec<f64> {
    if k_path.is_empty() {
        return Vec::new();
    }

    let recip = lattice.reciprocal();
    let mut distances = vec![0.0; k_path.len()];

    for i in 1..k_path.len() {
        let k_prev = recip.fractional_to_cartesian(k_path[i - 1]);
        let k_curr = recip.fractional_to_cartesian(k_path[i]);
        let dk = [k_curr[0] - k_prev[0], k_curr[1] - k_prev[1]];
        let dist = (dk[0] * dk[0] + dk[1] * dk[1]).sqrt();
        distances[i] = distances[i - 1] + dist;
    }

    distances
}

// ============================================================================
// Path Preset (for IO compatibility)
// ============================================================================

/// Preset path type for configuration files.
///
/// This is a simplified version of `BrillouinPath` for use in TOML configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PathPreset {
    /// Square lattice: Γ → X → M → Γ
    Square,
    /// Rectangular lattice: Γ → X → S → Y → Γ
    Rectangular,
    /// Triangular/hexagonal lattice: Γ → M → K → Γ
    Triangular,
    /// Alias for triangular
    #[serde(alias = "hex")]
    Hexagonal,
}

impl PathPreset {
    /// Convert to a `BrillouinPath`.
    pub fn to_path(self) -> BrillouinPath {
        match self {
            PathPreset::Square => BrillouinPath::Square,
            PathPreset::Rectangular => BrillouinPath::Rectangular,
            PathPreset::Triangular => BrillouinPath::Triangular,
            PathPreset::Hexagonal => BrillouinPath::Hexagonal,
        }
    }

    /// Get the default preset for a lattice type.
    pub fn for_lattice_type(lattice_type: LatticeType) -> Option<Self> {
        match lattice_type {
            LatticeType::Square => Some(PathPreset::Square),
            LatticeType::Rectangular => Some(PathPreset::Rectangular),
            LatticeType::Triangular => Some(PathPreset::Triangular),
            LatticeType::Oblique => None,
        }
    }
}

impl From<PathPreset> for BrillouinPath {
    fn from(preset: PathPreset) -> Self {
        preset.to_path()
    }
}

impl std::fmt::Display for PathPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PathPreset::Square => write!(f, "square"),
            PathPreset::Rectangular => write!(f, "rectangular"),
            PathPreset::Triangular => write!(f, "triangular"),
            PathPreset::Hexagonal => write!(f, "hexagonal"),
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
    fn square_path_points() {
        let path = BrillouinPath::Square;
        let points = path.raw_k_points();
        assert_eq!(points.len(), 4);
        assert_eq!(points[0], [0.0, 0.0]); // Γ
        assert_eq!(points[1], [0.5, 0.0]); // X
        assert_eq!(points[2], [0.5, 0.5]); // M
        assert_eq!(points[3], [0.0, 0.0]); // Γ
    }

    #[test]
    fn rectangular_path_points() {
        let path = BrillouinPath::Rectangular;
        let points = path.raw_k_points();
        assert_eq!(points.len(), 5);
        assert_eq!(points[0], [0.0, 0.0]); // Γ
        assert_eq!(points[1], [0.5, 0.0]); // X
        assert_eq!(points[2], [0.5, 0.5]); // S
        assert_eq!(points[3], [0.0, 0.5]); // Y
        assert_eq!(points[4], [0.0, 0.0]); // Γ
    }

    #[test]
    fn triangular_path_points() {
        let path = BrillouinPath::Triangular;
        let points = path.raw_k_points();
        assert_eq!(points.len(), 4);
        assert_eq!(points[0], [0.0, 0.0]); // Γ
        assert_eq!(points[1], [0.5, 0.0]); // M
        assert!((points[2][0] - 1.0 / 3.0).abs() < 1e-10); // K
        assert!((points[2][1] - 1.0 / 3.0).abs() < 1e-10);
        assert_eq!(points[3], [0.0, 0.0]); // Γ
    }

    #[test]
    fn path_densification() {
        let path = BrillouinPath::Square;
        let dense = generate_path(&path, 10);
        // 4 corners, 3 legs, 10 segments each = 3*10 + 1 = 31 points
        // (minus duplicates at corners = 31)
        assert_eq!(dense.len(), 31);
        // First and last should be Γ
        assert_eq!(dense[0], [0.0, 0.0]);
        assert_eq!(dense[dense.len() - 1], [0.0, 0.0]);
    }

    #[test]
    fn path_compatibility() {
        assert!(BrillouinPath::Square.is_compatible_with(LatticeType::Square));
        assert!(!BrillouinPath::Square.is_compatible_with(LatticeType::Rectangular));
        assert!(BrillouinPath::Rectangular.is_compatible_with(LatticeType::Rectangular));
        assert!(BrillouinPath::Custom(vec![]).is_compatible_with(LatticeType::Oblique));
    }

    #[test]
    fn for_lattice_type() {
        assert_eq!(
            BrillouinPath::for_lattice_type(LatticeType::Square),
            Some(BrillouinPath::Square)
        );
        assert_eq!(
            BrillouinPath::for_lattice_type(LatticeType::Rectangular),
            Some(BrillouinPath::Rectangular)
        );
        assert_eq!(BrillouinPath::for_lattice_type(LatticeType::Oblique), None);
    }

    #[test]
    fn path_distances_monotonic() {
        let lattice = BravaisLattice::square(1.0);
        let path = generate_path(&BrillouinPath::Square, 10);
        let distances = path_distances(&path, &lattice);

        // Distances should be monotonically increasing
        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }
}
