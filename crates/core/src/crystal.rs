//! Photonic crystal structure definition.
//!
//! This module provides a unified `PhotonicCrystal` type that combines:
//! - A Bravais lattice (the periodic structure)
//! - An atomic basis (the inclusions within each unit cell)
//! - Background dielectric constant
//!
//! This is the recommended entry point for defining photonic crystal geometries.
//!
//! # Example
//!
//! ```
//! use mpb2d_core::crystal::PhotonicCrystal;
//! use mpb2d_core::bravais::BravaisLattice;
//! use mpb2d_core::basis::{AtomicBasis, BasisAtom};
//!
//! // Create a square lattice photonic crystal with air holes in silicon
//! let crystal = PhotonicCrystal::builder()
//!     .lattice(BravaisLattice::square(1.0))
//!     .background_epsilon(11.9)  // Silicon
//!     .add_atom(BasisAtom::air_hole([0.0, 0.0], 0.3))
//!     .build();
//!
//! // Or use the convenience constructors
//! let square_holes = PhotonicCrystal::square_lattice_air_holes(11.9, 0.3);
//! let hex_holes = PhotonicCrystal::triangular_lattice_air_holes(13.0, 0.3);
//! ```

use serde::{Deserialize, Serialize};

use crate::basis::{AtomicBasis, BasisAtom};
use crate::bravais::{BravaisLattice, LatticeType};
use crate::brillouin::{BrillouinPath, PathPreset, generate_path};
use crate::geometry::Geometry2D;
use crate::lattice::Lattice2D;

// ============================================================================
// Photonic Crystal Structure
// ============================================================================

/// A complete 2D photonic crystal structure.
///
/// This combines a Bravais lattice with an atomic basis to fully specify
/// the geometry of a 2D photonic crystal.
///
/// # Components
///
/// - **Lattice**: The underlying 2D Bravais lattice (square, rectangular,
///   triangular, or oblique)
/// - **Basis**: The set of dielectric inclusions within each unit cell
/// - **Background ε**: The dielectric constant of the background material
///
/// # Coordinates
///
/// All positions are in fractional coordinates relative to the lattice vectors.
/// For a point r in the unit cell:
///
/// r_cart = f₁ · a₁ + f₂ · a₂
///
/// where 0 ≤ f₁, f₂ < 1 for the fundamental domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotonicCrystal {
    /// The Bravais lattice defining the periodic structure.
    lattice: BravaisLattice,
    /// The atomic basis (inclusions in each unit cell).
    basis: AtomicBasis,
    /// Background relative permittivity.
    #[serde(default = "default_epsilon")]
    epsilon_bg: f64,
}

fn default_epsilon() -> f64 {
    1.0
}

impl PhotonicCrystal {
    // ========================================================================
    // Construction
    // ========================================================================

    /// Create a new photonic crystal with the given components.
    pub fn new(lattice: BravaisLattice, basis: AtomicBasis, epsilon_bg: f64) -> Self {
        assert!(epsilon_bg > 0.0, "background permittivity must be positive");
        Self {
            lattice,
            basis,
            epsilon_bg,
        }
    }

    /// Create a builder for constructing a photonic crystal.
    pub fn builder() -> PhotonicCrystalBuilder {
        PhotonicCrystalBuilder::new()
    }

    /// Create a uniform dielectric (no inclusions).
    pub fn uniform(lattice: BravaisLattice, epsilon: f64) -> Self {
        Self::new(lattice, AtomicBasis::empty(), epsilon)
    }

    // ========================================================================
    // Convenience Constructors
    // ========================================================================

    /// Create a square lattice with a single centered air hole.
    ///
    /// This is the classic "air holes in dielectric" structure.
    ///
    /// # Arguments
    /// - `epsilon_bg`: Background dielectric constant
    /// - `radius`: Hole radius in units of lattice constant (r/a)
    pub fn square_lattice_air_holes(epsilon_bg: f64, radius: f64) -> Self {
        Self::new(
            BravaisLattice::square(1.0),
            AtomicBasis::single(BasisAtom::air_hole([0.0, 0.0], radius)),
            epsilon_bg,
        )
    }

    /// Create a triangular lattice with a single centered air hole.
    ///
    /// This is the classic hexagonal photonic crystal structure.
    ///
    /// # Arguments
    /// - `epsilon_bg`: Background dielectric constant
    /// - `radius`: Hole radius in units of lattice constant (r/a)
    pub fn triangular_lattice_air_holes(epsilon_bg: f64, radius: f64) -> Self {
        Self::new(
            BravaisLattice::triangular(1.0),
            AtomicBasis::single(BasisAtom::air_hole([0.0, 0.0], radius)),
            epsilon_bg,
        )
    }

    /// Alias for `triangular_lattice_air_holes`.
    pub fn hexagonal_lattice_air_holes(epsilon_bg: f64, radius: f64) -> Self {
        Self::triangular_lattice_air_holes(epsilon_bg, radius)
    }

    /// Create a rectangular lattice with a single centered air hole.
    ///
    /// # Arguments
    /// - `aspect_ratio`: Ratio b/a of lattice constants (a = 1)
    /// - `epsilon_bg`: Background dielectric constant
    /// - `radius`: Hole radius in units of lattice constant
    pub fn rectangular_lattice_air_holes(aspect_ratio: f64, epsilon_bg: f64, radius: f64) -> Self {
        Self::new(
            BravaisLattice::rectangular(aspect_ratio),
            AtomicBasis::single(BasisAtom::air_hole([0.0, 0.0], radius)),
            epsilon_bg,
        )
    }

    /// Create a square lattice with dielectric rods in air.
    ///
    /// This is the inverse of air holes - high-ε rods in low-ε background.
    ///
    /// # Arguments
    /// - `epsilon_rod`: Dielectric constant of the rods
    /// - `radius`: Rod radius in units of lattice constant
    pub fn square_lattice_rods(epsilon_rod: f64, radius: f64) -> Self {
        Self::new(
            BravaisLattice::square(1.0),
            AtomicBasis::single(BasisAtom::dielectric_rod([0.0, 0.0], radius, epsilon_rod)),
            1.0, // Air background
        )
    }

    /// Create a triangular lattice with dielectric rods in air.
    pub fn triangular_lattice_rods(epsilon_rod: f64, radius: f64) -> Self {
        Self::new(
            BravaisLattice::triangular(1.0),
            AtomicBasis::single(BasisAtom::dielectric_rod([0.0, 0.0], radius, epsilon_rod)),
            1.0,
        )
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Get the Bravais lattice.
    pub fn lattice(&self) -> &BravaisLattice {
        &self.lattice
    }

    /// Get the atomic basis.
    pub fn basis(&self) -> &AtomicBasis {
        &self.basis
    }

    /// Get the background permittivity.
    pub fn epsilon_bg(&self) -> f64 {
        self.epsilon_bg
    }

    /// Get the lattice type classification.
    pub fn lattice_type(&self) -> LatticeType {
        self.lattice.lattice_type()
    }

    // ========================================================================
    // High-Symmetry Paths
    // ========================================================================

    /// Get the recommended Brillouin zone path for this crystal.
    ///
    /// Returns `None` for oblique lattices which have no standard path.
    pub fn recommended_path(&self) -> Option<BrillouinPath> {
        BrillouinPath::for_lattice_type(self.lattice_type())
    }

    /// Get the recommended path preset.
    pub fn recommended_path_preset(&self) -> Option<PathPreset> {
        PathPreset::for_lattice_type(self.lattice_type())
    }

    /// Generate the k-path for band structure calculation.
    ///
    /// # Arguments
    /// - `segments_per_leg`: Number of k-points per segment
    ///
    /// # Returns
    /// - `Some(path)` with the densified k-path
    /// - `None` if no standard path exists (oblique lattice)
    pub fn generate_k_path(&self, segments_per_leg: usize) -> Option<Vec<[f64; 2]>> {
        self.recommended_path()
            .map(|p| generate_path(&p, segments_per_leg))
    }

    // ========================================================================
    // Dielectric Sampling
    // ========================================================================

    /// Get the relative permittivity at a fractional coordinate.
    ///
    /// Checks all atoms in the basis; returns background ε if outside all atoms.
    pub fn permittivity_at(&self, frac: [f64; 2]) -> f64 {
        for atom in self.basis.iter() {
            if atom.contains_point(frac, &self.lattice) {
                return atom.epsilon();
            }
        }
        self.epsilon_bg
    }

    // ========================================================================
    // Conversion to Legacy Types
    // ========================================================================

    /// Convert to the legacy `Geometry2D` type for backward compatibility.
    pub fn to_geometry(&self) -> Geometry2D {
        let legacy_lattice = Lattice2D::from(self.lattice);
        let legacy_atoms: Vec<crate::geometry::BasisAtom> = self
            .basis
            .iter()
            .map(|atom| crate::geometry::BasisAtom::from(atom.clone()))
            .collect();

        Geometry2D {
            lattice: legacy_lattice,
            eps_bg: self.epsilon_bg,
            atoms: legacy_atoms,
        }
    }
}

impl From<PhotonicCrystal> for Geometry2D {
    fn from(crystal: PhotonicCrystal) -> Self {
        crystal.to_geometry()
    }
}

impl From<Geometry2D> for PhotonicCrystal {
    fn from(geom: Geometry2D) -> Self {
        let lattice = BravaisLattice::from(geom.lattice);
        let basis: AtomicBasis = geom.atoms.into_iter().map(BasisAtom::from).collect();
        PhotonicCrystal::new(lattice, basis, geom.eps_bg)
    }
}

// ============================================================================
// Builder Pattern
// ============================================================================

/// Builder for constructing a `PhotonicCrystal`.
#[derive(Debug, Clone)]
pub struct PhotonicCrystalBuilder {
    lattice: Option<BravaisLattice>,
    atoms: Vec<BasisAtom>,
    epsilon_bg: f64,
}

impl PhotonicCrystalBuilder {
    /// Create a new builder with default values.
    pub fn new() -> Self {
        Self {
            lattice: None,
            atoms: Vec::new(),
            epsilon_bg: 1.0,
        }
    }

    /// Set the Bravais lattice.
    pub fn lattice(mut self, lattice: BravaisLattice) -> Self {
        self.lattice = Some(lattice);
        self
    }

    /// Set a square lattice with the given lattice constant.
    pub fn square(mut self, a: f64) -> Self {
        self.lattice = Some(BravaisLattice::square(a));
        self
    }

    /// Set a rectangular lattice with aspect ratio b (a = 1).
    pub fn rectangular(mut self, b: f64) -> Self {
        self.lattice = Some(BravaisLattice::rectangular(b));
        self
    }

    /// Set a triangular lattice with the given lattice constant.
    pub fn triangular(mut self, a: f64) -> Self {
        self.lattice = Some(BravaisLattice::triangular(a));
        self
    }

    /// Set an oblique lattice.
    pub fn oblique(mut self, b: f64, alpha: f64) -> Self {
        self.lattice = Some(BravaisLattice::oblique(b, alpha));
        self
    }

    /// Set the background permittivity.
    pub fn background_epsilon(mut self, epsilon: f64) -> Self {
        assert!(epsilon > 0.0, "permittivity must be positive");
        self.epsilon_bg = epsilon;
        self
    }

    /// Add an atom to the basis.
    pub fn add_atom(mut self, atom: BasisAtom) -> Self {
        self.atoms.push(atom);
        self
    }

    /// Add an air hole at the given position.
    pub fn add_air_hole(mut self, position: [f64; 2], radius: f64) -> Self {
        self.atoms.push(BasisAtom::air_hole(position, radius));
        self
    }

    /// Add a dielectric rod at the given position.
    pub fn add_rod(mut self, position: [f64; 2], radius: f64, epsilon: f64) -> Self {
        self.atoms
            .push(BasisAtom::dielectric_rod(position, radius, epsilon));
        self
    }

    /// Build the photonic crystal.
    ///
    /// # Panics
    /// Panics if no lattice has been set.
    pub fn build(self) -> PhotonicCrystal {
        let lattice = self
            .lattice
            .expect("lattice must be set before building PhotonicCrystal");
        PhotonicCrystal::new(lattice, AtomicBasis::new(self.atoms), self.epsilon_bg)
    }
}

impl Default for PhotonicCrystalBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_lattice_air_holes() {
        let crystal = PhotonicCrystal::square_lattice_air_holes(12.0, 0.3);
        assert_eq!(crystal.lattice_type(), LatticeType::Square);
        assert_eq!(crystal.epsilon_bg(), 12.0);
        assert_eq!(crystal.basis().len(), 1);
    }

    #[test]
    fn triangular_lattice_air_holes() {
        let crystal = PhotonicCrystal::triangular_lattice_air_holes(13.0, 0.3);
        assert_eq!(crystal.lattice_type(), LatticeType::Triangular);
        assert_eq!(crystal.epsilon_bg(), 13.0);
    }

    #[test]
    fn builder_pattern() {
        let crystal = PhotonicCrystal::builder()
            .square(1.0)
            .background_epsilon(10.0)
            .add_air_hole([0.0, 0.0], 0.2)
            .add_air_hole([0.5, 0.5], 0.15)
            .build();

        assert_eq!(crystal.lattice_type(), LatticeType::Square);
        assert_eq!(crystal.basis().len(), 2);
    }

    #[test]
    fn recommended_paths() {
        let square = PhotonicCrystal::square_lattice_air_holes(12.0, 0.3);
        assert!(square.recommended_path().is_some());
        assert!(matches!(
            square.recommended_path(),
            Some(BrillouinPath::Square)
        ));

        let hex = PhotonicCrystal::triangular_lattice_air_holes(12.0, 0.3);
        assert!(matches!(
            hex.recommended_path(),
            Some(BrillouinPath::Triangular)
        ));
    }

    #[test]
    fn geometry_roundtrip() {
        let crystal = PhotonicCrystal::square_lattice_air_holes(12.0, 0.3);
        let geom = crystal.to_geometry();
        let back = PhotonicCrystal::from(geom);

        assert_eq!(back.lattice_type(), crystal.lattice_type());
        assert!((back.epsilon_bg() - crystal.epsilon_bg()).abs() < 1e-10);
    }

    #[test]
    fn permittivity_sampling() {
        let crystal = PhotonicCrystal::square_lattice_air_holes(12.0, 0.3);

        // Inside the air hole (at origin)
        assert!((crystal.permittivity_at([0.0, 0.0]) - 1.0).abs() < 1e-10);

        // Point far from the hole at origin - need to pick a point that is
        // more than 0.3 away from the origin even with periodic wrapping
        // [0.5, 0.5] is at distance 0.5*sqrt(2) ≈ 0.707 from origin
        assert!((crystal.permittivity_at([0.5, 0.5]) - 12.0).abs() < 1e-10);
    }
}
