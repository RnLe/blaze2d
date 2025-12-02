//! Atomic basis for 2D photonic crystals.
//!
//! This module defines the atoms (inclusions) within the unit cell of a 2D
//! photonic crystal. Each atom is defined by its position within the unit cell
//! and its geometric/material properties.
//!
//! # Terminology
//!
//! - **Bravais lattice**: The underlying periodic lattice (square, rectangular, etc.)
//! - **Basis**: The set of atoms within each unit cell
//! - **Atom**: A single inclusion with a position and geometry
//!
//! In crystallography, a crystal structure = Bravais lattice + basis.
//! Similarly, a photonic crystal = Bravais lattice + atomic basis (dielectric inclusions).
//!
//! # Coordinates
//!
//! Atom positions are specified in **fractional coordinates** relative to the
//! unit cell. Fractional coordinates (f₁, f₂) map to Cartesian via:
//!
//! r_cart = f₁ · a₁ + f₂ · a₂
//!
//! where a₁, a₂ are the lattice vectors.
//!
//! For a position to be inside the unit cell: 0 ≤ f₁, f₂ < 1

use serde::{Deserialize, Serialize};

use crate::bravais::BravaisLattice;

// ============================================================================
// Atomic Geometry (Shape)
// ============================================================================

/// Geometry of an atom (inclusion) within the unit cell.
///
/// Currently only circular inclusions are supported, but this enum
/// is designed to be extended with other shapes in the future.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "lowercase")]
pub enum AtomGeometry {
    /// Circular inclusion with a given radius.
    Circle {
        /// Radius in fractional units (relative to lattice constant |a₁|).
        radius: f64,
    },
    // Future extensions:
    // Ellipse { a: f64, b: f64, angle: f64 },
    // Rectangle { width: f64, height: f64, angle: f64 },
    // Polygon { vertices: Vec<[f64; 2]> },
}

impl AtomGeometry {
    /// Create a circular geometry with the given radius.
    pub fn circle(radius: f64) -> Self {
        assert!(radius > 0.0, "radius must be positive");
        AtomGeometry::Circle { radius }
    }

    /// Get the radius if this is a circle.
    pub fn radius(&self) -> Option<f64> {
        match self {
            AtomGeometry::Circle { radius } => Some(*radius),
        }
    }

    /// Convert the fractional radius to Cartesian coordinates.
    ///
    /// For a circle, the radius is scaled by the lattice characteristic length.
    pub fn radius_cartesian(&self, lattice: &BravaisLattice) -> f64 {
        match self {
            AtomGeometry::Circle { radius } => radius * lattice.characteristic_length(),
        }
    }

    /// Check if a point (in fractional coordinates) is inside this geometry,
    /// given the atom's position.
    ///
    /// # Arguments
    /// - `point_frac`: Point to test in fractional coordinates
    /// - `center_frac`: Center of the atom in fractional coordinates  
    /// - `lattice`: The Bravais lattice for coordinate conversion
    pub fn contains_point(
        &self,
        point_frac: [f64; 2],
        center_frac: [f64; 2],
        lattice: &BravaisLattice,
    ) -> bool {
        match self {
            AtomGeometry::Circle { radius } => {
                // Compute distance in fractional coordinates, then convert to Cartesian
                let delta_frac = [
                    wrap_signed(point_frac[0] - center_frac[0]),
                    wrap_signed(point_frac[1] - center_frac[1]),
                ];
                let delta_cart = lattice.fractional_to_cartesian(delta_frac);
                let dist_sq = delta_cart[0] * delta_cart[0] + delta_cart[1] * delta_cart[1];
                let r_cart = radius * lattice.characteristic_length();
                dist_sq <= r_cart * r_cart
            }
        }
    }
}

impl Default for AtomGeometry {
    fn default() -> Self {
        AtomGeometry::Circle { radius: 0.3 }
    }
}

// ============================================================================
// Basis Atom
// ============================================================================

/// A single atom (inclusion) in the basis.
///
/// Each atom has a position within the unit cell and associated material/geometry.
/// The position is always in fractional coordinates.
///
/// # Examples
///
/// ```
/// use mpb2d_core::basis::{BasisAtom, AtomGeometry};
///
/// // Air hole at the center of the unit cell
/// let air_hole = BasisAtom::new([0.5, 0.5], AtomGeometry::circle(0.3), 1.0);
///
/// // Dielectric rod at the origin
/// let rod = BasisAtom::new([0.0, 0.0], AtomGeometry::circle(0.2), 12.0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasisAtom {
    /// Position in fractional coordinates [f₁, f₂] where 0 ≤ fᵢ < 1.
    position: [f64; 2],
    /// Geometry of the inclusion.
    geometry: AtomGeometry,
    /// Relative permittivity inside the atom.
    epsilon: f64,
}

impl BasisAtom {
    /// Create a new basis atom.
    ///
    /// # Arguments
    /// - `position`: Fractional coordinates within the unit cell
    /// - `geometry`: Shape of the inclusion
    /// - `epsilon`: Relative permittivity (dielectric constant) inside
    ///
    /// # Panics
    /// Panics if epsilon ≤ 0.
    pub fn new(position: [f64; 2], geometry: AtomGeometry, epsilon: f64) -> Self {
        assert!(epsilon > 0.0, "permittivity must be positive");
        Self {
            position: wrap_to_unit_cell(position),
            geometry,
            epsilon,
        }
    }

    /// Create an air hole (ε = 1) at the given position.
    pub fn air_hole(position: [f64; 2], radius: f64) -> Self {
        Self::new(position, AtomGeometry::circle(radius), 1.0)
    }

    /// Create a dielectric rod at the given position.
    pub fn dielectric_rod(position: [f64; 2], radius: f64, epsilon: f64) -> Self {
        Self::new(position, AtomGeometry::circle(radius), epsilon)
    }

    /// Get the position in fractional coordinates.
    #[inline]
    pub fn position(&self) -> [f64; 2] {
        self.position
    }

    /// Get the geometry.
    #[inline]
    pub fn geometry(&self) -> &AtomGeometry {
        &self.geometry
    }

    /// Get the relative permittivity.
    #[inline]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Get the radius in Cartesian coordinates (for circular atoms).
    pub fn radius_cartesian(&self, lattice: &BravaisLattice) -> f64 {
        self.geometry.radius_cartesian(lattice)
    }

    /// Check if a point is inside this atom.
    pub fn contains_point(&self, point_frac: [f64; 2], lattice: &BravaisLattice) -> bool {
        self.geometry
            .contains_point(point_frac, self.position, lattice)
    }

    /// Set a new position (automatically wraps to unit cell).
    pub fn with_position(mut self, position: [f64; 2]) -> Self {
        self.position = wrap_to_unit_cell(position);
        self
    }

    /// Set a new epsilon value.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        assert!(epsilon > 0.0, "permittivity must be positive");
        self.epsilon = epsilon;
        self
    }
}

// ============================================================================
// Atomic Basis (Collection)
// ============================================================================

/// A collection of atoms forming the basis of the crystal structure.
///
/// The basis defines all the atoms within a single unit cell. Together with
/// the Bravais lattice, this fully specifies the photonic crystal geometry.
///
/// # Invariants
///
/// - All atom positions are within the unit cell (0 ≤ f < 1)
/// - Epsilon values are positive
///
/// # Examples
///
/// ```
/// use mpb2d_core::basis::{AtomicBasis, BasisAtom, AtomGeometry};
///
/// // Single-atom basis (e.g., simple air hole lattice)
/// let single = AtomicBasis::single(BasisAtom::air_hole([0.0, 0.0], 0.3));
///
/// // Two-atom basis (e.g., honeycomb structure)
/// let honeycomb = AtomicBasis::new(vec![
///     BasisAtom::air_hole([1.0/3.0, 1.0/3.0], 0.15),
///     BasisAtom::air_hole([2.0/3.0, 2.0/3.0], 0.15),
/// ]);
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AtomicBasis {
    /// The atoms in the basis.
    atoms: Vec<BasisAtom>,
}

impl AtomicBasis {
    /// Create an empty basis (no atoms = uniform dielectric).
    pub fn empty() -> Self {
        Self { atoms: Vec::new() }
    }

    /// Create a basis with a single atom.
    pub fn single(atom: BasisAtom) -> Self {
        Self { atoms: vec![atom] }
    }

    /// Create a basis from a vector of atoms.
    pub fn new(atoms: Vec<BasisAtom>) -> Self {
        Self { atoms }
    }

    /// Create a single air hole centered at the origin.
    pub fn centered_air_hole(radius: f64) -> Self {
        Self::single(BasisAtom::air_hole([0.0, 0.0], radius))
    }

    /// Check if the basis is empty (no atoms).
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }

    /// Get the number of atoms in the basis.
    pub fn len(&self) -> usize {
        self.atoms.len()
    }

    /// Get an iterator over the atoms.
    pub fn iter(&self) -> impl Iterator<Item = &BasisAtom> {
        self.atoms.iter()
    }

    /// Get a mutable iterator over the atoms.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut BasisAtom> {
        self.atoms.iter_mut()
    }

    /// Add an atom to the basis.
    pub fn push(&mut self, atom: BasisAtom) {
        self.atoms.push(atom);
    }

    /// Get the atoms as a slice.
    pub fn atoms(&self) -> &[BasisAtom] {
        &self.atoms
    }
}

impl IntoIterator for AtomicBasis {
    type Item = BasisAtom;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.atoms.into_iter()
    }
}

impl<'a> IntoIterator for &'a AtomicBasis {
    type Item = &'a BasisAtom;
    type IntoIter = std::slice::Iter<'a, BasisAtom>;

    fn into_iter(self) -> Self::IntoIter {
        self.atoms.iter()
    }
}

impl FromIterator<BasisAtom> for AtomicBasis {
    fn from_iter<I: IntoIterator<Item = BasisAtom>>(iter: I) -> Self {
        Self {
            atoms: iter.into_iter().collect(),
        }
    }
}

// ============================================================================
// Offset Specification
// ============================================================================

/// Specification for an atom's position within the unit cell.
///
/// This enum provides two ways to specify atom positions:
/// 1. As an offset from the Bravais lattice point (the origin)
/// 2. As an absolute fractional coordinate
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PositionSpec {
    /// Offset from the Bravais lattice point [Δf₁, Δf₂].
    /// Final position = offset (wrapped to unit cell).
    Offset([f64; 2]),
    // Future: could add named positions like `Center`, `Corner`, etc.
}

impl PositionSpec {
    /// Resolve the position to fractional coordinates.
    pub fn to_fractional(&self) -> [f64; 2] {
        match self {
            PositionSpec::Offset(offset) => wrap_to_unit_cell(*offset),
        }
    }
}

impl From<[f64; 2]> for PositionSpec {
    fn from(offset: [f64; 2]) -> Self {
        PositionSpec::Offset(offset)
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Wrap a fractional coordinate to [0, 1).
fn wrap_unit(value: f64) -> f64 {
    value - value.floor()
}

/// Wrap fractional coordinates to the unit cell [0, 1)².
fn wrap_to_unit_cell(frac: [f64; 2]) -> [f64; 2] {
    [wrap_unit(frac[0]), wrap_unit(frac[1])]
}

/// Wrap a fractional delta to [-0.5, 0.5) for minimum image convention.
fn wrap_signed(mut delta: f64) -> f64 {
    while delta >= 0.5 {
        delta -= 1.0;
    }
    while delta < -0.5 {
        delta += 1.0;
    }
    delta
}

// ============================================================================
// Conversion from Legacy BasisAtom
// ============================================================================

impl From<crate::geometry::BasisAtom> for BasisAtom {
    fn from(old: crate::geometry::BasisAtom) -> Self {
        Self {
            position: old.pos,
            geometry: AtomGeometry::Circle { radius: old.radius },
            epsilon: old.eps_inside,
        }
    }
}

impl From<BasisAtom> for crate::geometry::BasisAtom {
    fn from(new: BasisAtom) -> Self {
        crate::geometry::BasisAtom {
            pos: new.position,
            radius: new.geometry.radius().unwrap_or(0.0),
            eps_inside: new.epsilon,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bravais::BravaisLattice;

    #[test]
    fn atom_construction() {
        let atom = BasisAtom::air_hole([0.5, 0.5], 0.3);
        assert_eq!(atom.position(), [0.5, 0.5]);
        assert_eq!(atom.epsilon(), 1.0);
        assert_eq!(atom.geometry().radius(), Some(0.3));
    }

    #[test]
    fn position_wrapping() {
        let atom = BasisAtom::air_hole([1.5, -0.3], 0.1);
        assert!((atom.position()[0] - 0.5).abs() < 1e-10);
        assert!((atom.position()[1] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn contains_point() {
        let lattice = BravaisLattice::square(1.0);
        let atom = BasisAtom::air_hole([0.5, 0.5], 0.2);

        // Center should be inside
        assert!(atom.contains_point([0.5, 0.5], &lattice));

        // Point just inside radius
        assert!(atom.contains_point([0.5 + 0.1, 0.5], &lattice));

        // Point outside
        assert!(!atom.contains_point([0.5 + 0.3, 0.5], &lattice));
    }

    #[test]
    fn atomic_basis_operations() {
        let mut basis = AtomicBasis::empty();
        assert!(basis.is_empty());

        basis.push(BasisAtom::air_hole([0.0, 0.0], 0.2));
        basis.push(BasisAtom::air_hole([0.5, 0.5], 0.2));

        assert_eq!(basis.len(), 2);
        assert!(!basis.is_empty());
    }

    #[test]
    fn conversion_roundtrip() {
        let old = crate::geometry::BasisAtom {
            pos: [0.3, 0.7],
            radius: 0.25,
            eps_inside: 2.5,
        };

        let new: BasisAtom = old.clone().into();
        let back: crate::geometry::BasisAtom = new.into();

        assert!((back.pos[0] - old.pos[0]).abs() < 1e-10);
        assert!((back.pos[1] - old.pos[1]).abs() < 1e-10);
        assert!((back.radius - old.radius).abs() < 1e-10);
        assert!((back.eps_inside - old.eps_inside).abs() < 1e-10);
    }
}
