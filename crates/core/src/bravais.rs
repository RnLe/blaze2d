//! Bravais lattice primitives for 2D photonic crystals.
//!
//! This module provides strongly-typed Bravais lattice definitions with clear
//! semantic separation between lattice type classification and the underlying
//! real-space basis vectors.
//!
//! # 2D Bravais Lattice Types
//!
//! In 2D, there are exactly 5 Bravais lattice types. We support 4 of them
//! (rhombic/centered-rectangular is not included):
//!
//! | Type        | Parameters                      | Symmetry | Example               |
//! |-------------|--------------------------------|----------|----------------------|
//! | Square      | a (lattice constant)            | C₄ᵥ      | Checkerboard pattern |
//! | Rectangular | a, b (a ≠ b)                    | C₂ᵥ      | Brick pattern        |
//! | Triangular  | a (lattice constant)            | C₆ᵥ      | Hexagonal honeycomb  |
//! | Oblique     | a, b, α (angle between a₁, a₂) | C₂       | General lattice      |
//!
//! # Conventions (Standard Crystallographic)
//!
//! All lattice types follow standard crystallographic conventions:
//!
//! - First basis vector a₁ always points along the x-axis: a₁ = [a, 0]
//! - Angles are measured counterclockwise from a₁ to a₂
//!
//! ## Primitive Vectors
//!
//! | Type        | a₁           | a₂                        | Angle |
//! |-------------|--------------|---------------------------|-------|
//! | Square      | [a, 0]       | [0, a]                    | 90°   |
//! | Rectangular | [a, 0]       | [0, b]                    | 90°   |
//! | Triangular  | [a, 0]       | [a/2, a√3/2]              | 60°   |
//! | Oblique     | [a, 0]       | [b·cos(α), b·sin(α)]      | α     |
//!
//! The triangular/hexagonal lattice uses the **60° convention** matching
//! Setyawan-Curtarolo, SeeK-path, and MPB's Python interface.
//!
//! ## High-Symmetry Points (Triangular/Hexagonal)
//!
//! With the 60° convention, the Brillouin zone high-symmetry points are:
//! - Γ = (0, 0)
//! - M = (1/2, 0)
//! - K = (1/3, 1/3)

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

// ============================================================================
// Lattice Type Classification
// ============================================================================

/// Classification of 2D Bravais lattice types.
///
/// This enum represents the crystallographic classification of the lattice,
/// which determines its point group symmetry and available high-symmetry paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LatticeType {
    /// Square lattice with C₄ᵥ point group symmetry.
    ///
    /// - Both basis vectors have equal length: |a₁| = |a₂| = a
    /// - Vectors are orthogonal: a₁ ⊥ a₂ (90° angle)
    /// - High-symmetry path: Γ → X → M → Γ
    Square,

    /// Rectangular lattice with C₂ᵥ point group symmetry.
    ///
    /// - Basis vectors have different lengths: |a₁| = a, |a₂| = b, a ≠ b
    /// - Vectors are orthogonal: a₁ ⊥ a₂ (90° angle)
    /// - High-symmetry path: Γ → X → S → Y → Γ
    Rectangular,

    /// Triangular (hexagonal) lattice with C₆ᵥ point group symmetry.
    ///
    /// - Both basis vectors have equal length: |a₁| = |a₂| = a
    /// - Vectors at 60° angle: a₁ = [a, 0], a₂ = [a/2, a√3/2]
    /// - High-symmetry path: Γ → M → K → Γ
    /// - High-symmetry points: M = (1/2, 0), K = (1/3, 1/3)
    Triangular,

    /// Oblique lattice with C₂ point group symmetry.
    ///
    /// - General lattice with arbitrary lengths and angle
    /// - No mirror symmetry, only 2-fold rotation
    /// - No standard high-symmetry path (must specify explicitly)
    Oblique,
}

impl LatticeType {
    /// Returns true if this lattice type has orthogonal basis vectors.
    pub fn is_orthogonal(&self) -> bool {
        matches!(self, LatticeType::Square | LatticeType::Rectangular)
    }

    /// Returns true if this lattice type has equal-length basis vectors.
    pub fn is_equilateral(&self) -> bool {
        matches!(self, LatticeType::Square | LatticeType::Triangular)
    }

    /// Returns the point group order (number of symmetry operations).
    pub fn point_group_order(&self) -> usize {
        match self {
            LatticeType::Square => 8,      // C₄ᵥ
            LatticeType::Rectangular => 4, // C₂ᵥ
            LatticeType::Triangular => 12, // C₆ᵥ
            LatticeType::Oblique => 2,     // C₂
        }
    }

    /// Returns the conventional angle (in radians) between basis vectors.
    ///
    /// For triangular/hexagonal lattices, returns 60° (π/3) matching the
    /// standard crystallographic convention: a₁ = [a, 0], a₂ = [a/2, a√3/2].
    pub fn conventional_angle(&self) -> Option<f64> {
        match self {
            LatticeType::Square => Some(PI / 2.0),      // 90°
            LatticeType::Rectangular => Some(PI / 2.0), // 90°
            LatticeType::Triangular => Some(PI / 3.0),  // 60°
            LatticeType::Oblique => None,               // Arbitrary
        }
    }
}

impl std::fmt::Display for LatticeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LatticeType::Square => write!(f, "square"),
            LatticeType::Rectangular => write!(f, "rectangular"),
            LatticeType::Triangular => write!(f, "triangular"),
            LatticeType::Oblique => write!(f, "oblique"),
        }
    }
}

// ============================================================================
// Lattice Parameters
// ============================================================================

/// Parameters for constructing a 2D Bravais lattice.
///
/// This enum encapsulates the minimal set of parameters needed to construct
/// each lattice type. Parameters are validated at construction time.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum LatticeParams {
    /// Square lattice: requires only the lattice constant (default: a = 1).
    Square {
        /// Lattice constant (length of both basis vectors).
        #[serde(default = "default_lattice_constant")]
        a: f64,
    },

    /// Rectangular lattice: requires a = 1 (fixed) and aspect ratio b.
    Rectangular {
        /// Second lattice constant (a is fixed at 1).
        b: f64,
    },

    /// Triangular lattice: requires only the lattice constant (default: a = 1).
    Triangular {
        /// Lattice constant (length of both basis vectors).
        #[serde(default = "default_lattice_constant")]
        a: f64,
    },

    /// Oblique lattice: requires b and angle α (a is fixed at 1).
    Oblique {
        /// Second lattice constant (a is fixed at 1).
        b: f64,
        /// Angle between a₁ and a₂ in radians (must be in (0, π)).
        alpha: f64,
    },

    /// Custom lattice specified by explicit basis vectors.
    ///
    /// Use this for advanced cases where the standard constructors don't apply.
    /// The lattice type will be inferred from the vectors.
    #[serde(rename = "custom")]
    Custom {
        /// First basis vector in Cartesian coordinates.
        a1: [f64; 2],
        /// Second basis vector in Cartesian coordinates.
        a2: [f64; 2],
    },
}

fn default_lattice_constant() -> f64 {
    1.0
}

impl LatticeParams {
    /// Get the lattice type for these parameters.
    pub fn lattice_type(&self) -> LatticeType {
        match self {
            LatticeParams::Square { .. } => LatticeType::Square,
            LatticeParams::Rectangular { .. } => LatticeType::Rectangular,
            LatticeParams::Triangular { .. } => LatticeType::Triangular,
            LatticeParams::Oblique { .. } => LatticeType::Oblique,
            LatticeParams::Custom { a1, a2 } => {
                // Infer type from vectors
                infer_lattice_type(*a1, *a2, CLASSIFICATION_TOL)
            }
        }
    }
}

// ============================================================================
// Bravais Lattice
// ============================================================================

/// A 2D Bravais lattice with strongly-typed construction.
///
/// This struct represents a 2D periodic lattice defined by two primitive
/// translation vectors a₁ and a₂. The lattice type is tracked alongside
/// the vectors for efficient high-symmetry path selection.
///
/// # Construction
///
/// Use one of the type-safe constructors:
/// - [`BravaisLattice::square()`] - Square lattice
/// - [`BravaisLattice::rectangular()`] - Rectangular lattice
/// - [`BravaisLattice::triangular()`] - Triangular/hexagonal lattice
/// - [`BravaisLattice::oblique()`] - General oblique lattice
/// - [`BravaisLattice::from_params()`] - From parameter enum
/// - [`BravaisLattice::from_vectors()`] - From explicit vectors (type inferred)
///
/// # Examples
///
/// ```
/// use mpb2d_core::bravais::BravaisLattice;
///
/// // Simple square lattice with a = 1
/// let square = BravaisLattice::square(1.0);
/// assert_eq!(square.a1(), [1.0, 0.0]);
/// assert_eq!(square.a2(), [0.0, 1.0]);
///
/// // Rectangular lattice with a = 1, b = 2
/// let rect = BravaisLattice::rectangular(2.0);
/// assert_eq!(rect.a1(), [1.0, 0.0]);
/// assert_eq!(rect.a2(), [0.0, 2.0]);
///
/// // Triangular lattice with a = 1 (60° convention)
/// let tri = BravaisLattice::triangular(1.0);
/// // a₁ = [1, 0], a₂ = [0.5, √3/2] at 60° from a₁
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BravaisLattice {
    /// First primitive translation vector (always along x-axis for standard types).
    a1: [f64; 2],
    /// Second primitive translation vector.
    a2: [f64; 2],
    /// Cached lattice type classification.
    #[serde(skip)]
    lattice_type: Option<LatticeType>,
}

// Classification tolerance for determining lattice type
const CLASSIFICATION_TOL: f64 = 1e-6;

impl BravaisLattice {
    // ========================================================================
    // Standard Constructors
    // ========================================================================

    /// Create a square lattice with the given lattice constant.
    ///
    /// The basis vectors are:
    /// - a₁ = [a, 0]
    /// - a₂ = [0, a]
    ///
    /// # Arguments
    /// - `a`: Lattice constant (default: 1.0)
    ///
    /// # Panics
    /// Panics if `a <= 0`.
    pub fn square(a: f64) -> Self {
        assert!(a > 0.0, "lattice constant must be positive");
        Self {
            a1: [a, 0.0],
            a2: [0.0, a],
            lattice_type: Some(LatticeType::Square),
        }
    }

    /// Create a rectangular lattice with a = 1 and the given b parameter.
    ///
    /// The basis vectors are:
    /// - a₁ = [1, 0]
    /// - a₂ = [0, b]
    ///
    /// # Arguments
    /// - `b`: Second lattice constant (must be positive and ≠ 1)
    ///
    /// # Panics
    /// Panics if `b <= 0` or if `b ≈ 1` (use `square()` instead).
    pub fn rectangular(b: f64) -> Self {
        assert!(b > 0.0, "lattice constant b must be positive");
        // Allow b = 1 but it will be classified as square
        Self {
            a1: [1.0, 0.0],
            a2: [0.0, b],
            lattice_type: Some(if (b - 1.0).abs() < CLASSIFICATION_TOL {
                LatticeType::Square
            } else {
                LatticeType::Rectangular
            }),
        }
    }

    /// Create a rectangular lattice with explicit a and b parameters.
    ///
    /// The basis vectors are:
    /// - a₁ = [a, 0]
    /// - a₂ = [0, b]
    ///
    /// This preserves backward compatibility with the old `Lattice2D::rectangular(a, b)`.
    pub fn rectangular_ab(a: f64, b: f64) -> Self {
        assert!(a > 0.0, "lattice constant a must be positive");
        assert!(b > 0.0, "lattice constant b must be positive");
        Self {
            a1: [a, 0.0],
            a2: [0.0, b],
            lattice_type: Some(if (a - b).abs() < CLASSIFICATION_TOL * a.max(b) {
                LatticeType::Square
            } else {
                LatticeType::Rectangular
            }),
        }
    }

    /// Create a triangular (hexagonal) lattice with the given lattice constant.
    ///
    /// The basis vectors use the standard 60° convention:
    /// - a₁ = [a, 0]
    /// - a₂ = [a/2, a√3/2]  (at 60° from a₁)
    ///
    /// This corresponds to the standard crystallographic convention used by
    /// DFT databases (Setyawan-Curtarolo, SeeK-path) and matches MPB's Python
    /// interface. The high-symmetry points in this convention are:
    /// - Γ = (0, 0)
    /// - M = (1/2, 0)
    /// - K = (1/3, 1/3)
    ///
    /// # Arguments
    /// - `a`: Lattice constant (default: 1.0)
    ///
    /// # Panics
    /// Panics if `a <= 0`.
    pub fn triangular(a: f64) -> Self {
        assert!(a > 0.0, "lattice constant must be positive");
        let half = 0.5 * a;
        let height = (3.0_f64).sqrt() * 0.5 * a;
        Self {
            a1: [a, 0.0],
            a2: [half, height],  // 60° convention: a₂ = [a/2, a√3/2]
            lattice_type: Some(LatticeType::Triangular),
        }
    }

    /// Alias for `triangular()` - creates a hexagonal lattice.
    ///
    /// In 2D, "hexagonal" and "triangular" refer to the same Bravais lattice.
    /// The name reflects the 6-fold rotational symmetry of the lattice.
    pub fn hexagonal(a: f64) -> Self {
        Self::triangular(a)
    }

    /// Create an oblique lattice with a = 1, given b and angle α.
    ///
    /// The basis vectors are:
    /// - a₁ = [1, 0]
    /// - a₂ = [b·cos(α), b·sin(α)]
    ///
    /// # Arguments
    /// - `b`: Length of the second basis vector
    /// - `alpha`: Angle from a₁ to a₂ in radians (must be in (0, π))
    ///
    /// # Panics
    /// Panics if `b <= 0` or if `alpha` is not in (0, π).
    pub fn oblique(b: f64, alpha: f64) -> Self {
        assert!(b > 0.0, "lattice constant b must be positive");
        assert!(
            alpha > 0.0 && alpha < PI,
            "angle alpha must be in (0, π) radians"
        );
        let a2 = [b * alpha.cos(), b * alpha.sin()];
        Self {
            a1: [1.0, 0.0],
            a2,
            lattice_type: Some(LatticeType::Oblique),
        }
    }

    /// Create a lattice from explicit basis vectors.
    ///
    /// The lattice type is automatically inferred from the vectors.
    /// Use this for backward compatibility or advanced use cases.
    ///
    /// # Arguments
    /// - `a1`: First basis vector
    /// - `a2`: Second basis vector
    ///
    /// # Panics
    /// Panics if the vectors are linearly dependent.
    pub fn from_vectors(a1: [f64; 2], a2: [f64; 2]) -> Self {
        let det = a1[0] * a2[1] - a1[1] * a2[0];
        assert!(
            det.abs() > f64::EPSILON,
            "basis vectors must be linearly independent"
        );
        let lattice_type = infer_lattice_type(a1, a2, CLASSIFICATION_TOL);
        Self {
            a1,
            a2,
            lattice_type: Some(lattice_type),
        }
    }

    /// Create a lattice from a parameters enum.
    ///
    /// # Panics
    /// Panics if the parameters are invalid.
    pub fn from_params(params: LatticeParams) -> Self {
        match params {
            LatticeParams::Square { a } => Self::square(a),
            LatticeParams::Rectangular { b } => Self::rectangular(b),
            LatticeParams::Triangular { a } => Self::triangular(a),
            LatticeParams::Oblique { b, alpha } => Self::oblique(b, alpha),
            LatticeParams::Custom { a1, a2 } => Self::from_vectors(a1, a2),
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Get the first basis vector.
    #[inline]
    pub fn a1(&self) -> [f64; 2] {
        self.a1
    }

    /// Get the second basis vector.
    #[inline]
    pub fn a2(&self) -> [f64; 2] {
        self.a2
    }

    /// Get the lattice type classification.
    pub fn lattice_type(&self) -> LatticeType {
        self.lattice_type
            .unwrap_or_else(|| infer_lattice_type(self.a1, self.a2, CLASSIFICATION_TOL))
    }

    /// Get the determinant of the lattice matrix (= area of unit cell).
    #[inline]
    pub fn determinant(&self) -> f64 {
        self.a1[0] * self.a2[1] - self.a1[1] * self.a2[0]
    }

    /// Get the area of the unit cell.
    #[inline]
    pub fn cell_area(&self) -> f64 {
        self.determinant().abs()
    }

    /// Get the characteristic length (|a₁|) used for scaling.
    pub fn characteristic_length(&self) -> f64 {
        (self.a1[0] * self.a1[0] + self.a1[1] * self.a1[1]).sqrt()
    }

    /// Get the length of the first basis vector.
    pub fn len_a1(&self) -> f64 {
        (self.a1[0] * self.a1[0] + self.a1[1] * self.a1[1]).sqrt()
    }

    /// Get the length of the second basis vector.
    pub fn len_a2(&self) -> f64 {
        (self.a2[0] * self.a2[0] + self.a2[1] * self.a2[1]).sqrt()
    }

    /// Get the angle between basis vectors (in radians).
    pub fn angle(&self) -> f64 {
        let dot = self.a1[0] * self.a2[0] + self.a1[1] * self.a2[1];
        let cos_angle = dot / (self.len_a1() * self.len_a2());
        cos_angle.clamp(-1.0, 1.0).acos()
    }

    // ========================================================================
    // Coordinate Transformations
    // ========================================================================

    /// Convert fractional coordinates to Cartesian coordinates.
    ///
    /// r_cart = f₁ · a₁ + f₂ · a₂
    #[inline]
    pub fn fractional_to_cartesian(&self, frac: [f64; 2]) -> [f64; 2] {
        [
            self.a1[0] * frac[0] + self.a2[0] * frac[1],
            self.a1[1] * frac[0] + self.a2[1] * frac[1],
        ]
    }

    /// Convert Cartesian coordinates to fractional coordinates.
    ///
    /// Solves r_cart = f₁ · a₁ + f₂ · a₂ for (f₁, f₂).
    #[inline]
    pub fn cartesian_to_fractional(&self, cart: [f64; 2]) -> [f64; 2] {
        let det = self.determinant();
        assert!(
            det.abs() > f64::EPSILON,
            "basis vectors are linearly dependent"
        );
        let inv_det = 1.0 / det;
        [
            (self.a2[1] * cart[0] - self.a2[0] * cart[1]) * inv_det,
            (-self.a1[1] * cart[0] + self.a1[0] * cart[1]) * inv_det,
        ]
    }

    // ========================================================================
    // Reciprocal Lattice
    // ========================================================================

    /// Compute the reciprocal lattice.
    ///
    /// The reciprocal lattice vectors satisfy:
    /// - a_i · b_j = 2π δ_ij
    ///
    /// For a 2D lattice with A = [a₁ | a₂], the reciprocal is:
    /// - b₁ = 2π (a₂^⊥) / det(A)
    /// - b₂ = 2π (-a₁^⊥) / det(A)
    ///
    /// where v^⊥ = [v_y, -v_x] is the 90° rotation.
    pub fn reciprocal(&self) -> ReciprocalLattice {
        let det = self.determinant();
        assert!(
            det.abs() > f64::EPSILON,
            "basis vectors are linearly dependent"
        );
        let inv = 2.0 * PI / det;
        ReciprocalLattice {
            b1: [self.a2[1] * inv, -self.a2[0] * inv],
            b2: [-self.a1[1] * inv, self.a1[0] * inv],
        }
    }
}

impl PartialEq for BravaisLattice {
    fn eq(&self, other: &Self) -> bool {
        const TOL: f64 = 1e-10;
        (self.a1[0] - other.a1[0]).abs() < TOL
            && (self.a1[1] - other.a1[1]).abs() < TOL
            && (self.a2[0] - other.a2[0]).abs() < TOL
            && (self.a2[1] - other.a2[1]).abs() < TOL
    }
}

// ============================================================================
// Reciprocal Lattice
// ============================================================================

/// A 2D reciprocal lattice.
///
/// The reciprocal lattice is the Fourier dual of the real-space Bravais lattice.
/// G-vectors are of the form G = n₁·b₁ + n₂·b₂ where n₁, n₂ are integers.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ReciprocalLattice {
    /// First reciprocal lattice vector.
    pub b1: [f64; 2],
    /// Second reciprocal lattice vector.
    pub b2: [f64; 2],
}

impl ReciprocalLattice {
    /// Convert fractional k-space coordinates to Cartesian Bloch wavevector.
    ///
    /// k_cart = k₁ · b₁ + k₂ · b₂
    ///
    /// # Important
    ///
    /// For non-orthogonal lattices, this is NOT simply [2π·k₁, 2π·k₂].
    /// The reciprocal lattice vectors must be used to get the correct result.
    #[inline]
    pub fn fractional_to_cartesian(&self, k_frac: [f64; 2]) -> [f64; 2] {
        [
            k_frac[0] * self.b1[0] + k_frac[1] * self.b2[0],
            k_frac[0] * self.b1[1] + k_frac[1] * self.b2[1],
        ]
    }
}

// ============================================================================
// Lattice Type Inference
// ============================================================================

/// Infer the lattice type from basis vectors.
fn infer_lattice_type(a1: [f64; 2], a2: [f64; 2], tol: f64) -> LatticeType {
    let len1 = (a1[0] * a1[0] + a1[1] * a1[1]).sqrt();
    let len2 = (a2[0] * a2[0] + a2[1] * a2[1]).sqrt();

    if len1 <= f64::EPSILON || len2 <= f64::EPSILON {
        return LatticeType::Oblique;
    }

    let dot = a1[0] * a2[0] + a1[1] * a2[1];
    let cos_angle = (dot / (len1 * len2)).clamp(-1.0, 1.0);
    let len_diff = (len1 - len2).abs() / len1.max(len2);
    let is_orthogonal = cos_angle.abs() <= tol;
    let is_equilateral = len_diff <= tol;

    if is_equilateral && is_orthogonal {
        return LatticeType::Square;
    }
    if is_orthogonal {
        return LatticeType::Rectangular;
    }
    // Triangular: equal lengths and 60° angle (cos = 0.5)
    // Note: cos = -0.5 (120°) also accepted for backward compatibility
    if is_equilateral && ((cos_angle - 0.5).abs() <= tol || (cos_angle + 0.5).abs() <= tol) {
        return LatticeType::Triangular;
    }
    LatticeType::Oblique
}

// ============================================================================
// Conversion from Legacy Lattice2D
// ============================================================================

impl From<crate::lattice::Lattice2D> for BravaisLattice {
    fn from(old: crate::lattice::Lattice2D) -> Self {
        Self::from_vectors(old.a1, old.a2)
    }
}

impl From<BravaisLattice> for crate::lattice::Lattice2D {
    fn from(new: BravaisLattice) -> Self {
        crate::lattice::Lattice2D {
            a1: new.a1,
            a2: new.a2,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TAU: f64 = 2.0 * PI;

    #[test]
    fn square_lattice_construction() {
        let lat = BravaisLattice::square(1.0);
        assert_eq!(lat.a1(), [1.0, 0.0]);
        assert_eq!(lat.a2(), [0.0, 1.0]);
        assert_eq!(lat.lattice_type(), LatticeType::Square);
    }

    #[test]
    fn rectangular_lattice_construction() {
        let lat = BravaisLattice::rectangular(2.0);
        assert_eq!(lat.a1(), [1.0, 0.0]);
        assert_eq!(lat.a2(), [0.0, 2.0]);
        assert_eq!(lat.lattice_type(), LatticeType::Rectangular);
    }

    #[test]
    fn triangular_lattice_construction() {
        let lat = BravaisLattice::triangular(1.0);
        assert_eq!(lat.a1(), [1.0, 0.0]);
        // 60° convention: a₂ = [0.5, √3/2]
        assert!((lat.a2()[0] - 0.5).abs() < 1e-10);
        assert!((lat.a2()[1] - (3.0_f64.sqrt() / 2.0)).abs() < 1e-10);
        assert_eq!(lat.lattice_type(), LatticeType::Triangular);
    }

    #[test]
    fn oblique_lattice_construction() {
        let lat = BravaisLattice::oblique(1.5, PI / 3.0); // 60°
        assert_eq!(lat.a1(), [1.0, 0.0]);
        assert!((lat.a2()[0] - 0.75).abs() < 1e-10); // 1.5 * cos(60°)
        assert!((lat.a2()[1] - 1.5 * (PI / 3.0).sin()).abs() < 1e-10);
        assert_eq!(lat.lattice_type(), LatticeType::Oblique);
    }

    #[test]
    fn reciprocal_of_square_lattice() {
        let lat = BravaisLattice::square(1.0);
        let recip = lat.reciprocal();
        assert!((recip.b1[0] - TAU).abs() < 1e-10);
        assert!(recip.b1[1].abs() < 1e-10);
        assert!(recip.b2[0].abs() < 1e-10);
        assert!((recip.b2[1] - TAU).abs() < 1e-10);
    }

    #[test]
    fn coordinate_roundtrip() {
        let lat = BravaisLattice::triangular(1.0);
        let frac = [0.3, 0.7];
        let cart = lat.fractional_to_cartesian(frac);
        let back = lat.cartesian_to_fractional(cart);
        assert!((back[0] - frac[0]).abs() < 1e-10);
        assert!((back[1] - frac[1]).abs() < 1e-10);
    }

    #[test]
    fn from_params_square() {
        let lat = BravaisLattice::from_params(LatticeParams::Square { a: 2.0 });
        assert_eq!(lat.lattice_type(), LatticeType::Square);
        assert_eq!(lat.a1(), [2.0, 0.0]);
    }

    #[test]
    fn type_inference_from_vectors() {
        // Square-like vectors
        let square = BravaisLattice::from_vectors([1.0, 0.0], [0.0, 1.0]);
        assert_eq!(square.lattice_type(), LatticeType::Square);

        // Rectangular-like vectors
        let rect = BravaisLattice::from_vectors([1.0, 0.0], [0.0, 2.0]);
        assert_eq!(rect.lattice_type(), LatticeType::Rectangular);

        // Triangular-like vectors (60° convention)
        let sqrt3_2 = 3.0_f64.sqrt() / 2.0;
        let tri = BravaisLattice::from_vectors([1.0, 0.0], [0.5, sqrt3_2]);
        assert_eq!(tri.lattice_type(), LatticeType::Triangular);
    }
}
