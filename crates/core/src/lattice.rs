//! Lattice primitives for 2D photonic crystals.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

const CLASSIFICATION_TOL: f64 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatticeClass {
    Square,
    Rectangular,
    Triangular,
    Oblique,
}

/// 2D lattice defined by primitive vectors a1 and a2.
///
/// # TOML Format
///
/// Supports two formats for deserialization:
///
/// ## Legacy format (explicit vectors):
/// ```toml
/// [geometry.lattice]
/// a1 = [1.0, 0.0]
/// a2 = [0.0, 1.0]
/// ```
///
/// ## Typed format (recommended):
/// ```toml
/// [geometry.lattice]
/// type = "square"
/// a = 1.0  # optional, default = 1.0
///
/// [geometry.lattice]
/// type = "rectangular"
/// a = 1.0  # optional
/// b = 1.5  # required for rectangular
///
/// [geometry.lattice]
/// type = "triangular"
/// a = 1.0  # optional
///
/// [geometry.lattice]
/// type = "hexagonal"  # alias for triangular
/// a = 1.0  # optional
///
/// [geometry.lattice]
/// type = "oblique"
/// a = 1.0      # optional
/// b = 1.2      # length of second vector
/// alpha = 60.0 # angle in degrees
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Lattice2D {
    pub a1: [f64; 2],
    pub a2: [f64; 2],
}

// Custom serialization to always use the legacy format (explicit vectors)
impl Serialize for Lattice2D {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Lattice2D", 2)?;
        state.serialize_field("a1", &self.a1)?;
        state.serialize_field("a2", &self.a2)?;
        state.end()
    }
}

// Custom deserialization to support both legacy and typed formats
impl<'de> Deserialize<'de> for Lattice2D {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;

        #[derive(Deserialize)]
        #[serde(untagged)]
        enum LatticeFormat {
            /// Legacy format with explicit vectors
            Legacy {
                a1: [f64; 2],
                a2: [f64; 2],
            },
            /// Typed format with lattice type and parameters
            Typed {
                #[serde(rename = "type")]
                lattice_type: String,
                #[serde(default = "default_lattice_constant")]
                a: f64,
                #[serde(default)]
                b: Option<f64>,
                #[serde(default)]
                alpha: Option<f64>,
            },
        }

        fn default_lattice_constant() -> f64 {
            1.0
        }

        match LatticeFormat::deserialize(deserializer)? {
            LatticeFormat::Legacy { a1, a2 } => Ok(Lattice2D { a1, a2 }),
            LatticeFormat::Typed {
                lattice_type,
                a,
                b,
                alpha,
            } => {
                let lat_type_lower = lattice_type.to_lowercase();
                match lat_type_lower.as_str() {
                    "square" => Ok(Lattice2D::square(a)),
                    "rectangular" => {
                        let b_val = b.unwrap_or(a * 1.5);
                        Ok(Lattice2D::rectangular(a, b_val))
                    }
                    "triangular" | "hexagonal" => Ok(Lattice2D::hexagonal(a)),
                    "oblique" => {
                        let b_val = b.ok_or_else(|| {
                            D::Error::custom("oblique lattice requires 'b' parameter")
                        })?;
                        let alpha_deg = alpha.ok_or_else(|| {
                            D::Error::custom("oblique lattice requires 'alpha' parameter (in degrees)")
                        })?;
                        let alpha_rad = alpha_deg.to_radians();
                        let a2 = [b_val * alpha_rad.cos(), b_val * alpha_rad.sin()];
                        Ok(Lattice2D::oblique([a, 0.0], a2))
                    }
                    _ => Err(D::Error::custom(format!(
                        "unknown lattice type: '{}'. Valid types are: square, rectangular, triangular, hexagonal, oblique",
                        lattice_type
                    ))),
                }
            }
        }
    }
}

impl Lattice2D {
    pub fn square(a: f64) -> Self {
        Self {
            a1: [a, 0.0],
            a2: [0.0, a],
        }
    }

    pub fn rectangular(a: f64, b: f64) -> Self {
        Self {
            a1: [a, 0.0],
            a2: [0.0, b],
        }
    }

    pub fn hexagonal(a: f64) -> Self {
        let half = 0.5 * a;
        let h = (3.0f64).sqrt() * 0.5 * a;
        Self {
            a1: [a, 0.0],
            a2: [half, h],  // 60° convention: a₂ = [a/2, a√3/2]
        }
    }

    pub fn oblique(a1: [f64; 2], a2: [f64; 2]) -> Self {
        Self { a1, a2 }
    }

    pub fn reciprocal(&self) -> ReciprocalLattice2D {
        let det = self.determinant();
        assert!(
            det.abs() > f64::EPSILON,
            "primitive vectors are linearly dependent"
        );
        let inv = 2.0 * std::f64::consts::PI / det;
        let b1 = [self.a2[1] * inv, -self.a2[0] * inv];
        let b2 = [-self.a1[1] * inv, self.a1[0] * inv];
        ReciprocalLattice2D { b1, b2 }
    }

    pub fn fractional_to_cartesian(&self, frac: [f64; 2]) -> [f64; 2] {
        [
            self.a1[0] * frac[0] + self.a2[0] * frac[1],
            self.a1[1] * frac[0] + self.a2[1] * frac[1],
        ]
    }

    pub fn cartesian_to_fractional(&self, cart: [f64; 2]) -> [f64; 2] {
        let det = self.determinant();
        assert!(
            det.abs() > f64::EPSILON,
            "primitive vectors are linearly dependent"
        );
        let inv_det = 1.0 / det;
        let inv = [
            [self.a2[1] * inv_det, -self.a2[0] * inv_det],
            [-self.a1[1] * inv_det, self.a1[0] * inv_det],
        ];
        [
            inv[0][0] * cart[0] + inv[0][1] * cart[1],
            inv[1][0] * cart[0] + inv[1][1] * cart[1],
        ]
    }

    pub fn characteristic_length(&self) -> f64 {
        (self.a1[0] * self.a1[0] + self.a1[1] * self.a1[1]).sqrt()
    }

    pub fn classify(&self) -> LatticeClass {
        self.classify_with_tolerance(CLASSIFICATION_TOL)
    }

    pub fn classify_with_tolerance(&self, tol: f64) -> LatticeClass {
        let len1 = vector_norm(self.a1);
        let len2 = vector_norm(self.a2);
        if len1 <= f64::EPSILON || len2 <= f64::EPSILON {
            return LatticeClass::Oblique;
        }
        let dot = self.a1[0] * self.a2[0] + self.a1[1] * self.a2[1];
        let cos = (dot / (len1 * len2)).clamp(-1.0, 1.0);
        let len_diff = (len1 - len2).abs() / len1.max(len2);
        let is_orthogonal = cos.abs() <= tol;
        if len_diff <= tol && is_orthogonal {
            return LatticeClass::Square;
        }
        if is_orthogonal {
            return LatticeClass::Rectangular;
        }
        if len_diff <= tol && ((cos - 0.5).abs() <= tol || (cos + 0.5).abs() <= tol) {
            return LatticeClass::Triangular;
        }
        LatticeClass::Oblique
    }

    pub fn cell_area(&self) -> f64 {
        self.determinant().abs()
    }

    fn determinant(&self) -> f64 {
        self.a1[0] * self.a2[1] - self.a1[1] * self.a2[0]
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ReciprocalLattice2D {
    pub b1: [f64; 2],
    pub b2: [f64; 2],
}

impl ReciprocalLattice2D {
    /// Convert fractional k-space coordinates to Cartesian Bloch wavevector.
    ///
    /// The fractional coordinates (k1, k2) represent a k-point as:
    ///   k_cart = k1 * b1 + k2 * b2
    ///
    /// where b1 and b2 are the reciprocal lattice vectors.
    ///
    /// # Important
    ///
    /// For non-orthogonal lattices (hexagonal, oblique), this transformation
    /// is NOT simply `[2π * k1, 2π * k2]`. The reciprocal lattice vectors
    /// must be used to get the correct Cartesian Bloch wavevector.
    ///
    /// # Examples
    ///
    /// For a square lattice with a=1: b1 = [2π, 0], b2 = [0, 2π]
    ///   - M-point (0.5, 0) → [π, 0] ✓
    ///
    /// For a hexagonal lattice (60° convention): b1 ≈ [2π, -2π/√3], b2 = [0, 4π/√3]
    ///   - M-point (0.5, 0) → [π, -π/√3] (NOT [π, 0]!)
    ///   - K-point (1/3, 1/3) → [(2/3)π, (2/3)π/√3]
    #[inline]
    pub fn fractional_to_cartesian(&self, k_frac: [f64; 2]) -> [f64; 2] {
        [
            k_frac[0] * self.b1[0] + k_frac[1] * self.b2[0],
            k_frac[0] * self.b1[1] + k_frac[1] * self.b2[1],
        ]
    }
}

fn vector_norm(v: [f64; 2]) -> f64 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}
