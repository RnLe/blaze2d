//! Schema v2 configuration for Blaze2D solves and parameter sweeps.
//!
//! One TOML schema for every consumer (CLI, native bulk driver, Python, WASM).
//! A config with no `[[sweeps]]` entries describes exactly one job.
//!
//! # Format
//!
//! ```toml
//! schema = 2
//!
//! [solver]
//! type = "maxwell"          # "maxwell" | "operator_data" (alias "ea")
//! precision = "f64"         # "f64" | "f32"
//! polarization = "TM"       # "TM" | "TE"
//!
//! [geometry]
//! eps_bg = 12.0             # required
//!
//! [geometry.lattice]
//! type = "triangular"       # square | rectangular | triangular | oblique | custom
//! a = 1.0
//!
//! [[geometry.atoms]]
//! pos = [0.0, 0.0]          # fractional, in [0, 1)
//! radius = 0.2              # units of a, in (0, 0.5)
//! eps_inside = 1.0
//!
//! [grid]
//! nx = 32                   # required, [4, 512]; ny defaults to nx
//!
//! [path]
//! preset = "auto"           # auto | square | rectangular | triangular | hexagonal
//! points_per_segment = 12
//! # points = [[0.0, 0.0], [0.5, 0.0], ...]   # XOR with preset
//!
//! [[sweeps]]                # optional; first entry = outermost loop
//! parameter = "atom0.radius"
//! min = 0.2
//! max = 0.4
//! step = 0.05
//! ```
//!
//! # Validation
//!
//! - Unknown keys are rejected everywhere (`deny_unknown_fields`).
//! - Physical constraints are validated at parse time; all violations are
//!   collected as [`Diagnostic`]s with dotted key paths (and byte spans for
//!   serde-level errors), so UIs can surface every problem at once.
//! - Files without `schema = 2` fail with a targeted migration message that
//!   lists the v1 constructs found in the file (see `docs/MIGRATION_V2.md`).

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;

use blaze2d_core::{
    brillouin::BrillouinPath,
    dielectric::{DielectricOptions, SmoothingMethod, SmoothingOptions},
    eigensolver::EigensolverConfig,
    grid::Grid2D,
    lattice::Lattice2D,
    polarization::Polarization,
};

/// The schema version this crate parses.
pub const SCHEMA_VERSION: u32 = 2;

// ============================================================================
// Diagnostics
// ============================================================================

/// A single validation problem, addressed by dotted key path.
///
/// `span` is a byte-offset range into the source TOML when the problem was
/// reported by the parser itself (syntax errors, unknown keys, type errors).
/// Hand-rolled physics checks carry only `path`.
#[derive(Debug, Clone, Serialize)]
pub struct Diagnostic {
    /// Dotted key path, e.g. `"geometry.lattice.b"` or `"sweeps[1].step"`.
    /// Empty when the error is not attributable to a specific key.
    pub path: String,
    /// Human-readable message.
    pub message: String,
    /// Byte offsets `(start, end)` into the source TOML, when known.
    pub span: Option<(usize, usize)>,
}

impl Diagnostic {
    fn at(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
            span: None,
        }
    }
}

impl std::fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.path.is_empty() {
            write!(f, "{}", self.message)
        } else {
            write!(f, "{}: {}", self.path, self.message)
        }
    }
}

// ============================================================================
// Solver Section
// ============================================================================

/// Type of eigenproblem to solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SolverType {
    /// Maxwell eigenproblem for photonic crystals:
    /// ∇ × (1/ε)∇ × H = (ω/c)² H. Requires geometry, path, polarization.
    #[default]
    Maxwell,

    /// Operator-data extraction from the Maxwell eigenproblem at one (R, k₀)
    /// point: velocity matrices, mass tensor, Born-Huang potential. Downstream
    /// users may consume these for envelope-approximation models.
    #[serde(rename = "operator_data", alias = "ea")]
    OperatorData,
}

impl std::fmt::Display for SolverType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverType::Maxwell => write!(f, "Maxwell"),
            SolverType::OperatorData => write!(f, "Operator-Data Extraction"),
        }
    }
}

/// Storage precision for the Maxwell eigensolver.
///
/// Picks between `CpuBackend<f32>` and `CpuBackend<f64>` at driver construction
/// time. Eigenvalues, dot-product accumulation, and Rayleigh-Ritz always run in
/// `f64`; this knob only controls **storage precision** for fields, FFTs, and
/// preconditioners (the f32-storage / f64-accumulation invariant).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    /// Single precision: `Complex<f32>` storage. ~2x bandwidth at the cost of
    /// ~7 digits of precision in eigenvectors. Eigenvalues remain f64.
    #[serde(alias = "single")]
    F32,
    /// Double precision: `Complex<f64>` storage throughout. (Default.)
    #[default]
    #[serde(alias = "double")]
    F64,
}

impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Precision::F32 => write!(f, "f32"),
            Precision::F64 => write!(f, "f64"),
        }
    }
}

/// `[solver]` section.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SolverSection {
    /// Solver type: "maxwell" or "operator_data" (alias "ea").
    #[serde(rename = "type")]
    pub solver_type: SolverType,

    /// Storage precision: "f32" or "f64" (default "f64").
    pub precision: Precision,

    /// Polarization mode: "TM" (default) or "TE".
    pub polarization: Polarization,
}

impl Default for SolverSection {
    fn default() -> Self {
        Self {
            solver_type: SolverType::default(),
            precision: Precision::default(),
            polarization: Polarization::TM,
        }
    }
}

// ============================================================================
// Run Section
// ============================================================================

/// `[run]` section: execution knobs that do not affect the physics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields, default)]
pub struct RunSection {
    /// Number of worker threads (native only; ignored in WASM).
    /// Default: all physical cores.
    pub threads: Option<usize>,

    /// Enable verbose progress logging.
    pub verbose: bool,

    /// Skip calculating the final Γ-point when the k-path loops back to Γ
    /// (copy the initial Γ result instead).
    pub skip_final_gamma: bool,

    /// Disable band tracking between k-points (polar decomposition +
    /// Hungarian matching). Use for non-sequential k-paths.
    pub disable_band_tracking: bool,
}

// ============================================================================
// Geometry Section
// ============================================================================

/// Lattice type in the `[geometry.lattice]` section.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LatticeKind {
    Square,
    Rectangular,
    /// Triangular (= hexagonal) Bravais lattice, 60-degree convention.
    #[serde(alias = "hexagonal")]
    Triangular,
    Oblique,
    /// Explicit basis vectors `a1`, `a2`.
    Custom,
}

impl std::fmt::Display for LatticeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LatticeKind::Square => write!(f, "square"),
            LatticeKind::Rectangular => write!(f, "rectangular"),
            LatticeKind::Triangular => write!(f, "triangular"),
            LatticeKind::Oblique => write!(f, "oblique"),
            LatticeKind::Custom => write!(f, "custom"),
        }
    }
}

/// `[geometry.lattice]` section.
///
/// Parsed as a flat struct and validated per type so errors can say exactly
/// which key is missing or extraneous for the chosen lattice type:
///
/// - `square` / `triangular`: `a` (default 1.0)
/// - `rectangular`: `a` (default 1.0), `b` (required)
/// - `oblique`: `a` (default 1.0), `b` (required), `alpha_deg` (required,
///   degrees, in (0, 180))
/// - `custom`: `a1`, `a2` (both required, linearly independent)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatticeSection {
    /// Lattice type.
    #[serde(rename = "type")]
    pub kind: LatticeKind,

    /// Lattice constant |a1| (default 1.0). Not allowed for `custom`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub a: Option<f64>,

    /// Second lattice constant |a2| (rectangular, oblique).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub b: Option<f64>,

    /// Angle from a1 to a2 in degrees (oblique only), in (0, 180).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha_deg: Option<f64>,

    /// First basis vector (custom only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub a1: Option<[f64; 2]>,

    /// Second basis vector (custom only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub a2: Option<[f64; 2]>,
}

impl LatticeSection {
    /// Lattice constant with default applied.
    pub fn a(&self) -> f64 {
        self.a.unwrap_or(1.0)
    }

    /// Validate key combinations and value ranges for the chosen type.
    fn validate_into(&self, diags: &mut Vec<Diagnostic>) {
        let p = |key: &str| format!("geometry.lattice.{}", key);
        let kind = self.kind;

        let reject = |diags: &mut Vec<Diagnostic>, present: bool, key: &str| {
            if present {
                diags.push(Diagnostic::at(
                    p(key),
                    format!("'{}' is not used by lattice type '{}'", key, kind),
                ));
            }
        };

        if let Some(a) = self.a {
            if !(a > 0.0) {
                diags.push(Diagnostic::at(p("a"), "lattice constant 'a' must be positive"));
            }
        }
        if let Some(b) = self.b {
            if !(b > 0.0) {
                diags.push(Diagnostic::at(p("b"), "lattice constant 'b' must be positive"));
            }
        }

        match kind {
            LatticeKind::Square | LatticeKind::Triangular => {
                reject(diags, self.b.is_some(), "b");
                reject(diags, self.alpha_deg.is_some(), "alpha_deg");
                reject(diags, self.a1.is_some(), "a1");
                reject(diags, self.a2.is_some(), "a2");
            }
            LatticeKind::Rectangular => {
                if self.b.is_none() {
                    diags.push(Diagnostic::at(
                        p("b"),
                        "lattice type 'rectangular' requires 'b'",
                    ));
                }
                reject(diags, self.alpha_deg.is_some(), "alpha_deg");
                reject(diags, self.a1.is_some(), "a1");
                reject(diags, self.a2.is_some(), "a2");
            }
            LatticeKind::Oblique => {
                if self.b.is_none() {
                    diags.push(Diagnostic::at(p("b"), "lattice type 'oblique' requires 'b'"));
                }
                match self.alpha_deg {
                    None => diags.push(Diagnostic::at(
                        p("alpha_deg"),
                        "lattice type 'oblique' requires 'alpha_deg' (degrees, in (0, 180))",
                    )),
                    Some(alpha) if !(alpha > 0.0 && alpha < 180.0) => {
                        diags.push(Diagnostic::at(
                            p("alpha_deg"),
                            format!("'alpha_deg' must be in (0, 180), got {}", alpha),
                        ))
                    }
                    _ => {}
                }
                reject(diags, self.a1.is_some(), "a1");
                reject(diags, self.a2.is_some(), "a2");
            }
            LatticeKind::Custom => {
                reject(diags, self.a.is_some(), "a");
                reject(diags, self.b.is_some(), "b");
                reject(diags, self.alpha_deg.is_some(), "alpha_deg");
                match (self.a1, self.a2) {
                    (Some(a1), Some(a2)) => {
                        let det = a1[0] * a2[1] - a1[1] * a2[0];
                        if det.abs() <= f64::EPSILON {
                            diags.push(Diagnostic::at(
                                p("a2"),
                                "'a1' and 'a2' must be linearly independent",
                            ));
                        }
                    }
                    _ => diags.push(Diagnostic::at(
                        p("a1"),
                        "lattice type 'custom' requires both 'a1' and 'a2'",
                    )),
                }
            }
        }
    }

    /// Build the runtime lattice, optionally overriding the kind
    /// (for `lattice_type` sweeps). Assumes `validate_into` passed.
    pub fn build(&self, kind_override: Option<LatticeKind>) -> Lattice2D {
        let kind = kind_override.unwrap_or(self.kind);
        let a = self.a();
        match kind {
            LatticeKind::Square => Lattice2D::square(a),
            LatticeKind::Rectangular => Lattice2D::rectangular(a, self.b.unwrap_or(a * 1.5)),
            LatticeKind::Triangular => Lattice2D::hexagonal(a),
            LatticeKind::Oblique => {
                let b = self.b.expect("validated: oblique requires b");
                let alpha = self
                    .alpha_deg
                    .expect("validated: oblique requires alpha_deg")
                    .to_radians();
                Lattice2D::oblique([a, 0.0], [b * alpha.cos(), b * alpha.sin()])
            }
            LatticeKind::Custom => Lattice2D::oblique(
                self.a1.expect("validated: custom requires a1"),
                self.a2.expect("validated: custom requires a2"),
            ),
        }
    }
}

/// One atom of the basis (`[[geometry.atoms]]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaseAtom {
    /// Position in fractional coordinates, each component in [0, 1).
    pub pos: [f64; 2],

    /// Radius in units of the lattice constant, in (0, 0.5).
    pub radius: f64,

    /// Relative permittivity inside the atom (>= 1.0, default 1.0).
    #[serde(default = "default_eps_inside")]
    pub eps_inside: f64,
}

fn default_eps_inside() -> f64 {
    1.0
}

/// `[geometry]` section.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometrySection {
    /// Background relative permittivity (required, >= 1.0).
    pub eps_bg: f64,

    /// Lattice specification.
    pub lattice: LatticeSection,

    /// Basis atoms.
    #[serde(default)]
    pub atoms: Vec<BaseAtom>,
}

// ============================================================================
// Grid Section
// ============================================================================

/// `[grid]` section. `ny` defaults to `nx`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GridSection {
    /// Grid points along x (plane-wave cutoff), required, in [4, 512].
    pub nx: usize,

    /// Grid points along y (default: nx), in [4, 512].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ny: Option<usize>,

    /// Physical size along x (default 1.0).
    #[serde(default = "default_length")]
    pub lx: f64,

    /// Physical size along y (default 1.0).
    #[serde(default = "default_length")]
    pub ly: f64,

    /// Centered coordinates (from -l/2 to l/2) instead of [0, l).
    #[serde(default)]
    pub centered: bool,
}

fn default_length() -> f64 {
    1.0
}

impl GridSection {
    /// Effective ny (defaults to nx).
    pub fn ny(&self) -> usize {
        self.ny.unwrap_or(self.nx)
    }

    /// Build the runtime grid at the given resolution override
    /// (used for `resolution` sweeps; None keeps nx/ny as configured).
    pub fn to_grid(&self, resolution_override: Option<usize>) -> Grid2D {
        let (nx, ny) = match resolution_override {
            Some(r) => (r, r),
            None => (self.nx, self.ny()),
        };
        Grid2D {
            nx,
            ny,
            lx: self.lx,
            ly: self.ly,
            centered: self.centered,
        }
    }
}

/// Valid resolution range (plane-wave grids outside this are either
/// physically meaningless or blow past memory budgets).
pub const RESOLUTION_RANGE: (usize, usize) = (4, 512);

// ============================================================================
// Path Section
// ============================================================================

/// K-path preset in `[path]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PathPresetSpec {
    /// Resolve the preset from the (possibly swept) lattice type per job.
    Auto,
    /// Γ → X → M → Γ
    Square,
    /// Γ → X → S → Y → Γ
    Rectangular,
    /// Γ → M → K → Γ
    Triangular,
    /// Alias for triangular.
    #[serde(alias = "hex")]
    Hexagonal,
}

impl PathPresetSpec {
    /// Resolve to a concrete Brillouin path for the given lattice kind.
    /// Returns None when `auto` cannot resolve (oblique/custom lattices).
    pub fn resolve(self, kind: LatticeKind) -> Option<BrillouinPath> {
        match self {
            PathPresetSpec::Auto => match kind {
                LatticeKind::Square => Some(BrillouinPath::Square),
                LatticeKind::Rectangular => Some(BrillouinPath::Rectangular),
                LatticeKind::Triangular => Some(BrillouinPath::Triangular),
                LatticeKind::Oblique | LatticeKind::Custom => None,
            },
            PathPresetSpec::Square => Some(BrillouinPath::Square),
            PathPresetSpec::Rectangular => Some(BrillouinPath::Rectangular),
            PathPresetSpec::Triangular | PathPresetSpec::Hexagonal => {
                Some(BrillouinPath::Triangular)
            }
        }
    }
}

impl std::fmt::Display for PathPresetSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PathPresetSpec::Auto => write!(f, "auto"),
            PathPresetSpec::Square => write!(f, "square"),
            PathPresetSpec::Rectangular => write!(f, "rectangular"),
            PathPresetSpec::Triangular => write!(f, "triangular"),
            PathPresetSpec::Hexagonal => write!(f, "hexagonal"),
        }
    }
}

/// `[path]` section: exactly one of `preset` / `points`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct PathSection {
    /// Preset high-symmetry path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset: Option<PathPresetSpec>,

    /// K-points per segment between high-symmetry corners (preset mode only,
    /// default 12).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub points_per_segment: Option<usize>,

    /// Explicit k-point list in fractional reciprocal coordinates.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub points: Vec<[f64; 2]>,
}

/// Default k-points per segment for preset paths.
pub const DEFAULT_POINTS_PER_SEGMENT: usize = 12;

impl PathSection {
    /// Effective points-per-segment (default 12).
    pub fn points_per_segment(&self) -> usize {
        self.points_per_segment.unwrap_or(DEFAULT_POINTS_PER_SEGMENT)
    }

    fn validate_into(&self, lattice_kind: LatticeKind, diags: &mut Vec<Diagnostic>) {
        let has_preset = self.preset.is_some();
        let has_points = !self.points.is_empty();

        match (has_preset, has_points) {
            (false, false) => diags.push(Diagnostic::at(
                "path",
                "specify exactly one of 'preset' or 'points'",
            )),
            (true, true) => diags.push(Diagnostic::at(
                "path.points",
                "'preset' and 'points' are mutually exclusive",
            )),
            (true, false) => {
                if let Some(pps) = self.points_per_segment {
                    if pps < 1 {
                        diags.push(Diagnostic::at(
                            "path.points_per_segment",
                            "'points_per_segment' must be >= 1",
                        ));
                    }
                }
                if matches!(lattice_kind, LatticeKind::Oblique | LatticeKind::Custom) {
                    diags.push(Diagnostic::at(
                        "path.preset",
                        format!(
                            "no standard high-symmetry path exists for lattice type '{}'; \
                             use 'path.points' instead",
                            lattice_kind
                        ),
                    ));
                }
            }
            (false, true) => {
                if self.points_per_segment.is_some() {
                    diags.push(Diagnostic::at(
                        "path.points_per_segment",
                        "'points_per_segment' only applies to preset paths",
                    ));
                }
                if self.points.len() < 2 {
                    diags.push(Diagnostic::at(
                        "path.points",
                        "'points' needs at least 2 k-points",
                    ));
                }
                for (i, p) in self.points.iter().enumerate() {
                    if !p[0].is_finite() || !p[1].is_finite() {
                        diags.push(Diagnostic::at(
                            format!("path.points[{}]", i),
                            "k-point components must be finite",
                        ));
                    }
                }
            }
        }
    }
}

// ============================================================================
// Sweeps
// ============================================================================

/// One `[[sweeps]]` entry: a parameter axis of the sweep.
///
/// Sweeps form nested loops in TOML order; the first entry is the outermost
/// loop. Exactly one of `(min, max, step)` or `values` must be given.
///
/// Valid parameters: `eps_bg`, `resolution`, `polarization`, `lattice_type`,
/// and `atomN.{radius, pos_x, pos_y, eps_inside}` where `atomN` must reference
/// an atom defined in `[[geometry.atoms]]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SweepSpec {
    /// Parameter path, e.g. `"eps_bg"` or `"atom0.radius"`.
    pub parameter: String,

    /// Minimum value (range mode, inclusive).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,

    /// Maximum value (range mode, inclusive).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,

    /// Step size (range mode, > 0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<f64>,

    /// Discrete values (values mode).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<toml::Value>>,
}

impl SweepSpec {
    /// True when min/max/step are all given.
    pub fn is_range(&self) -> bool {
        self.min.is_some() && self.max.is_some() && self.step.is_some()
    }

    /// True when discrete values are given.
    pub fn is_discrete(&self) -> bool {
        self.values.is_some()
    }

    /// Number of values on this axis.
    ///
    /// Uses the same epsilon-tolerant arithmetic as [`Self::range_values`] so
    /// the count always matches the materialized values (a plain `floor`
    /// undercounts ranges like 0.15..0.35 step 0.05 due to float rounding).
    pub fn count(&self) -> usize {
        if let Some(ref values) = self.values {
            values.len()
        } else if let (Some(min), Some(max), Some(step)) = (self.min, self.max, self.step) {
            if step <= 0.0 || max < min {
                0
            } else {
                ((max - min) / step + 1e-9).floor() as usize + 1
            }
        } else {
            0
        }
    }

    /// Generate the numeric values for a range sweep.
    fn range_values(&self) -> Vec<f64> {
        let (min, max, step) = match (self.min, self.max, self.step) {
            (Some(a), Some(b), Some(s)) => (a, b, s),
            _ => return Vec::new(),
        };
        let mut out = Vec::new();
        let mut v = min;
        while v <= max + step * 1e-9 {
            out.push(v);
            v += step;
        }
        out
    }

    fn validate_into(&self, index: usize, atom_count: usize, diags: &mut Vec<Diagnostic>) {
        let path = format!("sweeps[{}]", index);

        match (self.is_range(), self.is_discrete()) {
            (false, false) => {
                // Partially specified ranges get a more precise message.
                if self.min.is_some() || self.max.is_some() || self.step.is_some() {
                    diags.push(Diagnostic::at(
                        &path,
                        format!(
                            "sweep '{}' has an incomplete range; 'min', 'max' and 'step' \
                             must all be given",
                            self.parameter
                        ),
                    ));
                } else {
                    diags.push(Diagnostic::at(
                        &path,
                        format!(
                            "sweep '{}' must specify either (min, max, step) or 'values'",
                            self.parameter
                        ),
                    ));
                }
                return;
            }
            (true, true) => {
                diags.push(Diagnostic::at(
                    &path,
                    format!(
                        "sweep '{}' cannot have both a range and 'values'",
                        self.parameter
                    ),
                ));
                return;
            }
            _ => {}
        }

        if self.is_range() {
            let (min, max, step) = (
                self.min.unwrap(),
                self.max.unwrap(),
                self.step.unwrap(),
            );
            if step <= 0.0 {
                diags.push(Diagnostic::at(
                    format!("{}.step", path),
                    "'step' must be positive",
                ));
            }
            if min > max {
                diags.push(Diagnostic::at(
                    format!("{}.min", path),
                    format!("'min' ({}) is greater than 'max' ({})", min, max),
                ));
            }
        }

        if let Some(ref values) = self.values {
            if values.is_empty() {
                diags.push(Diagnostic::at(
                    format!("{}.values", path),
                    "'values' must not be empty",
                ));
            }
        }

        // Parameter path shape.
        if let Err(msg) = validate_parameter_path(&self.parameter) {
            diags.push(Diagnostic::at(format!("{}.parameter", path), msg));
            return;
        }

        // Atom references must point at defined atoms.
        if let Some((atom_idx, _)) = parse_atom_path(&self.parameter) {
            if atom_idx >= atom_count {
                diags.push(Diagnostic::at(
                    format!("{}.parameter", path),
                    format!(
                        "sweep references atom{} but only {} atom(s) are defined in \
                         [[geometry.atoms]]",
                        atom_idx, atom_count
                    ),
                ));
                return;
            }
        }

        // Every concrete value must satisfy the same physics rules as the
        // base config.
        match SweepDimension::from_spec(self, index) {
            Ok(dim) => {
                for (vi, value) in dim.values.iter().enumerate() {
                    if let Err(msg) = validate_sweep_value(&self.parameter, value) {
                        let key = if self.is_discrete() {
                            format!("{}.values[{}]", path, vi)
                        } else {
                            path.clone()
                        };
                        diags.push(Diagnostic::at(key, msg));
                    }
                }
            }
            Err(msg) => diags.push(Diagnostic::at(&path, msg)),
        }
    }
}

/// A concrete value in a sweep dimension (typed for output formatting).
#[derive(Debug, Clone, PartialEq)]
pub enum SweepValue {
    /// Floating-point value (eps_bg, radius, pos_x, ...).
    Float(f64),
    /// Integer value (resolution).
    Int(i64),
    /// String value (polarization, lattice_type).
    String(String),
}

impl SweepValue {
    /// Format for CSV output.
    pub fn to_csv_string(&self) -> String {
        match self {
            SweepValue::Float(v) => format!("{:.6}", v),
            SweepValue::Int(v) => v.to_string(),
            SweepValue::String(s) => s.clone(),
        }
    }

    /// As f64, if numeric.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            SweepValue::Float(v) => Some(*v),
            SweepValue::Int(v) => Some(*v as f64),
            SweepValue::String(_) => None,
        }
    }

    /// As i64, if numeric.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            SweepValue::Float(v) => Some(*v as i64),
            SweepValue::Int(v) => Some(*v),
            SweepValue::String(_) => None,
        }
    }

    /// As string slice, if a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            SweepValue::String(s) => Some(s),
            _ => None,
        }
    }
}

impl std::fmt::Display for SweepValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SweepValue::Float(v) => write!(f, "{}", v),
            SweepValue::Int(v) => write!(f, "{}", v),
            SweepValue::String(s) => write!(f, "{}", s),
        }
    }
}

/// A sweep axis with all values materialized, used during job expansion.
#[derive(Debug, Clone)]
pub struct SweepDimension {
    /// Parameter name.
    pub name: String,
    /// Loop order (0 = outermost).
    pub order: usize,
    /// All values on this axis.
    pub values: Vec<SweepValue>,
}

impl SweepDimension {
    /// Materialize a spec into concrete values.
    pub fn from_spec(spec: &SweepSpec, order: usize) -> Result<Self, String> {
        let values = if let Some(ref discrete) = spec.values {
            discrete
                .iter()
                .map(|v| toml_to_sweep_value(v, &spec.parameter))
                .collect::<Result<Vec<_>, _>>()?
        } else if spec.is_range() {
            let is_resolution = spec.parameter == "resolution";
            spec.range_values()
                .into_iter()
                .map(|v| {
                    if is_resolution {
                        SweepValue::Int(v.round() as i64)
                    } else {
                        SweepValue::Float(v)
                    }
                })
                .collect()
        } else {
            return Err(format!(
                "sweep '{}' has no valid range or values",
                spec.parameter
            ));
        };

        Ok(Self {
            name: spec.parameter.clone(),
            order,
            values,
        })
    }

    /// Number of values.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// True when there are no values.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

fn toml_to_sweep_value(value: &toml::Value, param: &str) -> Result<SweepValue, String> {
    match value {
        toml::Value::Float(f) => Ok(SweepValue::Float(*f)),
        toml::Value::Integer(i) => {
            if param == "resolution" {
                Ok(SweepValue::Int(*i))
            } else {
                Ok(SweepValue::Float(*i as f64))
            }
        }
        toml::Value::String(s) => Ok(SweepValue::String(s.clone())),
        _ => Err(format!(
            "unsupported value type for parameter '{}': {:?}",
            param, value
        )),
    }
}

/// Valid non-atom sweep parameters.
pub const VALID_GLOBAL_PARAMS: &[&str] = &["eps_bg", "resolution", "polarization", "lattice_type"];

/// Valid atom sweep properties.
pub const VALID_ATOM_PROPS: &[&str] = &["radius", "pos_x", "pos_y", "eps_inside"];

/// Validate the shape of a sweep parameter path.
pub fn validate_parameter_path(path: &str) -> Result<(), String> {
    if VALID_GLOBAL_PARAMS.contains(&path) {
        return Ok(());
    }

    if path.starts_with("atom") {
        let parts: Vec<&str> = path.splitn(2, '.').collect();
        if parts.len() != 2 {
            return Err(format!(
                "invalid atom parameter path '{}': expected 'atomN.property'",
                path
            ));
        }
        let index_str = parts[0].strip_prefix("atom").unwrap();
        if index_str.parse::<usize>().is_err() {
            return Err(format!(
                "invalid atom index in '{}': expected 'atomN' where N is a number",
                path
            ));
        }
        if !VALID_ATOM_PROPS.contains(&parts[1]) {
            return Err(format!(
                "invalid atom property '{}' in '{}': expected one of {:?}",
                parts[1], path, VALID_ATOM_PROPS
            ));
        }
        return Ok(());
    }

    Err(format!(
        "unknown parameter '{}': expected one of {:?} or an atom path like 'atom0.radius'",
        path, VALID_GLOBAL_PARAMS
    ))
}

/// Parse an atom parameter path into `(atom_index, property)`.
pub fn parse_atom_path(path: &str) -> Option<(usize, &str)> {
    if !path.starts_with("atom") {
        return None;
    }
    let parts: Vec<&str> = path.splitn(2, '.').collect();
    if parts.len() != 2 {
        return None;
    }
    let index = parts[0].strip_prefix("atom")?.parse().ok()?;
    Some((index, parts[1]))
}

/// Validate one concrete sweep value against the physics rules.
fn validate_sweep_value(param: &str, value: &SweepValue) -> Result<(), String> {
    let numeric = |value: &SweepValue| -> Result<f64, String> {
        value
            .as_f64()
            .ok_or_else(|| format!("parameter '{}' requires numeric values", param))
    };

    match param {
        "eps_bg" => {
            let v = numeric(value)?;
            if v < 1.0 {
                return Err(format!("eps_bg must be >= 1.0, got {}", v));
            }
        }
        "resolution" => {
            let v = value
                .as_i64()
                .ok_or_else(|| "resolution requires integer values".to_string())?;
            let (lo, hi) = RESOLUTION_RANGE;
            if v < lo as i64 || v > hi as i64 {
                return Err(format!("resolution must be in [{}, {}], got {}", lo, hi, v));
            }
        }
        "polarization" => {
            let s = value
                .as_str()
                .ok_or_else(|| "polarization values must be strings".to_string())?;
            if !matches!(s.to_uppercase().as_str(), "TM" | "TE") {
                return Err(format!("polarization must be \"TM\" or \"TE\", got \"{}\"", s));
            }
        }
        "lattice_type" => {
            let s = value
                .as_str()
                .ok_or_else(|| "lattice_type values must be strings".to_string())?;
            if parse_swept_lattice_kind(s).is_none() {
                return Err(format!(
                    "lattice_type must be one of \"square\", \"rectangular\", \"triangular\", \
                     \"hexagonal\", got \"{}\"",
                    s
                ));
            }
        }
        _ => {
            if let Some((_, prop)) = parse_atom_path(param) {
                let v = numeric(value)?;
                match prop {
                    "radius" => {
                        if !(v > 0.0 && v < 0.5) {
                            return Err(format!("radius must be in (0, 0.5), got {}", v));
                        }
                    }
                    "pos_x" | "pos_y" => {
                        if !(0.0..1.0).contains(&v) {
                            return Err(format!("{} must be in [0, 1), got {}", prop, v));
                        }
                    }
                    "eps_inside" => {
                        if v < 1.0 {
                            return Err(format!("eps_inside must be >= 1.0, got {}", v));
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

/// Parse a swept lattice-type string into a kind (sweeps cannot produce
/// oblique/custom lattices).
pub fn parse_swept_lattice_kind(s: &str) -> Option<LatticeKind> {
    match s.to_lowercase().as_str() {
        "square" => Some(LatticeKind::Square),
        "rectangular" => Some(LatticeKind::Rectangular),
        "triangular" | "hexagonal" => Some(LatticeKind::Triangular),
        _ => None,
    }
}

// ============================================================================
// Dielectric Section
// ============================================================================

/// Interface smoothing mode in `[dielectric]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SmoothingKind {
    /// Analytic geometry-aware smoothing (MPB-style, default).
    #[default]
    Analytic,
    /// Numerical subgrid sampling (uses `mesh_size`).
    Subgrid,
    /// No interface smoothing.
    #[serde(rename = "none")]
    Disabled,
}

/// `[dielectric]` section (flattened; replaces the v1 `[dielectric.smoothing]`
/// subtable and its hidden "enabled iff mesh_size > 1" rule).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct DielectricSection {
    /// Smoothing mode: "analytic" (default), "subgrid", or "none".
    pub smoothing: SmoothingKind,

    /// Subgrid mesh size (subgrid mode only, >= 2, default 3).
    pub mesh_size: usize,

    /// Tolerance for detecting material interfaces.
    pub interface_tolerance: f64,
}

impl Default for DielectricSection {
    fn default() -> Self {
        Self {
            smoothing: SmoothingKind::Analytic,
            mesh_size: 3,
            interface_tolerance: 1e-6,
        }
    }
}

impl DielectricSection {
    fn validate_into(&self, diags: &mut Vec<Diagnostic>) {
        if self.smoothing == SmoothingKind::Subgrid && self.mesh_size < 2 {
            diags.push(Diagnostic::at(
                "dielectric.mesh_size",
                "'mesh_size' must be >= 2 for subgrid smoothing",
            ));
        }
        if !(self.interface_tolerance > 0.0) {
            diags.push(Diagnostic::at(
                "dielectric.interface_tolerance",
                "'interface_tolerance' must be positive",
            ));
        }
    }

    /// Convert to the core runtime options (which encode "disabled" as
    /// `mesh_size <= 1`).
    pub fn to_options(&self) -> DielectricOptions {
        let smoothing = match self.smoothing {
            SmoothingKind::Analytic => SmoothingOptions {
                mesh_size: self.mesh_size.max(2),
                interface_tolerance: self.interface_tolerance,
                method: SmoothingMethod::Analytic,
            },
            SmoothingKind::Subgrid => SmoothingOptions {
                mesh_size: self.mesh_size,
                interface_tolerance: self.interface_tolerance,
                method: SmoothingMethod::Subgrid,
            },
            SmoothingKind::Disabled => SmoothingOptions {
                mesh_size: 1,
                interface_tolerance: self.interface_tolerance,
                method: SmoothingMethod::Analytic,
            },
        };
        DielectricOptions { smoothing }
    }
}

// ============================================================================
// Output Section
// ============================================================================

/// Output mode selection.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OutputMode {
    /// One CSV file per job with the complete band structure.
    #[default]
    Full,
    /// Single merged CSV with selected k-points and bands.
    Selective,
}

/// `[output.selective]` subsection.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields, default)]
pub struct SelectiveSpec {
    /// K-point indices to include (0-based).
    pub k_indices: Vec<usize>,

    /// K-point labels to include (e.g. "Gamma", "X", "M"); preset paths only.
    pub k_labels: Vec<String>,

    /// Band indices to include (1-based, matching band1, band2, ... columns).
    pub bands: Vec<usize>,
}

/// `[output]` section (native CSV writer; WASM consumers stream instead).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct OutputConfig {
    /// Output mode: "full" or "selective".
    pub mode: OutputMode,

    /// Output directory (full mode).
    pub directory: PathBuf,

    /// Output filename (selective mode).
    pub filename: PathBuf,

    /// Filename prefix (full mode).
    pub prefix: String,

    /// Selective output specification.
    pub selective: SelectiveSpec,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            mode: OutputMode::Full,
            directory: PathBuf::from("./bulk_output"),
            filename: PathBuf::from("./bulk_results.csv"),
            prefix: "job".to_string(),
            selective: SelectiveSpec::default(),
        }
    }
}

// ============================================================================
// Operator-Data Section
// ============================================================================

/// `[operator_data]` section (alias `[ea]`): configuration for operator-data
/// extraction (velocity matrices, mass tensor, Born-Huang potential) at one
/// (R, k₀) point. Only read when `solver.type = "operator_data"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct OperatorDataDriverConfig {
    /// Carrier momentum k₀ in Cartesian reciprocal-space units (2π/a).
    pub k0: [f64; 2],

    /// Number of retained bands in the active subspace.
    pub n_retained: usize,

    /// Number of remote bands for Löwdin corrections.
    pub n_remote: usize,

    /// Compute the Löwdin-corrected inverse mass tensor.
    pub compute_mass_tensor: bool,

    /// Compute the Born-Huang scalar potential.
    pub compute_born_huang: bool,

    /// Compute the TE slow-coefficient scalar potential.
    pub compute_slow_coefficient: bool,

    /// Compute dielectric derivatives for R-derivative matrix elements.
    pub compute_r_derivatives: bool,

    /// Which atom to differentiate w.r.t. for R-derivatives.
    pub atom_index: usize,

    /// Finite-difference step for dielectric derivatives (fractional coords).
    pub fd_step: f64,
}

impl Default for OperatorDataDriverConfig {
    fn default() -> Self {
        Self {
            k0: [0.0, 0.0],
            n_retained: 4,
            n_remote: 8,
            compute_mass_tensor: true,
            compute_born_huang: false,
            compute_slow_coefficient: false,
            compute_r_derivatives: true,
            atom_index: 0,
            fd_step: 0.001,
        }
    }
}

// ============================================================================
// The Config
// ============================================================================

/// Complete schema v2 configuration.
///
/// One schema for every consumer. A config with no `[[sweeps]]` describes a
/// single job; with sweeps, the axes form nested loops in TOML order (first
/// entry outermost).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Schema version marker; must be `2`.
    pub schema: u32,

    /// Execution knobs (threads, logging, band tracking).
    #[serde(default)]
    pub run: RunSection,

    /// Solver selection (type, precision, polarization).
    #[serde(default)]
    pub solver: SolverSection,

    /// Crystal geometry (required).
    pub geometry: GeometrySection,

    /// Computational grid (required).
    pub grid: GridSection,

    /// K-path (required for the Maxwell solver).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<PathSection>,

    /// Ordered parameter sweeps; first entry = outermost loop.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sweeps: Vec<SweepSpec>,

    /// Eigensolver configuration.
    #[serde(default)]
    pub eigensolver: EigensolverConfig,

    /// Dielectric smoothing configuration.
    #[serde(default)]
    pub dielectric: DielectricSection,

    /// Output configuration (native CSV writer).
    #[serde(default)]
    pub output: OutputConfig,

    /// Operator-data extraction configuration (alias `[ea]`); only read when
    /// `solver.type = "operator_data"`.
    #[serde(default, alias = "ea")]
    pub operator_data: OperatorDataDriverConfig,
}

/// Compatibility alias; `Config` is the v2 name.
pub type BulkConfig = Config;

impl Config {
    /// Load and validate a configuration from a TOML file.
    pub fn from_file(path: &std::path::Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        Self::from_str(&content)
    }

    /// Parse and validate a configuration from a TOML string.
    ///
    /// All diagnostics are joined into one `ConfigError::Invalid` message.
    /// Use [`parse_and_validate`] to get structured diagnostics instead.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, ConfigError> {
        parse_and_validate(s).map_err(|diags| {
            ConfigError::Invalid(
                diags
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
        })
    }

    /// Solver type.
    pub fn solver_type(&self) -> SolverType {
        self.solver.solver_type
    }

    /// Storage precision.
    pub fn precision(&self) -> Precision {
        self.solver.precision
    }

    /// True for the Maxwell solver.
    pub fn is_maxwell(&self) -> bool {
        self.solver.solver_type == SolverType::Maxwell
    }

    /// Materialize the sweep axes for expansion.
    pub fn build_sweep_dimensions(&self) -> Result<Vec<SweepDimension>, ConfigError> {
        self.sweeps
            .iter()
            .enumerate()
            .map(|(order, spec)| {
                SweepDimension::from_spec(spec, order).map_err(ConfigError::Invalid)
            })
            .collect()
    }

    /// Total number of jobs after sweep expansion.
    pub fn total_jobs(&self) -> usize {
        self.sweeps.iter().map(|s| s.count().max(1)).product()
    }

    /// Human-readable description of the sweep loop order.
    pub fn sweep_order_description(&self) -> Vec<String> {
        self.sweeps
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let loop_type = if i == 0 {
                    "outer"
                } else if i == self.sweeps.len() - 1 {
                    "inner"
                } else {
                    "middle"
                };
                format!(
                    "{}: {} ({} values, {} loop)",
                    i + 1,
                    s.parameter,
                    s.count(),
                    loop_type
                )
            })
            .collect()
    }

    /// Validate; joined-string error form for native consumers.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let diags = self.validate_diagnostics();
        if diags.is_empty() {
            Ok(())
        } else {
            Err(ConfigError::Invalid(
                diags
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join("\n"),
            ))
        }
    }

    /// Collect every physics/consistency violation (post-parse checks).
    pub fn validate_diagnostics(&self) -> Vec<Diagnostic> {
        let mut diags = Vec::new();

        if self.schema != SCHEMA_VERSION {
            diags.push(Diagnostic::at(
                "schema",
                format!(
                    "unsupported schema version {}; this build reads schema = {}",
                    self.schema, SCHEMA_VERSION
                ),
            ));
        }

        // --- geometry ---
        if self.geometry.eps_bg < 1.0 {
            diags.push(Diagnostic::at(
                "geometry.eps_bg",
                format!("'eps_bg' must be >= 1.0, got {}", self.geometry.eps_bg),
            ));
        }
        self.geometry.lattice.validate_into(&mut diags);
        for (i, atom) in self.geometry.atoms.iter().enumerate() {
            let p = |key: &str| format!("geometry.atoms[{}].{}", i, key);
            if !(atom.radius > 0.0 && atom.radius < 0.5) {
                diags.push(Diagnostic::at(
                    p("radius"),
                    format!("'radius' must be in (0, 0.5), got {}", atom.radius),
                ));
            }
            if atom.eps_inside < 1.0 {
                diags.push(Diagnostic::at(
                    p("eps_inside"),
                    format!("'eps_inside' must be >= 1.0, got {}", atom.eps_inside),
                ));
            }
            for (c, name) in [(atom.pos[0], "x"), (atom.pos[1], "y")] {
                if !(0.0..1.0).contains(&c) {
                    diags.push(Diagnostic::at(
                        p("pos"),
                        format!("fractional {}-position must be in [0, 1), got {}", name, c),
                    ));
                }
            }
        }

        // --- grid ---
        let (lo, hi) = RESOLUTION_RANGE;
        for (n, key) in [(self.grid.nx, "nx"), (self.grid.ny(), "ny")] {
            if n < lo || n > hi {
                diags.push(Diagnostic::at(
                    format!("grid.{}", key),
                    format!("'{}' must be in [{}, {}], got {}", key, lo, hi, n),
                ));
            }
        }
        for (l, key) in [(self.grid.lx, "lx"), (self.grid.ly, "ly")] {
            if !(l > 0.0) {
                diags.push(Diagnostic::at(
                    format!("grid.{}", key),
                    format!("'{}' must be positive", key),
                ));
            }
        }

        // --- eigensolver ---
        if self.eigensolver.n_bands < 1 {
            diags.push(Diagnostic::at(
                "eigensolver.n_bands",
                "'n_bands' must be >= 1",
            ));
        }
        if self.eigensolver.max_iter < 1 {
            diags.push(Diagnostic::at(
                "eigensolver.max_iter",
                "'max_iter' must be >= 1",
            ));
        }
        if !(self.eigensolver.tol > 0.0) {
            diags.push(Diagnostic::at("eigensolver.tol", "'tol' must be positive"));
        }
        if self.eigensolver.block_size != 0 && self.eigensolver.block_size < self.eigensolver.n_bands
        {
            diags.push(Diagnostic::at(
                "eigensolver.block_size",
                "'block_size' must be 0 (auto) or >= n_bands",
            ));
        }

        // --- dielectric ---
        self.dielectric.validate_into(&mut diags);

        // --- path (Maxwell only) ---
        if self.is_maxwell() {
            match &self.path {
                None => diags.push(Diagnostic::at(
                    "path",
                    "the Maxwell solver requires a [path] section ('preset' or 'points')",
                )),
                Some(path) => path.validate_into(self.geometry.lattice.kind, &mut diags),
            }
        }

        // --- output ---
        if self.output.mode == OutputMode::Selective {
            let spec = &self.output.selective;
            if spec.k_indices.is_empty() && spec.k_labels.is_empty() {
                diags.push(Diagnostic::at(
                    "output.selective",
                    "selective mode requires 'k_indices' or 'k_labels'",
                ));
            }
            if spec.bands.is_empty() {
                diags.push(Diagnostic::at(
                    "output.selective.bands",
                    "selective mode requires 'bands' (1-based indices)",
                ));
            }
            if spec.bands.iter().any(|&b| b == 0) {
                diags.push(Diagnostic::at(
                    "output.selective.bands",
                    "'bands' indices are 1-based; 0 is not a valid band",
                ));
            }
            if !spec.k_labels.is_empty() {
                let has_preset = self.path.as_ref().is_some_and(|p| p.preset.is_some());
                if !has_preset {
                    diags.push(Diagnostic::at(
                        "output.selective.k_labels",
                        "'k_labels' requires a preset path (explicit 'points' have no labels); \
                         use 'k_indices' instead",
                    ));
                }
            }
        }

        // --- sweeps ---
        let mut seen = HashSet::new();
        for (i, sweep) in self.sweeps.iter().enumerate() {
            if !seen.insert(sweep.parameter.as_str()) {
                diags.push(Diagnostic::at(
                    format!("sweeps[{}].parameter", i),
                    format!("duplicate sweep parameter '{}'", sweep.parameter),
                ));
            }
            sweep.validate_into(i, self.geometry.atoms.len(), &mut diags);
        }
        // lattice_type sweeps replace the lattice kind wholesale; that only
        // makes sense when the base lattice is a named type.
        if self.sweeps.iter().any(|s| s.parameter == "lattice_type")
            && self.geometry.lattice.kind == LatticeKind::Custom
        {
            diags.push(Diagnostic::at(
                "sweeps",
                "'lattice_type' cannot be swept when the base lattice is 'custom'",
            ));
        }

        // --- operator_data ---
        if self.solver.solver_type == SolverType::OperatorData {
            let od = &self.operator_data;
            if od.n_retained < 1 {
                diags.push(Diagnostic::at(
                    "operator_data.n_retained",
                    "'n_retained' must be >= 1",
                ));
            }
            if !(od.fd_step > 0.0) {
                diags.push(Diagnostic::at(
                    "operator_data.fd_step",
                    "'fd_step' must be positive",
                ));
            }
            if od.compute_r_derivatives && od.atom_index >= self.geometry.atoms.len().max(1) {
                diags.push(Diagnostic::at(
                    "operator_data.atom_index",
                    format!(
                        "'atom_index' {} is out of range for {} defined atom(s)",
                        od.atom_index,
                        self.geometry.atoms.len()
                    ),
                ));
            }
            if !self.sweeps.is_empty() {
                diags.push(Diagnostic::at(
                    "sweeps",
                    "parameter sweeps are not supported for the operator_data solver",
                ));
            }
        }

        diags
    }
}

// ============================================================================
// Parsing entry point
// ============================================================================

/// Parse and validate a schema v2 TOML string, collecting structured
/// diagnostics.
///
/// This is the single source of truth for configuration validity; the web
/// editor calls it through the WASM `validateConfig` export.
pub fn parse_and_validate(source: &str) -> Result<Config, Vec<Diagnostic>> {
    // 1. Syntax check via a generic parse (gives spans on syntax errors).
    let doc: toml::Value = match toml::from_str(source) {
        Ok(v) => v,
        Err(e) => return Err(vec![toml_error_to_diagnostic(&e)]),
    };

    // 2. Schema marker: absent -> v1 migration hint; wrong -> version error.
    match doc.get("schema") {
        None => return Err(vec![migration_diagnostic(source)]),
        Some(toml::Value::Integer(v)) if *v == SCHEMA_VERSION as i64 => {}
        Some(toml::Value::Integer(v)) => {
            return Err(vec![Diagnostic::at(
                "schema",
                format!(
                    "unsupported schema version {}; this build reads schema = {}",
                    v, SCHEMA_VERSION
                ),
            )]);
        }
        Some(_) => {
            return Err(vec![Diagnostic::at(
                "schema",
                format!("'schema' must be the integer {}", SCHEMA_VERSION),
            )]);
        }
    }

    // 3. Typed parse (deny_unknown_fields; spans from the toml crate).
    let config: Config = match toml::from_str(source) {
        Ok(c) => c,
        Err(e) => return Err(vec![toml_error_to_diagnostic(&e)]),
    };

    // 4. Physics and consistency checks, all collected.
    let diags = config.validate_diagnostics();
    if diags.is_empty() {
        Ok(config)
    } else {
        Err(diags)
    }
}

fn toml_error_to_diagnostic(e: &toml::de::Error) -> Diagnostic {
    Diagnostic {
        path: String::new(),
        message: e.message().to_string(),
        span: e.span().map(|r| (r.start, r.end)),
    }
}

/// Build the targeted migration message for a file without `schema = 2`.
fn migration_diagnostic(source: &str) -> Diagnostic {
    let mut hints: Vec<&str> = Vec::new();

    let mut check = |needle: &str, hint: &'static str| {
        if source.contains(needle) {
            hints.push(hint);
        }
    };

    check("[bulk]", "[bulk] is now [run] (and 'dry_run' moved to the CLI)");
    check("[ranges]", "[ranges] was removed; use ordered [[sweeps]] entries");
    check(
        "[defaults",
        "[defaults] was removed; sweeps read base values from [geometry], [grid] and [solver]",
    );
    check("k_path", "'k_path' is now 'points' inside [path]");
    check(
        "segments_per_leg",
        "'segments_per_leg' is now 'points_per_segment'",
    );
    check("io_mode", "'io_mode' and the [output] batching knobs were removed");
    check("batch_size", "'batch_size' was removed");
    check(
        "ea_hamiltonian",
        "'ea_hamiltonian' is now 'operator_data' (alias \"ea\")",
    );
    check(
        "[dielectric.smoothing]",
        "[dielectric.smoothing] was flattened into [dielectric] with smoothing = \
         \"analytic\" | \"subgrid\" | \"none\"",
    );
    if source.contains("\nalpha") || source.starts_with("alpha") {
        hints.push("lattice 'alpha' is now 'alpha_deg' (always degrees)");
    }
    // Top-level polarization: a line that starts with the bare key before any
    // section header.
    let top = source.split('[').next().unwrap_or("");
    if top.lines().any(|l| l.trim_start().starts_with("polarization")) {
        hints.push("top-level 'polarization' moved into [solver]");
    }

    let mut message = format!(
        "missing 'schema = {}': this looks like a v1 configuration. \
         See docs/MIGRATION_V2.md.",
        SCHEMA_VERSION
    );
    if !hints.is_empty() {
        message.push_str("\nDetected v1 constructs:");
        for h in &hints {
            message.push_str("\n  - ");
            message.push_str(h);
        }
    }

    Diagnostic {
        path: "schema".into(),
        message,
        span: None,
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Configuration errors (string form, for native consumers).
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid configuration:\n{0}")]
    Invalid(String),
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"
schema = 2

[geometry]
eps_bg = 12.0

[geometry.lattice]
type = "square"

[[geometry.atoms]]
pos = [0.5, 0.5]
radius = 0.3

[grid]
nx = 32

[path]
preset = "auto"
"#;

    #[test]
    fn minimal_config_parses() {
        let config = Config::from_str(MINIMAL).expect("minimal config must parse");
        assert_eq!(config.schema, 2);
        assert_eq!(config.solver.solver_type, SolverType::Maxwell);
        assert_eq!(config.solver.precision, Precision::F64);
        assert_eq!(config.solver.polarization, Polarization::TM);
        assert_eq!(config.grid.ny(), 32);
        assert_eq!(config.total_jobs(), 1);
    }

    #[test]
    fn missing_schema_gives_migration_hint() {
        let v1 = r#"
polarization = "TM"

[bulk]
threads = 4

[ranges]
eps_bg = { min = 10.0, max = 12.0, step = 1.0 }
"#;
        let err = parse_and_validate(v1).unwrap_err();
        assert_eq!(err.len(), 1);
        let msg = &err[0].message;
        assert!(msg.contains("MIGRATION_V2"), "got: {}", msg);
        assert!(msg.contains("[bulk] is now [run]"), "got: {}", msg);
        assert!(msg.contains("[ranges]"), "got: {}", msg);
        assert!(msg.contains("polarization"), "got: {}", msg);
    }

    #[test]
    fn unknown_key_rejected_with_span() {
        let bad = MINIMAL.replace("nx = 32", "nx = 32\nnnn = 7");
        let err = parse_and_validate(&bad).unwrap_err();
        assert!(err[0].message.contains("nnn"), "got: {}", err[0].message);
        assert!(err[0].span.is_some());
    }

    #[test]
    fn physics_violations_all_collected() {
        let bad = MINIMAL
            .replace("radius = 0.3", "radius = 0.7")
            .replace("eps_bg = 12.0", "eps_bg = 0.5");
        let err = parse_and_validate(&bad).unwrap_err();
        let paths: Vec<&str> = err.iter().map(|d| d.path.as_str()).collect();
        assert!(paths.contains(&"geometry.eps_bg"), "got: {:?}", paths);
        assert!(
            paths.contains(&"geometry.atoms[0].radius"),
            "got: {:?}",
            paths
        );
    }

    #[test]
    fn rectangular_requires_b() {
        let bad = MINIMAL.replace("type = \"square\"", "type = \"rectangular\"");
        let err = parse_and_validate(&bad).unwrap_err();
        assert!(
            err.iter()
                .any(|d| d.path == "geometry.lattice.b" && d.message.contains("requires 'b'")),
            "got: {:?}",
            err
        );
    }

    #[test]
    fn oblique_alpha_deg_range() {
        let bad = MINIMAL.replace(
            "type = \"square\"",
            "type = \"oblique\"\nb = 1.2\nalpha_deg = 210.0",
        );
        let err = parse_and_validate(&bad).unwrap_err();
        assert!(
            err.iter().any(|d| d.path == "geometry.lattice.alpha_deg"),
            "got: {:?}",
            err
        );
    }

    #[test]
    fn preset_rejected_for_custom_lattice() {
        let bad = MINIMAL.replace(
            "type = \"square\"",
            "type = \"custom\"\na1 = [1.0, 0.0]\na2 = [0.4, 0.9]",
        );
        let err = parse_and_validate(&bad).unwrap_err();
        assert!(
            err.iter()
                .any(|d| d.path == "path.preset" && d.message.contains("points")),
            "got: {:?}",
            err
        );
    }

    #[test]
    fn points_mode_works_and_excludes_preset() {
        let good = MINIMAL.replace(
            "preset = \"auto\"",
            "points = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.0]]",
        );
        let config = Config::from_str(&good).expect("points mode must parse");
        assert_eq!(config.path.unwrap().points.len(), 4);

        let bad = MINIMAL.replace(
            "preset = \"auto\"",
            "preset = \"square\"\npoints = [[0.0, 0.0], [0.5, 0.0]]",
        );
        let err = parse_and_validate(&bad).unwrap_err();
        assert!(
            err.iter().any(|d| d.message.contains("mutually exclusive")),
            "got: {:?}",
            err
        );
    }

    #[test]
    fn sweep_values_are_physics_checked() {
        let bad = format!(
            "{}\n[[sweeps]]\nparameter = \"atom0.radius\"\nmin = 0.3\nmax = 0.7\nstep = 0.1\n",
            MINIMAL
        );
        let err = parse_and_validate(&bad).unwrap_err();
        assert!(
            err.iter().any(|d| d.message.contains("(0, 0.5)")),
            "got: {:?}",
            err
        );
    }

    #[test]
    fn sweep_atom_reference_must_exist() {
        let bad = format!(
            "{}\n[[sweeps]]\nparameter = \"atom2.radius\"\nvalues = [0.2]\n",
            MINIMAL
        );
        let err = parse_and_validate(&bad).unwrap_err();
        assert!(
            err.iter().any(|d| d.message.contains("atom2")),
            "got: {:?}",
            err
        );
    }

    #[test]
    fn duplicate_sweep_parameter_rejected() {
        let bad = format!(
            "{}\n[[sweeps]]\nparameter = \"eps_bg\"\nvalues = [10.0]\n\
             [[sweeps]]\nparameter = \"eps_bg\"\nvalues = [11.0]\n",
            MINIMAL
        );
        let err = parse_and_validate(&bad).unwrap_err();
        assert!(
            err.iter().any(|d| d.message.contains("duplicate")),
            "got: {:?}",
            err
        );
    }

    #[test]
    fn total_jobs_multiplies_sweeps() {
        let cfg = format!(
            "{}\n[[sweeps]]\nparameter = \"eps_bg\"\nmin = 10.0\nmax = 12.0\nstep = 1.0\n\
             [[sweeps]]\nparameter = \"polarization\"\nvalues = [\"TM\", \"TE\"]\n",
            MINIMAL
        );
        let config = Config::from_str(&cfg).unwrap();
        assert_eq!(config.total_jobs(), 6);
    }

    #[test]
    fn solver_ea_alias_and_operator_data() {
        let cfg = MINIMAL.replace(
            "[geometry]",
            "[solver]\ntype = \"ea\"\npolarization = \"TE\"\n\n[geometry]",
        );
        let config = Config::from_str(&cfg).expect("ea alias must parse");
        assert_eq!(config.solver.solver_type, SolverType::OperatorData);
    }

    #[test]
    fn selective_output_validated() {
        let bad = format!("{}\n[output]\nmode = \"selective\"\n", MINIMAL);
        let err = parse_and_validate(&bad).unwrap_err();
        assert!(
            err.iter().any(|d| d.path.starts_with("output.selective")),
            "got: {:?}",
            err
        );
    }

    #[test]
    fn serialize_parse_roundtrip() {
        let config = Config::from_str(MINIMAL).unwrap();
        let serialized = toml::to_string(&config).expect("serialize");
        let reparsed = Config::from_str(&serialized).expect("reparse serialized config");
        assert_eq!(reparsed.geometry.eps_bg, config.geometry.eps_bg);
        assert_eq!(reparsed.geometry.lattice.kind, LatticeKind::Square);
        assert_eq!(reparsed.grid.nx, 32);
    }

    #[test]
    fn dielectric_none_disables_smoothing() {
        let cfg = format!("{}\n[dielectric]\nsmoothing = \"none\"\n", MINIMAL);
        let config = Config::from_str(&cfg).unwrap();
        assert!(!config.dielectric.to_options().smoothing_enabled());
    }
}
