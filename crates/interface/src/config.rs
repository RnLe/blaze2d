use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{CONFIG_SCHEMA, Diagnostic, InterfaceResult};

fn schema_version() -> String { CONFIG_SCHEMA.into() }
fn dimension() -> usize { 2 }
fn one() -> f64 { 1.0 }
fn band_count() -> usize { 8 }
fn intervals() -> usize { 15 }
fn retained() -> usize { 4 }
fn remote() -> usize { 8 }
fn fd_step() -> f64 { 0.001 }
fn mesh_size() -> usize { 3 }
fn interface_tolerance() -> f64 { 1e-6 }
fn zero_point() -> Vec<f64> { vec![0.0, 0.0] }
fn zero_registry() -> Vec<Vec<f64>> { vec![zero_point()] }
fn default_quantities() -> Vec<Quantity> { vec![Quantity::Velocity, Quantity::MassTensor] }

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub schema: String,
    pub task: Task,
    #[serde(default = "dimension")]
    pub dimension: usize,
    #[serde(default)]
    pub polarization: Polarization,
    pub geometry: Geometry,
    #[serde(default)]
    pub grid: Grid,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bands: Option<Bands>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operators: Option<Operators>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sweeps: Vec<Sweep>,
    #[serde(default)]
    pub eigensolver: Eigensolver,
    #[serde(default)]
    pub dielectric: Dielectric,
    #[serde(default)]
    pub results: Results,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            schema: schema_version(), task: Task::Bands, dimension: 2,
            polarization: Polarization::TM,
            geometry: Geometry { background_epsilon: 1.0, lattice: Lattice::default(),
                objects: vec![Object { name: "rod".into(), kind: ObjectKind::Circle,
                    center: zero_point(), radius: 0.2, epsilon: 8.9 }] },
            grid: Grid::default(), bands: Some(Bands::default()), operators: None,
            sweeps: vec![], eigensolver: Eigensolver::default(),
            dielectric: Dielectric::default(), results: Results::default(),
        }
    }
}

impl Config {
    pub fn from_toml(source: &str) -> InterfaceResult<Self> {
        toml::from_str(source).map_err(|error: toml::de::Error| Diagnostic {
            code: "invalid_toml".into(), path: String::new(),
            message: error.message().into(), span: error.span().map(|r| [r.start, r.end]),
        })
    }

    pub fn from_value(value: serde_json::Value) -> InterfaceResult<Self> {
        serde_json::from_value(value).map_err(|e| Diagnostic::new("invalid_config", "", e.to_string()))
    }

    pub fn to_toml(&self) -> InterfaceResult<String> {
        toml::to_string_pretty(self).map_err(|e| Diagnostic::new("serialization", "", e.to_string()))
    }
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Task { Bands, Operators }

#[derive(Debug, Default, Copy, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub enum Polarization { #[default] TM, TE }

#[derive(Debug, Default, Copy, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Precision { #[default] F64, F32 }

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Geometry {
    #[serde(default = "one")]
    pub background_epsilon: f64,
    pub lattice: Lattice,
    #[serde(default)]
    pub objects: Vec<Object>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Lattice {
    #[serde(rename = "type")]
    pub kind: LatticeKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub a: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub b: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub angle_deg: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vectors: Option<Vec<Vec<f64>>>,
}

impl Default for Lattice {
    fn default() -> Self { Self { kind: LatticeKind::Square, a: Some(1.0), b: None, angle_deg: None, vectors: None } }
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LatticeKind { Square, Triangular, Rectangular, Oblique, Custom }

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Object {
    pub name: String,
    pub kind: ObjectKind,
    #[serde(default = "zero_point")]
    pub center: Vec<f64>,
    pub radius: f64,
    pub epsilon: f64,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ObjectKind { Circle }

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Grid { #[serde(default)] pub resolution: Resolution }

impl Default for Grid { fn default() -> Self { Self { resolution: Resolution::default() } } }

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(untagged)]
pub enum Resolution { Uniform(usize), Axes(Vec<usize>) }
impl Default for Resolution { fn default() -> Self { Self::Uniform(32) } }

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Bands {
    #[serde(default = "band_count")]
    pub count: usize,
    #[serde(default)]
    pub path: KPath,
    #[serde(default = "default_tracking")]
    pub tracking: bool,
    #[serde(default)]
    pub symmetry: Symmetry,
    /// Selecting a metric ball changes the finite numerical pencil.
    #[serde(default, skip_serializing_if="Discretization::is_fft_rectangle")]
    pub discretization: Discretization,
}
fn default_tracking() -> bool { true }
impl Default for Bands {
    fn default() -> Self { Self { count: band_count(), path: KPath::default(), tracking: true, symmetry: Symmetry::default(), discretization: Discretization::default() } }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag="kind", rename_all="snake_case", deny_unknown_fields)]
pub enum Discretization {
    #[default]
    FftRectangle,
    MetricBall {
        /// Cartesian angular cutoff in inverse reference length; no implicit 2*pi.
        angular_cutoff: f64,
        #[serde(default="support_coefficients")]
        max_coefficients: usize,
        #[serde(default="support_embedding")]
        max_embedding_points: usize,
        #[serde(default="support_enumeration")]
        max_enumerated_indices: usize,
        /// Numerical payload admission, not a process RSS limit.
        #[serde(default="support_payload")]
        max_estimated_payload_bytes: usize,
        #[serde(default="support_dense")]
        max_dense_dimension: usize,
        #[serde(default="support_rounds")]
        max_allocation_rounds: usize,
    },
}
impl Discretization { pub fn is_fft_rectangle(&self)->bool {matches!(self,Self::FftRectangle)} }
fn support_coefficients()->usize {4096}
fn support_embedding()->usize {1_048_576}
fn support_enumeration()->usize {4_194_304}
fn support_payload()->usize {512*1024*1024}
fn support_dense()->usize {512}
fn support_rounds()->usize {32}

#[derive(Debug, Default, Copy, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all="snake_case")]
pub enum KReuse { #[default] Off, Identical, Unitary, TimeReversal }
#[derive(Debug, Default, Copy, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all="snake_case")]
pub enum KTraversal { #[default] Given, Nearest }
#[derive(Debug, Default, Copy, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all="snake_case")]
pub enum SeedTransfer { #[default] Copy, RankAware }
#[derive(Debug, Default, Copy, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all="snake_case")]
pub enum SectorPolicy { #[default] Off, ConservativeMirror, AdaptiveMirror, ConservativeGroup, AdaptiveGroup }
#[derive(Debug, Default, Copy, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all="snake_case")]
pub enum SymmetryArithmetic { #[default] SpatialComplex, FourierComplex, PtReal }
#[derive(Debug, Default, Copy, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all="snake_case")]
pub enum SectorStorage { #[default] FullGrid, Orbit }
#[derive(Debug, Default, Copy, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all="snake_case")]
pub enum SectorKernel { #[default] FullGrid, MirrorTm, MirrorTmTe }
fn projection_interval()->usize {1}
fn symmetry_cache_bytes()->usize {64*1024*1024}
fn symmetry_sector_bytes()->usize {blaze2d_core::symmetry::sectors::memory::DEFAULT_LIMIT_BYTES}
fn is_default_sector_bytes(value:&usize)->bool {*value==symmetry_sector_bytes()}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CoarseStart {
    pub resolution: [usize; 2],
    /// Coarse, lifted and prepared seed fields; solver and FFT scratch are extra.
    #[serde(default="symmetry_cache_bytes")]
    pub max_seed_field_bytes: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Symmetry {
    #[serde(default)]
    pub k_reuse: KReuse,
    #[serde(default)]
    pub k_traversal: KTraversal,
    #[serde(default="symmetry_cache_bytes")]
    pub max_traversal_field_bytes: usize,
    #[serde(default="symmetry_cache_bytes")]
    pub max_cache_bytes: usize,
    /// Numerical payload admission for rectangular sector plans and solves.
    /// Metric Fourier supports use bands.discretization's budget instead.
    #[serde(default="symmetry_sector_bytes",skip_serializing_if="is_default_sector_bytes")]
    pub max_sector_workspace_bytes: usize,
    #[serde(default,skip_serializing_if="Option::is_none")]
    pub plane_group: Option<String>,
    #[serde(default)]
    pub seed_transfer: SeedTransfer,
    #[serde(default,skip_serializing_if="Option::is_none")]
    pub coarse_start: Option<CoarseStart>,
    #[serde(default)]
    pub sectors: SectorPolicy,
    #[serde(default,skip_serializing_if="Option::is_none")]
    pub sector_guard_budget: Option<usize>,
    #[serde(default)]
    pub sector_storage: SectorStorage,
    #[serde(default)]
    pub sector_kernel: SectorKernel,
    #[serde(default)]
    pub reconstruct_partners: bool,
    #[serde(default)]
    pub share_sector_workspace: bool,
    #[serde(default="projection_interval")]
    pub correction_projection_interval: usize,
    #[serde(default)]
    pub arithmetic: SymmetryArithmetic,
}
impl Default for Symmetry {fn default()->Self {Self {k_reuse:KReuse::Off,k_traversal:KTraversal::Given,max_traversal_field_bytes:symmetry_cache_bytes(),max_cache_bytes:symmetry_cache_bytes(),max_sector_workspace_bytes:symmetry_sector_bytes(),plane_group:None,seed_transfer:SeedTransfer::Copy,coarse_start:None,sectors:SectorPolicy::Off,sector_guard_budget:None,sector_storage:SectorStorage::FullGrid,sector_kernel:SectorKernel::FullGrid,reconstruct_partners:false,share_sector_workspace:false,correction_projection_interval:1,arithmetic:SymmetryArithmetic::SpatialComplex}}}

#[derive(Debug, Default, Copy, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum KCoordinates { #[default] ReciprocalFractional, CartesianAngular }

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct KPath {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vertices: Option<Vec<Vec<f64>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub points: Option<Vec<Vec<f64>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intervals_per_segment: Option<usize>,
    #[serde(default)]
    pub basis: KCoordinates,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub labels: Vec<String>,
}
impl Default for KPath {
    fn default() -> Self {
        Self { preset: None, vertices: None, points: None,
            intervals_per_segment: Some(intervals()), basis: KCoordinates::default(), labels: vec![] }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Operators {
    #[serde(default)]
    pub band_lo: usize,
    #[serde(default = "retained")]
    pub retained_bands: usize,
    #[serde(default = "remote")]
    pub remote_bands: usize,
    pub k_point: KPoint,
    #[serde(default = "default_quantities")]
    pub quantities: Vec<Quantity>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub registry: Option<Registry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub k_stencil: Option<KStencil>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<Reference>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fail_on_residual: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct KPoint { pub value: Vec<f64>, pub basis: KCoordinates }

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag="source", rename_all="snake_case", deny_unknown_fields)]
pub enum Reference { External }

#[derive(Debug, Copy, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Quantity { Velocity, MassTensor, RDerivatives, BornHuang, SlowCoefficient, ExactTm, Overlap }

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Registry {
    pub object: String,
    #[serde(default = "zero_registry")]
    pub points: Vec<Vec<f64>>,
    #[serde(default = "fd_step")]
    pub fd_step: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct KStencil { pub points_per_axis: usize, pub half_width: f64 }

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Sweep {
    pub name: String,
    pub target: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linspace: Option<Linspace>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Linspace { pub start: f64, pub stop: f64, pub count: usize }

#[derive(Debug, Default, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Eigensolver {
    #[serde(default)]
    pub precision: Precision,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tolerance: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_iterations: Option<usize>,
    #[serde(default)]
    pub block_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Dielectric {
    #[serde(default)]
    pub source: DielectricSource,
    #[serde(default)]
    pub smoothing: Smoothing,
    #[serde(default = "mesh_size")]
    pub mesh_size: usize,
    #[serde(default = "interface_tolerance")]
    pub interface_tolerance: f64,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all="snake_case")]
pub enum DielectricSource { #[default] Geometry, External }
impl Default for Dielectric {
    fn default() -> Self {
        Self { source: DielectricSource::Geometry, smoothing: Smoothing::default(), mesh_size: mesh_size(), interface_tolerance: interface_tolerance() }
    }
}

#[derive(Debug, Default, Copy, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Smoothing { #[default] Analytic, Subgrid, None }

#[derive(Debug, Default, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Results { #[serde(default)] pub eigenvectors: bool }
