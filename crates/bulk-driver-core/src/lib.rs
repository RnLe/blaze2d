//! Platform-agnostic core types and logic for the Blaze bulk driver.
//!
//! This crate provides the shared foundation for every Blaze2D consumer
//! (CLI, native bulk driver, Python bindings, WASM):
//!
//! - **Configuration**: the schema v2 [`Config`] (one TOML schema for
//!   everything; a config without `[[sweeps]]` is a single job), with
//!   parse-time physics validation and structured [`Diagnostic`]s
//! - **Sweep types**: `SweepSpec`, `SweepDimension`, `SweepValue`
//! - **Result types**: `CompactBandResult`, `MaxwellResult`, `OperatorDataResult`
//! - **Job expansion**: [`expand_jobs`] turns a validated config into concrete
//!   job specifications
//! - **Filtering**: `SelectiveFilter` for k-point and band filtering
//!
//! # Configuration format (schema v2)
//!
//! ```toml
//! schema = 2
//!
//! [solver]
//! polarization = "TM"
//!
//! [geometry]
//! eps_bg = 12.0
//!
//! [geometry.lattice]
//! type = "square"
//!
//! [[geometry.atoms]]
//! pos = [0.5, 0.5]
//! radius = 0.3
//!
//! [grid]
//! nx = 32
//!
//! [path]
//! preset = "auto"
//!
//! # Optional sweeps: first entry = outermost loop
//! [[sweeps]]
//! parameter = "atom0.radius"
//! min = 0.2
//! max = 0.4
//! step = 0.1
//! ```
//!
//! See `docs/MIGRATION_V2.md` at the repository root for the v1 -> v2 rename
//! table.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     bulk-driver-core                            │
//! │  (Platform-agnostic: config, types, expansion, filtering)       │
//! └───────────────────────────┬─────────────────────────────────────┘
//!                             │
//!           ┌─────────────────┼─────────────────┐
//!           │                 │                 │
//!           ▼                 │                 ▼
//! ┌─────────────────┐         │       ┌─────────────────┐
//! │   bulk-driver   │         │       │  backend-wasm   │
//! │  (Native/Rayon) │         │       │ (Single-thread) │
//! └─────────────────┘         │       └─────────────────┘
//!           │
//!           ▼
//! ┌─────────────────┐
//! │ Python bindings │
//! └─────────────────┘
//! ```

pub mod config;
pub mod expansion;
pub mod filter;
pub mod result;

// Re-export all public types for convenient access
pub use config::{
    BaseAtom, BulkConfig, Config, ConfigError, DEFAULT_POINTS_PER_SEGMENT, Diagnostic,
    DielectricSection, GeometrySection, GridSection, LatticeKind, LatticeSection,
    OperatorDataDriverConfig, OutputConfig, OutputMode, PathPresetSpec, PathSection, Precision,
    RESOLUTION_RANGE, RunSection, SCHEMA_VERSION, SelectiveSpec, SmoothingKind, SolverSection,
    SolverType, SweepDimension, SweepSpec, SweepValue, VALID_ATOM_PROPS, VALID_GLOBAL_PARAMS,
    parse_and_validate, parse_atom_path, parse_swept_lattice_kind, validate_parameter_path,
};

pub use expansion::{
    AtomParams, ExpandedJob, ExpandedJobType, JobParams, OperatorDataJobSpec, expand_jobs,
};

pub use filter::SelectiveFilter;

pub use result::{
    CompactBandResult, CompactResultType, ComplexPair, MaxwellResult, OperatorDataResult,
};
