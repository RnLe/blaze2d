//! Platform-agnostic core types and logic for MPB2D bulk driver.
//!
//! This crate provides the shared foundation for both native and WASM bulk drivers:
//!
//! - **Configuration types**: `BulkConfig`, `SolverType`, `IoMode`, `OutputMode`
//! - **Sweep types**: `SweepSpec`, `SweepDimension`, `SweepValue` for ordered sweeps
//! - **Result types**: `CompactBandResult`, `MaxwellResult`, `EAResult`
//! - **Job expansion**: Convert parameter ranges into individual job specifications
//! - **Filtering**: `SelectiveFilter` for k-point and band filtering
//!
//! # Configuration Formats
//!
//! ## New Format (Recommended): Ordered Sweeps
//!
//! Use `[[sweeps]]` arrays for user-controlled loop nesting order:
//!
//! ```toml
//! [bulk]
//! verbose = true
//!
//! # First sweep = outermost loop, last = innermost
//! [[sweeps]]
//! parameter = "atom0.radius"
//! min = 0.2
//! max = 0.4  
//! step = 0.1
//!
//! [[sweeps]]
//! parameter = "eps_bg"
//! min = 10.0
//! max = 12.0
//! step = 1.0
//!
//! [defaults.geometry]
//! eps_bg = 12.0
//! # ... rest of config
//! ```
//!
//! ## Legacy Format
//!
//! The `[ranges]` section is still supported but uses hardcoded loop order.
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
//! │   bulk-driver   │         │       │ bulk-driver-wasm│
//! │  (Native/Rayon) │         │       │ (Single-thread) │
//! └─────────────────┘         │       └─────────────────┘
//!           │                 │                 │
//!           ▼                 │                 ▼
//! ┌─────────────────┐         │       ┌─────────────────┐
//! │ Python bindings │         │       │  backend-wasm   │
//! └─────────────────┘         │       └─────────────────┘
//! ```

pub mod config;
pub mod expansion;
pub mod filter;
pub mod result;

// Re-export all public types for convenient access
pub use config::{
    AtomRanges, BaseAtom, BaseGeometry, BaseLattice, BatchSettings, BulkConfig, BulkSection,
    ConfigError, DefaultsConfig, EAConfig, IoMode, LatticeTypeSpec, OutputConfig, OutputMode,
    ParameterRange, RangeSpec, SelectiveSpec, SolverSection, SolverType, SweepDimension,
    SweepSpec, SweepValue, ValueList, parse_atom_path, validate_parameter_path,
};

pub use expansion::{
    expand_jobs, AtomParams, EAJobSpec, ExpandedJob, ExpandedJobType, JobParams,
};

pub use filter::SelectiveFilter;

pub use result::{
    CompactBandResult, CompactResultType, ComplexPair, EAResult, MaxwellResult,
};
