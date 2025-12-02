//! Platform-agnostic core types and logic for MPB2D bulk driver.
//!
//! This crate provides the shared foundation for both native and WASM bulk drivers:
//!
//! - **Configuration types**: `BulkConfig`, `SolverType`, `IoMode`, `OutputMode`
//! - **Result types**: `CompactBandResult`, `MaxwellResult`, `EAResult`
//! - **Job expansion**: Convert parameter ranges into individual job specifications
//! - **Filtering**: `SelectiveFilter` for k-point and band filtering
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
//!
//! # Usage
//!
//! This crate is not meant to be used directly. Instead, use either:
//! - `mpb2d-bulk-driver` for native applications (with threading)
//! - `mpb2d-bulk-driver-wasm` for WebAssembly applications

pub mod config;
pub mod expansion;
pub mod filter;
pub mod result;

// Re-export all public types for convenient access
pub use config::{
    AtomRanges, BaseAtom, BaseGeometry, BaseLattice, BatchSettings, BulkConfig, BulkSection,
    ConfigError, EAConfig, IoMode, LatticeTypeSpec, OutputConfig, OutputMode, ParameterRange,
    RangeSpec, SelectiveSpec, SolverSection, SolverType, ValueList,
};

pub use expansion::{
    expand_jobs, AtomParams, EAJobSpec, ExpandedJob, ExpandedJobType, JobParams,
};

pub use filter::SelectiveFilter;

pub use result::{
    CompactBandResult, CompactResultType, EAResult, MaxwellResult,
};
