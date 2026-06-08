//! Job expansion from parameter ranges.
//!
//! Thin re-export of the platform-agnostic expansion logic from
//! `blaze2d_bulk_driver_core`. Native consumers import from here so the
//! native and WASM front-ends share a single source of truth for job
//! definitions and ordering.

pub use blaze2d_bulk_driver_core::expansion::{
    AtomParams, OperatorDataJobSpec, ExpandedJob, ExpandedJobType, JobParams, expand_jobs,
};
