//! MPB2D Bulk Driver - Smart multi-threaded parameter sweep driver.
//!
//! This crate provides a high-performance driver for running many photonic crystal
//! band structure calculations in parallel. It handles:
//!
//! - **Parameter space definition**: Define ranges for radius, epsilon, lattice type,
//!   multi-atom basis positions, resolution, and polarization
//! - **Job expansion**: Automatically expands parameter ranges into individual jobs
//! - **Thread pool management**: Efficient parallel execution with configurable thread count
//! - **Progress tracking**: Real-time progress logging without per-thread noise
//! - **Output batching**: High-performance CSV output in full or selective mode
//!
//! # Usage
//!
//! The bulk driver reads a specially-formatted TOML configuration file that specifies
//! parameter ranges instead of single values. The file must include `[bulk]` section
//! to be recognized as a bulk request.
//!
//! ## Full Mode
//!
//! Outputs one CSV file per solver run, containing complete band structure data.
//!
//! ## Selective Mode
//!
//! Outputs a single merged CSV file containing only specified k-points and bands,
//! with swept parameters as columns for easy analysis.

pub mod adaptive;
pub mod config;
pub mod driver;
pub mod expansion;
pub mod output;

pub use adaptive::{AdaptiveConfig, AdaptiveThreadManager};
pub use config::{BulkConfig, OutputConfig, OutputMode, ParameterRange, RangeSpec, SelectiveSpec};
pub use driver::{BulkDriver, ThreadMode};
pub use expansion::{expand_jobs, ExpandedJob};
pub use output::OutputWriter;
