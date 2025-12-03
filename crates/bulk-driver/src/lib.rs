//! MPB2D Bulk Driver - Smart multi-threaded parameter sweep driver.
//!
//! This crate provides a high-performance driver for running many photonic crystal
//! band structure calculations in parallel, as well as Envelope Approximation (EA)
//! eigenproblems for moiré lattice research. It handles:
//!
//! - **Parameter space definition**: Define ranges for radius, epsilon, lattice type,
//!   multi-atom basis positions, resolution, and polarization
//! - **Job expansion**: Automatically expands parameter ranges into individual jobs
//! - **Thread pool management**: Efficient parallel execution with configurable thread count
//! - **Progress tracking**: Real-time progress logging without per-thread noise
//! - **Output batching**: High-performance CSV output in full or selective mode
//!
//! # Solver Types
//!
//! The bulk driver supports two solver types:
//!
//! - **Maxwell** (default): Photonic crystal band structure calculations
//! - **EA**: Envelope Approximation eigenproblems for moiré lattices
//!
//! # Usage
//!
//! The bulk driver reads a specially-formatted TOML configuration file that specifies
//! parameter ranges instead of single values. The file must include `[bulk]` section
//! to be recognized as a bulk request.
//!
//! ## Output Modes
//!
//! - **Full Mode**: One CSV file per solver run, containing complete band structure data.
//! - **Selective Mode**: Single merged CSV with only specified k-points and bands.
//!
//! ## I/O Modes
//!
//! - **Sync**: Traditional synchronous I/O (default, current behavior)
//! - **Batch**: Buffer results in memory (~10 MB) and write in background thread
//! - **Stream**: Real-time emission for live consumers (Python plotting, WASM)
//!
//! # Example: Batch Mode
//!
//! ```ignore
//! use mpb2d_bulk_driver::{BulkDriver, BatchConfig, OutputChannel};
//! use std::sync::Arc;
//!
//! let driver = BulkDriver::new(config, None);
//! let stats = driver.run_batched(BatchConfig::default())?;
//! ```
//!
//! # Example: Streaming Mode
//!
//! ```ignore
//! let (receiver, handle) = driver.run_streaming()?;
//! for result in receiver {
//!     println!("Job {} complete", result.job_index);
//! }
//! handle.join().unwrap()?;
//! ```

pub mod adaptive;
pub mod batch;
pub mod channel;
pub mod config;
pub mod driver;
pub mod expansion;
pub mod output;
pub mod stream;

// Re-export adaptive thread management
pub use adaptive::{AdaptiveConfig, AdaptiveThreadManager};

// Re-export batch I/O
pub use batch::BatchChannel;

// Re-export channel types
pub use channel::{
    BackpressurePolicy, BatchConfig, ChannelError, ChannelStats, CompactBandResult,
    CompactBandResultExt, CompactResultType, EAResult, MaxwellResult, OutputChannel,
    OutputChannelSink, StreamConfig,
};

// Re-export configuration
pub use config::{
    BatchSettings, BulkConfig, BulkConfigNativeExt, EAConfig, IoMode, OutputConfig, OutputMode,
    ParameterRange, RangeSpec, SelectiveSpec, SolverSection, SolverType, SweepSpec, SweepValue,
};

// Re-export driver
pub use driver::{
    BulkDriver, DriverError, DriverStats, EAJobResult, JobError, JobResult, JobResultType,
    ThreadMode,
};

// Re-export job expansion
pub use expansion::{expand_jobs, EAJobSpec, ExpandedJob, ExpandedJobType, JobParams};

// Re-export legacy output writer
pub use output::OutputWriter;

// Re-export streaming
pub use stream::{
    CallbackSubscriber, ChannelSubscriber, CollectingSubscriber, FilteredStreamChannel,
    SelectiveFilter, SharedStreamChannel, StreamChannel, Subscriber,
};
