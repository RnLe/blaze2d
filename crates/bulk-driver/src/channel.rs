//! Output channel abstraction for batch and stream modes.
//!
//! This module provides a unified interface for outputting band structure results
//! in different modes:
//!
//! - **Batch Mode**: Buffer results in memory and write to disk in large chunks,
//!   minimizing I/O interference with the solver.
//! - **Stream Mode**: Emit results in real-time for live consumers (Python plots,
//!   WASM/React components).
//! - **Null Mode**: Discard output for pure benchmarking.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use thiserror::Error;

use crate::expansion::JobParams;

// ============================================================================
// Compact Band Result
// ============================================================================

/// Serializable band structure result optimized for efficient transfer.
///
/// This is a self-contained representation of a single band structure calculation,
/// including all metadata needed for output and analysis.
///
/// ## Memory Layout
///
/// For a typical calculation with 10 bands and 100 k-points:
/// - `k_path`: 100 × 2 × 8 = 1,600 bytes
/// - `distances`: 100 × 8 = 800 bytes
/// - `bands`: 100 × 10 × 8 = 8,000 bytes
/// - Metadata: ~200 bytes
/// - **Total**: ~10.6 KB per result
///
/// A 10 MB buffer can hold approximately 900-1000 results.
#[derive(Debug, Clone)]
pub struct CompactBandResult {
    /// Job index (matches ExpandedJob.index)
    pub job_index: usize,

    /// Parameter values used for this job
    pub params: JobParams,

    /// The result type (Maxwell with k-path data, or EA with eigenvalues only)
    pub result_type: CompactResultType,
}

/// Type of compact result.
#[derive(Debug, Clone)]
pub enum CompactResultType {
    /// Maxwell result with full band structure
    Maxwell(MaxwellResult),
    /// EA result with eigenvalues only (no k-path)
    EA(EAResult),
}

/// Maxwell band structure result.
#[derive(Debug, Clone)]
pub struct MaxwellResult {
    /// K-path in fractional coordinates
    pub k_path: Vec<[f64; 2]>,

    /// Cumulative distance along k-path
    pub distances: Vec<f64>,

    /// Computed eigenfrequencies organized as bands[k_index][band_index]
    /// Values are normalized frequencies (ω/2π)
    pub bands: Vec<Vec<f64>>,
}

/// EA eigenvalue result.
#[derive(Debug, Clone)]
pub struct EAResult {
    /// Computed eigenvalues
    pub eigenvalues: Vec<f64>,

    /// Number of iterations taken
    pub n_iterations: usize,

    /// Whether convergence was achieved
    pub converged: bool,
}

impl CompactBandResult {
    /// Create from a job result and expanded job.
    pub fn from_job_result(
        job: &crate::expansion::ExpandedJob,
        result: &crate::driver::JobResult,
    ) -> Self {
        let result_type = match &result.result {
            crate::driver::JobResultType::Maxwell(band_result) => {
                // Normalize frequencies (divide by 2π)
                let bands: Vec<Vec<f64>> = band_result
                    .bands
                    .iter()
                    .map(|k_bands| {
                        k_bands
                            .iter()
                            .map(|omega| omega / (2.0 * std::f64::consts::PI))
                            .collect()
                    })
                    .collect();

                CompactResultType::Maxwell(MaxwellResult {
                    k_path: band_result.k_path.clone(),
                    distances: band_result.distances.clone(),
                    bands,
                })
            }
            crate::driver::JobResultType::EA(ea_result) => {
                CompactResultType::EA(EAResult {
                    eigenvalues: ea_result.eigenvalues.clone(),
                    n_iterations: ea_result.n_iterations,
                    converged: ea_result.converged,
                })
            }
        };

        Self {
            job_index: job.index,
            params: job.params.clone(),
            result_type,
        }
    }

    /// Approximate size in bytes for buffer management.
    pub fn approx_size(&self) -> usize {
        // Base struct overhead
        let base = std::mem::size_of::<Self>();

        // params (rough estimate)
        let params_size = 200;

        let result_size = match &self.result_type {
            CompactResultType::Maxwell(m) => {
                // k_path: Vec<[f64; 2]>
                let k_path_size = m.k_path.len() * 16;
                // distances: Vec<f64>
                let distances_size = m.distances.len() * 8;
                // bands: Vec<Vec<f64>>
                let bands_size: usize = m.bands.iter().map(|b| b.len() * 8 + 24).sum();
                k_path_size + distances_size + bands_size
            }
            CompactResultType::EA(ea) => {
                // eigenvalues: Vec<f64>
                ea.eigenvalues.len() * 8 + 24
            }
        };

        base + params_size + result_size
    }

    /// Number of k-points in this result (Maxwell only).
    pub fn num_k_points(&self) -> usize {
        match &self.result_type {
            CompactResultType::Maxwell(m) => m.k_path.len(),
            CompactResultType::EA(_) => 1, // EA has no k-path concept
        }
    }

    /// Number of bands computed.
    pub fn num_bands(&self) -> usize {
        match &self.result_type {
            CompactResultType::Maxwell(m) => m.bands.first().map(|b| b.len()).unwrap_or(0),
            CompactResultType::EA(ea) => ea.eigenvalues.len(),
        }
    }

    /// Get k_path if this is a Maxwell result (for legacy compatibility).
    pub fn k_path(&self) -> Option<&Vec<[f64; 2]>> {
        match &self.result_type {
            CompactResultType::Maxwell(m) => Some(&m.k_path),
            CompactResultType::EA(_) => None,
        }
    }

    /// Get distances if this is a Maxwell result (for legacy compatibility).
    pub fn distances(&self) -> Option<&Vec<f64>> {
        match &self.result_type {
            CompactResultType::Maxwell(m) => Some(&m.distances),
            CompactResultType::EA(_) => None,
        }
    }

    /// Get bands if this is a Maxwell result (for legacy compatibility).
    pub fn bands(&self) -> Option<&Vec<Vec<f64>>> {
        match &self.result_type {
            CompactResultType::Maxwell(m) => Some(&m.bands),
            CompactResultType::EA(_) => None,
        }
    }
}

// ============================================================================
// Channel Configuration
// ============================================================================

/// Configuration for batch mode output.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Buffer size in bytes before triggering a flush (default: 10 MB)
    pub buffer_size_bytes: usize,

    /// Optional time-based flush interval
    pub flush_interval: Option<Duration>,

    /// Output format
    pub format: OutputFormat,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            buffer_size_bytes: 10 * 1024 * 1024, // 10 MB
            flush_interval: None,
            format: OutputFormat::Csv,
        }
    }
}

/// Configuration for stream mode output.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Channel capacity for buffering between producer and consumer
    pub channel_capacity: usize,

    /// Backpressure policy when channel is full
    pub backpressure: BackpressurePolicy,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            channel_capacity: 64,
            backpressure: BackpressurePolicy::Block,
        }
    }
}

/// Policy for handling backpressure when consumers are slow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackpressurePolicy {
    /// Block the producer until space is available (default)
    Block,
    /// Drop oldest results (ring buffer semantics)
    DropOldest,
    /// Drop newest results if channel is full
    DropNewest,
}

/// Output format for batch mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// CSV format (current default)
    Csv,
    /// Binary format (future: more compact)
    Binary,
    /// JSON format (for interop)
    Json,
}

// ============================================================================
// Channel Statistics
// ============================================================================

/// Statistics from channel operations.
#[derive(Debug, Clone, Default)]
pub struct ChannelStats {
    /// Total results sent through channel
    pub results_sent: usize,

    /// Total bytes written (for batch mode)
    pub bytes_written: usize,

    /// Number of flush operations (for batch mode)
    pub flush_count: usize,

    /// Results dropped due to backpressure (for stream mode)
    pub results_dropped: usize,

    /// Total write time (for batch mode)
    pub total_write_time: Duration,
}

impl ChannelStats {
    /// Merge with another stats instance.
    pub fn merge(&mut self, other: &ChannelStats) {
        self.results_sent += other.results_sent;
        self.bytes_written += other.bytes_written;
        self.flush_count += other.flush_count;
        self.results_dropped += other.results_dropped;
        self.total_write_time += other.total_write_time;
    }
}

// ============================================================================
// Channel Errors
// ============================================================================

/// Errors that can occur during channel operations.
#[derive(Debug, Error)]
pub enum ChannelError {
    /// Channel is closed
    #[error("channel closed")]
    Closed,

    /// Channel is full (non-blocking send)
    #[error("channel full")]
    Full,

    /// I/O error during write
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Background writer thread panicked
    #[error("writer thread panicked")]
    WriterPanic,
}

// ============================================================================
// Output Channel Trait
// ============================================================================

/// Trait for output channels supporting different I/O modes.
///
/// This abstraction allows the driver to use batch, stream, or null output
/// without knowing the concrete implementation.
pub trait OutputChannelSink: Send + Sync {
    /// Send a result through the channel.
    fn send(&self, result: CompactBandResult) -> Result<(), ChannelError>;

    /// Force flush any buffered data (batch mode).
    fn flush(&self) -> Result<(), ChannelError>;

    /// Close the channel and get final statistics.
    fn close(&self) -> Result<ChannelStats, ChannelError>;

    /// Check if the channel is still open.
    fn is_open(&self) -> bool;
}

// ============================================================================
// Null Channel (for benchmarking)
// ============================================================================

/// Null output channel that discards all results.
///
/// Used for pure performance benchmarking without I/O overhead.
pub struct NullChannel {
    count: AtomicUsize,
    closed: std::sync::atomic::AtomicBool,
}

impl NullChannel {
    /// Create a new null channel.
    pub fn new() -> Self {
        Self {
            count: AtomicUsize::new(0),
            closed: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

impl Default for NullChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputChannelSink for NullChannel {
    fn send(&self, _result: CompactBandResult) -> Result<(), ChannelError> {
        if self.closed.load(Ordering::Relaxed) {
            return Err(ChannelError::Closed);
        }
        self.count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn flush(&self) -> Result<(), ChannelError> {
        Ok(())
    }

    fn close(&self) -> Result<ChannelStats, ChannelError> {
        self.closed.store(true, Ordering::Relaxed);
        Ok(ChannelStats {
            results_sent: self.count.load(Ordering::Relaxed),
            ..Default::default()
        })
    }

    fn is_open(&self) -> bool {
        !self.closed.load(Ordering::Relaxed)
    }
}

// ============================================================================
// Output Channel Enum
// ============================================================================

/// Unified output channel supporting multiple I/O modes.
///
/// This enum dispatches to the appropriate implementation based on the
/// configured mode.
pub enum OutputChannel {
    /// Batch mode: buffer and write in chunks
    Batch(Arc<dyn OutputChannelSink>),
    /// Stream mode: real-time emission
    Stream(Arc<dyn OutputChannelSink>),
    /// Null mode: discard output
    Null(Arc<NullChannel>),
}

impl OutputChannel {
    /// Create a null channel for benchmarking.
    pub fn null() -> Self {
        OutputChannel::Null(Arc::new(NullChannel::new()))
    }

    /// Send a result through the channel.
    pub fn send(&self, result: CompactBandResult) -> Result<(), ChannelError> {
        match self {
            OutputChannel::Batch(ch) => ch.send(result),
            OutputChannel::Stream(ch) => ch.send(result),
            OutputChannel::Null(ch) => ch.send(result),
        }
    }

    /// Force flush any buffered data.
    pub fn flush(&self) -> Result<(), ChannelError> {
        match self {
            OutputChannel::Batch(ch) => ch.flush(),
            OutputChannel::Stream(ch) => ch.flush(),
            OutputChannel::Null(ch) => ch.flush(),
        }
    }

    /// Close the channel and get statistics.
    pub fn close(&self) -> Result<ChannelStats, ChannelError> {
        match self {
            OutputChannel::Batch(ch) => ch.close(),
            OutputChannel::Stream(ch) => ch.close(),
            OutputChannel::Null(ch) => ch.close(),
        }
    }

    /// Check if the channel is still open.
    pub fn is_open(&self) -> bool {
        match self {
            OutputChannel::Batch(ch) => ch.is_open(),
            OutputChannel::Stream(ch) => ch.is_open(),
            OutputChannel::Null(ch) => ch.is_open(),
        }
    }
}

impl Clone for OutputChannel {
    fn clone(&self) -> Self {
        match self {
            OutputChannel::Batch(ch) => OutputChannel::Batch(Arc::clone(ch)),
            OutputChannel::Stream(ch) => OutputChannel::Stream(Arc::clone(ch)),
            OutputChannel::Null(ch) => OutputChannel::Null(Arc::clone(ch)),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expansion::{AtomParams, JobParams};
    use mpb2d_core::polarization::Polarization;

    fn make_test_result(index: usize) -> CompactBandResult {
        CompactBandResult {
            job_index: index,
            params: JobParams {
                eps_bg: 12.0,
                resolution: 32,
                polarization: Polarization::TM,
                lattice_type: Some("square".to_string()),
                atoms: vec![AtomParams {
                    index: 0,
                    pos: [0.5, 0.5],
                    radius: 0.3,
                    eps_inside: 1.0,
                }],
            },
            result_type: CompactResultType::Maxwell(MaxwellResult {
                k_path: (0..100).map(|i| [i as f64 / 100.0, 0.0]).collect(),
                distances: (0..100).map(|i| i as f64 / 100.0).collect(),
                bands: (0..100)
                    .map(|_| (0..10).map(|b| 0.1 * b as f64).collect())
                    .collect(),
            }),
        }
    }

    #[test]
    fn test_compact_result_size() {
        let result = make_test_result(0);
        let size = result.approx_size();

        // Should be approximately 10-11 KB for 10 bands × 100 k-points
        assert!(size > 8000, "Size {} too small", size);
        assert!(size < 15000, "Size {} too large", size);
    }

    #[test]
    fn test_null_channel() {
        let channel = OutputChannel::null();

        for i in 0..100 {
            channel.send(make_test_result(i)).unwrap();
        }

        let stats = channel.close().unwrap();
        assert_eq!(stats.results_sent, 100);
    }

    #[test]
    fn test_null_channel_closed() {
        let channel = NullChannel::new();
        channel.send(make_test_result(0)).unwrap();
        channel.close().unwrap();

        // Should fail after close
        let result = channel.send(make_test_result(1));
        assert!(matches!(result, Err(ChannelError::Closed)));
    }

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.buffer_size_bytes, 10 * 1024 * 1024);
        assert_eq!(config.format, OutputFormat::Csv);
    }

    #[test]
    fn test_stream_config_default() {
        let config = StreamConfig::default();
        assert_eq!(config.channel_capacity, 64);
        assert_eq!(config.backpressure, BackpressurePolicy::Block);
    }
}
