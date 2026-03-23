//! High-performance batched I/O with background writer thread.
//!
//! This module provides a batch output channel that buffers results in memory
//! and writes them to disk in large chunks using a dedicated background thread.
//! This minimizes I/O interference with the solver threads.
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────┐     ┌──────────────┐     ┌─────────────────┐
//! │  Solver Pool   │────▶│ BatchChannel │────▶│ Writer Thread   │
//! │  (N threads)   │     │  (buffer)    │     │ (SPSC channel)  │
//! └────────────────┘     └──────────────┘     └─────────────────┘
//!                              │                      │
//!                              │                      ▼
//!                              │               ┌─────────────┐
//!                              └──────────────▶│  CSV Files  │
//!                               (on flush)     └─────────────┘
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! let config = BatchConfig {
//!     buffer_size_bytes: 10 * 1024 * 1024, // 10 MB
//!     ..Default::default()
//! };
//! let channel = BatchChannel::new(config, output_path);
//! 
//! // Send results (batched automatically)
//! channel.send(result)?;
//! 
//! // Close and wait for writer to finish
//! let stats = channel.close()?;
//! ```

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Receiver, Sender};
use log::{debug, error, info, warn};
use parking_lot::Mutex;

use crate::channel::{
    BatchConfig, ChannelError, ChannelStats, CompactBandResult, OutputChannelSink, OutputFormat,
};
use crate::config::OutputMode;

// ============================================================================
// Batch Commands
// ============================================================================

/// Commands sent to the background writer thread.
enum WriterCommand {
    /// Write a batch of results
    WriteBatch(Vec<CompactBandResult>),
    /// Force flush to disk
    Flush,
    /// Shutdown the writer thread
    Shutdown,
}

// ============================================================================
// Writer Statistics
// ============================================================================

/// Statistics from the background writer thread.
#[derive(Debug, Default)]
struct WriterStats {
    /// Number of batches written
    batches_written: usize,
    /// Total results written
    results_written: usize,
    /// Total bytes written
    bytes_written: usize,
    /// Total time spent writing
    write_time: Duration,
}

// ============================================================================
// Batch Channel
// ============================================================================

/// High-performance batch output channel with background writer.
///
/// Results are buffered in memory until the buffer reaches the configured size,
/// then sent to a background thread for writing. This decouples I/O from the
/// solver threads.
pub struct BatchChannel {
    /// Configuration
    config: BatchConfig,

    /// Output directory for full mode, or file path for selective mode
    #[allow(dead_code)]
    output_path: PathBuf,

    /// Output mode (full = multiple files, selective = single file)
    #[allow(dead_code)]
    output_mode: OutputMode,

    /// Command sender to background writer
    sender: Sender<WriterCommand>,

    /// In-memory buffer for results
    buffer: Mutex<Vec<CompactBandResult>>,

    /// Current buffer size in bytes (atomic for fast reads)
    buffer_size: AtomicUsize,

    /// Total results sent
    results_sent: AtomicUsize,

    /// Channel is closed
    closed: AtomicBool,

    /// Writer thread handle (for join on close)
    writer_handle: Mutex<Option<JoinHandle<WriterStats>>>,

    /// Last flush time (for time-based flushing)
    last_flush: Mutex<Instant>,
}

impl BatchChannel {
    /// Create a new batch channel.
    ///
    /// # Arguments
    ///
    /// * `config` - Batch configuration (buffer size, flush interval, format)
    /// * `output_path` - Output directory (full mode) or file path (selective mode)
    /// * `output_mode` - Whether to write one file per job or a single merged file
    pub fn new(config: BatchConfig, output_path: PathBuf, output_mode: OutputMode) -> Self {
        // Create output directory if needed
        if matches!(output_mode, OutputMode::Full) {
            if let Err(e) = fs::create_dir_all(&output_path) {
                warn!("failed to create output directory: {}", e);
            }
        } else if let Some(parent) = output_path.parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(e) = fs::create_dir_all(parent) {
                    warn!("failed to create output directory: {}", e);
                }
            }
        }

        // Create command channel (small queue, large batches)
        let (sender, receiver) = bounded(4);

        // Clone values for the writer thread
        let writer_path = output_path.clone();
        let writer_mode = output_mode.clone();
        let writer_format = config.format;

        // Spawn background writer thread
        let writer_handle = thread::spawn(move || {
            background_writer(receiver, writer_path, writer_mode, writer_format)
        });

        Self {
            config,
            output_path,
            output_mode,
            sender,
            buffer: Mutex::new(Vec::with_capacity(1024)),
            buffer_size: AtomicUsize::new(0),
            results_sent: AtomicUsize::new(0),
            closed: AtomicBool::new(false),
            writer_handle: Mutex::new(Some(writer_handle)),
            last_flush: Mutex::new(Instant::now()),
        }
    }

    /// Check if we should flush based on time interval.
    fn should_time_flush(&self) -> bool {
        if let Some(interval) = self.config.flush_interval {
            let last = *self.last_flush.lock();
            last.elapsed() >= interval
        } else {
            false
        }
    }

    /// Flush the buffer to the background writer.
    fn flush_buffer(&self) -> Result<(), ChannelError> {
        let mut buffer = self.buffer.lock();
        if buffer.is_empty() {
            return Ok(());
        }

        let batch = std::mem::take(&mut *buffer);
        self.buffer_size.store(0, Ordering::Relaxed);
        *self.last_flush.lock() = Instant::now();

        debug!("flushing batch of {} results to writer", batch.len());

        self.sender
            .send(WriterCommand::WriteBatch(batch))
            .map_err(|_| ChannelError::Closed)?;

        Ok(())
    }
}

impl OutputChannelSink for BatchChannel {
    fn send(&self, result: CompactBandResult) -> Result<(), ChannelError> {
        if self.closed.load(Ordering::Relaxed) {
            return Err(ChannelError::Closed);
        }

        let size = result.approx_size();

        // Add to buffer
        {
            let mut buffer = self.buffer.lock();
            buffer.push(result);
        }

        self.results_sent.fetch_add(1, Ordering::Relaxed);
        let total_size = self.buffer_size.fetch_add(size, Ordering::Relaxed) + size;

        // Check if we need to flush (size-based or time-based)
        if total_size >= self.config.buffer_size_bytes || self.should_time_flush() {
            self.flush_buffer()?;
        }

        Ok(())
    }

    fn flush(&self) -> Result<(), ChannelError> {
        self.flush_buffer()?;

        // Also tell the writer to flush to disk
        self.sender
            .send(WriterCommand::Flush)
            .map_err(|_| ChannelError::Closed)?;

        Ok(())
    }

    fn close(&self) -> Result<ChannelStats, ChannelError> {
        if self.closed.swap(true, Ordering::SeqCst) {
            // Already closed
            return Err(ChannelError::Closed);
        }

        // Flush remaining buffer
        self.flush_buffer()?;

        // Send shutdown command
        let _ = self.sender.send(WriterCommand::Shutdown);

        // Wait for writer thread to finish
        let mut handle_guard = self.writer_handle.lock();
        let writer_stats = if let Some(handle) = handle_guard.take() {
            match handle.join() {
                Ok(stats) => stats,
                Err(_) => {
                    error!("writer thread panicked");
                    return Err(ChannelError::WriterPanic);
                }
            }
        } else {
            WriterStats::default()
        };

        info!(
            "batch writer finished: {} results in {} batches, {:.2} MB written",
            writer_stats.results_written,
            writer_stats.batches_written,
            writer_stats.bytes_written as f64 / (1024.0 * 1024.0)
        );

        Ok(ChannelStats {
            results_sent: self.results_sent.load(Ordering::Relaxed),
            bytes_written: writer_stats.bytes_written,
            flush_count: writer_stats.batches_written,
            results_dropped: 0,
            total_write_time: writer_stats.write_time,
        })
    }

    fn is_open(&self) -> bool {
        !self.closed.load(Ordering::Relaxed)
    }
}

// ============================================================================
// Background Writer
// ============================================================================

/// Background writer thread function.
fn background_writer(
    receiver: Receiver<WriterCommand>,
    output_path: PathBuf,
    output_mode: OutputMode,
    format: OutputFormat,
) -> WriterStats {
    let mut stats = WriterStats::default();

    // For selective mode, we accumulate all results and write at the end
    let mut selective_buffer: Vec<CompactBandResult> = Vec::new();

    loop {
        match receiver.recv() {
            Ok(WriterCommand::WriteBatch(batch)) => {
                let start = Instant::now();
                let batch_len = batch.len();

                match output_mode {
                    OutputMode::Full => {
                        // Write each result to its own file
                        for result in batch {
                            let bytes = write_full_result(&output_path, &result, format);
                            stats.bytes_written += bytes;
                            stats.results_written += 1;
                        }
                    }
                    OutputMode::Selective => {
                        // Accumulate for later writing
                        selective_buffer.extend(batch);
                        stats.results_written += batch_len;
                    }
                }

                stats.batches_written += 1;
                stats.write_time += start.elapsed();

                debug!(
                    "wrote batch of {} results in {:.2}ms",
                    batch_len,
                    start.elapsed().as_secs_f64() * 1000.0
                );
            }
            Ok(WriterCommand::Flush) => {
                // For selective mode, write accumulated results
                if matches!(output_mode, OutputMode::Selective) && !selective_buffer.is_empty() {
                    let start = Instant::now();
                    let bytes = write_selective_results(&output_path, &selective_buffer, format);
                    stats.bytes_written += bytes;
                    stats.write_time += start.elapsed();
                    selective_buffer.clear();
                }
            }
            Ok(WriterCommand::Shutdown) => {
                // Write any remaining selective results
                if matches!(output_mode, OutputMode::Selective) && !selective_buffer.is_empty() {
                    let start = Instant::now();
                    let bytes = write_selective_results(&output_path, &selective_buffer, format);
                    stats.bytes_written += bytes;
                    stats.write_time += start.elapsed();
                }
                break;
            }
            Err(_) => {
                // Channel closed unexpectedly
                warn!("writer channel closed unexpectedly");
                break;
            }
        }
    }

    stats
}

/// Write a single result to its own CSV file (full mode).
fn write_full_result(output_dir: &PathBuf, result: &CompactBandResult, format: OutputFormat) -> usize {
    let filename = format!("job_{:06}.csv", result.job_index);
    let path = output_dir.join(filename);

    match format {
        OutputFormat::Csv => write_csv_full(&path, result),
        OutputFormat::Binary => {
            warn!("binary format not yet implemented, falling back to CSV");
            write_csv_full(&path, result)
        }
        OutputFormat::Json => {
            warn!("JSON format not yet implemented, falling back to CSV");
            write_csv_full(&path, result)
        }
    }
}

/// Write a CSV file for a single result.
fn write_csv_full(path: &PathBuf, result: &CompactBandResult) -> usize {
    use crate::channel::CompactResultType;

    let file = match File::create(path) {
        Ok(f) => f,
        Err(e) => {
            error!("failed to create {}: {}", path.display(), e);
            return 0;
        }
    };

    let mut writer = BufWriter::new(file);
    let mut bytes_written = 0;

    // Write header
    let param_cols = result.params.to_columns();
    let mut header = String::from("job_index");
    for (name, _) in &param_cols {
        header.push(',');
        header.push_str(name);
    }

    match &result.result_type {
        CompactResultType::Maxwell(m) => {
            header.push_str(",k_index,kx,ky,k_distance");
            let max_bands = m.bands.iter().map(|b| b.len()).max().unwrap_or(0);
            for i in 1..=max_bands {
                header.push_str(&format!(",band{}", i));
            }
            header.push('\n');

            if let Err(e) = writer.write_all(header.as_bytes()) {
                error!("failed to write header: {}", e);
                return 0;
            }
            bytes_written += header.len();

            // Write data rows
            for (k_idx, ((k_point, distance), bands)) in m
                .k_path
                .iter()
                .zip(m.distances.iter())
                .zip(m.bands.iter())
                .enumerate()
            {
                let mut row = format!("{}", result.job_index);
                for (_, value) in &param_cols {
                    row.push(',');
                    row.push_str(value);
                }
                row.push_str(&format!(
                    ",{},{},{},{}",
                    k_idx, k_point[0], k_point[1], distance
                ));

                for omega in bands {
                    row.push_str(&format!(",{}", omega));
                }

                // Pad with empty columns
                let max_bands = m.bands.iter().map(|b| b.len()).max().unwrap_or(0);
                for _ in bands.len()..max_bands {
                    row.push(',');
                }
                row.push('\n');

                if let Err(e) = writer.write_all(row.as_bytes()) {
                    error!("failed to write row: {}", e);
                    break;
                }
                bytes_written += row.len();
            }
        }
        CompactResultType::EA(ea) => {
            header.push_str(",n_iterations,converged,band_index,eigenvalue\n");

            if let Err(e) = writer.write_all(header.as_bytes()) {
                error!("failed to write header: {}", e);
                return 0;
            }
            bytes_written += header.len();

            // Write data rows (one per eigenvalue)
            for (band_idx, eigenvalue) in ea.eigenvalues.iter().enumerate() {
                let mut row = format!("{}", result.job_index);
                for (_, value) in &param_cols {
                    row.push(',');
                    row.push_str(value);
                }
                row.push_str(&format!(
                    ",{},{},{},{}",
                    ea.n_iterations,
                    ea.converged,
                    band_idx + 1,
                    eigenvalue
                ));
                row.push('\n');

                if let Err(e) = writer.write_all(row.as_bytes()) {
                    error!("failed to write row: {}", e);
                    break;
                }
                bytes_written += row.len();
            }
        }
    }

    if let Err(e) = writer.flush() {
        error!("failed to flush: {}", e);
    }

    bytes_written
}

/// Write all accumulated results to a single CSV file (selective mode).
fn write_selective_results(
    output_path: &PathBuf,
    results: &[CompactBandResult],
    format: OutputFormat,
) -> usize {
    if results.is_empty() {
        return 0;
    }

    match format {
        OutputFormat::Csv => write_csv_selective(output_path, results),
        OutputFormat::Binary => {
            warn!("binary format not yet implemented, falling back to CSV");
            write_csv_selective(output_path, results)
        }
        OutputFormat::Json => {
            warn!("JSON format not yet implemented, falling back to CSV");
            write_csv_selective(output_path, results)
        }
    }
}

/// Write selective mode CSV with all results merged.
fn write_csv_selective(path: &PathBuf, results: &[CompactBandResult]) -> usize {
    use crate::channel::CompactResultType;

    let file = match File::create(path) {
        Ok(f) => f,
        Err(e) => {
            error!("failed to create {}: {}", path.display(), e);
            return 0;
        }
    };

    let mut writer = BufWriter::new(file);
    let mut bytes_written = 0;

    // Determine columns from first result
    let first = &results[0];
    let param_cols = first.params.to_columns();

    // Check if any results are Maxwell type
    let has_maxwell = results.iter().any(|r| matches!(r.result_type, CompactResultType::Maxwell(_)));

    if has_maxwell {
        // Maxwell-style output with k-points
        let max_bands = results
            .iter()
            .filter_map(|r| match &r.result_type {
                CompactResultType::Maxwell(m) => Some(m.bands.iter().map(|b| b.len()).max().unwrap_or(0)),
                _ => None,
            })
            .max()
            .unwrap_or(0);

        // Write header
        let mut header = String::from("job_index");
        for (name, _) in &param_cols {
            header.push(',');
            header.push_str(name);
        }
        header.push_str(",k_index,kx,ky,k_distance");
        for i in 1..=max_bands {
            header.push_str(&format!(",band{}", i));
        }
        header.push('\n');

        if let Err(e) = writer.write_all(header.as_bytes()) {
            error!("failed to write header: {}", e);
            return 0;
        }
        bytes_written += header.len();

        // Write all results
        for result in results {
            if let CompactResultType::Maxwell(m) = &result.result_type {
                let param_values: Vec<_> = result.params.to_columns();

                for (k_idx, ((k_point, distance), bands)) in m
                    .k_path
                    .iter()
                    .zip(m.distances.iter())
                    .zip(m.bands.iter())
                    .enumerate()
                {
                    let mut row = format!("{}", result.job_index);
                    for (_, value) in &param_values {
                        row.push(',');
                        row.push_str(value);
                    }
                    row.push_str(&format!(
                        ",{},{},{},{}",
                        k_idx, k_point[0], k_point[1], distance
                    ));

                    for omega in bands {
                        row.push_str(&format!(",{}", omega));
                    }

                    for _ in bands.len()..max_bands {
                        row.push(',');
                    }
                    row.push('\n');

                    if let Err(e) = writer.write_all(row.as_bytes()) {
                        error!("failed to write row: {}", e);
                        break;
                    }
                    bytes_written += row.len();
                }
            }
        }
    } else {
        // EA-style output with eigenvalues
        let max_bands = results
            .iter()
            .filter_map(|r| match &r.result_type {
                CompactResultType::EA(ea) => Some(ea.eigenvalues.len()),
                _ => None,
            })
            .max()
            .unwrap_or(0);

        // Write header
        let mut header = String::from("job_index");
        for (name, _) in &param_cols {
            header.push(',');
            header.push_str(name);
        }
        header.push_str(",n_iterations,converged");
        for i in 1..=max_bands {
            header.push_str(&format!(",eigenvalue{}", i));
        }
        header.push('\n');

        if let Err(e) = writer.write_all(header.as_bytes()) {
            error!("failed to write header: {}", e);
            return 0;
        }
        bytes_written += header.len();

        // Write all results (one row per job for EA)
        for result in results {
            if let CompactResultType::EA(ea) = &result.result_type {
                let param_values: Vec<_> = result.params.to_columns();

                let mut row = format!("{}", result.job_index);
                for (_, value) in &param_values {
                    row.push(',');
                    row.push_str(value);
                }
                row.push_str(&format!(",{},{}", ea.n_iterations, ea.converged));

                for eigenvalue in &ea.eigenvalues {
                    row.push_str(&format!(",{}", eigenvalue));
                }

                for _ in ea.eigenvalues.len()..max_bands {
                    row.push(',');
                }
                row.push('\n');

                if let Err(e) = writer.write_all(row.as_bytes()) {
                    error!("failed to write row: {}", e);
                    break;
                }
                bytes_written += row.len();
            }
        }
    }

    if let Err(e) = writer.flush() {
        error!("failed to flush: {}", e);
    }

    info!(
        "wrote {} results ({:.2} MB) to {}",
        results.len(),
        bytes_written as f64 / (1024.0 * 1024.0),
        path.display()
    );

    bytes_written
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::{CompactResultType, MaxwellResult};
    use crate::expansion::{AtomParams, JobParams};
    use mpb2d_core::polarization::Polarization;
    use std::fs;
    use tempfile::tempdir;

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
                sweep_values: vec![],
            },
            result_type: CompactResultType::Maxwell(MaxwellResult {
                k_path: (0..10).map(|i| [i as f64 / 10.0, 0.0]).collect(),
                distances: (0..10).map(|i| i as f64 / 10.0).collect(),
                bands: (0..10)
                    .map(|_| (0..5).map(|b| 0.1 * b as f64).collect())
                    .collect(),
            }),
        }
    }

    #[test]
    fn test_batch_channel_full_mode() {
        let dir = tempdir().unwrap();
        let config = BatchConfig {
            buffer_size_bytes: 1024, // Small buffer for testing
            ..Default::default()
        };

        let channel = BatchChannel::new(config, dir.path().to_path_buf(), OutputMode::Full);

        // Send several results
        for i in 0..5 {
            channel.send(make_test_result(i)).unwrap();
        }

        // Close and get stats
        let stats = channel.close().unwrap();
        assert_eq!(stats.results_sent, 5);

        // Check files were created
        let files: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(files.len(), 5);
    }

    #[test]
    fn test_batch_channel_selective_mode() {
        let dir = tempdir().unwrap();
        let output_file = dir.path().join("results.csv");
        let config = BatchConfig::default();

        let channel = BatchChannel::new(config, output_file.clone(), OutputMode::Selective);

        // Send several results
        for i in 0..3 {
            channel.send(make_test_result(i)).unwrap();
        }

        // Close and get stats
        let stats = channel.close().unwrap();
        assert_eq!(stats.results_sent, 3);

        // Check single file was created
        assert!(output_file.exists());

        // Read and verify content
        let content = fs::read_to_string(&output_file).unwrap();
        assert!(content.contains("job_index"));
        assert!(content.contains("band1"));
    }

    #[test]
    fn test_batch_channel_buffer_flush() {
        let dir = tempdir().unwrap();
        let config = BatchConfig {
            buffer_size_bytes: 100, // Very small buffer to force flush
            ..Default::default()
        };

        let channel = BatchChannel::new(config, dir.path().to_path_buf(), OutputMode::Full);

        // Each result should trigger a flush due to small buffer
        for i in 0..3 {
            channel.send(make_test_result(i)).unwrap();
        }

        let stats = channel.close().unwrap();
        assert_eq!(stats.results_sent, 3);
        assert!(stats.flush_count >= 1); // Should have flushed at least once
    }

    #[test]
    fn test_batch_channel_closed_error() {
        let dir = tempdir().unwrap();
        let config = BatchConfig::default();
        let channel = BatchChannel::new(config, dir.path().to_path_buf(), OutputMode::Full);

        channel.close().unwrap();

        // Should error on send after close
        let result = channel.send(make_test_result(0));
        assert!(matches!(result, Err(ChannelError::Closed)));
    }
}
