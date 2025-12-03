//! Multi-threaded driver for bulk job execution.
//!
//! This module handles parallel execution of expanded jobs using a thread pool,
//! with adaptive thread management, progress tracking, and result collection.

use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, error, warn};
use parking_lot::Mutex;
use rayon::prelude::*;

use mpb2d_core::drivers::bandstructure::{self, BandStructureResult, RunOptions, Verbosity};
use mpb2d_core::drivers::ProgressInfo;
use mpb2d_core::profiler::print_profile;

#[cfg(feature = "cuda")]
use mpb2d_backend_cuda::CudaBackend;

#[cfg(not(feature = "cuda"))]
use mpb2d_backend_cpu::CpuBackend;

use crate::adaptive::{AdaptiveConfig, AdaptiveThreadManager, AdjustmentReason};
use crate::batch::BatchChannel;
use crate::channel::{
    BatchConfig, CompactBandResult, CompactBandResultExt, OutputChannel, StreamConfig,
};
use crate::config::{BulkConfig, BulkConfigNativeExt, OutputMode, SelectiveSpec, SolverType};
use crate::expansion::{expand_jobs, ExpandedJob, ExpandedJobType, EAJobSpec};
use crate::output::OutputWriter;
use crate::stream::{FilteredStreamChannel, SelectiveFilter, StreamChannel};

// ============================================================================
// Job Result
// ============================================================================

/// Result of a single job execution.
#[derive(Debug, Clone)]
pub struct JobResult {
    /// Job index (matches ExpandedJob.index)
    pub index: usize,

    /// The result (Maxwell or EA)
    pub result: JobResultType,

    /// Execution time
    pub duration: Duration,

    /// Any warnings or notes
    pub notes: Vec<String>,
}

/// Type of job result.
#[derive(Debug, Clone)]
pub enum JobResultType {
    /// Maxwell band structure result
    Maxwell(BandStructureResult),
    /// EA eigenvalue result
    EA(EAJobResult),
}

impl JobResult {
    /// Get the Maxwell result, if this is a Maxwell job.
    pub fn maxwell(&self) -> Option<&BandStructureResult> {
        match &self.result {
            JobResultType::Maxwell(r) => Some(r),
            _ => None,
        }
    }

    /// Get the EA result, if this is an EA job.
    pub fn ea(&self) -> Option<&EAJobResult> {
        match &self.result {
            JobResultType::EA(r) => Some(r),
            _ => None,
        }
    }
}

/// Result from an EA eigenvalue problem.
#[derive(Debug, Clone)]
pub struct EAJobResult {
    /// Computed eigenvalues
    pub eigenvalues: Vec<f64>,
    /// Computed eigenvectors as Field2D
    pub eigenvectors: Vec<mpb2d_core::field::Field2D>,
    /// Grid dimensions [nx, ny]
    pub grid_dims: [usize; 2],
    /// Number of iterations taken
    pub n_iterations: usize,
    /// Whether convergence was achieved
    pub converged: bool,
}

/// Error during job execution.
#[derive(Debug)]
pub struct JobError {
    /// Job index
    pub index: usize,

    /// Error message
    pub message: String,
}

// ============================================================================
// Pre-Run Report
// ============================================================================

/// Formatted pre-run report for display.
pub struct PreRunReport {
    lines: Vec<String>,
}

impl PreRunReport {
    /// Build a pre-run report from configuration and jobs.
    pub fn build(config: &BulkConfig, jobs: &[ExpandedJob], thread_mode: &str) -> Self {
        let mut lines = Vec::new();

        // Header
        lines.push(String::from("╭─────────────────────────────────────────────────╮"));
        lines.push(String::from("│            MPB2D Bulk Driver                    │"));
        lines.push(String::from("╰─────────────────────────────────────────────────╯"));
        lines.push(String::new());

        // Job summary with solver type
        let solver_str = match config.solver_type() {
            SolverType::Maxwell => "Maxwell",
            SolverType::EA => "EA",
        };
        lines.push(format!("  Solver: {}  │  Jobs: {}  │  Threads: {}", solver_str, jobs.len(), thread_mode));
        lines.push(String::new());

        // Fixed parameters (from base config) - only for Maxwell
        let mut fixed = Vec::new();
        if let Some(ref geometry) = config.geometry {
            if config.ranges.eps_bg.is_none() {
                fixed.push(format!("ε_bg={:.1}", geometry.eps_bg));
            }
            if config.ranges.resolution.is_none() {
                fixed.push(format!("grid={}×{}", config.grid.nx, config.grid.ny));
            }
            if config.ranges.lattice_type.is_none() {
                if let Some(ref lt) = geometry.lattice.lattice_type {
                    fixed.push(format!("lattice={}", lt));
                }
            }
            // Only show atoms info if there are atoms and they're not being swept
            if !geometry.atoms.is_empty() {
                let base_atoms_count = geometry.atoms.len();
                fixed.push(format!("atoms={}", base_atoms_count));
            }
        } else {
            // EA solver - show EA-specific fixed parameters
            fixed.push(format!("grid={}×{}", config.grid.nx, config.grid.ny));
            fixed.push(format!("η={:.4}", config.ea.eta));
            fixed.push(format!("n_bands={}", config.eigensolver.n_bands));
        }

        if !fixed.is_empty() {
            lines.push(format!("  Fixed: {}", fixed.join(", ")));
        }

        // Swept parameters
        let mut swept = Vec::new();
        if let Some(ref range) = config.ranges.eps_bg {
            swept.push(format!("ε_bg: {:.1}→{:.1} ({})", range.min, range.max, range.count()));
        }
        if let Some(ref range) = config.ranges.resolution {
            swept.push(format!("res: {}→{} ({})", range.min as i32, range.max as i32, range.count()));
        }
        if let Some(ref pols) = config.ranges.polarization {
            let pol_strs: Vec<_> = pols.iter().map(|p| format!("{:?}", p)).collect();
            swept.push(format!("pol: [{}]", pol_strs.join(",")));
        }
        if let Some(ref types) = config.ranges.lattice_type {
            let type_strs: Vec<_> = types.iter().map(|t| format!("{:?}", t)).collect();
            swept.push(format!("lattice: [{}]", type_strs.join(",")));
        }

        // Atom parameter sweeps
        for (i, atom_range) in config.ranges.atoms.iter().enumerate() {
            if let Some(ref r) = atom_range.radius {
                swept.push(format!("atom{}.r: {:.2}→{:.2} ({})", i, r.min, r.max, r.count()));
            }
            if let Some(ref px) = atom_range.pos_x {
                swept.push(format!("atom{}.x: {:.2}→{:.2} ({})", i, px.min, px.max, px.count()));
            }
            if let Some(ref py) = atom_range.pos_y {
                swept.push(format!("atom{}.y: {:.2}→{:.2} ({})", i, py.min, py.max, py.count()));
            }
            if let Some(ref eps) = atom_range.eps_inside {
                swept.push(format!("atom{}.ε: {:.1}→{:.1} ({})", i, eps.min, eps.max, eps.count()));
            }
        }

        if !swept.is_empty() {
            lines.push(format!("  Swept: {}", swept.join(", ")));
        }

        // Output mode
        let output_info = match config.output.mode {
            OutputMode::Full => String::from("full (one CSV per job)"),
            OutputMode::Selective => String::from("selective (merged CSV)"),
        };
        lines.push(format!("  Output: {} → {}", output_info, config.output.directory.display()));

        lines.push(String::new());

        Self { lines }
    }

    /// Print the report to stdout.
    pub fn print(&self) {
        for line in &self.lines {
            println!("{}", line);
        }
        let _ = io::stdout().flush();
    }
}

// ============================================================================
// Bulk Driver
// ============================================================================

/// High-performance driver for bulk parameter sweep calculations.
///
/// The driver handles:
/// - Expanding parameter ranges into individual jobs
/// - Parallel execution using a thread pool (with optional adaptive sizing)
/// - Progress tracking and logging
/// - Output batching and writing
pub struct BulkDriver {
    /// Configuration
    config: BulkConfig,

    /// Expanded jobs
    jobs: Vec<ExpandedJob>,

    /// Thread management mode
    thread_mode: ThreadMode,

    /// Verbose output (debug logs)
    verbose: bool,
}

/// Thread management mode.
#[derive(Debug, Clone)]
pub enum ThreadMode {
    /// Fixed number of threads.
    Fixed(usize),
    /// Adaptive thread management (starts at given count).
    Adaptive,
}

impl BulkDriver {
    /// Create a new bulk driver from configuration.
    pub fn new(config: BulkConfig, requested_threads: Option<i32>) -> Self {
        let thread_mode = match requested_threads {
            Some(-1) => ThreadMode::Adaptive,
            Some(n) if n > 0 => ThreadMode::Fixed(n as usize),
            _ => {
                // Default from config or use adaptive
                let t = config.effective_threads();
                if t == 0 {
                    ThreadMode::Adaptive
                } else {
                    ThreadMode::Fixed(t)
                }
            }
        };

        let verbose = config.bulk.verbose;
        let jobs = expand_jobs(&config);

        Self {
            config,
            jobs,
            thread_mode,
            verbose,
        }
    }

    /// Get the number of jobs to execute.
    pub fn job_count(&self) -> usize {
        self.jobs.len()
    }

    /// Get the expanded jobs (for dry run inspection).
    pub fn jobs(&self) -> &[ExpandedJob] {
        &self.jobs
    }

    /// Get thread mode description for display.
    fn thread_mode_str(&self, adaptive_mgr: Option<&AdaptiveThreadManager>) -> String {
        match &self.thread_mode {
            ThreadMode::Fixed(n) => format!("{}", n),
            ThreadMode::Adaptive => {
                if let Some(mgr) = adaptive_mgr {
                    format!("adaptive (start: {})", mgr.initial_threads())
                } else {
                    let num_cpus = num_cpus::get();
                    format!("adaptive ({}→{})", (num_cpus + 1) / 2, num_cpus)
                }
            }
        }
    }

    /// Execute all jobs and write results.
    ///
    /// Returns the number of successful jobs and any errors encountered.
    pub fn run(&self) -> Result<DriverStats, DriverError> {
        if self.jobs.is_empty() {
            warn!("no jobs to execute (parameter ranges resulted in zero configurations)");
            return Ok(DriverStats::default());
        }

        // Special case: single EA job gets detailed progress bar (no threading)
        if self.is_single_ea_job() {
            return self.run_single_ea_with_progress();
        }

        // Create adaptive thread manager
        let adaptive_mgr = match &self.thread_mode {
            ThreadMode::Fixed(n) => Arc::new(AdaptiveThreadManager::fixed(*n)),
            ThreadMode::Adaptive => Arc::new(AdaptiveThreadManager::new(AdaptiveConfig::default())),
        };

        // Print pre-run report
        let report = PreRunReport::build(
            &self.config,
            &self.jobs,
            &self.thread_mode_str(Some(&adaptive_mgr)),
        );
        report.print();

        // Setup thread pool with initial thread count
        // Note: For adaptive mode, we create with max threads but control parallelism
        let pool_threads = match &self.thread_mode {
            ThreadMode::Fixed(n) => *n,
            ThreadMode::Adaptive => num_cpus::get(),
        };

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(pool_threads)
            .build()
            .map_err(|e| DriverError::ThreadPoolError(e.to_string()))?;

        // Setup output writer
        let output_writer = Arc::new(Mutex::new(
            OutputWriter::new(&self.config.output, &self.jobs)
                .map_err(|e| DriverError::OutputError(e.to_string()))?,
        ));

        // Progress tracking
        let completed = Arc::new(AtomicUsize::new(0));
        let failed = Arc::new(AtomicUsize::new(0));
        let errors = Arc::new(Mutex::new(Vec::new()));

        // Create progress bar (always show, but minimal)
        let pb = ProgressBar::new(self.jobs.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("█▓░"),
        );

        let start_time = Instant::now();
        let adaptive_mgr_clone = adaptive_mgr.clone();
        let verbose = self.verbose;

        // Execute jobs in parallel
        pool.install(|| {
            self.jobs.par_iter().for_each(|expanded_job| {
                let job_start = Instant::now();
                let result = self.execute_job(expanded_job);
                let job_duration = job_start.elapsed();

                // Record timing for adaptive management
                if let Some(event) = adaptive_mgr_clone.record_job(job_duration) {
                    // Log thread adjustment
                    if event.reason != AdjustmentReason::Initial && verbose {
                        debug!(
                            "[adaptive] {} → {} threads ({})",
                            event.from_threads, event.to_threads, event.reason
                        );
                    }
                }

                match result {
                    Ok(job_result) => {
                        // Write result
                        let write_result = {
                            let mut writer = output_writer.lock();
                            writer.write_result(expanded_job, &job_result)
                        };

                        if let Err(e) = write_result {
                            error!("failed to write job {} result: {}", expanded_job.index, e);
                            failed.fetch_add(1, Ordering::Relaxed);
                            errors.lock().push(JobError {
                                index: expanded_job.index,
                                message: format!("output write error: {}", e),
                            });
                        } else {
                            completed.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    Err(e) => {
                        failed.fetch_add(1, Ordering::Relaxed);
                        errors.lock().push(e);
                    }
                }

                pb.inc(1);
            });
        });

        // Finalize output
        {
            let mut writer = output_writer.lock();
            writer
                .finalize()
                .map_err(|e| DriverError::OutputError(e.to_string()))?;
        }

        pb.finish_and_clear();

        let total_time = start_time.elapsed();
        let completed_count = completed.load(Ordering::Relaxed);
        let failed_count = failed.load(Ordering::Relaxed);

        // Get adaptive summary
        let adaptive_summary = adaptive_mgr.get_summary();

        // Print summary
        println!();
        if failed_count == 0 {
            println!(
                "✓ {} jobs completed in {:.2}s",
                completed_count,
                total_time.as_secs_f64()
            );
        } else {
            println!(
                "⚠ {}/{} jobs completed, {} failed in {:.2}s",
                completed_count,
                self.jobs.len(),
                failed_count,
                total_time.as_secs_f64()
            );
        }

        // Show adaptive thread summary if applicable
        if adaptive_mgr.is_adaptive() && adaptive_summary.total_adjustments > 0 {
            println!(
                "  threads: {} → {} ({} adjustments)",
                adaptive_summary.initial_threads,
                adaptive_summary.final_threads,
                adaptive_summary.total_adjustments
            );
            if self.verbose {
                println!("{}", adaptive_summary.format_log());
            }
        }

        if failed_count > 0 {
            let errs = errors.lock();
            for err in errs.iter().take(5) {
                error!("job {} failed: {}", err.index, err.message);
            }
            if errs.len() > 5 {
                error!("... and {} more errors", errs.len() - 5);
            }
        }

        Ok(DriverStats {
            total_jobs: self.jobs.len(),
            completed: completed_count,
            failed: failed_count,
            total_time,
            errors: Arc::try_unwrap(errors).unwrap_or_default().into_inner(),
            adaptive_summary: Some(adaptive_summary),
        })
    }

    /// Execute all jobs with a custom output channel.
    ///
    /// This is the core execution method that supports batch, stream, and null output modes.
    /// Results are sent to the provided channel instead of using the legacy OutputWriter.
    ///
    /// # Arguments
    ///
    /// * `channel` - Output channel (Batch, Stream, or Null)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Batch mode with 10MB buffer
    /// let channel = OutputChannel::Batch(Arc::new(BatchChannel::new(
    ///     BatchConfig::default(),
    ///     PathBuf::from("./output"),
    ///     OutputMode::Full,
    /// )));
    /// let stats = driver.run_with_channel(channel)?;
    ///
    /// // Stream mode
    /// let stream = Arc::new(StreamChannel::new(StreamConfig::default()));
    /// let receiver = stream.add_channel_subscriber();
    /// let channel = OutputChannel::Stream(stream);
    /// let stats = driver.run_with_channel(channel)?;
    /// ```
    pub fn run_with_channel(&self, channel: OutputChannel) -> Result<DriverStats, DriverError> {
        if self.jobs.is_empty() {
            warn!("no jobs to execute (parameter ranges resulted in zero configurations)");
            return Ok(DriverStats::default());
        }

        // Special case: single EA job gets detailed progress bar (no threading)
        if self.is_single_ea_job() {
            return self.run_single_ea_with_channel(channel);
        }

        // Create adaptive thread manager
        let adaptive_mgr = match &self.thread_mode {
            ThreadMode::Fixed(n) => Arc::new(AdaptiveThreadManager::fixed(*n)),
            ThreadMode::Adaptive => Arc::new(AdaptiveThreadManager::new(AdaptiveConfig::default())),
        };

        // Print pre-run report
        let io_mode = match &channel {
            OutputChannel::Batch(_) => "batch",
            OutputChannel::Stream(_) => "stream",
            OutputChannel::Null(_) => "null (benchmark)",
        };
        let report = PreRunReport::build(
            &self.config,
            &self.jobs,
            &self.thread_mode_str(Some(&adaptive_mgr)),
        );
        report.print();
        println!("  I/O Mode: {}", io_mode);
        println!();

        // Setup thread pool
        let pool_threads = match &self.thread_mode {
            ThreadMode::Fixed(n) => *n,
            ThreadMode::Adaptive => num_cpus::get(),
        };

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(pool_threads)
            .build()
            .map_err(|e| DriverError::ThreadPoolError(e.to_string()))?;

        // Progress tracking
        let completed = Arc::new(AtomicUsize::new(0));
        let failed = Arc::new(AtomicUsize::new(0));
        let errors = Arc::new(Mutex::new(Vec::new()));
        let channel_errors = Arc::new(AtomicUsize::new(0));

        // Create progress bar
        let pb = ProgressBar::new(self.jobs.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("█▓░"),
        );

        let start_time = Instant::now();
        let adaptive_mgr_clone = adaptive_mgr.clone();
        let verbose = self.verbose;

        // Wrap channel for thread-safe access
        let channel_arc: Arc<OutputChannel> = Arc::new(channel);

        // Execute jobs in parallel
        pool.install(|| {
            let channel = &channel_arc;
            self.jobs.par_iter().for_each(|expanded_job| {
                let job_start = Instant::now();
                let result = self.execute_job(expanded_job);
                let job_duration = job_start.elapsed();

                // Record timing for adaptive management
                if let Some(event) = adaptive_mgr_clone.record_job(job_duration) {
                    if event.reason != AdjustmentReason::Initial && verbose {
                        debug!(
                            "[adaptive] {} → {} threads ({})",
                            event.from_threads, event.to_threads, event.reason
                        );
                    }
                }

                match result {
                    Ok(job_result) => {
                        // Convert to compact result and send through channel
                        let compact = CompactBandResult::from_job_result(expanded_job, &job_result);

                        if let Err(e) = channel.send(compact) {
                            error!("failed to send job {} result: {}", expanded_job.index, e);
                            channel_errors.fetch_add(1, Ordering::Relaxed);
                        }
                        completed.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(e) => {
                        failed.fetch_add(1, Ordering::Relaxed);
                        errors.lock().push(e);
                    }
                }

                pb.inc(1);
            });
        });

        pb.finish_and_clear();

        // Close channel and get stats
        let channel_stats = match Arc::try_unwrap(channel_arc) {
            Ok(ch) => ch.close().map_err(|e| DriverError::OutputError(e.to_string()))?,
            Err(_) => {
                return Err(DriverError::OutputError("channel still in use".to_string()));
            }
        };

        let total_time = start_time.elapsed();
        let completed_count = completed.load(Ordering::Relaxed);
        let failed_count = failed.load(Ordering::Relaxed);
        let channel_error_count = channel_errors.load(Ordering::Relaxed);

        // Get adaptive summary
        let adaptive_summary = adaptive_mgr.get_summary();

        // Print summary
        println!();
        if failed_count == 0 && channel_error_count == 0 {
            println!(
                "✓ {} jobs completed in {:.2}s ({:.1} jobs/s)",
                completed_count,
                total_time.as_secs_f64(),
                completed_count as f64 / total_time.as_secs_f64()
            );
            if channel_stats.bytes_written > 0 {
                println!(
                    "  Output: {:.2} MB in {} flushes ({:.2}s write time)",
                    channel_stats.bytes_written as f64 / (1024.0 * 1024.0),
                    channel_stats.flush_count,
                    channel_stats.total_write_time.as_secs_f64()
                );
            }
        } else {
            println!(
                "⚠ {}/{} jobs completed, {} failed, {} channel errors in {:.2}s",
                completed_count,
                self.jobs.len(),
                failed_count,
                channel_error_count,
                total_time.as_secs_f64()
            );
        }

        // Show adaptive thread summary if applicable
        if adaptive_mgr.is_adaptive() && adaptive_summary.total_adjustments > 0 {
            println!(
                "  threads: {} → {} ({} adjustments)",
                adaptive_summary.initial_threads,
                adaptive_summary.final_threads,
                adaptive_summary.total_adjustments
            );
            if self.verbose {
                println!("{}", adaptive_summary.format_log());
            }
        }

        if failed_count > 0 {
            let errs = errors.lock();
            for err in errs.iter().take(5) {
                error!("job {} failed: {}", err.index, err.message);
            }
            if errs.len() > 5 {
                error!("... and {} more errors", errs.len() - 5);
            }
        }

        Ok(DriverStats {
            total_jobs: self.jobs.len(),
            completed: completed_count,
            failed: failed_count,
            total_time,
            errors: Arc::try_unwrap(errors).unwrap_or_default().into_inner(),
            adaptive_summary: Some(adaptive_summary),
        })
    }

    /// Run with batch mode: buffer results and write in large chunks.
    ///
    /// This minimizes I/O interference with the solver by using a background
    /// writer thread that handles file I/O asynchronously.
    ///
    /// # Arguments
    ///
    /// * `batch_config` - Batch configuration (buffer size, flush interval)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let stats = driver.run_batched(BatchConfig {
    ///     buffer_size_bytes: 10 * 1024 * 1024, // 10 MB
    ///     ..Default::default()
    /// })?;
    /// ```
    pub fn run_batched(&self, batch_config: BatchConfig) -> Result<DriverStats, DriverError> {
        let output_path = match self.config.output.mode {
            OutputMode::Full => self.config.output.directory.clone(),
            OutputMode::Selective => self.config.output.filename.clone(),
        };

        let batch_channel = BatchChannel::new(
            batch_config,
            output_path,
            self.config.output.mode.clone(),
        );

        let channel = OutputChannel::Batch(Arc::new(batch_channel));
        self.run_with_channel(channel)
    }

    /// Run with streaming mode: emit results in real-time.
    ///
    /// Returns a receiver channel that yields results as they complete.
    /// The computation runs in the calling thread, so use this from a
    /// background thread if you need non-blocking behavior.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let receiver = driver.run_streaming()?;
    /// 
    /// // Process results as they arrive
    /// for result in receiver {
    ///     println!("Job {} complete", result.job_index);
    /// }
    /// ```
    pub fn run_streaming(
        &self,
    ) -> Result<(crossbeam_channel::Receiver<CompactBandResult>, std::thread::JoinHandle<Result<DriverStats, DriverError>>), DriverError>
    where
        Self: Sync,
    {
        let stream = Arc::new(StreamChannel::new(StreamConfig::default()));
        let receiver = stream.add_channel_subscriber();

        // Clone self for the thread (requires that config and jobs are Send)
        let config = self.config.clone();
        let jobs = self.jobs.clone();
        let thread_mode = self.thread_mode.clone();
        let verbose = self.verbose;
        let stream_clone = stream.clone();

        let handle = std::thread::spawn(move || {
            let driver = BulkDriver {
                config,
                jobs,
                thread_mode,
                verbose,
            };
            let channel = OutputChannel::Stream(stream_clone);
            driver.run_with_channel(channel)
        });

        Ok((receiver, handle))
    }

    /// Run with streaming output and server-side selective filtering.
    ///
    /// This applies k-point and band filtering at the emission point,
    /// before broadcasting to subscribers. This is more efficient than
    /// filtering in the consumer when you only need a subset of data.
    ///
    /// Returns a receiver for filtered results and a join handle for the
    /// driver thread.
    ///
    /// # Arguments
    ///
    /// * `filter` - Filter specifying which k-points and bands to include
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Stream only specific k-points and first 4 bands
    /// let filter = SelectiveFilter::new(
    ///     vec![0, 10, 15],  // k-indices for Gamma, X, M
    ///     vec![0, 1, 2, 3], // bands 1-4 (0-based)
    /// );
    /// let (rx, handle) = driver.run_streaming_filtered(filter)?;
    ///
    /// for result in rx {
    ///     // result.bands contains only filtered data
    ///     assert_eq!(result.num_bands(), 4);
    /// }
    /// ```
    pub fn run_streaming_filtered(
        &self,
        filter: SelectiveFilter,
    ) -> Result<(crossbeam_channel::Receiver<CompactBandResult>, std::thread::JoinHandle<Result<DriverStats, DriverError>>), DriverError>
    where
        Self: Sync,
    {
        let stream = Arc::new(FilteredStreamChannel::new(filter, StreamConfig::default()));
        let receiver = stream.add_channel_subscriber();

        let config = self.config.clone();
        let jobs = self.jobs.clone();
        let thread_mode = self.thread_mode.clone();
        let verbose = self.verbose;
        let stream_clone = stream.clone();

        let handle = std::thread::spawn(move || {
            let driver = BulkDriver {
                config,
                jobs,
                thread_mode,
                verbose,
            };
            // FilteredStreamChannel implements OutputChannelSink, so use Stream variant
            let channel = OutputChannel::Stream(stream_clone);
            driver.run_with_channel(channel)
        });

        Ok((receiver, handle))
    }

    /// Run with streaming from a SelectiveSpec (from config).
    ///
    /// Convenience method that creates a filter from the bulk config's
    /// selective specification.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let spec = SelectiveSpec {
    ///     k_indices: vec![0, 10, 15],
    ///     k_labels: vec![],
    ///     bands: vec![1, 2, 3, 4], // 1-based
    /// };
    /// let (rx, handle) = driver.run_streaming_selective(&spec)?;
    /// ```
    pub fn run_streaming_selective(
        &self,
        spec: &SelectiveSpec,
    ) -> Result<(crossbeam_channel::Receiver<CompactBandResult>, std::thread::JoinHandle<Result<DriverStats, DriverError>>), DriverError>
    where
        Self: Sync,
    {
        self.run_streaming_filtered(SelectiveFilter::from_spec(spec))
    }

    /// Execute a single job.
    fn execute_job(&self, expanded: &ExpandedJob) -> Result<JobResult, JobError> {
        match &expanded.job_type {
            ExpandedJobType::Maxwell(job) => self.execute_maxwell_job(expanded.index, job),
            ExpandedJobType::EA(spec) => self.execute_ea_job(expanded.index, spec),
        }
    }

    /// Execute a Maxwell band structure job.
    fn execute_maxwell_job(&self, index: usize, job: &mpb2d_core::bandstructure::BandStructureJob) -> Result<JobResult, JobError> {
        let start = Instant::now();

        // Select backend
        #[cfg(feature = "cuda")]
        let backend = CudaBackend::new();
        #[cfg(not(feature = "cuda"))]
        let backend = CpuBackend::new();

        // Run the solver
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            bandstructure::run_with_options(
                backend,
                job,
                Verbosity::Quiet,
                RunOptions::default(),
            )
        }));

        match result {
            Ok(band_result) => Ok(JobResult {
                index,
                result: JobResultType::Maxwell(band_result),
                duration: start.elapsed(),
                notes: vec![],
            }),
            Err(e) => {
                let msg = if let Some(s) = e.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                Err(JobError {
                    index,
                    message: msg,
                })
            }
        }
    }

    /// Execute an EA eigenvalue problem.
    fn execute_ea_job(&self, index: usize, spec: &EAJobSpec) -> Result<JobResult, JobError> {
        use mpb2d_core::drivers::single_solve;
        use mpb2d_core::operators::EAOperatorBuilder;

        let start = Instant::now();

        // Select backend
        #[cfg(feature = "cuda")]
        let backend = CudaBackend::new();
        #[cfg(not(feature = "cuda"))]
        let backend = CpuBackend::new();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Read input data files
            let potential = read_f64_binary(&spec.potential_path)
                .map_err(|e| format!("failed to read potential: {}", e))?;
            let mass_inv = read_f64_binary(&spec.mass_inv_path)
                .map_err(|e| format!("failed to read mass_inv: {}", e))?;
            let vg = spec.vg_path.as_ref().map(|p| {
                read_f64_binary(p)
                    .map_err(|e| format!("failed to read vg: {}", e))
            }).transpose()?;

            let nx = spec.grid.nx;
            let ny = spec.grid.ny;
            let [lx, ly] = spec.domain_size;
            let dx = lx / nx as f64;
            let dy = ly / ny as f64;

            // Build EAOperator
            let mut builder = EAOperatorBuilder::new(backend.clone(), nx, ny)
                .with_spacing(dx, dy)
                .with_eta(spec.eta)
                .with_potential(potential)
                .with_mass_inv(mass_inv);

            if let Some(vg_data) = vg {
                builder = builder.with_vg(vg_data);
            }

            let mut operator = builder.build();

            // Run the solve
            let solve_result = single_solve::solve(&mut operator, None, &spec.solve_config);

            Ok::<_, String>(EAJobResult {
                eigenvalues: solve_result.eigenvalues,
                eigenvectors: solve_result.eigenvectors,
                grid_dims: [nx, ny],
                n_iterations: solve_result.iterations,
                converged: solve_result.converged,
            })
        }));

        match result {
            Ok(Ok(ea_result)) => Ok(JobResult {
                index,
                result: JobResultType::EA(ea_result),
                duration: start.elapsed(),
                notes: vec![],
            }),
            Ok(Err(msg)) => Err(JobError {
                index,
                message: msg,
            }),
            Err(e) => {
                let msg = if let Some(s) = e.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                Err(JobError {
                    index,
                    message: msg,
                })
            }
        }
    }

    /// Check if this is a single EA job (should run without threading with progress).
    fn is_single_ea_job(&self) -> bool {
        self.jobs.len() == 1
            && matches!(&self.jobs[0].job_type, ExpandedJobType::EA(_))
    }

    /// Execute a single EA job with a detailed progress bar.
    ///
    /// This is used when there's exactly one EA job. Threading is disabled
    /// and a detailed progress bar shows iteration count, trace, and relative
    /// trace change.
    fn run_single_ea_with_progress(&self) -> Result<DriverStats, DriverError> {
        use mpb2d_core::drivers::single_solve;
        use mpb2d_core::operators::EAOperatorBuilder;

        let expanded = &self.jobs[0];
        let spec = match &expanded.job_type {
            ExpandedJobType::EA(s) => s,
            _ => unreachable!("run_single_ea_with_progress called for non-EA job"),
        };

        // Select backend (no threading)
        #[cfg(feature = "cuda")]
        let backend = CudaBackend::new();
        #[cfg(not(feature = "cuda"))]
        let backend = CpuBackend::new();

        // Read input data
        let potential = read_f64_binary(&spec.potential_path)
            .map_err(|e| DriverError::ConfigError(format!("failed to read potential: {}", e)))?;
        let mass_inv = read_f64_binary(&spec.mass_inv_path)
            .map_err(|e| DriverError::ConfigError(format!("failed to read mass_inv: {}", e)))?;
        let vg = spec.vg_path.as_ref().map(|p| {
            read_f64_binary(p)
                .map_err(|e| DriverError::ConfigError(format!("failed to read vg: {}", e)))
        }).transpose()?;

        let nx = spec.grid.nx;
        let ny = spec.grid.ny;
        let [lx, ly] = spec.domain_size;
        let dx = lx / nx as f64;
        let dy = ly / ny as f64;

        // Build EAOperator
        let mut builder = EAOperatorBuilder::new(backend.clone(), nx, ny)
            .with_spacing(dx, dy)
            .with_eta(spec.eta)
            .with_potential(potential)
            .with_mass_inv(mass_inv);

        if let Some(vg_data) = vg {
            builder = builder.with_vg(vg_data);
        }

        let mut operator = builder.build();

        // Estimate spectral properties (15 power iterations)
        let (lambda_max, v_min, spectral_spread) = operator.estimate_condition_number(15);

        // Build adaptive FFT preconditioner (for display only)
        let preconditioner = operator.build_preconditioner();
        let precond_summary = preconditioner.format_summary();

        // Decide whether to use preconditioner based on eigenvalue sign
        // For negative eigenvalue problems (v_min < 0), the FFT preconditioner
        // doesn't help and may hurt convergence. Skip it.
        let use_preconditioner = v_min >= 0.0;
        let precond_note = if use_preconditioner {
            "(enabled)"
        } else {
            "(disabled for negative eigenvalue problem)"
        };

        // Print header with spectral info
        println!("╭─────────────────────────────────────────────────╮");
        println!("│      MPB2D Single EA Solve (with progress)     │");
        println!("╰─────────────────────────────────────────────────╯");
        println!();
        println!("  Grid: {}×{}", spec.grid.nx, spec.grid.ny);
        println!("  η: {:.4}", spec.eta);
        println!("  n_bands: {}", spec.solve_config.n_bands);
        println!("  max_iter: {}", spec.solve_config.max_iterations);
        println!("  tolerance: {:.2e}", spec.solve_config.tolerance);
        println!("  Spectrum: λ_max={:.6}, V_min={:.6}, spread={:.4}", lambda_max, v_min, spectral_spread);
        println!("  {} {}", precond_summary, precond_note);
        println!();

        let start_time = Instant::now();

        // Create progress bar for iterations
        let max_iter = spec.solve_config.max_iterations;
        let pb = ProgressBar::new(max_iter as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} iter {pos:>4}/{len} │ trace={msg}")
                .unwrap()
                .progress_chars("█▓░"),
        );

        // Run the solve
        // Use preconditioner only for positive eigenvalue problems
        let solve_result = if use_preconditioner {
            let mut precond = operator.build_preconditioner();
            single_solve::solve_with_progress(
                &mut operator,
                Some(&mut precond),
                &spec.solve_config,
                |progress: &ProgressInfo| {
                    pb.set_position((progress.iteration + 1) as u64);
                    let trace_str = match progress.trace_rel_change {
                        Some(change) if change < f64::INFINITY => {
                            format!("{:.6} (Δ={:.2e})", progress.trace, change)
                        }
                        _ => format!("{:.6}", progress.trace),
                    };
                    pb.set_message(trace_str);
                },
            )
        } else {
            // For negative eigenvalue problems, use solve without preconditioner
            // (still get progress via the progress bar based on iterations)
            single_solve::solve_with_progress(
                &mut operator,
                None,
                &spec.solve_config,
                |progress: &ProgressInfo| {
                    pb.set_position((progress.iteration + 1) as u64);
                    let trace_str = match progress.trace_rel_change {
                        Some(change) if change < f64::INFINITY => {
                            format!("{:.6} (Δ={:.2e})", progress.trace, change)
                        }
                        _ => format!("{:.6}", progress.trace),
                    };
                    pb.set_message(trace_str);
                },
            )
        };

        pb.finish_and_clear();

        let elapsed = start_time.elapsed();

        // Print result summary
        println!();
        if solve_result.converged {
            println!(
                "✓ Converged in {} iterations ({:.2}s)",
                solve_result.iterations,
                elapsed.as_secs_f64()
            );
        } else {
            println!(
                "⚠ Did not converge after {} iterations ({:.2}s)",
                solve_result.iterations,
                elapsed.as_secs_f64()
            );
        }

        println!();
        println!("Eigenvalues:");
        for (i, &ev) in solve_result.eigenvalues.iter().enumerate() {
            println!("  band {:>2}: {:.10}", i + 1, ev);
        }

        // Setup output writer and write result
        let output_writer = OutputWriter::new(&self.config.output, &self.jobs)
            .map_err(|e| DriverError::OutputError(e.to_string()))?;
        let mut output_writer = output_writer;

        let job_result = JobResult {
            index: expanded.index,
            result: JobResultType::EA(EAJobResult {
                eigenvalues: solve_result.eigenvalues,
                eigenvectors: solve_result.eigenvectors,
                grid_dims: [nx, ny],
                n_iterations: solve_result.iterations,
                converged: solve_result.converged,
            }),
            duration: elapsed,
            notes: vec![],
        };

        output_writer
            .write_result(expanded, &job_result)
            .map_err(|e| DriverError::OutputError(e.to_string()))?;
        output_writer
            .finalize()
            .map_err(|e| DriverError::OutputError(e.to_string()))?;

        println!();
        println!("Output written to: {}", self.config.output.directory.display());

        Ok(DriverStats {
            total_jobs: 1,
            completed: 1,
            failed: 0,
            total_time: elapsed,
            errors: vec![],
            adaptive_summary: None,
        })
    }

    /// Execute a single EA job with progress bar and channel output.
    ///
    /// This combines the detailed progress display with streaming/batch output.
    fn run_single_ea_with_channel(&self, channel: OutputChannel) -> Result<DriverStats, DriverError> {
        use mpb2d_core::drivers::single_solve;
        use mpb2d_core::operators::EAOperatorBuilder;

        let expanded = &self.jobs[0];
        let spec = match &expanded.job_type {
            ExpandedJobType::EA(s) => s,
            _ => unreachable!("run_single_ea_with_channel called for non-EA job"),
        };

        // Select backend (no threading)
        #[cfg(feature = "cuda")]
        let backend = CudaBackend::new();
        #[cfg(not(feature = "cuda"))]
        let backend = CpuBackend::new();

        // Read input data
        let potential = read_f64_binary(&spec.potential_path)
            .map_err(|e| DriverError::ConfigError(format!("failed to read potential: {}", e)))?;
        let mass_inv = read_f64_binary(&spec.mass_inv_path)
            .map_err(|e| DriverError::ConfigError(format!("failed to read mass_inv: {}", e)))?;
        let vg = spec.vg_path.as_ref().map(|p| {
            read_f64_binary(p)
                .map_err(|e| DriverError::ConfigError(format!("failed to read vg: {}", e)))
        }).transpose()?;

        let nx = spec.grid.nx;
        let ny = spec.grid.ny;
        let [lx, ly] = spec.domain_size;
        let dx = lx / nx as f64;
        let dy = ly / ny as f64;

        // Build EAOperator
        let mut builder = EAOperatorBuilder::new(backend.clone(), nx, ny)
            .with_spacing(dx, dy)
            .with_eta(spec.eta)
            .with_potential(potential)
            .with_mass_inv(mass_inv);

        if let Some(vg_data) = vg {
            builder = builder.with_vg(vg_data);
        }

        let mut operator = builder.build();

        // Estimate spectral properties (15 power iterations)
        let (lambda_max, v_min, spectral_spread) = operator.estimate_condition_number(15);

        // Build adaptive FFT preconditioner (for display only)
        let preconditioner = operator.build_preconditioner();
        let precond_summary = preconditioner.format_summary();

        // Decide whether to use preconditioner based on eigenvalue sign
        // For negative eigenvalue problems (v_min < 0), the FFT preconditioner
        // doesn't help and may hurt convergence. Skip it.
        let use_preconditioner = v_min >= 0.0;
        let precond_note = if use_preconditioner {
            "(enabled)"
        } else {
            "(disabled for negative eigenvalue problem)"
        };

        // Print header with spectral info
        println!("╭─────────────────────────────────────────────────╮");
        println!("│      MPB2D Single EA Solve (with progress)     │");
        println!("╰─────────────────────────────────────────────────╯");
        println!();
        println!("  Grid: {}×{}", spec.grid.nx, spec.grid.ny);
        println!("  η: {:.4}", spec.eta);
        println!("  n_bands: {}", spec.solve_config.n_bands);
        println!("  max_iter: {}", spec.solve_config.max_iterations);
        println!("  tolerance: {:.2e}", spec.solve_config.tolerance);
        println!("  Spectrum: λ_max={:.6}, V_min={:.6}, spread={:.4}", lambda_max, v_min, spectral_spread);
        println!("  {} {}", precond_summary, precond_note);
        println!();

        let start_time = Instant::now();

        // Create progress bar for iterations
        let max_iter = spec.solve_config.max_iterations;
        let pb = ProgressBar::new(max_iter as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} iter {pos:>4}/{len} │ trace={msg}")
                .unwrap()
                .progress_chars("█▓░"),
        );

        // Run the solve
        // Use preconditioner only for positive eigenvalue problems
        let solve_result = if use_preconditioner {
            let mut precond = operator.build_preconditioner();
            single_solve::solve_with_progress(
                &mut operator,
                Some(&mut precond),
                &spec.solve_config,
                |progress: &ProgressInfo| {
                    pb.set_position((progress.iteration + 1) as u64);
                    let trace_str = match progress.trace_rel_change {
                        Some(change) if change < f64::INFINITY => {
                            format!("{:.6} (Δ={:.2e})", progress.trace, change)
                        }
                        _ => format!("{:.6}", progress.trace),
                    };
                    pb.set_message(trace_str);
                },
            )
        } else {
            // For negative eigenvalue problems, use solve without preconditioner
            single_solve::solve_with_progress(
                &mut operator,
                None,
                &spec.solve_config,
                |progress: &ProgressInfo| {
                    pb.set_position((progress.iteration + 1) as u64);
                    let trace_str = match progress.trace_rel_change {
                        Some(change) if change < f64::INFINITY => {
                            format!("{:.6} (Δ={:.2e})", progress.trace, change)
                        }
                        _ => format!("{:.6}", progress.trace),
                    };
                    pb.set_message(trace_str);
                },
            )
        };

        pb.finish_and_clear();

        let elapsed = start_time.elapsed();

        // Print result summary
        println!();
        if solve_result.converged {
            println!(
                "✓ Converged in {} iterations ({:.2}s)",
                solve_result.iterations,
                elapsed.as_secs_f64()
            );
        } else {
            println!(
                "⚠ Did not converge after {} iterations ({:.2}s)",
                solve_result.iterations,
                elapsed.as_secs_f64()
            );
        }

        // Create job result and send through channel
        let job_result = JobResult {
            index: expanded.index,
            result: JobResultType::EA(EAJobResult {
                eigenvalues: solve_result.eigenvalues,
                eigenvectors: solve_result.eigenvectors,
                grid_dims: [nx, ny],
                n_iterations: solve_result.iterations,
                converged: solve_result.converged,
            }),
            duration: elapsed,
            notes: vec![],
        };

        // Send result through channel
        let compact = CompactBandResult::from_job_result(expanded, &job_result);
        if let Err(e) = channel.send(compact) {
            error!("failed to send EA result: {}", e);
        }

        // Close channel and get stats
        let _channel_stats = channel.close().map_err(|e| DriverError::OutputError(e.to_string()))?;

        Ok(DriverStats {
            total_jobs: 1,
            completed: 1,
            failed: 0,
            total_time: elapsed,
            errors: vec![],
            adaptive_summary: None,
        })
    }

    /// Perform a dry run: expand jobs and report statistics without executing.
    pub fn dry_run(&self) -> DryRunStats {
        let total_jobs = self.jobs.len();

        // Collect parameter statistics
        let mut param_counts = HashMap::new();

        if let Some(ref range) = self.config.ranges.eps_bg {
            param_counts.insert("eps_bg", range.count());
        }
        if let Some(ref range) = self.config.ranges.resolution {
            param_counts.insert("resolution", range.count());
        }
        if let Some(ref pols) = self.config.ranges.polarization {
            param_counts.insert("polarization", pols.len());
        }
        if let Some(ref types) = self.config.ranges.lattice_type {
            param_counts.insert("lattice_type", types.len());
        }

        for (i, atom_range) in self.config.ranges.atoms.iter().enumerate() {
            if atom_range.radius.is_some() {
                param_counts.insert(
                    Box::leak(format!("atom{}_radius", i).into_boxed_str()),
                    atom_range.radius.as_ref().unwrap().count(),
                );
            }
            if atom_range.pos_x.is_some() {
                param_counts.insert(
                    Box::leak(format!("atom{}_pos_x", i).into_boxed_str()),
                    atom_range.pos_x.as_ref().unwrap().count(),
                );
            }
            if atom_range.pos_y.is_some() {
                param_counts.insert(
                    Box::leak(format!("atom{}_pos_y", i).into_boxed_str()),
                    atom_range.pos_y.as_ref().unwrap().count(),
                );
            }
        }

        // Create adaptive manager to get initial thread count for display
        let adaptive_mgr = match &self.thread_mode {
            ThreadMode::Fixed(n) => AdaptiveThreadManager::fixed(*n),
            ThreadMode::Adaptive => AdaptiveThreadManager::new(AdaptiveConfig::default()),
        };

        DryRunStats {
            total_jobs,
            thread_mode: self.thread_mode_str(Some(&adaptive_mgr)),
            param_counts,
            config: self.config.clone(),
            jobs: self.jobs.clone(),
        }
    }

    /// Run a benchmark: real solves but skip all file output.
    /// Useful for pure performance measurement without I/O overhead.
    pub fn benchmark(&self) -> Result<DriverStats, DriverError> {
        if self.jobs.is_empty() {
            warn!("no jobs to execute (parameter ranges resulted in zero configurations)");
            return Ok(DriverStats::default());
        }

        // Create adaptive thread manager
        let adaptive_mgr = match &self.thread_mode {
            ThreadMode::Fixed(n) => Arc::new(AdaptiveThreadManager::fixed(*n)),
            ThreadMode::Adaptive => Arc::new(AdaptiveThreadManager::new(AdaptiveConfig::default())),
        };

        // Print pre-run report
        let report = PreRunReport::build(
            &self.config,
            &self.jobs,
            &self.thread_mode_str(Some(&adaptive_mgr)),
        );
        report.print();
        println!("  Mode: BENCHMARK (real solves, no file output)");
        println!();

        // Setup thread pool
        let pool_threads = match &self.thread_mode {
            ThreadMode::Fixed(n) => *n,
            ThreadMode::Adaptive => num_cpus::get(),
        };

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(pool_threads)
            .build()
            .map_err(|e| DriverError::ThreadPoolError(e.to_string()))?;

        // Progress tracking
        let completed = Arc::new(AtomicUsize::new(0));
        let failed = Arc::new(AtomicUsize::new(0));
        let errors = Arc::new(Mutex::new(Vec::new()));

        // Create progress bar
        let pb = ProgressBar::new(self.jobs.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("█▓░"),
        );

        let start_time = Instant::now();
        let adaptive_mgr_clone = adaptive_mgr.clone();
        let verbose = self.verbose;

        // Execute jobs in parallel (no output writing)
        pool.install(|| {
            self.jobs.par_iter().for_each(|expanded_job| {
                let job_start = Instant::now();
                let result = self.execute_job(expanded_job);
                let job_duration = job_start.elapsed();

                // Record timing for adaptive management
                if let Some(event) = adaptive_mgr_clone.record_job(job_duration) {
                    if event.reason != AdjustmentReason::Initial && verbose {
                        debug!(
                            "[adaptive] {} → {} threads ({})",
                            event.from_threads, event.to_threads, event.reason
                        );
                    }
                }

                match result {
                    Ok(_job_result) => {
                        // No output writing - just count
                        completed.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(e) => {
                        failed.fetch_add(1, Ordering::Relaxed);
                        errors.lock().push(e);
                    }
                }

                pb.inc(1);
            });
        });

        pb.finish_and_clear();

        let total_time = start_time.elapsed();
        let completed_count = completed.load(Ordering::Relaxed);
        let failed_count = failed.load(Ordering::Relaxed);

        // Get adaptive summary
        let adaptive_summary = adaptive_mgr.get_summary();

        // Print summary
        println!();
        if failed_count == 0 {
            println!(
                "✓ {} jobs benchmarked in {:.2}s ({:.1} jobs/s)",
                completed_count,
                total_time.as_secs_f64(),
                completed_count as f64 / total_time.as_secs_f64()
            );
        } else {
            println!(
                "⚠ {}/{} jobs completed, {} failed in {:.2}s",
                completed_count,
                self.jobs.len(),
                failed_count,
                total_time.as_secs_f64()
            );
        }

        // Show adaptive thread summary if applicable
        if adaptive_mgr.is_adaptive() && adaptive_summary.total_adjustments > 0 {
            println!(
                "  threads: {} → {} ({} adjustments)",
                adaptive_summary.initial_threads,
                adaptive_summary.final_threads,
                adaptive_summary.total_adjustments
            );
            if self.verbose {
                println!("{}", adaptive_summary.format_log());
            }
        }

        if failed_count > 0 {
            let errs = errors.lock();
            for err in errs.iter().take(5) {
                error!("job {} failed: {}", err.index, err.message);
            }
            if errs.len() > 5 {
                error!("... and {} more errors", errs.len() - 5);
            }
        }

        // Print profiling breakdown
        println!();
        print_profile();

        Ok(DriverStats {
            total_jobs: self.jobs.len(),
            completed: completed_count,
            failed: failed_count,
            total_time,
            errors: Arc::try_unwrap(errors).unwrap_or_default().into_inner(),
            adaptive_summary: Some(adaptive_summary),
        })
    }

    /// Run a stress test with simulated job execution (no actual computation).
    /// Useful for testing the adaptive thread manager.
    pub fn stress_test(&self, job_duration: Duration, vary_duration: bool) -> Result<DriverStats, DriverError> {
        if self.jobs.is_empty() {
            return Ok(DriverStats::default());
        }

        // Create adaptive thread manager
        let adaptive_mgr = match &self.thread_mode {
            ThreadMode::Fixed(n) => Arc::new(AdaptiveThreadManager::fixed(*n)),
            ThreadMode::Adaptive => Arc::new(AdaptiveThreadManager::new(AdaptiveConfig::default())),
        };

        // Print pre-run report
        let report = PreRunReport::build(
            &self.config,
            &self.jobs,
            &self.thread_mode_str(Some(&adaptive_mgr)),
        );
        report.print();
        
        let vary_str = if vary_duration { " ±50%" } else { "" };
        println!("  Mode: STRESS TEST (simulated {:.0}ms{} jobs)", job_duration.as_millis(), vary_str);
        println!();

        let pool_threads = match &self.thread_mode {
            ThreadMode::Fixed(n) => *n,
            ThreadMode::Adaptive => num_cpus::get(),
        };

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(pool_threads)
            .build()
            .map_err(|e| DriverError::ThreadPoolError(e.to_string()))?;

        let completed = Arc::new(AtomicUsize::new(0));
        let adaptive_mgr_clone = adaptive_mgr.clone();

        let pb = ProgressBar::new(self.jobs.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("█▓░"),
        );

        let start_time = Instant::now();
        let base_duration = job_duration;

        pool.install(|| {
            self.jobs.par_iter().enumerate().for_each(|(idx, _job)| {
                // Simulate job execution with optional variance
                let actual_duration = if vary_duration {
                    // Add ±50% variance based on job index for reproducibility
                    let factor = 0.5 + ((idx as f64 * 0.618033988749895) % 1.0);
                    Duration::from_secs_f64(base_duration.as_secs_f64() * factor)
                } else {
                    base_duration
                };
                
                std::thread::sleep(actual_duration);

                // Record for adaptive management
                if let Some(event) = adaptive_mgr_clone.record_job(actual_duration) {
                    if event.reason != AdjustmentReason::Initial {
                        // Use println for stress test visibility (goes above progress bar)
                        pb.suspend(|| {
                            println!(
                                "  [adaptive] {} → {} threads ({})",
                                event.from_threads, event.to_threads, event.reason
                            );
                        });
                    }
                }

                completed.fetch_add(1, Ordering::Relaxed);
                pb.inc(1);
            });
        });

        pb.finish_and_clear();

        let total_time = start_time.elapsed();
        let completed_count = completed.load(Ordering::Relaxed);
        let adaptive_summary = adaptive_mgr.get_summary();

        println!();
        println!(
            "✓ {} jobs simulated in {:.2}s",
            completed_count,
            total_time.as_secs_f64()
        );

        if adaptive_mgr.is_adaptive() {
            println!(
                "  threads: {} → {} ({} adjustments)",
                adaptive_summary.initial_threads,
                adaptive_summary.final_threads,
                adaptive_summary.total_adjustments
            );
            if adaptive_summary.total_adjustments > 0 {
                println!("{}", adaptive_summary.format_log());
            }
        }

        Ok(DriverStats {
            total_jobs: self.jobs.len(),
            completed: completed_count,
            failed: 0,
            total_time,
            errors: vec![],
            adaptive_summary: Some(adaptive_summary),
        })
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Read a binary file containing f64 values in row-major (C-order) layout.
fn read_f64_binary(path: &std::path::Path) -> Result<Vec<f64>, std::io::Error> {
    use std::fs::File;
    use std::io::Read;

    let mut file = File::open(path)?;
    let file_len = file.metadata()?.len() as usize;

    // Each f64 is 8 bytes
    if file_len % 8 != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "file size {} is not a multiple of 8 (expected f64 array)",
                file_len
            ),
        ));
    }

    let n_values = file_len / 8;
    let mut data = vec![0.0f64; n_values];

    // Read directly into the slice
    // SAFETY: f64 has no alignment requirements stricter than 8, and we're
    // reading exactly the right number of bytes
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, file_len)
    };
    file.read_exact(bytes)?;

    // Handle endianness if needed (assuming little-endian for now)
    // TODO: Add endianness detection/conversion if needed

    Ok(data)
}

// ============================================================================
// Statistics
// ============================================================================

/// Statistics from a completed bulk run.
#[derive(Debug, Default)]
pub struct DriverStats {
    /// Total number of jobs
    pub total_jobs: usize,

    /// Number of successfully completed jobs
    pub completed: usize,

    /// Number of failed jobs
    pub failed: usize,

    /// Total execution time
    pub total_time: Duration,

    /// Errors encountered
    pub errors: Vec<JobError>,

    /// Adaptive thread manager summary (if used)
    pub adaptive_summary: Option<crate::adaptive::AdaptiveSummary>,
}

/// Statistics from a dry run.
#[derive(Debug)]
pub struct DryRunStats {
    /// Total number of jobs that would be executed
    pub total_jobs: usize,

    /// Thread mode description
    pub thread_mode: String,

    /// Count of values for each swept parameter
    pub param_counts: HashMap<&'static str, usize>,

    /// Full configuration (for detailed report)
    pub config: BulkConfig,

    /// Expanded jobs
    pub jobs: Vec<ExpandedJob>,
}

impl DryRunStats {
    /// Print a rich report for dry run.
    pub fn print_report(&self) {
        let report = PreRunReport::build(&self.config, &self.jobs, &self.thread_mode);
        report.print();

        println!("  [DRY RUN - no jobs will be executed]");
        println!();

        if !self.param_counts.is_empty() {
            println!("Parameter sweep details:");
            for (param, count) in &self.param_counts {
                println!("  {}: {} values", param, count);
            }
        }
    }
}

impl std::fmt::Display for DryRunStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Dry Run Statistics")?;
        writeln!(f, "==================")?;
        writeln!(f, "Total jobs: {}", self.total_jobs)?;
        writeln!(f, "Threads: {}", self.thread_mode)?;
        writeln!(f, "\nSwept parameters:")?;
        for (param, count) in &self.param_counts {
            writeln!(f, "  {}: {} values", param, count)?;
        }
        Ok(())
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Driver-level errors.
#[derive(Debug, thiserror::Error)]
pub enum DriverError {
    #[error("failed to create thread pool: {0}")]
    ThreadPoolError(String),

    #[error("output error: {0}")]
    OutputError(String),

    #[error("configuration error: {0}")]
    ConfigError(String),
}
