//! Python streaming interface for bulk driver.
//!
//! This module provides Python bindings for streaming band structure results
//! in real-time, enabling live plotting and interactive analysis.
//!
//! # Example Usage (Python)
//!
//! ```python
//! from mpb2d import BulkDriver
//! import matplotlib.pyplot as plt
//!
//! driver = BulkDriver("sweep.toml")
//!
//! # Streaming iteration
//! plt.ion()
//! for result in driver.run_streaming():
//!     plt.clf()
//!     for band in result['bands']:
//!         plt.plot(result['distances'], band)
//!     plt.pause(0.01)
//!
//! # Or with callback
//! def on_result(result):
//!     print(f"Job {result['job_index']} complete")
//!
//! driver.run_with_callback(on_result)
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossbeam_channel::{Receiver, TryRecvError};
use parking_lot::Mutex;
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use mpb2d_bulk_driver::{
    BatchConfig, BulkConfig, BulkDriver, CompactBandResult, DriverError, DriverStats,
    OutputChannel, StreamChannel, StreamConfig,
};

// ============================================================================
// Python Result Wrapper
// ============================================================================

/// Convert a CompactBandResult to a Python dictionary.
fn result_to_py_dict(py: Python<'_>, result: &CompactBandResult) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);

    // Job metadata
    dict.set_item("job_index", result.job_index)?;

    // Parameters as nested dict
    let params = PyDict::new(py);
    params.set_item("eps_bg", result.params.eps_bg)?;
    params.set_item("resolution", result.params.resolution)?;
    params.set_item("polarization", format!("{:?}", result.params.polarization))?;
    if let Some(ref lt) = result.params.lattice_type {
        params.set_item("lattice_type", lt.as_str())?;
    }

    // Atom parameters
    let atoms_list = PyList::empty(py);
    for atom in &result.params.atoms {
        let atom_dict = PyDict::new(py);
        atom_dict.set_item("index", atom.index)?;
        atom_dict.set_item("pos", (atom.pos[0], atom.pos[1]))?;
        atom_dict.set_item("radius", atom.radius)?;
        atom_dict.set_item("eps_inside", atom.eps_inside)?;
        atoms_list.append(atom_dict)?;
    }
    params.set_item("atoms", atoms_list)?;
    dict.set_item("params", params)?;

    // K-path as list of tuples
    let k_path: Vec<(f64, f64)> = result.k_path.iter().map(|k| (k[0], k[1])).collect();
    dict.set_item("k_path", k_path)?;

    // Distances
    dict.set_item("distances", result.distances.clone())?;

    // Bands as 2D list [k_index][band_index]
    dict.set_item("bands", result.bands.clone())?;

    // Convenience accessors
    dict.set_item("num_k_points", result.num_k_points())?;
    dict.set_item("num_bands", result.num_bands())?;

    Ok(dict.into())
}

// ============================================================================
// Band Result Iterator
// ============================================================================

/// Iterator over streaming band structure results.
///
/// This class implements Python's iterator protocol, allowing use in
/// for-loops and comprehensions.
#[pyclass(name = "BandResultIterator")]
pub struct BandResultIterator {
    /// Receiver channel for results
    receiver: Receiver<CompactBandResult>,

    /// Handle to the computation thread
    handle: Arc<Mutex<Option<JoinHandle<Result<DriverStats, DriverError>>>>>,

    /// Whether iteration has completed
    finished: bool,
}

#[pymethods]
impl BandResultIterator {
    /// Python iterator protocol: return self
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Python iterator protocol: get next item
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        if self.finished {
            return Ok(None);
        }

        // Release GIL while waiting
        let result = py.allow_threads(|| self.receiver.recv());

        match result {
            Ok(band_result) => Ok(Some(result_to_py_dict(py, &band_result)?)),
            Err(_) => {
                // Channel closed - iteration complete
                self.finished = true;
                Ok(None)
            }
        }
    }

    /// Non-blocking attempt to get the next result.
    ///
    /// Returns None if no result is currently available, without blocking.
    /// Raises StopIteration if the computation is complete.
    #[pyo3(name = "try_next")]
    fn try_next(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        if self.finished {
            return Err(PyStopIteration::new_err("iteration complete"));
        }

        match self.receiver.try_recv() {
            Ok(result) => Ok(Some(result_to_py_dict(py, &result)?)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => {
                self.finished = true;
                Err(PyStopIteration::new_err("iteration complete"))
            }
        }
    }

    /// Get next result with timeout.
    ///
    /// Args:
    ///     timeout: Maximum time to wait in seconds
    ///
    /// Returns:
    ///     Result dict, or None if timeout elapsed
    ///
    /// Raises:
    ///     StopIteration: If computation is complete
    #[pyo3(name = "next_timeout")]
    fn next_timeout(&mut self, py: Python<'_>, timeout: f64) -> PyResult<Option<Py<PyDict>>> {
        if self.finished {
            return Err(PyStopIteration::new_err("iteration complete"));
        }

        let duration = Duration::from_secs_f64(timeout);
        let result = py.allow_threads(|| self.receiver.recv_timeout(duration));

        match result {
            Ok(band_result) => Ok(Some(result_to_py_dict(py, &band_result)?)),
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => Ok(None),
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                self.finished = true;
                Err(PyStopIteration::new_err("iteration complete"))
            }
        }
    }

    /// Wait for the computation to complete and return final statistics.
    ///
    /// This should be called after iterating to ensure cleanup.
    #[pyo3(name = "wait")]
    fn wait(&mut self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        // Drain any remaining results
        while self.receiver.try_recv().is_ok() {}

        // Join the computation thread
        let handle = {
            let mut guard = self.handle.lock();
            guard.take()
        };

        if let Some(h) = handle {
            let stats_result = py.allow_threads(|| h.join());

            match stats_result {
                Ok(Ok(stats)) => {
                    let dict = PyDict::new(py);
                    dict.set_item("total_jobs", stats.total_jobs)?;
                    dict.set_item("completed", stats.completed)?;
                    dict.set_item("failed", stats.failed)?;
                    dict.set_item("total_time_secs", stats.total_time.as_secs_f64())?;
                    Ok(dict.into())
                }
                Ok(Err(e)) => Err(PyRuntimeError::new_err(format!("driver error: {}", e))),
                Err(_) => Err(PyRuntimeError::new_err("computation thread panicked")),
            }
        } else {
            let dict = PyDict::new(py);
            dict.set_item("note", "already joined")?;
            Ok(dict.into())
        }
    }

    /// Check if more results are available.
    #[getter]
    fn is_finished(&self) -> bool {
        self.finished
    }

    /// Get approximate number of pending results.
    #[getter]
    fn pending(&self) -> usize {
        self.receiver.len()
    }
}

// ============================================================================
// Bulk Driver Python Wrapper
// ============================================================================

/// Python wrapper for the bulk driver with streaming support.
///
/// Example:
///     driver = BulkDriver("config.toml")
///     for result in driver.run_streaming():
///         print(f"Job {result['job_index']}: {result['num_bands']} bands")
#[pyclass(name = "BulkDriver")]
pub struct BulkDriverPy {
    /// Path to configuration file
    config_path: PathBuf,

    /// Parsed configuration (cached)
    config: Option<BulkConfig>,

    /// Requested thread count (-1 for adaptive, 0 for default)
    threads: i32,
}

#[pymethods]
impl BulkDriverPy {
    /// Create a new bulk driver from a TOML configuration file.
    ///
    /// Args:
    ///     config_path: Path to the bulk configuration TOML file
    ///     threads: Number of threads (-1 for adaptive, 0 for default)
    #[new]
    #[pyo3(signature = (config_path, threads=0))]
    fn new(config_path: &str, threads: i32) -> PyResult<Self> {
        let path = PathBuf::from(config_path);

        // Validate file exists
        if !path.exists() {
            return Err(PyValueError::new_err(format!(
                "configuration file not found: {}",
                config_path
            )));
        }

        // Try to parse config
        let config = BulkConfig::from_file(&path)
            .map_err(|e| PyValueError::new_err(format!("invalid configuration: {}", e)))?;

        Ok(Self {
            config_path: path,
            config: Some(config),
            threads,
        })
    }

    /// Get the number of jobs that will be executed.
    #[getter]
    fn job_count(&self) -> PyResult<usize> {
        let config = self
            .config
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("configuration not loaded"))?;
        Ok(config.total_jobs())
    }

    /// Get the configuration file path.
    #[getter]
    fn config_path(&self) -> String {
        self.config_path.to_string_lossy().to_string()
    }

    /// Run the computation with streaming output.
    ///
    /// Returns an iterator that yields band structure results as they complete.
    /// Results are dictionaries with keys:
    ///   - job_index: int
    ///   - params: dict (eps_bg, resolution, polarization, atoms, ...)
    ///   - k_path: list of (kx, ky) tuples
    ///   - distances: list of floats
    ///   - bands: 2D list [k_index][band_index]
    ///   - num_k_points: int
    ///   - num_bands: int
    ///
    /// Example:
    ///     for result in driver.run_streaming():
    ///         print(f"Job {result['job_index']}")
    #[pyo3(name = "run_streaming")]
    fn run_streaming(&self, _py: Python<'_>) -> PyResult<BandResultIterator> {
        let config = self
            .config
            .clone()
            .ok_or_else(|| PyRuntimeError::new_err("configuration not loaded"))?;

        let threads = if self.threads == 0 {
            None
        } else {
            Some(self.threads)
        };

        // Create driver and streaming channel
        let driver = BulkDriver::new(config, threads);
        let stream = Arc::new(StreamChannel::new(StreamConfig::default()));
        let receiver = stream.add_channel_subscriber();

        // Clone for the computation thread
        let stream_clone = stream.clone();

        // Spawn computation in background thread
        let handle = thread::spawn(move || {
            let channel = OutputChannel::Stream(stream_clone);
            driver.run_with_channel(channel)
        });

        Ok(BandResultIterator {
            receiver,
            handle: Arc::new(Mutex::new(Some(handle))),
            finished: false,
        })
    }

    /// Run with batch mode (buffered I/O).
    ///
    /// This runs the computation with I/O optimized for large sweeps,
    /// buffering results in memory before writing to disk.
    ///
    /// Args:
    ///     buffer_size_mb: Buffer size in megabytes (default: 10)
    ///
    /// Returns:
    ///     Statistics dictionary with total_jobs, completed, failed, total_time_secs
    #[pyo3(name = "run_batched", signature = (buffer_size_mb=10))]
    fn run_batched(&self, py: Python<'_>, buffer_size_mb: usize) -> PyResult<Py<PyDict>> {
        let config = self
            .config
            .clone()
            .ok_or_else(|| PyRuntimeError::new_err("configuration not loaded"))?;

        let threads = if self.threads == 0 {
            None
        } else {
            Some(self.threads)
        };

        let driver = BulkDriver::new(config, threads);

        let batch_config = BatchConfig {
            buffer_size_bytes: buffer_size_mb * 1024 * 1024,
            ..Default::default()
        };

        // Run with GIL released
        let result = py.allow_threads(|| driver.run_batched(batch_config));

        match result {
            Ok(stats) => {
                let dict = PyDict::new(py);
                dict.set_item("total_jobs", stats.total_jobs)?;
                dict.set_item("completed", stats.completed)?;
                dict.set_item("failed", stats.failed)?;
                dict.set_item("total_time_secs", stats.total_time.as_secs_f64())?;
                dict.set_item(
                    "jobs_per_second",
                    stats.completed as f64 / stats.total_time.as_secs_f64(),
                )?;
                Ok(dict.into())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("driver error: {}", e))),
        }
    }

    /// Run synchronously (legacy mode).
    ///
    /// Returns statistics dictionary after all jobs complete.
    #[pyo3(name = "run")]
    fn run(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let config = self
            .config
            .clone()
            .ok_or_else(|| PyRuntimeError::new_err("configuration not loaded"))?;

        let threads = if self.threads == 0 {
            None
        } else {
            Some(self.threads)
        };

        let driver = BulkDriver::new(config, threads);

        // Run with GIL released
        let result = py.allow_threads(|| driver.run());

        match result {
            Ok(stats) => {
                let dict = PyDict::new(py);
                dict.set_item("total_jobs", stats.total_jobs)?;
                dict.set_item("completed", stats.completed)?;
                dict.set_item("failed", stats.failed)?;
                dict.set_item("total_time_secs", stats.total_time.as_secs_f64())?;
                Ok(dict.into())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("driver error: {}", e))),
        }
    }

    /// Dry run: count jobs without executing.
    ///
    /// Returns statistics about what would be executed.
    #[pyo3(name = "dry_run")]
    fn dry_run(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let config = self
            .config
            .clone()
            .ok_or_else(|| PyRuntimeError::new_err("configuration not loaded"))?;

        let driver = BulkDriver::new(config, None);
        let stats = driver.dry_run();

        let dict = PyDict::new(py);
        dict.set_item("total_jobs", stats.total_jobs)?;
        dict.set_item("thread_mode", stats.thread_mode)?;

        let params = PyDict::new(py);
        for (name, count) in &stats.param_counts {
            params.set_item(*name, *count)?;
        }
        dict.set_item("parameter_counts", params)?;

        Ok(dict.into())
    }

    /// String representation.
    fn __repr__(&self) -> String {
        let job_count = self
            .config
            .as_ref()
            .map(|c| c.total_jobs())
            .unwrap_or(0);
        format!(
            "BulkDriver('{}', jobs={}, threads={})",
            self.config_path.display(),
            job_count,
            if self.threads == 0 {
                "auto".to_string()
            } else if self.threads == -1 {
                "adaptive".to_string()
            } else {
                self.threads.to_string()
            }
        )
    }
}

// ============================================================================
// Module Registration
// ============================================================================

/// Register the streaming classes with the Python module.
pub fn register_streaming(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BulkDriverPy>()?;
    m.add_class::<BandResultIterator>()?;
    Ok(())
}
