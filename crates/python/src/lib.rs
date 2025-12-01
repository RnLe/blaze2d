//! Python bindings for MPB2D.
//!
//! This crate provides Python bindings for the MPB2D photonic crystal band
//! structure solver, with special support for streaming results in real-time.
//!
//! # Features
//!
//! - **BulkDriver**: High-level driver for parameter sweep computations
//! - **Streaming**: Real-time iteration over results for live plotting
//! - **Batched I/O**: Optimized disk I/O for large sweeps
//!
//! # Example
//!
//! ```python
//! from mpb2d import BulkDriver
//!
//! driver = BulkDriver("sweep.toml")
//! print(f"Will run {driver.job_count} jobs")
//!
//! # Streaming mode - process results as they complete
//! for result in driver.run_streaming():
//!     print(f"Job {result['job_index']}: {result['num_bands']} bands")
//!
//! # Or batch mode for large sweeps
//! stats = driver.run_batched(buffer_size_mb=10)
//! print(f"Completed {stats['completed']} jobs in {stats['total_time_secs']:.2f}s")
//! ```

#[cfg(feature = "bindings")]
mod streaming;

#[cfg(feature = "bindings")]
mod py {
    use pyo3::prelude::*;

    use crate::streaming;

    /// MPB2D Python module.
    ///
    /// Provides access to the high-performance photonic crystal band structure
    /// solver with streaming support for real-time analysis.
    #[pymodule]
    fn mpb2d(m: &Bound<'_, PyModule>) -> PyResult<()> {
        // Module documentation
        m.add("__doc__", "MPB2D: Fast 2D photonic crystal band structure solver")?;
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;

        // Register streaming classes
        streaming::register_streaming(m)?;

        Ok(())
    }
}

#[cfg(not(feature = "bindings"))]
pub fn bindings_disabled() {
    log::warn!("mpb2d-python compiled without the \"bindings\" feature");
}
