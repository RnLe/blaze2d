//! Python bindings for BLAZE - Band-structure LOBPCG Accelerated Zone Eigensolver.
//!
//! This crate provides Python bindings for the BLAZE 2D photonic crystal band
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
//! from blaze import BulkDriver
//!
//! driver = BulkDriver("sweep.toml")
//! print(f"Will run {driver.job_count} jobs")
//!
//! # Streaming mode - process results as they complete
//! for result in driver.run_streaming():
//!     print(f"Job {result['job_index']}: {result['num_bands']} bands")
//!
//! # Or collect all results at once
//! results, stats = driver.run_collect()
//! print(f"Completed {stats['completed']} jobs in {stats['total_time_secs']:.2f}s")
//! ```

#[cfg(feature = "bindings")]
mod streaming;

#[cfg(feature = "bindings")]
mod py {
    use pyo3::prelude::*;

    use crate::streaming;

    /// BLAZE Python module.
    ///
    /// Provides access to the high-performance 2D photonic crystal band structure
    /// solver with streaming support for real-time analysis.
    #[pymodule]
    fn blaze(m: &Bound<'_, PyModule>) -> PyResult<()> {
        // Module documentation
        m.add("__doc__", "BLAZE: Band-structure LOBPCG Accelerated Zone Eigensolver for 2D Photonic Crystals")?;
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;

        // Register streaming classes
        streaming::register_streaming(m)?;

        Ok(())
    }
}

#[cfg(not(feature = "bindings"))]
pub fn bindings_disabled() {
    log::warn!("blaze2d compiled without the \"bindings\" feature");
}
