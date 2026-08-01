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
mod operator_data;
#[cfg(feature = "bindings")]
mod streaming;

#[cfg(feature = "bindings")]
mod py {
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    use crate::operator_data;
    use crate::streaming;

    /// Run `git <args>` in the blaze2d repo at runtime; None on any failure.
    fn runtime_git(args: &[&str]) -> Option<String> {
        // The installed module does not live in the repo, so anchor git at the
        // source directory recorded at compile time.
        let repo_dir = env!("CARGO_MANIFEST_DIR");
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(repo_dir)
            .args(args)
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let text = String::from_utf8(output.stdout).ok()?;
        Some(text.trim().to_string())
    }

    /// Build provenance of the native module.
    ///
    /// Returns a dict with:
    /// - ``version``: blaze2d crate version
    /// - ``git_sha``: git SHA of the blaze2d repo at build time (compile-time
    ///   embedded by build.rs; runtime ``git rev-parse`` fallback; "unknown"
    ///   if neither is available)
    /// - ``git_dirty``: "clean"/"dirty"/"unknown" working-tree state at build time
    /// - ``git_sha_source``: "build_script", "runtime_git", or "unknown"
    /// - ``profile``: "release" or "debug"
    #[pyfunction]
    fn build_info(py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("version", env!("CARGO_PKG_VERSION"))?;

        let (sha, source) = match option_env!("BLAZE2D_GIT_SHA") {
            Some(sha) => (sha.to_string(), "build_script"),
            None => match runtime_git(&["rev-parse", "HEAD"]) {
                Some(sha) => (sha, "runtime_git"),
                None => ("unknown".to_string(), "unknown"),
            },
        };
        dict.set_item("git_sha", sha)?;
        dict.set_item("git_sha_source", source)?;
        dict.set_item(
            "git_dirty",
            option_env!("BLAZE2D_GIT_DIRTY").unwrap_or("unknown"),
        )?;
        dict.set_item(
            "profile",
            if cfg!(debug_assertions) { "debug" } else { "release" },
        )?;
        Ok(dict.into())
    }

    /// BLAZE native Rust module (imported as blaze._native).
    ///
    /// Provides access to the high-performance 2D photonic crystal band structure
    /// solver with streaming support for real-time analysis.
    #[pymodule(name = "_native")]
    fn blaze_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
        // Module documentation
        m.add(
            "__doc__",
            "BLAZE: Band-structure LOBPCG Accelerated Zone Eigensolver for 2D Photonic Crystals",
        )?;
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;

        // Build provenance
        m.add_function(wrap_pyfunction!(build_info, m)?)?;

        // Register streaming classes
        streaming::register_streaming(m)?;

        // Register EA extraction classes
        operator_data::register_operator_data(m)?;

        Ok(())
    }
}

#[cfg(not(feature = "bindings"))]
pub fn bindings_disabled() {
    log::warn!("blaze2d compiled without the \"bindings\" feature");
}
