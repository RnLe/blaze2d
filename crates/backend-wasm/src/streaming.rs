//! WASM streaming interface for JavaScript/React consumers.
//!
//! This module provides WebAssembly bindings for streaming band structure
//! results to JavaScript applications, enabling real-time visualization
//! in web-based UIs. The interface is designed to mirror the Python bindings
//! for consistent cross-platform usage.
//!
//! # Architecture
//!
//! This WASM backend uses `bulk-driver-core` for shared types (config, results,
//! job expansion) and implements single-threaded job execution directly,
//! avoiding native threading dependencies.
//!
//! # Solver Types
//!
//! The bulk driver supports two solver types:
//!
//! - **Maxwell**: Photonic crystal band structure (k_path, distances, bands)
//! - **EA**: Envelope Approximation for moiré lattices (eigenvalues only)
//!
//! # I/O Modes
//!
//! The driver supports multiple I/O modes:
//!
//! - **Stream**: Real-time emission for live visualization (primary focus)
//! - **Selective**: Filter specific k-points and bands before emission
//! - **Collect**: Gather all results at once
//!
//! # Example Usage (JavaScript)
//!
//! ```javascript
//! import init, { WasmBulkDriver } from 'blaze2d-wasm';
//!
//! async function runSimulation() {
//!     await init();
//!     
//!     // Load config from TOML string
//!     const driver = new WasmBulkDriver(configToml);
//!     console.log(`Running ${driver.jobCount} jobs (solver: ${driver.solverType})`);
//!     
//!     // Callback-based streaming
//!     driver.runWithCallback((result) => {
//!         if (result.result_type === 'maxwell') {
//!             console.log(`Job ${result.job_index}: ${result.num_bands} bands`);
//!             updateBandPlot(result.distances, result.bands);
//!         } else {  // EA
//!             console.log(`Job ${result.job_index}: E₀ = ${result.eigenvalues[0]}`);
//!         }
//!     });
//! }
//! ```

use js_sys::{Array, Date, Function, Object, Reflect};
use wasm_bindgen::prelude::*;

use blaze2d_bulk_driver_core::{
    config::{BulkConfig, Config, Precision, SolverType, SweepValue, parse_and_validate},
    expansion::{ExpandedJob, ExpandedJobType, expand_jobs},
    filter::SelectiveFilter,
    result::{CompactBandResult, CompactResultType, MaxwellResult},
};

use blaze2d_core::backend::SpectralBackend;
use blaze2d_core::drivers::bandstructure::{self, RunOptions};

use blaze2d_backend_cpu::CpuBackend;

// ============================================================================
// WASM Driver Statistics
// ============================================================================

/// Statistics from a WASM bulk driver run.
#[derive(Debug, Clone, Default)]
pub struct WasmDriverStats {
    pub total_jobs: usize,
    pub completed: usize,
    pub failed: usize,
    pub total_time_ms: u128,
}

// ============================================================================
// JS Result Conversion
// ============================================================================

/// Convert a CompactBandResult to a JavaScript object.
fn result_to_js(result: &CompactBandResult) -> Result<JsValue, JsValue> {
    let obj = Object::new();

    // Job metadata
    Reflect::set(&obj, &"job_index".into(), &JsValue::from(result.job_index))?;

    // Parameters as nested object
    let params = Object::new();
    Reflect::set(
        &params,
        &"eps_bg".into(),
        &JsValue::from(result.params.eps_bg),
    )?;
    Reflect::set(
        &params,
        &"resolution".into(),
        &JsValue::from(result.params.resolution as u32),
    )?;
    Reflect::set(
        &params,
        &"polarization".into(),
        &JsValue::from_str(&format!("{:?}", result.params.polarization)),
    )?;
    if let Some(ref lt) = result.params.lattice_type {
        Reflect::set(&params, &"lattice_type".into(), &JsValue::from_str(lt))?;
    }

    // Atom parameters
    let atoms = Array::new();
    for atom in &result.params.atoms {
        let atom_obj = Object::new();
        Reflect::set(
            &atom_obj,
            &"index".into(),
            &JsValue::from(atom.index as u32),
        )?;

        let pos = Array::new();
        pos.push(&JsValue::from(atom.pos[0]));
        pos.push(&JsValue::from(atom.pos[1]));
        Reflect::set(&atom_obj, &"pos".into(), &pos)?;

        Reflect::set(&atom_obj, &"radius".into(), &JsValue::from(atom.radius))?;
        Reflect::set(
            &atom_obj,
            &"eps_inside".into(),
            &JsValue::from(atom.eps_inside),
        )?;
        atoms.push(&atom_obj);
    }
    Reflect::set(&params, &"atoms".into(), &atoms)?;

    // Sweep values (ordered parameter sweep tracking), mirroring the Python
    // bindings so the JS result shape matches `BulkDriver.run_streaming()`.
    let sweep_values = Object::new();
    for (name, value) in &result.params.sweep_values {
        let py_value: JsValue = match value {
            SweepValue::Float(f) => JsValue::from(*f),
            SweepValue::Int(i) => JsValue::from(*i as f64),
            SweepValue::String(s) => JsValue::from_str(s),
        };
        Reflect::set(&sweep_values, &JsValue::from_str(name), &py_value)?;
    }
    Reflect::set(&params, &"sweep_values".into(), &sweep_values)?;

    Reflect::set(&obj, &"params".into(), &params)?;

    // Sweep order string for convenience (e.g. "atom0.radius=0.3|eps_bg=12").
    Reflect::set(
        &obj,
        &"sweep_order".into(),
        &JsValue::from_str(&result.params.sweep_order_string()),
    )?;

    // Result type discriminator and type-specific data
    match &result.result_type {
        CompactResultType::Maxwell(maxwell) => {
            Reflect::set(&obj, &"result_type".into(), &JsValue::from_str("maxwell"))?;
            set_maxwell_fields(&obj, maxwell)?;
        }
        CompactResultType::OperatorData(_) => {
            return Err(JsValue::from_str(
                "operator-data extraction results are not yet exposed to WASM",
            ));
        }
    }

    Ok(obj.into())
}

/// Set Maxwell-specific fields on a JS object.
fn set_maxwell_fields(obj: &Object, maxwell: &MaxwellResult) -> Result<(), JsValue> {
    // K-path as array of [kx, ky] arrays
    let k_path = Array::new();
    for k in &maxwell.k_path {
        let point = Array::new();
        point.push(&JsValue::from(k[0]));
        point.push(&JsValue::from(k[1]));
        k_path.push(&point);
    }
    Reflect::set(obj, &"k_path".into(), &k_path)?;

    // Distances along k-path
    let distances = Array::new();
    for d in &maxwell.distances {
        distances.push(&JsValue::from(*d));
    }
    Reflect::set(obj, &"distances".into(), &distances)?;

    // Bands as 2D array [k_index][band_index]
    let bands = Array::new();
    for k_bands in &maxwell.bands {
        let band_arr = Array::new();
        for omega in k_bands {
            band_arr.push(&JsValue::from(*omega));
        }
        bands.push(&band_arr);
    }
    Reflect::set(obj, &"bands".into(), &bands)?;

    // Convenience properties
    Reflect::set(
        obj,
        &"num_k_points".into(),
        &JsValue::from(maxwell.k_path.len() as u32),
    )?;
    Reflect::set(
        obj,
        &"num_bands".into(),
        &JsValue::from(maxwell.bands.first().map(|b| b.len()).unwrap_or(0) as u32),
    )?;

    Ok(())
}

/// Convert driver statistics to a JS object.
fn stats_to_js(stats: &WasmDriverStats) -> Result<JsValue, JsValue> {
    let result = Object::new();
    Reflect::set(
        &result,
        &"total_jobs".into(),
        &JsValue::from(stats.total_jobs as u32),
    )?;
    Reflect::set(
        &result,
        &"completed".into(),
        &JsValue::from(stats.completed as u32),
    )?;
    Reflect::set(
        &result,
        &"failed".into(),
        &JsValue::from(stats.failed as u32),
    )?;
    Reflect::set(
        &result,
        &"total_time_ms".into(),
        &JsValue::from(stats.total_time_ms as u32),
    )?;

    // Jobs per second (if time > 0)
    let total_secs = stats.total_time_ms as f64 / 1000.0;
    if total_secs > 0.0 {
        Reflect::set(
            &result,
            &"jobs_per_second".into(),
            &JsValue::from(stats.completed as f64 / total_secs),
        )?;
    }

    Ok(result.into())
}

// ============================================================================
// Single-Threaded Job Execution
// ============================================================================

/// Run a single Maxwell job and convert to CompactBandResult.
///
/// Generic over the backend so the same code serves `CpuBackend<f32>` and
/// `CpuBackend<f64>`; the `[solver].precision` dispatch happens at the call
/// site (results are always f64 regardless of storage precision).
fn run_maxwell_job<B: SpectralBackend + Clone>(
    backend: B,
    job: &blaze2d_core::bandstructure::BandStructureJob,
    params: &blaze2d_bulk_driver_core::expansion::JobParams,
    job_index: usize,
) -> Result<CompactBandResult, String> {
    let band_result = bandstructure::run_with_options(backend, job, RunOptions::default());

    // Normalize frequencies (divide by 2π)
    let bands: Vec<Vec<f64>> = band_result
        .bands
        .iter()
        .map(|k_bands: &Vec<f64>| {
            k_bands
                .iter()
                .map(|omega| omega / (2.0 * std::f64::consts::PI))
                .collect()
        })
        .collect();

    Ok(CompactBandResult {
        job_index,
        params: params.clone(),
        result_type: CompactResultType::Maxwell(MaxwellResult {
            k_path: band_result.k_path.clone(),
            distances: band_result.distances.clone(),
            bands,
        }),
    })
}

/// Run a single Maxwell job with k-point streaming, calling callback for each k-point.
fn run_maxwell_job_with_k_streaming<B: SpectralBackend + Clone>(
    backend: B,
    job: &blaze2d_core::bandstructure::BandStructureJob,
    params: &blaze2d_bulk_driver_core::expansion::JobParams,
    job_index: usize,
    callback: &Function,
) -> Result<CompactBandResult, String> {
    use blaze2d_core::drivers::bandstructure::{KPointResult, RunOptions, run_with_k_streaming};

    // Clone params for use in closure
    let params_clone = params.clone();

    // Accumulate all k-point results for the final CompactBandResult
    let accumulated_bands: std::cell::RefCell<Vec<Vec<f64>>> = std::cell::RefCell::new(Vec::new());

    // Disable Γ-reuse so the closing Γ is solved with warm-start; this keeps
    // band-tracking continuity across crossings (otherwise the last segment
    // jumps to band indices from the very first Γ solve).
    let run_options = RunOptions {
        reuse_gamma: false,
        ..RunOptions::default()
    };

    let band_result = run_with_k_streaming(
        backend,
        job,
        run_options,
        |k_result: KPointResult| {
            // Convert KPointResult to JS object and emit
            if let Ok(js_obj) = k_result_to_js(&k_result, job_index, &params_clone) {
                let _ = callback.call1(&JsValue::NULL, &js_obj);
            }

            // Accumulate bands for final result
            // Normalize frequencies (divide by 2π)
            let normalized: Vec<f64> = k_result
                .omegas
                .iter()
                .map(|omega| omega / (2.0 * std::f64::consts::PI))
                .collect();
            accumulated_bands.borrow_mut().push(normalized);
        },
    );

    // Build final CompactBandResult
    let bands: Vec<Vec<f64>> = band_result
        .bands
        .iter()
        .map(|k_bands: &Vec<f64>| {
            k_bands
                .iter()
                .map(|omega| omega / (2.0 * std::f64::consts::PI))
                .collect()
        })
        .collect();

    Ok(CompactBandResult {
        job_index,
        params: params.clone(),
        result_type: CompactResultType::Maxwell(MaxwellResult {
            k_path: band_result.k_path.clone(),
            distances: band_result.distances.clone(),
            bands,
        }),
    })
}

/// Convert a KPointResult to a JavaScript object for streaming.
fn k_result_to_js(
    result: &blaze2d_core::drivers::bandstructure::KPointResult,
    job_index: usize,
    params: &blaze2d_bulk_driver_core::expansion::JobParams,
) -> Result<JsValue, JsValue> {
    let obj = Object::new();

    // Streaming type identifier
    Reflect::set(&obj, &"stream_type".into(), &JsValue::from_str("k_point"))?;

    // Job context
    Reflect::set(&obj, &"job_index".into(), &JsValue::from(job_index as u32))?;

    // K-point info
    Reflect::set(
        &obj,
        &"k_index".into(),
        &JsValue::from(result.k_index as u32),
    )?;
    Reflect::set(
        &obj,
        &"total_k_points".into(),
        &JsValue::from(result.total_k_points as u32),
    )?;

    // K-point coordinates
    let k_point = Array::new();
    k_point.push(&JsValue::from(result.k_point[0]));
    k_point.push(&JsValue::from(result.k_point[1]));
    Reflect::set(&obj, &"k_point".into(), &k_point)?;

    // Distance along path
    Reflect::set(&obj, &"distance".into(), &JsValue::from(result.distance))?;

    // Frequencies (normalized by 2π)
    let omegas = Array::new();
    for omega in &result.omegas {
        omegas.push(&JsValue::from(omega / (2.0 * std::f64::consts::PI)));
    }
    Reflect::set(&obj, &"omegas".into(), &omegas)?;

    // Convenience: also provide as "bands" for consistency
    Reflect::set(&obj, &"bands".into(), &omegas)?;

    // Metadata
    Reflect::set(
        &obj,
        &"iterations".into(),
        &JsValue::from(result.iterations as u32),
    )?;
    Reflect::set(&obj, &"is_gamma".into(), &JsValue::from(result.is_gamma))?;
    Reflect::set(
        &obj,
        &"num_bands".into(),
        &JsValue::from(result.omegas.len() as u32),
    )?;

    // Progress (0.0 to 1.0)
    let progress = (result.k_index + 1) as f64 / result.total_k_points as f64;
    Reflect::set(&obj, &"progress".into(), &JsValue::from(progress))?;

    // Parameters for context
    let params_obj = Object::new();
    Reflect::set(&params_obj, &"eps_bg".into(), &JsValue::from(params.eps_bg))?;
    Reflect::set(
        &params_obj,
        &"resolution".into(),
        &JsValue::from(params.resolution as u32),
    )?;
    Reflect::set(
        &params_obj,
        &"polarization".into(),
        &JsValue::from_str(&format!("{:?}", params.polarization)),
    )?;
    if let Some(ref lt) = params.lattice_type {
        Reflect::set(&params_obj, &"lattice_type".into(), &JsValue::from_str(lt))?;
    }

    // Atom parameters (mirrors the JobParams shape used in `result_to_js`).
    let atoms = Array::new();
    for atom in &params.atoms {
        let atom_obj = Object::new();
        Reflect::set(&atom_obj, &"index".into(), &JsValue::from(atom.index as u32))?;
        let pos = Array::new();
        pos.push(&JsValue::from(atom.pos[0]));
        pos.push(&JsValue::from(atom.pos[1]));
        Reflect::set(&atom_obj, &"pos".into(), &pos)?;
        Reflect::set(&atom_obj, &"radius".into(), &JsValue::from(atom.radius))?;
        Reflect::set(
            &atom_obj,
            &"eps_inside".into(),
            &JsValue::from(atom.eps_inside),
        )?;
        atoms.push(&atom_obj);
    }
    Reflect::set(&params_obj, &"atoms".into(), &atoms)?;

    // Sweep values for consistency with the final result shape.
    let sweep_values = Object::new();
    for (name, value) in &params.sweep_values {
        let py_value: JsValue = match value {
            SweepValue::Float(f) => JsValue::from(*f),
            SweepValue::Int(i) => JsValue::from(*i as f64),
            SweepValue::String(s) => JsValue::from_str(s),
        };
        Reflect::set(&sweep_values, &JsValue::from_str(name), &py_value)?;
    }
    Reflect::set(&params_obj, &"sweep_values".into(), &sweep_values)?;

    Reflect::set(&obj, &"params".into(), &params_obj)?;

    Ok(obj.into())
}

// ============================================================================
// WASM Bulk Driver
// ============================================================================

/// WebAssembly wrapper for the bulk driver with streaming support.
///
/// This provides a unified interface for running band structure calculations
/// in the browser, with support for both Maxwell and EA solvers.
#[wasm_bindgen]
pub struct WasmBulkDriver {
    config: BulkConfig,
    jobs: Vec<ExpandedJob>,
}

#[wasm_bindgen]
impl WasmBulkDriver {
    /// Create a new bulk driver from TOML configuration.
    ///
    /// @param config_str - Configuration as TOML string
    /// @throws Error if configuration is invalid
    #[wasm_bindgen(constructor)]
    pub fn new(config_str: &str) -> Result<WasmBulkDriver, JsValue> {
        let config = BulkConfig::from_str(config_str)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

        let jobs = expand_jobs(&config);

        Ok(Self { config, jobs })
    }

    // ========================================================================
    // Properties
    // ========================================================================

    /// Get the number of jobs that will be executed.
    #[wasm_bindgen(getter, js_name = "jobCount")]
    pub fn job_count(&self) -> usize {
        self.jobs.len()
    }

    /// Get the solver type ("maxwell" or "operator_data").
    #[wasm_bindgen(getter, js_name = "solverType")]
    pub fn solver_type(&self) -> String {
        match self.config.solver.solver_type {
            SolverType::Maxwell => "maxwell".to_string(),
            SolverType::OperatorData => "operator_data".to_string(),
        }
    }

    /// Get the storage precision ("f32" or "f64").
    #[wasm_bindgen(getter, js_name = "precision")]
    pub fn precision(&self) -> String {
        self.config.precision().to_string()
    }

    /// Check if this is an operator-data extraction solver.
    #[wasm_bindgen(getter, js_name = "isEA")]
    pub fn is_ea(&self) -> bool {
        matches!(self.config.solver.solver_type, SolverType::OperatorData)
    }

    /// Check if this is a Maxwell solver.
    #[wasm_bindgen(getter, js_name = "isMaxwell")]
    pub fn is_maxwell(&self) -> bool {
        matches!(self.config.solver.solver_type, SolverType::Maxwell)
    }

    /// Get the grid dimensions as [nx, ny].
    #[wasm_bindgen(getter, js_name = "gridSize")]
    pub fn grid_size(&self) -> Array {
        let arr = Array::new();
        arr.push(&JsValue::from(self.config.grid.nx as u32));
        arr.push(&JsValue::from(self.config.grid.ny() as u32));
        arr
    }

    /// Get the number of bands being computed.
    #[wasm_bindgen(getter, js_name = "numBands")]
    pub fn num_bands(&self) -> usize {
        self.config.eigensolver.n_bands
    }

    // ========================================================================
    // Streaming Execution
    // ========================================================================

    /// Run the computation with a callback for each result (STREAM mode).
    ///
    /// @param callback - Function(result) called for each completed job
    /// @returns Statistics object
    #[wasm_bindgen(js_name = "runWithCallback")]
    pub fn run_with_callback(&self, callback: Function) -> Result<JsValue, JsValue> {
        self.run_internal(None, callback)
    }

    /// Run with streaming output and server-side filtering (SELECTIVE mode).
    ///
    /// @param k_indices - Array of k-point indices to include (0-based), or null for all
    /// @param band_indices - Array of band indices to include (0-based), or null for all
    /// @param callback - Function(result) called for each filtered result
    /// @returns Statistics object
    #[wasm_bindgen(js_name = "runStreamingFiltered")]
    pub fn run_streaming_filtered(
        &self,
        k_indices: Option<Vec<usize>>,
        band_indices: Option<Vec<usize>>,
        callback: Function,
    ) -> Result<JsValue, JsValue> {
        let filter = SelectiveFilter::new(
            k_indices.unwrap_or_default(),
            band_indices.unwrap_or_default(),
        );
        self.run_internal(Some(filter), callback)
    }

    /// Run with K-POINT STREAMING: callback is called after EACH k-point is solved.
    ///
    /// This is the preferred mode for real-time visualization of band structure
    /// computation. The callback receives incremental results as each k-point
    /// completes, enabling smooth progressive rendering of the band diagram.
    ///
    /// @param callback - Function(kPointResult) called after each k-point solve
    /// @returns Statistics object with completion info
    ///
    /// The callback receives an object with:
    /// - `stream_type`: "k_point" (to distinguish from job-level streaming)
    /// - `k_index`: Index of this k-point (0-based)
    /// - `total_k_points`: Total number of k-points
    /// - `k_point`: [kx, ky] in fractional coordinates
    /// - `distance`: Cumulative path distance to this k-point
    /// - `omegas`: Array of frequencies for all bands at this k-point
    /// - `bands`: Same as omegas (alias for convenience)
    /// - `iterations`: Number of LOBPCG iterations for this k-point
    /// - `is_gamma`: Whether this is a Γ-point
    /// - `progress`: Completion fraction (0.0 to 1.0)
    /// - `params`: Job parameters (eps_bg, resolution, polarization, etc.)
    #[wasm_bindgen(js_name = "runWithKPointStreaming")]
    pub fn run_with_k_point_streaming(&self, callback: Function) -> Result<JsValue, JsValue> {
        let start = Date::now();
        let mut stats = WasmDriverStats {
            total_jobs: self.jobs.len(),
            ..Default::default()
        };

        for job in &self.jobs {
            match self.execute_job_with_k_streaming(job, &callback) {
                Ok(_result) => {
                    stats.completed += 1;
                }
                Err(_) => {
                    stats.failed += 1;
                }
            }
        }

        stats.total_time_ms = (Date::now() - start) as u128;
        stats_to_js(&stats)
    }

    /// Run and return all results as an array (COLLECT mode).
    ///
    /// @returns Object with `results` (array) and `stats`
    #[wasm_bindgen(js_name = "runCollect")]
    pub fn run_collect(&self) -> Result<JsValue, JsValue> {
        let start = Date::now();
        let mut results = Vec::new();
        let mut stats = WasmDriverStats {
            total_jobs: self.jobs.len(),
            ..Default::default()
        };

        for job in &self.jobs {
            match self.execute_job(job) {
                Ok(result) => {
                    results.push(result);
                    stats.completed += 1;
                }
                Err(_) => {
                    stats.failed += 1;
                }
            }
        }

        stats.total_time_ms = (Date::now() - start) as u128;

        // Convert results to JS array
        let js_results = Array::new();
        for result in &results {
            if let Ok(js_obj) = result_to_js(result) {
                js_results.push(&js_obj);
            }
        }

        // Build return object
        let output = Object::new();
        Reflect::set(&output, &"results".into(), &js_results)?;
        Reflect::set(&output, &"stats".into(), &stats_to_js(&stats)?)?;

        Ok(output.into())
    }

    /// Run and collect with optional filtering.
    #[wasm_bindgen(js_name = "runCollectFiltered")]
    pub fn run_collect_filtered(
        &self,
        k_indices: Option<Vec<usize>>,
        band_indices: Option<Vec<usize>>,
    ) -> Result<JsValue, JsValue> {
        let filter = SelectiveFilter::new(
            k_indices.unwrap_or_default(),
            band_indices.unwrap_or_default(),
        );

        let start = Date::now();
        let mut results = Vec::new();
        let mut stats = WasmDriverStats {
            total_jobs: self.jobs.len(),
            ..Default::default()
        };

        for job in &self.jobs {
            match self.execute_job(job) {
                Ok(result) => {
                    let filtered = filter.apply(&result);
                    results.push(filtered);
                    stats.completed += 1;
                }
                Err(_) => {
                    stats.failed += 1;
                }
            }
        }

        stats.total_time_ms = (Date::now() - start) as u128;

        // Convert results to JS array
        let js_results = Array::new();
        for result in &results {
            if let Ok(js_obj) = result_to_js(result) {
                js_results.push(&js_obj);
            }
        }

        let output = Object::new();
        Reflect::set(&output, &"results".into(), &js_results)?;
        Reflect::set(&output, &"stats".into(), &stats_to_js(&stats)?)?;

        Ok(output.into())
    }

    // ========================================================================
    // Dry Run / Inspection
    // ========================================================================

    /// Dry run: get job count and parameter info without executing.
    #[wasm_bindgen(js_name = "dryRun")]
    pub fn dry_run(&self) -> Result<JsValue, JsValue> {
        let result = Object::new();
        Reflect::set(
            &result,
            &"total_jobs".into(),
            &JsValue::from(self.jobs.len() as u32),
        )?;
        Reflect::set(
            &result,
            &"thread_mode".into(),
            &JsValue::from_str("single-threaded (WASM)"),
        )?;
        Reflect::set(
            &result,
            &"solver_type".into(),
            &JsValue::from_str(&self.solver_type()),
        )?;

        Ok(result.into())
    }

    /// Get the first N expanded job configurations.
    #[wasm_bindgen(js_name = "getJobConfigs")]
    pub fn get_job_configs(&self, n: usize) -> Result<Array, JsValue> {
        let result = Array::new();

        for job in self.jobs.iter().take(n) {
            let obj = Object::new();
            Reflect::set(&obj, &"index".into(), &JsValue::from(job.index as u32))?;

            let params = Object::new();
            Reflect::set(&params, &"eps_bg".into(), &JsValue::from(job.params.eps_bg))?;
            Reflect::set(
                &params,
                &"resolution".into(),
                &JsValue::from(job.params.resolution as u32),
            )?;
            Reflect::set(
                &params,
                &"polarization".into(),
                &JsValue::from_str(&format!("{:?}", job.params.polarization)),
            )?;
            if let Some(ref lt) = job.params.lattice_type {
                Reflect::set(&params, &"lattice_type".into(), &JsValue::from_str(lt))?;
            }

            let atoms = Array::new();
            for atom in &job.params.atoms {
                let atom_obj = Object::new();
                Reflect::set(
                    &atom_obj,
                    &"index".into(),
                    &JsValue::from(atom.index as u32),
                )?;
                let pos = Array::new();
                pos.push(&JsValue::from(atom.pos[0]));
                pos.push(&JsValue::from(atom.pos[1]));
                Reflect::set(&atom_obj, &"pos".into(), &pos)?;
                Reflect::set(&atom_obj, &"radius".into(), &JsValue::from(atom.radius))?;
                Reflect::set(
                    &atom_obj,
                    &"eps_inside".into(),
                    &JsValue::from(atom.eps_inside),
                )?;
                atoms.push(&atom_obj);
            }
            Reflect::set(&params, &"atoms".into(), &atoms)?;
            Reflect::set(&obj, &"params".into(), &params)?;
            result.push(&obj);
        }

        Ok(result)
    }

    // ========================================================================
    // Internal Helpers
    // ========================================================================

    /// Internal method for running with optional filtering and callback.
    fn run_internal(
        &self,
        filter: Option<SelectiveFilter>,
        callback: Function,
    ) -> Result<JsValue, JsValue> {
        let start = Date::now();
        let mut stats = WasmDriverStats {
            total_jobs: self.jobs.len(),
            ..Default::default()
        };

        for job in &self.jobs {
            match self.execute_job(job) {
                Ok(result) => {
                    let final_result = if let Some(ref f) = filter {
                        f.apply(&result)
                    } else {
                        result
                    };

                    if let Ok(js_obj) = result_to_js(&final_result) {
                        let _ = callback.call1(&JsValue::NULL, &js_obj);
                    }
                    stats.completed += 1;
                }
                Err(_) => {
                    stats.failed += 1;
                }
            }
        }

        stats.total_time_ms = (Date::now() - start) as u128;
        stats_to_js(&stats)
    }

    /// Execute a single expanded job on the precision-correct backend.
    fn execute_job(&self, job: &ExpandedJob) -> Result<CompactBandResult, String> {
        match &job.job_type {
            ExpandedJobType::Maxwell(maxwell_job) => match self.config.precision() {
                Precision::F32 => run_maxwell_job(
                    CpuBackend::<f32>::new(),
                    maxwell_job,
                    &job.params,
                    job.index,
                ),
                Precision::F64 => run_maxwell_job(
                    CpuBackend::<f64>::new(),
                    maxwell_job,
                    &job.params,
                    job.index,
                ),
            },
            _ => Err("Job type not supported on wasm backend".to_string()),
        }
    }

    /// Execute a single expanded job with k-point streaming.
    fn execute_job_with_k_streaming(
        &self,
        job: &ExpandedJob,
        callback: &Function,
    ) -> Result<CompactBandResult, String> {
        match &job.job_type {
            ExpandedJobType::Maxwell(maxwell_job) => match self.config.precision() {
                Precision::F32 => run_maxwell_job_with_k_streaming(
                    CpuBackend::<f32>::new(),
                    maxwell_job,
                    &job.params,
                    job.index,
                    callback,
                ),
                Precision::F64 => run_maxwell_job_with_k_streaming(
                    CpuBackend::<f64>::new(),
                    maxwell_job,
                    &job.params,
                    job.index,
                    callback,
                ),
            },
            _ => Err("Job type not supported on wasm backend".to_string()),
        }
    }
}

// ============================================================================
// Selective Filter WASM Wrapper
// ============================================================================

/// WASM wrapper for selective filtering configuration.
#[wasm_bindgen]
pub struct WasmSelectiveFilter {
    k_indices: Vec<usize>,
    band_indices: Vec<usize>,
}

#[wasm_bindgen]
impl WasmSelectiveFilter {
    /// Create a new selective filter (pass-through by default).
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            k_indices: Vec::new(),
            band_indices: Vec::new(),
        }
    }

    /// Set k-point indices to include.
    #[wasm_bindgen(js_name = "setKIndices")]
    pub fn set_k_indices(&mut self, indices: Vec<usize>) {
        self.k_indices = indices;
    }

    /// Set band indices to include.
    #[wasm_bindgen(js_name = "setBandIndices")]
    pub fn set_band_indices(&mut self, indices: Vec<usize>) {
        self.band_indices = indices;
    }

    /// Clear all filter settings.
    pub fn clear(&mut self) {
        self.k_indices.clear();
        self.band_indices.clear();
    }

    /// Check if the filter is active.
    #[wasm_bindgen(getter, js_name = "isActive")]
    pub fn is_active(&self) -> bool {
        !self.k_indices.is_empty() || !self.band_indices.is_empty()
    }

    /// Get the number of k-points in the filter.
    #[wasm_bindgen(getter, js_name = "kCount")]
    pub fn k_count(&self) -> Option<usize> {
        if self.k_indices.is_empty() {
            None
        } else {
            Some(self.k_indices.len())
        }
    }

    /// Get the number of bands in the filter.
    #[wasm_bindgen(getter, js_name = "bandCount")]
    pub fn band_count(&self) -> Option<usize> {
        if self.band_indices.is_empty() {
            None
        } else {
            Some(self.band_indices.len())
        }
    }

    /// Get k-indices as array.
    #[wasm_bindgen(getter, js_name = "kIndices")]
    pub fn k_indices(&self) -> Vec<usize> {
        self.k_indices.clone()
    }

    /// Get band indices as array.
    #[wasm_bindgen(getter, js_name = "bandIndices")]
    pub fn band_indices(&self) -> Vec<usize> {
        self.band_indices.clone()
    }
}

impl Default for WasmSelectiveFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Initialize the WASM module with proper panic handling.
/// Call this once at the start of your application.
///
/// This sets up the panic hook so that Rust panics are printed
/// to the browser console with full stack traces instead of
/// just showing "RuntimeError: unreachable".
#[wasm_bindgen(js_name = "initPanicHook")]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

/// Get the library version.
#[wasm_bindgen(js_name = "getVersion")]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Check if streaming mode is supported.
#[wasm_bindgen(js_name = "isStreamingSupported")]
pub fn is_streaming_supported() -> bool {
    true
}

/// Check if selective filtering is supported.
#[wasm_bindgen(js_name = "isSelectiveSupported")]
pub fn is_selective_supported() -> bool {
    true
}

/// Get supported solver types.
#[wasm_bindgen(js_name = "getSupportedSolvers")]
pub fn get_supported_solvers() -> Array {
    let arr = Array::new();
    arr.push(&JsValue::from_str("maxwell"));
    arr.push(&JsValue::from_str("operator_data"));
    arr
}

// ============================================================================
// Configuration Validation
// ============================================================================

/// Number of k-points the config's path resolves to (0 if not resolvable).
fn count_k_points(config: &Config) -> u32 {
    match &config.path {
        Some(p) if !p.points.is_empty() => p.points.len() as u32,
        Some(p) => p
            .preset
            .and_then(|preset| preset.resolve(config.geometry.lattice.kind))
            .map(|bp| {
                blaze2d_core::brillouin::generate_path(&bp, p.points_per_segment()).len() as u32
            })
            .unwrap_or(0),
        None => 0,
    }
}

/// Validate a schema v2 TOML configuration without running anything.
///
/// This runs THE parser: the exact same `parse_and_validate` the native
/// drivers use, so web editors get zero-drift validation. Returns:
///
/// ```text
/// {
///   ok: boolean,
///   errors: [{ path: string, message: string, span: [start, end] | null }],
///   summary?: {                     // present when ok
///     jobs, nx, ny, n_bands, k_points,
///     precision, solver_type, polarization,
///   }
/// }
/// ```
#[wasm_bindgen(js_name = "validateConfig")]
pub fn validate_config(config_str: &str) -> Result<JsValue, JsValue> {
    let out = Object::new();

    match parse_and_validate(config_str) {
        Ok(config) => {
            Reflect::set(&out, &"ok".into(), &JsValue::TRUE)?;
            Reflect::set(&out, &"errors".into(), &Array::new())?;

            let summary = Object::new();
            Reflect::set(
                &summary,
                &"jobs".into(),
                &JsValue::from(config.total_jobs() as u32),
            )?;
            Reflect::set(
                &summary,
                &"nx".into(),
                &JsValue::from(config.grid.nx as u32),
            )?;
            Reflect::set(
                &summary,
                &"ny".into(),
                &JsValue::from(config.grid.ny() as u32),
            )?;
            Reflect::set(
                &summary,
                &"n_bands".into(),
                &JsValue::from(config.eigensolver.n_bands as u32),
            )?;
            Reflect::set(
                &summary,
                &"k_points".into(),
                &JsValue::from(count_k_points(&config)),
            )?;
            Reflect::set(
                &summary,
                &"precision".into(),
                &JsValue::from_str(&config.precision().to_string()),
            )?;
            let solver_type = match config.solver_type() {
                SolverType::Maxwell => "maxwell",
                SolverType::OperatorData => "operator_data",
            };
            Reflect::set(&summary, &"solver_type".into(), &JsValue::from_str(solver_type))?;
            Reflect::set(
                &summary,
                &"polarization".into(),
                &JsValue::from_str(&format!("{:?}", config.solver.polarization)),
            )?;
            Reflect::set(&out, &"summary".into(), &summary)?;
        }
        Err(diags) => {
            Reflect::set(&out, &"ok".into(), &JsValue::FALSE)?;
            let errors = Array::new();
            for d in &diags {
                let err = Object::new();
                Reflect::set(&err, &"path".into(), &JsValue::from_str(&d.path))?;
                Reflect::set(&err, &"message".into(), &JsValue::from_str(&d.message))?;
                match d.span {
                    Some((start, end)) => {
                        let span = Array::new();
                        span.push(&JsValue::from(start as u32));
                        span.push(&JsValue::from(end as u32));
                        Reflect::set(&err, &"span".into(), &span)?;
                    }
                    None => {
                        Reflect::set(&err, &"span".into(), &JsValue::NULL)?;
                    }
                }
                errors.push(&err);
            }
            Reflect::set(&out, &"errors".into(), &errors)?;
        }
    }

    Ok(out.into())
}
