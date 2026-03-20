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
//! import init, { WasmBulkDriver } from 'mpb2d-wasm';
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

use js_sys::{Array, Function, Object, Reflect};
use wasm_bindgen::prelude::*;
use std::time::Instant;

use mpb2d_bulk_driver_core::{
    config::{BulkConfig, SolverType},
    expansion::{expand_jobs, ExpandedJob, ExpandedJobType},
    filter::SelectiveFilter,
    result::{CompactBandResult, CompactResultType, EAResult, MaxwellResult},
};

use mpb2d_core::drivers::bandstructure::{self, Verbosity};

use mpb2d_backend_cpu::CpuBackend;

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
    Reflect::set(&params, &"eps_bg".into(), &JsValue::from(result.params.eps_bg))?;
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
    Reflect::set(&obj, &"params".into(), &params)?;

    // Result type discriminator and type-specific data
    match &result.result_type {
        CompactResultType::Maxwell(maxwell) => {
            Reflect::set(&obj, &"result_type".into(), &JsValue::from_str("maxwell"))?;
            set_maxwell_fields(&obj, maxwell)?;
        }
        CompactResultType::EA(ea) => {
            Reflect::set(&obj, &"result_type".into(), &JsValue::from_str("ea"))?;
            set_ea_fields(&obj, ea)?;
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

/// Set EA-specific fields on a JS object.
fn set_ea_fields(obj: &Object, ea: &EAResult) -> Result<(), JsValue> {
    // Eigenvalues
    let eigenvalues = Array::new();
    for ev in &ea.eigenvalues {
        eigenvalues.push(&JsValue::from(*ev));
    }
    Reflect::set(obj, &"eigenvalues".into(), &eigenvalues)?;

    // Eigenvectors as 3D array: [band_index][grid_index][re, im]
    let eigenvectors = Array::new();
    for evec in &ea.eigenvectors {
        let vec_arr = Array::new();
        for &[re, im] in evec {
            let complex = Array::new();
            complex.push(&JsValue::from(re));
            complex.push(&JsValue::from(im));
            vec_arr.push(&complex);
        }
        eigenvectors.push(&vec_arr);
    }
    Reflect::set(obj, &"eigenvectors".into(), &eigenvectors)?;

    // Grid dimensions for reconstructing 2D structure
    let grid_dims = Array::new();
    grid_dims.push(&JsValue::from(ea.grid_dims[0] as u32));
    grid_dims.push(&JsValue::from(ea.grid_dims[1] as u32));
    Reflect::set(obj, &"grid_dims".into(), &grid_dims)?;

    // Solver info
    Reflect::set(
        obj,
        &"n_iterations".into(),
        &JsValue::from(ea.n_iterations as u32),
    )?;
    Reflect::set(obj, &"converged".into(), &JsValue::from(ea.converged))?;

    // Convenience properties
    Reflect::set(
        obj,
        &"num_eigenvalues".into(),
        &JsValue::from(ea.eigenvalues.len() as u32),
    )?;
    Reflect::set(
        obj,
        &"num_bands".into(),
        &JsValue::from(ea.eigenvalues.len() as u32),
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
fn run_maxwell_job(
    job: &mpb2d_core::bandstructure::BandStructureJob,
    params: &mpb2d_bulk_driver_core::expansion::JobParams,
    job_index: usize,
) -> Result<CompactBandResult, String> {
    let backend = CpuBackend::new();
    
    let band_result = bandstructure::run(backend, job, Verbosity::Quiet);
    
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

/// Run a single EA job and convert to CompactBandResult.
fn run_ea_job(
    _job: &mpb2d_bulk_driver_core::expansion::EAJobSpec,
    _params: &mpb2d_bulk_driver_core::expansion::JobParams,
    _job_index: usize,
) -> Result<CompactBandResult, String> {
    // EA solver implementation - placeholder for now
    Err("EA solver not yet implemented in WASM backend".to_string())
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

    /// Get the solver type ("maxwell" or "ea").
    #[wasm_bindgen(getter, js_name = "solverType")]
    pub fn solver_type(&self) -> String {
        match self.config.solver.solver_type {
            SolverType::Maxwell => "maxwell".to_string(),
            SolverType::EA => "ea".to_string(),
        }
    }

    /// Check if this is an EA solver.
    #[wasm_bindgen(getter, js_name = "isEA")]
    pub fn is_ea(&self) -> bool {
        matches!(self.config.solver.solver_type, SolverType::EA)
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
        arr.push(&JsValue::from(self.config.grid.ny as u32));
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

    /// Run and return all results as an array (COLLECT mode).
    ///
    /// @returns Object with `results` (array) and `stats`
    #[wasm_bindgen(js_name = "runCollect")]
    pub fn run_collect(&self) -> Result<JsValue, JsValue> {
        let start = Instant::now();
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
        
        stats.total_time_ms = start.elapsed().as_millis();
        
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
        
        let start = Instant::now();
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
        
        stats.total_time_ms = start.elapsed().as_millis();
        
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
        let start = Instant::now();
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
        
        stats.total_time_ms = start.elapsed().as_millis();
        stats_to_js(&stats)
    }

    /// Execute a single expanded job.
    fn execute_job(&self, job: &ExpandedJob) -> Result<CompactBandResult, String> {
        match &job.job_type {
            ExpandedJobType::Maxwell(maxwell_job) => {
                run_maxwell_job(maxwell_job, &job.params, job.index)
            }
            ExpandedJobType::EA(ea_job) => {
                run_ea_job(ea_job, &job.params, job.index)
            }
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
    arr.push(&JsValue::from_str("ea"));
    arr
}
