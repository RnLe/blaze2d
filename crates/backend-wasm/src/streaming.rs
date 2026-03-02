//! WASM streaming interface for React/JS consumers.
//!
//! This module provides WebAssembly bindings for streaming band structure
//! results to JavaScript applications, enabling real-time visualization
//! in web-based UIs.
//!
//! # Example Usage (JavaScript)
//!
//! ```javascript
//! import init, { WasmBulkDriver } from 'mpb2d-wasm';
//!
//! async function runSimulation() {
//!     await init();
//!     
//!     const config = { /* TOML config as JSON */ };
//!     const driver = new WasmBulkDriver(JSON.stringify(config));
//!     
//!     // Callback-based streaming
//!     driver.runWithCallback((result) => {
//!         console.log(`Job ${result.job_index} complete`);
//!         updateChart(result.bands, result.distances);
//!     });
//! }
//! ```
//!
//! # React Integration
//!
//! ```javascript
//! function BandPlot() {
//!     const [results, setResults] = useState([]);
//!     
//!     useEffect(() => {
//!         const driver = new WasmBulkDriver(config);
//!         driver.runWithCallback((result) => {
//!             setResults(prev => [...prev, result]);
//!         });
//!     }, []);
//!     
//!     return <Plot data={results} />;
//! }
//! ```

use js_sys::{Array, Function, Object, Reflect};
use wasm_bindgen::prelude::*;

use mpb2d_bulk_driver::{
    BulkConfig, BulkDriver, CompactBandResult, OutputChannel,
};

// ============================================================================
// JS Result Conversion
// ============================================================================

/// Convert a CompactBandResult to a JavaScript object.
fn result_to_js(result: &CompactBandResult) -> Result<JsValue, JsValue> {
    let obj = Object::new();

    // Job index
    Reflect::set(&obj, &"job_index".into(), &JsValue::from(result.job_index))?;

    // Parameters
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

    // K-path as array of [kx, ky] arrays
    let k_path = Array::new();
    for k in &result.k_path {
        let point = Array::new();
        point.push(&JsValue::from(k[0]));
        point.push(&JsValue::from(k[1]));
        k_path.push(&point);
    }
    Reflect::set(&obj, &"k_path".into(), &k_path)?;

    // Distances
    let distances = Array::new();
    for d in &result.distances {
        distances.push(&JsValue::from(*d));
    }
    Reflect::set(&obj, &"distances".into(), &distances)?;

    // Bands as 2D array [k_index][band_index]
    let bands = Array::new();
    for k_bands in &result.bands {
        let band_arr = Array::new();
        for omega in k_bands {
            band_arr.push(&JsValue::from(*omega));
        }
        bands.push(&band_arr);
    }
    Reflect::set(&obj, &"bands".into(), &bands)?;

    // Convenience properties
    Reflect::set(
        &obj,
        &"num_k_points".into(),
        &JsValue::from(result.num_k_points() as u32),
    )?;
    Reflect::set(
        &obj,
        &"num_bands".into(),
        &JsValue::from(result.num_bands() as u32),
    )?;

    Ok(obj.into())
}

// ============================================================================
// Logging Helper
// ============================================================================

#[cfg(target_arch = "wasm32")]
#[allow(dead_code)]
fn web_log(msg: &str) {
    web_sys::console::log_1(&JsValue::from_str(msg));
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn web_log(msg: &str) {
    log::info!("{}", msg);
}

// ============================================================================
// WASM Bulk Driver
// ============================================================================

/// WebAssembly wrapper for the bulk driver with callback-based streaming.
///
/// In WASM, we use a simpler single-threaded approach where results are
/// passed directly to a JavaScript callback as they complete.
///
/// @example
/// ```javascript
/// const driver = new WasmBulkDriver(configJson);
/// console.log(`Running ${driver.jobCount} jobs`);
///
/// driver.runWithCallback((result) => {
///     console.log(`Job ${result.job_index}: ${result.num_bands} bands`);
/// });
/// ```
#[wasm_bindgen]
pub struct WasmBulkDriver {
    config: BulkConfig,
}

#[wasm_bindgen]
impl WasmBulkDriver {
    /// Create a new bulk driver from JSON configuration.
    ///
    /// The configuration should be the TOML format converted to JSON,
    /// or a native JSON configuration object.
    ///
    /// @param config_json - Configuration as JSON string
    /// @throws Error if configuration is invalid
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> Result<WasmBulkDriver, JsValue> {
        // Try to parse as TOML first, then as JSON
        let config = if config_json.contains("[bulk]") {
            BulkConfig::from_str(config_json)
                .map_err(|e| JsValue::from_str(&format!("Invalid TOML config: {}", e)))?
        } else {
            // Parse JSON, convert to TOML-compatible structure
            serde_json::from_str::<BulkConfig>(config_json)
                .map_err(|e| JsValue::from_str(&format!("Invalid JSON config: {}", e)))?
        };

        Ok(Self { config })
    }

    /// Get the number of jobs that will be executed.
    #[wasm_bindgen(getter, js_name = "jobCount")]
    pub fn job_count(&self) -> usize {
        self.config.total_jobs()
    }

    /// Run the computation with a callback for each result.
    ///
    /// In WASM, this runs synchronously on the main thread, calling the
    /// callback after each job completes.
    ///
    /// The callback receives a result object with properties:
    /// - job_index: number
    /// - params: { eps_bg, resolution, polarization, atoms }
    /// - k_path: [[kx, ky], ...]
    /// - distances: number[]
    /// - bands: number[][] (bands[k_index][band_index])
    /// - num_k_points: number
    /// - num_bands: number
    ///
    /// @param callback - Function called for each completed job
    /// @returns Statistics object with total_jobs, completed, failed, total_time_ms
    #[wasm_bindgen(js_name = "runWithCallback")]
    pub fn run_with_callback(&self, callback: Function) -> Result<JsValue, JsValue> {
        let driver = BulkDriver::new(self.config.clone(), Some(1)); // Single thread for WASM

        // For WASM, we use the collecting subscriber and call the callback manually
        // since WASM is single-threaded and we can't use Send+Sync callbacks
        let collector = mpb2d_bulk_driver::CollectingSubscriber::new();
        let collector_arc = std::sync::Arc::new(collector);

        let stream = std::sync::Arc::new(mpb2d_bulk_driver::StreamChannel::new(
            mpb2d_bulk_driver::StreamConfig::default(),
        ));
        stream.subscribe(collector_arc.clone());

        // Run the computation
        let channel = OutputChannel::Stream(stream);
        let stats = driver
            .run_with_channel(channel)
            .map_err(|e| JsValue::from_str(&format!("Driver error: {}", e)))?;

        // Now call the callback for each collected result
        for result in collector_arc.results().iter() {
            if let Ok(js_obj) = result_to_js(result) {
                let _ = callback.call1(&JsValue::NULL, &js_obj);
            }
        }

        // Return statistics
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
            &JsValue::from(stats.total_time.as_millis() as u32),
        )?;

        Ok(result.into())
    }

    /// Run and return all results as an array.
    ///
    /// This collects all results and returns them at once, which may be
    /// more efficient for small job counts.
    ///
    /// @returns Array of result objects
    #[wasm_bindgen(js_name = "runCollect")]
    pub fn run_collect(&self) -> Result<Array, JsValue> {
        let driver = BulkDriver::new(self.config.clone(), Some(1));

        let collector = mpb2d_bulk_driver::CollectingSubscriber::new();
        let collector_arc = std::sync::Arc::new(collector);

        let stream = std::sync::Arc::new(mpb2d_bulk_driver::StreamChannel::new(
            mpb2d_bulk_driver::StreamConfig::default(),
        ));
        stream.subscribe(collector_arc.clone());

        let channel = OutputChannel::Stream(stream);
        driver
            .run_with_channel(channel)
            .map_err(|e| JsValue::from_str(&format!("Driver error: {}", e)))?;

        // Convert all results to JS array
        let results = Array::new();
        for result in collector_arc.results().iter() {
            if let Ok(js_obj) = result_to_js(result) {
                results.push(&js_obj);
            }
        }

        Ok(results)
    }

    /// Run dry-run to get job count and parameter info.
    ///
    /// @returns Object with total_jobs and parameter_counts
    #[wasm_bindgen(js_name = "dryRun")]
    pub fn dry_run(&self) -> Result<JsValue, JsValue> {
        let driver = BulkDriver::new(self.config.clone(), None);
        let stats = driver.dry_run();

        let result = Object::new();
        Reflect::set(
            &result,
            &"total_jobs".into(),
            &JsValue::from(stats.total_jobs as u32),
        )?;
        Reflect::set(
            &result,
            &"thread_mode".into(),
            &JsValue::from_str(&stats.thread_mode),
        )?;

        let params = Object::new();
        for (name, count) in &stats.param_counts {
            Reflect::set(
                &params,
                &JsValue::from_str(name),
                &JsValue::from(*count as u32),
            )?;
        }
        Reflect::set(&result, &"parameter_counts".into(), &params)?;

        Ok(result.into())
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Get version information.
#[wasm_bindgen(js_name = "getVersion")]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Check if streaming is supported.
#[wasm_bindgen(js_name = "isStreamingSupported")]
pub fn is_streaming_supported() -> bool {
    true
}
