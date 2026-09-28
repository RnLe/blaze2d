use blaze2d_interface as api;
use blaze2d_core::dielectric::Dielectric2D;
use js_sys::{Object, Reflect, Float64Array, Function};
use serde::Serialize;
use wasm_bindgen::prelude::*;

fn js(value: &impl Serialize) -> Result<JsValue,JsValue> {
    value.serialize(&serde_wasm_bindgen::Serializer::json_compatible()).map_err(|e|JsValue::from_str(&e.to_string()))
}
fn diagnostic(error: api::Diagnostic) -> JsValue { js(&error).unwrap_or_else(|_|JsValue::from_str(&error.message)) }
fn set(object: &JsValue,key: &str,value: &JsValue) -> Result<(),JsValue> {
    Reflect::set(object,&JsValue::from_str(key),value).map(|_|())
}

fn arrays(arrays: api::Arrays) -> Result<JsValue,JsValue> {
    let object = Object::new();
    for (name,array) in arrays {
        let descriptor = js(&serde_json::json!({"dtype":array.dtype,"shape":array.shape,
            "dimensions":array.dimensions,"order":array.order}))?;
        set(&descriptor,"data",&Float64Array::from(array.data.as_slice()))?;
        set(&object,&name,&descriptor)?;
    }
    Ok(object.into())
}

fn result_to_js(result: api::ResultRecord) -> Result<JsValue,JsValue> {
    let object = js(&serde_json::json!({"schema":result.schema,"task":result.task,
        "job_index":result.job_index,"metadata":result.metadata}))?;
    set(&object,"arrays",&arrays(result.arrays)?)?;
    if !result.samples.is_empty() {
        let samples = js_sys::Array::new();
        for sample in result.samples {
            let object = js(&serde_json::json!({"sample_index":sample.sample_index,"metadata":sample.metadata}))?;
            set(&object,"arrays",&arrays(sample.arrays)?)?;
            samples.push(&object);
        }
        set(&object,"samples",&samples)?;
    }
    Ok(object)
}

#[wasm_bindgen(js_name=buildInfo, unchecked_return_type="BuildInfo")]
pub fn build_info() -> Result<JsValue,JsValue> { js(&api::build_info()) }

/// Explanatory profiles are separate builds; ordinary browser modules expose
/// no instrumentation or reset API. Call outside active calculations.
#[cfg(feature="profiling")]
#[wasm_bindgen(js_name=resetProfile)]
pub fn reset_profile() { blaze2d_core::profiler::reset(); }

#[cfg(feature="profiling")]
#[wasm_bindgen(js_name=profileSnapshot)]
pub fn profile_snapshot() -> Result<JsValue,JsValue> {
    let timings:serde_json::Value = serde_json::from_str(&blaze2d_core::profiler::get_profile_json())
        .map_err(|e|JsValue::from_str(&e.to_string()))?;
    js(&serde_json::json!({"counters":blaze2d_core::profiler::get_all_counters(),
        "stage_counters":blaze2d_core::profiler::get_all_stage_counters(),"timings":timings}))
}
#[wasm_bindgen(unchecked_return_type="Capabilities")]
pub fn capabilities() -> Result<JsValue,JsValue> { js(&api::capabilities(api::Platform::Browser)) }
#[wasm_bindgen(js_name=initPanicHook)]
pub fn init_panic_hook() { console_error_panic_hook::set_once(); }
#[wasm_bindgen(js_name=defaultToml)]
pub fn default_toml() -> String { include_str!("../../../examples/calculations/square-rods.toml").into() }
#[wasm_bindgen(js_name=defaultConfig, unchecked_return_type="Config")]
pub fn default_config() -> Result<JsValue,JsValue> { js(&api::Config::default()) }
#[wasm_bindgen(js_name=validateToml, unchecked_return_type="ValidationReport")]
pub fn validate_toml(source: &str) -> Result<JsValue,JsValue> { js(&api::validate_toml(source,api::Platform::Browser)) }
#[wasm_bindgen(js_name=normalizeToml)]
pub fn normalize_toml(source: &str) -> Result<String,JsValue> {
    api::Plan::new(api::Config::from_toml(source).map_err(diagnostic)?,api::Platform::Browser)
        .map_err(diagnostic)?.config.to_toml().map_err(diagnostic)
}
#[wasm_bindgen(js_name=applyConfig)]
pub fn apply_config(source: &str,config_json: &str) -> Result<String,JsValue> {
    let value = serde_json::from_str(config_json).map_err(|e|JsValue::from_str(&e.to_string()))?;
    let config = api::Config::from_value(value).map_err(diagnostic)?;
    api::edit::apply_config(source,config,api::Platform::Browser).map_err(diagnostic)
}

#[wasm_bindgen(js_name=selectTask)]
pub fn select_task(source: &str,task: &str) -> Result<String,JsValue> {
    let mut config = api::Config::from_toml(source).map_err(diagnostic)?;
    config.task = serde_json::from_value(serde_json::json!(task)).map_err(|e| JsValue::from_str(&e.to_string()))?;
    match config.task {
        api::Task::Bands => { config.operators=None; config.bands=Some(api::Bands::default()); }
        api::Task::Operators => {
            config.bands=None;
            config.operators=Some(serde_json::from_value(serde_json::json!({
                "k_point":{"value":[0.0,0.0],"basis":"reciprocal_fractional"}
            })).map_err(|e| JsValue::from_str(&e.to_string()))?);
        }
    }
    api::edit::apply_config(source,config,api::Platform::Browser).map_err(diagnostic)
}

#[wasm_bindgen]
pub struct Calculation { plan: api::Plan, context: api::ExecutionContext }

#[wasm_bindgen]
impl Calculation {
    #[wasm_bindgen(constructor)]
    pub fn new(source: &str) -> Result<Calculation,JsValue> {
        let plan = api::Plan::new(api::Config::from_toml(source).map_err(diagnostic)?,api::Platform::Browser).map_err(diagnostic)?;
        let context = api::ExecutionContext::new(plan.config.bands.as_ref().map_or(0, |b|b.warm_start.as_ref().map_or(b.symmetry.max_continuation_bytes,|w|w.max_cache_bytes)));
        Ok(Self { plan, context })
    }
    #[wasm_bindgen(getter, unchecked_return_type="PlanSummary")]
    pub fn summary(&self) -> Result<JsValue,JsValue> { js(&self.plan.summary) }

    #[wasm_bindgen(js_name=runJob, unchecked_return_type="Result")]
    pub fn run_job(&mut self,index: usize,progress: &Function) -> Result<JsValue,JsValue> {
        let job = self.plan.job(index).map_err(diagnostic)?;
        let mut callback_error = None;
        let result = api::execute_in_context(crate::WasmBackend::new(),"wasm",&job,&mut self.context,|event| {
            if callback_error.is_none() {
                match js(&event).and_then(|v|progress.call1(&JsValue::NULL,&v)) {
                    Ok(_) => {}, Err(e) => callback_error=Some(e),
                }
            }
        });
        if let Some(error) = callback_error { return Err(error); }
        match result {
            Ok(result) => result_to_js(result),
            Err(failure) => {
                self.context.clear();
                let object = js(&serde_json::json!({"job_index":failure.job_index,"diagnostic":failure.diagnostic}))?;
                if let Some(result) = failure.partial_result { set(&object,"partial_result",&result_to_js(*result)?)?; }
                Err(object)
            }
        }
    }

    #[wasm_bindgen(js_name=geometryPreview, unchecked_return_type="Preview")]
    pub fn geometry_preview(&self) -> Result<JsValue,JsValue> {
        let job = self.plan.job(0).map_err(diagnostic)?;
        let geometry = job.geometry();
        let dielectric = Dielectric2D::from_geometry(&geometry,job.grid(),&job.dielectric());
        let value = js(&serde_json::json!({"resolution":job.resolved.resolution,
            "lattice_vectors":job.resolved.lattice_vectors,"sample_origin":[0.0,0.0]}))?;
        set(&value,"epsilon",&Float64Array::from(dielectric.eps()))?;
        Ok(value)
    }
}

#[wasm_bindgen(typescript_custom_section)]
const TYPES: &str = r#"
import type { Config, Capabilities, ValidationReport, PlanSummary } from "../../lib/contract/generated";
import type { Result } from "../../lib/contract/records";
import type { BuildInfo, Preview } from "../../lib/compute/protocol";
"#;
