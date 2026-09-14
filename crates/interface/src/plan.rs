use std::collections::{BTreeMap, HashSet};
use serde::{Deserialize, Serialize};
use schemars::JsonSchema;
use serde_json::{Value, json};
use crate::*;
use crate::normalize::invalid;

#[derive(Debug, Default, Copy, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Platform { #[default] Native, Browser }

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Capabilities {
    pub schema: String,
    pub version: String,
    pub platform: Platform,
    pub dimensions: Vec<usize>,
    pub tasks: Vec<Task>,
    pub precisions: Vec<Precision>,
    pub geometries: Vec<ObjectKind>,
    pub registry: bool,
    pub k_stencil: bool,
    pub external_inputs: bool,
    pub checkpoint_resume: bool,
}

pub fn capabilities(platform: Platform) -> Capabilities {
    Capabilities {
        schema: CONFIG_SCHEMA.into(), version: env!("CARGO_PKG_VERSION").into(), platform,
        dimensions: vec![2], tasks: vec![Task::Bands, Task::Operators],
        precisions: if platform == Platform::Native { vec![Precision::F64, Precision::F32] } else { vec![Precision::F64] },
        geometries: vec![ObjectKind::Circle], registry: true, k_stencil: true,
        external_inputs: platform == Platform::Native, checkpoint_resume: platform == Platform::Native,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PlanSummary {
    pub jobs: usize,
    pub solves: usize,
    pub estimated_peak_bytes: u64,
    pub resolution: [usize; 2],
    pub solved_bands: usize,
    pub task: Task,
    pub precision: Precision,
    pub sweep_shape: Vec<usize>,
    pub registry_points: usize,
}

#[derive(Debug, Clone)]
pub struct Plan {
    pub config: Config,
    pub summary: PlanSummary,
    pub platform: Platform,
    axis_lengths: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
pub struct PlannedJob {
    pub index: usize,
    pub sweep: BTreeMap<String, Value>,
    pub multi_index: Vec<usize>,
    pub registry_index: Option<usize>,
    pub registry: [f64; 2],
    pub resolved: ResolvedConfig,
}

impl Sweep {
    pub fn len(&self) -> InterfaceResult<usize> {
        match (&self.values, &self.linspace) {
            (Some(v), None) if !v.is_empty() => Ok(v.len()),
            (None, Some(l)) if l.count > 0 && l.start.is_finite() && l.stop.is_finite() => {
                if l.count == 1 && l.start != l.stop { return Err(invalid("sweeps.linspace", "a singleton linspace must have equal endpoints")); }
                Ok(l.count)
            }
            _ => Err(invalid("sweeps", "supply nonempty values or a finite linspace with a positive count")),
        }
    }

    pub fn value(&self, index: usize) -> InterfaceResult<Value> {
        let count = self.len()?;
        if index >= count { return Err(invalid("sweeps", "sample index is out of bounds")); }
        if let Some(v) = &self.values { return Ok(v[index].clone()); }
        let l = self.linspace.as_ref().unwrap();
        let value = if index == 0 { l.start } else if index == count - 1 { l.stop } else {
            let t = index as f64 / (count - 1) as f64;
            l.start * (1.0 - t) + l.stop * t
        };
        Ok(json!(value))
    }
}

fn integer_value(value: Value, target: &str) -> InterfaceResult<Value> {
    if let Some(v) = value.as_u64() { return Ok(json!(v)); }
    if let Some(v) = value.as_f64() {
        if v.is_finite() && v >= 0.0 && v < u64::MAX as f64 && v.fract() == 0.0 { return Ok(json!(v as u64)); }
    }
    Err(invalid(target, "expected a nonnegative integer"))
}

fn integer_target(target: &str) -> bool {
    matches!(target, "bands.count" | "operators.band_lo" | "operators.retained_bands" | "operators.remote_bands" |
        "eigensolver.max_iterations" | "eigensolver.block_size" | "grid.resolution")
}

pub fn set_parameter(config: &mut Config, target: &str, mut value: Value) -> InterfaceResult<()> {
    let parts: Vec<_> = target.split('.').collect();
    let mut root = serde_json::to_value(&*config).map_err(|e| invalid(target, e.to_string()))?;
    let mut path = parts.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let allowed = matches!(target, "polarization" | "geometry.background_epsilon" | "geometry.lattice.a" |
        "geometry.lattice.b" | "geometry.lattice.angle_deg" | "grid.resolution" | "bands.count" |
        "operators.band_lo" | "operators.retained_bands" | "operators.remote_bands" | "operators.k_point.value" |
        "eigensolver.tolerance" | "eigensolver.max_iterations" | "eigensolver.block_size");
    if parts.len() == 4 && parts[0] == "geometry" && parts[1] == "objects" && matches!(parts[3], "radius" | "epsilon" | "center") {
        let index = config.geometry.objects.iter().position(|o| o.name == parts[2])
            .ok_or_else(|| invalid(target, "unknown object name"))?;
        path[2] = index.to_string();
    } else if !allowed {
        return Err(Diagnostic::new("unsupported_sweep_target", target, "this field is not a supported sweep target"));
    }
    if integer_target(target) {
        value = if target == "grid.resolution" && value.is_array() {
            Value::Array(value.as_array().unwrap().iter().cloned().map(|v| integer_value(v, target)).collect::<InterfaceResult<_>>()?)
        } else { integer_value(value, target)? };
    }
    let mut node = &mut root;
    for part in &path[..path.len() - 1] {
        node = match node {
            Value::Array(array) => array.get_mut(part.parse::<usize>().map_err(|_| invalid(target, "invalid array index"))?)
                .ok_or_else(|| invalid(target, "array index out of bounds"))?
            ,
            Value::Object(object) => object.get_mut(part).filter(|v| !v.is_null()).ok_or_else(|| invalid(target, "target section is not active"))?,
            _ => return Err(invalid(target, "target section is not active")),
        };
    }
    let object = node.as_object_mut().ok_or_else(|| invalid(target, "target is not a field"))?;
    object.insert(path.last().unwrap().clone(), value);
    *config = Config::from_value(root).map_err(|e| invalid(target, e.message))?;
    Ok(())
}

impl Plan {
    pub fn new(config: Config, platform: Platform) -> InterfaceResult<Self> {
        let base = config.resolve()?;
        let config = base.config.clone();
        if config.dielectric.source == DielectricSource::External {
            if platform == Platform::Browser {
                return Err(Diagnostic::new("unsupported_capability", "dielectric.source", "external dielectric arrays require the native Python interface"));
            }
            if config.task != Task::Bands || base.k_points_fractional.len() != 1 || !config.geometry.objects.is_empty() {
                return Err(invalid("dielectric.source", "external dielectric input requires one sampled band point and an empty geometry object list"));
            }
        }
        if platform == Platform::Browser && config.operators.as_ref().is_some_and(|o|o.reference.is_some()) {
            return Err(Diagnostic::new("unsupported_capability", "operators.reference", "external reference fields require the native Python interface"));
        }
        if platform == Platform::Browser && config.eigensolver.precision != Precision::F64 {
            return Err(Diagnostic::new("unsupported_precision", "eigensolver.precision", "the browser backend supports f64; use Python or the native CLI for f32"));
        }
        let mut names = HashSet::new();
        let mut targets = HashSet::new();
        let mut axis_lengths = vec![];
        let mut combinations = 1usize;
        let mut envelope = base.clone();
        let mut min_samples = base.resolution[0] * base.resolution[1];
        let mut upper_window = config.clone();
        let mut min_explicit_block = if config.eigensolver.block_size == 0 { usize::MAX } else { config.eigensolver.block_size };
        for (axis_index, axis) in config.sweeps.iter().enumerate() {
            if axis.name.is_empty() || !names.insert(axis.name.clone()) { return Err(invalid("sweeps.name", "axis names must be nonempty and unique")); }
            if !targets.insert(axis.target.clone()) { return Err(invalid("sweeps.target", "each target may appear only once")); }
            let length = axis.len()?;
            combinations = combinations.checked_mul(length).ok_or_else(|| invalid("sweeps", "job count overflows"))?;
            axis_lengths.push(length);
            if let Some(l) = &axis.linspace {
                if integer_target(&axis.target) && l.count > 1 && ((l.stop - l.start) / (l.count - 1) as f64).fract() != 0.0 {
                    return Err(invalid("sweeps.linspace", "integer targets require an integer interval"));
                }
            }
            let indices: Vec<_> = if axis.values.is_some() { (0..length).collect() } else { vec![0, length - 1] };
            let mut max_integer = 0u64;
            if axis.target == "grid.resolution" { min_samples = usize::MAX; }
            if axis.target == "eigensolver.block_size" {
                min_explicit_block = usize::MAX;
                envelope.config.eigensolver.block_size = 0;
            }
            for i in indices {
                let mut candidate = config.clone();
                let value = axis.value(i)?;
                set_parameter(&mut candidate, &axis.target, value.clone()).map_err(|mut e| { e.path = format!("sweeps[{axis_index}].{}", e.path); e })?;
                let resolved = candidate.resolve_fields(false)?;
                if axis.target == "grid.resolution" {
                    min_samples = min_samples.min(resolved.resolution[0] * resolved.resolution[1]);
                }
                if axis.target == "eigensolver.block_size" && resolved.config.eigensolver.block_size > 0 {
                    min_explicit_block = min_explicit_block.min(resolved.config.eigensolver.block_size);
                }
                for axis in 0..2 {
                    envelope.resolution[axis] = envelope.resolution[axis].max(resolved.resolution[axis]);
                }
                if axis.target == "eigensolver.block_size" {
                    envelope.config.eigensolver.block_size = envelope.config.eigensolver.block_size.max(resolved.config.eigensolver.block_size);
                }
                if matches!(axis.target.as_str(), "bands.count" | "operators.band_lo" | "operators.retained_bands" | "operators.remote_bands") {
                    max_integer = max_integer.max(integer_value(value, &axis.target)?.as_u64().unwrap());
                }
            }
            if matches!(axis.target.as_str(), "bands.count" | "operators.band_lo" | "operators.retained_bands" | "operators.remote_bands") {
                set_parameter(&mut upper_window, &axis.target, json!(max_integer))?;
            }
        }
        let max_bands = upper_window.resolve_fields(false)?.solved_bands;
        if min_explicit_block < max_bands {
            return Err(invalid("sweeps", "every explicit block size must contain the largest swept band window"));
        }
        if envelope.config.eigensolver.block_size > min_samples {
            return Err(invalid("sweeps", "a swept block size exceeds the smallest swept grid"));
        }
        if max_bands >= min_samples { return Err(invalid("sweeps", "a swept band window does not fit the smallest swept grid")); }
        // The largest grid and solver block can occur together in a Cartesian
        // sweep. Estimating each axis independently underestimates that job.
        envelope.solved_bands = max_bands;
        let peak = estimate_memory(&envelope)?;
        let registry_points = config.operators.as_ref().and_then(|o| o.registry.as_ref()).map_or(1, |r| r.points.len());
        let jobs = combinations.checked_mul(registry_points).ok_or_else(|| invalid("sweeps", "job count overflows"))?;
        let per_job = match config.task {
            Task::Bands => base.k_points_fractional.len(),
            Task::Operators => config.operators.as_ref().and_then(|o| o.k_stencil.as_ref()).map_or(1, |s| s.points_per_axis * s.points_per_axis),
        };
        let solves = jobs.checked_mul(per_job).ok_or_else(|| invalid("sweeps", "solve count overflows"))?;
        let summary = PlanSummary { jobs, solves, estimated_peak_bytes: peak,
            resolution: base.resolution, solved_bands: base.solved_bands, task: config.task,
            precision: config.eigensolver.precision, sweep_shape: axis_lengths.clone(), registry_points };
        Ok(Self { config, summary, platform, axis_lengths })
    }

    pub fn job(&self, index: usize) -> InterfaceResult<PlannedJob> {
        if index >= self.summary.jobs { return Err(invalid("job_index", "out of bounds")); }
        let registry_index = index % self.summary.registry_points;
        let mut remainder = index / self.summary.registry_points;
        let mut multi_index = vec![0; self.axis_lengths.len()];
        for axis in (0..multi_index.len()).rev() {
            multi_index[axis] = remainder % self.axis_lengths[axis];
            remainder /= self.axis_lengths[axis];
        }
        let mut config = self.config.clone();
        let mut sweep = BTreeMap::new();
        for (axis, &sample) in self.config.sweeps.iter().zip(&multi_index) {
            let value = axis.value(sample)?;
            set_parameter(&mut config, &axis.target, value.clone())?;
            sweep.insert(axis.name.clone(), value);
        }
        config.sweeps.clear();
        let mut registry = [0.0; 2];
        let mut has_registry = false;
        if let Some(r) = config.operators.as_mut().and_then(|o| o.registry.as_mut()) {
            registry = crate::normalize::point(&r.points[registry_index], "operators.registry.points")?;
            r.points = vec![registry.to_vec()];
            has_registry = true;
        }
        Ok(PlannedJob { index, sweep, multi_index,
            registry_index: has_registry.then_some(registry_index), registry,
            resolved: config.resolve()? })
    }
}

fn estimate_memory(resolved: &ResolvedConfig) -> InterfaceResult<u64> {
    let samples = resolved.resolution[0] as u64 * resolved.resolution[1] as u64;
    let bands = resolved.solved_bands as u64;
    let block = blaze2d_core::eigensolver::EigensolverConfig {
        n_bands: resolved.solved_bands,
        block_size: resolved.config.eigensolver.block_size,
        ..Default::default()
    }.effective_block_size() as u64;
    let stencil = resolved.config.operators.as_ref().and_then(|o| o.k_stencil.as_ref()).map_or(1, |s| s.points_per_axis as u64 * s.points_per_axis as u64);
    let fields = block.checked_mul(24).and_then(|n| n.checked_add(16))
        .and_then(|n| bands.checked_mul(stencil)?.checked_mul(3)?.checked_add(n))
        .ok_or_else(|| invalid("configuration", "memory estimate overflows"))?;
    let fields = samples.checked_mul(fields).and_then(|n| n.checked_mul(16));
    let dense = block.checked_mul(block).and_then(|n| n.checked_mul(1024));
    let nk = resolved.k_points_fractional.len() as u64;
    let path_arrays = nk.checked_mul(bands.checked_mul(4).and_then(|n|n.checked_add(16))
        .ok_or_else(|| invalid("configuration", "memory estimate overflows"))?).and_then(|n|n.checked_mul(8));
    let retained_fields = if resolved.config.task == Task::Bands && resolved.config.results.eigenvectors {
        nk.checked_mul(bands).and_then(|n|n.checked_mul(samples)).and_then(|n|n.checked_mul(32))
    } else {Some(0)};
    fields.and_then(|n| n.checked_add(dense?)).and_then(|n|n.checked_add(path_arrays?))
        .and_then(|n|n.checked_add(retained_fields?)).ok_or_else(|| invalid("configuration", "memory estimate overflows"))
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct ValidationReport {
    pub ok: bool,
    pub errors: Vec<Diagnostic>,
    pub config: Option<Config>,
    pub resolved: Option<ResolvedConfig>,
    pub summary: Option<PlanSummary>,
    pub capabilities: Capabilities,
}

pub fn validate_toml(source: &str, platform: Platform) -> ValidationReport {
    match Config::from_toml(source).and_then(|config| Plan::new(config, platform)) {
        Ok(plan) => ValidationReport { ok: true, errors: vec![], resolved: plan.config.resolve().ok(),
            config: Some(plan.config), summary: Some(plan.summary), capabilities: capabilities(platform) },
        Err(error) => ValidationReport { ok: false, errors: vec![error], config: None, resolved: None,
            summary: None, capabilities: capabilities(platform) },
    }
}
