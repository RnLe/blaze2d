//! Lossless portable array records. Complex data uses interleaved real/imaginary f64.

use std::collections::BTreeMap;
use num_complex::Complex64;
use serde::{Serialize, Deserialize};
use serde_json::{Value, json};
use crate::{Diagnostic, InterfaceResult, PlannedJob, Task, RESULT_SCHEMA};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType { Float64, Complex128 }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Array {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub dimensions: Vec<String>,
    pub order: String,
    pub data: Vec<f64>,
}

impl Array {
    pub fn real(shape: &[usize], dimensions: &[&str], data: impl IntoIterator<Item=f64>) -> Self {
        Self { dtype: DType::Float64, shape: shape.into(),
            dimensions: dimensions.iter().map(|s| s.to_string()).collect(),
            order: "C".into(), data: data.into_iter().collect() }
    }
    pub fn complex(shape: &[usize], dimensions: &[&str], data: impl IntoIterator<Item=Complex64>) -> Self {
        let mut array = Self::real(shape, dimensions, data.into_iter().flat_map(|c| [c.re, c.im]));
        array.dtype = DType::Complex128;
        array
    }
    pub fn validate(&self, name: &str) -> InterfaceResult<()> {
        let count = self.shape.iter().try_fold(1usize, |a, &b| a.checked_mul(b))
            .and_then(|n| n.checked_mul(if self.dtype == DType::Complex128 {2} else {1}));
        if count != Some(self.data.len()) || self.dimensions.len() != self.shape.len() || self.order != "C" {
            return Err(Diagnostic::new("array_shape", name, "Array dimensions do not match its data"));
        }
        if self.data.iter().any(|v| !v.is_finite()) {
            return Err(Diagnostic::new("nonfinite_result", name, "Array contains non-finite values; lossless JSON export is unavailable"));
        }
        Ok(())
    }
}

pub type Arrays = BTreeMap<String, Array>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleRecord {
    pub sample_index: usize,
    pub metadata: Value,
    pub arrays: Arrays,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultRecord {
    pub schema: String,
    pub task: Task,
    pub job_index: usize,
    pub metadata: Value,
    pub arrays: Arrays,
    #[serde(default, skip_serializing_if="Vec::is_empty")]
    pub samples: Vec<SampleRecord>,
}

impl ResultRecord {
    pub fn new(job: &PlannedJob, backend: &str) -> Self {
        Self { schema: RESULT_SCHEMA.into(), task: job.resolved.config.task,
            job_index: job.index, arrays: Arrays::new(), samples: Vec::new(),
            metadata: json!({
                "config_schema": crate::CONFIG_SCHEMA, "build": crate::build_info(),
                "backend": backend, "storage_precision": job.resolved.config.eigensolver.precision,
                "accumulation_precision": "f64", "array_order": "C",
                "config": job.resolved.config, "sweep_parameters": job.sweep,
                "multi_index": job.multi_index, "registry_index": job.registry_index,
                "registry": job.registry, "resolution": job.resolved.resolution,
                "lattice_vectors": job.resolved.lattice_vectors,
                "coordinates": {
                    "length": "common reference length", "centers": "direct_fractional",
                    "wavevector": "cartesian_angular", "wavevector_transform": "k = 2*pi*A^(-T)*q",
                    "frequency": "sqrt(lambda)/(2*pi)", "frequency_unit": "c/reference_length",
                    "k_derivatives": "cartesian_angular", "registry_derivatives": "direct_fractional"
                },
                "gauge": if job.resolved.config.task == Task::Bands { "tracked_path" } else { "independent_registry; adjacent_transport_within_stencil" },
                "inner_product": if job.resolved.config.polarization == crate::Polarization::TM { "epsilon_weighted" } else { "euclidean" },
            }) }
    }
    pub fn validate_arrays(&self) -> InterfaceResult<()> {
        for (name, a) in &self.arrays { a.validate(name)?; }
        for s in &self.samples { for (name, a) in &s.arrays { a.validate(name)?; } }
        Ok(())
    }
    pub fn to_json(&self) -> InterfaceResult<String> {
        self.validate_arrays()?;
        serde_json::to_string(self).map_err(|e| Diagnostic::new("serialization", "", e.to_string()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobFailure {
    pub job_index: usize,
    pub diagnostic: Diagnostic,
    #[serde(skip_serializing_if="Option::is_none")]
    pub partial_result: Option<Box<ResultRecord>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag="event", rename_all="snake_case")]
pub enum Event {
    RunStart { jobs: usize, solves: usize },
    JobStart { job_index: usize },
    Progress { job_index: usize, sample_index: usize, completed: usize, total: usize, iterations: usize, converged: bool },
    Result { result: Box<ResultRecord> },
    JobFailure { error: Box<JobFailure> },
    Terminal { status: RunStatus, completed: usize, failed: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all="snake_case")]
pub enum RunStatus { Completed, CompletedWithErrors, Failed, Cancelled }
