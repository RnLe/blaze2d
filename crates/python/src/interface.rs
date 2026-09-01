use blaze2d_interface as api;
use numpy::{IntoPyArray, ndarray::{ArrayD, IxDyn}, PyArray1};
use pyo3::{prelude::*, types::{PyDict, PyList}, exceptions::{PyValueError, PyRuntimeError}};
use serde_json::{Value, json};

pyo3::create_exception!(_native, ConfigurationError, PyValueError);
pyo3::create_exception!(_native, CalculationError, PyRuntimeError);

pub fn value_to_py<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    let text = serde_json::to_string(value).map_err(|e| PyValueError::new_err(e.to_string()))?;
    py.import("json")?.call_method1("loads", (text,))
}

pub(crate) fn config_error(py: Python<'_>, d: api::Diagnostic) -> PyErr {
    let error = ConfigurationError::new_err(d.to_string());
    if let Ok(value) = value_to_py(py, &json!(d)) { let _ = error.value(py).setattr("diagnostic", value); }
    error
}

fn put_arrays(py: Python<'_>, dict: &Bound<'_, PyDict>, arrays: api::Arrays) -> PyResult<()> {
    let descriptors = PyDict::new(py);
    for (name, a) in arrays {
        a.validate_shape(&name).map_err(|e| PyValueError::new_err(e.to_string()))?;
        descriptors.set_item(&name, value_to_py(py, &json!({"dtype":a.dtype,"shape":a.shape,
            "dimensions":a.dimensions,"order":a.order}))?)?;
        match a.dtype {
            api::DType::Float64 => {
                let array = ArrayD::from_shape_vec(IxDyn(&a.shape), a.data)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                dict.set_item(name, array.into_pyarray(py))?;
            }
            api::DType::Complex128 => {
                let buffer = PyArray1::from_vec(py, a.data);
                let array = buffer.call_method1("view", ("complex128",))?.call_method1("reshape", (a.shape,))?;
                dict.set_item(name, array)?;
            }
        }
    }
    dict.set_item("array_info", descriptors)?;
    Ok(())
}

pub fn sample_to_py(py: Python<'_>, sample: api::SampleRecord) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("sample_index", sample.sample_index)?;
    dict.set_item("metadata", value_to_py(py, &sample.metadata)?)?;
    put_arrays(py, &dict, sample.arrays)?;
    Ok(dict.unbind())
}

pub fn result_to_py(py: Python<'_>, result: api::ResultRecord) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("schema", result.schema)?;
    dict.set_item("task", if result.task == api::Task::Bands {"bands"} else {"operators"})?;
    dict.set_item("job_index", result.job_index)?;
    dict.set_item("metadata", value_to_py(py, &result.metadata)?)?;
    put_arrays(py, &dict, result.arrays)?;
    if !result.samples.is_empty() {
        let samples = PyList::empty(py);
        for sample in result.samples { samples.append(sample_to_py(py, sample)?)?; }
        dict.set_item("samples", samples)?;
    }
    Ok(dict.unbind())
}

pub(crate) fn failure_to_py(py: Python<'_>, error: api::JobFailure) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("job_index", error.job_index)?;
    dict.set_item("diagnostic", value_to_py(py, &json!(error.diagnostic))?)?;
    if let Some(result) = error.partial_result { dict.set_item("partial_result", result_to_py(py, *result)?)?; }
    Ok(dict.unbind())
}

#[pyclass(frozen)]
pub struct Configuration { pub(crate) plan: api::Plan }

#[pymethods]
impl Configuration {
    #[new]
    #[pyo3(signature=(source, format="toml"))]
    fn new(py: Python<'_>, source: &str, format: &str) -> PyResult<Self> {
        let config = match format {
            "toml" => api::Config::from_toml(source),
            "json" => serde_json::from_str(source).map_err(|e| api::Diagnostic::new("invalid_json", "", e.to_string()))
                .and_then(api::Config::from_value),
            _ => Err(api::Diagnostic::new("input_format", "", "Use toml or json")),
        }.map_err(|d| config_error(py, d))?;
        let plan = api::Plan::new(config, api::Platform::Native).map_err(|d| config_error(py, d))?;
        Ok(Self { plan })
    }
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> { value_to_py(py, &json!(self.plan.config)) }
    fn to_toml(&self, py: Python<'_>) -> PyResult<String> { self.plan.config.to_toml().map_err(|d| config_error(py, d)) }
    #[getter]
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> { value_to_py(py, &json!(self.plan.summary)) }
    fn resolved<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        value_to_py(py, &json!(self.plan.config.resolve().map_err(|d| config_error(py, d))?))
    }
    fn solve(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        if self.plan.summary.jobs != 1 { return Err(PyValueError::new_err("solve requires exactly one job")); }
        let job = self.plan.job(0).map_err(|d| config_error(py,d))?;
        let outcome = py.detach(|| std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            match job.resolved.config.eigensolver.precision {
                api::Precision::F64 => api::execute(blaze2d_backend_cpu::CpuBackend::<f64>::new(), "cpu", &job, |_|{}),
                api::Precision::F32 => api::execute(blaze2d_backend_cpu::CpuBackend::<f32>::new(), "cpu", &job, |_|{}),
            }
        }))).map_err(|_| CalculationError::new_err("Internal solver failure"))?;
        match outcome {
            Ok(mut result) => {
                result.metadata["runner"] = json!({"threads":1,"queue_capacity":0,"error_policy":"stop",
                    "parallelism":"independent_configurations","nested_threads":false});
                result_to_py(py,result)
            },
            Err(failure) => {
                let error = CalculationError::new_err(failure.diagnostic.message.clone());
                let record = failure_to_py(py,failure)?;
                for (key,value) in record.bind(py).iter() { error.value(py).setattr(key.extract::<String>()?,value)?; }
                Err(error)
            }
        }
    }
    #[pyo3(signature=(options="{}"))]
    fn events(&self, options: &str) -> PyResult<Events> {
        let options = serde_json::from_str(options).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let stream = blaze2d_runner::start(self.plan.clone(), options).map_err(|d| PyValueError::new_err(d.to_string()))?;
        Ok(Events { stream, done: false })
    }
}

#[pyclass]
pub struct Events { stream: blaze2d_runner::RunStream, done: bool }
#[pymethods]
impl Events {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if self.done { return Ok(None); }
        let event = py.detach(|| self.stream.next_event());
        let Some(event) = event else { self.done = true; return Ok(None); };
        let dict = PyDict::new(py);
        match event {
            api::Event::Result {result} => {
                dict.set_item("event", "result")?;
                dict.set_item("result", result_to_py(py, *result)?)?;
            },
            api::Event::JobFailure {error} => {
                dict.set_item("event", "job_failure")?;
                dict.set_item("error", failure_to_py(py, *error)?)?;
            },
            other => {
                self.done = matches!(other, api::Event::Terminal {..});
                return Ok(Some(value_to_py(py, &json!(other))?.unbind()));
            }
        }
        Ok(Some(dict.into_any().unbind()))
    }
    fn cancel(&self) { self.stream.cancel(); }
}

#[pyfunction]
fn build_info(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> { value_to_py(py, &api::build_info()) }
#[pyfunction]
fn capabilities(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> { value_to_py(py, &json!(api::capabilities(api::Platform::Native))) }
#[pyfunction]
fn defaults(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> { value_to_py(py, &json!(api::Config::default())) }
#[pyfunction]
fn describe(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> { value_to_py(py, &json!(api::schema())) }

#[derive(serde::Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct SimpleOptions {
    lattice_type: Option<api::config::LatticeKind>, epsilon_background: Option<f64>,
    epsilon_atoms: Option<f64>, radius_atom: Option<f64>, polarization: Option<api::Polarization>,
    resolution: Option<api::Resolution>, n_bands: Option<usize>, k_path: Option<api::KPath>,
    tolerance: Option<f64>, max_iterations: Option<usize>, precision: Option<api::Precision>,
    eigenvectors: Option<bool>,
}

#[pyfunction]
fn simple_config(py: Python<'_>, options: &str) -> PyResult<Configuration> {
    let o: SimpleOptions = serde_json::from_str(options).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut c = api::Config::default();
    if let Some(v) = o.lattice_type { c.geometry.lattice.kind = v; }
    if let Some(v) = o.epsilon_background { c.geometry.background_epsilon = v; }
    if let Some(v) = o.epsilon_atoms { c.geometry.objects[0].epsilon = v; }
    if let Some(v) = o.radius_atom { c.geometry.objects[0].radius = v; }
    if let Some(v) = o.polarization { c.polarization = v; }
    if let Some(v) = o.resolution { c.grid.resolution = v; }
    if let Some(v) = o.n_bands { c.bands.as_mut().unwrap().count = v; }
    if let Some(v) = o.k_path { c.bands.as_mut().unwrap().path = v; }
    if let Some(v) = o.tolerance { c.eigensolver.tolerance = Some(v); }
    if let Some(v) = o.max_iterations { c.eigensolver.max_iterations = Some(v); }
    if let Some(v) = o.precision { c.eigensolver.precision = v; }
    if let Some(v) = o.eigenvectors { c.results.eigenvectors = v; }
    Ok(Configuration { plan: api::Plan::new(c, api::Platform::Native).map_err(|d| config_error(py, d))? })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ConfigurationError", m.py().get_type::<ConfigurationError>())?;
    m.add("CalculationError", m.py().get_type::<CalculationError>())?;
    m.add_class::<Configuration>()?;
    m.add_class::<Events>()?;
    m.add_function(wrap_pyfunction!(build_info,m)?)?;
    m.add_function(wrap_pyfunction!(capabilities,m)?)?;
    m.add_function(wrap_pyfunction!(defaults,m)?)?;
    m.add_function(wrap_pyfunction!(describe,m)?)?;
    m.add_function(wrap_pyfunction!(simple_config,m)?)?;
    Ok(())
}
