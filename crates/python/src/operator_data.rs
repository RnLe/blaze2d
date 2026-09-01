//! Native research inputs for shared planned calculations.
use blaze2d_interface as api;
use blaze2d_backend_cpu::CpuBackend;
use blaze2d_core::field::Field2D;
use numpy::{Complex64, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4, PyUntypedArrayMethods};
use pyo3::{prelude::*, types::PyDict, exceptions::PyValueError};
use crate::interface::{CalculationError, Configuration, config_error, failure_to_py, result_to_py};

fn single_job(py: Python<'_>, config: &Configuration) -> PyResult<api::PlannedJob> {
    if config.plan.summary.jobs != 1 { return Err(PyValueError::new_err("This entry point requires exactly one planned job")); }
    config.plan.job(0).map_err(|d|config_error(py,d))
}

fn fields(array: Option<PyReadonlyArray3<'_,Complex64>>, job: &api::PlannedJob) -> PyResult<Option<Vec<Field2D>>> {
    array.map(|a| {
        let [nx,ny] = job.resolved.resolution;
        if a.shape() != [job.resolved.solved_bands,ny,nx] {
            return Err(PyValueError::new_err("Expected fields with shape (solved_band, ny, nx)"));
        }
        Ok(a.as_slice()?.chunks_exact(nx*ny).map(|values| Field2D::from_f64_vec(job.grid(),values.to_vec())).collect())
    }).transpose()
}

#[pyfunction]
#[pyo3(signature=(config, reference_eigenvectors=None, warmstart_eigenvectors=None))]
fn _extract_with_reference(py: Python<'_>, config: &Configuration,
    reference_eigenvectors: Option<PyReadonlyArray3<'_,Complex64>>,
    warmstart_eigenvectors: Option<PyReadonlyArray3<'_,Complex64>>) -> PyResult<Py<PyDict>>
{
    let job = single_job(py,config)?;
    let reference = fields(reference_eigenvectors,&job)?;
    let warm = fields(warmstart_eigenvectors,&job)?;
    let outcome = py.detach(|| std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        match job.resolved.config.eigensolver.precision {
            api::Precision::F64 => api::execute_with_fields(CpuBackend::<f64>::new(),"cpu",&job,reference.as_deref(),warm.as_deref(),|_|{}),
            api::Precision::F32 => api::execute_with_fields(CpuBackend::<f32>::new(),"cpu",&job,reference.as_deref(),warm.as_deref(),|_|{}),
        }
    }))).map_err(|_|CalculationError::new_err("Internal solver failure"))?;
    match outcome {
        Ok(result) => result_to_py(py,result),
        Err(failure) => {
            let error = CalculationError::new_err(failure.diagnostic.message.clone());
            for (key,value) in failure_to_py(py,failure)?.bind(py).iter() {
                error.value(py).setattr(key.extract::<String>()?,value)?;
            }
            Err(error)
        }
    }
}

#[pyfunction]
#[pyo3(signature=(config, epsilon, inverse_epsilon_tensors=None))]
fn _solve_external_map(py: Python<'_>, config: &Configuration, epsilon: PyReadonlyArray2<'_,f64>,
    inverse_epsilon_tensors: Option<PyReadonlyArray4<'_,f64>>) -> PyResult<Py<PyDict>>
{
    let job = single_job(py,config)?;
    let [nx,ny] = job.resolved.resolution;
    if epsilon.shape() != [ny,nx] { return Err(PyValueError::new_err("Epsilon must have shape (ny, nx)")); }
    let epsilon = epsilon.as_slice()?.to_vec();
    let tensors = inverse_epsilon_tensors.map(|a| {
        if a.shape() != [ny,nx,2,2] { return Err(PyValueError::new_err("Inverse-epsilon tensors must have shape (ny, nx, 2, 2)")); }
        Ok(a.as_slice()?.chunks_exact(4).map(|p|[p[0],p[1],p[2],p[3]]).collect())
    }).transpose()?;
    let result = py.detach(|| std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        match job.resolved.config.eigensolver.precision {
            api::Precision::F64 => api::external::execute_sampled(CpuBackend::<f64>::new(),"cpu",&job,epsilon,tensors),
            api::Precision::F32 => api::external::execute_sampled(CpuBackend::<f32>::new(),"cpu",&job,epsilon,tensors),
        }
    }))).map_err(|_|CalculationError::new_err("Internal solver failure"))?
        .map_err(|d|config_error(py,d))?;
    result_to_py(py,result)
}

pub fn register_operator_data(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_extract_with_reference,m)?)?;
    m.add_function(wrap_pyfunction!(_solve_external_map,m)?)?;
    Ok(())
}
