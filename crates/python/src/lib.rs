//! Python bindings for the shared calculation and result contract.

#[cfg(feature = "bindings")]
mod interface;
#[cfg(feature = "bindings")]
mod operator_data;

#[cfg(feature = "bindings")]
use pyo3::prelude::*;

#[cfg(feature = "bindings")]
#[pymodule(name = "_native", gil_used = true)]
fn blaze_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    interface::register(m)?;
    operator_data::register_operator_data(m)?;
    Ok(())
}

#[cfg(not(feature = "bindings"))]
pub fn bindings_disabled() {}
