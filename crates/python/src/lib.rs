//! Python bindings placeholder crate.

#[cfg(feature = "bindings")]
mod py {
    use pyo3::prelude::*;

    #[pymodule]
    fn mpb2d_py(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add("__doc__", "mpb2d-lite python bindings placeholder")?;
        Ok(())
    }
}

#[cfg(not(feature = "bindings"))]
pub fn bindings_disabled() {
    eprintln!("mpb2d-python compiled without the \"bindings\" feature");
}
