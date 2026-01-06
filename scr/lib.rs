use pyo3::prelude::*;

#[pyfunction]
fn hello() -> String {
    "Hello from Rust!".to_string()
}

#[pymodule]
fn categorical_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    Ok(())
}
