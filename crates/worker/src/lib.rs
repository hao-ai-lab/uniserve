//! Rank-local inference execution with a Python numerical backend.
//!
//! The engine assigns calls and logical resource slots. This crate owns their
//! execution lifetime on a worker; Python retains model computation and tensor
//! operations. Native request objects are shared by serving and library callers.

use pyo3::prelude::*;

mod request;

/// Register the worker objects in the common native extension.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<request::Request>()?;
    module.add_class::<request::RequestPool>()?;
    Ok(())
}
