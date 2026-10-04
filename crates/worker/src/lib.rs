//! Rank-local inference execution with a Python numerical backend.
//!
//! The engine assigns calls and logical resource slots. This crate owns their
//! execution lifetime on a worker; Python retains model computation and tensor
//! operations. Native request and storage objects are shared by serving and
//! library callers.

use pyo3::prelude::*;

mod buffer;
mod error;
mod protocol;
mod request;
mod storage;

/// Register the worker objects in the common native extension.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<buffer::BufferBinding>()?;
    module.add_class::<buffer::BufferPool>()?;
    module.add_class::<request::Request>()?;
    module.add_class::<request::RequestPool>()?;
    module.add_class::<storage::Buffer>()?;
    module.add_class::<storage::TensorRead>()?;
    module.add_class::<storage::TensorImport>()?;
    module.add_class::<storage::TensorStore>()?;
    Ok(())
}
