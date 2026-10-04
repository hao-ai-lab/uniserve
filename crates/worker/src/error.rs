//! Preserve the worker's error codes across native and numerical execution.

use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(crate) fn invalid(py: Python<'_>, message: impl Into<String>) -> PyErr {
    match py.import("uniserve_worker.errors").and_then(|module| {
        module
            .getattr("invalid_descriptor")?
            .call1((message.into(),))
    }) {
        Ok(error) => PyErr::from_value(error),
        Err(error) => error,
    }
}

pub(crate) fn invariant(py: Python<'_>, message: impl Into<String>) -> PyErr {
    let error = (|| {
        let module = py.import("uniserve_worker.errors")?;
        let code = module
            .getattr("WorkerErrorCode")?
            .getattr("INVARIANT_VIOLATION")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("fatal", true)?;
        module
            .getattr("WorkerError")?
            .call((code, message.into()), Some(&kwargs))
    })();
    match error {
        Ok(error) => PyErr::from_value(error),
        Err(error) => error,
    }
}
