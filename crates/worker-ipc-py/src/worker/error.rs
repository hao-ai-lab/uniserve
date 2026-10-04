//! Translate worker failures at the Python interface.

use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(super) fn native_error(py: Python<'_>, error: uniserve_worker::Error) -> PyErr {
    match error {
        uniserve_worker::Error::Invalid(message) => invalid(py, message),
        uniserve_worker::Error::State(message) => {
            pyo3::exceptions::PyRuntimeError::new_err(message)
        }
        uniserve_worker::Error::Resource(message) => resource(py, message),
        uniserve_worker::Error::Unsupported(message) => unsupported(py, message),
        error @ uniserve_worker::Error::ReadBackpressure { .. } => resource(py, error.to_string()),
        uniserve_worker::Error::Invariant(message) => invariant(py, message),
    }
}

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

pub(crate) fn resource(py: Python<'_>, message: impl Into<String>) -> PyErr {
    match py
        .import("uniserve_worker.errors")
        .and_then(|module| module.getattr("resource_error")?.call1((message.into(),)))
    {
        Ok(error) => PyErr::from_value(error),
        Err(error) => error,
    }
}

pub(crate) fn unsupported(py: Python<'_>, message: impl Into<String>) -> PyErr {
    match py.import("uniserve_worker.errors").and_then(|module| {
        module
            .getattr("unsupported_setup")?
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
