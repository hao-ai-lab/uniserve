//! Overlap reserved input copies with numerical work on the caller's stream.

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use super::ModelExecutor;
use crate::worker::execution::close_all;
use crate::worker::host::with_context;
use crate::worker::stream::CUDAStream;

/// The caller reserves both ends of each copy through physical completion.
/// The scope joins on normal exit, a body error, or a partially submitted copy.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(super) struct InputCopies {
    pub(super) owner: Py<ModelExecutor>,
    pub(super) stream: Py<PyAny>,
    pub(super) transfers: Py<PyTuple>,
}

#[pymethods]
impl InputCopies {
    fn __enter__(&self, py: Python<'_>) -> PyResult<()> {
        if self.owner.borrow(py).closed {
            return Err(PyRuntimeError::new_err("model runner is closed"));
        }

        let stream = self.stream.bind(py).getattr("stream")?;
        let scope = py.import("torch.cuda")?.call_method1("stream", (stream,))?;
        let result = with_context(&scope, || {
            let options = PyDict::new(py);
            options.set_item("non_blocking", true)?;
            for pair in self.transfers.bind(py) {
                let destination = pair.get_item(0)?;
                let source = pair.get_item(1)?;
                if !destination.getattr("shape")?.eq(source.getattr("shape")?)?
                    || !destination.getattr("dtype")?.eq(source.getattr("dtype")?)?
                {
                    return Err(PyValueError::new_err(
                        "prepared input must match destination shape and dtype",
                    ));
                }
                destination.call_method("copy_", (source,), Some(&options))?;
            }
            Ok(())
        });

        // Python does not call __exit__ when __enter__ raises. Copies already
        // submitted still have to precede consumers and request retirement.
        match result {
            Ok(()) => Ok(()),
            Err(error) => close_all(py, [Err(error), self.join(py)]),
        }
    }

    fn __exit__(
        &self,
        py: Python<'_>,
        _kind: &Bound<'_, PyAny>,
        _error: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.join(py)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.owner)?;
        visit.call(&self.stream)?;
        visit.call(&self.transfers)
    }
}

impl InputCopies {
    fn join(&self, py: Python<'_>) -> PyResult<()> {
        let stream = self.stream.bind(py);
        let current = py
            .import("torch.cuda")?
            .call_method1("current_stream", (stream.getattr("device")?,))?;
        let event = stream
            .getattr("_native")?
            .cast_into::<CUDAStream>()?
            .borrow()
            .record(py, current.getattr("cuda_stream")?.extract()?)?;
        if let Some(event) = event {
            event.wait(py, Some(&current))?;
        }
        Ok(())
    }
}
