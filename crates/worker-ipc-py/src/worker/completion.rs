//! Python observers of native storage completion.

use std::ops::Deref;
use std::sync::Arc;
use std::time::Duration;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyBaseException, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyNone;
use uniserve_worker::{Completion as NativeCompletion, Outcome};

use super::error::native_error;

#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct Completion {
    inner: Arc<NativeCompletion<Py<PyBaseException>, Py<PyAny>>>,
}

#[pymethods]
impl Completion {
    #[new]
    pub(crate) fn new() -> Self {
        Self {
            inner: Arc::new(NativeCompletion::default()),
        }
    }

    pub(crate) fn done(&self) -> bool {
        self.inner.done()
    }

    pub(crate) fn succeeded(&self) -> bool {
        self.inner.succeeded()
    }

    fn cancelled(&self) -> bool {
        matches!(self.inner.outcome(), Some(Outcome::Cancelled))
    }

    fn cancel(slf: &Bound<'_, Self>) -> bool {
        let (cancelled, callbacks) = slf.borrow().inner.cancel();
        Self::notify(slf, callbacks);
        cancelled
    }

    #[pyo3(signature = (timeout=None))]
    pub(crate) fn result(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<()> {
        match self.wait(py, timeout)? {
            Outcome::Success => Ok(()),
            Outcome::Failed(error) => Err(PyErr::from_value(error.bind(py).clone().into_any())),
            Outcome::Cancelled => Err(cancelled(py)?),
        }
    }

    #[pyo3(signature = (timeout=None))]
    fn exception(
        &self,
        py: Python<'_>,
        timeout: Option<f64>,
    ) -> PyResult<Option<Py<PyBaseException>>> {
        match self.wait(py, timeout)? {
            Outcome::Success => Ok(None),
            Outcome::Failed(error) => Ok(Some(error.clone_ref(py))),
            Outcome::Cancelled => Err(cancelled(py)?),
        }
    }

    fn set_result(slf: &Bound<'_, Self>, _result: &Bound<'_, PyNone>) -> PyResult<()> {
        let callbacks = slf
            .borrow()
            .inner
            .complete(Ok(()))
            .map_err(|error| native_error(slf.py(), error))?;
        Self::notify(slf, callbacks);
        Ok(())
    }

    pub(crate) fn set_exception(slf: &Bound<'_, Self>, error: Py<PyBaseException>) -> PyResult<()> {
        let callbacks = slf
            .borrow()
            .inner
            .complete(Err(error))
            .map_err(|error| native_error(slf.py(), error))?;
        Self::notify(slf, callbacks);
        Ok(())
    }

    fn add_done_callback(slf: &Bound<'_, Self>, callback: Py<PyAny>) {
        let immediate = slf.borrow().inner.subscribe(callback);
        if let Some(callback) = immediate {
            Self::notify(slf, vec![callback]);
        }
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.inner.visit(|outcome, callbacks| {
            if let Some(Outcome::Failed(error)) = outcome {
                visit.call(error.as_ref())?;
            }
            for callback in callbacks {
                visit.call(callback)?;
            }
            Ok(())
        })
    }

    fn __clear__(&mut self) {
        self.inner = Arc::new(NativeCompletion::default());
    }
}

impl Completion {
    fn wait(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<Outcome<Py<PyBaseException>>> {
        // The common poll path neither releases the GIL nor enters a condvar.
        if let Some(outcome) = self.inner.outcome() {
            return Ok(outcome);
        }

        let timeout = timeout
            .map(|seconds| {
                Duration::try_from_secs_f64(seconds.max(0.0))
                    .map_err(|_| PyValueError::new_err("timeout must be finite"))
            })
            .transpose()?;
        py.detach(|| self.inner.wait(timeout))
            .ok_or_else(|| PyTimeoutError::new_err("operation has not completed"))
    }

    fn notify(slf: &Bound<'_, Self>, callbacks: Vec<Py<PyAny>>) {
        for callback in callbacks {
            if let Err(error) = callback.bind(slf.py()).call1((slf,)) {
                error.write_unraisable(slf.py(), Some(callback.bind(slf.py())));
            }
        }
    }
}

fn cancelled(py: Python<'_>) -> PyResult<PyErr> {
    Ok(PyErr::from_value(
        py.import("concurrent.futures")?
            .getattr("CancelledError")?
            .call0()?,
    ))
}

/// A native resource owner retains the Python wrapper for GC tracing while
/// reading completion state without acquiring the GIL or calling Python.
pub(crate) struct CompletionRef {
    pub(crate) owner: Py<Completion>,
    inner: Arc<NativeCompletion<Py<PyBaseException>, Py<PyAny>>>,
}

impl CompletionRef {
    pub(crate) fn new(py: Python<'_>, owner: Py<Completion>) -> Self {
        let inner = Arc::clone(&owner.borrow(py).inner);
        Self { owner, inner }
    }
}

impl Deref for CompletionRef {
    type Target = NativeCompletion<Py<PyBaseException>, Py<PyAny>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
