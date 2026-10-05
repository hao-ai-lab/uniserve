//! Numerical weight views and graph recorders for native peer prefetch.

use std::sync::{Mutex, PoisonError};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use uniserve_worker::{Error, WeightPrefetch as NativePrefetch};

type PeerCopy<'py> = (usize, Bound<'py, PyAny>, Bound<'py, PyAny>);

#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct WeightPrefetch {
    inner: NativePrefetch<Py<PyTuple>>,
    capture: Mutex<Option<Py<PyAny>>>,
}

#[pymethods]
impl WeightPrefetch {
    #[new]
    fn new(
        py: Python<'_>,
        device: i32,
        layers: Vec<usize>,
        copies: Vec<Vec<PeerCopy<'_>>>,
        backing: Vec<Py<PyAny>>,
    ) -> PyResult<Self> {
        let copies = copies
            .into_iter()
            .map(|layer| {
                layer
                    .into_iter()
                    .map(|(peer, destination, source)| {
                        let bytes = destination.call_method0("numel")?.extract::<usize>()?
                            * destination
                                .call_method0("element_size")?
                                .extract::<usize>()?;
                        Ok((
                            peer,
                            destination.call_method0("data_ptr")?.extract()?,
                            source.call_method0("data_ptr")?.extract()?,
                            bytes,
                        ))
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<Vec<_>>>()?;

        let backing = PyTuple::new(py, backing)?.unbind();
        Ok(Self {
            inner: NativePrefetch::new(device, layers, copies, backing).map_err(error)?,
            capture: Mutex::new(None),
        })
    }

    fn begin(&self) -> PyResult<()> {
        self.inner.begin().map_err(error)
    }

    fn end(&self, py: Python<'_>, stream: usize) -> PyResult<()> {
        py.detach(|| self.inner.end(stream)).map_err(error)
    }

    fn contains(&self, module: &Bound<'_, PyAny>) -> bool {
        self.inner.contains(module.as_ptr() as usize)
    }

    fn before(slf: Bound<'_, Self>, module: &Bound<'_, PyAny>, stream: usize) -> PyResult<()> {
        let py = slf.py();
        let owner = slf.borrow();
        let module = module.as_ptr() as usize;
        let capture = owner
            .capture
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .map(|capture| capture.clone_ref(py));

        if let Some(capture) = capture {
            let partial = py.import("functools")?.getattr("partial")?;
            let copy = slf.getattr("copy")?;
            owner
                .inner
                .before(module, stream, |index| {
                    let call = partial.call1((&copy, index))?;
                    capture.bind(py).call1((call,))?;
                    Ok(())
                })
                .map_err(error)?
        } else {
            let inner = &owner.inner;
            py.detach(|| inner.before(module, stream, |index| inner.copy(index)))
                .map_err(error)?
                .map_err(error)
        }
    }

    fn after(&self, py: Python<'_>, module: &Bound<'_, PyAny>, stream: usize) -> PyResult<()> {
        let module = module.as_ptr() as usize;
        py.detach(|| self.inner.after(module, stream))
            .map_err(error)
    }

    fn copy(&self, py: Python<'_>, index: usize) -> PyResult<()> {
        py.detach(|| self.inner.copy(index)).map_err(error)
    }

    fn set_capture(&self, capture: Option<Py<PyAny>>) -> Option<Py<PyAny>> {
        std::mem::replace(
            &mut *self.capture.lock().unwrap_or_else(PoisonError::into_inner),
            capture,
        )
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let retired = py.detach(|| self.inner.close()).map_err(error)?;
        let capture = self.set_capture(None);
        drop((retired, capture));
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.inner
            .visit(|backing| visit.call(backing))
            .unwrap_or(Ok(()))?;
        if let Some(capture) = &*self.capture.lock().unwrap_or_else(PoisonError::into_inner) {
            visit.call(capture)?;
        }
        Ok(())
    }

    fn __clear__(&self, py: Python<'_>) {
        if let Err(error) = self.close(py) {
            error.write_unraisable(py, None);
        }
    }
}

impl Drop for WeightPrefetch {
    fn drop(&mut self) {
        Python::try_attach(|py| {
            if let Err(failure) = py.detach(|| self.inner.close()) {
                error(failure).write_unraisable(py, None);
            }
        });
    }
}

fn error(error: Error) -> PyErr {
    match error {
        Error::Invalid(message) => PyValueError::new_err(message),
        error => PyRuntimeError::new_err(error.to_string()),
    }
}
