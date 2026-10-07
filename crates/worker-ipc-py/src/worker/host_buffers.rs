//! Python tensor allocation for the native host input buffer ring.

use std::sync::{Mutex, MutexGuard, TryLockError};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use uniserve_worker::{Error, HostBuffers as NativeHostBuffers};

use super::events::current_stream;

#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct HostBuffers {
    inner: Mutex<NativeHostBuffers<Py<PyAny>>>,
    #[pyo3(get)]
    device: Py<PyAny>,
}

#[pymethods]
impl HostBuffers {
    #[new]
    #[pyo3(signature = (shape, *, dtype, depth, device))]
    pub(super) fn new(
        py: Python<'_>,
        shape: &Bound<'_, PyAny>,
        dtype: &Bound<'_, PyAny>,
        depth: isize,
        device: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let device = py
            .import("uniserve.runtime.device")?
            .call_method1("canonical_device", (device,))?;
        let cuda = device.getattr("type")?.extract::<String>()? == "cuda";
        let index = if cuda {
            Some(device.getattr("index")?.extract()?)
        } else {
            None
        };
        let buffers = py
            .import("uniserve_worker.storage.host_buffers")?
            .call_method1("_allocate", (shape, dtype, depth, cuda))?
            .extract()?;

        Ok(Self {
            inner: Mutex::new(NativeHostBuffers::new(buffers, index).map_err(error)?),
            device: device.unbind(),
        })
    }

    pub(super) fn acquire(&self, py: Python<'_>) -> PyResult<(usize, Py<PyAny>)> {
        let mut buffers = self.lock(py)?;
        let owner = &mut *buffers;
        let (slot, tensor) = py.detach(|| owner.acquire()).map_err(error)?;
        Ok((slot, tensor.clone_ref(py)))
    }

    pub(super) fn record_copy(&self, py: Python<'_>, slot: usize) -> PyResult<()> {
        if self.lock(py)?.device().is_none() {
            return Ok(());
        }

        let stream = current_stream(py, self.device.bind(py))?;
        self.lock(py)?.record_copy(slot, stream).map_err(error)
    }

    pub(super) fn close(&self, py: Python<'_>) -> PyResult<()> {
        let retired = {
            let mut buffers = self.lock(py)?;
            let owner = &mut *buffers;
            py.detach(|| owner.close()).map_err(error)?
        };
        drop(retired);
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.device)?;
        let buffers = match self.inner.try_lock() {
            Ok(buffers) => buffers,
            Err(TryLockError::Poisoned(error)) => error.into_inner(),
            Err(TryLockError::WouldBlock) => return Ok(()),
        };
        for tensor in buffers.buffers() {
            visit.call(tensor)?;
        }
        Ok(())
    }

    fn __clear__(&self, py: Python<'_>) {
        if let Err(error) = self.close(py) {
            error.write_unraisable(py, None);
        }
    }
}

impl HostBuffers {
    fn lock(&self, py: Python<'_>) -> PyResult<MutexGuard<'_, NativeHostBuffers<Py<PyAny>>>> {
        self.inner
            .lock_py_attached(py)
            .map_err(|_| PyRuntimeError::new_err("host buffer lock is poisoned"))
    }
}

impl Drop for HostBuffers {
    fn drop(&mut self) {
        let buffers = self
            .inner
            .get_mut()
            .unwrap_or_else(|error| error.into_inner());
        // The last owner may disappear with a copy in flight. Release the GIL
        // during that wait just as acquire and explicit close do.
        Python::try_attach(|py| {
            if let Err(failure) = py.detach(|| buffers.close()) {
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
