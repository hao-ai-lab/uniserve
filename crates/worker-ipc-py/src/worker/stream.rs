//! PyTorch stream handles for native execution stream ownership.

use std::sync::{Mutex, MutexGuard};

use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use uniserve_worker::CUDAStream as NativeStream;

use super::events::CUDAEvent;

#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct CUDAStream {
    inner: Mutex<NativeStream>,
}

#[pymethods]
impl CUDAStream {
    #[new]
    #[pyo3(signature = (device, handle, event_slots=2))]
    fn new(device: i32, handle: usize, event_slots: usize) -> PyResult<Self> {
        NativeStream::borrowed(device, handle, event_slots)
            .map(Self::from)
            .map_err(error)
    }

    #[staticmethod]
    #[pyo3(signature = (device, origin, event_slots=2))]
    fn sibling(device: i32, origin: usize, event_slots: usize) -> PyResult<Self> {
        NativeStream::sibling(device, origin, event_slots)
            .map(Self::from)
            .map_err(error)
    }

    #[staticmethod]
    fn partition(device: i32, counts: Vec<u32>, slots: Vec<usize>) -> PyResult<Vec<Self>> {
        NativeStream::partition(device, &counts, &slots)
            .map(|streams| streams.into_iter().map(Self::from).collect())
            .map_err(error)
    }

    fn fork(&self, py: Python<'_>) -> PyResult<Self> {
        self.lock(py)?.fork().map(Self::from).map_err(error)
    }

    #[getter]
    fn handle(&self, py: Python<'_>) -> PyResult<usize> {
        self.lock(py)?.handle().map_err(error)
    }

    #[getter]
    fn device(&self, py: Python<'_>) -> PyResult<i32> {
        Ok(self.lock(py)?.device())
    }

    #[getter]
    fn sm_count(&self, py: Python<'_>) -> PyResult<u32> {
        Ok(self.lock(py)?.sm_count())
    }

    #[getter]
    fn full_device(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(self.lock(py)?.full_device())
    }

    #[getter]
    fn closed(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(self.lock(py)?.closed())
    }

    pub(super) fn wait(&self, py: Python<'_>, producer: usize) -> PyResult<()> {
        self.lock(py)?.wait(producer).map_err(error)
    }

    pub(super) fn record(&self, py: Python<'_>, consumer: usize) -> PyResult<Option<CUDAEvent>> {
        self.lock(py)?
            .record(consumer)
            .map(|event| event.map(|inner| CUDAEvent { inner }))
            .map_err(error)
    }

    fn synchronize(&self, py: Python<'_>) -> PyResult<()> {
        let stream = self.lock(py)?;
        let stream = &*stream;
        py.detach(|| stream.synchronize()).map_err(error)
    }

    #[pyo3(signature = (*, aborted=false))]
    pub(super) fn close(&self, py: Python<'_>, aborted: bool) -> PyResult<()> {
        let mut stream = self.lock(py)?;
        let stream = &mut *stream;
        py.detach(|| stream.close(aborted)).map_err(error)
    }
}

impl From<NativeStream> for CUDAStream {
    fn from(stream: NativeStream) -> Self {
        Self {
            inner: Mutex::new(stream),
        }
    }
}

impl CUDAStream {
    fn lock(&self, py: Python<'_>) -> PyResult<MutexGuard<'_, NativeStream>> {
        self.inner
            .lock_py_attached(py)
            .map_err(|_| error("CUDA stream lock is poisoned"))
    }
}

impl Drop for CUDAStream {
    fn drop(&mut self) {
        let stream = self
            .inner
            .get_mut()
            .unwrap_or_else(|error| error.into_inner());
        // Foreign final-reference release must not hold the GIL while GPU work
        // is draining. Aborted owners have already relinquished their handles.
        Python::try_attach(|py| {
            if let Err(failure) = py.detach(|| stream.close(false)) {
                error(failure).write_unraisable(py, None);
            }
        });
    }
}

fn error(message: impl Into<String>) -> PyErr {
    Python::attach(|py| {
        match py
            .import("uniserve.runtime.cuda")
            .and_then(|module| module.getattr("CUDAError"))
            .and_then(|class| class.call1((message.into(),)))
        {
            Ok(value) => PyErr::from_value(value),
            Err(error) => error,
        }
    })
}
