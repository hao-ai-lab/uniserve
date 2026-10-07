//! Execution streams, numerical views and their native communication resources.

use std::sync::{Arc, Mutex, MutexGuard};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use uniserve_worker::CUDAStream as NativeStream;

use super::communication::StreamCommunication;
use super::events::CUDAEvent;
use super::execution::close_all;

#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct CUDAStream {
    inner: Arc<Mutex<NativeStream>>,
    stream: Py<PyAny>,
    communication: Mutex<Option<Py<StreamCommunication>>>,
}

#[pymethods]
impl CUDAStream {
    #[new]
    #[pyo3(signature = (device, handle, event_slots=2))]
    fn new(py: Python<'_>, device: i32, handle: usize, event_slots: usize) -> PyResult<Self> {
        Self::from_native(
            py,
            NativeStream::borrowed(device, handle, event_slots).map_err(error)?,
            None,
        )
    }

    /// Retain a caller-owned PyTorch stream and bind native execution resources.
    #[staticmethod]
    #[pyo3(signature = (stream, *, event_slots=2))]
    fn external(py: Python<'_>, stream: Py<PyAny>, event_slots: usize) -> PyResult<Self> {
        let device = stream
            .bind(py)
            .getattr("device")?
            .getattr("index")?
            .extract()?;
        let handle = stream.bind(py).getattr("cuda_stream")?.extract()?;
        Self::from_native(
            py,
            NativeStream::borrowed(device, handle, event_slots).map_err(error)?,
            Some(stream),
        )
    }

    #[staticmethod]
    #[pyo3(signature = (device, origin, event_slots=2))]
    pub(super) fn sibling(
        py: Python<'_>,
        device: i32,
        origin: usize,
        event_slots: usize,
    ) -> PyResult<Self> {
        Self::from_native(
            py,
            NativeStream::sibling(device, origin, event_slots).map_err(error)?,
            None,
        )
    }

    #[staticmethod]
    fn partition(
        py: Python<'_>,
        device: i32,
        counts: Vec<u32>,
        slots: Vec<usize>,
    ) -> PyResult<Vec<Self>> {
        NativeStream::partition(device, &counts, &slots)
            .map_err(error)?
            .into_iter()
            .map(|stream| Self::from_native(py, stream, None))
            .collect()
    }

    pub(super) fn fork(&self, py: Python<'_>) -> PyResult<Self> {
        let stream = self.lock(py)?.fork().map_err(error)?;
        Self::from_native(py, stream, None)
    }

    #[getter]
    fn handle(&self, py: Python<'_>) -> PyResult<usize> {
        self.lock(py)?.handle().map_err(error)
    }

    #[getter]
    fn device(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.stream.getattr(py, "device")
    }

    #[getter]
    pub(super) fn stream(&self, py: Python<'_>) -> Py<PyAny> {
        self.stream.clone_ref(py)
    }

    #[getter]
    fn communication(&self, py: Python<'_>) -> PyResult<Py<StreamCommunication>> {
        if let Some(owner) = self
            .communication
            .lock_py_attached(py)
            .map_err(|_| error("stream communication lock is poisoned"))?
            .as_ref()
        {
            return Ok(owner.clone_ref(py));
        }
        let owner = Py::new(
            py,
            StreamCommunication::new(py, self.stream.clone_ref(py), Arc::clone(&self.inner)),
        )?;
        *self
            .communication
            .lock_py_attached(py)
            .map_err(|_| error("stream communication lock is poisoned"))? =
            Some(owner.clone_ref(py));
        Ok(owner)
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

    fn wait(&self, py: Python<'_>, producer: &Bound<'_, PyAny>) -> PyResult<()> {
        self.wait_for(py, producer.getattr("cuda_stream")?.extract()?)
    }

    fn record(&self, py: Python<'_>) -> PyResult<Option<CUDAEvent>> {
        let current = py
            .import("torch.cuda")?
            .call_method1("current_stream", (self.device(py)?,))?;
        self.record_for(py, current.getattr("cuda_stream")?.extract()?)
    }

    fn synchronize(&self, py: Python<'_>) -> PyResult<()> {
        let stream = self.lock(py)?;
        let stream = &*stream;
        py.detach(|| stream.synchronize()).map_err(error)
    }

    /// Retire communicators and windows before draining the owning stream.
    #[pyo3(signature = (*, aborted=false))]
    #[allow(clippy::mem_forget)]
    pub(super) fn close(&self, py: Python<'_>, aborted: bool) -> PyResult<()> {
        if self.closed(py)? {
            return Ok(());
        }
        let communication = self
            .communication
            .lock_py_attached(py)
            .map_err(|_| error("stream communication lock is poisoned"))?
            .as_ref()
            .map(|owner| owner.clone_ref(py));
        let communications = match communication {
            Some(owner) => StreamCommunication::close(owner.bind(py), aborted),
            None => Ok(()),
        };
        let aborted = aborted || communications.is_err();
        if aborted {
            // External PyTorch streams also own a handle; keep that reference
            // when failed device work prevents normal retirement.
            std::mem::forget(self.stream.clone_ref(py));
        }
        let mut stream = self.lock(py)?;
        let stream = &mut *stream;
        let stream = py.detach(|| stream.close(aborted)).map_err(error);
        close_all(py, [communications, stream])
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        if slf.closed(slf.py())? {
            return Err(error("CUDA stream is closed"));
        }
        Ok(slf)
    }

    fn __exit__(
        &self,
        py: Python<'_>,
        _kind: &Bound<'_, PyAny>,
        failure: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let bound = self
            .communication
            .lock_py_attached(py)
            .map_err(|_| error("stream communication lock is poisoned"))?
            .as_ref()
            .is_some_and(|owner| owner.borrow(py).has_communicators(py));
        let aborted = !failure.is_none()
            && (bound || !self.stream.call_method0(py, "query")?.is_truthy(py)?);
        if let Err(cleanup) = self.close(py, aborted) {
            if failure.is_none() {
                return Err(cleanup);
            }
            let _ = failure.call_method1(
                "add_note",
                (format!("CUDA stream cleanup failed: {cleanup}"),),
            );
        }
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.stream)?;
        if let Ok(owner) = self.communication.try_lock() {
            visit.call(&*owner)?;
        }
        Ok(())
    }
}

impl CUDAStream {
    fn from_native(
        py: Python<'_>,
        native: NativeStream,
        view: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let stream = match view {
            Some(view) => view,
            None => py
                .import("torch.cuda")?
                .call_method1(
                    "ExternalStream",
                    (native.handle().map_err(error)?, native.device()),
                )?
                .unbind(),
        };
        Ok(Self {
            inner: Arc::new(Mutex::new(native)),
            stream,
            communication: Mutex::new(None),
        })
    }

    fn lock(&self, py: Python<'_>) -> PyResult<MutexGuard<'_, NativeStream>> {
        self.inner
            .lock_py_attached(py)
            .map_err(|_| error("CUDA stream lock is poisoned"))
    }

    pub(super) fn wait_for(&self, py: Python<'_>, producer: usize) -> PyResult<()> {
        self.lock(py)?.wait(producer).map_err(error)
    }

    pub(super) fn record_for(
        &self,
        py: Python<'_>,
        consumer: usize,
    ) -> PyResult<Option<CUDAEvent>> {
        self.lock(py)?
            .record(consumer)
            .map(|event| event.map(|inner| CUDAEvent { inner }))
            .map_err(error)
    }
}

impl Drop for CUDAStream {
    fn drop(&mut self) {
        Python::try_attach(|py| {
            // Implicit disposal must not enter collective retirement; peers may
            // have already left. Plain private streams still drain normally.
            let bound = self
                .communication
                .get_mut()
                .unwrap_or_else(|error| error.into_inner())
                .as_ref()
                .is_some_and(|owner| owner.borrow(py).has_communicators(py));
            if let Err(failure) = self.close(py, bound) {
                failure.write_unraisable(py, None);
            }
        });
    }
}

fn error(message: impl Into<String>) -> PyErr {
    Python::attach(|py| {
        match py
            .import("uniserve.runtime.cuda")?
            .getattr("CUDAError")?
            .call1((message.into(),))
        {
            Ok(value) => Ok(PyErr::from_value(value)),
            Err(error) => Err(error),
        }
    })
    .unwrap_or_else(|error| error)
}
