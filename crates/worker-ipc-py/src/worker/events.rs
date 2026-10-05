//! Numerical stream selection and Python owners of native CUDA event leases.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError, TryLockError};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use uniserve_worker::cuda::{DeviceGuard, Event};
use uniserve_worker::{Error, EventPool as NativeEventPool};

pyo3::create_exception!(
    uniserve_worker._uniserve_ipc,
    EventPoolError,
    PyRuntimeError
);

#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct CUDAEvent {
    pub(crate) inner: Arc<Event>,
}

#[pymethods]
impl CUDAEvent {
    fn query(&self) -> PyResult<bool> {
        self.inner.ready().map_err(PyRuntimeError::new_err)
    }

    fn synchronize(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.wait())
            .map_err(PyRuntimeError::new_err)
    }

    #[pyo3(signature = (stream=None))]
    pub(crate) fn wait(&self, py: Python<'_>, stream: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let current;
        let stream = match stream {
            Some(stream) => stream,
            None => {
                current = py.import("torch.cuda")?.call_method0("current_stream")?;
                &current
            }
        };
        let device = stream.getattr("device")?.getattr("index")?.extract()?;
        let handle = stream.getattr("cuda_stream")?.extract()?;
        let _device = DeviceGuard::new(device).map_err(PyRuntimeError::new_err)?;
        self.inner.wait_on(handle).map_err(PyRuntimeError::new_err)
    }

    fn elapsed_time(&self, end: &Self) -> PyResult<f32> {
        self.inner
            .elapsed_time(&end.inner)
            .map_err(PyRuntimeError::new_err)
    }

    fn ipc_handle<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.inner.ipc_handle().map_err(PyRuntimeError::new_err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    #[staticmethod]
    fn from_ipc_handle(py: Python<'_>, device: &Bound<'_, PyAny>, handle: &[u8]) -> PyResult<Self> {
        let (_, index) = device_index(py, device)?;
        let bytes = handle
            .try_into()
            .map_err(|_| PyValueError::new_err("CUDA IPC event handle must contain 64 bytes"))?;
        let event = Event::from_ipc_handle(index, bytes).map_err(PyRuntimeError::new_err)?;
        Ok(Self {
            inner: Arc::new(event),
        })
    }
}

pub(crate) struct DeferredOwner {
    value: Py<PyAny>,
    completed: Option<Py<PyAny>>,
}

#[derive(Default)]
struct PoolState {
    pool: NativeEventPool<DeferredOwner>,
    wake: Option<Py<PyAny>>,
}

#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct EventPool {
    state: Mutex<PoolState>,
}

#[pymethods]
impl EventPool {
    #[new]
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(PoolState::default()),
        }
    }

    fn set_completion_wake(&self, wake_on_stream: Option<Py<PyAny>>) {
        let previous = std::mem::replace(&mut self.lock().wake, wake_on_stream);
        drop(previous);
    }

    /// Queue the installed native wake after work already on this stream.
    fn notify_stream(&self, py: Python<'_>, stream: usize) -> PyResult<()> {
        let wake = self.lock().wake.as_ref().map(|wake| wake.clone_ref(py));
        if let Some(wake) = wake {
            wake.bind(py).call1((stream,))?;
        }
        Ok(())
    }

    pub(crate) fn schedule_completion_wake(
        &self,
        py: Python<'_>,
        device: &Bound<'_, PyAny>,
        event: &CUDAEvent,
    ) -> PyResult<()> {
        let (_, device) = device_index(py, device)?;
        self.wake_event(py, device, &event.inner)
    }

    #[pyo3(signature = (device, *, timing=false, interprocess=false))]
    pub(crate) fn acquire(
        &self,
        py: Python<'_>,
        device: &Bound<'_, PyAny>,
        timing: bool,
        interprocess: bool,
    ) -> PyResult<CUDAEvent> {
        self.reap(py)?;
        self.acquire_event(py, device, timing, interprocess)
    }

    pub(crate) fn declare_stream(
        &self,
        py: Python<'_>,
        event: &CUDAEvent,
        device: &Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        let (target, device) = device_index(py, device)?;
        let stream = current_stream(py, &target)?;
        self.lock()
            .pool
            .declare_stream(&event.inner, device, stream)
            .map_err(pool_error)
    }

    pub(crate) fn record(
        &self,
        py: Python<'_>,
        event: &CUDAEvent,
        device: &Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        let (target, device) = device_index(py, device)?;
        let stream = current_stream(py, &target)?;
        let _device = DeviceGuard::new(device).map_err(PyRuntimeError::new_err)?;
        self.lock()
            .pool
            .record(&event.inner, device, stream)
            .map_err(pool_error)
    }

    #[pyo3(signature = (event, device, count=1))]
    pub(crate) fn retain(
        &self,
        py: Python<'_>,
        event: &CUDAEvent,
        device: &Bound<'_, PyAny>,
        count: usize,
    ) -> PyResult<()> {
        let (_, device) = device_index(py, device)?;
        self.lock()
            .pool
            .retain(&event.inner, device, count)
            .map_err(pool_error)
    }

    #[pyo3(signature = (event, count=1))]
    pub(crate) fn release(&self, event: &CUDAEvent, count: usize) -> PyResult<()> {
        self.lock()
            .pool
            .release(&event.inner, count)
            .map_err(pool_error)
    }

    #[pyo3(signature = (events, owner, *, completed=None))]
    fn defer_release(
        &self,
        py: Python<'_>,
        events: Vec<Py<CUDAEvent>>,
        owner: Py<PyAny>,
        completed: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let events = events
            .iter()
            .map(|event| Arc::clone(&event.borrow(py).inner))
            .collect();
        self.defer_events(events, owner, completed)?;
        self.reap(py)
    }

    pub(crate) fn reap(&self, py: Python<'_>) -> PyResult<()> {
        let (owners, result) = self.lock().pool.reap();
        let callbacks = notify(py, owners);
        result.map_err(pool_error)?;
        callbacks
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let wait = self.lock().pool.begin_close();
        py.detach(wait).map_err(pool_error)?;
        let owners = self.lock().pool.finish_close();
        let wake = self.lock().wake.take();
        drop(wake);
        notify(py, owners)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        let state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::Poisoned(error)) => error.into_inner(),
            Err(TryLockError::WouldBlock) => return Ok(()),
        };
        if let Some(wake) = &state.wake {
            visit.call(wake)?;
        }
        for owner in state.pool.deferred_owners() {
            visit.call(&owner.value)?;
            if let Some(completed) = &owner.completed {
                visit.call(completed)?;
            }
        }
        Ok(())
    }

    fn __clear__(&mut self, py: Python<'_>) {
        let state = std::mem::take(self.state.get_mut().unwrap_or_else(PoisonError::into_inner));
        py.detach(move || drop(state));
    }
}

impl EventPool {
    /// Compose native resource operations without dispatching Python callbacks.
    /// Callers reap completed owners before taking their own ownership lock.
    pub(crate) fn with_pool<T>(
        &self,
        operation: impl FnOnce(&mut NativeEventPool<DeferredOwner>) -> uniserve_worker::Result<T>,
    ) -> PyResult<T> {
        operation(&mut self.lock().pool).map_err(pool_error)
    }

    pub(crate) fn wake_event(
        &self,
        py: Python<'_>,
        device: i32,
        event: &Arc<Event>,
    ) -> PyResult<()> {
        let wake = self.lock().wake.as_ref().map(|wake| wake.clone_ref(py));
        let Some(wake) = wake else { return Ok(()) };
        let stream = {
            let _device = DeviceGuard::new(device).map_err(PyRuntimeError::new_err)?;
            self.lock()
                .pool
                .wake_stream(event, device)
                .map_err(pool_error)?
        };
        wake.bind(py).call1((stream.handle(),))?;
        Ok(())
    }

    // Storage owners reap callbacks before taking their own lock, then use
    // these operations while composing native resource changes.
    pub(crate) fn acquire_event(
        &self,
        py: Python<'_>,
        device: &Bound<'_, PyAny>,
        timing: bool,
        interprocess: bool,
    ) -> PyResult<CUDAEvent> {
        let (_, device) = device_index(py, device)?;
        let inner = self
            .lock()
            .pool
            .acquire(device, timing, interprocess)
            .map_err(pool_error)?;
        Ok(CUDAEvent { inner })
    }

    pub(crate) fn defer_events(
        &self,
        events: Vec<Arc<Event>>,
        owner: Py<PyAny>,
        completed: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let unused = self
            .lock()
            .pool
            .defer_release(
                events,
                DeferredOwner {
                    value: owner,
                    completed,
                },
            )
            .map_err(pool_error)?;
        drop(unused);
        Ok(())
    }

    fn lock(&self) -> MutexGuard<'_, PoolState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

fn notify(py: Python<'_>, owners: Vec<DeferredOwner>) -> PyResult<()> {
    let mut failure = Ok(());
    for owner in owners {
        if let Some(callback) = &owner.completed
            && let Err(error) = callback.bind(py).call0()
        {
            if failure.is_ok() {
                failure = Err(error);
            } else {
                error.write_unraisable(py, Some(callback.bind(py)));
            }
        }
    }
    failure
}

fn pool_error(error: Error) -> PyErr {
    match error {
        Error::Cuda(message) => PyRuntimeError::new_err(message),
        error => EventPoolError::new_err(error.to_string()),
    }
}

fn device_index<'py>(
    py: Python<'py>,
    device: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, i32)> {
    let target = py
        .import("uniserve.runtime.device")?
        .getattr("canonical_device")?
        .call1((device,))?;
    if target.getattr("type")?.extract::<String>()? != "cuda" {
        return Err(EventPoolError::new_err(
            "device event requires a CUDA device",
        ));
    }
    let index = target.getattr("index")?.extract()?;
    Ok((target, index))
}

pub(super) fn current_stream(py: Python<'_>, device: &Bound<'_, PyAny>) -> PyResult<usize> {
    py.import("torch.cuda")?
        .call_method1("current_stream", (device,))?
        .getattr("cuda_stream")?
        .extract()
}
