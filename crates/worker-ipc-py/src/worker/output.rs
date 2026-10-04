//! PyTorch copies into native output leases; completion uses the shared event pool.

use std::sync::{Arc, Mutex, MutexGuard, TryLockError};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{PyDict, PyTuple};
use uniserve_worker::{OutputBuffer as NativeBuffer, OutputPool as NativePool, OutputStorage};

use super::completion::Completion;
use super::error::{invariant, native_error};
use super::events::EventPool;

type Buffer = Arc<Mutex<NativeBuffer<Py<PyAny>>>>;

#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct OutputBuffer {
    inner: Buffer,
    #[pyo3(get)]
    event_pool: Py<EventPool>,
    #[pyo3(get)]
    devices: Py<PyTuple>,
    completion: Py<Completion>,
    #[pyo3(get)]
    logprob_layouts: Py<PyDict>,
    tokens: Py<PyDict>,
    logprobs: Py<PyDict>,
}

#[pymethods]
impl OutputBuffer {
    #[getter]
    pub(crate) fn sealed(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(self.lock(py)?.sealed())
    }

    fn register_device(&self, py: Python<'_>, device: &Bound<'_, PyAny>) -> PyResult<()> {
        let (device, _) = stream(py, device)?;
        self.lock(py)?
            .register_device(device)
            .map_err(|error| native_error(py, error))
    }

    fn begin_device(&self, py: Python<'_>, device: &Bound<'_, PyAny>) -> PyResult<()> {
        let (device, stream) = stream(py, device)?;
        let mut buffer = self.lock(py)?;
        self.event_pool
            .borrow(py)
            .with_pool(|pool| Ok(buffer.begin_device(device, stream, pool)))?
            .map_err(|error| native_error(py, error))
    }

    fn capture(&self, py: Python<'_>, tokens: &Bound<'_, PyAny>) -> PyResult<(usize, usize)> {
        let numerical = py.import("uniserve_worker.storage.output")?;
        let value = numerical.call_method1("_tokens", (tokens,))?;
        let count = value.call_method0("numel")?.extract()?;
        let (device, stream) = stream(py, &value.getattr("device")?)?;

        let mut buffer = self.lock(py)?;
        self.event_pool
            .borrow(py)
            .with_pool(|pool| Ok(buffer.begin_copy(device, stream, pool)))?
            .map_err(|error| native_error(py, error))?;
        let offset = buffer
            .reserve_tokens(count)
            .map_err(|error| native_error(py, error))?;
        let storage = buffer.storage().map_err(|error| native_error(py, error))?;
        numerical.call_method1("_copy_tokens", (storage.value.bind(py), offset, value))?;
        Ok((offset, count))
    }

    fn capture_bytes(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let numerical = py.import("uniserve_worker.storage.output")?;
        let value = numerical.call_method1("_bytes", (value,))?;
        let count = value.call_method0("numel")?.extract()?;
        let (device, stream) = stream(py, &value.getattr("device")?)?;

        let mut buffer = self.lock(py)?;
        self.event_pool
            .borrow(py)
            .with_pool(|pool| Ok(buffer.begin_copy(device, stream, pool)))?
            .map_err(|error| native_error(py, error))?;
        let offset = buffer
            .reserve_bytes(count)
            .map_err(|error| native_error(py, error))?;
        let storage = buffer.storage().map_err(|error| native_error(py, error))?;
        Ok(numerical
            .call_method1("_copy_bytes", (storage.value.bind(py), offset, value))?
            .unbind())
    }

    pub(super) fn seal(slf: &Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        let this = slf.get();
        let streams: Vec<_> = this
            .devices
            .bind(py)
            .iter()
            .map(|device| {
                let index = device.getattr("index")?.extract()?;
                stream(py, &device).map(|(_, stream)| (index, stream))
            })
            .collect::<PyResult<_>>()?;
        let events = {
            let mut buffer = this.lock(py)?;
            this.event_pool
                .borrow(py)
                .with_pool(|pool| Ok(buffer.seal(&streams, pool)))?
                .map_err(|error| native_error(py, error))?
        };
        let Some(events) = events else { return Ok(()) };
        if events.is_empty() {
            return this._copies_finished(py);
        }

        // Deferred release owns the lease before a wake can invoke observers.
        // The existing end fences notify storage consumers as well as result rows.
        let pool = this.event_pool.borrow(py);
        pool.defer_events(
            events,
            slf.clone().into_any().unbind(),
            Some(slf.getattr("_copies_finished")?.unbind()),
        )?;
        let fences: Vec<_> = this.lock(py)?.fences().cloned().collect();
        for event in fences {
            pool.wake_event(py, event.device(), &event)?;
        }
        pool.reap(py)
    }

    pub(super) fn ready(&self, py: Python<'_>) -> PyResult<bool> {
        let ready = self.query_ready(py)?;
        if ready {
            Completion::resolve(self.completion.bind(py))?;
        }
        Ok(ready)
    }

    pub(crate) fn completion(&self, py: Python<'_>) -> PyResult<Py<Completion>> {
        self.ready(py)?;
        Ok(self.completion.clone_ref(py))
    }

    pub(super) fn read_tokens(
        &self,
        py: Python<'_>,
        offset: usize,
        count: usize,
    ) -> PyResult<Py<PyAny>> {
        if let Some(cached) = self.tokens.bind(py).get_item((offset, count))? {
            return Ok(cached.unbind());
        }

        if !self.ready(py)? {
            return Err(invariant(
                py,
                "completion storage was observed before its copy event was ready",
            ));
        }
        let buffer = self.lock(py)?;
        buffer
            .token_range(offset, count)
            .map_err(|error| native_error(py, error))?;
        let host = &buffer
            .storage()
            .map_err(|error| native_error(py, error))?
            .value;
        let values = py
            .import("uniserve_worker.storage.output")?
            .call_method1("_read_tokens", (host.bind(py), offset, count))?;
        self.tokens.bind(py).set_item((offset, count), &values)?;
        Ok(values.unbind())
    }

    pub(super) fn logprob_values(
        &self,
        py: Python<'_>,
        span: (usize, usize, usize),
    ) -> PyResult<Py<PyAny>> {
        let (offset, count, row) = span;
        let key = (offset, count);
        let details = match self.logprobs.bind(py).get_item(key)? {
            Some(details) => details,
            None => {
                let values = self.read_tokens(py, offset, count)?;
                let layout = self
                    .logprob_layouts
                    .bind(py)
                    .get_item(key)?
                    .ok_or_else(|| invariant(py, "logprob capture has no row layout"))?;
                let details = py
                    .import("uniserve_worker.sampling.output")?
                    .call_method1("decode_logprobs", (values, layout))?;
                self.logprobs.bind(py).set_item(key, &details)?;
                details
            }
        };
        Ok(details.get_item(row)?.unbind())
    }

    pub(super) fn observe(&self, py: Python<'_>, row: usize) -> PyResult<(u64, u64)> {
        self.ready(py)?;
        let timing = self
            .lock(py)?
            .observe(row)
            .map_err(|error| native_error(py, error))?;
        Ok((timing[2], timing[3]))
    }

    pub(super) fn timing(&self, py: Python<'_>) -> PyResult<(u64, u64, u64, u64)> {
        let [queued, device, copy, host] = self
            .lock(py)?
            .timing()
            .map_err(|error| native_error(py, error))?;
        Ok((queued, device, copy, host))
    }

    pub(super) fn discard(&self, py: Python<'_>, row: usize) -> PyResult<()> {
        self.lock(py)?.discard(row);
        Ok(())
    }

    pub(crate) fn abandon(slf: &Bound<'_, Self>) -> PyResult<()> {
        Self::seal(slf)?;
        slf.get().lock(slf.py())?.abandon();
        Ok(())
    }

    fn retain_cpu_reader(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        slf.get()
            .lock(slf.py())?
            .retain_reader()
            .map_err(|error| native_error(slf.py(), error))?;
        Ok(slf.getattr("_release_cpu_reader")?.unbind())
    }

    fn _release_cpu_reader(&self, py: Python<'_>) -> PyResult<()> {
        self.lock(py)?
            .release_reader()
            .map_err(|error| native_error(py, error))
    }

    fn _copies_finished(&self, py: Python<'_>) -> PyResult<()> {
        self.lock(py)?.copies_finished();
        Completion::resolve(self.completion.bind(py))
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.event_pool)?;
        visit.call(&self.devices)?;
        visit.call(&self.completion)?;
        visit.call(&self.logprob_layouts)?;
        visit.call(&self.tokens)?;
        visit.call(&self.logprobs)?;
        let buffer = match self.inner.try_lock() {
            Ok(buffer) => buffer,
            Err(TryLockError::Poisoned(error)) => error.into_inner(),
            Err(TryLockError::WouldBlock) => return Ok(()),
        };
        if let Ok(storage) = buffer.storage() {
            visit.call(&storage.value)?;
        }
        Ok(())
    }
}

impl OutputBuffer {
    /// Observe copy fences without dispatching completion callbacks. Batch
    /// readiness borrows its input set; the event pool notifies its observers.
    pub(crate) fn query_ready(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(self
            .lock(py)?
            .ready_at()
            .map_err(|error| native_error(py, error))?
            .is_some())
    }

    fn lock(&self, py: Python<'_>) -> PyResult<MutexGuard<'_, NativeBuffer<Py<PyAny>>>> {
        self.inner
            .lock_py_attached(py)
            .map_err(|_| invariant(py, "output buffer lock poisoned"))
    }
}

#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct OutputPool {
    state: Mutex<NativePool<Py<PyAny>, Py<OutputBuffer>>>,
    #[pyo3(get)]
    event_pool: Py<EventPool>,
}

#[pymethods]
impl OutputPool {
    #[new]
    #[pyo3(signature = (*, capacity, max_words, event_pool))]
    fn new(capacity: usize, max_words: usize, event_pool: Py<EventPool>) -> PyResult<Self> {
        if capacity == 0 || max_words == 0 {
            return Err(PyValueError::new_err("output-pool bounds must be positive"));
        }
        Ok(Self {
            state: Mutex::new(NativePool::new(capacity, max_words)),
            event_pool,
        })
    }

    #[getter]
    fn capacity(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.lock(py)?.capacity())
    }

    #[getter]
    fn max_words(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.lock(py)?.max_words())
    }

    #[pyo3(signature = (rows, *, token_capacity, devices=Vec::new()))]
    fn acquire(
        &self,
        py: Python<'_>,
        rows: usize,
        token_capacity: usize,
        devices: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Py<OutputBuffer>> {
        if rows == 0 {
            return Err(PyValueError::new_err(
                "an output buffer must contain at least one call row",
            ));
        }

        self.event_pool.borrow(py).reap(py)?;

        // CPU readers can finish on a host lane while admission reclaims other
        // leases. Neither the GIL nor Python destruction belongs inside that wait.
        let owners = py.detach(|| {
            self.state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .reap()
        });
        drop(owners);

        let canonical = py
            .import("uniserve.runtime.device")?
            .getattr("canonical_device")?;
        let mut normalized = Vec::new();
        let mut indexes = Vec::new();
        for device in devices {
            let device = canonical.call1((device,))?;
            if device.getattr("type")?.extract::<String>()? != "cuda" {
                continue;
            }
            let index = device.getattr("index")?.extract::<i32>()?;
            if !indexes.contains(&index) {
                indexes.push(index);
                normalized.push(device);
            }
        }
        let timing = py
            .import("uniserve_worker.profiling")?
            .call_method0("timing_events_enabled")?
            .extract()?;

        let mut pool = self.lock(py)?;
        let storage = pool
            .take(rows, token_capacity)
            .map_err(|error| native_error(py, error))?;
        let pinned = !indexes.is_empty();
        let storage = match storage {
            Some(storage) if storage.words >= token_capacity && (!pinned || storage.pinned) => {
                storage
            }
            _ => OutputStorage {
                value: py
                    .import("uniserve_worker.storage.output")?
                    .call_method1("_allocate", (token_capacity, pinned))?
                    .unbind(),
                words: token_capacity,
                pinned,
            },
        };

        let inner = Arc::new(Mutex::new(NativeBuffer::new(
            storage, rows, &indexes, timing,
        )));
        let buffer = Py::new(
            py,
            OutputBuffer {
                inner: Arc::clone(&inner),
                event_pool: self.event_pool.clone_ref(py),
                devices: PyTuple::new(py, normalized)?.unbind(),
                completion: Py::new(py, Completion::new())?,
                logprob_layouts: PyDict::new(py).unbind(),
                tokens: PyDict::new(py).unbind(),
                logprobs: PyDict::new(py).unbind(),
            },
        )?;
        pool.insert(inner, buffer.clone_ref(py));
        Ok(buffer)
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let buffers: Vec<_> = {
            let mut pool = self.lock(py)?;
            pool.begin_close();
            pool.owners().map(|owner| owner.clone_ref(py)).collect()
        };

        let mut failure = Ok(());
        for buffer in buffers {
            let result = (|| {
                let buffer = buffer.bind(py);
                OutputBuffer::seal(buffer)?;
                let events = buffer.get().lock(py)?.events();
                py.detach(|| events.iter().try_for_each(|event| event.wait()))
                    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
                OutputBuffer::abandon(buffer)
            })();
            if result.is_err() && failure.is_ok() {
                failure = result;
            }
        }

        let reaped = self.event_pool.borrow(py).reap(py);
        failure.and(reaped)?;
        let retired = self.lock(py)?.finish_close();
        drop(retired);
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.event_pool)?;
        let pool = match self.state.try_lock() {
            Ok(pool) => pool,
            Err(TryLockError::Poisoned(error)) => error.into_inner(),
            Err(TryLockError::WouldBlock) => return Ok(()),
        };
        for owner in pool.owners() {
            visit.call(owner)?;
        }
        for storage in pool.storages() {
            visit.call(&storage.value)?;
        }
        Ok(())
    }

    fn __clear__(&self, py: Python<'_>) {
        if let Err(error) = self.close(py) {
            error.write_unraisable(py, None);
        }
    }
}

impl OutputPool {
    fn lock(
        &self,
        py: Python<'_>,
    ) -> PyResult<MutexGuard<'_, NativePool<Py<PyAny>, Py<OutputBuffer>>>> {
        self.state
            .lock_py_attached(py)
            .map_err(|_| invariant(py, "output pool lock poisoned"))
    }
}

fn stream(py: Python<'_>, device: &Bound<'_, PyAny>) -> PyResult<(Option<i32>, usize)> {
    let device = py
        .import("uniserve.runtime.device")?
        .call_method1("canonical_device", (device,))?;
    if device.getattr("type")?.extract::<String>()? != "cuda" {
        return Ok((None, 0));
    }

    let index = device.getattr("index")?.extract()?;
    let stream = py
        .import("torch.cuda")?
        .call_method1("current_stream", (device,))?
        .getattr("cuda_stream")?
        .extract()?;
    Ok((Some(index), stream))
}
