//! Registered transport buffers and native physical reclamation.

mod buffer;

use buffer::Retirement;
pub(crate) use buffer::TransportBuffer;

use std::sync::{Arc, Mutex, MutexGuard, PoisonError, TryLockError};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use uniserve_worker::{BufferRegistry as NativeBufferRegistry, HostLane, RegisteredBuffer};

use super::completion::{Completion, CompletionRef};
use super::error::{invalid, invariant, native_error};
use super::events::EventPool;

#[derive(Clone, Eq, Hash, PartialEq)]
enum BufferKey {
    Local(u64),
    Shm(String),
    Cuda(String),
}

struct Source {
    locator: Py<PyAny>,
    value: Py<TransportBuffer>,
}

type Registry = NativeBufferRegistry<BufferKey, Source, CompletionRef>;

/// Retain source storage until its producer and all granted readers finish.
/// Reclamation and retirement observers run outside the registry lock.
#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct BufferRegistry {
    #[pyo3(get)]
    name: String,
    events: Py<EventPool>,
    retirements: Mutex<Option<Arc<HostLane<Retirement>>>>,
    wake: Mutex<Option<Py<PyAny>>>,
    state: Mutex<Registry>,
}

#[pymethods]
impl BufferRegistry {
    #[new]
    #[pyo3(signature = (*, capacity, event_pool))]
    fn new(py: Python<'_>, capacity: usize, event_pool: Py<EventPool>) -> PyResult<Self> {
        let state =
            Registry::new(capacity).map_err(|error| PyValueError::new_err(error.to_string()))?;
        let suffix: String = py
            .import("uuid")?
            .call_method0("uuid4")?
            .getattr("hex")?
            .extract()?;

        Ok(Self {
            name: format!("uniserve-buffers-{suffix}"),
            events: event_pool,
            retirements: Mutex::new(None),
            wake: Mutex::new(None),
            state: Mutex::new(state),
        })
    }

    /// Register backing storage; the backend observes producer completion.
    fn register(
        &self,
        py: Python<'_>,
        locator: Py<PyAny>,
        source: Py<TransportBuffer>,
    ) -> PyResult<()> {
        let key = self.key(locator.bind(py))?;
        self.remove_finished(py)?;
        let retirement = Py::new(py, Completion::new())?;
        self.lock(py)?
            .register(
                key,
                Source {
                    locator,
                    value: source,
                },
                CompletionRef::new(py, retirement),
            )
            .map_err(|error| native_error(py, error))
    }

    /// Grant a local read. Its caller must release it after physical access
    /// ends, even if the buffer was revoked while the read was in flight.
    fn acquire(&self, py: Python<'_>, locator: &Bound<'_, PyAny>) -> PyResult<Py<TransportBuffer>> {
        let key = self.key(locator)?;
        let mut state = self.lock(py)?;
        Ok(Self::buffer(&mut state, &key, locator)?
            .acquire()
            .map_err(|error| native_error(py, error))?
            .value
            .clone_ref(py))
    }

    fn release_reader(&self, py: Python<'_>, locator: &Bound<'_, PyAny>) -> PyResult<()> {
        let key = self.key(locator)?;
        let mut state = self.lock(py)?;
        Self::buffer(&mut state, &key, locator)?
            .release_reader()
            .map_err(|error| native_error(py, error))?;
        drop(state);
        self.reclaim_buffer(py, &key)
    }

    /// Revoke future reads; the returned signal completes after physical
    /// retirement. A buffer already removed from the registry returns None.
    fn release(
        &self,
        py: Python<'_>,
        locator: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Py<Completion>>> {
        let key = self.key(locator)?;
        let mut state = self.lock(py)?;
        if state.get_mut(&key).is_none() {
            return Ok(None);
        }
        let buffer = Self::buffer(&mut state, &key, locator)?;
        buffer.release();
        let retirement = buffer.retirement().owner.clone_ref(py);
        drop(state);
        self.reclaim_buffer(py, &key)?;
        Ok(Some(retirement))
    }

    fn retirement(&self, py: Python<'_>, locator: &Bound<'_, PyAny>) -> PyResult<Py<Completion>> {
        let key = self.key(locator)?;
        let mut state = self.lock(py)?;
        Ok(Self::buffer(&mut state, &key, locator)?
            .retirement()
            .owner
            .clone_ref(py))
    }

    fn awaiting_acknowledgment(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(self.lock(py)?.awaiting_acknowledgment())
    }

    /// Sweep backend acknowledgments and discard successfully retired storage.
    fn reap(&self, py: Python<'_>) -> PyResult<()> {
        let keys: Vec<_> = self.lock(py)?.revoked().cloned().collect();
        for key in keys {
            self.reclaim_buffer(py, &key)?;
        }
        self.remove_finished(py)
    }

    fn set_completion_wake(&self, py: Python<'_>, wake: Option<Py<PyAny>>) {
        let previous = std::mem::replace(
            &mut *self.wake.lock().unwrap_or_else(PoisonError::into_inner),
            wake.as_ref().map(|wake| wake.clone_ref(py)),
        );
        drop(previous);

        let lane = self
            .retirements
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        if let Some(lane) = lane {
            lane.set_wake(wake);
        }
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        self.lock(py)?.close();
        let mut result = self.reap(py);
        let sources: Vec<_> = self
            .lock(py)?
            .buffers()
            .map(|buffer| buffer.source().value.clone_ref(py))
            .collect();
        for source in sources {
            result = result.and(source.get().drain(py, &self.events.borrow(py)));
        }
        result = result.and(self.reap(py));

        // Joining unregistration must not hold a registry or lane lock: its
        // completion observers may inspect returned storage or close peers.
        let lane = self
            .retirements
            .lock_py_attached(py)
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(lane) = lane {
            for error in py.detach(|| lane.close()) {
                result = result.and(Err(PyRuntimeError::new_err(error)));
            }
        }
        result = result.and(self.reap(py));
        result.and(
            self.lock(py)?
                .require_retired()
                .map_err(|error| native_error(py, error)),
        )
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.events)?;
        if let Ok(wake) = self.wake.try_lock() {
            visit.call(&*wake)?;
        }
        let state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::Poisoned(error)) => error.into_inner(),
            Err(TryLockError::WouldBlock) => return Ok(()),
        };
        for buffer in state.buffers() {
            visit.call(&buffer.source().locator)?;
            visit.call(&buffer.source().value)?;
            visit.call(&buffer.retirement().owner)?;
        }
        Ok(())
    }
}

impl BufferRegistry {
    fn lock(&self, py: Python<'_>) -> PyResult<MutexGuard<'_, Registry>> {
        self.state
            .lock_py_attached(py)
            .map_err(|_| invariant(py, "buffer registry lock is poisoned"))
    }

    fn key(&self, locator: &Bound<'_, PyAny>) -> PyResult<BufferKey> {
        let py = locator.py();
        let handle = locator.getattr("transport")?;
        if handle.getattr("endpoint")?.extract::<String>()? != self.name {
            return Err(invalid(py, "buffer belongs to another endpoint"));
        }
        match locator.getattr("backend")?.extract::<String>()?.as_str() {
            "local" => Ok(BufferKey::Local(handle.getattr("key")?.extract()?)),
            "shm" => Ok(BufferKey::Shm(handle.getattr("name")?.extract()?)),
            "cuda_vmm" => Ok(BufferKey::Cuda(handle.getattr("export_id")?.extract()?)),
            _ => Err(invalid(py, "transport does not register source storage")),
        }
    }

    fn buffer<'a>(
        state: &'a mut Registry,
        key: &BufferKey,
        locator: &Bound<'_, PyAny>,
    ) -> PyResult<&'a mut RegisteredBuffer<Source, CompletionRef>> {
        let buffer = state
            .get_mut(key)
            .ok_or_else(|| invalid(locator.py(), "buffer is no longer registered"))?;
        // The immutable locator describes the actual view, including its fence.
        // Comparing it directly avoids serializing and hashing tensor metadata.
        if !buffer.source().locator.bind(locator.py()).eq(locator)? {
            return Err(invalid(
                locator.py(),
                "buffer locator changed its registered view",
            ));
        }
        Ok(buffer)
    }

    fn reclaim_buffer(&self, py: Python<'_>, key: &BufferKey) -> PyResult<()> {
        let mut state = self.lock(py)?;
        let Some(buffer) = state.get_mut(key) else {
            return Ok(());
        };
        let Some((source, retirement)) =
            buffer.begin_reclaim(|source| Ok::<_, PyErr>(source.value.get().settled(py)))?
        else {
            return Ok(());
        };
        let source = source.value.clone_ref(py);
        let retirement = retirement.owner.clone_ref(py);
        drop(state);

        let action = Retirement {
            buffer: source.clone_ref(py),
            completion: Some(CompletionRef::new(py, retirement.clone_ref(py))),
        };
        let result = if source.get().asynchronous(py) {
            self.retire_shared(py, action)
        } else {
            TransportBuffer::reclaim(
                py,
                action.buffer,
                &self.events.borrow(py),
                action.completion,
            )
        };
        if let Err(error) = result {
            if !retirement.borrow(py).done() {
                Completion::set_exception(retirement.bind(py), error.value(py).clone().unbind())?;
            }
            return Err(error);
        }
        Ok(())
    }

    fn retire_shared(&self, py: Python<'_>, action: Retirement) -> PyResult<()> {
        let lane = {
            let mut installed = self
                .retirements
                .lock_py_attached(py)
                .unwrap_or_else(PoisonError::into_inner);
            if let Some(lane) = &*installed {
                Arc::clone(lane)
            } else {
                let lane = HostLane::new(256, 1, "uniserve-shm-retire")
                    .map_err(|error| native_error(py, error))?;
                lane.set_wake(
                    self.wake
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .as_ref()
                        .map(|wake| wake.clone_ref(py)),
                );
                let lane = Arc::new(lane);
                *installed = Some(Arc::clone(&lane));
                lane
            }
        };
        let task = lane.reserve().map_err(|error| native_error(py, error))?;
        let submitted = task.configure(action).and_then(|()| task.submit());
        if let Err(error) = submitted {
            if let Err(cleanup) = task.cancel(true) {
                PyRuntimeError::new_err(cleanup).write_unraisable(py, None);
            }
            return Err(native_error(py, error));
        }
        Ok(())
    }

    fn remove_finished(&self, py: Python<'_>) -> PyResult<()> {
        let retired = self.lock(py)?.take_finished();
        drop(retired);
        Ok(())
    }
}
