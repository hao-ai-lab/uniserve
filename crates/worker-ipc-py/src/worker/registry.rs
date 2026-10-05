//! Numerical sources and callbacks for the native buffer registry.

use std::sync::{Mutex, MutexGuard, TryLockError};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyBaseException, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use uniserve_worker::{BufferRegistry as NativeBufferRegistry, RegisteredBuffer};

use super::completion::{Completion, CompletionRef};
use super::error::{invalid, invariant, native_error};

#[derive(Clone, Eq, Hash, PartialEq)]
enum BufferKey {
    Local(u64),
    Shm(String),
    Cuda(String),
}

struct Source {
    locator: Py<PyAny>,
    value: Py<PyAny>,
}

type Registry = NativeBufferRegistry<BufferKey, Source, CompletionRef>;

/// Retain source storage until its producer and all granted readers finish.
/// Reclaim callbacks and retirement observers run outside the registry lock.
#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct BufferRegistry {
    #[pyo3(get)]
    name: String,
    reclaim: Py<PyAny>,
    drain: Py<PyAny>,
    settled: Py<PyAny>,
    state: Mutex<Registry>,
}

#[pymethods]
impl BufferRegistry {
    #[new]
    #[pyo3(signature = (*, capacity, reclaim, drain, settled))]
    fn new(
        py: Python<'_>,
        capacity: usize,
        reclaim: Py<PyAny>,
        drain: Py<PyAny>,
        settled: Py<PyAny>,
    ) -> PyResult<Self> {
        let state =
            Registry::new(capacity).map_err(|error| PyValueError::new_err(error.to_string()))?;
        let suffix: String = py
            .import("uuid")?
            .call_method0("uuid4")?
            .getattr("hex")?
            .extract()?;

        Ok(Self {
            name: format!("uniserve-buffers-{suffix}"),
            reclaim,
            drain,
            settled,
            state: Mutex::new(state),
        })
    }

    /// Register backing storage. A pending producer must call `complete`
    /// before the buffer can retire, including after a failed write.
    #[pyo3(signature = (locator, source, *, pending=false))]
    fn register(
        &self,
        py: Python<'_>,
        locator: Py<PyAny>,
        source: Py<PyAny>,
        pending: bool,
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
                pending,
            )
            .map_err(|error| native_error(py, error))
    }

    /// Inspect owner-held storage, including after reads have been revoked.
    fn source(&self, py: Python<'_>, locator: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let key = self.key(locator)?;
        let mut state = self.lock(py)?;
        Ok(Self::buffer(&mut state, &key, locator)?
            .source()
            .value
            .clone_ref(py))
    }

    /// Grant a local read. Its caller must release it after physical access
    /// ends, even if the buffer was revoked while the read was in flight.
    fn acquire(&self, py: Python<'_>, locator: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
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

    /// A failed producer with unknown completion retains its storage. Reporting
    /// the error must not release bytes that a device may still be writing.
    #[pyo3(signature = (locator, *, error=None, producer_completed=true))]
    fn complete(
        &self,
        py: Python<'_>,
        locator: &Bound<'_, PyAny>,
        error: Option<Py<PyBaseException>>,
        producer_completed: bool,
    ) -> PyResult<()> {
        let key = self.key(locator)?;
        let mut state = self.lock(py)?;
        let buffer = Self::buffer(&mut state, &key, locator)?;
        let callbacks = buffer
            .complete(error, producer_completed)
            .map_err(|error| native_error(py, error))?;
        let retirement = buffer.retirement().owner.clone_ref(py);
        drop(state);

        Completion::notify(retirement.bind(py), callbacks);
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

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        self.lock(py)?.close();
        self.reap(py)?;

        let sources: Vec<_> = self
            .lock(py)?
            .draining()
            .map(|source| source.value.clone_ref(py))
            .collect();
        for source in sources {
            self.drain.bind(py).call1((source,))?;
        }
        self.remove_finished(py)?;
        self.lock(py)?
            .require_retired()
            .map_err(|error| native_error(py, error))
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.reclaim)?;
        visit.call(&self.drain)?;
        visit.call(&self.settled)?;
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
        let Some((source, retirement)) = buffer
            .begin_reclaim(|source| self.settled.bind(py).call1((&source.value,))?.is_truthy())?
        else {
            return Ok(());
        };
        let source = source.value.clone_ref(py);
        let retirement = retirement.owner.clone_ref(py);
        drop(state);

        if let Err(error) = self.reclaim.bind(py).call1((source, &retirement)) {
            if !retirement.borrow(py).done() {
                Completion::set_exception(retirement.bind(py), error.value(py).clone().unbind())?;
            }
            return Err(error);
        }
        Ok(())
    }

    fn remove_finished(&self, py: Python<'_>) -> PyResult<()> {
        let retired = self.lock(py)?.take_finished();
        drop(retired);
        Ok(())
    }
}
