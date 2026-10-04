//! Registered transfer buffers and their outstanding readers.
//!
//! Backends supply fences and acknowledgment checks. The registry decides when
//! a revoked buffer has no remaining users and can be handed back to its owner.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, TryLockError};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyBaseException, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;

use super::completion::Completion;
use super::error::{invalid, invariant, resource};

#[derive(Clone, Eq, Hash, PartialEq)]
enum BufferKey {
    Local(u64),
    Shm(String),
    Cuda(String),
}

#[derive(PartialEq)]
enum Status {
    Active,
    Revoked,
    Reclaiming,
}

struct RegisteredBuffer {
    locator: Py<PyAny>,
    source: Py<PyAny>,
    retirement: Py<Completion>,
    readers: usize,
    pending: bool,
    failed: bool,
    status: Status,
}

struct RegistryState {
    buffers: HashMap<BufferKey, RegisteredBuffer>,
    closing: bool,
}

/// Retain source storage until its producer and all granted readers finish.
/// Reclaim callbacks and retirement observers run outside the registry lock.
#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct BufferRegistry {
    #[pyo3(get)]
    name: String,
    capacity: usize,
    reclaim: Py<PyAny>,
    drain: Py<PyAny>,
    settled: Py<PyAny>,
    state: Mutex<RegistryState>,
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
        if capacity == 0 {
            return Err(PyValueError::new_err("buffer capacity must be positive"));
        }
        let suffix: String = py
            .import("uuid")?
            .call_method0("uuid4")?
            .getattr("hex")?
            .extract()?;

        Ok(Self {
            name: format!("uniserve-buffers-{suffix}"),
            capacity,
            reclaim,
            drain,
            settled,
            state: Mutex::new(RegistryState {
                buffers: HashMap::new(),
                closing: false,
            }),
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
        let mut state = self.lock(py)?;
        if state.closing || state.buffers.len() >= self.capacity {
            return Err(resource(py, "transport buffer capacity is unavailable"));
        }
        if state.buffers.contains_key(&key) {
            return Err(invalid(py, "buffer is already registered"));
        }
        state.buffers.insert(
            key,
            RegisteredBuffer {
                locator,
                source,
                retirement,
                readers: 0,
                pending,
                failed: false,
                status: Status::Active,
            },
        );
        Ok(())
    }

    /// Inspect owner-held storage, including after reads have been revoked.
    fn source(&self, py: Python<'_>, locator: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let key = self.key(locator)?;
        let mut state = self.lock(py)?;
        Ok(Self::buffer(&mut state, &key, locator)?
            .source
            .clone_ref(py))
    }

    /// Grant a local read. Its caller must release it after physical access
    /// ends, even if the buffer was revoked while the read was in flight.
    fn acquire(&self, py: Python<'_>, locator: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let key = self.key(locator)?;
        let mut state = self.lock(py)?;
        let buffer = Self::buffer(&mut state, &key, locator)?;
        if buffer.status != Status::Active {
            return Err(invalid(py, "buffer is no longer available for reading"));
        }
        if buffer.failed {
            return Err(resource(py, "buffer producer failed before readiness"));
        }
        buffer.readers += 1;
        Ok(buffer.source.clone_ref(py))
    }

    fn release_reader(&self, py: Python<'_>, locator: &Bound<'_, PyAny>) -> PyResult<()> {
        let key = self.key(locator)?;
        let mut state = self.lock(py)?;
        let buffer = Self::buffer(&mut state, &key, locator)?;
        buffer.readers = buffer
            .readers
            .checked_sub(1)
            .ok_or_else(|| invariant(py, "buffer read was already released"))?;
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
        if !producer_completed && error.is_none() {
            return Err(PyValueError::new_err(
                "unknown producer completion requires a failure",
            ));
        }
        let key = self.key(locator)?;
        let mut state = self.lock(py)?;
        let buffer = Self::buffer(&mut state, &key, locator)?;
        buffer.pending = !producer_completed;
        buffer.failed = error.is_some();
        let retirement = buffer.retirement.clone_ref(py);
        drop(state);

        if !producer_completed
            && !retirement.borrow(py).done()
            && let Some(error) = error
        {
            Completion::set_exception(retirement.bind(py), error)?;
        }
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
        if !state.buffers.contains_key(&key) {
            return Ok(None);
        }
        let buffer = Self::buffer(&mut state, &key, locator)?;
        if buffer.status == Status::Active {
            buffer.status = Status::Revoked;
        }
        let retirement = buffer.retirement.clone_ref(py);
        drop(state);
        self.reclaim_buffer(py, &key)?;
        Ok(Some(retirement))
    }

    fn retirement(&self, py: Python<'_>, locator: &Bound<'_, PyAny>) -> PyResult<Py<Completion>> {
        let key = self.key(locator)?;
        let mut state = self.lock(py)?;
        Ok(Self::buffer(&mut state, &key, locator)?
            .retirement
            .clone_ref(py))
    }

    fn awaiting_acknowledgment(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(self
            .lock(py)?
            .buffers
            .values()
            .any(|buffer| buffer.status == Status::Revoked && !buffer.pending))
    }

    /// Sweep backend acknowledgments and discard successfully retired storage.
    fn reap(&self, py: Python<'_>) -> PyResult<()> {
        let keys: Vec<_> = self
            .lock(py)?
            .buffers
            .iter()
            .filter(|(_, buffer)| buffer.status == Status::Revoked)
            .map(|(key, _)| key.clone())
            .collect();
        for key in keys {
            self.reclaim_buffer(py, &key)?;
        }
        self.remove_finished(py)
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let mut state = self.lock(py)?;
        state.closing = true;
        for buffer in state.buffers.values_mut() {
            if buffer.status == Status::Active {
                buffer.status = Status::Revoked;
            }
        }
        drop(state);
        self.reap(py)?;

        let sources: Vec<_> = self
            .lock(py)?
            .buffers
            .values()
            .filter(|buffer| buffer.status == Status::Reclaiming)
            .map(|buffer| buffer.source.clone_ref(py))
            .collect();
        for source in sources {
            self.drain.bind(py).call1((source,))?;
        }
        self.remove_finished(py)?;
        if !self.lock(py)?.buffers.is_empty() {
            return Err(resource(
                py,
                "buffer completion or readers remain unresolved; storage is retained",
            ));
        }
        Ok(())
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
        for buffer in state.buffers.values() {
            visit.call(&buffer.locator)?;
            visit.call(&buffer.source)?;
            visit.call(&buffer.retirement)?;
        }
        Ok(())
    }
}

impl BufferRegistry {
    fn lock(&self, py: Python<'_>) -> PyResult<MutexGuard<'_, RegistryState>> {
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
            "cuda_vmm" => Ok(BufferKey::Cuda(
                handle.getattr("publication_id")?.extract()?,
            )),
            _ => Err(invalid(py, "transport does not register source storage")),
        }
    }

    fn buffer<'a>(
        state: &'a mut RegistryState,
        key: &BufferKey,
        locator: &Bound<'_, PyAny>,
    ) -> PyResult<&'a mut RegisteredBuffer> {
        let buffer = state
            .buffers
            .get_mut(key)
            .ok_or_else(|| invalid(locator.py(), "buffer is no longer registered"))?;
        // The immutable locator describes the actual view, including its fence.
        // Comparing it directly avoids serializing and hashing tensor metadata.
        if !buffer.locator.bind(locator.py()).eq(locator)? {
            return Err(invalid(
                locator.py(),
                "buffer locator changed its registered view",
            ));
        }
        Ok(buffer)
    }

    fn reclaim_buffer(&self, py: Python<'_>, key: &BufferKey) -> PyResult<()> {
        let mut state = self.lock(py)?;
        let Some(buffer) = state.buffers.get_mut(key) else {
            return Ok(());
        };
        if buffer.status != Status::Revoked || buffer.pending || buffer.readers != 0 {
            return Ok(());
        }
        // Hold the lock while inspecting mapped acknowledgment words: another
        // reaper must not unmap the source during this read-only backend query.
        if !self
            .settled
            .bind(py)
            .call1((&buffer.source,))?
            .is_truthy()?
        {
            return Ok(());
        }
        buffer.status = Status::Reclaiming;
        let source = buffer.source.clone_ref(py);
        let retirement = buffer.retirement.clone_ref(py);
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
        let mut state = self.lock(py)?;
        let mut finished = Vec::new();
        // Failed retirement keeps its backing storage held, so close can report
        // it without permitting reuse after an unknown device completion.
        for (key, buffer) in &state.buffers {
            if buffer.status == Status::Reclaiming && buffer.retirement.borrow(py).succeeded() {
                finished.push(key.clone());
            }
        }
        let retired: Vec<_> = finished
            .iter()
            .filter_map(|key| state.buffers.remove(key))
            .collect();
        drop(state);
        drop(retired);
        Ok(())
    }
}
