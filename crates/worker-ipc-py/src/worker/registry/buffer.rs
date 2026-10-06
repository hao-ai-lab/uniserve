//! Backing storage and retirement shared by registered transport buffers.

use std::os::fd::OwnedFd;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, TryLockError};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use uniserve_worker::{HostAction, HostTask, Outcome};

use super::super::completion::{Completion, CompletionRef};
use super::super::descriptor_grants::DescriptorGrants;
use super::super::events::{CUDAEvent, EventCallback, EventPool};
use super::super::shared_buffer::SharedBuffer;
use super::super::transfer::TransferCapacity;
use super::super::vmm_pool::{PoolChunk, VmmPool};

struct CudaSource {
    tensor: Py<PyAny>,
    event: Py<CUDAEvent>,
    copied: Option<Py<PyAny>>,
    pool: Option<(Py<VmmPool>, Py<PoolChunk>)>,
    consumers: Vec<usize>,
    grants: Option<Py<DescriptorGrants>>,
    export: String,
    descriptor: Option<OwnedFd>,
}

enum Backing {
    Local {
        tensor: Py<PyAny>,
        event: Option<Py<CUDAEvent>>,
    },
    Cuda(CudaSource),
    Shared(Py<SharedBuffer>),
}

/// One export's numerical views and physical backing. Reader admission belongs
/// to its registry; the backing returns bytes only after physical retirement.
#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct TransportBuffer {
    capacity: Py<TransferCapacity>,
    nbytes: u64,
    backing: Mutex<Backing>,
}

impl TransportBuffer {
    pub(in crate::worker) fn local(
        tensor: Py<PyAny>,
        event: Option<Py<CUDAEvent>>,
        nbytes: u64,
        capacity: Py<TransferCapacity>,
    ) -> Self {
        Self {
            capacity,
            nbytes,
            backing: Mutex::new(Backing::Local { tensor, event }),
        }
    }

    pub(in crate::worker) fn shared(
        py: Python<'_>,
        storage: Py<SharedBuffer>,
        capacity: Py<TransferCapacity>,
    ) -> Self {
        let nbytes = storage
            .get()
            .inner
            .lock_py_attached(py)
            .unwrap_or_else(PoisonError::into_inner)
            .nbytes() as u64;

        Self {
            capacity,
            nbytes,
            backing: Mutex::new(Backing::Shared(storage)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::worker) fn cuda(
        tensor: Py<PyAny>,
        event: Py<CUDAEvent>,
        nbytes: u64,
        capacity: Py<TransferCapacity>,
        descriptor: Option<OwnedFd>,
        copied_source: Option<Py<PyAny>>,
        pool: Option<(Py<VmmPool>, Py<PoolChunk>)>,
        consumers: Vec<usize>,
        grants: Option<Py<DescriptorGrants>>,
        export_id: &str,
    ) -> Self {
        Self {
            capacity,
            nbytes,
            backing: Mutex::new(Backing::Cuda(CudaSource {
                tensor,
                event,
                copied: copied_source,
                pool,
                consumers,
                grants,
                export: export_id.into(),
                descriptor,
            })),
        }
    }
}

#[pymethods]
impl TransportBuffer {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.capacity)?;
        let backing = match self.backing.try_lock() {
            Ok(backing) => backing,
            Err(TryLockError::Poisoned(error)) => error.into_inner(),
            Err(TryLockError::WouldBlock) => return Ok(()),
        };

        match &*backing {
            Backing::Local { tensor, event } => {
                visit.call(tensor)?;
                visit.call(event)?;
            }
            Backing::Shared(storage) => visit.call(storage)?,
            Backing::Cuda(source) => {
                visit.call(&source.tensor)?;
                visit.call(&source.event)?;
                visit.call(&source.copied)?;
                visit.call(&source.grants)?;
                if let Some((pool, chunk)) = &source.pool {
                    visit.call(pool)?;
                    visit.call(chunk)?;
                }
            }
        }

        Ok(())
    }
}

impl TransportBuffer {
    pub(in crate::worker) fn tensor(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &*self.lock(py) {
            Backing::Local { tensor, .. } => Ok(tensor.clone_ref(py)),
            Backing::Cuda(source) => Ok(source.tensor.clone_ref(py)),
            Backing::Shared(_) => Err(PyRuntimeError::new_err(
                "shared buffers expose their mapping through SharedRead",
            )),
        }
    }

    pub(in crate::worker) fn event(&self, py: Python<'_>) -> Option<Py<CUDAEvent>> {
        match &*self.lock(py) {
            Backing::Local { event, .. } => event.as_ref().map(|event| event.clone_ref(py)),
            Backing::Cuda(source) => Some(source.event.clone_ref(py)),
            Backing::Shared(_) => None,
        }
    }

    fn lock(&self, py: Python<'_>) -> MutexGuard<'_, Backing> {
        self.backing
            .lock_py_attached(py)
            .unwrap_or_else(PoisonError::into_inner)
    }

    pub(super) fn settled(&self, py: Python<'_>) -> bool {
        match &*self.lock(py) {
            Backing::Shared(storage) => storage
                .get()
                .inner
                .lock_py_attached(py)
                .unwrap_or_else(PoisonError::into_inner)
                .settled(),
            // Producer events are retained by EventPool; remote VMM readers
            // retain pool chunks independently of the original tensor views.
            _ => true,
        }
    }

    pub(super) fn asynchronous(&self, py: Python<'_>) -> bool {
        match &*self.lock(py) {
            Backing::Shared(storage) => storage
                .get()
                .inner
                .lock_py_attached(py)
                .unwrap_or_else(PoisonError::into_inner)
                .is_cuda(),
            _ => false,
        }
    }

    pub(super) fn drain(&self, py: Python<'_>, events: &EventPool) -> PyResult<()> {
        if let Some(event) = self.event(py) {
            let event = Arc::clone(&event.borrow(py).inner);
            py.detach(|| event.wait())
                .map_err(PyRuntimeError::new_err)?;
            events.reap(py)?;
        } else if let Backing::Shared(storage) = &*self.lock(py) {
            let storage = storage
                .get()
                .inner
                .lock_py_attached(py)
                .unwrap_or_else(PoisonError::into_inner);
            let storage = &*storage;
            py.detach(|| storage.synchronize())
                .map_err(PyRuntimeError::new_err)?;
        }

        Ok(())
    }

    pub(in crate::worker) fn reclaim(
        py: Python<'_>,
        buffer: Py<Self>,
        events: &EventPool,
        completion: Option<CompletionRef>,
    ) -> PyResult<()> {
        {
            let backing = buffer.get().lock(py);
            if let Backing::Cuda(source) = &*backing
                && let Some((pool, chunk)) = &source.pool
            {
                pool.borrow(py).retire(
                    py,
                    &chunk.borrow(py),
                    source.consumers.clone(),
                    &source.event.borrow(py),
                    source.grants.as_ref().map(|grants| grants.get()),
                    &source.export,
                )?;
            }
        }

        let event = buffer
            .get()
            .event(py)
            .map(|event| Arc::clone(&event.borrow(py).inner));
        let action = Retirement { buffer, completion };

        if let Some(event) = event {
            events.defer_callback(vec![Arc::clone(&event)], action)?;
            events.wake_event(py, event.device(), &event)?;
            events.reap(py)
        } else {
            py.detach(|| action.run()).map_err(PyRuntimeError::new_err)
        }
    }

    /// Called after the tensor fence, or on the shared host executor when
    /// host unregistration may wait. Completion observers run afterwards.
    fn finish(&self) -> Result<(), String> {
        let mut backing = self.backing.lock().unwrap_or_else(PoisonError::into_inner);
        let copied = match &mut *backing {
            Backing::Local { .. } => None,
            Backing::Shared(storage) => {
                storage
                    .get()
                    .inner
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .close()?;
                None
            }
            Backing::Cuda(source) => {
                if source.pool.is_none() {
                    if let Some(grants) = &source.grants {
                        grants.get().inner.release(&source.export);
                    }
                    source.descriptor.take();
                }
                source.copied.take()
            }
        };
        drop(backing);
        drop(copied);

        self.capacity
            .get()
            .inner
            .release(self.nbytes)
            .map_err(|error| error.to_string())
    }
}

/// The same completion operation serves event retirement and host unregistration.
pub(super) struct Retirement {
    pub(super) buffer: Py<TransportBuffer>,
    pub(super) completion: Option<CompletionRef>,
}

impl EventCallback for Retirement {
    fn complete(self: Box<Self>) -> PyResult<()> {
        self.run().map_err(PyRuntimeError::new_err)
    }

    fn visit(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.buffer)?;
        if let Some(completion) = &self.completion {
            visit.call(&completion.owner)?;
        }
        Ok(())
    }
}

impl HostAction for Retirement {
    type Output = ();
    type Error = String;
    type Callback = Box<dyn FnOnce() + Send>;
    type Wake = Py<PyAny>;

    fn ready(&self) -> Result<bool, String> {
        Ok(true)
    }

    fn run(&self) -> Result<(), String> {
        let result = self.buffer.get().finish();

        if let Some(completion) = &self.completion {
            let outcome = match &result {
                Ok(()) => Ok(()),
                Err(message) => Err(Python::attach(|py| {
                    PyRuntimeError::new_err(message.clone()).into_value(py)
                })),
            };
            let callbacks = completion
                .complete(outcome)
                .map_err(|error| error.to_string())?;
            if !callbacks.is_empty() {
                Python::attach(|py| Completion::notify(completion.owner.bind(py), callbacks));
            }
        }

        result
    }

    fn release(&self) -> Result<(), String> {
        Ok(())
    }

    fn input_outcome(&self) -> Option<Outcome<String>> {
        // Failed reclamation remains owned by the registry and its failed
        // completion. The host task itself has no separate input lease.
        Some(Outcome::Success(()))
    }

    fn defer_release(_: Arc<HostTask<Self>>) -> Result<(), String> {
        Err("transport reclamation has no deferred input".into())
    }

    fn notify(callbacks: Vec<Self::Callback>) {
        for callback in callbacks {
            callback();
        }
    }

    fn wake(wake: &Py<PyAny>) {
        Python::attach(|py| {
            if let Err(error) = wake.bind(py).call0() {
                error.write_unraisable(py, None);
            }
        });
    }

    fn report(error: String) {
        Python::attach(|py| PyRuntimeError::new_err(error).write_unraisable(py, None));
    }

    fn error(error: uniserve_worker::Error) -> String {
        error.to_string()
    }

    fn note_cleanup(error: &mut String, cleanup: String) {
        error.push_str(&format!("; {cleanup}"));
    }
}
