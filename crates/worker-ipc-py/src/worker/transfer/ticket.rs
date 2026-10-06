//! Borrowed views, consumer fences and native physical read ownership.

use std::collections::HashMap;
use std::ops::Deref;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use uniserve_worker::cuda::Event;
use uniserve_worker::{
    HostTask as NativeHostTask, Outcome, TransferRead, TransferTicket as NativeTransferTicket,
};

use super::super::error::{invalid, invariant, native_error, resource};
use super::super::events::{CUDAEvent, EventCallback, EventPool};
use super::super::registry::BufferRead;
use super::{PythonRead, TransferCapacity, gc_lock, notify, tensor_spans};

pub(super) struct ReadState {
    pub(super) ticket: NativeTransferTicket<Py<PyAny>, Py<PyAny>, Py<PyAny>>,
    source: Option<BufferRead>,
    pub(super) credit: Option<Py<TransferCapacity>>,
    pub(super) retained: Vec<Py<PyAny>>,
    pub(super) work: Option<Weak<NativeHostTask<TransferRead<PythonRead>>>>,
}

pub(super) type BorrowedReads = Mutex<Option<HashMap<usize, Weak<Read>>>>;

/// One read's stream readiness and physical retirement. Cancellation revokes
/// consumption; it cannot prove that a submitted device copy has stopped.
#[pyclass(frozen, weakref, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct TransferTicket {
    pub(super) inner: Arc<Read>,
}

/// Native ownership survives an unclosed Python ticket until its consumers end.
pub(crate) struct Read {
    pub(super) events: Py<EventPool>,
    pub(super) ready: Condvar,
    pub(super) state: Mutex<ReadState>,
    borrower: Option<Weak<BorrowedReads>>,
}

impl Deref for TransferTicket {
    type Target = Read;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Retain the transport object while native storage owners inspect physical
/// completion without entering Python or the ticket's numerical state lock.
pub(crate) struct TransferRef {
    pub(crate) owner: Py<TransferTicket>,
    retirement: Arc<uniserve_worker::Completion<Py<PyAny>, Py<PyAny>>>,
}

impl TransferRef {
    pub(crate) fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            owner: self.owner.clone_ref(py),
            retirement: Arc::clone(&self.retirement),
        }
    }

    pub(crate) fn new(py: Python<'_>, owner: Py<TransferTicket>) -> PyResult<Self> {
        let retirement = Arc::clone(&owner.get().lock(py)?.ticket.retirement);
        Ok(Self { owner, retirement })
    }
}

impl TransferTicket {
    /// Inspect cancellation while a native transport wait has released the GIL.
    pub(in crate::worker) fn check_active(&self) -> uniserve_worker::Result<()> {
        let state = self.state.lock().map_err(|_| {
            uniserve_worker::Error::Invariant("transfer ticket lock is poisoned".into())
        })?;
        if state.ticket.is_cancelled() {
            return Err(uniserve_worker::Error::Resource(
                "transfer read was cancelled",
            ));
        }
        Ok(())
    }
}

impl Deref for TransferRef {
    type Target = uniserve_worker::Completion<Py<PyAny>, Py<PyAny>>;

    fn deref(&self) -> &Self::Target {
        &self.retirement
    }
}

#[pymethods]
impl TransferTicket {
    pub(crate) fn ready(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(self.lock(py)?.ticket.ready())
    }

    pub(crate) fn retired(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(self.lock(py)?.ticket.retired())
    }

    pub(crate) fn retirement_ready(&self, py: Python<'_>) -> PyResult<bool> {
        let state = self.lock(py)?;
        state.ticket.retirement_ready().map_err(|error| {
            let error = native_error(py, error);
            if let Some(cause) = state.ticket.error() {
                error.set_cause(py, Some(PyErr::from_value(cause.bind(py).clone())));
            }
            error
        })
    }

    pub(crate) fn add_done_callback(&self, py: Python<'_>, callback: Py<PyAny>) -> PyResult<()> {
        let immediate = self.lock(py)?.ticket.add_done_callback(callback);
        if let Some(callback) = immediate {
            notify(py, vec![callback]);
        }
        Ok(())
    }

    pub(crate) fn add_retirement_callback(
        &self,
        py: Python<'_>,
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        let immediate = self.lock(py)?.ticket.retirement.subscribe(callback);
        if let Some(callback) = immediate {
            notify(py, vec![callback]);
        }
        Ok(())
    }

    pub(crate) fn cancel(&self, py: Python<'_>) -> PyResult<()> {
        let error = resource(py, "transfer read was cancelled")
            .into_value(py)
            .into_any();
        let mut state = self.lock(py)?;
        let callbacks = state.ticket.cancel(error);
        let work = state.work.as_ref().and_then(Weak::upgrade);
        drop(state);
        self.inner.ready.notify_all();
        notify(py, callbacks);
        if let Some(work) = work {
            py.detach(|| work.cancel(false))
                .map_err(|error| PyErr::from_value(error.into_bound(py)))?;
        }
        Ok(())
    }

    #[pyo3(name = "_require_active")]
    pub(super) fn require_active(&self, py: Python<'_>) -> PyResult<()> {
        if let Some(error) = self.lock(py)?.ticket.cancellation_error() {
            return Err(PyErr::from_value(error.bind(py).clone()));
        }
        Ok(())
    }

    /// Return views without a host wait. CUDA consumption waits on the read
    /// fence and records every view on the consuming allocator stream.
    #[pyo3(signature = (stream=None))]
    pub(crate) fn result(
        &self,
        py: Python<'_>,
        stream: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let mut state = self.lock(py)?;
        let value = match state
            .ticket
            .result()
            .map_err(|error| native_error(py, error))?
        {
            Outcome::Success(value) => value.clone_ref(py),
            Outcome::Failed(error) => return Err(PyErr::from_value(error.bind(py).clone())),
            Outcome::Cancelled => unreachable!("transfer cancellation retains its error"),
        };
        if state.ticket.producer().is_some() {
            let spans = tensor_spans(value.bind(py))?;
            let device = spans[0].getattr("device")?;
            let consumer = match stream {
                Some(stream) => stream,
                None => py
                    .import("torch")?
                    .getattr("cuda")?
                    .call_method1("current_stream", (&device,))?,
            };
            if !consumer.getattr("device")?.eq(&device)? {
                return Err(invalid(py, "transfer consumer stream is on another device"));
            }
            state
                .ticket
                .consume(
                    device.getattr("index")?.extract()?,
                    consumer.getattr("cuda_stream")?.extract()?,
                )
                .map_err(|error| native_error(py, error))?;
            for span in spans {
                span.call_method1("record_stream", (&consumer,))?;
            }
        }
        Ok(value)
    }

    /// Borrowed storage stays granted until every consumer stream completes.
    /// Copy destinations retain their ordinary tensor ownership.
    pub(crate) fn close(slf: Bound<'_, Self>) -> PyResult<()> {
        slf.get().inner.close(slf.py())
    }

    #[pyo3(name = "_complete", signature = (value, event=None))]
    pub(super) fn complete(
        &self,
        py: Python<'_>,
        value: Py<PyAny>,
        event: Option<Py<CUDAEvent>>,
    ) -> PyResult<()> {
        let event = event.map(|event| Arc::clone(&event.borrow(py).inner));
        self.complete_native(py, value, event)
    }

    #[pyo3(name = "_fail")]
    pub(super) fn fail(&self, py: Python<'_>, error: Py<PyAny>) -> PyResult<bool> {
        let mut state = self.lock(py)?;
        let (late, callbacks) = state.ticket.fail(error);
        drop(state);
        self.inner.ready.notify_all();
        notify(py, callbacks);
        Ok(late)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        // Native completion work is an external owner while a read is pending.
        if Arc::strong_count(&self.inner) == 1 {
            self.inner.visit(&visit)?;
        }
        Ok(())
    }
}

impl TransferTicket {
    /// Block a host-copy thread until a value or error is consumable. Physical
    /// retirement is separate and remains owned by the ticket's readers.
    pub(crate) fn wait_ready(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let state = self
                .state
                .lock()
                .map_err(|_| PyRuntimeError::new_err("transfer ticket lock is poisoned"))?;
            let _ready = self
                .ready
                .wait_while(state, |state| !state.ticket.ready())
                .map_err(|_| PyRuntimeError::new_err("transfer ticket lock is poisoned"))?;
            Ok(())
        })
    }
}

impl TransferTicket {
    pub(super) fn complete_native(
        &self,
        py: Python<'_>,
        value: Py<PyAny>,
        event: Option<Arc<Event>>,
    ) -> PyResult<()> {
        let mut state = self.lock(py)?;
        if let Some(event) = &event {
            self.events
                .borrow(py)
                .with_pool(|pool| pool.retain(event, event.device(), 1))?;
        }
        let callbacks = state.ticket.complete(value, event);
        drop(state);
        self.inner.ready.notify_all();
        notify(py, callbacks);
        Ok(())
    }
}

impl TransferTicket {
    pub(super) fn new(events: Py<EventPool>, borrower: Option<Weak<BorrowedReads>>) -> Self {
        Self {
            inner: Arc::new(Read {
                events,
                ready: Condvar::new(),
                state: Mutex::new(ReadState {
                    ticket: NativeTransferTicket::new(borrower.is_some()),
                    source: None,
                    credit: None,
                    retained: Vec::new(),
                    work: None,
                }),
                borrower,
            }),
        }
    }

    pub(in crate::worker) fn bind_source(
        &self,
        py: Python<'_>,
        source: BufferRead,
    ) -> PyResult<()> {
        let mut state = self.lock(py)?;
        if state.source.is_some() {
            drop(state);
            source.release(py)?;
            return Err(invariant(py, "transfer read already owns a source buffer"));
        }
        state.source = Some(source);
        Ok(())
    }
}

impl Read {
    pub(super) fn lock(&self, py: Python<'_>) -> PyResult<MutexGuard<'_, ReadState>> {
        self.state
            .lock_py_attached(py)
            .map_err(|_| invariant(py, "transfer ticket lock is poisoned"))
    }

    fn close(self: &Arc<Self>, py: Python<'_>) -> PyResult<()> {
        let pool = self.events.borrow(py);
        pool.reap(py)?;
        let events = {
            let mut state = self.lock(py)?;
            if !state.ticket.close() {
                return Ok(());
            }

            pool.with_pool(|pool| state.ticket.record_consumers(pool))?;
            state.ticket.consumer_events().to_vec()
        };
        if events.is_empty() {
            return self.finish_borrow(py);
        }

        // Retain the read before a notification can run its completion work.
        pool.defer_callback(events.clone(), Arc::clone(self))?;
        for event in &events {
            pool.wake_event(py, event.device(), event)?;
        }
        pool.reap(py)
    }

    pub(super) fn drain(self: &Arc<Self>, py: Python<'_>) -> PyResult<()> {
        self.close(py)?;
        let events = self.lock(py)?.ticket.consumer_events().to_vec();
        py.detach(|| {
            for event in events {
                event.wait().map_err(PyRuntimeError::new_err)?;
            }
            Ok::<_, PyErr>(())
        })?;
        self.events.borrow(py).reap(py)
    }

    pub(super) fn release_source(&self, py: Python<'_>) -> PyResult<()> {
        let (source, credit) = {
            let mut state = self.lock(py)?;
            (state.source.take(), state.credit.take())
        };
        let released = source.as_ref().map_or(Ok(()), |source| source.release(py));
        if released.is_err() {
            // A failed producer reclamation retains its registry and backing.
            self.lock(py)?.source = source;
        }
        let returned = credit.map_or(Ok(()), |capacity| {
            capacity
                .get()
                .inner
                .return_reads(1)
                .map_err(|error| native_error(py, error))
        });
        released.and(returned)
    }

    fn finish_borrow(&self, py: Python<'_>) -> PyResult<()> {
        self.lock(py)?.ticket.release_consumers();

        if let Some(borrower) = self.borrower.as_ref().and_then(Weak::upgrade)
            && let Some(reads) = borrower
                .lock_py_attached(py)
                .unwrap_or_else(|error| error.into_inner())
                .as_mut()
        {
            reads.remove(&(std::ptr::from_ref(self) as usize));
        }

        let result = self.release_source(py);
        let callbacks = {
            let state = self.lock(py)?;
            match &result {
                Ok(()) => state.ticket.retire(),
                Err(error) => state.ticket.retirement.complete(Err(error
                    .value(py)
                    .clone()
                    .into_any()
                    .unbind())),
            }
        }
        .map_err(|error| native_error(py, error))?;
        notify(py, callbacks);
        result
    }

    fn visit(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.events)?;
        if let Some(state) = gc_lock(&self.state) {
            state.ticket.visit(|value, error, callbacks| {
                visit.call(value)?;
                visit.call(error)?;
                for callback in callbacks {
                    visit.call(callback)?;
                }
                Ok(())
            })?;
            state.ticket.retirement.visit(|outcome, callbacks| {
                if let Some(Outcome::Failed(error)) = outcome {
                    visit.call(error.as_ref())?;
                }
                for callback in callbacks {
                    visit.call(callback)?;
                }
                Ok(())
            })?;
            if let Some(source) = &state.source {
                source.visit(visit)?;
            }
            visit.call(&state.credit)?;
            for value in &state.retained {
                visit.call(value)?;
            }
        }
        Ok(())
    }
}

impl EventCallback for Arc<Read> {
    fn complete(self: Box<Self>) -> PyResult<()> {
        Python::attach(|py| self.finish_borrow(py))
    }

    fn visit(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        if Arc::strong_count(self) == 1 {
            Read::visit(self, visit)?;
        }
        Ok(())
    }
}

impl Drop for TransferTicket {
    fn drop(&mut self) {
        Python::try_attach(|py| {
            if let Err(error) = self.inner.close(py) {
                error.write_unraisable(py, None);
            }
        });
    }
}

impl Drop for Read {
    fn drop(&mut self) {
        let state = self
            .state
            .get_mut()
            .unwrap_or_else(|error| error.into_inner());
        Python::try_attach(|py| {
            let cleanup = (|| -> PyResult<()> {
                let pool = self.events.borrow(py);
                let events = state.ticket.take_events();
                if !events.is_empty() {
                    pool.defer_events(
                        events,
                        state
                            .ticket
                            .value()
                            .map(|value| value.clone_ref(py))
                            .unwrap_or_else(|| py.None()),
                        None,
                    )?;
                }
                pool.reap(py)
            })();
            if let Err(error) = cleanup {
                error.write_unraisable(py, None);
            }
        });
    }
}
