//! Bounded asynchronous reads shared by tensor, KV, and latent transports.
//!
//! Consumable views and physical retirement are separate completions. Credits
//! return only after the backend stops accessing source and destination storage.

use std::ops::Deref;
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{PyCFunction, PyDict, PyTuple};
use uniserve_worker::cuda::{DeviceGuard, Event};
use uniserve_worker::{
    Outcome, ReadReservation as NativeReadReservation, TransferCapacity as NativeTransferCapacity,
    TransferPool as NativeTransferPool, TransferTicket as NativeTransferTicket,
};

use super::error::{invalid, invariant, native_error, resource};
use super::events::{CUDAEvent, EventPool};
use super::host::{HostLane, HostTask, with_context};

/// Python access to the rank's shared native transfer budget.
#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct TransferCapacity {
    inner: Arc<NativeTransferCapacity<Py<PyAny>>>,
}

#[pymethods]
impl TransferCapacity {
    #[new]
    fn new(byte_capacity: i64, ticket_capacity: isize) -> PyResult<Self> {
        if byte_capacity < 1 || ticket_capacity < 1 {
            return Err(PyValueError::new_err(
                "transfer byte and read capacities must be positive",
            ));
        }

        let inner = NativeTransferCapacity::new(
            byte_capacity as u64,
            ticket_capacity as usize,
            notify_capacity,
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?;

        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    #[getter]
    fn capacity(&self) -> u64 {
        self.inner.capacity()
    }

    #[getter]
    fn ticket_capacity(&self) -> usize {
        self.inner.ticket_capacity()
    }

    #[getter]
    fn used(&self) -> u64 {
        self.inner.used()
    }

    #[pyo3(signature = (count=1, *, message="asynchronous transfer ticket capacity is exhausted"))]
    fn take_reads(slf: Bound<'_, Self>, count: isize, message: &str) -> PyResult<()> {
        slf.get()
            .inner
            .take_reads(read_count(count)?)
            .map_err(|error| Self::read_error(&slf, error, message))
    }

    #[pyo3(signature = (count=1))]
    fn return_reads(&self, py: Python<'_>, count: isize) -> PyResult<()> {
        let count = usize::try_from(count)
            .map_err(|_| PyRuntimeError::new_err("read ticket return exceeds the tickets taken"))?;
        self.inner
            .return_reads(count)
            .map_err(|error| native_error(py, error))
    }

    #[pyo3(signature = (callback, *, after))]
    pub(super) fn notify_reads_returned(
        &self,
        py: Python<'_>,
        callback: Py<PyAny>,
        after: u64,
    ) -> PyResult<()> {
        if let Some(callback) = self.inner.notify_reads_returned(callback, after) {
            callback.bind(py).call0()?;
        }
        Ok(())
    }

    fn acquire(&self, py: Python<'_>, amount: i64) -> PyResult<()> {
        let amount = u64::try_from(amount)
            .map_err(|_| PyValueError::new_err("transfer byte reservation must not be negative"))?;
        self.inner
            .acquire(amount)
            .map_err(|error| native_error(py, error))
    }

    fn release(&self, py: Python<'_>, amount: i64) -> PyResult<()> {
        let amount = u64::try_from(amount).map_err(|_| {
            PyRuntimeError::new_err("transfer byte release exceeds the live reservation")
        })?;
        self.inner
            .release(amount)
            .map_err(|error| native_error(py, error))
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.inner.visit(|callbacks| {
            for callback in callbacks {
                visit.call(callback)?;
            }
            Ok(())
        })
    }
}

impl TransferCapacity {
    fn read_error(
        capacity: &Bound<'_, Self>,
        error: uniserve_worker::Error,
        message: &str,
    ) -> PyErr {
        let py = capacity.py();
        let uniserve_worker::Error::ReadBackpressure { returns } = error else {
            return native_error(py, error);
        };

        let converted = (|| -> PyResult<Bound<'_, PyAny>> {
            let kwargs = PyDict::new(py);
            kwargs.set_item("capacity", capacity)?;
            kwargs.set_item("returns", returns)?;
            py.import("uniserve_worker.transport.pool")?
                .getattr("ReadBackpressureError")?
                .call((message,), Some(&kwargs))
        })();

        match converted {
            Ok(error) => PyErr::from_value(error),
            Err(error) => error,
        }
    }
}

/// The native reservation returns unused credits even without an interpreter.
#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct ReadReservation {
    inner: NativeReadReservation<Py<PyAny>>,
    capacity: Py<TransferCapacity>,
}

#[pymethods]
impl ReadReservation {
    #[new]
    fn new(py: Python<'_>, capacity: Py<TransferCapacity>, count: isize) -> PyResult<Self> {
        let inner =
            NativeReadReservation::new(Arc::clone(&capacity.get().inner), read_count(count)?)
                .map_err(|error| {
                    TransferCapacity::read_error(
                        capacity.bind(py),
                        error,
                        "asynchronous transfer ticket capacity is exhausted",
                    )
                })?;

        Ok(Self { inner, capacity })
    }

    #[pyo3(name = "use")]
    fn use_read(&self, py: Python<'_>) -> PyResult<()> {
        self.inner
            .use_read()
            .map_err(|error| native_error(py, error))
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        self.inner.close().map_err(|error| native_error(py, error))
    }

    fn __enter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, py: Python<'_>, _args: &Bound<'_, PyTuple>) -> PyResult<()> {
        self.close(py)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.capacity)
    }
}

fn read_count(count: isize) -> PyResult<usize> {
    if count < 1 {
        return Err(PyValueError::new_err(
            "a read ticket reservation takes at least one",
        ));
    }
    Ok(count as usize)
}

fn notify_capacity(callbacks: Vec<Py<PyAny>>) {
    if !callbacks.is_empty() {
        Python::try_attach(|py| notify(py, callbacks));
    }
}

struct TicketState {
    ticket: NativeTransferTicket<Py<PyAny>, Py<PyAny>, Py<PyAny>>,
    release: Option<Py<PyAny>>,
    retained: Vec<Py<PyAny>>,
    work: Option<Py<HostTask>>,
}

/// One read's stream readiness and physical retirement. Cancellation revokes
/// consumption; it cannot prove that a submitted device copy has stopped.
#[pyclass(frozen, weakref, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct TransferTicket {
    events: Py<EventPool>,
    state: Mutex<TicketState>,
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

impl Deref for TransferRef {
    type Target = uniserve_worker::Completion<Py<PyAny>, Py<PyAny>>;

    fn deref(&self) -> &Self::Target {
        &self.retirement
    }
}

#[pymethods]
impl TransferTicket {
    #[new]
    #[pyo3(signature = (event_pool, *, release=None))]
    fn new(event_pool: Py<EventPool>, release: Option<Py<PyAny>>) -> Self {
        let ticket = NativeTransferTicket::new(release.is_some());
        Self {
            events: event_pool,
            state: Mutex::new(TicketState {
                ticket,
                release,
                retained: Vec::new(),
                work: None,
            }),
        }
    }

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
        let work = state.work.as_ref().map(|work| work.clone_ref(py));
        drop(state);
        notify(py, callbacks);
        if let Some(work) = work {
            work.borrow(py).cancel(py)?;
        }
        Ok(())
    }

    #[pyo3(name = "_require_active")]
    fn require_active(&self, py: Python<'_>) -> PyResult<()> {
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

    /// Borrowed storage stays granted until every observed consumer stream
    /// completes. Copy reads keep their ordinary destination tensor ownership.
    pub(crate) fn close(slf: Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        let owner = slf.get();
        let pool = owner.events.borrow(py);
        pool.reap(py)?;
        let events = {
            let mut state = owner.lock(py)?;
            if !state.ticket.close() {
                return Ok(());
            }

            pool.with_pool(|pool| state.ticket.record_consumers(pool))?;
            state.ticket.consumer_events().to_vec()
        };
        if events.is_empty() {
            return owner.events_released(py);
        }

        for event in &events {
            pool.wake_event(py, event.device(), event)?;
        }
        pool.defer_events(
            events,
            slf.clone().into_any().unbind(),
            Some(slf.getattr("events_released")?.unbind()),
        )?;
        pool.reap(py)
    }

    fn events_released(&self, py: Python<'_>) -> PyResult<()> {
        let release = {
            let mut state = self.lock(py)?;
            state.ticket.release_consumers();
            state.release.take()
        };
        if let Some(release) = release {
            release.bind(py).call0()?;
            self.retire(py)?;
        }
        Ok(())
    }

    #[pyo3(name = "_drain_consumers")]
    fn drain_consumers(slf: Bound<'_, Self>) -> PyResult<()> {
        Self::close(slf.clone())?;
        let py = slf.py();
        let owner = slf.get();
        let events = owner.lock(py)?.ticket.consumer_events().to_vec();
        py.detach(|| {
            for event in events {
                event.wait().map_err(PyRuntimeError::new_err)?;
            }
            Ok::<_, PyErr>(())
        })?;
        owner.events.borrow(py).reap(py)
    }

    #[pyo3(name = "_complete", signature = (value, event=None))]
    fn complete(
        &self,
        py: Python<'_>,
        value: Py<PyAny>,
        event: Option<Py<CUDAEvent>>,
    ) -> PyResult<()> {
        let event = event.map(|event| Arc::clone(&event.borrow(py).inner));
        self.complete_native(py, value, event)
    }

    #[pyo3(name = "_fail")]
    fn fail(&self, py: Python<'_>, error: Py<PyAny>) -> PyResult<bool> {
        let mut state = self.lock(py)?;
        let (late, callbacks) = state.ticket.fail(error);
        drop(state);
        notify(py, callbacks);
        Ok(late)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
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
            state.ticket.retirement.visit(|_, callbacks| {
                for callback in callbacks {
                    visit.call(callback)?;
                }
                Ok(())
            })?;
            visit.call(&state.release)?;
            visit.call(&state.work)?;
            for value in &state.retained {
                visit.call(value)?;
            }
        }
        Ok(())
    }
}

impl TransferTicket {
    /// Block a host-copy thread until a value or error is consumable. Physical
    /// retirement is separate and remains owned by the ticket's readers.
    pub(crate) fn wait_ready(&self, py: Python<'_>) -> PyResult<()> {
        if self.ready(py)? {
            return Ok(());
        }
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let ready = PyCFunction::new_closure(py, None, None, move |_args, _kwargs| {
            let _ = sender.send(());
            Ok::<_, PyErr>(())
        })?;
        self.add_done_callback(py, ready.into_any().unbind())?;
        py.detach(move || receiver.recv())
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }
}

impl TransferTicket {
    fn complete_native(
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
        notify(py, callbacks);
        Ok(())
    }

    fn lock(&self, py: Python<'_>) -> PyResult<MutexGuard<'_, TicketState>> {
        self.state
            .lock_py_attached(py)
            .map_err(|_| invariant(py, "transfer ticket lock is poisoned"))
    }

    fn retire(&self, py: Python<'_>) -> PyResult<()> {
        let callbacks = self
            .lock(py)?
            .ticket
            .retire()
            .map_err(|error| native_error(py, error))?;
        notify(py, callbacks);
        Ok(())
    }
}

impl Drop for TransferTicket {
    fn drop(&mut self) {
        let state = self
            .state
            .get_mut()
            .unwrap_or_else(|error| error.into_inner());
        Python::try_attach(|py| {
            let cleanup = (|| -> PyResult<()> {
                // An unclosed borrowed ticket can lose its last Python owner.
                // Its completion callback keeps the source grant and views
                // until every consumer fence finishes, without resurrecting it.
                let pool = self.events.borrow(py);
                if !state.ticket.closed()
                    && let Some(release) = state.release.take()
                {
                    state.ticket.close();
                    pool.with_pool(|pool| state.ticket.record_consumers(pool))?;
                    let events = state.ticket.consumer_events().to_vec();
                    for event in &events {
                        pool.wake_event(py, event.device(), event)?;
                    }
                    if events.is_empty() {
                        release.bind(py).call0()?;
                        notify(
                            py,
                            state
                                .ticket
                                .retire()
                                .map_err(|error| native_error(py, error))?,
                        );
                    } else {
                        let retained = Mutex::new(Some(release));
                        let retirement = Arc::clone(&state.ticket.retirement);
                        let complete = PyCFunction::new_closure(
                            py,
                            None,
                            None,
                            move |args: &Bound<'_, PyTuple>,
                                  _: Option<&Bound<'_, PyDict>>|
                                  -> PyResult<()> {
                                let py = args.py();
                                let completion = retained
                                    .lock()
                                    .unwrap_or_else(|error| error.into_inner())
                                    .take();
                                if let Some(release) = completion {
                                    release.bind(py).call0()?;
                                    let callbacks = retirement
                                        .complete(Ok(()))
                                        .map_err(|error| native_error(py, error))?;
                                    notify(py, callbacks);
                                }
                                Ok(())
                            },
                        )?;
                        pool.defer_events(
                            events,
                            state
                                .ticket
                                .value()
                                .map(|value| value.clone_ref(py))
                                .unwrap_or_else(|| py.None()),
                            Some(complete.into_any().unbind()),
                        )?;
                    }
                }

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
                pool.reap(py)?;
                Ok(())
            })();
            if let Err(error) = cleanup {
                error.write_unraisable(py, None);
            }
        });
    }
}

struct PoolState {
    pool: NativeTransferPool<Py<PyAny>, Py<TransferTicket>>,
    wake: Option<Py<PyAny>>,
}

/// Execute a backend's reads against the rank's shared credits. The native host
/// lane supplies threads; the pool orders copy streams and retires read credits.
#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct TransferPool {
    lane: HostLane,
    capacity: Py<TransferCapacity>,
    events: Py<EventPool>,
    state: Mutex<PoolState>,
}

#[pymethods]
impl TransferPool {
    #[new]
    #[pyo3(signature = (*, workers, capacity, name, event_pool))]
    fn new(
        py: Python<'_>,
        workers: usize,
        capacity: Py<TransferCapacity>,
        name: &str,
        event_pool: Py<EventPool>,
    ) -> PyResult<Self> {
        let limit = capacity.get().ticket_capacity();
        let lane = HostLane::new(py, limit, workers.min(limit), name)?;
        Ok(Self {
            lane,
            capacity,
            events: event_pool,
            state: Mutex::new(PoolState {
                pool: NativeTransferPool::default(),
                wake: None,
            }),
        })
    }

    fn set_completion_wake(&self, py: Python<'_>, wake: Option<Py<PyAny>>) -> PyResult<()> {
        self.lock(py)?.wake = wake;
        Ok(())
    }

    /// Capture the destination handoff on the submitting thread. A submitted
    /// task owns both credits until its terminal callback proves retirement.
    #[pyo3(signature = (call, *args, nbytes, destination=None, reservation=None))]
    fn submit(
        slf: Bound<'_, Self>,
        call: Py<PyAny>,
        args: &Bound<'_, PyTuple>,
        nbytes: i64,
        destination: Option<Bound<'_, PyAny>>,
        reservation: Option<Py<ReadReservation>>,
    ) -> PyResult<Py<TransferTicket>> {
        let py = slf.py();
        let owner = slf.get();
        {
            let state = owner.lock(py)?;
            if let Some(error) = state.pool.error() {
                return Err(PyErr::from_value(error.bind(py).clone()));
            }
        }
        if let Some(reservation) = reservation {
            if !Arc::ptr_eq(
                reservation.get().inner.capacity(),
                &owner.capacity.get().inner,
            ) {
                return Err(invalid(
                    py,
                    "read reservation belongs to another transfer capacity",
                ));
            }
            reservation.get().use_read(py)?;
        } else {
            TransferCapacity::take_reads(
                owner.capacity.bind(py).clone(),
                1,
                "asynchronous transfer ticket capacity is exhausted",
            )?;
        }
        if let Err(error) = owner.capacity.get().acquire(py, nbytes) {
            owner.capacity.get().return_reads(py, 1)?;
            return Err(error);
        }

        let mut work = None;
        let submitted = (|| -> PyResult<Py<TransferTicket>> {
            let task = owner.lane.reserve(py)?;
            work = Some(task.clone_ref(py));
            let ticket = Py::new(py, TransferTicket::new(owner.events.clone_ref(py), None))?;
            if let Some(destination) = destination {
                let spans = tensor_spans(&destination)?;
                if spans[0].getattr("is_cuda")?.extract::<bool>()? {
                    let device = spans[0].getattr("device")?;
                    let stream = py
                        .import("torch.cuda")?
                        .call_method1("current_stream", (&device,))?
                        .getattr("cuda_stream")?
                        .extract()?;
                    let device = device.getattr("index")?.extract()?;
                    let pool = owner.events.borrow(py);
                    pool.reap(py)?;
                    let mut state = ticket.get().lock(py)?;
                    pool.with_pool(|pool| state.ticket.prepare_copy(pool, device, stream))?;
                }
            }

            if let Some(wake) = owner.lock(py)?.wake.as_ref().map(|wake| wake.clone_ref(py)) {
                ticket.get().add_done_callback(py, wake.clone_ref(py))?;
                ticket.get().add_retirement_callback(py, wake)?;
            }

            let task_owner = slf.clone().unbind();
            let task_ticket = ticket.clone_ref(py);
            let arguments = args.clone().unbind();
            let run = PyCFunction::new_closure(
                py,
                None,
                None,
                move |invocation: &Bound<'_, PyTuple>,
                      _: Option<&Bound<'_, PyDict>>|
                      -> PyResult<()> {
                    let py = invocation.py();
                    task_owner.get().run(py, &task_ticket, &call, &arguments)
                },
            )?;

            let pool = slf.clone().unbind();
            let completed = ticket.clone_ref(py);
            let finished = PyCFunction::new_closure(
                py,
                None,
                None,
                move |args: &Bound<'_, PyTuple>, _: Option<&Bound<'_, PyDict>>| -> PyResult<()> {
                    pool.get().finished(
                        args.py(),
                        &completed,
                        nbytes,
                        &args.get_item(0)?.cast::<HostTask>()?.borrow(),
                    )
                },
            )?;

            HostTask::configure(
                task.bind(py),
                run.into_any().unbind(),
                Vec::new(),
                None,
                None,
                None,
                "uniserve.transfer",
            )?;

            ticket.get().lock(py)?.work = Some(task.clone_ref(py));
            task.borrow(py).submit_if_ready(py)?;
            HostTask::add_done_callback(task.bind(py), finished.into_any().unbind());
            Ok(ticket)
        })();

        if submitted.is_err() {
            if let Some(work) = work {
                work.borrow(py).abandon(py)?;
            }
            owner.capacity.get().release(py, nbytes)?;
            owner.capacity.get().return_reads(py, 1)?;
        }
        submitted
    }

    /// Copy on a transport thread. Readiness exposes a fence before this
    /// method drains its stream; its task retires only after the drain.
    #[pyo3(signature = (ticket, source, destination, producer=None, acknowledgment=None))]
    fn copy(
        &self,
        py: Python<'_>,
        ticket: Py<TransferTicket>,
        source: Bound<'_, PyAny>,
        destination: Bound<'_, PyAny>,
        producer: Option<Py<CUDAEvent>>,
        acknowledgment: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        ticket.get().require_active(py)?;
        let spans = tensor_spans(&destination)?;
        let device = spans[0].getattr("device")?;
        let numerical = py.import("uniserve_worker.transport.pool")?;
        if device.getattr("type")?.extract::<String>()? != "cuda" {
            copy_acknowledgment(py, acknowledgment.as_ref(), "CLAIMED", None)?;
            numerical
                .getattr("_copy_tensors")?
                .call1((&source, &destination, py.None()))?;
            copy_acknowledgment(py, acknowledgment.as_ref(), "ACKNOWLEDGED", None)?;
            return ticket.get().complete(py, destination.unbind(), None);
        }
        let device_index = device.getattr("index")?.extract()?;
        let stream = self
            .lock(py)?
            .pool
            .stream(device_index)
            .map_err(|error| native_error(py, error))?;
        let cuda = py.import("torch.cuda")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("device", &device)?;
        let numerical_stream = cuda
            .getattr("ExternalStream")?
            .call((stream.handle(),), Some(&kwargs))?;
        let producer = producer.map(|event| Arc::clone(&event.borrow(py).inner));
        self.events.borrow(py).reap(py)?;

        let copied = (|| -> PyResult<()> {
            let _device = DeviceGuard::new(device_index).map_err(PyRuntimeError::new_err)?;
            let completed = with_context(&cuda.call_method1("device", (&device,))?, || {
                with_context(&cuda.call_method1("stream", (&numerical_stream,))?, || {
                    // Claim before either handoff wait, so the producer cannot
                    // reuse the source while this copy is waiting to begin.
                    copy_acknowledgment(
                        py,
                        acknowledgment.as_ref(),
                        "CLAIMED",
                        Some(&numerical_stream),
                    )?;
                    if acknowledgment.is_some() {
                        py.detach(|| stream.wait())
                            .map_err(PyRuntimeError::new_err)?;
                    }

                    ticket
                        .get()
                        .lock(py)?
                        .ticket
                        .wait_for_copy(stream.handle())
                        .map_err(|error| native_error(py, error))?;
                    if let Some(producer) = &producer {
                        producer
                            .wait_on(stream.handle())
                            .map_err(PyRuntimeError::new_err)?;
                    }

                    numerical.getattr("_copy_tensors")?.call1((
                        &source,
                        &destination,
                        &numerical_stream,
                    ))?;
                    copy_acknowledgment(
                        py,
                        acknowledgment.as_ref(),
                        "ACKNOWLEDGED",
                        Some(&numerical_stream),
                    )?;
                    self.events.borrow(py).with_pool(|pool| {
                        let event = pool.acquire(device_index, false, false)?;
                        pool.record(&event, device_index, stream.handle())?;
                        Ok(event)
                    })
                })
            })?;
            ticket
                .get()
                .complete_native(py, destination.clone().unbind(), Some(completed))
        })();
        if let Err(error) = &copied {
            ticket
                .get()
                .fail(py, error.clone_ref(py).into_value(py).into_any())?;
        }

        let drained = py.detach(|| {
            let _device = DeviceGuard::new(device_index)?;
            stream.wait()
        });
        if let Err(message) = drained {
            // Unknown physical completion is not cancellation. Keep native
            // stream/fence owners and numerical allocations with the ticket.
            let mut state = ticket.get().lock(py)?;
            state.ticket.mark_undrained(Some(stream), producer);
            state
                .retained
                .extend([source.unbind(), destination.unbind()]);
            let error = PyRuntimeError::new_err(message);
            if let Err(cause) = copied {
                error.set_cause(py, Some(cause));
            }
            return Err(error);
        }
        copied
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        self.lane.close(py)?;
        let error = {
            let mut state = self.lock(py)?;
            state.pool.close();
            state.pool.error().map(|error| error.clone_ref(py))
        };
        if let Some(error) = error {
            return Err(PyErr::from_value(error.into_bound(py)));
        }
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.capacity)?;
        visit.call(&self.events)?;
        if let Some(state) = gc_lock(&self.state) {
            visit.call(state.pool.error())?;
            visit.call(&state.wake)?;
            for ticket in state.pool.unretired() {
                visit.call(ticket)?;
            }
        }
        Ok(())
    }
}

impl TransferPool {
    fn lock(&self, py: Python<'_>) -> PyResult<MutexGuard<'_, PoolState>> {
        self.state
            .lock_py_attached(py)
            .map_err(|_| invariant(py, "transfer pool lock is poisoned"))
    }

    fn run(
        &self,
        py: Python<'_>,
        ticket: &Py<TransferTicket>,
        call: &Py<PyAny>,
        args: &Py<PyTuple>,
    ) -> PyResult<()> {
        ticket.get().require_active(py)?;
        with_context(&py.import("torch")?.call_method0("inference_mode")?, || {
            let mut arguments = vec![ticket.bind(py).as_any().clone()];
            arguments.extend(args.bind(py).iter());
            call.bind(py).call1(PyTuple::new(py, arguments)?)?;
            Ok(())
        })
    }

    fn finished(
        &self,
        py: Python<'_>,
        ticket: &Py<TransferTicket>,
        nbytes: i64,
        work: &HostTask,
    ) -> PyResult<()> {
        let undrained = {
            let mut state = ticket.get().lock(py)?;
            state.work = None;
            state.ticket.undrained()
        };
        let error = if work.cancelled() {
            Some(resource(py, "transfer read was cancelled before submission").into_value(py))
        } else {
            work.exception(py, Some(0.0))?
        };

        if let Some(error) = error {
            let error = error.into_any();
            let late = ticket.get().fail(py, error.clone_ref(py))?;
            if late || undrained {
                let mut state = self.lock(py)?;
                state
                    .pool
                    .fail(error, undrained.then(|| ticket.clone_ref(py)));
                let wake = late
                    .then(|| state.wake.as_ref().map(|wake| wake.clone_ref(py)))
                    .flatten();
                drop(state);
                if let Some(wake) = wake {
                    notify(py, vec![wake]);
                }
            }
        }

        if !undrained {
            self.capacity.get().release(py, nbytes)?;
            self.capacity.get().return_reads(py, 1)?;
            ticket.get().retire(py)?;
        }
        Ok(())
    }
}

fn copy_acknowledgment(
    py: Python<'_>,
    word: Option<&Bound<'_, PyAny>>,
    state: &str,
    stream: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    if let Some(word) = word {
        let value = py
            .import("uniserve_worker.transport.vmm_pool")?
            .getattr(state)?;
        let numerical = py.import("uniserve_worker.transport.pool")?;
        let source = numerical.getattr("chunk_word")?.call1((value,))?;

        // The cached word outlives the native copy stream. Use explicit DMA
        // ownership: PyTorch's pinned allocator otherwise records an event on
        // that stream when the cached tensor is eventually freed.
        numerical
            .getattr("_copy_tensors")?
            .call1((source, word, stream))?;
    }
    Ok(())
}

fn tensor_spans<'py>(value: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    if let Ok(spans) = value.cast::<PyTuple>() {
        if spans.is_empty() {
            return Err(invalid(value.py(), "tensor destination has no spans"));
        }
        spans.extract()
    } else {
        Ok(vec![value.clone()])
    }
}

// Completion callbacks may inspect their ticket or submit another read. Run
// them after unlocking, and retain no callback once it has been notified.
fn notify(py: Python<'_>, callbacks: Vec<Py<PyAny>>) {
    for callback in callbacks {
        if let Err(error) = callback.bind(py).call0() {
            error.write_unraisable(py, Some(callback.bind(py)));
        }
    }
}

fn gc_lock<T>(mutex: &Mutex<T>) -> Option<MutexGuard<'_, T>> {
    match mutex.try_lock() {
        Ok(state) => Some(state),
        Err(TryLockError::Poisoned(error)) => Some(error.into_inner()),
        // An active native method retains its owner. GC cannot wait for a
        // thread that may need the interpreter before releasing this lock.
        Err(TryLockError::WouldBlock) => None,
    }
}
