//! Bounded asynchronous reads shared by tensor, KV, and latent transports.
//!
//! Consumable views and physical retirement are separate completions. Credits
//! return only after the backend stops accessing source and destination storage.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, TryLockError};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{PyCFunction, PyDict, PyTuple};

use super::error::{invalid, invariant, resource, unsupported};

/// Failed or cancelled retirement cannot authorize physical storage reuse.
pub(crate) fn retirement_succeeded(future: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(future.call_method0("done")?.is_truthy()?
        && !future.call_method0("cancelled")?.is_truthy()?
        && future.call_method0("exception")?.is_none())
}

struct CapacityState {
    used: u64,
    reads_free: usize,
    returns: u64,
    waiters: Vec<Py<PyAny>>,
}

/// One rank's byte and read credits, shared by every transport mechanism.
#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct TransferCapacity {
    #[pyo3(get)]
    capacity: u64,
    #[pyo3(get)]
    ticket_capacity: usize,
    state: Mutex<CapacityState>,
}

#[pymethods]
impl TransferCapacity {
    #[new]
    fn new(byte_capacity: i64, ticket_capacity: isize) -> PyResult<Self> {
        if byte_capacity < 1 || ticket_capacity < 1 {
            return Err(PyValueError::new_err(
                "transfer byte capacity must be positive",
            ));
        }
        Ok(Self {
            capacity: byte_capacity as u64,
            ticket_capacity: ticket_capacity as usize,
            state: Mutex::new(CapacityState {
                used: 0,
                reads_free: ticket_capacity as usize,
                returns: 0,
                waiters: Vec::new(),
            }),
        })
    }

    #[getter]
    fn used(&self, py: Python<'_>) -> PyResult<u64> {
        Ok(self.lock(py)?.used)
    }

    /// Reserve the entire fan-out before any physical read starts. A refusal
    /// carries the return sequence so registration cannot miss a credit wake.
    #[pyo3(signature = (count=1, *, message="asynchronous transfer ticket capacity is exhausted"))]
    fn take_reads(slf: Bound<'_, Self>, count: isize, message: &str) -> PyResult<()> {
        let py = slf.py();
        if count < 1 {
            return Err(PyValueError::new_err(
                "a read ticket reservation takes at least one",
            ));
        }
        let owner = slf.get();
        let count = count as usize;
        if count > owner.ticket_capacity {
            return Err(unsupported(
                py,
                format!(
                    "a fetch of {count} reads exceeds the rank's {} read tickets",
                    owner.ticket_capacity
                ),
            ));
        }
        let mut state = owner.lock(py)?;
        if count > state.reads_free {
            let returns = state.returns;
            drop(state);
            let kwargs = PyDict::new(py);
            kwargs.set_item("capacity", &slf)?;
            kwargs.set_item("returns", returns)?;
            return Err(PyErr::from_value(
                py.import("uniserve_worker.transport.pool")?
                    .getattr("ReadBackpressureError")?
                    .call((message,), Some(&kwargs))?,
            ));
        }
        state.reads_free -= count;
        Ok(())
    }

    #[pyo3(signature = (count=1))]
    fn return_reads(&self, py: Python<'_>, count: isize) -> PyResult<()> {
        let mut state = self.lock(py)?;
        if count < 0 || count as usize > self.ticket_capacity - state.reads_free {
            return Err(PyRuntimeError::new_err(
                "read ticket return exceeds the tickets taken",
            ));
        }
        state.reads_free += count as usize;
        state.returns = state
            .returns
            .checked_add(1)
            .ok_or_else(|| invariant(py, "transfer credit return sequence exhausted"))?;
        let callbacks = std::mem::take(&mut state.waiters);
        drop(state);
        notify(py, callbacks);
        Ok(())
    }

    #[pyo3(signature = (callback, *, after))]
    fn notify_reads_returned(
        &self,
        py: Python<'_>,
        callback: Py<PyAny>,
        after: u64,
    ) -> PyResult<()> {
        let mut state = self.lock(py)?;
        if state.returns == after {
            state.waiters.push(callback);
        } else {
            drop(state);
            callback.bind(py).call0()?;
        }
        Ok(())
    }

    fn acquire(&self, py: Python<'_>, amount: i64) -> PyResult<()> {
        if amount < 0 {
            return Err(PyValueError::new_err(
                "transfer byte reservation must not be negative",
            ));
        }
        let mut state = self.lock(py)?;
        let projected = state
            .used
            .checked_add(amount as u64)
            .ok_or_else(|| resource(py, "transfer byte capacity is exhausted"))?;
        if projected > self.capacity {
            return Err(resource(
                py,
                format!(
                    "transfer byte capacity is exhausted ({projected}>{})",
                    self.capacity
                ),
            ));
        }
        state.used = projected;
        Ok(())
    }

    fn release(&self, py: Python<'_>, amount: i64) -> PyResult<()> {
        let mut state = self.lock(py)?;
        if amount < 0 || amount as u64 > state.used {
            return Err(PyRuntimeError::new_err(
                "transfer byte release exceeds the live reservation",
            ));
        }
        state.used -= amount as u64;
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(state) = gc_lock(&self.state) {
            for callback in &state.waiters {
                visit.call(callback)?;
            }
        }
        Ok(())
    }
}

impl TransferCapacity {
    fn lock(&self, py: Python<'_>) -> PyResult<MutexGuard<'_, CapacityState>> {
        self.state
            .lock_py_attached(py)
            .map_err(|_| invariant(py, "transfer capacity lock is poisoned"))
    }
}

/// A fetch takes all its read credits together, then transfers one credit to
/// each submitted read. Closing returns only credits it has not handed off.
#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct ReadReservation {
    capacity: Py<TransferCapacity>,
    unused: Mutex<usize>,
}

#[pymethods]
impl ReadReservation {
    #[new]
    fn new(py: Python<'_>, capacity: Py<TransferCapacity>, count: isize) -> PyResult<Self> {
        TransferCapacity::take_reads(
            capacity.bind(py).clone(),
            count,
            "asynchronous transfer ticket capacity is exhausted",
        )?;
        Ok(Self {
            capacity,
            unused: Mutex::new(count as usize),
        })
    }

    #[pyo3(name = "use")]
    fn use_read(&self, py: Python<'_>) -> PyResult<()> {
        let mut unused = self
            .unused
            .lock_py_attached(py)
            .map_err(|_| invariant(py, "read reservation lock is poisoned"))?;
        if *unused == 0 {
            return Err(PyRuntimeError::new_err(
                "read reservation has no ticket left",
            ));
        }
        *unused -= 1;
        Ok(())
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let unused = {
            let mut unused = self
                .unused
                .lock_py_attached(py)
                .map_err(|_| invariant(py, "read reservation lock is poisoned"))?;
            std::mem::take(&mut *unused)
        };
        if unused != 0 {
            self.capacity.get().return_reads(py, unused as isize)?;
        }
        Ok(())
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

impl Drop for ReadReservation {
    fn drop(&mut self) {
        let unused = *self
            .unused
            .get_mut()
            .unwrap_or_else(|error| error.into_inner());
        if unused != 0 {
            Python::try_attach(|py| {
                if let Err(error) = self.capacity.get().return_reads(py, unused as isize) {
                    error.write_unraisable(py, None);
                }
            });
        }
    }
}

enum ReadAccess {
    Copy {
        destination_stream: Option<Py<PyAny>>,
    },
    Borrowed {
        release: Option<Py<PyAny>>,
        streams: HashMap<u64, Py<PyAny>>,
        events: Vec<Py<PyAny>>,
        closed: bool,
    },
}

struct TicketState {
    value: Option<Py<PyAny>>,
    error: Option<Py<PyAny>>,
    event: Option<Py<PyAny>>,
    cancelled: bool,
    retired: bool,
    unretired: Vec<Py<PyAny>>,
    work: Option<Py<PyAny>>,
    access: ReadAccess,
    done_callbacks: Vec<Py<PyAny>>,
    retirement_callbacks: Vec<Py<PyAny>>,
}

impl TicketState {
    fn ready(&self) -> bool {
        self.value.is_some() || self.error.is_some()
    }
}

/// One read's stream readiness and physical retirement. Cancellation revokes
/// consumption; it cannot prove that a submitted device copy has stopped.
#[pyclass(frozen, weakref, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct TransferTicket {
    events: Py<PyAny>,
    state: Mutex<TicketState>,
}

#[pymethods]
impl TransferTicket {
    #[new]
    #[pyo3(signature = (event_pool, *, release=None))]
    fn new(event_pool: Py<PyAny>, release: Option<Py<PyAny>>) -> Self {
        let access = match release {
            Some(release) => ReadAccess::Borrowed {
                release: Some(release),
                streams: HashMap::new(),
                events: Vec::new(),
                closed: false,
            },
            None => ReadAccess::Copy {
                destination_stream: None,
            },
        };
        Self {
            events: event_pool,
            state: Mutex::new(TicketState {
                value: None,
                error: None,
                event: None,
                cancelled: false,
                retired: false,
                unretired: Vec::new(),
                work: None,
                access,
                done_callbacks: Vec::new(),
                retirement_callbacks: Vec::new(),
            }),
        }
    }

    fn ready(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(self.lock(py)?.ready())
    }

    pub(crate) fn retired(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(self.lock(py)?.retired)
    }

    pub(crate) fn retirement_ready(&self, py: Python<'_>) -> PyResult<bool> {
        let state = self.lock(py)?;
        if !state.unretired.is_empty() {
            let error = resource(py, "transfer physical completion is unknown");
            if let Some(cause) = &state.error {
                error.set_cause(py, Some(PyErr::from_value(cause.bind(py).clone())));
            }
            return Err(error);
        }
        Ok(state.retired)
    }

    fn add_done_callback(&self, py: Python<'_>, callback: Py<PyAny>) -> PyResult<()> {
        let mut state = self.lock(py)?;
        if state.ready() {
            drop(state);
            notify(py, vec![callback]);
        } else {
            state.done_callbacks.push(callback);
        }
        Ok(())
    }

    fn add_retirement_callback(&self, py: Python<'_>, callback: Py<PyAny>) -> PyResult<()> {
        let mut state = self.lock(py)?;
        if state.retired {
            drop(state);
            notify(py, vec![callback]);
        } else {
            state.retirement_callbacks.push(callback);
        }
        Ok(())
    }

    pub(crate) fn cancel(&self, py: Python<'_>) -> PyResult<()> {
        let mut state = self.lock(py)?;
        let pending = !state.ready();
        state.cancelled = true;
        if state.error.is_none() {
            state.error = Some(
                resource(py, "transfer read was cancelled")
                    .into_value(py)
                    .into_any(),
            );
        }
        let callbacks = if pending {
            std::mem::take(&mut state.done_callbacks)
        } else {
            Vec::new()
        };
        let work = state.work.as_ref().map(|work| work.clone_ref(py));
        drop(state);
        notify(py, callbacks);
        if let Some(work) = work {
            work.bind(py).call_method0("cancel")?;
        }
        Ok(())
    }

    #[pyo3(name = "_require_active")]
    fn require_active(&self, py: Python<'_>) -> PyResult<()> {
        let state = self.lock(py)?;
        if state.cancelled {
            return Err(state
                .error
                .as_ref()
                .map(|error| PyErr::from_value(error.bind(py).clone()))
                .unwrap_or_else(|| invariant(py, "cancelled transfer has no error")));
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
        if !state.ready() {
            return Err(PyRuntimeError::new_err(
                "transfer ticket was observed before readiness",
            ));
        }
        if let Some(error) = &state.error {
            return Err(PyErr::from_value(error.bind(py).clone()));
        }
        if matches!(state.access, ReadAccess::Borrowed { closed: true, .. }) {
            return Err(PyRuntimeError::new_err(
                "transfer consumption has already closed",
            ));
        }
        let value = state
            .value
            .as_ref()
            .ok_or_else(|| invariant(py, "ready transfer has no views"))?
            .clone_ref(py);
        if let Some(event) = &state.event {
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
            consumer.call_method1("wait_event", (event,))?;
            for span in spans {
                span.call_method1("record_stream", (&consumer,))?;
            }
            if let ReadAccess::Borrowed { streams, .. } = &mut state.access {
                streams.insert(
                    consumer.getattr("cuda_stream")?.extract()?,
                    consumer.unbind(),
                );
            }
        }
        Ok(value)
    }

    /// Borrowed storage stays granted until every observed consumer stream
    /// completes. Copy reads keep their ordinary destination tensor ownership.
    fn close(slf: Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        let owner = slf.get();
        let streams = {
            let mut state = owner.lock(py)?;
            match &mut state.access {
                ReadAccess::Borrowed {
                    streams,
                    release: Some(_),
                    closed,
                    ..
                } if !*closed => {
                    *closed = true;
                    streams
                        .values()
                        .map(|stream| stream.clone_ref(py))
                        .collect::<Vec<_>>()
                }
                _ => return Ok(()),
            }
        };
        let events = record_consumers(py, &owner.events, streams)?;
        if events.is_empty() {
            return owner.events_released(py);
        }
        if let ReadAccess::Borrowed {
            events: retained, ..
        } = &mut owner.lock(py)?.access
        {
            retained.extend(events.iter().map(|event| event.clone_ref(py)));
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("completed", slf.getattr("events_released")?)?;
        owner.events.bind(py).call_method(
            "defer_release",
            (PyTuple::new(py, events)?, &slf),
            Some(&kwargs),
        )?;
        Ok(())
    }

    fn events_released(&self, py: Python<'_>) -> PyResult<()> {
        let release = {
            let mut state = self.lock(py)?;
            match &mut state.access {
                ReadAccess::Borrowed {
                    release,
                    streams,
                    events,
                    ..
                } => {
                    streams.clear();
                    events.clear();
                    release.take()
                }
                _ => None,
            }
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
        let events = match &owner.lock(py)?.access {
            ReadAccess::Borrowed { events, .. } => events
                .iter()
                .map(|event| event.clone_ref(py))
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        };
        for event in events {
            event.bind(py).call_method0("synchronize")?;
        }
        owner.events.bind(py).call_method0("reap")?;
        Ok(())
    }

    #[pyo3(name = "_complete", signature = (value, event=None))]
    fn complete(&self, py: Python<'_>, value: Py<PyAny>, event: Option<Py<PyAny>>) -> PyResult<()> {
        let mut state = self.lock(py)?;
        if let Some(event) = event {
            let device = tensor_spans(value.bind(py))?[0].getattr("device")?;
            self.events
                .bind(py)
                .call_method1("retain", (&event, device))?;
            state.event = Some(event);
        }
        let callbacks = if !state.ready() {
            state.value = Some(value);
            std::mem::take(&mut state.done_callbacks)
        } else {
            Vec::new()
        };
        drop(state);
        notify(py, callbacks);
        Ok(())
    }

    #[pyo3(name = "_fail")]
    fn fail(&self, py: Python<'_>, error: Py<PyAny>) -> PyResult<bool> {
        let mut state = self.lock(py)?;
        let late = state.value.is_some();
        let callbacks = if !state.ready() {
            std::mem::take(&mut state.done_callbacks)
        } else {
            Vec::new()
        };
        state.error = Some(error);
        drop(state);
        notify(py, callbacks);
        Ok(late)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.events)?;
        if let Some(state) = gc_lock(&self.state) {
            visit.call(&state.value)?;
            visit.call(&state.error)?;
            visit.call(&state.event)?;
            visit.call(&state.work)?;
            for value in state
                .unretired
                .iter()
                .chain(&state.done_callbacks)
                .chain(&state.retirement_callbacks)
            {
                visit.call(value)?;
            }
            match &state.access {
                ReadAccess::Copy { destination_stream } => visit.call(destination_stream)?,
                ReadAccess::Borrowed {
                    release,
                    streams,
                    events,
                    ..
                } => {
                    visit.call(release)?;
                    for value in streams.values().chain(events) {
                        visit.call(value)?;
                    }
                }
            }
        }
        Ok(())
    }
}

impl TransferTicket {
    fn lock(&self, py: Python<'_>) -> PyResult<MutexGuard<'_, TicketState>> {
        self.state
            .lock_py_attached(py)
            .map_err(|_| invariant(py, "transfer ticket lock is poisoned"))
    }

    fn retire(&self, py: Python<'_>) -> PyResult<()> {
        let mut state = self.lock(py)?;
        if state.retired || !state.unretired.is_empty() {
            return Err(invariant(
                py,
                "transfer cannot retire twice or with unknown physical completion",
            ));
        }
        state.retired = true;
        let callbacks = std::mem::take(&mut state.retirement_callbacks);
        drop(state);
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
                if let ReadAccess::Borrowed {
                    release,
                    streams,
                    closed: false,
                    ..
                } = &mut state.access
                    && let Some(release) = release.take()
                {
                    let streams = std::mem::take(streams).into_values().collect();
                    let events = record_consumers(py, &self.events, streams)?;
                    let callbacks = std::mem::take(&mut state.retirement_callbacks);
                    if events.is_empty() {
                        release.bind(py).call0()?;
                        notify(py, callbacks);
                    } else {
                        let retained = Mutex::new(Some((release, callbacks)));
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
                                if let Some((release, callbacks)) = completion {
                                    release.bind(py).call0()?;
                                    notify(py, callbacks);
                                }
                                Ok(())
                            },
                        )?;
                        let kwargs = PyDict::new(py);
                        kwargs.set_item("completed", complete)?;
                        self.events.bind(py).call_method(
                            "defer_release",
                            (PyTuple::new(py, events)?, &state.value),
                            Some(&kwargs),
                        )?;
                    }
                }
                if let Some(event) = state.event.take() {
                    self.events
                        .bind(py)
                        .call_method1("defer_release", ((event,), &state.value))?;
                }
                Ok(())
            })();
            if let Err(error) = cleanup {
                error.write_unraisable(py, None);
            }
        });
    }
}

struct PoolState {
    error: Option<Py<PyAny>>,
    wake: Option<Py<PyAny>>,
    unretired: Vec<Py<TransferTicket>>,
    streams: HashMap<(std::thread::ThreadId, String), Py<PyAny>>,
}

/// Execute a backend's reads against the rank's shared credits. The standard
/// Python thread executor supplies threads; Rust owns submission, cancellation,
/// stream ordering, failure classification, and credit retirement.
#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct TransferPool {
    executor: Py<PyAny>,
    capacity: Py<TransferCapacity>,
    events: Py<PyAny>,
    state: Mutex<PoolState>,
}

#[pymethods]
impl TransferPool {
    #[new]
    #[pyo3(signature = (*, workers, capacity, name, event_pool))]
    fn new(
        py: Python<'_>,
        workers: isize,
        capacity: Py<TransferCapacity>,
        name: &str,
        event_pool: Py<PyAny>,
    ) -> PyResult<Self> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("max_workers", workers)?;
        kwargs.set_item("thread_name_prefix", name)?;
        let executor = py
            .import("concurrent.futures")?
            .getattr("ThreadPoolExecutor")?
            .call((), Some(&kwargs))?
            .unbind();
        Ok(Self {
            executor,
            capacity,
            events: event_pool,
            state: Mutex::new(PoolState {
                error: None,
                wake: None,
                unretired: Vec::new(),
                streams: HashMap::new(),
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
            if let Some(error) = &state.error {
                return Err(PyErr::from_value(error.bind(py).clone()));
            }
        }
        if let Some(reservation) = reservation {
            if !reservation
                .get()
                .capacity
                .bind(py)
                .is(owner.capacity.bind(py))
            {
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
        let submitted = (|| -> PyResult<Py<TransferTicket>> {
            let ticket = Py::new(py, TransferTicket::new(owner.events.clone_ref(py), None))?;
            if let Some(destination) = destination {
                let spans = tensor_spans(&destination)?;
                if spans[0].getattr("is_cuda")?.extract::<bool>()? {
                    let stream = py
                        .import("torch")?
                        .getattr("cuda")?
                        .call_method1("current_stream", (spans[0].getattr("device")?,))?
                        .unbind();
                    ticket.get().lock(py)?.access = ReadAccess::Copy {
                        destination_stream: Some(stream),
                    };
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
            let completion = Mutex::new(Some((slf.clone().unbind(), ticket.clone_ref(py))));
            let finished = PyCFunction::new_closure(
                py,
                None,
                None,
                move |args: &Bound<'_, PyTuple>, _: Option<&Bound<'_, PyDict>>| -> PyResult<()> {
                    let py = args.py();
                    let finished = completion
                        .lock()
                        .unwrap_or_else(|error| error.into_inner())
                        .take();
                    if let Some((pool, ticket)) = finished {
                        pool.get()
                            .finished(py, &ticket, nbytes, &args.get_item(0)?)?;
                    }
                    Ok(())
                },
            )?;
            let work = owner.executor.bind(py).call_method1("submit", (run,))?;
            let cancelled = {
                let mut state = ticket.get().lock(py)?;
                state.work = Some(work.clone().unbind());
                state.cancelled
            };
            if cancelled {
                work.call_method0("cancel")?;
            }
            work.call_method1("add_done_callback", (finished,))?;
            Ok(ticket)
        })();
        if submitted.is_err() {
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
        producer: Option<Bound<'_, PyAny>>,
        acknowledgment: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        ticket.get().require_active(py)?;
        let spans = tensor_spans(&destination)?;
        let device = spans[0].getattr("device")?;
        let numerical = py.import("uniserve_worker.transport.pool")?;
        if device.getattr("type")?.extract::<String>()? != "cuda" {
            copy_acknowledgment(py, acknowledgment.as_ref(), "CLAIMED", false)?;
            numerical
                .getattr("_copy_tensors")?
                .call1((&source, &destination, py.None()))?;
            copy_acknowledgment(py, acknowledgment.as_ref(), "ACKNOWLEDGED", false)?;
            return ticket.get().complete(py, destination.unbind(), None);
        }
        let cuda = py.import("torch")?.getattr("cuda")?;
        let key = (
            std::thread::current().id(),
            device.str()?.to_str()?.to_owned(),
        );
        let stream = {
            let mut state = self.lock(py)?;
            match state.streams.get(&key) {
                Some(stream) => stream.clone_ref(py),
                None => {
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("device", &device)?;
                    let stream = cuda.getattr("Stream")?.call((), Some(&kwargs))?.unbind();
                    state.streams.insert(key, stream.clone_ref(py));
                    stream
                }
            }
        };
        let mut completed = None;
        let copied = (|| -> PyResult<()> {
            with_context(&cuda.call_method1("device", (&device,))?, || {
                with_context(&cuda.call_method1("stream", (&stream,))?, || {
                    // Claim on the idle read stream before waiting on either
                    // side of the handoff, so the producer cannot reuse it.
                    copy_acknowledgment(py, acknowledgment.as_ref(), "CLAIMED", false)?;
                    let destination_stream = match &mut ticket.get().lock(py)?.access {
                        ReadAccess::Copy { destination_stream } => destination_stream.take(),
                        ReadAccess::Borrowed { .. } => {
                            return Err(invariant(
                                py,
                                "a borrowed transfer cannot copy into storage",
                            ));
                        }
                    };
                    if let Some(destination_stream) = destination_stream {
                        stream
                            .bind(py)
                            .call_method1("wait_stream", (destination_stream,))?;
                    }
                    if let Some(producer) = &producer {
                        stream.bind(py).call_method1("wait_event", (producer,))?;
                    }
                    numerical
                        .getattr("_copy_tensors")?
                        .call1((&source, &destination, &stream))?;
                    copy_acknowledgment(py, acknowledgment.as_ref(), "ACKNOWLEDGED", true)?;
                    let event = self.events.bind(py).call_method1("acquire", (&device,))?;
                    self.events
                        .bind(py)
                        .call_method1("record", (&event, &device))?;
                    completed = Some(event.unbind());
                    Ok(())
                })
            })?;
            ticket.get().complete(
                py,
                destination.clone().unbind(),
                completed.as_ref().map(|event| event.clone_ref(py)),
            )?;
            if let Some(event) = &completed {
                event.bind(py).call_method0("synchronize")?;
            }
            Ok(())
        })();
        if let Err(error) = &copied {
            ticket
                .get()
                .fail(py, error.clone_ref(py).into_value(py).into_any())?;
        }
        if let Err(error) = stream.bind(py).call_method0("synchronize") {
            // Unknown physical completion is not cancellation. Retain every
            // allocation and fence, and leave its byte/read credits occupied.
            let mut state = ticket.get().lock(py)?;
            state
                .unretired
                .extend([source.unbind(), destination.unbind(), stream]);
            state.unretired.extend(producer.map(Bound::unbind));
            state.unretired.extend(completed);
            if let Err(cause) = copied {
                error.set_cause(py, Some(cause));
            }
            return Err(error);
        }
        copied
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("wait", true)?;
        kwargs.set_item("cancel_futures", false)?;
        self.executor
            .bind(py)
            .call_method("shutdown", (), Some(&kwargs))?;
        let (streams, error) = {
            let mut state = self.lock(py)?;
            (
                std::mem::take(&mut state.streams),
                state.error.as_ref().map(|error| error.clone_ref(py)),
            )
        };
        drop(streams);
        if let Some(error) = error {
            return Err(PyErr::from_value(error.into_bound(py)));
        }
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.executor)?;
        visit.call(&self.capacity)?;
        visit.call(&self.events)?;
        if let Some(state) = gc_lock(&self.state) {
            visit.call(&state.error)?;
            visit.call(&state.wake)?;
            for ticket in &state.unretired {
                visit.call(ticket)?;
            }
            for stream in state.streams.values() {
                visit.call(stream)?;
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
        let result = (|| -> PyResult<()> {
            ticket.get().require_active(py)?;
            with_context(&py.import("torch")?.call_method0("inference_mode")?, || {
                let mut arguments = vec![ticket.bind(py).as_any().clone()];
                arguments.extend(args.bind(py).iter());
                call.bind(py).call1(PyTuple::new(py, arguments)?)?;
                Ok(())
            })
        })();
        if let Err(error) = result {
            let error = error.into_value(py).into_any();
            let late = ticket.get().fail(py, error.clone_ref(py))?;
            let undrained = !ticket.get().lock(py)?.unretired.is_empty();
            if late || undrained {
                let mut state = self.lock(py)?;
                if state.error.is_none() {
                    state.error = Some(error);
                }
                if undrained {
                    state.unretired.push(ticket.clone_ref(py));
                }
                let wake = late
                    .then(|| state.wake.as_ref().map(|wake| wake.clone_ref(py)))
                    .flatten();
                drop(state);
                if let Some(wake) = wake {
                    notify(py, vec![wake]);
                }
            }
        }
        Ok(())
    }

    fn finished(
        &self,
        py: Python<'_>,
        ticket: &Py<TransferTicket>,
        nbytes: i64,
        work: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let undrained = {
            let mut state = ticket.get().lock(py)?;
            state.work = None;
            !state.unretired.is_empty()
        };
        if !undrained {
            self.capacity.get().release(py, nbytes)?;
            self.capacity.get().return_reads(py, 1)?;
            ticket.get().retire(py)?;
        }
        if work.call_method0("cancelled")?.extract::<bool>()? {
            ticket.get().fail(
                py,
                resource(py, "transfer read was cancelled before submission")
                    .into_value(py)
                    .into_any(),
            )?;
        }
        Ok(())
    }
}

fn copy_acknowledgment(
    py: Python<'_>,
    word: Option<&Bound<'_, PyAny>>,
    state: &str,
    non_blocking: bool,
) -> PyResult<()> {
    if let Some(word) = word {
        let value = py
            .import("uniserve_worker.transport.vmm_pool")?
            .getattr(state)?;
        let source = py
            .import("uniserve_worker.transport.pool")?
            .getattr("chunk_word")?
            .call1((value,))?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("non_blocking", non_blocking)?;
        word.call_method("copy_", (source,), Some(&kwargs))?;
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

fn record_consumers(
    py: Python<'_>,
    pool: &Py<PyAny>,
    streams: Vec<Py<PyAny>>,
) -> PyResult<Vec<Py<PyAny>>> {
    let cuda = py.import("torch")?.getattr("cuda")?;
    let mut events = Vec::new();
    for stream in streams {
        let device = stream.bind(py).getattr("device")?;
        with_context(&cuda.call_method1("device", (&device,))?, || {
            with_context(&cuda.call_method1("stream", (&stream,))?, || {
                let event = pool.bind(py).call_method1("acquire", (&device,))?;
                pool.bind(py).call_method1("retain", (&event, &device))?;
                pool.bind(py).call_method1("record", (&event, &device))?;
                pool.bind(py)
                    .call_method1("schedule_completion_wake", (&device, &event))?;
                events.push(event.unbind());
                Ok(())
            })
        })?;
    }
    Ok(events)
}

/// Enter PyTorch's thread-local inference/device/stream scopes. These contexts
/// restore their prior state on either exit and do not suppress exceptions.
fn with_context<T>(
    context: &Bound<'_, PyAny>,
    operation: impl FnOnce() -> PyResult<T>,
) -> PyResult<T> {
    let py = context.py();
    context.call_method0("__enter__")?;
    match operation() {
        Ok(value) => {
            context.call_method1("__exit__", (py.None(), py.None(), py.None()))?;
            Ok(value)
        }
        Err(error) => {
            context.call_method1(
                "__exit__",
                (error.get_type(py), error.value(py), error.traceback(py)),
            )?;
            Err(error)
        }
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
