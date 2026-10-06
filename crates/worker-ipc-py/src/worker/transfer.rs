//! Bounded asynchronous reads shared by tensor, KV, and latent transports.
//!
//! Consumable views and physical retirement are separate completions. Credits
//! return only after the backend stops accessing source and destination storage.

mod cuda;
mod ticket;

use ticket::BorrowedReads;
pub(crate) use ticket::{TransferRef, TransferTicket};

use std::sync::{Arc, Mutex, MutexGuard, TryLockError};
use std::time::Duration;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{PyDict, PyTuple};
use uniserve_worker::cuda::DeviceGuard;
use uniserve_worker::{
    ReadBackend, ReadReservation as NativeReadReservation,
    TransferCapacity as NativeTransferCapacity, TransferPool as NativeTransferPool,
    TransferTicket as NativeTransferTicket,
};

use super::error::{invalid, native_error, resource};
use super::events::{CUDAEvent, EventPool};
use super::host::with_context;
use super::registry::Registry;
use super::shared_buffer::SharedRead;

/// Python access to the rank's shared native transfer budget.
#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct TransferCapacity {
    pub(super) inner: Arc<NativeTransferCapacity<Py<PyAny>>>,
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
    pub(super) fn new(
        py: Python<'_>,
        capacity: Py<TransferCapacity>,
        count: isize,
    ) -> PyResult<Self> {
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

    pub(super) fn close(&self, py: Python<'_>) -> PyResult<()> {
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

/// Execute a backend's reads against the rank's shared credits. The native host
/// lane supplies threads; the pool orders copy streams and retires read credits.
#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct TransferPool {
    pool: Arc<NativeTransferPool<PythonRead>>,
    borrowed: Arc<BorrowedReads>,
    capacity: Py<TransferCapacity>,
    events: Py<EventPool>,
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
        let pool = NativeTransferPool::new(Arc::clone(&capacity.get().inner), workers, name)
            .map_err(|error| native_error(py, error))?;
        Ok(Self {
            pool: Arc::new(pool),
            borrowed: Arc::new(Mutex::new(Some(Default::default()))),
            capacity,
            events: event_pool,
        })
    }

    fn set_completion_wake(&self, wake: Option<Py<PyAny>>) {
        self.pool.set_completion_wake(wake);
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
        let ticket = Py::new(
            slf.py(),
            TransferTicket::new(slf.get().events.clone_ref(slf.py()), None),
        )?;
        let read = PythonRead {
            ticket,
            call: ReadCall::Python {
                call,
                args: args.clone().unbind(),
            },
        };
        Self::submit_read(&slf, read, nbytes, destination, reservation)
    }

    /// Grant and submit a local read without a Python ownership callback.
    #[pyo3(signature = (locator, *, device, destination=None, region=None, reservation=None))]
    fn fetch_local(
        slf: Bound<'_, Self>,
        locator: &Bound<'_, PyAny>,
        device: &Bound<'_, PyAny>,
        destination: Option<Bound<'_, PyAny>>,
        region: Option<Bound<'_, PyAny>>,
        reservation: Option<Py<ReadReservation>>,
    ) -> PyResult<Py<TransferTicket>> {
        let py = slf.py();
        let owner = slf.get();
        if let Some(error) = owner.pool.error() {
            return Err(PyErr::from_value(error.bind(py).clone()));
        }

        if locator.getattr("backend")?.extract::<String>()? != "local" {
            return Err(invalid(py, "local read requires a local locator"));
        }
        let (buffer, source) = Registry::acquire_read(locator)?;
        let borrowed = destination.is_none();
        let events = if borrowed {
            source.events(py)
        } else {
            owner.events.clone_ref(py)
        };
        let borrower = borrowed.then(|| Arc::downgrade(&owner.borrowed));
        let ticket = Py::new(py, TransferTicket::new(events, borrower))?;
        ticket.get().bind_source(py, source)?;

        let submitted = (|| -> PyResult<()> {
            let tensor = buffer.get().tensor(py)?;
            let producer = buffer.get().event(py);
            let (tensor, target): (Py<PyAny>, Option<Py<PyAny>>) = py
                .import("uniserve_worker.transport.local")?
                .getattr("_read_views")?
                .call1((
                    tensor,
                    locator,
                    device,
                    destination.as_ref(),
                    region.as_ref(),
                ))?
                .extract()?;

            if let Some(target) = target {
                let read = PythonRead {
                    ticket: ticket.clone_ref(py),
                    call: ReadCall::Copy {
                        pool: slf.clone().unbind(),
                        source: tensor,
                        destination: target.clone_ref(py),
                        producer,
                    },
                };
                Self::submit_read(
                    &slf,
                    read,
                    locator.getattr("nbytes")?.extract()?,
                    Some(target.into_bound(py)),
                    reservation,
                )?;
            } else {
                owner
                    .capacity
                    .get()
                    .inner
                    .take_read(reservation.as_ref().map(|value| &value.get().inner))
                    .map_err(|error| {
                        TransferCapacity::read_error(
                            owner.capacity.bind(py),
                            error,
                            "local borrowed-view ticket capacity is exhausted",
                        )
                    })?;
                ticket.get().lock(py)?.credit = Some(owner.capacity.clone_ref(py));
                {
                    let mut borrowed = owner
                        .borrowed
                        .lock_py_attached(py)
                        .unwrap_or_else(|error| error.into_inner());
                    let borrowed = borrowed
                        .as_mut()
                        .ok_or_else(|| resource(py, "transfer pool is closed"))?;
                    borrowed.insert(
                        Arc::as_ptr(&ticket.get().inner) as usize,
                        Arc::downgrade(&ticket.get().inner),
                    );
                }

                if let Some(wake) = owner.pool.completion_wake() {
                    ticket.get().add_done_callback(py, wake.clone_ref(py))?;
                    ticket
                        .get()
                        .add_retirement_callback(py, wake.clone_ref(py))?;
                }
                ticket.get().complete(py, tensor, producer)?;
            }
            Ok(())
        })();
        if let Err(error) = submitted {
            if let Err(cleanup) = ticket.get().inner.release_source(py) {
                cleanup.write_unraisable(py, None);
            }
            return Err(error);
        }
        Ok(ticket)
    }

    /// Submit a shared-memory read on the same lane as local copies. A private
    /// host tensor ends the source claim before destination DMA begins.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (locator, *, node, slot, device, destination=None, region=None, reservation=None))]
    fn fetch_shared(
        slf: Bound<'_, Self>,
        locator: Bound<'_, PyAny>,
        node: &str,
        slot: usize,
        device: Py<PyAny>,
        destination: Option<Bound<'_, PyAny>>,
        region: Option<Py<PyAny>>,
        reservation: Option<Py<ReadReservation>>,
    ) -> PyResult<Py<TransferTicket>> {
        let py = slf.py();
        if locator
            .getattr("source")?
            .getattr("node")?
            .extract::<String>()?
            != node
        {
            return Err(invalid(
                py,
                "shared storage transport requires the source node",
            ));
        }
        if locator.getattr("backend")?.extract::<String>()? != "shm" {
            return Err(invalid(py, "shared storage read requires a SHM locator"));
        }

        // Supplied destinations must be valid before recording their handoff.
        // Otherwise allocation stays on the read thread, after admission.
        let destination = destination
            .map(|destination| {
                py.import("uniserve_worker.transport.layout")?
                    .getattr("read_destination")?
                    .call1((&locator, &device, destination, region.as_ref()))
            })
            .transpose()?;
        let nbytes = locator.getattr("nbytes")?.extract()?;
        let ticket = Py::new(
            py,
            TransferTicket::new(slf.get().events.clone_ref(py), None),
        )?;
        let read = PythonRead {
            ticket,
            call: ReadCall::Shared {
                pool: slf.clone().unbind(),
                locator: locator.unbind(),
                slot,
                device,
                destination: destination.as_ref().map(|value| value.clone().unbind()),
                region,
            },
        };
        Self::submit_read(&slf, read, nbytes, destination, reservation)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (locator, *, source, slot, device, destination=None, region=None, reservation=None))]
    fn fetch_cuda(
        slf: Bound<'_, Self>,
        locator: Bound<'_, PyAny>,
        source: Bound<'_, PyAny>,
        slot: usize,
        device: Py<PyAny>,
        destination: Option<Bound<'_, PyAny>>,
        region: Option<Py<PyAny>>,
        reservation: Option<Py<ReadReservation>>,
    ) -> PyResult<Py<TransferTicket>> {
        cuda::CudaRead::submit(
            &slf,
            locator,
            source,
            slot,
            device,
            destination,
            region,
            reservation,
        )
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
        let borrowed = self
            .borrowed
            .lock_py_attached(py)
            .unwrap_or_else(|error| error.into_inner())
            .take()
            .unwrap_or_default();
        let mut result = py.detach(|| self.pool.close()).map_or(Ok(()), |error| {
            Err(PyErr::from_value(error.bind(py).clone()))
        });
        for read in borrowed.into_values().filter_map(|read| read.upgrade()) {
            result = result.and(read.drain(py));
        }
        result
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.capacity)?;
        visit.call(&self.events)?;
        self.pool
            .visit(|error, wake, unretired| {
                visit.call(error)?;
                visit.call(wake)?;
                for read in unretired {
                    visit.call(&read.ticket)?;
                    read.call.visit(&visit)?;
                }
                Ok(())
            })
            .unwrap_or(Ok(()))
    }
}

impl TransferPool {
    fn submit_read(
        owner: &Bound<'_, Self>,
        read: PythonRead,
        nbytes: i64,
        destination: Option<Bound<'_, PyAny>>,
        reservation: Option<Py<ReadReservation>>,
    ) -> PyResult<Py<TransferTicket>> {
        let py = owner.py();
        let owner_ref = owner.get();
        if let Some(error) = owner_ref.pool.error() {
            return Err(PyErr::from_value(error.bind(py).clone()));
        }
        let nbytes = u64::try_from(nbytes)
            .map_err(|_| PyValueError::new_err("transfer byte reservation must not be negative"))?;
        let ticket = read.ticket.clone_ref(py);
        let reservation = reservation.as_ref().map(|value| &value.get().inner);
        let task = py
            .detach(|| owner_ref.pool.reserve(read, nbytes, reservation))
            .map_err(|error| {
                TransferCapacity::read_error(
                    owner_ref.capacity.bind(py),
                    error,
                    "asynchronous transfer ticket capacity is exhausted",
                )
            })?;

        let submitted = (|| -> PyResult<()> {
            ticket.get().lock(py)?.work = Some(Arc::downgrade(&task));
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
                    let pool = owner_ref.events.borrow(py);
                    pool.reap(py)?;
                    let mut state = ticket.get().lock(py)?;
                    pool.with_pool(|pool| state.ticket.prepare_copy(pool, device, stream))?;
                }
            }

            if let Some(wake) = owner_ref.pool.completion_wake() {
                ticket.get().add_done_callback(py, wake.clone_ref(py))?;
                ticket
                    .get()
                    .add_retirement_callback(py, wake.clone_ref(py))?;
            }
            py.detach(|| task.submit())
                .map_err(|error| native_error(py, error))
        })();
        if let Err(error) = submitted {
            if let Err(cleanup) = py.detach(|| task.cancel(true)) {
                PyErr::from_value(cleanup.into_bound(py)).write_unraisable(py, None);
            }
            return Err(error);
        }
        Ok(ticket)
    }
}

struct PythonRead {
    ticket: Py<TransferTicket>,
    call: ReadCall,
}

enum ReadCall {
    Cuda(cuda::CudaRead),
    Python {
        call: Py<PyAny>,
        args: Py<PyTuple>,
    },
    Copy {
        pool: Py<TransferPool>,
        source: Py<PyAny>,
        destination: Py<PyAny>,
        producer: Option<Py<CUDAEvent>>,
    },
    Shared {
        pool: Py<TransferPool>,
        locator: Py<PyAny>,
        slot: usize,
        device: Py<PyAny>,
        destination: Option<Py<PyAny>>,
        region: Option<Py<PyAny>>,
    },
}

impl ReadCall {
    fn visit(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        match self {
            Self::Cuda(read) => read.visit(visit)?,
            Self::Python { call, args } => {
                visit.call(call)?;
                visit.call(args)?;
            }
            Self::Copy {
                pool,
                source,
                destination,
                producer,
            } => {
                visit.call(pool)?;
                visit.call(source)?;
                visit.call(destination)?;
                visit.call(producer)?;
            }
            Self::Shared {
                pool,
                locator,
                device,
                destination,
                region,
                ..
            } => {
                visit.call(pool)?;
                visit.call(locator)?;
                visit.call(device)?;
                visit.call(destination)?;
                visit.call(region)?;
            }
        }
        Ok(())
    }
}

impl ReadBackend for PythonRead {
    type Value = Py<PyAny>;
    type Error = Py<PyAny>;
    type Callback = Py<PyAny>;

    fn with_ticket<T>(
        &self,
        action: impl FnOnce(&mut NativeTransferTicket<Self::Value, Self::Error, Self::Callback>) -> T,
    ) -> T {
        let mut state = self
            .ticket
            .get()
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let result = action(&mut state.ticket);
        let ready = state.ticket.ready();
        drop(state);
        if ready {
            self.ticket.get().inner.ready.notify_all();
        }
        result
    }

    fn read(&self) -> Result<(), Self::Error> {
        Python::attach(|py| {
            let run = || -> PyResult<()> {
                with_context(
                    &py.import("torch")?.call_method0("inference_mode")?,
                    || match &self.call {
                        ReadCall::Cuda(read) => read.run(py, &self.ticket),
                        ReadCall::Python { call, args } => {
                            let mut arguments = vec![self.ticket.bind(py).as_any().clone()];
                            arguments.extend(args.bind(py).iter());
                            call.bind(py).call1(PyTuple::new(py, arguments)?)?;
                            Ok(())
                        }
                        ReadCall::Copy {
                            pool,
                            source,
                            destination,
                            producer,
                        } => pool.get().copy(
                            py,
                            self.ticket.clone_ref(py),
                            source.bind(py).clone(),
                            destination.bind(py).clone(),
                            producer.as_ref().map(|event| event.clone_ref(py)),
                            None,
                        ),
                        ReadCall::Shared {
                            pool,
                            locator,
                            slot,
                            device,
                            destination,
                            region,
                        } => {
                            let locator = locator.bind(py);
                            let name = locator.getattr("transport")?.getattr("name")?;
                            let read = Py::new(
                                py,
                                SharedRead::open(
                                    py,
                                    name.extract()?,
                                    locator.getattr("nbytes")?.extract()?,
                                    *slot,
                                    0,
                                    Duration::from_secs(120),
                                    || self.ticket.get().check_active(),
                                )?,
                            )?;

                            // The numerical call makes a private host copy.
                            // Release the source claim even if allocation or
                            // conversion fails; no device reads it directly.
                            let source = py
                                .import("uniserve_worker.transport.shm")?
                                .getattr("_copy_payload")?
                                .call1((&read, locator, device))
                                .map(Bound::unbind);
                            read.get().release(py)?;
                            let source = source?.into_bound(py);
                            let source = match region {
                                Some(region) => source.get_item(region)?,
                                None => source,
                            };
                            let target = match destination {
                                Some(destination) => destination.bind(py).clone(),
                                None => py
                                    .import("uniserve_worker.transport.layout")?
                                    .getattr("read_destination")?
                                    .call1((locator, device, py.None(), region.as_ref()))?,
                            };
                            pool.get().copy(
                                py,
                                self.ticket.clone_ref(py),
                                source,
                                target,
                                None,
                                None,
                            )
                        }
                    },
                )
            };
            run().map_err(|error| error.into_value(py).into_any())
        })
    }

    fn release(&self) -> Result<(), Self::Error> {
        Python::attach(|py| {
            self.ticket
                .get()
                .inner
                .release_source(py)
                .map_err(|error| error.into_value(py).into_any())
        })
    }

    fn notify(callbacks: Vec<Self::Callback>) {
        notify_capacity(callbacks);
    }

    fn wake(callback: &Self::Callback) {
        Python::try_attach(|py| {
            if let Err(error) = callback.bind(py).call0() {
                error.write_unraisable(py, Some(callback.bind(py)));
            }
        });
    }

    fn error(error: uniserve_worker::Error) -> Self::Error {
        Python::attach(|py| native_error(py, error).into_value(py).into_any())
    }

    fn report(error: Self::Error) {
        Python::try_attach(|py| PyErr::from_value(error.into_bound(py)).write_unraisable(py, None));
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
