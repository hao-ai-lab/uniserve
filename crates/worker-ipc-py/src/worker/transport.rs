//! Physical tensor exports and reads through one rank's native resources.

mod cuda;

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use super::completion::Completion;
use super::error::{invalid, native_error};
use super::events::{CUDAEvent, EventPool};
use super::registry::{BufferRegistry, TransportBuffer};
use super::shared_buffer::{SharedBuffer, SharedRead};
use super::transfer::{ReadReservation, TransferCapacity, TransferPool, TransferTicket};

enum Backend {
    Local {
        buffers: Py<BufferRegistry>,
        next: AtomicU64,
    },
    Shared {
        buffers: Py<BufferRegistry>,
        slot: usize,
        host_slots: HashSet<usize>,
    },
    Cuda(cuda::CudaExports),
    Channel {
        endpoint: String,
        host_slots: HashSet<usize>,
    },
}

/// Physical exports share byte capacity, read execution and retirement.
/// Python supplies borrowed tensor views and numerical copies.
#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct Transport {
    #[pyo3(get)]
    source: Py<PyAny>,
    #[pyo3(get)]
    pub(super) capacity: Py<TransferCapacity>,
    events: Py<EventPool>,
    reads: Py<TransferPool>,
    backend: Backend,
}

#[pymethods]
impl Transport {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (name, *, capacity, event_pool, source, acknowledgment_slot=0, host_slots=Vec::new(), cross_host_consumers=false))]
    fn new(
        py: Python<'_>,
        name: &str,
        capacity: Py<TransferCapacity>,
        event_pool: Py<EventPool>,
        source: Py<PyAny>,
        acknowledgment_slot: usize,
        host_slots: Vec<usize>,
        cross_host_consumers: bool,
    ) -> PyResult<Self> {
        let registry = |limit| Py::new(py, BufferRegistry::new(limit, event_pool.clone_ref(py))?);
        let backend = match name {
            "local" => Backend::Local {
                // Nonempty exports make the byte budget an upper bound on
                // registration count as well.
                buffers: registry(capacity.get().inner.capacity() as usize)?,
                next: AtomicU64::new(0),
            },
            "shm" => Backend::Shared {
                buffers: registry(256)?,
                slot: acknowledgment_slot,
                host_slots: host_slots.into_iter().collect(),
            },
            "cuda_vmm" => {
                py.import("uniserve_kernels.peer_storage")?
                    .call_method0("load")?;
                Backend::Cuda(cuda::CudaExports::new(
                    registry(256)?,
                    acknowledgment_slot,
                    cross_host_consumers,
                ))
            }
            "channel" => Backend::Channel {
                endpoint: format!("uniserve-channel-{}", uuid::Uuid::new_v4().simple()),
                host_slots: host_slots.into_iter().collect(),
            },
            _ => return Err(invalid(py, format!("unknown transport: {name}"))),
        };
        let reads = Py::new(
            py,
            TransferPool::new(
                py,
                if name == "channel" { 1 } else { 2 },
                capacity.clone_ref(py),
                &format!("uniserve-{name}-read"),
                event_pool.clone_ref(py),
            )?,
        )?;

        Ok(Self {
            source,
            capacity,
            events: event_pool,
            reads,
            backend,
        })
    }

    #[getter]
    pub(super) fn name(&self) -> &'static str {
        match self.backend {
            Backend::Local { .. } => "local",
            Backend::Shared { .. } => "shm",
            Backend::Cuda(_) => "cuda_vmm",
            Backend::Channel { .. } => "channel",
        }
    }

    fn endpoint(&self) -> &str {
        match &self.backend {
            Backend::Channel { endpoint, .. } => endpoint,
            Backend::Local { buffers, .. } | Backend::Shared { buffers, .. } => {
                buffers.get().name()
            }
            Backend::Cuda(cuda) => cuda.buffers.get().name(),
        }
    }

    /// Reserve bytes before preparing backing; successful registration owns
    /// the reservation until producer and consumer accesses have retired.
    #[pyo3(signature = (tensor, *, offset=None, consumers=Vec::new()))]
    pub(super) fn export<'py>(
        &self,
        py: Python<'py>,
        tensor: &Bound<'py, PyAny>,
        offset: Option<&Bound<'py, PyAny>>,
        consumers: Vec<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let view = ExportView::new(tensor, offset)?;
        match &self.backend {
            Backend::Cuda(cuda) => {
                cuda.check(py)?;
                cuda.validate(&view)?;
                self.events.borrow(py).reap(py)?;
            }
            Backend::Local { .. } => self.events.borrow(py).reap(py)?,
            Backend::Shared { buffers, .. } => buffers.get().reap(py)?,
            Backend::Channel { .. } => {}
        }
        self.capacity
            .get()
            .inner
            .acquire(view.nbytes)
            .map_err(|error| native_error(py, error))?;

        match &self.backend {
            Backend::Local { buffers, next } => self.export_local(&view, buffers, next),
            Backend::Shared { buffers, .. } => self.export_shared(&view, buffers, consumers),
            Backend::Cuda(cuda) => cuda.export(self, &view, consumers),
            Backend::Channel { endpoint, .. } => {
                let result = (|| {
                    let payload = numerical(py, "channel")?
                        .call_method1("_export_payload", (&view.tensor, &view.shape))?;
                    let handle = transfer_types(py)?
                        .getattr("ChannelTransfer")?
                        .call1((endpoint, payload))?;
                    view.locator(self, handle)
                })();
                self.return_bytes(py, view.nbytes)?;
                result
            }
        }
    }

    #[pyo3(signature = (locator, *, device, destination=None, region=None, reservation=None))]
    pub(super) fn fetch(
        &self,
        py: Python<'_>,
        locator: Bound<'_, PyAny>,
        device: Py<PyAny>,
        destination: Option<Bound<'_, PyAny>>,
        region: Option<Py<PyAny>>,
        reservation: Option<Py<ReadReservation>>,
    ) -> PyResult<Py<TransferTicket>> {
        let pool = self.reads.bind(py).clone();
        match &self.backend {
            Backend::Local { .. } => TransferPool::fetch_local(
                pool,
                &locator,
                device.bind(py),
                destination,
                region.map(|region| region.into_bound(py)),
                reservation,
            ),
            Backend::Shared { slot, .. } => TransferPool::fetch_shared(
                pool,
                locator,
                &self.source.bind(py).getattr("node")?.extract::<String>()?,
                *slot,
                device,
                destination,
                region,
                reservation,
            ),
            Backend::Cuda(cuda) => TransferPool::fetch_cuda(
                pool,
                locator,
                self.source.bind(py).clone(),
                cuda.slot,
                device,
                destination,
                region,
                reservation,
            ),
            Backend::Channel { .. } => {
                TransferPool::fetch_channel(pool, locator, device, destination, region, reservation)
            }
        }
    }

    /// Borrow shared host rows in place; release ends the reader's claim.
    #[pyo3(signature = (locator, region=None))]
    fn borrow(
        &self,
        py: Python<'_>,
        locator: &Bound<'_, PyAny>,
        region: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<SharedRead> {
        let Backend::Shared { slot, .. } = &self.backend else {
            return Err(invalid(py, "only shared storage supports host borrowing"));
        };
        if locator.getattr("backend")?.extract::<String>()? != "shm" {
            return Err(invalid(
                py,
                "shared storage borrow requires a shared storage locator",
            ));
        }
        if !locator
            .getattr("source")?
            .getattr("node")?
            .eq(self.source.bind(py).getattr("node")?)?
        {
            return Err(invalid(
                py,
                "shared storage transport requires the source node",
            ));
        }
        let (offset, nbytes) = py
            .import("uniserve_worker.transport.layout")?
            .call_method1("row_span", (locator, region))?
            .extract()?;
        let name: String = locator.getattr("transport")?.getattr("name")?.extract()?;
        SharedRead::open(
            py,
            &name,
            nbytes,
            *slot,
            offset,
            Duration::from_secs(120),
            || Ok(()),
        )
    }

    pub(super) fn release(
        &self,
        py: Python<'_>,
        locator: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Py<Completion>>> {
        if let Some(buffers) = self.buffers() {
            buffers.get().release(py, locator)
        } else {
            self.require_channel(locator)?;
            Ok(None)
        }
    }

    pub(super) fn retirement(
        &self,
        py: Python<'_>,
        locator: &Bound<'_, PyAny>,
    ) -> PyResult<Py<Completion>> {
        if let Some(buffers) = self.buffers() {
            buffers.get().retirement(py, locator)
        } else {
            self.require_channel(locator)?;
            let done = Py::new(py, Completion::new())?;
            Completion::set_result(done.bind(py), &py.None().into_bound(py).cast_into()?)?;
            Ok(done)
        }
    }

    fn set_completion_wake(&self, py: Python<'_>, wake: Option<Py<PyAny>>) {
        if let Backend::Shared { buffers, .. } = &self.backend {
            buffers
                .get()
                .set_completion_wake(py, wake.as_ref().map(|wake| wake.clone_ref(py)));
        }
        self.reads.get().set_completion_wake(wake);
    }

    fn reap(&self, py: Python<'_>) -> PyResult<()> {
        match &self.backend {
            Backend::Shared { buffers, .. } => buffers.get().reap(py),
            Backend::Cuda(cuda) => cuda.reap(py),
            _ => Ok(()),
        }
    }

    fn awaiting_acknowledgment(&self, py: Python<'_>) -> PyResult<bool> {
        match &self.backend {
            Backend::Shared { buffers, .. } => buffers.get().awaiting_acknowledgment(py),
            Backend::Cuda(cuda) => cuda.awaiting_acknowledgment(py),
            _ => Ok(false),
        }
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let mut result = self.reads.get().close(py);
        if let Some(buffers) = self.buffers()
            && let Err(error) = buffers.get().close(py)
        {
            // Active source readers still own pool chunks. Keep their
            // allocation and descriptor service until the registry drains.
            return result.and(Err(error));
        }
        if let Backend::Cuda(cuda) = &self.backend {
            result = result.and(cuda.close(py));
        }
        result
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.source)?;
        visit.call(&self.capacity)?;
        visit.call(&self.events)?;
        visit.call(&self.reads)?;
        if let Some(buffers) = self.buffers() {
            visit.call(buffers)?;
        }
        if let Backend::Cuda(cuda) = &self.backend {
            cuda.visit(&visit)?;
        }
        Ok(())
    }

    fn __clear__(&self, py: Python<'_>) {
        if let Err(error) = self.close(py) {
            error.write_unraisable(py, None);
        }
    }
}

impl Transport {
    pub(super) fn serves(&self, consumers: &[usize]) -> bool {
        if consumers.is_empty() {
            return true;
        }
        match &self.backend {
            Backend::Shared { host_slots, .. } => {
                consumers.iter().any(|slot| host_slots.contains(slot))
            }
            Backend::Channel { host_slots, .. } => {
                consumers.iter().any(|slot| !host_slots.contains(slot))
            }
            _ => true,
        }
    }

    fn buffers(&self) -> Option<&Py<BufferRegistry>> {
        match &self.backend {
            Backend::Local { buffers, .. } | Backend::Shared { buffers, .. } => Some(buffers),
            Backend::Cuda(cuda) => Some(&cuda.buffers),
            Backend::Channel { .. } => None,
        }
    }

    fn require_channel(&self, locator: &Bound<'_, PyAny>) -> PyResult<()> {
        if locator.getattr("backend")?.extract::<String>()? != "channel"
            || locator
                .getattr("transport")?
                .getattr("endpoint")?
                .extract::<String>()?
                != self.endpoint()
        {
            return Err(invalid(
                locator.py(),
                "channel export belongs to another endpoint",
            ));
        }
        Ok(())
    }

    fn return_bytes(&self, py: Python<'_>, nbytes: u64) -> PyResult<()> {
        self.capacity
            .get()
            .inner
            .release(nbytes)
            .map_err(|error| native_error(py, error))
    }

    fn record_event(
        &self,
        device: &Bound<'_, PyAny>,
        interprocess: bool,
    ) -> PyResult<Py<CUDAEvent>> {
        let py = device.py();
        let events = self.events.borrow(py);
        let event = Py::new(py, events.acquire_event(py, device, false, interprocess)?)?;
        events.record(py, &event.borrow(py), device)?;
        events.retain(py, &event.borrow(py), device, 1)?;
        Ok(event)
    }

    fn export_local<'py>(
        &self,
        view: &ExportView<'py>,
        buffers: &Py<BufferRegistry>,
        next: &AtomicU64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = view.tensor.py();
        let source = (|| {
            let event = if view.first.getattr("is_cuda")?.extract()? {
                Some(self.record_event(&view.first.getattr("device")?, false)?)
            } else {
                None
            };
            Py::new(
                py,
                TransportBuffer::local(
                    view.tensor.clone().unbind(),
                    event,
                    view.nbytes,
                    self.capacity.clone_ref(py),
                ),
            )
        })();
        let source = match source {
            Ok(source) => source,
            Err(error) => {
                self.return_bytes(py, view.nbytes)?;
                return Err(error);
            }
        };
        let result = (|| {
            let handle = transfer_types(py)?
                .getattr("LocalTransfer")?
                .call1((self.endpoint(), next.fetch_add(1, Ordering::Relaxed)))?;
            let locator = view.locator(self, handle)?;
            buffers
                .get()
                .register(py, locator.clone().unbind(), source.clone_ref(py))?;
            Ok(locator)
        })();
        if result.is_err() {
            TransportBuffer::reclaim(py, source, &self.events.borrow(py), None)?;
        }
        result
    }

    fn export_shared<'py>(
        &self,
        view: &ExportView<'py>,
        buffers: &Py<BufferRegistry>,
        consumers: Vec<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = view.tensor.py();
        let mut storage = None;
        let mut registered = None;
        let result: PyResult<_> = (|| {
            let stream = if view.first.getattr("is_cuda")?.extract()? {
                Some(
                    py.import("torch.cuda")?
                        .call_method1("current_stream", (view.first.getattr("device")?,))?,
                )
            } else {
                None
            };
            let device = stream
                .as_ref()
                .map(|stream| {
                    Ok::<_, PyErr>((
                        stream.getattr("device")?.getattr("index")?.extract()?,
                        stream.getattr("cuda_stream")?.extract()?,
                    ))
                })
                .transpose()?;
            let buffer = Py::new(
                py,
                SharedBuffer::new(py, view.nbytes as usize, consumers, device)?,
            )?;
            storage = Some(buffer.clone_ref(py));
            let handle = transfer_types(py)?
                .getattr("PosixShmTransfer")?
                .call1((self.endpoint(), buffer.get().name(py)?))?;
            let locator = view.locator(self, handle)?;
            let backing = Py::new(
                py,
                TransportBuffer::shared(py, buffer.clone_ref(py), self.capacity.clone_ref(py)),
            )?;
            buffers
                .get()
                .register(py, locator.clone().unbind(), backing)?;
            registered = Some(locator.clone());

            // Mark in-flight storage before the first numerical copy. Failed
            // copies must drain before CUDA unregistration or byte reuse.
            if stream.is_some() {
                buffer.get().begin_copy(py)?;
            }
            numerical(py, "shm")?.call_method1(
                "_export_payload",
                (&view.tensor, &view.shape, &buffer, stream.as_ref()),
            )?;
            buffer.get().mark_ready(py)?;
            if let Some((_, stream)) = device {
                self.events.borrow(py).notify_stream(py, stream)?;
            }
            Ok(locator)
        })();
        if let Err(error) = result {
            if let Some(storage) = storage
                && let Err(cleanup) = storage.get().close(py)
            {
                error.set_cause(py, Some(cleanup));
                return Err(error);
            }
            if let Some(locator) = registered {
                buffers.get().release(py, &locator)?;
            } else {
                self.return_bytes(py, view.nbytes)?;
            }
            return Err(error);
        }
        result
    }
}

/// Borrowed numerical spans and their logical extents for one export.
struct ExportView<'py> {
    tensor: Bound<'py, PyAny>,
    first: Bound<'py, PyAny>,
    shape: Bound<'py, PyTuple>,
    offset: Bound<'py, PyTuple>,
    nbytes: u64,
}

impl<'py> ExportView<'py> {
    fn new(tensor: &Bound<'py, PyAny>, offset: Option<&Bound<'py, PyAny>>) -> PyResult<Self> {
        let layout = tensor.py().import("uniserve_worker.transport.layout")?;
        let (tensor, shape, offset): (Bound<'py, PyAny>, Bound<'py, PyTuple>, Bound<'py, PyTuple>) =
            layout
                .call_method1("export_views", (tensor, offset))?
                .extract()?;
        let first = if tensor.is_instance_of::<PyTuple>() {
            tensor.get_item(0)?
        } else {
            tensor.clone()
        };
        let nbytes = layout
            .call_method1("tensor_nbytes", (&tensor,))?
            .extract()?;
        Ok(Self {
            tensor,
            first,
            shape,
            offset,
            nbytes,
        })
    }

    fn locator(&self, owner: &Transport, handle: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = self.tensor.py();
        let args = PyDict::new(py);
        args.set_item("source", &owner.source)?;
        args.set_item("transport", handle)?;
        args.set_item("nbytes", self.nbytes)?;
        args.set_item(
            "dtype",
            self.first
                .getattr("dtype")?
                .str()?
                .to_str()?
                .trim_start_matches("torch."),
        )?;
        args.set_item("shape", &self.shape)?;
        args.set_item("offset", &self.offset)?;
        args.set_item("device", self.first.getattr("device")?.str()?)?;
        transfer_types(py)?
            .getattr("Locator")?
            .call((), Some(&args))
    }
}

fn transfer_types(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("uniserve_worker.protocol.transfer")
}

fn numerical<'py>(py: Python<'py>, backend: &str) -> PyResult<Bound<'py, PyModule>> {
    py.import(format!("uniserve_worker.transport.{backend}"))
}
