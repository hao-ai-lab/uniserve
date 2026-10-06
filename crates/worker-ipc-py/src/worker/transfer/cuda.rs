//! CUDA VMM read admission, imported mappings and reader acknowledgments.

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::Arc;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use uniserve_worker::cuda::{DeviceGuard, Event, Stream};
use uniserve_worker_ipc::{TransferTransport, WorkerEndpoint};

use super::super::descriptor_grants::fetch_descriptor;
use super::super::error::invalid;
use super::super::events::CUDAEvent;
use super::super::host::with_context;
use super::super::locator::Locator;
use super::super::registry::Registry;
use super::{
    PythonRead, ReadCall, ReadReservation, TransferPool, TransferTicket, copy_acknowledgment,
};

pub(super) struct CudaRead {
    pool: Py<TransferPool>,
    locator: Py<Locator>,
    device: Py<PyAny>,
    destination: Option<Py<PyAny>>,
    region: Option<Py<PyAny>>,
    slot: usize,
    local: bool,
    same_node: bool,
}

impl CudaRead {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn submit(
        pool: &Bound<'_, TransferPool>,
        locator: Bound<'_, Locator>,
        source: &WorkerEndpoint,
        slot: usize,
        device: Py<PyAny>,
        destination: Option<Bound<'_, PyAny>>,
        region: Option<Py<PyAny>>,
        reservation: Option<Py<ReadReservation>>,
    ) -> PyResult<Py<TransferTicket>> {
        let py = pool.py();
        let value = &locator.get().inner;
        let TransferTransport::CudaVmm {
            allocation_handle, ..
        } = &value.transport
        else {
            return Err(invalid(py, "CUDA VMM read requires a CUDA VMM locator"));
        };
        let same_node = value.source.node == source.node;
        let local = value.source.address_space == source.address_space;
        if !same_node && allocation_handle.len() != 64 {
            return Err(invalid(
                py,
                "a device product crosses hosts only as a fabric handle",
            ));
        }
        if device.bind(py).getattr("type")?.extract::<String>()? != "cuda" {
            return Err(invalid(py, "CUDA VMM destination must be a CUDA device"));
        }
        let device = py
            .import("uniserve.runtime.device")?
            .getattr("canonical_device")?
            .call1((&device,))?
            .unbind();

        let destination = destination
            .map(|destination| {
                py.import("uniserve_worker.transport.layout")?
                    .getattr("read_destination")?
                    .call1((&locator, &device, destination, region.as_ref()))
            })
            .transpose()?;
        let nbytes = locator.get().inner.nbytes;
        let ticket = Py::new(
            py,
            TransferTicket::new(pool.get().events.clone_ref(py), None),
        )?;
        let read = Self {
            pool: pool.clone().unbind(),
            locator: locator.unbind(),
            device,
            destination: destination.as_ref().map(|value| value.clone().unbind()),
            region,
            slot,
            local,
            same_node,
        };
        TransferPool::submit_read(
            pool,
            PythonRead {
                ticket,
                call: ReadCall::Cuda(read),
            },
            nbytes,
            destination,
            reservation,
        )
    }

    pub(super) fn run(&self, py: Python<'_>, ticket: &Py<TransferTicket>) -> PyResult<()> {
        ticket.get().require_active(py)?;
        let locator = self.locator.bind(py);
        let device = self.device.bind(py);
        let layout = py.import("uniserve_worker.transport.layout")?;
        let destination = match &self.destination {
            Some(destination) => destination.bind(py).clone(),
            None => layout.getattr("read_destination")?.call1((
                locator,
                device,
                py.None(),
                self.region.as_ref(),
            ))?,
        };
        let cuda = py.import("torch.cuda")?;

        with_context(&cuda.call_method1("device", (device,))?, || {
            let mut import_device = device.clone();
            let (mut mapped, mut event, acknowledgment) = if self.local {
                let (buffer, source) = Registry::acquire_read(locator)?;
                ticket.get().bind_source(py, source)?;
                (
                    buffer.get().tensor(py)?.into_bound(py),
                    buffer.get().event(py),
                    None,
                )
            } else {
                let numerical = py.import("uniserve_worker.transport.cuda_vmm")?;
                let source_device = py
                    .import("torch")?
                    .getattr("device")?
                    .call1((&locator.get().inner.device,))?;
                if self.same_node
                    && source_device.getattr("type")?.extract::<String>()? == "cuda"
                    && !source_device.eq(device)?
                    && !numerical
                        .getattr("_can_access_peer")?
                        .call1((device.str()?, source_device.str()?))?
                        .extract::<bool>()?
                {
                    import_device = source_device;
                }

                let TransferTransport::CudaVmm {
                    endpoint,
                    export_id,
                    allocation_handle,
                    ready_event_handle,
                    ..
                } = &locator.get().inner.transport
                else {
                    return Err(invalid(py, "CUDA VMM read requires a CUDA VMM locator"));
                };
                let descriptor = if allocation_handle.len() == size_of::<i32>() {
                    let fd = fetch_descriptor(py, endpoint, export_id)?;
                    // The grant transfers this descriptor to the importing caller.
                    Some(unsafe { OwnedFd::from_raw_fd(fd) })
                } else {
                    None
                };
                let exported = match &descriptor {
                    Some(descriptor) => PyBytes::new(py, &descriptor.as_raw_fd().to_ne_bytes()),
                    None => PyBytes::new(py, allocation_handle),
                };
                let (mapped, acknowledgment): (Py<PyAny>, Option<Py<PyAny>>) = numerical
                    .getattr("_import_views")?
                    .call1((locator, &destination, &import_device, exported, self.slot))?
                    .extract()?;
                drop(descriptor);

                let bytes = ready_event_handle.as_slice();
                let event = if bytes.is_empty() {
                    None
                } else {
                    let device_index = import_device.getattr("index")?.extract()?;
                    let bytes = bytes
                        .try_into()
                        .map_err(|_| invalid(py, "CUDA IPC event handle must contain 64 bytes"))?;
                    let inner = Event::from_ipc_handle(device_index, bytes)
                        .map_err(PyRuntimeError::new_err)?;
                    Some(Py::new(
                        py,
                        CUDAEvent {
                            inner: Arc::new(inner),
                        },
                    )?)
                };
                (mapped.into_bound(py), event, acknowledgment)
            };

            if let Some(region) = &self.region {
                mapped = layout.getattr("region_view")?.call1((mapped, region))?;
            }
            let cross_device = !import_device.eq(device)?;
            if cross_device && let Some(producer) = event.take() {
                // Source-device IPC fences cannot order destination streams on
                // every topology. Finish the producer on its owning device.
                let producer = Arc::clone(&producer.borrow(py).inner);
                py.detach(|| producer.wait())
                    .map_err(PyRuntimeError::new_err)?;
            }

            if cross_device && let Some(word) = acknowledgment {
                let word = word.bind(py);
                self.acknowledge(py, ticket, &import_device, &mapped, word, "CLAIMED")?;
                self.pool.get().copy(
                    py,
                    ticket.clone_ref(py),
                    mapped.clone(),
                    destination,
                    event,
                    None,
                )?;
                self.acknowledge(py, ticket, &import_device, &mapped, word, "ACKNOWLEDGED")
            } else {
                self.pool.get().copy(
                    py,
                    ticket.clone_ref(py),
                    mapped,
                    destination,
                    event,
                    acknowledgment.map(|word| word.into_bound(py)),
                )
            }
        })
    }

    fn acknowledge(
        &self,
        py: Python<'_>,
        ticket: &Py<TransferTicket>,
        device: &Bound<'_, PyAny>,
        mapped: &Bound<'_, PyAny>,
        word: &Bound<'_, PyAny>,
        state: &str,
    ) -> PyResult<()> {
        let device_index = device.getattr("index")?.extract()?;
        let _device = DeviceGuard::new(device_index).map_err(PyRuntimeError::new_err)?;
        let numerical = py
            .import("torch.cuda")?
            .call_method1("current_stream", (device,))?;
        let stream = Arc::new(Stream::borrowed(
            numerical.getattr("cuda_stream")?.extract()?,
        ));
        let submitted = copy_acknowledgment(py, Some(word), state, Some(&numerical));
        if let Err(error) = py.detach(|| stream.wait()) {
            // An acknowledgment write may still reference the imported mapping.
            let mut read = ticket.get().lock(py)?;
            read.ticket.mark_undrained(Some(stream), None);
            read.retained.extend([
                mapped.clone().unbind(),
                word.clone().unbind(),
                numerical.unbind(),
            ]);
            return Err(PyRuntimeError::new_err(error));
        }
        submitted
    }

    pub(super) fn visit(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.pool)?;
        visit.call(&self.locator)?;
        visit.call(&self.device)?;
        visit.call(&self.destination)?;
        visit.call(&self.region)
    }
}
