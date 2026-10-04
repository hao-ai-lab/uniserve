//! Cross-call buffer visibility and physical retirement.
//!
//! A Buffer owns one allocation-backed value. TensorRead retains a generation
//! while its consumer uses it; TensorImport retains the transfers filling missing
//! coverage. TensorStore owns lookup and bounded relay storage. Numerical copies
//! and views use the same PyTorch backend as model computation.

use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, MutexGuard, TryLockError};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{PyDict, PyList, PyTuple};
use uniserve_core::CallId;
use uniserve_worker_ipc::{BufferId, RequestKey};

use crate::buffer::{BufferBinding, BufferPool};
use crate::error::{invalid, invariant, resource};

type BufferKey = (RequestKey, CallId, u16);
type CallKey = (RequestKey, CallId);
type LaneKey = (String, usize, usize);
type RelayKey = (LaneKey, String, usize);
type ArenaKey = (String, String, usize);
type RelayCall = (String, usize, RequestKey, CallId);
type ReadRequest<'py> = (
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Option<Bound<'py, PyAny>>,
);

fn buffer_key(id: BufferId) -> BufferKey {
    (id.owner, id.producer_call_id, id.output_index)
}

/// Writes become readable only after both production and result commit.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WriteState {
    Reserved,
    Deferred,
    Produced,
    Committed,
}

impl WriteState {
    fn produced(self) -> bool {
        matches!(self, Self::Produced | Self::Committed)
    }
}

enum Backing {
    Persistent(Py<BufferBinding>),
    Relay(RelayKey),
}

/// The event and stream that make a produced device value ready for a reader.
struct Fence {
    event: Py<PyAny>,
    stream: u64,
}

/// An allocation-backed value whose visibility and access lifetime are owned
/// by its store. Numerical consumers borrow the tensor and metadata.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct Buffer {
    id: BufferId,
    #[pyo3(get)]
    reference: Py<PyAny>,
    #[pyo3(get)]
    tensor: Py<PyAny>,
    device: String,
    logical_shape: Vec<usize>,
    #[pyo3(get)]
    region: Option<Py<PyAny>>,
    #[pyo3(get)]
    metadata: Option<Py<PyAny>>,
    #[pyo3(get)]
    feature: bool,
    backing: Option<Backing>,
    state: WriteState,
    extent: usize,
    value_shape: Vec<usize>,
    producer: Option<Fence>,
    readers: usize,
    reader_events: Vec<Py<PyAny>>,
    transfers: Vec<Py<PyAny>>,
    exports: Vec<Py<PyAny>>,
    released: bool,
}

#[pymethods]
impl Buffer {
    #[getter]
    fn logical_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.logical_shape)
    }

    #[getter]
    fn producer_recorded(&self) -> bool {
        self.state.produced()
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.reference)?;
        visit.call(&self.tensor)?;
        visit.call(&self.region)?;
        visit.call(&self.metadata)?;
        if let Some(Backing::Persistent(binding)) = &self.backing {
            visit.call(binding)?;
        }
        if let Some(fence) = &self.producer {
            visit.call(&fence.event)?;
        }
        for value in self
            .reader_events
            .iter()
            .chain(&self.transfers)
            .chain(&self.exports)
        {
            visit.call(value)?;
        }
        Ok(())
    }
}

impl Buffer {
    fn require_candidate(&self, py: Python<'_>) -> PyResult<()> {
        if self.backing.is_none() || self.state == WriteState::Committed {
            return Err(invariant(py, "device-product candidate is not live"));
        }
        if self.state == WriteState::Reserved {
            return Err(invariant(
                py,
                "completion packing found an unpublished device product",
            ));
        }
        Ok(())
    }

    fn defer(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.state.produced() {
            return Err(invariant(py, "a published write cannot be deferred"));
        }
        self.state = WriteState::Deferred;
        Ok(())
    }

    fn produced(&mut self, shape: Vec<usize>, extent: usize, fence: Option<Fence>) {
        self.value_shape = shape;
        self.extent = extent;
        self.producer = fence;
        self.state = WriteState::Produced;
    }

    fn retain_reader(&mut self, py: Python<'_>, event: &Bound<'_, PyAny>) -> PyResult<bool> {
        if self
            .reader_events
            .iter()
            .any(|current| current.bind(py).is(event))
        {
            return Ok(false);
        }
        self.reader_events.push(event.clone().unbind());
        Ok(true)
    }

    fn access_retired(&self, py: Python<'_>, events: &mut HashMap<usize, bool>) -> PyResult<bool> {
        if !self.released || self.readers != 0 {
            return Ok(false);
        }
        for transfer in &self.transfers {
            if !transfer
                .bind(py)
                .call_method0("retired")?
                .extract::<bool>()?
            {
                return Ok(false);
            }
        }
        for export in &self.exports {
            let export = export.bind(py);
            if !export.call_method0("done")?.extract::<bool>()?
                || !export.call_method0("exception")?.is_none()
            {
                return Ok(false);
            }
        }
        for event in self
            .producer
            .iter()
            .map(|fence| &fence.event)
            .chain(&self.reader_events)
        {
            let identity = event.as_ptr() as usize;
            let ready = match events.get(&identity) {
                Some(&ready) => ready,
                None => {
                    let ready = event.bind(py).call_method0("query")?.extract()?;
                    events.insert(identity, ready);
                    ready
                }
            };
            if !ready {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// One read of a specific buffer generation. Its tensor view stays stable
/// when an import extends the buffer's resident coverage.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct TensorRead {
    #[pyo3(get)]
    tensor: Py<PyAny>,
    #[pyo3(get)]
    region: Option<Py<PyAny>>,
    #[pyo3(get)]
    metadata: Option<Py<PyAny>>,
    buffer: Py<Buffer>,
    consumer: Option<CallId>,
    #[pyo3(get)]
    imported: Option<Py<TensorImport>>,
    complete: bool,
}

#[pymethods]
impl TensorRead {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.tensor)?;
        visit.call(&self.region)?;
        visit.call(&self.metadata)?;
        visit.call(&self.buffer)?;
        visit.call(&self.imported)
    }
}

/// Shared materialization of missing regions. The last read releases its
/// transfers; an abandoned destination remains held until they retire.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct TensorImport {
    buffer: Py<Buffer>,
    tensor: Py<PyAny>,
    tickets: Vec<Py<PyAny>>,
    metadata: Option<Py<PyAny>>,
    users: usize,
    committed: bool,
}

#[pymethods]
impl TensorImport {
    #[getter]
    fn tickets<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.tickets.iter().map(|ticket| ticket.bind(py)))
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.buffer)?;
        visit.call(&self.tensor)?;
        visit.call(&self.metadata)?;
        for ticket in &self.tickets {
            visit.call(ticket)?;
        }
        Ok(())
    }
}

/// A call owns its lane until every scalar field has retired.
struct Lane {
    call: CallKey,
    fields: HashSet<(String, usize)>,
}

struct StoreState {
    buffers: HashMap<BufferKey, Py<Buffer>>,
    calls: HashMap<CallKey, HashSet<BufferKey>>,
    imports: HashMap<BufferKey, Py<TensorImport>>,
    arenas: HashMap<ArenaKey, Py<PyAny>>,
    views: HashMap<RelayKey, Py<PyAny>>,
    lanes: HashMap<LaneKey, Lane>,
    call_lanes: HashMap<RelayCall, usize>,
    allocated_bytes: usize,
}

impl StoreState {
    fn require_buffer(&self, py: Python<'_>, buffer: &Bound<'_, Buffer>) -> PyResult<()> {
        let value = buffer.borrow();
        let current = self.buffers.get(&buffer_key(value.id));
        if value.backing.is_none() || !current.is_some_and(|current| current.bind(py).is(buffer)) {
            return Err(invariant(py, "stale device-product physical generation"));
        }
        Ok(())
    }

    fn require_reference(
        &self,
        py: Python<'_>,
        reference: &Bound<'_, PyAny>,
    ) -> PyResult<Py<Buffer>> {
        let id = crate::protocol::buffer_id(&reference.getattr("buffer_id")?)?;
        let buffer = self
            .buffers
            .get(&buffer_key(id))
            .filter(|buffer| buffer.borrow(py).state == WriteState::Committed);
        let buffer = buffer.ok_or_else(|| {
            let mut committed: Vec<_> = self.buffers.values().filter_map(|buffer| {
                let buffer = buffer.borrow(py);
                (buffer.id.owner.request_id == id.owner.request_id && buffer.state == WriteState::Committed)
                    .then_some((buffer.id.producer_call_id.batch_id, buffer.id.output_index))
            }).collect();
            committed.sort_unstable();
            invalid(py, format!(
                "unknown device-product reference: request {} produced by {:?} output {} generation {}; committed {committed:?}",
                id.owner.request_id.0, id.producer_call_id, id.output_index, id.generation,
            ))
        })?;
        if !buffer.borrow(py).reference.bind(py).eq(reference)? {
            return Err(invalid(py, "stale device-product logical generation"));
        }
        self.require_buffer(py, buffer.bind(py))?;
        Ok(buffer.clone_ref(py))
    }

    fn detach(&mut self, id: BufferId) {
        let call = (id.owner, id.producer_call_id);
        if let Some(buffers) = self.calls.get_mut(&call) {
            buffers.remove(&buffer_key(id));
            if buffers.is_empty() {
                self.calls.remove(&call);
            }
        }
    }

    fn release_backing(
        &mut self,
        py: Python<'_>,
        buffer: &Bound<'_, Buffer>,
        pool: &Py<BufferPool>,
    ) -> PyResult<()> {
        let backing = buffer
            .borrow_mut()
            .backing
            .take()
            .ok_or_else(|| invariant(py, "tensor storage was retired more than once"))?;
        match backing {
            Backing::Persistent(binding) => {
                pool.get().release(py, binding.bind(py))?;
            }
            Backing::Relay((lane_key, dtype, field)) => {
                let lane = self
                    .lanes
                    .get_mut(&lane_key)
                    .ok_or_else(|| invariant(py, "request-relay lane lost its owner"))?;
                if !lane.fields.remove(&(dtype, field)) {
                    return Err(invariant(
                        py,
                        "request-relay field was retired more than once",
                    ));
                }
                if lane.fields.is_empty() {
                    self.call_lanes.remove(&(
                        lane_key.0.clone(),
                        lane_key.1,
                        lane.call.0,
                        lane.call.1,
                    ));
                    self.lanes.remove(&lane_key);
                }
            }
        }
        Ok(())
    }

    fn reclaim(
        &mut self,
        py: Python<'_>,
        pool: &Py<BufferPool>,
        events: &Py<PyAny>,
    ) -> PyResult<()> {
        let mut queried = HashMap::new();
        let mut retired = Vec::new();
        for (&key, buffer) in &self.buffers {
            if buffer.borrow(py).access_retired(py, &mut queried)? {
                retired.push((key, buffer.clone_ref(py)));
            }
        }
        let mut released: HashMap<usize, (Py<PyAny>, usize)> = HashMap::new();
        for (key, buffer) in retired {
            self.require_buffer(py, buffer.bind(py))?;
            self.buffers.remove(&key);
            self.detach(buffer.borrow(py).id);
            self.release_backing(py, buffer.bind(py), pool)?;
            let buffer = buffer.borrow(py);
            for event in buffer
                .producer
                .iter()
                .map(|fence| &fence.event)
                .chain(&buffer.reader_events)
            {
                let value = released
                    .entry(event.as_ptr() as usize)
                    .or_insert_with(|| (event.clone_ref(py), 0));
                value.1 += 1;
            }
        }
        // Event references are returned only after every candidate has been
        // checked, so recycling cannot change a later query in this pass.
        for (event, count) in released.into_values() {
            events.bind(py).call_method1("release", (event, count))?;
        }
        Ok(())
    }
}

/// Own committed buffer lookup and retirement; borrow fixed physical arenas
/// and pooled CUDA events from the worker's resource owners.
#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct TensorStore {
    #[pyo3(get)]
    capacity: usize,
    #[pyo3(get)]
    byte_capacity: usize,
    #[pyo3(get)]
    max_entry_bytes: usize,
    #[pyo3(get)]
    devices: Py<PyTuple>,
    #[pyo3(get)]
    request_capacity: usize,
    #[pyo3(get)]
    relay_depth: usize,
    #[pyo3(get)]
    buffer_pool: Py<BufferPool>,
    #[pyo3(get)]
    event_pool: Py<PyAny>,
    #[pyo3(get)]
    exports: Py<PyDict>,
    state: Mutex<StoreState>,
}

impl TensorStore {
    fn wait_imported(
        &self,
        py: Python<'_>,
        state: &StoreState,
        read: &Bound<'_, TensorRead>,
    ) -> PyResult<()> {
        let read = read.borrow();
        if read.complete {
            return Err(invariant(
                py,
                "closed or ordinary tensor read is not an import",
            ));
        }
        let imported = read
            .imported
            .as_ref()
            .ok_or_else(|| invariant(py, "closed or ordinary tensor read is not an import"))?
            .borrow(py);
        state.require_buffer(py, imported.buffer.bind(py))?;
        for ticket in &imported.tickets {
            ticket.bind(py).call_method0("result")?;
        }
        self.wait_producer(py, &imported.buffer.borrow(py), &mut HashSet::new())
    }

    fn publish<'py>(
        &self,
        py: Python<'py>,
        buffer: &Bound<'py, Buffer>,
        value: &Bound<'py, PyAny>,
        event: Option<&Bound<'py, PyAny>>,
        metadata: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (reference, tensor, region, feature) = {
            let value = buffer.borrow();
            if value.state.produced() {
                return Err(invariant(py, "device product was published more than once"));
            }
            (
                value.reference.clone_ref(py),
                value.tensor.clone_ref(py),
                value.region.as_ref().map(|region| region.clone_ref(py)),
                value.feature,
            )
        };
        let view = numerical(py)?
            .getattr("_copy_value")?
            .call1((reference, tensor, region, feature, value, metadata))?;
        let shape = view.getattr("shape")?.extract()?;
        let extent = view.call_method0("numel")?.extract()?;
        let fence = self.producer_fence(py, &view.getattr("device")?, event, 1)?;
        let mut buffer = buffer.borrow_mut();
        buffer.produced(shape, extent, fence);
        buffer.metadata = metadata.map(|metadata| metadata.clone().unbind());
        Ok(view)
    }

    fn producer_fence(
        &self,
        py: Python<'_>,
        device: &Bound<'_, PyAny>,
        event: Option<&Bound<'_, PyAny>>,
        count: usize,
    ) -> PyResult<Option<Fence>> {
        if device.getattr("type")?.extract::<String>()? != "cuda" {
            return Ok(None);
        }
        let fence = match event {
            Some(event) => Fence {
                event: event.clone().unbind(),
                stream: self
                    .event_pool
                    .bind(py)
                    .call_method1("declare_stream", (event, device))?
                    .extract()?,
            },
            None => self.record_fence(py, device)?,
        };
        self.event_pool
            .bind(py)
            .call_method1("retain", (&fence.event, device, count))?;
        Ok(Some(fence))
    }

    fn record_fence(&self, py: Python<'_>, device: &Bound<'_, PyAny>) -> PyResult<Fence> {
        let event = self
            .event_pool
            .bind(py)
            .call_method1("acquire", (device,))?;
        let stream = self
            .event_pool
            .bind(py)
            .call_method1("record", (&event, device))?
            .extract()?;
        Ok(Fence {
            event: event.unbind(),
            stream,
        })
    }

    fn wait_producer(
        &self,
        py: Python<'_>,
        buffer: &Buffer,
        waited: &mut HashSet<(String, usize)>,
    ) -> PyResult<()> {
        if let Some(fence) = &buffer.producer {
            let key = (buffer.device.clone(), fence.event.as_ptr() as usize);
            if waited.insert(key) {
                let device = buffer.tensor.bind(py).getattr("device")?;
                let stream = current_stream(py, &device)?;
                if stream.getattr("cuda_stream")?.extract::<u64>()? != fence.stream {
                    stream.call_method1("wait_event", (&fence.event,))?;
                }
            }
        }
        Ok(())
    }

    fn wake_retirement(&self, py: Python<'_>, buffer: &Buffer) -> PyResult<()> {
        for event in buffer
            .producer
            .iter()
            .map(|fence| &fence.event)
            .chain(&buffer.reader_events)
        {
            if !event.bind(py).call_method0("query")?.extract::<bool>()? {
                self.event_pool
                    .bind(py)
                    .call_method1("schedule_completion_wake", (&buffer.device, event))?;
            }
        }
        Ok(())
    }

    fn release_buffer(
        &self,
        py: Python<'_>,
        state: &mut StoreState,
        buffer: &Bound<'_, Buffer>,
    ) -> PyResult<()> {
        let id = buffer.borrow().id;
        state.detach(id);
        buffer.borrow_mut().released = true;
        let value = buffer.borrow();
        self.wake_retirement(py, &value)?;
        if !value.state.produced() {
            for ticket in &value.transfers {
                ticket.bind(py).call_method0("cancel")?;
            }
        }
        Ok(())
    }
}

#[pymethods]
impl TensorStore {
    /// Share resident coverage and submit the missing regions as one transport
    /// reservation. A failed submission retains any in-flight destinations.
    #[pyo3(signature = (reference, tensor, *, device, bindings, request_slots, buffer_allocations, metadata=None))]
    #[allow(clippy::too_many_arguments)]
    fn import_tensor<'py>(
        &self,
        py: Python<'py>,
        reference: Bound<'py, PyAny>,
        tensor: Bound<'py, PyAny>,
        device: Bound<'py, PyAny>,
        bindings: Bound<'py, PyAny>,
        request_slots: Bound<'py, PyAny>,
        buffer_allocations: Bound<'py, PyAny>,
        metadata: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Py<TensorRead>> {
        let device = canonical_device(py, &device)?;
        let id = crate::protocol::buffer_id(&reference.getattr("buffer_id")?)?;
        let key = buffer_key(id);
        let shape: Vec<usize> = tensor.getattr("shape")?.extract()?;
        let mut state = self.lock(py)?;
        if let Some(pending) = state.imports.get(&key) {
            let value = pending.borrow(py);
            let buffer = value.buffer.borrow(py);
            if !buffer.reference.bind(py).eq(&reference)?
                || !value.tensor.bind(py).getattr("device")?.eq(&device)?
                || value
                    .tensor
                    .bind(py)
                    .getattr("shape")?
                    .extract::<Vec<usize>>()?
                    != shape
                || !optional_equal(py, value.metadata.as_ref(), metadata.as_ref())?
            {
                return Err(invalid(
                    py,
                    "product import conflicts with pending materialization",
                ));
            }
            if buffer.released {
                return Err(invalid(
                    py,
                    "product import requires a live published generation",
                ));
            }
            drop(buffer);
            let read = import_read(py, pending)?;
            value.buffer.borrow_mut(py).readers += 1;
            drop(value);
            pending.borrow_mut(py).users += 1;
            return Ok(read);
        }
        let existing = state
            .buffers
            .get(&key)
            .filter(|buffer| buffer.borrow(py).state == WriteState::Committed)
            .map(|buffer| buffer.clone_ref(py));
        let full = full_region(py, &shape)?;
        let (buffer, destination, missing) = if existing.is_some() {
            let buffer = state.require_reference(py, &reference)?;
            let value = buffer.borrow(py);
            if value.released {
                return Err(invalid(
                    py,
                    "product import requires a live published generation",
                ));
            }
            if device.str()?.to_str()? != value.device
                || !optional_equal(py, value.metadata.as_ref(), metadata.as_ref())?
            {
                return Err(invalid(
                    py,
                    "product import conflicts with resident ownership",
                ));
            }
            let (destination, missing) = match &value.region {
                None => {
                    if value.value_shape != shape {
                        return Err(invalid(py, "product import changes resident tensor shape"));
                    }
                    (
                        value_view(py, value.tensor.bind(py), &shape, value.extent)?,
                        Vec::new(),
                    )
                }
                Some(region) => {
                    let destination = match &value.backing {
                        Some(Backing::Persistent(binding)) => binding.bind(py).getattr("tensor")?,
                        _ => {
                            return Err(invalid(
                                py,
                                "product import exceeds its reserved logical storage",
                            ));
                        }
                    };
                    if destination.getattr("shape")?.extract::<Vec<usize>>()? != shape {
                        return Err(invalid(
                            py,
                            "product import exceeds its reserved logical storage",
                        ));
                    }
                    for ticket in &value.transfers {
                        if !ticket.bind(py).call_method0("retired")?.extract::<bool>()? {
                            return Err(resource(py, "product storage has pending physical reads"));
                        }
                    }
                    let missing: Vec<Bound<'py, PyAny>> = py
                        .import("uniserve._slices")?
                        .getattr("subtract")?
                        .call1((&full, region))?
                        .extract()?;
                    (destination, missing)
                }
            };
            drop(value);
            (buffer, destination, missing)
        } else {
            let shapes = PyDict::new(py);
            shapes.set_item(&reference, PyTuple::new(py, &shape)?)?;
            let feature = metadata
                .as_ref()
                .map(|metadata| metadata.is_instance(&numerical(py)?.getattr("FeatureMetadata")?))
                .transpose()?
                .unwrap_or(false);
            let bound = [(reference.clone(), device.clone())];
            let buffers = if feature {
                self.bind_persistent(
                    py,
                    &mut state,
                    &bound,
                    &buffer_allocations,
                    None,
                    Some(shapes.as_any()),
                    true,
                )?
            } else {
                self.bind_group(
                    py,
                    &mut state,
                    &bound,
                    Some(&request_slots),
                    Some(&buffer_allocations),
                    None,
                    Some(shapes.as_any()),
                )?
            };
            let buffer = buffers
                .into_iter()
                .next()
                .ok_or_else(|| invariant(py, "import destination was not reserved"))?;
            let extent = shape
                .iter()
                .try_fold(1_usize, |size, &dim| size.checked_mul(dim))
                .ok_or_else(|| invalid(py, "import tensor extent overflows"))?;
            let destination = value_view(py, buffer.borrow(py).tensor.bind(py), &shape, extent)?;
            (buffer, destination, vec![full.into_any()])
        };
        if destination
            .getattr("dtype")?
            .str()?
            .to_str()?
            .trim_start_matches("torch.")
            != tensor.getattr("dtype")?.extract::<String>()?
        {
            if existing.is_none() {
                buffer.borrow_mut(py).released = true;
                state.reclaim(py, &self.buffer_pool, &self.event_pool)?;
            }
            return Err(invalid(py, "product import changes resident tensor dtype"));
        }
        buffer.borrow_mut(py).readers += 1;
        let tickets = PyList::empty(py);
        let submitted = (|| -> PyResult<()> {
            let fetch = py.import("uniserve_worker.transport.fetch")?;
            let mut reads = Vec::<Bound<'py, PyAny>>::new();
            for region in missing {
                let kwargs = PyDict::new(py);
                kwargs.set_item("bindings", &bindings)?;
                kwargs.set_item("region", &region)?;
                reads.extend(
                    fetch
                        .getattr("plan_reads")?
                        .call((&tensor, destination.get_item(&region)?), Some(&kwargs))?
                        .extract::<Vec<Bound<'py, PyAny>>>()?,
                );
            }
            let kwargs = PyDict::new(py);
            kwargs.set_item("retain", tickets.getattr("append")?)?;
            fetch
                .getattr("submit_reads")?
                .call((PyTuple::new(py, reads)?,), Some(&kwargs))?;
            Ok(())
        })();
        let tickets: Vec<Py<PyAny>> = tickets.extract()?;
        buffer
            .borrow_mut(py)
            .transfers
            .extend(tickets.iter().map(|ticket| ticket.clone_ref(py)));
        if let Err(error) = submitted {
            for ticket in tickets {
                ticket.bind(py).call_method0("cancel")?;
                ticket.bind(py).call_method0("close")?;
            }
            buffer.borrow_mut(py).readers -= 1;
            if existing.is_none() {
                buffer.borrow_mut(py).released = true;
            }
            state.reclaim(py, &self.buffer_pool, &self.event_pool)?;
            return Err(error);
        }
        let committed = existing.is_some() && buffer.borrow(py).region.is_none();
        let imported = Py::new(
            py,
            TensorImport {
                buffer,
                tensor: destination.unbind(),
                tickets,
                metadata: metadata.map(Bound::unbind),
                users: 1,
                committed,
            },
        )?;
        let read = import_read(py, &imported)?;
        state.imports.insert(key, imported);
        Ok(read)
    }

    /// Tickets must be ready before this call; result() orders the current
    /// stream after their transfer fences and propagates transport failures.
    fn wait_import(&self, py: Python<'_>, read: &Bound<'_, TensorRead>) -> PyResult<()> {
        let state = self.lock(py)?;
        self.wait_imported(py, &state, read)
    }

    fn complete_import(&self, py: Python<'_>, read: &Bound<'_, TensorRead>) -> PyResult<()> {
        let mut state = self.lock(py)?;
        self.wait_imported(py, &state, read)?;
        let imported = read
            .borrow()
            .imported
            .as_ref()
            .ok_or_else(|| invariant(py, "closed or ordinary tensor read is not an import"))?
            .clone_ref(py);
        let value = imported.borrow(py);
        if value.committed {
            return Ok(());
        }
        let buffer = value.buffer.bind(py);
        let tensor = value.tensor.bind(py);
        if !buffer.borrow().state.produced() {
            self.publish(
                py,
                buffer,
                tensor,
                None,
                value.metadata.as_ref().map(|metadata| metadata.bind(py)),
            )?;
            let mut buffer = buffer.borrow_mut();
            buffer.state = WriteState::Committed;
            state
                .calls
                .entry((buffer.id.owner, buffer.id.producer_call_id))
                .or_default()
                .insert(buffer_key(buffer.id));
        } else {
            // Existing leases retain their original shard views. A new fence
            // covers the union, including every newly fetched region.
            let device = tensor.getattr("device")?;
            let fence = self.producer_fence(py, &device, None, 1)?;
            let previous = buffer
                .borrow()
                .producer
                .as_ref()
                .map(|fence| fence.event.clone_ref(py));
            if let Some(previous) = previous {
                self.event_pool
                    .bind(py)
                    .call_method1("defer_release", ((previous,), buffer))?;
            }
            let shape: Vec<usize> = tensor.getattr("shape")?.extract()?;
            let extent = tensor.call_method0("numel")?.extract()?;
            let mut buffer = buffer.borrow_mut();
            buffer.tensor = tensor.clone().unbind();
            buffer.region = None;
            buffer.value_shape = shape;
            buffer.extent = extent;
            buffer.producer = fence;
        }
        drop(value);
        imported.borrow_mut(py).committed = true;
        Ok(())
    }

    /// Copy a numerical value, then attach its producer fence. A supplied
    /// event is recorded by the caller after all writes in that invocation.
    #[pyo3(signature = (write, value, *, producer_event=None, metadata=None))]
    fn publish_write<'py>(
        &self,
        py: Python<'py>,
        write: Bound<'py, Buffer>,
        value: Bound<'py, PyAny>,
        producer_event: Option<Bound<'py, PyAny>>,
        metadata: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let state = self.lock(py)?;
        state.require_buffer(py, &write)?;
        self.publish(
            py,
            &write,
            &value,
            producer_event.as_ref(),
            metadata.as_ref(),
        )
    }

    #[pyo3(signature = (writes, values, *, producer_event=None))]
    fn publish_writes<'py>(
        &self,
        py: Python<'py>,
        writes: Vec<Bound<'py, Buffer>>,
        values: Bound<'py, PyAny>,
        producer_event: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let state = self.lock(py)?;
        let mut tensors = Vec::with_capacity(writes.len());
        let mut seen = HashSet::new();
        for write in &writes {
            state.require_buffer(py, write)?;
            if write.borrow().state.produced() || !seen.insert(write.as_ptr() as usize) {
                return Err(invariant(py, "device product was published more than once"));
            }
            tensors.push(write.borrow().tensor.clone_ref(py));
        }
        let tensors = PyTuple::new(py, tensors)?;
        if writes.is_empty() {
            return Ok(tensors);
        }
        numerical(py)?
            .getattr("_copy_scalars")?
            .call1((&tensors, values))?;
        let device = tensors.get_item(0)?.getattr("device")?;
        let fence = self.producer_fence(py, &device, producer_event.as_ref(), writes.len())?;
        for write in writes {
            write.borrow_mut().produced(
                vec![1],
                1,
                fence.as_ref().map(|fence| Fence {
                    event: fence.event.clone_ref(py),
                    stream: fence.stream,
                }),
            );
        }
        Ok(tensors)
    }

    #[pyo3(signature = (write, value, *, producer_event=None))]
    fn publish_scalar_write<'py>(
        &self,
        py: Python<'py>,
        write: Bound<'py, Buffer>,
        value: Bound<'py, PyAny>,
        producer_event: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let state = self.lock(py)?;
        state.require_buffer(py, &write)?;
        if write.borrow().state.produced() {
            return Err(invariant(py, "device product was published more than once"));
        }
        let tensor = write.borrow().tensor.clone_ref(py);
        let view = value_view(py, tensor.bind(py), &[1], 1)?;
        view.call_method1("fill_", (value,))?;
        let fence =
            self.producer_fence(py, &view.getattr("device")?, producer_event.as_ref(), 1)?;
        write.borrow_mut().produced(vec![1], 1, fence);
        Ok(view)
    }

    #[pyo3(signature = (reference, *, consumer_call_id, device=None))]
    fn consume<'py>(
        &self,
        py: Python<'py>,
        reference: Bound<'py, PyAny>,
        consumer_call_id: Bound<'py, PyAny>,
        device: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Py<TensorRead>> {
        let reads = self.consume_batch(py, vec![(reference, consumer_call_id, device)], None)?;
        Ok(reads.get_item(0)?.extract()?)
    }

    /// Resolve and validate the whole batch before acquiring any read lease.
    #[pyo3(signature = (requests, *, device=None))]
    fn consume_batch<'py>(
        &self,
        py: Python<'py>,
        requests: Vec<ReadRequest<'py>>,
        device: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let state = self.lock(py)?;
        let shared = device
            .as_ref()
            .map(|device| canonical_device(py, device))
            .transpose()?;
        let mut resolved = Vec::with_capacity(requests.len());
        let mut waited = HashSet::new();
        for (reference, consumer, requested) in requests {
            let buffer = state.require_reference(py, &reference)?;
            let value = buffer.borrow(py);
            if value.released {
                return Err(invalid(
                    py,
                    "device product was consumed after logical release",
                ));
            }
            let requested = requested
                .as_ref()
                .map(|device| canonical_device(py, device))
                .transpose()?;
            if let (Some(shared), Some(requested)) = (&shared, &requested)
                && !shared.eq(requested)?
            {
                return Err(invalid(
                    py,
                    "device-product batch names conflicting consumer devices",
                ));
            }
            if let Some(target) = shared.as_ref().or(requested.as_ref())
                && target.str()?.to_str()? != value.device
            {
                return Err(invalid(
                    py,
                    "device product consumer names a different device",
                ));
            }
            self.wait_producer(py, &value, &mut waited)?;
            let tensor =
                value_view(py, value.tensor.bind(py), &value.value_shape, value.extent)?.unbind();
            let read = Py::new(
                py,
                TensorRead {
                    tensor,
                    region: value.region.as_ref().map(|region| region.clone_ref(py)),
                    metadata: value
                        .metadata
                        .as_ref()
                        .map(|metadata| metadata.clone_ref(py)),
                    buffer: buffer.clone_ref(py),
                    consumer: Some(crate::protocol::call_id(&consumer)?),
                    imported: None,
                    complete: false,
                },
            )?;
            drop(value);
            resolved.push((buffer, read));
        }
        for (buffer, _) in &resolved {
            buffer.borrow_mut(py).readers += 1;
        }
        PyTuple::new(py, resolved.into_iter().map(|(_, read)| read))
    }

    /// End leases only after fencing their consuming streams. Outputs of the
    /// consuming call can supply the fence; otherwise record one per device.
    #[pyo3(signature = (reads, *, device=None, after_writes=Vec::new()))]
    fn complete_reads(
        &self,
        py: Python<'_>,
        reads: Vec<Bound<'_, TensorRead>>,
        device: Option<Bound<'_, PyAny>>,
        after_writes: Vec<Bound<'_, Buffer>>,
    ) -> PyResult<()> {
        let mut state = self.lock(py)?;
        let target = device
            .as_ref()
            .map(|device| canonical_device(py, device))
            .transpose()?;
        let mut seen = HashSet::new();
        let pending: Vec<_> = reads
            .into_iter()
            .filter(|read| !read.borrow().complete && seen.insert(read.as_ptr() as usize))
            .collect();
        if pending.is_empty() {
            return Ok(());
        }
        for read in &pending {
            let read = read.borrow();
            state.require_buffer(py, read.buffer.bind(py))?;
            if let Some(target) = &target
                && !read.tensor.bind(py).getattr("device")?.eq(target)?
            {
                return Err(invariant(
                    py,
                    "device-product reader completed on a different device",
                ));
            }
        }
        let mut write_fences = HashMap::new();
        for write in after_writes {
            state.require_buffer(py, &write)?;
            let value = write.borrow();
            if let Some(fence) = &value.producer {
                write_fences
                    .entry((
                        value.id.producer_call_id,
                        value.device.clone(),
                        fence.stream,
                    ))
                    .or_insert_with(|| fence.event.clone_ref(py));
            }
        }
        let mut recorded = HashMap::<String, Py<PyAny>>::new();
        for read in &pending {
            let value = read.borrow();
            let buffer = value.buffer.bind(py);
            let device = value.tensor.bind(py).getattr("device")?;
            if device.getattr("type")?.extract::<String>()? != "cuda" {
                continue;
            }
            let name = buffer.borrow().device.clone();
            let stream = current_stream(py, &device)?;
            let stream_id: u64 = stream.getattr("cuda_stream")?.extract()?;
            let event = value
                .consumer
                .and_then(|consumer| write_fences.get(&(consumer, name.clone(), stream_id)))
                .map(|event| event.clone_ref(py));
            let event = match event {
                Some(event) => event,
                None => match recorded.get(&name) {
                    Some(event) => event.clone_ref(py),
                    None => {
                        let event = self.record_fence(py, &device)?.event;
                        recorded.insert(name, event.clone_ref(py));
                        event
                    }
                },
            };
            if buffer.borrow_mut().retain_reader(py, event.bind(py))? {
                self.event_pool
                    .bind(py)
                    .call_method1("retain", (&event, &device))?;
            }
        }
        for read in pending {
            let (buffer, imported) = {
                let mut read = read.borrow_mut();
                read.complete = true;
                (read.buffer.clone_ref(py), read.imported.take())
            };
            buffer.borrow_mut(py).readers -= 1;
            if let Some(imported) = imported {
                let mut import = imported.borrow_mut(py);
                import.users -= 1;
                if import.users == 0 {
                    state.imports.remove(&buffer_key(buffer.borrow(py).id));
                    let committed = import.committed;
                    let tickets: Vec<_> = import
                        .tickets
                        .iter()
                        .map(|ticket| ticket.clone_ref(py))
                        .collect();
                    drop(import);
                    for ticket in tickets {
                        if !committed {
                            ticket.bind(py).call_method0("cancel")?;
                        }
                        ticket.bind(py).call_method0("close")?;
                    }
                    if !buffer.borrow(py).state.produced() {
                        buffer.borrow_mut(py).released = true;
                    }
                }
            }
            if buffer.borrow(py).released {
                self.wake_retirement(py, &buffer.borrow(py))?;
            }
        }
        state.reclaim(py, &self.buffer_pool, &self.event_pool)
    }

    fn release_calls(&self, py: Python<'_>, releases: Bound<'_, PyAny>) -> PyResult<()> {
        let mut state = self.lock(py)?;
        for release in releases.try_iter()? {
            let (request, call): (Bound<'_, PyAny>, Bound<'_, PyAny>) = release?.extract()?;
            let key = (
                crate::protocol::request_key(&request)?,
                crate::protocol::call_id(&call)?,
            );
            if let Some(buffers) = state.calls.remove(&key) {
                for key in buffers {
                    if let Some(buffer) = state.buffers.get(&key) {
                        buffer.borrow_mut(py).released = true;
                    }
                }
            }
        }
        Ok(())
    }

    fn release_buffers(&self, py: Python<'_>, buffers: Bound<'_, PyAny>) -> PyResult<()> {
        let buffers = PyTuple::new(py, buffers.try_iter()?.collect::<PyResult<Vec<_>>>()?)?;
        let selected = buffer_ids(&buffers)?;
        py.import("uniserve_worker.transport.exports")?
            .getattr("release_exports")?
            .call1((self.exports.bind(py), buffers))?;
        let mut state = self.lock(py)?;
        let release: Vec<_> = state
            .buffers
            .values()
            .filter(|buffer| selected.contains(&buffer.borrow(py).id))
            .map(|buffer| buffer.clone_ref(py))
            .collect();
        for buffer in release {
            self.release_buffer(py, &mut state, buffer.bind(py))?;
        }
        state.reclaim(py, &self.buffer_pool, &self.event_pool)
    }

    #[pyo3(signature = (requests, *, retained=None))]
    fn release_requests(
        &self,
        py: Python<'_>,
        requests: Bound<'_, PyAny>,
        retained: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let requests = request_keys(&requests)?;
        let retained = retained
            .as_ref()
            .map(buffer_ids)
            .transpose()?
            .unwrap_or_default();
        let mut state = self.lock(py)?;
        let mut release = Vec::new();
        for buffer in state.buffers.values() {
            let value = buffer.borrow(py);
            if requests.contains(&value.id.owner) {
                if retained.contains(&value.id) {
                    if !matches!(value.backing, Some(Backing::Persistent(_))) {
                        return Err(invalid(py, "finish cannot retain request-slot storage"));
                    }
                } else {
                    release.push(buffer.clone_ref(py));
                }
            }
        }
        for buffer in release {
            self.release_buffer(py, &mut state, buffer.bind(py))?;
        }
        state.reclaim(py, &self.buffer_pool, &self.event_pool)
    }

    #[pyo3(signature = (*, buffers, requests, retained=None))]
    fn retirement_ready(
        &self,
        py: Python<'_>,
        buffers: Bound<'_, PyAny>,
        requests: Bound<'_, PyAny>,
        retained: Option<Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        let buffers = buffer_ids(&buffers)?;
        let requests = request_keys(&requests)?;
        let retained = retained
            .as_ref()
            .map(buffer_ids)
            .transpose()?
            .unwrap_or_default();
        let selected = |buffer: &Buffer| {
            buffers.contains(&buffer.id)
                || (requests.contains(&buffer.id.owner) && !retained.contains(&buffer.id))
        };
        let mut state = self.lock(py)?;
        for buffer in state.buffers.values() {
            let value = buffer.borrow(py);
            if selected(&value) {
                for ticket in &value.transfers {
                    ticket.bind(py).call_method0("retirement_ready")?;
                }
                for export in &value.exports {
                    if export.bind(py).call_method0("done")?.extract::<bool>()? {
                        export.bind(py).call_method0("result")?;
                    }
                }
            }
        }
        state.reclaim(py, &self.buffer_pool, &self.event_pool)?;
        Ok(!state
            .buffers
            .values()
            .any(|buffer| selected(&buffer.borrow(py))))
    }

    #[pyo3(signature = (bindings, *, request_slots=None, buffer_allocations=None, regions=None, shapes=None))]
    fn bind_outputs<'py>(
        &self,
        py: Python<'py>,
        bindings: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>,
        request_slots: Option<Bound<'py, PyAny>>,
        buffer_allocations: Option<Bound<'py, PyAny>>,
        regions: Option<Bound<'py, PyAny>>,
        shapes: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let mut state = self.lock(py)?;
        let buffers = self.bind_group(
            py,
            &mut state,
            &bindings,
            request_slots.as_ref(),
            buffer_allocations.as_ref(),
            regions.as_ref(),
            shapes.as_ref(),
        )?;
        PyTuple::new(py, buffers)
    }

    #[pyo3(signature = (bindings, *, buffer_allocations, regions=None, shapes=None))]
    fn reserve_features<'py>(
        &self,
        py: Python<'py>,
        bindings: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>,
        buffer_allocations: Bound<'py, PyAny>,
        regions: Option<Bound<'py, PyAny>>,
        shapes: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let mut state = self.lock(py)?;
        let buffers = self.bind_persistent(
            py,
            &mut state,
            &bindings,
            &buffer_allocations,
            regions.as_ref(),
            shapes.as_ref(),
            true,
        )?;
        PyTuple::new(py, buffers)
    }

    #[pyo3(signature = (groups, *, request_slots=None, buffer_allocations=None, regions=None, shapes=None))]
    fn bind_output_groups<'py>(
        &self,
        py: Python<'py>,
        groups: Vec<Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>>,
        request_slots: Option<Bound<'py, PyAny>>,
        buffer_allocations: Option<Bound<'py, PyAny>>,
        regions: Option<Bound<'py, PyAny>>,
        shapes: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let mut state = self.lock(py)?;
        let mut outputs = Vec::new();
        let mut acquired = Vec::new();
        for group in groups.iter().filter(|group| !group.is_empty()) {
            match self.bind_group(
                py,
                &mut state,
                group,
                request_slots.as_ref(),
                buffer_allocations.as_ref(),
                regions.as_ref(),
                shapes.as_ref(),
            ) {
                Ok(buffers) => {
                    outputs.push(PyTuple::new(
                        py,
                        buffers.iter().map(|buffer| buffer.bind(py)),
                    )?);
                    acquired.extend(buffers);
                }
                Err(error) => {
                    self.rollback(py, &mut state, acquired)?;
                    return Err(error);
                }
            }
        }
        PyTuple::new(py, outputs)
    }

    fn producer_write_views<'py>(
        &self,
        py: Python<'py>,
        writes: Vec<Bound<'py, Buffer>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let state = self.lock(py)?;
        let mut tensors = Vec::with_capacity(writes.len());
        for write in writes {
            state.require_buffer(py, &write)?;
            let write = write.borrow();
            if write.state.produced() {
                return Err(invariant(py, "device product was published more than once"));
            }
            tensors.push(write.tensor.clone_ref(py));
        }
        PyTuple::new(py, tensors)
    }

    fn resident_bytes(&self, py: Python<'_>, device: &Bound<'_, PyAny>) -> PyResult<usize> {
        let name = py
            .import("torch")?
            .getattr("device")?
            .call1((device,))?
            .str()?
            .to_str()?
            .to_owned();
        let state = self.lock(py)?;
        let mut allocations = HashMap::new();
        for ((owner, _, _), tensor) in &state.arenas {
            if *owner == name {
                let storage = tensor.bind(py).call_method0("untyped_storage")?;
                let pointer: usize = storage.call_method0("data_ptr")?.extract()?;
                let bytes: usize = storage.call_method0("nbytes")?.extract()?;
                allocations.insert(pointer, bytes);
            }
        }
        Ok(allocations.values().sum())
    }

    #[new]
    #[pyo3(signature = (*, capacity=0, byte_capacity=None, max_entry_bytes=1, devices=None, request_capacity=0, relay_depth=0, buffer_pool, event_pool=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        capacity: isize,
        byte_capacity: Option<isize>,
        max_entry_bytes: isize,
        devices: Option<Vec<Bound<'_, PyAny>>>,
        request_capacity: isize,
        relay_depth: isize,
        buffer_pool: Py<BufferPool>,
        event_pool: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        if capacity < 0 || max_entry_bytes < 1 {
            return Err(PyValueError::new_err("tensor store capacities are invalid"));
        }
        let byte_capacity = match byte_capacity {
            Some(bytes) => bytes,
            None => buffer_pool.bind(py).getattr("byte_capacity")?.extract()?,
        };
        if byte_capacity < 1 {
            return Err(PyValueError::new_err(
                "device-product byte capacity must be positive",
            ));
        }
        if (request_capacity == 0) != (relay_depth == 0) {
            return Err(PyValueError::new_err(
                "request-relay dimensions must be complete",
            ));
        }
        if request_capacity < 0 || relay_depth < 0 {
            return Err(PyValueError::new_err(
                "request-relay dimensions must not be negative",
            ));
        }
        let canonical = py
            .import("uniserve.runtime.device")?
            .getattr("canonical_device")?;
        let devices = devices
            .unwrap_or_default()
            .into_iter()
            .map(|device| canonical.call1((device,)))
            .collect::<PyResult<Vec<_>>>()?;
        let event_pool = match event_pool {
            Some(pool) => pool,
            None => py
                .import("uniserve.runtime")?
                .getattr("EventPool")?
                .call0()?
                .unbind(),
        };
        Ok(Self {
            capacity: capacity as usize,
            byte_capacity: byte_capacity as usize,
            max_entry_bytes: max_entry_bytes as usize,
            devices: PyTuple::new(py, devices)?.unbind(),
            request_capacity: request_capacity as usize,
            relay_depth: relay_depth as usize,
            buffer_pool,
            event_pool,
            exports: PyDict::new(py).unbind(),
            state: Mutex::new(StoreState {
                buffers: HashMap::new(),
                calls: HashMap::new(),
                imports: HashMap::new(),
                arenas: HashMap::new(),
                views: HashMap::new(),
                lanes: HashMap::new(),
                call_lanes: HashMap::new(),
                allocated_bytes: 0,
            }),
        })
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        self.exports.bind(py).clear();
        let mut state = self.lock(py)?;
        state.buffers.clear();
        state.calls.clear();
        state.imports.clear();
        state.arenas.clear();
        state.views.clear();
        state.lanes.clear();
        state.call_lanes.clear();
        state.allocated_bytes = 0;
        Ok(())
    }

    fn defer_write(&self, py: Python<'_>, write: &Bound<'_, Buffer>) -> PyResult<()> {
        let state = self.lock(py)?;
        state.require_buffer(py, write)?;
        write.borrow_mut().defer(py)
    }

    fn validate_writes(&self, py: Python<'_>, writes: Vec<Bound<'_, Buffer>>) -> PyResult<()> {
        let state = self.lock(py)?;
        for write in writes {
            state.require_buffer(py, &write)?;
            write.borrow().require_candidate(py)?;
        }
        Ok(())
    }

    fn commit_writes(&self, py: Python<'_>, writes: Vec<Bound<'_, Buffer>>) -> PyResult<()> {
        let mut state = self.lock(py)?;
        let mut seen = HashSet::new();
        let mut ready = Vec::new();
        for write in writes {
            state.require_buffer(py, &write)?;
            let value = write.borrow();
            if value.state == WriteState::Deferred {
                continue;
            }
            value.require_candidate(py)?;
            if !seen.insert(buffer_key(value.id)) {
                return Err(invariant(
                    py,
                    "device-product commit repeats a product identity",
                ));
            }
            drop(value);
            ready.push(write);
        }
        for write in ready {
            let mut value = write.borrow_mut();
            value.state = WriteState::Committed;
            state
                .calls
                .entry((value.id.owner, value.id.producer_call_id))
                .or_default()
                .insert(buffer_key(value.id));
        }
        Ok(())
    }

    fn retain_publication(
        &self,
        py: Python<'_>,
        write: &Bound<'_, Buffer>,
        retirement: Py<PyAny>,
    ) -> PyResult<()> {
        let state = self.lock(py)?;
        state.require_buffer(py, write)?;
        let value = write.borrow();
        let mut retained = Vec::new();
        for export in &value.exports {
            let future = export.bind(py);
            if !future.call_method0("done")?.extract::<bool>()?
                || !future.call_method0("exception")?.is_none()
            {
                retained.push(export.clone_ref(py));
            }
        }
        retained.push(retirement);
        drop(value);
        write.borrow_mut().exports = retained;
        Ok(())
    }

    fn retain_transfer(
        &self,
        py: Python<'_>,
        write: &Bound<'_, Buffer>,
        ticket: Py<PyAny>,
    ) -> PyResult<()> {
        let state = self.lock(py)?;
        state.require_buffer(py, write)?;
        let mut value = write.borrow_mut();
        if value.state.produced() || value.released {
            return Err(invariant(py, "transfer destination already has a producer"));
        }
        value.transfers.push(ticket);
        Ok(())
    }

    fn abandon_writes(&self, py: Python<'_>, writes: Vec<Bound<'_, Buffer>>) -> PyResult<()> {
        let mut state = self.lock(py)?;
        for write in writes {
            if state.require_buffer(py, &write).is_ok() {
                write.borrow_mut().released = true;
            }
        }
        state.reclaim(py, &self.buffer_pool, &self.event_pool)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.devices)?;
        visit.call(&self.buffer_pool)?;
        visit.call(&self.event_pool)?;
        visit.call(&self.exports)?;
        let state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::Poisoned(error)) => error.into_inner(),
            Err(TryLockError::WouldBlock) => return Ok(()),
        };
        for buffer in state.buffers.values() {
            visit.call(buffer)?;
        }
        for import in state.imports.values() {
            visit.call(import)?;
        }
        for tensor in state.arenas.values().chain(state.views.values()) {
            visit.call(tensor)?;
        }
        Ok(())
    }
}

impl TensorStore {
    #[allow(clippy::too_many_arguments)]
    fn bind_group<'py>(
        &self,
        py: Python<'py>,
        state: &mut StoreState,
        bindings: &[(Bound<'py, PyAny>, Bound<'py, PyAny>)],
        request_slots: Option<&Bound<'py, PyAny>>,
        allocations: Option<&Bound<'py, PyAny>>,
        regions: Option<&Bound<'py, PyAny>>,
        shapes: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Vec<Py<Buffer>>> {
        if bindings.is_empty() {
            return Ok(Vec::new());
        }
        let slices = py.import("uniserve._slices")?;
        let mut persistent = 0;
        for (reference, _) in bindings {
            let shape = logical_shape(py, reference, shapes)?;
            let shape = PyTuple::new(py, shape)?;
            if !reference
                .getattr("shape_bound")?
                .call_method1("contains_shape", (&shape,))?
                .extract::<bool>()?
            {
                return Err(invalid(
                    py,
                    "tensor binding shape disagrees with its logical bounds",
                ));
            }
            if let Some(region) = mapping_value(regions, reference)?
                && !slices
                    .getattr("within")?
                    .call1((region, shape))?
                    .extract::<bool>()?
            {
                return Err(invalid(
                    py,
                    "tensor binding region disagrees with its logical bounds",
                ));
            }
            if mapping_value(allocations, &reference.getattr("buffer_id")?)?.is_some() {
                persistent += 1;
            }
        }
        if persistent > 0 {
            if persistent != bindings.len() {
                return Err(invalid(
                    py,
                    "persistent-buffer bindings cannot share a generic group",
                ));
            }
            let allocations = allocations
                .ok_or_else(|| invariant(py, "persistent buffers lost their placement"))?;
            return self.bind_persistent(py, state, bindings, allocations, regions, shapes, false);
        }
        if let Some(request_slots) = request_slots {
            return self.bind_relays(py, state, bindings, request_slots);
        }
        Err(invalid(
            py,
            "tensor output requires a buffer allocation or request relay slot",
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn bind_persistent<'py>(
        &self,
        py: Python<'py>,
        state: &mut StoreState,
        bindings: &[(Bound<'py, PyAny>, Bound<'py, PyAny>)],
        allocations: &Bound<'py, PyAny>,
        regions: Option<&Bound<'py, PyAny>>,
        shapes: Option<&Bound<'py, PyAny>>,
        feature: bool,
    ) -> PyResult<Vec<Py<Buffer>>> {
        state.reclaim(py, &self.buffer_pool, &self.event_pool)?;
        let mut seen = HashSet::new();
        let mut counts = HashMap::<String, usize>::new();
        for buffer in state.buffers.values() {
            let buffer = buffer.borrow(py);
            if !buffer.feature && matches!(buffer.backing, Some(Backing::Persistent(_))) {
                *counts.entry(buffer.device.clone()).or_default() += 1;
            }
        }
        let mut candidates = Vec::new();
        for (reference, raw_device) in bindings {
            let buffer = reference.getattr("buffer_id")?;
            let id = crate::protocol::buffer_id(&buffer)?;
            let key = buffer_key(id);
            if !seen.insert(key) {
                return Err(invalid(
                    py,
                    "tensor registration repeats an output identity",
                ));
            }
            if state.buffers.contains_key(&key) {
                return Err(invalid(py, "persistent output is already registered"));
            }
            let allocation = mapping_value(Some(allocations), &buffer)?
                .ok_or_else(|| invalid(py, "persistent output has no buffer allocation"))?;
            let device = canonical_device(py, raw_device)?;
            let name = device.str()?.to_str()?.to_owned();
            if feature {
                let dtype: String = reference.getattr("dtype")?.getattr("value")?.extract()?;
                if !matches!(dtype.as_str(), "f16" | "bf16" | "f32") {
                    return Err(invalid(py, "encoder feature dtype is unsupported"));
                }
                let bytes: usize = reference.getattr("max_bytes")?.extract()?;
                if bytes > self.max_entry_bytes {
                    return Err(resource(
                        py,
                        format!(
                            "encoder feature exceeds the fixed entry byte capacity: requested={bytes}, capacity={}",
                            self.max_entry_bytes
                        ),
                    ));
                }
                if !self.devices.bind(py).contains(&device)? {
                    return Err(invalid(py, "encoder feature names an undeclared device"));
                }
            } else {
                let count = counts.entry(name.clone()).or_default();
                *count += 1;
                if *count > self.capacity {
                    return Err(resource(
                        py,
                        format!(
                            "device-product arena for {name} has no query-ready free generation"
                        ),
                    ));
                }
            }
            candidates.push((reference, device, allocation));
        }

        let slices = py.import("uniserve._slices")?;
        let mut buffers = Vec::new();
        let result = (|| -> PyResult<()> {
            for (reference, device, allocation) in candidates {
                let dtype = tensor_dtype(py, reference)?;
                let shape = logical_shape(py, reference, shapes)?;
                let mut region = mapping_value(regions, reference)?;
                let full = full_region(py, &shape)?;
                if let Some(local) = &region
                    && local.eq(&full)?
                {
                    region = None;
                }
                let local_shape: Vec<usize> = match &region {
                    Some(region) => slices.getattr("shape")?.call1((region,))?.extract()?,
                    None => shape.clone(),
                };
                let full_storage = region.is_some()
                    && allocation.getattr("bytes")?.extract::<usize>()?
                        >= tensor_bytes(py, &shape, &dtype)?;
                let storage_shape = if full_storage { &shape } else { &local_shape };
                let storage_shape = storage_shape
                    .iter()
                    .map(|&dim| {
                        i64::try_from(dim).map_err(|_| {
                            invalid(py, "buffer tensor shape has an invalid byte extent")
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                let binding = self
                    .buffer_pool
                    .get()
                    .bind(py, reference, &allocation, &device, &dtype, storage_shape)?
                    .into_bound(py);
                let view = binding.getattr("tensor")?;
                let view = if full_storage {
                    match &region {
                        Some(region) => match view.get_item(region) {
                            Ok(view) => view,
                            Err(error) => {
                                self.buffer_pool.get().release(py, &binding)?;
                                return Err(error);
                            }
                        },
                        None => view,
                    }
                } else {
                    view
                };
                let buffer = create_buffer(
                    py,
                    reference,
                    view,
                    shape,
                    region,
                    feature,
                    Backing::Persistent(binding.clone().unbind()),
                );
                let buffer = match buffer {
                    Ok(buffer) => buffer,
                    Err(error) => {
                        self.buffer_pool.get().release(py, &binding)?;
                        return Err(error);
                    }
                };
                state
                    .buffers
                    .insert(buffer_key(buffer.borrow(py).id), buffer.clone_ref(py));
                buffers.push(buffer);
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.rollback(py, state, buffers)?;
            return Err(error);
        }
        Ok(buffers)
    }

    fn bind_relays<'py>(
        &self,
        py: Python<'py>,
        state: &mut StoreState,
        bindings: &[(Bound<'py, PyAny>, Bound<'py, PyAny>)],
        request_slots: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<Py<Buffer>>> {
        if self.request_capacity == 0 || self.relay_depth == 0 {
            return Err(resource(py, "worker has no request-relay arena"));
        }
        state.reclaim(py, &self.buffer_pool, &self.event_pool)?;
        let mut seen = HashSet::new();
        let mut fields = HashMap::<(RelayCall, String), usize>::new();
        let mut buffers = Vec::new();
        let result = (|| -> PyResult<()> {
            for (reference, raw_device) in bindings {
                let id = crate::protocol::buffer_id(&reference.getattr("buffer_id")?)?;
                let key = buffer_key(id);
                if !seen.insert(key) {
                    return Err(invalid(
                        py,
                        "request-relay registration repeats an output identity",
                    ));
                }
                if let Some(existing) = state.buffers.get(&key) {
                    let existing = existing.borrow(py);
                    let message = if existing.state != WriteState::Committed {
                        "request-relay output already has a candidate"
                    } else if !existing.reference.bind(py).eq(reference)? {
                        "stale request-relay logical generation"
                    } else {
                        "request-relay output is already registered"
                    };
                    return Err(invalid(py, message));
                }
                if id.generation == 0 {
                    return Err(invalid(
                        py,
                        "request-relay registration requires a positive logical generation",
                    ));
                }
                let slot = mapping_value(Some(request_slots), &reference.getattr("request_key")?)?
                    .map(|slot| slot.extract::<usize>())
                    .transpose()?
                    .unwrap_or(0);
                let shape = logical_shape(py, reference, None)?;
                if slot == 0 || slot > self.request_capacity || shape.iter().any(|&dim| dim != 1) {
                    return Err(invalid(
                        py,
                        "request-relay output has an invalid slot or scalar shape",
                    ));
                }
                let device = canonical_device(py, raw_device)?;
                let name = device.str()?.to_str()?.to_owned();
                let dtype = tensor_dtype(py, reference)?;
                let dtype_name = dtype.str()?.to_str()?.to_owned();
                let call = (name.clone(), slot, id.owner, id.producer_call_id);
                let field = fields
                    .entry((call.clone(), dtype_name.clone()))
                    .or_default();
                let field_index = *field;
                *field += 1;
                let lane = state
                    .call_lanes
                    .get(&call)
                    .copied()
                    .or_else(|| {
                        (0..self.relay_depth)
                            .find(|&lane| !state.lanes.contains_key(&(name.clone(), slot, lane)))
                    })
                    .ok_or_else(|| resource(py, "request-relay unresolved window is exhausted"))?;
                let lane_key = (name.clone(), slot, lane);
                if state
                    .lanes
                    .get(&lane_key)
                    .is_some_and(|lane| lane.fields.contains(&(dtype_name.clone(), field_index)))
                {
                    return Err(invariant(
                        py,
                        "request-relay lane was assigned more than once",
                    ));
                }
                let relay = (lane_key.clone(), dtype_name.clone(), field_index);
                let view = match state.views.get(&relay) {
                    Some(view) => view.clone_ref(py),
                    None => {
                        let arena_key = (name, dtype_name.clone(), field_index);
                        if !state.arenas.contains_key(&arena_key) {
                            let elements = self
                                .request_capacity
                                .checked_add(1)
                                .and_then(|rows| rows.checked_mul(self.relay_depth))
                                .ok_or_else(|| {
                                    resource(py, "request-relay byte capacity overflows")
                                })?;
                            let bytes = tensor_bytes(py, &[elements], &dtype)?;
                            let projected =
                                state.allocated_bytes.checked_add(bytes).ok_or_else(|| {
                                    resource(py, "request-relay byte capacity overflows")
                                })?;
                            if projected > self.byte_capacity {
                                return Err(resource(
                                    py,
                                    format!(
                                        "device-product byte capacity is exhausted ({projected}>{})",
                                        self.byte_capacity
                                    ),
                                ));
                            }
                            let empty = if device.getattr("type")?.extract::<String>()? == "cuda" {
                                py.import("uniserve_kernels.peer_storage")?
                                    .getattr("empty")?
                            } else {
                                py.import("torch")?.getattr("empty")?
                            };
                            let kwargs = PyDict::new(py);
                            kwargs.set_item("dtype", &dtype)?;
                            kwargs.set_item("device", &device)?;
                            let arena = empty.call(((elements,),), Some(&kwargs))?.unbind();
                            state.arenas.insert(arena_key.clone(), arena);
                            state.allocated_bytes = projected;
                        }
                        let arena = state
                            .arenas
                            .get(&arena_key)
                            .ok_or_else(|| invariant(py, "request-relay arena disappeared"))?;
                        let view = arena
                            .bind(py)
                            .call_method1("narrow", (0, slot * self.relay_depth + lane, 1))?
                            .unbind();
                        state.views.insert(relay.clone(), view.clone_ref(py));
                        view
                    }
                };
                let buffer = create_buffer(
                    py,
                    reference,
                    view.into_bound(py),
                    vec![1],
                    None,
                    false,
                    Backing::Relay(relay),
                )?;
                state
                    .lanes
                    .entry(lane_key)
                    .or_insert_with(|| Lane {
                        call: (id.owner, id.producer_call_id),
                        fields: HashSet::new(),
                    })
                    .fields
                    .insert((dtype_name, field_index));
                state.call_lanes.insert(call, lane);
                state.buffers.insert(key, buffer.clone_ref(py));
                buffers.push(buffer);
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.rollback(py, state, buffers)?;
            return Err(error);
        }
        Ok(buffers)
    }

    fn rollback(
        &self,
        py: Python<'_>,
        state: &mut StoreState,
        buffers: Vec<Py<Buffer>>,
    ) -> PyResult<()> {
        for buffer in buffers.into_iter().rev() {
            let id = buffer.borrow(py).id;
            state.buffers.remove(&buffer_key(id));
            state.release_backing(py, buffer.bind(py), &self.buffer_pool)?;
        }
        Ok(())
    }

    fn lock(&self, py: Python<'_>) -> PyResult<MutexGuard<'_, StoreState>> {
        self.state
            .lock_py_attached(py)
            .map_err(|_| invariant(py, "tensor store lock is poisoned"))
    }
}

fn mapping_value<'py>(
    mapping: Option<&Bound<'py, PyAny>>,
    key: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    match mapping {
        None => Ok(None),
        Some(mapping) => {
            let value = mapping.call_method1("get", (key,))?;
            Ok((!value.is_none()).then_some(value))
        }
    }
}

fn canonical_device<'py>(
    py: Python<'py>,
    device: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("uniserve.runtime.device")?
        .getattr("canonical_device")?
        .call1((device,))
}

fn tensor_dtype<'py>(
    py: Python<'py>,
    reference: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("uniserve_worker.storage.tensor_store")?
        .getattr("_device_dtype")?
        .call1((reference.getattr("dtype")?,))
}

fn logical_shape<'py>(
    py: Python<'py>,
    reference: &Bound<'py, PyAny>,
    shapes: Option<&Bound<'py, PyAny>>,
) -> PyResult<Vec<usize>> {
    match mapping_value(shapes, reference)? {
        Some(shape) => shape.extract(),
        None => py
            .import("uniserve_worker.storage.tensor_store")?
            .getattr("_device_shape")?
            .call1((reference,))?
            .extract(),
    }
}

fn tensor_bytes(py: Python<'_>, shape: &[usize], dtype: &Bound<'_, PyAny>) -> PyResult<usize> {
    let element_bytes: usize = dtype.getattr("itemsize")?.extract()?;
    shape
        .iter()
        .try_fold(element_bytes, |bytes, &dim| bytes.checked_mul(dim))
        .ok_or_else(|| invalid(py, "buffer tensor shape has an invalid byte extent"))
}

fn full_region<'py>(py: Python<'py>, shape: &[usize]) -> PyResult<Bound<'py, PyTuple>> {
    let slice = py.import("builtins")?.getattr("slice")?;
    PyTuple::new(
        py,
        shape
            .iter()
            .map(|&dim| slice.call1((0, dim)))
            .collect::<PyResult<Vec<_>>>()?,
    )
}

#[allow(clippy::too_many_arguments)]
fn create_buffer<'py>(
    py: Python<'py>,
    reference: &Bound<'py, PyAny>,
    tensor: Bound<'py, PyAny>,
    logical_shape: Vec<usize>,
    region: Option<Bound<'py, PyAny>>,
    feature: bool,
    backing: Backing,
) -> PyResult<Py<Buffer>> {
    Py::new(
        py,
        Buffer {
            id: crate::protocol::buffer_id(&reference.getattr("buffer_id")?)?,
            reference: reference.clone().unbind(),
            device: tensor.getattr("device")?.str()?.to_str()?.to_owned(),
            logical_shape,
            tensor: tensor.unbind(),
            region: region.map(Bound::unbind),
            metadata: None,
            feature,
            backing: Some(backing),
            state: WriteState::Reserved,
            extent: 0,
            value_shape: Vec::new(),
            producer: None,
            readers: 0,
            reader_events: Vec::new(),
            transfers: Vec::new(),
            exports: Vec::new(),
            released: false,
        },
    )
}

fn numerical(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("uniserve_worker.storage.tensor_store")
}

fn value_view<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyAny>,
    shape: &[usize],
    extent: usize,
) -> PyResult<Bound<'py, PyAny>> {
    numerical(py)?
        .getattr("_value_view")?
        .call1((tensor, PyTuple::new(py, shape)?, extent))
}

fn current_stream<'py>(py: Python<'py>, device: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    py.import("torch")?
        .getattr("cuda")?
        .call_method1("current_stream", (device,))
}

fn buffer_ids(values: &Bound<'_, PyAny>) -> PyResult<HashSet<BufferId>> {
    values
        .try_iter()?
        .map(|value| crate::protocol::buffer_id(&value?))
        .collect()
}

fn request_keys(values: &Bound<'_, PyAny>) -> PyResult<HashSet<RequestKey>> {
    values
        .try_iter()?
        .map(|value| crate::protocol::request_key(&value?))
        .collect()
}

fn optional_equal(
    py: Python<'_>,
    left: Option<&Py<PyAny>>,
    right: Option<&Bound<'_, PyAny>>,
) -> PyResult<bool> {
    match (left, right) {
        (None, None) => Ok(true),
        (Some(left), Some(right)) => left.bind(py).eq(right),
        _ => Ok(false),
    }
}

fn import_read(py: Python<'_>, imported: &Py<TensorImport>) -> PyResult<Py<TensorRead>> {
    let value = imported.borrow(py);
    Py::new(
        py,
        TensorRead {
            tensor: value.tensor.clone_ref(py),
            region: None,
            metadata: value
                .metadata
                .as_ref()
                .map(|metadata| metadata.clone_ref(py)),
            buffer: value.buffer.clone_ref(py),
            consumer: None,
            imported: Some(imported.clone_ref(py)),
            complete: false,
        },
    )
}
