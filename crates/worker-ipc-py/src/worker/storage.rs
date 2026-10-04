//! PyTorch views and transport submission for native tensor storage.

use std::collections::{HashMap, HashSet};
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, TryLockError};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{PyDict, PyList, PyTuple};
use uniserve_core::CallId;
use uniserve_worker_ipc::{BufferId, RequestKey};

use super::buffer::BufferPool;
use super::completion::{Completion, CompletionRef};
use super::error::{invalid, invariant, native_error, resource};
use super::events::{CUDAEvent, EventPool};
use super::transfer::{TransferRef, TransferTicket};
use uniserve_worker::cuda::Event;
use uniserve_worker::tensor::{self, Backing, Fence, WriteState, buffer_key};

type NativeBuffer = tensor::Buffer<TransferRef, CompletionRef>;
type NativeImport = tensor::TensorImport<BufferHandle, TransferRef>;
type NativeRead = tensor::TensorRead<BufferHandle, ImportHandle>;
type StoreState = tensor::TensorStore<BufferHandle, ImportHandle, Py<PyAny>>;
type ReadRequest<'py> = (
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Option<Bound<'py, PyAny>>,
);

struct BufferViews {
    reference: Py<PyAny>,
    tensor: Py<PyAny>,
    region: Option<Py<PyAny>>,
    metadata: Option<Py<PyAny>>,
    storage: Py<PyAny>,
}

/// A foreign owner and a direct native borrow. Each retained handle has one
/// Python reference, so GC sees the same ownership graph as native storage.
pub(crate) struct BufferHandle {
    owner: Py<Buffer>,
    inner: Arc<Mutex<NativeBuffer>>,
}

impl BufferHandle {
    fn new(owner: Py<Buffer>) -> Self {
        let inner = Arc::clone(&owner.get().inner);
        Self { owner, inner }
    }

    fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            owner: self.owner.clone_ref(py),
            inner: Arc::clone(&self.inner),
        }
    }
}

impl Deref for BufferHandle {
    type Target = Mutex<NativeBuffer>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub(crate) struct ImportHandle {
    owner: Py<TensorImport>,
    inner: Arc<Mutex<NativeImport>>,
}

impl ImportHandle {
    fn new(owner: Py<TensorImport>) -> Self {
        let inner = Arc::clone(&owner.get().inner);
        Self { owner, inner }
    }

    fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            owner: self.owner.clone_ref(py),
            inner: Arc::clone(&self.inner),
        }
    }
}

impl Deref for ImportHandle {
    type Target = Mutex<NativeImport>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct Buffer {
    inner: Arc<Mutex<NativeBuffer>>,
    views: Mutex<BufferViews>,
}

#[pymethods]
impl Buffer {
    #[getter]
    fn reference(&self, py: Python<'_>) -> Py<PyAny> {
        lock(py, &self.views).reference.clone_ref(py)
    }

    #[getter]
    fn tensor(&self, py: Python<'_>) -> Py<PyAny> {
        lock(py, &self.views).tensor.clone_ref(py)
    }

    #[getter]
    fn region(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        lock(py, &self.views)
            .region
            .as_ref()
            .map(|value| value.clone_ref(py))
    }

    #[getter]
    fn metadata(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        lock(py, &self.views)
            .metadata
            .as_ref()
            .map(|value| value.clone_ref(py))
    }

    #[getter]
    fn feature(&self, py: Python<'_>) -> bool {
        lock(py, &self.inner).feature
    }

    #[getter]
    fn logical_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &lock(py, &self.inner).logical_shape)
    }

    #[getter]
    fn producer_recorded(&self, py: Python<'_>) -> bool {
        lock(py, &self.inner).state.produced()
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        let Ok(views) = self.views.try_lock() else {
            return Ok(());
        };
        visit.call(&views.reference)?;
        visit.call(&views.tensor)?;
        visit.call(&views.region)?;
        visit.call(&views.metadata)?;
        visit.call(&views.storage)?;

        let Ok(buffer) = self.inner.try_lock() else {
            return Ok(());
        };
        for transfer in &buffer.transfers {
            visit.call(&transfer.owner)?;
        }
        for completion in &buffer.exports {
            visit.call(&completion.owner)?;
        }
        Ok(())
    }
}

struct BufferAccess<'a> {
    native: MutexGuard<'a, NativeBuffer>,
    views: MutexGuard<'a, BufferViews>,
}

impl Deref for BufferAccess<'_> {
    type Target = NativeBuffer;

    fn deref(&self) -> &NativeBuffer {
        &self.native
    }
}

impl DerefMut for BufferAccess<'_> {
    fn deref_mut(&mut self) -> &mut NativeBuffer {
        &mut self.native
    }
}

fn buffer_view<'a>(py: Python<'_>, buffer: &'a BufferHandle) -> BufferAccess<'a> {
    BufferAccess {
        native: lock(py, buffer),
        views: lock(py, &buffer.owner.get().views),
    }
}

struct ImportAccess<'a> {
    native: MutexGuard<'a, NativeImport>,
    views: &'a TensorImport,
}

impl Deref for ImportAccess<'_> {
    type Target = NativeImport;

    fn deref(&self) -> &NativeImport {
        &self.native
    }
}

impl DerefMut for ImportAccess<'_> {
    fn deref_mut(&mut self) -> &mut NativeImport {
        &mut self.native
    }
}

fn import_view<'a>(py: Python<'_>, imported: &'a ImportHandle) -> ImportAccess<'a> {
    ImportAccess {
        native: lock(py, imported),
        views: imported.owner.get(),
    }
}

#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct TensorRead {
    #[pyo3(get)]
    tensor: Py<PyAny>,
    #[pyo3(get)]
    region: Option<Py<PyAny>>,
    #[pyo3(get)]
    metadata: Option<Py<PyAny>>,
    inner: NativeRead,
}

impl Deref for TensorRead {
    type Target = NativeRead;

    fn deref(&self) -> &NativeRead {
        &self.inner
    }
}

impl DerefMut for TensorRead {
    fn deref_mut(&mut self) -> &mut NativeRead {
        &mut self.inner
    }
}

#[pymethods]
impl TensorRead {
    #[getter]
    fn imported(&self, py: Python<'_>) -> Option<Py<TensorImport>> {
        self.imported
            .as_ref()
            .map(|imported| imported.owner.clone_ref(py))
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.tensor)?;
        visit.call(&self.region)?;
        visit.call(&self.metadata)?;
        visit.call(&self.buffer.owner)?;
        if let Some(imported) = &self.imported {
            visit.call(&imported.owner)?;
        }
        Ok(())
    }
}

#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct TensorImport {
    inner: Arc<Mutex<NativeImport>>,
    tensor: Py<PyAny>,
    metadata: Option<Py<PyAny>>,
}

#[pymethods]
impl TensorImport {
    #[getter]
    fn tickets<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            lock(py, &self.inner)
                .tickets
                .iter()
                .map(|ticket| ticket.owner.bind(py)),
        )
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.tensor)?;
        visit.call(&self.metadata)?;
        let Ok(imported) = self.inner.try_lock() else {
            return Ok(());
        };
        visit.call(&imported.buffer.owner)?;
        for ticket in &imported.tickets {
            visit.call(&ticket.owner)?;
        }
        Ok(())
    }
}

fn lock<'a, T>(py: Python<'_>, value: &'a Mutex<T>) -> MutexGuard<'a, T> {
    // Numerical calls may release the GIL while holding a borrowed view.
    value
        .lock_py_attached(py)
        .unwrap_or_else(PoisonError::into_inner)
}

fn require_reference(
    py: Python<'_>,
    state: &StoreState,
    reference: &Bound<'_, PyAny>,
) -> PyResult<BufferHandle> {
    let id = super::protocol::buffer_id(&reference.getattr("buffer_id")?)?;
    let buffer = state
        .require_reference(id)
        .map_err(|error| native_error(py, error))?;
    if !lock(py, &buffer.owner.get().views)
        .reference
        .bind(py)
        .eq(reference)?
    {
        return Err(invalid(py, "stale device-product logical generation"));
    }
    Ok(buffer.clone_ref(py))
}

/// Own committed buffer lookup and retirement; borrow fixed physical arenas
/// and pooled CUDA events from the worker's resource owners.
#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct TensorStore {
    #[pyo3(get)]
    max_feature_bytes: usize,
    #[pyo3(get)]
    devices: Py<PyTuple>,
    #[pyo3(get)]
    buffer_pool: Py<BufferPool>,
    #[pyo3(get)]
    event_pool: Py<EventPool>,
    #[pyo3(get)]
    exports: Py<PyDict>,
    state: Mutex<StoreState>,
}

impl TensorStore {
    fn reclaim(&self, py: Python<'_>, state: &mut StoreState) -> PyResult<()> {
        let retired = state.reclaim().map_err(|error| native_error(py, error))?;
        let mut released: HashMap<usize, (Arc<Event>, usize)> = HashMap::new();
        for (buffer, backing) in retired {
            if let Backing::Persistent(binding) = backing {
                self.buffer_pool.get().release_binding(py, &binding)?;
            }
            for event in lock(py, &buffer).events() {
                let item = released
                    .entry(Arc::as_ptr(event) as usize)
                    .or_insert_with(|| (Arc::clone(event), 0));
                item.1 += 1;
            }
        }

        for (event, count) in released.into_values() {
            self.event_pool
                .borrow(py)
                .release(&CUDAEvent { inner: event }, count)?;
        }
        Ok(())
    }

    fn release_backing(
        &self,
        py: Python<'_>,
        state: &mut StoreState,
        buffer: &BufferHandle,
    ) -> PyResult<()> {
        let backing = state
            .release_backing(buffer)
            .map_err(|error| native_error(py, error))?;
        if let Backing::Persistent(binding) = backing {
            self.buffer_pool.get().release_binding(py, &binding)?;
        }
        Ok(())
    }

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
            .ok_or_else(|| invariant(py, "closed or ordinary tensor read is not an import"))?;
        let imported = lock(py, imported);
        state
            .require_buffer(&imported.buffer)
            .map_err(|error| native_error(py, error))?;
        for ticket in &imported.tickets {
            ticket.owner.get().result(py, None)?;
        }
        self.wait_producer(py, &lock(py, &imported.buffer), &mut HashSet::new())
    }

    fn publish<'py>(
        &self,
        py: Python<'py>,
        buffer: &BufferHandle,
        value: &Bound<'py, PyAny>,
        event: Option<&Bound<'py, PyAny>>,
        metadata: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (reference, tensor, region, feature) = {
            let value = buffer_view(py, buffer);
            if value.state.produced() {
                return Err(invariant(py, "device product was published more than once"));
            }
            (
                value.views.reference.clone_ref(py),
                value.views.tensor.clone_ref(py),
                value
                    .views
                    .region
                    .as_ref()
                    .map(|region| region.clone_ref(py)),
                value.feature,
            )
        };
        let view = numerical(py)?
            .getattr("_copy_value")?
            .call1((reference, tensor, region, feature, value, metadata))?;
        let shape = view.getattr("shape")?.extract()?;
        let extent = view.call_method0("numel")?.extract()?;
        let fence = self.producer_fence(py, &view.getattr("device")?, event, 1)?;
        let mut buffer = buffer_view(py, buffer);
        buffer.produced(shape, extent, fence);
        buffer.views.metadata = metadata.map(|metadata| metadata.clone().unbind());
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

        let pool = self.event_pool.borrow(py);
        let fence = match event {
            Some(event) => {
                let event: PyRef<'_, CUDAEvent> = event.extract()?;
                Fence {
                    stream: pool.declare_stream(py, &event, device)?,
                    event: Arc::clone(&event.inner),
                }
            }
            None => self.record_fence(py, device)?,
        };
        pool.retain(
            py,
            &CUDAEvent {
                inner: Arc::clone(&fence.event),
            },
            device,
            count,
        )?;
        Ok(Some(fence))
    }

    fn record_fence(&self, py: Python<'_>, device: &Bound<'_, PyAny>) -> PyResult<Fence> {
        let pool = self.event_pool.borrow(py);
        let event = pool.acquire_event(py, device, false, false)?;
        let stream = pool.record(py, &event, device)?;
        Ok(Fence {
            event: event.inner,
            stream,
        })
    }

    fn wait_producer(
        &self,
        py: Python<'_>,
        buffer: &NativeBuffer,
        waited: &mut HashSet<(String, usize)>,
    ) -> PyResult<()> {
        if let Some(fence) = &buffer.producer {
            let key = (buffer.device.clone(), Arc::as_ptr(&fence.event) as usize);
            if waited.insert(key) {
                let device = pyo3::types::PyString::new(py, &buffer.device);
                let stream = current_stream(py, device.as_any())?;
                if stream.getattr("cuda_stream")?.extract::<usize>()? != fence.stream {
                    CUDAEvent {
                        inner: Arc::clone(&fence.event),
                    }
                    .wait(py, Some(&stream))?;
                }
            }
        }
        Ok(())
    }

    fn wake_retirement(&self, py: Python<'_>, buffer: &BufferHandle) -> PyResult<()> {
        let (device, events) = {
            let value = lock(py, buffer);
            (
                value.device.clone(),
                value.events().cloned().collect::<Vec<_>>(),
            )
        };
        let device = pyo3::types::PyString::new(py, &device);
        for event in events {
            if !event
                .ready()
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)?
            {
                self.event_pool.borrow(py).schedule_completion_wake(
                    py,
                    device.as_any(),
                    &CUDAEvent { inner: event },
                )?;
            }
        }
        Ok(())
    }

    fn retire_buffer(&self, py: Python<'_>, buffer: &BufferHandle) -> PyResult<()> {
        self.wake_retirement(py, buffer)?;
        let transfers = {
            let value = lock(py, buffer);
            if value.state.produced() {
                Vec::new()
            } else {
                value
                    .transfers
                    .iter()
                    .map(|ticket| ticket.clone_ref(py))
                    .collect()
            }
        };
        for ticket in transfers {
            ticket.owner.get().cancel(py)?;
        }
        Ok(())
    }
}

#[pymethods]
impl TensorStore {
    #[getter]
    fn capacity(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.lock(py)?.capacity)
    }

    #[getter]
    fn byte_capacity(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.lock(py)?.byte_capacity)
    }

    #[getter]
    fn request_capacity(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.lock(py)?.request_capacity)
    }

    #[getter]
    fn relay_depth(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.lock(py)?.relay_depth)
    }

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
        let id = super::protocol::buffer_id(&reference.getattr("buffer_id")?)?;
        let key = buffer_key(id);
        let shape: Vec<usize> = tensor.getattr("shape")?.extract()?;
        let mut state = self.lock(py)?;
        if let Some(pending) = state.imports.get(&key) {
            let mut value = import_view(py, pending);
            let buffer = buffer_view(py, &value.buffer);
            if !buffer.views.reference.bind(py).eq(&reference)?
                || !value.views.tensor.bind(py).getattr("device")?.eq(&device)?
                || value
                    .views
                    .tensor
                    .bind(py)
                    .getattr("shape")?
                    .extract::<Vec<usize>>()?
                    != shape
                || !optional_equal(py, value.views.metadata.as_ref(), metadata.as_ref())?
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
            value.share();
            drop(value);
            return import_read(py, pending);
        }
        let existing = state
            .buffers
            .get(&key)
            .filter(|buffer| lock(py, buffer).state == WriteState::Committed)
            .map(|buffer| buffer.clone_ref(py));
        let full = full_region(py, &shape)?;
        let (buffer, destination, missing) = if existing.is_some() {
            let buffer = require_reference(py, &state, &reference)?;
            let value = buffer_view(py, &buffer);
            if value.released {
                return Err(invalid(
                    py,
                    "product import requires a live published generation",
                ));
            }
            if device.str()?.to_str()? != value.device
                || !optional_equal(py, value.views.metadata.as_ref(), metadata.as_ref())?
            {
                return Err(invalid(
                    py,
                    "product import conflicts with resident ownership",
                ));
            }
            let (destination, missing) = match &value.views.region {
                None => {
                    if value.value_shape != shape {
                        return Err(invalid(py, "product import changes resident tensor shape"));
                    }
                    (
                        value_view(py, value.views.tensor.bind(py), &shape, value.extent)?,
                        Vec::new(),
                    )
                }
                Some(region) => {
                    let destination = match &value.backing {
                        Some(Backing::Persistent(_)) => value.views.storage.bind(py).clone(),
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
                        if !ticket.succeeded() {
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
            let destination = value_view(
                py,
                buffer_view(py, &buffer).views.tensor.bind(py),
                &shape,
                extent,
            )?;
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
                lock(py, &buffer).released = true;
                self.reclaim(py, &mut state)?;
            }
            return Err(invalid(py, "product import changes resident tensor dtype"));
        }
        lock(py, &buffer).readers += 1;
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
        let tickets: Vec<TransferRef> = tickets
            .extract::<Vec<Py<TransferTicket>>>()?
            .into_iter()
            .map(|ticket| TransferRef::new(py, ticket))
            .collect::<PyResult<_>>()?;
        lock(py, &buffer)
            .transfers
            .extend(tickets.iter().map(|ticket| ticket.clone_ref(py)));
        if let Err(error) = submitted {
            lock(py, &buffer).readers -= 1;
            if existing.is_none() {
                lock(py, &buffer).released = true;
            }
            drop(state);

            for ticket in tickets {
                ticket.owner.get().cancel(py)?;
                TransferTicket::close(ticket.owner.into_bound(py))?;
            }
            let mut state = self.lock(py)?;
            self.reclaim(py, &mut state)?;
            return Err(error);
        }
        let committed = existing.is_some() && buffer_view(py, &buffer).views.region.is_none();
        let imported = ImportHandle::new(Py::new(
            py,
            TensorImport {
                inner: Arc::new(Mutex::new(NativeImport::new(buffer, tickets, committed))),
                tensor: destination.unbind(),
                metadata: metadata.map(Bound::unbind),
            },
        )?);
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
        self.event_pool.borrow(py).reap(py)?;

        let mut state = self.lock(py)?;
        self.wait_imported(py, &state, read)?;
        let imported = read
            .borrow()
            .imported
            .as_ref()
            .ok_or_else(|| invariant(py, "closed or ordinary tensor read is not an import"))?
            .clone_ref(py);
        let value = import_view(py, &imported);
        if value.committed {
            return Ok(());
        }
        let buffer = &value.buffer;
        let tensor = value.views.tensor.bind(py);
        if !lock(py, buffer).state.produced() {
            self.publish(
                py,
                buffer,
                tensor,
                None,
                value
                    .views
                    .metadata
                    .as_ref()
                    .map(|metadata| metadata.bind(py)),
            )?;
            state
                .commit_writes(&[buffer.clone_ref(py)])
                .map_err(|error| native_error(py, error))?;
        } else {
            // Existing leases retain their original shard views. A new fence
            // covers the union, including every newly fetched region.
            let device = tensor.getattr("device")?;
            let fence = self.producer_fence(py, &device, None, 1)?;
            let previous = buffer_view(py, buffer)
                .producer
                .as_ref()
                .map(|fence| Arc::clone(&fence.event));
            if let Some(previous) = previous {
                let owner = buffer.owner.clone_ref(py).into_any();
                self.event_pool
                    .borrow(py)
                    .defer_events(vec![previous], owner, None)?;
            }
            let shape: Vec<usize> = tensor.getattr("shape")?.extract()?;
            let extent = tensor.call_method0("numel")?.extract()?;
            let mut buffer = buffer_view(py, buffer);
            buffer.views.tensor = tensor.clone().unbind();
            buffer.views.region = None;
            buffer.value_shape = shape;
            buffer.extent = extent;
            buffer.producer = fence;
        }
        drop(value);
        lock(py, &imported).committed = true;
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
        self.event_pool.borrow(py).reap(py)?;

        let write = BufferHandle::new(write.clone().unbind());

        let state = self.lock(py)?;
        state
            .require_buffer(&write)
            .map_err(|error| native_error(py, error))?;
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
        self.event_pool.borrow(py).reap(py)?;

        let writes: Vec<_> = writes
            .iter()
            .map(|write| BufferHandle::new(write.clone().unbind()))
            .collect();

        let state = self.lock(py)?;
        let mut tensors = Vec::with_capacity(writes.len());
        let mut seen = HashSet::new();
        for write in &writes {
            state
                .require_buffer(write)
                .map_err(|error| native_error(py, error))?;
            if lock(py, write).state.produced() || !seen.insert(Arc::as_ptr(&write.inner) as usize)
            {
                return Err(invariant(py, "device product was published more than once"));
            }
            tensors.push(buffer_view(py, write).views.tensor.clone_ref(py));
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
            lock(py, &write).produced(vec![1], 1, fence.clone());
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
        self.event_pool.borrow(py).reap(py)?;

        let write = BufferHandle::new(write.clone().unbind());

        let state = self.lock(py)?;
        state
            .require_buffer(&write)
            .map_err(|error| native_error(py, error))?;
        if lock(py, &write).state.produced() {
            return Err(invariant(py, "device product was published more than once"));
        }
        let tensor = buffer_view(py, &write).views.tensor.clone_ref(py);
        let view = value_view(py, tensor.bind(py), &[1], 1)?;
        view.call_method1("fill_", (value,))?;
        let fence =
            self.producer_fence(py, &view.getattr("device")?, producer_event.as_ref(), 1)?;
        lock(py, &write).produced(vec![1], 1, fence);
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
            let buffer = require_reference(py, &state, &reference)?;
            let value = buffer_view(py, &buffer);
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
            let tensor = value_view(
                py,
                value.views.tensor.bind(py),
                &value.value_shape,
                value.extent,
            )?
            .unbind();
            let read = Py::new(
                py,
                TensorRead {
                    tensor,
                    region: value
                        .views
                        .region
                        .as_ref()
                        .map(|region| region.clone_ref(py)),
                    metadata: value
                        .views
                        .metadata
                        .as_ref()
                        .map(|metadata| metadata.clone_ref(py)),
                    inner: NativeRead {
                        buffer: buffer.clone_ref(py),
                        consumer: Some(super::protocol::call_id(&consumer)?),
                        imported: None,
                        complete: false,
                    },
                },
            )?;
            drop(value);
            resolved.push((buffer, read));
        }
        for (buffer, _) in &resolved {
            lock(py, buffer).readers += 1;
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
        self.event_pool.borrow(py).reap(py)?;

        let after_writes: Vec<_> = after_writes
            .iter()
            .map(|write| BufferHandle::new(write.clone().unbind()))
            .collect();

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
            state
                .require_buffer(&read.buffer)
                .map_err(|error| native_error(py, error))?;
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
            state
                .require_buffer(&write)
                .map_err(|error| native_error(py, error))?;
            let value = buffer_view(py, &write);
            if let Some(fence) = &value.producer {
                write_fences
                    .entry((
                        value.id.producer_call_id,
                        value.device.clone(),
                        fence.stream,
                    ))
                    .or_insert_with(|| Arc::clone(&fence.event));
            }
        }
        let mut recorded = HashMap::<String, Arc<Event>>::new();
        for read in &pending {
            let value = read.borrow();
            let buffer = &value.buffer;
            let device = value.tensor.bind(py).getattr("device")?;
            if device.getattr("type")?.extract::<String>()? != "cuda" {
                continue;
            }
            let name = lock(py, buffer).device.clone();
            let stream = current_stream(py, &device)?;
            let stream_id: usize = stream.getattr("cuda_stream")?.extract()?;
            let event = value
                .consumer
                .and_then(|consumer| write_fences.get(&(consumer, name.clone(), stream_id)))
                .map(Arc::clone);
            let event = match event {
                Some(event) => event,
                None => match recorded.get(&name) {
                    Some(event) => Arc::clone(event),
                    None => {
                        let event = self.record_fence(py, &device)?.event;
                        recorded.insert(name, Arc::clone(&event));
                        event
                    }
                },
            };
            if lock(py, buffer).retain_reader(&event) {
                self.event_pool
                    .borrow(py)
                    .retain(py, &CUDAEvent { inner: event }, &device, 1)?;
            }
        }
        let mut closing = Vec::new();
        let mut released = Vec::new();
        for read in pending {
            let (buffer, completed_import) = {
                let mut read = read.borrow_mut();
                let buffer = read.buffer.clone_ref(py);
                (buffer, read.inner.complete())
            };
            if let Some(imported) = completed_import {
                state.imports.remove(&buffer_key(lock(py, &buffer).id));
                let import = lock(py, &imported);
                closing.extend(
                    import
                        .tickets
                        .iter()
                        .map(|ticket| (ticket.clone_ref(py), !import.committed)),
                );
            }
            if lock(py, &buffer).released {
                released.push(buffer);
            }
        }
        drop(state);

        // Transport and event observers may reenter the store. Native read
        // dependencies retain backing while those observers run.
        for (ticket, cancel) in closing {
            if cancel {
                ticket.owner.get().cancel(py)?;
            }
            TransferTicket::close(ticket.owner.into_bound(py))?;
        }
        for buffer in released {
            self.wake_retirement(py, &buffer)?;
        }
        let mut state = self.lock(py)?;
        self.reclaim(py, &mut state)
    }

    fn release_calls(&self, py: Python<'_>, releases: Bound<'_, PyAny>) -> PyResult<()> {
        let calls = releases
            .try_iter()?
            .map(|release| {
                let (request, call): (Bound<'_, PyAny>, Bound<'_, PyAny>) = release?.extract()?;
                Ok((
                    super::protocol::request_key(&request)?,
                    super::protocol::call_id(&call)?,
                ))
            })
            .collect::<PyResult<Vec<_>>>()?;
        self.lock(py)?.release_calls(calls);
        Ok(())
    }

    fn release_buffers(&self, py: Python<'_>, buffers: Bound<'_, PyAny>) -> PyResult<()> {
        let buffers = PyTuple::new(py, buffers.try_iter()?.collect::<PyResult<Vec<_>>>()?)?;
        let selected = buffer_ids(&buffers)?;
        py.import("uniserve_worker.transport.exports")?
            .getattr("release_exports")?
            .call1((self.exports.bind(py), buffers))?;
        let mut state = self.lock(py)?;
        let released = state.release_buffers(&selected);
        let released = released
            .into_iter()
            .filter_map(|key| state.buffers.get(&key).map(|buffer| buffer.clone_ref(py)))
            .collect::<Vec<_>>();
        drop(state);

        for buffer in released {
            self.retire_buffer(py, &buffer)?;
        }
        let mut state = self.lock(py)?;
        self.reclaim(py, &mut state)
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
        let released = state
            .release_requests(&requests, &retained)
            .map_err(|error| native_error(py, error))?;
        let released = released
            .into_iter()
            .filter_map(|key| state.buffers.get(&key).map(|buffer| buffer.clone_ref(py)))
            .collect::<Vec<_>>();
        drop(state);

        for buffer in released {
            self.retire_buffer(py, &buffer)?;
        }
        let mut state = self.lock(py)?;
        self.reclaim(py, &mut state)
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
        let selected = |buffer: &NativeBuffer| {
            buffers.contains(&buffer.id)
                || (requests.contains(&buffer.id.owner) && !retained.contains(&buffer.id))
        };
        let mut state = self.lock(py)?;
        for buffer in state.buffers.values() {
            let value = lock(py, buffer);
            if selected(&value) {
                for ticket in &value.transfers {
                    ticket.owner.get().retirement_ready(py)?;
                }
                for export in &value.exports {
                    if export.done() {
                        export.owner.borrow(py).result(py, None)?;
                    }
                }
            }
        }
        self.reclaim(py, &mut state)?;
        Ok(!state
            .buffers
            .values()
            .any(|buffer| selected(&lock(py, buffer))))
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
        PyTuple::new(py, buffers.into_iter().map(|buffer| buffer.owner))
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
        PyTuple::new(py, buffers.into_iter().map(|buffer| buffer.owner))
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
                        buffers.iter().map(|buffer| buffer.owner.clone_ref(py)),
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
        let writes: Vec<_> = writes
            .iter()
            .map(|write| BufferHandle::new(write.clone().unbind()))
            .collect();

        let state = self.lock(py)?;
        let mut tensors = Vec::with_capacity(writes.len());
        for write in writes {
            state
                .require_buffer(&write)
                .map_err(|error| native_error(py, error))?;
            let write = buffer_view(py, &write);
            if write.state.produced() {
                return Err(invariant(py, "device product was published more than once"));
            }
            tensors.push(write.views.tensor.clone_ref(py));
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
    #[pyo3(signature = (*, capacity=0, byte_capacity=None, max_feature_bytes=1, devices=None, request_capacity=0, relay_depth=0, buffer_pool, event_pool=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        capacity: isize,
        byte_capacity: Option<isize>,
        max_feature_bytes: isize,
        devices: Option<Vec<Bound<'_, PyAny>>>,
        request_capacity: isize,
        relay_depth: isize,
        buffer_pool: Py<BufferPool>,
        event_pool: Option<Py<EventPool>>,
    ) -> PyResult<Self> {
        if capacity < 0 || max_feature_bytes < 1 {
            return Err(PyValueError::new_err("tensor store capacities are invalid"));
        }
        let byte_capacity = match byte_capacity {
            Some(bytes) => bytes,
            None => buffer_pool.get().byte_capacity(py)? as isize,
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
            None => Py::new(py, EventPool::new())?,
        };
        Ok(Self {
            max_feature_bytes: max_feature_bytes as usize,
            devices: PyTuple::new(py, devices)?.unbind(),
            buffer_pool,
            event_pool,
            exports: PyDict::new(py).unbind(),
            state: Mutex::new(StoreState::new(
                capacity as usize,
                byte_capacity as usize,
                request_capacity as usize,
                relay_depth as usize,
            )),
        })
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        self.exports.bind(py).clear();
        let retired = self.lock(py)?.close();
        drop(retired);
        Ok(())
    }

    fn defer_write(&self, py: Python<'_>, write: &Bound<'_, Buffer>) -> PyResult<()> {
        let write = BufferHandle::new(write.clone().unbind());

        let state = self.lock(py)?;
        state
            .require_buffer(&write)
            .map_err(|error| native_error(py, error))?;
        buffer_view(py, &write)
            .defer()
            .map_err(|error| native_error(py, error))
    }

    fn validate_writes(&self, py: Python<'_>, writes: Vec<Bound<'_, Buffer>>) -> PyResult<()> {
        let writes: Vec<_> = writes
            .iter()
            .map(|write| BufferHandle::new(write.clone().unbind()))
            .collect();

        self.lock(py)?
            .validate_writes(&writes)
            .map_err(|error| native_error(py, error))
    }

    fn commit_writes(&self, py: Python<'_>, writes: Vec<Bound<'_, Buffer>>) -> PyResult<()> {
        let writes: Vec<_> = writes
            .iter()
            .map(|write| BufferHandle::new(write.clone().unbind()))
            .collect();
        self.lock(py)?
            .commit_writes(&writes)
            .map_err(|error| native_error(py, error))
    }

    fn retain_export(
        &self,
        py: Python<'_>,
        write: &Bound<'_, Buffer>,
        retirement: Py<Completion>,
    ) -> PyResult<()> {
        let write = BufferHandle::new(write.clone().unbind());
        let state = self.lock(py)?;
        state
            .require_buffer(&write)
            .map_err(|error| native_error(py, error))?;
        lock(py, &write).retain_export(CompletionRef::new(py, retirement));
        Ok(())
    }

    fn retain_transfer(
        &self,
        py: Python<'_>,
        write: &Bound<'_, Buffer>,
        ticket: Py<TransferTicket>,
    ) -> PyResult<()> {
        let write = BufferHandle::new(write.clone().unbind());
        let state = self.lock(py)?;
        state
            .require_buffer(&write)
            .map_err(|error| native_error(py, error))?;
        lock(py, &write)
            .retain_transfer(TransferRef::new(py, ticket)?)
            .map_err(|error| native_error(py, error))
    }

    fn abandon_writes(&self, py: Python<'_>, writes: Vec<Bound<'_, Buffer>>) -> PyResult<()> {
        let writes: Vec<_> = writes
            .iter()
            .map(|write| BufferHandle::new(write.clone().unbind()))
            .collect();

        let mut state = self.lock(py)?;
        state.abandon_writes(&writes);
        self.reclaim(py, &mut state)
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
            visit.call(&buffer.owner)?;
        }
        for import in state.imports.values() {
            visit.call(&import.owner)?;
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
    ) -> PyResult<Vec<BufferHandle>> {
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
    ) -> PyResult<Vec<BufferHandle>> {
        self.reclaim(py, state)?;
        let mut native_bindings = Vec::new();
        let mut candidates = Vec::new();
        for (reference, raw_device) in bindings {
            let buffer = reference.getattr("buffer_id")?;
            let id = super::protocol::buffer_id(&buffer)?;
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
                if bytes > self.max_feature_bytes {
                    return Err(resource(
                        py,
                        format!(
                            "encoder feature exceeds the fixed feature byte capacity: requested={bytes}, capacity={}",
                            self.max_feature_bytes
                        ),
                    ));
                }
                if !self.devices.bind(py).contains(&device)? {
                    return Err(invalid(py, "encoder feature names an undeclared device"));
                }
            }
            native_bindings.push((id, name));
            candidates.push((reference, device, allocation));
        }

        state
            .validate_bindings(&native_bindings, feature)
            .map_err(|error| native_error(py, error))?;

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
                let view = binding.get().tensor.bind(py).clone();
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
                    Backing::Persistent(Arc::clone(&binding.get().binding)),
                    Some(binding.getattr("tensor")?.unbind()),
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
                    .insert(buffer_key(lock(py, &buffer).id), buffer.clone_ref(py));
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
    ) -> PyResult<Vec<BufferHandle>> {
        if state.request_capacity == 0 || state.relay_depth == 0 {
            return Err(resource(py, "worker has no request-relay arena"));
        }
        self.reclaim(py, state)?;
        let mut seen = HashSet::new();
        let mut fields = HashMap::<((String, usize, RequestKey, CallId), String), usize>::new();
        let mut buffers = Vec::new();
        let result = (|| -> PyResult<()> {
            for (reference, raw_device) in bindings {
                let id = super::protocol::buffer_id(&reference.getattr("buffer_id")?)?;
                let key = buffer_key(id);
                if !seen.insert(key) {
                    return Err(invalid(
                        py,
                        "request-relay registration repeats an output identity",
                    ));
                }
                if let Some(existing) = state.buffers.get(&key) {
                    let existing = buffer_view(py, existing);
                    let message = if existing.state != WriteState::Committed {
                        "request-relay output already has a candidate"
                    } else if !existing.views.reference.bind(py).eq(reference)? {
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
                if slot == 0 || slot > state.request_capacity || shape.iter().any(|&dim| dim != 1) {
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
                    .relay_lane(id, &name, slot)
                    .map_err(|error| native_error(py, error))?;
                let lane_key = (name.clone(), slot, lane);
                let relay = (lane_key.clone(), dtype_name.clone(), field_index);
                let view = match state.views.get(&relay) {
                    Some(view) => view.clone_ref(py),
                    None => {
                        let arena_key = (name, dtype_name.clone(), field_index);
                        if !state.arenas.contains_key(&arena_key) {
                            let elements = state
                                .request_capacity
                                .checked_add(1)
                                .and_then(|rows| rows.checked_mul(state.relay_depth))
                                .ok_or_else(|| {
                                    resource(py, "request-relay byte capacity overflows")
                                })?;
                            let bytes = tensor_bytes(py, &[elements], &dtype)?;
                            let projected = state
                                .arena_capacity(bytes)
                                .map_err(|error| native_error(py, error))?;
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
                            .call_method1("narrow", (0, slot * state.relay_depth + lane, 1))?
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
                    Backing::Relay(relay.clone()),
                    None,
                )?;
                state
                    .bind_relay(id, &relay)
                    .map_err(|error| native_error(py, error))?;
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
        buffers: Vec<BufferHandle>,
    ) -> PyResult<()> {
        for buffer in buffers.into_iter().rev() {
            let id = lock(py, &buffer).id;
            state.buffers.remove(&buffer_key(id));
            self.release_backing(py, state, &buffer)?;
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
    storage: Option<Py<PyAny>>,
) -> PyResult<BufferHandle> {
    let id = super::protocol::buffer_id(&reference.getattr("buffer_id")?)?;
    let device = tensor.getattr("device")?.str()?.to_str()?.to_owned();
    let inner = Arc::new(Mutex::new(NativeBuffer::new(
        id,
        device,
        logical_shape,
        feature,
        backing,
    )));
    let storage = storage.unwrap_or_else(|| tensor.clone().unbind());
    let owner = Py::new(
        py,
        Buffer {
            inner,
            views: Mutex::new(BufferViews {
                reference: reference.clone().unbind(),
                tensor: tensor.unbind(),
                region: region.map(Bound::unbind),
                metadata: None,
                storage,
            }),
        },
    )?;
    Ok(BufferHandle::new(owner))
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
        .map(|value| super::protocol::buffer_id(&value?))
        .collect()
}

fn request_keys(values: &Bound<'_, PyAny>) -> PyResult<HashSet<RequestKey>> {
    values
        .try_iter()?
        .map(|value| super::protocol::request_key(&value?))
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

fn import_read(py: Python<'_>, imported: &ImportHandle) -> PyResult<Py<TensorRead>> {
    let value = import_view(py, imported);
    Py::new(
        py,
        TensorRead {
            tensor: value.views.tensor.clone_ref(py),
            region: None,
            metadata: value
                .views
                .metadata
                .as_ref()
                .map(|metadata| metadata.clone_ref(py)),
            inner: NativeRead {
                buffer: value.buffer.clone_ref(py),
                consumer: None,
                imported: Some(imported.clone_ref(py)),
                complete: false,
            },
        },
    )
}
