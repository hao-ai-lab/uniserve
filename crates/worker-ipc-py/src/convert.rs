//! Typed conversions for steady-state worker IPC frames.
//!
//! The serve loop crosses the FFI boundary once per direction per batch. The
//! typed converters materialize the canonical `submit` request and `result`
//! response shapes directly: every dict key and enum string is interned
//! ([`pyo3::intern!`]), lists are preallocated at their known lengths, and byte
//! payloads stay on the `bytes` path.
//!
//! [`execute_request_to_py`] produces the mapping consumed by
//! `Run.from_mapping`. [`try_completion_response_from_py`] accepts the exact
//! completion-report mapping emitted by the Python worker. Administrative frame
//! kinds are handled by the schema-derived converter in the caller.

use std::collections::{BTreeMap, HashMap};

use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyList, PyString};
use uniserve_core::{ImageParams, SamplingParams};
#[cfg(test)]
use uniserve_worker_ipc::MediaGeometry;
use uniserve_worker_ipc::{
    ArRequestParams, ArtifactHandle, BatchCommand, BlockTable, BufferId, BufferPlacement,
    CachePageAllocation, Checkpoint, CheckpointPoint, CloseReason, DType, DecodePlacement,
    DiffusionRequestParams, DiffusionResult, DimBound, Disposition, DrawLayout, ErrorCode,
    ErrorOperationIdentity, FinishFlags, InlineValue, LatentPlacement, LogicalLengths, MediaOutput,
    ModelOutput, NewRequest, OpId, OpPayload, OpStatus, Operation, PointRange, ProductKind,
    ProductPayload, ProductRef, RegistrationAck, RequestKey, RequestKind, ResultData,
    ResultPayload, RowGeometry, Run, RunKind, RunResult, ShapeBound, StorageClass, TimingCounters,
    TokenSpan, TransferHandle, TransferLocator, TransferTransport, UmmRequestParams,
    WorkerForwardStats, WorkerRequest, WorkerResponse, WorkerResponseError,
};

#[cfg(test)]
use uniserve_worker_ipc::Bounds;

// ---------------------------------------------------------------------------
// Request -> Python (recv hot path)
// ---------------------------------------------------------------------------

/// Convert a `submit` [`WorkerRequest`] into the canonical Python IPC mapping.
pub(crate) fn execute_request_to_py<'py>(
    py: Python<'py>,
    request: &WorkerRequest,
) -> PyResult<Bound<'py, PyDict>> {
    let WorkerRequest::Submit { call_id, run } = request else {
        return Err(PyValueError::new_err(
            "native submit conversion requires a submit request",
        ));
    };
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "kind"), request_kind_py(py, request.kind()))?;
    dict.set_item(intern!(py, "call_id"), call_id)?;
    dict.set_item(intern!(py, "run"), run_to_py(py, run)?)?;
    Ok(dict)
}

// ---------------------------------------------------------------------------
// Native typed construction (recv hot path)
// ---------------------------------------------------------------------------

/// Cached handles to the worker's operation types and enum members.
///
/// The serve loop constructs one `Run` object per submission; every hot
/// record (operations, KV placements, commit/close/release controls) is built
/// by calling the operation dataclass constructors positionally, so the worker
/// never re-decodes those records from IPC maps. Rare members (admissions,
/// input products and placements) still cross as IPC maps and are decoded by
/// `native_run` on the Python side.
struct NativeRequestTypes {
    operation: Py<PyAny>,
    ar_payload: Py<PyAny>,
    encoder_payload: Py<PyAny>,
    diffusion_payload: Py<PyAny>,
    transfer_payload: Py<PyAny>,
    request_key: Py<PyAny>,
    version_ref: Py<PyAny>,
    fixed_point: Py<PyAny>,
    device_point: Py<PyAny>,
    product_ref: Py<PyAny>,
    buffer_id: Py<PyAny>,
    shape_bound: Py<PyAny>,
    static_dim: Py<PyAny>,
    device_dim: Py<PyAny>,
    point_range: Py<PyAny>,
    bounds: Py<PyAny>,
    rng: Py<PyAny>,
    block_table: Py<PyAny>,
    cache_page_allocation: Py<PyAny>,
    row_geometry: Py<PyAny>,
    start: Py<PyAny>,
    commit: Py<PyAny>,
    finish: Py<PyAny>,
    free: Py<PyAny>,
    native_run: Py<PyAny>,
    product_kinds: [Py<PyAny>; 10],
    storage_classes: [Py<PyAny>; 6],
    dtypes: [Py<PyAny>; 8],
    dispositions: [Py<PyAny>; 3],
    close_reasons: [Py<PyAny>; 4],
    draw_layouts: [Py<PyAny>; 3],
    works: [Py<PyAny>; 12],
}

static NATIVE_REQUEST_TYPES: std::sync::OnceLock<NativeRequestTypes> = std::sync::OnceLock::new();

fn enum_members<const N: usize>(
    module: &Bound<'_, PyModule>,
    name: &str,
    values: [&str; N],
) -> PyResult<[Py<PyAny>; N]> {
    let class = module.getattr(name)?;
    let mut members: Vec<Py<PyAny>> = Vec::with_capacity(N);
    for value in values {
        members.push(class.call1((value,))?.unbind());
    }
    Ok(members
        .try_into()
        .unwrap_or_else(|_| unreachable!("member count matches the declared array")))
}

impl NativeRequestTypes {
    fn build(py: Python<'_>) -> PyResult<Self> {
        let module = py.import("uniserve_worker.execution.batch")?;
        let class = |name: &str| -> PyResult<Py<PyAny>> { Ok(module.getattr(name)?.unbind()) };
        Ok(Self {
            operation: class("Operation")?,
            ar_payload: class("ArOpPayload")?,
            encoder_payload: class("EncoderOpPayload")?,
            diffusion_payload: class("DiffusionOpPayload")?,
            transfer_payload: class("TransferOpPayload")?,
            request_key: class("RequestKey")?,
            version_ref: class("Checkpoint")?,
            fixed_point: class("FixedCheckpoint")?,
            device_point: class("DeviceSelected")?,
            product_ref: class("ProductRef")?,
            buffer_id: class("BufferId")?,
            shape_bound: class("ShapeBound")?,
            static_dim: class("StaticDim")?,
            device_dim: class("DeviceDim")?,
            point_range: class("PointRange")?,
            bounds: class("Bounds")?,
            rng: class("Rng")?,
            block_table: class("BlockTable")?,
            cache_page_allocation: class("CachePageAllocation")?,
            row_geometry: class("RowGeometry")?,
            start: class("Start")?,
            commit: class("Commit")?,
            finish: class("Finish")?,
            free: class("Free")?,
            native_run: class("native_run")?,
            product_kinds: enum_members(
                &module,
                "ProductKind",
                [
                    "token",
                    "logprob",
                    "vision_feature",
                    "latent_feature",
                    "kv",
                    "latent",
                    "artifact",
                    "completion",
                    "sampling_state",
                    "selected_point",
                ],
            )?,
            storage_classes: enum_members(
                &module,
                "StorageClass",
                [
                    "device_tensor",
                    "request_relay",
                    "paged_kv",
                    "latent_arena",
                    "host_staging",
                    "pinned_output",
                ],
            )?,
            dtypes: enum_members(
                &module,
                "DType",
                ["u8", "u16", "u32", "i32", "i64", "f16", "bf16", "f32"],
            )?,
            dispositions: enum_members(&module, "Disposition", ["publish", "retain", "discard"])?,
            close_reasons: enum_members(
                &module,
                "CloseReason",
                ["completed", "cancelled", "error", "preempted"],
            )?,
            draw_layouts: enum_members(
                &module,
                "DrawLayout",
                ["target_sampling", "speculative_proposal", "flow_noise"],
            )?,
            works: enum_members(
                &module,
                "RunKind",
                [
                    "ar_extend",
                    "ar_decode",
                    "ar_verify",
                    "encoder_vision",
                    "encoder_latent",
                    "transfer_product",
                    "transfer_kv_publish",
                    "transfer_kv_install",
                    "diffusion_prepare",
                    "diffusion_step",
                    "diffusion_finalize",
                    "diffusion_decode",
                ],
            )?,
        })
    }

    fn get(py: Python<'_>) -> PyResult<&'static Self> {
        if let Some(types) = NATIVE_REQUEST_TYPES.get() {
            return Ok(types);
        }
        let built = Self::build(py)?;
        Ok(NATIVE_REQUEST_TYPES.get_or_init(|| built))
    }

    fn kind<'py>(&self, py: Python<'py>, kind: RunKind) -> Bound<'py, PyAny> {
        let index = kind as usize;
        self.works[index].bind(py).clone()
    }

    fn product_kind<'py>(&self, py: Python<'py>, kind: ProductKind) -> Bound<'py, PyAny> {
        let index = match kind {
            ProductKind::Token => 0,
            ProductKind::Logprob => 1,
            ProductKind::VisionFeature => 2,
            ProductKind::LatentFeature => 3,
            ProductKind::Kv => 4,
            ProductKind::Latent => 5,
            ProductKind::Artifact => 6,
            ProductKind::Completion => 7,
            ProductKind::SamplingState => 8,
            ProductKind::SelectedPoint => 9,
        };
        self.product_kinds[index].bind(py).clone()
    }

    fn storage_class<'py>(&self, py: Python<'py>, class: StorageClass) -> Bound<'py, PyAny> {
        let index = match class {
            StorageClass::DeviceTensor => 0,
            StorageClass::RequestRelay => 1,
            StorageClass::PagedKv => 2,
            StorageClass::LatentArena => 3,
            StorageClass::HostStaging => 4,
            StorageClass::PinnedOutput => 5,
        };
        self.storage_classes[index].bind(py).clone()
    }

    fn dtype<'py>(&self, py: Python<'py>, dtype: DType) -> Bound<'py, PyAny> {
        let index = match dtype {
            DType::U8 => 0,
            DType::U16 => 1,
            DType::U32 => 2,
            DType::I32 => 3,
            DType::I64 => 4,
            DType::F16 => 5,
            DType::BF16 => 6,
            DType::F32 => 7,
        };
        self.dtypes[index].bind(py).clone()
    }
}

/// Per-batch construction context: typed leaves shared across the batch's
/// records are built once and reused by identity.
struct NativeRequestConversion<'py> {
    py: Python<'py>,
    types: &'static NativeRequestTypes,
    request_keys: HashMap<RequestKey, Py<PyAny>>,
    shape_bounds: HashMap<ShapeBound, Py<PyAny>>,
    point_ranges: HashMap<PointRange, Py<PyAny>>,
}

impl<'py> NativeRequestConversion<'py> {
    fn new(py: Python<'py>) -> PyResult<Self> {
        Ok(Self {
            py,
            types: NativeRequestTypes::get(py)?,
            request_keys: HashMap::new(),
            shape_bounds: HashMap::new(),
            point_ranges: HashMap::new(),
        })
    }

    fn request_key(&mut self, key: RequestKey) -> PyResult<Bound<'py, PyAny>> {
        if let Some(value) = self.request_keys.get(&key) {
            return Ok(value.bind(self.py).clone());
        }
        let value = self.types.request_key.bind(self.py).call1((
            key.authority_id,
            key.request_id.0,
            key.epoch,
        ))?;
        self.request_keys.insert(key, value.clone().unbind());
        Ok(value)
    }

    fn shape_bound(&mut self, shape: &ShapeBound) -> PyResult<Bound<'py, PyAny>> {
        if let Some(value) = self.shape_bounds.get(shape) {
            return Ok(value.bind(self.py).clone());
        }
        let dims = shape
            .dims
            .iter()
            .map(|dim| match dim {
                DimBound::Static(extent) => self.types.static_dim.bind(self.py).call1((*extent,)),
                DimBound::Device { max } => self.types.device_dim.bind(self.py).call1((*max,)),
            })
            .collect::<PyResult<Vec<_>>>()?;
        let value = self
            .types
            .shape_bound
            .bind(self.py)
            .call1((pyo3::types::PyTuple::new(self.py, dims)?,))?;
        self.shape_bounds
            .insert(shape.clone(), value.clone().unbind());
        Ok(value)
    }

    fn point_range(&mut self, range: PointRange) -> PyResult<Bound<'py, PyAny>> {
        if let Some(value) = self.point_ranges.get(&range) {
            return Ok(value.bind(self.py).clone());
        }
        let value = self
            .types
            .point_range
            .bind(self.py)
            .call1((range.base_point, range.max_points))?;
        self.point_ranges.insert(range, value.clone().unbind());
        Ok(value)
    }

    fn product_ref(&mut self, product: &ProductRef) -> PyResult<Bound<'py, PyAny>> {
        let request_key = self.request_key(product.request_key)?;
        let shape_bound = self.shape_bound(&product.shape_bound)?;
        let point_range = self.point_range(product.point_range)?;
        self.types.product_ref.bind(self.py).call1((
            request_key,
            product.producer_op_id.0,
            product.output_index,
            product.generation,
            self.types.product_kind(self.py, product.kind),
            self.types.storage_class(self.py, product.storage_class),
            self.types.dtype(self.py, product.dtype),
            shape_bound,
            point_range,
        ))
    }

    fn buffer_id(&mut self, buffer: BufferId) -> PyResult<Bound<'py, PyAny>> {
        let owner = self.request_key(buffer.owner)?;
        self.types.buffer_id.bind(self.py).call1((
            owner,
            buffer.producer_op_id.0,
            buffer.output_index,
            buffer.generation,
        ))
    }

    fn checkpoint(&mut self, checkpoint: &Checkpoint) -> PyResult<Bound<'py, PyAny>> {
        let point = match checkpoint.point {
            CheckpointPoint::Fixed(point) => {
                self.types.fixed_point.bind(self.py).call1((point,))?
            }
            CheckpointPoint::DeviceSelected => self.types.device_point.bind(self.py).call0()?,
        };
        self.types
            .version_ref
            .bind(self.py)
            .call1((checkpoint.op_id.0, point))
    }

    fn operation(&mut self, operation: &Operation) -> PyResult<Bound<'py, PyAny>> {
        let request_key = self.request_key(operation.request_key)?;
        let parent = self.checkpoint(&operation.parent)?;
        let bounds = self.types.bounds.bind(self.py).call1((
            operation.bounds().max_points,
            operation.bounds().max_tokens,
            operation.bounds().max_kv_pages,
            operation.bounds().max_latent_bytes,
            operation.bounds().max_completion_bytes,
            operation.bounds().max_transfer_bytes,
        ))?;
        let inputs = operation
            .inputs()
            .iter()
            .map(|product| self.product_ref(product))
            .collect::<PyResult<Vec<_>>>()?;
        let outputs = operation
            .outputs()
            .iter()
            .map(|product| self.product_ref(product))
            .collect::<PyResult<Vec<_>>>()?;
        let predicate = operation
            .predicate()
            .as_ref()
            .map(|predicate| self.product_ref(predicate))
            .transpose()?;
        let rng = operation
            .rng()
            .as_ref()
            .map(|rng| {
                let layout = match rng.draw_layout {
                    DrawLayout::TargetSampling => 0,
                    DrawLayout::SpeculativeProposal => 1,
                    DrawLayout::FlowNoise => 2,
                };
                self.types.rng.bind(self.py).call1((
                    rng.seed,
                    rng.semantic_index_base,
                    self.types.draw_layouts[layout].bind(self.py).clone(),
                ))
            })
            .transpose()?;
        let py = self.py;
        let payload_type = match operation.payload {
            OpPayload::Ar { .. } => &self.types.ar_payload,
            OpPayload::Encoder { .. } => &self.types.encoder_payload,
            OpPayload::Diffusion { .. } => &self.types.diffusion_payload,
            OpPayload::Transfer { .. } => &self.types.transfer_payload,
        };
        let payload = payload_type.bind(py).call1((
            bounds,
            pyo3::types::PyTuple::new(py, inputs)?,
            pyo3::types::PyTuple::new(py, outputs)?,
            predicate
                .map(Bound::into_any)
                .unwrap_or_else(|| py.None().into_bound(py)),
            rng.map(Bound::into_any)
                .unwrap_or_else(|| py.None().into_bound(py)),
            operation.control_seq(),
        ))?;
        let arguments = pyo3::types::PyTuple::new(
            py,
            [
                request_key.into_any(),
                operation.op_id.0.into_pyobject(py)?.into_any(),
                parent.into_any(),
                self.types.kind(py, operation.kind),
                payload.into_any(),
            ],
        )?;
        self.types.operation.bind(py).call1(arguments)
    }

    fn block_table(&self, table: &BlockTable) -> PyResult<Bound<'py, PyAny>> {
        self.types.block_table.bind(self.py).call1((
            table.request_pool_idx,
            table.group_id,
            pyo3::types::PyTuple::new(self.py, table.page_ids.iter().map(|page| page.0))?,
            table.allocated_tokens,
        ))
    }

    fn cache_page_allocation(
        &self,
        allocation: &CachePageAllocation,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.types.cache_page_allocation.bind(self.py).call1((
            allocation.request_pool_idx,
            allocation.group_id,
            pyo3::types::PyTuple::new(self.py, allocation.page_ids.iter().map(|page| page.0))?,
        ))
    }

    fn row_geometry(&self, row: &RowGeometry) -> PyResult<Bound<'py, PyAny>> {
        self.types.row_geometry.bind(self.py).call1((
            row.operation_index,
            row.request_pool_index,
            row.seq_len,
            row.query_len,
        ))
    }

    fn command(&mut self, command: &BatchCommand) -> PyResult<Bound<'py, PyAny>> {
        match command {
            BatchCommand::Start { request } => {
                let request =
                    admission_to_py(self.py, request, &mut RequestConversion::new(self.py))?;
                let value = PyDict::new(self.py);
                value.set_item(intern!(self.py, "request"), request)?;
                self.types
                    .start
                    .bind(self.py)
                    .call_method1("from_mapping", (value,))
            }
            BatchCommand::Commit {
                request_key,
                control_seq,
                expected_parent,
                selected,
                public_event_limit,
                disposition,
            } => {
                let request = *request_key;
                let expected_parent = self.checkpoint(expected_parent)?;
                let selected = self.checkpoint(selected)?;
                let request_key = self.request_key(request)?;
                let disposition = match disposition {
                    Disposition::Publish => 0,
                    Disposition::Retain => 1,
                    Disposition::Discard => 2,
                };
                self.types.commit.bind(self.py).call1((
                    request_key,
                    *control_seq,
                    expected_parent,
                    selected,
                    *public_event_limit,
                    self.types.dispositions[disposition].bind(self.py).clone(),
                ))
            }
            BatchCommand::Finish {
                request_key,
                control_seq,
                cutoff,
                reason,
            } => {
                let request = *request_key;
                let cutoff = self.checkpoint(cutoff)?;
                let request_key = self.request_key(request)?;
                let reason = match reason {
                    CloseReason::Completed => 0,
                    CloseReason::Cancelled => 1,
                    CloseReason::Error => 2,
                    CloseReason::Preempted => 3,
                };
                self.types.finish.bind(self.py).call1((
                    request_key,
                    *control_seq,
                    cutoff,
                    self.types.close_reasons[reason].bind(self.py).clone(),
                ))
            }
            BatchCommand::Free { buffer } => {
                let buffer = self.buffer_id(*buffer)?;
                self.types.free.bind(self.py).call1((buffer,))
            }
        }
    }
}

fn run_to_py<'py>(py: Python<'py>, run: &Run) -> PyResult<Bound<'py, PyAny>> {
    let mut context = RequestConversion::new(py);
    let mut native = NativeRequestConversion::new(py)?;
    let operations = run
        .operations
        .iter()
        .map(|operation| native.operation(operation))
        .collect::<PyResult<Vec<_>>>()?;
    let block_tables = run
        .block_tables
        .iter()
        .map(|table| native.block_table(table))
        .collect::<PyResult<Vec<_>>>()?;
    let new_cache_pages = run
        .new_cache_pages
        .iter()
        .map(|allocation| native.cache_page_allocation(allocation))
        .collect::<PyResult<Vec<_>>>()?;
    let forward_rows = run
        .forward_rows
        .iter()
        .map(|row| native.row_geometry(row))
        .collect::<PyResult<Vec<_>>>()?;
    let commands = run
        .commands
        .iter()
        .map(|command| native.command(command))
        .collect::<PyResult<Vec<_>>>()?;
    let input_products = dict_list(py, &run.input_products, |payload| {
        product_payload_to_py(py, payload, &mut context)
    })?;
    native.types.native_run.bind(py).call1((
        run.batch_id,
        run.run_id,
        run.collective_seq,
        pyo3::types::PyTuple::new(py, operations)?,
        pyo3::types::PyTuple::new(py, block_tables)?,
        pyo3::types::PyTuple::new(py, new_cache_pages)?,
        pyo3::types::PyTuple::new(py, forward_rows)?,
        dict_list(py, &run.latent_placements, |placement| {
            latent_placement_to_py(py, placement, &mut context)
        })?,
        dict_list(py, &run.decode_placements, |placement| {
            decode_placement_to_py(py, placement, &mut context)
        })?,
        dict_list(py, &run.buffer_placements, |placement| {
            buffer_placement_to_py(py, placement, &mut context)
        })?,
        pyo3::types::PyTuple::new(py, commands)?,
        input_products,
    ))
}

fn latent_placement_to_py<'py>(
    py: Python<'py>,
    placement: &LatentPlacement,
    context: &mut RequestConversion<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(
        intern!(py, "request_key"),
        context.request_key(placement.request_key)?,
    )?;
    dict.set_item(intern!(py, "op_id"), placement.op_id.0)?;
    dict.set_item(
        intern!(py, "page_table"),
        u32_list(py, &placement.page_table)?,
    )?;
    dict.set_item(intern!(py, "latent_units"), placement.latent_units)?;
    dict.set_item(intern!(py, "height"), placement.height)?;
    dict.set_item(intern!(py, "width"), placement.width)?;
    dict.set_item(intern!(py, "start_step"), placement.start_step)?;
    dict.set_item(intern!(py, "step_count"), placement.step_count)?;
    Ok(dict)
}

fn decode_placement_to_py<'py>(
    py: Python<'py>,
    placement: &DecodePlacement,
    context: &mut RequestConversion<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(
        intern!(py, "request_key"),
        context.request_key(placement.request_key)?,
    )?;
    dict.set_item(intern!(py, "op_id"), placement.op_id.0)?;
    dict.set_item(intern!(py, "cursor"), placement.cursor)?;
    dict.set_item(intern!(py, "max_units"), placement.max_units)?;
    Ok(dict)
}

fn buffer_placement_to_py<'py>(
    py: Python<'py>,
    placement: &BufferPlacement,
    context: &mut RequestConversion<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    let id = placement.buffer;
    let buffer = PyDict::new(py);
    buffer.set_item(intern!(py, "owner"), context.request_key(id.owner)?)?;
    buffer.set_item(intern!(py, "producer_op_id"), id.producer_op_id.0)?;
    buffer.set_item(intern!(py, "output_index"), id.output_index)?;
    buffer.set_item(intern!(py, "generation"), id.generation)?;
    dict.set_item(intern!(py, "buffer"), buffer)?;
    dict.set_item(intern!(py, "offset"), placement.offset)?;
    dict.set_item(intern!(py, "bytes"), placement.bytes)?;
    Ok(dict)
}

fn dict_list<'py, T, F>(
    py: Python<'py>,
    items: &[T],
    mut convert: F,
) -> PyResult<Bound<'py, PyList>>
where
    F: FnMut(&T) -> PyResult<Bound<'py, PyDict>>,
{
    let converted = items
        .iter()
        .map(&mut convert)
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, converted)
}

struct RequestConversion<'py> {
    py: Python<'py>,
    request_keys: HashMap<RequestKey, Bound<'py, PyDict>>,
    shape_bounds: HashMap<ShapeBound, Bound<'py, PyDict>>,
    point_ranges: HashMap<PointRange, Bound<'py, PyDict>>,
}

impl<'py> RequestConversion<'py> {
    fn new(py: Python<'py>) -> Self {
        Self {
            py,
            request_keys: HashMap::new(),
            shape_bounds: HashMap::new(),
            point_ranges: HashMap::new(),
        }
    }

    fn request_key(&mut self, key: RequestKey) -> PyResult<Bound<'py, PyDict>> {
        if let Some(value) = self.request_keys.get(&key) {
            return Ok(value.clone());
        }
        let dict = PyDict::new(self.py);
        dict.set_item(intern!(self.py, "authority_id"), key.authority_id)?;
        dict.set_item(intern!(self.py, "request_id"), key.request_id.0)?;
        dict.set_item(intern!(self.py, "epoch"), key.epoch)?;
        self.request_keys.insert(key, dict.clone());
        Ok(dict)
    }

    fn shape_bound(&mut self, shape: &ShapeBound) -> PyResult<Bound<'py, PyDict>> {
        if let Some(value) = self.shape_bounds.get(shape) {
            return Ok(value.clone());
        }
        let dict = shape_bound_to_py(self.py, shape)?;
        self.shape_bounds.insert(shape.clone(), dict.clone());
        Ok(dict)
    }

    fn point_range(&mut self, range: PointRange) -> PyResult<Bound<'py, PyDict>> {
        if let Some(value) = self.point_ranges.get(&range) {
            return Ok(value.clone());
        }
        let dict = point_range_to_py(self.py, range)?;
        self.point_ranges.insert(range, dict.clone());
        Ok(dict)
    }
}

fn u32_list<'py>(py: Python<'py>, values: &[u32]) -> PyResult<Bound<'py, PyList>> {
    PyList::new(py, values.iter().copied())
}

fn admission_to_py<'py>(
    py: Python<'py>,
    admission: &NewRequest,
    context: &mut RequestConversion<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(
        intern!(py, "request_key"),
        context.request_key(admission.request_key)?,
    )?;
    dict.set_item(intern!(py, "request_pool_idx"), admission.request_pool_idx)?;
    dict.set_item(
        intern!(py, "ar"),
        admission
            .ar
            .as_ref()
            .map(|ar| ar_params_to_py(py, ar))
            .transpose()?,
    )?;
    dict.set_item(
        intern!(py, "umm"),
        admission
            .umm
            .as_ref()
            .map(|branch| umm_params_to_py(py, branch))
            .transpose()?,
    )?;
    dict.set_item(
        intern!(py, "diffusion"),
        admission
            .diffusion
            .as_ref()
            .map(|diffusion| diffusion_params_to_py(py, diffusion))
            .transpose()?,
    )?;
    Ok(dict)
}

fn ar_params_to_py<'py>(py: Python<'py>, ar: &ArRequestParams) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "sampling"), sampling_to_py(py, &ar.sampling)?)?;
    dict.set_item(
        intern!(py, "negative_token_ids"),
        u32_list(py, &ar.negative_token_ids)?,
    )?;
    dict.set_item(
        intern!(py, "finish_token_ids"),
        u32_list(py, &ar.finish_token_ids)?,
    )?;
    dict.set_item(intern!(py, "initial_position"), ar.initial_position)?;
    Ok(dict)
}

fn umm_params_to_py<'py>(
    py: Python<'py>,
    branch: &UmmRequestParams,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "image"), image_to_py(py, &branch.image)?)?;
    Ok(dict)
}

fn diffusion_params_to_py<'py>(
    py: Python<'py>,
    diffusion: &DiffusionRequestParams,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(
        intern!(py, "prompt_token_ids"),
        u32_list(py, &diffusion.prompt_token_ids)?,
    )?;
    dict.set_item(intern!(py, "seed"), diffusion.seed)?;
    let geometry = PyDict::new(py);
    geometry.set_item(intern!(py, "frame_count"), diffusion.geometry.frame_count)?;
    geometry.set_item(intern!(py, "decode_units"), diffusion.geometry.decode_units)?;
    geometry.set_item(
        intern!(py, "prompt_tokens"),
        diffusion.geometry.prompt_tokens,
    )?;
    geometry.set_item(
        intern!(py, "denoise_steps"),
        diffusion.geometry.denoise_steps,
    )?;
    dict.set_item(intern!(py, "geometry"), geometry)?;
    Ok(dict)
}

fn sampling_to_py<'py>(py: Python<'py>, sampling: &SamplingParams) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "temperature"), sampling.temperature)?;
    dict.set_item(intern!(py, "top_k"), sampling.top_k)?;
    dict.set_item(intern!(py, "top_p"), sampling.top_p)?;
    dict.set_item(intern!(py, "ignore_eos"), sampling.ignore_eos)?;
    dict.set_item(intern!(py, "seed"), sampling.seed)?;
    dict.set_item(intern!(py, "min_p"), sampling.min_p)?;
    dict.set_item(
        intern!(py, "repetition_penalty"),
        sampling.repetition_penalty,
    )?;
    dict.set_item(intern!(py, "frequency_penalty"), sampling.frequency_penalty)?;
    dict.set_item(intern!(py, "presence_penalty"), sampling.presence_penalty)?;
    // `(u32, f32)` pairs pythonize as Python tuples, not lists.
    dict.set_item(
        intern!(py, "logit_bias"),
        PyList::new(
            py,
            sampling
                .logit_bias
                .iter()
                .map(|(token, bias)| (*token, *bias)),
        )?,
    )?;
    dict.set_item(intern!(py, "min_tokens"), sampling.min_tokens)?;
    dict.set_item(intern!(py, "return_logprobs"), sampling.return_logprobs)?;
    dict.set_item(intern!(py, "n_logprobs"), sampling.n_logprobs)?;
    dict.set_item(
        intern!(py, "return_prompt_logprobs"),
        sampling.return_prompt_logprobs,
    )?;
    dict.set_item(intern!(py, "n_prompt_logprobs"), sampling.n_prompt_logprobs)?;
    dict.set_item(
        intern!(py, "logprob_token_ids"),
        u32_list(py, &sampling.logprob_token_ids)?,
    )?;
    let bad_words = sampling
        .bad_words_ids
        .iter()
        .map(|tokens| u32_list(py, tokens))
        .collect::<PyResult<Vec<_>>>()?;
    dict.set_item(intern!(py, "bad_words_ids"), PyList::new(py, bad_words)?)?;
    dict.set_item(
        intern!(py, "allowed_token_ids"),
        sampling
            .allowed_token_ids
            .as_deref()
            .map(|tokens| u32_list(py, tokens))
            .transpose()?,
    )?;
    dict.set_item(intern!(py, "typical_p"), sampling.typical_p)?;
    dict.set_item(
        intern!(py, "forced_token_ids"),
        u32_list(py, &sampling.forced_token_ids)?,
    )?;
    Ok(dict)
}

fn image_to_py<'py>(py: Python<'py>, image: &ImageParams) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "steps"), image.steps)?;
    dict.set_item(intern!(py, "cfg_text_scale"), image.cfg_text_scale)?;
    dict.set_item(intern!(py, "cfg_img_scale"), image.cfg_img_scale)?;
    dict.set_item(
        intern!(py, "cfg_renorm_type"),
        image.cfg_renorm_type.as_str(),
    )?;
    dict.set_item(intern!(py, "cfg_renorm_min"), image.cfg_renorm_min)?;
    // `(f32, f32)` pythonizes as a Python tuple.
    dict.set_item(intern!(py, "cfg_interval"), image.cfg_interval)?;
    dict.set_item(intern!(py, "timestep_shift"), image.timestep_shift)?;
    dict.set_item(intern!(py, "height"), image.height)?;
    dict.set_item(intern!(py, "width"), image.width)?;
    dict.set_item(intern!(py, "seed"), image.seed)?;
    dict.set_item(
        intern!(py, "negative_prompt"),
        image.negative_prompt.as_str(),
    )?;
    dict.set_item(intern!(py, "max_images"), image.max_images)?;
    dict.set_item(
        intern!(py, "image_prompts"),
        PyList::new(py, image.image_prompts.iter().map(|prompt| prompt.as_str()))?,
    )?;
    dict.set_item(intern!(py, "retain_images"), image.retain_images)?;
    Ok(dict)
}

fn product_ref_to_py<'py>(
    py: Python<'py>,
    product: &ProductRef,
    context: &mut RequestConversion<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(
        intern!(py, "request_key"),
        context.request_key(product.request_key)?,
    )?;
    dict.set_item(intern!(py, "producer_op_id"), product.producer_op_id.0)?;
    dict.set_item(intern!(py, "output_index"), product.output_index)?;
    dict.set_item(intern!(py, "generation"), product.generation)?;
    dict.set_item(intern!(py, "kind"), product_kind_py(py, product.kind))?;
    dict.set_item(
        intern!(py, "storage_class"),
        storage_class_py(py, product.storage_class),
    )?;
    dict.set_item(intern!(py, "dtype"), dtype_py(py, product.dtype))?;
    dict.set_item(
        intern!(py, "shape_bound"),
        context.shape_bound(&product.shape_bound)?,
    )?;
    dict.set_item(
        intern!(py, "point_range"),
        context.point_range(product.point_range)?,
    )?;
    Ok(dict)
}

fn shape_bound_to_py<'py>(py: Python<'py>, shape: &ShapeBound) -> PyResult<Bound<'py, PyDict>> {
    let dims = shape
        .dims
        .iter()
        .map(|dim| {
            let entry = PyDict::new(py);
            match dim {
                DimBound::Static(extent) => {
                    entry.set_item(intern!(py, "kind"), intern!(py, "static"))?;
                    entry.set_item(intern!(py, "value"), *extent)?;
                }
                DimBound::Device { max } => {
                    entry.set_item(intern!(py, "kind"), intern!(py, "device"))?;
                    let value = PyDict::new(py);
                    value.set_item(intern!(py, "max"), *max)?;
                    entry.set_item(intern!(py, "value"), value)?;
                }
            }
            Ok(entry)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "dims"), PyList::new(py, dims)?)?;
    Ok(dict)
}

fn point_range_to_py<'py>(py: Python<'py>, range: PointRange) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "base_point"), range.base_point)?;
    dict.set_item(intern!(py, "max_points"), range.max_points)?;
    Ok(dict)
}

fn product_payload_to_py<'py>(
    py: Python<'py>,
    payload: &ProductPayload,
    context: &mut RequestConversion<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(
        intern!(py, "product"),
        product_ref_to_py(py, &payload.product, context)?,
    )?;
    let value = PyDict::new(py);
    match &payload.value {
        InlineValue::Bytes(bytes) => {
            value.set_item(intern!(py, "kind"), "bytes")?;
            value.set_item(intern!(py, "value"), PyBytes::new(py, bytes))?;
        }
        InlineValue::Transfer(handle) => {
            value.set_item(intern!(py, "kind"), "transfer")?;
            value.set_item(intern!(py, "value"), transfer_handle_to_py(py, handle)?)?;
        }
    }
    dict.set_item(intern!(py, "value"), value)?;
    Ok(dict)
}

fn transfer_locator_to_py<'py>(
    py: Python<'py>,
    locator: &TransferLocator,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "nbytes"), locator.nbytes)?;
    dict.set_item(intern!(py, "dtype"), locator.dtype.as_str())?;
    dict.set_item(intern!(py, "shape"), PyList::new(py, &locator.shape)?)?;
    dict.set_item(intern!(py, "device"), locator.device.as_str())?;
    match &locator.transport {
        TransferTransport::Local { endpoint, key } => {
            dict.set_item(intern!(py, "transport"), "local")?;
            dict.set_item(intern!(py, "endpoint"), endpoint.as_str())?;
            dict.set_item(intern!(py, "key"), key)?;
        }
        TransferTransport::PosixShm {
            name,
            ready_header_bytes,
            ready_semaphore,
        } => {
            dict.set_item(intern!(py, "transport"), "posix_shm")?;
            dict.set_item(intern!(py, "name"), name.as_str())?;
            dict.set_item(intern!(py, "ready_header_bytes"), ready_header_bytes)?;
            dict.set_item(intern!(py, "ready_semaphore"), ready_semaphore.as_deref())?;
        }
        TransferTransport::CudaIpc {
            endpoint,
            publication_id,
            storage_handle,
            storage_size_bytes,
            storage_offset_bytes,
            tensor_offset,
            tensor_stride,
            ref_counter_handle,
            ref_counter_offset,
            event_handle,
            event_sync_required,
            ready_event_handle,
        } => {
            dict.set_item(intern!(py, "transport"), "cuda_ipc")?;
            dict.set_item(intern!(py, "endpoint"), endpoint.as_str())?;
            dict.set_item(intern!(py, "publication_id"), publication_id.as_str())?;
            dict.set_item(
                intern!(py, "storage_handle"),
                PyBytes::new(py, storage_handle),
            )?;
            dict.set_item(intern!(py, "storage_size_bytes"), storage_size_bytes)?;
            dict.set_item(intern!(py, "storage_offset_bytes"), storage_offset_bytes)?;
            dict.set_item(intern!(py, "tensor_offset"), tensor_offset)?;
            dict.set_item(
                intern!(py, "tensor_stride"),
                PyList::new(py, tensor_stride)?,
            )?;
            dict.set_item(
                intern!(py, "ref_counter_handle"),
                PyBytes::new(py, ref_counter_handle),
            )?;
            dict.set_item(intern!(py, "ref_counter_offset"), ref_counter_offset)?;
            dict.set_item(intern!(py, "event_handle"), PyBytes::new(py, event_handle))?;
            dict.set_item(intern!(py, "event_sync_required"), event_sync_required)?;
            dict.set_item(
                intern!(py, "ready_event_handle"),
                PyBytes::new(py, ready_event_handle),
            )?;
        }
    }
    Ok(dict)
}

fn checkpoint_mapping_to_py<'py>(
    py: Python<'py>,
    checkpoint: &Checkpoint,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "op_id"), checkpoint.op_id.0)?;
    let point = PyDict::new(py);
    match checkpoint.point {
        CheckpointPoint::Fixed(value) => {
            point.set_item(intern!(py, "kind"), "fixed")?;
            point.set_item(intern!(py, "value"), value)?;
        }
        CheckpointPoint::DeviceSelected => {
            point.set_item(intern!(py, "kind"), "device_selected")?;
            point.set_item(intern!(py, "value"), py.None())?;
        }
    }
    dict.set_item(intern!(py, "point"), point)?;
    Ok(dict)
}

fn transfer_handle_to_py<'py>(
    py: Python<'py>,
    handle: &TransferHandle,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    let value = PyDict::new(py);
    match handle {
        TransferHandle::Encoder {
            generation,
            height,
            width,
            payload_kind,
            locator,
        } => {
            dict.set_item(intern!(py, "kind"), "encoder")?;
            value.set_item(intern!(py, "generation"), generation)?;
            value.set_item(intern!(py, "height"), height)?;
            value.set_item(intern!(py, "width"), width)?;
            value.set_item(
                intern!(py, "payload_kind"),
                product_kind_py(py, *payload_kind),
            )?;
            value.set_item(intern!(py, "locator"), transfer_locator_to_py(py, locator)?)?;
        }
        TransferHandle::DeviceProduct {
            generation,
            height,
            width,
            value_range,
            locator,
        } => {
            dict.set_item(intern!(py, "kind"), "device_product")?;
            value.set_item(intern!(py, "generation"), generation)?;
            value.set_item(intern!(py, "height"), height)?;
            value.set_item(intern!(py, "width"), width)?;
            value.set_item(intern!(py, "value_range"), value_range.as_str())?;
            value.set_item(intern!(py, "locator"), transfer_locator_to_py(py, locator)?)?;
        }
        TransferHandle::Kv {
            generation,
            locators,
            source,
            destination,
            base,
            base_extent,
            published_extent,
            group_id,
            scale_identity,
        } => {
            dict.set_item(intern!(py, "kind"), "kv")?;
            value.set_item(intern!(py, "generation"), generation)?;
            let locators = locators
                .iter()
                .map(|locator| transfer_locator_to_py(py, locator))
                .collect::<PyResult<Vec<_>>>()?;
            value.set_item(intern!(py, "locators"), PyList::new(py, locators)?)?;
            value.set_item(intern!(py, "source"), checkpoint_mapping_to_py(py, source)?)?;
            value.set_item(intern!(py, "destination"), destination.as_str())?;
            value.set_item(
                intern!(py, "base"),
                base.as_ref()
                    .map(|checkpoint| checkpoint_mapping_to_py(py, checkpoint))
                    .transpose()?,
            )?;
            value.set_item(intern!(py, "base_extent"), base_extent)?;
            value.set_item(intern!(py, "published_extent"), published_extent)?;
            value.set_item(intern!(py, "group_id"), group_id)?;
            value.set_item(intern!(py, "scale_identity"), scale_identity.as_str())?;
        }
        TransferHandle::Latent {
            generation,
            height,
            width,
            latent_units,
            step,
            locator,
        } => {
            dict.set_item(intern!(py, "kind"), "latent")?;
            value.set_item(intern!(py, "generation"), generation)?;
            value.set_item(intern!(py, "height"), height)?;
            value.set_item(intern!(py, "width"), width)?;
            value.set_item(intern!(py, "latent_units"), latent_units)?;
            value.set_item(intern!(py, "step"), step)?;
            value.set_item(intern!(py, "locator"), transfer_locator_to_py(py, locator)?)?;
        }
    }
    dict.set_item(intern!(py, "value"), value)?;
    Ok(dict)
}

fn request_kind_py<'py>(py: Python<'py>, kind: RequestKind) -> &'py Bound<'py, PyString> {
    match kind {
        RequestKind::Info => intern!(py, "info"),
        RequestKind::Submit => intern!(py, "submit"),
        RequestKind::Poll => intern!(py, "poll"),
        RequestKind::Close => intern!(py, "close"),
    }
}

fn product_kind_py<'py>(py: Python<'py>, kind: ProductKind) -> &'py Bound<'py, PyString> {
    match kind {
        ProductKind::Token => intern!(py, "token"),
        ProductKind::Logprob => intern!(py, "logprob"),
        ProductKind::VisionFeature => intern!(py, "vision_feature"),
        ProductKind::LatentFeature => intern!(py, "latent_feature"),
        ProductKind::Kv => intern!(py, "kv"),
        ProductKind::Latent => intern!(py, "latent"),
        ProductKind::Artifact => intern!(py, "artifact"),
        ProductKind::Completion => intern!(py, "completion"),
        ProductKind::SamplingState => intern!(py, "sampling_state"),
        ProductKind::SelectedPoint => intern!(py, "selected_point"),
    }
}

fn storage_class_py<'py>(py: Python<'py>, class: StorageClass) -> &'py Bound<'py, PyString> {
    match class {
        StorageClass::DeviceTensor => intern!(py, "device_tensor"),
        StorageClass::RequestRelay => intern!(py, "request_relay"),
        StorageClass::PagedKv => intern!(py, "paged_kv"),
        StorageClass::LatentArena => intern!(py, "latent_arena"),
        StorageClass::HostStaging => intern!(py, "host_staging"),
        StorageClass::PinnedOutput => intern!(py, "pinned_output"),
    }
}

fn dtype_py<'py>(py: Python<'py>, dtype: DType) -> &'py Bound<'py, PyString> {
    match dtype {
        DType::U8 => intern!(py, "u8"),
        DType::U16 => intern!(py, "u16"),
        DType::U32 => intern!(py, "u32"),
        DType::I32 => intern!(py, "i32"),
        DType::I64 => intern!(py, "i64"),
        DType::F16 => intern!(py, "f16"),
        DType::BF16 => intern!(py, "bf16"),
        DType::F32 => intern!(py, "f32"),
    }
}

pub(crate) fn try_completion_response_from_py(
    response: &Bound<'_, PyAny>,
) -> PyResult<Option<WorkerResponse>> {
    let py = response.py();
    let dict = response
        .cast::<PyDict>()
        .map_err(|_| PyValueError::new_err("worker response must be a mapping"))?;
    let kind = dict
        .get_item(intern!(py, "kind"))?
        .ok_or_else(|| PyValueError::new_err("worker response has no kind"))?;
    let kind = kind
        .extract::<String>()
        .map_err(|_| PyValueError::new_err("worker response kind must be a string"))?;
    match kind.as_str() {
        "result" => decode_completion_response_from_py(response)
            .map(Some)
            .ok_or_else(|| PyValueError::new_err("invalid result worker response")),
        "error" => decode_error_response_from_py(response)
            .map(Some)
            .ok_or_else(|| PyValueError::new_err("invalid error worker response")),
        _ => Ok(None),
    }
}

fn decode_completion_response_from_py(response: &Bound<'_, PyAny>) -> Option<WorkerResponse> {
    let py = response.py();
    let dict = response.cast::<PyDict>().ok()?;
    let kind = str_field(dict, intern!(py, "kind"))?;
    if kind.to_str().ok()? != "result" {
        return None;
    }
    // Result reports reserve worker information for its response kind.
    for key in [intern!(py, "info")] {
        if !absent_or_none(dict, key)? {
            return None;
        }
    }
    let report = run_result_from_py(&get(dict, intern!(py, "result"))?)?;
    let identities = error_operations_from_py(dict)?;
    if !identities.is_empty()
        || opt_string(dict, intern!(py, "message"))?.is_some()
        || opt_string(dict, intern!(py, "code"))?.is_some()
        || opt_bool(dict, intern!(py, "retryable"))?.is_some()
        || opt_bool(dict, intern!(py, "fatal"))?.is_some()
        || opt_string(dict, intern!(py, "phase"))?.is_some()
        || opt_string(dict, intern!(py, "route"))?.is_some()
    {
        return None;
    }
    Some(WorkerResponse::Result {
        call_id: opt_u64(dict, intern!(py, "call_id"))?,
        result: report,
    })
}

fn decode_error_response_from_py(response: &Bound<'_, PyAny>) -> Option<WorkerResponse> {
    let py = response.py();
    let dict = response.cast::<PyDict>().ok()?;
    if str_field(dict, intern!(py, "kind"))?.to_str().ok()? != "error" {
        return None;
    }
    for key in [intern!(py, "info"), intern!(py, "result")] {
        if !absent_or_none(dict, key)? {
            return None;
        }
    }
    let identities = error_operations_from_py(dict)?;
    Some(WorkerResponse::Error {
        call_id: opt_u64(dict, intern!(py, "call_id"))?,
        error: WorkerResponseError {
            message: string_of(&get(dict, intern!(py, "message"))?)?,
            code: opt_string(dict, intern!(py, "code"))?,
            retryable: bool_of(&get(dict, intern!(py, "retryable"))?)?,
            fatal: bool_of(&get(dict, intern!(py, "fatal"))?)?,
            phase: opt_string(dict, intern!(py, "phase"))?,
            route: opt_string(dict, intern!(py, "route"))?,
            operations: identities,
        },
    })
}

fn run_result_from_py(value: &Bound<'_, PyAny>) -> Option<RunResult> {
    let py = value.py();
    let dict = value.cast::<PyDict>().ok()?;
    let completions = get(dict, intern!(py, "completions"))?;
    let completions = completions.cast::<PyList>().ok()?;
    let mut records = Vec::with_capacity(completions.len());
    for item in completions.iter() {
        records.push(completion_record_from_py(&item)?);
    }
    let products = get(dict, intern!(py, "products"))?;
    let products = products.cast::<PyList>().ok()?;
    let mut payloads = Vec::with_capacity(products.len());
    for item in products.iter() {
        payloads.push(product_payload_from_py(&item)?);
    }
    let registration = get(dict, intern!(py, "registration"))?;
    let registration = registration.cast::<PyDict>().ok()?;
    let registration = RegistrationAck {
        visible: bool_of(&get(registration, intern!(py, "visible"))?)?,
    };
    let forward_stats = match dict.get_item(intern!(py, "forward_stats")).ok()? {
        None => None,
        Some(value) if value.is_none() => None,
        Some(value) => Some(forward_stats_from_py(&value)?),
    };
    Some(RunResult {
        batch_id: u64_of(&get(dict, intern!(py, "batch_id"))?)?,
        run_id: u64_of(&get(dict, intern!(py, "run_id"))?)?,
        completions: records,
        products: payloads,
        registration,
        worker_exec_us: opt_u64(dict, intern!(py, "worker_exec_us"))?,
        forward_stats,
        done: bool_of(&get(dict, intern!(py, "done"))?)?,
    })
}

fn forward_stats_from_py(value: &Bound<'_, PyAny>) -> Option<WorkerForwardStats> {
    let py = value.py();
    let dict = value.cast::<PyDict>().ok()?;
    Some(WorkerForwardStats {
        mode_counts: u64_map(dict, intern!(py, "mode_counts"))?,
        mode_tokens: u64_map(dict, intern!(py, "mode_tokens"))?,
        mode_us: u64_map(dict, intern!(py, "mode_us"))?,
        component_us: u64_map(dict, intern!(py, "component_us"))?,
        attention_launches: u64_of(&get(dict, intern!(py, "attention_launches"))?)?,
        attention_us: u64_of(&get(dict, intern!(py, "attention_us"))?)?,
        attention_backend_counts: u64_map(dict, intern!(py, "attention_backend_counts"))?,
        cuda_graph_captures: u64_of(&get(dict, intern!(py, "cuda_graph_captures"))?)?,
        cuda_graph_replays: u64_of(&get(dict, intern!(py, "cuda_graph_replays"))?)?,
        cuda_graph_misses: u64_of(&get(dict, intern!(py, "cuda_graph_misses"))?)?,
        cuda_graph_fallbacks: u64_of(&get(dict, intern!(py, "cuda_graph_fallbacks"))?)?,
        cuda_graph_unpadded_tokens: u64_of(&get(dict, intern!(py, "cuda_graph_unpadded_tokens"))?)?,
        cuda_graph_padded_tokens: u64_of(&get(dict, intern!(py, "cuda_graph_padded_tokens"))?)?,
        cuda_graph_runtime_mode_counts: u64_map(
            dict,
            intern!(py, "cuda_graph_runtime_mode_counts"),
        )?,
        text_decode_token_relay_hits: u64_of(&get(
            dict,
            intern!(py, "text_decode_token_relay_hits"),
        )?)?,
        text_decode_token_relay_misses: u64_of(&get(
            dict,
            intern!(py, "text_decode_token_relay_misses"),
        )?)?,
        text_decode_position_relay_hits: u64_of(&get(
            dict,
            intern!(py, "text_decode_position_relay_hits"),
        )?)?,
        text_decode_position_relay_misses: u64_of(&get(
            dict,
            intern!(py, "text_decode_position_relay_misses"),
        )?)?,
        flashinfer_decode_plan_calls: u64_of(&get(
            dict,
            intern!(py, "flashinfer_decode_plan_calls"),
        )?)?,
        flashinfer_decode_plan_reuses: u64_of(&get(
            dict,
            intern!(py, "flashinfer_decode_plan_reuses"),
        )?)?,
        flashinfer_decode_plan_rows: u64_of(&get(
            dict,
            intern!(py, "flashinfer_decode_plan_rows"),
        )?)?,
        flashinfer_decode_plan_indices: u64_of(&get(
            dict,
            intern!(py, "flashinfer_decode_plan_indices"),
        )?)?,
        flashinfer_decode_graph_plan_calls: u64_of(&get(
            dict,
            intern!(py, "flashinfer_decode_graph_plan_calls"),
        )?)?,
        flashinfer_decode_graph_plan_reuses: u64_of(&get(
            dict,
            intern!(py, "flashinfer_decode_graph_plan_reuses"),
        )?)?,
        spec_verify_rows: u64_of(&get(dict, intern!(py, "spec_verify_rows"))?)?,
        spec_verify_draft_tokens: u64_of(&get(dict, intern!(py, "spec_verify_draft_tokens"))?)?,
        spec_verify_accepted_tokens: u64_of(&get(
            dict,
            intern!(py, "spec_verify_accepted_tokens"),
        )?)?,
        spec_verify_rejected_tokens: u64_of(&get(
            dict,
            intern!(py, "spec_verify_rejected_tokens"),
        )?)?,
        spec_verify_committed_tokens: u64_of(&get(
            dict,
            intern!(py, "spec_verify_committed_tokens"),
        )?)?,
        spec_verify_path_counts: u64_map(dict, intern!(py, "spec_verify_path_counts"))?,
    })
}

fn completion_record_from_py(value: &Bound<'_, PyAny>) -> Option<ModelOutput> {
    let py = value.py();
    let dict = value.cast::<PyDict>().ok()?;
    let status = str_field(dict, intern!(py, "status"))?;
    let status = match status.to_str().ok()? {
        "ok" => OpStatus::Ok,
        "predicated" => OpStatus::Predicated,
        "error" => OpStatus::Error,
        _ => return None,
    };
    let error_code = match get(dict, intern!(py, "error_code"))? {
        value if value.is_none() => None,
        value => Some(match value.cast::<PyString>().ok()?.to_str().ok()? {
            "invalid_operation" => ErrorCode::InvalidOperation,
            "resource_exhausted" => ErrorCode::ResourceExhausted,
            "compute_error" => ErrorCode::ComputeError,
            "cancelled" => ErrorCode::Cancelled,
            "internal" => ErrorCode::Internal,
            _ => return None,
        }),
    };
    let payload = get(dict, intern!(py, "payload"))?;
    let payload = payload.cast::<PyDict>().ok()?;
    let family = string_of(&get(payload, intern!(py, "family"))?)?;
    let payload_value = payload.get_item(intern!(py, "value")).ok()?;
    let payload = match &payload_value {
        Some(value) => value.cast::<PyDict>().ok()?,
        None => payload,
    };
    let lengths = get(payload, intern!(py, "logical_lengths"))?;
    let lengths = lengths.cast::<PyDict>().ok()?;
    let logical_lengths = LogicalLengths {
        token_len: u32_of(&get(lengths, intern!(py, "token_len"))?)?,
        kv_visible_len: u32_of(&get(lengths, intern!(py, "kv_visible_len"))?)?,
        kv_computed_len: u32_of(&get(lengths, intern!(py, "kv_computed_len"))?)?,
        latent_len: u32_of(&get(lengths, intern!(py, "latent_len"))?)?,
    };
    let span = get(payload, intern!(py, "token_span"))?;
    let span = span.cast::<PyDict>().ok()?;
    let token_span = TokenSpan {
        base: u32_of(&get(span, intern!(py, "base"))?)?,
        len: u32_of(&get(span, intern!(py, "len"))?)?,
    };
    let flags = get(payload, intern!(py, "finish_flags"))?;
    let flags = flags.cast::<PyDict>().ok()?;
    let finish_flags = FinishFlags {
        eos: bool_of(&get(flags, intern!(py, "eos"))?)?,
        length: bool_of(&get(flags, intern!(py, "length"))?)?,
        stop: bool_of(&get(flags, intern!(py, "stop"))?)?,
    };
    let timing = get(dict, intern!(py, "timing_counters"))?;
    let timing = timing.cast::<PyDict>().ok()?;
    let timing_counters = TimingCounters {
        queued_us: u64_of(&get(timing, intern!(py, "queued_us"))?)?,
        device_us: u64_of(&get(timing, intern!(py, "device_us"))?)?,
        copy_us: u64_of(&get(timing, intern!(py, "copy_us"))?)?,
        host_us: u64_of(&get(timing, intern!(py, "host_us"))?)?,
    };
    let media_output = if absent_or_none(payload, intern!(py, "media_output"))? {
        None
    } else {
        let output = get(payload, intern!(py, "media_output"))?;
        let output = output.cast::<PyDict>().ok()?;
        let handle = get(output, intern!(py, "handle"))?;
        let handle = handle.cast::<PyDict>().ok()?;
        if string_of(&get(handle, intern!(py, "transport"))?)? != "posix_shm" {
            return None;
        }
        let value = get(handle, intern!(py, "value"))?;
        let value = value.cast::<PyDict>().ok()?;
        Some(MediaOutput {
            handle: ArtifactHandle::PosixShm {
                name: string_of(&get(value, intern!(py, "name"))?)?,
            },
            bytes: u64_of(&get(output, intern!(py, "bytes"))?)?,
        })
    };
    let data = ResultData {
        logical_lengths,
        token_span,
        committed_tokens: u32_vec(&get(payload, intern!(py, "committed_tokens"))?)?,
        finish_flags,
        media_output,
    };
    let result_payload = match family.as_str() {
        "ar" => ResultPayload::Ar(data),
        "encoder" => ResultPayload::Encoder(data),
        "diffusion" => ResultPayload::Diffusion(DiffusionResult {
            data,
            next_cursor: u32_of(&get(payload, intern!(py, "next_cursor"))?)?,
            done: bool_of(&get(payload, intern!(py, "done"))?)?,
        }),
        "transfer" => ResultPayload::Transfer(data),
        _ => return None,
    };
    Some(ModelOutput {
        request_key: request_key_from_py(&get(dict, intern!(py, "request_key"))?)?,
        op_id: OpId(u64_of(&get(dict, intern!(py, "op_id"))?)?),
        completion_slot_generation: u32_of(&get(dict, intern!(py, "completion_slot_generation"))?)?,
        status,
        selected_point: u32_of(&get(dict, intern!(py, "selected_point"))?)?,
        product_generations: u32_vec(&get(dict, intern!(py, "product_generations"))?)?,
        error_code,
        timing_counters,
        payload: result_payload,
    })
}

fn product_payload_from_py(value: &Bound<'_, PyAny>) -> Option<ProductPayload> {
    let py = value.py();
    let dict = value.cast::<PyDict>().ok()?;
    let value = get(dict, intern!(py, "value"))?;
    let value = value.cast::<PyDict>().ok()?;
    let kind = str_field(value, intern!(py, "kind"))?;
    let value = match kind.to_str().ok()? {
        "bytes" => {
            let bytes = get(value, intern!(py, "value"))?;
            InlineValue::Bytes(bytes.cast::<PyBytes>().ok()?.as_bytes().to_vec())
        }
        "transfer" => {
            let transfer = get(value, intern!(py, "value"))?;
            let transfer = transfer.cast::<PyDict>().ok()?;
            let kind = str_field(transfer, intern!(py, "kind"))?;
            let payload = get(transfer, intern!(py, "value"))?;
            let payload = payload.cast::<PyDict>().ok()?;
            let generation = u32_of(&get(payload, intern!(py, "generation"))?)?;
            let handle = match kind.to_str().ok()? {
                "encoder" => TransferHandle::Encoder {
                    generation,
                    height: u32_of(&get(payload, intern!(py, "height"))?)?,
                    width: u32_of(&get(payload, intern!(py, "width"))?)?,
                    payload_kind: product_kind_from_py(&get(
                        payload,
                        intern!(py, "payload_kind"),
                    )?)?,
                    locator: transfer_locator_from_py(&get(payload, intern!(py, "locator"))?)?,
                },
                "device_product" => TransferHandle::DeviceProduct {
                    generation,
                    height: u32_of(&get(payload, intern!(py, "height"))?)?,
                    width: u32_of(&get(payload, intern!(py, "width"))?)?,
                    value_range: string_of(&get(payload, intern!(py, "value_range"))?)?,
                    locator: transfer_locator_from_py(&get(payload, intern!(py, "locator"))?)?,
                },
                "kv" => {
                    let raw_locators = get(payload, intern!(py, "locators"))?;
                    let raw_locators = raw_locators.cast::<PyList>().ok()?;
                    let mut locators = Vec::with_capacity(raw_locators.len());
                    for locator in raw_locators.iter() {
                        locators.push(transfer_locator_from_py(&locator)?);
                    }
                    TransferHandle::Kv {
                        generation,
                        locators,
                        source: checkpoint_mapping_from_py(&get(payload, intern!(py, "source"))?)?,
                        destination: string_of(&get(payload, intern!(py, "destination"))?)?,
                        base: if absent_or_none(payload, intern!(py, "base"))? {
                            None
                        } else {
                            Some(checkpoint_mapping_from_py(&get(
                                payload,
                                intern!(py, "base"),
                            )?)?)
                        },
                        base_extent: u32_of(&get(payload, intern!(py, "base_extent"))?)?,
                        published_extent: u32_of(&get(payload, intern!(py, "published_extent"))?)?,
                        group_id: u32_of(&get(payload, intern!(py, "group_id"))?)?,
                        scale_identity: string_of(&get(payload, intern!(py, "scale_identity"))?)?,
                    }
                }
                "latent" => TransferHandle::Latent {
                    generation,
                    height: u32_of(&get(payload, intern!(py, "height"))?)?,
                    width: u32_of(&get(payload, intern!(py, "width"))?)?,
                    latent_units: u32_of(&get(payload, intern!(py, "latent_units"))?)?,
                    step: u32_of(&get(payload, intern!(py, "step"))?)?,
                    locator: transfer_locator_from_py(&get(payload, intern!(py, "locator"))?)?,
                },
                _ => return None,
            };
            InlineValue::Transfer(handle)
        }
        _ => return None,
    };
    Some(ProductPayload {
        product: product_ref_from_py(&get(dict, intern!(py, "product"))?)?,
        value,
    })
}

fn transfer_locator_from_py(value: &Bound<'_, PyAny>) -> Option<TransferLocator> {
    let py = value.py();
    let dict = value.cast::<PyDict>().ok()?;
    let transport = match string_of(&get(dict, intern!(py, "transport"))?)?.as_str() {
        "local" => TransferTransport::Local {
            endpoint: string_of(&get(dict, intern!(py, "endpoint"))?)?,
            key: u64_of(&get(dict, intern!(py, "key"))?)?,
        },
        "posix_shm" => TransferTransport::PosixShm {
            name: string_of(&get(dict, intern!(py, "name"))?)?,
            ready_header_bytes: u32_of(&get(dict, intern!(py, "ready_header_bytes"))?)?,
            ready_semaphore: if absent_or_none(dict, intern!(py, "ready_semaphore"))? {
                None
            } else {
                Some(string_of(&get(dict, intern!(py, "ready_semaphore"))?)?)
            },
        },
        "cuda_ipc" => TransferTransport::CudaIpc {
            endpoint: string_of(&get(dict, intern!(py, "endpoint"))?)?,
            publication_id: string_of(&get(dict, intern!(py, "publication_id"))?)?,
            storage_handle: bytes_of(&get(dict, intern!(py, "storage_handle"))?)?,
            storage_size_bytes: u64_of(&get(dict, intern!(py, "storage_size_bytes"))?)?,
            storage_offset_bytes: u64_of(&get(dict, intern!(py, "storage_offset_bytes"))?)?,
            tensor_offset: u64_of(&get(dict, intern!(py, "tensor_offset"))?)?,
            tensor_stride: i64_vec(&get(dict, intern!(py, "tensor_stride"))?)?,
            ref_counter_handle: bytes_of(&get(dict, intern!(py, "ref_counter_handle"))?)?,
            ref_counter_offset: u64_of(&get(dict, intern!(py, "ref_counter_offset"))?)?,
            event_handle: bytes_of(&get(dict, intern!(py, "event_handle"))?)?,
            event_sync_required: bool_of(&get(dict, intern!(py, "event_sync_required"))?)?,
            ready_event_handle: bytes_of(&get(dict, intern!(py, "ready_event_handle"))?)?,
        },
        _ => return None,
    };
    Some(TransferLocator {
        transport,
        nbytes: u64_of(&get(dict, intern!(py, "nbytes"))?)?,
        dtype: string_of(&get(dict, intern!(py, "dtype"))?)?,
        shape: u64_vec(&get(dict, intern!(py, "shape"))?)?,
        device: string_of(&get(dict, intern!(py, "device"))?)?,
    })
}

fn checkpoint_mapping_from_py(value: &Bound<'_, PyAny>) -> Option<Checkpoint> {
    let py = value.py();
    let dict = value.cast::<PyDict>().ok()?;
    let point = get(dict, intern!(py, "point"))?;
    let point = point.cast::<PyDict>().ok()?;
    let point = match string_of(&get(point, intern!(py, "kind"))?)?.as_str() {
        "fixed" => CheckpointPoint::Fixed(u32_of(&get(point, intern!(py, "value"))?)?),
        "device_selected" => CheckpointPoint::DeviceSelected,
        _ => return None,
    };
    Some(Checkpoint {
        op_id: OpId(u64_of(&get(dict, intern!(py, "op_id"))?)?),
        point,
    })
}

fn product_ref_from_py(value: &Bound<'_, PyAny>) -> Option<ProductRef> {
    let py = value.py();
    let dict = value.cast::<PyDict>().ok()?;
    let kind = product_kind_from_py(&get(dict, intern!(py, "kind"))?)?;
    let storage_class = str_field(dict, intern!(py, "storage_class"))?;
    let storage_class = match storage_class.to_str().ok()? {
        "device_tensor" => StorageClass::DeviceTensor,
        "request_relay" => StorageClass::RequestRelay,
        "paged_kv" => StorageClass::PagedKv,
        "latent_arena" => StorageClass::LatentArena,
        "host_staging" => StorageClass::HostStaging,
        "pinned_output" => StorageClass::PinnedOutput,
        _ => return None,
    };
    let dtype = str_field(dict, intern!(py, "dtype"))?;
    let dtype = match dtype.to_str().ok()? {
        "u8" => DType::U8,
        "u16" => DType::U16,
        "u32" => DType::U32,
        "i32" => DType::I32,
        "i64" => DType::I64,
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        "f32" => DType::F32,
        _ => return None,
    };
    let shape = get(dict, intern!(py, "shape_bound"))?;
    let shape = shape.cast::<PyDict>().ok()?;
    let dims = get(shape, intern!(py, "dims"))?;
    let dims = dims.cast::<PyList>().ok()?;
    let mut shape_bound = ShapeBound {
        dims: Vec::with_capacity(dims.len()),
    };
    for item in dims.iter() {
        let entry = item.cast::<PyDict>().ok()?;
        let value = get(entry, intern!(py, "value"))?;
        let dim_kind = str_field(entry, intern!(py, "kind"))?;
        let dim = match dim_kind.to_str().ok()? {
            "static" => DimBound::Static(u32_of(&value)?),
            "device" => {
                let value = value.cast::<PyDict>().ok()?;
                DimBound::Device {
                    max: u32_of(&get(value, intern!(py, "max"))?)?,
                }
            }
            _ => return None,
        };
        shape_bound.dims.push(dim);
    }
    let range = get(dict, intern!(py, "point_range"))?;
    let range = range.cast::<PyDict>().ok()?;
    let point_range = PointRange {
        base_point: u32_of(&get(range, intern!(py, "base_point"))?)?,
        max_points: u32_of(&get(range, intern!(py, "max_points"))?)?,
    };
    Some(ProductRef {
        request_key: request_key_from_py(&get(dict, intern!(py, "request_key"))?)?,
        producer_op_id: OpId(u64_of(&get(dict, intern!(py, "producer_op_id"))?)?),
        output_index: u16_of(&get(dict, intern!(py, "output_index"))?)?,
        generation: u32_of(&get(dict, intern!(py, "generation"))?)?,
        kind,
        storage_class,
        dtype,
        shape_bound,
        point_range,
    })
}

fn product_kind_from_py(value: &Bound<'_, PyAny>) -> Option<ProductKind> {
    Some(match string_of(value)?.as_str() {
        "token" => ProductKind::Token,
        "logprob" => ProductKind::Logprob,
        "vision_feature" => ProductKind::VisionFeature,
        "latent_feature" => ProductKind::LatentFeature,
        "kv" => ProductKind::Kv,
        "latent" => ProductKind::Latent,
        "artifact" => ProductKind::Artifact,
        "completion" => ProductKind::Completion,
        "sampling_state" => ProductKind::SamplingState,
        "selected_point" => ProductKind::SelectedPoint,
        _ => return None,
    })
}

fn request_key_from_py(value: &Bound<'_, PyAny>) -> Option<RequestKey> {
    let py = value.py();
    let dict = value.cast::<PyDict>().ok()?;
    Some(RequestKey {
        authority_id: u64_of(&get(dict, intern!(py, "authority_id"))?)?,
        request_id: uniserve_core::RequestId(u64_of(&get(dict, intern!(py, "request_id"))?)?),
        epoch: u64_of(&get(dict, intern!(py, "epoch"))?)?,
    })
}

fn error_operation_from_py(value: &Bound<'_, PyAny>) -> Option<ErrorOperationIdentity> {
    let py = value.py();
    let dict = value.cast::<PyDict>().ok()?;
    Some(ErrorOperationIdentity {
        request_key: request_key_from_py(&get(dict, intern!(py, "request_key"))?)?,
        op_id: OpId(u64_of(&get(dict, intern!(py, "op_id"))?)?),
    })
}

fn error_operations_from_py(dict: &Bound<'_, PyDict>) -> Option<Vec<ErrorOperationIdentity>> {
    let py = dict.py();
    let Some(operations) = dict.get_item(intern!(py, "operations")).ok()? else {
        return Some(Vec::new());
    };
    if operations.is_none() {
        return Some(Vec::new());
    }
    let operations = operations.cast::<PyList>().ok()?;
    let mut identities = Vec::with_capacity(operations.len());
    for item in operations.iter() {
        identities.push(error_operation_from_py(&item)?);
    }
    Some(identities)
}

// --- extraction primitives -------------------------------------------------

fn get<'py>(dict: &Bound<'py, PyDict>, key: &Bound<'py, PyString>) -> Option<Bound<'py, PyAny>> {
    dict.get_item(key).ok().flatten()
}

fn absent_or_none(dict: &Bound<'_, PyDict>, key: &Bound<'_, PyString>) -> Option<bool> {
    match dict.get_item(key).ok()? {
        Some(value) => Some(value.is_none()),
        None => Some(true),
    }
}

fn str_field<'py>(
    dict: &Bound<'py, PyDict>,
    key: &Bound<'py, PyString>,
) -> Option<Bound<'py, PyString>> {
    get(dict, key)?.cast_into::<PyString>().ok()
}

fn u64_of(value: &Bound<'_, PyAny>) -> Option<u64> {
    // Integer fields reject Python booleans instead of coercing them
    // to zero or one.
    if value.cast::<PyBool>().is_ok() {
        return None;
    }
    value.extract().ok()
}

fn u32_of(value: &Bound<'_, PyAny>) -> Option<u32> {
    if value.cast::<PyBool>().is_ok() {
        return None;
    }
    value.extract().ok()
}

fn u16_of(value: &Bound<'_, PyAny>) -> Option<u16> {
    if value.cast::<PyBool>().is_ok() {
        return None;
    }
    value.extract().ok()
}

fn bool_of(value: &Bound<'_, PyAny>) -> Option<bool> {
    Some(value.cast::<PyBool>().ok()?.is_true())
}

fn string_of(value: &Bound<'_, PyAny>) -> Option<String> {
    Some(value.cast::<PyString>().ok()?.to_str().ok()?.to_owned())
}

fn u64_map(dict: &Bound<'_, PyDict>, key: &Bound<'_, PyString>) -> Option<BTreeMap<String, u64>> {
    let values = get(dict, key)?;
    let values = values.cast::<PyDict>().ok()?;
    let mut result = BTreeMap::new();
    for (key, value) in values.iter() {
        result.insert(string_of(&key)?, u64_of(&value)?);
    }
    Some(result)
}

fn u32_vec(value: &Bound<'_, PyAny>) -> Option<Vec<u32>> {
    let list = value.cast::<PyList>().ok()?;
    let mut values = Vec::with_capacity(list.len());
    for item in list.iter() {
        values.push(u32_of(&item)?);
    }
    Some(values)
}

fn u64_vec(value: &Bound<'_, PyAny>) -> Option<Vec<u64>> {
    let list = value.cast::<PyList>().ok()?;
    let mut values = Vec::with_capacity(list.len());
    for item in list.iter() {
        values.push(u64_of(&item)?);
    }
    Some(values)
}

fn i64_vec(value: &Bound<'_, PyAny>) -> Option<Vec<i64>> {
    let list = value.cast::<PyList>().ok()?;
    let mut values = Vec::with_capacity(list.len());
    for item in list.iter() {
        if item.cast::<PyBool>().is_ok() {
            return None;
        }
        values.push(item.extract().ok()?);
    }
    Some(values)
}

fn bytes_of(value: &Bound<'_, PyAny>) -> Option<Vec<u8>> {
    Some(value.cast::<PyBytes>().ok()?.as_bytes().to_vec())
}

fn opt_u64(dict: &Bound<'_, PyDict>, key: &Bound<'_, PyString>) -> Option<Option<u64>> {
    match dict.get_item(key).ok()? {
        None => Some(None),
        Some(value) if value.is_none() => Some(None),
        Some(value) => Some(Some(u64_of(&value)?)),
    }
}

fn opt_bool(dict: &Bound<'_, PyDict>, key: &Bound<'_, PyString>) -> Option<Option<bool>> {
    match dict.get_item(key).ok()? {
        None => Some(None),
        Some(value) if value.is_none() => Some(None),
        Some(value) => Some(Some(bool_of(&value)?)),
    }
}

fn opt_string(dict: &Bound<'_, PyDict>, key: &Bound<'_, PyString>) -> Option<Option<String>> {
    match dict.get_item(key).ok()? {
        None => Some(None),
        Some(value) if value.is_none() => Some(None),
        Some(value) => Some(Some(string_of(&value)?)),
    }
}

// ---------------------------------------------------------------------------
// Native boundary qualification
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::time::{Duration, SystemTime};

    use pythonize::pythonize;
    use uniserve_core::{BlockId, RequestId};
    use uniserve_worker_ipc::ClientEndpoint;

    use super::*;

    fn execute_request() -> WorkerRequest {
        let request_key = RequestKey::new(1, RequestId(2), 1);
        let admission = NewRequest::new(
            request_key,
            1,
            Some(ArRequestParams {
                sampling: SamplingParams {
                    temperature: 0.0,
                    ignore_eos: true,
                    ..SamplingParams::default()
                },
                negative_token_ids: Vec::new(),
                finish_token_ids: Vec::new(),
                initial_position: 0,
            }),
            None,
        )
        .unwrap();
        let input = ProductRef {
            request_key,
            producer_op_id: OpId(1),
            output_index: u16::MAX,
            generation: 1,
            kind: ProductKind::Token,
            storage_class: StorageClass::HostStaging,
            dtype: DType::U32,
            shape_bound: ShapeBound {
                dims: vec![DimBound::Static(2)],
            },
            point_range: PointRange::default(),
        };
        let token = ProductRef {
            request_key,
            producer_op_id: OpId(1),
            output_index: 0,
            generation: 5,
            kind: ProductKind::Token,
            storage_class: StorageClass::RequestRelay,
            dtype: DType::U32,
            shape_bound: ShapeBound::default(),
            point_range: PointRange {
                base_point: 0,
                max_points: 1,
            },
        };
        let operation = Operation {
            request_key,
            op_id: OpId(1),
            parent: Checkpoint::admission_root(OpId(0)),
            kind: RunKind::ArExtend,
            payload: OpPayload::new(
                RunKind::ArExtend,
                Bounds {
                    max_points: 1,
                    max_tokens: 2,
                    max_kv_pages: 1,
                    ..Bounds::default()
                },
                vec![input.clone()],
                vec![token],
                None,
                None,
                0,
            ),
        }
        .sealed();
        let block_tables = vec![BlockTable {
            request_pool_idx: 1,
            group_id: 0,
            page_ids: vec![BlockId(1)],
            allocated_tokens: 2,
        }];
        let new_cache_pages = vec![CachePageAllocation {
            request_pool_idx: 1,
            group_id: 0,
            page_ids: vec![BlockId(1)],
        }];
        let forward_rows = vec![RowGeometry {
            operation_index: 0,
            request_pool_index: 1,
            seq_len: 0,
            query_len: 2,
        }];
        let media_key = RequestKey::new(1, RequestId(3), 1);
        let media_prompt_token_ids = vec![17, 23, 65_537];
        let media_admission = NewRequest::new_media(
            media_key,
            2,
            DiffusionRequestParams {
                prompt_token_ids: media_prompt_token_ids,
                seed: 29,
                geometry: MediaGeometry {
                    frame_count: 22,
                    decode_units: 3,
                    prompt_tokens: 3,
                    denoise_steps: 4,
                },
            },
        )
        .unwrap();
        let media_operation = Operation {
            request_key: media_key,
            op_id: OpId(2),
            parent: Checkpoint::admission_root(OpId(0)),
            kind: RunKind::DiffusionPrepare,
            payload: OpPayload::new(
                RunKind::DiffusionPrepare,
                Bounds {
                    max_points: 1,
                    ..Bounds::default()
                },
                Vec::new(),
                Vec::new(),
                None,
                None,
                0,
            ),
        }
        .sealed();
        let latent_placements = vec![LatentPlacement {
            request_key: media_key,
            op_id: OpId(2),
            page_table: vec![1],
            latent_units: 64,
            height: 768,
            width: 1344,
            start_step: 0,
            step_count: 0,
        }];
        let mut run = Run::new(
            11,
            vec![admission, media_admission],
            vec![operation, media_operation],
        );
        run.block_tables = block_tables;
        run.new_cache_pages = new_cache_pages;
        run.forward_rows = forward_rows;
        run.latent_placements = latent_placements;
        let mut request = WorkerRequest::submit(run.with_input_products(vec![ProductPayload {
            product: input,
            value: InlineValue::Bytes(uniserve_worker_ipc::encode_token_product_bytes(&[7, 8])),
        }]));
        request.set_call_id(Some(9));
        request
    }

    fn result_response() -> WorkerResponse {
        let request_key = RequestKey::new(1, RequestId(2), 1);
        let mut response = WorkerResponse::result(RunResult {
            batch_id: 11,
            run_id: 11,
            completions: vec![ModelOutput {
                request_key,
                op_id: OpId(1),
                completion_slot_generation: 1,
                status: OpStatus::Ok,
                selected_point: 1,
                product_generations: vec![5],
                error_code: None,
                timing_counters: TimingCounters::default(),
                payload: ResultPayload::Ar(ResultData {
                    logical_lengths: LogicalLengths {
                        token_len: 2,
                        kv_visible_len: 2,
                        latent_len: 0,
                        kv_computed_len: 2,
                    },
                    token_span: TokenSpan { base: 0, len: 1 },
                    committed_tokens: vec![42],
                    finish_flags: FinishFlags::default(),
                    media_output: None,
                }),
            }],
            products: Vec::new(),
            registration: RegistrationAck { visible: true },
            worker_exec_us: Some(12),
            forward_stats: None,
            done: true,
        });
        response.set_call_id(Some(9));
        response
    }

    #[test]
    fn native_execute_and_result_round_trip_preserves_values() {
        Python::initialize();
        let nonce = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let service = format!("uniserve/ipc-py-test-{}-{nonce}", std::process::id());
        let server = crate::PyServer::new(&service, 1 << 20, 2).unwrap();
        let client = ClientEndpoint::connect(&service, 1 << 20, 2).unwrap();
        let request = execute_request();
        let pending = client.send_request(&request).unwrap();
        let expected = result_response();

        Python::attach(|py| {
            let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .canonicalize()
                .unwrap();
            py.import("sys")
                .unwrap()
                .getattr("path")
                .unwrap()
                .call_method1("insert", (0, repo_root.to_str().unwrap()))
                .unwrap();
            let native_request = server.recv(py).unwrap();
            let request_dict = native_request.bind(py).cast::<PyDict>().unwrap();
            let native_run = request_dict.get_item("run").unwrap().unwrap();
            assert_eq!(
                native_run
                    .getattr("run_id")
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                11
            );
            assert_eq!(native_run.getattr("operations").unwrap().len().unwrap(), 2);
            let admissions = native_run.getattr("admissions").unwrap();
            let media = admissions
                .get_item(1)
                .unwrap()
                .getattr("diffusion")
                .unwrap();
            assert_eq!(
                media
                    .getattr("prompt_token_ids")
                    .unwrap()
                    .extract::<Vec<u32>>()
                    .unwrap(),
                vec![17, 23, 65_537]
            );
            // The natively constructed batch must be exactly what the
            // canonical codec decodes from its own IPC form.
            let round_tripped = py
                .import("uniserve_worker.execution.batch")
                .unwrap()
                .getattr("Run")
                .unwrap()
                .call_method1(
                    "from_mapping",
                    (native_run.call_method0("to_mapping").unwrap(),),
                )
                .unwrap();
            assert!(
                round_tripped.eq(&native_run).unwrap(),
                "native batch construction diverged from the canonical codec"
            );

            let response = pythonize(py, &expected).unwrap();
            server.respond(py, &response).unwrap();
        });

        let response = client
            .recv_response_timeout(&pending, Duration::from_secs(5))
            .unwrap()
            .expect("native result response");
        assert_eq!(response.decode_response().unwrap(), expected);
    }
}
