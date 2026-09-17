//! Typed conversion between worker IPC frames and Python mappings.
//!
//! The conversion path crosses the FFI boundary once in each direction per
//! batch. Mapping keys and enum strings are interned, lists are preallocated,
//! and binary payloads remain Python `bytes`.
//!
//! [`execute_request_to_py`] builds worker input, while
//! [`try_completion_response_from_py`] strictly decodes worker output.

use std::collections::HashMap;
use uniserve_worker_ipc::{ForwardMode, PipelineStage, TransferMode};

use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyString};
use uniserve_core::{ImageParams, SamplingParams};
use uniserve_worker_ipc::{
    ArRequestParams, BatchCommand, BlockTable, BufferAllocation, BufferId, CachePageAllocation,
    Computation, ComputationId, DType, DecodeRange, DiffusionSamplingParams, DimBound, DrawLayout,
    FeatureKind, KvTransfer, LatentParams, Locator, NewRequest, RequestKey, RequestKind,
    ScheduleBatch, ScheduledRequest, ShapeBound, TensorPublication, TensorRef, TensorTransfer,
    TransferHandle, TransferTransport, UmmRequestParams, WorkerRequest,
};

#[cfg(test)]
use uniserve_worker_ipc::Bounds;

/// Converts a submit [`WorkerRequest`] into the Python worker mapping.
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

/// Cached handles to the worker's operation types and enum members.
///
/// The serve loop constructs one `ScheduleBatch` object per submission; every hot
/// record (operations, KV allocations, admission/close/release commands) is built
/// by calling the operation dataclass constructors positionally, so the worker
/// never re-decodes those records from IPC maps. Rare members (admissions,
/// input products and allocations) still cross as IPC maps and are decoded by
/// `native_run` on the Python side.
struct NativeRequestTypes {
    operation: Py<PyAny>,
    computation_id: Py<PyAny>,
    request_key: Py<PyAny>,
    tensor_ref: Py<PyAny>,
    buffer_id: Py<PyAny>,
    shape_bound: Py<PyAny>,
    static_dim: Py<PyAny>,
    device_dim: Py<PyAny>,
    bounds: Py<PyAny>,
    rng: Py<PyAny>,
    sampling_state: Py<PyAny>,
    block_table: Py<PyAny>,
    cache_page_allocation: Py<PyAny>,
    start: Py<PyAny>,
    finish: Py<PyAny>,
    free: Py<PyAny>,
    native_run: Py<PyAny>,
    dtypes: [Py<PyAny>; 7],
    draw_layouts: [Py<PyAny>; 3],
    forward_modes: [Py<PyAny>; ForwardMode::ALL.len()],
    pipeline_stages: [Py<PyAny>; PipelineStage::ALL.len()],
    transfer_modes: [Py<PyAny>; TransferMode::ALL.len()],
}

/// Process-wide cache of worker Python constructors and enum members.
static NATIVE_REQUEST_TYPES: std::sync::OnceLock<NativeRequestTypes> = std::sync::OnceLock::new();

/// Resolves named Python enum members into a fixed-size indexed cache.
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
    /// Imports worker model types and caches every constructor and enum member.
    fn build(py: Python<'_>) -> PyResult<Self> {
        // Resolve classes once so per-request conversion uses direct constructor
        // calls without repeated module or attribute lookup. Each name is
        // imported from the protocol module that owns it at the current Python
        // package structure.
        let batch = py.import("uniserve_worker.protocol.batch")?;
        let operation = py.import("uniserve_worker.protocol.operation")?;
        let identity = py.import("uniserve_worker.protocol.identity")?;
        let tensor = py.import("uniserve_worker.protocol.tensor")?;
        let class = |module: &Bound<'_, PyModule>, name: &str| -> PyResult<Py<PyAny>> {
            Ok(module.getattr(name)?.unbind())
        };

        Ok(Self {
            operation: class(&operation, "ScheduledRequest")?,
            computation_id: class(&identity, "ComputationId")?,
            request_key: class(&identity, "RequestKey")?,
            tensor_ref: class(&tensor, "TensorRef")?,
            buffer_id: class(&identity, "BufferId")?,
            shape_bound: class(&tensor, "ShapeBound")?,
            static_dim: class(&tensor, "StaticDim")?,
            device_dim: class(&tensor, "DeviceDim")?,
            bounds: class(&operation, "Bounds")?,
            rng: class(&operation, "Rng")?,
            sampling_state: class(&operation, "SamplingState")?,
            block_table: class(&batch, "BlockTable")?,
            cache_page_allocation: class(&batch, "CachePageAllocation")?,
            start: class(&batch, "Start")?,
            finish: class(&batch, "Finish")?,
            free: class(&batch, "Free")?,
            native_run: class(&batch, "native_run")?,

            // Enum members follow the stable Rust discriminant order used by
            // the indexed accessors below.
            dtypes: enum_members(
                &tensor,
                "DType",
                ["u8", "i32", "i64", "f16", "bf16", "f32", "i16"],
            )?,
            draw_layouts: enum_members(
                &operation,
                "DrawLayout",
                ["target_sampling", "speculative_proposal", "flow_noise"],
            )?,
            forward_modes: enum_members(
                &operation,
                "ForwardMode",
                ForwardMode::ALL.map(ForwardMode::as_str),
            )?,
            pipeline_stages: enum_members(
                &operation,
                "PipelineStage",
                PipelineStage::ALL.map(PipelineStage::as_str),
            )?,
            transfer_modes: enum_members(
                &operation,
                "TransferMode",
                TransferMode::ALL.map(TransferMode::as_str),
            )?,
        })
    }

    /// Returns the process-wide type cache, initializing it under the GIL.
    fn get(py: Python<'_>) -> PyResult<&'static Self> {
        if let Some(types) = NATIVE_REQUEST_TYPES.get() {
            return Ok(types);
        }
        let built = Self::build(py)?;
        Ok(NATIVE_REQUEST_TYPES.get_or_init(|| built))
    }

    /// Returns the Python enum member for a physical run kind.
    fn kind<'py>(&self, py: Python<'py>, kind: Computation) -> Bound<'py, PyAny> {
        let member = match kind {
            Computation::Forward(mode) => &self.forward_modes[mode as usize],
            Computation::Pipeline(stage) => &self.pipeline_stages[stage as usize],
            Computation::Transfer(mode) => &self.transfer_modes[mode as usize],
        };
        member.bind(py).clone()
    }

    /// Returns the Python enum member for an element type.
    fn dtype<'py>(&self, py: Python<'py>, dtype: DType) -> Bound<'py, PyAny> {
        let index = match dtype {
            DType::U8 => 0,
            DType::I32 => 1,
            DType::I64 => 2,
            DType::F16 => 3,
            DType::BF16 => 4,
            DType::F32 => 5,
            DType::I16 => 6,
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
    computation_ids: HashMap<ComputationId, Py<PyAny>>,
    shape_bounds: HashMap<ShapeBound, Py<PyAny>>,
}

impl<'py> NativeRequestConversion<'py> {
    /// Starts a batch conversion with shared Python types and empty value caches.
    fn new(py: Python<'py>) -> PyResult<Self> {
        Ok(Self {
            py,
            types: NativeRequestTypes::get(py)?,
            request_keys: HashMap::new(),
            computation_ids: HashMap::new(),
            shape_bounds: HashMap::new(),
        })
    }

    /// Shares repeated producer and predecessor coordinates within the physical batch.
    fn computation_id(&mut self, id: ComputationId) -> PyResult<Bound<'py, PyAny>> {
        if let Some(value) = self.computation_ids.get(&id) {
            return Ok(value.bind(self.py).clone());
        }
        let value = self
            .types
            .computation_id
            .bind(self.py)
            .call1((id.batch_id, id.request_index))?;
        self.computation_ids.insert(id, value.clone().unbind());
        Ok(value)
    }

    /// Returns a canonical Python request identity for this batch.
    fn request_key(&mut self, key: RequestKey) -> PyResult<Bound<'py, PyAny>> {
        if let Some(value) = self.request_keys.get(&key) {
            return Ok(value.bind(self.py).clone());
        }
        let value = self.types.request_key.bind(self.py).call1((
            key.engine_id,
            key.request_id.0,
            key.request_epoch,
        ))?;
        self.request_keys.insert(key, value.clone().unbind());
        Ok(value)
    }

    /// Returns a canonical Python shape bound for this batch.
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

    /// Constructs a typed Python product reference from shared leaf objects.
    fn tensor_ref(&mut self, product: &TensorRef) -> PyResult<Bound<'py, PyAny>> {
        let request_key = self.request_key(product.request_key)?;
        let shape_bound = self.shape_bound(&product.shape_bound)?;
        self.types.tensor_ref.bind(self.py).call1((
            request_key,
            self.computation_id(product.producer_op_id)?,
            product.output_index,
            product.generation,
            self.types.dtype(self.py, product.dtype),
            shape_bound,
        ))
    }

    /// Constructs a typed Python persistent-buffer identity.
    fn buffer_id(&mut self, buffer: BufferId) -> PyResult<Bound<'py, PyAny>> {
        let owner = self.request_key(buffer.owner)?;
        self.types.buffer_id.bind(self.py).call1((
            owner,
            self.computation_id(buffer.producer_op_id)?,
            buffer.output_index,
            buffer.generation,
        ))
    }

    /// Constructs a typed Python operation from its computation fields.
    fn operation(&mut self, operation: &ScheduledRequest) -> PyResult<Bound<'py, PyAny>> {
        // Resolve identity, lineage, resource bounds, and product references
        // before constructing the operation.
        let request_key = self.request_key(operation.request_key)?;
        let predecessor = operation
            .predecessor
            .map(|id| self.computation_id(id))
            .transpose()?;
        let bounds = self.types.bounds.bind(self.py).call1((
            operation.bounds.max_tokens,
            operation.bounds.max_kv_pages,
            operation.bounds.max_latent_bytes,
            operation.bounds.max_completion_bytes,
            operation.bounds.max_transfer_bytes,
        ))?;
        let inputs = operation
            .inputs
            .as_slice()
            .iter()
            .map(|product| self.tensor_ref(product))
            .collect::<PyResult<Vec<_>>>()?;
        let outputs = operation
            .outputs
            .as_slice()
            .iter()
            .map(|product| self.tensor_ref(product))
            .collect::<PyResult<Vec<_>>>()?;
        let predicate = operation
            .predicate
            .as_ref()
            .as_ref()
            .map(|predicate| self.tensor_ref(predicate))
            .transpose()?;
        let rng = operation
            .rng
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
        let sampling_state = operation
            .sampling_state
            .as_ref()
            .map(|state| {
                let allowed = state
                    .allowed_token_ids
                    .as_ref()
                    .map(|ids| pyo3::types::PyTuple::new(py, ids))
                    .transpose()?;
                self.types.sampling_state.bind(py).call1((
                    allowed,
                    pyo3::types::PyTuple::new(py, &state.suppressed_token_ids)?,
                    pyo3::types::PyTuple::new(py, &state.finish_token_ids)?,
                    pyo3::types::PyTuple::new(py, &state.transition_token_ids)?,
                    state.force_finish,
                ))
            })
            .transpose()?;
        let arguments = pyo3::types::PyTuple::new(
            py,
            [
                request_key.into_any(),
                self.computation_id(operation.op_id)?,
                predecessor.into_pyobject(py)?.into_any(),
                self.types.kind(py, operation.code),
                bounds.into_any(),
                operation.entry.clone().into_pyobject(py)?.into_any(),
                pyo3::types::PyTuple::new(py, inputs)?.into_any(),
                pyo3::types::PyTuple::new(py, outputs)?.into_any(),
                operation
                    .token_input
                    .as_ref()
                    .map(|tensor| self.tensor_ref(tensor))
                    .transpose()?
                    .unwrap_or_else(|| py.None().into_bound(py)),
                operation
                    .token_output
                    .as_ref()
                    .map(|tensor| self.tensor_ref(tensor))
                    .transpose()?
                    .unwrap_or_else(|| py.None().into_bound(py)),
                operation
                    .vision_input
                    .as_ref()
                    .map(|tensor| self.tensor_ref(tensor))
                    .transpose()?
                    .unwrap_or_else(|| py.None().into_bound(py)),
                operation
                    .latent_feature_input
                    .as_ref()
                    .map(|tensor| self.tensor_ref(tensor))
                    .transpose()?
                    .unwrap_or_else(|| py.None().into_bound(py)),
                operation
                    .encoder_output
                    .as_ref()
                    .map(|tensor| self.tensor_ref(tensor))
                    .transpose()?
                    .unwrap_or_else(|| py.None().into_bound(py)),
                operation
                    .latent_input
                    .as_ref()
                    .map(|tensor| self.tensor_ref(tensor))
                    .transpose()?
                    .unwrap_or_else(|| py.None().into_bound(py)),
                operation
                    .latent_output
                    .as_ref()
                    .map(|tensor| self.tensor_ref(tensor))
                    .transpose()?
                    .unwrap_or_else(|| py.None().into_bound(py)),
                operation
                    .image_input
                    .as_ref()
                    .map(|tensor| self.tensor_ref(tensor))
                    .transpose()?
                    .unwrap_or_else(|| py.None().into_bound(py)),
                operation
                    .image_output
                    .as_ref()
                    .map(|tensor| self.tensor_ref(tensor))
                    .transpose()?
                    .unwrap_or_else(|| py.None().into_bound(py)),
                operation
                    .completion_output
                    .as_ref()
                    .map(|tensor| self.tensor_ref(tensor))
                    .transpose()?
                    .unwrap_or_else(|| py.None().into_bound(py)),
                operation
                    .transition_output
                    .as_ref()
                    .map(|tensor| self.tensor_ref(tensor))
                    .transpose()?
                    .unwrap_or_else(|| py.None().into_bound(py)),
                predicate
                    .map(Bound::into_any)
                    .unwrap_or_else(|| py.None().into_bound(py)),
                rng.map(Bound::into_any)
                    .unwrap_or_else(|| py.None().into_bound(py)),
                sampling_state
                    .map(Bound::into_any)
                    .unwrap_or_else(|| py.None().into_bound(py)),
                pyo3::types::PyTuple::new(py, &operation.input_token_ids)?.into_any(),
                operation
                    .input_image
                    .as_deref()
                    .into_pyobject(py)?
                    .into_any(),
                operation
                    .kv_input
                    .map(|buffer| self.buffer_id(buffer))
                    .transpose()?
                    .unwrap_or_else(|| py.None().into_bound(py)),
                operation
                    .kv_output
                    .map(|buffer| self.buffer_id(buffer))
                    .transpose()?
                    .unwrap_or_else(|| py.None().into_bound(py)),
            ],
        )?;
        self.types.operation.bind(py).call1(arguments)
    }

    /// Constructs a typed Python KV block table.
    fn block_table(&self, table: &BlockTable) -> PyResult<Bound<'py, PyAny>> {
        self.types.block_table.bind(self.py).call1((
            table.request_pool_idx,
            table.group_id,
            pyo3::types::PyTuple::new(self.py, table.page_ids.iter().map(|page| page.0))?,
            table.allocated_tokens,
        ))
    }

    /// Constructs a typed Python KV page allocation.
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

    /// Constructs the typed Python variant for one batch control command.
    fn command(&mut self, command: &BatchCommand) -> PyResult<Bound<'py, PyAny>> {
        // Start retains its schema-shaped admission mapping; steady-state
        // controls use their cached typed constructors directly.
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

            BatchCommand::Finish {
                request_key,
                retained_buffers,
            } => {
                let request_key = self.request_key(*request_key)?;
                let retained = retained_buffers
                    .iter()
                    .map(|buffer| self.buffer_id(*buffer))
                    .collect::<PyResult<Vec<_>>>()?;
                self.types
                    .finish
                    .bind(self.py)
                    .call1((request_key, pyo3::types::PyTuple::new(self.py, retained)?))
            }
            BatchCommand::Free { buffer } => {
                let buffer = self.buffer_id(*buffer)?;
                self.types.free.bind(self.py).call1((buffer,))
            }
        }
    }
}

/// Constructs one Python run, using typed hot-path objects and mapped rare records.
fn run_to_py<'py>(py: Python<'py>, run: &ScheduleBatch) -> PyResult<Bound<'py, PyAny>> {
    let mut context = RequestConversion::new(py);
    let mut native = NativeRequestConversion::new(py)?;

    // Materialize the execution-critical records as Python model instances.
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
    let commands = run
        .commands
        .iter()
        .map(|command| native.command(command))
        .collect::<PyResult<Vec<_>>>()?;

    // Rare payloads retain their schema-shaped mapping representation.
    let input_products = dict_list(py, &run.input_products, |payload| {
        tensor_publication_to_py(py, payload, &mut context)
    })?;

    // `native_run` assembles both representations without reparsing typed leaves.
    let arguments = pyo3::types::PyTuple::new(
        py,
        [
            run.batch_id.into_pyobject(py)?.into_any(),
            run.run_id.into_pyobject(py)?.into_any(),
            run.collective_seq.into_pyobject(py)?.into_any(),
            pyo3::types::PyTuple::new(py, operations)?.into_any(),
            pyo3::types::PyTuple::new(py, block_tables)?.into_any(),
            pyo3::types::PyTuple::new(py, new_cache_pages)?.into_any(),
            (
                pyo3::types::PyTuple::new(py, &run.forward.operation_indices)?,
                pyo3::types::PyTuple::new(py, &run.forward.request_pool_indices)?,
                pyo3::types::PyTuple::new(py, &run.forward.seq_lens)?,
                pyo3::types::PyTuple::new(py, &run.forward.query_lens)?,
                pyo3::types::PyTuple::new(py, &run.forward.write_kv)?,
            )
                .into_pyobject(py)?
                .into_any(),
            dict_list(py, &run.latent_params, |params| {
                latent_params_to_py(py, params, &mut context)
            })?
            .into_any(),
            dict_list(py, &run.decode_ranges, |params| {
                decode_range_to_py(py, params, &mut context)
            })?
            .into_any(),
            dict_list(py, &run.buffer_allocations, |params| {
                buffer_allocation_to_py(py, params, &mut context)
            })?
            .into_any(),
            pyo3::types::PyTuple::new(py, commands)?.into_any(),
            input_products.into_any(),
            dict_list(py, &run.kv_inputs, |transfer| {
                kv_transfer_to_py(py, transfer)
            })?
            .into_any(),
        ],
    )?;
    native.types.native_run.bind(py).call1(arguments)
}

/// Converts a latent-page params into its Python mapping shape.
fn latent_params_to_py<'py>(
    py: Python<'py>,
    params: &LatentParams,
    context: &mut RequestConversion<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(
        intern!(py, "request_key"),
        context.request_key(params.request_key)?,
    )?;
    dict.set_item(
        intern!(py, "op_id"),
        computation_id_to_py(py, params.op_id)?,
    )?;
    dict.set_item(intern!(py, "page_table"), u32_list(py, &params.page_table)?)?;
    dict.set_item(intern!(py, "latent_units"), params.latent_units)?;
    dict.set_item(intern!(py, "height"), params.height)?;
    dict.set_item(intern!(py, "width"), params.width)?;
    dict.set_item(intern!(py, "start_step"), params.start_step)?;
    dict.set_item(intern!(py, "step_count"), params.step_count)?;
    Ok(dict)
}

/// Converts a diffusion decoder params into its Python mapping shape.
fn decode_range_to_py<'py>(
    py: Python<'py>,
    params: &DecodeRange,
    context: &mut RequestConversion<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(
        intern!(py, "request_key"),
        context.request_key(params.request_key)?,
    )?;
    dict.set_item(
        intern!(py, "op_id"),
        computation_id_to_py(py, params.op_id)?,
    )?;
    dict.set_item(intern!(py, "cursor"), params.cursor)?;
    dict.set_item(intern!(py, "max_units"), params.max_units)?;
    Ok(dict)
}

/// Converts a persistent-buffer byte span into its nested Python mapping.
fn buffer_allocation_to_py<'py>(
    py: Python<'py>,
    params: &BufferAllocation,
    context: &mut RequestConversion<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    let id = params.buffer;
    let buffer = PyDict::new(py);
    buffer.set_item(intern!(py, "owner"), context.request_key(id.owner)?)?;
    buffer.set_item(
        intern!(py, "producer_op_id"),
        computation_id_to_py(py, id.producer_op_id)?,
    )?;
    buffer.set_item(intern!(py, "output_index"), id.output_index)?;
    buffer.set_item(intern!(py, "generation"), id.generation)?;
    dict.set_item(intern!(py, "buffer"), buffer)?;
    dict.set_item(intern!(py, "offset"), params.offset)?;
    dict.set_item(intern!(py, "bytes"), params.bytes)?;
    Ok(dict)
}

/// Converts a Rust slice into a Python list of mapped dictionaries.
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

/// Per-batch cache for schema-shaped dictionary leaves shared by mapped records.
struct RequestConversion<'py> {
    py: Python<'py>,
    request_keys: HashMap<RequestKey, Bound<'py, PyDict>>,
    shape_bounds: HashMap<ShapeBound, Bound<'py, PyDict>>,
}

impl<'py> RequestConversion<'py> {
    /// Starts a mapped-record conversion with empty identity caches.
    fn new(py: Python<'py>) -> Self {
        Self {
            py,
            request_keys: HashMap::new(),
            shape_bounds: HashMap::new(),
        }
    }

    /// Returns a canonical request-identity mapping for this batch.
    fn request_key(&mut self, key: RequestKey) -> PyResult<Bound<'py, PyDict>> {
        if let Some(value) = self.request_keys.get(&key) {
            return Ok(value.clone());
        }
        let dict = PyDict::new(self.py);
        dict.set_item(intern!(self.py, "engine_id"), key.engine_id)?;
        dict.set_item(intern!(self.py, "request_id"), key.request_id.0)?;
        dict.set_item(intern!(self.py, "request_epoch"), key.request_epoch)?;
        self.request_keys.insert(key, dict.clone());
        Ok(dict)
    }

    /// Returns a canonical shape-bound mapping for this batch.
    fn shape_bound(&mut self, shape: &ShapeBound) -> PyResult<Bound<'py, PyDict>> {
        if let Some(value) = self.shape_bounds.get(shape) {
            return Ok(value.clone());
        }
        let dict = shape_bound_to_py(self.py, shape)?;
        self.shape_bounds.insert(shape.clone(), dict.clone());
        Ok(dict)
    }
}

/// Copies `u32` values into a Python list without an intermediate mapping.
fn u32_list<'py>(py: Python<'py>, values: &[u32]) -> PyResult<Bound<'py, PyList>> {
    PyList::new(py, values.iter().copied())
}

/// Converts a request admission and its selected parameter family.
fn admission_to_py<'py>(
    py: Python<'py>,
    admission: &NewRequest,
    context: &mut RequestConversion<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(
        intern!(py, "prompt_token_ids"),
        u32_list(py, &admission.prompt_token_ids)?,
    )?;
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

/// Converts autoregressive admission parameters into a Python mapping.
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

/// Converts unified-multimodal admission parameters into a Python mapping.
fn umm_params_to_py<'py>(
    py: Python<'py>,
    branch: &UmmRequestParams,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "image"), image_to_py(py, &branch.image)?)?;
    Ok(dict)
}

/// Converts diffusion admission parameters and resolved media geometry.
fn diffusion_params_to_py<'py>(
    py: Python<'py>,
    diffusion: &DiffusionSamplingParams,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "num_frames"), diffusion.num_frames)?;
    dict.set_item(
        intern!(py, "num_decode_chunks"),
        diffusion.num_decode_chunks,
    )?;
    dict.set_item(
        intern!(py, "num_inference_steps"),
        diffusion.num_inference_steps,
    )?;
    dict.set_item(intern!(py, "seed"), diffusion.seed)?;
    Ok(dict)
}

/// Converts sampling controls into the worker's Python mapping schema.
fn sampling_to_py<'py>(py: Python<'py>, sampling: &SamplingParams) -> PyResult<Bound<'py, PyDict>> {
    // Scalar controls are inserted directly under interned protocol keys.
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

    // Preserve the tuple shape expected for each token-bias pair.
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

    // Materialize nested token collections only after the scalar fields.
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

/// Converts image-generation controls into the worker's Python mapping schema.
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
    // The interval remains a tuple to match the Python parameter contract.
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

/// Converts a product identity, storage contract, and bounds into a mapping.
fn tensor_ref_to_py<'py>(
    py: Python<'py>,
    product: &TensorRef,
    context: &mut RequestConversion<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(
        intern!(py, "request_key"),
        context.request_key(product.request_key)?,
    )?;
    dict.set_item(
        intern!(py, "producer_op_id"),
        computation_id_to_py(py, product.producer_op_id)?,
    )?;
    dict.set_item(intern!(py, "output_index"), product.output_index)?;
    dict.set_item(intern!(py, "generation"), product.generation)?;
    dict.set_item(intern!(py, "dtype"), dtype_py(py, product.dtype))?;
    dict.set_item(
        intern!(py, "shape_bound"),
        context.shape_bound(&product.shape_bound)?,
    )?;
    Ok(dict)
}

/// Converts static and device-bounded dimensions into their tagged mappings.
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

/// Converts a tensor publication into its Python mapping.
fn tensor_publication_to_py<'py>(
    py: Python<'py>,
    payload: &TensorPublication,
    context: &mut RequestConversion<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(
        intern!(py, "product"),
        tensor_ref_to_py(py, &payload.product, context)?,
    )?;
    dict.set_item(
        intern!(py, "value"),
        transfer_handle_to_py(py, &payload.value)?,
    )?;
    Ok(dict)
}

/// Converts tensor metadata and transport coordinates into a Python mapping.
fn transfer_locator_to_py<'py>(py: Python<'py>, locator: &Locator) -> PyResult<Bound<'py, PyDict>> {
    // Tensor metadata is common to every transport family.
    let dict = PyDict::new(py);
    let source = PyDict::new(py);
    source.set_item("worker_id", &locator.source.worker_id)?;
    source.set_item("rank", locator.source.rank)?;
    source.set_item("node", &locator.source.node)?;
    source.set_item("address_space", &locator.source.address_space)?;
    source.set_item("incarnation", &locator.source.incarnation)?;
    dict.set_item("source", source)?;
    dict.set_item(intern!(py, "nbytes"), locator.nbytes)?;
    dict.set_item(intern!(py, "dtype"), locator.dtype.as_str())?;
    dict.set_item(intern!(py, "shape"), PyList::new(py, &locator.shape)?)?;
    dict.set_item(intern!(py, "device"), locator.device.as_str())?;
    dict.set_item(intern!(py, "offset"), PyList::new(py, &locator.offset)?)?;

    // The transport tag determines the remaining coordinate fields.
    match &locator.transport {
        TransferTransport::Local { endpoint, key } => {
            dict.set_item(intern!(py, "transport"), "local")?;
            dict.set_item(intern!(py, "endpoint"), endpoint.as_str())?;
            dict.set_item(intern!(py, "key"), key)?;
        }
        TransferTransport::PosixShm { endpoint, name } => {
            dict.set_item(intern!(py, "transport"), "posix_shm")?;
            dict.set_item(intern!(py, "endpoint"), endpoint.as_str())?;
            dict.set_item(intern!(py, "name"), name.as_str())?;
        }
        TransferTransport::CudaIpc {
            endpoint,
            publication_id,
            storage_size_bytes,
            storage_offsets_bytes,
            span_lengths,
            span_counts,
            tensor_stride,
            ready_event_handle,
        } => {
            dict.set_item(intern!(py, "transport"), "cuda_ipc")?;
            dict.set_item(intern!(py, "endpoint"), endpoint.as_str())?;
            dict.set_item(intern!(py, "publication_id"), publication_id.as_str())?;
            dict.set_item(intern!(py, "storage_size_bytes"), storage_size_bytes)?;
            dict.set_item(intern!(py, "storage_offsets_bytes"), storage_offsets_bytes)?;
            dict.set_item(intern!(py, "span_lengths"), span_lengths)?;
            dict.set_item(intern!(py, "span_counts"), span_counts)?;
            dict.set_item(
                intern!(py, "tensor_stride"),
                PyList::new(py, tensor_stride)?,
            )?;
            dict.set_item(
                intern!(py, "ready_event_handle"),
                PyBytes::new(py, ready_event_handle),
            )?;
        }
    }

    Ok(dict)
}

/// Encodes the persistent buffer that identifies a KV publication.
fn buffer_id_mapping_to_py<'py>(
    py: Python<'py>,
    buffer: &BufferId,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    let owner = PyDict::new(py);
    owner.set_item("engine_id", buffer.owner.engine_id)?;
    owner.set_item("request_id", buffer.owner.request_id.0)?;
    owner.set_item("request_epoch", buffer.owner.request_epoch)?;
    dict.set_item("owner", owner)?;
    dict.set_item(
        "producer_op_id",
        computation_id_to_py(py, buffer.producer_op_id)?,
    )?;
    dict.set_item("output_index", buffer.output_index)?;
    dict.set_item("generation", buffer.generation)?;
    Ok(dict)
}

/// Converts a product-family transfer handle into its tagged Python mapping.
fn transfer_handle_to_py<'py>(
    py: Python<'py>,
    handle: &TransferHandle,
) -> PyResult<Bound<'py, PyDict>> {
    // Preserve the tagged-union shape expected by the worker: the outer mapping
    // carries the family tag and the inner mapping carries family metadata.
    let dict = PyDict::new(py);
    let value = PyDict::new(py);
    match handle {
        TransferHandle::Encoder {
            height,
            width,
            payload_kind,
            tensor,
        } => {
            dict.set_item(intern!(py, "kind"), "encoder")?;
            value.set_item(intern!(py, "height"), height)?;
            value.set_item(intern!(py, "width"), width)?;
            value.set_item(
                intern!(py, "payload_kind"),
                feature_kind_py(py, *payload_kind),
            )?;
            value.set_item(intern!(py, "tensor"), tensor_transfer_to_py(py, tensor)?)?;
        }
        TransferHandle::DeviceProduct {
            height,
            width,
            value_range,
            tensor,
        } => {
            dict.set_item(intern!(py, "kind"), "device_product")?;
            value.set_item(intern!(py, "height"), height)?;
            value.set_item(intern!(py, "width"), width)?;
            value.set_item(intern!(py, "value_range"), value_range.as_str())?;
            value.set_item(intern!(py, "tensor"), tensor_transfer_to_py(py, tensor)?)?;
        }
        TransferHandle::Latent {
            height,
            width,
            latent_units,
            step,
            tensor,
        } => {
            dict.set_item(intern!(py, "kind"), "latent")?;
            value.set_item(intern!(py, "height"), height)?;
            value.set_item(intern!(py, "width"), width)?;
            value.set_item(intern!(py, "latent_units"), latent_units)?;
            value.set_item(intern!(py, "step"), step)?;
            value.set_item(intern!(py, "tensor"), tensor_transfer_to_py(py, tensor)?)?;
        }
    }

    dict.set_item(intern!(py, "value"), value)?;
    Ok(dict)
}

/// Returns the interned Python spelling for a request kind.
fn request_kind_py<'py>(py: Python<'py>, kind: RequestKind) -> &'py Bound<'py, PyString> {
    match kind {
        RequestKind::Info => intern!(py, "info"),
        RequestKind::Submit => intern!(py, "submit"),
        RequestKind::Poll => intern!(py, "poll"),
        RequestKind::Close => intern!(py, "close"),
    }
}

/// Returns the interned Python spelling for a product family.
fn feature_kind_py<'py>(py: Python<'py>, kind: FeatureKind) -> &'py Bound<'py, PyString> {
    match kind {
        FeatureKind::Vision => intern!(py, "vision_feature"),
        FeatureKind::Latent => intern!(py, "latent_feature"),
    }
}

/// Returns the interned Python spelling for a product storage class.

/// Returns the interned Python spelling for an element type.
fn dtype_py<'py>(py: Python<'py>, dtype: DType) -> &'py Bound<'py, PyString> {
    match dtype {
        DType::U8 => intern!(py, "u8"),
        DType::I32 => intern!(py, "i32"),
        DType::I64 => intern!(py, "i64"),
        DType::F16 => intern!(py, "f16"),
        DType::BF16 => intern!(py, "bf16"),
        DType::F32 => intern!(py, "f32"),
        DType::I16 => intern!(py, "i16"),
    }
}

fn computation_id_to_py(py: Python<'_>, id: ComputationId) -> PyResult<Bound<'_, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "batch_id"), id.batch_id)?;
    dict.set_item(intern!(py, "request_index"), id.request_index)?;
    Ok(dict)
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, SystemTime};

    use pythonize::pythonize;
    use uniserve_core::{BlockId, RequestId, TokenLogprob};
    use uniserve_worker_ipc::{
        BatchOutput, ClientEndpoint, FinishFlags, OpStatus, RegistrationAck, RequestOutput,
        TimingCounters, WorkerResponse,
    };

    use super::*;

    fn kv_publication(source: BufferId) -> KvTransfer {
        let tensors = ["keys", "values"]
            .into_iter()
            .map(|name| TensorTransfer {
                shape: vec![2, 1, 1, 4],
                locations: vec![Locator {
                    source: uniserve_worker_ipc::WorkerInfo::default().endpoint,
                    transport: TransferTransport::PosixShm {
                        endpoint: "publisher".into(),
                        name: name.into(),
                    },
                    nbytes: 16,
                    dtype: "bfloat16".into(),
                    offset: vec![0; 4],
                    shape: vec![2, 1, 1, 4],
                    device: "cpu".into(),
                }],
            })
            .collect();
        KvTransfer {
            tensors,
            source,
            destination: "decoder".into(),
            base: None,
            base_extent: 0,
            published_extent: 2,
            group_id: 0,
            compute_dtype: "bfloat16".into(),
            page_size: 4,
        }
    }

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
        let token = TensorRef {
            request_key,
            producer_op_id: ComputationId::new(11, 0),
            output_index: 0,
            generation: 5,
            dtype: DType::I64,
            shape_bound: ShapeBound::default(),
        };
        let operation = ScheduledRequest {
            token_input: None,

            token_output: Some(token),
            vision_input: None,
            latent_feature_input: None,
            encoder_output: None,
            latent_input: None,
            latent_output: None,
            image_input: None,
            image_output: None,
            completion_output: None,
            transition_output: None,

            input_image: None,
            kv_input: None,
            kv_output: None,
            input_token_ids: vec![7, 8],
            sampling_state: Some(uniserve_worker_ipc::SamplingState {
                allowed_token_ids: Some(Vec::new()),
                suppressed_token_ids: vec![3, 9],
                finish_token_ids: vec![11],
                transition_token_ids: vec![13, 29],
                force_finish: true,
            }),
            request_key,
            op_id: ComputationId::new(11, 0),
            predecessor: Some(ComputationId::new(0, 0)),
            entry: "model".into(),
            code: Computation::Forward(ForwardMode::Prefill),
            bounds: Bounds {
                max_tokens: 2,
                max_kv_pages: 1,
                ..Bounds::default()
            },
            inputs: Vec::new(),
            outputs: Vec::new(),
            predicate: None,
            rng: None,
        };
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
        let forward = uniserve_worker_ipc::ForwardBatch {
            operation_indices: vec![0],
            request_pool_indices: vec![1],
            seq_lens: vec![2],
            query_lens: vec![2],
            write_kv: vec![true],
        };
        let media_key = RequestKey::new(1, RequestId(3), 1);
        let media_prompt_token_ids = vec![17, 23, 65_537];
        let media_admission = NewRequest::new_media(
            media_key,
            2,
            media_prompt_token_ids,
            DiffusionSamplingParams {
                num_frames: 22,
                num_decode_chunks: 3,
                num_inference_steps: 4,
                seed: 29,
            },
        )
        .unwrap();
        let media_operation = ScheduledRequest {
            token_input: None,

            token_output: None,
            vision_input: None,
            latent_feature_input: None,
            encoder_output: None,
            latent_input: None,
            latent_output: None,
            image_input: None,
            image_output: None,
            completion_output: None,
            transition_output: None,

            input_image: None,
            kv_input: None,
            kv_output: None,
            input_token_ids: Vec::new(),
            sampling_state: None,
            request_key: media_key,
            op_id: ComputationId::new(11, 1),
            predecessor: Some(ComputationId::new(0, 0)),
            entry: "model".into(),
            code: Computation::Pipeline(PipelineStage::LatentPreparation),
            bounds: Bounds {
                ..Bounds::default()
            },
            inputs: Vec::new(),
            outputs: Vec::new(),
            predicate: None,
            rng: None,
        };
        let latent_params = vec![LatentParams {
            request_key: media_key,
            op_id: ComputationId::new(11, 1),
            page_table: vec![1],
            latent_units: 64,
            height: 768,
            width: 1344,
            start_step: 0,
            step_count: 0,
        }];
        let kv_key = RequestKey::new(1, RequestId(4), 1);
        let kv_source = BufferId {
            owner: kv_key,
            producer_op_id: ComputationId::new(10, 0),
            output_index: 0,
            generation: 3,
        };
        let kv_operation = ScheduledRequest {
            token_input: None,
            token_output: None,
            vision_input: None,
            latent_feature_input: None,
            encoder_output: None,
            latent_input: None,
            latent_output: None,
            image_input: None,
            image_output: None,
            completion_output: None,
            transition_output: None,

            request_key: kv_key,
            op_id: ComputationId::new(11, 2),
            predecessor: Some(ComputationId::new(0, 0)),
            entry: "decoder".into(),
            code: Computation::Transfer(TransferMode::KvInstall),
            bounds: Bounds {
                max_transfer_bytes: 32,
                ..Bounds::default()
            },
            kv_input: Some(kv_source),
            kv_output: Some(BufferId {
                producer_op_id: ComputationId::new(11, 2),
                generation: 4,
                ..kv_source
            }),
            input_image: None,
            input_token_ids: Vec::new(),
            sampling_state: None,
            inputs: Vec::new(),
            outputs: Vec::new(),
            predicate: None,
            rng: None,
        };
        let mut run = ScheduleBatch::new(
            11,
            vec![admission, media_admission],
            vec![operation, media_operation, kv_operation],
        );
        run.kv_inputs = vec![kv_publication(kv_source)];
        run.block_tables = block_tables;
        run.new_cache_pages = new_cache_pages;
        run.forward = forward;
        run.latent_params = latent_params;
        let mut request = WorkerRequest::submit(run);
        request.set_call_id(Some(9));
        request
    }

    fn result_response() -> WorkerResponse {
        let request_key = RequestKey::new(1, RequestId(2), 1);
        let mut response = WorkerResponse::result(BatchOutput {
            batch_id: 11,
            run_id: 11,
            completions: vec![RequestOutput {
                sampled_logprob: Some(-0.25),
                top_logprobs: vec![TokenLogprob {
                    token_id: 42,
                    logprob: -0.25,
                    rank: 1,
                }],
                prompt_logprobs: vec![vec![TokenLogprob {
                    token_id: 7,
                    logprob: f32::NEG_INFINITY,
                    rank: 19,
                }]],
                request_key,
                op_id: ComputationId::new(11, 0),

                status: OpStatus::Ok,

                product_generations: vec![5],
                error_code: None,
                timing_counters: TimingCounters::default(),
                code: Computation::Forward(ForwardMode::Decode),
                position: 2,
                kv_visible_len: 2,
                num_completed_steps: 0,
                kv_computed_len: 2,

                committed_tokens: vec![42],
                finish_flags: FinishFlags::default(),
                media_output: None,
                kv_output: None,
            }],
            products: Vec::new(),
            registration: RegistrationAck { visible: true },
            worker_exec_us: Some(12),
            forward_stats: None,
            done: true,
        });
        let WorkerResponse::Result { result, .. } = &mut response else {
            unreachable!();
        };
        let mut publication = result.completions[0].clone();
        publication.request_key = RequestKey::new(1, RequestId(4), 1);
        publication.op_id = ComputationId::new(11, 2);
        publication.code = Computation::Transfer(TransferMode::KvPublish);
        publication.committed_tokens.clear();

        publication.sampled_logprob = None;
        publication.top_logprobs.clear();
        publication.prompt_logprobs.clear();
        publication.product_generations.clear();
        publication.kv_output = Some(kv_publication(BufferId {
            owner: publication.request_key,
            producer_op_id: publication.op_id,
            output_index: 0,
            generation: 4,
        }));
        result.completions.push(publication);
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
            assert_eq!(native_run.getattr("operations").unwrap().len().unwrap(), 3);
            assert_eq!(
                native_run
                    .getattr("operations")
                    .unwrap()
                    .get_item(0)
                    .unwrap()
                    .getattr("input_token_ids")
                    .unwrap()
                    .extract::<Vec<u32>>()
                    .unwrap(),
                vec![7, 8]
            );
            let sampling = native_run
                .getattr("operations")
                .unwrap()
                .get_item(0)
                .unwrap()
                .getattr("sampling_state")
                .unwrap();
            assert_eq!(
                sampling
                    .getattr("allowed_token_ids")
                    .unwrap()
                    .extract::<Vec<u32>>()
                    .unwrap(),
                Vec::<u32>::new()
            );
            assert_eq!(
                sampling
                    .getattr("suppressed_token_ids")
                    .unwrap()
                    .extract::<Vec<u32>>()
                    .unwrap(),
                vec![3, 9]
            );
            assert_eq!(
                sampling
                    .getattr("finish_token_ids")
                    .unwrap()
                    .extract::<Vec<u32>>()
                    .unwrap(),
                vec![11]
            );
            assert_eq!(
                sampling
                    .getattr("transition_token_ids")
                    .unwrap()
                    .extract::<Vec<u32>>()
                    .unwrap(),
                vec![13, 29]
            );
            assert!(
                sampling
                    .getattr("force_finish")
                    .unwrap()
                    .extract::<bool>()
                    .unwrap()
            );
            let imported_kv = native_run
                .getattr("kv_inputs")
                .unwrap()
                .get_item(0)
                .unwrap();
            let imported_mapping = imported_kv.call_method0("to_mapping").unwrap();
            let WorkerRequest::Submit { run, .. } = &request else {
                unreachable!();
            };
            let expected_source = pythonize(py, &run.kv_inputs[0].source).unwrap();
            assert!(
                imported_kv
                    .getattr("source")
                    .unwrap()
                    .call_method0("to_mapping")
                    .unwrap()
                    .eq(expected_source)
                    .unwrap()
            );
            let kv_operation = native_run
                .getattr("operations")
                .unwrap()
                .get_item(2)
                .unwrap();
            assert!(
                kv_operation
                    .getattr("kv_input")
                    .unwrap()
                    .eq(imported_kv.getattr("source").unwrap())
                    .unwrap()
            );
            let admissions = native_run.getattr("admissions").unwrap();
            let media = admissions.get_item(1).unwrap();
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
                .import("uniserve_worker.protocol.batch")
                .unwrap()
                .getattr("ScheduleBatch")
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
            // Relay the imported physical metadata through the public result
            // mapping. The final Rust equality checks every location and extent
            // after both native directions, not just Python serialization itself.
            let publication = response
                .get_item("result")
                .unwrap()
                .get_item("completions")
                .unwrap()
                .get_item(1)
                .unwrap();
            let output_source = publication
                .get_item("kv_output")
                .unwrap()
                .get_item("source")
                .unwrap();
            imported_mapping.set_item("source", output_source).unwrap();
            publication.set_item("kv_output", imported_mapping).unwrap();
            server.respond(py, &response).unwrap();
        });

        let response = client
            .recv_response_timeout(&pending, Duration::from_secs(5))
            .unwrap()
            .expect("native result response");
        assert_eq!(response.decode_response().unwrap(), expected);
    }
}

fn tensor_transfer_to_py<'py>(
    py: Python<'py>,
    tensor: &TensorTransfer,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("shape", PyList::new(py, &tensor.shape)?)?;
    let locations = tensor
        .locations
        .iter()
        .map(|location| transfer_locator_to_py(py, location))
        .collect::<PyResult<Vec<_>>>()?;
    dict.set_item("locations", PyList::new(py, locations)?)?;
    Ok(dict)
}

fn kv_transfer_to_py<'py>(py: Python<'py>, transfer: &KvTransfer) -> PyResult<Bound<'py, PyDict>> {
    let KvTransfer {
        tensors,
        source,
        destination,
        base,
        base_extent,
        published_extent,
        group_id,
        compute_dtype,
        page_size,
    } = transfer;
    let value = PyDict::new(py);
    // KV publications may contain multiple physical tensors but share
    // one published buffer and destination contract.
    let tensors = tensors
        .iter()
        .map(|tensor| tensor_transfer_to_py(py, tensor))
        .collect::<PyResult<Vec<_>>>()?;
    value.set_item(intern!(py, "tensors"), PyList::new(py, tensors)?)?;
    value.set_item(intern!(py, "source"), buffer_id_mapping_to_py(py, source)?)?;
    value.set_item(intern!(py, "destination"), destination.as_str())?;
    value.set_item(
        intern!(py, "base"),
        base.as_ref()
            .map(|buffer| buffer_id_mapping_to_py(py, buffer))
            .transpose()?,
    )?;
    value.set_item(intern!(py, "base_extent"), base_extent)?;
    value.set_item(intern!(py, "published_extent"), published_extent)?;
    value.set_item(intern!(py, "group_id"), group_id)?;
    value.set_item(intern!(py, "compute_dtype"), compute_dtype.as_str())?;
    value.set_item(intern!(py, "page_size"), page_size)?;
    Ok(value)
}
