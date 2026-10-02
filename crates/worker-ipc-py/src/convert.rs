//! Typed conversion between worker IPC messages and the Python worker's
//! protocol records.
//!
//! This module converts submit requests into Python records and decodes
//! result and error responses from Python, each in one call per message.
//! Every other message kind uses the schema-derived serde representation
//! (`pythonize_request` and `PyServer::respond` in the crate root choose the
//! path).
//!
//! - Inbound, [`execute_request_to_py`] turns a decoded submit request into
//!   `{kind, message_id, batch}`, where `batch` is a fully constructed
//!   `uniserve_worker.protocol.batch.Batch`. Python classes and enum members
//!   are resolved once per process (`RequestTypes`); outside `kv_inputs`,
//!   each distinct request key, call id, and shape bound is constructed once
//!   per batch (`RequestConversion`); and `batch_from_validated` assembles the
//!   batch without running `Batch.__post_init__`.
//! - Outbound, [`try_completion_response_from_py`] decodes the mappings the
//!   worker's `to_mapping` methods emit. That encoding differs from the serde
//!   form: for example `Locator.to_mapping` writes a plain `transport` string
//!   beside flattened transport fields, whereas `TransferTransport`
//!   deserializes only from its adjacently tagged `transport`/`value` form.
//!
//! The encoder constructs many records positionally, so its argument order is
//! coupled to the field order of the Python dataclasses in
//! `uniserve_worker.protocol` and to the parameters of `batch_from_validated`.
//! The round-trip test at the bottom of this file checks that each native
//! batch equals `Batch.from_mapping(batch.to_mapping())`.
//!
//! Decoding is strict: integers reject Python `bool`, strings must be `str`,
//! byte payloads must be `bytes`, and sequences must be `list` (except the
//! logprob iterables). Any mismatch rejects the whole response with one
//! generic `ValueError` that does not name the offending field.

use std::collections::{BTreeMap, HashMap};
use uniserve_worker_ipc::{ForwardMode, MediaCall, TransferMode};

use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyList, PyString, PyTuple};
use uniserve_core::{
    AudioClip, Canvas, ConditionMedia, ConditionRole, ImageParams, SamplingParams, TokenLogprob,
    VideoCondition,
};
use uniserve_worker_ipc::{
    ArRequestParams, ArtifactHandle, Batch, BatchCommand, BatchOutput, BlockTable,
    BufferAllocation, BufferId, CachePageAllocation, Call, CallId, CallKind, CallStatus, DType,
    DecodeRange, DiffusionSamplingParams, DimBound, DrawLayout, ErrorCallIdentity, ErrorCode,
    FeatureKind, FinishFlags, ForwardStats, KvTransfer, LatentParams, Locator, MediaOutput,
    NewRequest, RequestKey, RequestKind, RequestOutput, ShapeBound, TensorPublication, TensorRef,
    TensorTransfer, TimingCounters, TransferHandle, TransferTransport, VideoAdmission,
    WorkerEndpoint, WorkerRequest, WorkerResponse, WorkerResponseError,
};

#[cfg(test)]
use uniserve_worker_ipc::Bounds;

/// Converts a submit [`WorkerRequest`] into the Python worker mapping.
///
/// Returns a `dict` with `kind`, `message_id`, and a typed `batch`. Fails with
/// `ValueError` for any other request kind, and propagates errors from
/// resolving the Python types and exceptions raised by record constructors.
/// Nested records that define `__post_init__` still run it; only `Batch`
/// itself skips it.
pub(crate) fn execute_request_to_py<'py>(
    py: Python<'py>,
    request: &WorkerRequest,
) -> PyResult<Bound<'py, PyDict>> {
    let WorkerRequest::Submit { message_id, batch } = request else {
        return Err(PyValueError::new_err(
            "native submit conversion requires a submit request",
        ));
    };
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "kind"), request_kind_py(py, request.kind()))?;
    dict.set_item(intern!(py, "message_id"), message_id)?;
    dict.set_item(intern!(py, "batch"), batch_to_py(py, batch)?)?;
    Ok(dict)
}

/// Cached constructors and enum members for transport-validated records.
///
/// `records` holds the classes built through [`construct`] by keyword; the
/// classes and the function in the other fields are called positionally. The
/// enum arrays are indexed by `RequestTypes::kind`, `RequestTypes::dtype`,
/// and the draw-layout match in `RequestConversion::call`.
struct RequestTypes {
    records: HashMap<&'static str, Py<PyAny>>,
    call: Py<PyAny>,
    computation_id: Py<PyAny>,
    request_key: Py<PyAny>,
    tensor_ref: Py<PyAny>,
    buffer_id: Py<PyAny>,
    shape_bound: Py<PyAny>,
    static_dim: Py<PyAny>,
    device_dim: Py<PyAny>,
    bounds: Py<PyAny>,
    call_coordinates: Py<PyAny>,
    rng: Py<PyAny>,
    sampling_state: Py<PyAny>,
    block_table: Py<PyAny>,
    cache_page_allocation: Py<PyAny>,
    start: Py<PyAny>,
    finish: Py<PyAny>,
    free: Py<PyAny>,
    batch_from_validated: Py<PyAny>,
    dtypes: [Py<PyAny>; 7],
    draw_layouts: [Py<PyAny>; 3],
    forward_modes: [Py<PyAny>; ForwardMode::ALL.len()],
    media_calls: [Py<PyAny>; MediaCall::ALL.len()],
    transfer_modes: [Py<PyAny>; TransferMode::ALL.len()],
}

/// Process-wide cache of worker Python constructors and enum members.
static REQUEST_TYPES: std::sync::OnceLock<RequestTypes> = std::sync::OnceLock::new();

/// Resolves named Python enum members into a fixed-size indexed cache.
///
/// Members are looked up by value (`Enum(value)`), so each spelling must be a
/// value of the Python enum `name`, and the array order is the order of
/// `values`.
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

impl RequestTypes {
    /// Imports worker model types and caches every constructor and enum member.
    fn build(py: Python<'_>) -> PyResult<Self> {
        // Resolve classes once so per-request conversion uses direct constructor
        // calls without repeated module or attribute lookup. Each name is
        // imported from the module that defines it.
        let batch = py.import("uniserve_worker.protocol.batch")?;
        let call = py.import("uniserve_worker.protocol.call")?;
        let identity = py.import("uniserve_worker.protocol.identity")?;
        let tensor = py.import("uniserve_worker.protocol.tensor")?;
        let class = |module: &Bound<'_, PyModule>, name: &str| -> PyResult<Py<PyAny>> {
            Ok(module.getattr(name)?.unbind())
        };

        let mut records = HashMap::new();
        let module = py.import("uniserve_worker.protocol.batch")?;
        records.insert("LatentParams", class(&module, "LatentParams")?);
        records.insert("DecodeRange", class(&module, "DecodeRange")?);
        records.insert("BufferAllocation", class(&module, "BufferAllocation")?);
        records.insert("NewRequest", class(&module, "NewRequest")?);
        records.insert("GenerationParams", class(&module, "GenerationParams")?);
        records.insert("DiffusionParams", class(&module, "DiffusionParams")?);
        records.insert("TensorPublication", class(&module, "TensorPublication")?);
        let module = py.import("uniserve_worker.protocol.video")?;
        for name in [
            "VideoAdmission",
            "VideoCondition",
            "MediaLocator",
            "ImageFit",
            "VideoClip",
            "AudioClip",
            "ConditionVision",
        ] {
            records.insert(name, class(&module, name)?);
        }
        let module = py.import("uniserve.media.image")?;
        records.insert("Raster", class(&module, "Config")?);
        let module = py.import("uniserve_worker.protocol.call")?;
        records.insert("ImageParams", class(&module, "ImageParams")?);
        let module = py.import("uniserve.sampling")?;
        records.insert("SamplingParams", class(&module, "SamplingParams")?);
        let module = py.import("uniserve_worker.protocol.transfer")?;
        records.insert("WorkerEndpoint", class(&module, "WorkerEndpoint")?);
        records.insert("Locator", class(&module, "Locator")?);
        records.insert("LocalTransfer", class(&module, "LocalTransfer")?);
        records.insert("PosixShmTransfer", class(&module, "PosixShmTransfer")?);
        records.insert("CudaVmmTransfer", class(&module, "CudaVmmTransfer")?);
        records.insert("ChannelTransfer", class(&module, "ChannelTransfer")?);
        records.insert(
            "EncoderTransferValue",
            class(&module, "EncoderTransferValue")?,
        );
        records.insert(
            "DeviceProductTransferValue",
            class(&module, "DeviceProductTransferValue")?,
        );
        records.insert(
            "LatentTransferValue",
            class(&module, "LatentTransferValue")?,
        );
        records.insert("TensorTransfer", class(&module, "TensorTransfer")?);
        records.insert("KvTransfer", class(&module, "KvTransfer")?);
        Ok(Self {
            records,
            call: class(&call, "Call")?,
            computation_id: class(&identity, "CallId")?,
            request_key: class(&identity, "RequestKey")?,
            tensor_ref: class(&tensor, "TensorRef")?,
            buffer_id: class(&identity, "BufferId")?,
            shape_bound: class(&tensor, "ShapeBound")?,
            static_dim: class(&tensor, "StaticDim")?,
            device_dim: class(&tensor, "DeviceDim")?,
            bounds: class(&call, "Bounds")?,
            call_coordinates: class(&call, "CallCoordinates")?,
            rng: class(&call, "Rng")?,
            sampling_state: class(&call, "SamplingState")?,
            block_table: class(&batch, "BlockTable")?,
            cache_page_allocation: class(&batch, "CachePageAllocation")?,
            start: class(&batch, "Start")?,
            finish: class(&batch, "Finish")?,
            free: class(&batch, "Free")?,
            batch_from_validated: class(
                &py.import("uniserve_worker.protocol.construction")?,
                "batch_from_validated",
            )?,

            // The `dtypes` and `draw_layouts` spellings sit at the indices that
            // `dtype` and the draw-layout match in `RequestConversion::call`
            // hardcode (the Rust discriminant values). `kind` indexes the other
            // three with `as usize`, which relies on each `ALL` array listing
            // variants in declaration order.
            dtypes: enum_members(
                &tensor,
                "DType",
                ["u8", "i32", "i64", "f16", "bf16", "f32", "i16"],
            )?,
            draw_layouts: enum_members(
                &call,
                "DrawLayout",
                ["target_sampling", "speculative_proposal", "flow_noise"],
            )?,
            forward_modes: enum_members(
                &call,
                "ForwardMode",
                ForwardMode::ALL.map(ForwardMode::as_str),
            )?,
            media_calls: enum_members(&call, "MediaCall", MediaCall::ALL.map(MediaCall::as_str))?,
            transfer_modes: enum_members(
                &call,
                "TransferMode",
                TransferMode::ALL.map(TransferMode::as_str),
            )?,
        })
    }

    /// Returns the process-wide type cache, building it on first use.
    ///
    /// A failed build is not cached, so a later call retries the imports.
    fn get(py: Python<'_>) -> PyResult<&'static Self> {
        if let Some(types) = REQUEST_TYPES.get() {
            return Ok(types);
        }
        // `build` is fallible and runs outside `get_or_init`, so concurrent
        // first callers may each build a cache; the first one stored wins and
        // the others are dropped.
        let built = Self::build(py)?;
        Ok(REQUEST_TYPES.get_or_init(|| built))
    }

    /// Returns the Python `ForwardMode`, `MediaCall`, or `TransferMode` member
    /// for a call kind.
    fn kind<'py>(&self, py: Python<'py>, kind: CallKind) -> Bound<'py, PyAny> {
        let member = match kind {
            CallKind::Forward(mode) => &self.forward_modes[mode as usize],
            CallKind::Media(call) => &self.media_calls[call as usize],
            CallKind::Transfer(mode) => &self.transfer_modes[mode as usize],
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

/// Constructs the record class registered as `name` with `fields` as keyword
/// arguments.
///
/// The dictionary keys must match the class's field names. Panics if `name`
/// was not registered in `RequestTypes::build`; every caller passes a name
/// registered there.
fn construct<'py>(
    py: Python<'py>,
    name: &str,
    fields: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    RequestTypes::get(py)?.records[name]
        .bind(py)
        .call((), Some(fields))
}

/// Per-batch construction context: repeated typed leaves are built once.
///
/// Equal request keys, call ids, and shape bounds converted through one
/// context map to the same Python object. Sharing is safe because those
/// records are frozen dataclasses. Tensor references and buffer ids are constructed anew on each
/// use from the cached leaves.
struct RequestConversion<'py> {
    py: Python<'py>,
    types: &'static RequestTypes,
    request_keys: HashMap<RequestKey, Py<PyAny>>,
    computation_ids: HashMap<CallId, Py<PyAny>>,
    shape_bounds: HashMap<ShapeBound, Py<PyAny>>,
}

impl<'py> RequestConversion<'py> {
    /// Starts a batch conversion with shared Python types and empty value caches.
    fn new(py: Python<'py>) -> PyResult<Self> {
        Ok(Self {
            py,
            types: RequestTypes::get(py)?,
            request_keys: HashMap::new(),
            computation_ids: HashMap::new(),
            shape_bounds: HashMap::new(),
        })
    }

    /// Returns the batch-canonical Python `CallId` for `id`.
    ///
    /// Call ids repeat across a batch, for example as call identities, as the
    /// `call_id` of latent and decode params, and as the producers named by
    /// tensor references and buffer ids.
    fn computation_id(&mut self, id: CallId) -> PyResult<Bound<'py, PyAny>> {
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
            self.computation_id(product.producer_call_id)?,
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
            self.computation_id(buffer.producer_call_id)?,
            buffer.output_index,
            buffer.generation,
        ))
    }

    /// Constructs a typed Python call from its computation fields.
    ///
    /// `Call` is constructed positionally: the argument tuple below follows
    /// the field order of the `Call` dataclass in
    /// `uniserve_worker.protocol.call`, and `Call` has no `__post_init__` to
    /// reject a misordered argument. Reordering fields on either side requires
    /// updating the other.
    fn call(&mut self, call: &Call) -> PyResult<Bound<'py, PyAny>> {
        let request_key = self.request_key(call.request_key)?;
        let coordinates = self.types.call_coordinates.bind(self.py).call1((
            call.coordinates.logical_position,
            call.coordinates.kv_visible_len,
            call.coordinates.kv_computed_len,
            call.coordinates.flow_step,
        ))?;
        let bounds = self.types.bounds.bind(self.py).call1((
            call.bounds.max_tokens,
            call.bounds.max_kv_pages,
            call.bounds.max_latent_bytes,
            call.bounds.max_completion_bytes,
            call.bounds.max_transfer_bytes,
        ))?;
        let inputs = call
            .inputs
            .as_slice()
            .iter()
            .map(|product| self.tensor_ref(product))
            .collect::<PyResult<Vec<_>>>()?;
        let outputs = call
            .outputs
            .as_slice()
            .iter()
            .map(|product| self.tensor_ref(product))
            .collect::<PyResult<Vec<_>>>()?;
        let predicate = call
            .predicate
            .as_ref()
            .as_ref()
            .map(|predicate| self.tensor_ref(predicate))
            .transpose()?;
        let rng = call
            .rng
            .as_ref()
            .map(|rng| {
                // Indices follow the `draw_layouts` order in `RequestTypes::build`.
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
        let sampling_state = call
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
                self.computation_id(call.call_id)?,
                coordinates.into_any(),
                self.types.kind(py, call.code),
                bounds.into_any(),
                call.component.clone().into_pyobject(py)?.into_any(),
                pyo3::types::PyTuple::new(py, inputs)?.into_any(),
                pyo3::types::PyTuple::new(py, outputs)?.into_any(),
                call.token_input
                    .as_ref()
                    .map(|tensor| self.tensor_ref(tensor))
                    .transpose()?
                    .unwrap_or_else(|| py.None().into_bound(py)),
                call.token_output
                    .as_ref()
                    .map(|tensor| self.tensor_ref(tensor))
                    .transpose()?
                    .unwrap_or_else(|| py.None().into_bound(py)),
                call.vision_input
                    .as_ref()
                    .map(|tensor| self.tensor_ref(tensor))
                    .transpose()?
                    .unwrap_or_else(|| py.None().into_bound(py)),
                call.latent_feature_input
                    .as_ref()
                    .map(|tensor| self.tensor_ref(tensor))
                    .transpose()?
                    .unwrap_or_else(|| py.None().into_bound(py)),
                call.encoder_output
                    .as_ref()
                    .map(|tensor| self.tensor_ref(tensor))
                    .transpose()?
                    .unwrap_or_else(|| py.None().into_bound(py)),
                call.latent_input
                    .as_ref()
                    .map(|tensor| self.tensor_ref(tensor))
                    .transpose()?
                    .unwrap_or_else(|| py.None().into_bound(py)),
                call.latent_output
                    .as_ref()
                    .map(|tensor| self.tensor_ref(tensor))
                    .transpose()?
                    .unwrap_or_else(|| py.None().into_bound(py)),
                call.image_input
                    .as_ref()
                    .map(|tensor| self.tensor_ref(tensor))
                    .transpose()?
                    .unwrap_or_else(|| py.None().into_bound(py)),
                call.image_output
                    .as_ref()
                    .map(|tensor| self.tensor_ref(tensor))
                    .transpose()?
                    .unwrap_or_else(|| py.None().into_bound(py)),
                call.completion_output
                    .as_ref()
                    .map(|tensor| self.tensor_ref(tensor))
                    .transpose()?
                    .unwrap_or_else(|| py.None().into_bound(py)),
                call.transition_output
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
                pyo3::types::PyTuple::new(py, &call.input_token_ids)?.into_any(),
                call.input_image.as_deref().into_pyobject(py)?.into_any(),
                call.kv_input
                    .map(|buffer| self.buffer_id(buffer))
                    .transpose()?
                    .unwrap_or_else(|| py.None().into_bound(py)),
                call.kv_output
                    .map(|buffer| self.buffer_id(buffer))
                    .transpose()?
                    .unwrap_or_else(|| py.None().into_bound(py)),
                pyo3::types::PyTuple::new(py, &call.consumer_slots)?.into_any(),
            ],
        )?;
        self.types.call.bind(py).call1(arguments)
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
        match command {
            BatchCommand::Start { request } => {
                let request = admission_to_py(self.py, request, self)?;
                self.types.start.bind(self.py).call1((request,))
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

/// Constructs a fully typed Python batch from the validated wire record.
///
/// One `RequestConversion` spans the whole batch, so identity leaves are
/// shared between calls, commands, input products, and the per-batch
/// parameter records. `kv_inputs` are the exception: `kv_transfer_to_py`
/// builds their buffer ids through `buffer_id_to_py`.
fn batch_to_py<'py>(py: Python<'py>, run: &Batch) -> PyResult<Bound<'py, PyAny>> {
    let mut native = RequestConversion::new(py)?;

    let calls = run
        .calls
        .iter()
        .map(|call| native.call(call))
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

    let input_products = record_tuple(py, &run.input_products, |payload| {
        tensor_publication_to_py(py, payload, &mut native)
    })?;

    // `batch_from_validated` sets these fields on a bare `Batch` without
    // running `Batch.__post_init__`, because the frame was validated when it
    // was decoded. Arguments are positional and follow its parameter order;
    // the forward tuple is unpacked by index into `forward_call_indices`,
    // `request_pool_indices`, `seq_lens`, `query_lens`, and `write_kv`.
    let arguments = pyo3::types::PyTuple::new(
        py,
        [
            run.batch_id.into_pyobject(py)?.into_any(),
            run.collective_seq.into_pyobject(py)?.into_any(),
            pyo3::types::PyTuple::new(py, calls)?.into_any(),
            pyo3::types::PyTuple::new(py, block_tables)?.into_any(),
            pyo3::types::PyTuple::new(py, new_cache_pages)?.into_any(),
            (
                pyo3::types::PyTuple::new(py, &run.forward.call_indices)?,
                pyo3::types::PyTuple::new(py, &run.forward.request_pool_indices)?,
                pyo3::types::PyTuple::new(py, &run.forward.seq_lens)?,
                pyo3::types::PyTuple::new(py, &run.forward.query_lens)?,
                pyo3::types::PyTuple::new(py, &run.forward.write_kv)?,
            )
                .into_pyobject(py)?
                .into_any(),
            record_tuple(py, &run.latent_params, |params| {
                latent_params_to_py(py, params, &mut native)
            })?
            .into_any(),
            record_tuple(py, &run.decode_ranges, |params| {
                decode_range_to_py(py, params, &mut native)
            })?
            .into_any(),
            record_tuple(py, &run.buffer_allocations, |params| {
                buffer_allocation_to_py(py, params, &mut native)
            })?
            .into_any(),
            pyo3::types::PyTuple::new(py, commands)?.into_any(),
            input_products.into_any(),
            record_tuple(py, &run.kv_inputs, |transfer| {
                kv_transfer_to_py(py, transfer)
            })?
            .into_any(),
        ],
    )?;
    native.types.batch_from_validated.bind(py).call1(arguments)
}

/// Converts one trajectory's solver-step range and latent page table into a
/// Python `LatentParams` record.
fn latent_params_to_py<'py>(
    py: Python<'py>,
    params: &LatentParams,
    context: &mut RequestConversion<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item(
        intern!(py, "request_key"),
        context.request_key(params.request_key)?,
    )?;
    dict.set_item(
        intern!(py, "call_id"),
        context.computation_id(params.call_id)?,
    )?;
    dict.set_item(
        intern!(py, "page_table"),
        u32_tuple(py, &params.page_table)?,
    )?;
    dict.set_item(intern!(py, "latent_units"), params.latent_units)?;
    dict.set_item(intern!(py, "height"), params.height)?;
    dict.set_item(intern!(py, "width"), params.width)?;
    dict.set_item(intern!(py, "start_step"), params.start_step)?;
    dict.set_item(intern!(py, "step_count"), params.step_count)?;
    construct(py, "LatentParams", &dict)
}

/// Converts the cursor and unit bound of one diffusion decode call into a
/// Python `DecodeRange` record.
fn decode_range_to_py<'py>(
    py: Python<'py>,
    params: &DecodeRange,
    context: &mut RequestConversion<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item(
        intern!(py, "request_key"),
        context.request_key(params.request_key)?,
    )?;
    dict.set_item(
        intern!(py, "call_id"),
        context.computation_id(params.call_id)?,
    )?;
    dict.set_item(intern!(py, "cursor"), params.cursor)?;
    dict.set_item(intern!(py, "max_units"), params.max_units)?;
    construct(py, "DecodeRange", &dict)
}

/// Converts a persistent-buffer byte span into its Python allocation record.
fn buffer_allocation_to_py<'py>(
    py: Python<'py>,
    params: &BufferAllocation,
    context: &mut RequestConversion<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("buffer", context.buffer_id(params.buffer)?)?;
    dict.set_item(intern!(py, "offset"), params.offset)?;
    dict.set_item(intern!(py, "bytes"), params.bytes)?;
    construct(py, "BufferAllocation", &dict)
}

/// Converts a Rust slice into a tuple of Python records.
fn record_tuple<'py, T, F>(
    py: Python<'py>,
    items: &[T],
    mut convert: F,
) -> PyResult<Bound<'py, PyTuple>>
where
    F: FnMut(&T) -> PyResult<Bound<'py, PyAny>>,
{
    let converted = items
        .iter()
        .map(&mut convert)
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, converted)
}

/// Copies `u32` values into a Python tuple.
fn u32_tuple<'py>(py: Python<'py>, values: &[u32]) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(py, values.iter().copied())
}

/// Converts a request admission and its selected parameter family.
fn admission_to_py<'py>(
    py: Python<'py>,
    admission: &NewRequest,
    context: &mut RequestConversion<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item(
        intern!(py, "prompt_token_ids"),
        u32_tuple(py, &admission.prompt_token_ids)?,
    )?;
    dict.set_item(
        intern!(py, "request_key"),
        context.request_key(admission.request_key)?,
    )?;
    dict.set_item(intern!(py, "request_pool_idx"), admission.request_pool_idx)?;
    dict.set_item(intern!(py, "input_images"), admission.input_images)?;
    dict.set_item(
        intern!(py, "generation"),
        admission
            .ar
            .as_ref()
            .map(|ar| ar_params_to_py(py, ar))
            .transpose()?,
    )?;
    dict.set_item(
        intern!(py, "image"),
        admission
            .image
            .as_ref()
            .map(|branch| image_to_py(py, branch))
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
    dict.set_item(
        intern!(py, "video"),
        admission
            .video
            .as_ref()
            .map(|video| video_admission_to_py(py, video))
            .transpose()?,
    )?;
    construct(py, "NewRequest", &dict)
}

/// Converts a video request's task, presentation tags and conditions.
fn video_admission_to_py<'py>(
    py: Python<'py>,
    video: &VideoAdmission,
) -> PyResult<Bound<'py, PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "task"), video.task.as_str())?;
    dict.set_item(
        intern!(py, "text_tags"),
        PyTuple::new(py, video.text_tags.iter().copied())?,
    )?;
    let conditions = video
        .conditions
        .iter()
        .map(|condition| video_condition_to_py(py, condition))
        .collect::<PyResult<Vec<_>>>()?;
    dict.set_item(intern!(py, "conditions"), PyTuple::new(py, conditions)?)?;
    construct(py, "VideoAdmission", &dict)
}

/// Converts a raster into the library's `image.Config`.
fn raster_to_py<'py>(py: Python<'py>, canvas: Canvas) -> PyResult<Bound<'py, PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "height"), canvas.height)?;
    dict.set_item(intern!(py, "width"), canvas.width)?;
    construct(py, "Raster", &dict)
}

fn audio_clip_to_py<'py>(py: Python<'py>, clip: &AudioClip) -> PyResult<Bound<'py, PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "sample_rate"), clip.sample_rate)?;
    dict.set_item(intern!(py, "start_sample"), clip.start_sample)?;
    dict.set_item(intern!(py, "source_samples"), clip.source_samples)?;
    dict.set_item(intern!(py, "samples"), clip.samples)?;
    construct(py, "AudioClip", &dict)
}

/// Converts one condition. Its media becomes the worker record's `image`,
/// `video` and `audio` fields: an image alone, a video with its optional
/// soundtrack, or audio alone.
fn video_condition_to_py<'py>(
    py: Python<'py>,
    condition: &VideoCondition,
) -> PyResult<Bound<'py, PyAny>> {
    let dict = PyDict::new(py);
    let role = match condition.role {
        ConditionRole::FirstFrame => "first_frame",
        ConditionRole::LastFrame => "last_frame",
        ConditionRole::Reference => "reference",
    };
    dict.set_item(intern!(py, "role"), role)?;

    let source = PyDict::new(py);
    source.set_item(intern!(py, "name"), &condition.source.name)?;
    source.set_item(intern!(py, "bytes"), condition.source.bytes)?;
    dict.set_item(
        intern!(py, "source"),
        construct(py, "MediaLocator", &source)?,
    )?;

    let (image, video, audio) = match &condition.media {
        ConditionMedia::Image(fit) => {
            let fields = PyDict::new(py);
            fields.set_item(intern!(py, "resized"), raster_to_py(py, fit.resized)?)?;
            fields.set_item(intern!(py, "left"), fit.left)?;
            fields.set_item(intern!(py, "top"), fit.top)?;
            fields.set_item(intern!(py, "size"), raster_to_py(py, fit.size)?)?;
            (Some(construct(py, "ImageFit", &fields)?), None, None)
        }
        ConditionMedia::Video { clip, soundtrack } => {
            let fields = PyDict::new(py);
            fields.set_item(intern!(py, "canvas"), raster_to_py(py, clip.canvas)?)?;
            fields.set_item(intern!(py, "start_frame"), clip.start_frame)?;
            fields.set_item(intern!(py, "frames"), clip.frames)?;
            fields.set_item(intern!(py, "vae_frames"), clip.vae_frames)?;
            (
                None,
                Some(construct(py, "VideoClip", &fields)?),
                soundtrack
                    .as_ref()
                    .map(|track| audio_clip_to_py(py, track))
                    .transpose()?,
            )
        }
        ConditionMedia::Audio(clip) => (None, None, Some(audio_clip_to_py(py, clip)?)),
    };
    dict.set_item(intern!(py, "image"), image)?;
    dict.set_item(intern!(py, "video"), video)?;
    dict.set_item(intern!(py, "audio"), audio)?;

    let vision = condition
        .vision
        .as_ref()
        .map(|vision| -> PyResult<Bound<'py, PyAny>> {
            let fields = PyDict::new(py);
            fields.set_item(
                intern!(py, "grid"),
                PyTuple::new(py, [vision.grid.t, vision.grid.h, vision.grid.w])?,
            )?;
            fields.set_item(intern!(py, "tokens"), vision.tokens)?;
            fields.set_item(
                intern!(py, "frame_indices"),
                u32_tuple(py, &vision.frame_indices)?,
            )?;
            construct(py, "ConditionVision", &fields)
        })
        .transpose()?;
    dict.set_item(intern!(py, "vision"), vision)?;
    dict.set_item(
        intern!(py, "latent_units"),
        u32_tuple(py, &condition.latent_units)?,
    )?;
    dict.set_item(intern!(py, "audio_rows"), condition.audio_rows)?;
    construct(py, "VideoCondition", &dict)
}

/// Converts autoregressive admission parameters into a Python record.
fn ar_params_to_py<'py>(py: Python<'py>, ar: &ArRequestParams) -> PyResult<Bound<'py, PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "sampling"), sampling_to_py(py, &ar.sampling)?)?;
    dict.set_item(
        intern!(py, "negative_token_ids"),
        u32_tuple(py, &ar.negative_token_ids)?,
    )?;
    dict.set_item(
        intern!(py, "finish_token_ids"),
        u32_tuple(py, &ar.finish_token_ids)?,
    )?;
    dict.set_item(intern!(py, "initial_position"), ar.initial_position)?;
    construct(py, "GenerationParams", &dict)
}

/// Converts diffusion admission parameters and resolved media geometry.
fn diffusion_params_to_py<'py>(
    py: Python<'py>,
    diffusion: &DiffusionSamplingParams,
) -> PyResult<Bound<'py, PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "num_frames"), diffusion.num_frames)?;
    dict.set_item(intern!(py, "video_units"), diffusion.video_units)?;
    dict.set_item(
        intern!(py, "num_inference_steps"),
        diffusion.num_inference_steps,
    )?;
    dict.set_item(intern!(py, "seed"), diffusion.seed)?;
    dict.set_item(intern!(py, "width"), diffusion.width)?;
    dict.set_item(intern!(py, "height"), diffusion.height)?;
    construct(py, "DiffusionParams", &dict)
}

/// Converts sampling controls into the worker's typed parameters.
fn sampling_to_py<'py>(py: Python<'py>, sampling: &SamplingParams) -> PyResult<Bound<'py, PyAny>> {
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

    // Python `SamplingParams.logit_bias` is a tuple of `(token_id, bias)`
    // pairs rather than a mapping.
    dict.set_item(
        intern!(py, "logit_bias"),
        PyTuple::new(
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
        u32_tuple(py, &sampling.logprob_token_ids)?,
    )?;

    let bad_words = sampling
        .bad_words_ids
        .iter()
        .map(|tokens| u32_tuple(py, tokens))
        .collect::<PyResult<Vec<_>>>()?;
    dict.set_item(intern!(py, "bad_words_ids"), PyTuple::new(py, bad_words)?)?;
    dict.set_item(
        intern!(py, "allowed_token_ids"),
        sampling
            .allowed_token_ids
            .as_deref()
            .map(|tokens| u32_tuple(py, tokens))
            .transpose()?,
    )?;
    dict.set_item(intern!(py, "typical_p"), sampling.typical_p)?;
    dict.set_item(
        intern!(py, "forced_token_ids"),
        u32_tuple(py, &sampling.forced_token_ids)?,
    )?;
    construct(py, "SamplingParams", &dict)
}

/// Converts image-generation controls into the worker's typed parameters.
fn image_to_py<'py>(py: Python<'py>, image: &ImageParams) -> PyResult<Bound<'py, PyAny>> {
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
        PyTuple::new(py, image.image_prompts.iter().map(|prompt| prompt.as_str()))?,
    )?;
    dict.set_item(intern!(py, "retain_images"), image.retain_images)?;
    construct(py, "ImageParams", &dict)
}

/// Converts a tensor publication into its Python record.
fn tensor_publication_to_py<'py>(
    py: Python<'py>,
    payload: &TensorPublication,
    context: &mut RequestConversion<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item(
        intern!(py, "product"),
        context.tensor_ref(&payload.product)?,
    )?;
    dict.set_item(
        intern!(py, "value"),
        transfer_handle_to_py(py, &payload.value)?,
    )?;
    construct(py, "TensorPublication", &dict)
}

/// Converts tensor metadata and transport coordinates into a Python record.
///
/// The transport becomes a nested typed record (`LocalTransfer`,
/// `PosixShmTransfer`, `CudaVmmTransfer`, or `ChannelTransfer`); the CUDA VMM
/// event and allocation handles and channel payloads cross as `bytes`.
/// `dtype` here is the torch dtype name string, not the `DType` enum used by
/// tensor references.
fn transfer_locator_to_py<'py>(py: Python<'py>, locator: &Locator) -> PyResult<Bound<'py, PyAny>> {
    // Tensor metadata is common to every transport family.
    let dict = PyDict::new(py);
    let source = PyDict::new(py);
    source.set_item("worker_id", &locator.source.worker_id)?;
    source.set_item("rank", locator.source.rank)?;
    source.set_item("node", &locator.source.node)?;
    source.set_item("address_space", &locator.source.address_space)?;
    source.set_item("incarnation", &locator.source.incarnation)?;
    dict.set_item("source", construct(py, "WorkerEndpoint", &source)?)?;
    dict.set_item(intern!(py, "nbytes"), locator.nbytes)?;
    dict.set_item(intern!(py, "dtype"), locator.dtype.as_str())?;
    dict.set_item(intern!(py, "shape"), PyTuple::new(py, &locator.shape)?)?;
    dict.set_item(intern!(py, "device"), locator.device.as_str())?;
    dict.set_item(intern!(py, "offset"), PyTuple::new(py, &locator.offset)?)?;

    // The transport tag determines the remaining coordinate fields.
    let handle = PyDict::new(py);
    let name = match &locator.transport {
        TransferTransport::Local { endpoint, key } => {
            handle.set_item(intern!(py, "endpoint"), endpoint.as_str())?;
            handle.set_item(intern!(py, "key"), key)?;
            "LocalTransfer"
        }
        TransferTransport::PosixShm { endpoint, name } => {
            handle.set_item(intern!(py, "endpoint"), endpoint.as_str())?;
            handle.set_item(intern!(py, "name"), name.as_str())?;
            "PosixShmTransfer"
        }
        TransferTransport::CudaVmm {
            endpoint,
            publication_id,
            storage_size_bytes,
            storage_offsets_bytes,
            span_lengths,
            span_counts,
            tensor_stride,
            ready_event_handle,
            allocation_handle,
            acknowledgment_offset,
        } => {
            handle.set_item(intern!(py, "endpoint"), endpoint.as_str())?;
            handle.set_item(intern!(py, "publication_id"), publication_id.as_str())?;
            handle.set_item(intern!(py, "storage_size_bytes"), storage_size_bytes)?;
            handle.set_item(
                intern!(py, "storage_offsets_bytes"),
                PyTuple::new(py, storage_offsets_bytes)?,
            )?;
            handle.set_item(intern!(py, "span_lengths"), PyTuple::new(py, span_lengths)?)?;
            handle.set_item(intern!(py, "span_counts"), PyTuple::new(py, span_counts)?)?;
            handle.set_item(
                intern!(py, "tensor_stride"),
                PyTuple::new(py, tensor_stride)?,
            )?;
            handle.set_item(
                intern!(py, "ready_event_handle"),
                PyBytes::new(py, ready_event_handle),
            )?;
            handle.set_item(
                intern!(py, "allocation_handle"),
                PyBytes::new(py, allocation_handle),
            )?;
            handle.set_item(intern!(py, "acknowledgment_offset"), acknowledgment_offset)?;
            "CudaVmmTransfer"
        }
        TransferTransport::Channel { endpoint, payload } => {
            handle.set_item(intern!(py, "endpoint"), endpoint.as_str())?;
            handle.set_item(intern!(py, "payload"), PyBytes::new(py, payload))?;
            "ChannelTransfer"
        }
    };
    dict.set_item("transport", construct(py, name, &handle)?)?;

    construct(py, "Locator", &dict)
}

/// Encodes the persistent buffer that identifies a KV publication.
///
/// Uses a fresh `RequestConversion`, so the owner request key and producer
/// call id are equal to, but not the same objects as, the batch's cached
/// leaves for the same identities.
fn buffer_id_to_py<'py>(py: Python<'py>, buffer: &BufferId) -> PyResult<Bound<'py, PyAny>> {
    RequestConversion::new(py)?.buffer_id(*buffer)
}

/// Converts a product-family transfer handle into its typed Python record.
fn transfer_handle_to_py<'py>(
    py: Python<'py>,
    handle: &TransferHandle,
) -> PyResult<Bound<'py, PyAny>> {
    let value = PyDict::new(py);
    let name = match handle {
        TransferHandle::Encoder {
            height,
            width,
            payload_kind,
            tensor,
        } => {
            value.set_item(intern!(py, "height"), height)?;
            value.set_item(intern!(py, "width"), width)?;
            value.set_item(
                intern!(py, "payload_kind"),
                feature_kind_py(py, *payload_kind),
            )?;
            value.set_item(intern!(py, "tensor"), tensor_transfer_to_py(py, tensor)?)?;
            "EncoderTransferValue"
        }
        TransferHandle::DeviceProduct {
            height,
            width,
            value_range,
            tensor,
        } => {
            value.set_item(intern!(py, "height"), height)?;
            value.set_item(intern!(py, "width"), width)?;
            value.set_item(intern!(py, "value_range"), value_range.as_str())?;
            value.set_item(intern!(py, "tensor"), tensor_transfer_to_py(py, tensor)?)?;
            "DeviceProductTransferValue"
        }
        TransferHandle::Latent {
            height,
            width,
            latent_units,
            step,
            tensor,
        } => {
            value.set_item(intern!(py, "height"), height)?;
            value.set_item(intern!(py, "width"), width)?;
            value.set_item(intern!(py, "latent_units"), latent_units)?;
            value.set_item(intern!(py, "step"), step)?;
            value.set_item(intern!(py, "tensor"), tensor_transfer_to_py(py, tensor)?)?;
            "LatentTransferValue"
        }
    };
    construct(py, name, &value)
}

/// Returns the interned Python spelling for a request kind.
fn request_kind_py<'py>(py: Python<'py>, kind: RequestKind) -> &'py Bound<'py, PyString> {
    match kind {
        RequestKind::Info => intern!(py, "info"),
        RequestKind::Submit => intern!(py, "submit"),
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

/// Decodes a Python result or error mapping with strict field typing.
///
/// Returns `None` for any other `kind`, which the caller decodes with the
/// schema-derived converter. Fails with `ValueError` when the response is not
/// a `dict`, has no `kind`, has a non-string `kind`, or is a result or error
/// mapping that fails strict decoding; the last case carries no field detail.
/// The semantic checks of `BatchOutput::validate` run afterwards, in
/// `codec::encode_response` in the IPC crate, when the result is published.
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

/// Decodes a result response and rejects fields reserved for other variants.
fn decode_completion_response_from_py(response: &Bound<'_, PyAny>) -> Option<WorkerResponse> {
    let py = response.py();
    let dict = response.cast::<PyDict>().ok()?;
    let kind = str_field(dict, intern!(py, "kind"))?;
    if kind.to_str().ok()? != "result" {
        return None;
    }
    // Fields of the other response kinds must carry no data: `info` and the
    // error fields below must be absent or `None`, and `calls` may also be an
    // empty list.
    for key in [intern!(py, "info")] {
        if !absent_or_none(dict, key)? {
            return None;
        }
    }
    let report = run_result_from_py(&get(dict, intern!(py, "result"))?)?;
    let identities = error_calls_from_py(dict)?;
    if !identities.is_empty()
        || opt_string(dict, intern!(py, "message"))?.is_some()
        || opt_string(dict, intern!(py, "code"))?.is_some()
        || opt_bool(dict, intern!(py, "fatal"))?.is_some()
        || opt_string(dict, intern!(py, "phase"))?.is_some()
        || opt_string(dict, intern!(py, "route"))?.is_some()
    {
        return None;
    }
    Some(WorkerResponse::Result {
        message_id: opt_u64(dict, intern!(py, "message_id"))?,
        result: report,
    })
}

/// Decodes an error response and rejects success payloads.
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
    let identities = error_calls_from_py(dict)?;
    Some(WorkerResponse::Error {
        message_id: opt_u64(dict, intern!(py, "message_id"))?,
        error: WorkerResponseError {
            message: string_of(&get(dict, intern!(py, "message"))?)?,
            code: opt_string(dict, intern!(py, "code"))?,
            fatal: bool_of(&get(dict, intern!(py, "fatal"))?)?,
            phase: opt_string(dict, intern!(py, "phase"))?,
            route: opt_string(dict, intern!(py, "route"))?,
            calls: identities,
        },
    })
}

/// Decodes a batch result, preserving completion and product order.
///
/// `completions` and `products` must be lists; `forward_stats` may be absent
/// or `None`, but when present every counter is required.
fn run_result_from_py(value: &Bound<'_, PyAny>) -> Option<BatchOutput> {
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
        payloads.push(tensor_publication_from_py(&item)?);
    }
    let forward_stats = match dict.get_item(intern!(py, "forward_stats")).ok()? {
        None => None,
        Some(value) if value.is_none() => None,
        Some(value) => Some(forward_stats_from_py(&value)?),
    };
    Some(BatchOutput {
        batch_id: u64_of(&get(dict, intern!(py, "batch_id"))?)?,
        completions: records,
        products: payloads,
        worker_exec_us: opt_u64(dict, intern!(py, "worker_exec_us"))?,
        forward_stats,
    })
}

/// Decodes the complete set of worker forward-path counters.
fn forward_stats_from_py(value: &Bound<'_, PyAny>) -> Option<ForwardStats> {
    let py = value.py();
    let dict = value.cast::<PyDict>().ok()?;
    Some(ForwardStats {
        // Aggregate execution-mode counters.
        mode_counts: u64_map(dict, intern!(py, "mode_counts"))?,
        mode_tokens: u64_map(dict, intern!(py, "mode_tokens"))?,
        mode_us: u64_map(dict, intern!(py, "mode_us"))?,
        component_us: u64_map(dict, intern!(py, "component_us"))?,

        // Attention backend activity.
        attention_launches: u64_of(&get(dict, intern!(py, "attention_launches"))?)?,
        attention_us: u64_of(&get(dict, intern!(py, "attention_us"))?)?,
        attention_backend_counts: u64_map(dict, intern!(py, "attention_backend_counts"))?,

        // CUDA graph lifecycle and padding behavior.
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

        // Decode relay cache effectiveness.
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

        // FlashInfer planning activity.
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

        // Speculative-verification outcomes.
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

/// Decodes a sequence of `{token_id, logprob, rank}` mappings.
///
/// Unlike the list helpers, this accepts any iterable.
fn token_logprobs_from_py(value: &Bound<'_, PyAny>) -> Option<Vec<TokenLogprob>> {
    value
        .try_iter()
        .ok()?
        .map(|entry| {
            let entry = entry.ok()?;
            let dict = entry.cast::<PyDict>().ok()?;
            let py = entry.py();
            Some(TokenLogprob {
                token_id: u32_of(&get(dict, intern!(py, "token_id"))?)?,
                logprob: get(dict, intern!(py, "logprob"))?.extract::<f32>().ok()?,
                rank: u32_of(&get(dict, intern!(py, "rank"))?)?,
            })
        })
        .collect()
}

/// Decodes one completion with its accepted progress and output fields.
fn completion_record_from_py(value: &Bound<'_, PyAny>) -> Option<RequestOutput> {
    let py = value.py();
    let dict = value.cast::<PyDict>().ok()?;

    // Status and error code are closed string enums; an unknown spelling
    // rejects the record.
    let status = str_field(dict, intern!(py, "status"))?;
    let status = match status.to_str().ok()? {
        "ok" => CallStatus::Ok,
        "predicated" => CallStatus::Predicated,
        "error" => CallStatus::Error,
        _ => return None,
    };
    let error_code = match get(dict, intern!(py, "error_code"))? {
        value if value.is_none() => None,
        value => Some(match value.cast::<PyString>().ok()?.to_str().ok()? {
            "invalid_call" => ErrorCode::InvalidCall,
            "resource_exhausted" => ErrorCode::ResourceExhausted,
            "compute_error" => ErrorCode::ComputeError,
            "cancelled" => ErrorCode::Cancelled,
            "internal" => ErrorCode::Internal,
            _ => return None,
        }),
    };

    let flags = get(dict, intern!(py, "finish_flags"))?;
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

    // `media_output` may be absent or `None`. A present handle must use
    // `posix_shm`, the only `ArtifactHandle` transport, in the
    // `{"transport": ..., "value": {...}}` form.
    let media_output = if absent_or_none(dict, intern!(py, "media_output"))? {
        None
    } else {
        let output = get(dict, intern!(py, "media_output"))?;
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
    Some(RequestOutput {
        sampled_logprob: {
            let value = get(dict, intern!(py, "sampled_logprob"))?;
            if value.is_none() {
                None
            } else {
                Some(value.extract::<f32>().ok()?)
            }
        },
        top_logprobs: token_logprobs_from_py(&get(dict, intern!(py, "top_logprobs"))?)?,
        prompt_logprobs: get(dict, intern!(py, "prompt_logprobs"))?
            .try_iter()
            .ok()?
            .map(|position| token_logprobs_from_py(&position.ok()?))
            .collect::<Option<Vec<_>>>()?,
        request_key: request_key_from_py(&get(dict, intern!(py, "request_key"))?)?,
        call_id: computation_id_from_py(&get(dict, intern!(py, "call_id"))?)?,
        status,
        product_generations: u32_vec(&get(dict, intern!(py, "product_generations"))?)?,
        error_code,
        timing_counters,
        code: {
            let name = string_of(&get(dict, intern!(py, "code"))?)?;
            CallKind::ALL
                .into_iter()
                .find(|code| code.as_str() == name)?
        },
        position: u32_of(&get(dict, intern!(py, "position"))?)?,
        kv_visible_len: u32_of(&get(dict, intern!(py, "kv_visible_len"))?)?,
        kv_computed_len: u32_of(&get(dict, intern!(py, "kv_computed_len"))?)?,
        num_completed_steps: u32_of(&get(dict, intern!(py, "num_completed_steps"))?)?,
        committed_tokens: u32_vec(&get(dict, intern!(py, "committed_tokens"))?)?,
        finish_flags,
        media_output,
        kv_output: if absent_or_none(dict, intern!(py, "kv_output"))? {
            None
        } else {
            Some(kv_transfer_from_py(&get(dict, intern!(py, "kv_output"))?)?)
        },
    })
}

/// Decodes a tensor publication: the product reference and its transfer
/// handle, given as `{"kind": ..., "value": {...}}`.
fn tensor_publication_from_py(value: &Bound<'_, PyAny>) -> Option<TensorPublication> {
    let py = value.py();
    let dict = value.cast::<PyDict>().ok()?;

    let transfer = get(dict, intern!(py, "value"))?;
    let transfer = transfer.cast::<PyDict>().ok()?;
    let kind = str_field(transfer, intern!(py, "kind"))?;
    let payload = get(transfer, intern!(py, "value"))?;
    let payload = payload.cast::<PyDict>().ok()?;

    let value = match kind.to_str().ok()? {
        "encoder" => TransferHandle::Encoder {
            height: u32_of(&get(payload, intern!(py, "height"))?)?,
            width: u32_of(&get(payload, intern!(py, "width"))?)?,
            payload_kind: feature_kind_from_py(&get(payload, intern!(py, "payload_kind"))?)?,
            tensor: tensor_transfer_from_py(&get(payload, intern!(py, "tensor"))?)?,
        },
        "device_product" => TransferHandle::DeviceProduct {
            height: u32_of(&get(payload, intern!(py, "height"))?)?,
            width: u32_of(&get(payload, intern!(py, "width"))?)?,
            value_range: string_of(&get(payload, intern!(py, "value_range"))?)?,
            tensor: tensor_transfer_from_py(&get(payload, intern!(py, "tensor"))?)?,
        },
        "latent" => TransferHandle::Latent {
            height: u32_of(&get(payload, intern!(py, "height"))?)?,
            width: u32_of(&get(payload, intern!(py, "width"))?)?,
            latent_units: u32_of(&get(payload, intern!(py, "latent_units"))?)?,
            step: u32_of(&get(payload, intern!(py, "step"))?)?,
            tensor: tensor_transfer_from_py(&get(payload, intern!(py, "tensor"))?)?,
        },
        _ => return None,
    };

    Some(TensorPublication {
        product: tensor_ref_from_py(&get(dict, intern!(py, "product"))?)?,
        value,
    })
}

/// Decodes transport coordinates and their common logical tensor metadata.
///
/// Reads the flat form `Locator.to_mapping` emits: the `transport` value is a
/// plain tag string and the transport's fields sit beside the common tensor
/// metadata in the same mapping. In the serde form, `Locator.transport` is
/// instead the adjacently tagged `{"transport": tag, "value": {...}}`
/// mapping, so the schema-derived converter rejects the worker's locators.
fn transfer_locator_from_py(value: &Bound<'_, PyAny>) -> Option<Locator> {
    let py = value.py();
    let dict = value.cast::<PyDict>().ok()?;

    // The transport tag determines which coordinate set must be present.
    let transport = match string_of(&get(dict, intern!(py, "transport"))?)?.as_str() {
        "local" => TransferTransport::Local {
            endpoint: string_of(&get(dict, intern!(py, "endpoint"))?)?,
            key: u64_of(&get(dict, intern!(py, "key"))?)?,
        },
        "posix_shm" => TransferTransport::PosixShm {
            endpoint: string_of(&get(dict, intern!(py, "endpoint"))?)?,
            name: string_of(&get(dict, intern!(py, "name"))?)?,
        },
        "cuda_vmm" => TransferTransport::CudaVmm {
            endpoint: string_of(&get(dict, intern!(py, "endpoint"))?)?,
            publication_id: string_of(&get(dict, intern!(py, "publication_id"))?)?,
            storage_size_bytes: u64_of(&get(dict, intern!(py, "storage_size_bytes"))?)?,
            storage_offsets_bytes: u64_vec(&get(dict, intern!(py, "storage_offsets_bytes"))?)?,
            span_lengths: u64_vec(&get(dict, intern!(py, "span_lengths"))?)?,
            span_counts: u32_vec(&get(dict, intern!(py, "span_counts"))?)?,
            tensor_stride: i64_vec(&get(dict, intern!(py, "tensor_stride"))?)?,
            ready_event_handle: bytes_of(&get(dict, intern!(py, "ready_event_handle"))?)?,
            allocation_handle: bytes_of(&get(dict, intern!(py, "allocation_handle"))?)?,
            acknowledgment_offset: i64_of(&get(dict, intern!(py, "acknowledgment_offset"))?)?,
        },
        "channel" => TransferTransport::Channel {
            endpoint: string_of(&get(dict, intern!(py, "endpoint"))?)?,
            payload: bytes_of(&get(dict, intern!(py, "payload"))?)?,
        },
        _ => return None,
    };

    let source_value = get(dict, intern!(py, "source"))?;
    let source = source_value.cast::<PyDict>().ok()?;
    Some(Locator {
        source: WorkerEndpoint {
            worker_id: string_of(&get(source, intern!(py, "worker_id"))?)?,
            rank: u32_of(&get(source, intern!(py, "rank"))?)?,
            node: string_of(&get(source, intern!(py, "node"))?)?,
            address_space: string_of(&get(source, intern!(py, "address_space"))?)?,
            incarnation: string_of(&get(source, intern!(py, "incarnation"))?)?,
        },
        transport,
        nbytes: u64_of(&get(dict, intern!(py, "nbytes"))?)?,
        dtype: string_of(&get(dict, intern!(py, "dtype"))?)?,
        shape: u64_vec(&get(dict, intern!(py, "shape"))?)?,
        device: string_of(&get(dict, intern!(py, "device"))?)?,
        offset: u64_vec(&get(dict, intern!(py, "offset"))?)?,
    })
}

/// Decodes the persistent buffer that identifies a KV publication.
fn buffer_id_mapping_from_py(value: &Bound<'_, PyAny>) -> Option<BufferId> {
    let py = value.py();
    let dict = value.cast::<PyDict>().ok()?;
    Some(BufferId {
        owner: request_key_from_py(&get(dict, intern!(py, "owner"))?)?,
        producer_call_id: computation_id_from_py(&get(dict, intern!(py, "producer_call_id"))?)?,
        output_index: u16_of(&get(dict, intern!(py, "output_index"))?)?,
        generation: u32_of(&get(dict, intern!(py, "generation"))?)?,
    })
}

/// Decodes a product identity, storage contract, and bounded shape.
fn tensor_ref_from_py(value: &Bound<'_, PyAny>) -> Option<TensorRef> {
    let py = value.py();
    let dict = value.cast::<PyDict>().ok()?;

    let dtype = str_field(dict, intern!(py, "dtype"))?;
    let dtype = match dtype.to_str().ok()? {
        "u8" => DType::U8,
        "i32" => DType::I32,
        "i64" => DType::I64,
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        "f32" => DType::F32,
        "i16" => DType::I16,
        _ => return None,
    };

    // Each dimension is `{"kind": "static", "value": extent}` or
    // `{"kind": "device", "value": {"max": bound}}`.
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
    Some(TensorRef {
        request_key: request_key_from_py(&get(dict, intern!(py, "request_key"))?)?,
        producer_call_id: computation_id_from_py(&get(dict, intern!(py, "producer_call_id"))?)?,
        output_index: u16_of(&get(dict, intern!(py, "output_index"))?)?,
        generation: u32_of(&get(dict, intern!(py, "generation"))?)?,
        dtype,
        shape_bound,
    })
}

/// Decodes a Python product-family spelling.
fn feature_kind_from_py(value: &Bound<'_, PyAny>) -> Option<FeatureKind> {
    Some(match string_of(value)?.as_str() {
        "vision_feature" => FeatureKind::Vision,
        "latent_feature" => FeatureKind::Latent,
        _ => return None,
    })
}

/// Decodes a call id from its `batch_id` and `request_index` fields.
fn computation_id_from_py(value: &Bound<'_, PyAny>) -> Option<CallId> {
    let py = value.py();
    let dict = value.cast::<PyDict>().ok()?;
    Some(CallId::new(
        u64_of(&get(dict, intern!(py, "batch_id"))?)?,
        u32_of(&get(dict, intern!(py, "request_index"))?)?,
    ))
}

/// Decodes a request identity from its Python mapping.
fn request_key_from_py(value: &Bound<'_, PyAny>) -> Option<RequestKey> {
    let py = value.py();
    let dict = value.cast::<PyDict>().ok()?;
    Some(RequestKey {
        engine_id: u64_of(&get(dict, intern!(py, "engine_id"))?)?,
        request_id: uniserve_core::RequestId(u64_of(&get(dict, intern!(py, "request_id"))?)?),
        request_epoch: u64_of(&get(dict, intern!(py, "request_epoch"))?)?,
    })
}

/// Decodes one request and call identity attached to an error.
fn error_call_from_py(value: &Bound<'_, PyAny>) -> Option<ErrorCallIdentity> {
    let py = value.py();
    let dict = value.cast::<PyDict>().ok()?;
    Some(ErrorCallIdentity {
        request_key: request_key_from_py(&get(dict, intern!(py, "request_key"))?)?,
        call_id: computation_id_from_py(&get(dict, intern!(py, "call_id"))?)?,
    })
}

/// Decodes an optional ordered list of call identities attached to an error.
///
/// An absent or `None` `calls` field decodes as an empty list.
fn error_calls_from_py(dict: &Bound<'_, PyDict>) -> Option<Vec<ErrorCallIdentity>> {
    let py = dict.py();
    let Some(calls) = dict.get_item(intern!(py, "calls")).ok()? else {
        return Some(Vec::new());
    };
    if calls.is_none() {
        return Some(Vec::new());
    }
    let calls = calls.cast::<PyList>().ok()?;
    let mut identities = Vec::with_capacity(calls.len());
    for item in calls.iter() {
        identities.push(error_call_from_py(&item)?);
    }
    Some(identities)
}

/// Returns a required mapping value, collapsing lookup errors and absence.
fn get<'py>(dict: &Bound<'py, PyDict>, key: &Bound<'py, PyString>) -> Option<Bound<'py, PyAny>> {
    dict.get_item(key).ok().flatten()
}

/// Returns whether a mapping key is absent or explicitly set to `None`.
///
/// Returns `None` only when the lookup itself raises.
fn absent_or_none(dict: &Bound<'_, PyDict>, key: &Bound<'_, PyString>) -> Option<bool> {
    match dict.get_item(key).ok()? {
        Some(value) => Some(value.is_none()),
        None => Some(true),
    }
}

/// Extracts a required mapping value as a borrowed Python string.
fn str_field<'py>(
    dict: &Bound<'py, PyDict>,
    key: &Bound<'py, PyString>,
) -> Option<Bound<'py, PyString>> {
    get(dict, key)?.cast_into::<PyString>().ok()
}

/// Extracts an `i64` while rejecting Python booleans as integers.
fn i64_of(value: &Bound<'_, PyAny>) -> Option<i64> {
    // Python `bool` subclasses `int`, so `extract` alone would accept `True`
    // and `False`. The protocol treats booleans as a distinct type; the other
    // integer helpers apply the same check.
    if value.cast::<PyBool>().is_ok() {
        return None;
    }
    value.extract().ok()
}

/// Extracts a `u64` while rejecting Python booleans as integers.
fn u64_of(value: &Bound<'_, PyAny>) -> Option<u64> {
    if value.cast::<PyBool>().is_ok() {
        return None;
    }
    value.extract().ok()
}

/// Extracts a `u32` while rejecting Python booleans as integers.
fn u32_of(value: &Bound<'_, PyAny>) -> Option<u32> {
    if value.cast::<PyBool>().is_ok() {
        return None;
    }
    value.extract().ok()
}

/// Extracts a `u16` while rejecting Python booleans as integers.
fn u16_of(value: &Bound<'_, PyAny>) -> Option<u16> {
    if value.cast::<PyBool>().is_ok() {
        return None;
    }
    value.extract().ok()
}

/// Extracts a strict Python boolean.
fn bool_of(value: &Bound<'_, PyAny>) -> Option<bool> {
    Some(value.cast::<PyBool>().ok()?.is_true())
}

/// Copies a strict Python string into owned Rust storage.
fn string_of(value: &Bound<'_, PyAny>) -> Option<String> {
    Some(value.cast::<PyString>().ok()?.to_str().ok()?.to_owned())
}

/// Decodes a required string-to-`u64` mapping with deterministic key order.
fn u64_map(dict: &Bound<'_, PyDict>, key: &Bound<'_, PyString>) -> Option<BTreeMap<String, u64>> {
    let values = get(dict, key)?;
    let values = values.cast::<PyDict>().ok()?;
    let mut result = BTreeMap::new();
    for (key, value) in values.iter() {
        result.insert(string_of(&key)?, u64_of(&value)?);
    }
    Some(result)
}

/// Copies a Python list into a strictly typed `u32` vector.
fn u32_vec(value: &Bound<'_, PyAny>) -> Option<Vec<u32>> {
    let list = value.cast::<PyList>().ok()?;
    let mut values = Vec::with_capacity(list.len());
    for item in list.iter() {
        values.push(u32_of(&item)?);
    }
    Some(values)
}

/// Copies a Python list into a strictly typed `u64` vector.
fn u64_vec(value: &Bound<'_, PyAny>) -> Option<Vec<u64>> {
    let list = value.cast::<PyList>().ok()?;
    let mut values = Vec::with_capacity(list.len());
    for item in list.iter() {
        values.push(u64_of(&item)?);
    }
    Some(values)
}

/// Copies a Python list into an `i64` vector while rejecting booleans.
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

/// Copies strict Python `bytes` into owned Rust storage.
fn bytes_of(value: &Bound<'_, PyAny>) -> Option<Vec<u8>> {
    Some(value.cast::<PyBytes>().ok()?.as_bytes().to_vec())
}

/// Decodes an absent, `None`, or strict `u64` mapping field.
fn opt_u64(dict: &Bound<'_, PyDict>, key: &Bound<'_, PyString>) -> Option<Option<u64>> {
    match dict.get_item(key).ok()? {
        None => Some(None),
        Some(value) if value.is_none() => Some(None),
        Some(value) => Some(Some(u64_of(&value)?)),
    }
}

/// Decodes an absent, `None`, or strict boolean mapping field.
fn opt_bool(dict: &Bound<'_, PyDict>, key: &Bound<'_, PyString>) -> Option<Option<bool>> {
    match dict.get_item(key).ok()? {
        None => Some(None),
        Some(value) if value.is_none() => Some(None),
        Some(value) => Some(Some(bool_of(&value)?)),
    }
}

/// Decodes an absent, `None`, or strict string mapping field.
fn opt_string(dict: &Bound<'_, PyDict>, key: &Bound<'_, PyString>) -> Option<Option<String>> {
    match dict.get_item(key).ok()? {
        None => Some(None),
        Some(value) if value.is_none() => Some(None),
        Some(value) => Some(Some(string_of(&value)?)),
    }
}

/// Converts a logical tensor shape and its shard or replica locators into a
/// Python `TensorTransfer` record.
fn tensor_transfer_to_py<'py>(
    py: Python<'py>,
    tensor: &TensorTransfer,
) -> PyResult<Bound<'py, PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("shape", PyTuple::new(py, &tensor.shape)?)?;
    let locations = tensor
        .locations
        .iter()
        .map(|location| transfer_locator_to_py(py, location))
        .collect::<PyResult<Vec<_>>>()?;
    dict.set_item("locations", PyTuple::new(py, locations)?)?;
    construct(py, "TensorTransfer", &dict)
}

/// Decodes a tensor transfer and rejects it unless `TensorTransfer::validate`
/// accepts the assembled descriptor.
fn tensor_transfer_from_py(value: &Bound<'_, PyAny>) -> Option<TensorTransfer> {
    let py = value.py();
    let dict = value.cast::<PyDict>().ok()?;
    let raw = get(dict, intern!(py, "locations"))?;
    let locations = raw
        .cast::<PyList>()
        .ok()?
        .iter()
        .map(|location| transfer_locator_from_py(&location))
        .collect::<Option<Vec<_>>>()?;
    let tensor = TensorTransfer {
        shape: u64_vec(&get(dict, intern!(py, "shape"))?)?,
        locations,
    };
    tensor.validate().ok()?;
    Some(tensor)
}

/// Converts a KV publication imported by a batch into a Python `KvTransfer`
/// record.
fn kv_transfer_to_py<'py>(py: Python<'py>, transfer: &KvTransfer) -> PyResult<Bound<'py, PyAny>> {
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
    // Tensor order is keys, values, then scales for quantized storage (see
    // `KvTransfer::tensors`); all of them belong to the one `source` buffer.
    let tensors = tensors
        .iter()
        .map(|tensor| tensor_transfer_to_py(py, tensor))
        .collect::<PyResult<Vec<_>>>()?;
    value.set_item(intern!(py, "tensors"), PyTuple::new(py, tensors)?)?;
    value.set_item(intern!(py, "source"), buffer_id_to_py(py, source)?)?;
    value.set_item(intern!(py, "destination"), destination.as_str())?;
    value.set_item(
        intern!(py, "base"),
        base.as_ref()
            .map(|buffer| buffer_id_to_py(py, buffer))
            .transpose()?,
    )?;
    value.set_item(intern!(py, "base_extent"), base_extent)?;
    value.set_item(intern!(py, "published_extent"), published_extent)?;
    value.set_item(intern!(py, "group_id"), group_id)?;
    value.set_item(intern!(py, "compute_dtype"), compute_dtype.as_str())?;
    value.set_item(intern!(py, "page_size"), page_size)?;
    construct(py, "KvTransfer", &value)
}

/// Decodes the KV publication a completion reports in `kv_output`.
///
/// Each tensor is checked by `tensor_transfer_from_py`. `KvTransfer::validate`
/// runs later, through `RequestOutput::validate`, when the result is
/// published.
fn kv_transfer_from_py(value: &Bound<'_, PyAny>) -> Option<KvTransfer> {
    let py = value.py();
    let payload = value.cast::<PyDict>().ok()?;
    // Preserve tensor order: keys, values, then scales for quantized storage.
    let raw_tensors = get(payload, intern!(py, "tensors"))?;
    let raw_tensors = raw_tensors.cast::<PyList>().ok()?;
    let mut tensors = Vec::with_capacity(raw_tensors.len());
    for tensor in raw_tensors.iter() {
        tensors.push(tensor_transfer_from_py(&tensor)?);
    }
    Some(KvTransfer {
        tensors,
        source: buffer_id_mapping_from_py(&get(payload, intern!(py, "source"))?)?,
        destination: string_of(&get(payload, intern!(py, "destination"))?)?,
        base: if absent_or_none(payload, intern!(py, "base"))? {
            None
        } else {
            Some(buffer_id_mapping_from_py(&get(
                payload,
                intern!(py, "base"),
            )?)?)
        },
        base_extent: u32_of(&get(payload, intern!(py, "base_extent"))?)?,
        published_extent: u32_of(&get(payload, intern!(py, "published_extent"))?)?,
        group_id: u32_of(&get(payload, intern!(py, "group_id"))?)?,
        compute_dtype: string_of(&get(payload, intern!(py, "compute_dtype"))?)?,
        page_size: u32_of(&get(payload, intern!(py, "page_size"))?)?,
    })
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, SystemTime};

    use pythonize::pythonize;
    use uniserve_core::{BlockId, RequestId};
    use uniserve_worker_ipc::ClientEndpoint;

    use super::*;

    /// An unquantized two-token KV publication of `source`: `bfloat16` keys
    /// and values tensors of shape `[2, 1, 1, 4]`, each on a single
    /// `posix_shm` locator.
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

    /// One submission per computation.
    ///
    /// A batch carries one computation through one component, so the native
    /// conversion is exercised with a token call, a media call and a KV
    /// transfer in three batches (batch ids 11, 12, and 13 under message ids
    /// 9, 10, and 11) rather than one mixed submission.
    fn execute_requests() -> Vec<WorkerRequest> {
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
            2,
        )
        .unwrap();
        let token = TensorRef {
            request_key,
            producer_call_id: CallId::new(11, 0),
            output_index: 0,
            generation: 5,
            dtype: DType::I64,
            shape_bound: ShapeBound::default(),
        };
        let call = Call {
            consumer_slots: Vec::new(),
            coordinates: uniserve_worker_ipc::CallCoordinates::default(),
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
            call_id: CallId::new(11, 0),
            component: "model".into(),
            code: CallKind::Forward(ForwardMode::Prefill),
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
            call_indices: vec![0],
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
                video_units: 3,
                num_inference_steps: 4,
                seed: 29,
                width: 1344,
                height: 768,
            },
            VideoAdmission {
                task: uniserve_core::VideoTask::T2va,
                text_tags: vec![1; 3],
                conditions: Vec::new(),
            },
        )
        .unwrap();
        let media_call = Call {
            consumer_slots: Vec::new(),
            coordinates: uniserve_worker_ipc::CallCoordinates::default(),
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
            call_id: CallId::new(12, 0),
            component: "model".into(),
            code: CallKind::Media(MediaCall::LatentPreparation),
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
            call_id: CallId::new(12, 0),
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
            producer_call_id: CallId::new(10, 0),
            output_index: 0,
            generation: 3,
        };
        let kv_call = Call {
            consumer_slots: Vec::new(),
            coordinates: uniserve_worker_ipc::CallCoordinates::default(),
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
            call_id: CallId::new(13, 0),
            component: "decoder".into(),
            code: CallKind::Transfer(TransferMode::KvInstall),
            bounds: Bounds {
                max_transfer_bytes: 32,
                ..Bounds::default()
            },
            kv_input: Some(kv_source),
            kv_output: Some(BufferId {
                producer_call_id: CallId::new(13, 0),
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
        let mut token_batch = Batch::new(11, vec![admission], vec![call]);
        token_batch.block_tables = block_tables;
        token_batch.new_cache_pages = new_cache_pages;
        token_batch.forward = forward;

        let mut media_batch = Batch::new(12, vec![media_admission], vec![media_call]);
        media_batch.latent_params = latent_params;

        let mut kv_batch = Batch::new(13, Vec::new(), vec![kv_call]);
        kv_batch.kv_inputs = vec![kv_publication(kv_source)];

        [token_batch, media_batch, kv_batch]
            .into_iter()
            .enumerate()
            .map(|(index, batch)| {
                let mut request = WorkerRequest::submit(batch);
                request.set_call_id(Some(9 + index as u64));
                request
            })
            .collect()
    }

    /// The expected reply to the KV batch (message id 11).
    ///
    /// It holds a decode completion with sampled, top, and prompt logprobs,
    /// and a KV-publish completion whose publication is owned by that
    /// completion's own request and call, as `RequestOutput::validate`
    /// requires.
    fn result_response() -> WorkerResponse {
        let request_key = RequestKey::new(1, RequestId(2), 1);
        let mut response = WorkerResponse::result(BatchOutput {
            batch_id: 13,
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
                call_id: CallId::new(13, 0),
                status: CallStatus::Ok,
                product_generations: vec![5],
                error_code: None,
                timing_counters: TimingCounters::default(),
                code: CallKind::Forward(ForwardMode::Decode),
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
            worker_exec_us: Some(12),
            forward_stats: None,
        });
        let WorkerResponse::Result { result, .. } = &mut response else {
            unreachable!();
        };

        let mut publication = result.completions[0].clone();
        publication.request_key = RequestKey::new(1, RequestId(4), 1);
        publication.call_id = CallId::new(13, 1);
        publication.code = CallKind::Transfer(TransferMode::KvPublish);
        publication.committed_tokens.clear();
        publication.sampled_logprob = None;
        publication.top_logprobs.clear();
        publication.prompt_logprobs.clear();
        publication.product_generations.clear();
        publication.kv_output = Some(kv_publication(BufferId {
            owner: publication.request_key,
            producer_call_id: publication.call_id,
            output_index: 0,
            generation: 4,
        }));
        result.completions.push(publication);
        response.set_call_id(Some(11));
        response
    }

    /// A minimal result for a batch whose values this test does not inspect.
    ///
    /// Its message id maps batch ids 11 and 12 to the message ids 9 and 10
    /// that `execute_requests` assigned.
    fn acknowledgement(py: Python<'_>, batch_id: u64) -> Bound<'_, PyAny> {
        let mut response = WorkerResponse::result(BatchOutput {
            batch_id,
            completions: Vec::new(),
            products: Vec::new(),
            worker_exec_us: None,
            forward_stats: None,
        });
        response.set_call_id(Some(9 + batch_id - 11));
        pythonize(py, &response).unwrap()
    }

    /// Drives the three submit requests through a real `PyServer` and back.
    ///
    /// Each natively constructed batch must equal what the Python codec
    /// decodes from its own mapping, and the reply to the KV batch must decode
    /// in Rust to `result_response()`.
    #[test]
    fn native_execute_and_result_round_trip_preserves_values() {
        Python::initialize();

        // The process id and a timestamp keep concurrent test runs on
        // separate IPC services.
        let nonce = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let service = format!("uniserve/ipc-py-test-{}-{nonce}", std::process::id());
        let server = crate::PyServer::new(
            &service,
            1 << 20,
            4,
            uniserve_worker_ipc::SHARED_STORAGE_CHANNEL,
        )
        .unwrap();
        let client = ClientEndpoint::connect(&service, 1 << 20, 4).unwrap();

        let requests = execute_requests();
        let pending = requests
            .iter()
            .map(|request| client.send_request(request).unwrap())
            .collect::<Vec<_>>();
        let expected = result_response();

        Python::attach(|py| {
            // Import `uniserve_worker` from this repository's source tree.
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
            let canonical = py
                .import("uniserve_worker.protocol.batch")
                .unwrap()
                .getattr("Batch")
                .unwrap();

            for (index, request) in requests.iter().enumerate() {
                let native_request = server.recv(py).unwrap();
                let request_dict = native_request.bind(py).cast::<PyDict>().unwrap();
                let native_batch = request_dict.get_item("batch").unwrap().unwrap();
                let WorkerRequest::Submit { batch, .. } = request else {
                    unreachable!();
                };
                assert_eq!(
                    native_batch
                        .getattr("batch_id")
                        .unwrap()
                        .extract::<u64>()
                        .unwrap(),
                    batch.batch_id
                );
                assert_eq!(native_batch.getattr("calls").unwrap().len().unwrap(), 1);

                // The natively constructed batch must be exactly what the
                // canonical codec decodes from its own IPC form.
                let round_tripped = canonical
                    .call_method1(
                        "from_mapping",
                        (native_batch.call_method0("to_mapping").unwrap(),),
                    )
                    .unwrap();
                assert!(
                    round_tripped.eq(&native_batch).unwrap(),
                    "native batch construction diverged from the canonical codec"
                );

                let call = native_batch.getattr("calls").unwrap().get_item(0).unwrap();
                match index {
                    // The token call carries host-staged inputs and the
                    // sampling state the worker reads per call; its
                    // admission carries the request's input image count.
                    0 => {
                        assert_eq!(
                            native_batch
                                .getattr("admissions")
                                .unwrap()
                                .get_item(0)
                                .unwrap()
                                .getattr("input_images")
                                .unwrap()
                                .extract::<u32>()
                                .unwrap(),
                            2
                        );
                        assert_eq!(
                            call.getattr("input_token_ids")
                                .unwrap()
                                .extract::<Vec<u32>>()
                                .unwrap(),
                            vec![7, 8]
                        );
                        let sampling = call.getattr("sampling_state").unwrap();
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
                        server
                            .respond(py, &acknowledgement(py, batch.batch_id))
                            .unwrap();
                    }
                    // The media call travels with its own admission.
                    1 => {
                        let media = native_batch
                            .getattr("admissions")
                            .unwrap()
                            .get_item(0)
                            .unwrap();
                        assert_eq!(
                            media
                                .getattr("prompt_token_ids")
                                .unwrap()
                                .extract::<Vec<u32>>()
                                .unwrap(),
                            vec![17, 23, 65_537]
                        );
                        server
                            .respond(py, &acknowledgement(py, batch.batch_id))
                            .unwrap();
                    }
                    // The KV install call reads the batch's imported
                    // publication; the reply relays its tensors back in a
                    // KV-publish completion.
                    _ => {
                        let imported_kv = native_batch
                            .getattr("kv_inputs")
                            .unwrap()
                            .get_item(0)
                            .unwrap();
                        let imported_mapping = imported_kv.call_method0("to_mapping").unwrap();
                        let expected_source = pythonize(py, &batch.kv_inputs[0].source).unwrap();
                        assert!(
                            imported_kv
                                .getattr("source")
                                .unwrap()
                                .call_method0("to_mapping")
                                .unwrap()
                                .eq(expected_source)
                                .unwrap()
                        );
                        assert!(
                            call.getattr("kv_input")
                                .unwrap()
                                .eq(imported_kv.getattr("source").unwrap())
                                .unwrap()
                        );

                        // `pythonize` renders the expected publication in the
                        // serde form, whose nested locator transports the
                        // native decoder rejects. Substitute the imported
                        // publication's `to_mapping` form, re-owned by the
                        // publishing completion. Its tensors equal the
                        // expected ones, so the final Rust equality checks
                        // every locator and extent after both native
                        // directions.
                        let response = pythonize(py, &expected).unwrap();
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
                    }
                }
            }
        });

        let response = client
            .recv_response_timeout(pending.last().unwrap(), Duration::from_secs(5))
            .unwrap()
            .expect("native result response");
        assert_eq!(response.decode_response().unwrap(), expected);
    }
}
