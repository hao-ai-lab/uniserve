//! Typed conversions for steady-state worker IPC frames.
//!
//! The serve loop crosses the FFI boundary once per direction per batch. The
//! typed converters materialize the canonical `execute` request and `result`
//! response shapes directly: every dict key and enum string is interned
//! ([`pyo3::intern!`]), lists are preallocated at their known lengths, and byte
//! payloads stay on the `bytes` path.
//!
//! [`execute_request_to_py`] produces the mapping consumed by
//! `Batch.from_wire`. [`try_completion_response_from_py`] accepts the exact
//! completion-report mapping emitted by the Python worker. Administrative frame
//! kinds are handled by the schema-derived converter in the caller.

use std::collections::{BTreeMap, HashMap};

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyList, PyString};
use uniserve_core::{ImageParams, SamplingParams};
use uniserve_worker_wire::{
    Admission, AttentionRegime, Batch, BatchPartition, Bounds, CacheCopy, CacheGroupPlacement,
    CloseReason, CompletionRecord, CompletionReport, Control, DType, DimBound, Disposition, Domain,
    DrawLayout, EncodeMode, ErrorCode, ErrorOperationIdentity, ExecutionCapability, FinishFlags,
    GenAdmission, GenMode, KvAdmission, KvBranchPlacement, KvPlacement, LatentPlacement,
    LogicalLengths, OpId, OpStatus, Operation, PartitionCompletion, Point, PointRange, ProductKind,
    ProductPayload, ProductRef, RecoveryPlacement, RegistrationAck, RequestKey, RequestKind,
    ResponseKind, Rng, ShapeBound, SnapshotRef, StorageClass, TimingCounters, TokenMode, TokenSpan,
    TransferMode, UndAdmission, VersionRef, Work, WorkerForwardStats, WorkerRequest,
    WorkerResponse,
};

// ---------------------------------------------------------------------------
// Request -> Python (recv hot path)
// ---------------------------------------------------------------------------

/// Convert an `execute` [`WorkerRequest`] into the canonical Python wire mapping.
pub(crate) fn execute_request_to_py<'py>(
    py: Python<'py>,
    request: &WorkerRequest,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "kind"), request_kind_py(py, request.kind))?;
    dict.set_item(intern!(py, "call_id"), request.call_id)?;
    dict.set_item(
        intern!(py, "batch"),
        request
            .batch
            .as_ref()
            .map(|batch| batch_to_py(py, batch))
            .transpose()?,
    )?;
    dict.set_item(intern!(py, "step_id"), request.step_id)?;
    dict.set_item(intern!(py, "session_id"), request.session_id.map(|id| id.0))?;
    dict.set_item(
        intern!(py, "copies"),
        request
            .copies
            .as_ref()
            .map(|copies| dict_list(py, copies, |copy| cache_copy_to_py(py, copy)))
            .transpose()?,
    )?;
    match &request.product_handles {
        Some(handles) => dict.set_item(
            intern!(py, "product_handles"),
            PyList::new(py, handles.iter().copied())?,
        )?,
        None => dict.set_item(intern!(py, "product_handles"), py.None())?,
    }
    dict.set_item(
        intern!(py, "snapshot"),
        request
            .snapshot
            .as_ref()
            .map(|snapshot| snapshot_to_py(py, snapshot))
            .transpose()?,
    )?;
    dict.set_item(
        intern!(py, "recovery_placement"),
        request
            .recovery_placement
            .as_ref()
            .map(|placement| recovery_placement_to_py(py, placement))
            .transpose()?,
    )?;
    Ok(dict)
}

// ---------------------------------------------------------------------------
// Native typed construction (recv hot path)
// ---------------------------------------------------------------------------

/// Cached handles to the worker's protocol types and enum members.
///
/// The serve loop constructs one `Batch` object per submission; every hot
/// record (operations, KV placements, commit/close/release controls) is built
/// by calling the protocol dataclass constructors positionally, so the worker
/// never re-decodes those records from wire maps. Rare members (admissions,
/// input products, branch and latent placements) still cross as wire maps and
/// are decoded by `native_batch`/`native_partition` on the Python side.
struct NativeRequestTypes {
    operation: Py<PyAny>,
    request_key: Py<PyAny>,
    version_ref: Py<PyAny>,
    fixed_point: Py<PyAny>,
    device_point: Py<PyAny>,
    product_ref: Py<PyAny>,
    shape_bound: Py<PyAny>,
    static_dim: Py<PyAny>,
    device_dim: Py<PyAny>,
    point_range: Py<PyAny>,
    bounds: Py<PyAny>,
    rng: Py<PyAny>,
    kv_placement: Py<PyAny>,
    commit: Py<PyAny>,
    close: Py<PyAny>,
    release: Py<PyAny>,
    native_partition: Py<PyAny>,
    native_batch: Py<PyAny>,
    domains: [Py<PyAny>; 3],
    product_kinds: [Py<PyAny>; 14],
    storage_classes: [Py<PyAny>; 5],
    dtypes: [Py<PyAny>; 8],
    dispositions: [Py<PyAny>; 3],
    close_reasons: [Py<PyAny>; 4],
    draw_layouts: [Py<PyAny>; 3],
    execution_capabilities: [Py<PyAny>; 2],
    attention_regimes: [Py<PyAny>; 4],
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
        let module = py.import("uniserve_worker.batch")?;
        let class = |name: &str| -> PyResult<Py<PyAny>> { Ok(module.getattr(name)?.unbind()) };
        let work = module.getattr("Work")?;
        let work_variant = |kind: &str, mode: Option<&str>| -> PyResult<Py<PyAny>> {
            Ok(work.call1((kind, mode))?.unbind())
        };
        Ok(Self {
            operation: class("Operation")?,
            request_key: class("RequestKey")?,
            version_ref: class("VersionRef")?,
            fixed_point: class("FixedPoint")?,
            device_point: class("DevicePoint")?,
            product_ref: class("ProductRef")?,
            shape_bound: class("ShapeBound")?,
            static_dim: class("StaticDim")?,
            device_dim: class("DeviceDim")?,
            point_range: class("PointRange")?,
            bounds: class("Bounds")?,
            rng: class("Rng")?,
            kv_placement: class("KvPlacement")?,
            commit: class("Commit")?,
            close: class("Close")?,
            release: class("Release")?,
            native_partition: class("native_partition")?,
            native_batch: class("native_batch")?,
            domains: enum_members(&module, "Domain", ["prefill", "decode", "flow"])?,
            product_kinds: enum_members(
                &module,
                "ProductKind",
                [
                    "token",
                    "logprob",
                    "draft",
                    "vision_feature",
                    "latent_feature",
                    "kv",
                    "latent",
                    "artifact",
                    "completion",
                    "sampling_state",
                    "finish",
                    "selected_point",
                    "accepted_span",
                    "continuation",
                ],
            )?,
            storage_classes: enum_members(
                &module,
                "StorageClass",
                [
                    "device_tensor",
                    "paged_kv",
                    "latent_arena",
                    "host_staging",
                    "completion_arena",
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
            execution_capabilities: enum_members(
                &module,
                "ExecutionCapability",
                ["domain_homogeneous", "tensorized_mixed"],
            )?,
            attention_regimes: enum_members(
                &module,
                "AttentionRegime",
                ["none", "causal", "bidirectional", "hybrid"],
            )?,
            works: [
                work_variant("token", Some("extend"))?,
                work_variant("token", Some("decode"))?,
                work_variant("token", Some("verify"))?,
                work_variant("draft", None)?,
                work_variant("encode", Some("vision"))?,
                work_variant("encode", Some("latent"))?,
                work_variant("transfer", Some("product"))?,
                work_variant("transfer", Some("kv_publish"))?,
                work_variant("transfer", Some("kv_install"))?,
                work_variant("gen", Some("transition"))?,
                work_variant("gen", Some("flow"))?,
                work_variant("materialize", None)?,
            ],
        })
    }

    fn get(py: Python<'_>) -> PyResult<&'static Self> {
        if let Some(types) = NATIVE_REQUEST_TYPES.get() {
            return Ok(types);
        }
        let built = Self::build(py)?;
        Ok(NATIVE_REQUEST_TYPES.get_or_init(|| built))
    }

    fn domain<'py>(&self, py: Python<'py>, domain: Domain) -> Bound<'py, PyAny> {
        let index = match domain {
            Domain::Prefill => 0,
            Domain::Decode => 1,
            Domain::Flow => 2,
        };
        self.domains[index].bind(py).clone()
    }

    fn work<'py>(&self, py: Python<'py>, work: Work) -> Bound<'py, PyAny> {
        let index = match work {
            Work::Token(TokenMode::Extend) => 0,
            Work::Token(TokenMode::Decode) => 1,
            Work::Token(TokenMode::Verify) => 2,
            Work::Draft => 3,
            Work::Encode(EncodeMode::Vision) => 4,
            Work::Encode(EncodeMode::Latent) => 5,
            Work::Transfer(TransferMode::Product) => 6,
            Work::Transfer(TransferMode::KvPublish) => 7,
            Work::Transfer(TransferMode::KvInstall) => 8,
            Work::Gen(GenMode::Transition) => 9,
            Work::Gen(GenMode::Flow) => 10,
            Work::Materialize => 11,
        };
        self.works[index].bind(py).clone()
    }

    fn product_kind<'py>(&self, py: Python<'py>, kind: ProductKind) -> Bound<'py, PyAny> {
        let index = match kind {
            ProductKind::Token => 0,
            ProductKind::Logprob => 1,
            ProductKind::Draft => 2,
            ProductKind::VisionFeature => 3,
            ProductKind::LatentFeature => 4,
            ProductKind::Kv => 5,
            ProductKind::Latent => 6,
            ProductKind::Artifact => 7,
            ProductKind::Completion => 8,
            ProductKind::SamplingState => 9,
            ProductKind::Finish => 10,
            ProductKind::SelectedPoint => 11,
            ProductKind::AcceptedSpan => 12,
            ProductKind::Continuation => 13,
        };
        self.product_kinds[index].bind(py).clone()
    }

    fn storage_class<'py>(&self, py: Python<'py>, class: StorageClass) -> Bound<'py, PyAny> {
        let index = match class {
            StorageClass::DeviceTensor => 0,
            StorageClass::PagedKv => 1,
            StorageClass::LatentArena => 2,
            StorageClass::HostStaging => 3,
            StorageClass::CompletionArena => 4,
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
            key.session_id.0,
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

    fn version_ref(&mut self, version: &VersionRef) -> PyResult<Bound<'py, PyAny>> {
        let point = match &version.point {
            Point::Fixed {
                point_index,
                semantic_digest,
            } => self
                .types
                .fixed_point
                .bind(self.py)
                .call1((*point_index, semantic_digest.as_str()))?,
            Point::Device {
                point_index,
                selected_point,
                producer_plan_digest,
            } => {
                let selected = selected_point
                    .as_ref()
                    .map(|selected| self.product_ref(selected))
                    .transpose()?;
                self.types.device_point.bind(self.py).call1((
                    *point_index,
                    selected,
                    producer_plan_digest.as_str(),
                ))?
            }
        };
        let request_key = self.request_key(version.request_key)?;
        self.types
            .version_ref
            .bind(self.py)
            .call1((request_key, version.producer_op_id.0, point))
    }

    fn operation(&mut self, operation: &Operation) -> PyResult<Bound<'py, PyAny>> {
        let request_key = self.request_key(operation.request_key)?;
        let parent = self.version_ref(&operation.parent)?;
        let bounds = self.types.bounds.bind(self.py).call1((
            operation.bounds.max_points,
            operation.bounds.max_tokens,
            operation.bounds.max_kv_pages,
            operation.bounds.max_latent_bytes,
            operation.bounds.max_completion_bytes,
            operation.bounds.max_transfer_bytes,
        ))?;
        let inputs = operation
            .inputs
            .iter()
            .map(|product| self.product_ref(product))
            .collect::<PyResult<Vec<_>>>()?;
        let outputs = operation
            .outputs
            .iter()
            .map(|product| self.product_ref(product))
            .collect::<PyResult<Vec<_>>>()?;
        let predicate = operation
            .predicate
            .as_ref()
            .map(|predicate| self.product_ref(predicate))
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
        let arguments = pyo3::types::PyTuple::new(
            py,
            [
                request_key.into_any(),
                operation.op_id.0.into_pyobject(py)?.into_any(),
                parent.into_any(),
                self.types.work(py, operation.work),
                operation.route.0.into_pyobject(py)?.into_any(),
                self.types.domain(py, operation.domain),
                PyBool::new(py, operation.advances_state)
                    .to_owned()
                    .into_any(),
                bounds.into_any(),
                pyo3::types::PyTuple::new(py, inputs)?.into_any(),
                pyo3::types::PyTuple::new(py, outputs)?.into_any(),
                operation.kv_capacity_pages.into_pyobject(py)?.into_any(),
                predicate
                    .map(Bound::into_any)
                    .unwrap_or_else(|| py.None().into_bound(py)),
                rng.map(Bound::into_any)
                    .unwrap_or_else(|| py.None().into_bound(py)),
                operation.control_seq.into_pyobject(py)?.into_any(),
                operation.plan_digest.as_str().into_pyobject(py)?.into_any(),
            ],
        )?;
        self.types.operation.bind(py).call1(arguments)
    }

    fn kv_placement(&mut self, placement: &KvPlacement) -> PyResult<Bound<'py, PyAny>> {
        let request_key = self.request_key(placement.request_key)?;
        self.types.kv_placement.bind(self.py).call1((
            request_key,
            placement.op_id.0,
            placement.group_id,
            pyo3::types::PyTuple::new(self.py, placement.block_table.iter().map(|block| block.0))?,
            placement.block_table_update,
            pyo3::types::PyTuple::new(
                self.py,
                placement.pages_to_zero.iter().map(|block| block.0),
            )?,
            placement.prefix_length,
            placement.input_length,
            placement.visible_length,
            placement.resulting_length,
        ))
    }

    fn control(&mut self, control: &Control) -> PyResult<Bound<'py, PyAny>> {
        match control {
            Control::Commit {
                request_key,
                control_seq,
                expected_parent,
                selected,
                public_event_limit,
                disposition,
            } => {
                let request_key = self.request_key(*request_key)?;
                let expected_parent = self.version_ref(expected_parent)?;
                let selected = self.version_ref(selected)?;
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
                    control.content_digest().as_str(),
                ))
            }
            Control::Close {
                request_key,
                control_seq,
                cutoff,
                reason,
            } => {
                let request_key = self.request_key(*request_key)?;
                let cutoff = self.version_ref(cutoff)?;
                let reason = match reason {
                    CloseReason::Completed => 0,
                    CloseReason::Cancelled => 1,
                    CloseReason::Error => 2,
                    CloseReason::Preempted => 3,
                };
                self.types.close.bind(self.py).call1((
                    request_key,
                    *control_seq,
                    cutoff,
                    self.types.close_reasons[reason].bind(self.py).clone(),
                    control.content_digest().as_str(),
                ))
            }
            Control::Release { request_key, op_id } => {
                let request_key = self.request_key(*request_key)?;
                self.types.release.bind(self.py).call1((
                    request_key,
                    op_id.0,
                    control.content_digest().as_str(),
                ))
            }
        }
    }

    fn partition(
        &mut self,
        partition: &BatchPartition,
        context: &mut RequestConversion<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let operations = partition
            .operations
            .iter()
            .map(|operation| self.operation(operation))
            .collect::<PyResult<Vec<_>>>()?;
        let kv_placements = partition
            .kv_placements
            .iter()
            .map(|placement| self.kv_placement(placement))
            .collect::<PyResult<Vec<_>>>()?;
        let execution = match partition.execution {
            ExecutionCapability::DomainHomogeneous => 0,
            ExecutionCapability::TensorizedMixed => 1,
        };
        let attention = match partition.attention {
            AttentionRegime::None => 0,
            AttentionRegime::Causal => 1,
            AttentionRegime::Bidirectional => 2,
            AttentionRegime::Hybrid => 3,
        };
        let py = self.py;
        let arguments = pyo3::types::PyTuple::new(
            py,
            [
                partition.partition_id.into_pyobject(py)?.into_any(),
                partition.submission_group.into_pyobject(py)?.into_any(),
                partition.collective_seq.into_pyobject(py)?.into_any(),
                self.types.domain(py, partition.domain),
                partition.route.0.into_pyobject(py)?.into_any(),
                self.types.execution_capabilities[execution]
                    .bind(py)
                    .clone(),
                self.types.attention_regimes[attention].bind(py).clone(),
                partition.shape_class.into_pyobject(py)?.into_any(),
                pyo3::types::PyTuple::new(py, operations)?.into_any(),
                pyo3::types::PyTuple::new(py, partition.request_pool_indices.iter().copied())?
                    .into_any(),
                pyo3::types::PyTuple::new(py, kv_placements)?.into_any(),
                dict_list(py, &partition.kv_branch_placements, |placement| {
                    kv_branch_placement_to_py(py, placement, context)
                })?
                .into_any(),
                dict_list(py, &partition.latent_placements, |placement| {
                    latent_placement_to_py(py, placement, context)
                })?
                .into_any(),
            ],
        )?;
        self.types.native_partition.bind(py).call1(arguments)
    }
}

fn batch_to_py<'py>(py: Python<'py>, batch: &Batch) -> PyResult<Bound<'py, PyAny>> {
    let mut context = RequestConversion::new(py);
    let mut native = NativeRequestConversion::new(py)?;
    let partitions = batch
        .partitions
        .iter()
        .map(|partition| native.partition(partition, &mut context))
        .collect::<PyResult<Vec<_>>>()?;
    let controls = batch
        .controls
        .iter()
        .map(|control| native.control(control))
        .collect::<PyResult<Vec<_>>>()?;
    let admissions = dict_list(py, &batch.admissions, |admission| {
        admission_to_py(py, admission, &mut context)
    })?;
    let input_products = dict_list(py, &batch.input_products, |payload| {
        product_payload_to_py(py, payload, &mut context)
    })?;
    native.types.native_batch.bind(py).call1((
        batch.step_id,
        admissions,
        pyo3::types::PyTuple::new(py, partitions)?,
        pyo3::types::PyTuple::new(py, controls)?,
        input_products,
    ))
}

fn kv_branch_placement_to_py<'py>(
    py: Python<'py>,
    placement: &KvBranchPlacement,
    context: &mut RequestConversion<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(
        intern!(py, "request_key"),
        context.request_key(placement.request_key)?,
    )?;
    dict.set_item(intern!(py, "op_id"), placement.op_id.0)?;
    dict.set_item(intern!(py, "branch_index"), placement.branch_index)?;
    dict.set_item(intern!(py, "group_id"), placement.group_id)?;
    dict.set_item(
        intern!(py, "block_table"),
        PyList::new(py, placement.block_table.iter().map(|block| block.0))?,
    )?;
    dict.set_item(
        intern!(py, "pages_to_zero"),
        PyList::new(py, placement.pages_to_zero.iter().map(|block| block.0))?,
    )?;
    Ok(dict)
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
        dict.set_item(intern!(self.py, "session_id"), key.session_id.0)?;
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
    admission: &Admission,
    context: &mut RequestConversion<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(
        intern!(py, "request_key"),
        context.request_key(admission.request_key)?,
    )?;
    dict.set_item(intern!(py, "request_pool_idx"), admission.request_pool_idx)?;
    dict.set_item(intern!(py, "digest"), admission.digest.as_str())?;
    dict.set_item(
        intern!(py, "und"),
        admission
            .und
            .as_ref()
            .map(|und| und_admission_to_py(py, und))
            .transpose()?,
    )?;
    dict.set_item(
        intern!(py, "gen_admission"),
        admission
            .gen_admission
            .as_ref()
            .map(|branch| gen_admission_to_py(py, branch))
            .transpose()?,
    )?;
    Ok(dict)
}

fn und_admission_to_py<'py>(py: Python<'py>, und: &UndAdmission) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "sampling"), sampling_to_py(py, &und.sampling)?)?;
    dict.set_item(
        intern!(py, "negative_token_ids"),
        u32_list(py, &und.negative_token_ids)?,
    )?;
    dict.set_item(
        intern!(py, "finish_token_ids"),
        u32_list(py, &und.finish_token_ids)?,
    )?;
    dict.set_item(intern!(py, "kv"), kv_admission_to_py(py, &und.kv)?)?;
    Ok(dict)
}

fn gen_admission_to_py<'py>(
    py: Python<'py>,
    branch: &GenAdmission,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "image"), image_to_py(py, &branch.image)?)?;
    Ok(dict)
}

fn kv_admission_to_py<'py>(py: Python<'py>, kv: &KvAdmission) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "prefix_len"), kv.prefix_len)?;
    dict.set_item(intern!(py, "group_id"), kv.group_id)?;
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

fn version_ref_to_py<'py>(
    py: Python<'py>,
    version: &VersionRef,
    context: &mut RequestConversion<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let point = PyDict::new(py);
    match &version.point {
        Point::Fixed {
            point_index,
            semantic_digest,
        } => {
            point.set_item(intern!(py, "kind"), intern!(py, "fixed"))?;
            let value = PyDict::new(py);
            value.set_item(intern!(py, "point_index"), *point_index)?;
            value.set_item(intern!(py, "semantic_digest"), semantic_digest.as_str())?;
            point.set_item(intern!(py, "value"), value)?;
        }
        Point::Device {
            point_index,
            selected_point,
            producer_plan_digest,
        } => {
            point.set_item(intern!(py, "kind"), intern!(py, "device"))?;
            let value = PyDict::new(py);
            value.set_item(intern!(py, "point_index"), *point_index)?;
            value.set_item(
                intern!(py, "selected_point"),
                selected_point
                    .as_ref()
                    .map(|selected_point| product_ref_to_py(py, selected_point, context))
                    .transpose()?,
            )?;
            value.set_item(
                intern!(py, "producer_plan_digest"),
                producer_plan_digest.as_str(),
            )?;
            point.set_item(intern!(py, "value"), value)?;
        }
    }
    let dict = PyDict::new(py);
    dict.set_item(
        intern!(py, "request_key"),
        context.request_key(version.request_key)?,
    )?;
    dict.set_item(intern!(py, "producer_op_id"), version.producer_op_id.0)?;
    dict.set_item(intern!(py, "point"), point)?;
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
    // `serde_bytes` pythonizes to `bytes`; one buffer copy, no per-element walk.
    dict.set_item(intern!(py, "bytes"), PyBytes::new(py, &payload.bytes))?;
    Ok(dict)
}

fn snapshot_to_py<'py>(py: Python<'py>, snapshot: &SnapshotRef) -> PyResult<Bound<'py, PyDict>> {
    let mut context = RequestConversion::new(py);
    let dict = PyDict::new(py);
    dict.set_item(
        intern!(py, "version"),
        version_ref_to_py(py, &snapshot.version, &mut context)?,
    )?;
    dict.set_item(intern!(py, "digest"), snapshot.digest.as_str())?;
    dict.set_item(intern!(py, "locator"), snapshot.locator.as_str())?;
    Ok(dict)
}

fn cache_copy_to_py<'py>(py: Python<'py>, copy: &CacheCopy) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "group_id"), copy.group_id)?;
    dict.set_item(intern!(py, "source_page"), copy.source_page.0)?;
    dict.set_item(intern!(py, "destination_page"), copy.destination_page.0)?;
    Ok(dict)
}

fn cache_group_placement_to_py<'py>(
    py: Python<'py>,
    placement: &CacheGroupPlacement,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "group_id"), placement.group_id)?;
    dict.set_item(
        intern!(py, "page_ids"),
        u32_list(
            py,
            &placement
                .page_ids
                .iter()
                .map(|page| page.0)
                .collect::<Vec<_>>(),
        )?,
    )?;
    dict.set_item(intern!(py, "length"), placement.length)?;
    Ok(dict)
}

fn recovery_placement_to_py<'py>(
    py: Python<'py>,
    placement: &RecoveryPlacement,
) -> PyResult<Bound<'py, PyDict>> {
    let mut context = RequestConversion::new(py);
    let dict = PyDict::new(py);
    dict.set_item(
        intern!(py, "request_key"),
        context.request_key(placement.request_key)?,
    )?;
    dict.set_item(intern!(py, "request_pool_idx"), placement.request_pool_idx)?;
    dict.set_item(
        intern!(py, "cache_groups"),
        dict_list(py, &placement.cache_groups, |group| {
            cache_group_placement_to_py(py, group)
        })?,
    )?;
    dict.set_item(
        intern!(py, "latent_page_table"),
        u32_list(py, &placement.latent_page_table)?,
    )?;
    Ok(dict)
}

fn request_kind_py<'py>(py: Python<'py>, kind: RequestKind) -> &'py Bound<'py, PyString> {
    match kind {
        RequestKind::GetCapabilities => intern!(py, "get_capabilities"),
        RequestKind::Execute => intern!(py, "execute"),
        RequestKind::PollCompletions => intern!(py, "poll_completions"),
        RequestKind::DropSession => intern!(py, "drop_session"),
        RequestKind::Shutdown => intern!(py, "shutdown"),
        RequestKind::CopyKv => intern!(py, "copy_kv"),
        RequestKind::ReleaseProducts => intern!(py, "release_products"),
        RequestKind::GetPressure => intern!(py, "get_pressure"),
        RequestKind::SnapshotSession => intern!(py, "snapshot_session"),
        RequestKind::RestoreSession => intern!(py, "restore_session"),
    }
}

fn product_kind_py<'py>(py: Python<'py>, kind: ProductKind) -> &'py Bound<'py, PyString> {
    match kind {
        ProductKind::Token => intern!(py, "token"),
        ProductKind::Logprob => intern!(py, "logprob"),
        ProductKind::Draft => intern!(py, "draft"),
        ProductKind::VisionFeature => intern!(py, "vision_feature"),
        ProductKind::LatentFeature => intern!(py, "latent_feature"),
        ProductKind::Kv => intern!(py, "kv"),
        ProductKind::Latent => intern!(py, "latent"),
        ProductKind::Artifact => intern!(py, "artifact"),
        ProductKind::Completion => intern!(py, "completion"),
        ProductKind::SamplingState => intern!(py, "sampling_state"),
        ProductKind::Finish => intern!(py, "finish"),
        ProductKind::SelectedPoint => intern!(py, "selected_point"),
        ProductKind::AcceptedSpan => intern!(py, "accepted_span"),
        ProductKind::Continuation => intern!(py, "continuation"),
    }
}

fn storage_class_py<'py>(py: Python<'py>, class: StorageClass) -> &'py Bound<'py, PyString> {
    match class {
        StorageClass::DeviceTensor => intern!(py, "device_tensor"),
        StorageClass::PagedKv => intern!(py, "paged_kv"),
        StorageClass::LatentArena => intern!(py, "latent_arena"),
        StorageClass::HostStaging => intern!(py, "host_staging"),
        StorageClass::CompletionArena => intern!(py, "completion_arena"),
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
) -> Option<WorkerResponse> {
    let py = response.py();
    let dict = response.cast::<PyDict>().ok()?;
    let kind = str_field(dict, intern!(py, "kind"))?;
    if kind.to_str().ok()? != "result" {
        return None;
    }
    // Result reports reserve these fields for their respective response kinds.
    for key in [
        intern!(py, "capabilities"),
        intern!(py, "pressure"),
        intern!(py, "snapshot"),
    ] {
        if !absent_or_none(dict, key)? {
            return None;
        }
    }
    let report = match get(dict, intern!(py, "completion_report"))? {
        value if value.is_none() => None,
        value => Some(completion_report_from_py(&value)?),
    };
    let operations = get(dict, intern!(py, "operations"))?;
    let operations = operations.cast::<PyList>().ok()?;
    let mut identities = Vec::with_capacity(operations.len());
    for item in operations.iter() {
        identities.push(error_operation_from_py(&item)?);
    }
    Some(WorkerResponse {
        kind: ResponseKind::Result,
        call_id: opt_u64(dict, intern!(py, "call_id"))?,
        capabilities: None,
        completion_report: report,
        pressure: None,
        message: opt_string(dict, intern!(py, "message"))?,
        code: opt_string(dict, intern!(py, "code"))?,
        retryable: opt_bool(dict, intern!(py, "retryable"))?,
        fatal: opt_bool(dict, intern!(py, "fatal"))?,
        phase: opt_string(dict, intern!(py, "phase"))?,
        route: opt_string(dict, intern!(py, "route"))?,
        operations: identities,
        snapshot: None,
    })
}

fn completion_report_from_py(value: &Bound<'_, PyAny>) -> Option<CompletionReport> {
    let py = value.py();
    let dict = value.cast::<PyDict>().ok()?;
    let partitions = get(dict, intern!(py, "partitions"))?;
    let partitions = partitions.cast::<PyList>().ok()?;
    let mut partition_reports = Vec::with_capacity(partitions.len());
    for item in partitions.iter() {
        partition_reports.push(partition_completion_from_py(&item)?);
    }
    Some(CompletionReport {
        step_id: u64_of(&get(dict, intern!(py, "step_id"))?)?,
        partitions: partition_reports,
    })
}

fn partition_completion_from_py(value: &Bound<'_, PyAny>) -> Option<PartitionCompletion> {
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
    Some(PartitionCompletion {
        partition_id: u32_of(&get(dict, intern!(py, "partition_id"))?)?,
        completions: records,
        products: payloads,
        registration,
        worker_exec_us: opt_u64(dict, intern!(py, "worker_exec_us"))?,
        forward_stats,
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

fn completion_record_from_py(value: &Bound<'_, PyAny>) -> Option<CompletionRecord> {
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
    let lengths = get(dict, intern!(py, "logical_lengths"))?;
    let lengths = lengths.cast::<PyDict>().ok()?;
    let logical_lengths = LogicalLengths {
        token_len: u32_of(&get(lengths, intern!(py, "token_len"))?)?,
        kv_visible_len: u32_of(&get(lengths, intern!(py, "kv_visible_len"))?)?,
        latent_len: u32_of(&get(lengths, intern!(py, "latent_len"))?)?,
        kv_reserved_len: u32_of(&get(lengths, intern!(py, "kv_reserved_len"))?)?,
        kv_initialized_len: u32_of(&get(lengths, intern!(py, "kv_initialized_len"))?)?,
        kv_committed_len: u32_of(&get(lengths, intern!(py, "kv_committed_len"))?)?,
        kv_published_len: u32_of(&get(lengths, intern!(py, "kv_published_len"))?)?,
    };
    let span = get(dict, intern!(py, "token_span"))?;
    let span = span.cast::<PyDict>().ok()?;
    let token_span = TokenSpan {
        base: u32_of(&get(span, intern!(py, "base"))?)?,
        len: u32_of(&get(span, intern!(py, "len"))?)?,
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
    Some(CompletionRecord {
        request_key: request_key_from_py(&get(dict, intern!(py, "request_key"))?)?,
        op_id: OpId(u64_of(&get(dict, intern!(py, "op_id"))?)?),
        completion_slot_generation: u32_of(&get(dict, intern!(py, "completion_slot_generation"))?)?,
        status,
        selected_point: u32_of(&get(dict, intern!(py, "selected_point"))?)?,
        logical_lengths,
        token_span,
        committed_tokens: u32_vec(&get(dict, intern!(py, "committed_tokens"))?)?,
        finish_flags,
        product_generations: u32_vec(&get(dict, intern!(py, "product_generations"))?)?,
        semantic_digest: string_of(&get(dict, intern!(py, "semantic_digest"))?)?,
        error_code,
        timing_counters,
    })
}

fn product_payload_from_py(value: &Bound<'_, PyAny>) -> Option<ProductPayload> {
    let py = value.py();
    let dict = value.cast::<PyDict>().ok()?;
    let bytes = get(dict, intern!(py, "bytes"))?;
    let bytes = bytes.cast::<PyBytes>().ok()?.as_bytes().to_vec();
    Some(ProductPayload {
        product: product_ref_from_py(&get(dict, intern!(py, "product"))?)?,
        bytes,
    })
}

fn product_ref_from_py(value: &Bound<'_, PyAny>) -> Option<ProductRef> {
    let py = value.py();
    let dict = value.cast::<PyDict>().ok()?;
    let kind = str_field(dict, intern!(py, "kind"))?;
    let kind = match kind.to_str().ok()? {
        "token" => ProductKind::Token,
        "logprob" => ProductKind::Logprob,
        "draft" => ProductKind::Draft,
        "vision_feature" => ProductKind::VisionFeature,
        "latent_feature" => ProductKind::LatentFeature,
        "kv" => ProductKind::Kv,
        "latent" => ProductKind::Latent,
        "artifact" => ProductKind::Artifact,
        "completion" => ProductKind::Completion,
        "sampling_state" => ProductKind::SamplingState,
        "finish" => ProductKind::Finish,
        "selected_point" => ProductKind::SelectedPoint,
        "accepted_span" => ProductKind::AcceptedSpan,
        "continuation" => ProductKind::Continuation,
        _ => return None,
    };
    let storage_class = str_field(dict, intern!(py, "storage_class"))?;
    let storage_class = match storage_class.to_str().ok()? {
        "device_tensor" => StorageClass::DeviceTensor,
        "paged_kv" => StorageClass::PagedKv,
        "latent_arena" => StorageClass::LatentArena,
        "host_staging" => StorageClass::HostStaging,
        "completion_arena" => StorageClass::CompletionArena,
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

fn request_key_from_py(value: &Bound<'_, PyAny>) -> Option<RequestKey> {
    let py = value.py();
    let dict = value.cast::<PyDict>().ok()?;
    Some(RequestKey {
        authority_id: u64_of(&get(dict, intern!(py, "authority_id"))?)?,
        session_id: uniserve_core::RequestId(u64_of(&get(dict, intern!(py, "session_id"))?)?),
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
    // Protocol integer fields reject Python booleans instead of coercing them
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
    use uniserve_worker_ipc_core::ClientEndpoint;

    use super::*;

    fn execute_request() -> WorkerRequest {
        let request_key = RequestKey::new(1, RequestId(2), 1);
        let admission = Admission::new(
            request_key,
            1,
            Some(UndAdmission {
                sampling: SamplingParams {
                    temperature: 0.0,
                    ignore_eos: true,
                    ..SamplingParams::default()
                },
                negative_token_ids: Vec::new(),
                finish_token_ids: Vec::new(),
                kv: KvAdmission::default(),
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
            storage_class: StorageClass::DeviceTensor,
            dtype: DType::U32,
            shape_bound: ShapeBound::default(),
            point_range: PointRange {
                base_point: 0,
                max_points: 1,
            },
        };
        let finish = ProductRef {
            request_key,
            producer_op_id: OpId(1),
            output_index: 1,
            generation: 6,
            kind: ProductKind::Finish,
            storage_class: StorageClass::DeviceTensor,
            dtype: DType::U8,
            shape_bound: ShapeBound::default(),
            point_range: PointRange::default(),
        };
        let operation = Operation::registered(
            request_key,
            OpId(1),
            VersionRef::admission_root(request_key, OpId(0), admission.digest.clone()),
            Work::Token(TokenMode::Extend),
            uniserve_worker_wire::RouteId(0),
            Domain::Prefill,
            Bounds {
                max_points: 1,
                max_tokens: 2,
                max_kv_pages: 1,
                ..Bounds::default()
            },
            vec![input.clone()],
            vec![token, finish],
            1,
            None,
            None,
            0,
        );
        let partition = BatchPartition {
            partition_id: 1,
            submission_group: 1,
            collective_seq: 1,
            domain: Domain::Prefill,
            route: uniserve_worker_wire::RouteId(0),
            execution: ExecutionCapability::DomainHomogeneous,
            attention: AttentionRegime::Causal,
            shape_class: 0,
            operations: vec![operation],
            request_pool_indices: vec![1],
            kv_placements: vec![KvPlacement {
                request_key,
                op_id: OpId(1),
                group_id: 0,
                block_table: vec![BlockId(1)],
                block_table_update: true,
                pages_to_zero: vec![BlockId(1)],
                prefix_length: 0,
                input_length: 2,
                visible_length: 0,
                resulting_length: 2,
            }],
            kv_branch_placements: Vec::new(),
            latent_placements: Vec::new(),
        };
        let mut request = WorkerRequest::execute(
            Batch::new(11, vec![admission], vec![partition]).with_input_products(vec![
                ProductPayload {
                    product: input,
                    bytes: uniserve_worker_wire::encode_token_product_bytes(&[7, 8]),
                },
            ]),
        );
        request.call_id = Some(9);
        request
    }

    fn result_response() -> WorkerResponse {
        let request_key = RequestKey::new(1, RequestId(2), 1);
        let mut response = WorkerResponse::completion_report(CompletionReport {
            step_id: 11,
            partitions: vec![PartitionCompletion {
                partition_id: 1,
                completions: vec![CompletionRecord {
                    request_key,
                    op_id: OpId(1),
                    completion_slot_generation: 1,
                    status: OpStatus::Ok,
                    selected_point: 1,
                    logical_lengths: LogicalLengths {
                        token_len: 2,
                        kv_visible_len: 2,
                        latent_len: 0,
                        kv_reserved_len: 2,
                        kv_initialized_len: 2,
                        kv_committed_len: 2,
                        kv_published_len: 2,
                    },
                    token_span: TokenSpan { base: 0, len: 1 },
                    committed_tokens: vec![42],
                    finish_flags: FinishFlags::default(),
                    product_generations: vec![5, 6],
                    semantic_digest: "b".repeat(64),
                    error_code: None,
                    timing_counters: TimingCounters::default(),
                }],
                products: Vec::new(),
                registration: RegistrationAck { visible: true },
                worker_exec_us: Some(12),
                forward_stats: None,
            }],
        });
        response.call_id = Some(9);
        response
    }

    #[test]
    fn native_execute_and_result_round_trip_preserves_protocol_values() {
        Python::initialize();
        let nonce = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let service = format!("uniserve/ipc-py-contract-{}-{nonce}", std::process::id());
        let server = crate::PyServer::new(&service, 1 << 20, 2).unwrap();
        let client = ClientEndpoint::connect(&service, 1 << 20, 2).unwrap();
        let request = execute_request();
        let pending = client.send_request(&request).unwrap();
        let expected = result_response();

        Python::attach(|py| {
            let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../..")
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
            let native_batch = request_dict.get_item("batch").unwrap().unwrap();
            assert_eq!(
                native_batch
                    .getattr("step_id")
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                11
            );
            assert_eq!(
                native_batch.getattr("operations").unwrap().len().unwrap(),
                1
            );
            // The natively constructed batch must be exactly what the
            // canonical codec decodes from its own wire form.
            let round_tripped = py
                .import("uniserve_worker.batch")
                .unwrap()
                .getattr("Batch")
                .unwrap()
                .call_method1(
                    "from_wire",
                    (native_batch.call_method0("to_wire").unwrap(),),
                )
                .unwrap();
            assert!(
                round_tripped.eq(&native_batch).unwrap(),
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
