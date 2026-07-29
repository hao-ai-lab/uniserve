//! Hand-rolled conversions for the hot worker-IPC frames.
//!
//! The serve loop crosses the FFI boundary once per direction per batch. The
//! reflective `pythonize`/`depythonize` walk costs milliseconds at decode batch
//! sizes, dominated by per-field `PyString` creation and serde dispatch. The
//! converters here build the exact same Python values directly: every dict key
//! and enum string is interned ([`pyo3::intern!`]), lists are preallocated at
//! their known lengths, and byte payloads stay on the `bytes` fast path.
//!
//! Contract: for an `execute` request, [`execute_request_to_py`] produces a
//! Python object deep-equal (`==`) to `pythonize(&WorkerRequest)`; for a
//! `result` response in the shape the Python worker emits,
//! [`try_completion_response_from_py`] produces the same `WorkerResponse` as
//! `depythonize`. Both properties are asserted field-exhaustively by the tests
//! in this module. Anything outside those hot shapes keeps the reflective
//! path: rare request kinds are still pythonized by the caller, and the
//! response extractor returns `None` on any unexpected shape so the caller
//! falls back to `depythonize` (identical values, identical errors).

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyList, PyString};
use uniserve_core::{ImageParams, SamplingParams};
use uniserve_worker_wire::{
    Admission, Batch, Bounds, CloseReason, CompletionRecord, CompletionReport, Control, DType,
    DimBound, Disposition, Domain, DrawLayout, EncodeMode, ErrorCode, ErrorOperationIdentity,
    FinishFlags, GenAdmission, GenMode, KvAllocation, LogicalLengths, OpId, OpStatus, Operation,
    Point, PointRange, ProductKind, ProductPayload, ProductRef, RegistrationAck, RequestKey,
    RequestKind, ResponseKind, Rng, ShapeBound, SnapshotRef, StorageClass, TimingCounters,
    TokenMode, TokenSpan, TransferMode, UndAdmission, VersionRef, WorkerRequest, WorkerResponse,
    Work,
};

// ---------------------------------------------------------------------------
// Request -> Python (recv hot path)
// ---------------------------------------------------------------------------

/// Convert an `execute` [`WorkerRequest`] into the exact Python object
/// `pythonize` would produce. Total over the wire types (they are closed), so
/// the caller only needs the reflective fallback for other request kinds.
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
    dict.set_item(intern!(py, "session_id"), request.session_id.map(|id| id.0))?;
    match &request.copies {
        Some(copies) => dict.set_item(
            intern!(py, "copies"),
            PyList::new(py, copies.iter().map(|(src, dst)| (src.0, dst.0)))?,
        )?,
        None => dict.set_item(intern!(py, "copies"), py.None())?,
    }
    dict.set_item(intern!(py, "adapter_id"), request.adapter_id)?;
    dict.set_item(intern!(py, "adapter_path"), request.adapter_path.as_deref())?;
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
    Ok(dict)
}

fn batch_to_py<'py>(py: Python<'py>, batch: &Batch) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "step_id"), batch.step_id)?;
    dict.set_item(
        intern!(py, "admissions"),
        dict_list(py, &batch.admissions, admission_to_py)?,
    )?;
    dict.set_item(
        intern!(py, "operations"),
        dict_list(py, &batch.operations, operation_to_py)?,
    )?;
    dict.set_item(
        intern!(py, "controls"),
        dict_list(py, &batch.controls, control_to_py)?,
    )?;
    dict.set_item(
        intern!(py, "input_products"),
        dict_list(py, &batch.input_products, product_payload_to_py)?,
    )?;
    Ok(dict)
}

fn dict_list<'py, T>(
    py: Python<'py>,
    items: &[T],
    convert: fn(Python<'py>, &T) -> PyResult<Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyList>> {
    let converted = items
        .iter()
        .map(|item| convert(py, item))
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, converted)
}

fn u32_list<'py>(py: Python<'py>, values: &[u32]) -> PyResult<Bound<'py, PyList>> {
    PyList::new(py, values.iter().copied())
}

fn admission_to_py<'py>(py: Python<'py>, admission: &Admission) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(
        intern!(py, "request_key"),
        request_key_to_py(py, admission.request_key)?,
    )?;
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
    dict.set_item(intern!(py, "adapter_id"), admission.adapter_id)?;
    Ok(dict)
}

fn und_admission_to_py<'py>(py: Python<'py>, und: &UndAdmission) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "sampling"), sampling_to_py(py, &und.sampling)?)?;
    dict.set_item(
        intern!(py, "negative_token_ids"),
        u32_list(py, &und.negative_token_ids)?,
    )?;
    dict.set_item(intern!(py, "kv"), kv_allocation_to_py(py, &und.kv)?)?;
    Ok(dict)
}

fn gen_admission_to_py<'py>(py: Python<'py>, branch: &GenAdmission) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "image"), image_to_py(py, &branch.image)?)?;
    Ok(dict)
}

fn kv_allocation_to_py<'py>(py: Python<'py>, kv: &KvAllocation) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(
        intern!(py, "block_ids"),
        PyList::new(py, kv.block_ids.iter().map(|block| block.0))?,
    )?;
    dict.set_item(intern!(py, "prefix_len"), kv.prefix_len)?;
    dict.set_item(intern!(py, "group_id"), kv.group_id)?;
    Ok(dict)
}

fn sampling_to_py<'py>(
    py: Python<'py>,
    sampling: &SamplingParams,
) -> PyResult<Bound<'py, PyDict>> {
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
        PyList::new(py, sampling.logit_bias.iter().map(|(token, bias)| (*token, *bias)))?,
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
    Ok(dict)
}

fn image_to_py<'py>(py: Python<'py>, image: &ImageParams) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "steps"), image.steps)?;
    dict.set_item(intern!(py, "cfg_text_scale"), image.cfg_text_scale)?;
    dict.set_item(intern!(py, "cfg_img_scale"), image.cfg_img_scale)?;
    dict.set_item(intern!(py, "cfg_renorm_type"), image.cfg_renorm_type.as_str())?;
    dict.set_item(intern!(py, "cfg_renorm_min"), image.cfg_renorm_min)?;
    // `(f32, f32)` pythonizes as a Python tuple.
    dict.set_item(intern!(py, "cfg_interval"), image.cfg_interval)?;
    dict.set_item(intern!(py, "timestep_shift"), image.timestep_shift)?;
    dict.set_item(intern!(py, "height"), image.height)?;
    dict.set_item(intern!(py, "width"), image.width)?;
    dict.set_item(intern!(py, "seed"), image.seed)?;
    dict.set_item(intern!(py, "negative_prompt"), image.negative_prompt.as_str())?;
    dict.set_item(intern!(py, "max_images"), image.max_images)?;
    dict.set_item(
        intern!(py, "image_prompts"),
        PyList::new(py, image.image_prompts.iter().map(|prompt| prompt.as_str()))?,
    )?;
    dict.set_item(intern!(py, "retain_images"), image.retain_images)?;
    Ok(dict)
}

fn operation_to_py<'py>(py: Python<'py>, operation: &Operation) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(
        intern!(py, "request_key"),
        request_key_to_py(py, operation.request_key)?,
    )?;
    dict.set_item(intern!(py, "op_id"), operation.op_id.0)?;
    dict.set_item(intern!(py, "parent"), version_ref_to_py(py, &operation.parent)?)?;
    dict.set_item(intern!(py, "work"), work_to_py(py, operation.work)?)?;
    dict.set_item(intern!(py, "route"), operation.route.0)?;
    dict.set_item(intern!(py, "domain"), domain_py(py, operation.domain))?;
    dict.set_item(intern!(py, "advances_state"), operation.advances_state)?;
    dict.set_item(intern!(py, "bounds"), bounds_to_py(py, &operation.bounds)?)?;
    dict.set_item(
        intern!(py, "inputs"),
        dict_list(py, &operation.inputs, product_ref_to_py)?,
    )?;
    dict.set_item(
        intern!(py, "outputs"),
        dict_list(py, &operation.outputs, product_ref_to_py)?,
    )?;
    dict.set_item(
        intern!(py, "new_kv_blocks"),
        PyList::new(py, operation.new_kv_blocks.iter().map(|block| block.0))?,
    )?;
    dict.set_item(
        intern!(py, "predicate"),
        operation
            .predicate
            .as_ref()
            .map(|predicate| product_ref_to_py(py, predicate))
            .transpose()?,
    )?;
    dict.set_item(
        intern!(py, "rng"),
        operation.rng.as_ref().map(|rng| rng_to_py(py, rng)).transpose()?,
    )?;
    dict.set_item(intern!(py, "control_seq"), operation.control_seq)?;
    dict.set_item(intern!(py, "plan_digest"), operation.plan_digest.as_str())?;
    Ok(dict)
}

fn request_key_to_py<'py>(py: Python<'py>, key: RequestKey) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "authority_id"), key.authority_id)?;
    dict.set_item(intern!(py, "session_id"), key.session_id.0)?;
    dict.set_item(intern!(py, "epoch"), key.epoch)?;
    Ok(dict)
}

fn version_ref_to_py<'py>(py: Python<'py>, version: &VersionRef) -> PyResult<Bound<'py, PyDict>> {
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
            selected_point,
            producer_plan_digest,
        } => {
            point.set_item(intern!(py, "kind"), intern!(py, "device"))?;
            let value = PyDict::new(py);
            value.set_item(
                intern!(py, "selected_point"),
                product_ref_to_py(py, selected_point)?,
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
        request_key_to_py(py, version.request_key)?,
    )?;
    dict.set_item(intern!(py, "producer_op_id"), version.producer_op_id.0)?;
    dict.set_item(intern!(py, "point"), point)?;
    Ok(dict)
}

fn product_ref_to_py<'py>(py: Python<'py>, product: &ProductRef) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(
        intern!(py, "request_key"),
        request_key_to_py(py, product.request_key)?,
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
        shape_bound_to_py(py, &product.shape_bound)?,
    )?;
    dict.set_item(
        intern!(py, "point_range"),
        point_range_to_py(py, product.point_range)?,
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

fn bounds_to_py<'py>(py: Python<'py>, bounds: &Bounds) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "max_points"), bounds.max_points)?;
    dict.set_item(intern!(py, "max_tokens"), bounds.max_tokens)?;
    dict.set_item(intern!(py, "max_kv_pages"), bounds.max_kv_pages)?;
    dict.set_item(intern!(py, "max_latent_bytes"), bounds.max_latent_bytes)?;
    dict.set_item(
        intern!(py, "max_completion_bytes"),
        bounds.max_completion_bytes,
    )?;
    dict.set_item(intern!(py, "max_transfer_bytes"), bounds.max_transfer_bytes)?;
    Ok(dict)
}

fn rng_to_py<'py>(py: Python<'py>, rng: &Rng) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "seed"), rng.seed)?;
    dict.set_item(intern!(py, "semantic_index_base"), rng.semantic_index_base)?;
    dict.set_item(intern!(py, "draw_layout"), draw_layout_py(py, rng.draw_layout))?;
    Ok(dict)
}

fn work_to_py<'py>(py: Python<'py>, work: Work) -> PyResult<Bound<'py, PyDict>> {
    // Adjacently tagged serde enums emit only the tag for unit variants; the
    // `value` key is present exactly when the variant carries content.
    let dict = PyDict::new(py);
    match work {
        Work::Token(mode) => {
            dict.set_item(intern!(py, "kind"), intern!(py, "token"))?;
            let value = match mode {
                TokenMode::Extend => intern!(py, "extend"),
                TokenMode::Decode => intern!(py, "decode"),
                TokenMode::Verify => intern!(py, "verify"),
            };
            dict.set_item(intern!(py, "value"), value)?;
        }
        Work::Draft => dict.set_item(intern!(py, "kind"), intern!(py, "draft"))?,
        Work::Encode(mode) => {
            dict.set_item(intern!(py, "kind"), intern!(py, "encode"))?;
            let value = match mode {
                EncodeMode::Vision => intern!(py, "vision"),
                EncodeMode::Latent => intern!(py, "latent"),
            };
            dict.set_item(intern!(py, "value"), value)?;
        }
        Work::Transfer(mode) => {
            dict.set_item(intern!(py, "kind"), intern!(py, "transfer"))?;
            let value = match mode {
                TransferMode::Product => intern!(py, "product"),
                TransferMode::KvPublish => intern!(py, "kv_publish"),
                TransferMode::KvInstall => intern!(py, "kv_install"),
            };
            dict.set_item(intern!(py, "value"), value)?;
        }
        Work::Gen(mode) => {
            dict.set_item(intern!(py, "kind"), intern!(py, "gen"))?;
            let value = match mode {
                GenMode::Transition => intern!(py, "transition"),
                GenMode::Flow => intern!(py, "flow"),
            };
            dict.set_item(intern!(py, "value"), value)?;
        }
        Work::Materialize => dict.set_item(intern!(py, "kind"), intern!(py, "materialize"))?,
    }
    Ok(dict)
}

fn control_to_py<'py>(py: Python<'py>, control: &Control) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    let value = PyDict::new(py);
    match control {
        Control::Commit {
            request_key,
            control_seq,
            expected_parent,
            selected,
            public_event_limit,
            disposition,
        } => {
            dict.set_item(intern!(py, "kind"), intern!(py, "commit"))?;
            value.set_item(intern!(py, "request_key"), request_key_to_py(py, *request_key)?)?;
            value.set_item(intern!(py, "control_seq"), *control_seq)?;
            value.set_item(
                intern!(py, "expected_parent"),
                version_ref_to_py(py, expected_parent)?,
            )?;
            value.set_item(intern!(py, "selected"), version_ref_to_py(py, selected)?)?;
            value.set_item(intern!(py, "public_event_limit"), *public_event_limit)?;
            let disposition = match disposition {
                Disposition::Publish => intern!(py, "publish"),
                Disposition::Retain => intern!(py, "retain"),
                Disposition::Discard => intern!(py, "discard"),
            };
            value.set_item(intern!(py, "disposition"), disposition)?;
        }
        Control::Close {
            request_key,
            control_seq,
            cutoff,
            reason,
        } => {
            dict.set_item(intern!(py, "kind"), intern!(py, "close"))?;
            value.set_item(intern!(py, "request_key"), request_key_to_py(py, *request_key)?)?;
            value.set_item(intern!(py, "control_seq"), *control_seq)?;
            value.set_item(intern!(py, "cutoff"), version_ref_to_py(py, cutoff)?)?;
            let reason = match reason {
                CloseReason::Completed => intern!(py, "completed"),
                CloseReason::Cancelled => intern!(py, "cancelled"),
                CloseReason::Error => intern!(py, "error"),
                CloseReason::Preempted => intern!(py, "preempted"),
            };
            value.set_item(intern!(py, "reason"), reason)?;
        }
        Control::Release { request_key, op_id } => {
            dict.set_item(intern!(py, "kind"), intern!(py, "release"))?;
            value.set_item(intern!(py, "request_key"), request_key_to_py(py, *request_key)?)?;
            value.set_item(intern!(py, "op_id"), op_id.0)?;
        }
    }
    dict.set_item(intern!(py, "value"), value)?;
    Ok(dict)
}

fn product_payload_to_py<'py>(
    py: Python<'py>,
    payload: &ProductPayload,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "product"), product_ref_to_py(py, &payload.product)?)?;
    // `serde_bytes` pythonizes to `bytes`; one buffer copy, no per-element walk.
    dict.set_item(intern!(py, "bytes"), PyBytes::new(py, &payload.bytes))?;
    Ok(dict)
}

fn snapshot_to_py<'py>(py: Python<'py>, snapshot: &SnapshotRef) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(intern!(py, "session_id"), snapshot.session_id.0)?;
    dict.set_item(intern!(py, "epoch"), snapshot.epoch)?;
    dict.set_item(intern!(py, "version"), snapshot.version)?;
    dict.set_item(intern!(py, "digest"), snapshot.digest.as_str())?;
    dict.set_item(intern!(py, "locator"), snapshot.locator.as_str())?;
    Ok(dict)
}

fn request_kind_py<'py>(py: Python<'py>, kind: RequestKind) -> &'py Bound<'py, PyString> {
    match kind {
        RequestKind::GetCapabilities => intern!(py, "get_capabilities"),
        RequestKind::Execute => intern!(py, "execute"),
        RequestKind::DropSession => intern!(py, "drop_session"),
        RequestKind::Shutdown => intern!(py, "shutdown"),
        RequestKind::CopyKv => intern!(py, "copy_kv"),
        RequestKind::LoadAdapter => intern!(py, "load_adapter"),
        RequestKind::UnloadAdapter => intern!(py, "unload_adapter"),
        RequestKind::ReleaseProducts => intern!(py, "release_products"),
        RequestKind::ResetPrefixCache => intern!(py, "reset_prefix_cache"),
        RequestKind::GetMetrics => intern!(py, "get_metrics"),
        RequestKind::GetPressure => intern!(py, "get_pressure"),
        RequestKind::SnapshotSession => intern!(py, "snapshot_session"),
        RequestKind::RestoreSession => intern!(py, "restore_session"),
    }
}

fn domain_py<'py>(py: Python<'py>, domain: Domain) -> &'py Bound<'py, PyString> {
    match domain {
        Domain::Und => intern!(py, "und"),
        Domain::Gen => intern!(py, "gen"),
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

fn draw_layout_py<'py>(py: Python<'py>, layout: DrawLayout) -> &'py Bound<'py, PyString> {
    match layout {
        DrawLayout::TargetSampling => intern!(py, "target_sampling"),
        DrawLayout::SpeculativeProposal => intern!(py, "speculative_proposal"),
        DrawLayout::FlowNoise => intern!(py, "flow_noise"),
    }
}

// ---------------------------------------------------------------------------
// Python -> Response (respond hot path)
// ---------------------------------------------------------------------------

/// Extract a `result` [`WorkerResponse`] from the dict shape the Python worker
/// emits (`app.py::_response` + `CompletionReport.to_wire`). Returns `None` on
/// any shape outside that contract; the caller then falls back to
/// `depythonize`, which reproduces the reflective values and errors exactly.
/// Where this extractor is stricter than `depythonize` (bools must be `bool`,
/// ints must not be `bool`), the fallback — not a panic — decides the outcome.
pub(crate) fn try_completion_response_from_py(
    response: &Bound<'_, PyAny>,
) -> Option<WorkerResponse> {
    let py = response.py();
    let dict = response.cast::<PyDict>().ok()?;
    let kind = str_field(dict, intern!(py, "kind"))?;
    if kind.to_str().ok()? != "result" {
        return None;
    }
    // Fields a `result` response never populates: accept only absent/None so
    // anything unexpected falls back to the reflective path.
    for key in [
        intern!(py, "capabilities"),
        intern!(py, "metrics"),
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
        metrics: None,
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
    // The Python worker never attaches forward stats to a completion report;
    // anything but absent/None goes back through the reflective path.
    if !absent_or_none(dict, intern!(py, "forward_stats"))? {
        return None;
    }
    Some(CompletionReport {
        step_id: u64_of(&get(dict, intern!(py, "step_id"))?)?,
        completions: records,
        products: payloads,
        registration,
        worker_exec_us: opt_u64(dict, intern!(py, "worker_exec_us"))?,
        forward_stats: None,
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
        completion_slot_generation: u32_of(&get(
            dict,
            intern!(py, "completion_slot_generation"),
        )?)?,
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
    // `depythonize` rejects bools for integer fields; extracting them as 0/1
    // here would silently diverge, so route them to the reflective fallback.
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
// Equality proofs and micro-benchmarks
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use pythonize::{depythonize, pythonize};
    use uniserve_core::{BlockId, RequestId};
    use uniserve_worker_wire::{ResourcePressure, WorkVariant, WorkerMetrics};

    use super::*;

    const PRODUCT_KINDS: [ProductKind; 9] = [
        ProductKind::Token,
        ProductKind::Logprob,
        ProductKind::Draft,
        ProductKind::VisionFeature,
        ProductKind::LatentFeature,
        ProductKind::Kv,
        ProductKind::Latent,
        ProductKind::Artifact,
        ProductKind::Completion,
    ];
    const STORAGE_CLASSES: [StorageClass; 5] = [
        StorageClass::DeviceTensor,
        StorageClass::PagedKv,
        StorageClass::LatentArena,
        StorageClass::HostStaging,
        StorageClass::CompletionArena,
    ];
    const DTYPES: [DType; 8] = [
        DType::U8,
        DType::U16,
        DType::U32,
        DType::I32,
        DType::I64,
        DType::F16,
        DType::BF16,
        DType::F32,
    ];
    const DRAW_LAYOUTS: [DrawLayout; 3] = [
        DrawLayout::TargetSampling,
        DrawLayout::SpeculativeProposal,
        DrawLayout::FlowNoise,
    ];

    fn digest(seed: u64) -> String {
        format!("{seed:064x}")
    }

    fn request_key(seed: u64) -> RequestKey {
        RequestKey::new(1_000 + seed, RequestId(2_000 + seed), seed % 3)
    }

    fn product_ref(seed: u64) -> ProductRef {
        let dims = match seed % 3 {
            0 => vec![DimBound::Static(4), DimBound::Device { max: 64 + seed as u32 }],
            1 => vec![DimBound::Static(seed as u32)],
            _ => Vec::new(),
        };
        ProductRef {
            request_key: request_key(seed),
            producer_op_id: OpId(300 + seed),
            output_index: (seed % 7) as u16,
            generation: (seed % 5) as u32,
            kind: PRODUCT_KINDS[seed as usize % PRODUCT_KINDS.len()],
            storage_class: STORAGE_CLASSES[seed as usize % STORAGE_CLASSES.len()],
            dtype: DTYPES[seed as usize % DTYPES.len()],
            shape_bound: ShapeBound { dims },
            point_range: PointRange {
                base_point: seed as u32,
                max_points: 4 + seed as u32 % 3,
            },
        }
    }

    fn fixed_parent(seed: u64) -> VersionRef {
        VersionRef {
            request_key: request_key(seed),
            producer_op_id: OpId(400 + seed),
            point: Point::Fixed {
                point_index: seed as u32,
                semantic_digest: digest(seed),
            },
        }
    }

    fn device_parent(seed: u64) -> VersionRef {
        VersionRef {
            request_key: request_key(seed),
            producer_op_id: OpId(500 + seed),
            point: Point::Device {
                selected_point: product_ref(seed),
                producer_plan_digest: digest(seed + 1),
            },
        }
    }

    fn full_sampling() -> SamplingParams {
        SamplingParams {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            ignore_eos: true,
            seed: Some(u64::MAX),
            min_p: 0.05,
            repetition_penalty: 1.1,
            frequency_penalty: 0.25,
            presence_penalty: -0.5,
            logit_bias: vec![(11, 1.5), (u32::MAX, -100.0)],
            min_tokens: 3,
            return_logprobs: true,
            n_logprobs: 5,
            return_prompt_logprobs: true,
            n_prompt_logprobs: 2,
            logprob_token_ids: vec![7, 8, 9],
            bad_words_ids: vec![vec![1, 2], vec![3]],
            allowed_token_ids: Some(vec![4, 5, 6]),
        }
    }

    fn full_image() -> ImageParams {
        ImageParams {
            steps: 28,
            cfg_text_scale: 4.0,
            cfg_img_scale: 1.5,
            cfg_renorm_type: "global".to_owned(),
            cfg_renorm_min: 0.125,
            cfg_interval: (0.4, 1.0),
            timestep_shift: 3.0,
            height: 1024,
            width: 512,
            seed: Some(42),
            negative_prompt: "blurry, low quality".to_owned(),
            max_images: 2,
            image_prompts: vec!["a cat".to_owned(), "on a mat".to_owned()],
            retain_images: false,
        }
    }

    fn operation(seed: u64, work: Work) -> Operation {
        let inputs = match seed % 3 {
            0 => vec![product_ref(seed + 10), product_ref(seed + 11)],
            1 => vec![product_ref(seed + 12)],
            _ => Vec::new(),
        };
        let parent = if seed.is_multiple_of(2) {
            fixed_parent(seed)
        } else {
            device_parent(seed)
        };
        let rng = (seed.is_multiple_of(2)).then(|| Rng {
            seed: 900 + seed,
            semantic_index_base: seed * 17,
            draw_layout: DRAW_LAYOUTS[seed as usize % DRAW_LAYOUTS.len()],
        });
        let predicate = (seed.is_multiple_of(4)).then(|| product_ref(seed + 20));
        let new_kv_blocks = if seed.is_multiple_of(2) {
            vec![BlockId(seed as u32), BlockId(seed as u32 + 1)]
        } else {
            Vec::new()
        };
        let key = request_key(seed);
        let mut outputs = vec![product_ref(seed + 30), product_ref(seed + 31)];
        for output in &mut outputs {
            output.request_key = key;
            output.producer_op_id = OpId(600 + seed);
        }
        Operation::registered(
            key,
            OpId(600 + seed),
            parent,
            work,
            uniserve_worker_wire::RouteId(seed as u32 % 4),
            if seed.is_multiple_of(2) { Domain::Und } else { Domain::Gen },
            Bounds {
                max_points: 4,
                max_tokens: 16,
                max_kv_pages: 2,
                max_latent_bytes: 1 << 20,
                max_completion_bytes: 4096,
                max_transfer_bytes: 1 << 22,
            },
            inputs,
            outputs,
            new_kv_blocks,
            predicate,
            rng,
            seed,
        )
    }

    /// A batch exercising every closed wire variant: all 12 work variants over
    /// fixed and device parents, und+gen admissions with every sampling and
    /// image field populated, all three control kinds across every disposition
    /// and close reason, and input products with non-trivial bytes.
    fn comprehensive_batch() -> Batch {
        let admissions = vec![
            Admission::new(
                request_key(0),
                Some(UndAdmission {
                    sampling: full_sampling(),
                    negative_token_ids: vec![100, 200],
                    kv: KvAllocation {
                        block_ids: vec![BlockId(1), BlockId(9)],
                        prefix_len: 64,
                        group_id: 1,
                    },
                }),
                None,
                Some(7),
            )
            .unwrap(),
            Admission::new(
                request_key(1),
                None,
                Some(GenAdmission { image: full_image() }),
                None,
            )
            .unwrap(),
            Admission::new(
                request_key(2),
                Some(UndAdmission {
                    sampling: SamplingParams::default(),
                    negative_token_ids: Vec::new(),
                    kv: KvAllocation::default(),
                }),
                Some(GenAdmission { image: full_image() }),
                None,
            )
            .unwrap(),
        ];
        let operations = WorkVariant::ALL
            .iter()
            .enumerate()
            .map(|(index, variant)| operation(index as u64, Work::from_variant(*variant)))
            .collect();
        let mut controls = vec![
            Control::Release {
                request_key: request_key(3),
                op_id: OpId(77),
            },
        ];
        for (index, disposition) in [
            Disposition::Publish,
            Disposition::Retain,
            Disposition::Discard,
        ]
        .into_iter()
        .enumerate()
        {
            controls.push(Control::Commit {
                request_key: request_key(index as u64),
                control_seq: 10 + index as u64,
                // Device points stay covered through operation parents; a
                // commit must select a fixed version to be protocol-valid.
                expected_parent: device_parent(index as u64),
                selected: fixed_parent(index as u64 + 1),
                public_event_limit: 1 << 30,
                disposition,
            });
        }
        for (index, reason) in [
            CloseReason::Completed,
            CloseReason::Cancelled,
            CloseReason::Error,
            CloseReason::Preempted,
        ]
        .into_iter()
        .enumerate()
        {
            controls.push(Control::Close {
                request_key: request_key(index as u64),
                control_seq: 20 + index as u64,
                cutoff: fixed_parent(index as u64 + 2),
                reason,
            });
        }
        Batch::new(11, admissions, operations)
            .with_controls(controls)
            .with_input_products(vec![
                ProductPayload {
                    product: product_ref(90),
                    bytes: (0..=255).collect(),
                },
                ProductPayload {
                    product: product_ref(91),
                    bytes: Vec::new(),
                },
            ])
    }

    fn assert_matches_pythonize(py: Python<'_>, request: &WorkerRequest) {
        let reflective = pythonize(py, request).unwrap();
        let hand_rolled = execute_request_to_py(py, request).unwrap();
        assert!(
            hand_rolled.eq(&reflective).unwrap(),
            "hand-rolled dict diverges from pythonize:\n hand: {hand_rolled}\n refl: {reflective}",
        );
    }

    #[test]
    fn execute_request_matches_pythonize() {
        Python::initialize();
        Python::attach(|py| {
            let mut request = WorkerRequest::execute(comprehensive_batch());
            request.call_id = Some(u64::MAX);
            assert_matches_pythonize(py, &request);
        });
    }

    #[test]
    fn execute_request_matches_pythonize_with_every_side_field() {
        Python::initialize();
        Python::attach(|py| {
            // Production execute frames leave these None; the converter is
            // still total over the WorkerRequest struct.
            let mut request = WorkerRequest::execute(comprehensive_batch());
            request.call_id = Some(3);
            request.session_id = Some(RequestId(u64::MAX));
            request.copies = Some(vec![(BlockId(1), BlockId(2)), (BlockId(3), BlockId(4))]);
            request.adapter_id = Some(5);
            request.adapter_path = Some("/adapters/a".to_owned());
            request.product_handles = Some(vec![1, u64::MAX]);
            request.snapshot = Some(SnapshotRef {
                session_id: RequestId(6),
                epoch: 7,
                version: 8,
                digest: digest(9),
                locator: "snap://9".to_owned(),
            });
            assert_matches_pythonize(py, &request);
        });
    }

    #[test]
    fn execute_request_matches_pythonize_when_bare() {
        Python::initialize();
        Python::attach(|py| {
            let mut request = WorkerRequest::execute(Batch::new(
                0,
                Vec::new(),
                vec![operation(1, Work::Token(TokenMode::Extend))],
            ));
            assert_matches_pythonize(py, &request);
            request.batch = None;
            assert_matches_pythonize(py, &request);
        });
    }

    /// The hand-rolled dict must be consumable by the worker's real decoder,
    /// `uniserve_worker.batch.Batch.from_wire`, which re-validates every
    /// Rust-computed plan and admission digest byte-for-byte in Python. This
    /// closes the loop across the actual language boundary contract, not just
    /// against `pythonize`.
    #[test]
    fn hand_rolled_dict_feeds_worker_from_wire() {
        Python::initialize();
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
            let batch_type = py
                .import("uniserve_worker.batch")
                .unwrap()
                .getattr("Batch")
                .unwrap();
            let request = WorkerRequest::execute(comprehensive_batch());
            let dict = execute_request_to_py(py, &request).unwrap();
            let wire_batch = dict.get_item("batch").unwrap().unwrap();
            let decoded = batch_type
                .call_method1("from_wire", (wire_batch,))
                .unwrap_or_else(|err| panic!("worker rejected the hand-rolled batch: {err}"));
            let step_id: u64 = decoded.getattr("step_id").unwrap().extract().unwrap();
            assert_eq!(step_id, 11);
            let operations = decoded.getattr("operations").unwrap();
            assert_eq!(operations.len().unwrap(), WorkVariant::ALL.len());
        });
    }

    fn completion_record(seed: u64, status: OpStatus) -> CompletionRecord {
        CompletionRecord {
            request_key: request_key(seed),
            op_id: OpId(700 + seed),
            completion_slot_generation: seed as u32 % 4,
            status,
            selected_point: seed as u32 % 3,
            logical_lengths: LogicalLengths {
                token_len: 100 + seed as u32,
                kv_visible_len: 200 + seed as u32,
                latent_len: seed as u32 % 2,
            },
            token_span: TokenSpan {
                base: 100 + seed as u32,
                len: 1,
            },
            committed_tokens: vec![10_000 + seed as u32],
            finish_flags: FinishFlags {
                eos: seed.is_multiple_of(2),
                length: seed.is_multiple_of(3),
                stop: seed.is_multiple_of(5),
            },
            product_generations: vec![seed as u32, seed as u32 + 1],
            semantic_digest: digest(800 + seed),
            error_code: (status == OpStatus::Error).then_some(ErrorCode::ComputeError),
            timing_counters: TimingCounters {
                queued_us: seed,
                device_us: 2 * seed,
                copy_us: 3 * seed,
                host_us: u64::MAX - seed,
            },
        }
    }

    fn completion_response(records: usize) -> WorkerResponse {
        let error_codes = [
            ErrorCode::InvalidOperation,
            ErrorCode::ResourceExhausted,
            ErrorCode::ComputeError,
            ErrorCode::Cancelled,
            ErrorCode::Internal,
        ];
        let completions = (0..records)
            .map(|index| {
                let status = match index % 3 {
                    0 => OpStatus::Ok,
                    1 => OpStatus::Predicated,
                    _ => OpStatus::Error,
                };
                let mut record = completion_record(index as u64, status);
                if status == OpStatus::Error {
                    record.error_code = Some(error_codes[index % error_codes.len()]);
                }
                record
            })
            .collect();
        let mut response = WorkerResponse::completion_report(CompletionReport {
            step_id: 42,
            completions,
            products: vec![
                ProductPayload {
                    product: product_ref(95),
                    bytes: vec![1, 2, 3, 254, 255],
                },
                ProductPayload {
                    product: product_ref(96),
                    bytes: Vec::new(),
                },
            ],
            registration: RegistrationAck { visible: true },
            worker_exec_us: Some(1234),
            forward_stats: None,
        });
        response.call_id = Some(9);
        response
    }

    /// The dict shape `app.py::_finalize_response` actually emits: identical
    /// to `pythonize(&WorkerResponse)` except the Python `to_wire` writes no
    /// `forward_stats` key at all.
    fn python_worker_shaped_dict<'py>(
        py: Python<'py>,
        response: &WorkerResponse,
    ) -> Bound<'py, PyAny> {
        let dict = pythonize(py, response).unwrap();
        let report = dict
            .cast::<PyDict>()
            .unwrap()
            .get_item("completion_report")
            .unwrap()
            .unwrap();
        report
            .cast::<PyDict>()
            .unwrap()
            .del_item("forward_stats")
            .unwrap();
        dict
    }

    #[test]
    fn result_extractor_matches_depythonize() {
        Python::initialize();
        Python::attach(|py| {
            for response in [completion_response(6), WorkerResponse::completion_report(
                CompletionReport {
                    step_id: 0,
                    completions: Vec::new(),
                    products: Vec::new(),
                    registration: RegistrationAck::default(),
                    worker_exec_us: None,
                    forward_stats: None,
                },
            )] {
                for dict in [
                    pythonize(py, &response).unwrap(),
                    python_worker_shaped_dict(py, &response),
                ] {
                    let reflective: WorkerResponse = depythonize(&dict).unwrap();
                    let extracted = try_completion_response_from_py(&dict)
                        .expect("extractor must accept the worker's result shape");
                    assert_eq!(extracted, reflective);
                    assert_eq!(extracted, response);
                }
            }
        });
    }

    #[test]
    fn result_extractor_accepts_report_none() {
        Python::initialize();
        Python::attach(|py| {
            let mut response = WorkerResponse::ok();
            response.kind = ResponseKind::Result;
            let dict = pythonize(py, &response).unwrap();
            let reflective: WorkerResponse = depythonize(&dict).unwrap();
            let extracted = try_completion_response_from_py(&dict).unwrap();
            assert_eq!(extracted, reflective);
        });
    }

    #[test]
    fn extractor_falls_back_on_non_result_kinds() {
        Python::initialize();
        Python::attach(|py| {
            let mut error = WorkerResponse::ok();
            error.kind = ResponseKind::Error;
            error.message = Some("boom".to_owned());
            error.code = Some("internal".to_owned());
            error.retryable = Some(false);
            error.fatal = Some(true);
            error.phase = Some("execute".to_owned());
            error.route = Some("decode".to_owned());
            error.operations = vec![ErrorOperationIdentity {
                request_key: request_key(1),
                op_id: OpId(5),
            }];
            let mut metrics = WorkerResponse::ok();
            metrics.kind = ResponseKind::Metrics;
            metrics.metrics = Some(WorkerMetrics::default());
            let mut pressure = WorkerResponse::ok();
            pressure.kind = ResponseKind::Pressure;
            pressure.pressure = Some(vec![ResourcePressure {
                class: uniserve_worker_wire::ResourceClass::KvBlock,
                total: 10,
                used: 5,
                evictable: 3,
                free: 2,
            }]);
            for response in [error, WorkerResponse::ok(), metrics, pressure] {
                let dict = pythonize(py, &response).unwrap();
                assert!(try_completion_response_from_py(&dict).is_none());
                // The reflective fallback still round-trips the value.
                let reflective: WorkerResponse = depythonize(&dict).unwrap();
                assert_eq!(reflective, response);
            }
        });
    }

    #[test]
    fn extractor_falls_back_on_unexpected_shapes() {
        Python::initialize();
        Python::attach(|py| {
            let response = completion_response(2);

            fn first_completion_record<'py>(dict: &Bound<'py, PyAny>) -> Bound<'py, PyDict> {
                dict.cast::<PyDict>()
                    .unwrap()
                    .get_item("completion_report")
                    .unwrap()
                    .unwrap()
                    .cast_into::<PyDict>()
                    .unwrap()
                    .get_item("completions")
                    .unwrap()
                    .unwrap()
                    .cast_into::<PyList>()
                    .unwrap()
                    .get_item(0)
                    .unwrap()
                    .cast_into::<PyDict>()
                    .unwrap()
            }

            // A bool where an integer belongs is never silently coerced by
            // the extractor.
            let dict = pythonize(py, &response).unwrap();
            let record = first_completion_record(&dict);
            record
                .set_item("committed_tokens", PyList::new(py, [true]).unwrap())
                .unwrap();
            assert!(try_completion_response_from_py(&dict).is_none());
            // The reflective fallback decides the outcome for that shape: it
            // extracts `True` as token 1, so falling back (rather than
            // coercing here) preserves depythonize's semantics exactly.
            let reflective: WorkerResponse = depythonize(&dict).unwrap();
            let report = reflective.completion_report.unwrap();
            assert_eq!(report.completions[0].committed_tokens, vec![1]);

            // A shape both paths reject: a string where tokens belong.
            let dict = pythonize(py, &response).unwrap();
            let record = first_completion_record(&dict);
            record
                .set_item("committed_tokens", PyList::new(py, ["x"]).unwrap())
                .unwrap();
            assert!(try_completion_response_from_py(&dict).is_none());
            assert!(depythonize::<WorkerResponse>(&dict).is_err());

            // A populated field a result never carries goes reflective.
            let dict = pythonize(py, &response).unwrap();
            dict.cast::<PyDict>()
                .unwrap()
                .set_item("metrics", PyDict::new(py))
                .unwrap();
            assert!(try_completion_response_from_py(&dict).is_none());

            // A missing required key goes reflective (which then errors).
            let dict = pythonize(py, &response).unwrap();
            dict.cast::<PyDict>().unwrap().del_item("operations").unwrap();
            assert!(try_completion_response_from_py(&dict).is_none());
            assert!(depythonize::<WorkerResponse>(&dict).is_err());

            // Non-dict input goes reflective.
            let list = PyList::new(py, [1, 2]).unwrap();
            assert!(try_completion_response_from_py(list.as_any()).is_none());
        });
    }

    /// A steady-state r16 decode step: 35 token-decode operations over device
    /// parents, a couple of controls, and small forced-token input payloads.
    fn decode_batch_request(ops: u64) -> WorkerRequest {
        let operations = (0..ops)
            .map(|index| {
                let seed = 1 + index;
                let key = request_key(seed);
                let mut outputs = vec![product_ref(seed + 30), product_ref(seed + 31)];
                for output in &mut outputs {
                    output.request_key = key;
                    output.producer_op_id = OpId(600 + seed);
                }
                Operation::registered(
                    key,
                    OpId(600 + seed),
                    device_parent(seed),
                    Work::Token(TokenMode::Decode),
                    uniserve_worker_wire::RouteId(0),
                    Domain::Und,
                    Bounds {
                        max_points: 1,
                        max_tokens: 1,
                        max_kv_pages: 1,
                        max_latent_bytes: 0,
                        max_completion_bytes: 4096,
                        max_transfer_bytes: 0,
                    },
                    Vec::new(),
                    outputs,
                    if seed.is_multiple_of(4) {
                        vec![BlockId(seed as u32)]
                    } else {
                        Vec::new()
                    },
                    None,
                    Some(Rng {
                        seed,
                        semantic_index_base: seed * 3,
                        draw_layout: DrawLayout::TargetSampling,
                    }),
                    seed,
                )
            })
            .collect();
        let controls = vec![
            Control::Commit {
                request_key: request_key(1),
                control_seq: 5,
                expected_parent: fixed_parent(1),
                selected: fixed_parent(2),
                public_event_limit: 128,
                disposition: Disposition::Publish,
            },
            Control::Release {
                request_key: request_key(2),
                op_id: OpId(11),
            },
        ];
        let input_products = (0..4)
            .map(|index| ProductPayload {
                product: product_ref(80 + index),
                bytes: vec![0xAB; 8],
            })
            .collect();
        WorkerRequest::execute(
            Batch::new(77, Vec::new(), operations)
                .with_controls(controls)
                .with_input_products(input_products),
        )
    }

    /// Micro-benchmark for the 35-op decode shapes. Run explicitly:
    /// `cargo test -p uniserve-ipc-py --release -- bench_35 --ignored --nocapture`
    #[test]
    #[ignore = "micro-benchmark; run with --release --ignored --nocapture"]
    fn bench_35_op_decode_shapes() {
        const OPS: u64 = 35;
        const ITERS: u32 = 400;
        Python::initialize();
        Python::attach(|py| {
            let request = decode_batch_request(OPS);
            assert_matches_pythonize(py, &request);
            let response = completion_response(OPS as usize);
            let response_dict = python_worker_shaped_dict(py, &response);
            assert_eq!(
                try_completion_response_from_py(&response_dict).unwrap(),
                depythonize::<WorkerResponse>(&response_dict).unwrap()
            );

            let time = |label: &str, mut run: Box<dyn FnMut() + '_>| {
                for _ in 0..ITERS / 4 {
                    run();
                }
                let started = std::time::Instant::now();
                for _ in 0..ITERS {
                    run();
                }
                let nanos = started.elapsed().as_nanos() / u128::from(ITERS);
                println!("{label}: {nanos} ns/conversion");
            };
            time(
                "recv  pythonize (reflective)",
                Box::new(|| {
                    pythonize(py, &request).unwrap();
                }),
            );
            time(
                "recv  hand-rolled",
                Box::new(|| {
                    execute_request_to_py(py, &request).unwrap();
                }),
            );
            time(
                "send  depythonize (reflective)",
                Box::new(|| {
                    depythonize::<WorkerResponse>(&response_dict).unwrap();
                }),
            );
            time(
                "send  hand-rolled extractor",
                Box::new(|| {
                    try_completion_response_from_py(&response_dict).unwrap();
                }),
            );
        });
    }
}
