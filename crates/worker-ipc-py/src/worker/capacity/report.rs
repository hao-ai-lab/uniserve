//! One capacity report drives resource allocation and the engine handshake.

mod tokens;

use std::collections::{BTreeMap, BTreeSet};

use pythonize::pythonize;
use uniserve_worker_ipc::{CallKind, ComponentInfo, WorkerInfo};

use super::pools::*;
use super::*;
use crate::convert::{mapping_from_py, record_from_py, records_from_py};

/// Resolved rank-local backing and the logical capacity advertised to the engine.
#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct WorkerLayout {
    pub(in crate::worker) info: WorkerInfo,
    pub(in crate::worker) arena: native::ArenaCapacity,
    #[pyo3(get)]
    pub(in crate::worker) input_config: Option<Py<PyAny>>,
    #[pyo3(get)]
    pub(in crate::worker) fixed_device_bytes: Vec<(String, u64)>,
    #[pyo3(get)]
    pub(in crate::worker) physical_buffer_pool_bytes: u64,
    pub(in crate::worker) latent_plan: Option<LatentPoolPlan>,
}

#[pymethods]
impl WorkerLayout {
    #[getter]
    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        info_to_py(py, &self.info)
    }

    #[getter]
    fn arena(&self) -> ArenaCapacity {
        ArenaCapacity { inner: self.arena }
    }
}

pub(in crate::worker) fn info_to_py<'py>(
    py: Python<'py>,
    info: &WorkerInfo,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("uniserve_worker.protocol.worker_info")?
        .getattr("WorkerInfo")?
        .call_method1("from_mapping", (pythonize(py, info)?,))
}

/// Resolve storage before admission. Numerical layouts remain owned by the
/// model library; native dimensions below also feed the actual pool allocators.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (model, worker_config, *, image_processor=None, model_name=None,
    queue_depth=1, endpoint=None, capacity_group=None,
    allowed_calls=None, transfer_backends=vec!["local".to_owned()], components=None,
    attention_backend="", bindings=None, state_buffers=None))]
pub(in crate::worker) fn build_worker_layout(
    model: &Bound<'_, PyAny>,
    worker_config: &Bound<'_, PyAny>,
    image_processor: Option<&Bound<'_, PyAny>>,
    model_name: Option<String>,
    queue_depth: usize,
    endpoint: Option<&Bound<'_, PyAny>>,
    capacity_group: Option<&Bound<'_, PyAny>>,
    allowed_calls: Option<&Bound<'_, PyAny>>,
    transfer_backends: Vec<String>,
    components: Option<&Bound<'_, PyAny>>,
    attention_backend: &str,
    bindings: Option<&Bound<'_, PyAny>>,
    state_buffers: Option<&Bound<'_, PyAny>>,
) -> PyResult<WorkerLayout> {
    let py = model.py();
    let config = worker_config;
    let config_native = crate::worker::config::native(config)?;
    if queue_depth == 0 {
        return Err(unsupported(py, "worker pipeline depth must be positive"));
    }

    let empty = PyDict::new(py).into_any();
    let bindings = bindings.unwrap_or(&empty);
    let components = components
        .cloned()
        .unwrap_or_else(|| PyTuple::empty(py).into_any());
    let held: Vec<String> = components
        .try_iter()?
        .map(|item| item?.get_item(0)?.extract())
        .collect::<PyResult<_>>()?;
    let declarations = py.import("uniserve_worker.bootstrap.components")?;
    let dedicated = config_native.role == "experts";
    let supported: Vec<CallKind> = if dedicated {
        Vec::new()
    } else {
        let supported = declarations.call_method1("supported_calls", (model, &held))?;
        CallKind::ALL
            .into_iter()
            .filter_map(|kind| {
                let selected = supported.contains(kind.as_str()).and_then(|present| {
                    Ok(present
                        && allowed_calls
                            .map(|calls| calls.contains(kind.as_str()))
                            .transpose()?
                            .unwrap_or(true))
                });
                match selected {
                    Ok(true) => Some(Ok(kind)),
                    Ok(false) => None,
                    Err(error) => Some(Err(error)),
                }
            })
            .collect::<PyResult<_>>()?
    };
    if supported.is_empty() && !dedicated {
        return Err(unsupported(
            py,
            "worker model implements none of the requested call kinds",
        ));
    }

    let endpoint = match endpoint {
        Some(endpoint) => record_from_py(endpoint)?,
        None => {
            let options = PyDict::new(py);
            options.set_item("rank", config_native.rank)?;
            record_from_py(
                &py.import("uniserve_worker.protocol.transfer")?
                    .getattr("WorkerEndpoint")?
                    .call_method("local", (), Some(&options))?,
            )?
        }
    };
    let model_name = match model_name {
        Some(name) => name,
        None => format!(
            "{}.{}",
            model.get_type().getattr("__module__")?,
            model.get_type().getattr("__qualname__")?
        ),
    };
    let entry_outputs = outputs(model, config)?;
    let builder = media_builder(model, config)?;
    let state_buffers = match state_buffers {
        Some(schema) => schema.clone(),
        None => py
            .import("uniserve_worker.model_executor.resources")?
            .call_method1("media_state_buffers", (bindings, &builder))?,
    };
    let latent_plan = latent_pool_plan(model, config)?;
    let mut media = BTreeMap::new();
    if !dedicated {
        let routes = declarations.call_method1("media_components", (model, &held))?;
        for item in routes.call_method0("items")?.try_iter()? {
            let item = item?;
            let call = mapping_from_py(&item.get_item(0)?)?;
            if supported.contains(&CallKind::Media(call)) {
                media.insert(call, item.get_item(1)?.extract()?);
            }
        }
    }
    let mut placed = Vec::new();
    for item in components.try_iter()? {
        let item = item?;
        let name: String = item.get_item(0)?.extract()?;
        let output = entry_outputs.call_method1("get", (&name, PyTuple::empty(py)))?;
        placed.push(ComponentInfo {
            name,
            config: mapping_from_py(&item.get_item(1)?.call_method0("to_dict")?)?,
            outputs: records_from_py(&output)?,
        });
    }

    // Report loaded tensor formats, independently of checkpoint parameter names.
    let quantized = py
        .import("uniserve.quantization")?
        .getattr("QuantizedTensor")?;
    let linear = py.import("uniserve.nn")?.getattr("Linear")?;
    let mut weight_formats = BTreeSet::new();
    for value in model.call_method0("parameters")?.try_iter()? {
        let value = value?;
        weight_formats.insert(if value.is_instance(&quantized)? {
            value.getattr("quantizer")?.getattr("format")?.extract()?
        } else {
            "dense".to_owned()
        });
    }
    let mut activation_formats = BTreeSet::new();
    for child in model.call_method0("modules")?.try_iter()? {
        let child = child?;
        if child.is_instance(&linear)? {
            let quantizer = child.getattr("input_quantizer")?;
            activation_formats.insert(if quantizer.is_none() {
                "dense".to_owned()
            } else {
                quantizer.getattr("format")?.extract()?
            });
        }
    }
    let device = config_native.device.as_str().into_pyobject(py)?.into_any();
    let fabric_handles = if is_cuda(&device)? {
        let index = py
            .import("torch")?
            .getattr("device")?
            .call1((&device,))?
            .getattr("index")?
            .extract::<Option<usize>>()?
            .unwrap_or(0);
        py.import("uniserve_kernels.peer_storage")?
            .call_method1("exports_fabric_handles", (index,))?
            .extract()?
    } else {
        false
    };
    let backend = if attention_backend.is_empty() {
        config_native
            .attention_backend
            .clone()
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| "auto".into())
    } else {
        attention_backend.to_owned()
    };
    let mut info = WorkerInfo {
        model_name,
        endpoint,
        device: device.str()?.extract()?,
        world_size: u32::try_from(config_native.world_size)
            .map_err(|error| PyValueError::new_err(error.to_string()))?,
        model_dtype: config_native.model_dtype.clone(),
        attention_backend: backend,
        weight_formats: weight_formats.into_iter().collect(),
        activation_formats: activation_formats.into_iter().collect(),
        supported_calls: supported,
        components: placed,
        media_components: media,
        transfer_backends,
        fabric_handles,
        queue_depth: queue_depth as u32,
        max_batch_calls: u32::try_from(config_native.max_batch_calls)
            .map_err(|error| PyValueError::new_err(error.to_string()))?,
        max_batch_tokens: u32::try_from(config_native.max_batch_tokens)
            .map_err(|error| PyValueError::new_err(error.to_string()))?,
        request_slots: u32::try_from(config_native.max_request_pool_size)
            .map_err(|error| PyValueError::new_err(error.to_string()))?,
        max_prefill_calls: 0,
        max_decode_calls: 0,
        kv_cache: None,
        latent_page_units: latent_plan
            .as_ref()
            .map_or(0, |plan| plan.page_units as u32),
        latent_pages: latent_plan.as_ref().map_or(0, |plan| plan.num_pages as u32),
        buffer_pool_bytes: 0,
        encoder_cache_entries: 0,
        encoder_entry_bytes: 0,
        max_unresolved_calls: 0,
        num_inference_steps: 0,
        video_denoiser: None,
        host_lane_capacity: 1,
    };

    let mut layout =
        if !capability(model, "VideoPostprocessor")?.is_none() || state_buffers.is_truthy()? {
            request_layout(
                model,
                config,
                info,
                latent_plan,
                bindings,
                &state_buffers,
                &entry_outputs,
                &builder,
            )?
        } else {
            tokens::layout(
                model,
                config,
                info,
                latent_plan,
                image_processor,
                capacity_group,
                bindings,
                &entry_outputs,
            )?
        };
    info = layout.info;
    // Every eligible lane must be able to execute an admitted batch. Storage
    // still covers warmup and graph capture at their configured dimensions.
    for lane in &config_native.lanes {
        if let Some(calls) = lane.max_batch_calls {
            info.max_batch_calls = info.max_batch_calls.min(
                u32::try_from(calls).map_err(|error| PyValueError::new_err(error.to_string()))?,
            );
        }
        if let Some(tokens) = lane.max_batch_tokens {
            info.max_batch_tokens = info.max_batch_tokens.min(
                u32::try_from(tokens).map_err(|error| PyValueError::new_err(error.to_string()))?,
            );
        }
    }
    info.max_prefill_calls = info.max_prefill_calls.min(info.max_batch_calls);
    info.max_decode_calls = info.max_decode_calls.min(info.max_batch_calls);
    layout.info = info;
    Ok(layout)
}

#[allow(clippy::too_many_arguments)]
fn request_layout(
    model: &Bound<'_, PyAny>,
    config: &Bound<'_, PyAny>,
    mut info: WorkerInfo,
    latent_plan: Option<LatentPoolPlan>,
    bindings: &Bound<'_, PyAny>,
    state_buffers: &Bound<'_, PyAny>,
    outputs: &Bound<'_, PyAny>,
    builder: &Bound<'_, PyAny>,
) -> PyResult<WorkerLayout> {
    let py = model.py();
    let slots = info.request_slots as usize;
    info.max_unresolved_calls = native::request_tensor_window(info.queue_depth as usize, slots)
        .map_err(value_error)? as u32;
    info.max_batch_calls = info.max_batch_calls.min(info.request_slots);
    info.max_batch_tokens = info.max_batch_calls;
    info.buffer_pool_bytes = slots as u64 * product_storage_bytes(outputs)?;
    info.num_inference_steps = builder.getattr("num_steps")?.extract()?;
    info.video_denoiser = Some(record_from_py(
        &py.import("uniserve_worker.bootstrap.report")?
            .call_method1("video_denoiser_info", (builder.getattr("denoiser")?,))?,
    )?);
    let local = local_product_storage_bytes(outputs, bindings, &media_components(model, config)?)?;
    // Only ranks with diffusion request state back the latent pool. All ranks
    // advertise the same logical pages for the engine's request ledger.
    let latent_bytes = if py
        .import("uniserve_worker.model_executor.resources")?
        .call_method1("holds_samples", (state_buffers, builder))?
        .is_truthy()?
    {
        latent_plan
            .as_ref()
            .map(|plan| plan.capacity_bytes(py))
            .transpose()?
            .unwrap_or(0)
    } else {
        0
    };
    let arena = native::ArenaCapacity::for_requests(
        info.queue_depth as usize,
        info.max_batch_calls as usize,
        slots,
        info.world_size as usize,
        scalar_bytes(py)?,
        local,
        artifact_import_regions(model, config, bindings)?,
        latent_bytes,
    )
    .map_err(value_error)?;
    let host = py
        .import("uniserve_worker.bootstrap.components")?
        .call_method1("holds_host_components", (bindings,))?
        .is_truthy()?;
    info.host_lane_capacity = if host {
        1
    } else {
        arena.host_lane_inflight as u32
    };
    Ok(WorkerLayout {
        info,
        arena,
        input_config: None,
        fixed_device_bytes: Vec::new(),
        physical_buffer_pool_bytes: slots as u64 * local,
        latent_plan,
    })
}
