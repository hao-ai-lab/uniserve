//! Page, product and transfer reservations for token and media workers.

use std::collections::{HashMap, HashSet};

use super::*;

pub(super) fn media_builder<'py>(
    model: &Bound<'py, PyAny>,
    config: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    model
        .py()
        .import("uniserve_worker.bootstrap.inputs")?
        .call_method1("media_builder", (model, config))
}

pub(super) fn image_builder<'py>(model: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    model
        .py()
        .import("uniserve_worker.bootstrap.inputs")?
        .call_method1("image_builder", (model,))
}

pub(super) fn outputs<'py>(
    model: &Bound<'py, PyAny>,
    config: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    model
        .py()
        .import("uniserve_worker.bootstrap.outputs")?
        .call_method1("resolve_outputs", (model, config))
}

pub(super) fn media_components<'py>(
    model: &Bound<'py, PyAny>,
    config: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    model
        .py()
        .import("uniserve_worker.bootstrap.components")?
        .call_method1(
            "media_components",
            (
                model,
                &crate::worker::config::native(config)?.deployment_components,
            ),
        )
}

pub(super) fn scalar_bytes(py: Python<'_>) -> PyResult<u64> {
    let options = PyDict::new(py);
    options.set_item("max_value_bytes", 1)?;
    let bytes = py
        .import("uniserve_worker.storage.tensor_store")?
        .call_method("device_product_capacity_bytes", (1, 1), Some(&options))?
        .extract::<u64>()?;
    Ok(bytes - 1)
}

pub(super) fn product_storage_bytes(entry_outputs: &Bound<'_, PyAny>) -> PyResult<u64> {
    let mut bytes = 0;
    for outputs in entry_outputs.call_method0("values")?.try_iter()? {
        bytes += aligned_outputs(&outputs?)?;
    }
    Ok(bytes)
}

fn aligned_outputs(outputs: &Bound<'_, PyAny>) -> PyResult<u64> {
    let mut bytes = 0;
    for output in outputs.try_iter()? {
        bytes += output?
            .getattr("max_bytes")?
            .extract::<u64>()?
            .div_ceil(256)
            * 256;
    }
    Ok(bytes)
}

#[pyfunction]
#[pyo3(signature = (entry_outputs, *, bindings, media_components))]
pub(super) fn local_product_storage_bytes(
    entry_outputs: &Bound<'_, PyAny>,
    bindings: &Bound<'_, PyAny>,
    media_components: &Bound<'_, PyAny>,
) -> PyResult<u64> {
    if !bindings.is_truthy()? {
        return product_storage_bytes(entry_outputs);
    }

    let components: HashMap<String, String> = media_components.extract()?;
    let mut consumers: HashMap<&str, HashSet<&str>> = HashMap::new();
    for (source, destination) in [
        ("media_reading", "vision_encoding"),
        ("media_reading", "latent_encoding"),
        ("vision_encoding", "text_encoding"),
        ("latent_encoding", "latent_preparation"),
        ("text_encoding", "latent_preparation"),
        ("denoising", "video_decoding"),
        ("denoising", "audio_decoding"),
        ("video_decoding", "video_encoding"),
        ("audio_decoding", "audio_encoding"),
        ("video_encoding", "muxing"),
    ] {
        if let (Some(source), Some(destination)) =
            (components.get(source), components.get(destination))
        {
            consumers.entry(source).or_default().insert(destination);
        }
    }

    let mut bytes = 0;
    for item in entry_outputs.call_method0("items")?.try_iter()? {
        let item = item?;
        let name: String = item.get_item(0)?.extract()?;
        let producer = bindings.call_method1("get", (&name,))?;
        let produces = !producer.is_none()
            && producer
                .getattr("output_ranks")?
                .contains(producer.getattr("process_group")?.getattr("rank")?)?;
        let mut consumes = false;
        if let Some(consumers) = consumers.get(name.as_str()) {
            for consumer in consumers {
                let binding = bindings.call_method1("get", (consumer,))?;
                if !binding.is_none() && binding.getattr("owns")?.is_truthy()? {
                    consumes = true;
                    break;
                }
            }
        }

        // Streaming changes transfer granularity, not the extent retained by
        // a local producer or consumer until its last region has retired.
        if produces || consumes {
            bytes += aligned_outputs(&item.get_item(1)?)?;
        }
    }
    Ok(bytes)
}

pub(super) fn latent_pool_plan(
    model: &Bound<'_, PyAny>,
    worker_config: &Bound<'_, PyAny>,
) -> PyResult<Option<LatentPoolPlan>> {
    let worker_config_native = crate::worker::config::native(worker_config)?;
    let py = model.py();
    let slots = worker_config_native.max_request_pool_size;
    let flow = image_builder(model)?;
    if !flow.is_none() {
        let dtype_name: String = worker_config_native.model_dtype.clone();
        let torch = py.import("torch")?;
        let dtype = torch.getattr(dtype_name.trim_start_matches("torch."))?;
        if !dtype.is_instance(&torch.getattr("dtype")?)? {
            return Err(unsupported(
                py,
                format!("unsupported latent dtype {dtype_name:?}"),
            ));
        }

        let per_image: usize = flow.getattr("max_tokens")?.extract()?;
        let budget: Option<usize> = worker_config_native.kv_token_capacity;
        let units = if per_image == 0 {
            0
        } else {
            per_image.max(budget.unwrap_or(per_image))
        };
        let page_units = worker_config_native
            .block_size
            .ok_or_else(|| unsupported(py, "KV page size has not been resolved"))?;
        let denoiser = flow.getattr("denoiser")?;
        let channels: usize = denoiser.getattr("latent_channels")?.extract()?;
        let patch: usize = denoiser.getattr("patch_size")?.extract()?;
        return Ok(Some(LatentPoolPlan {
            request_pool_size: slots,
            num_pages: units.div_ceil(page_units) + 1,
            page_units,
            latent_width: channels * patch * patch,
            dtype: dtype.unbind(),
            with_workspace: true,
        }));
    }

    let builder = media_builder(model, worker_config)?;
    if builder.is_none() {
        return Ok(None);
    }
    let pages = builder.getattr("sample_pages")?;
    Ok(Some(LatentPoolPlan {
        request_pool_size: slots,
        num_pages: slots * pages.getattr("pages")?.extract::<usize>()? + 1,
        page_units: pages.getattr("page_units")?.extract()?,
        latent_width: 1,
        dtype: pages.getattr("dtype")?.unbind(),
        with_workspace: false,
    }))
}

pub(super) fn artifact_import_regions(
    model: &Bound<'_, PyAny>,
    worker_config: &Bound<'_, PyAny>,
    bindings: &Bound<'_, PyAny>,
) -> PyResult<usize> {
    let routes = media_components(model, worker_config)?;
    let encoder = routes.call_method1("get", ("video_encoding",))?;
    let binding = bindings.call_method1("get", (encoder,))?;
    let decoder = capability(model, "VideoDecoder")?;
    let builder = media_builder(model, worker_config)?;
    if binding.is_none() || decoder.is_none() || builder.is_none() {
        return Ok(0);
    }

    let ranks = binding.getattr("config")?.getattr("ranks")?.len()?;
    let units = decoder
        .call_method1(
            "frame_slices",
            (builder.getattr("maximum")?.getattr("num_frames")?,),
        )?
        .len()?;
    Ok(units.div_ceil(ranks) * ranks.saturating_sub(1))
}

#[pyfunction]
#[pyo3(signature = (model, worker_config, *, queue_depth, capacity_group, bindings=None, state_buffers=None))]
pub(in crate::worker) fn resolve_request_capacity<'py>(
    model: &Bound<'py, PyAny>,
    worker_config: &Bound<'py, PyAny>,
    queue_depth: usize,
    capacity_group: Option<&Bound<'py, PyAny>>,
    bindings: Option<&Bound<'py, PyAny>>,
    state_buffers: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let worker_config_native = crate::worker::config::native(worker_config)?;
    let py = model.py();
    let device = worker_config_native
        .device
        .as_str()
        .into_pyobject(py)?
        .into_any();
    if !is_cuda(&device)? {
        return Ok(worker_config.clone());
    }

    let available: u64 = py
        .import("uniserve.runtime.device")?
        .call_method1(
            "device_storage_budget",
            (&device, worker_config_native.kv_storage_fraction),
        )?
        .get_item(0)?
        .extract()?;
    let mut resolved = (*worker_config_native).clone();
    resolved.pool_storage_bytes = Some(available as usize);
    let empty = PyDict::new(py).into_any();
    let bindings = bindings.unwrap_or(&empty);
    let builder = media_builder(model, worker_config)?;
    let resources = py.import("uniserve_worker.model_executor.resources")?;
    let schema = match state_buffers {
        Some(schema) => schema.clone(),
        None => resources.call_method1("media_state_buffers", (bindings, &builder))?,
    };
    if schema.is_truthy()? || !capability(model, "VideoPostprocessor")?.is_none() {
        let group = capacity_group
            .ok_or_else(|| unsupported(py, "request tensor sizing requires its rank group"))?;
        let maximum = worker_config_native
            .max_request_pool_size
            .min(queue_depth / 3);
        let minimum = worker_config_native.min_request_pool_size;
        if minimum == 0 || maximum < minimum {
            return Err(PyValueError::new_err(
                "request tensor capacity requires valid slot bounds",
            ));
        }

        // Numerical layouts and placement do not vary with request concurrency.
        // Resolve them once, then price every candidate using native scalars.
        let state_bytes = schema_bytes(&schema, true)?;
        let product_bytes = local_product_storage_bytes(
            &outputs(model, worker_config)?,
            bindings,
            &media_components(model, worker_config)?,
        )?;
        let plan = if resources
            .call_method1("holds_samples", (&schema, &builder))?
            .is_truthy()?
        {
            latent_pool_plan(model, worker_config)?
        } else {
            None
        };
        let element_bytes = plan
            .as_ref()
            .map(|plan| plan.dtype.bind(py).getattr("itemsize")?.extract())
            .transpose()?
            .unwrap_or(0);
        let scalar_bytes = scalar_bytes(py)?;
        let calls = worker_config_native.max_batch_calls;
        let ranks = worker_config_native.world_size;
        let mut requirements = Vec::with_capacity(maximum - minimum + 1);
        for count in minimum..=maximum {
            let arena = native::ArenaCapacity::for_requests(
                queue_depth,
                calls.min(count),
                count,
                ranks,
                scalar_bytes,
                product_bytes,
                0,
                0,
            )
            .map_err(value_error)?;
            let latent_bytes = if let Some(plan) = &plan {
                let pages = if plan.with_workspace {
                    plan.num_pages
                } else {
                    count * ((plan.num_pages - 1) / plan.request_pool_size) + 1
                };
                native::latent_pool_bytes(
                    count,
                    pages,
                    plan.page_units,
                    plan.latent_width,
                    element_bytes,
                    plan.with_workspace,
                )
                .map_err(value_error)?
            } else {
                0
            };
            requirements.push(
                count as u64 * (state_bytes + product_bytes)
                    + arena.device_product_bytes
                    + latent_bytes,
            );
        }

        let slots = tensor_slot_capacity(requirements, group, minimum, available)?;
        resolved.max_request_pool_size = slots;
        resolved.max_batch_calls = calls.min(slots);
        resolved.max_batch_tokens = worker_config_native.max_batch_tokens.min(slots);
    }
    crate::worker::config::updated(worker_config, resolved)
}

/// Media slots reserve two unresolved outputs and one retirement position.
#[pyfunction]
pub(super) fn loaded_worker_config<'py>(
    model: &Bound<'py, PyAny>,
    config: &Bound<'py, PyAny>,
    queue_depth: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let config_native = crate::worker::config::native(config)?;
    let py = model.py();
    let decoder = py.import("uniserve.model")?.getattr("VideoDecoder")?;
    let mut media = false;
    for module in model.call_method0("modules")?.try_iter()? {
        if module?.is_instance(&decoder)? {
            media = true;
            break;
        }
    }
    if !media {
        return Ok(config.clone());
    }

    let slots = config_native.max_batch_calls.min(queue_depth / 3);
    if slots < 2 {
        return Err(unsupported(
            py,
            format!(
                "resident media execution requires two slots of three queue positions each; queue depth {queue_depth} holds {}",
                queue_depth / 3,
            ),
        ));
    }
    let mut resolved = (*config_native).clone();
    resolved.kv_token_capacity = None;
    resolved.attention_backend = None;
    resolved.generation_device = None;
    resolved.max_batch_calls = slots;
    resolved.max_batch_tokens = slots;
    resolved.max_request_pool_size = slots;
    resolved.min_request_pool_size = 2;
    crate::worker::config::updated(config, resolved)
}
