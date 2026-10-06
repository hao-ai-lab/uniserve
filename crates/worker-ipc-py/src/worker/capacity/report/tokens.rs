//! Token, canvas and image workers share one KV and fixed-storage budget.

use super::*;
use crate::worker::block_tables::GroupShape;
use crate::worker::graph_shapes;

#[allow(clippy::too_many_arguments)]
pub(super) fn layout(
    model: &Bound<'_, PyAny>,
    config: &Bound<'_, PyAny>,
    mut info: WorkerInfo,
    latent_plan: Option<LatentPoolPlan>,
    processor: Option<&Bound<'_, PyAny>>,
    group: Option<&Bound<'_, PyAny>>,
    bindings: &Bound<'_, PyAny>,
    outputs: &Bound<'_, PyAny>,
) -> PyResult<WorkerLayout> {
    let config_native = crate::worker::config::native(config)?;
    let py = model.py();
    let text = capability(model, "CausalLM")?;
    let flow = image_builder(model)?;
    let latent_bytes = latent_plan
        .as_ref()
        .map(|plan| plan.capacity_bytes(py))
        .transpose()?
        .unwrap_or(0);
    let dtype = py
        .import("torch")?
        .getattr(info.model_dtype.trim_start_matches("torch."))?;
    let latent_features = if flow.is_none() {
        0
    } else {
        flow.getattr("max_tokens")?.extract::<u64>()?
            * latent_plan
                .as_ref()
                .map_or(0, |plan| plan.latent_width as u64)
            * dtype.getattr("itemsize")?.extract::<u64>()?
    };
    let vision_tokens: u64 = py
        .import("uniserve_worker.bootstrap.capacity")?
        .call_method1("vision_tokens", (model, processor))?
        .extract()?;
    let vision = capability(model, "PatchEncoder")?;
    let vision_features = if vision.is_none() || vision_tokens == 0 {
        0
    } else {
        let stride = vision.getattr("patch_size")?.extract::<usize>()?
            * vision.getattr("downsample")?.extract::<usize>()?;
        let image = py
            .import("uniserve.media.image")?
            .getattr("Config")?
            .call1((stride, stride))?;
        let feature = vision
            .call_method1("output_layout", (image,))?
            .get_item("features")?;
        vision_tokens
            * feature.getattr("shape")?.get_item(1)?.extract::<u64>()?
            * feature
                .getattr("dtype")?
                .getattr("itemsize")?
                .extract::<u64>()?
    };
    let feature_bytes = latent_features.max(vision_features);
    info.encoder_cache_entries = if processor.is_some() {
        u32::try_from(config_native.encoder_cache_entries)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
    } else {
        0
    };
    info.encoder_entry_bytes = if info.encoder_cache_entries == 0 {
        0
    } else {
        feature_bytes
    };
    info.buffer_pool_bytes = (u64::from(info.encoder_cache_entries) + 1) * feature_bytes
        + u64::from(info.request_slots) * product_storage_bytes(outputs)?;
    info.max_unresolved_calls =
        native::call_window(info.queue_depth as usize, info.max_batch_calls as usize)
            .map_err(value_error)? as u32;

    let devices = devices(config)?;
    let mut arena = native::ArenaCapacity::for_tokens(
        info.queue_depth as usize,
        info.max_batch_calls as usize,
        info.request_slots as usize,
        devices.len(),
        scalar_bytes(py)?,
        latent_bytes,
        feature_bytes,
    )
    .map_err(value_error)?;
    let mut fixed = BTreeMap::new();
    for device in &devices {
        fixed.insert(
            device.str()?.extract::<String>()?,
            info.buffer_pool_bytes + arena.device_product_bytes / devices.len() as u64,
        );
    }
    let generation = config_native
        .generation_device
        .clone()
        .unwrap_or_else(|| info.device.clone());
    *fixed.entry(generation).or_default() += latent_bytes;
    let input = if text.is_none() {
        None
    } else {
        Some(inputs::input_buffer_config(model, config, processor)?)
    };
    if let Some(input) = &input {
        let planner = py.import("uniserve_worker.bootstrap.cache")?;
        let planes = planner.call_method1("plan_cache", (&text, config))?;
        let groups = inputs::groups(&planes)?;
        let options = PyDict::new(py);
        options.set_item(
            "num_units",
            1 + groups
                .iter()
                .map(|group| group.units_per_page)
                .max()
                .unwrap_or(0),
        )?;
        let mut cache: uniserve_worker_ipc::KvCacheInfo =
            record_from_py(&planner.call_method("cache_info", (&text, config), Some(&options))?)?;
        let (prefill, decode) = reserve_inputs(
            model,
            config,
            input,
            processor,
            bindings,
            !flow.is_none(),
            &mut fixed,
        )?;
        *fixed.entry(info.device.clone()).or_default() +=
            shared_bytes(&text, config, &planes, &groups, &dtype, &info)?;
        let pages: Vec<_> = groups
            .iter()
            .map(|group| (group.page_tokens, group.units_per_page))
            .collect();
        let mut reserved_units = 0;
        for &(tokens, units) in &pages {
            reserved_units += uniserve_worker::config::graph_padding_block_count(tokens as usize)
                as u64
                * u64::from(units);
        }
        reserved_units += uniserve_worker::graph_storage_budget_bytes(device_total_bytes(
            &config_native.device.as_str().into_pyobject(py)?.into_any(),
        )? as i64)
        .div_ceil(cache.unit_bytes);
        reserved_units += fixed[&info.device].div_ceil(cache.unit_bytes);
        let mut capacity = derive_runtime_kv_capacity(
            pages.clone(),
            config_native.kv_token_capacity.map(|tokens| tokens as u64),
            cache.unit_bytes,
            Some(&config_native.device.as_str().into_pyobject(py)?.into_any()),
            config_native.pool_storage_bytes.map(|bytes| bytes as u64),
            1,
            None,
            1,
            reserved_units,
        )?
        .inner;
        capacity.num_units = minimum_capacity(group, capacity.num_units)?;
        cache.num_units = u32::try_from(capacity.num_units)
            .map_err(|_| PyValueError::new_err("KV pool exceeds the unit index range"))?;
        let row_units = groups
            .iter()
            .map(|group| u64::from(group.units_per_page))
            .sum::<u64>();
        let max_rows = info.max_batch_calls.min(info.request_slots) as usize;
        if prefill {
            let rows = max_rows.min(((capacity.num_units - 1) / row_units) as usize);
            let shapes = prefill_shapes(
                config,
                input,
                processor,
                !flow.is_none(),
                rows,
                Some((&pages, capacity.num_units - 1)),
            )?;
            info.max_prefill_calls = shapes
                .iter()
                .map(|shape| shape.row_bucket - 1)
                .max()
                .map(|limit| limit.min(rows) as u32)
                .unwrap_or(0);
        }
        if decode {
            info.max_decode_calls = graph_shapes::decode_shapes(
                config,
                max_rows,
                row_units as usize,
                capacity.num_units as usize,
            )?
            .into_iter()
            .max()
            .unwrap_or(0) as u32;
        }
        // The arena's slot counts are independent of KV capacity. Only its
        // transfer byte reservation changes after the pool has been fitted.
        arena.transfer_bytes = (capacity.num_units * cache.unit_bytes)
            .max(feature_bytes)
            .max(1)
            * arena.transfer_tickets as u64;
        info.kv_cache = Some(cache);
    }
    info.host_lane_capacity = arena.host_lane_inflight as u32;
    let physical_buffer_pool_bytes = info.buffer_pool_bytes;
    Ok(WorkerLayout {
        info,
        arena,
        input_config: input.map(Bound::unbind),
        fixed_device_bytes: fixed.into_iter().collect(),
        physical_buffer_pool_bytes,
        latent_plan,
    })
}

fn prefill_shapes(
    config: &Bound<'_, PyAny>,
    input: &Bound<'_, PyAny>,
    processor: Option<&Bound<'_, PyAny>>,
    image: bool,
    rows: usize,
    pool: Option<(&[(u32, u32)], u64)>,
) -> PyResult<Vec<uniserve_worker::PrefillShape>> {
    let pages = pool.map(|(pages, _)| {
        pages
            .iter()
            .map(|&(tokens, units)| (tokens as usize, units as usize))
            .collect::<Vec<_>>()
    });
    graph_shapes::configured_prefill(
        config,
        rows,
        input.getattr("max_text_tokens")?.extract()?,
        image,
        processor
            .map(|processor| {
                processor
                    .getattr("feature_injection")
                    .map(|value| !value.is_none())
            })
            .transpose()?
            .unwrap_or(false),
        true,
        pages
            .as_ref()
            .zip(pool)
            .map(|(pages, (_, units))| (pages.as_slice(), units as usize)),
    )
}

#[allow(clippy::too_many_arguments)]
fn reserve_inputs(
    model: &Bound<'_, PyAny>,
    config: &Bound<'_, PyAny>,
    input: &Bound<'_, PyAny>,
    processor: Option<&Bound<'_, PyAny>>,
    bindings: &Bound<'_, PyAny>,
    image: bool,
    fixed: &mut BTreeMap<String, u64>,
) -> PyResult<(bool, bool)> {
    let config_native = crate::worker::config::native(config)?;
    let py = model.py();
    let declarations = py.import("uniserve_worker.bootstrap.components")?;
    let buffers = py.import("uniserve_worker.model_executor.input_buffers")?;
    let options = PyDict::new(py);
    options.set_item("diffusion", image)?;
    let buffered = buffers
        .call_method("buffered_kinds", (), Some(&options))?
        .try_iter()?
        .map(|kind| {
            let kind = kind?;
            Ok((kind.extract::<String>()?, kind))
        })
        .collect::<PyResult<BTreeMap<_, _>>>()?;
    let device: String = config_native.device.clone();
    let generation = config_native
        .generation_device
        .clone()
        .unwrap_or_else(|| device.clone());
    let max_rows = config_native
        .max_batch_calls
        .min(config_native.max_request_pool_size);
    let copies = config_native.expert_microbatches as u64;
    let lanes: Vec<_> = if config_native.lanes.is_empty() {
        vec![None]
    } else {
        config_native.lanes.iter().map(Some).collect()
    };
    let mut prefill = false;
    let mut decode = false;
    for item in declarations
        .call_method1("describe_components", (model,))?
        .call_method0("items")?
        .try_iter()?
    {
        let item = item?;
        let placement = bindings.call_method1("get", (item.get_item(0)?,))?;
        if bindings.is_truthy()?
            && (placement.is_none() || !placement.getattr("owns")?.is_truthy()?)
        {
            continue;
        }
        for call in item.get_item(1)?.try_iter()? {
            let call = call?;
            let kinds: BTreeSet<String> = declarations
                .call_method1("call_kinds", (PyTuple::new(py, [&call])?,))?
                .extract()?;
            let kinds: BTreeSet<_> = kinds
                .into_iter()
                .filter(|kind| buffered.contains_key(kind))
                .collect();
            let target = if kinds.contains("latent_encoding") || kinds.contains("image_decoding") {
                generation.clone()
            } else if placement.is_none() {
                device.clone()
            } else {
                placement.getattr("device")?.str()?.extract()?
            };
            let target_device = py.import("torch")?.getattr("device")?.call1((&target,))?;
            let cuda = target_device.getattr("type")?.extract::<String>()? == "cuda";
            for lane in &lanes {
                let selected = match lane {
                    None => kinds.clone(),
                    Some(lane) => {
                        let allowed: BTreeSet<_> = lane
                            .call_kinds
                            .iter()
                            .map(|kind| kind.as_str().to_owned())
                            .collect();
                        kinds.intersection(&allowed).cloned().collect()
                    }
                };
                let Some(kind) = selected.first() else {
                    continue;
                };
                prefill |= selected.contains("prefill") && cuda;
                decode |= selected.contains("decode") && cuda;
                let mut fields = input.clone();
                let mut rows = input.getattr("max_rows")?.extract::<usize>()?;
                if selected.contains("prefill") && cuda {
                    let shapes = prefill_shapes(config, input, processor, image, max_rows, None)?;
                    for shape in shapes {
                        rows = rows.max(shape.row_bucket);
                    }
                } else if selected.contains("token_denoising")
                    && cuda
                    && config_native.graph_policy != "off"
                {
                    rows = py
                        .import("uniserve_worker.model_executor.canvas_runner")?
                        .call_method1("canvas_buffer_rows", (rows,))?
                        .extract()?;
                }
                if rows != input.getattr("max_rows")?.extract::<usize>()? {
                    let options = PyDict::new(py);
                    options.set_item("max_rows", rows)?;
                    fields = py.import("dataclasses")?.call_method(
                        "replace",
                        (&fields,),
                        Some(&options),
                    )?;
                }
                let allocation = buffers
                    .call_method1("input_buffer_config", (&buffered[kind], &fields))?
                    .get_item(1)?;
                let count = if selected
                    .iter()
                    .any(|kind| matches!(kind.as_str(), "prefill" | "decode" | "token_denoising"))
                {
                    copies
                } else {
                    1
                };
                *fixed.entry(target.clone()).or_default() +=
                    count * schema_bytes(&allocation.call_method0("buffers")?, false)?;
            }

            if kinds.contains("token_denoising") {
                let slots = py.import("uniserve_worker.storage.canvas_slots")?;
                let denoiser =
                    slots.call_method1("generating_denoiser", (call.getattr("module")?,))?;
                if let Some(sampling) = &config_native.canvas_sampling
                    && !denoiser.is_none()
                {
                    let options = PyDict::new(py);
                    options.set_item("request_pool_size", config_native.max_request_pool_size)?;
                    options.set_item("history_depth", sampling.stability_threshold)?;
                    let resident: u64 = slots
                        .getattr("CanvasSlots")?
                        .call_method("denoiser_bytes", (&denoiser,), Some(&options))?
                        .extract()?;
                    options.del_item("request_pool_size")?;
                    options.set_item("max_rows", input.getattr("max_rows")?)?;
                    options.set_item("device_type", target_device.getattr("type")?)?;
                    let scratch: u64 = py
                        .import("uniserve_worker.model_executor.canvas_runner")?
                        .getattr("CanvasRunner")?
                        .call_method("sampler_bytes", (&denoiser,), Some(&options))?
                        .extract()?;
                    *fixed.entry(target).or_default() += resident + copies * scratch;
                }
            }
        }
    }
    Ok((prefill, decode))
}

fn shared_bytes(
    text: &Bound<'_, PyAny>,
    config: &Bound<'_, PyAny>,
    planes: &Bound<'_, PyAny>,
    groups: &[uniserve_worker::GroupShape],
    dtype: &Bound<'_, PyAny>,
    info: &WorkerInfo,
) -> PyResult<u64> {
    let config_native = crate::worker::config::native(config)?;
    let py = config.py();
    let shapes = groups
        .iter()
        .map(|shape| Py::new(py, GroupShape { shape: *shape }))
        .collect::<PyResult<Vec<_>>>()?;
    let options = PyDict::new(py);
    options.set_item("groups", PyTuple::new(py, shapes)?)?;
    options.set_item("request_pool_size", info.request_slots)?;
    options.set_item(
        "width",
        inputs::resident_width(planes, config_native.max_sequence_tokens as u64)?,
    )?;
    let tables = py
        .import("uniserve_worker.storage.block_tables")?
        .getattr("BlockTables")?
        .call_method("buffers", (), Some(&options))?;
    let options = PyDict::new(py);
    options.set_item("request_pool_size", info.request_slots)?;
    options.set_item(
        "vocab_size",
        text.getattr("backbone")?.getattr("vocab_size")?,
    )?;
    options.set_item("continuation_width", 1)?;
    options.set_item("logits_dtype", dtype)?;
    let decode = py
        .import("uniserve_worker.storage.decode_state")?
        .getattr("DecodeState")?
        .call_method("buffers", (), Some(&options))?;
    let mut elements = 0;
    let mut scales = 0;
    for group in planes.getattr("groups")?.try_iter()? {
        let group = group?;
        let rows = group.getattr("page_tokens")?.extract::<u64>()?
            * group.getattr("layers")?.len()? as u64
            * group.getattr("num_kv_heads")?.extract::<u64>()?;
        scales = scales.max(rows);
        elements = elements.max(rows * group.getattr("head_dim")?.extract::<u64>()?);
    }
    let options = PyDict::new(py);
    options.set_item("page_elements", elements)?;
    options.set_item("page_scales", scales)?;
    options.set_item("capacity", info.max_unresolved_calls)?;
    let transfers: u64 = py
        .import("uniserve_worker.storage.cache_imports")?
        .call_method("cache_transfer_workspace_bytes", (), Some(&options))?
        .extract()?;
    Ok(schema_bytes(&tables, false)? + schema_bytes(&decode, false)? + transfers)
}
