//! Input and block-table bounds from public numerical layouts.

use uniserve_worker::GroupShape;

use super::*;

pub(super) fn groups(planes: &Bound<'_, PyAny>) -> PyResult<Vec<GroupShape>> {
    planes
        .getattr("groups")?
        .try_iter()?
        .map(|group| {
            let group = group?;
            Ok(GroupShape {
                page_tokens: group.getattr("page_tokens")?.extract()?,
                units_per_page: group.getattr("units_per_page")?.extract()?,
                window: group.getattr("window")?.extract()?,
            })
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (planes, *, max_sequence_tokens, max_query_tokens))]
pub(super) fn table_widths<'py>(
    planes: &Bound<'py, PyAny>,
    max_sequence_tokens: u64,
    max_query_tokens: u64,
) -> PyResult<Bound<'py, PyTuple>> {
    let widths = groups(planes)?.into_iter().flat_map(|group| {
        std::iter::repeat_n(
            group.table_width(max_sequence_tokens, max_query_tokens),
            group.units_per_page as usize,
        )
    });
    PyTuple::new(planes.py(), widths.collect::<Vec<_>>())
}

#[pyfunction]
#[pyo3(signature = (planes, *, max_sequence_tokens))]
pub(in crate::worker) fn resident_width(
    planes: &Bound<'_, PyAny>,
    max_sequence_tokens: u64,
) -> PyResult<u64> {
    Ok(groups(planes)?
        .iter()
        .map(|group| group.resident_width(max_sequence_tokens))
        .max()
        .unwrap_or(1))
}

#[pyfunction]
pub(in crate::worker) fn graph_table_widths<'py>(
    model: &Bound<'py, PyAny>,
    worker_config: &Bound<'py, PyAny>,
    pool: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let py = model.py();
    let sequence: u64 = worker_config.getattr("max_sequence_tokens")?.extract()?;
    if pool.is_none() || sequence == 0 || capability(model, "CausalLM")?.is_none() {
        return Ok(PyTuple::empty(py));
    }

    let query = worker_config.getattr("max_batch_tokens")?.extract()?;
    let units = pool
        .getattr("info")?
        .getattr("num_units")?
        .extract::<u64>()?
        - 1;
    let widths = groups(&pool.getattr("cache")?.getattr("planes")?)?
        .into_iter()
        .flat_map(|group| {
            let capacity = (units / u64::from(group.units_per_page)).max(1);
            std::iter::repeat_n(
                group.table_width(sequence, query).min(capacity),
                group.units_per_page as usize,
            )
        });
    PyTuple::new(py, widths.collect::<Vec<_>>())
}

#[pyfunction]
#[pyo3(signature = (model, config, *, processor=None))]
pub(super) fn input_buffer_config<'py>(
    model: &Bound<'py, PyAny>,
    config: &Bound<'py, PyAny>,
    processor: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = model.py();
    let text = capability(model, "CausalLM")?;
    if text.is_none() {
        return Err(PyValueError::new_err(
            "text input buffers require a causal language model",
        ));
    }

    let calls: usize = config.getattr("max_batch_calls")?.extract()?;
    let tokens: usize = config.getattr("max_batch_tokens")?.extract()?;
    let mut max_rows = calls.min(config.getattr("max_request_pool_size")?.extract()?);
    let mut max_tokens = tokens;
    for lane in config.getattr("lanes")?.try_iter()? {
        let lane = lane?;
        max_rows = max_rows.min(
            lane.getattr("max_batch_calls")?
                .extract::<Option<usize>>()?
                .unwrap_or(calls),
        );
        max_tokens = max_tokens.min(
            lane.getattr("max_batch_tokens")?
                .extract::<Option<usize>>()?
                .unwrap_or(tokens),
        );
    }

    let flow = py
        .import("uniserve_worker.bootstrap.inputs")?
        .call_method1("image_builder", (model,))?;
    let branches = if flow.is_none() {
        1
    } else {
        py.import("uniserve.diffusion")?.getattr("Branch")?.len()?
    };
    let image_tokens = if flow.is_none() {
        0
    } else {
        flow.getattr("max_tokens")?.extract::<usize>()?
    };
    let injection = match processor {
        Some(processor) => processor.getattr("feature_injection")?,
        None => py.None().into_bound(py),
    };
    let mut image_span = 0;
    if !injection.is_none() {
        let vision: usize = py
            .import("uniserve_worker.bootstrap.capacity")?
            .call_method1("vision_tokens", (model, processor))?
            .extract()?;
        image_span = image_tokens.max(vision);
        if injection.getattr("layout")?.eq(py
            .import("uniserve.processing")?
            .getattr("FeatureLayout")?
            .getattr("FRAMED")?)?
        {
            image_span += 2;
        }
    }

    let mut flow_tokens = 0;
    if !flow.is_none() {
        flow_tokens = image_tokens + flow.getattr("framing")?.extract::<usize>()?;
        let image = py.import("uniserve.media.image")?.getattr("Config")?;
        for shape in config.getattr("flow_graph_shapes")?.try_iter()? {
            let (height, width): (usize, usize) = shape?.extract()?;
            let size = image.call1((height, width))?;
            flow_tokens =
                flow_tokens.max(flow.call_method1("sequence_length", (size,))?.extract()?);
        }
        flow_tokens *= branches;
    }

    // Independent text and diffusion calls borrow the same storage, so its
    // token extent is their maximum rather than their sum.
    let text_tokens = max_tokens + image_span;
    let call_tokens = text_tokens.max(flow_tokens);
    let planes = py
        .import("uniserve_worker.bootstrap.cache")?
        .call_method1("plan_cache", (&text, config))?;
    let options = PyDict::new(py);
    options.set_item("max_rows", max_rows * branches)?;
    options.set_item("max_tokens", call_tokens)?;
    options.set_item("max_text_tokens", text_tokens)?;
    options.set_item(
        "table_widths",
        table_widths(
            &planes,
            config.getattr("max_sequence_tokens")?.extract()?,
            call_tokens as u64,
        )?,
    )?;
    options.set_item(
        "hidden_size",
        text.getattr("backbone")?.getattr("hidden_size")?,
    )?;
    let dtype: String = config.getattr("model_dtype")?.extract()?;
    options.set_item(
        "embedding_dtype",
        py.import("torch")?
            .getattr(dtype.trim_start_matches("torch."))?,
    )?;
    py.import("uniserve_worker.model_executor.input_buffers")?
        .getattr("TokenBufferConfig")?
        .call((), Some(&options))
}

/// Select a common base page size whose numerical layouts every rank can read.
#[pyfunction]
#[pyo3(signature = (model, config, attention, *, group=None))]
pub(in crate::worker) fn resolve_page_size<'py>(
    model: &Bound<'py, PyAny>,
    config: &Bound<'py, PyAny>,
    attention: &Bound<'py, PyAny>,
    group: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = model.py();
    if !config.getattr("block_size")?.is_none() {
        return Ok(config.clone());
    }
    let text = capability(model, "CausalLM")?;
    let mut size = 64;
    if !text.is_none() {
        let class = py.import("uniserve.nn.attention")?.getattr("Attention")?;
        let readers = PyDict::new(py);
        for module in text.call_method0("modules")?.try_iter()? {
            let module = module?;
            if module.is_instance(&class)? {
                let name = module.getattr("cache_name")?;
                if !name.is_none() {
                    readers.set_item(name, module)?;
                }
            }
        }
        let layers = text.getattr("cache_config")?.getattr("layers")?;
        for name in layers.try_iter()? {
            if !readers.contains(name?)? {
                return Err(unsupported(
                    py,
                    "every resident cache layer requires an attention layer reading it",
                ));
            }
        }
        let cache = py.import("uniserve_worker.bootstrap.cache")?;
        let page_type = py
            .import("uniserve.runtime.backends.attention")?
            .getattr("CachePages")?;
        loop {
            let planes = cache.call_method1("_plan", (&text, config, size))?;
            let mut readable = true;
            'groups: for group in planes.getattr("groups")?.try_iter()? {
                let group = group?;
                let pages = page_type.call1((
                    group.getattr("page_tokens")?,
                    group.getattr("dtype")?,
                    planes.getattr("quantized")?,
                ))?;
                for name in group.getattr("layers")?.try_iter()? {
                    let name = name?;
                    let reader = readers.as_any().get_item(&name)?;
                    let options = PyDict::new(py);
                    options.set_item("num_heads", reader.getattr("local_heads")?)?;
                    options.set_item("num_kv_heads", reader.getattr("local_kv_heads")?)?;
                    options.set_item("head_dim", reader.getattr("head_dim")?)?;
                    options.set_item("window", reader.getattr("window")?)?;
                    options.set_item("dtype", layers.get_item(&name)?.getattr("compute_dtype")?)?;
                    options.set_item("pages", &pages)?;
                    if !attention
                        .call_method("reads_pages", (), Some(&options))?
                        .is_truthy()?
                    {
                        readable = false;
                        break 'groups;
                    }
                }
            }
            if readable {
                break;
            }
            size /= 2;
            if size == 0 {
                return Err(unsupported(
                    py,
                    "no KV page size of at most 64 tokens is read by the attention of every resident cache layer",
                ));
            }
        }
    }
    size = minimum_capacity(group, size)?;
    let options = PyDict::new(py);
    options.set_item("block_size", size)?;
    py.import("dataclasses")?
        .call_method("replace", (config,), Some(&options))
}
