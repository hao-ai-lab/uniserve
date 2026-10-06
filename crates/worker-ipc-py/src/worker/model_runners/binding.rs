//! Bind fixed numerical inputs, contexts and graph pools to execution lanes.

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use uniserve_worker_ipc::{CallKind, ForwardMode, MediaCall};

use super::{ModelRunners, experts};
use crate::worker::error::native_error;
use crate::worker::execution::close_all;
use crate::worker::graph_shapes::{TextShapes, configured_prefill, decode_shapes};
use crate::worker::host::with_context;

#[allow(clippy::too_many_arguments)]
pub(super) fn configure(
    slf: &Bound<'_, ModelRunners>,
    owner: &Bound<'_, PyAny>,
    input_config: &Bound<'_, PyAny>,
    kv_cache: &Bound<'_, PyAny>,
    latent_pool: &Bound<'_, PyAny>,
    predicates: &Bound<'_, PyAny>,
    max_calls: usize,
    request_slots: usize,
    latent_capacity_units: usize,
    table_widths: Vec<usize>,
    max_inflight: usize,
) -> PyResult<()> {
    let py = slf.py();
    if slf.borrow().inner.iter().next().is_some() {
        return Err(PyRuntimeError::new_err(
            "input execution resources are already bound",
        ));
    }
    super::modules::ensure_open(owner)?;
    owner.setattr("kv_cache", kv_cache)?;
    owner.setattr("decode_predicates", predicates)?;
    let widths = PyTuple::new(py, table_widths)?;
    owner.setattr("table_widths", &widths)?;

    let config = owner.getattr("worker_config")?;
    let config_native = crate::worker::config::native(&config)?;
    let streams = Arc::clone(&slf.borrow().streams);
    streams.initialize(owner, Some(max_inflight + 1))?;
    if config_native.expert_exchange == "dwdp" {
        if streams.lane_count() > 1 {
            return Err(PyValueError::new_err(
                "DWDP weight buffers require one execution lane",
            ));
        }
        owner.setattr(
            "expert_weights",
            py.import("uniserve.runtime.weight_prefetch")?
                .call_method1("WeightPrefetch", (owner.getattr("model")?,))?,
        )?;
    }
    let max_tokens: usize = input_config.getattr("max_tokens")?.extract()?;
    experts::configure(slf, owner, max_tokens)?;
    let has_experts = !slf.borrow().exchanges.is_empty();
    let microbatches: usize = config_native.expert_microbatches;

    let max_rows = max_calls.min(request_slots);
    let num_units: usize = kv_cache.getattr("info")?.getattr("num_units")?.extract()?;
    let row_units: usize = kv_cache.getattr("row_units")?.extract()?;
    let pool_rows = (num_units - 1) / row_units;
    let decode_sizes = decode_shapes(&config, max_rows, row_units, num_units)?;
    let pages = kv_cache
        .getattr("shapes")?
        .try_iter()?
        .map(|shape| {
            let shape = shape?;
            Ok((
                shape.getattr("page_tokens")?.extract()?,
                shape.getattr("units_per_page")?.extract()?,
            ))
        })
        .collect::<PyResult<Vec<(usize, usize)>>>()?;
    let text_tokens = input_config
        .getattr("max_text_tokens")?
        .extract::<usize>()?
        .min(kv_cache.getattr("token_capacity")?.extract()?);
    let builder = owner.getattr("image_builder")?;
    let processor = owner.getattr("processor")?;
    let feature_injection =
        !processor.is_none() && !processor.getattr("feature_injection")?.is_none();
    if !builder.is_none() {
        configure_images(
            owner,
            &builder,
            latent_pool,
            max_rows,
            max_tokens,
            latent_capacity_units,
        )?;
    }

    let attention = owner.getattr("attention")?;
    let cache = kv_cache.getattr("cache")?;
    let storage = owner.getattr("graph_storage")?;
    let capture = config_native.graph_policy != "off";
    let runtime = py.import("uniserve.runtime")?;
    let models = py.import("uniserve.model")?;
    let buffers = py.import("uniserve_worker.model_executor.input_buffers")?;
    let text_type = py
        .import("uniserve_worker.model_executor.text_runner")?
        .getattr("TextRunner")?;
    let canvas = py.import("uniserve_worker.model_executor.canvas_runner")?;
    let canvas_type = canvas.getattr("CanvasRunner")?;
    let calls = py.import("uniserve_worker.protocol.call")?;
    let replace = py.import("dataclasses")?.getattr("replace")?;
    let modules: Vec<_> = slf.borrow().modules.values().cloned().collect();

    // Serial calls on one stream share intermediates. Independent streams,
    // including expert microbatches, must retain separate graph pools.
    let mut token_pools: HashMap<(String, usize), Py<PyAny>> = HashMap::new();
    for module in modules {
        let mut entry_kinds: Vec<_> = module
            .kinds
            .iter()
            .copied()
            .filter(|kind| match kind {
                CallKind::Forward(_) => true,
                CallKind::Media(
                    MediaCall::VisionEncoding
                    | MediaCall::LatentEncoding
                    | MediaCall::ImageDecoding,
                ) => true,
                CallKind::Media(MediaCall::Denoising) => !builder.is_none(),
                _ => false,
            })
            .collect();
        entry_kinds.sort_by_key(|kind| kind.as_str());
        if entry_kinds.is_empty() {
            continue;
        }
        let target = if entry_kinds.iter().any(|kind| {
            matches!(
                kind,
                CallKind::Media(MediaCall::LatentEncoding | MediaCall::ImageDecoding)
            )
        }) {
            let generation = config_native
                .generation_device
                .as_deref()
                .unwrap_or(&config_native.device);
            py.import("uniserve.runtime.device")?
                .call_method1("canonical_device", (generation,))?
        } else {
            module.binding.bind(py).getattr("device")?
        };
        let cuda = super::streams::cuda_index(&target)?.is_some();
        let runner_type = module.runner_type.bind(py);
        let is_text = runner_type
            .cast::<pyo3::types::PyType>()?
            .is_subclass(&text_type)?;
        let is_canvas = runner_type
            .cast::<pyo3::types::PyType>()?
            .is_subclass(&canvas_type)?;
        let token_runner = is_text || is_canvas;
        let call = module.call.bind(py);
        let model = call.getattr("module")?;
        let reads_cache = model.is_instance(&models.getattr("CausalLM")?)?
            || model.is_instance(&models.getattr("ImageDenoiser")?)?
            || model.is_instance(&models.getattr("TokenDenoiser")?)?;
        let mut peers = Vec::new();
        for lane in streams.batch_streams(&target, token_runner && has_experts)? {
            let kinds: Vec<_> = entry_kinds
                .iter()
                .copied()
                .filter(|kind| {
                    lane.kinds
                        .as_ref()
                        .is_none_or(|covered| covered.contains(kind))
                })
                .collect();
            if kinds.is_empty() {
                continue;
            }
            let rows = max_rows.min(lane.max_rows.filter(|&rows| rows != 0).unwrap_or(max_rows));
            let decode = if cuda && kinds.contains(&CallKind::Forward(ForwardMode::Decode)) {
                decode_sizes
                    .iter()
                    .copied()
                    .filter(|&size| size <= rows)
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            if cuda
                && capture
                && kinds.contains(&CallKind::Forward(ForwardMode::Decode))
                && decode.is_empty()
            {
                return Err(PyValueError::new_err(format!(
                    "no configured decode graph size fits {rows} rows and a KV pool of {num_units} units, whose rows hold {row_units} units each"
                )));
            }
            let prefill = if cuda && kinds.contains(&CallKind::Forward(ForwardMode::Prefill)) {
                configured_prefill(
                    &config,
                    rows.min(pool_rows),
                    text_tokens,
                    !builder.is_none(),
                    feature_injection,
                    attention.getattr("device_causality")?.extract()?,
                    Some((&pages, num_units - 1)),
                )?
            } else {
                Vec::new()
            };

            // Prefill adds one padding sequence; canvas readout can double
            // the numerical rows. Neither consumes scheduler request slots.
            let input_rows = input_config.getattr("max_rows")?.extract::<usize>()?;
            let buffer_rows = if !prefill.is_empty() {
                prefill
                    .iter()
                    .map(|shape| shape.row_bucket)
                    .fold(input_rows, usize::max)
            } else if cuda
                && capture
                && kinds.contains(&CallKind::Forward(ForwardMode::TokenDenoising))
            {
                canvas
                    .call_method1("canvas_buffer_rows", (input_rows,))?
                    .extract()?
            } else {
                input_rows
            };
            let fields = if buffer_rows != input_rows {
                let options = PyDict::new(py);
                options.set_item("max_rows", buffer_rows)?;
                replace.call((input_config,), Some(&options))?
            } else {
                input_config.clone()
            };
            let py_kinds = PyTuple::new(
                py,
                kinds
                    .iter()
                    .map(|kind| {
                        calls
                            .getattr(match kind {
                                CallKind::Forward(_) => "ForwardMode",
                                _ => "MediaCall",
                            })?
                            .call1((kind.as_str(),))
                    })
                    .collect::<PyResult<Vec<_>>>()?,
            )?;
            let stream = lane.stream.bind(py);
            let pool_key = (target.str()?.extract::<String>()?, stream.as_ptr() as usize);
            let mut resources = Vec::new();
            let result = (|| {
                let scope = if stream.is_none() {
                    py.import("contextlib")?.call_method0("nullcontext")?
                } else {
                    let cuda = py.import("torch.cuda")?;
                    stream.call_method1(
                        "wait",
                        (cuda.call_method1("current_stream", (&target,))?,),
                    )?;
                    cuda.call_method1("stream", (stream.getattr("stream")?,))?
                };
                let bound = with_context(&scope, || {
                    let selected = buffers
                        .call_method1("input_buffer_config", (py_kinds.get_item(0)?, &fields))?;
                    let buffer_type = selected.get_item(0)?;
                    let options = PyDict::new(py);
                    options.set_item("config", selected.get_item(1)?)?;
                    options.set_item("device", &target)?;
                    options.set_item("max_inflight", max_inflight)?;
                    if buffer_type.is(&buffers.getattr("TokenBuffers")?)
                        || buffer_type.is(&buffers.getattr("DiffusionBuffers")?)
                    {
                        options.set_item("image_builder", &builder)?;
                    }
                    let inputs = buffer_type.call((), Some(&options))?;
                    resources.push(inputs.clone());

                    let options = PyDict::new(py);
                    options.set_item(
                        "cache",
                        if reads_cache {
                            cache.clone()
                        } else {
                            py.None().into_bound(py)
                        },
                    )?;
                    options.set_item("attention", &attention)?;
                    options.set_item("stream", stream)?;
                    options.set_item("groups", call.getattr("groups")?)?;
                    // Host lengths come from request preparation, never from
                    // a device readback in the attention planning path.
                    options.set_item("derive_host_lengths", false)?;
                    options.set_item(
                        "experts",
                        slf.borrow()
                            .exchanges
                            .get(lane.microbatch)
                            .map(|exchange| exchange.bind(py)),
                    )?;
                    options.set_item("weights", owner.getattr("expert_weights")?)?;
                    let context = runtime
                        .getattr("ExecutionContext")?
                        .call((&model,), Some(&options))?;
                    resources.push(context.clone());

                    let options = PyDict::new(py);
                    options.set_item("storage", &storage)?;
                    options.set_item("exact_graphs", config_native.flow_cuda_graph)?;
                    options.set_item("cache", &cache)?;
                    options.set_item("predicates", predicates)?;
                    options.set_item("rank", config_native.rank)?;
                    let mut devices = Vec::new();
                    if capture && cuda {
                        devices.push(target.clone());
                        devices.extend(super::streams::capture_devices(owner, &target)?);
                    }
                    options.set_item("devices", PyTuple::new(py, devices)?)?;
                    options.set_item(
                        "share",
                        if token_runner {
                            token_pools.get(&pool_key).map(|runner| runner.bind(py))
                        } else {
                            None
                        },
                    )?;
                    let bound = runner_type.call(
                        (
                            &module.name,
                            call,
                            &target,
                            &py_kinds,
                            stream,
                            &context,
                            &inputs,
                        ),
                        Some(&options),
                    )?;
                    // The runner takes ownership of its context and buffers.
                    resources.clear();
                    resources.push(bound.clone());
                    let execution = bound.getattr("execution")?;
                    let size = if reads_cache {
                        models.call_method1(
                            "TextSize",
                            (fields.getattr("max_tokens")?, fields.getattr("max_rows")?),
                        )?
                    } else {
                        py.None().into_bound(py)
                    };
                    with_context(&storage.call_method1("allocate", (&execution,))?, || {
                        context.call_method1("prepare", (size,))
                    })?;
                    storage.call_method0("check")?;
                    if token_runner && execution.getattr("pools")?.is_truthy()? {
                        token_pools
                            .entry(pool_key.clone())
                            .or_insert_with(|| bound.clone().unbind());
                    }
                    Ok(bound)
                })?;
                if is_text {
                    bound.setattr(
                        "shapes",
                        Py::new(
                            py,
                            TextShapes {
                                inner: uniserve_worker::TextShapes::new(decode, prefill),
                            },
                        )?,
                    )?;
                }
                bound.setattr("table_widths", &widths)?;
                if is_canvas {
                    bound.setattr("pool_rows", pool_rows)?;
                }
                slf.borrow_mut()
                    .inner
                    .bind(
                        &module.name,
                        if lane.microbatch == 0 { &kinds } else { &[] },
                        bound.clone().unbind(),
                    )
                    .map_err(|error| native_error(py, error))?;
                Ok(bound)
            })();
            match result {
                Ok(runner) => peers.push(runner),
                Err(error) => {
                    // Before runner ownership transfers, close whichever
                    // allocations succeeded, in reverse construction order.
                    return close_all(
                        py,
                        std::iter::once(Err(error)).chain(
                            resources
                                .iter()
                                .rev()
                                .map(|resource| resource.call_method0("close").map(drop)),
                        ),
                    );
                }
            }
        }
        if token_runner && microbatches > 1 {
            slf.borrow()
                .bind_microbatches(py, &PyTuple::new(py, peers)?)?;
        }
    }
    Ok(())
}

/// Numerical image sizes are queried once, then filtered against physical
/// token and latent capacity for every configured batch and guidance count.
fn configure_images(
    owner: &Bound<'_, PyAny>,
    builder: &Bound<'_, PyAny>,
    pool: &Bound<'_, PyAny>,
    max_rows: usize,
    max_tokens: usize,
    per_image: usize,
) -> PyResult<()> {
    let py = owner.py();
    let config = owner.getattr("worker_config")?;
    let config_native = crate::worker::config::native(&config)?;
    let sizes: Vec<(usize, usize)> = config_native.flow_graph_shapes.clone();
    let rows: Vec<usize> = config_native.flow_graph_batch_sizes.clone();
    let capacity: usize = pool.getattr("capacity_units")?.extract()?;
    let image = py.import("uniserve.media.image")?.getattr("Config")?;
    let shape = py
        .import("uniserve_worker.model_executor.graph_inputs")?
        .getattr("DiffusionShape")?;
    let mut shapes = Vec::new();
    for (height, width) in sizes {
        let size = image.call1((height, width))?;
        let tokens: usize = builder
            .call_method1("sequence_length", (&size,))?
            .extract()?;
        let units: usize = builder
            .getattr("denoiser")?
            .call_method1("latent_shape", ("image", &size))?
            .get_item(0)?
            .extract()?;
        for &rows in &rows {
            for branches in 1..=3 {
                if rows > 0
                    && rows <= max_rows
                    && rows * tokens * branches <= max_tokens
                    && units <= per_image
                    && rows * units <= capacity
                {
                    shapes.push(shape.call1((rows, height, width, branches))?);
                }
            }
        }
    }
    owner.setattr("flow_cfg_branches", (1, 2, 3))?;
    owner.setattr("flow_captures", PyTuple::new(py, shapes)?)
}
