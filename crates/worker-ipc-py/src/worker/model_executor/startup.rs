//! Startup ordering for numerical preparation and expert participation.

use std::cmp::Reverse;
use std::collections::HashSet;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};
use std::sync::Arc;
use uniserve_worker_ipc::{CallKind, ForwardMode, MediaCall};

use super::{ModelExecutor, context, execution, experts, graphs, resources};

/// Seal captured capacities only after budget checks and startup reporting.
pub(super) fn complete(owner: &Bound<'_, ModelExecutor>) -> PyResult<()> {
    let py = owner.py();
    let storage = owner.borrow().graph_storage(py)?;
    storage.borrow().check(py)?;
    py.import("uniserve_worker.execution.kernel_table")?
        .call_method1("log_startup_memory", (storage,))?;

    owner.borrow_mut().kernel_choices = py
        .import("uniserve.runtime.backends")?
        .call_method0("kernel_choices")?
        .extract()?;
    let kernels = owner.borrow().kernels.bind(py).clone();
    kernels.call_method1("add", (ModelExecutor::all(owner)?,))?;
    log_kernels(owner, "startup")?;
    seal(owner)
}

impl ModelExecutor {
    /// Collect diagnostics only when a serving call selected another kernel.
    pub(super) fn report_new_kernels(owner: &Bound<'_, Self>) -> PyResult<()> {
        if !owner.borrow().sealed {
            return Ok(());
        }
        let py = owner.py();
        let choices = py
            .import("uniserve.runtime.backends")?
            .call_method0("kernel_choices")?
            .extract()?;
        {
            let mut owner = owner.borrow_mut();
            if choices == owner.kernel_choices {
                return Ok(());
            }
            owner.kernel_choices = choices;
        }
        let kernels = owner.borrow().kernels.bind(py).clone();
        if kernels
            .call_method1("add", (Self::all(owner)?,))?
            .is_truthy()?
        {
            log_kernels(owner, "serving")?;
        }
        Ok(())
    }
}

fn log_kernels(owner: &Bound<'_, ModelExecutor>, stage: &str) -> PyResult<()> {
    let py = owner.py();
    let (config, kernels) = {
        let owner = owner.borrow();
        (Arc::clone(&owner.config), owner.kernels.bind(py).clone())
    };
    let options = PyDict::new(py);
    options.set_item("stage", stage)?;
    options.set_item("rank", config.rank)?;
    options.set_item("device", &config.device)?;
    let table = kernels.call_method("table", (), Some(&options))?;
    let text = py
        .import("uniserve_worker.execution.kernel_table")?
        .call_method1("format_kernel_table", (table,))?;
    py.import("logging")?
        .call_method1("getLogger", ("uniserve_worker.execution.model_executor",))?
        .call_method1("info", ("%s", text))?;
    Ok(())
}
use crate::worker::block_tables::{GroupShape, GroupTable};
use crate::worker::execution::{Execution, on_stream};
use crate::worker::expert_exchange::ExpertExchange;
use crate::worker::graph_shapes::PrefillShape;
use crate::worker::host::{with_context, with_entered};

pub(super) fn prepare(
    slf: &Bound<'_, ModelExecutor>,
    tokenizer: &Bound<'_, PyAny>,
    latents: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let py = slf.py();
    with_context(&py.import("torch")?.call_method0("inference_mode")?, || {
        experts::bind(slf)?;
        let (exchange, expert) = {
            let runners = slf.borrow();
            (
                runners.exchanges.first().map(|value| value.clone_ref(py)),
                runners
                    .expert_executions
                    .first()
                    .map(|value| value.clone_ref(py)),
            )
        };
        let control = exchange
            .as_ref()
            .map(|exchange| -> PyResult<_> {
                Ok(exchange
                    .bind(py)
                    .getattr("_control")?
                    .cast_into::<ExpertExchange>()?)
            })
            .transpose()?;
        if let (Some(expert), Some(control)) = (expert, &control) {
            // Model ranks announce actual eager calls, including warmup for
            // captures. Expert-only ranks execute the matching layer sequence.
            loop {
                let capacity = control.borrow().warmup(py, 0)?;
                if capacity == 0 {
                    break;
                }
                expert.borrow(py).join_expert_step(py, capacity)?;
            }
        } else {
            let runners = slf
                .borrow()
                .inner
                .iter()
                .map(|runner| runner.clone_ref(py))
                .collect::<Vec<_>>();
            let mut prepared = Vec::new();
            for runner in runners {
                let kinds = runner
                    .bind(py)
                    .getattr("call_kinds")?
                    .try_iter()?
                    .map(|kind| Ok(pythonize::depythonize(&kind?)?))
                    .collect::<PyResult<HashSet<CallKind>>>()?;
                prepared.push((runner, kinds));
            }
            for kind in [
                CallKind::Forward(ForwardMode::Prefill),
                CallKind::Forward(ForwardMode::Decode),
                CallKind::Forward(ForwardMode::TokenDenoising),
                CallKind::Media(MediaCall::Denoising),
            ] {
                for (runner, kinds) in &prepared {
                    if !kinds.contains(&kind) {
                        continue;
                    }
                    let runner = runner.bind(py);
                    match kind {
                        CallKind::Forward(ForwardMode::Prefill) => {
                            let mut shapes = runner.getattr("shapes")?.getattr("prefill")?;
                            if !shapes.is_truthy()? {
                                // One causal token prepares an eager runner
                                // whose policy has no prefill graph buckets.
                                let shape = py
                                    .import("uniserve_worker.model_executor.graph_inputs")?
                                    .getattr("PrefillShape")?
                                    .call1((1, 1, 1))?;
                                shapes = PyTuple::new(py, [shape])?.into_any();
                            }
                            prepare_prefill(slf, runner, &shapes)?;
                        }
                        CallKind::Forward(ForwardMode::Decode) => {
                            prepare_decode(slf, runner)?;
                        }
                        CallKind::Forward(ForwardMode::TokenDenoising) => {
                            prepare_canvas(slf, runner)?;
                        }
                        CallKind::Media(MediaCall::Denoising) => {
                            prepare_flow(slf, runner, latents, tokenizer)?;
                        }
                        _ => unreachable!("fixed startup call kinds"),
                    }
                }
            }
            prepare_images(slf, &prepared, latents)?;
            if let Some(control) = control
                && (slf.borrow().attention_ranks != 0)
            {
                control.borrow().warmup(py, 0)?;
            }
        }
        experts::capture(slf)?;
        resources::synchronize(slf)
    })
}

pub(super) fn seal(slf: &Bound<'_, ModelExecutor>) -> PyResult<()> {
    slf.borrow().graph_storage(slf.py())?.borrow_mut().seal();
    slf.borrow_mut().sealed = true;
    for runner in ModelExecutor::all(slf)? {
        execution(&runner)?.borrow_mut().sealed = true;
    }
    for expert in &slf.borrow().expert_executions {
        expert.borrow_mut(slf.py()).sealed = true;
    }
    Ok(())
}

/// Borrow whole scratch pages, bind numerical tables, and retire the lease on
/// either exit. Request allocation remains with the scheduler.
fn with_tables<'py, T>(
    cache: &Bound<'py, PyAny>,
    lengths: &[usize],
    operation: impl FnOnce(Bound<'py, PyTuple>) -> PyResult<T>,
) -> PyResult<T> {
    let py = cache.py();
    if cache.is_none() {
        return Err(PyValueError::new_err(
            "graph preparation requires the worker KV cache",
        ));
    }
    let groups: Vec<Py<GroupShape>> = cache.getattr("shapes")?.extract()?;
    let count: usize = lengths
        .iter()
        .flat_map(|&length| {
            groups.iter().map(move |group| {
                let shape = group.get().shape;
                length.max(1).div_ceil(shape.page_tokens as usize) * shape.units_per_page as usize
            })
        })
        .sum();
    with_entered(&cache.call_method1("startup_units", (count,))?, |scratch| {
        let units: Vec<u32> = scratch.extract()?;
        let mut cursor = 0;
        let mut tables = Vec::with_capacity(lengths.len());
        for &length in lengths {
            let mut row = Vec::with_capacity(groups.len());
            for group in &groups {
                let shape = group.get().shape;
                let pages = length.max(1).div_ceil(shape.page_tokens as usize);
                let count = pages * shape.units_per_page as usize;
                row.push(Py::new(
                    py,
                    GroupTable::new(
                        group.get(),
                        0,
                        units[cursor..cursor + count].to_vec(),
                        pages as u32 * shape.page_tokens,
                    ),
                )?);
                cursor += count;
            }
            tables.push(PyTuple::new(py, row)?);
        }
        operation(PyTuple::new(py, tables)?)
    })
}

fn eager(runner: &Bound<'_, PyAny>, batch: &Bound<'_, PyAny>) -> PyResult<()> {
    on_stream(&context(runner)?, &runner.getattr("device")?, || {
        graphs::run_eager(runner, batch, &runner.getattr("batch_forward")?).map(drop)
    })
}

fn capture(runner: &Bound<'_, PyAny>, batch: &Bound<'_, PyAny>) -> PyResult<()> {
    graphs::capture(runner, batch, &runner.getattr("batch_forward")?)
}

fn prepare_prefill(
    owner: &Bound<'_, ModelExecutor>,
    runner: &Bound<'_, PyAny>,
    shapes: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let py = owner.py();
    let backend = py.import("uniserve_worker.model_executor.startup")?;
    let config_native = Arc::clone(&owner.borrow().config);
    let row_tokens = config_native
        .max_sequence_tokens
        .min(config_native.max_batch_tokens)
        .max(1);
    let cache = owner.borrow().kv_cache.bind(owner.py()).clone();
    let groups: Vec<Py<GroupShape>> = cache.getattr("shapes")?.extract()?;
    let pages: Vec<_> = groups
        .iter()
        .map(|group| {
            let shape = group.get().shape;
            (shape.page_tokens as usize, shape.units_per_page as usize)
        })
        .collect();
    let units = cache
        .getattr("info")?
        .getattr("num_units")?
        .extract::<usize>()?
        - 1;
    let mut shapes: Vec<Py<PrefillShape>> = shapes.extract()?;
    shapes.sort_by_key(|shape| {
        Reverse((
            shape.get().inner.token_bucket * shape.get().inner.row_bucket,
            shape.get().inner.token_bucket,
        ))
    });
    let selection = py
        .import("uniserve_worker.sampling.metadata")?
        .getattr("TokenSelection")?;
    // Largest footprints first let smaller captures reuse output backing and
    // workspaces without pinning successive allocator growth in separate graphs.
    for shape in shapes {
        let lengths = shape.get().inner.scratch_lengths(row_tokens, &pages, units);
        with_tables(&cache, &lengths, |tables| {
            let options = PyDict::new(py);
            options.set_item(
                "selection",
                selection.getattr(if shape.get().inner.outputs {
                    "LAST_LOGITS"
                } else {
                    "CACHE"
                })?,
            )?;
            options.set_item("causal", shape.get().inner.causal)?;
            options.set_item("embeddings", shape.get().inner.embeddings)?;
            let tokens = PyTuple::new(py, lengths.iter().map(|&length| vec![0i64; length]))?;
            let batch = backend.call_method(
                "make_text_batch",
                (runner.getattr("input_buffers")?, tokens, tables),
                Some(&options),
            )?;
            capture(runner, &batch)
        })?;
    }
    Ok(())
}

fn prepare_decode(owner: &Bound<'_, ModelExecutor>, runner: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = owner.py();
    let backend = py.import("uniserve_worker.model_executor.startup")?;
    let buffers = runner.getattr("input_buffers")?;
    let cache = owner.borrow().kv_cache.bind(owner.py()).clone();
    let mut rows: Vec<usize> = runner.getattr("shapes")?.getattr("decode")?.extract()?;
    if rows.is_empty() {
        rows.push(1);
    }
    for rows in rows.into_iter().rev() {
        with_tables(&cache, &vec![2; rows], |tables| {
            let tokens = PyTuple::new(py, vec![(0i64,); rows])?;
            let prompt = backend.call_method1("make_text_batch", (&buffers, &tokens, &tables))?;
            // Decode attends to real model-produced K/V, including the eager
            // representative used when this runner has no decode graphs.
            eager(runner, &prompt)?;
            let options = PyDict::new(py);
            options.set_item("prefixes", vec![1; rows])?;
            options.set_item("decode", true)?;
            let batch = backend.call_method(
                "make_text_batch",
                (&buffers, tokens, tables),
                Some(&options),
            )?;
            let predicates = owner.borrow().decode_predicates.bind(owner.py()).clone();
            let saved = if predicates.is_none() {
                None
            } else {
                Some(predicates.call_method0("clone")?)
            };
            let result = (|| {
                if saved.is_some() {
                    predicates.set_item(PySlice::new(py, 1, (rows + 1) as isize, 1), true)?;
                }
                capture(runner, &batch)
            })();
            let restored = saved.map_or(Ok(()), |saved| {
                predicates.call_method1("copy_", (saved,)).map(drop)
            });
            result?;
            restored
        })?;
    }
    Ok(())
}

fn prepare_canvas(owner: &Bound<'_, ModelExecutor>, runner: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = owner.py();
    let backend = py.import("uniserve_worker.model_executor.startup")?;
    let cache = owner.borrow().kv_cache.bind(owner.py()).clone();
    let slots = runner.getattr("canvas_slots")?;
    let lengths: Vec<usize> = runner.getattr("readout_lengths")?.extract()?;
    let mut kinds: Vec<_> = lengths
        .into_iter()
        .rev()
        .map(|length| (py.None(), 0, length))
        .collect();
    if !slots.is_none() {
        let length = runner.getattr("canvas_length")?.extract()?;
        for step in [0, 1] {
            kinds.push((slots.getattr("constants")?.unbind(), step, length));
        }
    }
    let rows: Vec<usize> = if execution(runner)?.borrow().pools.bind(py).is_empty() {
        vec![1]
    } else {
        runner.getattr("canvas_rows")?.extract()?
    };
    for rows in rows.into_iter().rev() {
        for (sampling, step, length) in &kinds {
            with_tables(&cache, &vec![1; rows], |tables| {
                let options = PyDict::new(py);
                options.set_item("length", length)?;
                options.set_item("sampling", sampling)?;
                options.set_item("step", step)?;
                let batch = backend.call_method(
                    "make_canvas_batch",
                    (runner.getattr("input_buffers")?, tables),
                    Some(&options),
                )?;
                capture(runner, &batch)
            })?;
        }
    }
    Ok(())
}

fn prepare_flow(
    slf: &Bound<'_, ModelExecutor>,
    runner: &Bound<'_, PyAny>,
    latents: &Bound<'_, PyAny>,
    tokenizer: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let py = slf.py();
    let backend = py.import("uniserve_worker.model_executor.startup")?;
    let config_native = Arc::clone(&slf.borrow().config);
    let capture_enabled = config_native.graph_policy != "off" && config_native.flow_cuda_graph;
    let builder = slf.borrow().image_builder.bind(slf.py()).clone();
    let capacity = builder
        .getattr("max_tokens")?
        .extract::<usize>()?
        .min(latents.getattr("capacity_units")?.extract()?);
    let side = capacity.isqrt().max(1)
        * builder
            .getattr("denoiser")?
            .getattr("downsample")?
            .extract::<usize>()?;
    let configured: Vec<Py<PyAny>> =
        { slf.borrow().flow_captures.bind(slf.py()).clone() }.extract()?;
    let mut shapes = if capture_enabled && !configured.is_empty() {
        configured
    } else {
        let branches: Vec<usize> = slf.borrow().flow_cfg_branches.clone();
        let mut selected = Vec::new();
        for branches in branches {
            let mut matching = None;
            for shape in configured.iter().rev() {
                if shape.bind(py).getattr("cfg_branches")?.extract::<usize>()? == branches {
                    matching = Some(shape.clone_ref(py));
                    break;
                }
            }
            selected.push(match matching {
                Some(shape) => shape,
                None => py
                    .import("uniserve_worker.model_executor.graph_inputs")?
                    .getattr("DiffusionShape")?
                    .call1((1, side, side, branches))?
                    .unbind(),
            });
        }
        selected
    };
    let mut ordered = Vec::new();
    for shape in shapes.drain(..) {
        let mut footprint = 1usize;
        for name in ["rows", "height", "width", "cfg_branches"] {
            footprint *= shape.bind(py).getattr(name)?.extract::<usize>()?;
        }
        ordered.push((footprint, shape));
    }
    ordered.sort_by_key(|(footprint, _)| Reverse(*footprint));
    let component: String = runner.getattr("name")?.extract()?;
    let prefix = slf
        .borrow()
        .inner
        .get(&component, CallKind::Forward(ForwardMode::Prefill))
        .map(|runner| runner.clone_ref(py))
        .ok_or_else(|| PyValueError::new_err("image denoising requires a prefill runner"))?;
    let prefix = prefix.bind(py);
    let cache = slf.borrow().kv_cache.bind(slf.py()).clone();
    on_stream(&context(runner)?, &runner.getattr("device")?, || {
        for (_, shape) in ordered {
            let shape = shape.bind(py);
            let rows: usize = shape.getattr("rows")?.extract()?;
            let branches: usize = shape.getattr("cfg_branches")?.extract()?;
            let prepared = backend.call_method1("flow_prefixes", (slf, shape, tokenizer))?;
            let schedule = prepared.get_item(0)?;
            let prefixes = prepared.get_item(1)?.cast_into::<PyTuple>()?;
            let lengths = prefixes
                .iter()
                .map(|prefix| prefix.len())
                .collect::<PyResult<Vec<_>>>()?;
            let size = py
                .import("uniserve.media.image")?
                .getattr("Config")?
                .call1((shape.getattr("height")?, shape.getattr("width")?))?;
            let units = builder
                .getattr("denoiser")?
                .call_method1("latent_shape", ("image", size))?
                .get_item(0)?;
            with_tables(&cache, &lengths, |tables| {
                with_entered(
                    &latents.call_method1("startup_values", (rows, units))?,
                    |values| {
                        let selected: Vec<_> = lengths
                            .iter()
                            .enumerate()
                            .filter_map(|(index, &length)| (length > 0).then_some(index))
                            .collect();
                        if !selected.is_empty() {
                            let options = PyDict::new(py);
                            options.set_item(
                                "selection",
                                py.import("uniserve_worker.sampling.metadata")?
                                    .getattr("TokenSelection")?
                                    .getattr("HIDDEN")?,
                            )?;
                            options.set_item(
                                "slots",
                                selected
                                    .iter()
                                    .map(|index| 1 + index / branches)
                                    .collect::<Vec<_>>(),
                            )?;
                            let tokens = selected
                                .iter()
                                .map(|&index| prefixes.get_item(index))
                                .collect::<PyResult<Vec<_>>>()?;
                            let prefix_tables = selected
                                .iter()
                                .map(|&index| tables.get_item(index))
                                .collect::<PyResult<Vec<_>>>()?;
                            let batch = backend.call_method(
                                "make_text_batch",
                                (
                                    prefix.getattr("input_buffers")?,
                                    PyTuple::new(py, tokens)?,
                                    PyTuple::new(py, prefix_tables)?,
                                ),
                                Some(&options),
                            )?;
                            eager(prefix, &batch)?;
                        }
                        let batch = backend.call_method1(
                            "make_flow_batch",
                            (slf, runner, shape, schedule, prefixes, tables, values),
                        )?;
                        if capture_enabled {
                            capture(runner, &batch)
                        } else {
                            eager(runner, &batch)
                        }
                    },
                )
            })?;
        }
        Ok(())
    })
}

fn prepare_images(
    owner: &Bound<'_, ModelExecutor>,
    runners: &[(Py<PyAny>, HashSet<CallKind>)],
    latents: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let py = owner.py();
    let backend = py.import("uniserve_worker.model_executor.startup")?;
    let builder = owner.borrow().image_builder.bind(owner.py()).clone();
    let side = if builder.is_none() {
        64
    } else {
        2 * builder
            .getattr("denoiser")?
            .getattr("downsample")?
            .extract::<usize>()?
    };
    let size = py
        .import("uniserve.media.image")?
        .getattr("Config")?
        .call1((side, side))?;
    let features = !builder.is_none()
        && !latents.is_none()
        && builder.getattr("framing")?.extract::<usize>()? == 2
        && runners
            .iter()
            .any(|(_, kinds)| kinds.contains(&CallKind::Media(MediaCall::LatentEncoding)));
    let processor = owner.borrow().processor.bind(owner.py()).clone();
    for (runner, kinds) in runners {
        let runner = runner.bind(py);
        on_stream(&context(runner)?, &runner.getattr("device")?, || {
            for (kind, name) in [
                (MediaCall::VisionEncoding, "VISION_ENCODING"),
                (MediaCall::LatentEncoding, "LATENT_ENCODING"),
            ] {
                if !kinds.contains(&CallKind::Media(kind)) || processor.is_none() {
                    continue;
                }
                let kind_value = py
                    .import("uniserve_worker.protocol.call")?
                    .getattr("MediaCall")?
                    .getattr(name)?;
                let prepared = backend
                    .call_method1("make_image_batch", (&processor, runner, kind_value, side))?;
                if kind == MediaCall::VisionEncoding
                    && runner
                        .getattr_opt("packs_images")?
                        .map(|value| value.is_truthy())
                        .transpose()?
                        .unwrap_or(false)
                {
                    Execution::capture_images(
                        &execution(runner)?,
                        runner,
                        Arc::clone(&owner.borrow().config).max_batch_calls,
                        &prepared.get_item(1)?,
                    )?;
                } else {
                    eager(runner, &prepared.get_item(0)?)?;
                }
            }
            if kinds.contains(&CallKind::Media(MediaCall::ImageDecoding))
                && !builder.is_none()
                && !latents.is_none()
            {
                let units = builder
                    .getattr("denoiser")?
                    .call_method1("latent_shape", ("image", &size))?
                    .get_item(0)?;
                with_entered(
                    &latents.call_method1("startup_values", (1, units))?,
                    |values| {
                        let batch = backend.call_method1(
                            "make_decoding_batch",
                            (runner, side, values.get_item(0)?),
                        )?;
                        eager(runner, &batch)
                    },
                )?;
            }
            if features && kinds.contains(&CallKind::Media(MediaCall::Denoising)) {
                let query = builder
                    .call_method1("sequence_length", (&size,))?
                    .extract()?;
                let units = builder
                    .getattr("denoiser")?
                    .call_method1("latent_shape", ("image", &size))?
                    .get_item(0)?;
                with_tables(
                    &{ owner.borrow().kv_cache.bind(owner.py()).clone() },
                    &[query],
                    |tables| {
                        with_entered(
                            &latents.call_method1("startup_values", (1, units))?,
                            |values| {
                                let batch = backend.call_method1(
                                    "make_latent_feature_batch",
                                    (owner, runner, &size, tables, values.get_item(0)?),
                                )?;
                                eager(runner, &batch)
                            },
                        )
                    },
                )?;
            }
            Ok(())
        })?;
    }
    Ok(())
}
