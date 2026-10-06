//! Standalone denoiser ownership, layout residency and startup ordering.

use std::time::Instant;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use super::modules::{ensure_open, input_error};
use super::{ModelRunners, dispatch, execution, graphs};
use crate::worker::execution::Execution;
use crate::worker::host::with_context;
use crate::worker::model_results::ExecutionOutput;

pub(super) fn runner<'py>(
    slf: &Bound<'py, ModelRunners>,
    owner: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    ensure_open(owner)?;
    let py = slf.py();
    if let Some(runner) = &slf.borrow().diffusion {
        return Ok(runner.bind(py).clone());
    }

    let binding = owner.getattr("_denoiser")?;
    let pool = owner.getattr("latent_pool")?;
    if binding.is_none() || pool.is_none() {
        return Err(input_error(py, "rank does not own denoising computation"));
    }

    let name: String = binding.get_item(0)?.extract()?;
    let call = binding.get_item(2)?;
    let method: String = call.getattr("entry_point")?.getattr("method")?.extract()?;
    let path: String = call.getattr("path")?.extract()?;
    let module = slf.borrow().module(py, &name, Some(&method), Some(&path))?;
    let stream = module.stream(owner)?;
    let device = binding.get_item(1)?.getattr("device")?;
    let config = owner.getattr("worker_config")?;
    let config_native = crate::worker::config::native(&config)?;
    let captures = !stream.is_none() && config_native.graph_policy != "off";

    let builder = owner.getattr("media_builder")?;
    let options = PyDict::new(py);
    options.set_item("device", &device)?;
    options.set_item("stream", &stream)?;
    options.set_item("storage", owner.getattr("graph_storage")?)?;
    let mut devices = Vec::new();
    if captures {
        devices.push(device);
        devices.extend(super::streams::capture_devices(owner, &devices[0])?);
    }
    options.set_item("devices", PyTuple::new(py, devices)?)?;
    options.set_item(
        "bank",
        captures
            .then(|| owner.getattr("diffusion_bank"))
            .transpose()?,
    )?;
    options.set_item("slots", config_native.max_request_pool_size)?;
    options.set_item("pool", pool)?;
    options.set_item("pages", builder.getattr("sample_pages")?.getattr("pages")?)?;
    options.set_item("attention", owner.getattr("attention")?)?;

    let runner = py
        .import("uniserve_worker.model_executor.diffusion_runner")?
        .getattr("DiffusionRunner")?
        .call_method(
            "for_layouts",
            (name, call, builder.getattr("maximum_layout")?),
            Some(&options),
        )?;

    let mut runners = slf.borrow_mut();
    runners.diffusion = Some(runner.clone().unbind());
    runners.diffusion_layouts = Some(PyDict::new(py).unbind());
    Ok(runner)
}

pub(super) fn prepare_layouts<'py>(
    slf: &Bound<'py, ModelRunners>,
    owner: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let runner = runner(slf, owner)?;
    let builder = owner.getattr("media_builder")?;
    let layouts = builder.call_method0("layouts")?.cast_into::<PyTuple>()?;
    for layout in std::iter::once(builder.getattr("maximum_layout")?).chain(layouts.iter()) {
        Execution::prepare_denoising(
            &execution(&runner)?,
            &runner,
            &layout,
            builder
                .call_method1("layout_pages", (&layout,))?
                .extract()?,
        )?;
    }
    owner.getattr("graph_storage")?.call_method0("check")?;
    Ok(layouts)
}

pub(super) fn layout<'py>(
    slf: &Bound<'py, ModelRunners>,
    owner: &Bound<'py, PyAny>,
    key: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let py = slf.py();
    let runner = runner(slf, owner)?;
    let execution = execution(&runner)?;
    let serving = slf
        .borrow()
        .diffusion_layouts
        .as_ref()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("denoising runner is closed"))?
        .bind(py)
        .clone();
    if let Some(value) = serving.get_item(key)? {
        serving.del_item(key)?;
        serving.set_item(key, &value)?;
        return Ok(value.unbind());
    }
    if execution.borrow().has_denoising_layout(py, key)? {
        return Ok(Execution::denoising_layout(&execution, key)?
            .into_any()
            .unbind());
    }

    let capacity: usize =
        crate::worker::config::native(&owner.getattr("worker_config")?)?.max_request_pool_size;
    if serving.len() >= capacity
        && let Some((key, _)) = serving.iter().next()
    {
        Execution::retire_denoising(&execution, &key)?;
        serving.del_item(key)?;
    }

    let pages = owner
        .getattr("media_builder")?
        .call_method1("layout_pages", (key,))?
        .extract()?;
    let buffers = Execution::prepare_denoising(&execution, &runner, key, pages)?;
    serving.set_item(key, &buffers)?;
    Ok(buffers.into_any().unbind())
}

pub(super) fn run(
    slf: &Bound<'_, ModelRunners>,
    owner: &Bound<'_, PyAny>,
    ladder: &Bound<'_, PyAny>,
    index: usize,
    bank: i64,
) -> PyResult<Py<ExecutionOutput>> {
    let py = slf.py();
    let runner = runner(slf, owner)?;
    let started = Instant::now();
    let rank: usize = crate::worker::config::native(&owner.getattr("worker_config")?)?.rank;
    let (values, path) = with_context(
        &dispatch::profile(
            py,
            &format!("uniserve.model.denoise rank={rank} work=denoiser"),
        )?,
        || Execution::step_denoising(&execution(&runner)?, &runner, ladder, index, bank),
    )?;

    let result = runner
        .call_method1("result", (values,))?
        .cast_into::<ExecutionOutput>()?
        .unbind();
    graphs::record_stats(py, &result, "denoiser", path, started)?;
    Ok(result)
}

pub(super) fn prepare(
    slf: &Bound<'_, ModelRunners>,
    owner: &Bound<'_, PyAny>,
    storage: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let py = slf.py();
    with_context(&py.import("torch")?.call_method0("inference_mode")?, || {
        let started = Instant::now();
        let layouts = prepare_layouts(slf, owner)?;
        let prepared = started.elapsed().as_secs_f64();
        let runner = runner(slf, owner)?;
        let execution = execution(&runner)?;
        let builder = owner.getattr("media_builder")?;
        let numerical = py.import("uniserve_worker.execution.media")?;
        let schedules = numerical
            .call_method1("open_state", (owner, builder.getattr("maximum")?))?
            .getattr("schedules")?;

        let bind = |layout: &Bound<'_, PyAny>, initialize: bool| {
            numerical
                .call_method1(
                    "denoising_inputs",
                    (owner, storage.get_item(0)?, &schedules, layout, initialize),
                )
                .map(Bound::unbind)
        };
        let graph_storage = owner.getattr("graph_storage")?;
        let maximum = builder.getattr("maximum_layout")?;

        // Every layout warms before any graph captures, because persistent
        // plans and captured intermediates share the same allocation pool.
        let extra = (!layouts.contains(&maximum)?).then_some(maximum);
        for layout in extra.into_iter().chain(layouts.iter()) {
            Execution::warm_denoising(&execution, &runner, &bind(&layout, true)?.into_bound(py))?;
            graph_storage.call_method0("check")?;
        }
        let warmed = started.elapsed().as_secs_f64();
        let captures = runner.getattr("captures")?.is_truthy()?;
        if captures {
            for layout in layouts.iter() {
                Execution::capture_denoising(
                    &execution,
                    &runner,
                    &bind(&layout, false)?.into_bound(py),
                )?;
                graph_storage.call_method0("check")?;
            }
        }

        let resident = graph_storage
            .call_method0("pool_bytes")?
            .cast_into::<PyDict>()?
            .values()
            .iter()
            .map(|value| value.extract::<u64>())
            .sum::<PyResult<u64>>()?;
        let finished = started.elapsed().as_secs_f64();
        let logger = py
            .import("logging")?
            .call_method1("getLogger", ("uniserve_worker.execution.media",))?;
        let graph_count = if captures { layouts.len() } else { 0 };
        logger.call_method1(
            "info",
            (
                concat!(
                    "prepared %d denoiser layouts (%d step graphs) in %.1f s ",
                    "(contexts %.1f s, warm steps %.1f s, capture %.1f s); ",
                    "graph storage %.2f GiB"
                ),
                layouts.len(),
                graph_count,
                finished,
                prepared,
                warmed - prepared,
                finished - warmed,
                resident as f64 / (1u64 << 30) as f64,
            ),
        )?;
        Ok(())
    })
}
