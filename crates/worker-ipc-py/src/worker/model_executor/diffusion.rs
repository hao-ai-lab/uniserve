//! Standalone denoiser ownership, layout residency and startup ordering.

use std::time::Instant;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::sync::Arc;

use super::modules::{ensure_open, input_error};
use super::{ModelExecutor, dispatch, execution, graphs};
use crate::worker::execution::Execution;
use crate::worker::host::with_context;
use crate::worker::model_results::ExecutionOutput;

pub(super) fn runner<'py>(slf: &Bound<'py, ModelExecutor>) -> PyResult<Bound<'py, PyAny>> {
    ensure_open(slf)?;
    let py = slf.py();
    if let Some(runner) = &slf.borrow().diffusion {
        return Ok(runner.bind(py).clone());
    }

    let pool = slf.borrow().latent_pool.bind(py).clone();
    let module = slf
        .borrow()
        .modules
        .values()
        .rev()
        .find(|module| module.denoiser)
        .cloned();
    let Some(module) = module.filter(|_| slf.borrow().denoises(py) && !pool.is_none()) else {
        return Err(input_error(py, "rank does not own denoising computation"));
    };
    let name = &module.name;
    let call = module.call.bind(py);
    let stream = module.stream(slf)?;
    let device = module.binding.bind(py).getattr("device")?;
    let config_native = Arc::clone(&slf.borrow().config);
    let captures = !stream.is_none() && config_native.graph_policy != "off";

    let builder = slf.borrow().media_builder.bind(slf.py()).clone();
    let options = PyDict::new(py);
    options.set_item("device", &device)?;
    options.set_item("stream", &stream)?;
    let storage = slf.borrow().graph_storage(py)?;
    options.set_item("storage", storage)?;
    let mut devices = Vec::new();
    if captures {
        devices.push(device);
        devices.extend(super::streams::capture_devices(slf, &devices[0])?);
    }
    options.set_item("devices", PyTuple::new(py, devices)?)?;
    options.set_item(
        "bank",
        captures.then(|| slf.borrow().diffusion_bank.bind(py).clone()),
    )?;
    options.set_item("slots", config_native.max_request_pool_size)?;
    options.set_item("pool", pool)?;
    options.set_item("pages", builder.getattr("sample_pages")?.getattr("pages")?)?;
    options.set_item("attention", {
        slf.borrow().attention.bind(slf.py()).clone()
    })?;

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
    slf: &Bound<'py, ModelExecutor>,
) -> PyResult<Bound<'py, PyTuple>> {
    let runner = runner(slf)?;
    let builder = slf.borrow().media_builder.bind(slf.py()).clone();
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
    slf.borrow()
        .graph_storage(slf.py())?
        .borrow()
        .check(slf.py())?;
    Ok(layouts)
}

pub(super) fn layout<'py>(
    slf: &Bound<'py, ModelExecutor>,
    key: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let py = slf.py();
    let runner = runner(slf)?;
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

    let capacity: usize = Arc::clone(&slf.borrow().config).max_request_pool_size;
    if serving.len() >= capacity
        && let Some((key, _)) = serving.iter().next()
    {
        Execution::retire_denoising(&execution, &key)?;
        serving.del_item(key)?;
    }

    let pages = { slf.borrow().media_builder.bind(slf.py()).clone() }
        .call_method1("layout_pages", (key,))?
        .extract()?;
    let buffers = Execution::prepare_denoising(&execution, &runner, key, pages)?;
    serving.set_item(key, &buffers)?;
    Ok(buffers.into_any().unbind())
}

pub(super) fn run(
    slf: &Bound<'_, ModelExecutor>,
    ladder: &Bound<'_, PyAny>,
    index: usize,
    bank: i64,
) -> PyResult<Py<ExecutionOutput>> {
    let py = slf.py();
    let runner = runner(slf)?;
    let started = Instant::now();
    let rank: usize = Arc::clone(&slf.borrow().config).rank;
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

pub(super) fn prepare(slf: &Bound<'_, ModelExecutor>, storage: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = slf.py();
    with_context(&py.import("torch")?.call_method0("inference_mode")?, || {
        let started = Instant::now();
        let layouts = prepare_layouts(slf)?;
        let prepared = started.elapsed().as_secs_f64();
        let runner = runner(slf)?;
        let execution = execution(&runner)?;
        let builder = slf.borrow().media_builder.bind(slf.py()).clone();
        let numerical = py.import("uniserve_worker.execution.media")?;
        let schedules = numerical
            .call_method1("open_state", (slf, builder.getattr("maximum")?))?
            .getattr("schedules")?;

        let bind = |layout: &Bound<'_, PyAny>, initialize: bool| {
            numerical
                .call_method1(
                    "denoising_inputs",
                    (slf, storage.get_item(0)?, &schedules, layout, initialize),
                )
                .map(Bound::unbind)
        };
        let graph_storage = slf.borrow().graph_storage(slf.py())?;
        let maximum = builder.getattr("maximum_layout")?;

        // Every layout warms before any graph captures, because persistent
        // plans and captured intermediates share the same allocation pool.
        let extra = (!layouts.contains(&maximum)?).then_some(maximum);
        for layout in extra.into_iter().chain(layouts.iter()) {
            Execution::warm_denoising(&execution, &runner, &bind(&layout, true)?.into_bound(py))?;
            graph_storage.borrow().check(py)?;
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
                graph_storage.borrow().check(py)?;
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
