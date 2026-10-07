//! Resolve numerical modules and provision their input capacity at startup.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::worker::config::WorkerConfig;

/// Ordinary module traversal deduplicates modules shared under several paths.
#[pyfunction(name = "model_capability")]
pub(in crate::worker) fn capability<'py>(
    model: &Bound<'py, PyAny>,
    kind: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut found = None;
    for module in model.call_method0("modules")?.try_iter()? {
        let module = module?;
        if module.is_instance(kind)? {
            if found.is_some() {
                return Err(PyValueError::new_err(format!(
                    "worker requires an unambiguous {} capability",
                    kind.getattr("__name__")?,
                )));
            }
            found = Some(module);
        }
    }
    Ok(found.unwrap_or_else(|| model.py().None().into_bound(model.py())))
}

#[pyfunction(name = "image_input_builder")]
pub(in crate::worker) fn image_builder<'py>(
    model: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = model.py();
    let denoiser = capability(
        model,
        &py.import("uniserve.model")?.getattr("ImageDenoiser")?,
    )?;
    if denoiser.is_none() {
        return Ok(denoiser);
    }
    py.import("uniserve_worker.model_executor.diffusion_inputs")?
        .call_method1("ImageBuilder", (denoiser,))
}

/// A multi-task checkpoint can carry several denoisers; placement chooses one.
#[pyfunction]
pub(in crate::worker) fn video_denoiser<'py>(
    model: &Bound<'py, PyAny>,
    config: &Bound<'py, WorkerConfig>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = model.py();
    let kind = py.import("uniserve.model")?.getattr("VideoDenoiser")?;
    let mut candidates = Vec::new();
    for module in model.call_method0("modules")?.try_iter()? {
        let module = module?;
        if module.is_instance(&kind)? {
            candidates.push(module);
        }
    }
    if candidates.len() <= 1 {
        return Ok(candidates.pop().unwrap_or_else(|| py.None().into_bound(py)));
    }

    let config = Arc::clone(&config.borrow().inner);
    let declared = py
        .import("uniserve_worker.bootstrap.components")?
        .call_method1("describe_components", (model,))?
        .cast_into::<PyDict>()?;
    let mut placed = Vec::new();
    let mut names = Vec::new();
    for (name, calls) in &declared {
        let name: String = name.extract()?;
        let mut owns_denoiser = false;
        for call in calls.try_iter()? {
            let module = call?.getattr("module")?;
            if let Some(index) = candidates.iter().position(|candidate| module.is(candidate)) {
                owns_denoiser = true;
                if config.deployment_components.contains(&name) && !placed.contains(&index) {
                    placed.push(index);
                }
            }
        }
        if owns_denoiser {
            names.push(name);
        }
    }
    if let [index] = placed.as_slice() {
        return Ok(candidates.swap_remove(*index));
    }

    names.sort();
    Err(PyValueError::new_err(format!(
        "a deployment places exactly one of the model's video denoisers {names:?}; it places {}",
        placed.len(),
    )))
}

fn tasks(denoiser: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let mut tasks: Vec<String> = denoiser.getattr("tasks")?.extract()?;
    tasks.retain(|task| matches!(task.as_str(), "t2va" | "fl2va" | "ref2va"));
    Ok(tasks)
}

#[pyfunction]
pub(in crate::worker) fn executed_video_tasks<'py>(
    denoiser: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(denoiser.py(), tasks(denoiser)?)
}

/// Prompt-only generation needs no condition rows, regardless of its allowance.
#[pyfunction]
pub(in crate::worker) fn condition_capacity(
    denoiser: &Bound<'_, PyAny>,
    config: &Bound<'_, WorkerConfig>,
) -> PyResult<usize> {
    Ok(if tasks(denoiser)?.iter().any(|task| task != "t2va") {
        config.borrow().inner.max_condition_rows
    } else {
        0
    })
}

#[pyfunction(name = "media_input_builder")]
pub(in crate::worker) fn media_builder<'py>(
    model: &Bound<'py, PyAny>,
    config: &Bound<'py, WorkerConfig>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = model.py();
    let denoiser = video_denoiser(model, config)?;
    if denoiser.is_none() {
        return Ok(denoiser);
    }
    let output = capability(
        model,
        &py.import("uniserve.model")?.getattr("VideoPostprocessor")?,
    )?;
    if output.is_none() {
        return Err(PyValueError::new_err(
            "media input construction requires its output sampling clock",
        ));
    }

    let conditions = condition_capacity(&denoiser, config)?;
    let config = Arc::clone(&config.borrow().inner);
    let rate: f64 = output.getattr("frame_rate")?.extract()?;
    // The HTTP request planner uses the same ties-to-even rule for duration.
    let frames = |seconds: f64| (seconds * rate).round_ties_even() as usize;
    let options = PyDict::new(py);
    options.set_item("max_frames", frames(config.max_video_seconds))?;
    options.set_item("min_frames", config.min_video_seconds.map_or(1, frames))?;
    options.set_item("max_text_tokens", config.max_sequence_tokens)?;
    options.set_item("text_capacities", &config.video_text_capacities)?;
    options.set_item("condition_rows", conditions)?;
    py.import("uniserve_worker.model_executor.media_inputs")?
        .getattr("MediaBuilder")?
        .call((denoiser,), Some(&options))
}

pub(in crate::worker) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(capability, module)?)?;
    module.add_function(wrap_pyfunction!(image_builder, module)?)?;
    module.add_function(wrap_pyfunction!(video_denoiser, module)?)?;
    module.add_function(wrap_pyfunction!(executed_video_tasks, module)?)?;
    module.add_function(wrap_pyfunction!(condition_capacity, module)?)?;
    module.add_function(wrap_pyfunction!(media_builder, module)?)?;
    Ok(())
}
