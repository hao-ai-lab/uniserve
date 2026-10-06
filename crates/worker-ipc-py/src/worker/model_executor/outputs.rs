//! Resolve numerical output layouts and this rank's scheduled media units.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};

use super::{ModelExecutor, input_error};

#[allow(clippy::too_many_arguments)]
pub(super) fn layout(
    owner: &Bound<'_, ModelExecutor>,
    component: &str,
    index: usize,
    media: &Bound<'_, PyAny>,
    decode: &Bound<'_, PyAny>,
    prompt_tokens: usize,
    conditions: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let py = owner.py();
    let (bindings, outputs, declarations, config, builder, clock, decoder) = {
        let owner = owner.borrow();
        (
            owner.bindings.bind(py).clone(),
            owner.outputs.bind(py).clone(),
            owner.declarations.bind(py).clone(),
            owner.worker_config.bind(py).clone(),
            owner.media_builder.bind(py).clone(),
            owner.video_postprocessor.bind(py).clone(),
            owner.video_decoder.bind(py).clone(),
        )
    };
    let binding = bindings.call_method1("get", (component,))?;
    if !binding.is_none()
        && !binding
            .getattr("output_ranks")?
            .contains(binding.getattr("process_group")?.getattr("rank")?)?
    {
        return Ok(py.None());
    }

    let condition = py.import("uniserve_worker.execution.conditions")?;
    let declared = outputs.call_method1("get", (component, PyTuple::empty(py)))?;
    if index < declared.len()? {
        let output = declared.get_item(index)?;
        if condition
            .getattr("CONDITION_PRODUCTS")?
            .contains(output.getattr("name")?)?
        {
            let conditions = conditions
                .ok_or_else(|| input_error(py, "a condition product needs its conditions"))?;
            return Ok(condition
                .call_method1("condition_layout", (output, conditions, decode, &binding))?
                .unbind());
        }
    }

    let resources = py.import("uniserve_worker.model_executor.resources")?;
    let mut results = Vec::new();
    let video_codec: String = py
        .import("uniserve_worker.bootstrap.components")?
        .getattr("VIDEO_CODEC_COMPONENT")?
        .extract()?;
    if component == video_codec {
        // A codec row bounds the largest encoded media unit at every canvas.
        let mut layouts = Vec::new();
        for size in builder.call_method0("video_sizes")?.try_iter()? {
            layouts.push(resources.call_method1("encoded_units_layout", (&decoder, size?))?);
        }
        let layout = resources.call_method1("bounding_layout", (PyTuple::new(py, layouts)?,))?;
        results.push((py.None().into_bound(py), layout));
    } else {
        let options = PyDict::new(py);
        options.set_item("builder", &builder)?;
        options.set_item("clock", &clock)?;
        options.set_item(
            "frames",
            if media.is_none() {
                py.None().into_bound(py)
            } else {
                media.getattr("num_frames")?
            },
        )?;
        options.set_item(
            "canvas",
            if media.is_none() {
                py.None().into_bound(py)
            } else {
                media.getattr("canvas")?
            },
        )?;
        options.set_item("prompt_tokens", prompt_tokens)?;
        let conditions = match conditions {
            Some(conditions) => condition.call_method1("library_conditions", (conditions,))?,
            None => PyTuple::empty(py).into_any(),
        };
        options.set_item("conditions", conditions)?;
        let calls = declarations
            .get_item(component)?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(component.to_owned()))?;
        for call in calls.try_iter()? {
            let call = call?;
            let module = call.getattr("module")?;
            let layouts = resources
                .call_method("output_layouts", (&config, &call), Some(&options))?
                .cast_into::<PyDict>()?;
            results.extend(layouts.iter().map(|(_, layout)| (module.clone(), layout)));
        }
    }
    let (module, layout) = results.get(index).ok_or_else(|| {
        pyo3::exceptions::PyIndexError::new_err("output index exceeds component outputs")
    })?;
    let types = py.import("uniserve.model")?;
    if module.is_instance(&types.getattr("Denoiser")?)? {
        for axis in layout.getattr("local_slice")?.try_iter()? {
            let axis = axis?;
            if axis.getattr("stop")?.eq(axis.getattr("start")?)? {
                return Ok(py.None());
            }
        }
    }
    if binding.is_none()
        || binding
            .getattr("config")?
            .getattr("distribution")?
            .is_none()
    {
        return Ok(layout.clone().unbind());
    }

    // Video leads with media units; audio leads with samples. Placement gives
    // the audio unit count, and the decoder maps each unit to its sample span.
    let audio = module.is_instance(&types.getattr("AudioDecoder")?)?;
    let shape = layout.getattr("shape")?.cast_into::<PyTuple>()?;
    let total = if audio {
        let config = binding.getattr("config")?;
        config.getattr("ranks")?.len()?
            * config.getattr("units_per_rank")?.extract::<usize>()?.max(1)
    } else {
        shape.get_item(0)?.extract()?
    };
    let units = if decode.is_none() {
        total
    } else {
        decode.getattr("max_units")?.extract()?
    };
    let run = binding.call_method1("media_units", (0, units))?;
    if !run.is_truthy()? {
        return Ok(py.None());
    }
    let start: usize = run.getattr("start")?.extract()?;
    let stop: usize = run.getattr("stop")?.extract()?;
    let options = PyDict::new(py);
    let (first, last) = if audio {
        let spans = module.call_method1("unit_samples", (shape.get_item(0)?, units))?;
        (
            spans.get_item(start)?.getattr("start")?.extract()?,
            spans.get_item(stop - 1)?.getattr("stop")?.extract()?,
        )
    } else {
        let shape =
            std::iter::once(units.into_pyobject(py)?.into_any()).chain(shape.iter().skip(1));
        options.set_item("shape", PyTuple::new(py, shape.collect::<Vec<_>>())?)?;
        (start, stop)
    };
    let axes = layout.getattr("local_slice")?.cast_into::<PyTuple>()?;
    let axes =
        std::iter::once(py.get_type::<PySlice>().call1((first, last))?).chain(axes.iter().skip(1));
    options.set_item("local_slice", PyTuple::new(py, axes.collect::<Vec<_>>())?)?;
    Ok(py
        .import("dataclasses")?
        .getattr("replace")?
        .call((layout,), Some(&options))?
        .unbind())
}
