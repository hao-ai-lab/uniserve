//! Validate component placement and bind rank-local numerical calls.

use std::collections::BTreeMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use super::ComponentBinding;
use crate::worker::error::unsupported;
use crate::worker::placement::ComponentConfig;

/// Check deployment policy against a model's numerical declarations before
/// constructing process groups, or reuse declarations from a loaded model.
#[pyfunction]
#[pyo3(signature = (model, components, *, entries=None, declarations=None))]
pub(in crate::worker) fn validate_components<'py>(
    model: &Bound<'py, PyAny>,
    components: &Bound<'py, PyAny>,
    entries: Option<&Bound<'py, PyAny>>,
    declarations: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let py = model.py();
    let bootstrap = py.import("uniserve_worker.bootstrap.components")?;
    let declared = if let Some(declared) = declarations {
        declared.clone()
    } else {
        let options = PyDict::new(py);
        options.set_item("entries", entries)?;
        bootstrap
            .call_method("describe_components", (model,), Some(&options))?
            .cast_into::<PyDict>()?
    };
    let placements = placements(components)?;
    let unknown = placements
        .keys()
        .filter_map(|name| match declared.contains(name) {
            Ok(true) => None,
            Ok(false) => Some(Ok(name)),
            Err(error) => Some(Err(error)),
        })
        .collect::<PyResult<Vec<_>>>()?;
    if !unknown.is_empty() {
        return Err(unsupported(py, format!("unknown components {unknown:?}")));
    }

    let types = py.import("uniserve.model")?;
    let video_decoder = types.getattr("VideoDecoder")?;
    let audio_decoder = types.getattr("AudioDecoder")?;
    let video_encoder = types.getattr("VideoEncoder")?;
    let host = bootstrap.getattr("HOST_COMPONENTS")?;
    let muxer: String = bootstrap.getattr("MUXER_COMPONENT")?.extract()?;
    let reader: String = bootstrap.getattr("MEDIA_READER_COMPONENT")?.extract()?;
    let codec: String = bootstrap.getattr("VIDEO_CODEC_COMPONENT")?.extract()?;
    for (name, placement) in placements {
        let placement = &placement.get().inner;
        let calls = declared.as_any().get_item(&name)?;
        let temporal = placement.distribution.is_some();
        if calls.is_empty()? {
            if !host.contains(&name)? {
                return Err(unsupported(
                    py,
                    format!("component {name:?} has no numerical methods"),
                ));
            }
            if name == muxer && temporal {
                return Err(unsupported(
                    py,
                    "the muxer assembles one artifact and is not distributed",
                ));
            }
            if name == reader && temporal {
                return Err(unsupported(
                    py,
                    "the media reader reads one request's conditions on one rank and is not distributed",
                ));
            }
            if name == codec && (!temporal || placement.units_per_rank != 1) {
                return Err(unsupported(
                    py,
                    "video encoding distributes by temporal_units with one media unit per rank, each rank one codec slot",
                ));
            }
            continue;
        }

        let mut video = false;
        let mut media = false;
        for call in calls.try_iter()? {
            let module = call?.getattr("module")?;
            video |= module.is_instance(&video_decoder)?;
            media |= module.is_instance(&audio_decoder)? || module.is_instance(&video_encoder)?;
        }
        if video {
            if !temporal || placement.units_per_rank != 1 {
                return Err(unsupported(
                    py,
                    "video decoding requires temporal_units with one native unit per rank",
                ));
            }
        } else if !media && temporal {
            return Err(unsupported(
                py,
                format!("entry {name:?} requires model-parallel membership"),
            ));
        }
        // ComponentConfig already limits distribution to temporal units and
        // requires a positive unit count. Audio and encoding add no restriction.
    }
    Ok(declared)
}

fn placements(components: &Bound<'_, PyAny>) -> PyResult<BTreeMap<String, Py<ComponentConfig>>> {
    components
        .call_method0("items")?
        .try_iter()?
        .map(|item| {
            let item = item?;
            Ok((item.get_item(0)?.extract()?, item.get_item(1)?.extract()?))
        })
        .collect()
}

/// Borrow numerical calls only on their participating pipeline stages. Calls
/// retain communicator views; ProcessGroups remains the owner of their handles.
#[pyfunction]
#[pyo3(signature = (model, bindings, *, entries=None, declarations=None))]
pub(in crate::worker) fn bind_components(
    model: &Bound<'_, PyAny>,
    bindings: &Bound<'_, PyAny>,
    entries: Option<&Bound<'_, PyAny>>,
    declarations: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    let py = model.py();
    let mut local = Vec::new();
    let components = PyDict::new(py);
    for item in bindings.call_method0("items")?.try_iter()? {
        let item = item?;
        let name: String = item.get_item(0)?.extract()?;
        let binding: Py<ComponentBinding> = item.get_item(1)?.extract()?;
        components.set_item(&name, &binding.borrow(py).config)?;
        local.push((name, binding));
    }
    let declared = validate_components(model, components.as_any(), entries, declarations)?;
    let bootstrap = py.import("uniserve_worker.bootstrap.components")?;
    let muxer: String = bootstrap.getattr("MUXER_COMPONENT")?.extract()?;
    let postprocessor = py.import("uniserve.model")?.getattr("VideoPostprocessor")?;
    let distributed = py.import("uniserve.distributed")?;
    let replace = py.import("dataclasses")?.getattr("replace")?;

    for (name, binding) in local {
        let (mesh, units) = {
            let mut binding = binding.borrow_mut(py);
            binding.calls = PyTuple::empty(py).unbind();
            if !binding.owns() {
                continue;
            }
            let Some(mesh) = &binding.mesh else { continue };
            (
                mesh.clone_ref(py),
                binding.units.as_ref().map(|units| units.clone_ref(py)),
            )
        };
        let mesh = mesh.bind(py);
        let pipeline = mesh.call_method1("get_group", ("pp",))?;
        let rank: usize = pipeline.getattr("rank")?.extract()?;
        let size: usize = pipeline.getattr("size")?.extract()?;
        let mut calls = Vec::new();
        for call in declared.as_any().get_item(&name)?.try_iter()? {
            let call = call?;
            let point = call.getattr("entry_point")?;
            let stage: String = point.getattr("stage")?.extract()?;
            if (stage == "first" && rank != 0) || (stage == "last" && rank != size - 1) {
                continue;
            }
            let module = call.getattr("module")?;
            let groups = PyDict::new(py);
            for group in distributed
                .call_method1("communicators", (&module,))?
                .try_iter()?
            {
                let group = group?;
                let axes: String = group.getattr("name")?.extract()?;
                if stage == "all" || !axes.split('.').any(|axis| axis == "pp") {
                    groups.set_item(group.call_method0("_require")?, &group)?;
                }
            }

            // A temporal decoder has a local numerical mesh but still exchanges
            // reconstruction overlap with its neighbors through the unit ring.
            if let Some(units) = &units
                && module.is_instance(&postprocessor)?
            {
                let units = units.bind(py);
                module.setattr("units", units)?;
                if units.getattr("size")?.extract::<usize>()? > 1 {
                    groups.set_item(units.call_method0("_require")?, units)?;
                }
            }
            for axes in point
                .call_method1("communication_axes", (mesh,))?
                .try_iter()?
            {
                let group = mesh.call_method1("get_group", (axes?,))?;
                if group.getattr("size")?.extract::<usize>()? > 1 {
                    groups.set_item(group.call_method0("_require")?, &group)?;
                }
            }
            let fields = PyDict::new(py);
            fields.set_item("groups", PyTuple::new(py, groups.values())?)?;
            calls.push(replace.call((call,), Some(&fields))?);
        }
        let calls = PyTuple::new(py, calls)?;
        let kinds = if name == muxer {
            bootstrap.getattr("MUXER_CALL_KINDS")?
        } else {
            super::calls::call_kinds(calls.as_any())?.into_any()
        };
        let kinds = PyTuple::new(py, kinds.try_iter()?.collect::<PyResult<Vec<_>>>()?)?;
        let mut binding = binding.borrow_mut(py);
        binding.calls = calls.unbind();
        binding.call_kinds = kinds.unbind();
    }
    Ok(())
}
