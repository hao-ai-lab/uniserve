//! Numerical call classification and component routes advertised to the engine.

use std::collections::{BTreeMap, BTreeSet};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet};
use uniserve_worker_ipc::{CallKind, ForwardMode, MediaCall, TransferMode};

use crate::worker::error::unsupported;

pub(in crate::worker) const HOST_COMPONENTS: [(&str, &[MediaCall]); 3] = [
    ("media_reader", &[MediaCall::MediaReading]),
    ("video_codec", &[MediaCall::VideoEncoding]),
    ("muxer", &[MediaCall::AudioEncoding, MediaCall::Muxing]),
];

#[pyfunction]
pub(in crate::worker) fn is_host_component(name: &str) -> bool {
    HOST_COMPONENTS.iter().any(|(host, _)| *host == name)
}

#[pyfunction]
pub(in crate::worker) fn holds_host_components(names: &Bound<'_, PyAny>) -> PyResult<bool> {
    for name in names.try_iter()? {
        if is_host_component(name?.extract::<String>()?.as_str()) {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(in crate::worker) fn kinds(calls: &Bound<'_, PyAny>) -> PyResult<BTreeSet<CallKind>> {
    use CallKind::{Forward, Media};
    use MediaCall::*;

    let types = calls.py().import("uniserve.model")?;
    let patch_vae = calls
        .py()
        .import("uniserve.nn.vae")?
        .getattr("PatchAutoencoder")?;
    let mut kinds = BTreeSet::new();
    for call in calls.try_iter()? {
        let call = call?;
        let module = call.getattr("module")?;
        let method: String = call.getattr("entry_point")?.getattr("method")?.extract()?;
        let is = |name| module.is_instance(&types.getattr(name)?);
        match method.as_str() {
            "forward" => {
                if is("CausalLM")? {
                    kinds.extend([
                        Forward(ForwardMode::Prefill),
                        Forward(ForwardMode::Decode),
                        Forward(ForwardMode::Verify),
                    ]);
                } else if is("TokenDenoiser")? {
                    kinds.insert(Forward(ForwardMode::TokenDenoising));
                } else if is("Denoiser")? {
                    kinds.extend([Media(LatentPreparation), Media(Denoising)]);
                } else if is("VideoPostprocessor")? {
                    kinds.insert(Media(VideoDecoding));
                }
            }
            "encode" => {
                let kind = if is("TextEncoder")? {
                    Some(TextEncoding)
                } else if is("PatchEncoder")? {
                    Some(VisionEncoding)
                } else if module.is_instance(&patch_vae)?
                    || is("VideoEncoder")?
                    || is("AudioEncoder")?
                {
                    Some(LatentEncoding)
                } else if is("Encoder")? {
                    Some(LatentPreparation)
                } else {
                    None
                };
                kinds.extend(kind.map(Media));
            }
            "decode" => {
                let kind = if is("ImageDecoder")? {
                    Some(ImageDecoding)
                } else if is("VideoDecoder")? {
                    Some(VideoDecoding)
                } else if is("AudioDecoder")? {
                    Some(AudioDecoding)
                } else {
                    None
                };
                kinds.extend(kind.map(Media));
            }
            _ => {}
        }
    }
    Ok(kinds)
}

#[pyfunction]
pub(in crate::worker) fn call_kinds<'py>(
    calls: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyFrozenSet>> {
    python_kinds(calls.py(), kinds(calls)?)
}

fn components<'py>(
    model: &Bound<'py, PyAny>,
    held: &[String],
) -> PyResult<Vec<(String, Bound<'py, PyAny>)>> {
    let declared = model
        .py()
        .import("uniserve_worker.bootstrap.components")?
        .call_method1("describe_components", (model,))?;
    let mut result = Vec::new();
    for item in declared.cast::<PyDict>()?.iter() {
        let name: String = item.0.extract()?;
        if held.is_empty() || held.contains(&name) {
            result.push((name, item.1));
        }
    }
    Ok(result)
}

fn held_names(held: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<String>> {
    match held {
        Some(held) => held.try_iter()?.map(|name| name?.extract()).collect(),
        None => Ok(Vec::new()),
    }
}

pub(in crate::worker) fn supported(
    model: &Bound<'_, PyAny>,
    held: &[String],
) -> PyResult<BTreeSet<CallKind>> {
    let causal = model.py().import("uniserve.model")?.getattr("CausalLM")?;
    let mut result = BTreeSet::from([CallKind::Transfer(TransferMode::Tensor)]);
    for (name, calls) in components(model, held)? {
        result.extend(kinds(&calls)?);
        if let Some((_, kinds)) = HOST_COMPONENTS.iter().find(|(host, _)| *host == name) {
            result.extend(kinds.iter().copied().map(CallKind::Media));
        }
        for call in calls.try_iter()? {
            if call?.getattr("module")?.is_instance(&causal)? {
                result.extend([
                    CallKind::Transfer(TransferMode::KvExport),
                    CallKind::Transfer(TransferMode::KvInstall),
                ]);
                break;
            }
        }
    }
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (model, held=None))]
pub(in crate::worker) fn supported_calls<'py>(
    model: &Bound<'py, PyAny>,
    held: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyFrozenSet>> {
    python_kinds(model.py(), supported(model, &held_names(held)?)?)
}

pub(in crate::worker) fn routes(
    model: &Bound<'_, PyAny>,
    held: &[String],
) -> PyResult<BTreeMap<MediaCall, String>> {
    let mut owners = BTreeMap::<MediaCall, Vec<String>>::new();
    for (name, calls) in components(model, held)? {
        let kinds = if let Some((_, kinds)) = HOST_COMPONENTS.iter().find(|(host, _)| *host == name)
        {
            kinds.iter().copied().collect::<BTreeSet<_>>()
        } else {
            kinds(&calls)?
                .into_iter()
                .filter_map(|kind| match kind {
                    CallKind::Media(kind) => Some(kind),
                    _ => None,
                })
                .collect()
        };
        for kind in kinds {
            owners.entry(kind).or_default().push(name.clone());
        }
    }
    // Ambiguous computations have no single component route. The engine checks
    // that the selected deployment supplies every required media operation.
    let routes: BTreeMap<_, _> = owners
        .into_iter()
        .filter_map(|(kind, names)| match names.as_slice() {
            [name] => Some((kind, name.clone())),
            _ => None,
        })
        .collect();
    if let (Some(prepare), Some(denoise)) = (
        routes.get(&MediaCall::LatentPreparation),
        routes.get(&MediaCall::Denoising),
    ) && prepare != denoise
    {
        return Err(unsupported(
            model.py(),
            "latent preparation must participate in the denoiser component",
        ));
    }
    Ok(routes)
}

#[pyfunction]
#[pyo3(signature = (model, held=None))]
pub(in crate::worker) fn media_components<'py>(
    model: &Bound<'py, PyAny>,
    held: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(model.py());
    for (kind, name) in routes(model, &held_names(held)?)? {
        result.set_item(
            crate::convert::call_kind_to_py(model.py(), CallKind::Media(kind))?,
            name,
        )?;
    }
    Ok(result)
}

pub(in crate::worker) fn python_kinds(
    py: Python<'_>,
    kinds: impl IntoIterator<Item = CallKind>,
) -> PyResult<Bound<'_, PyFrozenSet>> {
    PyFrozenSet::new(
        py,
        kinds
            .into_iter()
            .map(|kind| crate::convert::call_kind_to_py(py, kind))
            .collect::<PyResult<Vec<_>>>()?,
    )
}

/// Python declaration inspection uses the same host operation table as routing.
#[pyfunction]
fn host_components(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let result = PyDict::new(py);
    for (name, kinds) in HOST_COMPONENTS {
        result.set_item(
            name,
            python_kinds(py, kinds.iter().copied().map(CallKind::Media))?,
        )?;
    }
    Ok(result)
}

pub(in crate::worker) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(call_kinds, module)?)?;
    module.add_function(wrap_pyfunction!(host_components, module)?)?;
    module.add_function(wrap_pyfunction!(is_host_component, module)?)?;
    module.add_function(wrap_pyfunction!(holds_host_components, module)?)?;
    module.add_function(wrap_pyfunction!(supported_calls, module)?)?;
    module.add_function(wrap_pyfunction!(media_components, module)?)?;
    Ok(())
}
