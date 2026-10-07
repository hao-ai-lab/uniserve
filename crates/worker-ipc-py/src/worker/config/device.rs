//! Device and tower placement normalization at the PyTorch loading interface.

use std::collections::BTreeMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(super) fn normalize(py: Python<'_>, value: &str, option: &str) -> PyResult<String> {
    let device = py
        .import("torch")?
        .call_method1("device", (value,))
        .map_err(|error| {
            PyValueError::new_err(format!(
                "{option} must name a torch device, got {value:?}: {error}"
            ))
        })?;
    // A rank scoped by CUDA_VISIBLE_DEVICES owns the visible device zero.
    if device.getattr("type")?.extract::<String>()? == "cuda" && device.getattr("index")?.is_none()
    {
        Ok("cuda:0".into())
    } else {
        device.str()?.extract()
    }
}

pub(super) fn generation(py: Python<'_>, mesh: &str, device: &str) -> PyResult<Option<String>> {
    if mesh.trim().is_empty() {
        return Ok(None);
    }

    let mut generation = None;
    for entry in mesh.split(',') {
        let (key, value) = entry.split_once('=').ok_or_else(|| {
            PyValueError::new_err(format!(
                "invalid --mesh entry {entry:?}; expected key=value"
            ))
        })?;
        let key = key.trim().to_lowercase().replace('_', "-");
        if key != "tower" {
            return Err(PyValueError::new_err(format!("unknown --mesh key {key:?}")));
        }
        if generation.is_some() {
            return Err(PyValueError::new_err("duplicate --mesh key tower"));
        }

        let mut parameters = BTreeMap::new();
        for part in value.split(';') {
            let (name, target) = part.split_once(':').ok_or_else(|| {
                PyValueError::new_err("tower params must use text:<device>;gen:<device>")
            })?;
            let name = name.trim().to_lowercase();
            if !matches!(name.as_str(), "text" | "gen") || target.trim().is_empty() {
                return Err(PyValueError::new_err(
                    "tower params must use text:<device>;gen:<device>",
                ));
            }
            if parameters.insert(name.clone(), target.trim()).is_some() {
                return Err(PyValueError::new_err(format!(
                    "duplicate tower params {name:?}"
                )));
            }
        }
        let target = parameters
            .get("gen")
            .ok_or_else(|| PyValueError::new_err("tower params requires gen:<device>"))?;
        let text = normalize(
            py,
            parameters.get("text").copied().unwrap_or(device),
            "--mesh tower text",
        )?;
        if text != device {
            return Err(PyValueError::new_err(
                "text expert device must match the Worker rank device",
            ));
        }
        let target = normalize(py, target, "--mesh tower gen")?;
        if target == text {
            return Err(PyValueError::new_err(
                "tower text and gen devices must be different",
            ));
        }
        generation = Some(target);
    }
    Ok(generation)
}
