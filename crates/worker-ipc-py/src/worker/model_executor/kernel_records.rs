//! Native accumulation and rendering of the kernels selected by numerical layers.

use std::collections::{BTreeMap, BTreeSet};

use indexmap::IndexMap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use serde_json::{Map, Value, json};

use crate::worker::graph_storage::GraphStorage;

const TAG: &str = "uniserve-kernel-table";

type LayerRecords = IndexMap<String, Vec<Map<String, Value>>>;

/// A retired runner keeps its last reported selections. Equal records share a
/// call site in the log without turning serialized JSON into internal keys.
#[derive(Default)]
pub(super) struct KernelRecords {
    runners: IndexMap<String, (String, LayerRecords)>,
}

impl KernelRecords {
    pub(super) fn gather(runners: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let mut current = Self::default();
        for runner in runners {
            let mut label: String = runner.getattr("name")?.extract()?;
            if let Some(entry) = runner
                .getattr("call")?
                .getattr_opt("entry_point")?
                .filter(|entry| !entry.is_none())
            {
                label.push('.');
                label.push_str(&entry.getattr("method")?.extract::<String>()?);
            }
            let mut kinds = Vec::new();
            if let Some(values) = runner.getattr_opt("call_kinds")? {
                for kind in values.try_iter()? {
                    kinds.push(kind?.str()?.extract::<String>()?);
                }
            }
            kinds.sort();
            if !kinds.is_empty() {
                label.push_str(&format!("[{}]", kinds.join(",")));
            }
            let device: String = runner.getattr("device")?.str()?.extract()?;
            let (_, paths) = current
                .runners
                .entry(label)
                .or_insert_with(|| (device, IndexMap::new()));
            for record in runner.call_method0("kernels")?.try_iter()? {
                let mut record: Map<String, Value> = pythonize::depythonize(&record?)?;
                let path = record
                    .remove("path")
                    .and_then(|path| path.as_str().map(str::to_owned))
                    .ok_or_else(|| PyValueError::new_err("kernel records require a layer path"))?;
                let values = paths.entry(path).or_default();
                if !values.contains(&record) {
                    values.push(record);
                }
            }
        }
        Ok(current)
    }

    pub(super) fn merge(&mut self, current: Self) -> bool {
        let mut changed = false;
        for (label, (device, paths)) in current.runners {
            let (_, stored) = self
                .runners
                .entry(label)
                .or_insert_with(|| (device, IndexMap::new()));
            for (path, records) in paths {
                if stored.get(&path) != Some(&records) {
                    stored.insert(path, records);
                    changed = true;
                }
            }
        }
        changed
    }

    pub(super) fn log(
        &self,
        py: Python<'_>,
        stage: &str,
        rank: usize,
        device: &str,
    ) -> PyResult<()> {
        let mut runners = Vec::new();
        let mut portable = Vec::new();
        for (label, (device, paths)) in &self.runners {
            let mut sites: Vec<(&Map<String, Value>, Vec<&str>)> = Vec::new();
            for (path, records) in paths {
                let prepared = records
                    .iter()
                    .any(|record| record["op"] == "attention" && !record["provider"].is_null());
                for record in records {
                    if prepared && record["op"] == "attention" && record["provider"].is_null() {
                        continue;
                    }
                    if let Some((_, layers)) = sites.iter_mut().find(|(value, _)| *value == record)
                    {
                        layers.push(path);
                    } else {
                        sites.push((record, vec![path]));
                    }
                }
            }
            let mut calls = Vec::new();
            for (record, paths) in sites {
                let layers = layer_paths(paths);
                let mut call = record.clone();
                call.insert("layers".to_owned(), json!(layers));
                calls.push(call);
                let served: Vec<&str> = if record["provider"] == "torch" {
                    vec!["every input"]
                } else if record["provider"].is_null() {
                    Vec::new()
                } else {
                    record
                        .get("inputs")
                        .and_then(Value::as_object)
                        .into_iter()
                        .flat_map(|inputs| {
                            inputs
                                .iter()
                                .filter(|(_, provider)| **provider == "torch")
                                .map(|(name, _)| name.as_str())
                        })
                        .collect()
                };
                if !served.is_empty() && (device == "cuda" || device.starts_with("cuda:")) {
                    portable.push(json!({ "runner": label, "op": record["op"], "layers": layers, "inputs": served }));
                }
            }
            runners.push(json!({ "runner": label, "device": device, "call_sites": calls }));
        }
        let table = json!({ "stage": stage, "rank": rank, "device": device, "runners": runners, "portable": portable });
        let text = serde_json::to_string(&table)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        logger(py)?.call_method1("info", ("%s %s", TAG, text))?;
        Ok(())
    }
}

fn layer_paths(paths: Vec<&str>) -> Vec<String> {
    let mut templates: IndexMap<Vec<&str>, Vec<Vec<&str>>> = IndexMap::new();
    for path in paths {
        let parts: Vec<_> = path.split('.').collect();
        let key = parts
            .iter()
            .map(|&part| {
                if part.parse::<u64>().is_ok() {
                    "*"
                } else {
                    part
                }
            })
            .collect();
        templates.entry(key).or_default().push(parts);
    }
    let mut result = Vec::new();
    for (key, members) in templates {
        let varying: Vec<_> = key
            .iter()
            .enumerate()
            .filter(|&(position, &part)| {
                part == "*"
                    && members
                        .iter()
                        .any(|member| member[position] != members[0][position])
            })
            .map(|(position, _)| position)
            .collect();
        if varying.len() > 1 {
            result.extend(members.iter().map(|parts| parts.join(".")));
            continue;
        }
        let mut parts: Vec<_> = members[0].iter().map(|&part| part.to_owned()).collect();
        if let Some(&position) = varying.first() {
            let numbers: BTreeSet<_> = members
                .iter()
                .filter_map(|member| member[position].parse::<u64>().ok())
                .collect();
            let mut runs: Vec<(u64, u64)> = Vec::new();
            for number in numbers {
                if let Some((_, stop)) = runs.last_mut()
                    && stop.checked_add(1) == Some(number)
                {
                    *stop = number;
                } else {
                    runs.push((number, number));
                }
            }
            let ranges: Vec<_> = runs
                .into_iter()
                .map(|(start, stop)| {
                    if start == stop {
                        start.to_string()
                    } else {
                        format!("{start}-{stop}")
                    }
                })
                .collect();
            parts[position] = format!("{{{}}}", ranges.join(","));
        }
        result.push(parts.join("."));
    }
    result
}

pub(super) fn log_memory(storage: &Bound<'_, GraphStorage>) -> PyResult<()> {
    let py = storage.py();
    let logger = logger(py)?;
    let device_backend = py.import("uniserve.runtime.device")?;
    let cuda = py.import("torch.cuda")?;
    let pools = storage.borrow().pool_bytes(py)?;
    let mut devices = pools
        .iter()
        .map(|(device, bytes)| {
            Ok((
                device.str()?.extract::<String>()?,
                device,
                bytes.extract::<f64>()?,
            ))
        })
        .collect::<PyResult<Vec<_>>>()?;
    devices.sort_by(|left, right| left.0.cmp(&right.0));
    for (_, device, pooled) in devices {
        let process: f64 = device_backend
            .call_method1("process_device_bytes", (&device,))?
            .extract()?;
        let reserved: f64 = cuda
            .call_method1("memory_reserved", (&device,))?
            .extract()?;
        logger.call_method1("info", ("device storage on %s: %.2f GiB held by this process at startup, %.2f GiB of it reserved by the caching allocator and %.2f GiB of that in graph pools", device, process / 2_f64.powi(30), reserved / 2_f64.powi(30), pooled / 2_f64.powi(30)))?;
    }
    let mut totals = BTreeMap::<(String, String), u64>::new();
    for (key, bytes) in storage.borrow().owner_bytes(py)?.iter() {
        let device = key.get_item(1)?.str()?.extract()?;
        let label = key.get_item(0)?.getattr("name")?.extract()?;
        *totals.entry((device, label)).or_default() += bytes.extract::<u64>()?;
    }
    for ((device, label), used) in totals {
        logger.call_method1(
            "info",
            (
                "graph storage on %s: %s holds %.2f GiB at startup",
                device,
                label,
                used as f64 / 2_f64.powi(30),
            ),
        )?;
    }
    Ok(())
}

fn logger(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    py.import("logging")?
        .call_method1("getLogger", ("uniserve_worker.execution.model_executor",))
}
