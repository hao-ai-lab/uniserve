//! Bind component communication in collective order across the process world.

use std::collections::{BTreeMap, BTreeSet};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use super::ProcessGroups;
use crate::worker::component_binding::ComponentBinding;
use crate::worker::placement::ComponentConfig;

#[pyfunction]
#[pyo3(signature = (groups, components, *, declarations=None))]
pub(in crate::worker) fn initialize_components<'py>(
    groups: &Bound<'py, ProcessGroups>,
    components: &Bound<'py, PyAny>,
    declarations: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let py = groups.py();
    let mut owner = groups.borrow_mut();
    let device = owner.device.clone_ref(py);
    let mut placements = BTreeMap::<String, Py<ComponentConfig>>::new();
    for item in components.call_method0("items")?.try_iter()? {
        let item = item?;
        placements.insert(item.get_item(0)?.extract()?, item.get_item(1)?.extract()?);
    }
    let owns = placements
        .values()
        .any(|config| config.get().inner.ranks.contains(&owner.rank));
    let result = PyDict::new(py);
    let distributed = py.import("uniserve.distributed")?;
    let causal = py.import("uniserve.model")?.getattr("CausalLM")?;
    for (name, config) in placements {
        let placement = &config.get().inner;
        let member = placement.ranks.contains(&owner.rank);
        let temporal = placement.distribution.is_some();
        let host = crate::worker::component_binding::calls::is_host_component(&name);
        let mut units = None;
        if temporal && !host {
            // Every process creates the overlap ring, including non-members.
            // Each member then uses a rank-local mesh for numerical execution.
            let topology = mesh(
                py,
                &placement.ranks,
                &[("units", placement.ranks.len())],
                owner.rank,
            )?;
            let ring = owner.bind(py, &topology, device.bind(py), None)?;
            if member {
                units = Some(
                    ring.bind(py)
                        .call_method1("get_group", ("units",))?
                        .unbind(),
                );
            }
        }

        let mut bound = None;
        let mut participation = Vec::new();
        if !temporal || member {
            let local = [owner.rank];
            let ranks = if temporal {
                &local[..]
            } else {
                &placement.ranks
            };
            let dimensions = placement.parallel_config.dimensions();
            let topology = mesh(py, ranks, &dimensions, owner.rank)?;
            let mut axes = BTreeSet::<Vec<String>>::new();
            match declarations {
                None => axes.extend(dimensions.iter().map(|(axis, _)| vec![(*axis).to_owned()])),
                Some(declarations) => {
                    let calls = declarations.call_method1("get", (&name, PyTuple::empty(py)))?;
                    let attention = py
                        .import("uniserve_worker.bootstrap.model_loader")?
                        .call_method1("attention_parallel", (&config,))?;
                    let options = PyDict::new(py);
                    options.set_item("attention", attention)?;
                    for call in calls.try_iter()? {
                        let call = call?;
                        let module = call.getattr("module")?;
                        let numerical = distributed.call_method(
                            "communication_axes",
                            (&module, &topology),
                            Some(&options),
                        )?;
                        let entry = call
                            .getattr("entry_point")?
                            .call_method1("communication_axes", (&topology,))?;
                        for selection in [&numerical, &entry] {
                            for axis in selection.try_iter()? {
                                axes.insert(axis?.extract()?);
                            }
                        }
                        // Sampling needs TP even for modules with replicated logits.
                        if module.is_instance(&causal)? {
                            axes.insert(vec!["tp".to_owned()]);
                        }
                    }
                }
            }
            let selected = PyTuple::new(
                py,
                axes.iter()
                    .map(|axis| PyTuple::new(py, axis))
                    .collect::<PyResult<Vec<_>>>()?,
            )?;
            let topology = owner.bind(py, &topology, device.bind(py), Some(selected.as_any()))?;
            if member {
                for axes in selected.iter() {
                    participation.push(topology.bind(py).call_method1("get_group", (axes,))?);
                }
                bound = Some(topology);
            }
        }

        let binding = ComponentBinding::new(
            py,
            name.clone(),
            config,
            owner.instance.clone_ref(py),
            bound,
            device.clone_ref(py),
            units,
            Some(PyTuple::new(py, participation)?.unbind()),
            None,
            None,
        )?;
        result.set_item(name, binding)?;
    }
    // Check after collective creation, so an unplaced rank cannot leave peers
    // blocked in a group that still needs its participation.
    if !owns {
        return Err(PyValueError::new_err("rank has no configured component"));
    }
    Ok(result)
}

fn mesh<'py>(
    py: Python<'py>,
    ranks: &[usize],
    dimensions: &[(&str, usize)],
    rank: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let options = PyDict::new(py);
    options.set_item("ranks", PyTuple::new(py, ranks)?)?;
    options.set_item(
        "shape",
        PyTuple::new(py, dimensions.iter().map(|(_, size)| size))?,
    )?;
    options.set_item(
        "axes",
        PyTuple::new(py, dimensions.iter().map(|(axis, _)| axis))?,
    )?;
    options.set_item("rank", rank)?;
    py.import("uniserve.distributed")?
        .getattr("DeviceMesh")?
        .call((), Some(&options))
}
