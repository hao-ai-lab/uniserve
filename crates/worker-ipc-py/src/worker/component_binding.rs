//! Borrowed numerical modules and native component placement.

pub(super) mod calls;
mod model;

pub(super) use model::{bind_components, validate_components};

use std::ops::Range;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use super::placement::ComponentConfig;

/// Placement remains available on non-members for routing and output sizing.
/// Members borrow their mesh and numerical calls; process groups own handles.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct ComponentBinding {
    #[pyo3(get)]
    pub(super) name: String,
    #[pyo3(get)]
    pub(super) config: Py<ComponentConfig>,
    #[pyo3(get)]
    pub(super) process_group: Py<PyAny>,
    #[pyo3(get)]
    pub(super) mesh: Option<Py<PyAny>>,
    #[pyo3(get)]
    pub(super) device: Py<PyAny>,
    #[pyo3(get)]
    pub(super) units: Option<Py<PyAny>>,
    #[pyo3(get)]
    pub(super) groups: Py<PyTuple>,
    #[pyo3(get, set)]
    pub(super) call_kinds: Py<PyTuple>,
    #[pyo3(get, set)]
    pub(super) calls: Py<PyTuple>,
    rank: usize,
}

#[pymethods]
impl ComponentBinding {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (name, config, process_group, mesh, device, units=None, groups=None, call_kinds=None, calls=None))]
    pub(super) fn new(
        py: Python<'_>,
        name: String,
        config: Py<ComponentConfig>,
        process_group: Py<PyAny>,
        mesh: Option<Py<PyAny>>,
        device: Py<PyAny>,
        units: Option<Py<PyAny>>,
        groups: Option<Py<PyTuple>>,
        call_kinds: Option<Py<PyTuple>>,
        calls: Option<Py<PyTuple>>,
    ) -> PyResult<Self> {
        let placement = &config.get().inner;
        let size = process_group.bind(py).getattr("size")?.extract::<usize>()?;
        let rank = process_group.bind(py).getattr("rank")?.extract::<usize>()?;
        if placement.ranks.iter().any(|&rank| rank >= size) {
            return Err(PyValueError::new_err(format!(
                "component {name} members lie outside its Worker"
            )));
        }

        // Temporal units execute locally; only model-parallel components use
        // their full placement as the numerical mesh.
        if placement.distribution.is_none() {
            if mesh.is_some() != placement.ranks.contains(&rank) {
                return Err(PyValueError::new_err(format!(
                    "component {name} requires its local mesh"
                )));
            }
            if let Some(mesh) = &mesh {
                let mesh = mesh.bind(py);
                let ranks: Vec<usize> = mesh.getattr("ranks")?.extract()?;
                let axes: Vec<String> = mesh.getattr("axes")?.extract()?;
                let shape: Vec<usize> = mesh.getattr("shape")?.extract()?;
                let dimensions: Vec<_> = axes.iter().map(String::as_str).zip(shape).collect();
                if ranks != placement.ranks || dimensions != placement.parallel_config.dimensions()
                {
                    return Err(PyValueError::new_err(format!(
                        "component {name} mesh disagrees with configuration"
                    )));
                }
            }
        }

        Ok(Self {
            name,
            config,
            process_group,
            mesh,
            device,
            units,
            rank,
            groups: groups.unwrap_or_else(|| PyTuple::empty(py).unbind()),
            call_kinds: call_kinds.unwrap_or_else(|| PyTuple::empty(py).unbind()),
            calls: calls.unwrap_or_else(|| PyTuple::empty(py).unbind()),
        })
    }

    #[getter]
    pub(super) fn owns(&self) -> bool {
        self.config.get().inner.ranks.contains(&self.rank)
    }

    #[getter]
    fn communicators<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            self.groups
                .bind(py)
                .iter()
                .chain(self.units.iter().map(|value| value.bind(py).clone()))
                .collect::<Vec<_>>(),
        )
    }

    #[getter]
    fn input_ranks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.config.get().inner.input_ranks())
    }

    #[getter]
    fn output_ranks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.config.get().inner.output_ranks())
    }

    /// Assign contiguous temporal units using worker-local rank order.
    fn media_units<'py>(
        &self,
        py: Python<'py>,
        cursor: usize,
        count: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let range = self.unit_range(cursor, count)?;
        py.import("builtins")?
            .call_method1("range", (range.start, range.end))
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.config)?;
        visit.call(&self.process_group)?;
        visit.call(&self.mesh)?;
        visit.call(&self.device)?;
        visit.call(&self.units)?;
        visit.call(&self.groups)?;
        visit.call(&self.call_kinds)?;
        visit.call(&self.calls)
    }
}

impl ComponentBinding {
    pub(super) fn unit_range(&self, cursor: usize, count: usize) -> PyResult<Range<usize>> {
        let placement = &self.config.get().inner;
        if placement.distribution.is_none() {
            return Ok(cursor..cursor + count);
        }

        let index = placement
            .ranks
            .iter()
            .position(|&rank| rank == self.rank)
            .ok_or_else(|| {
                PyValueError::new_err("rank is not a member of the distributed component")
            })?;
        let offset = index * placement.units_per_rank;
        Ok(cursor + offset..cursor + (offset + placement.units_per_rank).min(count))
    }
}
