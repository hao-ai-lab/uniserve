//! Component placement uses the engine's native parallel configuration.

use std::collections::BTreeMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use uniserve_core::{
    ComponentConfig as NativeComponent, ComponentDistribution, ParallelConfig as NativeParallel,
    SequenceParallel,
};

use crate::convert::mapping_from_py;

#[pyclass(frozen, eq, module = "uniserve_worker._uniserve_ipc")]
#[derive(PartialEq)]
pub(crate) struct SequenceConfig {
    inner: SequenceParallel,
}

#[pymethods]
impl SequenceConfig {
    #[new]
    #[pyo3(signature = (kind="local", degrees=None))]
    fn new(kind: &str, degrees: Option<Vec<usize>>) -> PyResult<Self> {
        let degrees = degrees.unwrap_or_default();
        let inner = match (kind, degrees.as_slice()) {
            ("local", []) => SequenceParallel::Local,
            ("ulysses", &[ulysses_degree]) => SequenceParallel::Ulysses { ulysses_degree },
            ("allgather", &[allgather_degree]) => SequenceParallel::Allgather { allgather_degree },
            ("hybrid", &[ulysses_degree, allgather_degree]) => SequenceParallel::Hybrid {
                ulysses_degree,
                allgather_degree,
            },
            _ => {
                return Err(PyValueError::new_err(
                    "sequence strategy requires its declared degrees",
                ));
            }
        };
        inner.size().map_err(error)?;
        Ok(Self { inner })
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            SequenceParallel::Local => "local",
            SequenceParallel::Ulysses { .. } => "ulysses",
            SequenceParallel::Allgather { .. } => "allgather",
            SequenceParallel::Hybrid { .. } => "hybrid",
        }
    }

    #[getter]
    fn degrees<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let values = match self.inner {
            SequenceParallel::Local => vec![],
            SequenceParallel::Ulysses { ulysses_degree } => vec![ulysses_degree],
            SequenceParallel::Allgather { allgather_degree } => vec![allgather_degree],
            SequenceParallel::Hybrid {
                ulysses_degree,
                allgather_degree,
            } => vec![ulysses_degree, allgather_degree],
        };
        PyTuple::new(py, values)
    }

    #[getter]
    fn size(&self) -> PyResult<usize> {
        self.inner.size().map_err(error)
    }

    #[getter]
    fn dimensions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.dimensions())
    }

    #[staticmethod]
    fn from_dict(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner: SequenceParallel = mapping_from_py(value)?;
        inner.size().map_err(error)?;
        Ok(Self { inner })
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(pythonize::pythonize(py, &self.inner)?)
    }
}

#[pyclass(frozen, eq, module = "uniserve_worker._uniserve_ipc")]
#[derive(PartialEq)]
pub(crate) struct ParallelConfig {
    inner: NativeParallel,
}

#[pymethods]
impl ParallelConfig {
    #[new]
    #[pyo3(signature = (tensor_parallel_size=1, pipeline_parallel_size=1, sequence_parallel=None))]
    fn new(
        tensor_parallel_size: usize,
        pipeline_parallel_size: usize,
        sequence_parallel: Option<PyRef<'_, SequenceConfig>>,
    ) -> PyResult<Self> {
        let inner = NativeParallel {
            tensor_parallel_size,
            pipeline_parallel_size,
            sequence_parallel: sequence_parallel
                .map(|value| value.inner.clone())
                .unwrap_or_default(),
        };
        inner.world_size().map_err(error)?;
        Ok(Self { inner })
    }

    #[getter]
    fn tensor_parallel_size(&self) -> usize {
        self.inner.tensor_parallel_size
    }

    #[getter]
    fn pipeline_parallel_size(&self) -> usize {
        self.inner.pipeline_parallel_size
    }

    #[getter]
    fn sequence_parallel(&self) -> SequenceConfig {
        SequenceConfig {
            inner: self.inner.sequence_parallel.clone(),
        }
    }

    #[getter]
    fn sequence_parallel_size(&self) -> PyResult<usize> {
        self.inner.sequence_parallel.size().map_err(error)
    }

    #[getter]
    fn world_size(&self) -> PyResult<usize> {
        self.inner.world_size().map_err(error)
    }

    #[getter]
    fn dimensions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.dimensions())
    }

    #[staticmethod]
    fn from_dict(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner: NativeParallel = mapping_from_py(value)?;
        inner.world_size().map_err(error)?;
        Ok(Self { inner })
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(pythonize::pythonize(py, &self.inner)?)
    }
}

#[pyclass(frozen, eq, module = "uniserve_worker._uniserve_ipc")]
#[derive(PartialEq)]
pub(crate) struct ComponentConfig {
    pub(super) inner: NativeComponent,
}

#[pymethods]
impl ComponentConfig {
    #[new]
    #[pyo3(signature = (ranks, parallel_config=None, distribution=None, units_per_rank=1))]
    pub(super) fn new(
        ranks: Vec<usize>,
        parallel_config: Option<PyRef<'_, ParallelConfig>>,
        distribution: Option<&str>,
        units_per_rank: usize,
    ) -> PyResult<Self> {
        let distribution = match distribution {
            None => None,
            Some("temporal_units") => Some(ComponentDistribution::TemporalUnits),
            Some(_) => return Err(PyValueError::new_err("unsupported component distribution")),
        };
        let inner = NativeComponent {
            ranks,
            parallel_config: parallel_config
                .map(|value| value.inner.clone())
                .unwrap_or_default(),
            distribution,
            units_per_rank,
        };
        inner.validate().map_err(error)?;
        Ok(Self { inner })
    }

    #[getter]
    fn ranks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.ranks)
    }

    #[getter]
    fn parallel_config(&self) -> ParallelConfig {
        ParallelConfig {
            inner: self.inner.parallel_config.clone(),
        }
    }

    #[getter]
    fn distribution(&self) -> Option<&'static str> {
        self.inner.distribution.map(|_| "temporal_units")
    }

    #[getter]
    fn units_per_rank(&self) -> usize {
        self.inner.units_per_rank
    }

    #[staticmethod]
    fn from_dict(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner: NativeComponent = mapping_from_py(value)?;
        inner.validate().map_err(error)?;
        Ok(Self { inner })
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(pythonize::pythonize(py, &self.inner)?)
    }
}

/// Resolve rank placement once at launch, before collective creation begins.
#[pyfunction]
#[pyo3(signature = (value, world_size, *, role="model"))]
pub(super) fn parse_components<'py>(
    value: &Bound<'py, PyDict>,
    world_size: usize,
    role: &str,
) -> PyResult<Bound<'py, PyTuple>> {
    let py = value.py();
    let components: BTreeMap<String, NativeComponent> = mapping_from_py(value)?;
    if role == "experts" {
        if !components.is_empty() {
            return Err(PyValueError::new_err(
                "expert workers have no request components",
            ));
        }
        return Ok(PyTuple::empty(py));
    }
    if components.is_empty() {
        return Err(PyValueError::new_err(
            "component configuration must not be empty",
        ));
    }

    let mut result = Vec::with_capacity(components.len());
    for (name, inner) in components {
        inner.validate().map_err(error)?;
        if name.is_empty() || inner.ranks.iter().any(|&rank| rank >= world_size) {
            return Err(PyValueError::new_err(
                "component contains ranks outside the process world or an empty name",
            ));
        }
        result.push((name, ComponentConfig { inner }));
    }
    PyTuple::new(py, result)
}

fn error(error: uniserve_core::ParallelConfigError) -> PyErr {
    PyValueError::new_err(error.to_string())
}
