//! Captured expert participation for ranks and microbatches without tokens.

use std::collections::BTreeSet;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyFrozenSet;

use crate::worker::cuda_graph::CUDAGraph;

use super::{GraphBucket, close_all};
use crate::worker::host::with_context;

/// All ranks warm and capture the same capacities, largest first. Graphs
/// borrow the supplied numerical context and retire before that context.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct JoinGraphs {
    graphs: Py<GraphBucket>,
}

#[pymethods]
impl JoinGraphs {
    #[new]
    #[pyo3(signature = (context, exchange, capacities, *, pools=None, step=None, warm=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        context: &Bound<'_, PyAny>,
        exchange: &Bound<'_, PyAny>,
        capacities: &Bound<'_, PyAny>,
        pools: Option<&Bound<'_, PyAny>>,
        step: Option<&Bound<'_, PyAny>>,
        warm: bool,
    ) -> PyResult<Self> {
        let capacities = capacities
            .try_iter()?
            .map(|capacity| capacity?.extract())
            .collect::<PyResult<BTreeSet<usize>>>()?;
        let graphs = Py::new(py, GraphBucket::new(None))?;
        let result = (|| {
            let partial = py.import("functools")?.getattr("partial")?;
            let join = py.get_type::<Self>().getattr("_join")?;

            for capacity in capacities.into_iter().rev() {
                let call = partial.call1((&join, context, exchange, capacity, step))?;
                if warm {
                    with_context(&context.call_method0("activate")?, || call.call0())?;
                }
                let graph = Py::new(py, CUDAGraph::new(py, context.clone().unbind(), pools)?)?;
                graphs
                    .borrow_mut(py)
                    .graphs
                    .insert(Some(capacity), graph.clone_ref(py).into_any());
                graph.borrow_mut(py).capture(py, call.unbind(), None)?;
            }
            Ok(())
        })();
        if let Err(error) = result {
            close_all(py, [Err(error), GraphBucket::close(graphs.bind(py))])?;
        }
        Ok(Self { graphs })
    }

    /// A capture callback borrows its arguments only for the numerical call.
    #[staticmethod]
    fn _join(
        context: &Bound<'_, PyAny>,
        exchange: &Bound<'_, PyAny>,
        capacity: usize,
        step: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        if let Some(step) = step {
            return step.call1((capacity,)).map(drop);
        }
        let control = exchange
            .getattr("_control")?
            .cast_into::<crate::worker::expert_exchange::ExpertExchange>()?;
        control.borrow().begin(capacity)?;
        let joined = context.call_method0("join_expert_layers").map(drop);
        control.borrow().end();
        joined
    }

    #[getter]
    fn capacities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        let capacities: Vec<_> = self
            .graphs
            .borrow(py)
            .graphs
            .keys()
            .flatten()
            .copied()
            .collect();
        PyFrozenSet::new(py, capacities)
    }

    pub(in crate::worker) fn replay(&self, py: Python<'_>, capacity: usize) -> PyResult<()> {
        let graph = self
            .graphs
            .borrow(py)
            .get(py, Some(capacity))
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "no captured expert join serves step capacity {capacity}"
                ))
            })?;
        graph
            .bind(py)
            .cast::<CUDAGraph>()?
            .borrow()
            .replay(py)
            .map(drop)
    }

    pub(in crate::worker) fn close(&self, py: Python<'_>) -> PyResult<()> {
        GraphBucket::close(self.graphs.bind(py))
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.graphs)
    }
}
