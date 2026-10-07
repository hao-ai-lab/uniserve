//! Capture segments around numerical submissions CUDA cannot record in a graph.

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::worker::execution_context::ExecutionContext;

use super::super::execution::close_all;
use super::super::host::with_context;

#[pyclass]
pub(super) struct Capture {
    stream: Py<PyAny>,
    computation: Py<PyAny>,
    pool: Option<Py<PyAny>>,
    graphs: Vec<Py<PyAny>>,
    steps: Vec<Py<PyAny>>,
    active: Option<Py<PyAny>>,
}

impl Capture {
    pub(super) fn new(stream: Py<PyAny>, computation: Py<PyAny>, pool: Option<Py<PyAny>>) -> Self {
        Self {
            stream,
            computation,
            pool,
            graphs: Vec::new(),
            steps: Vec::new(),
            active: None,
        }
    }

    fn begin(&mut self, py: Python<'_>) -> PyResult<()> {
        let cuda = py.import("torch.cuda")?;
        let options = PyDict::new(py);
        options.set_item("keep_graph", true)?;
        let graph = cuda.getattr("CUDAGraph")?.call((), Some(&options))?;
        self.graphs.push(graph.clone().unbind());
        with_context(&cuda.call_method1("stream", (&self.stream,))?, || {
            let options = PyDict::new(py);
            options.set_item("pool", &self.pool)?;
            graph.call_method("capture_begin", (), Some(&options))?;
            self.active = Some(graph.clone().unbind());
            self.computation
                .call_method1(py, "wait_stream", (&self.stream,))?;
            Ok(())
        })
    }

    fn end(&mut self, py: Python<'_>) -> PyResult<()> {
        let Some(graph) = self.active.take() else {
            return Ok(());
        };
        let result = with_context(
            &py.import("torch.cuda")?
                .call_method1("stream", (&self.stream,))?,
            || {
                self.stream
                    .call_method1(py, "wait_stream", (&self.computation,))?;
                graph.call_method0(py, "capture_end")?;
                Ok(())
            },
        );
        result?;
        if self.pool.is_none() {
            self.pool = Some(graph.call_method0(py, "pool")?);
        }
        self.steps.push(graph.getattr(py, "replay")?);
        Ok(())
    }

    pub(super) fn replay(&self, py: Python<'_>) -> PyResult<()> {
        for step in &self.steps {
            step.call0(py)?;
        }
        Ok(())
    }

    pub(super) fn reset(&mut self, py: Python<'_>) -> PyResult<()> {
        close_all(
            py,
            self.graphs
                .iter()
                .map(|graph| graph.call_method0(py, "reset").map(drop)),
        )?;
        self.active = None;
        self.steps.clear();
        self.graphs.clear();
        Ok(())
    }
}

#[pymethods]
impl Capture {
    /// Copy submissions between segments use external event dependencies.
    /// Their order and their shared allocator pool are fixed by capture.
    fn submit(&mut self, py: Python<'_>, call: Py<PyAny>) -> PyResult<()> {
        self.end(py)?;
        self.steps.push(call);
        self.begin(py)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.stream)?;
        visit.call(&self.computation)?;
        visit.call(&self.pool)?;
        visit.call(&self.active)?;
        for graph in &self.graphs {
            visit.call(graph)?;
        }
        for step in &self.steps {
            visit.call(step)?;
        }
        Ok(())
    }
}

pub(super) fn run(
    sequence: &Bound<'_, Capture>,
    context: &Bound<'_, ExecutionContext>,
    call: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let py = context.py();
    let invoke = || {
        sequence.borrow_mut().begin(py)?;
        let computation = sequence.borrow().computation.clone_ref(py);
        let result = with_context(
            &py.import("torch.cuda")?
                .call_method1("stream", (&computation,))?,
            || ExecutionContext::with_active(context, || call.call0().map(Bound::unbind)),
        );
        let ended = sequence.borrow_mut().end(py);
        ended?;
        result
    };
    let weights = context.getattr("weights")?;
    let output = if weights.is_none() {
        invoke()?
    } else {
        with_context(
            &weights.call_method1("capture", (sequence.getattr("submit")?,))?,
            invoke,
        )?
    };
    for graph in &sequence.borrow().graphs {
        graph.call_method0(py, "instantiate")?;
    }
    Ok(output)
}
