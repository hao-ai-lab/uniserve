//! Thread-local numerical bindings for nested, independent invocations.

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use super::ExecutionContext;
use super::transfers::DeviceTransfers;

#[pyclass]
pub(super) struct Activation {
    context: Py<ExecutionContext>,
    scope: Option<Py<PyAny>>,
}

impl Activation {
    pub(super) fn new(context: Py<ExecutionContext>) -> Self {
        Self {
            context,
            scope: None,
        }
    }

    pub(super) fn enter(&mut self, py: Python<'_>) -> PyResult<()> {
        let owner = self.context.borrow(py);
        let state = owner.open()?;
        let scope = py.import("contextlib")?.call_method0("ExitStack")?;
        scope.call_method0("__enter__")?;
        let entered: PyResult<()> = (|| {
            let stream = state.stream.bind(py);
            let collectives = if stream.is_none() {
                py.None().into_bound(py)
            } else {
                let cuda = py.import("torch.cuda")?;
                scope.call_method1(
                    "enter_context",
                    (cuda.call_method1("device", (stream.getattr("device")?,))?,),
                )?;
                scope.call_method1(
                    "enter_context",
                    (cuda.call_method1("stream", (stream.getattr("stream")?,))?,),
                )?;
                stream.getattr("communication")?.getattr("communicators")?
            };
            scope.call_method1(
                "enter_context",
                (py.import("uniserve.runtime.communication")?
                    .call_method1("stream_collective_scope", (collectives,))?,),
            )?;
            if let Some(transfers) = DeviceTransfers::scope(state.transfers.bind(py)) {
                scope.call_method1("enter_context", (Py::new(py, transfers)?,))?;
            }

            let parallel = py.import("uniserve.nn.attention._parallel")?;
            scope.call_method1(
                "enter_context",
                (parallel.call_method1("output_scope", (&state.vsa_output,))?,),
            )?;
            scope.call_method1(
                "enter_context",
                (parallel.call_method1("context_scope", (&state.vsa_context,))?,),
            )?;
            let variables = py.import("uniserve.nn._binding")?;
            for (name, values) in [
                ("matmul", &state.operators),
                ("merged_matmul", &state.merged),
                ("attention", &state.attention),
                ("vsa", &state.vsa),
                ("moe", &state.moe),
                ("attention_storage", &state.exchange),
                ("linear_chunks", &state.chunks),
            ] {
                let variable = variables.getattr(name)?;
                let token = variable.call_method1("set", (values,))?;
                scope.call_method1("callback", (variable.getattr("reset")?, token))?;
            }
            if !state.weights.is_none(py) {
                scope.call_method1(
                    "enter_context",
                    (state.weights.call_method0(py, "activate")?,),
                )?;
            }
            Ok(())
        })();
        if let Err(error) = entered {
            scope.call_method1(
                "__exit__",
                (error.get_type(py), error.value(py), error.traceback(py)),
            )?;
            return Err(error);
        }
        self.scope = Some(scope.unbind());
        Ok(())
    }

    pub(super) fn exit(
        &mut self,
        py: Python<'_>,
        kind: &Bound<'_, PyAny>,
        error: &Bound<'_, PyAny>,
        traceback: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let scope = self
            .scope
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("numerical activation was not entered"))?;
        scope.call_method1(py, "__exit__", (kind, error, traceback))?;
        Ok(())
    }
}

#[pymethods]
impl Activation {
    fn __enter__(&mut self, py: Python<'_>) -> PyResult<()> {
        self.enter(py)
    }

    fn __exit__(
        &mut self,
        py: Python<'_>,
        kind: &Bound<'_, PyAny>,
        error: &Bound<'_, PyAny>,
        traceback: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.exit(py, kind, error, traceback)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.context)?;
        visit.call(&self.scope)
    }
}
