//! Numerical runner bindings and batched dispatch.

mod execute;
mod forward;

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use uniserve_worker::ModelRunners as NativeModelRunners;
use uniserve_worker_ipc::CallKind;

use super::error::native_error;
use super::model_results::ExecutionOutput;

#[pyclass(module = "uniserve_worker._uniserve_ipc")]
#[derive(Default)]
pub(crate) struct ModelRunners {
    inner: NativeModelRunners<Py<PyAny>>,
}

#[pymethods]
impl ModelRunners {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (owner, tasks, *, cache, tables, states))]
    fn forward(
        &self,
        py: Python<'_>,
        owner: Py<PyAny>,
        tasks: Py<PyTuple>,
        cache: Py<PyAny>,
        tables: Py<PyAny>,
        states: Py<PyAny>,
    ) -> PyResult<forward::ModelBatches> {
        forward::prepare(py, &self.inner, owner, tasks, cache, tables, states)
    }

    /// Execute prepared numerical rows with native stream and result ordering.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (owner, runner, rows, *, calls, cache, tables, states))]
    fn run_batch<'py>(
        &self,
        py: Python<'py>,
        owner: &Bound<'py, PyAny>,
        runner: &Bound<'py, PyAny>,
        rows: &Bound<'py, PyTuple>,
        calls: &Bound<'py, PyTuple>,
        cache: &Bound<'py, PyAny>,
        tables: &Bound<'py, PyAny>,
        states: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, ExecutionOutput>> {
        execute::run_batch(py, owner, runner, rows, calls, cache, tables, states, false)
    }

    fn clear(&mut self) {
        self.inner.clear();
    }

    fn bind(
        &mut self,
        py: Python<'_>,
        component: &str,
        kinds: &Bound<'_, PyAny>,
        runner: Py<PyAny>,
    ) -> PyResult<()> {
        let kinds = kinds
            .try_iter()?
            .map(|kind| pythonize::depythonize(&kind?).map_err(Into::into))
            .collect::<PyResult<Vec<CallKind>>>()?;
        self.inner
            .bind(component, &kinds, runner)
            .map_err(|error| native_error(py, error))
    }

    fn get(
        &self,
        py: Python<'_>,
        component: &str,
        kind: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Py<PyAny>>> {
        Ok(self
            .inner
            .get(component, pythonize::depythonize(kind)?)
            .map(|runner| runner.clone_ref(py)))
    }

    fn first(&self, py: Python<'_>, kind: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
        Ok(self
            .inner
            .first(pythonize::depythonize(kind)?)
            .map(|runner| runner.clone_ref(py)))
    }
}
