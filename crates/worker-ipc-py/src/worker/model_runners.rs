//! Numerical runner bindings and batched dispatch.

mod dispatch;
mod execute;
mod forward;
mod graphs;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use uniserve_worker::ModelRunners as NativeModelRunners;
use uniserve_worker_ipc::CallKind;

use super::error::native_error;
use super::execution::Execution;
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

    /// Evaluate a prepared numerical batch in its context, including experts.
    /// The caller orders input and output accesses with the runner's stream.
    fn run_eager(
        &self,
        py: Python<'_>,
        runner: &Bound<'_, PyAny>,
        batch: &Bound<'_, PyAny>,
        forward: &Bound<'_, PyAny>,
    ) -> PyResult<Py<ExecutionOutput>> {
        super::host::with_context(&py.import("torch")?.call_method0("inference_mode")?, || {
            super::host::with_context(&context(runner)?.call_method0("activate")?, || {
                graphs::run_eager(runner, batch, forward)
            })
        })
    }

    /// Capture every expert-capacity variant of a selected startup bucket.
    /// Fails after startup is sealed; joins the caller's stream on every exit.
    fn capture(
        &self,
        py: Python<'_>,
        runner: &Bound<'_, PyAny>,
        batch: &Bound<'_, PyAny>,
        forward: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        super::host::with_context(&py.import("torch")?.call_method0("inference_mode")?, || {
            graphs::capture(runner, batch, forward)
        })
    }

    /// Evaluate a standalone numerical signature and return owned tensors.
    fn run_module(
        &self,
        py: Python<'_>,
        runner: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: &Bound<'_, PyDict>,
    ) -> PyResult<Py<ExecutionOutput>> {
        super::host::with_context(&py.import("torch")?.call_method0("inference_mode")?, || {
            graphs::run_module(runner, args, kwargs)
        })
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

    /// Bind whole-row numerical peers to their shared native execution group.
    fn bind_microbatches(&self, py: Python<'_>, peers: &Bound<'_, PyTuple>) -> PyResult<()> {
        let executions = peers
            .iter()
            .map(|peer| execution(&peer).map(Bound::unbind))
            .collect::<PyResult<Vec<_>>>()?;
        Execution::bind_microbatches(py, executions)?;
        for peer in peers {
            peer.setattr("peers", peers)?;
        }
        Ok(())
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

pub(super) fn execution<'py>(runner: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Execution>> {
    Ok(runner.getattr("execution")?.cast_into()?)
}

pub(super) fn context<'py>(runner: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    Ok(execution(runner)?
        .borrow()
        .context
        .bind(runner.py())
        .clone())
}
