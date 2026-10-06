//! Numerical runner bindings and batched dispatch.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyTuple, PyType};
use uniserve_worker::ModelRunners as NativeModelRunners;
use uniserve_worker_ipc::CallKind;

use crate::calls::Call;

use super::error::native_error;

type InputRow<'py> = (
    PyRef<'py, Call>,
    Bound<'py, PyAny>,
    Bound<'py, PyType>,
    Vec<usize>,
    Option<bool>,
);
type ModelBatch<'py> = (Py<PyAny>, Bound<'py, PyTuple>, bool);

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

    /// Input type and dimensions are host metadata; grouping never reads a tensor.
    fn group<'py>(
        &self,
        py: Python<'py>,
        rows: &Bound<'py, PyAny>,
    ) -> PyResult<(Vec<usize>, Vec<ModelBatch<'py>>)> {
        let mut native = Vec::new();
        for row in rows.try_iter()? {
            let (call, kind, input_type, shape, causal): InputRow<'_> = row?.extract()?;
            native.push((
                Arc::clone(&call.inner),
                pythonize::depythonize::<CallKind>(&kind)?,
                (input_type, shape),
                causal,
            ));
        }

        let (missing, batches) = self.inner.group(native.iter().map(
            |(call, kind, (input_type, shape), causal)| {
                (
                    call.as_ref(),
                    *kind,
                    (input_type.as_ptr() as usize, shape),
                    *causal,
                )
            },
        ));
        let batches = batches
            .into_iter()
            .map(|batch| {
                Ok((
                    batch.runner.clone_ref(py),
                    PyTuple::new(py, batch.rows)?,
                    batch.preserve_output,
                ))
            })
            .collect::<PyResult<_>>()?;

        Ok((missing, batches))
    }
}
