//! Ordered numerical batches with outputs retained until their consumer runs.

use std::collections::VecDeque;
use std::sync::Arc;

use super::ModelExecutor;
use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use uniserve_worker::ModelRunners;
use uniserve_worker_ipc::CallKind;

use crate::calls::Call;
use crate::worker::error::invalid;
use crate::worker::model_results::ExecutionOutput;

use super::execute;

type Group = (Py<PyAny>, Vec<usize>, bool);

/// The iterator retains numerical resources and runs only when its consumer
/// requests the next result. A failed consumer therefore starts no later work.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(in crate::worker) struct ModelBatches {
    owner: Option<Py<ModelExecutor>>,
    tasks: Py<PyTuple>,
    cache: Py<PyAny>,
    tables: Py<PyAny>,
    states: Py<PyAny>,
    missing: VecDeque<usize>,
    groups: VecDeque<Group>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare(
    py: Python<'_>,
    runners: &ModelRunners<Py<PyAny>>,
    owner: Py<ModelExecutor>,
    tasks: Py<PyTuple>,
    cache: Py<PyAny>,
    tables: Py<PyAny>,
    states: Py<PyAny>,
) -> PyResult<ModelBatches> {
    let inputs = py.import("uniserve_worker.model_executor.input_batch")?;
    let images = py.import("uniserve_worker.model_executor.image_inputs")?;
    let diffusion = py.import("uniserve_worker.model_executor.diffusion_inputs")?;
    let token = inputs.getattr("TokenRow")?;
    let vision = images.getattr("VisionRow")?;
    let decode = images.getattr("DecodeRow")?;
    let flow = diffusion.getattr("DiffusionRow")?;

    let mut rows = Vec::with_capacity(tasks.bind(py).len());
    for task in tasks.bind(py) {
        let (row, call): (Bound<'_, PyAny>, PyRef<'_, Call>) = task.extract()?;
        let shape = if row.is_instance(&vision)? {
            row.getattr("encode_pixels")?
                .getattr("shape")?
                .extract::<Vec<usize>>()?
        } else if row.is_instance(&flow)? || row.is_instance(&decode)? {
            vec![
                row.getattr("image_height")?.extract()?,
                row.getattr("image_width")?.extract()?,
            ]
        } else {
            Vec::new()
        };
        let causal = row.is_instance(&token)? && row.getattr("causal")?.extract::<bool>()?;
        rows.push((
            Arc::clone(&call.inner),
            pythonize::depythonize::<CallKind>(&row.getattr("forward_mode")?)?,
            row.get_type(),
            shape,
            Some(causal),
        ));
    }
    let (missing, groups) = runners.group(rows.iter().map(|(call, kind, class, shape, causal)| {
        (
            call.as_ref(),
            *kind,
            (class.as_ptr() as usize, shape),
            *causal,
        )
    }));

    Ok(ModelBatches {
        owner: Some(owner),
        tasks,
        cache,
        tables,
        states,
        missing: missing.into(),
        groups: groups
            .into_iter()
            .map(|group| {
                (
                    group.runner.clone_ref(py),
                    group.rows,
                    group.preserve_output,
                )
            })
            .collect(),
    })
}

#[pymethods]
impl ModelBatches {
    fn __iter__(this: PyRef<'_, Self>) -> PyRef<'_, Self> {
        this
    }

    fn __next__<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Option<(Bound<'py, PyTuple>, Py<PyAny>)>> {
        self.next(py)?
            .map(|(indices, result)| Ok((PyTuple::new(py, indices)?, result)))
            .transpose()
    }

    fn __clear__(&mut self, py: Python<'_>) {
        self.owner = None;
        self.tasks = PyTuple::empty(py).unbind();
        self.cache = py.None();
        self.tables = py.None();
        self.states = py.None();
        self.missing.clear();
        self.groups.clear();
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.owner)?;
        visit.call(&self.tasks)?;
        visit.call(&self.cache)?;
        visit.call(&self.tables)?;
        visit.call(&self.states)?;
        for (runner, _, _) in &self.groups {
            visit.call(runner)?;
        }
        Ok(())
    }
}

impl ModelBatches {
    /// Advance one group; native consumers retain its host row indexes directly.
    pub(in crate::worker) fn next<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Option<(Vec<usize>, Py<PyAny>)>> {
        let Some(owner) = &self.owner else {
            return Ok(None);
        };
        if let Some(index) = self.missing.pop_front() {
            let task = self.tasks.bind(py).get_item(index)?;
            let call = task.get_item(1)?.extract::<PyRef<'_, Call>>()?;
            let error = invalid(
                py,
                format!(
                    "execution has no {:?} binding for {:?}",
                    call.inner.code.as_str(),
                    call.inner.component,
                ),
            );
            return Ok(Some((vec![index], error.into_value(py).into_any())));
        }
        let Some((runner, indices, preserve)) = self.groups.pop_front() else {
            self.__clear__(py);
            return Ok(None);
        };
        let runner = runner.bind(py);
        let tasks = self.tasks.bind(py);
        let mut rows = Vec::with_capacity(indices.len());
        let mut calls = Vec::with_capacity(indices.len());
        for &index in &indices {
            let task = tasks.get_item(index)?;
            rows.push(task.get_item(0)?);
            calls.push(task.get_item(1)?);
        }
        let rows = PyTuple::new(py, rows)?;
        let calls = PyTuple::new(py, calls)?;
        let result: PyResult<Py<ExecutionOutput>> = (|| {
            let result = execute::run_batch(
                py,
                owner.bind(py),
                runner,
                &rows,
                &calls,
                self.cache.bind(py),
                self.tables.bind(py),
                self.states.bind(py),
                true,
            )?;
            if preserve {
                // run_batch already joined the producer on this stream.
                // Copy the views without inserting a second event wait.
                return ExecutionOutput::copy(&result);
            }
            Ok(result.unbind())
        })();
        match result {
            Ok(result) => Ok(Some((indices, result.into_any()))),
            Err(error) => {
                let fatal: bool = py
                    .import("uniserve_worker.errors")?
                    .call_method1("classify", (error.value(py),))?
                    .getattr("fatal")?
                    .extract()?;
                if fatal {
                    self.__clear__(py);
                    Err(error)
                } else {
                    Ok(Some((indices, error.into_value(py).into_any())))
                }
            }
        }
    }
}
