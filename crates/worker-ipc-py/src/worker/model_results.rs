//! Numerical result views, their completion fence and native observations.

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use uniserve_worker_ipc::ForwardStats as NativeStats;

use crate::stats::ForwardStats;

use super::events::CUDAEvent;

/// Tensor views retain their backing; the event orders reads on another stream.
/// Numerical copies and vocabulary gathers stay with the tensor backend.
/// Vocabulary shards and optional tensor layouts align with `values`. A
/// prepared batch has one value per request row; standalone modules return
/// their own tensor tuples. `greedy` holds graph-computed decode results.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct ExecutionOutput {
    #[pyo3(get)]
    pub(super) values: Py<PyTuple>,
    #[pyo3(get)]
    vocabularies: Py<PyTuple>,
    #[pyo3(get)]
    pub(super) request_pool_indices: Option<Py<PyAny>>,
    #[pyo3(get)]
    pub(super) output_event: Option<Py<CUDAEvent>>,
    #[pyo3(get)]
    pub(super) stats: Option<Py<ForwardStats>>,
    #[pyo3(get)]
    pub(super) greedy: Option<Py<PyAny>>,
    #[pyo3(get)]
    pub(super) layouts: Py<PyTuple>,
}

impl ExecutionOutput {
    pub(super) fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            values: self.values.clone_ref(py),
            vocabularies: self.vocabularies.clone_ref(py),
            request_pool_indices: self
                .request_pool_indices
                .as_ref()
                .map(|value| value.clone_ref(py)),
            output_event: self.output_event.as_ref().map(|value| value.clone_ref(py)),
            stats: self.stats.as_ref().map(|value| value.clone_ref(py)),
            greedy: self.greedy.as_ref().map(|value| value.clone_ref(py)),
            layouts: self.layouts.clone_ref(py),
        }
    }

    /// Copy numerical views after the caller has joined their producer.
    pub(super) fn copy(slf: &Bound<'_, Self>) -> PyResult<Py<Self>> {
        Ok(backend(slf.py())?
            .call_method1("_clone", (slf,))?
            .extract()?)
    }

    fn validate_rows(&mut self, py: Python<'_>) -> PyResult<()> {
        let count = self.values.bind(py).len();
        for (column, name) in [
            (&mut self.vocabularies, "vocabulary metadata"),
            (&mut self.layouts, "output layouts"),
        ] {
            if column.bind(py).is_empty() {
                *column = PyTuple::new(py, (0..count).map(|_| py.None()))?.unbind();
            } else if column.bind(py).len() != count {
                return Err(PyValueError::new_err(format!(
                    "{name} must align with output rows"
                )));
            }
        }

        if self
            .vocabularies
            .bind(py)
            .iter()
            .any(|value| !value.is_none())
        {
            backend(py)?.call_method1(
                "_validate_vocabularies",
                (self.values.bind(py), self.vocabularies.bind(py)),
            )?;
        }

        Ok(())
    }

    /// Join once before consuming the borrowed views. Callers retaining the
    /// result itself keep the event too, so independent consumers also join it.
    pub(super) fn wait(&self, py: Python<'_>) -> PyResult<()> {
        if let Some(event) = &self.output_event {
            let producer = self.values.bind(py).get_item(0).map_err(|_| {
                PyRuntimeError::new_err("forward output has a fence without a producer tensor")
            })?;
            let current = py
                .import("torch.cuda")?
                .call_method1("current_stream", (producer.getattr("device")?,))?;
            event.borrow(py).wait(py, Some(&current))?;
        }
        Ok(())
    }
}

#[pymethods]
impl ExecutionOutput {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (values, vocabularies=None, request_pool_indices=None, output_event=None, stats=None, greedy=None, layouts=None))]
    fn new(
        py: Python<'_>,
        values: Py<PyTuple>,
        vocabularies: Option<Py<PyTuple>>,
        request_pool_indices: Option<Py<PyAny>>,
        output_event: Option<Py<CUDAEvent>>,
        stats: Option<Py<ForwardStats>>,
        greedy: Option<Py<PyAny>>,
        layouts: Option<Py<PyTuple>>,
    ) -> PyResult<Self> {
        let mut output = Self {
            values,
            vocabularies: vocabularies.unwrap_or_else(|| PyTuple::empty(py).unbind()),
            request_pool_indices,
            output_event,
            stats,
            greedy,
            layouts: layouts.unwrap_or_else(|| PyTuple::empty(py).unbind()),
        };
        output.validate_rows(py)?;
        Ok(output)
    }

    /// Derive another result, rechecking tensor metadata only when it changes.
    #[pyo3(signature = (**fields))]
    fn replace(&self, py: Python<'_>, fields: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut output = self.clone_ref(py);
        let mut changed_rows = false;

        if let Some(fields) = fields {
            for (name, value) in fields {
                match name.extract::<&str>()? {
                    "values" => {
                        output.values = value.extract()?;
                        changed_rows = true;
                    }
                    "vocabularies" => {
                        output.vocabularies = value.extract()?;
                        changed_rows = true;
                    }
                    "layouts" => {
                        output.layouts = value.extract()?;
                        changed_rows = true;
                    }
                    "request_pool_indices" => output.request_pool_indices = value.extract()?,
                    "output_event" => output.output_event = value.extract()?,
                    "stats" => output.stats = value.extract()?,
                    "greedy" => output.greedy = value.extract()?,
                    name => {
                        return Err(PyTypeError::new_err(format!(
                            "unknown ExecutionOutput field {name:?}"
                        )));
                    }
                }
            }
        }

        if changed_rows {
            output.validate_rows(py)?;
        }
        Ok(output)
    }

    /// Concatenate completed microbatches after joining their streams.
    /// Graph-greedy outputs carry four row-length completion sections.
    #[staticmethod]
    pub(super) fn combine(py: Python<'_>, outputs: &Bound<'_, PyAny>) -> PyResult<Py<Self>> {
        let outputs = outputs
            .try_iter()?
            .map(|output| Ok(output?.extract::<Py<Self>>()?))
            .collect::<PyResult<Vec<_>>>()?;
        if let [output] = outputs.as_slice() {
            return Ok(output.clone_ref(py));
        }

        let mut values = Vec::new();
        let mut vocabularies = Vec::new();
        let mut layouts = Vec::new();
        let mut greedy = Vec::new();
        let mut stats = NativeStats::default();
        for output in &outputs {
            let output = output.borrow(py);
            values.extend(output.values.bind(py).iter().map(Bound::unbind));
            vocabularies.extend(output.vocabularies.bind(py).iter().map(Bound::unbind));
            layouts.extend(output.layouts.bind(py).iter().map(Bound::unbind));
            if let Some(value) = &output.greedy {
                greedy.push(value.clone_ref(py));
            }
            if let Some(value) = &output.stats {
                stats.merge(&value.borrow(py).inner);
            }
        }

        let greedy = if !outputs.is_empty() && greedy.len() == outputs.len() {
            Some(
                backend(py)?
                    .call_method1("_combine_greedy", (PyTuple::new(py, greedy)?,))?
                    .unbind(),
            )
        } else {
            None
        };

        // Every source already has aligned rows; combining only concatenates
        // those columns and leaves the numerical tensor views borrowed.
        Py::new(
            py,
            Self {
                values: PyTuple::new(py, values)?.unbind(),
                vocabularies: PyTuple::new(py, vocabularies)?.unbind(),
                layouts: PyTuple::new(py, layouts)?.unbind(),
                request_pool_indices: None,
                output_event: None,
                stats: Some(Py::new(py, ForwardStats::from(stats))?),
                greedy,
            },
        )
    }

    pub(super) fn materialize(slf: &Bound<'_, Self>) -> PyResult<Py<Self>> {
        let py = slf.py();
        slf.borrow().wait(py)?;
        Ok(backend(py)?
            .call_method1("_materialize", (slf,))?
            .extract()?)
    }

    fn clone(slf: &Bound<'_, Self>) -> PyResult<Py<Self>> {
        let py = slf.py();
        slf.borrow().wait(py)?;
        Self::copy(slf)
    }

    pub(super) fn validate_for(&self, py: Python<'_>, batch: &Bound<'_, PyAny>) -> PyResult<()> {
        if self.values.bind(py).len() != batch.getattr("row_count")?.extract::<usize>()? {
            return Err(PyValueError::new_err(
                "model output count does not match forward rows",
            ));
        }
        let tensor = py.import("torch")?.getattr("Tensor")?;
        for value in self.values.bind(py) {
            if !value.is_instance(&tensor)? {
                return Err(PyTypeError::new_err("model output values must be tensors"));
            }
        }
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.values)?;
        visit.call(&self.vocabularies)?;
        visit.call(&self.layouts)?;
        visit.call(&self.request_pool_indices)?;
        visit.call(&self.output_event)?;
        visit.call(&self.stats)?;
        visit.call(&self.greedy)
    }

    fn __clear__(&mut self, py: Python<'_>) {
        self.values = PyTuple::empty(py).unbind();
        self.vocabularies = PyTuple::empty(py).unbind();
        self.layouts = PyTuple::empty(py).unbind();
        self.request_pool_indices = None;
        self.output_event = None;
        self.stats = None;
        self.greedy = None;
    }
}

fn backend(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("uniserve_worker.model_executor.output")
}
