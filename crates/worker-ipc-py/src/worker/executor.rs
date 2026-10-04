//! Python numerical backend for the shared native batch executor.

use std::collections::HashSet;
use std::sync::Arc;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyBaseException, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use uniserve_worker::{Backend, Batch, Executor as NativeExecutor, Submission as NativeSubmission};

use super::error::native_error;

#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct Submission {
    submission: Arc<NativeSubmission>,
}

#[pymethods]
impl Submission {
    #[getter]
    fn batch_id(&self) -> u64 {
        self.submission.batch_id()
    }

    fn notify_ready(&self) {
        self.submission.notify_ready();
    }
}

struct PythonBackend {
    runner: Py<PyAny>,
    batch_type: Py<PyAny>,
}

impl PythonBackend {
    fn call(&self, method: &str, batch: &Py<PyAny>) -> Result<Py<PyAny>, Py<PyBaseException>> {
        Python::attach(|py| {
            self.runner
                .bind(py)
                .call_method1(method, (batch,))
                .map(Bound::unbind)
                .map_err(|error| error.into_value(py))
        })
    }

    fn input_call(
        &self,
        method: &str,
        batch: &Py<PyAny>,
        submission: &Arc<NativeSubmission>,
    ) -> Result<Py<PyAny>, Py<PyBaseException>> {
        Python::attach(|py| {
            let call = || -> PyResult<_> {
                let submission = Py::new(
                    py,
                    Submission {
                        submission: Arc::clone(submission),
                    },
                )?;
                self.runner
                    .bind(py)
                    .call_method1(method, (batch, submission))
                    .map(Bound::unbind)
            };
            call().map_err(|error| error.into_value(py))
        })
    }
}

impl Backend for PythonBackend {
    type Batch = Py<PyAny>;
    type Output = Py<PyAny>;
    type Error = Py<PyBaseException>;

    fn error(&self, error: uniserve_worker::Error) -> Self::Error {
        Python::attach(|py| native_error(py, error).into_value(py))
    }

    fn classify(&self, error: Self::Error, batch: &Self::Batch, context: &str) -> Self::Error {
        Python::attach(|py| {
            let classify = || -> PyResult<_> {
                let kwargs = PyDict::new(py);
                kwargs.set_item("context", context)?;
                kwargs.set_item("route", batch.bind(py).getattr("route")?)?;
                py.import("uniserve_worker.errors")?
                    .getattr("classify")?
                    .call((error,), Some(&kwargs))?
                    .cast_into::<PyBaseException>()
                    .map(Bound::unbind)
                    .map_err(Into::into)
            };
            classify().unwrap_or_else(|error| error.into_value(py))
        })
    }

    fn note_cleanup(&self, error: &mut Self::Error, cleanup: Self::Error) {
        Python::attach(|py| {
            let _ = error.bind(py).call_method1(
                "add_note",
                (format!("batch cleanup failed: {}", cleanup.bind(py)),),
            );
        });
    }

    fn admit(&mut self, batch: &mut Self::Batch) -> Result<(), Self::Error> {
        self.call("admit", batch).map(drop)
    }

    fn prepare(&mut self, batch: &mut Self::Batch) -> Result<(), Self::Error> {
        self.call("prepare", batch).map(drop)
    }

    fn prepare_inputs(
        &mut self,
        batch: &mut Self::Batch,
        submission: &Arc<NativeSubmission>,
    ) -> Result<bool, Self::Error> {
        let ready = self.input_call("prepare_inputs", batch, submission)?;
        Python::attach(|py| {
            ready
                .bind(py)
                .extract::<bool>()
                .map_err(|error| error.into_value(py))
        })
    }

    fn await_inputs(
        &mut self,
        batch: &mut Self::Batch,
        submission: &Arc<NativeSubmission>,
    ) -> Result<(), Self::Error> {
        self.input_call("await_inputs", batch, submission).map(drop)
    }

    fn execute(&mut self, batch: &mut Self::Batch) -> Result<(), Self::Error> {
        self.call("execute", batch).map(drop)
    }

    fn begin_retirement(&mut self, batch: &mut Self::Batch) -> Result<(), Self::Error> {
        self.call("begin_retirement", batch).map(drop)
    }

    fn poll(&mut self, batch: &mut Self::Batch) -> Result<(bool, bool), Self::Error> {
        let progress = self.call("poll", batch)?;
        Python::attach(|py| {
            progress
                .bind(py)
                .extract::<(bool, bool)>()
                .map_err(|error| error.into_value(py))
        })
    }

    fn result(&mut self, batch: &mut Self::Batch) -> Result<Self::Output, Self::Error> {
        Python::attach(|py| {
            batch
                .bind(py)
                .call_method0("result")
                .map(Bound::unbind)
                .map_err(|error| error.into_value(py))
        })
    }

    fn close(&mut self, batch: &mut Self::Batch) -> Result<(), Self::Error> {
        self.call("close", batch).map(drop)
    }

    fn reap(&mut self) -> Result<(), Self::Error> {
        Python::attach(|py| {
            self.runner
                .bind(py)
                .call_method0("reap")
                .map(drop)
                .map_err(|error| error.into_value(py))
        })
    }
}

#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct Executor {
    executor: Option<NativeExecutor<PythonBackend>>,
}

#[pymethods]
impl Executor {
    #[new]
    fn new(py: Python<'_>, worker: &Bound<'_, PyAny>) -> PyResult<Self> {
        let runner = py
            .import("uniserve_worker.execution.batch_runner")?
            .getattr("BatchRunner")?
            .call1((worker,))?;
        let capacity = worker.getattr("info")?.getattr("queue_depth")?.extract()?;
        let distributed = worker
            .getattr("worker_config")?
            .getattr("world_size")?
            .extract::<usize>()?
            > 1;
        let collective = runner.getattr("collective")?.extract()?;
        let batch_type = py
            .import("uniserve_worker.execution.batch")?
            .getattr("BatchState")?
            .unbind();
        let backend = PythonBackend {
            runner: runner.unbind(),
            batch_type,
        };
        let executor = NativeExecutor::new(backend, capacity, distributed, collective)
            .map_err(|error| PyErr::from_value(error.into_bound(py).into_any()))?;
        Ok(Self {
            executor: Some(executor),
        })
    }

    #[pyo3(signature = (batch, *, propagate_errors=false))]
    fn submit(
        &mut self,
        py: Python<'_>,
        batch: &Bound<'_, PyAny>,
        propagate_errors: bool,
    ) -> PyResult<Submission> {
        let executor = self.executor_mut()?;
        let id = batch.getattr("batch_id")?.extract()?;
        let calls = batch.getattr("calls")?;
        let collective_seq = if calls.len()? == 0 {
            None
        } else {
            Some(batch.getattr("collective_seq")?.extract()?)
        };
        let mut requests = HashSet::new();
        let mut producers = HashSet::new();
        for call in calls.try_iter()? {
            let call = call?;
            requests.insert(
                call.getattr("request_key")?
                    .getattr("request_id")?
                    .extract()?,
            );
            for input in call.call_method0("tensor_inputs")?.try_iter()? {
                producers.insert(
                    input?
                        .getattr("producer_call_id")?
                        .getattr("batch_id")?
                        .extract()?,
                );
            }
            for field in ["predicate", "kv_input"] {
                let input = call.getattr(field)?;
                if !input.is_none() {
                    producers.insert(
                        input
                            .getattr("producer_call_id")?
                            .getattr("batch_id")?
                            .extract()?,
                    );
                }
            }
        }
        for command in batch.getattr("commands")?.try_iter()? {
            requests.insert(
                command?
                    .getattr("request_key")?
                    .getattr("request_id")?
                    .extract()?,
            );
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("propagate_errors", propagate_errors)?;
        let state = executor
            .backend()
            .batch_type
            .bind(py)
            .call((batch,), Some(&kwargs))?
            .unbind();
        let submission = executor
            .submit(
                Batch::new(id, collective_seq, requests, producers, state),
                propagate_errors,
            )
            .map_err(|error| PyErr::from_value(error.into_bound(py).into_any()))?;
        Ok(Submission { submission })
    }

    fn advance(&mut self, py: Python<'_>) -> PyResult<bool> {
        self.executor_mut()?
            .advance()
            .map_err(|error| PyErr::from_value(error.into_bound(py).into_any()))
    }

    fn poll(&mut self, py: Python<'_>, submission: &Submission) -> PyResult<Option<Py<PyAny>>> {
        self.executor_mut()?
            .poll(&submission.submission)
            .map_err(|error| PyErr::from_value(error.into_bound(py).into_any()))
    }

    #[getter]
    fn has_work(&self) -> bool {
        self.executor.as_ref().is_some_and(NativeExecutor::has_work)
    }

    #[getter]
    fn started(&self) -> bool {
        self.executor.as_ref().is_some_and(NativeExecutor::started)
    }

    fn reset(&mut self, py: Python<'_>) -> PyResult<()> {
        self.executor_mut()?
            .reset()
            .map_err(|error| PyErr::from_value(error.into_bound(py).into_any()))
    }

    fn drop_request(&mut self, py: Python<'_>, request_id: u64) -> PyResult<()> {
        self.executor_mut()?
            .backend()
            .runner
            .bind(py)
            .call_method1("drop_request", (request_id,))?;
        Ok(())
    }

    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        if let Some(mut executor) = self.executor.take() {
            executor
                .close()
                .map_err(|error| PyErr::from_value(error.into_bound(py).into_any()))?;
        }
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(executor) = &self.executor {
            visit.call(&executor.backend().runner)?;
            visit.call(&executor.backend().batch_type)?;
            for (batch, error) in executor.batches() {
                visit.call(batch)?;
                visit.call(error)?;
            }
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.executor = None;
    }
}

impl Executor {
    fn executor_mut(&mut self) -> PyResult<&mut NativeExecutor<PythonBackend>> {
        self.executor
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("worker executor is closed"))
    }
}
