//! Python numerical backend for the shared native batch executor.

use std::sync::Arc;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyBaseException, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pythonize::depythonize;
use uniserve_worker::{
    Backend, Batch, Executor as NativeExecutor, Service, ServiceBackend,
    Submission as NativeSubmission,
};
use uniserve_worker_ipc::{
    Batch as BatchPlan, BatchOutput, RequestKind, WorkerInfo, WorkerResponseError,
};

use super::host::with_context;
use super::request::RequestPool;
use crate::{PyServer, convert};

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
    requests: Py<RequestPool>,
    model_runner: Py<PyAny>,
    transports: Vec<Py<PyAny>>,
    info: WorkerInfo,
}

impl PythonBackend {
    fn batch_state(
        &self,
        py: Python<'_>,
        batch: &Bound<'_, PyAny>,
        propagate_errors: bool,
    ) -> PyResult<Py<PyAny>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("propagate_errors", propagate_errors)?;
        self.batch_type
            .bind(py)
            .call((batch,), Some(&kwargs))
            .map(Bound::unbind)
    }

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

impl ServiceBackend for PythonBackend {
    fn batch(&self, plan: &BatchPlan) -> Result<Py<PyAny>, Py<PyBaseException>> {
        Python::attach(|py| {
            let build = || {
                let range = py
                    .import("uniserve.profiling")?
                    .getattr("profile_range")?
                    .call1(("uniserve.worker.batch_decode",))?;
                with_context(&range, || {
                    let batch = convert::batch_to_py(py, plan)?;
                    self.batch_state(py, &batch, false)
                })
            };
            build().map_err(|error: PyErr| error.into_value(py))
        })
    }

    fn output(&self, output: Py<PyAny>) -> Result<BatchOutput, Py<PyBaseException>> {
        Python::attach(|py| {
            let convert = || -> PyResult<_> {
                let output = output.bind(py).call_method0("to_mapping")?;
                convert::run_result_from_py(&output)
                    .ok_or_else(|| PyValueError::new_err("invalid worker batch output"))
            };
            convert().map_err(|error| error.into_value(py))
        })
    }

    fn response_error(
        &self,
        kind: RequestKind,
        error: Py<PyBaseException>,
    ) -> Result<WorkerResponseError, Py<PyBaseException>> {
        Python::attach(|py| {
            let report = || -> PyResult<_> {
                let errors = py.import("uniserve_worker.errors")?;
                let unexpected = !error
                    .bind(py)
                    .is_instance(&errors.getattr("WorkerError")?)?;
                let classified = if unexpected {
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("context", kind.as_str())?;
                    errors.getattr("classify")?.call((error,), Some(&kwargs))?
                } else {
                    error.into_bound(py).into_any()
                };
                let kwargs = PyDict::new(py);
                kwargs.set_item("unexpected", unexpected)?;
                py.import("uniserve_worker.profiling")?
                    .getattr("record_failure")?
                    .call((kind.as_str(), &classified), Some(&kwargs))?;
                depythonize(&classified.call_method0("to_mapping")?).map_err(Into::into)
            };
            report().map_err(|error| error.into_value(py))
        })
    }

    fn has_open_requests(&self) -> Result<bool, Py<PyBaseException>> {
        Python::attach(|py| {
            self.requests
                .borrow(py)
                .has_open_requests(py)
                .map_err(|error| error.into_value(py))
        })
    }

    fn awaiting_acknowledgment(&self) -> Result<bool, Py<PyBaseException>> {
        Python::attach(|py| {
            for transport in &self.transports {
                let waiting = transport
                    .bind(py)
                    .call_method0("awaiting_acknowledgment")
                    .and_then(|value| value.extract::<bool>())
                    .map_err(|error| error.into_value(py))?;
                if waiting {
                    return Ok(true);
                }
            }
            Ok(false)
        })
    }

    fn join_expert_step(&self, leaving: bool) -> Result<(bool, bool), Py<PyBaseException>> {
        Python::attach(|py| {
            let join = || -> PyResult<_> {
                let runner = self.model_runner.bind(py);
                let kwargs = PyDict::new(py);
                kwargs.set_item("leaving", leaving)?;
                let advanced = runner
                    .call_method("join_expert_step", (), Some(&kwargs))?
                    .extract()?;
                let released = runner.getattr("experts")?.getattr("released")?.extract()?;
                Ok((advanced, released))
            };
            join().map_err(|error| error.into_value(py))
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
            requests: worker.getattr("requests")?.extract()?,
            model_runner: worker.getattr("runner")?.unbind(),
            transports: worker
                .getattr("transports")?
                .call_method0("values")?
                .try_iter()?
                .map(|value| value.map(Bound::unbind))
                .collect::<PyResult<_>>()?,
            info: depythonize(&worker.getattr("info")?.call_method0("to_mapping")?)?,
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
        let plan = convert::batch_from_py(&batch.call_method0("to_mapping")?)?;
        let state = executor
            .backend()
            .batch_state(py, batch, propagate_errors)?;
        let submission = executor
            .submit(Batch::from_plan(&plan, state), propagate_errors)
            .map_err(|error| PyErr::from_value(error.into_bound(py).into_any()))?;
        Ok(Submission { submission })
    }

    /// Serve the rank's native channel without Python request/response envelopes.
    /// The resource owner retains both executor and endpoint after this returns.
    fn serve(&mut self, py: Python<'_>, server: &PyServer) -> PyResult<()> {
        let executor = self.executor_mut()?;
        let info = executor.backend().info.clone();
        let experts = !executor
            .backend()
            .model_runner
            .bind(py)
            .getattr("experts")?
            .is_none();
        let gc = py.import("gc")?;
        let gc_enabled: bool = gc.call_method0("isenabled")?.extract()?;
        let mut endpoint = server.take_endpoint()?;

        // Cyclic collection traverses resident models and graphs. Keep it
        // outside serving and restore the caller's mode on either exit.
        let result = (|| -> PyResult<()> {
            gc.call_method0("disable")?;
            py.detach(|| Service::new(executor, &mut endpoint, info, experts).run())
                .map_err(|error| PyErr::from_value(error.into_bound(py).into_any()))
        })();
        let restored = server.replace_endpoint(endpoint);
        if gc_enabled {
            gc.call_method0("enable")?;
        }
        restored?;
        result
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
            visit.call(&executor.backend().requests)?;
            visit.call(&executor.backend().model_runner)?;
            for transport in &executor.backend().transports {
                visit.call(transport)?;
            }
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
