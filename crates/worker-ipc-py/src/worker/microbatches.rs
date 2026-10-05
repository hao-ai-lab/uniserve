//! Cooperative numerical calls on the shared native host executor.

use std::sync::Arc;
use std::time::Duration;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyBaseException, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use uniserve_worker::{
    Error, HostAction, HostLane, HostTask, Microbatches as NativeMicrobatches, Outcome,
};

use super::host::PythonAction;

struct Microbatch {
    action: PythonAction,
    turn: Option<(Arc<NativeMicrobatches>, usize)>,
}

impl HostAction for Microbatch {
    type Output = Py<PyAny>;
    type Error = Py<PyBaseException>;
    type Callback = ();
    type Wake = ();

    fn ready(&self) -> Result<bool, Self::Error> {
        Ok(true)
    }

    fn run(&self) -> Result<Self::Output, Self::Error> {
        match &self.turn {
            Some((owner, index)) => owner
                .run(*index, || self.action.run())
                .map_err(Self::error)?,
            None => self.action.run(),
        }
    }

    // Calls borrow prepared contexts, with no asynchronous input lease of
    // their own. The invocation's CUDA stream join orders their device work.
    fn release(&self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn input_outcome(&self) -> Option<Outcome<Self::Error>> {
        Some(Outcome::Success(()))
    }

    fn defer_release(_task: Arc<HostTask<Self>>) -> Result<(), Self::Error> {
        Ok(())
    }

    fn notify(_callbacks: Vec<()>) {}

    fn wake(_wake: &()) {}

    fn report(error: Self::Error) {
        PythonAction::report(error);
    }

    fn error(error: Error) -> Self::Error {
        Python::attach(|py| native_error(error).into_value(py))
    }

    fn note_cleanup(error: &mut Self::Error, cleanup: Self::Error) {
        PythonAction::note_cleanup(error, cleanup);
    }
}

type Task = Arc<HostTask<Microbatch>>;

/// Persistent host threads and cooperative turns for borrowed CUDA contexts.
/// Contexts outlive this owner and their graphs. Close after captured readers retire.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct Microbatches {
    control: Arc<NativeMicrobatches>,
    lanes: Vec<HostLane<Microbatch>>,
    #[pyo3(get)]
    contexts: Py<PyTuple>,
    #[pyo3(get)]
    device: Py<PyAny>,
}

#[pymethods]
impl Microbatches {
    #[new]
    fn new(py: Python<'_>, contexts: Vec<Py<PyAny>>) -> PyResult<Self> {
        let control = Arc::new(NativeMicrobatches::new(contexts.len()).map_err(native_error)?);
        let contexts = PyTuple::new(py, contexts)?;
        let backend = backend(py)?;
        let device = backend.call_method1("_prepare", (&contexts,))?.unbind();
        let lanes = (0..contexts.len())
            .map(|index| HostLane::new(1, 1, &format!("microbatch-{index}")))
            .collect::<uniserve_worker::Result<Vec<_>>>()
            .map_err(native_error)?;
        let owner = Self {
            control,
            lanes,
            contexts: contexts.unbind(),
            device,
        };

        // Warm the same fixed host threads used for numerical calls. cuBLAS
        // creates thread-local handles that must exist before graph capture.
        let warmed = (|| {
            let partial = py.import("functools")?.getattr("partial")?;
            for (index, context) in owner.contexts.bind(py).iter().enumerate() {
                let call = partial
                    .call1((backend.getattr("_warm")?, context))?
                    .unbind();
                let task = owner.submit(index, call, false)?;
                match py.detach(|| task.completion.wait(None)) {
                    Some(Outcome::Success(_)) => {}
                    Some(Outcome::Failed(error)) => return Err(exception(py, &error)),
                    _ => return Err(PyRuntimeError::new_err("microbatch warmup cancelled")),
                }
            }
            Ok(())
        })();
        if let Err(error) = warmed {
            owner.close(py)?;
            return Err(error);
        }
        Ok(owner)
    }

    /// Run one numerical call per context and return results in input order.
    /// Failures wake suspended peers and all host turns retire before return.
    fn __call__(&self, py: Python<'_>, calls: Vec<Py<PyAny>>) -> PyResult<Vec<Py<PyAny>>> {
        self.control.begin(calls.len()).map_err(native_error)?;
        let mut tasks = Vec::with_capacity(calls.len());
        let mut current = None;
        let submitted = (|| {
            let backend = backend(py)?;
            current = Some(
                backend
                    .call_method1("_fork", (self.contexts.bind(py), self.device.bind(py)))?
                    .unbind(),
            );
            let partial = py.import("functools")?.getattr("partial")?;
            let copy_context = py.import("contextvars")?.getattr("copy_context")?;
            let execute = backend.getattr("_execute")?;
            for (index, call) in calls.into_iter().enumerate() {
                let context = self.contexts.bind(py).get_item(index)?;
                let action = partial
                    .call1((
                        copy_context.call0()?.getattr("run")?,
                        &execute,
                        context,
                        call,
                    ))?
                    .unbind();
                tasks.push(self.submit(index, action, true)?);
            }
            Ok(())
        })();

        let mut interrupted = None;
        if let Err(error) = submitted
            && self.control.abort()
        {
            interrupted = Some(error);
        }
        for task in &tasks {
            loop {
                if py
                    .detach(|| task.completion.wait(Some(Duration::from_millis(50))))
                    .is_some()
                {
                    break;
                }
                // The main Python thread must observe interruption even if
                // a numerical call is waiting for another host participant.
                if let Err(error) = py.check_signals()
                    && self.control.abort()
                {
                    interrupted = Some(error);
                }
            }
        }

        let completed = (|| {
            if let Some(current) = current {
                backend(py)?.call_method1("_join", (current, self.contexts.bind(py)))?;
            }
            if let Some(error) = interrupted {
                return Err(error);
            }
            if let Some(index) = self.control.failure()
                && let Some(Outcome::Failed(error)) = tasks[index].completion.outcome()
            {
                return Err(exception(py, &error));
            }

            tasks
                .iter()
                .map(|task| match task.completion.outcome() {
                    Some(Outcome::Success(value)) => Ok(value.clone_ref(py)),
                    Some(Outcome::Failed(error)) => Err(exception(py, &error)),
                    _ => Err(PyRuntimeError::new_err("microbatch execution cancelled")),
                })
                .collect()
        })();
        self.control.end();
        completed
    }

    /// Join persistent threads after invocations and captured readers retire.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        self.control.close().map_err(native_error)?;
        let errors = py.detach(|| {
            self.lanes
                .iter()
                .flat_map(HostLane::close)
                .collect::<Vec<_>>()
        });
        if let Some(error) = errors.first() {
            return Err(exception(py, error));
        }
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.contexts)?;
        visit.call(&self.device)
    }

    fn __clear__(&mut self, py: Python<'_>) {
        if let Err(error) = self.close(py) {
            error.write_unraisable(py, None);
        }
        self.contexts = PyTuple::empty(py).unbind();
    }
}

impl Microbatches {
    fn submit(&self, index: usize, call: Py<PyAny>, rotate: bool) -> PyResult<Task> {
        let task = self.lanes[index].reserve().map_err(native_error)?;
        let action = Microbatch {
            action: PythonAction::numerical(call, "uniserve.microbatch"),
            turn: rotate.then(|| (Arc::clone(&self.control), index)),
        };
        if let Err(error) = task.configure(action).and_then(|()| task.submit()) {
            let _ = task.cancel(true);
            return Err(native_error(error));
        }
        Ok(task)
    }
}

#[pyfunction]
pub(super) fn yield_microbatch(py: Python<'_>) -> PyResult<()> {
    py.detach(uniserve_worker::yield_microbatch)
        .map_err(native_error)
}

fn backend(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("uniserve.runtime.microbatches")
}

fn native_error(error: Error) -> PyErr {
    match error {
        Error::Invalid(message) => PyValueError::new_err(message),
        error => PyRuntimeError::new_err(error.to_string()),
    }
}

fn exception(py: Python<'_>, error: &Py<PyBaseException>) -> PyErr {
    PyErr::from_value(error.bind(py).clone().into_any())
}
