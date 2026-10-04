//! Python numerical actions and observers for the native host lane.

use std::sync::Arc;
use std::time::Duration;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyBaseException, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyTuple};
use uniserve_worker::{
    Completion as NativeCompletion, HostAction, HostLane as NativeHostLane,
    HostTask as NativeHostTask, Outcome,
};

use super::completion::{Completion, CompletionRef, cancelled};
use super::error::native_error;

enum Dependency {
    Task {
        owner: Py<HostTask>,
        task: Arc<NativeHostTask<PythonAction>>,
    },
    Completion(CompletionRef),
}

struct PythonAction {
    action: Py<PyAny>,
    dependencies: Vec<Dependency>,
    input_ready: Option<Py<PyAny>>,
    input_completion: Option<CompletionRef>,
    release: Option<Py<PyAny>>,
    profile_name: String,
}

impl HostAction for PythonAction {
    type Output = Py<PyAny>;
    type Error = Py<PyBaseException>;
    type Callback = (Py<PyAny>, Py<HostTask>);
    type Wake = Py<PyAny>;

    fn ready(&self) -> Result<bool, Self::Error> {
        Python::attach(|py| {
            self.input_ready
                .as_ref()
                .map_or(Ok(true), |ready| ready.bind(py).call0()?.is_truthy())
                .map_err(|error| error.into_value(py))
        })
    }

    fn run(&self) -> Result<Self::Output, Self::Error> {
        for dependency in &self.dependencies {
            match dependency {
                Dependency::Task { task, .. } => wait_for(&task.completion)?,
                Dependency::Completion(completion) => wait_for(completion)?,
            }
        }

        Python::attach(|py| {
            let run = || -> PyResult<_> {
                let scope = py
                    .import("uniserve.profiling")?
                    .getattr("profile_range")?
                    .call1((&self.profile_name,))?;
                with_context(&scope, || self.action.bind(py).call0().map(Bound::unbind))
            };
            run().map_err(|error| error.into_value(py))
        })
    }

    fn release(&self) -> Result<(), Self::Error> {
        Python::attach(|py| {
            self.release
                .as_ref()
                .map_or(Ok(()), |release| release.bind(py).call0().map(drop))
                .map_err(|error| error.into_value(py))
        })
    }

    fn input_outcome(&self) -> Option<Outcome<Self::Error>> {
        self.input_completion
            .as_ref()
            .map_or(Some(Outcome::Success(())), |completion| {
                completion.outcome()
            })
    }

    fn defer_release(task: Arc<NativeHostTask<Self>>) -> Result<(), Self::Error> {
        Python::attach(|py| {
            let register = || -> PyResult<()> {
                let Some(action) = task.action() else {
                    return Ok(());
                };
                let Some(completion) = &action.input_completion else {
                    return Ok(());
                };
                let callback = PyCFunction::new_closure(py, None, None, move |_args, _kwargs| {
                    task.retire_input().map(drop).map_err(|error| {
                        Python::attach(|py| PyErr::from_value(error.bind(py).clone().into_any()))
                    })
                })?;
                Completion::add_done_callback(
                    completion.owner.bind(py),
                    callback.into_any().unbind(),
                );
                Ok(())
            };
            register().map_err(|error| error.into_value(py))
        })
    }

    fn notify(callbacks: Vec<Self::Callback>) {
        Python::attach(|py| {
            for (callback, task) in callbacks {
                if let Err(error) = callback.bind(py).call1((task,)) {
                    error.write_unraisable(py, Some(callback.bind(py)));
                }
            }
        });
    }

    fn wake(wake: &Self::Wake) {
        Python::attach(|py| {
            if let Err(error) = wake.bind(py).call0() {
                error.write_unraisable(py, Some(wake.bind(py)));
            }
        });
    }

    fn report(error: Self::Error) {
        Python::attach(|py| {
            PyErr::from_value(error.bind(py).clone().into_any()).write_unraisable(py, None)
        });
    }

    fn error(error: uniserve_worker::Error) -> Self::Error {
        Python::attach(|py| native_error(py, error).into_value(py))
    }

    fn note_cleanup(error: &mut Self::Error, cleanup: Self::Error) {
        Python::attach(|py| {
            let _ = error.bind(py).call_method1(
                "add_note",
                (format!(
                    "Host input release also failed: {}",
                    cleanup.bind(py)
                ),),
            );
        });
    }
}

fn wait_for<C, T: Clone>(
    completion: &NativeCompletion<Py<PyBaseException>, C, T>,
) -> Result<(), Py<PyBaseException>> {
    match completion.wait(None) {
        Some(Outcome::Success(_)) => Ok(()),
        Some(Outcome::Failed(error)) => Err(Python::attach(|py| error.clone_ref(py))),
        Some(Outcome::Cancelled) => Err(Python::attach(|py| {
            cancelled(py).unwrap_or_else(|error| error).into_value(py)
        })),
        None => unreachable!("an unbounded wait returns a completed outcome"),
    }
}

#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct HostLane {
    lane: NativeHostLane<PythonAction>,
    #[pyo3(get)]
    max_inflight: usize,
}

#[pymethods]
impl HostLane {
    #[new]
    #[pyo3(signature = (*, max_inflight, workers, name="worker-host-lane"))]
    pub(crate) fn new(
        py: Python<'_>,
        max_inflight: usize,
        workers: usize,
        name: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            lane: NativeHostLane::new(max_inflight, workers, name)
                .map_err(|error| native_error(py, error))?,
            max_inflight,
        })
    }

    #[getter]
    fn reserved(&self) -> usize {
        self.lane.reserved()
    }

    fn set_completion_wake(&self, wake: Option<Py<PyAny>>) {
        self.lane.set_wake(wake);
    }

    pub(crate) fn reserve(&self, py: Python<'_>) -> PyResult<Py<HostTask>> {
        let task = self
            .lane
            .reserve()
            .map_err(|error| native_error(py, error))?;
        Py::new(
            py,
            HostTask {
                task: Arc::clone(&task),
            },
        )
        .inspect_err(|_| {
            let _ = task.cancel(true);
        })
    }

    fn abort(&self) {
        self.lane.abort();
    }

    pub(crate) fn close(&self, py: Python<'_>) -> PyResult<()> {
        let mut errors = py.detach(|| self.lane.close()).into_iter();
        if let Some(mut error) = errors.next() {
            for cleanup in errors {
                PythonAction::note_cleanup(&mut error, cleanup);
            }
            return Err(PyErr::from_value(error.bind(py).clone().into_any()));
        }
        Ok(())
    }
}

#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct HostTask {
    task: Arc<NativeHostTask<PythonAction>>,
}

#[pymethods]
impl HostTask {
    #[pyo3(signature = (action, *, dependencies=Vec::new(), input_ready=None, input_completion=None, release=None, profile_name="uniserve.host"))]
    pub(crate) fn configure<'py>(
        slf: &Bound<'py, Self>,
        action: Py<PyAny>,
        dependencies: Vec<Py<PyAny>>,
        input_ready: Option<Py<PyAny>>,
        input_completion: Option<Py<PyAny>>,
        release: Option<Py<PyAny>>,
        profile_name: &str,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let dependencies = dependencies
            .into_iter()
            .map(|dependency| {
                if let Ok(owner) = dependency.extract::<Py<HostTask>>(py) {
                    let task = Arc::clone(&owner.borrow(py).task);
                    Ok(Dependency::Task { owner, task })
                } else {
                    dependency
                        .extract::<Py<Completion>>(py)
                        .map(|owner| Dependency::Completion(CompletionRef::new(py, owner)))
                        .map_err(Into::into)
                }
            })
            .collect::<PyResult<_>>()?;
        let input_completion = input_completion
            .map(|completion| {
                let owner = completion.bind(py).call0()?.extract()?;
                Ok::<_, PyErr>(CompletionRef::new(py, owner))
            })
            .transpose()?;
        slf.borrow()
            .task
            .configure(PythonAction {
                action,
                dependencies,
                input_ready,
                input_completion,
                release,
                profile_name: profile_name.to_owned(),
            })
            .map_err(|error| native_error(py, error))?;
        Ok(slf.clone())
    }

    #[pyo3(signature = (function, *args, **kwargs))]
    fn submit<'py>(
        slf: &Bound<'py, Self>,
        function: Py<PyAny>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let positional = PyTuple::new(
            py,
            std::iter::once(function.bind(py).clone())
                .chain(args.iter())
                .collect::<Vec<_>>(),
        )?;
        let action = py
            .import("functools")?
            .getattr("partial")?
            .call(positional, kwargs)?
            .unbind();
        Self::configure(slf, action, Vec::new(), None, None, None, "uniserve.host")?;
        slf.borrow()
            .task
            .submit()
            .map_err(|error| native_error(py, error))?;
        Ok(slf.clone())
    }

    pub(crate) fn submit_if_ready(&self, py: Python<'_>) -> PyResult<()> {
        self.task
            .submit_if_ready()
            .map_err(|error| PyErr::from_value(error.bind(py).clone().into_any()))
    }

    fn done(&self) -> bool {
        self.task.completion.done()
    }

    pub(crate) fn cancelled(&self) -> bool {
        matches!(self.task.completion.outcome(), Some(Outcome::Cancelled))
    }

    #[pyo3(signature = (timeout=None))]
    fn result(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<Py<PyAny>> {
        match self.outcome(py, timeout)? {
            Outcome::Success(value) => Ok(value.clone_ref(py)),
            Outcome::Failed(error) => Err(PyErr::from_value(error.bind(py).clone().into_any())),
            Outcome::Cancelled => Err(cancelled(py)?),
        }
    }

    #[pyo3(signature = (timeout=None))]
    pub(crate) fn exception(
        &self,
        py: Python<'_>,
        timeout: Option<f64>,
    ) -> PyResult<Option<Py<PyBaseException>>> {
        match self.outcome(py, timeout)? {
            Outcome::Success(_) => Ok(None),
            Outcome::Failed(error) => Ok(Some(error.clone_ref(py))),
            Outcome::Cancelled => Err(cancelled(py)?),
        }
    }

    pub(crate) fn add_done_callback(slf: &Bound<'_, Self>, callback: Py<PyAny>) {
        let immediate = slf
            .borrow()
            .task
            .completion
            .subscribe((callback, slf.clone().unbind()));
        if let Some(callback) = immediate {
            PythonAction::notify(vec![callback]);
        }
    }

    pub(crate) fn abandon(&self, py: Python<'_>) -> PyResult<()> {
        self.task
            .cancel(true)
            .map(drop)
            .map_err(|error| PyErr::from_value(error.bind(py).clone().into_any()))
    }

    pub(crate) fn cancel(&self, py: Python<'_>) -> PyResult<bool> {
        self.task
            .cancel(false)
            .map_err(|error| PyErr::from_value(error.bind(py).clone().into_any()))
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        // Native queues and running actions are external owners. Their Python
        // references become GC edges only after those owners have released them.
        if Arc::strong_count(&self.task) != 1 {
            return Ok(());
        }
        if let Some(result) = self.task.visit_action(|action| {
            visit.call(&action.action)?;
            visit.call(&action.input_ready)?;
            visit.call(&action.release)?;
            if let Some(completion) = &action.input_completion {
                visit.call(&completion.owner)?;
            }
            for dependency in &action.dependencies {
                match dependency {
                    Dependency::Task { owner, .. } => visit.call(owner)?,
                    Dependency::Completion(completion) => visit.call(&completion.owner)?,
                }
            }
            Ok(())
        }) {
            result?;
        }
        self.task.completion.visit(|outcome, callbacks| {
            match outcome {
                Some(Outcome::Success(value)) => visit.call(value.as_ref())?,
                Some(Outcome::Failed(error)) => visit.call(error.as_ref())?,
                _ => {}
            }
            for (callback, owner) in callbacks {
                visit.call(callback)?;
                visit.call(owner)?;
            }
            Ok(())
        })
    }

    fn __clear__(&mut self) {
        if let Some(task) = Arc::get_mut(&mut self.task) {
            task.clear();
        }
    }
}

impl HostTask {
    fn outcome(
        &self,
        py: Python<'_>,
        timeout: Option<f64>,
    ) -> PyResult<Outcome<Py<PyBaseException>, Arc<Py<PyAny>>>> {
        if let Some(outcome) = self.task.completion.outcome() {
            return Ok(outcome);
        }
        let timeout = timeout
            .map(|seconds| {
                Duration::try_from_secs_f64(seconds.max(0.0))
                    .map_err(|_| PyValueError::new_err("timeout must be finite"))
            })
            .transpose()?;
        py.detach(|| self.task.completion.wait(timeout))
            .ok_or_else(|| PyTimeoutError::new_err("host task has not completed"))
    }
}

/// Enter PyTorch's thread-local inference/device/stream scopes. These contexts
/// restore their prior state on either exit and do not suppress exceptions.
pub(super) fn with_context<T>(
    context: &Bound<'_, PyAny>,
    operation: impl FnOnce() -> PyResult<T>,
) -> PyResult<T> {
    let py = context.py();
    context.call_method0("__enter__")?;
    match operation() {
        Ok(value) => {
            context.call_method1("__exit__", (py.None(), py.None(), py.None()))?;
            Ok(value)
        }
        Err(error) => {
            context.call_method1(
                "__exit__",
                (error.get_type(py), error.value(py), error.traceback(py)),
            )?;
            Err(error)
        }
    }
}
