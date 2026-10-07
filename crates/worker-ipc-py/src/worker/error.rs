//! Translate worker failures at the Python interface.

use std::collections::HashSet;

use pyo3::exceptions::{
    PyAssertionError, PyBaseException, PyException, PyIndexError, PyKeyError,
    PyNotImplementedError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use uniserve_worker_ipc::{
    Batch, Call, CallKind, ErrorCallIdentity, RequestKey, WorkerResponseError,
};

use super::cuda_graph::CUDAGraphError;

/// Context-corrupting CUDA failures require teardown, even if allocation also failed.
const FATAL_CUDA_MESSAGES: &[&str] = &[
    "illegal memory access",
    "cudaerrorillegaladdress",
    "device-side assert",
    "cudaerrorlaunchfailure",
    "unspecified launch failure",
    "unrecoverable",
    "uncorrectable ecc",
    "misaligned address",
];

#[pyfunction]
pub(super) fn should_capture_trace(code: &str) -> bool {
    matches!(
        code,
        "ComputeError" | "ResourceError" | "InvariantViolation" | "FatalWorkerFailure"
    )
}

/// Preserve classified failures and fill only missing context. Python exception
/// objects remain at the language interface; classification and fatality belong here.
#[pyfunction(name = "classify_error", signature = (exc, *, context=None, **kw))]
pub(super) fn classify<'py>(
    exc: &Bound<'py, PyBaseException>,
    context: Option<&str>,
    kw: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = exc.py();
    let module = py.import("uniserve_worker.errors")?;
    if exc.is_instance(&module.getattr("WorkerError")?)? {
        if let Some(kw) = kw {
            for (name, value) in kw {
                let name: String = name.extract()?;
                if !value.is_none()
                    && exc.getattr(name.as_str()).map_or_else(
                        |error| {
                            if error.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) {
                                Ok(true)
                            } else {
                                Err(error)
                            }
                        },
                        |value| Ok(value.is_none()),
                    )?
                {
                    exc.setattr(name.as_str(), value)?;
                }
            }
        }
        return Ok(exc.clone().into_any());
    }

    let mut message = message(exc)?;
    if let Some(context) = context.filter(|context| !context.is_empty()) {
        message = format!("{context}: {message}");
    }
    let lowered = message.to_lowercase();
    let code = if FATAL_CUDA_MESSAGES
        .iter()
        .any(|text| lowered.contains(text))
    {
        "FatalWorkerFailure"
    } else if ["out of memory", "cuda oom", "cublas_status_alloc_failed"]
        .iter()
        .any(|text| lowered.contains(text))
        || exc.get_type().getattr("__mro__")?.try_iter()?.try_fold(
            false,
            |found, class| -> PyResult<bool> {
                Ok(found
                    || class?
                        .getattr("__name__")?
                        .extract::<String>()?
                        .to_lowercase()
                        .contains("outofmemory"))
            },
        )?
    {
        "ResourceError"
    } else if exc.is_instance(&py.import("uniserve.runtime")?.getattr("EventPoolError")?)? {
        "InvariantViolation"
    } else if exc.is_instance_of::<PyNotImplementedError>() {
        "UnsupportedCall"
    } else if exc.is_instance_of::<PyKeyError>()
        || exc.is_instance_of::<PyIndexError>()
        || exc.is_instance_of::<PyTypeError>()
        || exc.is_instance_of::<PyValueError>()
    {
        "InputError"
    } else if exc.is_instance_of::<PyAssertionError>() {
        "InvariantViolation"
    } else {
        "ComputeError"
    };
    make(py, code, &message, kw)
}

fn message(exc: &Bound<'_, PyBaseException>) -> PyResult<String> {
    let message = exc.str()?.to_string();
    if message.is_empty() {
        exc.get_type().getattr("__name__")?.extract()
    } else {
        Ok(message)
    }
}

fn make<'py>(
    py: Python<'py>,
    code: &str,
    message: &str,
    kw: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let module = py.import("uniserve_worker.errors")?;
    if matches!(code, "InputError" | "ComputeError" | "ResourceError") {
        return module.getattr(code)?.call((message,), kw);
    }

    let kw = kw.map_or_else(|| Ok(PyDict::new(py)), PyDictMethods::copy)?;
    if !kw.contains("fatal")? {
        kw.set_item(
            "fatal",
            matches!(code, "InvariantViolation" | "FatalWorkerFailure"),
        )?;
    }
    let code = module.getattr("WorkerErrorCode")?.call1((code,))?;
    module
        .getattr("WorkerError")?
        .call((code, message), Some(&kw))
}

fn call_keys<'py, 'a>(
    py: Python<'py>,
    calls: impl IntoIterator<Item = &'a Call>,
) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(
        py,
        calls
            .into_iter()
            .map(|call| {
                let request = call.request_key;
                Ok((
                    request.engine_id,
                    request.request_id.0,
                    request.request_epoch,
                    Py::new(
                        py,
                        crate::ids::CallId {
                            inner: call.call_id,
                        },
                    )?,
                ))
            })
            .collect::<PyResult<Vec<_>>>()?,
    )
}

/// Attach a numerical phase after the execution stream has joined failed work.
pub(super) fn model_failure(
    py: Python<'_>,
    error: PyErr,
    input: bool,
    kind: CallKind,
    calls: &Bound<'_, PyTuple>,
) -> PyErr {
    if !error.is_instance_of::<PyException>(py) {
        return error;
    }
    let classified = (|| {
        let module = py.import("uniserve_worker.errors")?;
        let class = if input { "InputError" } else { "WorkerError" };
        if error.value(py).is_instance(&module.getattr(class)?)? {
            return Ok(error.value(py).clone().into_any());
        }

        let kw = PyDict::new(py);
        let borrowed = calls
            .iter()
            .map(|call| Ok(call.extract::<PyRef<'_, crate::calls::Call>>()?))
            .collect::<PyResult<Vec<_>>>()?;
        kw.set_item(
            "calls",
            call_keys(py, borrowed.iter().map(|call| call.inner.as_ref()))?,
        )?;
        kw.set_item("route", kind.as_str())?;
        let (code, phase) = if input {
            ("InputError", "input_preparation")
        } else {
            let classified = classify(error.value(py), None, None)?;
            let code: String = classified.getattr("code")?.extract()?;
            if error.is_instance_of::<CUDAGraphError>(py)
                || matches!(code.as_str(), "ResourceError" | "FatalWorkerFailure")
            {
                kw.set_item("fatal", classified.getattr("fatal")?)?;
                ("ResourceError", "graph_or_device")
            } else {
                ("ComputeError", "neural_execution")
            }
        };
        kw.set_item("phase", phase)?;
        make(py, code, &message(error.value(py))?, Some(&kw))
    })();

    match classified {
        Ok(value) if value.is(error.value(py)) => error,
        Ok(value) => {
            let classified = PyErr::from_value(value);
            classified.set_cause(py, Some(error));
            classified
        }
        Err(classification) => classification,
    }
}

/// Fail only the affected batch before commit; after visibility, readers can no
/// longer be rolled back and the worker must stop accepting work.
pub(super) fn batch_failure<'py>(
    error: &Bound<'py, PyBaseException>,
    phase: &str,
    batch: &Batch,
    committed: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let py = error.py();
    let kw = PyDict::new(py);
    kw.set_item("calls", call_keys(py, &batch.calls)?)?;
    kw.set_item("route", batch.calls.first().map(|call| call.code.as_str()))?;
    let classified = if committed {
        kw.set_item("phase", "batch export")?;
        make(
            py,
            "InvariantViolation",
            &format!("batch export failed after visibility began: {error}"),
            Some(&kw),
        )?
    } else {
        kw.set_item("phase", phase)?;
        if let [call] = batch.calls.as_slice() {
            kw.set_item("req_id", call.request_key.request_id.0)?;
            kw.set_item(
                "call_id",
                Py::new(
                    py,
                    crate::ids::CallId {
                        inner: call.call_id,
                    },
                )?,
            )?;
            kw.set_item("call_kind", call.code.as_str())?;
        }
        classify(error, Some(phase), Some(&kw))?
    };
    let code: String = classified.getattr("code")?.extract()?;
    let kwargs = PyDict::new(py);
    if should_capture_trace(&code) {
        kwargs.set_item(
            "exc_info",
            (error.get_type(), error, error.getattr("__traceback__")?),
        )?;
    }
    py.import("logging")?
        .call_method1("getLogger", ("uniserve_worker.errors",))?
        .call_method(
            if should_capture_trace(&code) {
                "error"
            } else {
                "warning"
            },
            (
                "batch failed: %s [code=%s route=%s calls=%s]",
                classified.getattr("message")?,
                &code,
                classified.getattr("route")?,
                classified.getattr("calls")?,
            ),
            Some(&kwargs),
        )?;

    // Log records may outlive the CUDA stream. Keep traceback locations, but
    // release frame locals that could otherwise retain borrowed device buffers.
    let traceback = py.import("traceback")?;
    let mut current = error.clone().into_any();
    let mut seen = HashSet::new();
    while !current.is_none() && seen.insert(current.as_ptr()) {
        traceback.call_method1("clear_frames", (current.getattr("__traceback__")?,))?;
        let cause = current.getattr("__cause__")?;
        current = if cause.is_none() {
            current.getattr("__context__")?
        } else {
            cause
        };
    }
    Ok(classified)
}

pub(super) fn record_failure(
    kind: &str,
    error: &Bound<'_, PyAny>,
    unexpected: bool,
) -> PyResult<()> {
    let py = error.py();
    let code: String = error.getattr("code")?.extract()?;
    let kwargs = PyDict::new(py);
    let trace = unexpected || should_capture_trace(&code);
    if trace {
        kwargs.set_item(
            "exc_info",
            (error.get_type(), error, error.getattr("__traceback__")?),
        )?;
    }
    py.import("logging")?
        .call_method1("getLogger", ("uniserve_worker.profiling",))?
        .call_method(
            if trace { "error" } else { "warning" },
            (
                "worker request %r failed: %s [code=%s request_id=%s call_id=%s call=%s]",
                kind,
                error.getattr("message")?,
                code,
                error.getattr("req_id")?,
                error.getattr("call_id")?,
                error.getattr("call_kind")?,
            ),
            Some(&kwargs),
        )?;
    Ok(())
}

/// Translate the public exception fields directly into the native IPC response.
pub(super) fn response(error: &Bound<'_, PyAny>) -> PyResult<WorkerResponseError> {
    let calls = error
        .getattr("calls")?
        .try_iter()?
        .map(|call| {
            let (engine, request, epoch, id): (u64, u64, u64, Py<crate::ids::CallId>) =
                call?.extract()?;
            Ok(ErrorCallIdentity {
                request_key: RequestKey::new(engine, uniserve_core::RequestId(request), epoch),
                call_id: id.get().inner,
            })
        })
        .collect::<PyResult<_>>()?;

    Ok(WorkerResponseError {
        message: error.getattr("message")?.extract()?,
        code: Some(error.getattr("code")?.extract()?),
        fatal: error.getattr("fatal")?.extract()?,
        phase: error.getattr("phase")?.extract()?,
        route: error.getattr("route")?.extract()?,
        calls,
    })
}

#[pyfunction]
pub(super) fn worker_error_mapping<'py>(error: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    Ok(pythonize::pythonize(error.py(), &response(error)?)?)
}

pub(super) fn native_error(py: Python<'_>, error: uniserve_worker::Error) -> PyErr {
    match error {
        uniserve_worker::Error::Cuda(message) | uniserve_worker::Error::Transport(message) => {
            pyo3::exceptions::PyRuntimeError::new_err(message)
        }
        uniserve_worker::Error::Invalid(message) => invalid(py, message),
        uniserve_worker::Error::State(message) => {
            pyo3::exceptions::PyRuntimeError::new_err(message)
        }
        uniserve_worker::Error::Resource(message) => resource(py, message),
        uniserve_worker::Error::Unsupported(message) => unsupported(py, message),
        error @ uniserve_worker::Error::ReadBackpressure { .. } => resource(py, error.to_string()),
        uniserve_worker::Error::Invariant(message) => invariant(py, message),
    }
}

pub(crate) fn invalid(py: Python<'_>, message: impl Into<String>) -> PyErr {
    match py.import("uniserve_worker.errors").and_then(|module| {
        module
            .getattr("invalid_descriptor")?
            .call1((message.into(),))
    }) {
        Ok(error) => PyErr::from_value(error),
        Err(error) => error,
    }
}

/// Numerical library callers distinguish invalid inputs from worker failures.
pub(super) fn input_error(py: Python<'_>, message: impl Into<String>) -> PyErr {
    match py
        .import("uniserve_worker.errors")
        .and_then(|module| module.getattr("InputError"))
        .and_then(|class| class.call1((message.into(),)))
    {
        Ok(error) => PyErr::from_value(error),
        Err(error) => error,
    }
}

pub(crate) fn resource(py: Python<'_>, message: impl Into<String>) -> PyErr {
    match py
        .import("uniserve_worker.errors")
        .and_then(|module| module.getattr("resource_error")?.call1((message.into(),)))
    {
        Ok(error) => PyErr::from_value(error),
        Err(error) => error,
    }
}

pub(crate) fn unsupported(py: Python<'_>, message: impl Into<String>) -> PyErr {
    match py.import("uniserve_worker.errors").and_then(|module| {
        module
            .getattr("unsupported_setup")?
            .call1((message.into(),))
    }) {
        Ok(error) => PyErr::from_value(error),
        Err(error) => error,
    }
}

pub(crate) fn invariant(py: Python<'_>, message: impl Into<String>) -> PyErr {
    let error = (|| {
        let module = py.import("uniserve_worker.errors")?;
        let code = module
            .getattr("WorkerErrorCode")?
            .getattr("INVARIANT_VIOLATION")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("fatal", true)?;
        module
            .getattr("WorkerError")?
            .call((code, message.into()), Some(&kwargs))
    })();
    match error {
        Ok(error) => PyErr::from_value(error),
        Err(error) => error,
    }
}
