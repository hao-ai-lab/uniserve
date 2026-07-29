//! PyO3 bridge between the Rust IPC transport and the Python worker.
//!
//! # Boundary conversions (hot path)
//!
//! Each IPC call crosses **two distinct boundaries**, and the bridge performs
//! exactly **one conversion per boundary** — there is no redundant or duplicated
//! reflective conversion to drop:
//!
//! 1. The shared-memory wire boundary (iceoryx2), which carries flatbuffer
//!    bytes. `Frame::decode_request` / `ServerEndpoint::respond` handle this
//!    via the zero-copy-friendly flatbuffer codec in `worker-wire::flat`.
//! 2. The Rust↔Python FFI boundary, crossed once on the inbound path
//!    ([`PyServer::recv`] / [`PyServer::try_recv`]) and once on the outbound
//!    path ([`PyServer::respond`]).
//!
//! The FFI conversion itself is split by frame heat. The steady-state serve
//! loop exchanges one `execute` batch and one `result` completion report per
//! step, and at decode batch sizes the reflective serde walk
//! (`pythonize`/`depythonize`) dominates the boundary cost, so those two frame
//! shapes take the hand-rolled converters in [`convert`]: interned dict keys
//! and enum strings, preallocated lists, and direct scalar conversions that
//! produce values deep-equal to the reflective ones (asserted by the equality
//! tests in `convert`). Every other frame kind is rare (capabilities, admin,
//! metrics, errors) and keeps the reflective path: requests that are not
//! `execute` are pythonized, and any response outside the exact `result`
//! contract falls back to `depythonize`, preserving reflective values and
//! errors byte-for-byte.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

mod convert;

use std::sync::Mutex;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use pythonize::{depythonize, pythonize};
use uniserve_worker_ipc_core::ServerEndpoint;
use uniserve_worker_wire::{RequestKind, WorkerRequest, WorkerResponse};

#[pyclass(name = "Server")]
struct PyServer {
    inner: Mutex<Option<ServerEndpoint>>,
}

#[pymethods]
impl PyServer {
    #[new]
    #[pyo3(signature = (service_name, max_payload = 1048576, max_inflight = 1))]
    fn new(service_name: &str, max_payload: usize, max_inflight: usize) -> PyResult<Self> {
        let inner = ServerEndpoint::bind(service_name, max_payload, max_inflight)
            .map_err(|err| py_runtime(format!("failed to bind IPC service: {err:#}")))?;
        Ok(Self {
            inner: Mutex::new(Some(inner)),
        })
    }

    fn recv(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut endpoint = self.take_endpoint()?;
        let (endpoint, result) = py.detach(move || {
            let result = (|| {
                let frame = endpoint.recv()?;
                frame.decode_request()
            })();
            (endpoint, result)
        });
        self.replace_endpoint(endpoint)?;
        let req = result.map_err(|err: anyhow::Error| {
            py_runtime(format!("failed to receive IPC request: {err:#}"))
        })?;
        pythonize_request(py, &req)
    }

    fn try_recv(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let mut endpoint = self.take_endpoint()?;
        let (endpoint, result) = py.detach(move || {
            let result = (|| {
                let Some(frame) = endpoint.try_recv()? else {
                    return Ok(None);
                };
                Ok(Some(frame.decode_request()?))
            })();
            (endpoint, result)
        });
        self.replace_endpoint(endpoint)?;
        let Some(req) = result.map_err(|err: anyhow::Error| {
            py_runtime(format!("failed to receive IPC request: {err:#}"))
        })?
        else {
            return Ok(None);
        };
        Ok(Some(pythonize_request(py, &req)?))
    }

    fn respond(&self, py: Python<'_>, response: &Bound<'_, PyAny>) -> PyResult<()> {
        // Hot path: the per-step completion report, extracted without the
        // reflective serde walk. Anything else (or any unexpected shape)
        // falls back to `depythonize` for identical values and errors.
        let resp: WorkerResponse = match convert::try_completion_response_from_py(response) {
            Some(resp) => resp,
            None => depythonize(response).map_err(|err| {
                PyErr::new::<PyValueError, _>(format!("invalid response: {err}"))
            })?,
        };
        let mut endpoint = self.take_endpoint()?;
        let (endpoint, result) = py.detach(move || {
            let result = endpoint.respond(&resp);
            (endpoint, result)
        });
        self.replace_endpoint(endpoint)?;
        result.map_err(|err| py_runtime(format!("failed to send IPC response: {err:#}")))?;
        Ok(())
    }
}

fn pythonize_request(py: Python<'_>, request: &WorkerRequest) -> PyResult<Py<PyAny>> {
    // Hot path: `execute` batches take the hand-rolled converter, which
    // produces an object deep-equal to `pythonize`'s. Rare request kinds keep
    // the reflective conversion.
    if request.kind == RequestKind::Execute {
        return Ok(convert::execute_request_to_py(py, request)?.into_any().unbind());
    }
    let object = pythonize(py, request)
        .map_err(|err| py_runtime(format!("failed to pythonize IPC request: {err}")))?;
    Ok(object.unbind())
}

impl PyServer {
    fn take_endpoint(&self) -> PyResult<ServerEndpoint> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| py_runtime("IPC server mutex poisoned"))?;
        guard
            .take()
            .ok_or_else(|| py_runtime("IPC server endpoint is already in use"))
    }

    fn replace_endpoint(&self, endpoint: ServerEndpoint) -> PyResult<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| py_runtime("IPC server mutex poisoned"))?;
        *guard = Some(endpoint);
        Ok(())
    }
}

#[pymodule]
fn _uniserve_ipc(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyServer>()?;
    Ok(())
}

fn py_runtime(message: impl ToString) -> PyErr {
    PyErr::new::<PyRuntimeError, _>(message.to_string())
}
