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
//! 2. The Rust↔Python FFI boundary, crossed by a single `pythonize` on the
//!    inbound path ([`PyServer::recv`] / [`PyServer::try_recv`]) and a single
//!    `depythonize` on the outbound path ([`PyServer::respond`]).
//!
//! So per IPC round-trip the bridge does one reflective serde conversion *per
//! direction* (not two on the same value): the inbound `pythonize` produces the
//! `WorkerRequest` object the Python worker consumes, and the outbound
//! `depythonize` consumes the `WorkerResponse` object the Python worker
//! produces. Each is the minimum needed to materialize a serde-defined type on
//! the far side of the FFI boundary.
//!
//! Eliminating the reflective `pythonize`/`depythonize` would require
//! hand-written, per-variant `IntoPyObject`/`FromPyObject` impls on every
//! `WorkerRequest`/`WorkerResponse` field (kept in lockstep with the wire
//! schema) — a feature, not a behavior-preserving cleanup — so it is
//! intentionally left as the reflective path here.

use std::sync::Mutex;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use pythonize::{depythonize, pythonize};
use uniserve_worker_ipc_core::ServerEndpoint;
use uniserve_worker_wire::WorkerResponse;

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
        let obj = pythonize(py, &req)
            .map_err(|err| py_runtime(format!("failed to pythonize IPC request: {err}")))?;
        Ok(obj.unbind())
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
        let obj = pythonize(py, &req)
            .map_err(|err| py_runtime(format!("failed to pythonize IPC request: {err}")))?;
        Ok(Some(obj.unbind()))
    }

    fn respond(&self, py: Python<'_>, response: &Bound<'_, PyAny>) -> PyResult<()> {
        let resp: WorkerResponse = depythonize(response)
            .map_err(|err| PyErr::new::<PyValueError, _>(format!("invalid response: {err}")))?;
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
