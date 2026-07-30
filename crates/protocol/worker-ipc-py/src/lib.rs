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
//! Each direction uses the serde-defined wire type directly. Inbound requests
//! are converted with `pythonize`, and outbound responses are converted with
//! `depythonize`. Execute batches carry process-local validation provenance
//! after their Rust wire contract has been validated.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::sync::Mutex;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};
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

fn pythonize_request(py: Python<'_>, request: &WorkerRequest) -> PyResult<Py<PyAny>> {
    let object = pythonize(py, request)
        .map_err(|err| py_runtime(format!("failed to pythonize IPC request: {err}")))?;
    if request.kind == RequestKind::Execute {
        mark_validated_batch(py, object.cast::<PyDict>()?)?;
    }
    Ok(object.unbind())
}

fn mark_validated_batch(py: Python<'_>, request: &Bound<'_, PyDict>) -> PyResult<()> {
    if let Some(batch) = request.get_item("batch")? {
        let batch = batch.cast::<PyDict>()?;
        let module = py.import("uniserve_worker.batch")?;
        batch.set_item(
            module.getattr("_WIRE_VALIDATION_KEY")?,
            module.getattr("_WIRE_VALIDATION_TOKEN")?,
        )?;
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_batch_carries_validated_wire_provenance() {
        Python::initialize();
        Python::attach(|py| {
            let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../..")
                .canonicalize()
                .unwrap();
            py.import("sys")
                .unwrap()
                .getattr("path")
                .unwrap()
                .call_method1("insert", (0, repo_root.to_str().unwrap()))
                .unwrap();
            let request = PyDict::new(py);
            let batch = PyDict::new(py);
            request.set_item("batch", &batch).unwrap();
            mark_validated_batch(py, &request).unwrap();
            let module = py.import("uniserve_worker.batch").unwrap();
            let key = module.getattr("_WIRE_VALIDATION_KEY").unwrap();
            let token = module.getattr("_WIRE_VALIDATION_TOKEN").unwrap();
            assert!(batch.get_item(key).unwrap().unwrap().is(&token));
        });
    }
}
