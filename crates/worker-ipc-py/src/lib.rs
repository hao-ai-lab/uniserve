//! Python access to native worker execution, storage and rank channels.
//!
//! `Server` owns a rank's shared-storage or TCP endpoint. The native service
//! borrows it for the serving run and calls Python for numerical preparation
//! and execution. `Client` exposes the head's end of the same channel for
//! direct callers. Standalone channel operations use the public Python
//! protocol mappings; their transfer records are converted by `convert`.
//!
//! Completion callbacks use native wakes and CUDA stream signals. The
//! Python surface is declared in `uniserve_worker/_uniserve_ipc.pyi`.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

mod batches;
mod calls;
mod client;
mod convert;
mod ids;
mod worker;

use std::os::fd::AsRawFd;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use pyo3::wrap_pyfunction;
use pythonize::pythonize;
use uniserve_worker::cuda::{StreamSignal, schedule_completion_wake};
use uniserve_worker_ipc::{RankServer, SHARED_STORAGE_CHANNEL, Wake};
use uniserve_worker_ipc::{RequestKind, WorkerRequest};

#[pyclass(name = "Server", weakref)]
/// Python-facing owner of one worker-side IPC endpoint.
struct PyServer {
    /// Endpoint and completion wake. A call that releases the GIL takes the
    /// endpoint out (`take_endpoint`) and restores it afterwards
    /// (`replace_endpoint`), so the mutex is never held across a transport
    /// call and a concurrent endpoint call fails instead of waiting.
    inner: Mutex<ServerState>,
}

/// Server lifecycle, encoded in which fields are set: both while open and
/// idle, only `completion_wake` while a GIL-free call holds the endpoint, and
/// neither after close. `completion_wake` is therefore the closed flag.
struct ServerState {
    /// Wake source used by CPU, transfer, and device completion callbacks.
    /// Declared first, and taken first by `close`: on a shared-storage
    /// channel it holds a port of the endpoint's node, which must be released
    /// before the endpoint releases the node.
    completion_wake: Option<Wake>,
    endpoint: Option<RankServer>,
}

/// A selector-compatible one-shot signal fired by a CUDA stream host callback.
#[pyclass(name = "StreamSignal")]
struct PyStreamSignal {
    inner: StreamSignal,
}

#[pymethods]
impl PyStreamSignal {
    #[new]
    fn new() -> PyResult<Self> {
        let inner = StreamSignal::new()
            .map_err(|error| py_runtime(format!("creating CUDA stream signal failed: {error}")))?;
        Ok(Self { inner })
    }

    fn fileno(&self) -> i32 {
        self.inner.as_raw_fd()
    }

    fn schedule(&self, stream: usize) -> PyResult<()> {
        self.inner.schedule(stream).map_err(py_runtime)
    }

    fn consume(&self) -> PyResult<()> {
        self.inner
            .consume()
            .map_err(|error| py_runtime(format!("consuming CUDA stream signal failed: {error}")))
    }
}

#[pymethods]
impl PyServer {
    #[new]
    #[pyo3(signature = (service_name, max_payload = 1048576, max_inflight = 1, transport = SHARED_STORAGE_CHANNEL))]
    /// Binds this rank's channel with bounded payload and inflight capacity.
    ///
    /// `transport` is the mechanism the placement calls for: a rank on the
    /// head's host serves shared storage, and a rank elsewhere serves a socket,
    /// where `service_name` is the host to bind rather than a service.
    fn new(
        service_name: &str,
        max_payload: usize,
        max_inflight: usize,
        transport: &str,
    ) -> PyResult<Self> {
        let inner = RankServer::bind(transport, service_name, max_payload, max_inflight)
            .map_err(|err| py_runtime(format!("failed to bind IPC service: {err:#}")))?;
        let completion_wake = inner.completion_wake();
        Ok(Self {
            inner: Mutex::new(ServerState {
                endpoint: Some(inner),
                completion_wake: Some(completion_wake),
            }),
        })
    }

    /// Borrows this open endpoint for a scope that owns its eventual closure.
    fn __enter__(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        if slf.closed()? {
            return Err(py_runtime("IPC server endpoint is closed"));
        }
        Ok(slf)
    }

    /// Closes the endpoint without replacing an exception raised inside the scope.
    fn __exit__(
        &self,
        _exc_type: &Bound<'_, PyAny>,
        exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if let Err(error) = self.close() {
            if exc_value.is_none() {
                return Err(error);
            }
            // Exception notes are diagnostic: even a user-defined add_note()
            // failure must not replace the original exception.
            let _ = exc_value.call_method1(
                "add_note",
                (format!("IPC endpoint cleanup also failed: {error}"),),
            );
        }
        Ok(())
    }

    /// Returns the endpoint the head binds, which the rank reports.
    ///
    /// A shared-storage endpoint is the service it was given; a socket endpoint
    /// is the address its bind produced, which the caller could not know.
    /// Raises `RuntimeError` while another call holds the endpoint and after
    /// close.
    fn endpoint(&self, service: &str) -> PyResult<String> {
        let state = self
            .inner
            .lock()
            .map_err(|_| py_runtime("IPC server mutex poisoned"))?;
        state
            .endpoint
            .as_ref()
            .map(|endpoint| endpoint.endpoint(service))
            .ok_or_else(|| py_runtime("IPC server endpoint is already in use"))
    }

    /// Releases the service after its caller has stopped all endpoint calls.
    /// Repeated close is harmless; closing during a blocking call is rejected.
    fn close(&self) -> PyResult<()> {
        let mut state = self
            .inner
            .lock()
            .map_err(|_| py_runtime("IPC server mutex poisoned"))?;
        if state.completion_wake.is_none() {
            return Ok(());
        }
        if state.endpoint.is_none() {
            return Err(py_runtime("IPC server endpoint is already in use"));
        }
        state.completion_wake.take();
        state.endpoint.take();
        Ok(())
    }

    #[getter]
    /// Reports whether the endpoint owner has released this service.
    fn closed(&self) -> PyResult<bool> {
        let state = self
            .inner
            .lock()
            .map_err(|_| py_runtime("IPC server mutex poisoned"))?;
        Ok(state.completion_wake.is_none())
    }

    /// Waits for one request and converts it into its Python representation.
    fn recv(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut endpoint = self.take_endpoint()?;

        // Move exclusive endpoint ownership into the GIL-free blocking section.
        let (endpoint, result) = py.detach(move || {
            let result: anyhow::Result<WorkerRequest> = (|| {
                let frame = endpoint.recv()?;
                Ok(frame.decode_request()?)
            })();
            (endpoint, result)
        });

        // Restore endpoint ownership before surfacing transport, decode, or
        // validation errors, so a failed receive does not leave the endpoint
        // taken.
        self.replace_endpoint(endpoint)?;
        let req =
            result.map_err(|err| py_runtime(format!("failed to receive IPC request: {err:#}")))?;
        pythonize_request(py, &req)
    }

    /// Attempts one request receive without blocking.
    fn try_recv(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let mut endpoint = self.take_endpoint()?;

        // Keep the endpoint unavailable to concurrent Python calls while the
        // non-blocking transport attempt runs without the GIL.
        let (endpoint, result) = py.detach(move || {
            let result: anyhow::Result<Option<WorkerRequest>> = (|| {
                let Some(frame) = endpoint.try_recv()? else {
                    return Ok(None);
                };
                Ok(Some(frame.decode_request()?))
            })();
            (endpoint, result)
        });
        self.replace_endpoint(endpoint)?;
        let Some(req) =
            result.map_err(|err| py_runtime(format!("failed to receive IPC request: {err:#}")))?
        else {
            return Ok(None);
        };
        Ok(Some(pythonize_request(py, &req)?))
    }

    /// Waits for request or completion readiness while releasing the GIL.
    fn wait_incoming(&self, py: Python<'_>, timeout_us: u64) -> PyResult<()> {
        let endpoint = self.take_endpoint()?;
        let (endpoint, result) = py.detach(move || {
            // A socket endpoint accepts and reads through this call, so the
            // owned endpoint is mutable while the GIL is released.
            let mut endpoint = endpoint;
            let result = endpoint.wait_incoming(std::time::Duration::from_micros(timeout_us));
            (endpoint, result)
        });
        self.replace_endpoint(endpoint)?;
        result.map_err(|err| py_runtime(format!("failed to wait for IPC command: {err:#}")))?;
        Ok(())
    }

    /// Signals that asynchronous worker progress is ready to consume.
    fn wake(&self) -> PyResult<()> {
        self.completion_wake()?.wake();
        Ok(())
    }

    /// Schedules the worker completion wake on a CUDA stream.
    fn wake_on_stream(&self, stream: usize) -> PyResult<()> {
        let wake = self.completion_wake()?;
        schedule_completion_wake(stream, wake).map_err(py_runtime)
    }

    /// Converts and publishes one response for the active request.
    fn respond(&self, py: Python<'_>, response: &Bound<'_, PyAny>) -> PyResult<()> {
        let resp = convert::response_from_py(response)?;

        // Publish without the GIL while retaining exclusive endpoint ownership.
        // Encoding validates info and result payloads, so a response that
        // converts but violates the protocol surfaces as `RuntimeError`.
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

/// Converts a request through the typed submit path or schema-derived fallback.
fn pythonize_request(py: Python<'_>, request: &WorkerRequest) -> PyResult<Py<PyAny>> {
    // Submitted runs use the typed converter, which constructs the worker's
    // Python call objects directly after Rust validation.
    if request.kind() == RequestKind::Submit {
        let object = convert::execute_request_to_py(py, request)?;
        return Ok(object.into_any().unbind());
    }
    let object = pythonize(py, request)
        .map_err(|err| py_runtime(format!("failed to pythonize IPC request: {err}")))?;
    Ok(object.unbind())
}

impl PyServer {
    /// Takes exclusive endpoint ownership for a call that releases the GIL.
    fn take_endpoint(&self) -> PyResult<RankServer> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| py_runtime("IPC server mutex poisoned"))?;
        guard.endpoint.take().ok_or_else(|| {
            py_runtime(if guard.completion_wake.is_none() {
                "IPC server endpoint is closed"
            } else {
                "IPC server endpoint is already in use"
            })
        })
    }

    /// Restores endpoint ownership after a GIL-free call.
    fn replace_endpoint(&self, endpoint: RankServer) -> PyResult<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| py_runtime("IPC server mutex poisoned"))?;
        guard.endpoint = Some(endpoint);
        Ok(())
    }

    /// Clones the completion wake without taking the endpoint, so `wake` and
    /// `wake_on_stream` work while another thread holds the endpoint.
    fn completion_wake(&self) -> PyResult<Wake> {
        let state = self
            .inner
            .lock()
            .map_err(|_| py_runtime("IPC server mutex poisoned"))?;
        state
            .completion_wake
            .clone()
            .ok_or_else(|| py_runtime("IPC server endpoint is closed"))
    }
}

#[pyfunction]
/// Returns the shared-storage service name for one endpoint identifier.
///
/// A rank names its own channel endpoint and reports it to the head, so both
/// sides have to spell the name the same way; this is that one spelling.
fn service_name(id: &str) -> String {
    uniserve_worker_ipc::service_name(id)
}

/// Resolves one aligned 32-bit word inside a writable buffer.
///
/// A shared-storage segment's header words are read and written by different
/// processes, and the readiness word is written after the payload it
/// announces. Python cannot order those stores, so the words are accessed
/// through release and acquire atomics here.
fn buffer_word(buffer: &PyBuffer<u8>, offset: usize) -> PyResult<*mut u32> {
    if buffer.readonly() {
        return Err(py_runtime("atomic word requires a writable buffer"));
    }
    if !buffer.is_c_contiguous()
        || offset
            .checked_add(4)
            .is_none_or(|end| end > buffer.len_bytes())
    {
        return Err(py_runtime(
            "atomic word requires four bytes in a contiguous buffer",
        ));
    }
    // SAFETY: contiguous storage and the checked byte range permit this address;
    // the owning PyBuffer remains alive through the atomic operation.
    let word = unsafe { buffer.buf_ptr().cast::<u8>().add(offset).cast::<u32>() };
    if !word.is_aligned() {
        return Err(py_runtime("atomic word address is not aligned"));
    }
    Ok(word)
}

#[pyfunction]
/// Stores `value` at `offset` with release ordering: every write the calling
/// thread made before the store, to any storage, is visible to a process
/// whose `atomic_load_u32` observes `value`.
fn atomic_store_u32(buffer: PyBuffer<u8>, offset: usize, value: u32) -> PyResult<()> {
    let word = buffer_word(&buffer, offset)?;
    // SAFETY: `buffer_word` checked alignment and bounds; the buffer stays
    // mapped for the call.
    unsafe { AtomicU32::from_ptr(word) }.store(value, Ordering::Release);
    Ok(())
}

#[pyfunction]
/// Loads the word at `offset` with acquire ordering, so that every write the
/// storing process made before its release store is visible afterwards.
fn atomic_load_u32(buffer: PyBuffer<u8>, offset: usize) -> PyResult<u32> {
    let word = buffer_word(&buffer, offset)?;
    // SAFETY: as in `atomic_store_u32`.
    Ok(unsafe { AtomicU32::from_ptr(word) }.load(Ordering::Acquire))
}

#[pymodule]
/// Registers the worker IPC Python extension module.
fn _uniserve_ipc(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ids::RequestKey>()?;
    m.add_class::<ids::CallId>()?;
    m.add_class::<ids::BufferId>()?;
    m.add_class::<calls::Call>()?;
    m.add_class::<batches::Batch>()?;
    m.add_class::<batches::CanvasSampling>()?;
    worker::register(m)?;
    m.add_class::<PyServer>()?;
    m.add_class::<client::PyClient>()?;
    m.add_class::<PyStreamSignal>()?;
    m.add_function(wrap_pyfunction!(service_name, m)?)?;
    m.add_function(wrap_pyfunction!(atomic_store_u32, m)?)?;
    m.add_function(wrap_pyfunction!(atomic_load_u32, m)?)?;
    // Whether this extension was compiled with debug assertions, which an
    // unoptimized (debug-profile) build has. The worker refuses to serve with
    // such a build, since every call crosses this module.
    m.add("DEBUG_BUILD", cfg!(debug_assertions))?;
    Ok(())
}

/// Converts contextual Rust failures into Python runtime errors.
fn py_runtime(message: impl ToString) -> PyErr {
    PyErr::new::<PyRuntimeError, _>(message.to_string())
}
