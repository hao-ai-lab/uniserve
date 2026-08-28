//! PyO3 bridge between the Rust IPC transport and the Python worker.
//!
//! # Boundary conversions
//!
//! Each IPC call crosses two boundaries:
//!
//! 1. The shared-memory IPC boundary (iceoryx2), which carries flatbuffer
//!    bytes. `Frame::decode_request` / `ServerEndpoint::respond` handle this
//!    via the zero-copy-friendly flatbuffer codec in `uniserve-worker-ipc::codec`.
//! 2. The Rust↔Python FFI boundary, crossed once on the inbound path
//!    ([`PyServer::recv`] / [`PyServer::try_recv`]) and once on the outbound
//!    path ([`PyServer::respond`]).
//!
//! The steady-state `execute` and `result` frames use typed converters with
//! interned keys, preallocated lists, and direct scalar extraction. Worker-info,
//! control, pressure, snapshot, and error frames use the schema-derived serde
//! converter. Frame kind determines exactly one conversion path.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

mod convert;

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use pythonize::{depythonize, pythonize};
use uniserve_worker_ipc::{RequestKind, WorkerRequest, WorkerResponse};
use uniserve_worker_ipc::{ServerEndpoint, WakeSender};

#[pyclass(name = "Server")]
struct PyServer {
    inner: Mutex<Option<ServerEndpoint>>,
    completion_wake: WakeSender,
}

struct StreamSignalState {
    fd: i32,
    scheduled: AtomicBool,
}

impl Drop for StreamSignalState {
    fn drop(&mut self) {
        // SAFETY: this state exclusively owns the eventfd.
        unsafe {
            libc::close(self.fd);
        }
    }
}

/// A selector-compatible one-shot signal fired by a CUDA stream host callback.
#[pyclass(name = "StreamSignal")]
struct PyStreamSignal {
    state: Arc<StreamSignalState>,
}

type CudaLaunchHostFunc = unsafe extern "C" fn(
    stream: *mut c_void,
    callback: Option<unsafe extern "C" fn(*mut c_void)>,
    user_data: *mut c_void,
) -> i32;

struct CudaRuntime {
    _library: libloading::Library,
    launch_host_func: CudaLaunchHostFunc,
}

static CUDA_RUNTIME: OnceLock<Result<CudaRuntime, String>> = OnceLock::new();

fn cuda_runtime() -> Result<&'static CudaRuntime, String> {
    CUDA_RUNTIME
        .get_or_init(|| {
            let mut failure = String::new();
            for name in ["libcudart.so", "libcudart.so.13", "libcudart.so.12"] {
                // SAFETY: the library handle remains owned by `CudaRuntime` for the
                // process lifetime and the copied symbol has the CUDA runtime ABI.
                let library = match unsafe { libloading::Library::new(name) } {
                    Ok(library) => library,
                    Err(error) => {
                        failure = format!("{name}: {error}");
                        continue;
                    }
                };
                // SAFETY: `cudaLaunchHostFunc` has the signature declared by the
                // CUDA runtime API and the library stays live in the result.
                let launch_host_func = unsafe {
                    *library
                        .get::<CudaLaunchHostFunc>(b"cudaLaunchHostFunc\0")
                        .map_err(|error| format!("loading cudaLaunchHostFunc: {error}"))?
                };
                return Ok(CudaRuntime {
                    _library: library,
                    launch_host_func,
                });
            }
            Err(format!(
                "loading CUDA runtime for host callbacks failed: {failure}"
            ))
        })
        .as_ref()
        .map_err(Clone::clone)
}

unsafe extern "C" fn completion_callback(user_data: *mut c_void) {
    // SAFETY: `schedule_completion_wake` passes ownership of exactly one boxed
    // `WakeSender` to CUDA, which invokes this callback exactly once.
    let wake = unsafe { Box::from_raw(user_data.cast::<WakeSender>()) };
    wake.wake();
}

unsafe extern "C" fn stream_signal_callback(user_data: *mut c_void) {
    // SAFETY: `PyStreamSignal::schedule` transfers one boxed Arc to CUDA and
    // CUDA invokes this callback exactly once after preceding stream work.
    let state = unsafe { Box::from_raw(user_data.cast::<Arc<StreamSignalState>>()) };
    let value = 1_u64.to_ne_bytes();
    // SAFETY: `state` keeps the eventfd alive for this call. The descriptor is
    // non-blocking and an eventfd counter cannot saturate under a one-shot use.
    let _ = unsafe { libc::write(state.fd, value.as_ptr().cast(), value.len()) };
}

fn schedule_completion_wake(stream: usize, wake: WakeSender) -> Result<(), String> {
    let runtime = cuda_runtime()?;
    let user_data = Box::into_raw(Box::new(wake)).cast::<c_void>();
    // SAFETY: `stream` is the native CUDA stream address supplied by PyTorch;
    // `user_data` remains owned by CUDA until `completion_callback` runs.
    let status = unsafe {
        (runtime.launch_host_func)(stream as *mut c_void, Some(completion_callback), user_data)
    };
    if status != 0 {
        // SAFETY: CUDA rejected the callback and therefore did not take ownership.
        drop(unsafe { Box::from_raw(user_data.cast::<WakeSender>()) });
        return Err(format!(
            "cudaLaunchHostFunc failed with CUDA status {status}"
        ));
    }
    Ok(())
}

#[pymethods]
impl PyStreamSignal {
    #[new]
    fn new() -> PyResult<Self> {
        // SAFETY: eventfd has no pointer arguments and returns an owned fd.
        let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
        if fd < 0 {
            return Err(py_runtime(format!(
                "creating CUDA stream signal failed: {}",
                std::io::Error::last_os_error()
            )));
        }
        Ok(Self {
            state: Arc::new(StreamSignalState {
                fd,
                scheduled: AtomicBool::new(false),
            }),
        })
    }

    fn fileno(&self) -> i32 {
        self.state.fd
    }

    fn schedule(&self, stream: usize) -> PyResult<()> {
        self.state
            .scheduled
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| py_runtime("CUDA stream signal was scheduled more than once"))?;
        let runtime = cuda_runtime().map_err(py_runtime)?;
        let user_data = Box::into_raw(Box::new(Arc::clone(&self.state))).cast::<c_void>();
        // SAFETY: `stream` is a native CUDA stream address supplied by PyTorch;
        // the boxed Arc remains owned by CUDA until the callback runs.
        let status = unsafe {
            (runtime.launch_host_func)(
                stream as *mut c_void,
                Some(stream_signal_callback),
                user_data,
            )
        };
        if status != 0 {
            // SAFETY: CUDA rejected the callback and did not take ownership.
            drop(unsafe { Box::from_raw(user_data.cast::<Arc<StreamSignalState>>()) });
            self.state.scheduled.store(false, Ordering::Release);
            return Err(py_runtime(format!(
                "cudaLaunchHostFunc failed with CUDA status {status}"
            )));
        }
        Ok(())
    }

    fn consume(&self) -> PyResult<()> {
        let mut value = 0_u64;
        // SAFETY: the state owns a valid non-blocking eventfd and `value` is a
        // writable eight-byte destination, as required by eventfd.
        let read = unsafe {
            libc::read(
                self.state.fd,
                (&mut value as *mut u64).cast(),
                std::mem::size_of::<u64>(),
            )
        };
        if read == std::mem::size_of::<u64>() as isize && value > 0 {
            return Ok(());
        }
        Err(py_runtime(if read < 0 {
            format!(
                "consuming CUDA stream signal failed: {}",
                std::io::Error::last_os_error()
            )
        } else {
            "CUDA stream signal carried an invalid eventfd value".to_owned()
        }))
    }
}

#[pymethods]
impl PyServer {
    #[new]
    #[pyo3(signature = (service_name, max_payload = 1048576, max_inflight = 1))]
    fn new(service_name: &str, max_payload: usize, max_inflight: usize) -> PyResult<Self> {
        let inner = ServerEndpoint::bind(service_name, max_payload, max_inflight)
            .map_err(|err| py_runtime(format!("failed to bind IPC service: {err:#}")))?;
        let completion_wake = inner.completion_wake();
        Ok(Self {
            inner: Mutex::new(Some(inner)),
            completion_wake,
        })
    }

    fn recv(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut endpoint = self.take_endpoint()?;
        let (endpoint, result) = py.detach(move || {
            let result: anyhow::Result<WorkerRequest> = (|| {
                let frame = endpoint.recv()?;
                Ok(frame.decode_request()?)
            })();
            (endpoint, result)
        });
        self.replace_endpoint(endpoint)?;
        let req =
            result.map_err(|err| py_runtime(format!("failed to receive IPC request: {err:#}")))?;
        pythonize_request(py, &req)
    }

    fn try_recv(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let mut endpoint = self.take_endpoint()?;
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

    fn wait_incoming(&self, py: Python<'_>, timeout_us: u64) -> PyResult<()> {
        let endpoint = self.take_endpoint()?;
        let (endpoint, result) = py.detach(move || {
            let result = endpoint.wait_incoming(std::time::Duration::from_micros(timeout_us));
            (endpoint, result)
        });
        self.replace_endpoint(endpoint)?;
        result.map_err(|err| py_runtime(format!("failed to wait for IPC command: {err:#}")))?;
        Ok(())
    }

    fn wake(&self) {
        self.completion_wake.wake();
    }

    fn wake_on_stream(&self, stream: usize) -> PyResult<()> {
        let wake = self.completion_wake.clone();
        schedule_completion_wake(stream, wake).map_err(py_runtime)
    }

    fn respond(&self, py: Python<'_>, response: &Bound<'_, PyAny>) -> PyResult<()> {
        // Per-step result reports use the typed extractor. Every other response
        // kind is decoded by the schema-derived converter.
        let resp: WorkerResponse = match convert::try_completion_response_from_py(response)? {
            Some(resp) => resp,
            None => depythonize(response)
                .map_err(|err| PyErr::new::<PyValueError, _>(format!("invalid response: {err}")))?,
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
    // Execute batches use the typed converter, which constructs the worker's
    // Python operation objects directly: the decoded Rust batch has already passed
    // `Batch::validate`.
    if request.kind() == RequestKind::Execute {
        let object = convert::execute_request_to_py(py, request)?;
        return Ok(object.into_any().unbind());
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
    m.add_class::<PyStreamSignal>()?;
    Ok(())
}

fn py_runtime(message: impl ToString) -> PyErr {
    PyErr::new::<PyRuntimeError, _>(message.to_string())
}
