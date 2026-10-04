//! CUDA completion events and host notifications shared by worker backends.

use std::ffi::c_void;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use libloading::Library;
use uniserve_worker_ipc::Wake;

type Handle = *mut c_void;
type Status = i32;

struct Driver {
    _library: Library,
    context: unsafe extern "C" fn(*mut Handle) -> Status,
    push: unsafe extern "C" fn(Handle) -> Status,
    pop: unsafe extern "C" fn(*mut Handle) -> Status,
    create: unsafe extern "C" fn(*mut Handle, u32) -> Status,
    record: unsafe extern "C" fn(Handle, Handle) -> Status,
    query: unsafe extern "C" fn(Handle) -> Status,
    synchronize: unsafe extern "C" fn(Handle) -> Status,
    wait: unsafe extern "C" fn(Handle, Handle, u32) -> Status,
    destroy: unsafe extern "C" fn(Handle) -> Status,
    launch: unsafe extern "C" fn(Handle, unsafe extern "C" fn(Handle), Handle) -> Status,
}

fn driver() -> Result<&'static Driver, String> {
    static DRIVER: OnceLock<Result<Driver, String>> = OnceLock::new();
    DRIVER
        .get_or_init(|| {
            // The CUDA driver has a stable system soname. No toolkit or user
            // installation path is involved; the library owns every loaded symbol.
            unsafe {
                let library = Library::new("libcuda.so.1").map_err(|e| e.to_string())?;
                Ok(Driver {
                    context: *library
                        .get(b"cuCtxGetCurrent\0")
                        .map_err(|e| e.to_string())?,
                    push: *library
                        .get(b"cuCtxPushCurrent_v2\0")
                        .map_err(|e| e.to_string())?,
                    pop: *library
                        .get(b"cuCtxPopCurrent_v2\0")
                        .map_err(|e| e.to_string())?,
                    create: *library.get(b"cuEventCreate\0").map_err(|e| e.to_string())?,
                    record: *library.get(b"cuEventRecord\0").map_err(|e| e.to_string())?,
                    query: *library.get(b"cuEventQuery\0").map_err(|e| e.to_string())?,
                    synchronize: *library
                        .get(b"cuEventSynchronize\0")
                        .map_err(|e| e.to_string())?,
                    wait: *library
                        .get(b"cuStreamWaitEvent\0")
                        .map_err(|e| e.to_string())?,
                    destroy: *library
                        .get(b"cuEventDestroy_v2\0")
                        .map_err(|e| e.to_string())?,
                    launch: *library
                        .get(b"cuLaunchHostFunc\0")
                        .map_err(|e| e.to_string())?,
                    _library: library,
                })
            }
        })
        .as_ref()
        .map_err(Clone::clone)
}

fn check(status: Status, operation: &str) -> Result<(), String> {
    if status == 0 {
        Ok(())
    } else {
        Err(format!("{operation} failed with CUDA status {status}"))
    }
}

/// An owned completion event recorded once on a borrowed CUDA stream.
///
/// The backend keeps the stream's context alive through this event's lifetime.
/// Dropping the event releases its handle without waiting for device work;
/// storage owners must observe completion before reusing the accessed memory.
pub struct Event {
    handle: Handle,
    context: Handle,
}

impl Event {
    /// Records a fence on the backend's current context and supplied stream.
    pub fn record(stream: usize) -> Result<Self, String> {
        let driver = driver()?;
        let mut event = Self {
            handle: std::ptr::null_mut(),
            context: std::ptr::null_mut(),
        };

        // The backend has initialized the calling thread's context and
        // submitted numerical work. This fence follows it on the same stream.
        unsafe {
            check((driver.context)(&mut event.context), "cuCtxGetCurrent")?;
            // CU_EVENT_DISABLE_TIMING: this event only tracks completion.
            check((driver.create)(&mut event.handle, 2), "cuEventCreate")?;
            check(
                (driver.record)(event.handle, stream as Handle),
                "cuEventRecord",
            )?;
        }
        Ok(event)
    }

    fn in_context<T>(
        &self,
        operation: impl FnOnce(&Driver) -> Result<T, String>,
    ) -> Result<T, String> {
        let driver = driver()?;
        unsafe {
            check((driver.push)(self.context), "cuCtxPushCurrent")?;
        }
        let result = operation(driver);
        let mut previous = std::ptr::null_mut();
        let restored = unsafe { check((driver.pop)(&mut previous), "cuCtxPopCurrent") };
        restored?;
        result
    }

    /// Queries completion without waiting; device failures remain errors.
    pub fn ready(&self) -> Result<bool, String> {
        self.in_context(|driver| {
            let status = unsafe { (driver.query)(self.handle) };
            if status == 600 {
                // CUDA_ERROR_NOT_READY
                return Ok(false);
            }
            check(status, "cuEventQuery")?;
            Ok(true)
        })
    }

    /// Waits on the host for this event alone.
    pub fn wait(&self) -> Result<(), String> {
        self.in_context(|driver| unsafe {
            check((driver.synchronize)(self.handle), "cuEventSynchronize")
        })
    }

    /// Orders a consumer stream after the work captured by this event.
    pub fn wait_on(&self, stream: usize) -> Result<(), String> {
        self.in_context(|driver| unsafe {
            check(
                (driver.wait)(stream as Handle, self.handle, 0),
                "cuStreamWaitEvent",
            )
        })
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let _ = self.in_context(|driver| unsafe {
                check((driver.destroy)(self.handle), "cuEventDestroy")
            });
        }
    }
}

struct SignalState {
    fd: OwnedFd,
    scheduled: AtomicBool,
}

/// A one-shot, selector-compatible notification of CUDA stream completion.
///
/// CUDA retains the signal until its host function runs, even if the caller
/// drops this owner. Scheduling and consumption do not acquire the interpreter.
pub struct StreamSignal {
    state: Arc<SignalState>,
}

impl StreamSignal {
    pub fn new() -> io::Result<Self> {
        // SAFETY: eventfd has no pointer arguments and returns an owned fd.
        let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }

        Ok(Self {
            state: Arc::new(SignalState {
                // SAFETY: this is the only owner of the newly created descriptor.
                fd: unsafe { OwnedFd::from_raw_fd(fd) },
                scheduled: AtomicBool::new(false),
            }),
        })
    }

    /// Schedules a single notification after preceding work on this stream.
    ///
    /// The caller supplies a live stream in its current CUDA context, outside
    /// graph capture. A rejected launch leaves the signal available for retry.
    pub fn schedule(&self, stream: usize) -> Result<(), String> {
        self.state
            .scheduled
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| "CUDA stream signal was scheduled more than once".to_owned())?;

        let state = Arc::clone(&self.state);
        let result = launch(stream, move || {
            let value = 1_u64.to_ne_bytes();
            loop {
                // SAFETY: state owns the nonblocking eventfd and value is eight
                // readable bytes. Its counter cannot saturate after one write.
                let written = unsafe {
                    libc::write(state.fd.as_raw_fd(), value.as_ptr().cast(), value.len())
                };
                if written >= 0 || io::Error::last_os_error().kind() != io::ErrorKind::Interrupted {
                    break;
                }
            }
        });

        if result.is_err() {
            self.state.scheduled.store(false, Ordering::Release);
        }
        result
    }

    /// Consumes the notification; returns WouldBlock if it has not arrived.
    pub fn consume(&self) -> io::Result<()> {
        let mut value = 0_u64;
        loop {
            // SAFETY: the state owns this eventfd and value is an eight-byte
            // writable destination as required by eventfd.
            let read = unsafe {
                libc::read(
                    self.as_raw_fd(),
                    (&mut value as *mut u64).cast(),
                    std::mem::size_of::<u64>(),
                )
            };
            if read >= 0 {
                return Ok(());
            }

            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(error);
            }
        }
    }
}

impl AsRawFd for StreamSignal {
    fn as_raw_fd(&self) -> RawFd {
        self.state.fd.as_raw_fd()
    }
}

/// Wakes a worker channel after preceding work on a live CUDA stream.
///
/// The caller must be outside graph capture: this notification transfers its
/// ownership to one host-function invocation, not to a replayable graph.
pub fn schedule_completion_wake(stream: usize, wake: Wake) -> Result<(), String> {
    launch(stream, move || wake.wake())
}

// Only native notification actions enter this helper. CUDA host functions must
// not call CUDA or acquire the GIL; either can deadlock outstanding device work.
fn launch<F: FnOnce() + Send + 'static>(stream: usize, action: F) -> Result<(), String> {
    unsafe extern "C" fn invoke<F: FnOnce()>(data: Handle) {
        // SAFETY: launch transfers one boxed action to this single invocation.
        let action = unsafe { Box::from_raw(data.cast::<F>()) };
        action();
    }

    let driver = driver()?;
    let data = Box::into_raw(Box::new(action));
    // SAFETY: the backend supplies a live stream. CUDA owns data on successful
    // submission; invoke consumes it exactly once after preceding stream work.
    let status = unsafe { (driver.launch)(stream as Handle, invoke::<F>, data.cast()) };
    if status != 0 {
        // SAFETY: a rejected launch did not transfer ownership to CUDA.
        drop(unsafe { Box::from_raw(data) });
    }

    check(status, "cuLaunchHostFunc")
}
