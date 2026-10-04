//! Native completion fences for tensors borrowed by the numerical callback.

use std::ffi::c_void;
use std::sync::OnceLock;

use libloading::Library;

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

pub struct Event {
    handle: Handle,
    context: Handle,
}

impl Event {
    pub fn record(stream: Handle) -> Result<Self, String> {
        let driver = driver()?;
        let mut event = Self {
            handle: std::ptr::null_mut(),
            context: std::ptr::null_mut(),
        };

        // PyTorch has initialized the calling thread's context and submitted
        // numerical work. This fence follows that work on the same stream.
        unsafe {
            check((driver.context)(&mut event.context), "cuCtxGetCurrent")?;
            check((driver.create)(&mut event.handle, 2), "cuEventCreate")?;
            check((driver.record)(event.handle, stream), "cuEventRecord")?;
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

    pub fn wait(&self) -> Result<(), String> {
        self.in_context(|driver| unsafe {
            check((driver.synchronize)(self.handle), "cuEventSynchronize")
        })
    }

    pub fn wait_on(&self, stream: Handle) -> Result<(), String> {
        self.in_context(|driver| unsafe {
            check((driver.wait)(stream, self.handle, 0), "cuStreamWaitEvent")
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
