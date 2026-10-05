//! CUDA streams, completion events and host notifications shared by worker backends.

mod green;

use std::ffi::c_void;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use libloading::Library;
use uniserve_worker_ipc::Wake;

type Handle = *mut c_void;
type Status = i32;

const CUDA_ERROR_NOT_READY: Status = 600;
const CU_EVENT_DISABLE_TIMING: u32 = 2;
const CU_EVENT_INTERPROCESS: u32 = 4;

#[repr(C)]
struct IpcEventHandle {
    reserved: [u8; 64],
}

struct Driver {
    _library: Library,
    init: unsafe extern "C" fn(u32) -> Status,
    primary_retain: unsafe extern "C" fn(*mut Handle, i32) -> Status,
    primary_release: unsafe extern "C" fn(i32) -> Status,
    stream_context: unsafe extern "C" fn(Handle, *mut Handle) -> Status,
    stream_capture: unsafe extern "C" fn(Handle, *mut i32) -> Status,
    push: unsafe extern "C" fn(Handle) -> Status,
    pop: unsafe extern "C" fn(*mut Handle) -> Status,
    create: unsafe extern "C" fn(*mut Handle, u32) -> Status,
    record: unsafe extern "C" fn(Handle, Handle, u32) -> Status,
    query: unsafe extern "C" fn(Handle) -> Status,
    synchronize: unsafe extern "C" fn(Handle) -> Status,
    wait: unsafe extern "C" fn(Handle, Handle, u32) -> Status,
    destroy: unsafe extern "C" fn(Handle) -> Status,
    elapsed: unsafe extern "C" fn(*mut f32, Handle, Handle) -> Status,
    export_event: unsafe extern "C" fn(*mut IpcEventHandle, Handle) -> Status,
    import_event: unsafe extern "C" fn(*mut Handle, IpcEventHandle) -> Status,
    create_stream: unsafe extern "C" fn(*mut Handle, u32) -> Status,
    synchronize_stream: unsafe extern "C" fn(Handle) -> Status,
    destroy_stream: unsafe extern "C" fn(Handle) -> Status,
    stream_priority: unsafe extern "C" fn(Handle, *mut i32) -> Status,
    create_priority_stream: unsafe extern "C" fn(*mut Handle, u32, i32) -> Status,
    stream_green: unsafe extern "C" fn(Handle, *mut Handle) -> Status,
    create_green_stream: unsafe extern "C" fn(*mut Handle, Handle, u32, i32) -> Status,
    launch: unsafe extern "C" fn(Handle, unsafe extern "C" fn(Handle), Handle) -> Status,
    register_host: unsafe extern "C" fn(Handle, usize, u32) -> Status,
    unregister_host: unsafe extern "C" fn(Handle) -> Status,
    allocate_host: unsafe extern "C" fn(*mut Handle, usize, u32) -> Status,
    free_host: unsafe extern "C" fn(Handle) -> Status,
    memset: unsafe extern "C" fn(u64, u32, usize, Handle) -> Status,
}

fn driver() -> Result<&'static Driver, String> {
    static DRIVER: OnceLock<Result<Driver, String>> = OnceLock::new();
    DRIVER
        .get_or_init(|| {
            // The CUDA driver has a stable system soname. No toolkit or user
            // installation path is involved; the library owns every loaded symbol.
            unsafe {
                let library = Library::new("libcuda.so.1").map_err(|e| e.to_string())?;
                let driver = Driver {
                    allocate_host: *library
                        .get(b"cuMemHostAlloc\0")
                        .map_err(|e| e.to_string())?,
                    free_host: *library.get(b"cuMemFreeHost\0").map_err(|e| e.to_string())?,
                    memset: *library
                        .get(b"cuMemsetD32Async\0")
                        .map_err(|e| e.to_string())?,
                    register_host: *library
                        .get(b"cuMemHostRegister_v2\0")
                        .map_err(|e| e.to_string())?,
                    unregister_host: *library
                        .get(b"cuMemHostUnregister\0")
                        .map_err(|e| e.to_string())?,
                    push: *library
                        .get(b"cuCtxPushCurrent_v2\0")
                        .map_err(|e| e.to_string())?,
                    pop: *library
                        .get(b"cuCtxPopCurrent_v2\0")
                        .map_err(|e| e.to_string())?,
                    create: *library.get(b"cuEventCreate\0").map_err(|e| e.to_string())?,
                    record: *library
                        .get(b"cuEventRecordWithFlags\0")
                        .map_err(|e| e.to_string())?,
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
                    stream_context: *library
                        .get(b"cuStreamGetCtx\0")
                        .map_err(|e| e.to_string())?,
                    stream_capture: *library
                        .get(b"cuStreamIsCapturing\0")
                        .map_err(|e| e.to_string())?,
                    elapsed: *library
                        .get(b"cuEventElapsedTime\0")
                        .map_err(|e| e.to_string())?,
                    export_event: *library
                        .get(b"cuIpcGetEventHandle\0")
                        .map_err(|e| e.to_string())?,
                    import_event: *library
                        .get(b"cuIpcOpenEventHandle\0")
                        .map_err(|e| e.to_string())?,
                    create_stream: *library
                        .get(b"cuStreamCreate\0")
                        .map_err(|e| e.to_string())?,
                    synchronize_stream: *library
                        .get(b"cuStreamSynchronize\0")
                        .map_err(|e| e.to_string())?,
                    destroy_stream: *library
                        .get(b"cuStreamDestroy_v2\0")
                        .map_err(|e| e.to_string())?,
                    stream_priority: *library
                        .get(b"cuStreamGetPriority\0")
                        .map_err(|e| e.to_string())?,
                    create_priority_stream: *library
                        .get(b"cuStreamCreateWithPriority\0")
                        .map_err(|e| e.to_string())?,
                    stream_green: *library
                        .get(b"cuStreamGetGreenCtx\0")
                        .map_err(|e| e.to_string())?,
                    create_green_stream: *library
                        .get(b"cuGreenCtxStreamCreate\0")
                        .map_err(|e| e.to_string())?,
                    init: *library.get(b"cuInit\0").map_err(|e| e.to_string())?,
                    primary_retain: *library
                        .get(b"cuDevicePrimaryCtxRetain\0")
                        .map_err(|e| e.to_string())?,
                    primary_release: *library
                        .get(b"cuDevicePrimaryCtxRelease_v2\0")
                        .map_err(|e| e.to_string())?,
                    _library: library,
                };
                check((driver.init)(0), "cuInit")?;
                Ok(driver)
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

fn in_context<T>(
    context: Handle,
    operation: impl FnOnce(&Driver) -> Result<T, String>,
) -> Result<T, String> {
    let driver = driver()?;
    // SAFETY: the backend keeps this borrowed context alive.
    unsafe { check((driver.push)(context), "cuCtxPushCurrent")? };
    let result = operation(driver);
    let mut previous = std::ptr::null_mut();
    unsafe { check((driver.pop)(&mut previous), "cuCtxPopCurrent")? };
    result
}

/// An owned CUDA event, allocated on its first recording.
///
/// The backend keeps the CUDA context alive. Queries, waits and destruction
/// use the event handle directly, without changing the calling thread's context.
pub struct Event {
    device: i32,
    timing: bool,
    interprocess: bool,
    external: bool,
    handle: Mutex<Option<Arc<EventHandle>>>,
    // Output fences may outlive the stream that produced them.
    _green: Option<Arc<green::GreenContext>>,
}

struct EventHandle {
    raw: Handle,
    context: Handle,
    primary_device: Option<i32>,
}

// SAFETY: CUDA event handles can be used from multiple host threads. Arc keeps
// the event alive during each driver operation; only Drop destroys the handle.
// Context lifetime remains with the backend, and context changes are per-thread.
unsafe impl Send for EventHandle {}
unsafe impl Sync for EventHandle {}

impl Event {
    pub fn new(device: i32, timing: bool, interprocess: bool) -> Self {
        Self {
            device,
            timing,
            interprocess,
            external: false,
            handle: Mutex::new(None),
            _green: None,
        }
    }

    /// An event whose record and wait remain explicit nodes across graph
    /// segments, including work submitted between those segments.
    pub fn external(device: i32) -> Self {
        Self {
            external: true,
            ..Self::new(device, false, false)
        }
    }

    pub fn device(&self) -> i32 {
        self.device
    }

    pub fn timing(&self) -> bool {
        self.timing
    }

    pub fn interprocess(&self) -> bool {
        self.interprocess
    }

    /// Records on a borrowed stream. The caller selects the device's current
    /// context when passing a default stream; explicit streams identify theirs.
    pub fn record(&self, stream: usize) -> Result<(), String> {
        let driver = driver()?;
        let mut context = std::ptr::null_mut();
        unsafe {
            check(
                (driver.stream_context)(stream as Handle, &mut context),
                "cuStreamGetCtx",
            )?;
        }

        let mut retained = self.handle.lock().unwrap_or_else(PoisonError::into_inner);
        let event = match &mut *retained {
            Some(event) if event.context == context => event,
            slot => {
                let mut flags = 0;
                if !self.timing {
                    flags |= CU_EVENT_DISABLE_TIMING;
                }
                if self.interprocess {
                    flags |= CU_EVENT_INTERPROCESS;
                }

                let mut raw = std::ptr::null_mut();
                in_context(context, |driver| unsafe {
                    check((driver.create)(&mut raw, flags), "cuEventCreate")
                })?;
                slot.insert(Arc::new(EventHandle {
                    raw,
                    context,
                    primary_device: None,
                }))
            }
        };
        unsafe {
            check(
                (driver.record)(event.raw, stream as Handle, self.capture_flags(stream)?),
                "cuEventRecordWithFlags",
            )
        }
    }

    fn handle(&self) -> Option<Arc<EventHandle>> {
        self.handle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn capture_flags(&self, stream: usize) -> Result<u32, String> {
        if !self.external {
            return Ok(0);
        }

        // CUDA rejects EXTERNAL outside capture. As with PyTorch events,
        // eager record/wait use ordinary flags on these same stable handles.
        let mut status = 0;
        unsafe {
            check(
                (driver()?.stream_capture)(stream as Handle, &mut status),
                "cuStreamIsCapturing",
            )?;
        }
        Ok(u32::from(status != 0))
    }

    /// Queries completion without waiting; device failures remain errors.
    pub fn ready(&self) -> Result<bool, String> {
        let Some(event) = self.handle() else {
            return Ok(true);
        };
        let status = unsafe { (driver()?.query)(event.raw) };
        if status == CUDA_ERROR_NOT_READY {
            return Ok(false);
        }
        check(status, "cuEventQuery")?;
        Ok(true)
    }

    /// Waits on the host for this event alone.
    pub fn wait(&self) -> Result<(), String> {
        let Some(event) = self.handle() else {
            return Ok(());
        };
        unsafe { check((driver()?.synchronize)(event.raw), "cuEventSynchronize") }
    }

    /// Orders a consumer stream after the event's most recent recording.
    pub fn wait_on(&self, stream: usize) -> Result<(), String> {
        let Some(event) = self.handle() else {
            return Ok(());
        };
        unsafe {
            check(
                (driver()?.wait)(stream as Handle, event.raw, self.capture_flags(stream)?),
                "cuStreamWaitEvent",
            )
        }
    }

    /// Returns milliseconds between recorded timing events, without host waiting.
    pub fn elapsed_time(&self, end: &Self) -> Result<f32, String> {
        let start = self
            .handle()
            .ok_or("CUDA timing event has not been recorded")?;
        let end = end
            .handle()
            .ok_or("CUDA timing event has not been recorded")?;
        let mut milliseconds = 0.0;
        unsafe {
            check(
                (driver()?.elapsed)(&mut milliseconds, start.raw, end.raw),
                "cuEventElapsedTime",
            )?;
        }
        Ok(milliseconds)
    }

    pub fn ipc_handle(&self) -> Result<[u8; 64], String> {
        let event = self
            .handle()
            .ok_or("CUDA IPC event has not been recorded")?;
        let mut bytes = IpcEventHandle { reserved: [0; 64] };
        unsafe {
            check(
                (driver()?.export_event)(&mut bytes, event.raw),
                "cuIpcGetEventHandle",
            )?
        };
        Ok(bytes.reserved)
    }

    /// Imports an event while retaining its device's primary context.
    pub fn from_ipc_handle(device: i32, bytes: [u8; 64]) -> Result<Self, String> {
        let context = primary_context(device)?;
        let mut event = EventHandle {
            raw: std::ptr::null_mut(),
            context,
            primary_device: Some(device),
        };
        in_context(context, |driver| unsafe {
            check(
                (driver.import_event)(&mut event.raw, IpcEventHandle { reserved: bytes }),
                "cuIpcOpenEventHandle",
            )
        })?;
        Ok(Self {
            device,
            timing: false,
            interprocess: true,
            external: false,
            handle: Mutex::new(Some(Arc::new(event))),
            _green: None,
        })
    }
}

impl Drop for EventHandle {
    fn drop(&mut self) {
        if let Ok(driver) = driver() {
            // CUDA defers physical destruction if the recorded work is pending.
            if !self.raw.is_null() {
                unsafe { (driver.destroy)(self.raw) };
            }
            if let Some(device) = self.primary_device {
                unsafe { (driver.primary_release)(device) };
            }
        }
    }
}

fn primary_context(device: i32) -> Result<Handle, String> {
    let mut context = std::ptr::null_mut();
    unsafe {
        check(
            (driver()?.primary_retain)(&mut context, device),
            "cuDevicePrimaryCtxRetain",
        )?;
    }
    Ok(context)
}

/// Selects a primary CUDA context for native operations on the calling thread.
///
/// CUDA runtime device selection can leave a fresh host thread without a
/// current driver context. This guard binds it explicitly and restores the
/// previous context on drop; it does not change PyTorch's stream selection.
pub struct DeviceGuard {
    device: i32,
    // A context stack must be restored on the thread that selected it.
    _context: Handle,
}

impl DeviceGuard {
    pub fn new(device: i32) -> Result<Self, String> {
        let context = primary_context(device)?;
        let driver = driver()?;
        let status = unsafe { (driver.push)(context) };
        if status != 0 {
            unsafe { (driver.primary_release)(device) };
            check(status, "cuCtxPushCurrent")?;
        }
        Ok(Self {
            device,
            _context: context,
        })
    }
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        if let Ok(driver) = driver() {
            let mut previous = std::ptr::null_mut();
            unsafe {
                (driver.pop)(&mut previous);
                (driver.primary_release)(self.device);
            }
        }
    }
}

/// A nonblocking stream owned by worker execution or transport infrastructure.
pub struct Stream {
    handle: Handle,
    owned: bool,
    primary_device: Option<i32>,
    green: Option<Arc<green::GreenContext>>,
}

// SAFETY: the CUDA driver supports stream operations from multiple host threads.
// Shared references retain the handle; destruction requires exclusive ownership.
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    /// Creates a stream retaining the device's primary context through destruction.
    pub fn new(device: i32) -> Result<Self, String> {
        let context = primary_context(device)?;
        let mut handle = std::ptr::null_mut();
        let created = in_context(context, |driver| unsafe {
            check((driver.create_stream)(&mut handle, 1), "cuStreamCreate")
        });
        if let Err(error) = created {
            unsafe { (driver()?.primary_release)(device) };
            return Err(error);
        }

        Ok(Self {
            handle,
            owned: true,
            primary_device: Some(device),
            green: None,
        })
    }

    /// Borrows a stream whose creator retains its CUDA context and handle.
    pub fn borrowed(handle: usize) -> Self {
        Self {
            handle: handle as Handle,
            owned: false,
            primary_device: None,
            green: None,
        }
    }

    /// Creates an independent stream with the origin's context and priority.
    /// A borrowed origin's context must outlive the returned stream.
    pub fn sibling(origin: usize) -> Result<Self, String> {
        let driver = driver()?;
        let mut green = std::ptr::null_mut();
        let mut priority = 0;
        let mut handle = std::ptr::null_mut();
        unsafe {
            check(
                (driver.stream_green)(origin as Handle, &mut green),
                "cuStreamGetGreenCtx",
            )?;
            check(
                (driver.stream_priority)(origin as Handle, &mut priority),
                "cuStreamGetPriority",
            )?;
        }

        if green.is_null() {
            let mut context = std::ptr::null_mut();
            unsafe {
                check(
                    (driver.stream_context)(origin as Handle, &mut context),
                    "cuStreamGetCtx",
                )?;
            }
            in_context(context, |driver| unsafe {
                check(
                    (driver.create_priority_stream)(&mut handle, 1, priority),
                    "cuStreamCreateWithPriority",
                )
            })?;
        } else {
            unsafe {
                check(
                    (driver.create_green_stream)(&mut handle, green, 1, priority),
                    "cuGreenCtxStreamCreate",
                )?;
            }
        }

        Ok(Self {
            handle,
            owned: true,
            primary_device: None,
            green: None,
        })
    }

    /// Forks this stream while retaining its owned CUDA context or SM partition.
    pub fn fork(&self) -> Result<Self, String> {
        let mut stream = Self::sibling(self.handle())?;
        if let Some(device) = self.primary_device {
            primary_context(device)?;
            stream.primary_device = Some(device);
        }
        stream.green = self.green.clone();
        Ok(stream)
    }

    /// Allocates disjoint SM partitions and one stream per partition.
    pub fn partition(device: i32, counts: &[u32]) -> Result<Vec<Self>, String> {
        green::partition(device, counts)
    }

    pub fn sm_count(&self) -> Result<u32, String> {
        match &self.green {
            Some(green) => Ok(green.sm_count),
            None => green::stream_sms(self.handle),
        }
    }

    pub fn partitioned(&self) -> Result<bool, String> {
        let mut green = std::ptr::null_mut();
        unsafe {
            check(
                (driver()?.stream_green)(self.handle, &mut green),
                "cuStreamGetGreenCtx",
            )?;
        }
        Ok(!green.is_null())
    }

    /// Creates a fence retaining the stream's owned CUDA context.
    pub fn event(&self, device: i32) -> Event {
        Event {
            _green: self.green.clone(),
            ..Event::new(device, false, false)
        }
    }

    pub fn handle(&self) -> usize {
        self.handle as usize
    }

    pub fn wait(&self) -> Result<(), String> {
        unsafe {
            check(
                (driver()?.synchronize_stream)(self.handle),
                "cuStreamSynchronize",
            )
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if self.owned
            && let Ok(driver) = driver()
        {
            unsafe { (driver.destroy_stream)(self.handle) };
            if let Some(device) = self.primary_device {
                unsafe { (driver.primary_release)(device) };
            }
        }
    }
}

#[repr(C)]
struct CopyAttributes {
    source_order: i32,
    source_location: [i32; 2],
    destination_location: [i32; 2],
    flags: u32,
}

type CopyBatchFn = unsafe extern "C" fn(
    *mut u64,
    *mut u64,
    *mut usize,
    usize,
    *mut CopyAttributes,
    *mut usize,
    usize,
    Handle,
) -> Status;

/// A fixed batch of disjoint CUDA copies. The caller retains source and
/// destination allocations until every submission on its stream completes.
pub struct CopyBatch {
    destinations: Vec<u64>,
    sources: Vec<u64>,
    sizes: Vec<usize>,
}

impl CopyBatch {
    pub fn new(copies: impl IntoIterator<Item = (usize, usize, usize)>) -> Self {
        let mut batch = Self {
            destinations: Vec::new(),
            sources: Vec::new(),
            sizes: Vec::new(),
        };
        for (destination, source, size) in copies {
            batch.destinations.push(destination as u64);
            batch.sources.push(source as u64);
            batch.sizes.push(size);
        }
        batch
    }

    /// Submit one DMA batch with source accesses ordered on `stream`.
    /// CUDA 13's batch API is resolved only for users of batched copies.
    pub fn copy_on(&self, stream: usize) -> Result<(), String> {
        if self.sizes.is_empty() {
            return Ok(());
        }
        static COPY: OnceLock<Result<CopyBatchFn, String>> = OnceLock::new();
        let copy = COPY
            .get_or_init(|| unsafe {
                driver()?
                    ._library
                    .get::<CopyBatchFn>(b"cuMemcpyBatchAsync_v2\0")
                    .map(|symbol| *symbol)
                    .map_err(|error| error.to_string())
            })
            .as_ref()
            .map_err(Clone::clone)?;
        let mut attributes = CopyAttributes {
            source_order: 1, // CU_MEMCPY_SRC_ACCESS_ORDER_STREAM
            source_location: [0; 2],
            destination_location: [0; 2],
            flags: 0,
        };
        let mut first = 0;

        // CUDA reads these host arrays during submission; it neither writes
        // them nor retains their addresses for the asynchronous device work.
        unsafe {
            check(
                copy(
                    self.destinations.as_ptr().cast_mut(),
                    self.sources.as_ptr().cast_mut(),
                    self.sizes.as_ptr().cast_mut(),
                    self.sizes.len(),
                    &mut attributes,
                    &mut first,
                    1,
                    stream as Handle,
                ),
                "cuMemcpyBatchAsync",
            )
        }
    }
}

/// Pinned bytes used by native DMA consumers. The owner drains every copy
/// before reading, resizing or dropping this allocation.
pub struct PinnedBuffer {
    address: usize,
    bytes: usize,
}

impl PinnedBuffer {
    pub fn new(bytes: usize) -> Result<Self, String> {
        let mut address = std::ptr::null_mut();
        // Portable host storage can be used by copies in any device context.
        unsafe {
            check(
                (driver()?.allocate_host)(&mut address, bytes, 1),
                "cuMemHostAlloc",
            )?;
        }
        Ok(Self {
            address: address as usize,
            bytes,
        })
    }

    pub fn address(&self) -> usize {
        self.address
    }

    pub fn size(&self) -> usize {
        self.bytes
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        if let Ok(driver) = driver() {
            // SAFETY: this allocation is owned here and its DMA has ended.
            unsafe { (driver.free_host)(self.address as Handle) };
        }
    }
}

/// Fill aligned device words after preceding work on the borrowed stream.
/// The caller retains the writable span through stream completion.
///
/// # Safety
/// `address` must identify `words` writable, aligned u32 values in the current
/// device context, and `stream` must belong to that context.
pub unsafe fn fill_words(
    address: usize,
    value: u32,
    words: usize,
    stream: usize,
) -> Result<(), String> {
    unsafe {
        check(
            (driver()?.memset)(address as u64, value, words, stream as Handle),
            "cuMemsetD32Async",
        )
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
pub(crate) fn launch<F: FnOnce() + Send + 'static>(stream: usize, action: F) -> Result<(), String> {
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

/// Register a live host mapping for DMA. Its owner must drain all device
/// accesses before unregistering or unmapping it.
pub(crate) unsafe fn register_host(address: usize, size: usize) -> Result<(), String> {
    unsafe {
        check(
            (driver()?.register_host)(address as Handle, size, 0),
            "cuMemHostRegister",
        )
    }
}

pub(crate) unsafe fn unregister_host(address: usize) -> Result<(), String> {
    unsafe {
        check(
            (driver()?.unregister_host)(address as Handle),
            "cuMemHostUnregister",
        )
    }
}
