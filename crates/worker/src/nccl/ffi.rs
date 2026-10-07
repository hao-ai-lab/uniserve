//! NCCL's C ABI. The runtime loader resolves its installed soname.

use std::ffi::{CStr, c_char, c_void};
use std::sync::OnceLock;

use libloading::Library;

pub(super) type Handle = *mut c_void;
type Status = i32;

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct UniqueId {
    pub bytes: [u8; 128],
}

/// ncclConfig_t from the pinned NCCL 2.30 headers. Unspecified fields use
/// NCCL_CONFIG_UNDEF_INT, preserving NCCL's environment and topology tuning.
#[repr(C)]
pub(super) struct Config {
    size: usize,
    magic: u32,
    version: u32,
    blocking: i32,
    cga_cluster_size: i32,
    min_ctas: i32,
    max_ctas: i32,
    net_name: *const c_char,
    split_share: i32,
    traffic_class: i32,
    comm_name: *const c_char,
    collnet_enable: i32,
    cta_policy: i32,
    shrink_share: i32,
    nvls_ctas: i32,
    channels_per_net_peer: i32,
    nvlink_centric_sched: i32,
    graph_usage_mode: i32,
    rma_contexts: i32,
    max_p2p_peers: i32,
}

impl Config {
    pub(super) fn new(full_device: bool) -> Self {
        Self {
            size: std::mem::size_of::<Self>(),
            magic: 0xcafebeef,
            version: 23004,
            blocking: i32::MIN,
            cga_cluster_size: i32::MIN,
            min_ctas: i32::MIN,
            max_ctas: i32::MIN,
            net_name: std::ptr::null(),
            split_share: i32::MIN,
            traffic_class: i32::MIN,
            comm_name: std::ptr::null(),
            collnet_enable: i32::MIN,
            // Green Contexts cannot batch-copy mapped peer windows on the
            // deployed driver. Their collectives use NCCL's CTA algorithms.
            cta_policy: if full_device { 0x02 } else { 0 },
            shrink_share: i32::MIN,
            nvls_ctas: i32::MIN,
            channels_per_net_peer: i32::MIN,
            nvlink_centric_sched: i32::MIN,
            graph_usage_mode: i32::MIN,
            rma_contexts: i32::MIN,
            max_p2p_peers: i32::MIN,
        }
    }
}

type Collective = unsafe extern "C" fn(Handle, Handle, usize, i32, Handle, Handle) -> Status;
type Reduction = unsafe extern "C" fn(Handle, Handle, usize, i32, i32, Handle, Handle) -> Status;
type PointToPoint = unsafe extern "C" fn(Handle, usize, i32, i32, Handle, Handle) -> Status;

pub(super) struct LibraryApi {
    _library: Library,
    error: unsafe extern "C" fn(Status) -> *const c_char,
    pub unique_id: unsafe extern "C" fn(*mut UniqueId) -> Status,
    pub init: unsafe extern "C" fn(*mut Handle, i32, UniqueId, i32, *mut Config) -> Status,
    pub destroy: unsafe extern "C" fn(Handle) -> Status,
    pub abort: unsafe extern "C" fn(Handle) -> Status,
    pub register: unsafe extern "C" fn(Handle, Handle, usize, *mut Handle, i32) -> Status,
    pub deregister: unsafe extern "C" fn(Handle, Handle) -> Status,
    pub all_reduce: Reduction,
    pub reduce_scatter: Reduction,
    pub all_gather: Collective,
    pub all_to_all: Collective,
    pub broadcast: Reduction,
    pub send: PointToPoint,
    pub recv: PointToPoint,
    pub group_start: unsafe extern "C" fn() -> Status,
    pub group_end: unsafe extern "C" fn() -> Status,
}

pub(super) fn api() -> Result<&'static LibraryApi, String> {
    static API: OnceLock<Result<LibraryApi, String>> = OnceLock::new();
    API.get_or_init(|| unsafe {
        // The numerical backend loads its installed NCCL before constructing
        // streams. Reopen that soname without guessing package or user paths.
        let library = Library::new("libnccl.so.2").map_err(|error| error.to_string())?;
        Ok(LibraryApi {
            error: *library
                .get(b"ncclGetErrorString\0")
                .map_err(|error| error.to_string())?,
            unique_id: *library
                .get(b"ncclGetUniqueId\0")
                .map_err(|error| error.to_string())?,
            init: *library
                .get(b"ncclCommInitRankConfig\0")
                .map_err(|error| error.to_string())?,
            destroy: *library
                .get(b"ncclCommDestroy\0")
                .map_err(|error| error.to_string())?,
            abort: *library
                .get(b"ncclCommAbort\0")
                .map_err(|error| error.to_string())?,
            register: *library
                .get(b"ncclCommWindowRegister\0")
                .map_err(|error| error.to_string())?,
            deregister: *library
                .get(b"ncclCommWindowDeregister\0")
                .map_err(|error| error.to_string())?,
            all_reduce: *library
                .get(b"ncclAllReduce\0")
                .map_err(|error| error.to_string())?,
            reduce_scatter: *library
                .get(b"ncclReduceScatter\0")
                .map_err(|error| error.to_string())?,
            all_gather: *library
                .get(b"ncclAllGather\0")
                .map_err(|error| error.to_string())?,
            all_to_all: *library
                .get(b"ncclAlltoAll\0")
                .map_err(|error| error.to_string())?,
            broadcast: *library
                .get(b"ncclBroadcast\0")
                .map_err(|error| error.to_string())?,
            send: *library
                .get(b"ncclSend\0")
                .map_err(|error| error.to_string())?,
            recv: *library
                .get(b"ncclRecv\0")
                .map_err(|error| error.to_string())?,
            group_start: *library
                .get(b"ncclGroupStart\0")
                .map_err(|error| error.to_string())?,
            group_end: *library
                .get(b"ncclGroupEnd\0")
                .map_err(|error| error.to_string())?,
            _library: library,
        })
    })
    .as_ref()
    .map_err(Clone::clone)
}

pub(super) fn check(status: Status, operation: &str) -> Result<(), String> {
    if status == 0 {
        return Ok(());
    }
    // NCCL returns a static, terminated description for every result code.
    let message = unsafe { CStr::from_ptr((api()?.error)(status)) }.to_string_lossy();
    Err(format!("{operation} failed: {message}"))
}

pub(super) fn grouped<T>(call: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    unsafe {
        check((api()?.group_start)(), "ncclGroupStart")?;
    }
    let result = call();
    let ended = unsafe { check((api()?.group_end)(), "ncclGroupEnd") };
    ended?;
    result
}
