//! Stream ownership and device dependencies through the shared native core.

use std::sync::{Arc, Mutex};

use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::{Array, Function, Object, ObjectArc, Result};
use uniserve_worker::{
    cuda::{DeviceGuard, Event},
    CUDAStream as NativeStream,
};

use crate::execution::{failure, lock};
use crate::{method, object};

#[repr(C)]
#[derive(Object)]
#[type_key = "uniserve.ffi.CUDAStream"]
pub struct CUDAStreamObj {
    object: Object,
    stream: Mutex<NativeStream>,
}

#[derive(Clone, ObjectRef)]
pub struct CUDAStream {
    data: ObjectArc<CUDAStreamObj>,
}

impl From<NativeStream> for CUDAStream {
    fn from(stream: NativeStream) -> Self {
        Self {
            data: ObjectArc::new(CUDAStreamObj {
                object: Object::new(),
                stream: Mutex::new(stream),
            }),
        }
    }
}

#[repr(C)]
#[derive(Object)]
#[type_key = "uniserve.ffi.CUDAEvent"]
pub struct CUDAEventObj {
    object: Object,
    event: Arc<Event>,
}

#[derive(Clone, ObjectRef)]
pub struct CUDAEvent {
    data: ObjectArc<CUDAEventObj>,
}

impl CUDAEvent {
    pub(crate) fn native(&self) -> &Event {
        &self.data.event
    }
}

fn current_stream(device: i32) -> usize {
    unsafe { tvm_ffi::tvm_ffi_sys::TVMFFIEnvGetStream(2, device) as usize }
}

pub fn partition_streams(
    device: i32,
    counts: Array<i64>,
    event_slots: usize,
) -> Result<Array<CUDAStream>> {
    let counts = counts
        .iter()
        .map(u32::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| failure(error.to_string()))?;

    let streams = NativeStream::partition(device, &counts, &vec![event_slots; counts.len()])
        .map_err(failure)?;
    Ok(streams.into_iter().map(CUDAStream::from).collect())
}

pub fn register() -> Result<()> {
    object::<CUDAStreamObj>();
    object::<CUDAEventObj>();

    method::<CUDAStreamObj>(
        "__ffi_init__",
        Function::from_typed(|device: i32, handle: usize, slots: usize| {
            NativeStream::borrowed(device, handle, slots)
                .map(CUDAStream::from)
                .map_err(failure)
        }),
        "Bind execution fences to a caller-owned CUDA stream.",
    )?;
    method::<CUDAStreamObj>(
        "fork",
        Function::from_typed(|owner: CUDAStream| {
            lock(&owner.data.stream)?
                .fork()
                .map(CUDAStream::from)
                .map_err(failure)
        }),
        "Create an independent stream retaining this SM partition.",
    )?;
    method::<CUDAStreamObj>(
        "handle",
        Function::from_typed(|owner: CUDAStream| {
            lock(&owner.data.stream)?.handle().map_err(failure)
        }),
        "Borrow the CUDA handle for a numerical stream view.",
    )?;
    method::<CUDAStreamObj>(
        "sm_count",
        Function::from_typed(|owner: CUDAStream| -> Result<u32> {
            Ok(lock(&owner.data.stream)?.sm_count())
        }),
        "Return the number of SMs allocated to this stream.",
    )?;
    method::<CUDAStreamObj>(
        "wait",
        Function::from_typed(|owner: CUDAStream| {
            let stream = lock(&owner.data.stream)?;
            stream
                .wait(current_stream(stream.device()))
                .map_err(failure)
        }),
        "Order this stream after the producer selected through TVM-FFI.",
    )?;
    method::<CUDAStreamObj>(
        "record",
        Function::from_typed(|owner: CUDAStream| -> Result<Option<CUDAEvent>> {
            let mut stream = lock(&owner.data.stream)?;
            let consumer = current_stream(stream.device());
            let event = stream.record(consumer).map_err(failure)?;
            Ok(event.map(|event| CUDAEvent {
                data: ObjectArc::new(CUDAEventObj {
                    object: Object::new(),
                    event,
                }),
            }))
        }),
        "Fence this submission for a later consumer-stream join.",
    )?;
    method::<CUDAStreamObj>(
        "close",
        Function::from_typed(|owner: CUDAStream| {
            lock(&owner.data.stream)?.close(false).map_err(failure)
        }),
        "Drain submitted work and release the stream outside the GIL.",
    )?;
    method::<CUDAEventObj>(
        "query",
        Function::from_typed(|owner: CUDAEvent| owner.data.event.ready().map_err(failure)),
        "Query completion without waiting for the device.",
    )?;
    method::<CUDAEventObj>(
        "wait",
        Function::from_typed(|owner: CUDAEvent| {
            let _device = DeviceGuard::new(owner.data.event.device()).map_err(failure)?;
            owner
                .data
                .event
                .wait_on(current_stream(owner.data.event.device()))
                .map_err(failure)
        }),
        "Order the consumer selected through TVM-FFI after this event.",
    )?;
    Ok(())
}
