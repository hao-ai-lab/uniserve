//! Shared host storage exposed as a DLPack tensor through TVM-FFI.

use std::sync::Mutex;
use std::time::Duration;

use tvm_ffi::collections::tensor::NDAllocator;
use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::dtype::AsDLDataType;
use tvm_ffi::tvm_ffi_sys::dlpack::{DLDevice, DLDeviceType, DLTensor};
use tvm_ffi::{Array, Function, Object, ObjectArc, Result, Tensor};
use uniserve_worker::{SharedBuffer as NativeBuffer, SharedRead as NativeRead, SHM_HEADER_BYTES};

use crate::execution::{failure, lock};
use crate::{method, object};

#[repr(C)]
#[derive(Object)]
#[type_key = "uniserve.ffi.SharedBuffer"]
pub struct SharedBufferObj {
    object: Object,
    buffer: Mutex<NativeBuffer>,
}

#[derive(Clone, ObjectRef)]
pub struct SharedBuffer {
    data: ObjectArc<SharedBufferObj>,
}

// The tensor borrows payload bytes but owns a mapping reference. In
// particular, its final DLPack release never unregisters CUDA host memory.
struct SharedView<T> {
    _owner: T,
    address: usize,
}

unsafe impl<T: 'static> NDAllocator for SharedView<T> {
    const MIN_ALIGN: usize = 1;

    unsafe fn alloc_data(&mut self, _: &DLTensor) -> *mut core::ffi::c_void {
        self.address as *mut _
    }

    unsafe fn free_data(&mut self, _: &DLTensor) {}
}

impl SharedBuffer {
    fn new(bytes: i64, consumers: Array<i64>, device: Option<i64>) -> Result<Self> {
        let bytes = usize::try_from(bytes).map_err(|error| failure(error.to_string()))?;
        let consumers = consumers
            .iter()
            .map(usize::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|error| failure(error.to_string()))?;
        let device = device
            .map(i32::try_from)
            .transpose()
            .map_err(|error| failure(error.to_string()))?;
        let device = device.map(|device| {
            let stream = unsafe { tvm_ffi::tvm_ffi_sys::TVMFFIEnvGetStream(2, device) };
            (device, stream as usize)
        });
        let buffer = NativeBuffer::new(bytes, consumers, device).map_err(failure)?;

        Ok(Self {
            data: ObjectArc::new(SharedBufferObj {
                object: Object::new(),
                buffer: Mutex::new(buffer),
            }),
        })
    }

    fn tensor(&self) -> Result<Tensor> {
        let buffer = lock(&self.data.buffer)?;
        let mapping = buffer.mapping().map_err(failure)?;
        let address = mapping.address() + SHM_HEADER_BYTES;
        Ok(Tensor::from_nd_alloc(
            SharedView {
                _owner: mapping,
                address,
            },
            &[buffer.nbytes() as i64],
            u8::DL_DATA_TYPE,
            DLDevice::new(DLDeviceType::kDLCPU, 0),
        ))
    }
}

#[repr(C)]
#[derive(Object)]
#[type_key = "uniserve.ffi.SharedRead"]
pub struct SharedReadObj {
    object: Object,
    read: Mutex<NativeRead>,
}

#[derive(Clone, ObjectRef)]
pub struct SharedRead {
    data: ObjectArc<SharedReadObj>,
}

impl SharedRead {
    fn new(
        name: tvm_ffi::String,
        bytes: i64,
        slot: i64,
        offset: i64,
        timeout: f64,
    ) -> Result<Self> {
        let bytes = usize::try_from(bytes).map_err(|error| failure(error.to_string()))?;
        let slot = usize::try_from(slot).map_err(|error| failure(error.to_string()))?;
        let offset = usize::try_from(offset).map_err(|error| failure(error.to_string()))?;
        let timeout =
            Duration::try_from_secs_f64(timeout).map_err(|error| failure(error.to_string()))?;
        let read = NativeRead::open(name.as_str(), offset, bytes, slot)
            .map_err(|error| failure(error.to_string()))?;
        read.wait(timeout, || Ok(()))
            .map_err(|error| failure(error.to_string()))?;

        Ok(Self {
            data: ObjectArc::new(SharedReadObj {
                object: Object::new(),
                read: Mutex::new(read),
            }),
        })
    }

    fn tensor(&self) -> Result<Tensor> {
        let read = lock(&self.data.read)?;
        let mapping = read.mapping().map_err(|error| failure(error.to_string()))?;
        let address = mapping.address() + read.offset();

        // Retain the read grant, not just the mapping: final tensor release
        // must acknowledge only after the consumer has finished using it.
        Ok(Tensor::from_nd_alloc(
            SharedView {
                _owner: self.clone(),
                address,
            },
            &[read.nbytes() as i64],
            u8::DL_DATA_TYPE,
            DLDevice::new(DLDeviceType::kDLCPU, 0),
        ))
    }
}

pub fn register() -> Result<()> {
    object::<SharedBufferObj>();
    method::<SharedBufferObj>(
        "__ffi_init__",
        Function::from_typed(SharedBuffer::new),
        "Allocate shared bytes, optionally registered for the current device stream.",
    )?;
    method::<SharedBufferObj>(
        "tensor",
        Function::from_typed(|buffer: SharedBuffer| buffer.tensor()),
        "Borrow payload bytes through an owning DLPack tensor reference.",
    )?;
    method::<SharedBufferObj>(
        "name",
        Function::from_typed(|buffer: SharedBuffer| -> Result<tvm_ffi::String> {
            Ok(lock(&buffer.data.buffer)?.name().map_err(failure)?.into())
        }),
        "The POSIX name used by readers to open the segment.",
    )?;
    method::<SharedBufferObj>(
        "begin_copy",
        Function::from_typed(|buffer: SharedBuffer| -> Result<()> {
            lock(&buffer.data.buffer)?.begin_copy();
            Ok(())
        }),
        "Retain possible DMA before the numerical backend submits its copy.",
    )?;
    method::<SharedBufferObj>(
        "mark_ready",
        Function::from_typed(|buffer: SharedBuffer| {
            lock(&buffer.data.buffer)?.mark_ready().map_err(failure)
        }),
        "Announce completed writes without entering Python from the CUDA callback.",
    )?;
    method::<SharedBufferObj>(
        "settled",
        Function::from_typed(|buffer: SharedBuffer| -> Result<bool> {
            Ok(lock(&buffer.data.buffer)?.settled())
        }),
        "Inspect producer completion and the granted readers' acknowledgments.",
    )?;
    method::<SharedBufferObj>(
        "close",
        Function::from_typed(|buffer: SharedBuffer| {
            lock(&buffer.data.buffer)?.close().map_err(failure)
        }),
        "Drain producer accesses, unregister host memory and unlink its name.",
    )?;

    object::<SharedReadObj>();
    method::<SharedReadObj>(
        "__ffi_init__",
        Function::from_typed(SharedRead::new),
        "Claim a shared byte range and wait for producer readiness outside Python.",
    )?;
    method::<SharedReadObj>(
        "tensor",
        Function::from_typed(|read: SharedRead| read.tensor()),
        "Borrow the payload as a DLPack tensor that retains its read grant.",
    )?;
    method::<SharedReadObj>(
        "release",
        Function::from_typed(|read: SharedRead| -> Result<()> {
            lock(&read.data.read)?.release();
            Ok(())
        }),
        "Acknowledge after the consumer's last read.",
    )?;
    Ok(())
}
