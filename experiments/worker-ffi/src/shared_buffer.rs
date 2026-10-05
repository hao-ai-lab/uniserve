//! Shared host storage exposed as a DLPack tensor through TVM-FFI.

use std::sync::{Arc, Mutex};

use tvm_ffi::collections::tensor::NDAllocator;
use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::dtype::AsDLDataType;
use tvm_ffi::tvm_ffi_sys::dlpack::{DLDevice, DLDeviceType, DLTensor};
use tvm_ffi::{Array, Function, Object, ObjectArc, Result, Tensor};
use uniserve_worker::{SharedBuffer as NativeBuffer, SharedMapping, SHM_HEADER_BYTES};

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
struct SharedView(Arc<SharedMapping>);

unsafe impl NDAllocator for SharedView {
    const MIN_ALIGN: usize = SHM_HEADER_BYTES;

    unsafe fn alloc_data(&mut self, _: &DLTensor) -> *mut core::ffi::c_void {
        (self.0.address() + SHM_HEADER_BYTES) as *mut _
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
        Ok(Tensor::from_nd_alloc(
            SharedView(mapping),
            &[buffer.nbytes() as i64],
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
    Ok(())
}
