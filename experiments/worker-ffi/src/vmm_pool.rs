//! Export-pool ownership and numerical views through TVM-FFI alone.

use std::sync::{Arc, Mutex};

use tvm_ffi::collections::tensor::NDAllocator;
use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::dtype::AsDLDataType;
use tvm_ffi::tvm_ffi_sys::dlpack::{DLDeviceType, DLTensor};
use tvm_ffi::{Array, Function, Object, ObjectArc, Result, Tensor, TensorView};
use uniserve_worker::vmm_pool::{PoolChunk as NativeChunk, VmmPool as NativePool, HEADER_BYTES};

use crate::descriptor_grants::DescriptorGrants;
use crate::execution::{failure, lock};
use crate::stream::CUDAEvent;
use crate::{method, object};

#[repr(C)]
#[derive(Object)]
#[type_key = "uniserve.ffi.VmmPool"]
pub struct VmmPoolObj {
    object: Object,
    pool: Mutex<NativePool<Tensor>>,
    device: i32,
}

#[derive(Clone, ObjectRef)]
pub struct VmmPool {
    data: ObjectArc<VmmPoolObj>,
}

#[repr(C)]
#[derive(Object)]
#[type_key = "uniserve.ffi.PoolChunk"]
pub struct PoolChunkObj {
    object: Object,
    chunk: Arc<NativeChunk>,
    storage: Tensor,
    address: usize,
}

#[derive(Clone, ObjectRef)]
pub struct PoolChunk {
    data: ObjectArc<PoolChunkObj>,
}

struct Payload {
    _storage: Tensor,
    address: usize,
}

unsafe impl NDAllocator for Payload {
    const MIN_ALIGN: usize = 1;

    unsafe fn alloc_data(&mut self, _: &DLTensor) -> *mut core::ffi::c_void {
        self.address as *mut _
    }

    unsafe fn free_data(&mut self, _: &DLTensor) {}
}

impl VmmPool {
    fn new(storage: Tensor) -> Result<Self> {
        let view = TensorView::from(&storage);
        if view.device().device_type != DLDeviceType::kDLCUDA
            || view.dtype() != u8::DL_DATA_TYPE
            || view.ndim() != 1
            || !view.is_contiguous()
            || view.numel() == 0
        {
            return Err(failure(
                "VMM backing must be a nonempty contiguous CUDA byte tensor",
            ));
        }
        let device = view.device().device_id;
        let address = unsafe { view.data_ptr() as usize + (*view.as_raw()).byte_offset as usize };
        if address % size_of::<u32>() != 0 {
            return Err(failure("VMM backing must align its acknowledgment words"));
        }
        let capacity = view.numel();
        let pool = unsafe { NativePool::new(storage, device, address, capacity) }
            .map_err(|error| failure(error.to_string()))?;

        Ok(Self {
            data: ObjectArc::new(VmmPoolObj {
                object: Object::new(),
                pool: Mutex::new(pool),
                device,
            }),
        })
    }

    fn reserve(&self, bytes: i64) -> Result<PoolChunk> {
        let bytes = usize::try_from(bytes).map_err(|error| failure(error.to_string()))?;
        let stream =
            unsafe { tvm_ffi::tvm_ffi_sys::TVMFFIEnvGetStream(2, self.data.device) as usize };
        let mut pool = lock(&self.data.pool)?;
        let chunk = pool
            .reserve(bytes, stream)
            .map_err(|error| failure(error.to_string()))?
            .ok_or_else(|| failure("payload does not fit the VMM pool"))?;
        let storage = pool
            .backing()
            .map_err(|error| failure(error.to_string()))?
            .clone();
        let view = TensorView::from(&storage);
        let address = unsafe { view.data_ptr() as usize + (*view.as_raw()).byte_offset as usize };

        Ok(PoolChunk {
            data: ObjectArc::new(PoolChunkObj {
                object: Object::new(),
                chunk,
                storage,
                address,
            }),
        })
    }
}

impl PoolChunk {
    fn tensor(&self, header: bool) -> Tensor {
        let (offset, bytes) = if header {
            (self.data.chunk.offset, HEADER_BYTES)
        } else {
            (
                self.data.chunk.payload_offset(),
                self.data.chunk.payload_bytes,
            )
        };

        Tensor::from_nd_alloc(
            Payload {
                _storage: self.data.storage.clone(),
                address: self.data.address + offset,
            },
            &[bytes as i64],
            u8::DL_DATA_TYPE,
            self.data.storage.device(),
        )
    }
}

pub fn register() -> Result<()> {
    object::<VmmPoolObj>();
    object::<PoolChunkObj>();
    method::<VmmPoolObj>(
        "__ffi_init__",
        Function::from_typed(VmmPool::new),
        "Bind native allocation and retirement to caller-supplied exportable CUDA storage.",
    )?;
    method::<VmmPoolObj>(
        "reserve",
        Function::from_typed(|pool: VmmPool, bytes: i64| pool.reserve(bytes)),
        "Reserve a payload and clear its reader words on the current FFI stream.",
    )?;
    method::<VmmPoolObj>(
        "release",
        Function::from_typed(|pool: VmmPool, chunk: PoolChunk| {
            lock(&pool.data.pool)?
                .release(&chunk.data.chunk)
                .map_err(|error| failure(error.to_string()))
        }),
        "Return a range after its local accesses have ended.",
    )?;
    method::<VmmPoolObj>(
        "retire",
        Function::from_typed(
            |pool: VmmPool,
             chunk: PoolChunk,
             consumers: Array<i64>,
             producer: CUDAEvent,
             grants: Option<DescriptorGrants>,
             export: tvm_ffi::String| {
                let consumers = consumers
                    .iter()
                    .map(usize::try_from)
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|error| failure(error.to_string()))?;
                let grant = grants.map(|grants| (grants.native(), export.as_str().to_owned()));
                lock(&pool.data.pool)?
                    .retire(&chunk.data.chunk, &consumers, producer.native(), grant)
                    .map_err(|error| failure(error.to_string()))
            },
        ),
        "Retain a revoked export until producer and reader accesses finish.",
    )?;
    method::<VmmPoolObj>(
        "reap",
        Function::from_typed(|pool: VmmPool| {
            lock(&pool.data.pool)?
                .reap()
                .map_err(|error| failure(error.to_string()))
        }),
        "Observe completed readback without waiting for device work.",
    )?;
    method::<VmmPoolObj>(
        "awaiting_acknowledgment",
        Function::from_typed(|pool: VmmPool| -> Result<bool> {
            Ok(lock(&pool.data.pool)?.awaiting_acknowledgment())
        }),
        "Report whether any revoked export is still retained.",
    )?;
    method::<VmmPoolObj>(
        "close",
        Function::from_typed(|pool: VmmPool| {
            let backing = lock(&pool.data.pool)?
                .close()
                .map_err(|error| failure(error.to_string()))?;
            drop(backing);
            Ok(())
        }),
        "Drain owned device accesses and release pool backing.",
    )?;
    method::<PoolChunkObj>(
        "tensor",
        Function::from_typed(|chunk: PoolChunk| -> Result<Tensor> { Ok(chunk.tensor(false)) }),
        "Borrow the payload as a DLPack byte tensor retaining its allocation.",
    )?;
    method::<PoolChunkObj>(
        "acknowledgments",
        Function::from_typed(|chunk: PoolChunk| -> Result<Tensor> { Ok(chunk.tensor(true)) }),
        "Borrow the wire header as a DLPack byte tensor.",
    )?;
    Ok(())
}
