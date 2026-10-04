//! TVM-FFI interface experiment for native batch execution.
//!
//! Only this module registers cross-language objects and converts FFI values.
//! The production worker extension is not linked or loaded by this library.

use std::sync::{Arc, Mutex};

use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::tvm_ffi_sys::{TVMFFIByteArray, TVMFFIMethodInfo};
use tvm_ffi::{Any, Bytes, Function, Object, ObjectArc, ObjectCore, Result, Tensor};
use uniserve_worker_ipc::{codec, WorkerRequest as Request};

use execution::{failure, lock};

mod cuda;
mod execution;

#[repr(C)]
#[derive(Object)]
#[type_key = "uniserve.ffi.Batch"]
pub struct BatchObj {
    object: Object,
    execution: Arc<Mutex<execution::Execution>>,
}

#[derive(Clone, ObjectRef)]
pub struct Batch {
    data: ObjectArc<BatchObj>,
}

impl Batch {
    fn new(execution: Arc<Mutex<execution::Execution>>) -> Self {
        Self {
            data: ObjectArc::new(BatchObj {
                object: Object::new(),
                execution,
            }),
        }
    }

    fn result(&self) -> Result<Tensor> {
        lock(&self.data.execution)?.result()
    }

    fn retired(&self) -> Result<bool> {
        lock(&self.data.execution)?.retired()
    }

    fn wait(&self) -> Result<()> {
        lock(&self.data.execution)?.wait()
    }

    fn cancel(&self) -> Result<()> {
        lock(&self.data.execution)?.cancel();
        Ok(())
    }
}

#[repr(C)]
#[derive(Object)]
#[type_key = "uniserve.ffi.Executor"]
pub struct ExecutorObj {
    object: Object,
    executor: execution::Executor,
}

#[derive(Clone, ObjectRef)]
pub struct Executor {
    data: ObjectArc<ExecutorObj>,
}

impl Executor {
    fn new(forward: Function, capacity: i64) -> Result<Self> {
        Ok(Self {
            data: ObjectArc::new(ExecutorObj {
                object: Object::new(),
                executor: execution::Executor::new(forward, capacity)?,
            }),
        })
    }

    fn submit(&self, input: Tensor) -> Result<Batch> {
        self.data.executor.submit(input).map(Batch::new)
    }

    fn close(&self) -> Result<()> {
        self.data.executor.close()
    }
}

#[repr(C)]
#[derive(Object)]
#[type_key = "uniserve.ffi.WorkerRequest"]
pub struct WorkerRequestObj {
    object: Object,
    request: Request,
}

#[derive(Clone, ObjectRef)]
pub struct WorkerRequest {
    data: ObjectArc<WorkerRequestObj>,
}

impl WorkerRequest {
    fn decode(bytes: Bytes) -> Result<Self> {
        let request =
            codec::decode_request(bytes.as_slice()).map_err(|e| failure(e.to_string()))?;
        Ok(Self {
            data: ObjectArc::new(WorkerRequestObj {
                object: Object::new(),
                request,
            }),
        })
    }

    fn encode(&self) -> Result<Bytes> {
        codec::encode_request(&self.data.request)
            .map(Bytes::from)
            .map_err(|e| failure(e.to_string()))
    }

    fn kind(&self) -> Result<tvm_ffi::String> {
        Ok(tvm_ffi::String::from(self.data.request.kind().as_str()))
    }

    fn batch_id(&self) -> Result<i64> {
        match &self.data.request {
            Request::Submit { batch, .. } => {
                i64::try_from(batch.batch_id).map_err(|e| failure(e.to_string()))
            }
            _ => Err(failure("request does not contain a batch")),
        }
    }
}

extern "C" {
    // The upstream Rust sys crate exposes the metadata struct but not this
    // stable C entry point. Registration copies the name, docs, and function.
    fn TVMFFITypeRegisterMethod(type_index: i32, info: *const TVMFFIMethodInfo) -> i32;
    fn TVMFFITypeGetOrAllocIndex(
        type_key: *const TVMFFIByteArray,
        static_type_index: i32,
        type_depth: i32,
        num_child_slots: i32,
        child_slots_can_overflow: i32,
        parent_type_index: i32,
    ) -> i32;
}

fn object<T: ObjectCore>() {
    // The Rust derive resolves registered types; allocation uses the shared
    // C registry. These objects are final direct descendants of ffi.Object.
    unsafe {
        let key = TVMFFIByteArray::from_str(T::TYPE_KEY);
        TVMFFITypeGetOrAllocIndex(&key, -1, T::TYPE_DEPTH, 0, 0, Object::type_index());
    }
}

fn method<T: ObjectCore>(name: &str, function: Function, doc: &str) -> Result<()> {
    let value = unsafe { Any::into_raw_ffi_any(Any::from(function)) };
    // These byte views remain alive until the registration copies them.
    let info = unsafe {
        TVMFFIMethodInfo {
            name: TVMFFIByteArray::from_str(name),
            doc: TVMFFIByteArray::from_str(doc),
            metadata: TVMFFIByteArray::from_str("{}"),
            flags: if name == "__ffi_init__" { 1 << 2 } else { 0 },
            method: value,
        }
    };
    let status = unsafe { TVMFFITypeRegisterMethod(T::type_index(), &info) };
    // Restore the owning value after the registration borrows it. The runtime
    // retained its own function reference on success.
    drop(unsafe { Any::from_raw_ffi_any(value) });
    if status != 0 {
        return Err(tvm_ffi::Error::from_raised());
    }
    Ok(())
}

fn register() -> Result<()> {
    object::<ExecutorObj>();
    object::<BatchObj>();
    object::<WorkerRequestObj>();

    method::<ExecutorObj>(
        "__ffi_init__",
        Function::from_typed(Executor::new),
        "Create a bounded numerical executor.",
    )?;
    method::<ExecutorObj>(
        "submit",
        Function::from_typed(|owner: Executor, input: Tensor| owner.submit(input)),
        "Run a numerical callback and retain its asynchronous accesses.",
    )?;
    method::<ExecutorObj>(
        "close",
        Function::from_typed(|owner: Executor| owner.close()),
        "Drain retained work and release the callback.",
    )?;

    method::<BatchObj>(
        "result",
        Function::from_typed(|batch: Batch| batch.result()),
        "Return the tensor with its producer fence on the current FFI stream.",
    )?;
    method::<BatchObj>(
        "retired",
        Function::from_typed(|batch: Batch| batch.retired()),
        "Query physical completion without a host synchronization.",
    )?;
    method::<BatchObj>(
        "wait",
        Function::from_typed(|batch: Batch| batch.wait()),
        "Wait for physical completion without calling Python.",
    )?;
    method::<BatchObj>(
        "cancel",
        Function::from_typed(|batch: Batch| batch.cancel()),
        "Revoke the result while retaining in-flight storage.",
    )?;

    method::<WorkerRequestObj>(
        "__ffi_init__",
        Function::from_typed(WorkerRequest::decode),
        "Decode the existing worker wire format into Rust state.",
    )?;
    method::<WorkerRequestObj>(
        "encode",
        Function::from_typed(|request: WorkerRequest| request.encode()),
        "Serialize the native request using the shared worker codec.",
    )?;
    method::<WorkerRequestObj>(
        "kind",
        Function::from_typed(|request: WorkerRequest| request.kind()),
        "Return the worker request kind.",
    )?;
    method::<WorkerRequestObj>(
        "batch_id",
        Function::from_typed(|request: WorkerRequest| request.batch_id()),
        "Read the batch identity from native request state.",
    )?;
    Ok(())
}

tvm_ffi::tvm_ffi_dll_export_typed_func!(register, register);
