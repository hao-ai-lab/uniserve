//! TVM-FFI interface experiment for native worker execution.
//!
//! This module registers the cross-language objects and their public methods.
//! The production worker extension is not linked or loaded by this library.

use std::sync::{Arc, Mutex};

use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::tvm_ffi_sys::{TVMFFIByteArray, TVMFFIMethodInfo};
use tvm_ffi::{Any, Array, Bytes, Function, Object, ObjectArc, ObjectCore, Result, Tensor};
use uniserve_worker::{HostAction, Outcome};
use uniserve_worker_ipc::{codec, BatchCommand, WorkerRequest as Request};

use execution::{failure, lock};

mod cuda;
mod execution;
mod host;

#[repr(C)]
#[derive(Object)]
#[type_key = "uniserve.ffi.HostLane"]
pub struct HostLaneObj {
    object: Object,
    lane: uniserve_worker::HostLane<host::Action>,
}

#[derive(Clone, ObjectRef)]
pub struct HostLane {
    data: ObjectArc<HostLaneObj>,
}

impl HostLane {
    fn new(capacity: i64, workers: i64) -> Result<Self> {
        let capacity = usize::try_from(capacity).map_err(|error| failure(error.to_string()))?;
        let workers = usize::try_from(workers).map_err(|error| failure(error.to_string()))?;
        let lane = uniserve_worker::HostLane::new(capacity, workers, "worker-host-lane")
            .map_err(|error| failure(error.to_string()))?;
        Ok(Self {
            data: ObjectArc::new(HostLaneObj {
                object: Object::new(),
                lane,
            }),
        })
    }

    fn submit(&self, action: Function) -> Result<HostTask> {
        let task = self
            .data
            .lane
            .reserve()
            .map_err(|error| failure(error.to_string()))?;

        if let Err(error) = task
            .configure(host::Action(action))
            .and_then(|()| task.submit())
        {
            task.cancel(true).map_err(|error| error.to_ffi())?;
            return Err(failure(error.to_string()));
        }

        Ok(HostTask {
            data: ObjectArc::new(HostTaskObj {
                object: Object::new(),
                task,
            }),
        })
    }

    fn close(&self) -> Result<()> {
        let mut errors = self.data.lane.close().into_iter();
        if let Some(mut error) = errors.next() {
            for cleanup in errors {
                host::Action::note_cleanup(&mut error, cleanup);
            }
            return Err(error.to_ffi());
        }

        Ok(())
    }
}

#[repr(C)]
#[derive(Object)]
#[type_key = "uniserve.ffi.HostTask"]
pub struct HostTaskObj {
    object: Object,
    task: Arc<uniserve_worker::HostTask<host::Action>>,
}

#[derive(Clone, ObjectRef)]
pub struct HostTask {
    data: ObjectArc<HostTaskObj>,
}

impl HostTask {
    fn result(&self) -> Result<Option<Bytes>> {
        match self.data.task.completion.wait(None) {
            Some(Outcome::Success(value)) => Ok(value.as_ref().as_ref().map(Bytes::from)),
            Some(Outcome::Failed(error)) => Err(error.to_ffi()),
            Some(Outcome::Cancelled) => Err(failure("host task was cancelled")),
            None => unreachable!("an unbounded wait returns a completed outcome"),
        }
    }

    fn add_done_callback(&self, callback: Function) {
        if let Some(callback) = self
            .data
            .task
            .completion
            .subscribe((callback, self.clone()))
        {
            host::Action::notify(vec![callback]);
        }
    }
}

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

#[repr(C)]
#[derive(Object)]
#[type_key = "uniserve.ffi.RequestPool"]
pub struct RequestPoolObj {
    object: Object,
    pool: Mutex<uniserve_worker::RequestPool>,
}

#[derive(Clone, ObjectRef)]
pub struct RequestPool {
    data: ObjectArc<RequestPoolObj>,
}

impl RequestPool {
    fn new(capacity: i64) -> Result<Self> {
        let capacity = usize::try_from(capacity).map_err(|error| failure(error.to_string()))?;
        let pool = uniserve_worker::RequestPool::new(capacity)
            .map_err(|error| failure(error.to_string()))?;
        Ok(Self {
            data: ObjectArc::new(RequestPoolObj {
                object: Object::new(),
                pool: Mutex::new(pool),
            }),
        })
    }

    fn apply_commands(&self, request: &WorkerRequest) -> Result<Array<i64>> {
        let batch = request
            .data
            .request
            .batch()
            .ok_or_else(|| failure("request does not contain a batch"))?;
        let mut pool = lock(&self.data.pool)?;
        let mut admitted = Vec::new();
        for command in &batch.commands {
            match command {
                BatchCommand::Start { request } => {
                    if let Some(slot) = pool
                        .start((**request).clone())
                        .map_err(|error| failure(error.to_string()))?
                    {
                        // Slots originate in the wire's u32 request index.
                        admitted.push(slot as i64);
                    }
                }
                BatchCommand::Finish { request_key, .. } => {
                    pool.finish(*request_key)
                        .map_err(|error| failure(error.to_string()))?;
                }
                BatchCommand::Free { .. } => {}
            }
        }
        Ok(admitted.into_iter().collect())
    }

    fn retire(&self, request_id: i64) -> Result<()> {
        let request_id = u64::try_from(request_id).map_err(|error| failure(error.to_string()))?;
        lock(&self.data.pool)?
            .retire(request_id)
            .map_err(|error| failure(error.to_string()))
    }

    fn has_open_requests(&self) -> Result<bool> {
        lock(&self.data.pool)?
            .has_open_requests()
            .map_err(|error| failure(error.to_string()))
    }

    fn close(&self) -> Result<()> {
        lock(&self.data.pool)?.close();
        Ok(())
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
    object::<RequestPoolObj>();
    object::<HostLaneObj>();
    object::<HostTaskObj>();

    method::<HostLaneObj>(
        "__ffi_init__",
        Function::from_typed(HostLane::new),
        "Create bounded native threads for host callbacks.",
    )?;
    method::<HostLaneObj>(
        "submit",
        Function::from_typed(|lane: HostLane, action: Function| lane.submit(action)),
        "Submit a callback producing completed host bytes or None.",
    )?;
    method::<HostLaneObj>(
        "close",
        Function::from_typed(|lane: HostLane| lane.close()),
        "Drain host actions and release their callback references.",
    )?;
    method::<HostTaskObj>(
        "done",
        Function::from_typed(|task: HostTask| Ok(task.data.task.completion.done())),
        "Whether the task has a result, error, or cancellation.",
    )?;
    method::<HostTaskObj>(
        "result",
        Function::from_typed(|task: HostTask| task.result()),
        "Wait for the host result without holding the Python GIL.",
    )?;
    method::<HostTaskObj>(
        "cancel",
        Function::from_typed(|task: HostTask| {
            task.data.task.cancel(false).map_err(|error| error.to_ffi())
        }),
        "Remove queued work; running actions finish normally.",
    )?;
    method::<HostTaskObj>(
        "add_done_callback",
        Function::from_typed(|task: HostTask, callback: Function| {
            task.add_done_callback(callback);
            Ok(())
        }),
        "Invoke an observer with this task after completion.",
    )?;

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
    method::<RequestPoolObj>(
        "__ffi_init__",
        Function::from_typed(RequestPool::new),
        "Create the production native request pool.",
    )?;
    method::<RequestPoolObj>(
        "apply_commands",
        Function::from_typed(|pool: RequestPool, request: WorkerRequest| {
            pool.apply_commands(&request)
        }),
        "Apply the native IPC batch's request commands and return admitted slots.",
    )?;
    method::<RequestPoolObj>(
        "retire",
        Function::from_typed(|pool: RequestPool, request_id: i64| pool.retire(request_id)),
        "Retire a closed request after its physical readers drain.",
    )?;
    method::<RequestPoolObj>(
        "has_open_requests",
        Function::from_typed(|pool: RequestPool| pool.has_open_requests()),
        "Whether the pool holds requests that can accept calls.",
    )?;
    method::<RequestPoolObj>(
        "close",
        Function::from_typed(|pool: RequestPool| pool.close()),
        "Release drained request state.",
    )?;
    Ok(())
}

tvm_ffi::tvm_ffi_dll_export_typed_func!(register, register);
