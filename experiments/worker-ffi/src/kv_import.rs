//! KV copy admission and workspace retirement through the production importer.

use std::sync::Arc;

use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::{Function, Object, ObjectArc, Result};
use uniserve_worker::cuda::{DeviceGuard, Stream};
use uniserve_worker::{
    Completion, HostAction, HostLane, HostTask, ImportBackend, ImportCopy,
    KVImport as NativeImport, KVImporter as NativeImporter, Outcome,
};

use crate::execution::failure;
use crate::host::{Action, CallbackError};
use crate::{method, object, Request, WorkerRequest};

type Read = Arc<Completion<CallbackError, Function>>;
type Import = NativeImport<Read, Workspace>;
type Importer = NativeImporter<Arc<Import>, Workspace>;
type Task = HostTask<ImportCopy<Copy>>;

struct Workspace {
    index: usize,
    device: Option<i32>,
    stream: Option<Stream>,
}

struct Copy {
    reset: Function,
    copy: Function,
}

impl ImportBackend for Copy {
    type Error = CallbackError;
    type Callback = Function;
    type Read = Read;
    type Workspace = Workspace;
    type Import = Arc<Import>;

    fn reset(&self, workspace: &Workspace) -> std::result::Result<(), CallbackError> {
        workspace.call(&self.reset)
    }

    fn copy(&self, workspace: &Workspace) -> std::result::Result<(), CallbackError> {
        workspace.call(&self.copy)
    }

    fn drain(&self, workspace: &Workspace) -> std::result::Result<(), CallbackError> {
        if let Some(stream) = &workspace.stream {
            let _device = workspace
                .device
                .map(DeviceGuard::new)
                .transpose()
                .map_err(failure)?;
            stream.wait().map_err(failure)?;
        }
        Ok(())
    }

    fn retired(&self, _: Vec<Arc<Import>>) -> std::result::Result<(), CallbackError> {
        Ok(())
    }

    fn error(error: uniserve_worker::Error) -> CallbackError {
        error.into()
    }

    fn report(error: CallbackError) {
        Action::report(error);
    }

    fn note_cleanup(error: &mut CallbackError, cleanup: CallbackError) {
        Action::note_cleanup(error, cleanup);
    }
}

impl Workspace {
    fn call(&self, callback: &Function) -> std::result::Result<(), CallbackError> {
        let _device = self
            .device
            .map(DeviceGuard::new)
            .transpose()
            .map_err(failure)?;
        let stream = self.stream.as_ref().map_or(0, Stream::handle);
        callback.call_tuple((self.index, stream))?;
        Ok(())
    }
}

#[repr(C)]
#[derive(Object)]
#[type_key = "uniserve.ffi.KVImporter"]
pub struct KVImporterObj {
    object: Object,
    importer: Arc<Importer>,
    tasks: HostLane<ImportCopy<Copy>>,
}

#[derive(Clone, ObjectRef)]
pub struct KVImporter {
    data: ObjectArc<KVImporterObj>,
}

impl KVImporter {
    fn new(capacity: usize, workers: usize, device: Option<i32>) -> Result<Self> {
        let tasks = HostLane::new(capacity, workers, "ffi-kv-import")
            .map_err(|error| failure(error.to_string()))?;
        let workspaces = (0..workers)
            .map(|index| {
                Ok(Arc::new(Workspace {
                    index,
                    device,
                    stream: device.map(Stream::new).transpose().map_err(failure)?,
                }))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            data: ObjectArc::new(KVImporterObj {
                object: Object::new(),
                importer: Arc::new(Importer::new(workspaces)),
                tasks,
            }),
        })
    }

    fn reserve(
        &self,
        request: &WorkerRequest,
        input: usize,
        request_pool_idx: usize,
        reset: Function,
        copy: Function,
    ) -> Result<KVImport> {
        let Request::Submit { batch, .. } = &request.data.request else {
            return Err(failure("KV import requires a batch submission"));
        };
        let export = batch
            .kv_inputs
            .get(input)
            .ok_or_else(|| failure("KV input index is outside the batch"))?;
        let write = Arc::new(Import::new(export.source, request_pool_idx, true));
        let task = self
            .data
            .tasks
            .reserve()
            .map_err(|error| failure(error.to_string()))?;

        let submitted = (|| {
            task.configure(ImportCopy::new(
                Arc::clone(&self.data.importer),
                Arc::clone(&write),
                Copy { reset, copy },
            ))?;
            self.data.importer.reserve(Arc::clone(&write), || Ok(()))?;
            task.submit()
        })();
        if let Err(error) = submitted {
            if let Err(cleanup) = task.cancel(true) {
                Copy::report(cleanup);
            }
            self.data.importer.abandon(&write);
            self.data.importer.reap();
            return Err(failure(error.to_string()));
        }

        Ok(KVImport {
            data: ObjectArc::new(KVImportObj {
                object: Object::new(),
                importer: Arc::clone(&self.data.importer),
                write,
                task,
            }),
        })
    }

    fn adopt(&self, write: &KVImport) -> Result<()> {
        if !write.data.task.completion.done() {
            return Err(failure("KV import was observed before input readiness"));
        }
        write.result()?;
        self.data
            .importer
            .adopt(&write.data.write)
            .map_err(|error| failure(error.to_string()))?;
        self.data.importer.reap();
        Ok(())
    }

    fn abandon(&self, write: &KVImport) {
        self.data.importer.abandon(&write.data.write);
        self.data.importer.reap();
    }
}

impl KVImporterObj {
    fn stop(&self) -> Result<()> {
        self.importer.stop();
        let mut errors = self.tasks.close().into_iter();
        self.importer.reap();
        if let Some(mut error) = errors.next() {
            for cleanup in errors {
                Copy::note_cleanup(&mut error, cleanup);
            }
            return Err(error.to_ffi());
        }
        Ok(())
    }
}

impl Drop for KVImporterObj {
    fn drop(&mut self) {
        if let Err(error) = self.stop() {
            Copy::report(error.into());
        }
    }
}

#[repr(C)]
#[derive(Object)]
#[type_key = "uniserve.ffi.KVImport"]
pub struct KVImportObj {
    object: Object,
    importer: Arc<Importer>,
    write: Arc<Import>,
    task: Arc<Task>,
}

#[derive(Clone, ObjectRef)]
pub struct KVImport {
    data: ObjectArc<KVImportObj>,
}

impl KVImport {
    fn result(&self) -> Result<()> {
        match self.data.task.completion.wait(None) {
            Some(Outcome::Success(_)) => Ok(()),
            Some(Outcome::Failed(error)) => Err(error.to_ffi()),
            Some(Outcome::Cancelled) => Err(failure("KV copy was cancelled")),
            None => unreachable!("unbounded completion wait"),
        }
    }
}

pub fn register() -> Result<()> {
    object::<KVImporterObj>();
    object::<KVImportObj>();
    method::<KVImporterObj>(
        "__ffi_init__",
        Function::from_typed(KVImporter::new),
        "Own bounded KV copy workspaces on CPU or one CUDA device.",
    )?;
    method::<KVImporterObj>(
        "reserve",
        Function::from_typed(
            |pool: KVImporter,
             request: WorkerRequest,
             input: usize,
             slot: usize,
             reset: Function,
             copy: Function| { pool.reserve(&request, input, slot, reset, copy) },
        ),
        "Reserve a decoded KV input; callbacks receive workspace index and native stream.",
    )?;
    method::<KVImporterObj>(
        "stop",
        Function::from_typed(|pool: KVImporter| pool.data.stop()),
        "Cancel imports and join their physical copies outside the GIL.",
    )?;
    method::<KVImportObj>(
        "done",
        Function::from_typed(|write: KVImport| Ok(write.data.task.completion.done())),
        "Query copy completion without blocking.",
    )?;
    method::<KVImportObj>(
        "result",
        Function::from_typed(|write: KVImport| write.result()),
        "Wait for copy completion and propagate its numerical error.",
    )?;
    method::<KVImportObj>(
        "cancel",
        Function::from_typed(|write: KVImport| {
            write
                .data
                .task
                .cancel(false)
                .map_err(|error| error.to_ffi())
        }),
        "Withdraw a queued copy; running copies finish normally.",
    )?;
    method::<KVImporterObj>(
        "adopt",
        Function::from_typed(|pool: KVImporter, write: KVImport| pool.adopt(&write)),
        "Commit a successful copy and release the import reservation.",
    )?;
    method::<KVImporterObj>(
        "abandon",
        Function::from_typed(|pool: KVImporter, write: KVImport| {
            pool.abandon(&write);
            Ok(())
        }),
        "Discard an import while retaining any outstanding device accesses.",
    )?;
    method::<KVImportObj>(
        "retired",
        Function::from_typed(|write: KVImport| Ok(!write.data.importer.owns(&write.data.write))),
        "Query whether the destination reservation and workspace have retired.",
    )?;
    Ok(())
}
