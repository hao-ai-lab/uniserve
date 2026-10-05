//! Numerical KV copies on native import lanes and bounded conversion storage.

use std::collections::{HashMap, HashSet};
use std::ops::Deref;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use uniserve_worker::cuda::{DeviceGuard, Stream};
use uniserve_worker::{
    HostLane as NativeLane, HostTask as NativeTask, ImportBackend, ImportCopy,
    KVImport as NativeImport, KVImporter as NativeImporter, Outcome,
};

use super::block_tables::GroupTable;
use super::completion::{Completion, CompletionRef, cancelled};
use super::error::{invariant, native_error};
use super::host::with_context;
use super::kv_cache::KVCacheManager;
use super::protocol::{buffer_id, request_key};
use super::transfer::{TransferRef, TransferTicket};

struct Workspace {
    values: Py<PyAny>,
    stream: Option<Stream>,
    torch_stream: Py<PyAny>,
}

impl Workspace {
    fn visit(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.values)?;
        visit.call(&self.torch_stream)
    }
}

type CopyTask = NativeTask<ImportCopy<Copy>>;

#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct KVImport {
    inner: Arc<NativeImport<TransferRef, Workspace>>,
    #[pyo3(get)]
    tables: Py<PyTuple>,
    #[pyo3(get)]
    initialized_units: Py<PyTuple>,
    #[pyo3(get)]
    export: Py<PyAny>,
    task: Option<Arc<CopyTask>>,
    #[pyo3(get)]
    retirement: Py<Completion>,
}

#[pymethods]
impl KVImport {
    #[getter]
    fn request_pool_idx(&self) -> usize {
        self.inner.request_pool_idx
    }

    #[getter]
    fn cancelled(&self) -> bool {
        self.inner.cancelled()
            || self
                .task
                .as_ref()
                .is_some_and(|task| matches!(task.completion.outcome(), Some(Outcome::Cancelled)))
    }

    #[getter]
    pub(crate) fn released(&self) -> bool {
        self.inner.released()
    }

    pub(crate) fn done(&self) -> bool {
        self.task.as_ref().is_none_or(|task| task.completion.done())
    }

    #[pyo3(signature = (timeout=None))]
    pub(crate) fn result(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<()> {
        let Some(task) = self.task.as_ref() else {
            return Ok(());
        };
        let timeout = timeout
            .map(|seconds| Duration::try_from_secs_f64(seconds.max(0.0)))
            .transpose()
            .map_err(|_| PyValueError::new_err("timeout must be finite"))?;
        let outcome = task
            .completion
            .outcome()
            .or_else(|| py.detach(|| task.completion.wait(timeout)))
            .ok_or_else(|| PyTimeoutError::new_err("KV import has not completed"))?;

        match outcome {
            Outcome::Success(_) => Ok(()),
            Outcome::Failed(error) => Err(PyErr::from_value(error.bind(py).clone().into_any())),
            Outcome::Cancelled => Err(cancelled(py)?),
        }
    }

    fn cancel(&self, py: Python<'_>) -> PyResult<bool> {
        self.task.as_ref().map_or(Ok(false), |task| {
            task.cancel(false)
                .map_err(|error| PyErr::from_value(error.bind(py).clone().into_any()))
        })
    }

    pub(crate) fn add_done_callback(slf: &Bound<'_, Self>, callback: Py<PyAny>) {
        let owner = slf.clone().unbind();
        let notify: Box<dyn FnOnce() + Send> = Box::new(move || {
            Python::attach(|py| {
                if let Err(error) = callback.bind(py).call1((owner,)) {
                    error.write_unraisable(py, None);
                }
            });
        });
        let immediate = match &slf.get().task {
            Some(task) => task.completion.subscribe(notify),
            None => Some(notify),
        };
        if let Some(notify) = immediate {
            notify();
        }
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.tables)?;
        visit.call(&self.initialized_units)?;
        visit.call(&self.export)?;
        visit.call(&self.retirement)?;
        if let Some(task) = self
            .task
            .as_ref()
            .filter(|task| Arc::strong_count(task) == 1)
        {
            task.completion.visit(|outcome, _| {
                if let Some(Outcome::Failed(error)) = outcome {
                    visit.call(error.as_ref())?;
                }
                Ok(())
            })?;
        }
        self.inner.visit(|reads, workspace| {
            for read in reads {
                visit.call(&read.owner)?;
            }
            if let Some(workspace) = workspace {
                workspace.visit(&visit)?;
            }
            Ok(())
        })
    }
}

struct ImportRef {
    owner: Py<KVImport>,
    inner: Arc<NativeImport<TransferRef, Workspace>>,
}

impl Deref for ImportRef {
    type Target = NativeImport<TransferRef, Workspace>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct KVImporter {
    inner: Arc<NativeImporter<ImportRef, Workspace>>,
    pool: Py<PyAny>,
    accesses: Py<KVCacheManager>,
    tasks: NativeLane<ImportCopy<Copy>>,
    wake: Mutex<Option<Py<PyAny>>>,
}

#[pymethods]
impl KVImporter {
    #[new]
    #[pyo3(signature = (pool, *, capacity))]
    fn new(py: Python<'_>, pool: Py<PyAny>, capacity: usize) -> PyResult<Self> {
        let workers = capacity.min(4);
        let tasks = NativeLane::new(capacity, workers, "worker-kv-import")
            .map_err(|error| native_error(py, error))?;
        let device = pool.bind(py).getattr("cache")?.getattr("device")?;
        let device = py
            .import("uniserve.runtime.device")?
            .call_method1("canonical_device", (device,))?;
        let cuda = device.getattr("type")?.extract::<String>()? == "cuda";
        let mut streams = Vec::new();
        let mut views = Vec::new();
        for _ in 0..workers {
            if cuda {
                let _device = DeviceGuard::new(device.getattr("index")?.extract()?)
                    .map_err(PyRuntimeError::new_err)?;
                let stream = Stream::new(device.getattr("index")?.extract()?)
                    .map_err(PyRuntimeError::new_err)?;
                let kwargs = PyDict::new(py);
                kwargs.set_item("device", &device)?;
                let view = py
                    .import("torch.cuda")?
                    .getattr("ExternalStream")?
                    .call((stream.handle(),), Some(&kwargs))?
                    .unbind();
                streams.push(Some(stream));
                views.push(view);
            } else {
                streams.push(None);
                views.push(py.None());
            }
        }

        let values: Vec<Py<PyAny>> = py
            .import("uniserve_worker.storage.cache_imports")?
            .call_method1("_allocate", (pool.bind(py), workers))?
            .extract()?;
        let workspaces = values
            .into_iter()
            .zip(streams)
            .zip(views)
            .map(|((values, stream), torch_stream)| {
                Arc::new(Workspace {
                    values,
                    stream,
                    torch_stream,
                })
            })
            .collect();
        let accesses = pool.bind(py).getattr("_manager")?.extract()?;
        Ok(Self {
            inner: Arc::new(NativeImporter::new(workspaces)),
            pool,
            accesses,
            tasks,
            wake: Mutex::new(None),
        })
    }

    fn set_completion_wake(&self, py: Python<'_>, wake: Option<Py<PyAny>>) {
        self.tasks.set_wake(wake.as_ref().map(|wake| {
            let wake = wake.clone_ref(py);
            Box::new(move || {
                Python::attach(|py| {
                    if let Err(error) = wake.bind(py).call0() {
                        error.write_unraisable(py, None);
                    }
                });
            }) as Box<dyn Fn() + Send + Sync>
        }));
        let previous = std::mem::replace(
            &mut *self.wake.lock().unwrap_or_else(PoisonError::into_inner),
            wake,
        );
        drop(previous);
    }

    #[pyo3(signature = (export, *, request_pool_idx, tables, initialized_units, transports))]
    fn reserve(
        slf: &Bound<'_, Self>,
        export: Py<PyAny>,
        request_pool_idx: usize,
        tables: Py<PyTuple>,
        initialized_units: Py<PyTuple>,
        transports: Py<PyAny>,
    ) -> PyResult<Py<KVImport>> {
        let py = slf.py();
        let this = slf.borrow();
        let source = buffer_id(&export.bind(py).getattr("source")?)?;
        let extent: u64 = export.bind(py).getattr("exported_extent")?.extract()?;
        let groups = export.bind(py).getattr("groups")?;
        let mut ranges = HashMap::new();
        for (table, group) in tables.bind(py).iter().zip(groups.try_iter()?) {
            let start: u64 = group?.getattr("start")?.extract()?;
            let table = table.extract::<PyRef<'_, GroupTable>>()?;
            let count = extent.checked_sub(start).ok_or_else(|| {
                super::error::invalid(py, "KV import starts beyond its exported extent")
            })?;
            for (unit, offset, count) in table
                .table
                .spans(start, count)
                .map_err(|error| native_error(py, error))?
            {
                ranges.insert(unit, (offset, count));
            }
        }
        let reset: Vec<(u32, u32, u32)> = this
            .pool
            .bind(py)
            .call_method1("unit_spans", (initialized_units.bind(py),))?
            .extract()?;
        ranges.extend(
            reset
                .into_iter()
                .map(|(unit, offset, count)| (unit, (offset, count))),
        );
        let ranges: Vec<_> = ranges
            .into_iter()
            .map(|(unit, (offset, count))| (unit, offset, count))
            .collect();
        let copy = !initialized_units.bind(py).is_empty()
            || export.bind(py).getattr("tensors")?.is_truthy()?;

        let retirement = Py::new(py, Completion::new())?;
        let task = copy
            .then(|| this.tasks.reserve())
            .transpose()
            .map_err(|error| native_error(py, error))?;
        let write = Py::new(
            py,
            KVImport {
                inner: Arc::new(NativeImport::new(source, request_pool_idx, copy)),
                tables,
                initialized_units,
                export,
                task: task.clone(),
                retirement,
            },
        );
        let write = match write {
            Ok(write) => write,
            Err(error) => {
                if let Some(task) = task
                    && let Err(cleanup) = task.cancel(true)
                {
                    Copy::report(cleanup);
                }
                return Err(error);
            }
        };
        if let Some(task) = &task {
            let configured = task.configure(ImportCopy::new(
                Arc::clone(&this.inner),
                Arc::clone(&write.get().inner),
                Copy {
                    owner: slf.clone().unbind(),
                    write: write.clone_ref(py),
                    transports,
                },
            ));
            if let Err(error) = configured {
                task.cancel(true)
                    .map_err(|cause| PyErr::from_value(cause.bind(py).clone().into_any()))?;
                return Err(native_error(py, error));
            }
        }

        let retirement = CompletionRef::new(py, write.get().retirement.clone_ref(py));
        let reserved = {
            let mut accesses = this.accesses.borrow_mut(py);
            this.inner.reserve(
                ImportRef {
                    owner: write.clone_ref(py),
                    inner: Arc::clone(&write.get().inner),
                },
                || accesses.inner.reserve_import(source, &ranges, retirement),
            )
        };
        if let Err(error) = reserved {
            if let Some(task) = &task {
                task.cancel(true)
                    .map_err(|cause| PyErr::from_value(cause.bind(py).clone().into_any()))?;
            }
            return Err(native_error(py, error));
        }

        if let Some(task) = task
            && let Err(error) = task.submit()
        {
            task.cancel(true)
                .map_err(|cause| PyErr::from_value(cause.bind(py).clone().into_any()))?;
            this.abandon(py, write.get())?;
            return Err(native_error(py, error));
        }
        Ok(write)
    }

    fn owns(&self, write: &KVImport) -> bool {
        self.inner.owns(&write.inner)
    }

    fn adopt(&self, py: Python<'_>, write: &KVImport) -> PyResult<()> {
        if !write.done() {
            return Err(PyRuntimeError::new_err(
                "KV import was observed before input readiness",
            ));
        }
        write.result(py, None)?;
        self.inner
            .adopt(&write.inner)
            .map_err(|error| native_error(py, error))?;
        self._reap(py)
    }

    pub(crate) fn abandon(&self, py: Python<'_>, write: &KVImport) -> PyResult<()> {
        let cancelled = cancel_reads(py, self.inner.abandon(&write.inner));
        let reaped = self._reap(py);
        cancelled.and(reaped)
    }

    fn release(&self, py: Python<'_>, buffers: &Bound<'_, PyAny>) -> PyResult<()> {
        let buffers = buffers
            .try_iter()?
            .map(|buffer| buffer_id(&buffer?))
            .collect::<PyResult<HashSet<_>>>()?;
        self.release_buffers(py, &buffers)
    }

    #[pyo3(signature = (requests, *, retained=None))]
    fn cancel_requests(
        &self,
        py: Python<'_>,
        requests: &Bound<'_, PyAny>,
        retained: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let requests = requests
            .try_iter()?
            .map(|key| request_key(&key?))
            .collect::<PyResult<HashSet<_>>>()?;
        let retained = retained
            .map(|buffers| {
                buffers
                    .try_iter()?
                    .map(|buffer| buffer_id(&buffer?))
                    .collect::<PyResult<HashSet<_>>>()
            })
            .transpose()?
            .unwrap_or_default();
        self.cancel_request_imports(py, &requests, &retained)
    }

    /// Revoke reads before joining the lane. Callbacks retire tasks cancelled
    /// before their numerical action began, as well as completed copies.
    fn stop(&self, py: Python<'_>) -> PyResult<()> {
        let cancelled = cancel_reads(py, self.inner.stop());
        let mut errors = py.detach(|| self.tasks.close()).into_iter();
        let stopped = errors.next().map_or(Ok(()), |mut error| {
            for cleanup in errors {
                Copy::note_cleanup(&mut error, cleanup);
            }
            Err(PyErr::from_value(error.bind(py).clone().into_any()))
        });
        let reaped = self._reap(py);
        cancelled.and(stopped).and(reaped)
    }

    fn require_retired(&self, py: Python<'_>) -> PyResult<()> {
        self._reap(py)?;
        self.inner
            .require_retired()
            .map_err(|error| native_error(py, error))
    }

    fn _require_active(&self, py: Python<'_>, write: &KVImport) -> PyResult<()> {
        self.inner
            .require_active(&write.inner)
            .map_err(|error| native_error(py, error))
    }

    fn _retain(
        slf: &Bound<'_, Self>,
        write: &KVImport,
        ticket: Py<TransferTicket>,
    ) -> PyResult<()> {
        let py = slf.py();
        let read = Arc::new(TransferRef::new(py, ticket.clone_ref(py))?);
        let cancelled = write.inner.retain(read);
        ticket
            .get()
            .add_retirement_callback(py, slf.getattr("_reap")?.unbind())?;
        if cancelled {
            ticket.get().cancel(py)?;
        }
        Ok(())
    }

    fn _consume(
        &self,
        py: Python<'_>,
        write: &KVImport,
        tickets: Vec<Py<TransferTicket>>,
    ) -> PyResult<()> {
        let workspace = write
            .inner
            .workspace()
            .ok_or_else(|| invariant(py, "KV import has no copy workspace"))?;
        let stream = workspace.torch_stream.bind(py);
        for ticket in tickets {
            ticket.get().wait_ready(py)?;
            ticket.get().result(
                py,
                if stream.is_none() {
                    None
                } else {
                    Some(stream.clone())
                },
            )?;
        }
        Ok(())
    }

    fn _drain(&self, py: Python<'_>, write: &KVImport) -> PyResult<()> {
        if let Some(workspace) = write.inner.workspace() {
            drain(py, &workspace)?;
        }
        Ok(())
    }

    fn _reap(&self, py: Python<'_>) -> PyResult<()> {
        self.notify_retired(py, self.inner.reap())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.pool)?;
        visit.call(&self.accesses)?;
        if let Some(wake) = self
            .wake
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
        {
            visit.call(wake)?;
        }
        self.inner
            .visit(|imports, workspaces| {
                for write in imports.values() {
                    visit.call(&write.owner)?;
                }
                for workspace in workspaces {
                    workspace.visit(&visit)?;
                }
                Ok(())
            })
            .unwrap_or(Ok(()))
    }

    fn __clear__(&mut self, py: Python<'_>) {
        if let Err(error) = self.stop(py) {
            error.write_unraisable(py, None);
        }
        if self.inner.require_retired().is_ok() {
            self.pool = py.None();
        }
    }
}

fn drain(py: Python<'_>, workspace: &Workspace) -> PyResult<()> {
    if let Some(stream) = &workspace.stream {
        py.detach(|| stream.wait())
            .map_err(PyRuntimeError::new_err)?;
    }
    Ok(())
}

fn cancel_reads(py: Python<'_>, reads: Vec<Arc<TransferRef>>) -> PyResult<()> {
    let mut failure = Ok(());
    for read in reads {
        if let Err(error) = read.owner.get().cancel(py)
            && failure.is_ok()
        {
            failure = Err(error);
        }
    }
    failure
}

impl KVImporter {
    fn notify_retired(&self, py: Python<'_>, retired: Vec<ImportRef>) -> PyResult<()> {
        let changed = !retired.is_empty();
        for write in retired {
            Completion::resolve(write.owner.get().retirement.bind(py))?;
        }

        if changed {
            let wake = self
                .wake
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .as_ref()
                .map(|wake| wake.clone_ref(py));
            if let Some(wake) = wake {
                wake.bind(py).call0()?;
            }
        }
        Ok(())
    }

    pub(super) fn cancel_request_imports(
        &self,
        py: Python<'_>,
        requests: &HashSet<uniserve_worker_ipc::RequestKey>,
        retained: &HashSet<uniserve_worker_ipc::BufferId>,
    ) -> PyResult<()> {
        let cancelled = cancel_reads(py, self.inner.cancel_requests(requests, retained));
        let reaped = self._reap(py);
        cancelled.and(reaped)
    }

    pub(super) fn release_buffers(
        &self,
        py: Python<'_>,
        buffers: &HashSet<uniserve_worker_ipc::BufferId>,
    ) -> PyResult<()> {
        let cancelled = cancel_reads(py, self.inner.release(buffers));
        let reaped = self._reap(py);
        cancelled.and(reaped)
    }
}

struct Copy {
    owner: Py<KVImporter>,
    write: Py<KVImport>,
    transports: Py<PyAny>,
}

impl ImportBackend for Copy {
    type Error = Py<PyAny>;
    type Callback = Py<PyAny>;
    type Read = TransferRef;
    type Workspace = Workspace;
    type Import = ImportRef;

    fn reset(&self, workspace: &Workspace) -> Result<(), Self::Error> {
        Python::attach(|py| {
            let units = self.write.get().initialized_units.bind(py);
            if units.is_empty() {
                return Ok(());
            }
            numerical_scope(py, workspace, || {
                self.owner
                    .borrow(py)
                    .pool
                    .bind(py)
                    .getattr("cache")?
                    .call_method1("zero_units", (units,))?;
                Ok(())
            })
            .map_err(|error| error.into_value(py).into_any())
        })
    }

    fn copy(&self, workspace: &Workspace) -> Result<(), Self::Error> {
        Python::attach(|py| {
            numerical_scope(py, workspace, || {
                py.import("uniserve_worker.storage.cache_imports")?
                    .call_method1(
                        "_copy",
                        (
                            self.owner.borrow(py).pool.bind(py),
                            self.owner.bind(py),
                            self.write.bind(py),
                            self.transports.bind(py),
                            workspace.values.bind(py),
                        ),
                    )?;
                Ok(())
            })
            .map_err(|error| error.into_value(py).into_any())
        })
    }

    fn drain(&self, workspace: &Workspace) -> Result<(), Self::Error> {
        if let Some(stream) = &workspace.stream {
            stream.wait().map_err(|error| {
                Python::attach(|py| PyRuntimeError::new_err(error).into_value(py).into_any())
            })?;
        }
        Ok(())
    }

    fn retired(&self, imports: Vec<ImportRef>) -> Result<(), Self::Error> {
        Python::attach(|py| {
            self.owner
                .borrow(py)
                .notify_retired(py, imports)
                .map_err(|error| error.into_value(py).into_any())
        })
    }

    fn error(error: uniserve_worker::Error) -> Self::Error {
        Python::attach(|py| native_error(py, error).into_value(py).into_any())
    }

    fn report(error: Self::Error) {
        Python::attach(|py| {
            PyErr::from_value(error.bind(py).clone().into_any()).write_unraisable(py, None);
        });
    }

    fn note_cleanup(error: &mut Self::Error, cleanup: Self::Error) {
        Python::attach(|py| {
            let _ = error.bind(py).call_method1(
                "add_note",
                (format!(
                    "KV import cleanup also failed: {}",
                    cleanup.bind(py)
                ),),
            );
        });
    }
}

fn numerical_scope<T>(
    py: Python<'_>,
    workspace: &Workspace,
    action: impl FnOnce() -> PyResult<T>,
) -> PyResult<T> {
    let stream = workspace.torch_stream.bind(py);
    let scope = if stream.is_none() {
        py.import("contextlib")?.call_method0("nullcontext")?
    } else {
        py.import("torch.cuda")?.call_method1("stream", (stream,))?
    };
    with_context(&scope, action)
}
