//! Numerical KV copies on native import lanes and bounded conversion storage.

use std::collections::{HashMap, HashSet};
use std::ops::Deref;
use std::sync::{Arc, Mutex, PoisonError};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use uniserve_worker::cuda::{DeviceGuard, Stream};
use uniserve_worker::{KVImport as NativeImport, KVImporter as NativeImporter};

use super::block_tables::GroupTable;
use super::completion::{Completion, CompletionRef};
use super::error::{invariant, native_error};
use super::host::{HostLane, HostTask, with_context};
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

pub(super) enum CopyCompletion {
    Task(Py<HostTask>),
    Immediate(Py<Completion>),
}

impl CopyCompletion {
    pub(crate) fn done(&self, py: Python<'_>) -> bool {
        match self {
            Self::Task(task) => task.borrow(py).done(),
            Self::Immediate(completion) => completion.borrow(py).done(),
        }
    }

    fn result(&self, py: Python<'_>) -> PyResult<()> {
        match self {
            Self::Task(task) => task.borrow(py).result(py, None).map(drop),
            Self::Immediate(completion) => completion.borrow(py).result(py, None),
        }
    }

    fn owner(&self, py: Python<'_>) -> Py<PyAny> {
        match self {
            Self::Task(task) => task.clone_ref(py).into_any(),
            Self::Immediate(completion) => completion.clone_ref(py).into_any(),
        }
    }
}

#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct KVImport {
    inner: Arc<NativeImport<TransferRef, Workspace>>,
    #[pyo3(get)]
    tables: Py<PyTuple>,
    #[pyo3(get)]
    initialized_units: Py<PyTuple>,
    #[pyo3(get)]
    publication: Py<PyAny>,
    pub(super) completion: CopyCompletion,
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
    }

    #[getter]
    pub(crate) fn released(&self) -> bool {
        self.inner.released()
    }

    #[getter]
    fn completion(&self, py: Python<'_>) -> Py<PyAny> {
        self.completion.owner(py)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.tables)?;
        visit.call(&self.initialized_units)?;
        visit.call(&self.publication)?;
        visit.call(&self.retirement)?;
        match &self.completion {
            CopyCompletion::Task(task) => visit.call(task)?,
            CopyCompletion::Immediate(completion) => visit.call(completion)?,
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
    inner: NativeImporter<ImportRef, Workspace>,
    pool: Py<PyAny>,
    accesses: Py<KVCacheManager>,
    tasks: Py<HostLane>,
    wake: Mutex<Option<Py<PyAny>>>,
}

#[pymethods]
impl KVImporter {
    #[new]
    #[pyo3(signature = (pool, *, capacity))]
    fn new(py: Python<'_>, pool: Py<PyAny>, capacity: usize) -> PyResult<Self> {
        let workers = capacity.min(4);
        let tasks = Py::new(
            py,
            HostLane::new(py, capacity, workers, "worker-kv-import")?,
        )?;
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
                let stream = Stream::new().map_err(PyRuntimeError::new_err)?;
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
            inner: NativeImporter::new(workspaces),
            pool,
            accesses,
            tasks,
            wake: Mutex::new(None),
        })
    }

    fn set_completion_wake(&self, py: Python<'_>, wake: Option<Py<PyAny>>) {
        self.tasks
            .borrow(py)
            .set_completion_wake(wake.as_ref().map(|wake| wake.clone_ref(py)));
        let previous = std::mem::replace(
            &mut *self.wake.lock().unwrap_or_else(PoisonError::into_inner),
            wake,
        );
        drop(previous);
    }

    #[pyo3(signature = (publication, *, request_pool_idx, tables, initialized_units, transports))]
    fn reserve(
        slf: &Bound<'_, Self>,
        publication: Py<PyAny>,
        request_pool_idx: usize,
        tables: Py<PyTuple>,
        initialized_units: Py<PyTuple>,
        transports: Py<PyAny>,
    ) -> PyResult<Py<KVImport>> {
        let py = slf.py();
        let this = slf.borrow();
        let source = buffer_id(&publication.bind(py).getattr("source")?)?;
        let extent: u64 = publication
            .bind(py)
            .getattr("published_extent")?
            .extract()?;
        let groups = publication.bind(py).getattr("groups")?;
        let mut ranges = HashMap::new();
        for (table, group) in tables.bind(py).iter().zip(groups.try_iter()?) {
            let start: u64 = group?.getattr("start")?.extract()?;
            let table = table.extract::<PyRef<'_, GroupTable>>()?;
            let count = extent.checked_sub(start).ok_or_else(|| {
                super::error::invalid(py, "KV import starts beyond its published extent")
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
            || publication.bind(py).getattr("tensors")?.is_truthy()?;

        let completion = if copy {
            CopyCompletion::Task(this.tasks.borrow(py).reserve(py)?)
        } else {
            let completion = Py::new(py, Completion::new())?;
            Completion::resolve(completion.bind(py))?;
            CopyCompletion::Immediate(completion)
        };
        let write = Py::new(
            py,
            KVImport {
                inner: Arc::new(NativeImport::new(source, request_pool_idx, copy)),
                tables,
                initialized_units,
                publication,
                completion,
                retirement: Py::new(py, Completion::new())?,
            },
        )?;
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
            if let CopyCompletion::Task(task) = &write.get().completion {
                task.borrow(py).abandon(py)?;
            }
            return Err(native_error(py, error));
        }

        if let CopyCompletion::Task(task) = &write.get().completion {
            let submit = (|| {
                let partial = py.import("functools")?.getattr("partial")?;
                let callback = partial.call1((slf.getattr("_task_done")?, write.bind(py)))?;
                HostTask::add_done_callback(task.bind(py), callback.unbind());
                let action = partial.call1((slf.getattr("_copy")?, write.bind(py), transports))?;
                HostTask::configure(
                    task.bind(py),
                    action.unbind(),
                    Vec::new(),
                    None,
                    None,
                    None,
                    "uniserve.kv_import",
                )?;
                task.borrow(py).submit_if_ready(py)
            })();
            if let Err(error) = submit {
                task.borrow(py).abandon(py)?;
                if task.borrow(py).done() {
                    write.get().inner.task_done();
                }
                this.abandon(py, write.get())?;
                return Err(error);
            }
        }
        Ok(write)
    }

    fn owns(&self, write: &KVImport) -> bool {
        self.inner.owns(&write.inner)
    }

    fn adopt(&self, py: Python<'_>, write: &KVImport) -> PyResult<()> {
        if !write.completion.done(py) {
            return Err(PyRuntimeError::new_err(
                "KV import was observed before input readiness",
            ));
        }
        write.completion.result(py)?;
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
        let stopped = self.tasks.borrow(py).close(py);
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

    fn _copy(slf: &Bound<'_, Self>, write: Py<KVImport>, transports: Py<PyAny>) -> PyResult<()> {
        let py = slf.py();
        let this = slf.borrow();
        let write = write.bind(py);
        let inner = Arc::clone(&write.get().inner);
        let importer = &this.inner;
        inner.start();

        let workspace = match py.detach(|| importer.acquire(&inner)) {
            Ok(workspace) => workspace,
            Err(error) => {
                inner.finish(true);
                this._reap(py)?;
                return Err(native_error(py, error));
            }
        };

        let copied = (|| {
            let stream = workspace.torch_stream.bind(py);
            let scope = if stream.is_none() {
                py.import("contextlib")?.call_method0("nullcontext")?
            } else {
                py.import("torch.cuda")?.call_method1("stream", (stream,))?
            };
            with_context(&scope, || {
                this._require_active(py, write.get())?;
                let units = write.get().initialized_units.bind(py);
                if !units.is_empty() {
                    this.pool
                        .bind(py)
                        .getattr("cache")?
                        .call_method1("zero_units", (units,))?;
                }
                // Transfer backends use separate copy streams. Reset destination
                // units before any of those streams can start writing them.
                drain(py, &workspace)?;
                py.import("uniserve_worker.storage.cache_imports")?
                    .call_method1(
                        "_copy",
                        (
                            this.pool.bind(py),
                            slf,
                            write,
                            transports,
                            workspace.values.bind(py),
                        ),
                    )?;
                Ok(())
            })
        })();

        let drained = drain(py, &workspace);
        write.get().inner.finish(drained.is_ok());
        this._reap(py)?;
        if let Err(error) = drained {
            if let Err(cause) = copied {
                error.set_cause(py, Some(cause));
            }
            return Err(error);
        }
        copied
    }

    fn _task_done(&self, py: Python<'_>, write: &KVImport, _task: &HostTask) -> PyResult<()> {
        write.inner.task_done();
        self._reap(py)
    }

    fn _reap(&self, py: Python<'_>) -> PyResult<()> {
        let retired = self.inner.reap();
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

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.pool)?;
        visit.call(&self.accesses)?;
        visit.call(&self.tasks)?;
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
