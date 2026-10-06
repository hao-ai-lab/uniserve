//! Typed numerical owners retained by the native batch input set.

use std::sync::Arc;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use uniserve_worker::{BatchInputs as NativeInputs, InputReady, InputWait as NativeWait};
use uniserve_worker_ipc::BufferId;

use super::completion::Completion;
use super::host::HostTask;
use super::kv_import::{KVImport, KVImporter};
use super::latent::{LatentImport, LatentPool};
use super::output::OutputBuffer;
use super::protocol::buffer_id;
use super::storage::{TensorRead, TensorStore};
use super::transfer::TransferTicket;

pub(super) enum Input {
    Tensor(Py<TensorRead>),
    Latent(Py<LatentImport>),
    Cache(Py<KVImport>),
    Borrowed,
}

impl Input {
    fn clone_ref(&self, py: Python<'_>) -> Self {
        match self {
            Self::Tensor(read) => Self::Tensor(read.clone_ref(py)),
            Self::Latent(write) => Self::Latent(write.clone_ref(py)),
            Self::Cache(write) => Self::Cache(write.clone_ref(py)),
            Self::Borrowed => Self::Borrowed,
        }
    }

    fn dependencies(&self, py: Python<'_>) -> Vec<Dependency> {
        match self {
            Self::Tensor(read) => read
                .borrow(py)
                .tickets(py)
                .into_iter()
                .map(Dependency::Transfer)
                .collect(),
            Self::Latent(write) => write
                .get()
                .inner
                .transfers()
                .iter()
                .map(|ticket| Dependency::Transfer(ticket.owner.clone_ref(py)))
                .collect(),
            Self::Cache(write) => vec![Dependency::Cache(write.clone_ref(py))],
            Self::Borrowed => Vec::new(),
        }
    }

    fn visit(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        match self {
            Self::Tensor(read) => visit.call(read),
            Self::Latent(write) => visit.call(write),
            Self::Cache(write) => visit.call(write),
            Self::Borrowed => Ok(()),
        }
    }
}

impl InputReady for Input {
    type Error = PyErr;

    fn ready(&self) -> PyResult<bool> {
        Python::attach(|py| match self {
            Self::Tensor(read) => {
                for ticket in read.borrow(py).tickets(py) {
                    if !ticket.get().ready(py)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            Self::Latent(write) => {
                for ticket in write.get().inner.transfers() {
                    if !ticket.owner.get().ready(py)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            Self::Cache(write) => Ok(write.get().done()),
            Self::Borrowed => Ok(true),
        })
    }
}

enum Dependency {
    Cache(Py<KVImport>),
    Completion(Py<Completion>),
    Task(Py<HostTask>),
    Transfer(Py<TransferTicket>),
    Predicate(Py<OutputBuffer>),
}

impl Dependency {
    fn clone_ref(&self, py: Python<'_>) -> Self {
        match self {
            Self::Cache(value) => Self::Cache(value.clone_ref(py)),
            Self::Completion(value) => Self::Completion(value.clone_ref(py)),
            Self::Task(value) => Self::Task(value.clone_ref(py)),
            Self::Transfer(value) => Self::Transfer(value.clone_ref(py)),
            Self::Predicate(value) => Self::Predicate(value.clone_ref(py)),
        }
    }

    fn subscribe(&self, py: Python<'_>, callback: Py<PyAny>) -> PyResult<()> {
        match self {
            Self::Cache(value) => KVImport::add_done_callback(value.bind(py), callback),
            Self::Completion(value) => Completion::add_done_callback(value.bind(py), callback),
            Self::Task(value) => HostTask::add_done_callback(value.bind(py), callback),
            Self::Transfer(value) => value.get().add_done_callback(py, callback)?,
            Self::Predicate(value) => {
                Completion::add_done_callback(value.get().completion(py)?.bind(py), callback)
            }
        }
        Ok(())
    }

    fn visit(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        match self {
            Self::Cache(value) => visit.call(value),
            Self::Completion(value) => visit.call(value),
            Self::Task(value) => visit.call(value),
            Self::Transfer(value) => visit.call(value),
            Self::Predicate(value) => visit.call(value),
        }
    }
}

impl InputReady for Dependency {
    type Error = PyErr;

    fn ready(&self) -> PyResult<bool> {
        Python::attach(|py| match self {
            Self::Cache(value) => Ok(value.get().done()),
            Self::Completion(value) => Ok(value.borrow(py).done()),
            Self::Task(value) => Ok(value.borrow(py).done()),
            Self::Transfer(value) => value.get().ready(py),
            Self::Predicate(value) => value.get().query_ready(py),
        })
    }
}

#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
struct InputWait {
    inner: Arc<NativeWait>,
    callback: Py<PyAny>,
}

#[pymethods]
impl InputWait {
    #[pyo3(signature = (*_args))]
    fn __call__(&self, py: Python<'_>, _args: &Bound<'_, PyTuple>) -> PyResult<()> {
        if self.inner.arrive() {
            self.callback.bind(py).call0()?;
        }
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.callback)
    }
}

/// Physical inputs borrowed by one batch. Resource owners retain unfinished
/// device accesses after this set relinquishes its numerical consumers.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct BatchInputs {
    inner: NativeInputs<Input, Dependency>,
}

impl BatchInputs {
    pub(super) fn get(&self, py: Python<'_>, buffer: BufferId) -> Option<Input> {
        self.inner
            .inputs
            .get(&buffer)
            .map(|input| input.clone_ref(py))
    }

    pub(super) fn cache_import(&self, py: Python<'_>, buffer: BufferId) -> Option<Py<KVImport>> {
        match self.inner.inputs.get(&buffer) {
            Some(Input::Cache(write)) => Some(write.clone_ref(py)),
            _ => None,
        }
    }

    pub(super) fn cache_imports(&self, py: Python<'_>) -> Vec<Py<KVImport>> {
        self.inner
            .inputs
            .values()
            .filter_map(|input| match input {
                Input::Cache(write) => Some(write.clone_ref(py)),
                _ => None,
            })
            .collect()
    }

    /// Slots made resident by this batch's latent imports. Failed execution
    /// releases them using the import owner's adoption state.
    pub(super) fn imported_slots(&self) -> Vec<i64> {
        self.inner
            .inputs
            .values()
            .filter_map(|input| match input {
                Input::Latent(write) if write.get().inner.adopted() => {
                    Some(write.get().inner.request_pool_idx as i64)
                }
                _ => None,
            })
            .collect()
    }

    pub(super) fn insert(&mut self, buffer: BufferId, input: Input) {
        self.inner.inputs.insert(buffer, input);
    }

    pub(super) fn remove(&mut self, buffer: BufferId) {
        self.inner.inputs.remove(&buffer);
    }

    pub(super) fn started(&self) -> usize {
        self.inner.started
    }

    pub(super) fn set_started(&mut self, started: usize) {
        self.inner.started = started;
    }

    pub(super) fn awaiting_reads(&self) -> bool {
        self.inner.awaiting_reads
    }

    pub(super) fn set_awaiting_reads(&mut self, waiting: bool) {
        self.inner.awaiting_reads = waiting;
    }
}

#[pymethods]
impl BatchInputs {
    #[new]
    pub(super) fn new() -> Self {
        Self {
            inner: NativeInputs::default(),
        }
    }

    #[getter]
    pub(super) fn closed(&self) -> bool {
        self.inner.closed()
    }

    #[getter]
    pub(super) fn submitted(&self) -> bool {
        self.inner.submitted
    }

    #[setter]
    pub(super) fn set_submitted(&mut self, submitted: bool) {
        self.inner.submitted = submitted;
    }

    #[getter]
    pub(super) fn predicate(&self, py: Python<'_>) -> Option<Py<OutputBuffer>> {
        match &self.inner.predicate {
            Some(Dependency::Predicate(buffer)) => Some(buffer.clone_ref(py)),
            _ => None,
        }
    }

    #[setter]
    pub(super) fn set_predicate(&mut self, buffer: Option<Py<OutputBuffer>>) {
        self.inner.predicate = buffer.map(Dependency::Predicate);
    }

    pub(super) fn set_dependencies(&mut self, dependencies: Vec<Py<Completion>>) {
        self.inner.dependencies = dependencies
            .into_iter()
            .map(Dependency::Completion)
            .collect();
    }

    pub(super) fn storage_ready(&self) -> PyResult<bool> {
        self.inner.storage_ready()
    }

    pub(super) fn require_storage(&self, py: Python<'_>) -> PyResult<()> {
        for dependency in &self.inner.dependencies {
            if let Dependency::Completion(value) = dependency {
                value.borrow(py).result(py, None)?;
            }
        }
        Ok(())
    }

    pub(super) fn ready(&self) -> PyResult<bool> {
        self.inner.ready()
    }

    fn input_ready(&self, buffer: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner.input_ready(&buffer_id(buffer)?)
    }

    pub(super) fn is_borrowed(&self, buffer: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(matches!(
            self.inner.inputs.get(&buffer_id(buffer)?),
            Some(Input::Borrowed)
        ))
    }

    #[pyo3(signature = (buffer, value=None))]
    fn add(&mut self, buffer: &Bound<'_, PyAny>, value: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let value = match value {
            None => Input::Borrowed,
            Some(value) if value.is_instance_of::<TensorRead>() => Input::Tensor(value.extract()?),
            Some(value) if value.is_instance_of::<LatentImport>() => {
                Input::Latent(value.extract()?)
            }
            Some(value) => Input::Cache(value.extract()?),
        };
        self.insert(buffer_id(buffer)?, value);
        Ok(())
    }

    fn tensor(
        &self,
        py: Python<'_>,
        buffer: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Py<TensorRead>>> {
        Ok(match self.inner.inputs.get(&buffer_id(buffer)?) {
            Some(Input::Tensor(read)) => Some(read.clone_ref(py)),
            _ => None,
        })
    }

    fn cache(&self, py: Python<'_>, buffer: &Bound<'_, PyAny>) -> PyResult<Option<Py<KVImport>>> {
        Ok(self.cache_import(py, buffer_id(buffer)?))
    }

    pub(super) fn on_ready(slf: &Bound<'_, Self>, callback: Py<PyAny>) -> PyResult<()> {
        let py = slf.py();
        let (dependencies, wait) = {
            let this = slf.borrow();
            if this.inner.closed() {
                return Ok(());
            }

            let mut dependencies: Vec<_> = this
                .inner
                .dependencies
                .iter()
                .chain(this.inner.tasks.values())
                .map(|value| value.clone_ref(py))
                .collect();
            for input in this.inner.inputs.values() {
                dependencies.extend(input.dependencies(py));
            }

            // Unsealed predicates still need the executor to capture their
            // transferred inputs. Their copy fence joins the next snapshot.
            if let Some(Dependency::Predicate(buffer)) = &this.inner.predicate
                && buffer.get().sealed(py)?
            {
                dependencies.push(Dependency::Predicate(buffer.clone_ref(py)));
            }
            let wait = this.inner.wait(dependencies.len());
            (dependencies, wait)
        };

        // Subscription can invoke already-complete dependencies immediately.
        // Release the input borrow first so observers may close the batch.
        let callback = Py::new(
            py,
            InputWait {
                inner: wait,
                callback,
            },
        )?;
        for dependency in dependencies {
            dependency.subscribe(py, callback.clone_ref(py).into_any())?;
        }
        callback.get().__call__(py, &PyTuple::empty(py))
    }

    pub(super) fn close(
        slf: &Bound<'_, Self>,
        tensor_store: &TensorStore,
        latent_pool: Option<&Bound<'_, LatentPool>>,
        kv_importer: Option<&Bound<'_, KVImporter>>,
    ) -> PyResult<()> {
        let py = slf.py();
        let (tasks, tensors, latents, caches, predicate) = {
            let mut this = slf.borrow_mut();
            if !this.inner.close() {
                return Ok(());
            }

            let tasks: Vec<_> = this.inner.tasks.drain().map(|(_, task)| task).collect();
            let mut tensors = Vec::new();
            let mut latents = Vec::new();
            let mut caches = Vec::new();
            for input in this.inner.inputs.values() {
                match input {
                    Input::Tensor(read) => tensors.push(read.clone_ref(py)),
                    Input::Latent(write) => latents.push(write.clone_ref(py)),
                    Input::Cache(write) => caches.push(write.clone_ref(py)),
                    Input::Borrowed => {}
                }
            }
            let predicate = this.inner.predicate.take();
            (tasks, tensors, latents, caches, predicate)
        };

        // Owners may invoke completion observers while releasing a resource.
        // Attempt every release without retaining a mutable input borrow.
        let mut failure = None;
        let mut collect = |result: PyResult<()>| {
            if let Err(error) = result {
                if let Some(first) = &failure {
                    let first: &PyErr = first;
                    let _ = first
                        .value(py)
                        .call_method1("add_note", (format!("input cleanup failed: {error}"),));
                } else {
                    failure = Some(error);
                }
            }
        };
        for task in tasks {
            if let Dependency::Task(task) = task {
                collect(task.borrow(py).cancel(py).map(drop));
            }
        }

        for write in &latents {
            if !write.get().inner.adopted() {
                collect(
                    latent_pool
                        .ok_or_else(|| super::error::invariant(py, "latent input has no pool"))
                        .and_then(|pool| pool.borrow_mut().abandon_import(py, write.bind(py))),
                );
            }
        }

        for write in caches {
            if !write.get().released() {
                collect(
                    kv_importer
                        .ok_or_else(|| super::error::invariant(py, "KV input has no importer"))
                        .and_then(|importer| importer.borrow().abandon(py, write.get())),
                );
            }
        }

        if let Some(Dependency::Predicate(buffer)) = predicate {
            collect(OutputBuffer::abandon(buffer.bind(py)));
        }

        if !tensors.is_empty() {
            collect(tensor_store.complete_reads(
                py,
                tensors.iter().map(|read| read.bind(py).clone()).collect(),
                None,
                Vec::new(),
            ));
        }

        for write in latents {
            for ticket in write.get().inner.transfers() {
                collect(TransferTicket::close(ticket.owner.bind(py).clone()));
            }
        }

        failure.map_or(Ok(()), Err)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for input in self.inner.inputs.values() {
            input.visit(&visit)?;
        }

        for dependency in self
            .inner
            .dependencies
            .iter()
            .chain(self.inner.tasks.values())
            .chain(self.inner.predicate.iter())
        {
            dependency.visit(&visit)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.inner.close();
        self.inner.inputs.clear();
        self.inner.tasks.clear();
        self.inner.dependencies.clear();
        self.inner.predicate = None;
    }
}

impl BatchInputs {
    pub(super) fn add_image(&mut self, call: uniserve_core::CallId, task: Py<HostTask>) {
        self.inner.tasks.insert(call, Dependency::Task(task));
    }

    pub(super) fn image(
        &self,
        py: Python<'_>,
        call: uniserve_core::CallId,
    ) -> Option<Py<HostTask>> {
        match self.inner.tasks.get(&call) {
            Some(Dependency::Task(task)) => Some(task.clone_ref(py)),
            _ => None,
        }
    }
}
