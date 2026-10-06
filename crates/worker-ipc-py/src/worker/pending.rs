//! Resolve completed numerical work and retire its pinned output row.

use std::sync::{Arc, Mutex, MutexGuard};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyException, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use uniserve_core::MediaSource;
use uniserve_worker::PendingOutput as NativeOutput;
use uniserve_worker_ipc::{BufferId, Call, CallStatus, ErrorCode, KvTransfer, RequestOutput};

use crate::convert;

use super::block_tables::BlockTables;
use super::error::{native_error, unsupported};
use super::host::HostTask;
use super::latent::LatentBuffer;
use super::output::OutputBuffer;
use super::request::{Request, RequestPool, RequestProgress};

/// Borrowed sampling views and device coordinates awaiting batch commit.
pub(crate) struct TokenUpdate {
    pub(super) sampled: Option<Py<PyAny>>,
    pub(super) logical_position: Py<PyAny>,
    pub(super) sampling_position: Py<PyAny>,
    pub(super) penalty_base: Option<Py<PyAny>>,
    pub(super) decode_increment: bool,
    pub(super) cache_length: Option<Py<PyAny>>,
    pub(super) prompt_logits: Option<Py<PyAny>>,
}

impl TokenUpdate {
    fn new(py: Python<'_>) -> Self {
        Self {
            sampled: None,
            logical_position: 0_u64.into_pyobject(py).unwrap().into_any().unbind(),
            sampling_position: 0_u64.into_pyobject(py).unwrap().into_any().unbind(),
            penalty_base: None,
            decode_increment: false,
            cache_length: None,
            prompt_logits: None,
        }
    }

    /// Release the object's borrow before numerical callbacks can run.
    pub(super) fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            sampled: self.sampled.as_ref().map(|value| value.clone_ref(py)),
            logical_position: self.logical_position.clone_ref(py),
            sampling_position: self.sampling_position.clone_ref(py),
            penalty_base: self.penalty_base.as_ref().map(|value| value.clone_ref(py)),
            decode_increment: self.decode_increment,
            cache_length: self.cache_length.as_ref().map(|value| value.clone_ref(py)),
            prompt_logits: self.prompt_logits.as_ref().map(|value| value.clone_ref(py)),
        }
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.sampled)?;
        visit.call(&self.logical_position)?;
        visit.call(&self.sampling_position)?;
        visit.call(&self.penalty_base)?;
        visit.call(&self.cache_length)?;
        visit.call(&self.prompt_logits)
    }

    fn clear(&mut self, py: Python<'_>) {
        *self = Self::new(py);
    }
}

/// One call's result, with borrowed numerical views until execution commits.
///
/// The output lease and host tasks own completion storage. RequestPool accepts
/// the resolved progress only after every output in the batch is ready.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct PendingOutput {
    state: Arc<Mutex<NativeOutput>>,
    buffer: Option<Py<OutputBuffer>>,
    #[pyo3(get)]
    pub(super) call: Py<PyAny>,
    #[pyo3(get)]
    pub(super) request: Py<Request>,
    pub(super) token_update: TokenUpdate,
    #[pyo3(get)]
    pub(super) latent_params: Option<Py<PyAny>>,
    #[pyo3(get)]
    pub(super) latent_buffer: Option<Py<LatentBuffer>>,
    pub(super) latent_exports: Py<PyDict>,
    pub(super) host_tasks: Vec<Py<HostTask>>,
    // Video unit positions within this decode round, assigned to this rank.
    pub(super) media_units: Vec<usize>,
    host_finish: Option<Py<PyAny>>,
    pub(super) tensor_exports: Py<PyDict>,
    #[pyo3(get)]
    pub(super) cache_exports: Py<PyDict>,
    #[pyo3(get)]
    pub(super) exported_locators: Py<PyList>,
    pub(super) cache_export: Option<Arc<KvTransfer>>,
    pub(super) cache_installation: Option<(BufferId, Arc<KvTransfer>)>,
    #[pyo3(get)]
    pub(super) device_reads: Py<PyList>,
    #[pyo3(get)]
    pub(super) feature_reads: Py<PyList>,
    #[pyo3(get)]
    pub(super) writes: Py<PyList>,
    #[pyo3(get, set)]
    pub(super) predicate: Option<Py<PyAny>>,
    #[pyo3(get, set)]
    pub(super) token_write: Option<Py<PyAny>>,
    #[pyo3(get, set)]
    pub(super) transition_write: Option<Py<PyAny>>,
    #[pyo3(get, set)]
    pub(super) completion_write: Option<Py<PyAny>>,
    #[pyo3(get, set)]
    pub(super) producer_write: Option<Py<PyAny>>,
}

impl PendingOutput {
    pub(super) fn set_sampling(
        &self,
        py: Python<'_>,
        sampling: (usize, usize, usize),
        logprobs: Option<(usize, usize, usize)>,
    ) -> PyResult<()> {
        let mut state = self.lock(py)?;
        state.sampling_range = Some(sampling);
        state.logprob_range = logprobs;
        Ok(())
    }

    pub(super) fn for_call(
        py: Python<'_>,
        call: Py<PyAny>,
        plan: &Call,
        request: Py<Request>,
        buffer: Py<OutputBuffer>,
        row: usize,
    ) -> PyResult<Self> {
        let native_request = request.borrow(py);
        let previous = native_request
            .request
            .progress()
            .map_err(|error| native_error(py, error))?;
        drop(native_request);

        Ok(Self {
            state: Arc::new(Mutex::new(
                NativeOutput::new(plan, previous, row).map_err(|error| native_error(py, error))?,
            )),
            buffer: Some(buffer),
            call,
            request,
            token_update: TokenUpdate::new(py),
            latent_params: None,
            latent_buffer: None,
            latent_exports: PyDict::new(py).unbind(),
            host_tasks: Vec::new(),
            media_units: Vec::new(),
            host_finish: None,
            tensor_exports: PyDict::new(py).unbind(),
            cache_exports: PyDict::new(py).unbind(),
            exported_locators: PyList::empty(py).unbind(),
            cache_export: None,
            cache_installation: None,
            device_reads: PyList::empty(py).unbind(),
            feature_reads: PyList::empty(py).unbind(),
            writes: PyList::empty(py).unbind(),
            predicate: None,
            token_write: None,
            transition_write: None,
            completion_write: None,
            producer_write: None,
        })
    }
}

#[pymethods]
impl PendingOutput {
    #[new]
    fn new(
        py: Python<'_>,
        call: Py<PyAny>,
        request: Py<Request>,
        buffer: Py<OutputBuffer>,
        row: usize,
    ) -> PyResult<Self> {
        let plan = Arc::clone(&call.extract::<PyRef<'_, crate::calls::Call>>(py)?.inner);
        Self::for_call(py, call, &plan, request, buffer, row)
    }

    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let value = {
            let state = self.lock(py)?;
            state.resolved().then(|| state.output.clone())
        };
        value
            .as_ref()
            .map(|value| convert::request_output_to_py(py, value))
            .transpose()
    }

    #[getter]
    fn request_key<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.call.bind(py).getattr("request_key")
    }

    #[getter]
    fn call_id<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.call.bind(py).getattr("call_id")
    }

    #[getter]
    fn kind<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.call.bind(py).getattr("kind")
    }

    #[getter]
    fn progress(&self, py: Python<'_>) -> PyResult<RequestProgress> {
        Ok(RequestProgress {
            inner: self.lock(py)?.progress,
        })
    }

    fn set_cache_length(&self, py: Python<'_>, length: u64) -> PyResult<()> {
        self.lock(py)?.set_cache_length(length);
        Ok(())
    }

    fn cache_coordinates(&self, tables: &Bound<'_, PyAny>) -> PyResult<(u32, u64, u32)> {
        let py = tables.py();
        if tables.is_none() {
            return Err(unsupported(py, "call requires request-to-token storage"));
        }
        let tables = tables.getattr("_tables")?;
        let tables = tables.extract::<PyRef<'_, BlockTables>>()?;
        let slot = self.request.borrow(py).request.slot() as u32;
        let visible = self.lock(py)?.progress.kv_visible_len;
        tables
            .tables
            .coordinates(slot, visible)
            .map_err(|error| native_error(py, error))
    }

    #[getter]
    fn status(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let status = self.lock(py)?.output.status;
        call_enum(py, "CallStatus", &status)
    }

    #[getter]
    fn error_code(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let code = self.lock(py)?.output.error_code;
        code.as_ref()
            .map(|code| call_enum(py, "ErrorCode", code))
            .transpose()
    }

    #[getter]
    fn media_units<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.media_units)
    }

    #[getter]
    fn host_tasks(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        PyTuple::new(py, self.host_tasks.iter().map(|task| task.bind(py))).map(Bound::unbind)
    }

    #[pyo3(signature = (tasks, finish=None))]
    fn set_host_tasks(slf: &Bound<'_, Self>, tasks: Vec<Py<HostTask>>, finish: Option<Py<PyAny>>) {
        let retired = {
            let mut this = slf.borrow_mut();
            (
                std::mem::replace(&mut this.host_tasks, tasks),
                std::mem::replace(&mut this.host_finish, finish),
            )
        };
        drop(retired);
    }

    pub(super) fn ready(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let py = slf.py();
        let (buffer, tasks) = {
            let this = slf.borrow();
            if this.lock(py)?.resolved() {
                return Ok(true);
            }
            (this.buffer(py), this.clone_host_tasks(py))
        };
        let Some(buffer) = buffer else {
            return Ok(false);
        };
        // Readiness may dispatch completion observers. No result borrow may
        // span a callback, including the output buffer's notification.
        if !buffer.get().ready(py)? {
            return Ok(false);
        }
        Ok(tasks.iter().all(|task| task.borrow(py).done()))
    }

    fn materialize(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        Self::resolve(slf)?;
        let value = slf.borrow().result(slf.py())?;
        let result = convert::request_output_to_py(slf.py(), &value)?;
        slf.borrow().handoff_media(slf.py())?;
        Ok(result)
    }

    pub(super) fn abandon(slf: &Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        let (tasks, finish, buffer, row) = {
            let mut this = slf.borrow_mut();
            let row = {
                let mut state = this.lock(py)?;
                state.discard_media();
                state.row
            };
            (
                std::mem::take(&mut this.host_tasks),
                this.host_finish.take(),
                this.buffer.take(),
                row,
            )
        };
        drop(finish);

        // Submitted host work retains its input until it actually finishes.
        // Try every owner even when a task's release callback fails.
        let mut failure: Option<PyErr> = None;
        for result in tasks
            .iter()
            .map(|task| task.borrow(py).abandon(py))
            .chain(buffer.iter().map(|buffer| buffer.get().discard(py, row)))
        {
            if let Err(error) = result {
                if let Some(first) = &failure {
                    let _ = first
                        .value(py)
                        .call_method1("add_note", (error.to_string(),));
                } else {
                    failure = Some(error);
                }
            }
        }
        failure.map_or(Ok(()), Err)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.buffer)?;
        visit.call(&self.call)?;
        visit.call(&self.request)?;
        self.token_update.traverse(&visit)?;
        visit.call(&self.latent_params)?;
        visit.call(&self.latent_buffer)?;
        visit.call(&self.latent_exports)?;
        for task in &self.host_tasks {
            visit.call(task)?;
        }
        visit.call(&self.host_finish)?;
        visit.call(&self.tensor_exports)?;
        visit.call(&self.cache_exports)?;
        visit.call(&self.exported_locators)?;
        visit.call(&self.device_reads)?;
        visit.call(&self.feature_reads)?;
        visit.call(&self.writes)?;
        visit.call(&self.predicate)?;
        visit.call(&self.token_write)?;
        visit.call(&self.transition_write)?;
        visit.call(&self.completion_write)?;
        visit.call(&self.producer_write)
    }

    fn __clear__(&mut self, py: Python<'_>) {
        self.token_update.clear(py);
        self.latent_params = None;
        self.latent_buffer = None;
        self.latent_exports.bind(py).clear();
        self.host_tasks.clear();
        self.host_finish = None;
    }
}

impl PendingOutput {
    pub(super) fn release_execution_references(&mut self, py: Python<'_>) -> PyResult<()> {
        // A host task may still be filling its reserved buffers after batch
        // commit. Keep only those writes until its completion callback runs.
        let writes = self
            .writes
            .bind(py)
            .iter()
            .map(|write| write.cast_into::<super::storage::Buffer>())
            .collect::<Result<Vec<_>, _>>()?;
        self.writes = PyList::new(
            py,
            writes.into_iter().filter(|write| write.get().deferred(py)),
        )?
        .unbind();
        self.tensor_exports.bind(py).clear();
        self.cache_exports.bind(py).clear();
        self.exported_locators.bind(py).call_method0("clear")?;
        self.cache_export = None;
        self.cache_installation = None;

        self.latent_exports.bind(py).clear();
        self.latent_params = None;
        self.latent_buffer = None;

        self.predicate = None;
        self.token_write = None;
        self.transition_write = None;
        self.completion_write = None;
        self.producer_write = None;
        self.token_update.clear(py);
        Ok(())
    }

    pub(super) fn revoke_exports(&self, transports: &Bound<'_, PyAny>) -> PyResult<()> {
        for locator in self.exported_locators.bind(transports.py()) {
            transports
                .get_item(locator.getattr("backend")?)?
                .call_method1("release", (locator,))?;
        }
        Ok(())
    }

    pub(super) fn lock(&self, py: Python<'_>) -> PyResult<MutexGuard<'_, NativeOutput>> {
        self.state
            .lock_py_attached(py)
            .map_err(|_| PyRuntimeError::new_err("pending output lock poisoned"))
    }

    fn clone_host_tasks(&self, py: Python<'_>) -> Vec<Py<HostTask>> {
        self.host_tasks
            .iter()
            .map(|task| task.clone_ref(py))
            .collect()
    }

    fn buffer(&self, py: Python<'_>) -> Option<Py<OutputBuffer>> {
        self.buffer.as_ref().map(|buffer| buffer.clone_ref(py))
    }

    pub(super) fn result(&self, py: Python<'_>) -> PyResult<RequestOutput> {
        let result = self.lock(py)?.result().cloned();
        result.map_err(|error| native_error(py, error))
    }

    pub(super) fn handoff_media(&self, py: Python<'_>) -> PyResult<()> {
        self.lock(py)?.handoff_media();
        Ok(())
    }

    pub(super) fn take_media(&self, py: Python<'_>) -> PyResult<Option<MediaSource>> {
        Ok(self.lock(py)?.take_media())
    }

    pub(super) fn validate_output(&self, py: Python<'_>, call: &Call) -> PyResult<()> {
        let result = {
            let buffer = self
                .buffer
                .as_ref()
                .ok_or_else(lost_buffer)?
                .get()
                .lock(py)?;
            self.lock(py)?.validate_output(&buffer, call)
        };
        result.map_err(|error| native_error(py, error))
    }

    pub(super) fn cancel(&self, py: Python<'_>, requests: &mut RequestPool) -> PyResult<()> {
        let result = self.lock(py)?.cancel(&mut requests.pool);
        result.map_err(|error| native_error(py, error))
    }

    pub(super) fn resolve(slf: &Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        if slf.borrow().lock(py)?.resolved() {
            return Ok(());
        }
        if !Self::ready(slf)? {
            return Err(PyRuntimeError::new_err(
                "completion was resolved before query-ready",
            ));
        }

        let (state, buffer, request) = {
            let this = slf.borrow();
            (
                Arc::clone(&this.state),
                this.buffer(py).ok_or_else(lost_buffer)?,
                this.request.clone_ref(py),
            )
        };
        let parent = request
            .borrow(py)
            .request
            .progress()
            .map_err(|error| native_error(py, error))?;
        let (status, reports) = {
            let state = state
                .lock_py_attached(py)
                .map_err(|_| PyRuntimeError::new_err("pending output lock poisoned"))?;
            (state.output.status, state.reports_output)
        };

        // A host callback may update this result. Drop every native lock and
        // Python borrow before invoking it or dispatching completion observers.
        let completed = if status == CallStatus::Ok {
            resolve_host(slf, reports)
        } else {
            Ok(())
        };
        let completed = completed.and_then(|()| {
            buffer.get().with_readback(py, |buffer| {
                state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .resolve(parent, buffer)
            })
        });
        match completed {
            Err(error) if error.is_instance_of::<PyException>(py) => {
                log_failure(slf, &error)?;
                {
                    let mut state = state
                        .lock_py_attached(py)
                        .map_err(|_| PyRuntimeError::new_err("pending output lock poisoned"))?;
                    state.output.status = CallStatus::Error;
                    state.output.error_code = Some(ErrorCode::ComputeError);
                }
                buffer.get().with_readback(py, |buffer| {
                    state
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .resolve(parent, buffer)
                })?;
            }
            Err(error) => return Err(error),
            Ok(()) => {}
        }

        let retired = {
            let mut this = slf.borrow_mut();
            this.writes.bind(py).call_method0("clear")?;
            (
                this.buffer.take(),
                std::mem::take(&mut this.host_tasks),
                this.host_finish.take(),
            )
        };
        drop(retired);
        Ok(())
    }

    pub(super) fn submit_host_tasks(slf: &Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        let tasks = {
            let this = slf.borrow();
            if this.lock(py)?.resolved() {
                return Ok(());
            }
            this.clone_host_tasks(py)
        };
        for task in tasks {
            task.borrow(py).submit_if_ready(py)?;
        }
        Ok(())
    }

    pub(super) fn accept(&self, py: Python<'_>, requests: &mut RequestPool) -> PyResult<()> {
        let result = self.lock(py)?.accept(&mut requests.pool);
        result.map_err(|error| native_error(py, error))
    }
}

fn resolve_host(output: &Bound<'_, PendingOutput>, reports: bool) -> PyResult<()> {
    let py = output.py();
    let tasks = output.borrow().clone_host_tasks(py);
    let results = tasks
        .iter()
        .map(|task| task.borrow(py).result(py, None))
        .collect::<PyResult<Vec<_>>>()?;
    let finish = output.borrow_mut().host_finish.take();
    if let Some(finish) = finish {
        finish.call1(py, (PyTuple::new(py, results)?,))?;
        return Ok(());
    }

    if reports {
        for result in results {
            if let Ok(bytes) = result.bind(py).cast::<PyBytes>() {
                // Encoders and muxers return bytes; export and ownership
                // stay native. PyBytes keeps this immutable borrow alive while
                // allocation and the copy run outside the interpreter.
                let payload = bytes.as_bytes();
                let source = py.detach(|| MediaSource::publish(payload))?;
                output
                    .borrow()
                    .lock(py)?
                    .set_media(source)
                    .map_err(|error| native_error(py, error))?;
            }
        }
    }
    Ok(())
}

/// Standalone producers hand ownership to their receiving process directly.
#[pyfunction]
pub(super) fn store_media_bytes(py: Python<'_>, payload: &[u8]) -> PyResult<String> {
    if payload.is_empty() {
        return Err(PyValueError::new_err("published media must not be empty"));
    }
    Ok(py
        .detach(|| MediaSource::publish(payload))?
        .into_locator()
        .name)
}

fn lost_buffer() -> PyErr {
    PyRuntimeError::new_err("completion lost its pinned output buffer")
}

fn call_enum(py: Python<'_>, name: &str, value: &impl serde::Serialize) -> PyResult<Py<PyAny>> {
    py.import("uniserve_worker.protocol.call")?
        .getattr(name)?
        .call1((pythonize::pythonize(py, value)?,))
        .map(Bound::unbind)
}

fn log_failure(output: &Bound<'_, PendingOutput>, error: &PyErr) -> PyResult<()> {
    let py = output.py();
    let call = output.borrow().call.clone_ref(py);
    let call = call.bind(py);
    let kwargs = PyDict::new(py);
    kwargs.set_item(
        "exc_info",
        (error.get_type(py), error.value(py), error.traceback(py)),
    )?;
    py.import("logging")?
        .call_method1("getLogger", ("uniserve_worker.execution.output",))?
        .call_method(
            "error",
            (
                "completion materialization failed: request=%s call=%s computation=%s",
                call.getattr("request_key")?,
                call.getattr("call_id")?,
                call.getattr("kind")?,
            ),
            Some(&kwargs),
        )?;
    Ok(())
}
