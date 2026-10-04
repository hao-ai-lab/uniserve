//! Resolve completed numerical work and retire its pinned output row.

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyException, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use uniserve_core::CallId;
use uniserve_worker::RequestProgress;
use uniserve_worker_ipc::{CallStatus, ErrorCode, RequestKey};

use super::error::native_error;
use super::host::HostTask;
use super::output::OutputBuffer;
use super::protocol::{call_id, request_key};
use super::request::{Request, RequestPool, progress_from_py, progress_to_py};

// Layouts produced by sampling_columns and the canvas step's device commit.
const SAMPLING_FIELDS: usize = 4;
const STEP_STOPPED: i64 = 1;
const STEP_SKIPPED: i64 = 2;

/// One call's result, with borrowed numerical views until execution commits.
///
/// The output lease and host tasks own completion storage. RequestPool accepts
/// the resolved progress only after every output in the batch is ready.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct PendingOutput {
    key: RequestKey,
    id: CallId,
    progress: RequestProgress,
    accepted: Option<RequestProgress>,
    status: CallStatus,
    error_code: Option<ErrorCode>,
    buffer: Option<Py<OutputBuffer>>,
    row: usize,
    #[pyo3(get)]
    call: Py<PyAny>,
    #[pyo3(get)]
    request: Py<Request>,
    #[pyo3(get)]
    token: Py<PyAny>,
    #[pyo3(get)]
    latent: Py<PyAny>,
    #[pyo3(get)]
    host: Py<PyAny>,
    #[pyo3(get, set)]
    finish_flags: Py<PyAny>,
    #[pyo3(get, set)]
    product_generations: Py<PyTuple>,
    #[pyo3(get)]
    tensor_exports: Py<PyDict>,
    #[pyo3(get)]
    cache_exports: Py<PyDict>,
    #[pyo3(get)]
    exported_locators: Py<PyList>,
    #[pyo3(get, set)]
    cache_publication: Option<Py<PyAny>>,
    #[pyo3(get, set)]
    cache_installation: Option<Py<PyAny>>,
    #[pyo3(get)]
    device_reads: Py<PyList>,
    #[pyo3(get)]
    feature_reads: Py<PyList>,
    #[pyo3(get)]
    writes: Py<PyList>,
    #[pyo3(get, set)]
    predicate: Option<Py<PyAny>>,
    #[pyo3(get, set)]
    token_write: Option<Py<PyAny>>,
    #[pyo3(get, set)]
    transition_write: Option<Py<PyAny>>,
    #[pyo3(get, set)]
    completion_write: Option<Py<PyAny>>,
    #[pyo3(get, set)]
    producer_write: Option<Py<PyAny>>,
    #[pyo3(get, set)]
    kv_output: Option<Py<PyAny>>,
    #[pyo3(get, set)]
    products: Py<PyTuple>,
    #[pyo3(get, set)]
    _reports_output: bool,
    #[pyo3(get)]
    value: Option<Py<PyAny>>,
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
        let numerical = py.import("uniserve_worker.execution.output")?;
        let native_request = request.borrow(py);
        let previous = native_request
            .request
            .progress()
            .map_err(|error| native_error(py, error))?;
        let coordinates = call.bind(py).getattr("coordinates")?;
        let update = py
            .import("uniserve_worker._uniserve_ipc")?
            .getattr("LatentUpdate")?
            .call1((native_request.request.slot(),))?;
        drop(native_request);

        Ok(Self {
            key: request_key(&call.bind(py).getattr("request_key")?)?,
            id: call_id(&call.bind(py).getattr("call_id")?)?,
            progress: RequestProgress {
                logical_position: coordinates.getattr("logical_position")?.extract()?,
                flow_step: coordinates.getattr("flow_step")?.extract()?,
                kv_visible_len: coordinates.getattr("kv_visible_len")?.extract()?,
                kv_computed_len: coordinates.getattr("kv_computed_len")?.extract()?,
                ..previous
            },
            accepted: None,
            status: CallStatus::Ok,
            error_code: None,
            buffer: Some(buffer),
            row,
            call,
            request,
            token: numerical.getattr("TokenResult")?.call0()?.unbind(),
            latent: numerical
                .getattr("LatentResult")?
                .call1((update,))?
                .unbind(),
            host: numerical.getattr("HostResult")?.call0()?.unbind(),
            finish_flags: py
                .import("uniserve_worker.protocol.output")?
                .getattr("FinishFlags")?
                .call0()?
                .unbind(),
            product_generations: PyTuple::empty(py).unbind(),
            tensor_exports: PyDict::new(py).unbind(),
            cache_exports: PyDict::new(py).unbind(),
            exported_locators: PyList::empty(py).unbind(),
            cache_publication: None,
            cache_installation: None,
            device_reads: PyList::empty(py).unbind(),
            feature_reads: PyList::empty(py).unbind(),
            writes: PyList::empty(py).unbind(),
            predicate: None,
            token_write: None,
            transition_write: None,
            completion_write: None,
            producer_write: None,
            kv_output: None,
            products: PyTuple::empty(py).unbind(),
            _reports_output: true,
            value: None,
        })
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
    fn progress(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        progress_to_py(py, self.progress).map(Bound::unbind)
    }

    #[setter]
    fn set_progress(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.progress = progress_from_py(value)?;
        Ok(())
    }

    #[getter]
    fn status(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        call_enum(py, "CallStatus", &self.status)
    }

    #[setter]
    fn set_status(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.status = pythonize::depythonize(&value.getattr("value")?)?;
        Ok(())
    }

    #[getter]
    fn error_code(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.error_code
            .as_ref()
            .map(|code| call_enum(py, "ErrorCode", code))
            .transpose()
    }

    #[setter]
    fn set_error_code(&mut self, value: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.error_code = value
            .map(|value| pythonize::depythonize(&value.getattr("value")?).map_err(PyErr::from))
            .transpose()?;
        Ok(())
    }

    #[getter(_buffer)]
    fn buffer(&self, py: Python<'_>) -> Option<Py<OutputBuffer>> {
        self.buffer.as_ref().map(|buffer| buffer.clone_ref(py))
    }

    fn release_execution_references(&mut self, py: Python<'_>) -> PyResult<()> {
        self.writes.bind(py).call_method0("clear")?;
        self.tensor_exports.bind(py).clear();
        self.cache_exports.bind(py).clear();
        self.exported_locators.bind(py).call_method0("clear")?;
        self.cache_publication = None;
        self.cache_installation = None;

        let latent = self.latent.bind(py);
        latent.getattr("exports")?.call_method0("clear")?;
        latent.setattr("input_params", py.None())?;
        latent.setattr("staging", py.None())?;
        latent.setattr("imported", false)?;

        self.predicate = None;
        self.token_write = None;
        self.transition_write = None;
        self.completion_write = None;
        self.producer_write = None;
        let token = self.token.bind(py);
        for field in [
            "sampled",
            "runtime_penalty_base",
            "runtime_prompt_logits",
            "runtime_cache_length",
        ] {
            token.setattr(field, py.None())?;
        }
        token.setattr("runtime_logical_position", 0)?;
        token.setattr("runtime_sampling_position", 0)?;
        Ok(())
    }

    pub(super) fn ready(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let py = slf.py();
        let (buffer, host) = {
            let this = slf.borrow();
            if this.value.is_some() {
                return Ok(true);
            }
            (this.buffer(py), this.host.clone_ref(py))
        };
        let Some(buffer) = buffer else {
            return Ok(false);
        };
        // Readiness may dispatch completion observers. No result borrow may
        // span a callback, including the output buffer's notification.
        if !buffer.get().ready(py)? {
            return Ok(false);
        }
        Ok(host_tasks(host.bind(py))?
            .iter()
            .all(|task| task.borrow(py).done()))
    }

    pub(super) fn materialize(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if let Some(value) = &slf.borrow().value {
            return Ok(value.clone_ref(py));
        }
        if !Self::ready(slf)? {
            return Err(PyRuntimeError::new_err(
                "completion was resolved before query-ready",
            ));
        }
        let (token, host, buffer, reports, mut status, mut error_code, mut progress, request) = {
            let this = slf.borrow();
            (
                this.token.clone_ref(py),
                this.host.clone_ref(py),
                this.buffer(py).ok_or_else(lost_buffer)?,
                this._reports_output,
                this.status,
                this.error_code,
                this.progress,
                this.request.clone_ref(py),
            )
        };
        let accepted_parent = request
            .borrow(py)
            .request
            .progress()
            .map_err(|error| native_error(py, error))?;
        let token = token.bind(py);
        let host = host.bind(py);
        let mut tokens: Vec<i64> = token.getattr("committed_tokens")?.extract()?;
        let mut suppressed = status == CallStatus::Predicated;
        let mut skipped = false;
        let scores = PyDict::new(py);

        if !suppressed {
            let decoded = resolve_host(py, host, reports)
                .and_then(|()| decode_scores(py, token, buffer.get(), &scores));
            match decoded {
                Err(error) if error.is_instance_of::<PyException>(py) => {
                    log_failure(slf, &error)?;
                    status = CallStatus::Error;
                    error_code = Some(ErrorCode::ComputeError);
                    suppressed = true;
                }
                Err(error) => return Err(error),
                Ok(canvas) => {
                    if let Some((outcome, canvas)) = canvas {
                        // Canvas completion is [step outcome | token ids]. A
                        // step behind a stopped block was a device-side no-op.
                        skipped = outcome == STEP_SKIPPED;
                        tokens = if outcome == STEP_STOPPED {
                            canvas
                        } else {
                            Vec::new()
                        };
                    }
                    if let Some((offset, extent, row)) = token
                        .getattr("sampling_range")?
                        .extract::<Option<(usize, usize, usize)>>()?
                    {
                        let values = buffer.get().read_tokens(py, offset, extent)?;
                        let values = values.bind(py).cast::<PyTuple>()?;
                        let count = extent / SAMPLING_FIELDS;

                        // Rows share the cached [valid | active | token | accepted]
                        // column. Read this row only, without converting the
                        // entire sampled batch again for each result.
                        // The predicate wins over an invalid distribution in
                        // an inactive graph row: that row never sampled.
                        if values.get_item(count + row)?.extract::<i64>()? == 0 {
                            status = CallStatus::Predicated;
                            suppressed = true;
                        } else if values.get_item(row)?.extract::<i64>()? == 0 {
                            status = CallStatus::Error;
                            error_code = Some(ErrorCode::InvalidCall);
                            suppressed = true;
                        } else {
                            tokens = accepted_tokens(
                                token,
                                values.get_item(count * 2 + row)?.extract()?,
                                values.get_item(count * 3 + row)?.extract()?,
                            )?;
                            accept_speculation(token, tokens.len(), &mut progress)?;
                        }
                    }
                }
            }
        }

        if skipped && status == CallStatus::Ok {
            status = CallStatus::Predicated;
            suppressed = true;
        }
        if status == CallStatus::Predicated {
            progress = accepted_parent;
            error_code = None;
        }
        if suppressed {
            tokens.clear();
        }

        let row = slf.borrow().row;
        buffer.get().observe(py, row)?;
        let timing = buffer.get().timing(py)?;
        slf.borrow_mut().buffer = None;

        let protocol = py.import("uniserve_worker.protocol.output")?;
        let kwargs = PyDict::new(py);
        let this = slf.borrow();
        kwargs.set_item("request_key", this.request_key(py)?)?;
        kwargs.set_item("call_id", this.call_id(py)?)?;
        kwargs.set_item("status", call_enum(py, "CallStatus", &status)?)?;
        kwargs.set_item("kind", this.kind(py)?)?;
        kwargs.set_item(
            "error_code",
            error_code
                .as_ref()
                .map(|code| call_enum(py, "ErrorCode", code))
                .transpose()?,
        )?;
        kwargs.set_item(
            "timing_counters",
            protocol.getattr("TimingCounters")?.call1(timing)?,
        )?;
        kwargs.set_item("position", progress.logical_position)?;
        kwargs.set_item("kv_visible_len", progress.kv_visible_len)?;
        kwargs.set_item("kv_computed_len", progress.kv_computed_len)?;
        kwargs.set_item("num_completed_steps", progress.flow_step)?;
        kwargs.set_item("committed_tokens", PyTuple::new(py, tokens)?)?;
        kwargs.set_item("media_output", host.getattr("media")?)?;
        if suppressed {
            kwargs.set_item("product_generations", PyTuple::empty(py))?;
            kwargs.set_item("finish_flags", protocol.getattr("FinishFlags")?.call0()?)?;
        } else {
            kwargs.set_item("product_generations", &this.product_generations)?;
            kwargs.set_item("finish_flags", &this.finish_flags)?;
            kwargs.set_item("kv_output", &this.kv_output)?;
            if reports {
                kwargs.update(scores.as_mapping())?;
            }
        }
        drop(this);

        let value = protocol.getattr("RequestOutput")?.call((), Some(&kwargs))?;
        value.call_method0("validate")?;
        host.setattr("tasks", PyTuple::empty(py))?;
        let mut this = slf.borrow_mut();
        this.accepted = Some(
            if matches!(status, CallStatus::Predicated | CallStatus::Error) {
                accepted_parent
            } else {
                progress
            },
        );
        this.status = status;
        this.error_code = error_code;
        this.value = Some(value.clone().unbind());
        Ok(value.unbind())
    }

    fn abandon(slf: &Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        let (host, buffer, row) = {
            let mut this = slf.borrow_mut();
            (this.host.clone_ref(py), this.buffer.take(), this.row)
        };
        let host = host.bind(py);
        let tasks = host_tasks(host)?;
        host.setattr("tasks", PyTuple::empty(py))?;
        host.setattr("finish", py.None())?;

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
        visit.call(&self.token)?;
        visit.call(&self.latent)?;
        visit.call(&self.host)?;
        visit.call(&self.finish_flags)?;
        visit.call(&self.product_generations)?;
        visit.call(&self.tensor_exports)?;
        visit.call(&self.cache_exports)?;
        visit.call(&self.exported_locators)?;
        visit.call(&self.cache_publication)?;
        visit.call(&self.cache_installation)?;
        visit.call(&self.device_reads)?;
        visit.call(&self.feature_reads)?;
        visit.call(&self.writes)?;
        visit.call(&self.predicate)?;
        visit.call(&self.token_write)?;
        visit.call(&self.transition_write)?;
        visit.call(&self.completion_write)?;
        visit.call(&self.producer_write)?;
        visit.call(&self.kv_output)?;
        visit.call(&self.products)?;
        visit.call(&self.value)
    }

    fn __clear__(&mut self, py: Python<'_>) {
        self.token = py.None();
        self.latent = py.None();
        self.host = py.None();
    }
}

impl PendingOutput {
    pub(super) fn submit_host_tasks(slf: &Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        let host = {
            let this = slf.borrow();
            if this.value.is_some() {
                return Ok(());
            }
            this.host.clone_ref(py)
        };
        for task in host_tasks(host.bind(py))? {
            task.borrow(py).submit_if_ready(py)?;
        }
        Ok(())
    }

    pub(super) fn accept(&self, py: Python<'_>, requests: &mut RequestPool) -> PyResult<()> {
        requests
            .pool
            .apply_result(self.key, self.id, self.status, self.accepted)
            .map_err(|error| native_error(py, error))
    }
}

fn host_tasks(host: &Bound<'_, PyAny>) -> PyResult<Vec<Py<HostTask>>> {
    host.getattr("tasks")?.extract()
}

fn resolve_host(py: Python<'_>, host: &Bound<'_, PyAny>, reports: bool) -> PyResult<()> {
    let results = host_tasks(host)?
        .iter()
        .map(|task| task.borrow(py).result(py, None))
        .collect::<PyResult<Vec<_>>>()?;
    let finish = host.getattr("finish")?;
    host.setattr("finish", py.None())?;
    if !finish.is_none() {
        finish.call1((PyTuple::new(py, results)?,))?;
        return Ok(());
    }

    let protocol = py.import("uniserve_worker.protocol.output")?;
    let media_type = protocol.getattr("MediaOutput")?;
    for result in results {
        let mut result = result.into_bound(py);
        if reports && let Ok(bytes) = result.cast::<PyBytes>() {
            let name = py
                .import("uniserve_worker.media.storage")?
                .call_method1("publish_media_bytes", (&result,))?;
            let artifact = protocol.getattr("PosixShmArtifact")?.call1((name,))?;
            result = media_type.call1((artifact, bytes.as_bytes().len()))?;
        }
        if result.is_instance(&media_type)? {
            if !host.getattr("media")?.is_none() {
                return Err(PyRuntimeError::new_err(
                    "completion produced more than one media output",
                ));
            }
            host.setattr("media", result)?;
        }
    }
    Ok(())
}

/// Decode copied score columns and an optional canvas completion. Tensor
/// storage is already query-ready; these routines never wait for device work.
fn decode_scores(
    py: Python<'_>,
    token: &Bound<'_, PyAny>,
    buffer: &OutputBuffer,
    scores: &Bound<'_, PyDict>,
) -> PyResult<Option<(i64, Vec<i64>)>> {
    if let Some(span) = token.getattr("logprob_range")?.extract()? {
        let values = buffer.logprob_values(py, span)?;
        scores.set_item("sampled_logprob", values.bind(py).get_item(0)?)?;
        scores.set_item("top_logprobs", values.bind(py).get_item(1)?)?;
    }
    let spans: Vec<(usize, usize, usize)> = token.getattr("prompt_logprob_ranges")?.extract()?;
    let prompt = spans
        .into_iter()
        .map(|span| {
            buffer
                .logprob_values(py, span)?
                .bind(py)
                .get_item(1)
                .map(Bound::unbind)
        })
        .collect::<PyResult<Vec<_>>>()?;
    scores.set_item("prompt_logprobs", PyTuple::new(py, prompt)?)?;

    if let Some((offset, count)) = token.getattr("candidate_range")?.extract()? {
        let words: Vec<i64> = buffer.read_tokens(py, offset, count)?.extract(py)?;
        // Candidate scores are float32 bits sign-extended from int32.
        let candidates = words.into_iter().map(|word| f32::from_bits(word as u32));
        scores.set_item("candidate_logprobs", PyTuple::new(py, candidates)?)?;
    }
    if let Some((offset, count)) = token.getattr("canvas_range")?.extract()? {
        let words: Vec<i64> = buffer.read_tokens(py, offset, count)?.extract(py)?;
        let Some((&outcome, tokens)) = words.split_first() else {
            return Err(PyRuntimeError::new_err(
                "canvas completion has no step outcome",
            ));
        };
        return Ok(Some((outcome, tokens.to_vec())));
    }
    Ok(None)
}

fn accepted_tokens(token: &Bound<'_, PyAny>, sampled: i64, accepted: i64) -> PyResult<Vec<i64>> {
    let draft: Option<Vec<i64>> = token.getattr("draft_tokens")?.extract()?;
    let Some(mut draft) = draft.filter(|draft| !draft.is_empty()) else {
        return Ok(vec![sampled]);
    };
    if accepted < 0 || accepted as usize > draft.len() {
        return Err(PyRuntimeError::new_err(
            "speculative acceptance count is outside the draft span",
        ));
    }
    let accepted = accepted as usize;
    draft.truncate(accepted);
    let terminal: Option<usize> = token.getattr("terminal_prefix")?.extract()?;
    if terminal.is_none_or(|terminal| accepted < terminal) {
        draft.push(sampled);
    }
    Ok(draft)
}

fn accept_speculation(
    token: &Bound<'_, PyAny>,
    accepted: usize,
    progress: &mut RequestProgress,
) -> PyResult<()> {
    let draft = token.getattr("draft_tokens")?;
    if draft.is_none() {
        return Ok(());
    }
    let visible = token.getattr("base_kv_visible")?.extract::<u64>()? + accepted as u64;
    let initialized: u64 = token.getattr("initialized_kv")?.extract()?;
    if accepted > draft.len()? + 1
        || progress.kv_computed_len != initialized
        || visible > initialized
    {
        return Err(PyRuntimeError::new_err(
            "speculative acceptance exceeds initialized KV state",
        ));
    }

    // Rejected drafts stay initialized but invisible. Advance all accepted
    // coordinates together, using the prefix that verification started from.
    progress.logical_position =
        token.getattr("base_logical_position")?.extract::<u64>()? + accepted as u64;
    progress.rng_counter = token.getattr("base_rng_counter")?.extract::<u64>()? + accepted as u64;
    progress.kv_visible_len = visible;
    Ok(())
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
