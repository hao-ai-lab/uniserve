//! Admission, request dependencies, accepted progress, and slot retirement.
//!
//! A request is shared with numerical output observers through its Python
//! reference, so finishing or evicting its pool slot does not invalidate those
//! observers. Only the pool mutates lifecycle state. Numerical diffusion state
//! is released after its host staging and submitted calls have finished.

use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;
use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use uniserve_core::{CallId, RequestId};
use uniserve_worker_ipc::RequestKey;

/// One admitted request epoch, including its borrowed numerical state.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct Request {
    key: RequestKey,
    slot: usize,
    admission: Py<PyAny>,
    progress: Py<PyAny>,
    accepted_call: CallId,
    state_call: CallId,
    // Insertion order is commit order, which need not equal call-id order.
    pending: IndexMap<CallId, bool>,
    #[pyo3(get, set)]
    diffusion: Option<Py<PyAny>>,
    #[pyo3(get)]
    closed: bool,
    #[pyo3(get)]
    retired: bool,
}

#[pymethods]
impl Request {
    #[getter]
    fn request_id(&self) -> u64 {
        self.key.request_id.0
    }

    #[getter]
    fn request_key<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.admission.bind(py).getattr("request_key")
    }

    #[getter]
    fn request_pool_idx(&self) -> usize {
        self.slot
    }

    #[getter]
    fn admission(&self, py: Python<'_>) -> Py<PyAny> {
        self.admission.clone_ref(py)
    }

    #[getter]
    fn sampling<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let generation = self.admission.bind(py).getattr("generation")?;
        if generation.is_none() {
            Ok(py.None().into_bound(py))
        } else {
            generation.getattr("sampling")
        }
    }

    #[getter]
    fn image<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.admission.bind(py).getattr("image")
    }

    #[getter]
    fn negative_token_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.generation_tokens(py, "negative_token_ids")
    }

    #[getter]
    fn finish_token_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.generation_tokens(py, "finish_token_ids")
    }

    #[getter]
    fn accepted_progress(&self, py: Python<'_>) -> Py<PyAny> {
        self.progress.clone_ref(py)
    }

    #[getter]
    fn prompt_logits_ready(&self, py: Python<'_>) -> PyResult<bool> {
        self.progress
            .bind(py)
            .getattr("prompt_logits_ready")?
            .extract()
    }

    #[getter]
    fn rng_counter(&self, py: Python<'_>) -> PyResult<u64> {
        self.progress.bind(py).getattr("rng_counter")?.extract()
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.admission)?;
        visit.call(&self.progress)?;
        visit.call(&self.diffusion)
    }

    fn __clear__(&mut self) {
        self.diffusion = None;
    }
}

impl Request {
    fn generation_tokens<'py>(&self, py: Python<'py>, field: &str) -> PyResult<Bound<'py, PyAny>> {
        let generation = self.admission.bind(py).getattr("generation")?;
        if generation.is_none() {
            Ok(PyTuple::empty(py).into_any())
        } else {
            generation.getattr(field)
        }
    }
}

/// Own scheduler-assigned request slots and accept completed call progress.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct RequestPool {
    #[pyo3(get)]
    max_request_pool_size: usize,
    // Slot zero is graph padding. A slot contains only the resident request id;
    // requests and numerical observers share the same native Request object.
    slots: Vec<Option<u64>>,
    requests: HashMap<u64, Py<Request>>,
    #[pyo3(get)]
    storage: Py<PyAny>,
    closed: bool,
}

#[pymethods]
impl RequestPool {
    #[new]
    #[pyo3(signature = (max_request_pool_size, *, state_buffers=None, device=None))]
    fn new(
        py: Python<'_>,
        max_request_pool_size: isize,
        state_buffers: Option<Py<PyAny>>,
        device: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        if max_request_pool_size < 1 {
            return Err(PyValueError::new_err(
                "request-pool capacity must be positive",
            ));
        }
        let size = max_request_pool_size as usize;
        let kwargs = PyDict::new(py);
        kwargs.set_item("state_buffers", state_buffers)?;
        if let Some(device) = device {
            kwargs.set_item("device", device)?;
        } else {
            kwargs.set_item("device", "cpu")?;
        }
        let storage = py
            .import("uniserve_worker.storage.request_slots")?
            .getattr("RequestSlots")?
            .call((size,), Some(&kwargs))?
            .unbind();

        Ok(Self {
            max_request_pool_size: size,
            slots: vec![None; size + 1],
            requests: HashMap::new(),
            storage,
            closed: false,
        })
    }

    /// Release drained request views before the backing storage owner closes.
    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        for request in self.requests.values() {
            let mut request = request.borrow_mut(py);
            request.diffusion = None;
            request.pending.clear();
        }
        self.requests.clear();
        self.slots.fill(None);
        self.storage.bind(py).call_method0("close")?;
        Ok(())
    }

    fn get(&self, py: Python<'_>, request_id: u64) -> PyResult<Py<Request>> {
        self.requests
            .get(&request_id)
            .map(|request| request.clone_ref(py))
            .ok_or_else(|| invalid(py, format!("unknown request {request_id}")))
    }

    fn peek(&self, py: Python<'_>, request_id: u64) -> Option<Py<Request>> {
        self.requests
            .get(&request_id)
            .map(|request| request.clone_ref(py))
    }

    fn request_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let mut ids: Vec<_> = self.requests.keys().copied().collect();
        ids.sort_unstable();
        PyTuple::new(py, ids)
    }

    fn has_open_requests(&self, py: Python<'_>) -> bool {
        self.requests.values().any(|request| {
            let request = request.borrow(py);
            !request.closed && !request.retired
        })
    }

    /// Validate epoch and slot ownership before constructing numerical inputs.
    fn bind_calls<'py>(
        &self,
        py: Python<'py>,
        calls: Vec<Bound<'py, PyAny>>,
        request_pool_indices: Vec<isize>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        if calls.len() != request_pool_indices.len() {
            return Err(invalid(
                py,
                "request-pool indices are not aligned with calls",
            ));
        }
        let mut keys = HashSet::new();
        let mut slots = HashSet::new();
        let mut requests = Vec::with_capacity(calls.len());
        for (call, slot) in calls.iter().zip(request_pool_indices) {
            let key = request_key(&call.getattr("request_key")?)?;
            let id = call_id(&call.getattr("call_id")?)?;
            let slot = self.validate_slot(py, slot)?;
            if !keys.insert(key) {
                return Err(invalid(py, "a batch repeats a request"));
            }
            if !slots.insert(slot) {
                return Err(invalid(py, "a batch repeats a request-pool index"));
            }
            let request = self.get(py, key.request_id.0)?;
            {
                let request = request.borrow(py);
                if request.key != key {
                    return Err(invalid(py, format!("call {id:?} has a stale request key")));
                }
                if request.slot != slot || self.slots[slot] != Some(key.request_id.0) {
                    return Err(invalid(py, format!("call {id:?} has a stale request slot")));
                }
                if request.closed {
                    return Err(invalid(py, format!("call {id:?} targets a closed request")));
                }
                if request.pending.contains_key(&id) {
                    return Err(invalid(py, "request call is already executing"));
                }
            }
            requests.push(request);
        }
        PyTuple::new(py, requests)
    }

    /// Check every call before any resource or request state becomes visible.
    fn validate_pending(&self, py: Python<'_>, calls: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        self.pending_calls(py, calls).map(|_| ())
    }

    fn add_pending(&mut self, py: Python<'_>, calls: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        let calls = self.pending_calls(py, calls)?;
        for (request, id, advances) in calls {
            request.borrow_mut(py).pending.insert(id, advances);
        }
        Ok(())
    }

    /// Independent media encoders have no state predecessor. State consumers
    /// follow the last committed state call, or the accepted admission root.
    fn predecessors<'py>(
        &self,
        py: Python<'py>,
        calls: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        let call_type = py
            .import("uniserve_worker.protocol.identity")?
            .getattr("CallId")?;
        for call in calls {
            let key = request_key(&call.getattr("request_key")?)?;
            let advances: bool = call.getattr("advances_state")?.extract()?;
            let predecessor = match self.requests.get(&key.request_id.0) {
                Some(request) => {
                    let request = request.borrow(py);
                    let media = !request.admission.bind(py).getattr("diffusion")?.is_none();
                    if request.key != key || (media && !advances) {
                        None
                    } else {
                        Some(
                            request
                                .pending
                                .iter()
                                .rev()
                                .find_map(|(&id, &advances)| advances.then_some(id))
                                .unwrap_or(request.state_call),
                        )
                    }
                }
                None => None,
            };
            let value = match predecessor {
                Some(id) => call_type.call1((id.batch_id, id.request_index))?,
                None => py.None().into_bound(py),
            };
            result.set_item(call.getattr("call_id")?, value)?;
        }
        Ok(result)
    }

    /// Ignore stale epochs and duplicate results; late results may release a
    /// pending call but cannot roll accepted progress back.
    fn apply_result(&mut self, py: Python<'_>, result: &Bound<'_, PyAny>) -> PyResult<()> {
        let key = request_key(&result.getattr("request_key")?)?;
        let id = call_id(&result.getattr("call_id")?)?;
        let Some(request) = self.requests.get(&key.request_id.0) else {
            return Ok(());
        };
        let mut request = request.borrow_mut(py);
        if request.key != key {
            return Ok(());
        }
        // Decode the whole result before consuming its pending identity.
        let status: String = result.getattr("status")?.getattr("value")?.extract()?;
        let progress = result.getattr("progress")?;
        let Some(advances) = request.pending.shift_remove(&id) else {
            return Ok(());
        };
        if !progress.is_none() && id > request.accepted_call {
            request.progress = progress.unbind();
            request.accepted_call = id;
        }
        if status == "ok" && advances && id > request.state_call {
            request.state_call = id;
        }
        if status == "error" {
            request.closed = true;
        }
        Ok(())
    }

    fn cancel_calls(&mut self, py: Python<'_>, calls: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        for call in calls {
            let key = request_key(&call.getattr("request_key")?)?;
            let id = call_id(&call.getattr("call_id")?)?;
            if let Some(request) = self.requests.get(&key.request_id.0) {
                let mut request = request.borrow_mut(py);
                if request.key == key {
                    request.pending.shift_remove(&id);
                    request.closed = true;
                }
            }
        }
        Ok(())
    }

    /// Admit immutable parameters at the scheduler's exact slot. Only retired
    /// epochs may be evicted; an identical admission does not reset its tensors.
    fn start(&mut self, py: Python<'_>, admission: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
        let key = request_key(&admission.getattr("request_key")?)?;
        let slot = self.validate_slot(py, admission.getattr("request_pool_idx")?.extract()?)?;
        let base = self.requests.get(&key.request_id.0);
        if base.is_some_and(|request| {
            let request = request.borrow(py);
            request.retired && request.key != key
        }) {
            self.drop_request(py, key.request_id.0);
        }
        if let Some(id) = self.slots[slot] {
            let occupant = self.get(py, id)?;
            let occupant = occupant.borrow(py);
            if occupant.retired && occupant.key != key {
                self.drop_request(py, id);
            }
        }
        if let Some(base) = self.requests.get(&key.request_id.0) {
            let base = base.borrow(py);
            if !base.admission.bind(py).eq(admission)? || self.slots[slot] != Some(key.request_id.0)
            {
                return Err(invalid(
                    py,
                    "request admission conflicts with resident state",
                ));
            }
            return Ok(None);
        }
        if self.slots[slot].is_some() {
            return Err(invalid(
                py,
                format!("request-pool index {slot} is occupied"),
            ));
        }
        let generation = admission.getattr("generation")?;
        let prefix = if generation.is_none() {
            0
        } else {
            generation.getattr("initial_position")?.extract::<u64>()?
        };
        let kwargs = PyDict::new(py);
        kwargs.set_item("logical_position", prefix)?;
        kwargs.set_item("kv_visible_len", prefix)?;
        kwargs.set_item("kv_computed_len", prefix)?;
        let progress = py
            .import("uniserve_worker.execution.request")?
            .getattr("RequestProgress")?
            .call((), Some(&kwargs))?
            .unbind();
        let request = Py::new(
            py,
            Request {
                key,
                slot,
                admission: admission.clone().unbind(),
                progress,
                accepted_call: CallId::default(),
                state_call: CallId::default(),
                pending: IndexMap::new(),
                diffusion: None,
                closed: false,
                retired: false,
            },
        )?;
        self.requests.insert(key.request_id.0, request);
        self.slots[slot] = Some(key.request_id.0);
        Ok(Some(slot))
    }

    fn finish(&mut self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<()> {
        let key = request_key(key)?;
        if let Some(request) = self.requests.get(&key.request_id.0) {
            let mut request = request.borrow_mut(py);
            if request.key == key {
                request.closed = true;
            }
        }
        Ok(())
    }

    /// Free commands belong to the buffer owners, not to request slots.
    fn apply_commands<'py>(
        &mut self,
        py: Python<'py>,
        commands: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let protocol = py.import("uniserve_worker.protocol.batch")?;
        let start = protocol.getattr("Start")?;
        let finish = protocol.getattr("Finish")?;
        let mut slots = Vec::new();
        for command in commands {
            if command.is_instance(&start)? {
                if let Some(slot) = self.start(py, &command.getattr("request")?)? {
                    slots.push(slot);
                }
            } else if command.is_instance(&finish)? {
                self.finish(py, &command.getattr("request_key")?)?;
            }
        }
        PyTuple::new(py, slots)
    }

    fn retirement_ready(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        let key = request_key(key)?;
        Ok(self.requests.get(&key.request_id.0).is_none_or(|request| {
            let request = request.borrow(py);
            request.key != key || request.pending.is_empty()
        }))
    }

    #[pyo3(name = "drop")]
    fn drop_request(&mut self, py: Python<'_>, request_id: u64) {
        if let Some(request) = self.requests.remove(&request_id) {
            self.slots[request.borrow(py).slot] = None;
        }
    }

    /// Wait for host staging before releasing diffusion state. Device readers
    /// are drained by the executor before it asks the pool to retire a request.
    fn retire(&mut self, py: Python<'_>, request_id: u64) -> PyResult<()> {
        let request = self.get(py, request_id)?;
        let staging = {
            let request = request.borrow(py);
            if request.retired {
                return Ok(());
            }
            if !request.closed || !request.pending.is_empty() {
                return Err(PyRuntimeError::new_err(
                    "request retirement requires closed, completed execution",
                ));
            }
            match &request.diffusion {
                Some(diffusion) => {
                    let slot = diffusion.bind(py).getattr("slot")?;
                    if slot.is_none() {
                        None
                    } else {
                        Some(slot.getattr("staging")?)
                    }
                }
                None => None,
            }
        };
        if let Some(staging) = staging.filter(|staging| !staging.is_none()) {
            // A failed staging operation still owns its destination until it
            // finishes. wait(), unlike result(), does not rethrow its error.
            py.import("concurrent.futures")?
                .getattr("wait")?
                .call1(((staging,),))?;
        }
        let mut request = request.borrow_mut(py);
        request.diffusion = None;
        request.retired = true;
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.storage)?;
        for request in self.requests.values() {
            visit.call(request)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.requests.clear();
        self.slots.fill(None);
    }
}

impl RequestPool {
    fn validate_slot(&self, py: Python<'_>, slot: isize) -> PyResult<usize> {
        if self.closed {
            return Err(PyRuntimeError::new_err("request pool is closed"));
        }
        if slot < 1 || slot as usize > self.max_request_pool_size {
            return Err(invalid(
                py,
                format!("request-pool index {slot} exceeds capacity"),
            ));
        }
        Ok(slot as usize)
    }

    fn pending_calls(
        &self,
        py: Python<'_>,
        calls: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Vec<(Py<Request>, CallId, bool)>> {
        let mut seen = HashSet::new();
        let mut pending = Vec::with_capacity(calls.len());
        for call in calls {
            let key = request_key(&call.getattr("request_key")?)?;
            let id = call_id(&call.getattr("call_id")?)?;
            let advances = call.getattr("advances_state")?.extract()?;
            let request = self.get(py, key.request_id.0)?;
            {
                let request = request.borrow(py);
                if request.key != key {
                    return Err(PyRuntimeError::new_err(
                        "request commit lost its admitted slot",
                    ));
                }
                if request.pending.contains_key(&id) || !seen.insert((key, id)) {
                    return Err(PyRuntimeError::new_err(
                        "request commit repeats an executing call",
                    ));
                }
            }
            pending.push((request, id, advances));
        }
        Ok(pending)
    }
}

fn request_key(value: &Bound<'_, PyAny>) -> PyResult<RequestKey> {
    Ok(RequestKey::new(
        value.getattr("engine_id")?.extract()?,
        RequestId(value.getattr("request_id")?.extract()?),
        value.getattr("request_epoch")?.extract()?,
    ))
}

fn call_id(value: &Bound<'_, PyAny>) -> PyResult<CallId> {
    Ok(CallId::new(
        value.getattr("batch_id")?.extract()?,
        value.getattr("request_index")?.extract()?,
    ))
}

fn invalid(py: Python<'_>, message: impl Into<String>) -> PyErr {
    match py.import("uniserve_worker.errors").and_then(|module| {
        module
            .getattr("invalid_descriptor")?
            .call1((message.into(),))
    }) {
        Ok(error) => PyErr::from_value(error),
        Err(error) => error,
    }
}
