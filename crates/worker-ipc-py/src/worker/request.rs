//! Numerical request views backed by the native request pool.

use std::sync::Arc;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use uniserve_core::CallId;
use uniserve_worker::{Request as NativeRequest, RequestPool as NativePool, RequestProgress};
use uniserve_worker_ipc::{CallStatus, RequestKey};

use super::error::{invalid, native_error};
use super::protocol::{call_id, new_request, request_key};

/// Retain numerical parameters and tensors alongside a native request epoch.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct Request {
    pub(super) request: Arc<NativeRequest>,
    #[pyo3(get)]
    admission: Py<PyAny>,
    #[pyo3(get, set)]
    pub(super) diffusion: Option<Py<PyAny>>,
}

#[pymethods]
impl Request {
    #[getter]
    fn request_id(&self) -> u64 {
        self.request.key().request_id.0
    }

    #[getter]
    fn request_key<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.admission.bind(py).getattr("request_key")
    }

    #[getter]
    fn request_pool_idx(&self) -> usize {
        self.request.slot()
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
    fn negative_token_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            self.request
                .admission()
                .ar
                .as_ref()
                .map_or(&[][..], |ar| ar.negative_token_ids.as_slice()),
        )
    }

    #[getter]
    fn finish_token_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            self.request
                .admission()
                .ar
                .as_ref()
                .map_or(&[][..], |ar| ar.finish_token_ids.as_slice()),
        )
    }

    #[getter]
    fn accepted_progress<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let progress = self
            .request
            .progress()
            .map_err(|error| native_error(py, error))?;
        progress_to_py(py, progress)
    }

    #[getter]
    fn prompt_logits_ready(&self, py: Python<'_>) -> PyResult<bool> {
        self.request
            .progress()
            .map(|progress| progress.prompt_logits_ready)
            .map_err(|error| native_error(py, error))
    }

    #[getter]
    fn rng_counter(&self, py: Python<'_>) -> PyResult<u64> {
        self.request
            .progress()
            .map(|progress| progress.rng_counter)
            .map_err(|error| native_error(py, error))
    }

    #[getter]
    fn closed(&self, py: Python<'_>) -> PyResult<bool> {
        self.request
            .closed()
            .map_err(|error| native_error(py, error))
    }

    #[getter]
    fn retired(&self, py: Python<'_>) -> PyResult<bool> {
        self.request
            .retired()
            .map_err(|error| native_error(py, error))
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.admission)?;
        visit.call(&self.diffusion)
    }

    fn __clear__(&mut self) {
        self.diffusion = None;
    }
}

/// Bind the native pool to its Python numerical storage and slot views.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct RequestPool {
    pub(super) pool: NativePool,
    // Cached language views carry numerical state, never a second lifecycle.
    views: Vec<Option<Py<Request>>>,
    #[pyo3(get)]
    storage: Py<PyAny>,
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
        let pool = NativePool::new(size).map_err(|error| native_error(py, error))?;
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
            pool,
            views: (0..=size).map(|_| None).collect(),
            storage,
        })
    }

    #[getter]
    fn max_request_pool_size(&self) -> usize {
        self.pool.capacity()
    }

    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        for request in self.views.iter().flatten() {
            request.borrow_mut(py).diffusion = None;
        }
        self.views.iter_mut().for_each(|view| *view = None);
        self.pool.close();
        self.storage.bind(py).call_method0("close")?;
        Ok(())
    }

    pub(super) fn get(&self, py: Python<'_>, request_id: u64) -> PyResult<Py<Request>> {
        self.peek(py, request_id)
            .ok_or_else(|| invalid(py, format!("unknown request {request_id}")))
    }

    fn peek(&self, py: Python<'_>, request_id: u64) -> Option<Py<Request>> {
        let request = self.pool.peek(request_id)?;
        self.views[request.slot()]
            .as_ref()
            .map(|view| view.clone_ref(py))
    }

    fn request_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.pool.request_ids())
    }

    pub(crate) fn has_open_requests(&self, py: Python<'_>) -> PyResult<bool> {
        self.pool
            .has_open_requests()
            .map_err(|error| native_error(py, error))
    }

    fn bind_calls<'py>(
        &self,
        py: Python<'py>,
        calls: Vec<Bound<'py, PyAny>>,
        request_pool_indices: Vec<usize>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        if calls.len() != request_pool_indices.len() {
            return Err(invalid(
                py,
                "request-pool indices are not aligned with calls",
            ));
        }
        let rows = calls
            .iter()
            .zip(request_pool_indices)
            .map(|(call, slot)| {
                Ok((
                    request_key(&call.getattr("request_key")?)?,
                    call_id(&call.getattr("call_id")?)?,
                    slot,
                ))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let requests = self
            .pool
            .bind_calls(&rows)
            .map_err(|error| native_error(py, error))?;
        let views = requests
            .iter()
            .map(|request| self.get(py, request.key().request_id.0))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, views)
    }

    fn validate_pending(&self, py: Python<'_>, calls: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        self.pool
            .validate_pending(&pending_calls(calls)?)
            .map_err(|error| native_error(py, error))
    }

    fn add_pending(&mut self, py: Python<'_>, calls: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        self.pool
            .add_pending(&pending_calls(calls)?)
            .map_err(|error| native_error(py, error))
    }

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
            let advances = call.getattr("advances_state")?.extract()?;
            let predecessor = self
                .pool
                .predecessor(key, advances)
                .map_err(|error| native_error(py, error))?;
            let value = match predecessor {
                Some(id) => call_type.call1((id.batch_id, id.request_index))?,
                None => py.None().into_bound(py),
            };
            result.set_item(call.getattr("call_id")?, value)?;
        }
        Ok(result)
    }

    fn apply_result(&mut self, py: Python<'_>, result: &Bound<'_, PyAny>) -> PyResult<()> {
        let key = request_key(&result.getattr("request_key")?)?;
        let id = call_id(&result.getattr("call_id")?)?;
        let status =
            pythonize::depythonize::<CallStatus>(&result.getattr("status")?.getattr("value")?)?;
        let value = result.getattr("progress")?;
        let progress = if value.is_none() {
            None
        } else {
            Some(progress_from_py(&value)?)
        };
        self.pool
            .apply_result(key, id, status, progress)
            .map_err(|error| native_error(py, error))
    }

    fn cancel_calls(&mut self, py: Python<'_>, calls: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        let calls = calls
            .iter()
            .map(|call| {
                Ok((
                    request_key(&call.getattr("request_key")?)?,
                    call_id(&call.getattr("call_id")?)?,
                ))
            })
            .collect::<PyResult<Vec<_>>>()?;
        self.pool
            .cancel_calls(&calls)
            .map_err(|error| native_error(py, error))
    }

    fn start(&mut self, py: Python<'_>, admission: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
        let parameters = new_request(admission)?;
        let request_id = parameters.request_key.request_id.0;
        let previous_slot = self.pool.peek(request_id).map(|request| request.slot());
        let slot = self
            .pool
            .start(parameters)
            .map_err(|error| native_error(py, error))?;
        if let Some(slot) = slot {
            if let Some(previous) = previous_slot {
                self.views[previous] = None;
            }
            let request = Arc::clone(
                self.pool
                    .get(request_id)
                    .map_err(|error| native_error(py, error))?,
            );
            self.views[slot] = Some(Py::new(
                py,
                Request {
                    request,
                    admission: admission.clone().unbind(),
                    diffusion: None,
                },
            )?);
        }
        Ok(slot)
    }

    fn finish(&mut self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<()> {
        self.pool
            .finish(request_key(key)?)
            .map_err(|error| native_error(py, error))
    }

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
        self.pool
            .retirement_ready(request_key(key)?)
            .map_err(|error| native_error(py, error))
    }

    #[pyo3(name = "drop")]
    pub(super) fn drop_request(&mut self, request_id: u64) {
        if let Some(request) = self.pool.peek(request_id) {
            self.views[request.slot()] = None;
        }
        self.pool.remove(request_id);
    }

    /// The execution owner drains numerical staging before retiring this slot.
    pub(super) fn retire(&mut self, py: Python<'_>, request_id: u64) -> PyResult<()> {
        self.pool
            .retire(request_id)
            .map_err(|error| native_error(py, error))?;
        self.get(py, request_id)?.borrow_mut(py).diffusion = None;
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.storage)?;
        for request in self.views.iter().flatten() {
            visit.call(request)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.views.iter_mut().for_each(|view| *view = None);
        self.pool.close();
    }
}

fn pending_calls(calls: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<(RequestKey, CallId, bool)>> {
    calls
        .iter()
        .map(|call| {
            Ok((
                request_key(&call.getattr("request_key")?)?,
                call_id(&call.getattr("call_id")?)?,
                call.getattr("advances_state")?.extract()?,
            ))
        })
        .collect()
}

pub(super) fn progress_from_py(value: &Bound<'_, PyAny>) -> PyResult<RequestProgress> {
    Ok(RequestProgress {
        logical_position: value.getattr("logical_position")?.extract()?,
        rng_counter: value.getattr("rng_counter")?.extract()?,
        flow_step: value.getattr("flow_step")?.extract()?,
        kv_visible_len: value.getattr("kv_visible_len")?.extract()?,
        kv_computed_len: value.getattr("kv_computed_len")?.extract()?,
        prompt_logits_ready: value.getattr("prompt_logits_ready")?.extract()?,
    })
}

pub(super) fn progress_to_py(
    py: Python<'_>,
    progress: RequestProgress,
) -> PyResult<Bound<'_, PyAny>> {
    py.import("uniserve_worker.execution.request")?
        .getattr("RequestProgress")?
        .call1((
            progress.logical_position,
            progress.rng_counter,
            progress.flow_step,
            progress.kv_visible_len,
            progress.kv_computed_len,
            progress.prompt_logits_ready,
        ))
}
