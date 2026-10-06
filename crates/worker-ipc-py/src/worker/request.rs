//! Numerical request views backed by the native request pool.

use std::sync::Arc;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use uniserve_core::CallId;
use uniserve_worker::{
    Request as NativeRequest, RequestPool as NativePool, RequestProgress as NativeProgress,
};
use uniserve_worker_ipc::{CallStatus, NewRequest, RequestKey};

use super::error::{invalid, native_error};
use super::protocol::{call_id, new_request, request_key};

/// Immutable observation of accepted or projected native request coordinates.
#[pyclass(frozen, eq, module = "uniserve_worker._uniserve_ipc")]
#[derive(PartialEq)]
pub(crate) struct RequestProgress {
    pub(super) inner: NativeProgress,
}

#[pymethods]
impl RequestProgress {
    #[new]
    #[pyo3(signature = (logical_position=0, rng_counter=0, flow_step=0, kv_visible_len=0, kv_computed_len=0, prompt_logits_ready=false))]
    fn new(
        py: Python<'_>,
        logical_position: u64,
        rng_counter: u64,
        flow_step: u64,
        kv_visible_len: u64,
        kv_computed_len: u64,
        prompt_logits_ready: bool,
    ) -> PyResult<Self> {
        if kv_visible_len > kv_computed_len {
            return Err(invalid(py, "request KV extents are not contained"));
        }

        Ok(Self {
            inner: NativeProgress {
                logical_position,
                rng_counter,
                flow_step,
                kv_visible_len,
                kv_computed_len,
                prompt_logits_ready,
            },
        })
    }

    #[getter]
    fn logical_position(&self) -> u64 {
        self.inner.logical_position
    }

    #[getter]
    fn rng_counter(&self) -> u64 {
        self.inner.rng_counter
    }

    #[getter]
    fn flow_step(&self) -> u64 {
        self.inner.flow_step
    }

    #[getter]
    fn kv_visible_len(&self) -> u64 {
        self.inner.kv_visible_len
    }

    #[getter]
    fn kv_computed_len(&self) -> u64 {
        self.inner.kv_computed_len
    }

    #[getter]
    fn prompt_logits_ready(&self) -> bool {
        self.inner.prompt_logits_ready
    }
}

/// Retain numerical parameters and tensors alongside a native request epoch.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct Request {
    pub(super) request: Arc<NativeRequest>,
    #[pyo3(get)]
    pub(super) admission: Py<PyAny>,
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
    pub(super) fn sampling<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
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
    fn accepted_progress(&self, py: Python<'_>) -> PyResult<RequestProgress> {
        let inner = self
            .request
            .progress()
            .map_err(|error| native_error(py, error))?;
        Ok(RequestProgress { inner })
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

    fn add_pending(&mut self, py: Python<'_>, calls: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        self.pool
            .add_pending(&pending_calls(calls)?)
            .map_err(|error| native_error(py, error))
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
            Some(value.extract::<PyRef<'_, RequestProgress>>()?.inner)
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

    #[pyo3(name = "start")]
    fn start_py(
        &mut self,
        py: Python<'_>,
        admission: &Bound<'_, PyAny>,
    ) -> PyResult<Option<usize>> {
        self.start(py, new_request(admission)?, admission)
    }

    fn finish(&mut self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<()> {
        self.pool
            .finish(request_key(key)?)
            .map_err(|error| native_error(py, error))
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

    /// The execution owner drains numerical work before retiring this slot.
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

impl RequestPool {
    pub(super) fn start(
        &mut self,
        py: Python<'_>,
        parameters: NewRequest,
        admission: &Bound<'_, PyAny>,
    ) -> PyResult<Option<usize>> {
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
