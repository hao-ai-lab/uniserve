//! Batch command retirement and request-slot release on the executor thread.

use std::collections::HashSet;
use std::sync::Arc;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use uniserve_worker::cuda::Event;
use uniserve_worker_ipc::{Batch, BatchCommand, BufferId, RequestKey};

use super::super::error::native_error;
use super::super::events::CUDAEvent;
use super::super::exports;
use super::super::latent::LatentPool;
use super::super::protocol::buffer_id;
use super::PythonBackend;
use crate::convert;

pub(super) struct Retirement {
    requests: HashSet<RequestKey>,
    buffers: HashSet<BufferId>,
    retained: HashSet<BufferId>,
    local_requests: Vec<RequestKey>,
    exports: Vec<Py<PyAny>>,
    events: Vec<Arc<Event>>,
    complete: bool,
}

impl Retirement {
    pub(super) fn new(batch: &Batch) -> Self {
        let mut requests = HashSet::new();
        let mut buffers = HashSet::new();
        let mut retained = HashSet::new();
        for command in &batch.commands {
            match command {
                BatchCommand::Finish {
                    request_key,
                    retained_buffers,
                } => {
                    requests.insert(*request_key);
                    retained.extend(retained_buffers);
                }
                BatchCommand::Free { buffer } => {
                    buffers.insert(*buffer);
                }
                BatchCommand::Start { .. } => {}
            }
        }
        retained.retain(|buffer| !buffers.contains(buffer));

        Self {
            requests,
            buffers,
            retained,
            local_requests: Vec::new(),
            exports: Vec::new(),
            events: Vec::new(),
            complete: false,
        }
    }

    pub(super) fn begin(&mut self, py: Python<'_>, backend: &PythonBackend) -> PyResult<()> {
        if self.requests.is_empty() && self.buffers.is_empty() {
            self.complete = true;
            return Ok(());
        }

        // Remote epochs can own exports without having request slots here.
        self.local_requests = self
            .requests
            .iter()
            .copied()
            .filter(|key| {
                backend
                    .requests
                    .borrow(py)
                    .pool
                    .peek(key.request_id.0)
                    .is_some_and(|request| request.key() == *key)
            })
            .collect();
        backend
            .tensors
            .get()
            .release_request_set(py, &self.requests, &self.retained)?;
        if let Some(latents) = &backend.latents {
            latents
                .borrow_mut(py)
                .cancel_request_imports(py, &self.requests)?;
        }

        for directory in &backend.exports {
            self.exports.extend(exports::select(
                directory.bind(py),
                &self.buffers,
                &self.requests,
                &self.retained,
            )?);
        }
        for directory in &backend.exports {
            exports::release(directory.bind(py), &self.exports)?;
        }
        if let Some(latents) = &backend.latents {
            LatentPool::release_buffers(
                latents.bind(py).clone(),
                self.exports.iter().map(|key| key.clone_ref(py)).collect(),
            )?;
        }
        if let Some(imports) = &backend.cache_imports {
            imports
                .borrow(py)
                .cancel_request_imports(py, &self.requests, &self.retained)?;
            let buffers = self
                .exports
                .iter()
                .map(|key| buffer_id(key.bind(py)))
                .collect::<PyResult<HashSet<_>>>()?;
            imports.borrow(py).release_buffers(py, &buffers)?;
            if let Some(cache) = &backend.cache {
                cache
                    .borrow_mut(py)
                    .inner
                    .release_exports(&buffers.into_iter().collect::<Vec<_>>());
            }
        }

        // Finish includes request-state writes issued after output capture.
        // Retain each event before recording so failure cleanup owns every
        // fence already acquired by this batch.
        if !self.requests.is_empty() {
            for device in &backend.retirement_devices {
                let pool = backend.events.borrow(py);
                let event = pool.acquire(py, device.bind(py), false, false)?;
                pool.retain(py, &event, device.bind(py), 1)?;
                self.events.push(Arc::clone(&event.inner));
                pool.record(py, &event, device.bind(py))?;
                pool.schedule_completion_wake(py, device.bind(py), &event)?;
            }
        }
        Ok(())
    }

    pub(super) fn poll(&mut self, py: Python<'_>, backend: &PythonBackend) -> PyResult<bool> {
        backend.reap_resources(py)?;
        for event in &self.events {
            if !event.ready().map_err(PyRuntimeError::new_err)? {
                return Ok(false);
            }
        }
        while let Some(event) = self.events.last() {
            backend.events.borrow(py).release(
                &CUDAEvent {
                    inner: Arc::clone(event),
                },
                1,
            )?;
            self.events.pop();
        }
        if self.complete {
            return Ok(true);
        }

        for key in &self.local_requests {
            if !backend
                .requests
                .borrow(py)
                .pool
                .retirement_ready(*key)
                .map_err(|error| native_error(py, error))?
            {
                return Ok(false);
            }
        }
        if !backend.tensors.get().retirement_ready_for(
            py,
            &self.buffers,
            &self.requests,
            &self.retained,
        )? {
            return Ok(false);
        }
        if let Some(latents) = &backend.latents
            && !latents
                .borrow_mut(py)
                .retirement_ready_for(py, &self.requests)?
        {
            return Ok(false);
        }
        if let Some(cache) = &backend.cache
            && !cache.borrow_mut(py).retirement_ready_for(
                py,
                &self.buffers,
                &self.requests,
                &self.retained,
            )?
        {
            return Ok(false);
        }

        for directory in &backend.exports {
            exports::forget(directory.bind(py), &self.exports)?;
        }
        backend.retire_requests(py, &self.local_requests, &self.retained)?;
        self.complete = true;
        Ok(true)
    }

    pub(super) fn close(
        &mut self,
        py: Python<'_>,
        backend: &PythonBackend,
        numerical: &Py<PyAny>,
    ) -> PyResult<()> {
        if !self.events.is_empty() {
            backend.events.borrow(py).defer_events(
                self.events.clone(),
                numerical.clone_ref(py),
                None,
            )?;
            self.events.clear();
        }
        Ok(())
    }

    pub(super) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        for key in &self.exports {
            visit.call(key)?;
        }
        Ok(())
    }
}

impl PythonBackend {
    pub(super) fn reap_resources(&self, py: Python<'_>) -> PyResult<()> {
        self.events.borrow(py).reap(py)?;
        // Remote consumers acknowledge exported chunks without a local wake.
        for transport in &self.transports {
            transport.bind(py).call_method0("reap")?;
        }
        Ok(())
    }

    fn retire_requests(
        &self,
        py: Python<'_>,
        keys: &[RequestKey],
        retained: &HashSet<BufferId>,
    ) -> PyResult<()> {
        let mut live = Vec::new();
        for key in keys {
            if let Some(request) = self.requests.borrow(py).pool.peek(key.request_id.0)
                && request.key() == *key
                && !request.retired().map_err(|error| native_error(py, error))?
            {
                live.push(key.request_id.0);
            }
        }
        if live.is_empty() {
            return Ok(());
        }

        // Resets precede result delivery on the current stream. Later batch
        // streams wait for it before reusing slots. Waiting for the resets
        // themselves would delay this result behind later queued batches.
        self.release_requests(py, &live, retained)?;
        for request_id in live {
            let request = self.requests.borrow(py).get(py, request_id)?;
            let diffusion = request
                .borrow(py)
                .diffusion
                .as_ref()
                .map(|value| value.clone_ref(py));
            if let Some(diffusion) = diffusion {
                diffusion.bind(py).call_method0("close")?;
            }
            self.requests.borrow_mut(py).retire(py, request_id)?;
        }
        Ok(())
    }

    pub(super) fn release_requests(
        &self,
        py: Python<'_>,
        ids: &[u64],
        retained: &HashSet<BufferId>,
    ) -> PyResult<()> {
        let (keys, slots): (HashSet<_>, Vec<_>) = ids
            .iter()
            .filter_map(|id| {
                self.requests
                    .borrow(py)
                    .pool
                    .peek(*id)
                    .map(|request| (request.key(), request.slot()))
            })
            .unzip();
        let worker = self.worker.bind(py);
        let slots = PyTuple::new(py, slots)?;
        if !keys.is_empty() {
            if let Some(imports) = &self.cache_imports {
                imports
                    .borrow(py)
                    .cancel_request_imports(py, &keys, retained)?;
            }
            for name in ["decode_state", "canvas_slots"] {
                let state = worker.getattr(name)?;
                if !state.is_none() {
                    state.call_method1("reset", (&slots,))?;
                }
            }
            let tables = worker.getattr("block_tables")?;
            if !tables.is_none() {
                tables.call_method1("release", (&slots,))?;
            }
        }

        let cache = worker.getattr("kv_cache")?;
        if !cache.is_none() {
            for id in ids {
                cache.call_method1("drop", (*id,))?;
            }
        }
        let tables = worker.getattr("block_tables")?;
        if !tables.is_none() {
            for key in &keys {
                tables
                    .call_method1("release_prefixes", (convert::request_key_to_py(py, *key)?,))?;
            }
        }

        let owners: HashSet<_> = ids.iter().copied().collect();
        for name in ["tensor_store", "kv_cache", "latent_pool"] {
            let store = worker.getattr(name)?;
            if store.is_none() {
                continue;
            }
            let directory = store.getattr("exports")?.cast_into::<PyDict>()?;
            let mut selected = Vec::new();
            for (key, _) in &directory {
                let buffer = buffer_id(&key)?;
                if owners.contains(&buffer.owner.request_id.0) && !retained.contains(&buffer) {
                    selected.push(key);
                }
            }
            store.call_method1("release_buffers", (PyTuple::new(py, selected)?,))?;
        }

        if !keys.is_empty() {
            self.tensors
                .get()
                .release_request_set(py, &keys, retained)?;
        }
        let media = worker.getattr("media_mux")?;
        if !media.is_none() {
            for id in ids {
                media.call_method1("drop", (*id,))?;
            }
        }
        if !keys.is_empty()
            && let Some(latents) = &self.latents
        {
            latents.bind(py).call_method1("release_slots", (slots,))?;
        }
        Ok(())
    }
}
