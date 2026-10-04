//! Request commands and input ownership around numerical batch execution.

use std::collections::{BTreeSet, HashSet};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use uniserve_core::CallId;
use uniserve_worker_ipc::{BatchCommand, CallKind, MediaCall};

use super::super::error::{invalid, native_error};
use super::super::inputs::BatchInputs;
use super::{BatchState, PythonBackend};

impl PythonBackend {
    pub(super) fn prepare_batch(&self, py: Python<'_>, batch: &mut BatchState) -> PyResult<()> {
        let unsupported = batch
            .plan
            .calls
            .iter()
            .filter(|call| !self.info.supported_calls.contains(&call.code))
            .map(|call| call.code.as_str())
            .collect::<BTreeSet<_>>();
        if !unsupported.is_empty() {
            return Err(invalid(
                py,
                format!(
                    "execution batch contains call kinds unsupported by this worker: {:?}",
                    unsupported.into_iter().collect::<Vec<_>>()
                ),
            ));
        }

        // Install Starts before resolving predecessors. Even if a later Start
        // or its numerical noise preparation fails, reset all admitted slots.
        let commands = batch
            .numerical
            .bind(py)
            .getattr("batch")?
            .getattr("commands")?;
        let mut admitted = Vec::new();
        let started = (|| -> PyResult<()> {
            for (index, command) in batch.plan.commands.iter().enumerate() {
                let BatchCommand::Start { request } = command else {
                    continue;
                };
                let admission = commands.get_item(index)?.getattr("request")?;
                let slot =
                    self.requests
                        .borrow_mut(py)
                        .start(py, request.as_ref().clone(), &admission)?;
                if let Some(slot) = slot {
                    admitted.push(slot);
                    if request.diffusion.is_some() {
                        let view = self
                            .requests
                            .borrow(py)
                            .get(py, request.request_key.request_id.0)?;
                        py.import("uniserve_worker.execution.media")?
                            .getattr("begin_noise")?
                            .call1((&self.model_runner, view, &self.requests))?;
                    }
                }
            }
            Ok(())
        })();
        self.reset_slots(py, &admitted)?;
        started?;

        let requests = self.requests.borrow(py);
        batch.predecessors = batch
            .plan
            .calls
            .iter()
            .map(|call| {
                requests
                    .pool
                    .predecessor(call.request_key, call.advances_state())
                    .map_err(|error| native_error(py, error))
            })
            .collect::<PyResult<_>>()?;
        drop(requests);

        // Numerical staging consumes a snapshot; request ordering remains
        // native and also determines input release and failed-call progress.
        let call_type = py
            .import("uniserve_worker.protocol.identity")?
            .getattr("CallId")?;
        let predecessors = PyDict::new(py);
        for (call, previous) in batch.plan.calls.iter().zip(&batch.predecessors) {
            let key = call_type.call1((call.call_id.batch_id, call.call_id.request_index))?;
            let value = match previous {
                Some(previous) => call_type.call1((previous.batch_id, previous.request_index))?,
                None => py.None().into_bound(py),
            };
            predecessors.set_item(key, value)?;
        }
        batch
            .numerical
            .bind(py)
            .setattr("predecessors", predecessors)?;
        self.validate_batch(py, batch)?;

        for command in &batch.plan.commands {
            if let BatchCommand::Finish { request_key, .. } = command {
                self.requests
                    .borrow_mut(py)
                    .pool
                    .finish(*request_key)
                    .map_err(|error| native_error(py, error))?;
            }
        }

        self.release_predecessors(py, batch, false)?;
        batch.retirement.revoke(py, self)?;
        batch.imports = self
            .runner
            .bind(py)
            .call_method1("prepare", (&batch.numerical,))?
            .extract()?;
        Ok(())
    }

    fn validate_batch(&self, py: Python<'_>, batch: &BatchState) -> PyResult<()> {
        let plan = &batch.plan;
        if let Some(allocation) = plan
            .buffer_allocations
            .iter()
            .find(|allocation| allocation.offset + allocation.bytes > self.info.buffer_pool_bytes)
        {
            return Err(invalid(
                py,
                format!(
                    "batch buffer allocation exceeds the worker buffer pool: offset={}, bytes={}, capacity={}, buffer={:?}",
                    allocation.offset,
                    allocation.bytes,
                    self.info.buffer_pool_bytes,
                    allocation.buffer,
                ),
            ));
        }

        let config = self.worker.bind(py).getattr("worker_config")?;
        let rank: usize = config.getattr("rank")?.extract()?;
        if !self.info.components.is_empty() {
            for call in &plan.calls {
                if !self.info.components.iter().any(|component| {
                    component.name == call.component && component.config.ranks.contains(&rank)
                }) {
                    return Err(invalid(
                        py,
                        format!(
                            "call targets component {:?} outside this rank",
                            call.component,
                        ),
                    ));
                }
            }
        }

        let max_calls: usize = config.getattr("max_batch_calls")?.extract()?;
        if plan.calls.len() > max_calls {
            return Err(invalid(
                py,
                "execution batch exceeds the worker_config call limit",
            ));
        }
        let max_slots: u32 = config.getattr("max_request_pool_size")?.extract()?;
        if plan
            .block_tables
            .iter()
            .map(|table| table.request_pool_idx)
            .chain(plan.forward.request_pool_indices.iter().copied())
            .any(|slot| slot > max_slots)
        {
            return Err(invalid(py, "execution batch exceeds request-slot capacity"));
        }

        // Latent preparation opens the video trajectory, so it must follow
        // admission rather than an already submitted state-producing call.
        if !self
            .model_runner
            .bind(py)
            .getattr("video_postprocessor")?
            .is_none()
        {
            for (call, previous) in plan.calls.iter().zip(&batch.predecessors) {
                if call.code == CallKind::Media(MediaCall::LatentPreparation)
                    && *previous != Some(CallId::new(0, 0))
                {
                    return Err(invalid(
                        py,
                        "video preparation does not follow its request root",
                    ));
                }
            }
        }
        Ok(())
    }

    fn release_predecessors(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        inputs_acquired: bool,
    ) -> PyResult<()> {
        let mut consumed = HashSet::new();
        for call in &batch.plan.calls {
            consumed.extend(
                call.tensor_inputs()
                    .chain(call.predicate.iter())
                    .map(|input| (input.request_key, input.producer_call_id)),
            );
            if let Some(input) = call.kv_input {
                consumed.insert((input.owner, input.producer_call_id));
            }
        }
        let releases = batch
            .plan
            .calls
            .iter()
            .zip(&batch.predecessors)
            .filter_map(|(call, previous)| {
                let previous = previous.as_ref()?;
                let source = (call.request_key, *previous);
                // The admission root has no outputs. Read predecessors stay
                // visible through acquisition; unread ones can retire early.
                (previous.batch_id > 0 && consumed.contains(&source) == inputs_acquired)
                    .then_some(source)
            })
            .collect::<Vec<_>>();
        self.tensors.get().release_calls(py, &releases)?;
        if let Some(cache) = &self.cache {
            let released = cache.borrow_mut(py).inner.release_calls(&releases);
            self.release_buffers(py, &released.into_iter().collect())?;
        }
        Ok(())
    }

    pub(super) fn execute_batch<'py>(
        &self,
        py: Python<'py>,
        batch: &mut BatchState,
    ) -> PyResult<Bound<'py, PyAny>> {
        if batch.inputs.borrow(py).closed() {
            return Err(PyRuntimeError::new_err(
                "batch inputs have already been consumed",
            ));
        }
        if !batch.inputs.borrow(py).ready()? {
            return Err(PyRuntimeError::new_err(
                "batch was observed before dependency readiness",
            ));
        }

        let executed = (|| {
            batch.inputs.borrow(py).require_storage(py)?;
            let failure = self.run_batch(py, batch)?;
            self.release_predecessors(py, batch, true)?;

            // Predicates outside the request's predecessor are consumed by
            // this batch too, but releasing that other call would be too broad.
            let predicates = batch
                .plan
                .calls
                .iter()
                .zip(&batch.predecessors)
                .filter_map(|(call, previous)| {
                    call.predicate
                        .as_ref()
                        .filter(|value| Some(value.producer_call_id) != *previous)
                        .map(|value| value.buffer_id())
                })
                .collect();
            self.tensors.get().release_buffers(py, &predicates)?;
            Ok(failure)
        })();

        let closed = BatchInputs::close(
            batch.inputs.bind(py),
            self.tensors.get(),
            self.latents.as_ref().map(|pool| pool.bind(py)),
            self.cache_imports.as_ref().map(|imports| imports.bind(py)),
        );
        match (executed, closed) {
            (Err(error), Err(cleanup)) => {
                let error: PyErr = error;
                let _ = error.value(py).call_method1(
                    "add_note",
                    (format!("batch input cleanup failed: {cleanup}"),),
                );
                Err(error)
            }
            (Err(error), _) | (_, Err(error)) => Err(error),
            (Ok(output), Ok(())) => Ok(output),
        }
    }
}
