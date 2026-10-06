//! Commit batch resources together, or retire provisional work after failure.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use uniserve_worker_ipc::CallStatus;

use super::{BatchState, PythonBackend};
use crate::worker::error::native_error;
use crate::worker::exports;
use crate::worker::host::with_context;
use crate::worker::output::OutputBuffer;
use crate::worker::pending::PendingOutput;
use crate::worker::storage::{Buffer, TensorRead};

impl PythonBackend {
    pub(super) fn commit_batch(&self, py: Python<'_>, batch: &mut BatchState) -> PyResult<()> {
        let numerical = batch.numerical.clone_ref(py);
        let scope = super::super::batch::BatchState::scope(numerical.bind(py))?;
        with_context(scope.bind(py), || {
            let started = std::time::Instant::now();
            let outputs = batch.pending_outputs(py);
            self.finish_reads(py, &outputs)?;
            self.write_completion_predicates(py, &outputs)?;

            for (output, call) in outputs.iter().zip(&batch.plan.calls) {
                output.borrow().validate_output(py, call)?;
            }

            let writes = output_writes(py, &outputs)?;
            self.tensors.get().validate_writes(py, writes.clone())?;
            let mut updates = Vec::new();
            for output in &outputs {
                if let Some(update) = &output.borrow().lock(py)?.latent_update {
                    updates.push(update.clone());
                }
            }
            if let Some(latents) = &self.latents {
                latents.borrow_mut(py).validate(py, &updates)?;
            } else if !updates.is_empty() {
                return Err(PyRuntimeError::new_err(
                    "latent export has no physical pool",
                ));
            }
            let buffer = numerical.borrow(py).output_buffer(py)?;
            OutputBuffer::seal(buffer.bind(py))?;

            let mut exports = Vec::new();
            let mut installations = Vec::new();
            let tensor_exports = PyDict::new(py);
            let cache_exports = PyDict::new(py);
            let latent_exports = PyDict::new(py);
            // Host results belong to the component's first member rank.
            let reports_output = self.info.endpoint.rank as usize
                == self.output_rank(py, &batch.plan.calls[0].component)?;
            for output in &outputs {
                let pending = output.borrow();
                pending.lock(py)?.reports_output = reports_output;
                if let Some(transfer) = &pending.cache_export {
                    exports.push((transfer.source, (**transfer).clone()));
                }
                if let Some((buffer, transfer)) = &pending.cache_installation {
                    installations.push((transfer.source, *buffer, (**transfer).clone()));
                }
                tensor_exports.update(pending.tensor_exports.bind(py).as_mapping())?;
                cache_exports.update(pending.cache_exports.bind(py).as_mapping())?;
                latent_exports.update(pending.latent_exports.bind(py).as_mapping())?;
            }

            // Keep execution/commit timings at their established sampling
            // point, before cross-resource preflight and device-state updates.
            let started_ns = numerical.borrow(py).started_ns;
            batch.record_component(py, "commit_lane", started)?;
            let elapsed = (py
                .import("time")?
                .call_method0("perf_counter_ns")?
                .extract::<u64>()?
                - started_ns)
                / 1000;
            let stats = numerical.borrow(py).execution_stats(py)?.inner;
            if let Some(cache) = &self.cache {
                cache
                    .borrow(py)
                    .inner
                    .validate_exports(&exports, &installations)
                    .map_err(|error| native_error(py, error))?;
            } else if !exports.is_empty() || !installations.is_empty() {
                return Err(PyRuntimeError::new_err(
                    "cache export has no backing KV resources",
                ));
            }

            let calls: Vec<_> = batch
                .plan
                .calls
                .iter()
                .map(|call| (call.request_key, call.call_id, call.advances_state()))
                .collect();
            self.requests
                .borrow(py)
                .pool
                .validate_pending(&calls)
                .map_err(|error| native_error(py, error))?;
            let mut directories = Vec::new();
            for (name, exports) in [
                ("tensor_store", tensor_exports),
                ("kv_cache", cache_exports),
                ("latent_pool", latent_exports),
            ] {
                let owner = self.worker.bind(py).getattr(name)?;
                if owner.is_none() {
                    if !exports.is_empty() {
                        return Err(PyRuntimeError::new_err(
                            "transport export has no backing storage",
                        ));
                    }
                } else {
                    let resident = owner.getattr("exports")?.cast_into::<PyDict>()?;
                    exports::validate_exports(&resident, &exports)?;
                    directories.push((resident, exports));
                }
            }

            // Every owner has accepted its proposed update. Any failure from
            // this point is fatal, including a device-state update failure.
            batch.committed = true;
            self.tensors.get().commit_writes(py, writes)?;
            if let Some(latents) = &self.latents {
                latents.borrow_mut(py).apply(py, &updates)?;
            }

            for output in &outputs {
                let output = output.borrow();
                let mut state = output.lock(py)?;
                if let Some(update) = state.latent_update.take() {
                    state.apply_latent_update(&update);
                }
            }

            if let Some(cache) = &self.cache {
                cache
                    .borrow_mut(py)
                    .inner
                    .apply_exports(exports, installations);
            }
            for (resident, exports) in directories {
                resident.update(exports.as_mapping())?;
            }
            self.apply_decode_state(py, &outputs)?;
            for output in outputs {
                output.borrow_mut().release_execution_references(py)?;
            }
            self.requests
                .borrow_mut(py)
                .pool
                .add_pending(&calls)
                .map_err(|error| native_error(py, error))?;
            batch.record_execution(py, elapsed, stats)
        })
    }

    fn write_completion_predicates(
        &self,
        py: Python<'_>,
        outputs: &[Bound<'_, PendingOutput>],
    ) -> PyResult<()> {
        let mut writes = Vec::new();
        for output in outputs {
            let output = output.borrow();
            if output.lock(py)?.output.status == CallStatus::Predicated {
                continue;
            }
            if let Some(write) = &output.completion_write {
                let write = write.bind(py).cast::<Buffer>()?;
                if !write.get().producer_recorded(py) {
                    writes.push(write.clone());
                }
            }
        }
        let Some(first) = writes.first() else {
            return Ok(());
        };

        // Models may already have written a device decision. Only unresolved
        // completion scalars receive true, through the store's shared copy.
        let tensor = first.get().tensor(py);
        let options = PyDict::new(py);
        options.set_item("device", tensor.bind(py).getattr("device")?)?;
        options.set_item("dtype", tensor.bind(py).getattr("dtype")?)?;
        let values = py
            .import("torch")?
            .call_method("ones", ((writes.len(),),), Some(&options))?;
        self.tensors.get().write_scalars(py, writes, values, None)?;
        Ok(())
    }

    fn apply_decode_state(
        &self,
        py: Python<'_>,
        outputs: &[Bound<'_, PendingOutput>],
    ) -> PyResult<()> {
        let updates: Vec<_> = outputs
            .iter()
            .map(|output| {
                let output = output.borrow();
                (
                    output.request.borrow(py).request.slot(),
                    output.token_update.clone_ref(py),
                )
            })
            .collect();
        let Some(state) = &self.decode_state else {
            if updates.iter().any(|(_, update)| {
                update.sampled.is_some()
                    || update.prompt_logits.is_some()
                    || update.cache_length.is_some()
            }) {
                return Err(PyRuntimeError::new_err(
                    "runtime state export has no backing storage",
                ));
            }
            return Ok(());
        };
        let state = state.bind(py);

        // Install verified extents before advancing tokens. Tensor lengths and
        // coordinates remain on device, including speculative acceptance.
        for (slot, update) in &updates {
            if let Some(length) = &update.cache_length {
                state.call_method1("set_cache_length", (slot, length))?;
            }
        }

        let decode: Vec<_> = updates
            .iter()
            .filter_map(|(slot, update)| {
                update
                    .sampled
                    .as_ref()
                    .filter(|_| update.decode_increment)
                    .map(|sample| (slot, update, sample.bind(py)))
            })
            .collect();
        if !decode.is_empty() {
            let samples = PyTuple::new(py, decode.iter().map(|(_, _, sample)| sample))?;
            let indices = decode
                .iter()
                .map(|(_, _, sample)| {
                    let index = sample.getattr("request_pool_index")?;
                    if index.is_none() {
                        return Err(PyRuntimeError::new_err(
                            "decode samples have no device request slots",
                        ));
                    }
                    Ok(index)
                })
                .collect::<PyResult<Vec<_>>>()?;
            let indices = PyTuple::new(py, indices)?;
            let columns: [Bound<'_, PyAny>; 4] = py
                .import("uniserve_worker.sampling.result")?
                .call_method1(
                    "sample_columns",
                    (samples, ("tokens", "continuation", "valid", "active")),
                )?
                .extract()?;
            let options = PyDict::new(py);
            for (name, column) in ["tokens", "predicates", "valid", "active"]
                .into_iter()
                .zip(columns)
            {
                options.set_item(name, column)?;
            }
            options.set_item(
                "device_slots",
                py.import("uniserve.tensors")?
                    .call_method1("concatenate_views", (indices,))?,
            )?;
            options.set_item(
                "penalty_bases",
                PyTuple::new(
                    py,
                    decode.iter().map(|(_, update, _)| {
                        update.penalty_base.as_ref().map(|value| value.bind(py))
                    }),
                )?,
            )?;
            let slots = PyTuple::new(py, decode.iter().map(|(slot, _, _)| slot))?;
            state.call_method("apply_tokens", (slots,), Some(&options))?;
        }

        // Prefill and verification have explicit per-call coordinates; decode
        // above advances all of its rows together using their device slots.
        for (slot, update) in &updates {
            if let Some(sample) = &update.sampled
                && !update.decode_increment
            {
                let sample = sample.bind(py);
                let options = PyDict::new(py);
                for (name, field) in [
                    ("tokens", "tokens"),
                    ("predicates", "continuation"),
                    ("valid", "valid"),
                    ("active", "active"),
                ] {
                    options.set_item(name, sample.getattr(field)?)?;
                }
                options.set_item("logical_position", &update.logical_position)?;
                options.set_item("sampling_position", &update.sampling_position)?;
                options.set_item(
                    "penalty_bases",
                    (update.penalty_base.as_ref().map(|value| value.bind(py)),),
                )?;
                state.call_method("apply_tokens", ((slot,),), Some(&options))?;
            }
        }
        for (slot, update) in &updates {
            if let Some(logits) = &update.prompt_logits {
                state.call_method1("set_prompt_logits", (slot, logits))?;
            }
        }
        Ok(())
    }

    pub(super) fn discard_batch(&self, py: Python<'_>, batch: &BatchState) -> PyResult<()> {
        let numerical = batch.numerical.bind(py);
        let buffer = numerical
            .borrow()
            .buffer
            .as_ref()
            .map(|buffer| buffer.clone_ref(py));
        let Some(buffer) = buffer else {
            // The numerical reservation owns allocation failures before it
            // binds outputs. There are no batch resources to retire yet.
            return Ok(());
        };
        let scope = super::super::batch::BatchState::scope(numerical)?;
        with_context(scope.bind(py), || {
            let outputs = batch.pending_outputs(py);
            self.finish_reads(py, &outputs)?;
            let buffer = buffer.bind(py);
            OutputBuffer::seal(buffer)?;
            for output in &outputs {
                PendingOutput::abandon(output)?;
            }

            OutputBuffer::abandon(buffer)?;
            self.tensors
                .get()
                .abandon_writes(py, output_writes(py, &outputs)?)?;

            let buffers = batch
                .plan
                .calls
                .iter()
                .flat_map(|call| {
                    call.kv_output
                        .into_iter()
                        .chain(call.tensor_outputs().map(|output| output.buffer_id()))
                })
                .collect();
            self.release_buffers(py, &buffers)?;
            if let Some(latents) = &self.latents {
                let slots = batch.inputs.borrow(py).imported_slots();
                if !slots.is_empty() {
                    latents.borrow_mut(py).release_slots(py, slots)?;
                }
            }

            let transports = self.worker.bind(py).getattr("transports")?;
            for output in outputs {
                output.borrow().revoke_exports(&transports)?;
                output.borrow_mut().release_execution_references(py)?;
            }
            Ok(())
        })
    }

    fn finish_reads(&self, py: Python<'_>, outputs: &[Bound<'_, PendingOutput>]) -> PyResult<()> {
        let mut reads = Vec::new();
        let mut after_writes = Vec::new();
        let mut feature_reads = Vec::new();
        for output in outputs {
            let pending = output.borrow();
            for read in pending.device_reads.bind(py) {
                reads.push(read.cast_into::<TensorRead>()?);
                if let Some(write) = &pending.producer_write {
                    after_writes.push(write.bind(py).clone().cast_into::<Buffer>()?);
                }
            }
            for read in pending.feature_reads.bind(py) {
                feature_reads.push(read.cast_into::<TensorRead>()?);
            }
        }
        // A source may belong to another request. Reuse the consuming call's
        // producer fence only when every read has a corresponding write.
        if !reads.is_empty() {
            if after_writes.len() != reads.len() {
                after_writes.clear();
            }
            self.tensors
                .get()
                .complete_reads(py, reads, None, after_writes)?;
        }
        if !feature_reads.is_empty() {
            self.tensors
                .get()
                .complete_reads(py, feature_reads, None, Vec::new())?;
        }
        for output in outputs {
            let pending = output.borrow();
            pending.device_reads.bind(py).call_method0("clear")?;
            pending.feature_reads.bind(py).call_method0("clear")?;
        }
        Ok(())
    }
}

fn output_writes<'py>(
    py: Python<'py>,
    outputs: &[Bound<'py, PendingOutput>],
) -> PyResult<Vec<Bound<'py, Buffer>>> {
    let mut writes = Vec::new();
    for output in outputs {
        for write in output.borrow().writes.bind(py) {
            writes.push(write.cast_into::<Buffer>()?);
        }
    }
    Ok(writes)
}
