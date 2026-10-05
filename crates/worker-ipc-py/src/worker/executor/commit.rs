//! Commit batch resources together, or retire provisional work after failure.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use uniserve_worker_ipc::{CallKind, MediaCall};

use super::{BatchState, PythonBackend};
use crate::worker::error::{native_error, unsupported};
use crate::worker::exports;
use crate::worker::host::with_context;
use crate::worker::latent::{LatentUpdate, lower_updates};
use crate::worker::output::OutputBuffer;
use crate::worker::pending::PendingOutput;
use crate::worker::storage::{Buffer, TensorRead};

impl PythonBackend {
    pub(super) fn commit_batch(&self, py: Python<'_>, batch: &mut BatchState) -> PyResult<()> {
        let numerical = batch.numerical.clone_ref(py);
        let scope = super::super::batch::BatchState::scope(numerical.bind(py))?;
        with_context(scope.bind(py), || {
            let started: u64 = py
                .import("time")?
                .call_method0("perf_counter_ns")?
                .extract()?;
            let outputs = batch.pending_outputs(py);
            self.finish_reads(py, &outputs)?;
            self.runner
                .bind(py)
                .call_method1("prepare_outputs", (&numerical,))?;

            for (output, call) in outputs.iter().zip(&batch.plan.calls) {
                output.borrow().validate_output(py, call)?;
            }

            let writes = output_writes(py, &outputs)?;
            self.tensors.get().validate_writes(py, writes.clone())?;
            let updates = outputs
                .iter()
                .map(|output| {
                    Ok(output
                        .borrow()
                        .latent
                        .bind(py)
                        .getattr("update")?
                        .extract()?)
                })
                .collect::<PyResult<Vec<Py<LatentUpdate>>>>()?;
            let updates = lower_updates(py, &updates)?;
            if let Some(latents) = &self.latents {
                latents.borrow_mut(py).validate(py, &updates)?;
            } else if updates.iter().any(|update| update.params.is_some()) {
                return Err(PyRuntimeError::new_err(
                    "latent export has no physical pool",
                ));
            }
            let buffer = numerical.borrow(py).output_buffer(py)?;
            OutputBuffer::seal(buffer.bind(py))?;

            let products = PyList::empty(py);
            let mut exports = Vec::new();
            let mut installations = Vec::new();
            let tensor_exports = PyDict::new(py);
            let cache_exports = PyDict::new(py);
            let latent_exports = PyDict::new(py);
            let rank: usize = self
                .worker
                .bind(py)
                .getattr("worker_config")?
                .getattr("rank")?
                .extract()?;
            // All calls share one component. Its first rank reports host
            // results; every participating rank retains its score rows.
            let component_name = &batch.plan.calls[0].component;
            let component = self
                .info
                .components
                .iter()
                .find(|component| &component.name == component_name);
            let reports_output = match component {
                Some(component) => component.config.ranks.first() == Some(&rank),
                None if self.info.world_size == 1 => rank == 0,
                None => {
                    return Err(unsupported(
                        py,
                        format!("component {component_name:?} has no export owner",),
                    ));
                }
            };
            for output in &outputs {
                let pending = output.borrow();
                pending.lock(py)?.reports_output = reports_output;
                for product in pending.products.bind(py) {
                    products.append(product)?;
                }
                if let Some(transfer) = &pending.cache_export {
                    exports.push((transfer.source, (**transfer).clone()));
                }
                if let Some((buffer, transfer)) = &pending.cache_installation {
                    installations.push((transfer.source, *buffer, (**transfer).clone()));
                }
                tensor_exports.update(pending.tensor_exports.bind(py).as_mapping())?;
                cache_exports.update(pending.cache_exports.bind(py).as_mapping())?;
                latent_exports.update(
                    pending
                        .latent
                        .bind(py)
                        .getattr("exports")?
                        .cast::<PyDict>()?
                        .as_mapping(),
                )?;
            }

            // Keep execution/commit timings at their established sampling
            // point, before cross-resource preflight and device-state updates.
            let started_ns = numerical.borrow(py).started_ns;
            let stats = self
                .runner
                .bind(py)
                .call_method1("execution_stats", (&numerical, started_ns, started))?;
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

            for (output, update) in outputs.iter().zip(&updates) {
                output.borrow().lock(py)?.apply_latent_update(update);
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
            self.runner
                .bind(py)
                .call_method1("apply_outputs", (&numerical,))?;
            for output in outputs {
                output.borrow_mut().release_execution_references(py)?;
            }
            self.requests
                .borrow_mut(py)
                .pool
                .add_pending(&calls)
                .map_err(|error| native_error(py, error))?;
            numerical.borrow_mut(py).products = PyTuple::new(py, products.iter())?.unbind();
            batch.record_execution(py, &stats)
        })
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

            let media_mux = self.worker.bind(py).getattr("media_mux")?;
            if !media_mux.is_none() {
                for call in &batch.plan.calls {
                    if call.code == CallKind::Media(MediaCall::LatentPreparation) {
                        media_mux.call_method1("drop", (call.request_key.request_id.0,))?;
                    }
                }
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
                let locators = output.borrow().exported_locators.clone_ref(py);
                for locator in locators.bind(py) {
                    transports
                        .get_item(locator.getattr("backend")?)?
                        .call_method1("release", (locator,))?;
                }
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
