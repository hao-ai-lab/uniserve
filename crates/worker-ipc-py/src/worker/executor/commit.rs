//! Commit batch resources together, or retire provisional work after failure.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use uniserve_worker_ipc::{CallKind, MediaCall};

use super::{BatchState, PythonBackend};
use crate::worker::error::{native_error, unsupported};
use crate::worker::exports;
use crate::worker::host::with_context;
use crate::worker::kv_cache::{installations_from_py, publications_from_py};
use crate::worker::latent::{LatentUpdate, lower_updates};
use crate::worker::output::OutputBuffer;
use crate::worker::pending::PendingOutput;
use crate::worker::storage::{Buffer, TensorRead};

impl PythonBackend {
    pub(super) fn commit_batch(&self, py: Python<'_>, batch: &mut BatchState) -> PyResult<()> {
        let numerical = batch.numerical.clone_ref(py);
        let scope = numerical.bind(py).call_method0("scope")?;
        with_context(&scope, || {
            let started: u64 = py
                .import("time")?
                .call_method0("perf_counter_ns")?
                .extract()?;
            let outputs = batch.pending_outputs(py)?;
            self.finish_reads(py, &outputs)?;
            self.runner
                .bind(py)
                .call_method1("prepare_outputs", (&numerical,))?;

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
                    "latent publication has no physical pool",
                ));
            }
            let buffer = numerical.bind(py).getattr("output_buffer")?;
            OutputBuffer::seal(buffer.cast()?)?;

            let products = PyList::empty(py);
            let publications = PyList::empty(py);
            let installations = PyList::empty(py);
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
                        format!("component {component_name:?} has no publication owner",),
                    ));
                }
            };
            for output in &outputs {
                let mut pending = output.borrow_mut();
                pending.reports_output = reports_output;
                for product in pending.products.bind(py) {
                    products.append(product)?;
                }
                if let Some(value) = &pending.cache_publication {
                    publications.append(value)?;
                }
                if let Some(value) = &pending.cache_installation {
                    installations.append(value)?;
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
            let stats = self.runner.bind(py).call_method1(
                "execution_stats",
                (
                    &numerical,
                    numerical.bind(py).getattr("started_ns")?,
                    started,
                ),
            )?;
            let publications = publications_from_py(publications.as_any())?;
            let installations = installations_from_py(installations.as_any())?;
            if let Some(cache) = &self.cache {
                cache
                    .borrow(py)
                    .inner
                    .validate_publications(&publications, &installations)
                    .map_err(|error| native_error(py, error))?;
            } else if !publications.is_empty() || !installations.is_empty() {
                return Err(PyRuntimeError::new_err(
                    "cache publication has no backing KV resources",
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
            if let Some(cache) = &self.cache {
                cache
                    .borrow_mut(py)
                    .inner
                    .apply_publications(publications, installations);
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
            numerical
                .bind(py)
                .setattr("products", PyTuple::new(py, products.iter())?)?;
            batch.record_execution(py, &stats)
        })
    }

    pub(super) fn discard_batch(&self, py: Python<'_>, batch: &BatchState) -> PyResult<()> {
        let numerical = batch.numerical.bind(py);
        let buffer = numerical.getattr("buffer")?;
        if buffer.is_none() {
            // The numerical reservation owns allocation failures before it
            // binds outputs. There are no batch resources to retire yet.
            return Ok(());
        }
        let scope = numerical.call_method0("scope")?;
        with_context(&scope, || {
            let outputs = batch.pending_outputs(py)?;
            self.finish_reads(py, &outputs)?;
            let buffer = buffer.cast::<OutputBuffer>()?;
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
                let mut slots = Vec::new();
                for output in &outputs {
                    let pending = output.borrow();
                    if pending
                        .latent
                        .bind(py)
                        .getattr("imported")?
                        .extract::<bool>()?
                    {
                        slots.push(pending.request.borrow(py).request.slot() as i64);
                    }
                }
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
