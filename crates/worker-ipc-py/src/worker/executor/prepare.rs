//! Request commands and input ownership around numerical batch execution.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use uniserve_core::CallId;
use uniserve_worker_ipc::{BatchCommand, CallKind, MediaCall, TransferMode};

use super::super::error::{invalid, native_error};
use super::super::inputs::BatchInputs;
use super::{BatchState, PythonBackend};
use crate::convert::{RequestConversion, admission_to_py};

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
        let mut views = RequestConversion::new(py)?;
        let mut admitted = Vec::new();
        let started = (|| -> PyResult<()> {
            for command in &batch.plan.commands {
                let BatchCommand::Start { request } = command else {
                    continue;
                };
                let admission = admission_to_py(py, request, &mut views)?;
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
                        self.prepare_video_noise(py, &view)?;
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
        self.prepare_storage(py, batch)?;
        self.prepare_images(py, batch)?;
        Ok(())
    }

    /// Resolve imported KV and storage hazards before any numerical write.
    /// All Starts are already installed; these queries reserve no storage.
    fn prepare_storage(&self, py: Python<'_>, batch: &mut BatchState) -> PyResult<()> {
        let plan = &batch.plan;
        let requests = self.requests.borrow(py);
        let mut dependencies = Vec::new();

        if let Some(pool) = &self.latents {
            let mut pool = pool.borrow_mut(py);
            pool.inner.reap();
            let writes: HashSet<_> = plan
                .calls
                .iter()
                .filter(|call| {
                    matches!(
                        call.code,
                        CallKind::Media(MediaCall::LatentPreparation | MediaCall::Denoising)
                    )
                })
                .map(|call| (call.request_key, call.call_id))
                .collect();

            for params in &plan.latent_params {
                if !writes.contains(&(params.request_key, params.call_id)) {
                    continue;
                }
                let slot = requests
                    .pool
                    .get(params.request_key.request_id.0)
                    .map_err(|error| native_error(py, error))?
                    .slot();
                let slot = pool
                    .inner
                    .slot(slot as i64)
                    .map_err(|error| native_error(py, error))?;
                let pages = params
                    .page_table
                    .iter()
                    .map(|&page| page as usize)
                    .collect::<Vec<_>>();
                dependencies.extend(
                    pool.inner
                        .write_dependencies(slot, &pages)
                        .iter()
                        .map(|completion| completion.owner.clone_ref(py)),
                );
            }
        }

        // Supplied transfers take precedence. Retain resident descriptors by
        // shared reference so channel payloads are not copied during lookup.
        let mut resident = Vec::new();
        let mut supplied: HashSet<_> = plan.kv_inputs.iter().map(|export| export.source).collect();
        for call in &plan.calls {
            if call.code != CallKind::Transfer(TransferMode::KvInstall) {
                continue;
            }
            let source = call
                .kv_input
                .ok_or_else(|| invalid(py, "KV installation requires a source export"))?;
            if supplied.insert(source) {
                let cache = self
                    .cache
                    .as_ref()
                    .ok_or_else(|| invalid(py, "KV installation requires cache export storage"))?;
                let cache = cache.borrow(py);
                let export = cache
                    .inner
                    .resident(source)
                    .ok_or_else(|| invalid(py, "KV export buffer is not resident"))?;
                resident.push(Arc::clone(export));
            }
        }

        if let (Some(cache), Some(tables)) = (&self.cache, &self.tables) {
            let mut cache = cache.borrow_mut(py);
            if cache.inner.has_pending_accesses() {
                let tables = tables.borrow(py);
                let tables = &tables.tables;
                let unit_tokens = tables
                    .groups()
                    .iter()
                    .map(|group| group.page_tokens)
                    .max()
                    .unwrap_or(0);
                let mut spans = Vec::new();

                // A recycled unit can have belonged to another cache group.
                // Cover its full token capacity before its new owner writes.
                for allocation in &plan.new_cache_units {
                    spans.extend(
                        allocation
                            .unit_ids
                            .iter()
                            .map(|unit| (unit.0, 0, unit_tokens)),
                    );
                }
                for (row, &write_kv) in plan.forward.write_kv.iter().enumerate() {
                    if !write_kv {
                        continue;
                    }
                    let length = u64::from(plan.forward.query_lens[row]);
                    let start = u64::from(plan.forward.seq_lens[row]) - length;
                    for table in tables
                        .for_batch(plan.forward.request_pool_indices[row], &plan.block_tables)
                        .map_err(|error| native_error(py, error))?
                    {
                        spans.extend(
                            table
                                .spans(start, length)
                                .map_err(|error| native_error(py, error))?,
                        );
                    }
                }

                let imports: HashMap<_, _> = plan
                    .kv_inputs
                    .iter()
                    .chain(resident.iter().map(AsRef::as_ref))
                    .map(|export| (export.source, export))
                    .collect();
                for call in &plan.calls {
                    if call.code != CallKind::Transfer(TransferMode::KvInstall) {
                        continue;
                    }
                    let Some(export) = call.kv_input.and_then(|source| imports.get(&source)) else {
                        continue;
                    };
                    let slot = requests
                        .pool
                        .get(call.request_key.request_id.0)
                        .map_err(|error| native_error(py, error))?
                        .slot() as u32;
                    for table in tables
                        .for_batch(slot, &plan.block_tables)
                        .map_err(|error| native_error(py, error))?
                    {
                        let start = u64::from(export.base_extent)
                            .max(u64::from(table.start_page) * u64::from(table.shape.page_tokens));
                        let end = u64::from(export.exported_extent);
                        if start < end {
                            spans.extend(
                                table
                                    .spans(start, end - start)
                                    .map_err(|error| native_error(py, error))?,
                            );
                        }
                    }
                }

                // Query the combined footprint once. One execution completion
                // can cover many rows and groups in the same batch.
                dependencies.extend(
                    cache
                        .inner
                        .write_dependencies(&spans)
                        .into_iter()
                        .map(|completion| completion.owner.clone_ref(py)),
                );
            }
        }
        drop(requests);

        batch.imports =
            !plan.input_products.is_empty() || !plan.kv_inputs.is_empty() || !resident.is_empty();
        batch.inputs.borrow_mut(py).set_dependencies(dependencies);
        batch.resident_kv = resident;
        Ok(())
    }

    fn validate_batch(&self, py: Python<'_>, batch: &BatchState) -> PyResult<()> {
        let plan = &batch.plan;
        if let Some(cache) = &self.info.kv_cache
            && plan
                .new_cache_units
                .iter()
                .flat_map(|allocation| &allocation.unit_ids)
                .any(|unit| unit.0 >= cache.num_units)
        {
            return Err(invalid(py, "KV allocation exceeds the fixed physical pool"));
        }
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
        let config_native = crate::worker::config::native(&config)?;
        let rank: usize = config_native.rank;
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

        let max_calls: usize = config_native.max_batch_calls;
        if plan.calls.len() > max_calls {
            return Err(invalid(
                py,
                "execution batch exceeds the worker_config call limit",
            ));
        }
        let max_slots = u32::try_from(config_native.max_request_pool_size)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
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
        if !{
            self.model_runner
                .borrow(py)
                .video_postprocessor
                .bind(py)
                .clone()
        }
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
