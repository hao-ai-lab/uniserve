//! Select active calls and launch the batch's numerical operations.

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet, PyTuple};
use uniserve_worker_ipc::{CallKind, CallStatus, MediaCall, TransferMode};

use super::{BatchState, PythonBackend};
use crate::convert;
use crate::worker::error::{invalid, unsupported};
use crate::worker::host::with_context;
use crate::worker::kv_cache::KVCacheManager;

impl PythonBackend {
    pub(super) fn run_batch<'py>(
        &self,
        py: Python<'py>,
        batch: &mut BatchState,
    ) -> PyResult<Bound<'py, PyAny>> {
        if batch.plan.calls.is_empty() {
            return Ok(py.None().into_bound(py));
        }

        let numerical = batch.numerical.clone_ref(py);
        let clock = py.import("time")?.getattr("perf_counter_ns")?;
        let started: u64 = clock.call0()?.extract()?;
        let profiler = self.worker.bind(py).getattr("profiler")?;
        let scope = if profiler.is_none() {
            None
        } else {
            let first = &batch.plan.calls[0];
            Some(profiler.call_method1(
                "step",
                (format!("batch:{}:{}", first.code.as_str(), first.component),),
            )?)
        };
        let execute = || {
            let mut phase = "batch registration";
            let executed = (|| {
                self.reserve_batch(py, batch)?;
                phase = "batch execution";
                self.execute_calls(py, batch)?;
                phase = "batch commit";
                self.commit_batch(py, batch)?;
                Ok(())
            })();

            let Err(error): PyResult<()> = executed else {
                return Ok(py.None().into_bound(py));
            };
            let kwargs = PyDict::new(py);
            kwargs.set_item("phase", phase)?;
            kwargs.set_item("state", &numerical)?;
            kwargs.set_item("committed", batch.committed)?;
            let classified = py.import("uniserve_worker.errors")?.call_method(
                "classify_batch_failure",
                (error.value(py),),
                Some(&kwargs),
            )?;

            // Once stores become visible, rollback could invalidate a reader.
            // Earlier failures retire all reservations on the consuming stream.
            if !batch.committed {
                self.discard_batch(py, batch)?;
            }
            if batch.propagate_errors || classified.getattr("fatal")?.extract::<bool>()? {
                return Err(PyErr::from_value(classified));
            }

            let bound_at = numerical.borrow(py).started_ns;
            let elapsed = (clock.call0()?.extract::<u64>()?
                - if bound_at == 0 { started } else { bound_at })
                / 1000;
            let stats = numerical.borrow(py).execution_stats(py)?.inner;
            batch.record_execution(py, elapsed, stats)?;
            Ok(classified)
        };
        if let Some(scope) = scope {
            with_context(&scope, execute)
        } else {
            let mut execute = execute;
            execute()
        }
    }

    fn execute_calls(&self, py: Python<'_>, batch: &BatchState) -> PyResult<()> {
        let mut active = Vec::new();
        for (index, output) in batch.pending_outputs(py).iter().enumerate() {
            if output.borrow().lock(py)?.output.status != CallStatus::Predicated {
                active.push(index);
            }
        }

        if active.is_empty() {
            return Ok(());
        }

        // Numerical calls and native transfers record completion on the same
        // producer devices before launching any work into the output buffer.
        let scope = super::super::batch::BatchState::scope(batch.numerical.bind(py))?;
        with_context(scope.bind(py), || {
            let numerical = batch.numerical.borrow(py);
            let calls = numerical.batch.bind(py).getattr("calls")?;
            let buffer = numerical.output_buffer(py)?;
            drop(numerical);
            let devices = PySet::empty(py)?;
            for &index in &active {
                let selected = self
                    .model_runner
                    .bind(py)
                    .call_method1("call_devices", (calls.get_item(index)?,))?;
                for device in selected.try_iter()? {
                    let device = device?;
                    if !devices.contains(&device)? {
                        devices.add(&device)?;
                        buffer.get().begin_device(py, &device)?;
                    }
                }
            }
            Ok(())
        })?;

        if matches!(
            batch.plan.calls[0].code,
            CallKind::Transfer(TransferMode::KvExport | TransferMode::KvInstall)
        ) {
            let scope = super::super::batch::BatchState::scope(batch.numerical.bind(py))?;
            with_context(scope.bind(py), || self.execute_kv(py, batch, &active))?;
        } else if self.forward_calls.contains(&batch.plan.calls[0].code) {
            self.execute_forward(py, batch, &active)?;
        } else if batch.plan.calls[0].code == CallKind::Transfer(TransferMode::Tensor) {
            let scope = super::super::batch::BatchState::scope(batch.numerical.bind(py))?;
            with_context(scope.bind(py), || self.execute_tensors(py, batch, &active))?;
        } else {
            self.execute_media(py, batch, &active)?;
        }
        if active.iter().any(|&index| {
            batch.plan.calls[index].latent_output.is_some()
                || batch.plan.calls[index].encoder_output.is_some()
        }) {
            let scope = super::super::batch::BatchState::scope(batch.numerical.bind(py))?;
            with_context(scope.bind(py), || {
                self.export_latents(py, batch, &active)?;
                self.export_features(py, batch, &active)
            })?;
        }
        Ok(())
    }

    /// Run one homogeneous numerical batch at a time. Denoising intervals
    /// may have different lengths; only calls with another step participate.
    /// Request progress remains provisional until the entire batch commits.
    fn execute_forward(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        active: &[usize],
    ) -> PyResult<()> {
        let denoising = batch.plan.calls[0].code == CallKind::Media(MediaCall::Denoising);
        let mut intervals = vec![(0, 0); batch.plan.calls.len()];
        if denoising {
            let params: HashMap<_, _> = batch
                .plan
                .latent_params
                .iter()
                .map(|params| ((params.request_key, params.call_id), params))
                .collect();
            for &index in active {
                let call = &batch.plan.calls[index];
                // IPC requires an interval for each trajectory; numerical
                // reservation checks it against the admitted solver schedule.
                let params = params[&(call.request_key, call.call_id)];
                intervals[index] = (params.start_step, params.step_count);
            }
        } else {
            for &index in active {
                intervals[index] = (0, 1);
            }
        }
        let trajectories = if denoising {
            self.initialize_trajectories(py, batch, active)?
        } else {
            HashMap::new()
        };
        let step_count = intervals.iter().map(|&(_, count)| count).max().unwrap_or(0);

        for offset in 0..step_count {
            let steps: Vec<_> = intervals
                .iter()
                .enumerate()
                .filter(|&(_, &(_, count))| offset < count)
                .map(|(index, &(start, _))| (index, start + offset))
                .collect();
            let integrated = self.forward_step(py, batch, &steps, &trajectories)?;

            if denoising {
                // Pipeline ranks without a numerical output have no latent
                // result to capture, even on the interval's final step.
                let finished: Vec<_> = integrated
                    .into_iter()
                    .filter(|&index| offset + 1 == intervals[index].1)
                    .collect();
                if !finished.is_empty() {
                    self.finish_diffusion(py, batch, &finished)?;
                }
            }
        }
        Ok(())
    }

    /// KV calls advance storage and request state without a numerical model
    /// invocation. Tensor views and verified-length copies use pool callbacks.
    fn execute_kv(&self, py: Python<'_>, batch: &BatchState, active: &[usize]) -> PyResult<()> {
        let cache = self
            .cache
            .as_ref()
            .ok_or_else(|| invalid(py, "KV transfer requires cache storage"))?;
        let outputs = batch.pending_outputs(py);
        for &index in active {
            let call = &batch.plan.calls[index];
            let output = &outputs[index];
            let buffer = call
                .kv_output
                .ok_or_else(|| invalid(py, "KV transfer requires an output buffer"))?;
            if call.code == CallKind::Transfer(TransferMode::KvExport) {
                let transports = self.worker.bind(py).getattr("export_transports")?;
                if !transports.is_truthy()? {
                    return Err(unsupported(py, "KV export requires a configured transport"));
                }
                let (slot, visible) = {
                    let pending = output.borrow();
                    let slot = pending.request.borrow(py).request.slot() as u32;
                    let visible = pending.lock(py)?.progress.kv_visible_len;
                    (slot, visible)
                };
                let (transfer, locators) = KVCacheManager::export(
                    cache.bind(py),
                    slot,
                    visible,
                    "gen",
                    buffer,
                    &transports,
                    &call.consumer_slots,
                )?;
                let transfer = Arc::new(transfer);
                let mut pending = output.borrow_mut();
                let mut locations = Vec::new();
                for locator in locators.bind(py) {
                    pending.exported_locators.bind(py).append(&locator)?;
                    locations.push((transports.get_item(locator.getattr("backend")?)?, locator));
                }
                pending.cache_exports.bind(py).set_item(
                    convert::buffer_id_to_py(py, &buffer)?,
                    PyTuple::new(py, locations)?,
                )?;
                pending.lock(py)?.output.kv_output = Some((*transfer).clone());
                pending.cache_export = Some(transfer);
            } else {
                let source = call
                    .kv_input
                    .ok_or_else(|| invalid(py, "KV installation requires an input buffer"))?;
                let write = batch
                    .inputs
                    .borrow(py)
                    .cache_import(py, source)
                    .ok_or_else(|| invalid(py, "KV installation has no reserved physical input"))?;
                let imports = self
                    .cache_imports
                    .as_ref()
                    .ok_or_else(|| invalid(py, "KV installation requires cache imports"))?;
                let transfer = imports.borrow(py).adopt(py, write.get(), buffer)?;
                let mut pending = output.borrow_mut();
                pending
                    .lock(py)?
                    .set_cache_length(u64::from(transfer.exported_extent));
                pending.cache_installation = Some((buffer, transfer));
            }
        }
        Ok(())
    }
}
