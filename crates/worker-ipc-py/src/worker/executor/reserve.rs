//! Bind a batch's output rows and physical resources before numerical launch.

use std::collections::HashMap;
use std::time::Instant;

use crate::worker::model_executor::ModelExecutor;
use indexmap::IndexMap;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PySlice, PyTuple};
use uniserve_worker_ipc::{CallKind, CallStatus, DType, MediaCall, TransferMode};

use super::{BatchState, PythonBackend};
use crate::worker::batch::BatchState as NumericalBatch;
use crate::worker::error::unsupported;
use crate::worker::host::with_context;
use crate::worker::output::OutputBuffer;
use crate::worker::pending::PendingOutput;
use crate::worker::storage::{Buffer, TensorRead};

impl PythonBackend {
    pub(super) fn reserve_batch(&self, py: Python<'_>, batch: &BatchState) -> PyResult<()> {
        let numerical = batch.numerical.bind(py);
        let predicated = NumericalBatch::resolve_predicates(numerical)?;
        let active: Vec<_> = batch
            .plan
            .calls
            .iter()
            .enumerate()
            .filter(|(_, call)| !predicated.contains(&call.request_key.request_id.0))
            .map(|(index, _)| index)
            .collect();
        let values = numerical.borrow().batch.clone_ref(py);
        let calls = values.bind(py).getattr("calls")?.cast_into::<PyTuple>()?;
        let model = self.model_runner.bind(py);
        let devices = calls
            .iter()
            .map(|call| {
                model
                    .call_method1("call_devices", (call,))?
                    .extract::<[Bound<'_, PyAny>; 3]>()
            })
            .collect::<PyResult<Vec<_>>>()?;

        if let Some(&index) = active.first() {
            let stream = ModelExecutor::call_stream(
                model,
                &*calls
                    .get_item(index)?
                    .extract::<PyRef<crate::calls::Call>>()?,
            )?
            .into_bound(py);
            if !stream.is_none() {
                // Slot resets run on the current stream. Independent model
                // streams join those resets before borrowing request storage.
                let current = py
                    .import("torch.cuda")?
                    .call_method1("current_stream", (stream.getattr("device")?,))?;
                stream.call_method1("wait_stream", (current,))?;
                numerical.borrow_mut().stream = Some(stream.unbind());
            }
        }

        let scope = NumericalBatch::scope(numerical)?;
        with_context(scope.bind(py), || {
            let started = Instant::now();
            // Payload capacity uses four-byte units in addition to int64
            // sampling columns. Saturated requests are rejected by OutputPool.
            let words = batch
                .plan
                .calls
                .iter()
                .fold(
                    self.sampling_columns.saturating_mul(batch.plan.calls.len()),
                    |total, call| {
                        total.saturating_add(
                            usize::try_from(call.bounds.max_completion_bytes.div_ceil(4))
                                .unwrap_or(usize::MAX),
                        )
                    },
                )
                .max(1);
            let buffer = self.output_pool.get().acquire(
                py,
                batch.plan.calls.len(),
                words,
                devices.iter().flat_map(|row| row.iter().cloned()).collect(),
            )?;
            if let Err(error) = NumericalBatch::bind_outputs(
                numerical,
                self.requests.bind(py),
                buffer.clone_ref(py),
                started,
                predicated,
            ) {
                OutputBuffer::abandon(buffer.bind(py))?;
                return Err(error);
            }

            // Once rows are bound, the executor's failure path retires every
            // accepted task, read and write through its normal resource owner.
            let outputs = batch.pending_outputs(py);
            self.reserve_host_tasks(py, batch, &active, &outputs)?;
            if !active.is_empty() {
                match (&self.cache, &self.tables) {
                    (Some(cache), Some(tables)) => {
                        let cache_views = self.worker.bind(py).getattr("kv_cache")?;
                        let table_views = self.worker.bind(py).getattr("block_tables")?;
                        numerical.borrow().bind_cache(
                            py,
                            cache.bind(py),
                            tables.bind(py),
                            &table_views.getattr("_copy_tables")?,
                            &cache_views.getattr("cache")?.getattr("recycle_units")?,
                        )?;
                    }
                    _ => {
                        if batch
                            .plan
                            .forward
                            .call_indices
                            .iter()
                            .any(|index| active.contains(&(*index as usize)))
                        {
                            return Err(unsupported(
                                py,
                                "KV-free execution received cache forward rows",
                            ));
                        }
                    }
                }
                let images = { model.borrow().image_builder.bind(py).clone() };
                let media = {
                    model
                        .borrow()
                        .media_builder
                        .as_ref()
                        .map(|builder| builder.bind(py).clone())
                };
                NumericalBatch::bind_latents(
                    numerical,
                    self.latents.as_ref().map(|pool| pool.bind(py)),
                    (!images.is_none()).then_some(&images),
                    media.as_ref(),
                )?;
            }
            self.bind_writes(py, batch, &outputs, &devices)?;
            NumericalBatch::complete_inputs(
                numerical,
                self.tensors.get(),
                self.latents.as_ref().map(|pool| pool.bind(py)),
                self.cache.as_ref().map(|cache| cache.bind(py)),
            )?;
            self.bind_predicates(py, batch, &active, &outputs, &devices)?;

            // Skipped calls still forward a false decision to their consumers.
            for output in &outputs {
                let writes = {
                    let output = output.borrow();
                    if output.lock(py)?.output.status != CallStatus::Predicated {
                        continue;
                    }
                    [&output.completion_write, &output.transition_write]
                        .into_iter()
                        .flatten()
                        .map(|write| {
                            write
                                .bind(py)
                                .extract::<Bound<'_, Buffer>>()
                                .map_err(Into::into)
                        })
                        .collect::<PyResult<Vec<_>>>()?
                };
                for write in writes {
                    self.tensors.get().write_scalar(
                        py,
                        write,
                        PyBool::new(py, false).to_owned().into_any(),
                        None,
                    )?;
                }
            }
            batch.record_component(py, "open_lane", started)?;
            Ok(())
        })
    }

    fn reserve_host_tasks(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        active: &[usize],
        outputs: &[Bound<'_, PendingOutput>],
    ) -> PyResult<()> {
        if active.is_empty() {
            return Ok(());
        }

        let call = &batch.plan.calls[0];
        if !matches!(
            call.code,
            CallKind::Media(
                MediaCall::ImageDecoding
                    | MediaCall::MediaReading
                    | MediaCall::VideoEncoding
                    | MediaCall::AudioEncoding
                    | MediaCall::Muxing
            )
        ) {
            return Ok(());
        }
        let rank = self.info.endpoint.rank as usize;
        let component = self
            .info
            .components
            .iter()
            .find(|component| component.name == call.component);
        let distributed =
            component.is_some_and(|component| component.config.distribution.is_some());
        if call.code != CallKind::Media(MediaCall::ImageDecoding)
            && !distributed
            && rank != self.output_rank(py, &call.component)?
        {
            return Ok(());
        }

        for &index in active {
            let call = &batch.plan.calls[index];
            let mut count = 1;
            if call.code == CallKind::Media(MediaCall::VideoEncoding)
                && let Some(component) = component
            {
                let position = component
                    .config
                    .ranks
                    .iter()
                    .position(|&member| member == rank)
                    .ok_or_else(|| {
                        PyValueError::new_err("video encoder rank is outside its component")
                    })?;
                let units = component.config.units_per_rank;
                let limit = batch
                    .plan
                    .decode_ranges
                    .iter()
                    .find(|params| {
                        params.request_key == call.request_key && params.call_id == call.call_id
                    })
                    .ok_or_else(|| {
                        PyRuntimeError::new_err("video encoding has no decode parameters")
                    })?
                    .max_units as usize;
                let first = position * units;
                let positions: Vec<_> = (first..(first + units).min(limit)).collect();
                count = positions.len();
                outputs[index].borrow_mut().media_units = positions;
            }
            for _ in 0..count {
                let task = self.host_tasks.get().reserve(py)?;
                outputs[index].borrow_mut().host_tasks.push(task);
            }
        }
        Ok(())
    }

    pub(super) fn output_rank(&self, py: Python<'_>, component: &str) -> PyResult<usize> {
        match self
            .info
            .components
            .iter()
            .find(|value| value.name == component)
        {
            Some(value) => Ok(value.config.ranks[0]),
            None if self.info.world_size == 1 => Ok(0),
            None => Err(unsupported(
                py,
                format!("component {component:?} has no export owner"),
            )),
        }
    }

    fn bind_writes(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        outputs: &[Bound<'_, PendingOutput>],
        devices: &[[Bound<'_, PyAny>; 3]],
    ) -> PyResult<()> {
        let values = batch.numerical.borrow(py).batch.clone_ref(py);
        let calls = values.bind(py).getattr("calls")?;
        let decodes = values.bind(py).getattr("decode_ranges")?;
        let decode_rows: HashMap<_, _> = batch
            .plan
            .decode_ranges
            .iter()
            .enumerate()
            .map(|(index, params)| (params.request_key.request_id.0, index))
            .collect();
        let regions = PyDict::new(py);
        let shapes = PyDict::new(py);
        let allocations = PyDict::new(py);
        for allocation in values.bind(py).getattr("buffer_allocations")?.try_iter()? {
            let allocation = allocation?;
            allocations.set_item(allocation.getattr("buffer")?, allocation)?;
        }
        let slots = PyDict::new(py);
        let mut scalar_groups = IndexMap::new();
        let mut persistent = Vec::new();
        let mut features = Vec::new();

        for (index, call) in batch.plan.calls.iter().enumerate() {
            let device = &devices[index][2];
            let device_name = device.str()?.to_str()?.to_owned();
            let value = calls.get_item(index)?;
            let (request, skipped) = {
                let output = outputs[index].borrow();
                (
                    output.request.clone_ref(py),
                    output.lock(py)?.output.status == CallStatus::Predicated,
                )
            };
            let (admission, slot) = {
                let request = request.borrow(py);
                (request.admission.clone_ref(py), request.request.slot())
            };
            let admission = admission.bind(py);
            slots.set_item(admission.getattr("request_key")?, slot)?;

            if !skipped {
                let decode = decode_rows
                    .get(&call.request_key.request_id.0)
                    .map_or_else(
                        || Ok(py.None().into_bound(py)),
                        |&row| decodes.get_item(row),
                    )?;
                let declared = value.getattr("outputs")?;
                for (row, output) in call.outputs.iter().enumerate() {
                    let reference = declared.get_item(row)?;
                    if call.code != CallKind::Transfer(TransferMode::Tensor) {
                        // Models describe mathematical layouts; execution binds
                        // each rank's local slice to the declared result buffer.
                        let conditions = admission.getattr("video")?;
                        let layout = ModelExecutor::output_layout(
                            self.model_runner.bind(py),
                            &call.component,
                            output.output_index as usize,
                            &admission.getattr("diffusion")?,
                            &decode,
                            admission.getattr("prompt_token_ids")?.len()?,
                            (!conditions.is_none()).then_some(&conditions),
                        )?
                        .into_bound(py);
                        if layout.is_none() {
                            continue;
                        }
                        let shape = layout.getattr("shape")?;
                        let region = layout.getattr("local_slice")?;
                        let whole = PyTuple::new(
                            py,
                            shape
                                .try_iter()?
                                .map(|extent| py.get_type::<PySlice>().call1((0, extent?)))
                                .collect::<PyResult<Vec<_>>>()?,
                        )?;
                        shapes.set_item(&reference, shape)?;
                        if !region.eq(whole)? {
                            regions.set_item(&reference, region)?;
                        }
                    }
                    persistent.push((reference, device.clone()));
                }
                if call.image_output.is_some() {
                    persistent.push((value.getattr("image_output")?, device.clone()));
                }
                if call.encoder_output.is_some() {
                    features.push((value.getattr("encoder_output")?, device.clone()));
                }
            }

            // A skipped call retains completion and transition outputs only.
            for (name, scalar) in [
                (
                    "token_output",
                    call.token_output.as_ref().filter(|_| !skipped),
                ),
                ("completion_output", call.completion_output.as_ref()),
                ("transition_output", call.transition_output.as_ref()),
            ] {
                if let Some(scalar) = scalar {
                    scalar_groups
                        .entry((
                            device_name.clone(),
                            scalar.dtype,
                            scalar.shape_bound.clone(),
                        ))
                        .or_insert_with(Vec::new)
                        .push((value.getattr(name)?, device.clone()));
                }
            }
        }

        let mut groups: Vec<_> = scalar_groups.into_values().collect();
        if !persistent.is_empty() {
            groups.push(persistent);
        }
        let bound = self.tensors.get().bind_output_groups(
            py,
            groups,
            Some(slots.into_any()),
            Some(allocations.clone().into_any()),
            Some(regions.into_any()),
            Some(shapes.into_any()),
        )?;
        let bound: Vec<Vec<Bound<'_, Buffer>>> = bound.extract()?;
        let owners: HashMap<_, _> = batch
            .plan
            .calls
            .iter()
            .enumerate()
            .map(|(index, call)| (call.request_key.request_id.0, index))
            .collect();

        // Retain every write before feature admission can fail. Native IDs
        // also select each write's output role without rebuilding Python keys.
        for write in bound.iter().flatten() {
            let id = write.get().id(py);
            let index = owners[&id.owner.request_id.0];
            let call = &batch.plan.calls[index];
            let mut output = outputs[index].borrow_mut();
            output.writes.bind(py).append(write)?;
            let transition = call
                .transition_output
                .as_ref()
                .is_some_and(|value| value.buffer_id() == id);
            if call
                .token_output
                .as_ref()
                .is_some_and(|value| value.buffer_id() == id)
            {
                output.token_write = Some(write.clone().into_any().unbind());
            } else if transition {
                output.transition_write = Some(write.clone().into_any().unbind());
            }
            if call
                .completion_output
                .as_ref()
                .is_some_and(|value| value.buffer_id() == id)
            {
                output.completion_write = Some(write.clone().into_any().unbind());
            }
            if !transition && output.producer_write.is_none() {
                output.producer_write = Some(write.clone().into_any().unbind());
            }
        }
        let features = self.tensors.get().reserve_features(
            py,
            features,
            allocations.into_any(),
            None,
            None,
        )?;
        for write in features.iter() {
            let write = write.cast_into::<Buffer>()?;
            let index = owners[&write.get().id(py).owner.request_id.0];
            outputs[index].borrow().writes.bind(py).append(write)?;
        }
        Ok(())
    }

    fn bind_predicates(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        active: &[usize],
        outputs: &[Bound<'_, PendingOutput>],
        devices: &[[Bound<'_, PyAny>; 3]],
    ) -> PyResult<()> {
        let mut groups: IndexMap<String, (Bound<'_, PyAny>, Vec<usize>)> = IndexMap::new();
        for &index in active {
            if batch.plan.calls[index].predicate.is_none() {
                continue;
            }
            let device = &devices[index][0];
            groups
                .entry(device.str()?.to_str()?.to_owned())
                .or_insert_with(|| (device.clone(), Vec::new()))
                .1
                .push(index);
        }
        let values = batch.numerical.borrow(py).batch.clone_ref(py);
        let calls = values.bind(py).getattr("calls")?;
        for (_, (device, indices)) in groups {
            let requests = indices
                .iter()
                .map(|&index| {
                    let call = calls.get_item(index)?;
                    Ok((call.getattr("predicate")?, call.getattr("call_id")?, None))
                })
                .collect::<PyResult<_>>()?;
            let reads = self
                .tensors
                .get()
                .consume_batch(py, requests, Some(device))?;
            let reads: Vec<Bound<'_, TensorRead>> = reads.extract()?;
            for (index, read) in indices.into_iter().zip(reads) {
                let tagged = batch.plan.calls[index]
                    .predicate
                    .as_ref()
                    .is_some_and(|value| value.dtype == DType::I64);
                let tensor = read.borrow().tensor.clone_ref(py);
                let mut output = outputs[index].borrow_mut();
                output.device_reads.bind(py).append(&read)?;
                output.predicate = Some((tensor, tagged).into_pyobject(py)?.into_any().unbind());
            }
        }
        Ok(())
    }
}
