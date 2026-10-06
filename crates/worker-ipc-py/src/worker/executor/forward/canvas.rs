//! Prepare readout and generation canvases, then capture their device results.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};

use super::{BatchState, ForwardRow, PythonBackend};
use crate::batches::CanvasSampling;
use crate::worker::error::{invalid, native_error};
use crate::worker::storage::Buffer;

impl PythonBackend {
    pub(super) fn prepare_canvas_rows(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        index: usize,
    ) -> PyResult<Vec<ForwardRow>> {
        let call = &batch.plan.calls[index];
        let pending = batch.pending(py, index);
        let output = pending.borrow(py);
        let request = output.request.borrow(py);
        let slot = request.request.slot();
        let visible = self.visible_length(py, &output)?;
        let position = output.lock(py)?.progress.logical_position;
        let numerical = batch.numerical.borrow(py);
        let descriptors = &numerical.forward_indices[index];
        let inputs = &batch.plan.forward;
        if descriptors.is_empty()
            || descriptors.iter().any(|&row| {
                inputs.write_kv[row]
                    || inputs.request_pool_indices[row] as usize != slot
                    || u64::from(inputs.seq_lens[row] - inputs.query_lens[row]) != visible
            })
        {
            return Err(invalid(
                py,
                "canvas rows must read the request's unchanged prefix",
            ));
        }

        let row_type = py.import("uniserve_worker.model_executor.input_batch")?;
        let options = PyDict::new(py);
        options.set_item(
            "forward_mode",
            crate::convert::RequestConversion::new(py)?.kind(call.code),
        )?;
        options.set_item("request_pool_idx", slot)?;
        options.set_item("seq_len", visible)?;
        options.set_item("write_kv", false)?;
        options.set_item("causal", false)?;
        let arange = py.import("torch")?.getattr("arange")?;

        if let Some(step) = call.canvas {
            let admission =
                request.request.admission().ar.as_ref().ok_or_else(|| {
                    invalid(py, "a canvas step requires admitted canvas sampling")
                })?;
            let sampling = admission
                .canvas
                .as_ref()
                .ok_or_else(|| invalid(py, "a canvas step requires admitted canvas sampling"))?;
            let seed = admission
                .sampling
                .seed
                .ok_or_else(|| invalid(py, "a canvas step requires a seeded request"))?;
            let slots = { self.model_runner.borrow(py).canvas_slots.bind(py).clone() };
            if slots.is_none() {
                return Err(invalid(py, "this worker keeps no generating canvases"));
            }
            let served = slots.getattr("served")?.cast_into::<CanvasSampling>()?;
            if &served.get().inner != sampling {
                return Err(invalid(
                    py,
                    "the admitted canvas sampling is not the sampling this worker serves",
                ));
            }
            let length = sampling.canvas_length;
            if descriptors.len() != 1
                || call.bounds.max_tokens != length
                || inputs.query_lens[descriptors[0]] != length
            {
                return Err(invalid(
                    py,
                    "a canvas step is one read-only canvas row over its prefix",
                ));
            }
            if step.step >= sampling.max_steps {
                return Err(invalid(py, "the canvas step exceeds its step limit"));
            }
            self.requests
                .borrow(py)
                .pool
                .advance_canvas(call.request_key, step)
                .map_err(|error| native_error(py, error))?;

            options.set_item(
                "positions",
                arange.call1((position, position + u64::from(length)))?,
            )?;
            options.set_item("canvas_length", length)?;
            // The sampler stores the unsigned seed in an int64 device column.
            options.set_item("seed", seed as i64)?;
            options.set_item("block", step.block)?;
            options.set_item("step", step.step)?;
            options.set_item("sampling", slots.getattr("constants")?)?;
            let row = row_type
                .getattr("CanvasStepRow")?
                .call((), Some(&options))?;
            return Ok(vec![ForwardRow::new(index, row.unbind())]);
        }

        let readout = call
            .readout
            .as_ref()
            .ok_or_else(|| invalid(py, "a canvas pass requires a readout"))?;
        if descriptors
            .iter()
            .map(|&row| inputs.query_lens[row] as usize)
            .sum::<usize>()
            != call.input_token_ids.len()
        {
            return Err(invalid(
                py,
                "canvas rows must partition the call's tokens over its prefix",
            ));
        }
        let tokens = py
            .import("uniserve_worker.execution.canvas")?
            .call_method1("token_values", (PyTuple::new(py, &call.input_token_ids)?,))?;
        let mut positions = HashMap::new();
        let mut rows = Vec::with_capacity(descriptors.len());
        let (mut first, mut slot_index) = (0usize, 0usize);
        for &descriptor in descriptors {
            let length = inputs.query_lens[descriptor] as usize;
            let end = first + length;
            let mut last = slot_index;
            while last < readout.slot_tokens.len() && (readout.slot_tokens[last] as usize) < end {
                last += 1;
            }
            if last == slot_index {
                return Err(invalid(py, "every canvas row must read a slot"));
            }
            let position_values = match positions.entry(length) {
                std::collections::hash_map::Entry::Occupied(value) => value.into_mut(),
                std::collections::hash_map::Entry::Vacant(value) => {
                    value.insert(arange.call1((position, position + length as u64))?)
                }
            };
            let offsets = &readout.candidate_offsets;
            options.set_item("positions", position_values.clone())?;
            options.set_item(
                "token_ids",
                tokens.get_item(PySlice::new(py, first as isize, end as isize, 1))?,
            )?;
            options.set_item(
                "slot_tokens",
                PyTuple::new(
                    py,
                    readout.slot_tokens[slot_index..last]
                        .iter()
                        .map(|token| *token as usize - first),
                )?,
            )?;
            options.set_item(
                "candidate_offsets",
                PyTuple::new(
                    py,
                    offsets[slot_index..=last]
                        .iter()
                        .map(|offset| offset - offsets[slot_index]),
                )?,
            )?;
            options.set_item(
                "candidate_ids",
                PyTuple::new(
                    py,
                    &readout.candidate_ids[offsets[slot_index] as usize..offsets[last] as usize],
                )?,
            )?;
            let row = row_type.getattr("CanvasRow")?.call((), Some(&options))?;
            rows.push(ForwardRow::new(index, row.unbind()));
            first = end;
            slot_index = last;
        }
        Ok(rows)
    }

    pub(super) fn capture_canvas_readout(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        index: usize,
        values: Vec<Py<PyAny>>,
    ) -> PyResult<()> {
        let call = &batch.plan.calls[index];
        let count = call
            .readout
            .as_ref()
            .ok_or_else(|| invalid(py, "a canvas pass requires a readout"))?
            .candidate_ids
            .len();
        let values = py
            .import("uniserve_worker.execution.canvas")?
            .call_method1("readout_values", (PyTuple::new(py, values)?, count))?;
        let buffer = batch.numerical.borrow(py).output_buffer(py)?;
        let span = buffer.get().capture(py, &values)?;
        let pending = batch.pending(py, index);
        let output = pending.borrow(py);
        let visible = self.visible_length(py, &output)?;
        let mut state = output.lock(py)?;
        state.candidate_range = Some(span);
        state.set_cache_length(visible);
        Ok(())
    }

    pub(super) fn capture_canvas_steps(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        steps: Vec<(usize, Py<PyAny>)>,
    ) -> PyResult<()> {
        if steps.is_empty() {
            return Ok(());
        }
        let widths: Vec<_> = steps
            .iter()
            .map(|(index, _)| 1 + batch.plan.calls[*index].bounds.max_tokens as usize)
            .collect();
        let values = PyTuple::new(py, steps.iter().map(|(_, value)| value.bind(py)))?;
        let numerical = py.import("uniserve_worker.execution.canvas")?;
        let block = numerical.call_method1("step_values", (values, PyTuple::new(py, &widths)?))?;
        let buffer = batch.numerical.borrow(py).output_buffer(py)?;
        let (offset, _) = buffer.get().capture(py, &block)?;
        let width = widths[0];
        let mut writes = Vec::new();
        let mut written_rows = Vec::new();
        for (row, (index, _)) in steps.into_iter().enumerate() {
            let pending = batch.pending(py, index);
            let output = pending.borrow(py);
            let visible = self.visible_length(py, &output)?;
            {
                let mut state = output.lock(py)?;
                state.canvas_range = Some((offset + row * width, width));
                state.set_cache_length(visible);
            }
            if let Some(write) = &output.completion_write {
                writes.push(write.bind(py).cast::<Buffer>()?.clone());
                written_rows.push(row);
            }
        }
        if let Some(first) = writes.first() {
            let dtype = first.get().tensor(py).bind(py).getattr("dtype")?;
            let values = numerical
                .call_method1("step_continuations", (&block, width, written_rows, dtype))?;
            self.tensors.get().write_scalars(py, writes, values, None)?;
        }
        Ok(())
    }
}
