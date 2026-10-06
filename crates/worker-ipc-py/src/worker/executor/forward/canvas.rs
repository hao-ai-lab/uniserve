//! Capture canvas results and bind device continuation outputs.

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use super::{BatchState, PythonBackend};
use crate::worker::error::invalid;
use crate::worker::storage::Buffer;

impl PythonBackend {
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
