//! Route model results to request outputs and sample remaining token rows.

use std::time::Instant;

use indexmap::IndexMap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use uniserve_worker_ipc::{CallKind, ForwardMode};

use super::{BatchState, ForwardRow, ForwardValue, PythonBackend, SampleCandidate, Trajectories};
use crate::worker::sampling;

impl PythonBackend {
    pub(super) fn consume_forward_values(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        rows: Vec<ForwardRow>,
        values: Vec<Option<ForwardValue>>,
        trajectories: &Trajectories,
    ) -> PyResult<IndexMap<usize, Vec<Py<PyAny>>>> {
        let image = py.import("uniserve_worker.execution.image")?;
        let mut predictions = IndexMap::<usize, Vec<Py<PyAny>>>::new();
        let mut readouts = IndexMap::<usize, Vec<Py<PyAny>>>::new();
        let mut contexts = IndexMap::<usize, Vec<(ForwardRow, ForwardValue)>>::new();
        let mut canvas_steps = Vec::new();
        let mut samples = Vec::new();

        for (row, result) in rows.into_iter().zip(values) {
            let Some(result) = result else { continue };
            let index = row.index;
            let plan = &batch.plan.calls[index];
            if trajectories.contains_key(&index) {
                predictions.entry(index).or_default().push(result.value);
            } else if plan.code == CallKind::Forward(ForwardMode::TokenDenoising) {
                if plan.canvas.is_some() {
                    canvas_steps.push((index, result.value));
                } else {
                    readouts.entry(index).or_default().push(result.value);
                }
            } else if plan.writes_context() {
                // A context is committed only after all text/vision segments
                // return; accepting an individual row would hide later blocks.
                contexts.entry(index).or_default().push((row, result));
            } else if matches!(plan.code, CallKind::Forward(_)) {
                let metadata = if result.selection.is_some() {
                    self.record_token_kv(
                        py,
                        &mut batch.pending(py, index).borrow_mut(py),
                        row.task.bind(py),
                        1,
                        false,
                    )?;
                    None
                } else {
                    let Some(metadata) = self.prepare_sample(py, batch, &row, &result)? else {
                        continue;
                    };
                    Some(metadata)
                };
                samples.push(SampleCandidate {
                    index,
                    task: row.task,
                    logits: result.value,
                    metadata,
                    selection: result.selection,
                });
            } else if let Some(prepared) = &row.image {
                let call = batch.call(py, index)?;
                let options = PyDict::new(py);
                options.set_item("tensor_store", &self.tensors)?;
                options.set_item("state", &batch.numerical)?;
                image.call_method(
                    "write_features",
                    (&call, prepared, result.value),
                    Some(&options),
                )?;
            } else {
                let call = batch.call(py, index)?;
                let layout = result.layout.bind(py);
                let range = if layout.is_none() {
                    py.None().into_bound(py)
                } else {
                    layout.getattr("value_range")?
                };
                if range.is_none() {
                    return Err(PyValueError::new_err(
                        "image decoder must declare its numerical range",
                    ));
                }
                let options = PyDict::new(py);
                options.set_item("tensor_store", &self.tensors)?;
                options.set_item("state", &batch.numerical)?;
                image.call_method(
                    "export_image",
                    (&call, result.value.bind(py).call_method0("detach")?, range),
                    Some(&options),
                )?;
            }
        }

        for (index, rows) in contexts {
            if let Some(sample) = self.finish_context(py, batch, index, &rows)? {
                samples.push(sample);
            }
        }
        self.capture_canvas_steps(py, batch, canvas_steps)?;
        for (index, values) in readouts {
            self.capture_canvas_readout(py, batch, index, values)?;
        }
        self.finish_samples(py, batch, samples)?;
        Ok(predictions)
    }

    fn finish_samples(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        samples: Vec<SampleCandidate>,
    ) -> PyResult<()> {
        if samples.is_empty() {
            return Ok(());
        }
        let started = Instant::now();
        let tasks = samples
            .iter()
            .filter_map(|sample| sample.metadata.as_ref().map(|value| value.clone_ref(py)))
            .collect();
        let broadcast = py.import("functools")?.getattr("partial")?.call1((
            py.import("uniserve_worker.sampling.sampler")?
                .getattr("broadcast_selection")?,
            self.worker.bind(py).getattr("sampling_group")?,
        ))?;
        let selected = sampling::sample(py, tasks, Some(broadcast.unbind()))?;
        let mut selected = selected.iter();
        let values = samples
            .iter()
            .map(|sample| {
                sample.selection.as_ref().map_or_else(
                    || {
                        selected
                            .next()
                            .map(Bound::unbind)
                            .ok_or_else(|| PyValueError::new_err("sampler returned too few rows"))
                    },
                    |value| Ok(value.clone_ref(py)),
                )
            })
            .collect::<PyResult<Vec<_>>>()?;
        let output = batch.numerical.borrow(py).output_buffer(py)?;
        output.get().capture_samples(
            py,
            &values,
            samples
                .iter()
                .map(|sample| batch.pending(py, sample.index))
                .collect(),
        )?;
        batch.record_component(py, "text_sample", started)?;

        let started = Instant::now();
        self.write_sample_tensors(py, batch, &samples, &values)?;
        for (sample, selected) in samples.iter().zip(values) {
            self.finish_sample(py, batch, sample, selected)?;
        }
        batch.record_component(py, "text_finalize", started)
    }
}
