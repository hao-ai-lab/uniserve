//! Route model results to request outputs and sample remaining token rows.

use std::time::Instant;

use indexmap::IndexMap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use uniserve_worker_ipc::{CallKind, ForwardMode};

use super::{BatchState, ForwardRow, ForwardValue, PythonBackend, SampleCandidate, Trajectories};
use crate::worker::pending::PendingOutput;
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
        let token = py.import("uniserve_worker.execution.token")?;
        let image = py.import("uniserve_worker.execution.image")?;
        let canvas = py.import("uniserve_worker.execution.canvas")?;
        let mut predictions = IndexMap::<usize, Vec<Py<PyAny>>>::new();
        let mut readouts = IndexMap::<usize, Vec<Py<PyAny>>>::new();
        let mut contexts = IndexMap::<usize, Vec<(Py<PyAny>, Py<PyAny>, Py<PyAny>)>>::new();
        let mut canvas_steps = Vec::new();
        let mut samples = Vec::new();

        for (row, result) in rows.into_iter().zip(values) {
            let Some(result) = result else { continue };
            let index = row.index;
            let call = batch.call(py, index)?;
            let plan = &batch.plan.calls[index];
            if trajectories.contains_key(&index) {
                predictions.entry(index).or_default().push(result.value);
            } else if plan.code == CallKind::Forward(ForwardMode::TokenDenoising) {
                if plan.canvas.is_some() {
                    canvas_steps.push((call.unbind(), result.value));
                } else {
                    readouts.entry(index).or_default().push(result.value);
                }
            } else if plan.writes_context() {
                // A context is committed only after all text/vision segments
                // return; accepting an individual row would hide later blocks.
                contexts
                    .entry(index)
                    .or_default()
                    .push((row.task, result.value, result.slot));
            } else if matches!(plan.code, CallKind::Forward(_)) {
                let options = self.token_options(py, batch)?;
                let metadata = if result.selection.is_some() {
                    options.del_item("state")?;
                    options.set_item("publish_runtime", false)?;
                    token.call_method(
                        "commit_kv",
                        (&row.task, 1, batch.pending(py, index)),
                        Some(&options),
                    )?;
                    None
                } else {
                    options.set_item("request_pool_index", &result.slot)?;
                    options.set_item("tensor_store", &self.tensors)?;
                    options.set_item(
                        "image_builder",
                        self.model_runner.bind(py).getattr("image_builder")?,
                    )?;
                    let prepared = token.call_method(
                        "prepare_sampling",
                        (&call, &row.task, &result.value),
                        Some(&options),
                    )?;
                    if prepared.is_instance_of::<PendingOutput>() {
                        continue;
                    }
                    Some(prepared.extract()?)
                };
                samples.push(SampleCandidate {
                    index,
                    task: row.task,
                    logits: result.value,
                    metadata,
                    selection: result.selection,
                });
            } else if let Some(prepared) = &row.image {
                let options = PyDict::new(py);
                options.set_item("tensor_store", &self.tensors)?;
                options.set_item("state", &batch.numerical)?;
                image.call_method(
                    "write_features",
                    (&call, prepared, result.value),
                    Some(&options),
                )?;
            } else {
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
            let options = self.token_options(py, batch)?;
            let rows = PyTuple::new(py, rows)?;
            let result = token.call_method(
                "finish_context",
                (batch.call(py, index)?, rows),
                Some(&options),
            )?;
            if !result.is_instance_of::<PendingOutput>() {
                samples.push(SampleCandidate {
                    index,
                    task: result.get_item(0)?.unbind(),
                    logits: result.get_item(1)?.unbind(),
                    metadata: Some(result.get_item(2)?.extract()?),
                    selection: None,
                });
            }
        }

        let options = PyDict::new(py);
        options.set_item(
            "request_tables",
            self.worker.bind(py).getattr("block_tables")?,
        )?;
        options.set_item("state", &batch.numerical)?;
        options.set_item("tensor_store", &self.tensors)?;
        canvas.call_method(
            "publish_steps",
            (PyTuple::new(py, canvas_steps)?,),
            Some(&options),
        )?;
        options.del_item("tensor_store")?;
        for (index, values) in readouts {
            canvas.call_method(
                "capture_readout",
                (batch.call(py, index)?, PyList::new(py, values)?),
                Some(&options),
            )?;
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
            values.iter().map(|value| value.clone_ref(py)).collect(),
            samples
                .iter()
                .map(|sample| batch.pending(py, sample.index))
                .collect(),
        )?;
        batch.record_component(py, "text_sample", started)?;

        let started = Instant::now();
        let token = py.import("uniserve_worker.execution.token")?;
        let calls = samples
            .iter()
            .map(|sample| batch.call(py, sample.index))
            .collect::<PyResult<Vec<_>>>()?;
        let options = PyDict::new(py);
        options.set_item("tensor_store", &self.tensors)?;
        options.set_item("state", &batch.numerical)?;
        token.call_method(
            "publish_token_products",
            (
                PyTuple::new(py, calls)?,
                PyTuple::new(py, values.iter().map(|value| value.bind(py)))?,
            ),
            Some(&options),
        )?;
        let options = self.token_options(py, batch)?;
        options.set_item(
            "image_builder",
            self.model_runner.bind(py).getattr("image_builder")?,
        )?;
        for (sample, selected) in samples.into_iter().zip(values) {
            token.call_method(
                "publish_sample",
                (
                    batch.call(py, sample.index)?,
                    sample.task,
                    sample.logits,
                    sample.metadata,
                    selected,
                ),
                Some(&options),
            )?;
        }
        batch.record_component(py, "text_finalize", started)
    }
}
