//! Batch numerical execution: prepare rows, run homogeneous groups and bind results.

mod canvas;
mod context;
mod diffusion;
mod inputs;
mod results;
mod token;

use std::collections::HashMap;
use std::time::Instant;

use pyo3::exceptions::{PyBaseException, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PySlice, PyTuple};

use super::{BatchState, PythonBackend};
use crate::calls::Call;
use crate::worker::model_executor::ModelExecutor;
use crate::worker::model_results::ExecutionOutput;
use crate::worker::pending::PendingOutput;
use crate::worker::sampling::{self, SamplingMetadata};

/// One numerical row may share a call with other context or guidance rows.
struct ForwardRow {
    index: usize,
    task: Py<PyAny>,
    // Encoder preparation is retained until its features are written.
    image: Option<Py<PyAny>>,
}

impl ForwardRow {
    fn new(index: usize, task: Py<PyAny>) -> Self {
        Self {
            index,
            task,
            image: None,
        }
    }
}

struct ForwardValue {
    value: Py<PyAny>,
    slot: Py<PyAny>,
    selection: Option<Py<PyAny>>,
    layout: Py<PyAny>,
}

struct SampleCandidate {
    index: usize,
    task: Py<PyAny>,
    logits: Py<PyAny>,
    metadata: Option<Py<SamplingMetadata>>,
    selection: Option<Py<PyAny>>,
}

struct DiffusionStep {
    branches: Vec<String>,
    timestep: Py<PyAny>,
    step: u32,
}

type Trajectories = HashMap<usize, Py<PyAny>>;

impl BatchState {
    pub(super) fn call<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, Call>> {
        self.numerical
            .borrow(py)
            .batch
            .bind(py)
            .getattr("calls")?
            .get_item(index)?
            .cast_into::<Call>()
            .map_err(Into::into)
    }

    pub(super) fn pending(&self, py: Python<'_>, index: usize) -> Py<PendingOutput> {
        self.numerical.borrow(py).outputs[index].clone_ref(py)
    }

    /// Merge native host intervals with the numerical backend's component timers.
    pub(super) fn record_component(
        &self,
        py: Python<'_>,
        name: &str,
        started: Instant,
    ) -> PyResult<()> {
        let elapsed = started.elapsed().as_micros() as u64;
        let numerical = self.numerical.borrow(py);
        let components = numerical.component_us.bind(py);
        let previous = components
            .get_item(name)?
            .map_or(Ok(0), |value| value.extract::<u64>())?;
        components.set_item(name, previous.saturating_add(elapsed))
    }
}

impl PythonBackend {
    pub(super) fn forward_step(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        steps: &[(usize, u32)],
        trajectories: &Trajectories,
    ) -> PyResult<Vec<usize>> {
        let inputs = if trajectories.is_empty() {
            HashMap::new()
        } else {
            self.prepare_diffusion_step(py, batch, steps, trajectories)?
        };
        let rows = self.prepare_forward_rows(py, batch, steps, &inputs, trajectories)?;
        let values = self.forward_values(py, batch, &rows)?;
        let predictions = self.consume_forward_values(py, batch, rows, values, trajectories)?;
        self.integrate_predictions(py, batch, &predictions, &inputs, trajectories)?;
        Ok(predictions.into_keys().collect())
    }

    /// Materialize eager groups after their producer fence and vocabulary gather.
    /// Captured greedy groups retain their device selection instead. Every result
    /// keeps its original call index, including repeated guidance/context rows.
    fn forward_values(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        rows: &[ForwardRow],
    ) -> PyResult<Vec<Option<ForwardValue>>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let model = self.model_runner.bind(py);
        let buffer = batch.numerical.borrow(py).output_buffer(py)?;
        let mut tasks = Vec::with_capacity(rows.len());
        for row in rows {
            let call = batch.call(py, row.index)?;
            let device = model
                .borrow()
                .call_devices(py, &*call.extract::<PyRef<crate::calls::Call>>()?)?
                .into_bound(py)
                .get_item(1)?;
            buffer.get().register_device(py, &device)?;
            tasks.push((row.task.bind(py), call));
        }
        let runners = model;
        let mut groups = ModelExecutor::forward(
            runners,
            PyTuple::new(py, tasks)?.unbind(),
            self.worker.bind(py).getattr("kv_cache")?.unbind(),
            self.worker.bind(py).getattr("block_tables")?.unbind(),
            self.decode_state
                .as_ref()
                .map_or_else(|| py.None(), |value| value.clone_ref(py)),
        )?;
        let sampling_group = self.worker.bind(py).getattr("sampling_group")?;
        let mut results: Vec<Option<ForwardValue>> = (0..rows.len()).map(|_| None).collect();

        while let Some((indexes, output)) = groups.next(py)? {
            let output = output.into_bound(py);
            if output.is_instance_of::<PyBaseException>() {
                return Err(PyErr::from_value(output));
            }
            let output = output.cast_into::<ExecutionOutput>()?;
            let (stats, slots, greedy) = {
                let output = output.borrow();
                let (Some(stats), Some(slots)) = (&output.stats, &output.request_pool_indices)
                else {
                    return Err(PyRuntimeError::new_err(
                        "numerical forward lost statistics or request slot views",
                    ));
                };
                (
                    stats.clone_ref(py),
                    slots.clone_ref(py),
                    output.greedy.as_ref().map(|value| value.clone_ref(py)),
                )
            };
            let calls = indexes
                .iter()
                .map(|&index| batch.call(py, rows[index].index).map(Bound::unbind))
                .collect::<PyResult<Vec<_>>>()?;
            let pending = indexes
                .iter()
                .map(|&index| batch.pending(py, rows[index].index))
                .collect();
            let tasks = PyTuple::new(py, indexes.iter().map(|&index| rows[index].task.bind(py)))?;
            let selected = sampling::sample_graph(
                py,
                calls,
                pending,
                &tasks,
                greedy
                    .as_ref()
                    .map_or_else(|| py.None().into_bound(py), |value| value.bind(py).clone())
                    .as_any(),
                &sampling_group,
                slots.bind(py),
            )?;
            let output = if selected.is_none() {
                ExecutionOutput::materialize(&output)?.into_bound(py)
            } else {
                output
            };
            batch
                .numerical
                .borrow_mut(py)
                .forward_stats
                .merge(&stats.borrow(py).inner);
            let output = output.borrow();
            for (local, index) in indexes.into_iter().enumerate() {
                results[index] = Some(ForwardValue {
                    value: output.values.bind(py).get_item(local)?.unbind(),
                    slot: slots
                        .bind(py)
                        .get_item(PySlice::new(py, local as isize, local as isize + 1, 1))?
                        .unbind(),
                    selection: if selected.is_none() {
                        None
                    } else {
                        Some(selected.get_item(local)?.unbind())
                    },
                    layout: output.layouts.bind(py).get_item(local)?.unbind(),
                });
            }
        }
        Ok(results)
    }
}
