//! Coordinate guidance prefixes and completed solver predictions.

use std::collections::HashMap;

use indexmap::IndexMap;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};

use super::{BatchState, DiffusionStep, ForwardRow, PythonBackend, Trajectories};
use crate::worker::error::invalid;
use crate::worker::pending::PendingOutput;

impl PythonBackend {
    pub(in super::super) fn initialize_trajectories(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        indices: &[usize],
    ) -> PyResult<Trajectories> {
        let diffusion = py.import("uniserve_worker.execution.diffusion")?;
        let options = PyDict::new(py);
        options.set_item("kv_cache", self.worker.bind(py).getattr("kv_cache")?)?;
        options.set_item("latent_pool", &self.latents)?;
        options.set_item(
            "request_tables",
            self.worker.bind(py).getattr("block_tables")?,
        )?;
        options.set_item("model_runner", &self.model_runner)?;
        options.set_item("state", &batch.numerical)?;
        indices
            .iter()
            .map(|&index| {
                let trajectory = diffusion.call_method(
                    "initialize",
                    (batch.call(py, index)?,),
                    Some(&options),
                )?;
                Ok((index, trajectory.unbind()))
            })
            .collect()
    }

    pub(in super::super) fn finish_diffusion(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        indices: &[usize],
        trajectories: &Trajectories,
    ) -> PyResult<()> {
        let diffusion = py.import("uniserve_worker.execution.diffusion")?;
        let options = PyDict::new(py);
        options.set_item("latent_pool", &self.latents)?;
        options.set_item("state", &batch.numerical)?;
        for &index in indices {
            diffusion.call_method(
                "finish",
                (batch.call(py, index)?, &trajectories[&index]),
                Some(&options),
            )?;
        }
        Ok(())
    }

    pub(super) fn prepare_diffusion_step(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        steps: &[(usize, u32)],
        trajectories: &Trajectories,
    ) -> PyResult<HashMap<usize, DiffusionStep>> {
        let diffusion = py.import("uniserve_worker.execution.diffusion")?;
        let options = PyDict::new(py);
        options.set_item(
            "request_tables",
            self.worker.bind(py).getattr("block_tables")?,
        )?;
        options.set_item("model_runner", &self.model_runner)?;
        options.set_item("latent_pool", &self.latents)?;
        options.set_item("tokenizer", self.worker.bind(py).getattr("tokenizer")?)?;
        options.set_item("state", &batch.numerical)?;
        let mut inputs = HashMap::new();
        let mut prefixes = Vec::new();
        let mut branches = Vec::new();
        for &(index, step) in steps {
            let prepared = diffusion.call_method(
                "prepare_step",
                (batch.call(py, index)?, &trajectories[&index], step),
                Some(&options),
            )?;
            inputs.insert(
                index,
                DiffusionStep {
                    guide: prepared.get_item(0)?.unbind(),
                    timestep: prepared.get_item(1)?.unbind(),
                    step,
                },
            );
            for prefix in prepared.get_item(2)?.try_iter()? {
                let prefix = prefix?;
                branches.push(prefix.get_item(0)?.unbind());
                prefixes.push(ForwardRow::new(index, prefix.get_item(1)?.unbind()));
            }
        }
        let values = self.forward_values(py, batch, &prefixes)?;
        for ((prefix, branch), result) in prefixes.into_iter().zip(branches).zip(values) {
            if result.is_none() {
                continue;
            }
            // Guidance writes its own KV slot, without advancing request KV.
            let task = prefix.task.bind(py);
            let count: usize = task.getattr("query_tokens")?.extract()?;
            self.record_token_kv(
                py,
                &mut batch.pending(py, prefix.index).borrow_mut(py),
                task,
                count as u64,
                false,
            )?;
            let kv = diffusion.call_method1("kv_conditioning", (&trajectories[&prefix.index],))?;
            let entries = kv.getattr("entries")?;
            let (slot, length, capacity): (usize, usize, usize) =
                entries.get_item(&branch)?.extract()?;
            entries.set_item(branch, (slot, length + count, capacity))?;
        }
        Ok(inputs)
    }

    pub(super) fn integrate_predictions(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        predictions: &IndexMap<usize, Vec<Py<PyAny>>>,
        inputs: &HashMap<usize, DiffusionStep>,
        trajectories: &Trajectories,
    ) -> PyResult<()> {
        for (&index, values) in predictions {
            let step = &inputs[&index];
            let values = PyTuple::new(py, values.iter().map(|value| value.bind(py)))?;
            let latent = self.latent_values(py, &batch.pending(py, index))?;
            let runner = self
                .model_runner
                .bind(py)
                .call_method1("diffusion_entry", (batch.call(py, index)?,))?;
            runner.call_method1(
                "integrate",
                (
                    &trajectories[&index],
                    latent,
                    &step.timestep,
                    values,
                    step.step,
                ),
            )?;
        }
        Ok(())
    }

    /// Borrow the model-visible latent rows, leaving physical page padding intact.
    pub(super) fn latent_values<'py>(
        &self,
        py: Python<'py>,
        output: &Py<PendingOutput>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let output = output.bind(py);
        let params = output.getattr("latent_params")?;
        let buffer = output.getattr("latent_buffer")?;
        if params.is_none() || buffer.is_none() {
            return Err(invalid(py, "trajectory call has no bound latent inputs"));
        }
        let count = params.getattr("latent_units")?.extract()?;
        buffer
            .getattr("value")?
            .get_item(PySlice::new(py, 0, count, 1))
    }
}
