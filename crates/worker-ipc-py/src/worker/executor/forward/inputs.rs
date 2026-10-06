//! Select the numerical input builder for each scheduled call.

use std::collections::HashMap;
use std::time::Instant;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use uniserve_worker_ipc::{CallKind, ForwardMode, MediaCall};

use super::{BatchState, DiffusionStep, ForwardRow, PythonBackend, Trajectories};

impl PythonBackend {
    pub(super) fn prepare_forward_rows(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        steps: &[(usize, u32)],
        step_inputs: &HashMap<usize, DiffusionStep>,
        trajectories: &Trajectories,
    ) -> PyResult<Vec<ForwardRow>> {
        let model = self.model_runner.bind(py);
        let image = py.import("uniserve_worker.execution.image")?;
        let token = py.import("uniserve_worker.execution.token")?;
        let canvas = py.import("uniserve_worker.execution.canvas")?;
        let mut forward = Vec::new();

        for &(index, _) in steps {
            let call = batch.call(py, index)?;
            let plan = &batch.plan.calls[index];
            if let Some(trajectory) = trajectories.get(&index) {
                let input = &step_inputs[&index];
                let output = batch.pending(py, index);
                let values = self.latent_values(py, &output)?;
                let options = PyDict::new(py);
                let position = output.borrow(py).lock(py)?.progress.logical_position;
                options.set_item("conditioning_position", position)?;
                options.set_item(
                    "device",
                    model.call_method1("call_devices", (&call,))?.get_item(1)?,
                )?;
                let diffusion = py.import("uniserve_worker.execution.diffusion")?;
                let rows = diffusion.call_method(
                    "flow_rows",
                    (
                        diffusion.call_method1("require_inputs", (model,))?,
                        trajectory,
                        values,
                        &input.guide,
                        &input.timestep,
                    ),
                    Some(&options),
                )?;
                for task in rows.try_iter()? {
                    forward.push(ForwardRow::new(index, task?.unbind()));
                }
            } else if plan.code == CallKind::Forward(ForwardMode::TokenDenoising) {
                let options = PyDict::new(py);
                options.set_item("state", &batch.numerical)?;
                options.set_item(
                    "request_tables",
                    self.worker.bind(py).getattr("block_tables")?,
                )?;
                if plan.canvas.is_some() {
                    options.set_item("canvas_slots", model.getattr("canvas_slots")?)?;
                    let task = canvas.call_method("prepare_step", (&call,), Some(&options))?;
                    forward.push(ForwardRow::new(index, task.unbind()));
                } else {
                    let rows = canvas.call_method("prepare_rows", (&call,), Some(&options))?;
                    for task in rows.try_iter()? {
                        forward.push(ForwardRow::new(index, task?.unbind()));
                    }
                }
            } else if matches!(plan.code, CallKind::Forward(_)) {
                let started = Instant::now();
                let options = PyDict::new(py);
                options.set_item("tensor_store", &self.tensors)?;
                options.set_item(
                    "request_tables",
                    self.worker.bind(py).getattr("block_tables")?,
                )?;
                options.set_item("model_runner", model)?;
                options.set_item("state", &batch.numerical)?;
                if plan.writes_context() {
                    let rows = token.call_method("prepare_context", (&call,), Some(&options))?;
                    for task in rows.try_iter()? {
                        forward.push(ForwardRow::new(index, task?.unbind()));
                    }
                } else {
                    options.set_item("decode_state", &self.decode_state)?;
                    let task = token.call_method("prepare_forward", (&call,), Some(&options))?;
                    forward.push(ForwardRow::new(index, task.unbind()));
                }
                batch.record_component(py, "text_build_batch", started)?;
            } else if matches!(
                plan.code,
                CallKind::Media(MediaCall::VisionEncoding | MediaCall::LatentEncoding)
            ) {
                let options = PyDict::new(py);
                options.set_item("tensor_store", &self.tensors)?;
                options.set_item("model_runner", model)?;
                options.set_item("state", &batch.numerical)?;
                let prepared = image.call_method("prepare_features", (&call,), Some(&options))?;
                let task = image.call_method1("encode_row", (call.getattr("kind")?, &prepared))?;
                forward.push(ForwardRow {
                    index,
                    task: task.unbind(),
                    image: Some(prepared.unbind()),
                });
            } else if plan.latent_input.is_none() {
                // A resident decoded image needs only encoding, with no model row.
                let options = PyDict::new(py);
                options.set_item("tensor_store", &self.tensors)?;
                options.set_item("model_runner", model)?;
                options.set_item("state", &batch.numerical)?;
                image.call_method("diffusion_finalize_frames", (&call,), Some(&options))?;
            } else {
                let pool = self.latents.as_ref().ok_or_else(|| {
                    PyRuntimeError::new_err("image decoding requires a latent pool")
                })?;
                let options = PyDict::new(py);
                options.set_item("latent_pool", pool)?;
                options.set_item("model_runner", model)?;
                options.set_item("state", &batch.numerical)?;
                let latent =
                    image.call_method("materialization_latent", (&call,), Some(&options))?;
                let output = batch.pending(py, index);
                let params = output.bind(py).getattr("latent_params")?;
                let row = PyDict::new(py);
                row.set_item("forward_mode", call.getattr("kind")?)?;
                row.set_item("latent", latent)?;
                row.set_item("image_height", params.getattr("height")?)?;
                row.set_item("image_width", params.getattr("width")?)?;
                let task = py
                    .import("uniserve_worker.model_executor.image_inputs")?
                    .getattr("DecodeRow")?
                    .call((), Some(&row))?;
                forward.push(ForwardRow::new(index, task.unbind()));
            }
        }
        Ok(forward)
    }
}
