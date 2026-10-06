//! Select the numerical input builder for each scheduled call.

use std::collections::HashMap;
use std::time::Instant;

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
        let image = py.import("uniserve_worker.execution.image")?;
        let mut forward = Vec::new();

        for &(index, _) in steps {
            let plan = &batch.plan.calls[index];
            if let Some(trajectory) = trajectories.get(&index) {
                forward.extend(self.prepare_diffusion_rows(
                    py,
                    batch,
                    index,
                    &step_inputs[&index],
                    trajectory,
                )?);
            } else if plan.code == CallKind::Forward(ForwardMode::TokenDenoising) {
                forward.extend(self.prepare_canvas_rows(py, batch, index)?);
            } else if matches!(plan.code, CallKind::Forward(_)) {
                let started = Instant::now();
                if plan.writes_context() {
                    forward.extend(self.prepare_context_rows(py, batch, index)?);
                } else if plan.writes_visual_state() {
                    forward.push(self.prepare_visual_row(py, batch, index)?);
                } else {
                    forward.push(self.prepare_token_row(py, batch, index)?);
                }
                batch.record_component(py, "text_build_batch", started)?;
            } else if matches!(
                plan.code,
                CallKind::Media(MediaCall::VisionEncoding | MediaCall::LatentEncoding)
            ) {
                let call = batch.call(py, index)?;
                let prepared = self.prepare_features(py, batch, index)?;
                let task = image.call_method1("encode_row", (call.getattr("kind")?, &prepared))?;
                forward.push(ForwardRow {
                    index,
                    task: task.unbind(),
                    image: Some(prepared),
                });
            } else if plan.latent_input.is_none() {
                // A resident decoded image needs only encoding, with no model row.
                self.finalize_image(py, batch, index)?;
            } else {
                let call = batch.call(py, index)?;
                let latent = self.image_latent(py, batch, index)?;
                let params = batch.latent_params(py, index)?;
                let row = PyDict::new(py);
                row.set_item("forward_mode", call.getattr("kind")?)?;
                row.set_item("latent", latent)?;
                row.set_item("image_height", params.height)?;
                row.set_item("image_width", params.width)?;
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
