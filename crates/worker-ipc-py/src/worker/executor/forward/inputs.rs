//! Select the numerical input builder for each scheduled call.

use std::collections::HashMap;
use std::time::Instant;

use pyo3::prelude::*;
use uniserve_worker_ipc::{CallKind, ForwardMode, MediaCall};

use super::{BatchState, DiffusionStep, ForwardRow, PythonBackend, Trajectories};
use crate::worker::model_inputs::{DecodeRow, InputRow, VisionRow};

impl PythonBackend {
    pub(super) fn prepare_forward_rows(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        steps: &[(usize, u32)],
        step_inputs: &HashMap<usize, DiffusionStep>,
        trajectories: &Trajectories,
    ) -> PyResult<Vec<ForwardRow>> {
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
                let prepared = self.prepare_features(py, batch, index)?;
                let pixels = prepared.bind(py);
                let row = VisionRow {
                    encode_pixels: pixels.getattr("pixels")?.unbind(),
                    encode_grid: pixels.getattr("grid")?.extract()?,
                    encode_grid_shape: pixels.getattr("grid_shape")?.extract()?,
                };
                let input = InputRow {
                    kind: plan.code,
                    request_pool_idx: 0,
                };
                let task =
                    Bound::new(py, PyClassInitializer::from(input).add_subclass(row))?.into_any();
                forward.push(ForwardRow {
                    index,
                    task: task.unbind(),
                    image: Some(prepared),
                });
            } else if plan.latent_input.is_none() {
                // A resident decoded image needs only encoding, with no model row.
                self.finalize_image(py, batch, index)?;
            } else {
                let latent = self.image_latent(py, batch, index)?;
                let params = batch.latent_params(py, index)?;
                let row = DecodeRow {
                    latent,
                    image_height: params.height as usize,
                    image_width: params.width as usize,
                };
                let input = InputRow {
                    kind: plan.code,
                    request_pool_idx: 0,
                };
                let task =
                    Bound::new(py, PyClassInitializer::from(input).add_subclass(row))?.into_any();
                forward.push(ForwardRow::new(index, task.unbind()));
            }
        }
        Ok(forward)
    }
}
