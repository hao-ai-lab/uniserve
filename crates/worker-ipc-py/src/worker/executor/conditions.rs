//! Execute text, vision and latent condition calls over retained tensor reads.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use uniserve_worker_ipc::{CallKind, MediaCall};

use super::{BatchState, PythonBackend};
use crate::worker::error::invalid;
use crate::worker::model_results::ExecutionOutput;
use crate::worker::pending::PendingOutput;

impl PythonBackend {
    pub(super) fn encode_conditions(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        index: usize,
    ) -> PyResult<()> {
        let call = &batch.plan.calls[index];
        let output = batch.pending(py, index);
        let request = output.borrow(py).request.clone_ref(py);
        let native = Arc::clone(&request.borrow(py).request);
        let admission = native.admission();
        let numerical = py.import("uniserve_worker.execution.conditions")?;
        let model = self.model_runner.bind(py);

        if call.code == CallKind::Media(MediaCall::TextEncoding) {
            if admission.diffusion.is_none() || admission.prompt_token_ids.is_empty() {
                return Err(invalid(
                    py,
                    "text conditioning requires admitted prompt tokens",
                ));
            }
            if call.outputs.is_empty() {
                return Err(invalid(py, "text encoding requires conditioning outputs"));
            }
            let options = PyDict::new(py);
            if !call.inputs.is_empty() {
                if call.inputs.len() != 1 {
                    return Err(invalid(py, "text encoding reads one vision product"));
                }
                let video = conditioned_video(py, &request.borrow(py))?;
                let visual = self.media_inputs(py, batch, index, 0..1)?;
                options.set_item("visual", &visual[0])?;
                options.update(
                    numerical
                        .call_method1("vision_grids", (video,))?
                        .cast::<PyDict>()?
                        .as_mapping(),
                )?;
            }
            let result = model
                .call_method(
                    "encode_text",
                    (PyTuple::new(py, &admission.prompt_token_ids)?,),
                    Some(&options),
                )?
                .cast_into::<ExecutionOutput>()?;
            let values = result.borrow().values.bind(py).iter().collect();
            self.export_conditions(py, batch, index, values, &result)?;
            return Ok(());
        }

        let video = conditioned_video(py, &request.borrow(py))?;
        if call.inputs.len() != 1 || call.outputs.len() != 1 {
            return Err(invalid(
                py,
                "condition encoding requires one input and one output",
            ));
        }
        let source = self.media_inputs(py, batch, index, 0..1)?;
        let encoded = match call.code {
            CallKind::Media(MediaCall::VisionEncoding) => {
                numerical.call_method1("vision_features", (video, &source[0], model))?
            }
            CallKind::Media(MediaCall::LatentEncoding) => {
                let component = self
                    .info
                    .components
                    .iter()
                    .find(|component| component.name == call.component)
                    .ok_or_else(|| invalid(py, "condition encoding has no component"))?;
                let declared = component
                    .outputs
                    .get(call.outputs[0].output_index as usize)
                    .ok_or_else(|| invalid(py, "condition encoding has no declared output"))?;
                match declared.name.as_str() {
                    "condition_video_latents" => {
                        let decode = batch
                            .plan
                            .decode_ranges
                            .iter()
                            .find(|decode| {
                                decode.request_key == call.request_key
                                    && decode.call_id == call.call_id
                            })
                            .ok_or_else(|| {
                                invalid(py, "condition encoding has no unit interval")
                            })?;
                        let total: usize = admission
                            .video
                            .iter()
                            .flat_map(|video| &video.conditions)
                            .map(|condition| condition.latent_units.len())
                            .sum();
                        if decode.max_units == 0
                            || decode.cursor as usize + decode.max_units as usize > total
                        {
                            return Err(invalid(
                                py,
                                "condition encoding exceeds the admitted units",
                            ));
                        }
                        let run = model
                            .getattr("bindings")?
                            .get_item(&call.component)?
                            .call_method1("media_units", (decode.cursor, decode.max_units))?;
                        if !run.is_truthy()? {
                            return Err(invalid(
                                py,
                                "condition encoding assigns this rank no unit",
                            ));
                        }
                        numerical.call_method1("encode_units", (video, run, &source[0], model))?
                    }
                    "condition_audio_latents" => {
                        numerical.call_method1("encode_tracks", (video, &source[0], model))?
                    }
                    _ => return Err(invalid(py, "condition encoding requires a latent output")),
                }
            }
            _ => return Err(invalid(py, "unsupported condition encoding operation")),
        };
        let result = encoded.get_item(1)?.cast_into::<ExecutionOutput>()?;
        self.export_conditions(py, batch, index, vec![encoded.get_item(0)?], &result)
    }

    fn export_conditions(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        index: usize,
        values: Vec<Bound<'_, PyAny>>,
        result: &Bound<'_, ExecutionOutput>,
    ) -> PyResult<()> {
        PendingOutput::export_tensors(
            batch.pending(py, index).bind(py),
            &batch.plan.calls[index],
            values,
            self.tensors.get(),
            &self.worker.bind(py).getattr("export_transports")?,
            false,
            None,
        )?;
        batch.record_result(py, result)
    }
}

fn conditioned_video<'py>(
    py: Python<'py>,
    request: &crate::worker::request::Request,
) -> PyResult<Bound<'py, PyAny>> {
    if request
        .request
        .admission()
        .video
        .as_ref()
        .is_none_or(|video| video.conditions.is_empty())
    {
        return Err(invalid(
            py,
            "condition encoding requires admitted conditions",
        ));
    }
    request.admission.bind(py).getattr("video")
}
