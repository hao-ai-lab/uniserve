//! Execute standalone video trajectories and reconstruct their media units.

use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use uniserve_worker_ipc::{CallKind, MediaCall};

use super::{BatchState, PythonBackend};
use crate::worker::diffusion_state::DiffusionState;
use crate::worker::error::invalid;
use crate::worker::host::HostTask;
use crate::worker::model_executor::ModelExecutor;
use crate::worker::model_results::ExecutionOutput;
use crate::worker::model_runner::{DenoisingSequence, DiffusionRunner};
use crate::worker::request::Request;

impl PythonBackend {
    /// Resolve immutable numerical dimensions once for this admitted request.
    fn video_state(&self, py: Python<'_>, request: &Py<Request>) -> PyResult<Py<DiffusionState>> {
        if let Some(state) = &request.borrow(py).diffusion {
            return Ok(state.clone_ref(py));
        }

        let model = self.model_runner.bind(py);
        let media = py.import("uniserve_worker.execution.media")?;
        let admission = request.borrow(py).admission.clone_ref(py);
        let size = media.call_method1("video_shape", (model, admission))?;
        let state = media
            .call_method1("open_state", (model, size))?
            .extract::<Py<DiffusionState>>()?;
        let mut request = request.borrow_mut(py);
        request.diffusion = Some(state.clone_ref(py));
        Ok(state)
    }

    /// Borrow retained slot views; runners own their constants and workspace.
    fn video_views(
        &self,
        py: Python<'_>,
        request: &Py<Request>,
        size: &Bound<'_, PyAny>,
        denoising: bool,
    ) -> PyResult<Py<PyAny>> {
        let model = self.model_runner.bind(py);
        if denoising && !model.borrow().denoises() {
            return Err(PyRuntimeError::new_err(
                "rank does not own denoising execution",
            ));
        }
        {
            let owner = request.borrow(py);
            let video = &owner.video;
            let views = if denoising {
                &video.denoising
            } else {
                &video.overlap
            };
            if let Some(views) = views {
                return Ok(views.clone_ref(py));
            }
        }

        let buffers = if denoising {
            { model.borrow().media_inputs(py)? }
                .borrow()
                .buffers(size)?
                .into_bound(py)
                .into_any()
        } else {
            { model.borrow().video_postprocessor.bind(py).clone() }
                .call_method1("state_buffers", (video_config(py, size)?,))?
        };
        let slot = request.borrow(py).request.slot();
        let storage = self.requests.borrow(py).storage.clone_ref(py);
        let views = storage
            .borrow(py)
            .tensors(py, slot as isize)?
            .borrow(py)
            .view(py, &buffers)?;
        let mut owner = request.borrow_mut(py);
        let video = &mut owner.video;
        if denoising {
            video.denoising = Some(views.clone_ref(py));
        } else {
            video.overlap = Some(views.clone_ref(py));
        }
        Ok(views)
    }

    /// Draw noise and fill CPU tables while independent device work proceeds.
    pub(super) fn prepare_video_noise(
        &self,
        py: Python<'_>,
        request: &Py<Request>,
    ) -> PyResult<()> {
        let model = self.model_runner.bind(py);
        let Some(lane) = model
            .borrow()
            .noise_draws
            .as_ref()
            .map(|lane| lane.clone_ref(py))
        else {
            return Ok(());
        };
        if !{ model.borrow().state_buffers.bind(py).clone() }.is_truthy()? {
            return Ok(());
        }
        let state = self.video_state(py, request)?;
        let size = state.borrow(py).size.bind(py).clone();
        let views = self.video_views(py, request, &size, true)?;
        let seed = request
            .borrow(py)
            .request
            .admission()
            .diffusion
            .as_ref()
            .ok_or_else(|| invalid(py, "video call has no admitted media dimensions"))?
            .seed;
        let options = PyDict::new(py);
        options.set_item("seed", seed)?;
        let args = PyTuple::new(py, [size, views.bind(py).clone()])?;
        let function = { model.borrow().media_inputs(py)? }
            .getattr("prepare_request")?
            .unbind();
        let task = lane.get().reserve(py)?;
        if let Err(error) = HostTask::submit(task.bind(py), function, &args, Some(&options)) {
            task.borrow(py).abandon(py)?;
            return Err(error);
        }
        request.borrow_mut(py).video.preparation = Some(task);
        Ok(())
    }

    pub(super) fn execute_video(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        index: usize,
    ) -> PyResult<()> {
        let call = &batch.plan.calls[index];
        let pending = batch.pending(py, index);
        let request = pending.borrow(py).request.clone_ref(py);
        let admitted = Arc::clone(&request.borrow(py).request);
        let media = admitted
            .admission()
            .diffusion
            .as_ref()
            .ok_or_else(|| invalid(py, "video call has no admitted media dimensions"))?;
        let state = self.video_state(py, &request)?;
        let size = state.borrow(py).size.bind(py).clone();
        let model = self.model_runner.bind(py);
        let numerical = py.import("uniserve_worker.execution.media")?;
        let slot = admitted.slot();

        match call.code {
            CallKind::Media(MediaCall::LatentPreparation | MediaCall::Denoising) => {
                let params = batch.latent_params(py, index)?;
                let views = self.video_views(py, &request, &size, true)?;
                let builder = { model.borrow().media_inputs(py)? };
                let layout = builder.call_method1("layout", (&size,))?;
                let context = ModelExecutor::diffusion_layout(model, &layout)?.into_bound(py);
                let pool = self
                    .latents
                    .as_ref()
                    .ok_or_else(|| invalid(py, "video trajectory requires a latent pool"))?;
                let pages = params
                    .page_table
                    .iter()
                    .map(|&page| i64::from(page))
                    .collect::<Vec<_>>();

                if call.code == CallKind::Media(MediaCall::LatentPreparation) {
                    let inputs = self.media_inputs(py, batch, index, 0..call.inputs.len())?;
                    let admission = request.borrow(py).admission.clone_ref(py);
                    let video = admission.bind(py).getattr("video")?;
                    let conditioned =
                        !video.is_none() && video.getattr("conditions")?.is_truthy()?;
                    if inputs.is_empty() || (inputs.len() > 1) != conditioned {
                        return Err(invalid(
                            py,
                            "video preparation reads its conditioning and its conditions",
                        ));
                    }
                    let preparation = request
                        .borrow(py)
                        .video
                        .preparation
                        .as_ref()
                        .map(|task| task.clone_ref(py));
                    if let Some(preparation) = preparation {
                        preparation.borrow(py).result(py, None)?;
                    } else {
                        let options = PyDict::new(py);
                        options.set_item("seed", media.seed)?;
                        builder.call_method("prepare_request", (&size, &views), Some(&options))?;
                    }

                    let samples = {
                        let mut pool = pool.borrow_mut(py);
                        let bank = pool.initial_bank(
                            py,
                            slot as i64,
                            pages.clone(),
                            i64::from(params.latent_units),
                        )?;
                        pool.bank_view(py, i64::from(bank), pages)?
                    };
                    let conditions = if conditioned {
                        py.import("uniserve_worker.execution.conditions")?
                            .call_method1(
                                "condition_latents",
                                (video, PyTuple::new(py, &inputs[1..])?),
                            )?
                    } else {
                        PyTuple::empty(py).into_any()
                    };
                    let result = numerical.call_method1(
                        "initialize_video",
                        (
                            model, &size, &views, context, samples, &inputs[0], conditions,
                        ),
                    )?;
                    if !result.is_none() {
                        batch.record_result(py, &result.cast_into::<ExecutionOutput>()?)?;
                    }
                    batch
                        .numerical
                        .borrow(py)
                        .complete_latent(py, call.request_key.request_id.0)?;
                } else {
                    let (source, _) = pool.borrow_mut(py).step_banks(
                        py,
                        slot as i64,
                        pages.clone(),
                        i64::from(params.start_step),
                        i64::from(params.start_step) + 1,
                        i64::from(params.latent_units),
                        i64::from(params.height),
                        i64::from(params.width),
                    )?;
                    let sequence = request
                        .borrow(py)
                        .video
                        .sequence
                        .as_ref()
                        .map(|value| value.clone_ref(py));
                    let diffusion = ModelExecutor::diffusion(model)?;
                    let reusable = match &sequence {
                        Some(sequence) => DiffusionRunner::binds(&diffusion, sequence.borrow(py))?,
                        None => false,
                    };
                    let sequence = match sequence {
                        Some(sequence) if reusable => sequence,
                        _ => {
                            let bound = numerical
                                .call_method1(
                                    "bind_video",
                                    (model, &state, views, slot, PyTuple::new(py, pages)?),
                                )?
                                .unbind();
                            let bound: Py<DenoisingSequence> = bound.extract(py)?;
                            request.borrow_mut(py).video.sequence = Some(bound.clone_ref(py));
                            bound
                        }
                    };
                    let result = ModelExecutor::run_denoising(
                        model,
                        sequence.bind(py),
                        params.start_step as usize,
                        i64::from(source),
                    )?;
                    batch.record_result(py, result.bind(py))?;
                    batch
                        .numerical
                        .borrow(py)
                        .complete_latent(py, call.request_key.request_id.0)?;

                    if !call.outputs.is_empty() {
                        if params.start_step + params.step_count != media.num_inference_steps {
                            return Err(invalid(
                                py,
                                "final latent products require completed denoising",
                            ));
                        }
                        self.export_video_result(py, batch, index, result.bind(py), false)?;
                    }
                }
            }
            CallKind::Media(MediaCall::VideoDecoding | MediaCall::AudioDecoding) => {
                let params = batch
                    .plan
                    .decode_ranges
                    .iter()
                    .find(|params| {
                        params.request_key == call.request_key && params.call_id == call.call_id
                    })
                    .ok_or_else(|| invalid(py, "video decode call has no decode params"))?;
                if call.inputs.len() != 1 {
                    return Err(invalid(
                        py,
                        "media reconstruction requires one Tensor input",
                    ));
                }
                let inputs = self.media_inputs(py, batch, index, 0..call.inputs.len())?;
                let result = if call.code == CallKind::Media(MediaCall::VideoDecoding) {
                    let windows = { model.borrow().video_decoder.bind(py).clone() }
                        .call_method1("frame_slices", (media.num_frames,))?;
                    let units = numerical.call_method1(
                        "assigned_units",
                        (
                            model,
                            &call.component,
                            params.cursor,
                            params.max_units,
                            windows.len()?,
                        ),
                    )?;
                    let window = windows.get_item(units.getattr("start")?)?;
                    let views = self.video_views(py, &request, &size, false)?;
                    let results = numerical.call_method1(
                        "reconstruct_video",
                        (
                            model,
                            &call.component,
                            &inputs[0],
                            window,
                            video_config(py, &size)?,
                            views,
                            params.max_units,
                        ),
                    )?;
                    let decoded = results.get_item(0)?.cast_into::<ExecutionOutput>()?;
                    batch.record_result(py, &decoded)?;
                    results.get_item(1)?.cast_into::<ExecutionOutput>()?
                } else {
                    let options = PyDict::new(py);
                    options.set_item("cursor", params.cursor)?;
                    options.set_item("count", params.max_units)?;
                    let samples =
                        numerical.call_method1("audio_samples", (model, media.num_frames))?;
                    numerical
                        .call_method(
                            "decode_audio",
                            (model, &call.component, &inputs[0], samples),
                            Some(&options),
                        )?
                        .cast_into::<ExecutionOutput>()?
                };
                batch.record_result(py, &result)?;
                self.export_video_result(py, batch, index, &result, true)?;
            }
            _ => return Err(invalid(py, "unsupported video computation")),
        }
        Ok(())
    }

    fn export_video_result(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        index: usize,
        result: &Bound<'_, ExecutionOutput>,
        host: bool,
    ) -> PyResult<()> {
        batch.numerical.borrow(py).export_tensors(
            py,
            batch.plan.calls[index].request_key.request_id.0,
            result.borrow().values.bind(py).iter().collect(),
            self.tensors.get(),
            &self.worker.bind(py).getattr("export_transports")?,
            host,
            None,
        )
    }
}

fn video_config<'py>(py: Python<'py>, size: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    py.import("uniserve.media.video")?
        .getattr("Config")?
        .call1((size.getattr("num_frames")?, size.getattr("canvas")?))
}
