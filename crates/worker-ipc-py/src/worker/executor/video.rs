//! Execute standalone video trajectories and reconstruct their media units.

use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use uniserve_worker_ipc::{CallKind, MediaCall};

use super::{BatchState, PythonBackend};
use crate::worker::error::invalid;
use crate::worker::host::{HostLane, HostTask};
use crate::worker::model_results::ExecutionOutput;
use crate::worker::request::Request;
use crate::worker::storage::TensorRead;

impl PythonBackend {
    /// Resolve immutable numerical dimensions once for this admitted request.
    fn video_state(&self, py: Python<'_>, request: &Py<Request>) -> PyResult<Py<PyAny>> {
        if let Some(state) = &request.borrow(py).diffusion {
            return Ok(state.clone_ref(py));
        }

        let model = self.model_runner.bind(py);
        let media = py.import("uniserve_worker.execution.media")?;
        let admission = request.borrow(py).admission.clone_ref(py);
        let size = media.call_method1("video_shape", (model, admission))?;
        let state = media.call_method1("open_state", (model, size))?.unbind();
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
        if denoising && !model.getattr("denoises")?.is_truthy()? {
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
            model
                .getattr("media_builder")?
                .call_method1("buffers", (size,))?
        } else {
            model
                .getattr("video_postprocessor")?
                .call_method1("state_buffers", (video_config(py, size)?,))?
        };
        let slot = request.borrow(py).request.slot();
        let storage = self.requests.borrow(py).storage.clone_ref(py);
        let views = storage
            .bind(py)
            .call_method1("tensors", (slot,))?
            .call_method1("view", (buffers,))?
            .unbind();
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
        let lane = model.getattr("noise_draws")?;
        if lane.is_none() || !model.getattr("state_buffers")?.is_truthy()? {
            return Ok(());
        }
        let state = self.video_state(py, request)?;
        let size = state.bind(py).getattr("size")?;
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
        let function = model
            .getattr("media_builder")?
            .getattr("prepare_request")?
            .unbind();
        let task = lane.cast::<HostLane>()?.borrow().reserve(py)?;
        if let Err(error) = HostTask::submit(task.bind(py), function, &args, Some(&options)) {
            task.borrow(py).abandon(py)?;
            return Err(error);
        }
        request.borrow_mut(py).video.preparation = Some(task);
        Ok(())
    }

    /// Acquire complete tensors together and retain every read through completion.
    fn video_inputs(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        index: usize,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let call = batch.call(py, index)?;
        let device = self
            .model_runner
            .bind(py)
            .call_method1("call_devices", (&call,))?
            .get_item(0)?;
        let id = call.getattr("call_id")?;
        let inputs = call
            .getattr("inputs")?
            .try_iter()?
            .map(|reference| Ok((reference?, id.clone(), Some(device.clone()))))
            .collect::<PyResult<Vec<_>>>()?;
        let reads = self.tensors.get().consume_batch(py, inputs, None)?;
        let reads = reads.extract::<Vec<Py<TensorRead>>>()?;
        let pending = batch.pending(py, index);
        let pending = pending.borrow(py);
        for read in &reads {
            pending.device_reads.bind(py).append(read.bind(py))?;
        }

        reads
            .iter()
            .map(|read| {
                let read = read.borrow(py);
                if read.region.is_some() || read.tensor.bind(py).is_none() {
                    return Err(invalid(
                        py,
                        "video computation requires complete input coverage",
                    ));
                }
                Ok(read.tensor.clone_ref(py))
            })
            .collect()
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
        let size = state.bind(py).getattr("size")?;
        let model = self.model_runner.bind(py);
        let numerical = py.import("uniserve_worker.execution.media")?;
        let slot = admitted.slot();

        match call.code {
            CallKind::Media(MediaCall::LatentPreparation | MediaCall::Denoising) => {
                let params = batch.latent_params(py, index)?;
                let views = self.video_views(py, &request, &size, true)?;
                let builder = model.getattr("media_builder")?;
                let layout = builder.call_method1("layout", (&size,))?;
                let context = model.call_method1("diffusion_layout", (&layout,))?;
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
                    let inputs = self.video_inputs(py, batch, index)?;
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
                        record_result(py, batch, &result.cast_into::<ExecutionOutput>()?)?;
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
                    let mut ladder = request
                        .borrow(py)
                        .video
                        .ladder
                        .as_ref()
                        .map(|value| value.clone_ref(py));
                    let diffusion = model.getattr("diffusion")?;
                    let reusable = match &ladder {
                        Some(ladder) => diffusion.call_method1("binds", (ladder,))?.is_truthy()?,
                        None => false,
                    };
                    if !reusable {
                        let bound = numerical
                            .call_method1(
                                "bind_video",
                                (model, &state, views, slot, PyTuple::new(py, pages)?),
                            )?
                            .unbind();
                        request.borrow_mut(py).video.ladder = Some(bound.clone_ref(py));
                        ladder = Some(bound);
                    }
                    let result = model
                        .call_method1("run_denoising", (ladder, params.start_step, source))?
                        .cast_into::<ExecutionOutput>()?;
                    record_result(py, batch, &result)?;
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
                        self.export_video_result(py, batch, index, &result, false)?;
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
                let inputs = self.video_inputs(py, batch, index)?;
                let result = if call.code == CallKind::Media(MediaCall::VideoDecoding) {
                    let windows = model
                        .getattr("video_decoder")?
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
                    record_result(py, batch, &decoded)?;
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
                record_result(py, batch, &result)?;
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

fn record_result(
    py: Python<'_>,
    batch: &BatchState,
    result: &Bound<'_, ExecutionOutput>,
) -> PyResult<()> {
    let result = result.borrow();
    let stats = result
        .stats
        .as_ref()
        .ok_or_else(|| PyRuntimeError::new_err("module output has no execution statistics"))?;
    batch
        .numerical
        .borrow_mut(py)
        .record_forward(stats.borrow(py));
    Ok(())
}
