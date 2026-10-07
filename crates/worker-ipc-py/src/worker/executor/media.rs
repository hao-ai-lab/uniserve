//! Submit media readers and codecs using request and batch-owned resources.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};
use uniserve_worker_ipc::{CallKind, MediaCall, TensorTransfer, TransferHandle, TransferTransport};

use super::{BatchState, PythonBackend};
use crate::worker::error::invalid;
use crate::worker::host::HostTask;
use crate::worker::locator::Locator;
use crate::worker::media::{
    HostTensors, MediaTask, MuxSession, host_array, read_encoded_unit, read_units,
};
use crate::worker::request::Request;
use crate::worker::shared_buffer::SharedRead;
use crate::worker::storage::{Buffer, TensorRead};
use crate::worker::transport::Transport;

impl PythonBackend {
    /// Retain whole input tensors through the enclosing call's completion.
    pub(super) fn media_inputs(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        index: usize,
        indices: std::ops::Range<usize>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let call = batch.call(py, index)?;
        let device = self
            .model_runner
            .borrow(py)
            .call_devices(py, &*call.extract::<PyRef<crate::calls::Call>>()?)?
            .into_bound(py)
            .get_item(0)?;
        let id = call.getattr("call_id")?;
        let values = call.getattr("inputs")?;
        let inputs = indices
            .map(|index| Ok((values.get_item(index)?, id.clone(), Some(device.clone()))))
            .collect::<PyResult<Vec<_>>>()?;
        let reads = self
            .tensors
            .get()
            .consume_batch(py, inputs, None)?
            .extract::<Vec<Py<TensorRead>>>()?;
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
                        "media computation requires complete input coverage",
                    ));
                }
                Ok(read.tensor.clone_ref(py))
            })
            .collect()
    }

    fn media_write(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        index: usize,
        output: usize,
    ) -> PyResult<Py<Buffer>> {
        let reference = &batch.plan.calls[index].outputs[output];
        let pending = batch.pending(py, index);
        let write = pending
            .borrow(py)
            .write_buffer(py, reference.buffer_id(), false)?;
        self.tensors.get().defer_write(py, write.bind(py))?;
        Ok(write)
    }

    pub(super) fn read_media(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        index: usize,
    ) -> PyResult<()> {
        let call = &batch.plan.calls[index];
        let pending = batch.pending(py, index);
        let request = pending.borrow(py).request.clone_ref(py);
        let video = request.borrow(py).admission.bind(py).getattr("video")?;
        if video.is_none() || !video.getattr("conditions")?.is_truthy()? {
            return Err(invalid(py, "media reading requires admitted conditions"));
        }

        let tasks = pending
            .borrow(py)
            .host_tasks
            .iter()
            .map(|task| task.clone_ref(py))
            .collect::<Vec<_>>();
        if tasks.len() != 1 {
            return Err(invalid(py, "media reading requires one reserved host task"));
        }

        let model = self.model_runner.borrow(py).model.bind(py).clone();
        let library = py.import("uniserve.model")?;
        let vision = crate::worker::model_executor::discovery::inputs::capability(
            &model,
            &library.getattr("PatchEncoder")?,
        )?;
        let audio = crate::worker::model_executor::discovery::inputs::capability(
            &model,
            &library.getattr("AudioEncoder")?,
        )?;
        if vision.is_none() || audio.is_none() {
            return Err(invalid(py, "media reading requires the condition encoders"));
        }

        let component = self
            .info
            .components
            .iter()
            .find(|component| component.name == call.component)
            .ok_or_else(|| invalid(py, "media reading has no declared component"))?;

        let options = PyDict::new(py);
        for name in ["pixels", "samples", "patches"] {
            options.set_item(name, py.None())?;
        }
        let mut values = Vec::new();
        for (index_in_call, output) in call.outputs.iter().enumerate() {
            let info = component
                .outputs
                .get(output.output_index as usize)
                .ok_or_else(|| invalid(py, "media reading declares no such product"))?;
            let name = match info.name.as_str() {
                "condition_pixels" => "pixels",
                "condition_samples" => "samples",
                "vision_pixels" => "patches",
                name => {
                    return Err(invalid(
                        py,
                        format!("media reading does not produce {name}"),
                    ));
                }
            };
            let write = self.media_write(py, batch, index, index_in_call)?;
            let value = write.get().tensor(py);
            options.set_item(name, &value)?;
            values.push(value);
        }

        options.set_item("vision", vision)?;
        options.set_item("sample_rate", audio.getattr("sample_rate")?)?;
        options.set_item("ffmpeg", &self.model_runner.borrow(py).config.ffmpeg)?;

        let conditions = video.getattr("conditions")?;
        let function = py
            .import("uniserve_worker.execution.media_reader")?
            .getattr("read_conditions")?;
        let action = py
            .import("functools")?
            .getattr("partial")?
            .call((function, &conditions), Some(&options))?;
        HostTask::configure(
            tasks[0].bind(py),
            action.unbind(),
            Vec::new(),
            None,
            None,
            None,
            &format!(
                "uniserve.host.read request={} conditions={}",
                call.request_key.request_id.0,
                conditions.len()?
            ),
        )?;

        // Finish uses call-owned tensors directly; it retains no batch callback.
        pending.borrow_mut(py).host_tensors = Some(HostTensors {
            values,
            encoded: false,
            store: self.tensors.clone_ref(py),
            transports: self.worker.bind(py).getattr("export_transports")?.unbind(),
        });
        Ok(())
    }

    pub(super) fn encode_media(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        index: usize,
    ) -> PyResult<()> {
        let call = &batch.plan.calls[index];
        let pending = batch.pending(py, index);
        let (request, tasks, positions) = {
            let pending = pending.borrow(py);
            (
                pending.request.clone_ref(py),
                pending
                    .host_tasks
                    .iter()
                    .map(|task| task.clone_ref(py))
                    .collect::<Vec<_>>(),
                pending.media_units.clone(),
            )
        };
        let config = self.media_config(py, &request)?;
        let transports = self
            .worker
            .bind(py)
            .getattr("export_transports")?
            .cast_into::<PyDict>()?;
        let profile = |operation, kind| {
            format!(
                "uniserve.host.{operation} request={}:{}:{} step={} call={} kind={kind} rank={}",
                call.request_key.engine_id,
                call.request_key.request_id.0,
                call.request_key.request_epoch,
                call.call_id.batch_id,
                call.call_id.request_index,
                self.info.endpoint.rank
            )
        };

        match call.code {
            CallKind::Media(MediaCall::VideoEncoding) => {
                if call.inputs.len() != 1
                    || call.outputs.len() != 1
                    || positions.len() != tasks.len()
                {
                    return Err(invalid(
                        py,
                        "video encoding requires an input round and one task per assigned unit",
                    ));
                }

                let cursor = batch
                    .plan
                    .decode_ranges
                    .iter()
                    .find(|params| {
                        params.call_id == call.call_id && params.request_key == call.request_key
                    })
                    .ok_or_else(|| invalid(py, "video encoding has no decoder interval"))?
                    .cursor as usize;
                let borrowed =
                    batch
                        .inputs
                        .borrow(py)
                        .is_borrowed(&crate::convert::buffer_id_to_py(
                            py,
                            &call.inputs[0].buffer_id(),
                        )?)?;
                let imported = if borrowed {
                    None
                } else {
                    self.media_inputs(py, batch, index, 0..1)?
                        .into_iter()
                        .next()
                };

                let write = self.media_write(py, batch, index, 0)?;
                let rows = write.get().tensor(py);
                if rows
                    .bind(py)
                    .getattr("shape")?
                    .extract::<Vec<usize>>()?
                    .first()
                    != Some(&positions.len())
                {
                    return Err(invalid(
                        py,
                        "encoded media output must reserve one row per unit",
                    ));
                }

                let height = config.bind(py).getattr("height")?.extract::<usize>()?;
                let width = config.bind(py).getattr("width")?.extract::<usize>()?;
                let frames = config.bind(py).getattr("video_unit_frames")?;

                for (position, task) in positions.iter().zip(tasks) {
                    let unit = cursor + position;
                    let expected = frames.get_item(unit)?.extract::<usize>()? * height * width * 3;
                    let source = if let Some(imported) = &imported {
                        let value = imported.bind(py).get_item(position)?;
                        let array = host_array(&value)?;
                        if array.len()? < expected {
                            return Err(invalid(
                                py,
                                "decoded media unit holds fewer bytes than its frames",
                            ));
                        }
                        array
                            .get_item(PySlice::new(py, 0, expected as isize, 1))?
                            .unbind()
                    } else {
                        let input = input_tensor(py, batch, index, 0)?;
                        let source = borrow_unit(py, input, *position, expected, &transports)?;
                        source.into_any()
                    };
                    let borrowed = source
                        .bind(py)
                        .cast::<SharedRead>()
                        .ok()
                        .map(|read| read.clone().unbind());
                    if let Err(error) = task.borrow(py).configure_media(
                        py,
                        MediaTask::Video {
                            config: config.clone_ref(py),
                            source,
                        },
                        format!("{} unit={unit}", profile("encode", "video")),
                    ) {
                        if let Some(borrowed) = borrowed {
                            borrowed.get().release(py)?;
                        }
                        return Err(error);
                    }
                }

                pending.borrow_mut(py).host_tensors = Some(HostTensors {
                    values: vec![rows],
                    encoded: true,
                    store: self.tensors.clone_ref(py),
                    transports: transports.into_any().unbind(),
                });
            }
            CallKind::Media(MediaCall::AudioEncoding | MediaCall::Muxing) => {
                if tasks.len() != 1 {
                    return Err(invalid(
                        py,
                        "media assembly requires one reserved host task",
                    ));
                }

                let session = self.mux_session(py, &request, config)?;
                let (action, name) =
                    if call.code == CallKind::Media(MediaCall::AudioEncoding) {
                        if call.inputs.len() != 1 {
                            return Err(invalid(py, "audio encoding requires one PCM input"));
                        }
                        let inputs = self.media_inputs(py, batch, index, 0..1)?;
                        let array = host_array(inputs[0].bind(py))?;
                        let pcm = array
                            .call_method1("view", (py.import("numpy")?.getattr("int16")?,))?
                            .unbind();
                        (
                            MediaTask::Audio { session, pcm },
                            profile("encode", "audio"),
                        )
                    } else {
                        let mut units = Vec::new();
                        for (input, reference) in call.inputs.iter().enumerate() {
                            if batch.inputs.borrow(py).is_borrowed(
                                &crate::convert::buffer_id_to_py(py, &reference.buffer_id())?,
                            )? {
                                units.extend(read_units(
                                    py,
                                    input_tensor(py, batch, index, input)?,
                                    &transports,
                                )?);
                            } else {
                                let values =
                                    self.media_inputs(py, batch, index, input..input + 1)?;
                                for row in values[0]
                                    .bind(py)
                                    .call_method1("unbind", (0,))?
                                    .try_iter()?
                                {
                                    units.push(read_encoded_unit(&row?)?);
                                }
                            }
                        }
                        if call.inputs.is_empty() {
                            (MediaTask::Finalize { session }, profile("mux", "artifact"))
                        } else {
                            (
                                MediaTask::Append { session, units },
                                profile("mux", "units"),
                            )
                        }
                    };
                tasks[0].borrow(py).configure_media(py, action, name)?;
            }
            _ => return Err(invalid(py, "unsupported host media operation")),
        }
        Ok(())
    }

    fn media_config(&self, py: Python<'_>, request: &Py<Request>) -> PyResult<Py<PyAny>> {
        let admission = request.borrow(py).admission.clone_ref(py);
        let media = admission.bind(py).getattr("diffusion")?;
        py.import("uniserve_worker.execution.media")?
            .call_method1("mux_config", (self.model_runner.bind(py), media))
            .map(Bound::unbind)
    }

    fn mux_session(
        &self,
        py: Python<'_>,
        request: &Py<Request>,
        config: Py<PyAny>,
    ) -> PyResult<Arc<MuxSession>> {
        if let Some(session) = &request.borrow(py).mux {
            return Ok(Arc::clone(session));
        }
        let session = Arc::new(MuxSession::new(py, config)?);
        request.borrow_mut(py).mux = Some(Arc::clone(&session));
        Ok(session)
    }
}

fn input_tensor<'a>(
    py: Python<'_>,
    batch: &'a BatchState,
    index: usize,
    input: usize,
) -> PyResult<&'a TensorTransfer> {
    let reference = &batch.plan.calls[index].inputs[input];
    let export = batch
        .plan
        .input_products
        .iter()
        .find(|export| export.product == *reference)
        .ok_or_else(|| invalid(py, "host media input has no transfer locations"))?;
    match &export.value {
        TransferHandle::DeviceProduct { tensor, .. } => Ok(tensor),
        _ => Err(invalid(py, "host media input requires a tensor")),
    }
}

fn borrow_unit(
    py: Python<'_>,
    tensor: &TensorTransfer,
    row: usize,
    nbytes: usize,
    transports: &Bound<'_, PyDict>,
) -> PyResult<Py<SharedRead>> {
    let transport = transports
        .get_item("shm")?
        .ok_or_else(|| invalid(py, "video encoding requires local shared storage"))?
        .cast_into::<Transport>()?;
    let location = tensor
        .locations
        .iter()
        .find(|location| {
            matches!(location.transport, TransferTransport::PosixShm { .. })
                && location.source.node == transport.get().source.node
                && location.offset[0] <= row as u64
                && (row as u64) < location.offset[0] + location.shape[0]
        })
        .ok_or_else(|| invalid(py, "decoded media unit has no local shared storage"))?;
    let region = PyTuple::new(
        py,
        std::iter::once(PySlice::new(py, row as isize, row as isize + 1, 1))
            .chain(
                tensor.shape[1..]
                    .iter()
                    .map(|&extent| PySlice::new(py, 0, extent as isize, 1)),
            )
            .collect::<Vec<_>>(),
    )?;
    let locator = Locator::wrap(py, location.clone())?;
    let read = transport
        .get()
        .borrow(py, &locator, Some(region.as_any()))?;
    read.truncate(py, nbytes)?;
    Py::new(py, read)
}
