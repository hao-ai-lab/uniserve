//! Image preparation, feature writes and output encoding on native owners.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::{BatchState, PythonBackend};
use crate::worker::error::invalid;
use crate::worker::host::HostTask;
use crate::worker::media::MediaTask;

impl PythonBackend {
    /// Prepare inline inputs while other batches execute. BatchInputs owns every
    /// reservation before it can fail or begin writing its pinned host tensors.
    pub(super) fn prepare_images(&self, py: Python<'_>, batch: &BatchState) -> PyResult<()> {
        let calls = batch
            .plan
            .calls
            .iter()
            .enumerate()
            .filter(|(_, call)| call.input_image.is_some())
            .collect::<Vec<_>>();
        if calls.is_empty() {
            return Ok(());
        }
        let model = self.model_runner.bind(py);
        let processor = model.borrow().image_processor(py)?.into_bound(py);
        let prepare = py
            .import("uniserve_worker.model_executor.image_inputs")?
            .getattr("prepare_host_image")?;
        let partial = py.import("functools")?.getattr("partial")?;

        for (index, plan) in calls {
            let request = self
                .requests
                .borrow(py)
                .get(py, plan.request_key.request_id.0)?;
            let count = request.borrow(py).request.admission().input_images;
            if count == 0 {
                return Err(invalid(py, "request admission declares no input images"));
            }
            let call = batch.call(py, index)?;
            let device = model
                .borrow()
                .call_devices(py, &*call.extract::<PyRef<crate::calls::Call>>()?)?
                .into_bound(py)
                .get_item(1)?;
            let task = self.host_tasks.get().reserve(py)?;
            batch
                .inputs
                .borrow_mut(py)
                .add_image(plan.call_id, task.clone_ref(py));

            let options = PyDict::new(py);
            options.set_item("input_images", count)?;
            options.set_item("pin", device.getattr("type")?.eq("cuda")?)?;
            let action = partial.call(
                (
                    &prepare,
                    &processor,
                    call.getattr("kind")?,
                    call.getattr("input_image")?,
                ),
                Some(&options),
            )?;
            HostTask::configure(
                task.bind(py),
                action.unbind(),
                Vec::new(),
                None,
                None,
                None,
                "uniserve.host.prepare_image",
            )?;
            task.borrow(py).submit_if_ready(py)?;
        }
        Ok(())
    }

    /// Borrow a resident image or copy the completed host preparation. The
    /// tensor store orders producer and consumer streams without a host wait.
    pub(super) fn prepare_features(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        index: usize,
    ) -> PyResult<Py<PyAny>> {
        let plan = &batch.plan.calls[index];
        if plan.encoder_output.is_none() {
            return Err(invalid(py, "image encoder requires a feature output"));
        }
        let call = batch.call(py, index)?;
        let model = self.model_runner.bind(py);
        let devices = model
            .borrow()
            .call_devices(py, &*call.extract::<PyRef<crate::calls::Call>>()?)?
            .into_bound(py);
        let target = devices.get_item(1)?;
        let numerical = py.import("uniserve_worker.model_executor.image_inputs")?;

        if plan.input_image.is_some() {
            let task = batch
                .inputs
                .borrow(py)
                .image(py, plan.call_id)
                .ok_or_else(|| invalid(py, "inline image has no host preparation"))?;
            let prepared = task.borrow(py).result(py, None)?;
            return numerical
                .call_method1("copy_image", (prepared, target))
                .map(Bound::unbind);
        }

        let read = batch.numerical.borrow(py).consume_tensor(
            py,
            plan.request_key.request_id.0,
            self.tensors.get(),
            devices.get_item(0)?,
        )?;
        let read = read.borrow(py);
        let metadata = image_metadata(py, read.metadata.as_ref())?;
        let range = metadata.getattr("value_range")?;
        if range.is_none() {
            return Err(invalid(py, "resident image has no numerical range"));
        }
        let options = PyDict::new(py);
        options.set_item("device", target)?;
        options.set_item("signed_unit", range.eq((-1.0, 1.0))?)?;
        numerical
            .call_method(
                "prepare_tensor_image",
                (
                    model.borrow().image_processor(py)?.into_bound(py),
                    call.getattr("kind")?,
                    &read.tensor,
                ),
                Some(&options),
            )
            .map(Bound::unbind)
    }

    pub(super) fn write_features(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        index: usize,
        prepared: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let reference = batch.plan.calls[index]
            .encoder_output
            .as_ref()
            .ok_or_else(|| invalid(py, "image encoder requires a feature output"))?;
        let output = batch.pending(py, index);
        let write = output
            .borrow(py)
            .write_buffer(py, reference.buffer_id(), true)?;
        let metadata = py
            .import("uniserve_worker.storage.tensor_store")?
            .getattr("FeatureMetadata")?
            .call1((prepared.getattr("height")?, prepared.getattr("width")?))?;
        self.tensors.get().write(
            py,
            write.into_bound(py),
            value.call_method0("detach")?,
            None,
            Some(metadata),
        )?;
        Ok(())
    }

    /// Gather only the final committed latent bank for numerical decoding.
    pub(super) fn image_latent(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        index: usize,
    ) -> PyResult<Py<PyAny>> {
        let call = &batch.plan.calls[index];
        let input = call
            .latent_input
            .as_ref()
            .ok_or_else(|| invalid(py, "image decoding requires a latent input"))?;
        let params = batch.latent_params(py, index)?;
        let pending = batch.pending(py, index);
        let pending = pending.borrow(py);
        let request = pending.request.borrow(py);
        let image = request
            .request
            .admission()
            .image
            .as_ref()
            .ok_or_else(|| invalid(py, "image decoding has no admitted image parameters"))?;
        if params.start_step != u32::from(image.steps) {
            return Err(invalid(
                py,
                "image decoding requires a completed trajectory",
            ));
        }
        let buffer = pending
            .latent_buffer
            .as_ref()
            .ok_or_else(|| invalid(py, "image decoding has no latent buffer"))?;
        let pool = self
            .latents
            .as_ref()
            .ok_or_else(|| invalid(py, "image decoding requires a latent pool"))?;
        pool.borrow_mut(py).gather_current(
            py,
            request.request.slot() as i64,
            buffer.bind(py),
            params.start_step as i64,
            input.generation as i64,
            params.latent_units as i64,
            params.height as i64,
            params.width as i64,
        )
    }

    pub(super) fn export_image(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        index: usize,
        value: &Bound<'_, PyAny>,
        range: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let call = &batch.plan.calls[index];
        let params = batch.latent_params(py, index)?;
        let value = value.call_method0("detach")?;
        let output = batch.pending(py, index);
        if let Some(reference) = &call.image_output {
            let write = output
                .borrow(py)
                .write_buffer(py, reference.buffer_id(), false)?;
            let metadata = py
                .import("uniserve_worker.storage.tensor_store")?
                .getattr("ImageMetadata")?
                .call1((params.height, params.width, range))?;
            let options = PyDict::new(py);
            options.set_item("dtype", write.get().tensor(py).bind(py).getattr("dtype")?)?;
            let stored = value.call_method("to", (), Some(&options))?;
            self.tensors
                .get()
                .write(py, write.bind(py).clone(), stored, None, Some(metadata))?;
            let mut output = output.borrow_mut(py);
            if output.producer_write.is_none() {
                output.producer_write = Some(write.into_any());
            }
        }

        self.encode_image(py, batch, index, &value, range)?;
        batch
            .numerical
            .borrow(py)
            .complete_latent(py, call.request_key.request_id.0)
    }

    pub(super) fn finalize_image(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        index: usize,
    ) -> PyResult<()> {
        let call = batch.call(py, index)?;
        let device = self
            .model_runner
            .borrow(py)
            .call_devices(py, &*call.extract::<PyRef<crate::calls::Call>>()?)?
            .into_bound(py)
            .get_item(0)?;
        let read = batch.numerical.borrow(py).consume_tensor(
            py,
            batch.plan.calls[index].request_key.request_id.0,
            self.tensors.get(),
            device,
        )?;
        let read = read.borrow(py);
        let metadata = image_metadata(py, read.metadata.as_ref())?;
        let mut range = metadata.getattr("value_range")?;
        if range.is_none() {
            range = (-1.0, 1.0).into_pyobject(py)?.into_any();
        }
        self.encode_image(py, batch, index, read.tensor.bind(py), &range)
    }

    fn encode_image(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        index: usize,
        value: &Bound<'_, PyAny>,
        range: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let maximum = batch.plan.calls[index].bounds.max_completion_bytes as usize;
        let options = PyDict::new(py);
        options.set_item("value_range", range)?;
        let quantized = py.import("uniserve_worker.media.codec")?.call_method(
            "quantize_image_hwc",
            (value,),
            Some(&options),
        )?;
        if maximum == 0 || quantized.call_method0("numel")?.extract::<usize>()? > maximum {
            return Err(invalid(
                py,
                "image preparation exceeds its completion byte bound",
            ));
        }
        let output = batch.pending(py, index);
        let task = {
            let output = output.borrow(py);
            if output.host_tasks.len() != 1 {
                return Err(invalid(
                    py,
                    "image encoding requires one reserved host task",
                ));
            }
            output.host_tasks[0].clone_ref(py)
        };
        let buffer = batch.numerical.borrow(py).output_buffer(py)?;
        let pixels = buffer.get().capture_bytes(py, &quantized)?;

        // The CPU reader survives cancellation until the device copy completes.
        // Ownership transfers to the task only after successful configuration.
        buffer.get().retain_reader(py)?;
        let result = task.borrow(py).configure_media(
            py,
            MediaTask::Image {
                pixels,
                buffer: buffer.clone_ref(py),
                maximum,
            },
            "uniserve.image.encode".to_owned(),
        );
        if result.is_err() {
            buffer.get().release_reader(py)?;
        }
        result
    }
}

fn image_metadata<'py>(
    py: Python<'py>,
    metadata: Option<&Py<PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let metadata = metadata
        .ok_or_else(|| invalid(py, "image input has no spatial metadata"))?
        .bind(py);
    let kind = py
        .import("uniserve_worker.storage.tensor_store")?
        .getattr("ImageMetadata")?;
    if !metadata.is_instance(&kind)?
        || metadata.getattr("height")?.extract::<i64>()? < 1
        || metadata.getattr("width")?.extract::<i64>()? < 1
    {
        return Err(invalid(py, "image input has incomplete spatial metadata"));
    }
    Ok(metadata.clone())
}
