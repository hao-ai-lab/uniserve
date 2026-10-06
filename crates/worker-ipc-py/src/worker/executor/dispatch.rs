//! Select media operations without a second Python batch dispatcher.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use uniserve_worker_ipc::{CallKind, MediaCall};

use super::{BatchState, PythonBackend};
use crate::worker::error::invalid;
use crate::worker::host::with_context;

impl PythonBackend {
    pub(super) fn prepare_images(&self, py: Python<'_>, batch: &BatchState) -> PyResult<()> {
        let options = PyDict::new(py);
        options.set_item("host_tasks", &self.host_tasks)?;
        options.set_item("request_pool", &self.requests)?;
        options.set_item("model_runner", &self.model_runner)?;
        py.import("uniserve_worker.execution.image")?.call_method(
            "reserve_images",
            (&batch.numerical,),
            Some(&options),
        )?;
        Ok(())
    }

    pub(super) fn execute_media(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        active: &[usize],
    ) -> PyResult<()> {
        let model = self.model_runner.bind(py);
        let images = !model.getattr("image_builder")?.is_none();
        let videos = !model.getattr("video_postprocessor")?.is_none();
        for &index in active {
            let call = batch.call(py, index)?;
            let scope = super::super::batch::BatchState::scope(batch.numerical.bind(py))?;
            with_context(scope.bind(py), || {
                let options = PyDict::new(py);
                options.set_item("model_runner", model)?;
                options.set_item("state", &batch.numerical)?;
                let kind = batch.plan.calls[index].code;
                if kind == CallKind::Media(MediaCall::LatentPreparation) && images {
                    options.set_item("kv_cache", self.worker.bind(py).getattr("kv_cache")?)?;
                    options.set_item("latent_pool", &self.latents)?;
                    options.set_item(
                        "request_tables",
                        self.worker.bind(py).getattr("block_tables")?,
                    )?;
                    py.import("uniserve_worker.execution.diffusion")?
                        .call_method("prepare_latent", (&call,), Some(&options))?;
                    return Ok(());
                }
                options.set_item("tensor_store", &self.tensors)?;
                options.set_item(
                    "export_transports",
                    self.worker.bind(py).getattr("export_transports")?,
                )?;
                let (module, method) = match kind {
                    CallKind::Media(MediaCall::MediaReading) => {
                        ("uniserve_worker.execution.media_reader", "execute")
                    }
                    CallKind::Media(MediaCall::VisionEncoding) if videos => {
                        ("uniserve_worker.execution.conditions", "encode_vision")
                    }
                    CallKind::Media(MediaCall::LatentEncoding) if videos => {
                        ("uniserve_worker.execution.conditions", "encode_latents")
                    }
                    CallKind::Media(MediaCall::TextEncoding) => {
                        ("uniserve_worker.execution.image", "text")
                    }
                    CallKind::Media(
                        MediaCall::VideoEncoding | MediaCall::AudioEncoding | MediaCall::Muxing,
                    ) => {
                        options
                            .set_item("media_mux", self.worker.bind(py).getattr("media_mux")?)?;
                        options
                            .set_item("transports", self.worker.bind(py).getattr("transports")?)?;
                        ("uniserve_worker.execution.host_media", "execute")
                    }
                    _ if videos => {
                        options.set_item("request_pool", &self.requests)?;
                        ("uniserve_worker.execution.media", "execute")
                    }
                    _ => return Err(invalid(py, format!("unsupported call {}", kind.as_str()))),
                };
                py.import(module)?
                    .call_method(method, (&call,), Some(&options))?;
                Ok(())
            })?;
        }
        Ok(())
    }
}
