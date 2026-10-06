//! Select media operations without a second Python batch dispatcher.

use pyo3::prelude::*;
use uniserve_worker_ipc::{CallKind, MediaCall};

use super::{BatchState, PythonBackend};
use crate::worker::error::invalid;
use crate::worker::host::with_context;

impl PythonBackend {
    pub(super) fn execute_media(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        active: &[usize],
    ) -> PyResult<()> {
        let model = self.model_runner.bind(py);
        let images = !{ model.borrow().image_builder.bind(py).clone() }.is_none();
        let videos = !{ model.borrow().video_postprocessor.bind(py).clone() }.is_none();
        for &index in active {
            let scope = super::super::batch::BatchState::scope(batch.numerical.bind(py))?;
            with_context(scope.bind(py), || {
                let kind = batch.plan.calls[index].code;
                if kind == CallKind::Media(MediaCall::LatentPreparation) && images {
                    self.prepare_image_latent(py, batch, index)?;
                    return Ok(());
                }

                if videos
                    && matches!(
                        kind,
                        CallKind::Media(
                            MediaCall::LatentPreparation
                                | MediaCall::Denoising
                                | MediaCall::VideoDecoding
                                | MediaCall::AudioDecoding
                        )
                    )
                {
                    return self.execute_video(py, batch, index);
                }

                match kind {
                    CallKind::Media(MediaCall::MediaReading) => self.read_media(py, batch, index),
                    CallKind::Media(MediaCall::VisionEncoding | MediaCall::LatentEncoding)
                        if videos =>
                    {
                        self.encode_conditions(py, batch, index)
                    }
                    CallKind::Media(MediaCall::TextEncoding) => {
                        self.encode_conditions(py, batch, index)
                    }
                    CallKind::Media(
                        MediaCall::VideoEncoding | MediaCall::AudioEncoding | MediaCall::Muxing,
                    ) => self.encode_media(py, batch, index),
                    _ => Err(invalid(py, format!("unsupported call {}", kind.as_str()))),
                }
            })?;
        }
        Ok(())
    }
}
