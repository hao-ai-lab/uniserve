//! Convert immutable values at the Python numerical backend boundary.

use pyo3::prelude::*;
use serde::de::DeserializeOwned;
use uniserve_core::{CallId, ConditionMedia, ConditionVision, VideoCondition, VisionGrid};
use uniserve_worker_ipc::{BufferId, NewRequest, RequestKey, VideoAdmission};

/// Decode immutable admission parameters once, before the pool mutates state.
pub(super) fn new_request(value: &Bound<'_, PyAny>) -> PyResult<NewRequest> {
    let video = value.getattr("video")?;
    Ok(NewRequest {
        request_key: request_key(&value.getattr("request_key")?)?,
        request_pool_idx: value.getattr("request_pool_idx")?.extract()?,
        ar: parameters(&value.getattr("generation")?)?,
        image: parameters(&value.getattr("image")?)?,
        diffusion: parameters(&value.getattr("diffusion")?)?,
        video: if video.is_none() {
            None
        } else {
            Some(video_admission(&video)?)
        },
        prompt_token_ids: value.getattr("prompt_token_ids")?.extract()?,
        input_images: value.getattr("input_images")?.extract()?,
    })
}

fn parameters<T: DeserializeOwned>(value: &Bound<'_, PyAny>) -> PyResult<Option<T>> {
    if value.is_none() {
        Ok(None)
    } else {
        pythonize::depythonize(&value.call_method0("to_mapping")?)
            .map(Some)
            .map_err(Into::into)
    }
}

fn video_admission(value: &Bound<'_, PyAny>) -> PyResult<VideoAdmission> {
    let mapping = value.call_method0("to_mapping")?;
    let mut conditions = Vec::new();

    for condition in mapping.get_item("conditions")?.try_iter()? {
        let condition = condition?;
        let image = condition.get_item("image")?;
        let video = condition.get_item("video")?;
        let audio = condition.get_item("audio")?;
        // Python exposes the tracks as separate optional fields; the native
        // admission groups a video's picture and soundtrack in one variant.
        let media = if !image.is_none() {
            ConditionMedia::Image(pythonize::depythonize(&image)?)
        } else if !video.is_none() {
            ConditionMedia::Video {
                clip: pythonize::depythonize(&video)?,
                soundtrack: pythonize::depythonize(&audio)?,
            }
        } else {
            ConditionMedia::Audio(pythonize::depythonize(&audio)?)
        };
        let vision = condition.get_item("vision")?;
        let vision = if vision.is_none() {
            None
        } else {
            let (t, h, w) = pythonize::depythonize(&vision.get_item("grid")?)?;
            Some(ConditionVision {
                grid: VisionGrid { t, h, w },
                tokens: vision.get_item("tokens")?.extract()?,
                frame_indices: vision.get_item("frame_indices")?.extract()?,
            })
        };
        conditions.push(VideoCondition {
            role: pythonize::depythonize(&condition.get_item("role")?)?,
            source: pythonize::depythonize(&condition.get_item("source")?)?,
            media,
            vision,
            latent_units: condition.get_item("latent_units")?.extract()?,
            audio_rows: condition.get_item("audio_rows")?.extract()?,
        });
    }

    Ok(VideoAdmission {
        task: pythonize::depythonize(&mapping.get_item("task")?)?,
        text_tags: mapping.get_item("text_tags")?.extract()?,
        conditions,
    })
}

pub(crate) fn request_key(value: &Bound<'_, PyAny>) -> PyResult<RequestKey> {
    Ok(value.extract::<PyRef<'_, crate::ids::RequestKey>>()?.inner)
}

pub(crate) fn call_id(value: &Bound<'_, PyAny>) -> PyResult<CallId> {
    Ok(value.extract::<PyRef<'_, crate::ids::CallId>>()?.inner)
}

pub(crate) fn buffer_id(value: &Bound<'_, PyAny>) -> PyResult<BufferId> {
    Ok(value.extract::<PyRef<'_, crate::ids::BufferId>>()?.inner)
}
