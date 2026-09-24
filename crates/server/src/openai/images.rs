//! Collection of image-generation output into OpenAI responses.
//!
//! The `/v1/images/generations` route submits an `ImageGenerationRequest`
//! through the serving runtime and hands the resulting `RequestOutput` stream
//! to [`collect_image_generation`], which buffers completed images until the
//! request finishes and returns one non-streaming response.

use crate::serving::RequestOutput;
use futures::{Stream, StreamExt as _};

use crate::openai::error::{ApiError, serve_error_to_api};
use crate::openai::types::{GeneratedImageData, ImageGenerationResponse};

/// Collects image events into one OpenAI-compatible response.
///
/// Consumes `stream` until its `Finished` event. Each `ImageDone` event
/// contributes one entry to the response `data`, in arrival order; `created`
/// is the response timestamp in Unix seconds. Lifecycle, text, reasoning,
/// output-block, tool-call, image-progress, and usage events are ignored.
///
/// # Errors
///
/// Returns a stream error mapped by [`serve_error_to_api`], the category
/// chosen by [`ApiError::rejected`] for an admission rejection, and a server
/// error when the request fails, is cancelled or aborted, finishes with an
/// error status, emits a video event, reports an image without its
/// dimensions, byte count, or PNG payload, or ends without a `Finished` event
/// or without any image.
pub async fn collect_image_generation(
    stream: impl Stream<Item = crate::serving::Result<RequestOutput>> + Send,
    created: u64,
) -> Result<ImageGenerationResponse, ApiError> {
    futures::pin_mut!(stream);

    let mut data = Vec::new();
    let mut finished = false;
    while let Some(event) = stream.next().await {
        match event.map_err(serve_error_to_api)? {
            // `ImageDone` metadata is optional in the serving protocol; an
            // image response cannot be built without all four fields.
            RequestOutput::ImageDone {
                height,
                width,
                bytes,
                pixels_png_b64,
                ..
            } => data.push(GeneratedImageData {
                b64_json: required_image_field(pixels_png_b64, "PNG artifact")?,
                revised_prompt: None,
                height: required_image_field(height, "height")?,
                width: required_image_field(width, "width")?,
                bytes: required_image_field(bytes, "byte count")?,
            }),
            RequestOutput::Finished { reason, .. } => {
                if matches!(reason, crate::serving::FinishStatus::Error) {
                    return Err(ApiError::server_error(
                        "image generation terminated with an execution error".to_string(),
                    ));
                }
                finished = true;
                break;
            }
            RequestOutput::Rejected { kind, message, .. } => {
                return Err(ApiError::rejected(kind, message));
            }
            RequestOutput::Failed { message, .. } => return Err(ApiError::server_error(message)),
            RequestOutput::Cancelled { .. } => {
                return Err(ApiError::server_error(
                    "image generation request was cancelled".to_string(),
                ));
            }
            RequestOutput::Aborted { .. } => {
                return Err(ApiError::server_error(
                    "image generation request was aborted".to_string(),
                ));
            }
            // Artifact and media-progress events belong to video generation.
            RequestOutput::Artifact(_) | RequestOutput::MediaProgress { .. } => {
                return Err(ApiError::server_error(
                    "image request received a video event".to_owned(),
                ));
            }
            RequestOutput::Accepted { .. }
            | RequestOutput::Scheduled { .. }
            | RequestOutput::TextDelta { .. }
            | RequestOutput::InternalTextDelta { .. }
            | RequestOutput::ReasoningDelta { .. }
            | RequestOutput::OutputBlockStart { .. }
            | RequestOutput::OutputBlockEnd { .. }
            | RequestOutput::ToolCallStart { .. }
            | RequestOutput::ToolCallArgumentsDelta { .. }
            | RequestOutput::ToolCallEnd { .. }
            | RequestOutput::ImageBegin { .. }
            | RequestOutput::ImageStep { .. }
            | RequestOutput::ImageCommit { .. }
            | RequestOutput::Usage { .. } => {}
        }
    }

    if !finished || data.is_empty() {
        return Err(ApiError::server_error(
            "image generation completed without a committed image".to_string(),
        ));
    }
    Ok(ImageGenerationResponse { created, data })
}

/// Returns a required `ImageDone` field, or a server error naming the missing
/// metadata.
fn required_image_field<T>(value: Option<T>, field: &str) -> Result<T, ApiError> {
    value.ok_or_else(|| {
        ApiError::server_error(format!(
            "image generation completed without required {field} metadata"
        ))
    })
}
