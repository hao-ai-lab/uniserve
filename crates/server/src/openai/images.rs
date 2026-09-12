//! Collection of image-generation output into OpenAI responses.

use crate::serving::RequestOutput;
use futures::{Stream, StreamExt as _};

use crate::openai::error::{ApiError, serve_error_to_api};
use crate::openai::types::{GeneratedImageData, ImageGenerationResponse};

/// Collects image events into one OpenAI-compatible response.
pub async fn collect_image_generation(
    stream: impl Stream<Item = crate::serving::Result<RequestOutput>> + Send,
    created: u64,
) -> Result<ImageGenerationResponse, ApiError> {
    futures::pin_mut!(stream);
    let mut data = Vec::new();
    let mut finished = false;
    while let Some(event) = stream.next().await {
        match event.map_err(serve_error_to_api)? {
            RequestOutput::ImageDone {
                height,
                width,
                bytes,
                sha256,
                pixels_png_b64,
                ..
            } => data.push(GeneratedImageData {
                b64_json: required_image_field(pixels_png_b64, "PNG artifact")?,
                revised_prompt: None,
                height: required_image_field(height, "height")?,
                width: required_image_field(width, "width")?,
                bytes: required_image_field(bytes, "byte count")?,
                sha256: required_image_field(sha256, "SHA-256")?,
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
            RequestOutput::Rejected { message, .. } => {
                return Err(ApiError::invalid_request(message, None));
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

/// Returns a required image field or a validation error.
fn required_image_field<T>(value: Option<T>, field: &str) -> Result<T, ApiError> {
    value.ok_or_else(|| {
        ApiError::server_error(format!(
            "image generation completed without required {field} metadata"
        ))
    })
}
