//! Video-generation request validation and serving-input construction.

use crate::serving::{ServeRequestId, VideoGenerationInput};

use crate::openai::error::ApiError;
use crate::openai::types::VideoGenerationRequest;
use crate::openai::utils::{ResolvedRequestContext, check_model_served};

/// Validates a video request and constructs a serving input.
pub fn lower_video_generation_request(
    request: VideoGenerationRequest,
    served_model_name: &str,
    context: ResolvedRequestContext,
) -> Result<VideoGenerationInput, ApiError> {
    check_model_served(&request.model, served_model_name)?;
    if request.prompt.trim().is_empty() {
        return Err(ApiError::invalid_request(
            "prompt must not be empty".to_string(),
            Some("prompt"),
        ));
    }
    Ok(VideoGenerationInput {
        request_id: ServeRequestId::from(format!("vid-{}", context.request_id)),
        prompt: request.prompt,
        seed: request.seed,
        seconds: request.seconds,
    })
}
