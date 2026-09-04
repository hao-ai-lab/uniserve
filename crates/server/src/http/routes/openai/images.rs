//! HTTP handler for OpenAI-compatible image generation.

use std::sync::Arc;

use crate::openai::ImageGenerationRequest;
use crate::openai::images::{collect_image_generation, lower_image_generation_request};
use crate::openai::serve_error_to_api;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};

use crate::AppState;
use crate::http::routes::openai::utils::validated_json::ValidatedJson;
use crate::http::utils::{resolve_request_context, unix_timestamp};
use crate::openai::ApiError;

/// Validates, submits, and collects one image-generation request.
pub(crate) async fn images_generations(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<ImageGenerationRequest>,
) -> Response {
    let request_context = resolve_request_context(&headers);
    let input =
        match lower_image_generation_request(body, state.served_model_name(), request_context) {
            Ok(input) => input,
            Err(error) => return ApiError::from(error).into_response(),
        };
    let serve_stream = match state.runtime().generate(input).await {
        Ok(stream) => stream,
        Err(error) => return ApiError::from(serve_error_to_api(error)).into_response(),
    };
    match collect_image_generation(serve_stream, unix_timestamp()).await {
        Ok(response) => axum::Json(response).into_response(),
        Err(error) => ApiError::from(error).into_response(),
    }
}
