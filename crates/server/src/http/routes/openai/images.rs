//! HTTP handler for OpenAI-compatible image generation.

use std::sync::Arc;

use crate::openai::ImageGenerationRequest;
use crate::openai::images::collect_image_generation;
use axum::Extension;
use axum::extract::State;
use axum::response::{IntoResponse, Response};

use crate::AppState;
use crate::http::middleware::RequestId;
use crate::http::routes::openai::utils::validated_json::ValidatedJson;
use crate::http::utils::unix_timestamp;

/// Validates, submits, and collects one image-generation request.
pub(crate) async fn images_generations(
    State(state): State<Arc<AppState>>,
    Extension(RequestId(base_id)): Extension<RequestId>,
    ValidatedJson(body): ValidatedJson<ImageGenerationRequest>,
) -> Response {
    let request_id = format!("img-{base_id}");
    let serve_stream = match state
        .runtime()
        .generate_image(request_id.into(), body)
        .await
    {
        Ok(stream) => stream,
        Err(error) => return error.into_response(),
    };
    match collect_image_generation(serve_stream, unix_timestamp()).await {
        Ok(response) => axum::Json(response).into_response(),
        Err(error) => error.into_response(),
    }
}
