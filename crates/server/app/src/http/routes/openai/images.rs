use std::sync::Arc;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use uniserve_openai_types::ImageGenerationRequest;
use uniserve_protocol_adapters::openai::images::{
    collect_image_generation, prepare_image_generation_request,
};
use uniserve_protocol_adapters::openai::serve_error_to_api;

use crate::AppState;
use crate::http::error::ApiError;
use crate::http::routes::openai::utils::validated_json::ValidatedJson;
use crate::http::utils::{resolve_request_context, unix_timestamp};

pub(crate) async fn images_generations(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<ImageGenerationRequest>,
) -> Response {
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let prepared =
        match prepare_image_generation_request(body, state.served_model_names(), request_context) {
            Ok(prepared) => prepared,
            Err(error) => return ApiError::from(error).into_response(),
        };
    let serve_stream = match state.runtime().generate(prepared.input).await {
        Ok(stream) => stream,
        Err(error) => return ApiError::from(serve_error_to_api(error)).into_response(),
    };
    match collect_image_generation(serve_stream, unix_timestamp()).await {
        Ok(response) => axum::Json(response).into_response(),
        Err(error) => ApiError::from(error).into_response(),
    }
}
