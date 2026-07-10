use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::Sse;
use axum::response::{IntoResponse, Response};
use tracing::info;
use tracing_futures::Instrument as _;
use uniserve_serving::{RequestMetadata, ServeRequest};

use crate::AppState;
use crate::http::error::ApiError;
use crate::http::routes::openai::utils::validated_json::ValidatedJson;
use crate::http::utils::resolve_request_context;
use uniserve_protocol_adapters::raw_generate::{
    GenerateRequest, collect_generate, collect_generate_events, finish_status_as_str,
    generate_chunk_stream, generate_sse_stream, prepare_generate_request, serve_error_to_api,
};

/// Validate one token-in/token-out request and run it through the serving runtime.
pub(crate) async fn generate(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<GenerateRequest>,
) -> Response {
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let lora_resolution = state.resolve_model_with_loras(body.model.as_deref()).await;
    let prepared = match prepare_generate_request(body, &lora_resolution, request_context) {
        Ok(prepared) => prepared,
        Err(error) => return ApiError::from(error).into_response(),
    };
    let request_span = tracing::info_span!(
        "generate",
        request_id = %prepared.request_id,
        engine_request_id = tracing::field::Empty,
    );

    let log_request = state.enable_log_requests();
    let include_logprobs = prepared.include_logprobs;
    let include_prompt_logprobs = prepared.include_prompt_logprobs;
    let stream = prepared.stream;

    let mut serve_request = match ServeRequest::from_text_request(
        prepared.text_request,
        RequestMetadata {
            protocol_adapter: Some("native_generate".to_string()),
            route: Some("generate".to_string()),
            ..RequestMetadata::default()
        },
    ) {
        Ok(request) => request,
        Err(error) => return ApiError::from(serve_error_to_api(error)).into_response(),
    };
    serve_request.generation.intermediate = stream;

    let serve_stream = match state
        .runtime()
        .serve(serve_request)
        .instrument(request_span.clone())
        .await
    {
        Ok(stream) => stream,
        Err(error) => return ApiError::from(serve_error_to_api(error)).into_response(),
    };

    if stream {
        let chunk_stream = generate_chunk_stream(
            serve_stream,
            prepared.request_id,
            log_request,
            prepared.include_usage,
            prepared.include_continuous_usage,
            include_logprobs,
        );
        let sse_stream = generate_sse_stream(chunk_stream).instrument(request_span);

        return Sse::new(sse_stream).into_response();
    }

    let collected = match collect_generate_events(serve_stream)
        .instrument(request_span.clone())
        .await
    {
        Ok(collected) => collected,
        Err(error) => return ApiError::from(error).into_response(),
    };

    if log_request {
        info!(
            parent: &request_span,
            prompt_tokens = collected.prompt_token_ids.len(),
            output_tokens = collected.token_ids.len(),
            finish_reason = finish_status_as_str(&collected.finish_reason),
            "generate finished"
        );
    }

    let response = match collect_generate(
        collected,
        prepared.request_id,
        include_logprobs,
        include_prompt_logprobs,
    ) {
        Ok(response) => response,
        Err(error) => return ApiError::from(error).into_response(),
    };

    Json(response).into_response()
}
