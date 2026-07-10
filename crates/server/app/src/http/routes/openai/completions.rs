use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::Sse;
use axum::response::{IntoResponse, Response};
use tracing::info;
use tracing_futures::Instrument as _;
use uniserve_openai_types::CompletionRequest;
use uniserve_protocol_adapters::openai::completions::{
    collect_completion, completion_chunk_stream, completion_sse_stream, prepare_completion_request,
};
use uniserve_protocol_adapters::openai::serve_error_to_api;

use crate::AppState;
use crate::http::error::ApiError;
use crate::http::routes::openai::utils::validated_json::ValidatedJson;
use crate::http::utils::{resolve_request_context, unix_timestamp};

/// Validate one completions request and proxy it into the shared `text`
/// stack.
pub(crate) async fn completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<CompletionRequest>,
) -> Response {
    let stream = body.stream;
    let logprobs = body.logprobs;
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let lora_resolution = state.resolve_model_with_loras(Some(&body.model)).await;

    let prepared = match prepare_completion_request(body, &lora_resolution, request_context) {
        Ok(prepared) => prepared,
        Err(error) => return ApiError::from(error).into_response(),
    };
    let request_span = tracing::info_span!(
        "completions",
        request_id = %prepared.request_id,
        engine_request_id = tracing::field::Empty,
    );

    let created = unix_timestamp();
    let include_prompt_logprobs = prepared.serve_request.generation.prompt_logprobs.is_some();
    let log_request = state.enable_log_requests();

    let serve_request = prepared.serve_request;

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
        let chunk_stream = completion_chunk_stream(
            serve_stream,
            prepared.request_id,
            prepared.response_model,
            created,
            log_request,
            prepared.include_usage,
            prepared.echo,
            logprobs,
            prepared.return_token_ids,
            prepared.return_tokens_as_token_ids,
        );
        let sse_stream = completion_sse_stream(chunk_stream).instrument(request_span);

        Sse::new(sse_stream).into_response()
    } else {
        let response = match collect_completion(
            serve_stream,
            prepared.request_id,
            prepared.response_model,
            created,
            prepared.echo,
            logprobs,
            include_prompt_logprobs,
            prepared.return_token_ids,
            prepared.return_tokens_as_token_ids,
        )
        .instrument(request_span.clone())
        .await
        {
            Ok(response) => response,
            Err(error) => return ApiError::from(error).into_response(),
        };

        if log_request {
            let usage = response.usage.as_ref();
            info!(
                parent: &request_span,
                model = %response.model,
                prompt_tokens = usage.map_or(0, |u| u.prompt_tokens),
                output_tokens = usage.and_then(|u| u.completion_tokens).unwrap_or(0),
                finish_reason = response.choices.first().and_then(|c| c.finish_reason.as_deref()).unwrap_or("unknown"),
                "completion finished"
            );
        }

        Json(response).into_response()
    }
}
