use std::sync::Arc;

use crate::openai::ChatCompletionRequest;
use crate::openai::chat_completions::{
    chat_completion_chunk_stream, chat_completion_sse_stream, collect_chat_completion,
    lower_chat_request,
};
use crate::openai::serve_error_to_api;
use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use tracing::info;
use tracing_futures::Instrument as _;

use crate::AppState;
use crate::http::routes::openai::utils::validated_json::ValidatedJson;
use crate::http::utils::{resolve_request_context, unix_timestamp};
use crate::openai::ApiError;

/// Validate one chat completion request and run it through the serving runtime.
pub(crate) async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<ChatCompletionRequest>,
) -> Response {
    let stream = body.stream;
    let request_context = resolve_request_context(&headers);
    let (input, response) =
        match lower_chat_request(body, state.served_model_names(), request_context) {
            Ok(lowered) => lowered,
            Err(error) => return ApiError::from(error).into_response(),
        };
    let request_span = tracing::info_span!(
        "chat_completions",
        request_id = %response.request_id,
        engine_request_id = tracing::field::Empty,
    );

    let created = unix_timestamp();
    let log_request = state.enable_log_requests();

    let serve_stream = match state
        .runtime()
        .generate(input)
        .instrument(request_span.clone())
        .await
    {
        Ok(stream) => stream,
        Err(error) => return ApiError::from(serve_error_to_api(error)).into_response(),
    };

    if stream {
        let chunk_stream = chat_completion_chunk_stream(
            serve_stream,
            response.request_id,
            response.response_model,
            created,
            log_request,
            response.include_usage,
            response.requested_logprobs,
            response.include_reasoning,
            response.return_token_ids,
            response.return_tokens_as_token_ids,
        );
        let sse_stream = chat_completion_sse_stream(chunk_stream).instrument(request_span);

        // Emit periodic SSE keep-alive comments so long prefills (no chunks yet)
        // do not get torn down by idle-timeout proxies between client and server.
        Sse::new(sse_stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        let response = match collect_chat_completion(
            serve_stream,
            response.request_id,
            response.response_model,
            created,
            response.requested_logprobs,
            response.include_prompt_logprobs,
            response.include_reasoning,
            response.return_token_ids,
            response.return_tokens_as_token_ids,
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
                "chat completion finished"
            );
        }

        Json(response).into_response()
    }
}
