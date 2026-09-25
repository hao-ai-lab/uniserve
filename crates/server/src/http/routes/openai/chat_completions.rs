//! HTTP handler for OpenAI-compatible chat completions.

use std::sync::Arc;

use crate::openai::ChatCompletionRequest;
use crate::openai::chat_completions::{
    ChatResponseContext, chat_completion_chunk_stream, chat_completion_sse_stream,
    collect_chat_completion,
};
use axum::extract::State;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use tracing::info;
use tracing_futures::Instrument as _;

use crate::AppState;
use crate::http::middleware::RequestId;
use crate::http::routes::openai::utils::validated_json::ValidatedJson;
use crate::http::utils::unix_timestamp;

/// Validates one chat completion request and runs it through the serving runtime.
///
/// Returns an SSE stream when `stream` is set and one JSON completion
/// otherwise. Errors from submission, and from collection in the non-streaming
/// case, become `ApiError` responses; errors after an SSE stream has started
/// are reported inside the stream.
pub(crate) async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Extension(RequestId(base_id)): Extension<RequestId>,
    ValidatedJson(body): ValidatedJson<ChatCompletionRequest>,
) -> Response {
    // `body` moves into the runtime below, so every option that shapes the
    // response is captured first. Responses report the served model name.
    let stream = body.stream;
    let request_id = format!("chatcmpl-{base_id}");
    let response =
        ChatResponseContext::from_request(&body, request_id.clone(), state.served_model_name());
    let request_span = tracing::info_span!(
        "chat_completions",
        request_id = %response.request_id,
        engine_request_id = tracing::field::Empty,
    );

    // One timestamp for the whole response, shared by every streamed chunk.
    let created = unix_timestamp();
    let log_request = state.enable_log_requests();

    let serve_stream = match state
        .runtime()
        .generate_chat(request_id.into(), body)
        .instrument(request_span.clone())
        .await
    {
        Ok(stream) => stream,
        Err(error) => return error.into_response(),
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
            Err(error) => return error.into_response(),
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
