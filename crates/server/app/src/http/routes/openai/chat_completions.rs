use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use tracing::info;
use tracing_futures::Instrument as _;
use uniserve_engine_gateway::transport::GenerationConstraint;
use uniserve_openai_types::ChatCompletionRequest;
use uniserve_protocol_adapters::openai::chat_completions::{
    chat_completion_chunk_stream, chat_completion_sse_stream, collect_chat_completion,
    native_chat_constraint, prepare_chat_request, prepare_native_chat_request,
};
use uniserve_protocol_adapters::openai::serve_error_to_api;

use crate::AppState;
use crate::http::error::ApiError;
use crate::http::routes::openai::utils::validated_json::ValidatedJson;
use crate::http::utils::{resolve_request_context, unix_timestamp};

/// Validate one chat completion request and run it through the serving runtime.
pub(crate) async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<ChatCompletionRequest>,
) -> Response {
    let stream = body.stream;
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let lora_resolution = state.resolve_model_with_loras(Some(&body.model)).await;

    let supports_und_only = state
        .runtime()
        .profile()
        .generation_dialect
        .as_ref()
        .is_some_and(|profile| profile.supports_constraint(GenerationConstraint::UndOnly));
    let prepared = match native_chat_constraint(supports_und_only, &body) {
        Some(constraint) => {
            prepare_native_chat_request(body, &lora_resolution, request_context, constraint)
        }
        None => prepare_chat_request(body, &lora_resolution, request_context),
    };
    let prepared = match prepared {
        Ok(prepared) => prepared,
        Err(error) => return ApiError::from(error).into_response(),
    };
    let request_span = tracing::info_span!(
        "chat_completions",
        request_id = %prepared.request_id,
        engine_request_id = tracing::field::Empty,
    );

    let created = unix_timestamp();
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
        let chunk_stream = chat_completion_chunk_stream(
            serve_stream,
            prepared.request_id,
            prepared.response_model,
            created,
            log_request,
            prepared.include_usage,
            prepared.requested_logprobs,
            prepared.include_reasoning,
            prepared.echo,
            prepared.return_token_ids,
            prepared.return_tokens_as_token_ids,
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
            prepared.request_id,
            prepared.response_model,
            created,
            prepared.requested_logprobs,
            prepared.include_prompt_logprobs,
            prepared.include_reasoning,
            prepared.echo,
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
                "chat completion finished"
            );
        }

        Json(response).into_response()
    }
}

/// Compile the exact OpenAI chat payload into its sanitized runtime plan
/// without submitting work to the engine.
pub(crate) async fn chat_completion_plan(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<ChatCompletionRequest>,
) -> Response {
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let lora_resolution = state.resolve_model_with_loras(Some(&body.model)).await;
    let supports_und_only = state
        .runtime()
        .profile()
        .generation_dialect
        .as_ref()
        .is_some_and(|profile| profile.supports_constraint(GenerationConstraint::UndOnly));

    let serve_request =
        if let Some(native_constraint) = native_chat_constraint(supports_und_only, &body) {
            match prepare_native_chat_request(
                body,
                &lora_resolution,
                request_context,
                native_constraint,
            ) {
                Ok(prepared) => prepared.serve_request,
                Err(error) => return ApiError::from(error).into_response(),
            }
        } else {
            match prepare_chat_request(body, &lora_resolution, request_context) {
                Ok(prepared) => prepared.serve_request,
                Err(error) => return ApiError::from(error).into_response(),
            }
        };

    match state.runtime().compile_async(serve_request).await {
        Ok(plan) => Json(plan.inspect().clone()).into_response(),
        Err(error) => ApiError::invalid_request(error.to_string(), None).into_response(),
    }
}
