use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::StreamExt as _;
use serde_json::json;
use uniserve_protocol_adapters::native::events::event_json;
use uniserve_protocol_adapters::native::{
    NativeGenerateBody, NativeRequestResolution, into_serve_request,
};
use uniserve_protocol_adapters::openai::serve_error_to_api;
use uniserve_serving::{AdapterSelection, RequestMetadata};

use crate::AppState;
use crate::http::error::ApiError;
use crate::http::routes::openai::utils::validated_json::ValidatedJson;
use crate::http::utils::resolve_request_context;

/// Execute one typed UniServe-native request and stream protocol-neutral events.
pub(crate) async fn native_generate(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(mut body): ValidatedJson<NativeGenerateBody>,
) -> Response {
    let request_context = resolve_request_context(&headers, None);
    body.data_parallel_rank = body
        .data_parallel_rank
        .or(request_context.data_parallel_rank);
    for (key, value) in request_context.trace_context {
        body.trace_context.entry(key).or_insert(value);
    }

    let requested_adapter = body.adapter.clone();
    let lora_resolution = state
        .resolve_model_with_loras(requested_adapter.as_deref())
        .await;
    if let Some(requested) = requested_adapter.as_ref()
        && !matches!(
            &lora_resolution.adapter,
            AdapterSelection::Adapter { name, .. } if name == requested
        )
    {
        return ApiError::invalid_request(
            format!("requested adapter `{requested}` is not loaded"),
            Some("adapter"),
        )
        .into_response();
    }
    let profile_id = state
        .runtime()
        .profile()
        .generation_dialect
        .as_ref()
        .map_or_else(
            || state.runtime().profile().profile_id().to_string(),
            |dialect| dialect.id.clone(),
        );
    let request = match into_serve_request(
        request_context.request_id,
        body,
        NativeRequestResolution {
            profile_id,
            requested_adapter,
            adapter: lora_resolution.adapter,
        },
        RequestMetadata {
            protocol_adapter: Some("native".to_string()),
            route: Some("inference.native.generate".to_string()),
            ..RequestMetadata::default()
        },
    ) {
        Ok(request) => request,
        Err(error) => {
            return ApiError::invalid_request(error.to_string(), None).into_response();
        }
    };

    let stream = match state.runtime().serve(request).await {
        Ok(stream) => stream,
        Err(error) => return ApiError::from(serve_error_to_api(error)).into_response(),
    };
    let events = stream.map(|result| {
        let payload = match result {
            Ok(event) => event_json(&event),
            Err(error) => json!({"type":"error","message":error.to_string()}),
        };
        Ok::<_, Infallible>(Event::default().data(payload.to_string()))
    });
    Sse::new(events)
        .keep_alive(KeepAlive::default())
        .into_response()
}
