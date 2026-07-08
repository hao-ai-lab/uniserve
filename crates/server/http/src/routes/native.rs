//! Native-generation helpers used by OpenAI-compatible image endpoints.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use futures::StreamExt as _;
use serde_json::{Value, json};

use uniserve_engine_client::GenerationConstraint;
use uniserve_native_api::{NativeGenerateBody, NativeImageBody, NativeRequestBuilder};
use uniserve_server_app::AppState;
use uniserve_serving::{ServeError, ServeEvent};

use crate::error::ApiError;

fn serve_event_is_terminal(event: &ServeEvent) -> bool {
    matches!(
        event,
        ServeEvent::Finished { .. } | ServeEvent::Rejected { .. } | ServeEvent::Failed { .. }
    )
}

fn serve_error_to_api(error: ServeError) -> ApiError {
    ApiError::server_error(serve_error_message(error))
}

fn serve_error_message(error: ServeError) -> String {
    match error {
        ServeError::UnsupportedRuntimeExtension { key, .. } => {
            format!("unsupported runtime extension `{key}`")
        }
        ServeError::UnsupportedOutputCount { requested, .. } => {
            format!("only one output is supported, got {requested}")
        }
        ServeError::Text(error) => format!("text runtime error: {error}"),
        ServeError::Chat(error) => format!("chat runtime error: {error}"),
        ServeError::Engine(message) => format!("engine runtime error: {message}"),
    }
}

/// OpenAI-style pure text-to-image generation.
pub(crate) async fn images_generations(
    State(st): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Response {
    let prompt = body
        .get("prompt")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .to_string();
    let parsed_size = body
        .get("size")
        .and_then(|value| value.as_str())
        .and_then(|size| {
            size.split_once('x')
                .and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?)))
        });
    let image = NativeImageBody {
        width: parsed_size.map(|(w, _)| w),
        height: parsed_size.map(|(_, h)| h),
        steps: body
            .get("steps")
            .and_then(|value| value.as_u64())
            .map(|value| value as u16),
        seed: body.get("seed").and_then(|value| value.as_u64()),
        negative_prompt: body
            .get("negative_prompt")
            .and_then(|value| value.as_str())
            .map(str::to_owned),
        ..Default::default()
    };
    let native = NativeGenerateBody {
        prompt,
        constraint: Some(GenerationConstraint::GenOnly),
        image: Some(image),
        ..Default::default()
    };
    let tokenizer = st.chat().text().tokenizer();
    let request = match NativeRequestBuilder::new(tokenizer, st.native_profile()).build(&native) {
        Ok(request) => request,
        Err(error) => {
            return ApiError::invalid_request(error.message().to_string(), None).into_response();
        }
    };

    let mut serve_stream = match st
        .runtime()
        .serve_native("images-generations".to_string(), request)
        .await
    {
        Ok(stream) => stream,
        Err(error) => {
            return serve_error_to_api(error).into_response();
        }
    };

    let mut data = Vec::new();
    while let Some(ev) = serve_stream.next().await {
        match ev {
            Ok(ServeEvent::ImageDone {
                height,
                width,
                bytes,
                sha256,
                pixels_png_b64,
                ..
            }) => data.push(json!({
                "b64_json": pixels_png_b64.unwrap_or_default(),
                "height": height.unwrap_or_default(),
                "width": width.unwrap_or_default(),
                "bytes": bytes.unwrap_or_default(),
                "sha256": sha256.unwrap_or_default()
            })),
            Ok(event) if serve_event_is_terminal(&event) => break,
            Err(error) => return serve_error_to_api(error).into_response(),
            _ => {}
        }
    }

    Json(json!({"created": 0, "data": data})).into_response()
}
