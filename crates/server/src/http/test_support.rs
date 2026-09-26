//! Application state over the in-process engine simulator for HTTP route
//! tests.
//!
//! Requests travel through the public router (`routes::build_router`) into a
//! real `ServingRuntime`, whose engine drives `SimEngine` through
//! `SimExecutor`. The default simulator provides no video media components,
//! so the engine refuses every video submission as an invalid request.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract::Request;
use axum::http::{HeaderMap, StatusCode};
use tower::ServiceExt as _;
use uniserve_core::RuntimeFamily;
use uniserve_engine::{EngineConfig, SimEngine, SimExecutor};

use crate::AppState;
use crate::engine_client::EngineClient;
use crate::profile::{ModelConfig, ModelParameters, SamplingDefaults};
use crate::serving::chat::{ChatTemplateContentFormatOption, HfChatRenderer};
use crate::serving::media::{ImageFetchPolicy, ImageFetcher};
use crate::serving::test_support::configured_tokenizer;
use crate::serving::{InputProcessor, ServedSamplingControl, ServingRuntime, WorkerCapabilities};

/// Served model name the simulated state answers to.
pub(crate) const SERVED_MODEL: &str = "sim-model";

/// Builds application state for a model with `parameters` over a default
/// `SimEngine`.
///
/// The tokenizer is `serving::test_support::configured_tokenizer`, whose
/// `<|im_end|>` (ID 2) is the end-of-sequence token, and the chat template
/// renders only the first message's content. A MiniMax H3 model runs four
/// denoising steps on a diffusion runtime, as a video deployment does.
pub(crate) fn sim_state(parameters: ModelParameters) -> AppState {
    let mut config = EngineConfig::sim(SERVED_MODEL);
    if matches!(parameters, ModelParameters::MiniMaxH3 { .. }) {
        config.runtime_family = RuntimeFamily::Diffusion;
    }
    let client = Arc::new(
        EngineClient::connect_with_executor(config, Box::new(SimExecutor::new(SimEngine::new())))
            .unwrap(),
    );
    let renderer = HfChatRenderer::new(
        Some("{{ messages[0].content }}".to_string()),
        HashMap::new(),
        ChatTemplateContentFormatOption::String,
    )
    .unwrap();
    let processor = InputProcessor::new(
        ModelConfig {
            served_name: SERVED_MODEL.to_string(),
            parameters,
            sampling_defaults: SamplingDefaults::default(),
            max_model_tokens: Some(4096),
            primary_eos_token_id: Some(2),
            eos_token_ids: BTreeSet::from([2]),
        },
        configured_tokenizer(),
        Some(renderer),
        WorkerCapabilities {
            limits: client.generation_limits(),
            sampling_controls: ServedSamplingControl::ALL.to_vec(),
            max_model_tokens: 4096,
            denoise_steps: 4,
        },
        false,
    )
    .unwrap();

    let images = ImageFetcher::new(ImageFetchPolicy::default()).unwrap();
    AppState::new(ServingRuntime::new(processor, client, images, false))
}

/// Sends `request` through `router` and returns the status, the headers, and
/// the body parsed as JSON (`Null` for an empty body).
pub(crate) async fn send(
    router: &Router,
    request: Request<Body>,
) -> (StatusCode, HeaderMap, serde_json::Value) {
    let response = router.clone().oneshot(request).await.unwrap();
    let (parts, body) = response.into_parts();
    let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
    let json = if bytes.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap()
    };
    (parts.status, parts.headers, json)
}

/// Builds a JSON `POST` request to `path`.
pub(crate) fn post_json(path: &str, body: &serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}
