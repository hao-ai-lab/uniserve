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
use crate::serving::test_support::configured_tokenizer;
use crate::serving::video::VideoService;
use crate::serving::video::plan::VisionConfig;
use crate::serving::{InputProcessor, ServedSamplingControl, ServingRuntime, WorkerCapabilities};

/// Served model name the simulated state answers to.
pub(crate) const SERVED_MODEL: &str = "sim-model";

/// Builds application state for a model with `parameters` over a default
/// `SimEngine`.
///
/// The tokenizer is `serving::test_support::configured_tokenizer`, whose
/// `<|im_end|>` (ID 2) is the end-of-sequence token, and the chat template
/// renders only the first message's content. A MiniMax H3 model runs on a
/// diffusion runtime with the video service of a four-step text-to-video
/// denoiser (`sim_video_service`), as a video deployment does.
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
    let video = match &parameters {
        ModelParameters::MiniMaxH3 { max_video_seconds } => {
            Some(sim_video_service(*max_video_seconds))
        }
        _ => None,
    };
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
        },
        video,
        false,
    )
    .unwrap();

    AppState::new(ServingRuntime::new(processor, client, false))
}

/// The video service of a four-step text-to-video denoiser that generates
/// every canvas of the canvas rule, with the released checkpoint's vision
/// processor geometry and the default media policy.
pub(crate) fn sim_video_service(max_video_seconds: f64) -> VideoService {
    VideoService::new(
        uniserve_engine::VideoDenoiserInfo {
            tasks: vec!["t2va".to_owned()],
            schedule_points: 5,
            video_shift: 12.0,
            audio_shift: 3.0,
            canvases: Vec::new(),
            max_sequence_rows: None,
        },
        VisionConfig {
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            image_min_pixels: 65_536,
            image_max_pixels: 16_777_216,
            video_min_pixels: 4_096,
            video_max_pixels: 25_165_824,
        },
        max_video_seconds,
        &crate::VideoMediaSettings::default(),
        configured_tokenizer(),
    )
    .unwrap()
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
