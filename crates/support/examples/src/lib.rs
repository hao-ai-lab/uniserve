//! Shared example builders for documented UniServe configurations.

#![deny(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
use std::time::Duration;

use uniserve_server_app::{
    Config, EngineBackendKind, EngineSettings, HttpListenerMode, ParserSelection, RendererSelection,
};
use uniserve_serving::{GenerationPolicy, OutputDetail, ServeRequest};

/// Build a local GPU-free HTTP serving config suitable for examples and smoke
/// tests.
pub fn sim_http_config(model: impl Into<String>) -> Config {
    Config {
        model: model.into(),
        served_model_name: Vec::new(),
        listener_mode: HttpListenerMode::BindTcp {
            host: "127.0.0.1".to_string(),
            port: 8000,
        },
        engine: EngineSettings {
            backend: EngineBackendKind::Sim,
            ..EngineSettings::default()
        },
        tool_call_parser: ParserSelection::None,
        uniserve_reasoning_parser: ParserSelection::None,
        renderer: RendererSelection::Hf,
        language_model_only: true,
        chat_template: None,
        default_chat_template_kwargs: None,
        chat_template_content_format: Default::default(),
        enable_log_requests: false,
        enable_request_id_headers: true,
        disable_log_stats: true,
        grpc_port: None,
        shutdown_timeout: Duration::from_secs(0),
    }
}

/// Build a semantic text-generation request for examples that exercise the
/// runtime surface directly.
pub fn semantic_text_request(
    request_id: impl Into<String>,
    prompt: impl Into<String>,
) -> ServeRequest {
    let mut request = ServeRequest::text(request_id, prompt);
    request.generation = GenerationPolicy {
        max_tokens: Some(32),
        temperature: Some(0.0),
        output_detail: OutputDetail::Tokens,
        ..GenerationPolicy::default()
    };
    request
}
