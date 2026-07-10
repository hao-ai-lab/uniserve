//! Shared example builders for documented UniServe configurations.

#![deny(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
use std::time::Duration;

use futures::TryStreamExt as _;
use uniserve_server::{
    Config, EngineBackendKind, EngineSettings, HttpListenerMode, ParserSelection, RendererSelection,
};
use uniserve_serving::{
    ContextRole, ContextSegment, GenerationPolicy, ImageGenerationPolicy, ImageInput,
    ModalityPolicy, ModelContext, OutputDetail, PlanInspection, ServeEvent, ServeRequest,
    ServeRequestId, ServingRuntime,
};

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
        tokenizer_mode: Default::default(),
        language_model_only: true,
        chat_template: None,
        default_chat_template_kwargs: None,
        chat_template_content_format: Default::default(),
        enable_log_requests: false,
        enable_request_id_headers: true,
        disable_log_stats: true,
        api_key: None,
        admin_api_key: None,
        request_timeout: None,
        max_concurrent_requests: None,
        server_dev_mode: false,
        enable_lora: false,
        lora_allowed_path_prefixes: Vec::new(),
        grpc_port: None,
        shutdown_timeout: Duration::from_secs(0),
    }
}

/// Build a semantic text-generation request for examples that exercise the
/// runtime surface directly.
pub fn semantic_text_request(
    request_id: impl Into<ServeRequestId>,
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

/// Build an Und-only request with an image in its ordered semantic context.
pub fn semantic_text_image_request(
    request_id: impl Into<ServeRequestId>,
    prompt: impl Into<String>,
    input_image_b64: impl Into<String>,
) -> ServeRequest {
    let mut request = ServeRequest::text(request_id, "");
    request.model_context = ModelContext::Segments(vec![
        ContextSegment::Text {
            role: ContextRole::User,
            text: prompt.into(),
        },
        ContextSegment::Image(ImageInput {
            b64: input_image_b64.into(),
            placement: None,
        }),
    ]);
    request.generation = GenerationPolicy {
        constraint: uniserve_core::GenerationConstraint::UndOnly,
        max_tokens: Some(128),
        temperature: Some(0.0),
        ..GenerationPolicy::default()
    };
    request.modalities = ModalityPolicy {
        input_text: true,
        input_image: true,
        output_text: true,
        output_image: false,
    };
    request
}

/// Build a Gen-only text-to-image request.
pub fn semantic_image_request(
    request_id: impl Into<ServeRequestId>,
    prompt: impl Into<String>,
) -> ServeRequest {
    let mut request = ServeRequest::text(request_id, prompt);
    request.generation = GenerationPolicy {
        constraint: uniserve_core::GenerationConstraint::GenOnly,
        max_tokens: Some(128),
        image: ImageGenerationPolicy {
            max_images: Some(1),
            retain_images: Some(true),
            ..ImageGenerationPolicy::default()
        },
        ..GenerationPolicy::default()
    };
    request.modalities = ModalityPolicy {
        input_text: true,
        input_image: false,
        output_text: false,
        output_image: true,
    };
    request
}

/// Build a default interleaved text-and-image generation request.
pub fn semantic_interleaved_request(
    request_id: impl Into<ServeRequestId>,
    prompt: impl Into<String>,
    max_images: u16,
) -> ServeRequest {
    let mut request = ServeRequest::text(request_id, prompt);
    request.generation = GenerationPolicy {
        constraint: uniserve_core::GenerationConstraint::Default,
        max_tokens: Some(512),
        image: ImageGenerationPolicy {
            max_images: Some(max_images),
            retain_images: Some(true),
            ..ImageGenerationPolicy::default()
        },
        ..GenerationPolicy::default()
    };
    request.modalities = ModalityPolicy {
        input_text: true,
        input_image: false,
        output_text: true,
        output_image: true,
    };
    request
}

/// Compile and inspect a semantic request without submitting it to the engine.
pub fn inspect_semantic_request(
    runtime: &ServingRuntime,
    request: ServeRequest,
) -> uniserve_serving::Result<PlanInspection> {
    runtime.compile(request).map(|plan| plan.inspect().clone())
}

/// Invoke the semantic runtime directly and collect its protocol-neutral events.
pub async fn run_semantic_request(
    runtime: &ServingRuntime,
    request: ServeRequest,
) -> uniserve_serving::Result<Vec<ServeEvent>> {
    runtime.serve(request).await?.try_collect().await
}

/// Convert the public native adapter schema into a semantic runtime request.
pub fn native_adapter_request(
    request_id: impl Into<ServeRequestId>,
    profile_id: impl Into<String>,
    body: uniserve_protocol_adapters::native::NativeGenerateBody,
) -> Result<ServeRequest, uniserve_protocol_adapters::native::NativeAdapterError> {
    uniserve_protocol_adapters::native::into_serve_request(
        request_id,
        body,
        uniserve_protocol_adapters::native::NativeRequestResolution::base(profile_id),
        uniserve_serving::RequestMetadata {
            route: Some("example.native".to_string()),
            protocol_adapter: Some("native".to_string()),
            ..uniserve_serving::RequestMetadata::default()
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multimodal_example_builders_express_constraints_and_ordered_context() {
        let text_image = semantic_text_image_request("u", "describe", "aW1hZ2U=");
        assert_eq!(
            text_image.generation.constraint,
            uniserve_core::GenerationConstraint::UndOnly
        );
        assert!(text_image.modalities.input_image);
        assert!(!text_image.modalities.output_image);
        let ModelContext::Segments(segments) = text_image.model_context else {
            panic!("text-plus-image builder must use ordered segments");
        };
        assert!(matches!(segments[0], ContextSegment::Text { .. }));
        assert!(matches!(segments[1], ContextSegment::Image(_)));

        let image = semantic_image_request("g", "paint");
        assert_eq!(
            image.generation.constraint,
            uniserve_core::GenerationConstraint::GenOnly
        );
        assert!(!image.modalities.output_text);
        assert!(image.modalities.output_image);

        let interleaved = semantic_interleaved_request("d", "guide", 4);
        assert_eq!(
            interleaved.generation.constraint,
            uniserve_core::GenerationConstraint::Default
        );
        assert!(interleaved.modalities.output_text);
        assert!(interleaved.modalities.output_image);
        assert_eq!(interleaved.generation.image.max_images, Some(4));
    }
}
