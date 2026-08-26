//! Shared example builders for the configured UniServe generate funnel.

#![deny(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
use std::time::Duration;

use futures::TryStreamExt as _;
use uniserve_server::ModelDescription;
use uniserve_server::{Config, EngineBackendKind, EngineSettings, HttpListenerMode};
use uniserve_serving::{
    GenerateReqInput, ImageGenControls, ImageInput, ModalitySelection, OutputContract,
    SamplingConfig, ServeEvent, ServeRequestId, ServingRuntime,
};

/// Build a local GPU-free HTTP serving config suitable for examples and smoke
/// tests.
pub fn sim_http_config(model: impl Into<String>) -> Config {
    Config {
        model: model.into(),
        model_description: ModelDescription::SenseNova,
        served_model_name: None,
        media_spool: "/tmp/uniserve-media".into(),
        listener_mode: HttpListenerMode::BindTcp {
            host: "127.0.0.1".to_string(),
            port: 8000,
        },
        engine: EngineSettings {
            backend: EngineBackendKind::Sim,
            ..EngineSettings::default()
        },
        chat_template: None,
        default_chat_template_kwargs: None,
        chat_template_content_format: Default::default(),
        enable_log_requests: false,
        enable_request_id_headers: true,
        disable_log_stats: true,
        api_key: None,
        request_timeout: None,
        max_concurrent_requests: None,
        shutdown_timeout: Duration::ZERO,
        reasoning_parsing: true,
    }
}

/// Build a text generation request for examples that exercise the runtime
/// surface directly.
pub fn semantic_text_request(
    request_id: impl Into<ServeRequestId>,
    prompt: impl Into<String>,
) -> GenerateReqInput {
    let mut request = GenerateReqInput::text(request_id, prompt);
    request.sampling = SamplingConfig {
        max_tokens: Some(32),
        temperature: Some(0.0),
        ..SamplingConfig::default()
    };
    request.output = OutputContract::Tokens;
    request
}

/// Build a text-output request with one input image.
pub fn semantic_text_image_request(
    request_id: impl Into<ServeRequestId>,
    prompt: impl Into<String>,
    input_image_b64: impl Into<String>,
) -> GenerateReqInput {
    let mut request = GenerateReqInput::text(request_id, prompt);
    request.images.push(ImageInput {
        b64: input_image_b64.into(),
    });
    request.sampling.max_tokens = Some(128);
    request.sampling.temperature = Some(0.0);
    request
}

/// Build a text-to-image generation request.
pub fn semantic_image_request(
    request_id: impl Into<ServeRequestId>,
    prompt: impl Into<String>,
) -> GenerateReqInput {
    let mut request = GenerateReqInput::text(request_id, prompt);
    request.sampling.max_tokens = Some(128);
    request.modalities = ModalitySelection {
        output_text: false,
        output_image: true,
    };
    request.image_gen = Some(ImageGenControls {
        max_images: Some(1),
        retain_images: Some(true),
        ..ImageGenControls::default()
    });
    request
}

/// Build an interleaved text-and-image generation request.
pub fn semantic_interleaved_request(
    request_id: impl Into<ServeRequestId>,
    prompt: impl Into<String>,
    max_images: u16,
) -> GenerateReqInput {
    let mut request = GenerateReqInput::text(request_id, prompt);
    request.sampling.max_tokens = Some(512);
    request.modalities = ModalitySelection {
        output_text: true,
        output_image: true,
    };
    request.image_gen = Some(ImageGenControls {
        max_images: Some(max_images),
        retain_images: Some(true),
        ..ImageGenControls::default()
    });
    request
}

/// Invoke the serving runtime and collect its protocol-neutral events.
pub async fn run_semantic_request(
    runtime: &ServingRuntime,
    request: GenerateReqInput,
) -> uniserve_serving::Result<Vec<ServeEvent>> {
    runtime.generate(request).await?.try_collect().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multimodal_builders_express_the_requested_io_contract() {
        let text_image = semantic_text_image_request("u", "describe", "aW1hZ2U=");
        assert!(text_image.modalities.output_text);
        assert!(!text_image.modalities.output_image);
        assert_eq!(text_image.images.len(), 1);

        let image = semantic_image_request("g", "paint");
        assert!(!image.modalities.output_text);
        assert!(image.modalities.output_image);

        let interleaved = semantic_interleaved_request("d", "guide", 4);
        assert!(interleaved.modalities.output_text);
        assert!(interleaved.modalities.output_image);
        assert_eq!(
            interleaved
                .image_gen
                .as_ref()
                .and_then(|controls| controls.max_images),
            Some(4)
        );
    }
}
