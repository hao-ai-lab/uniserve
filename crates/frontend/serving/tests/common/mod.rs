//! Shared fixtures and mock-engine harness for the serving funnel oracles.
#![allow(clippy::unwrap_used, clippy::expect_used, dead_code, unreachable_pub)]

use std::collections::HashMap;
use std::sync::Arc;

use base64::Engine as _;
use futures::StreamExt as _;
use uniserve_core::GenerationRuntimeCapabilities;
use uniserve_engine_gateway::transport::EngineCoreClient;
use uniserve_engine_gateway::transport::protocol::generation::GenerationFinish;
use uniserve_engine_gateway::transport::protocol::{
    EngineCoreEvent, EngineCoreEventType, EngineCoreFinishReason, EngineCoreOutput,
    EngineCoreOutputs, EngineCoreRequest, StopReason,
};
use uniserve_engine_gateway::transport::test_utils::spawn_mock_engine_task;
use uniserve_engine_gateway::{EngineGateway, MockEngine};
use uniserve_model_profile::ModelProfile;
use uniserve_model_profile::tokenizer::{DynTokenizer, Tokenizer};
use uniserve_serving::chat::{ChatMessage, ChatTemplateContentFormatOption, HfChatRenderer};
use uniserve_serving::{
    GenerateReqInput, ImageInput, ModalitySelection, ResolvedModel, ServeEvent, ServingRuntime,
};

pub const MAX_MODEL_TOKENS: u32 = 4096;

const CHATML_TEMPLATE: &str = "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

/// Deterministic byte tokenizer with a handful of special tokens.
#[derive(Debug)]
pub struct ByteTokenizer;

impl Tokenizer for ByteTokenizer {
    fn encode(
        &self,
        text: &str,
        _add_special_tokens: bool,
    ) -> uniserve_model_profile::tokenizer::Result<Vec<u32>> {
        Ok(text.bytes().map(u32::from).collect())
    }

    fn decode(
        &self,
        token_ids: &[u32],
        _skip_special_tokens: bool,
    ) -> uniserve_model_profile::tokenizer::Result<String> {
        Ok(
            String::from_utf8_lossy(&token_ids.iter().map(|id| *id as u8).collect::<Vec<_>>())
                .into_owned(),
        )
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match token {
            "<|im_start|>" => Some(1),
            "<|im_end|>" => Some(2),
            "<|vision_start|>" => Some(3),
            "<|vision_end|>" => Some(4),
            "<think>" => Some(5),
            "</think>" => Some(6),
            _ => None,
        }
    }
}

/// Tokenizer that encodes `<img>`/`</img>` atomically, like the real omni
/// tokenizers, and byte-encodes everything else.
#[derive(Debug)]
pub struct SenseNovaTokenizer;

impl Tokenizer for SenseNovaTokenizer {
    fn encode(
        &self,
        text: &str,
        _add_special_tokens: bool,
    ) -> uniserve_model_profile::tokenizer::Result<Vec<u32>> {
        const MARKERS: [(&str, u32); 2] = [("<img>", 151670), ("</img>", 151671)];
        let mut ids = Vec::new();
        let mut rest = text;
        'outer: while !rest.is_empty() {
            for (marker, id) in MARKERS {
                if let Some(stripped) = rest.strip_prefix(marker) {
                    ids.push(id);
                    rest = stripped;
                    continue 'outer;
                }
            }
            let mut chars = rest.chars();
            let ch = chars.next().expect("non-empty");
            let mut buf = [0u8; 4];
            ids.extend(ch.encode_utf8(&mut buf).bytes().map(u32::from));
            rest = chars.as_str();
        }
        Ok(ids)
    }

    fn decode(
        &self,
        token_ids: &[u32],
        _skip_special_tokens: bool,
    ) -> uniserve_model_profile::tokenizer::Result<String> {
        Ok(
            String::from_utf8_lossy(&token_ids.iter().map(|id| *id as u8).collect::<Vec<_>>())
                .into_owned(),
        )
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match token {
            "<img>" => Some(151670),
            "</img>" => Some(151671),
            "<|im_start|>" => Some(151644),
            "<|im_end|>" => Some(151645),
            _ => None,
        }
    }
}

fn renderer() -> HfChatRenderer {
    HfChatRenderer::new(
        Some(CHATML_TEMPLATE.to_string()),
        HashMap::new(),
        ChatTemplateContentFormatOption::Auto,
    )
    .expect("renderer")
}

fn capabilities() -> GenerationRuntimeCapabilities {
    GenerationRuntimeCapabilities {
        supports_understanding: true,
        supports_vision_encode: true,
        supports_latent_encode: true,
        supports_image_generation: true,
        max_latent_units: 65_536,
        latent_downsample: 16,
        max_vae_grid_tokens: 4_096,
        max_vit_grid_tokens: 2_048,
        max_latent_feature_bytes: 1 << 28,
        max_vision_feature_bytes: 1 << 28,
        commit_marker_tokens: 2,
        max_cfg_branches: 3,
        scratch_capacity_tokens: 65_536,
        scratch_block_size: 64,
        encoder_cache_entries: 256,
    }
}

/// Resolve a Qwen3 (text/chat) model with a byte tokenizer.
pub fn resolve_qwen3() -> ResolvedModel {
    let tokenizer: DynTokenizer = Arc::new(ByteTokenizer);
    let profile = ModelProfile::text_only("qwen3-test");
    ResolvedModel::resolve(
        profile,
        tokenizer,
        renderer(),
        capabilities(),
        MAX_MODEL_TOKENS,
    )
    .expect("resolve qwen3")
}

pub fn resolve_unsupported_text() -> uniserve_serving::Result<ResolvedModel> {
    let tokenizer: DynTokenizer = Arc::new(ByteTokenizer);
    ResolvedModel::resolve(
        ModelProfile::text_only("unsupported-text-family"),
        tokenizer,
        renderer(),
        capabilities(),
        MAX_MODEL_TOKENS,
    )
}

fn resolve_omni(dialect_key: &str) -> ResolvedModel {
    let tokenizer: DynTokenizer = Arc::new(SenseNovaTokenizer);
    let dialect = uniserve_model_profile::dialect::resolve_generation_dialect_for_model(
        dialect_key,
        tokenizer.as_ref(),
    )
    .expect("dialect resolution")
    .expect("omni dialect");
    let profile = ModelProfile::text_only("omni-test").with_generation_dialect(dialect);
    ResolvedModel::resolve(
        profile,
        tokenizer,
        renderer(),
        capabilities(),
        MAX_MODEL_TOKENS,
    )
    .expect("resolve omni")
}

pub fn resolve_sensenova() -> ResolvedModel {
    resolve_omni("sensenova-u1")
}

pub fn resolve_bagel() -> ResolvedModel {
    resolve_omni("bagel")
}

// ---- GenerateReqInput builders -------------------------------------------

pub fn text_input(request_id: &str, prompt: &str) -> GenerateReqInput {
    GenerateReqInput::text(request_id, prompt)
}

pub fn chat_input(request_id: &str, messages: Vec<ChatMessage>) -> GenerateReqInput {
    GenerateReqInput::chat(request_id, messages)
}

/// Text -> image generation request.
pub fn t2i_input(request_id: &str, prompt: &str) -> GenerateReqInput {
    let mut request = GenerateReqInput::text(request_id, prompt);
    request.modalities = ModalitySelection {
        output_text: false,
        output_image: true,
    };
    request
}

/// Image -> text (understanding) request carrying one input image.
pub fn i2t_input(request_id: &str, prompt: &str) -> GenerateReqInput {
    let mut request = GenerateReqInput::text(request_id, prompt);
    request.modalities = ModalitySelection {
        output_text: true,
        output_image: false,
    };
    request.images = vec![ImageInput {
        b64: png_b64(64, 64),
        placement: None,
    }];
    request
}

pub fn png_b64(width: u32, height: u32) -> String {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, width, height);
        encoder.set_color(png::ColorType::Grayscale);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().expect("png header");
        writer
            .write_image_data(&vec![0; (width * height) as usize])
            .expect("png pixels");
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

// ---- Mock-engine harness --------------------------------------------------

pub fn build_runtime(model: ResolvedModel) -> (ServingRuntime, MockEngine) {
    let (client, mock) = EngineCoreClient::connect_mock("test-model");
    (ServingRuntime::new(model, EngineGateway::new(client)), mock)
}

pub fn scheduling_output(request_id: &str) -> EngineCoreOutput {
    EngineCoreOutput {
        request_id: request_id.to_string(),
        events: Some(vec![
            EngineCoreEvent {
                r#type: EngineCoreEventType::Queued,
                timestamp: 1.0,
            },
            EngineCoreEvent {
                r#type: EngineCoreEventType::Scheduled,
                timestamp: 1.001,
            },
        ]),
        ..Default::default()
    }
}

pub fn token_output(
    request_id: &str,
    new_token_ids: Vec<u32>,
    finish_reason: Option<EngineCoreFinishReason>,
    stop_reason: Option<StopReason>,
) -> EngineCoreOutput {
    EngineCoreOutput {
        request_id: request_id.to_string(),
        new_token_ids,
        finish_reason,
        stop_reason,
        ..Default::default()
    }
}

pub fn canonicalize_outputs(
    prompt_token_count: usize,
    mut batch: EngineCoreOutputs,
) -> EngineCoreOutputs {
    let mut completion_tokens = 0_u64;
    for output in &mut batch.outputs {
        completion_tokens = completion_tokens.saturating_add(output.new_token_ids.len() as u64);
        let Some(finish_reason) = output.finish_reason else {
            continue;
        };
        if let Some(StopReason::TokenId(stop_token_id)) = output.stop_reason.as_ref()
            && !output.new_token_ids.contains(stop_token_id)
        {
            completion_tokens = completion_tokens.saturating_add(1);
        }
        let reason = match finish_reason {
            EngineCoreFinishReason::Stop => "stop",
            EngineCoreFinishReason::Length => "max_tokens",
            EngineCoreFinishReason::Abort | EngineCoreFinishReason::Aborted => "aborted",
            EngineCoreFinishReason::Error => "error",
            EngineCoreFinishReason::Repetition => "repetition",
            EngineCoreFinishReason::Cancelled => "cancelled",
        };
        output
            .generation
            .get_or_insert_with(Default::default)
            .finish = Some(GenerationFinish {
            reason: reason.to_string(),
            prompt_tokens: prompt_token_count as u64,
            completion_tokens,
            images: 0,
            message: None,
        });
    }
    batch
}

/// Drive `generate` over a scripted mock engine and collect the resulting
/// events. `script` builds the engine outputs from the received request.
pub async fn drive(
    runtime: &ServingRuntime,
    mock: MockEngine,
    request: GenerateReqInput,
    script: impl FnOnce(EngineCoreRequest) -> EngineCoreOutputs + Send + 'static,
) -> Vec<ServeEvent> {
    let (shutdown_tx, engine_task) = spawn_mock_engine_task(mock, move |mut mock| {
        Box::pin(async move {
            let request = mock.recv_request().await;
            let outputs = script(request);
            mock.send_outputs(outputs);
        })
    });
    let stream = runtime.generate(request).await.expect("generate stream");
    let events: Vec<ServeEvent> = stream
        .map(|event| event.expect("serve event"))
        .collect()
        .await;
    let _ = shutdown_tx.send(());
    let _ = engine_task.await;
    events
}

/// Assert exactly one terminal event and that it is a successful `Finished`.
pub fn assert_terminal_success(events: &[ServeEvent]) {
    use uniserve_serving::FinishStatus;
    let terminals: Vec<&ServeEvent> = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                ServeEvent::Finished { .. }
                    | ServeEvent::Rejected { .. }
                    | ServeEvent::Cancelled { .. }
                    | ServeEvent::Aborted { .. }
                    | ServeEvent::Failed { .. }
            )
        })
        .collect();
    assert_eq!(terminals.len(), 1, "expected exactly one terminal event");
    assert!(
        matches!(
            terminals[0],
            ServeEvent::Finished {
                reason: FinishStatus::Stop { .. } | FinishStatus::Length,
                ..
            }
        ),
        "expected a successful terminal, got {:?}",
        terminals[0]
    );
    assert!(
        matches!(events.last(), Some(ServeEvent::Finished { .. })),
        "the terminal event must be last"
    );
}
