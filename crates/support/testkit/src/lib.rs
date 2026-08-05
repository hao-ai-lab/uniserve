//! Shared test fixtures and harness utilities for UniServe integration tests.
//!
//! Production crates should depend on this package only from dev-dependencies.

#![deny(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
use std::collections::BTreeSet;
use std::io::Cursor;
use std::ops::Deref;

use base64::Engine as _;
use futures::TryStreamExt as _;
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use uniserve_engine_api::GenEvent;

mod stub_executor;
pub use stub_executor::StubExecutor;

pub fn generate_input_fixture(
    request_id: impl Into<uniserve_serving::ServeRequestId>,
    prompt: impl Into<String>,
) -> uniserve_serving::GenerateReqInput {
    uniserve_serving::GenerateReqInput::text(request_id, prompt)
}

pub fn mock_model_profile(profile_id: impl Into<String>) -> uniserve_model_profile::ModelProfile {
    let mut profile = uniserve_model_profile::ModelProfile::text_only(profile_id);
    profile.identity.family_id = "qwen3".to_string();
    profile
}

pub fn mock_engine_gateway(
    model_name: impl Into<String>,
) -> (
    uniserve_engine_gateway::EngineGateway,
    uniserve_engine_gateway::MockEngine,
) {
    let (client, engine) = uniserve_engine_gateway::EngineCoreClient::connect_mock(model_name);
    (uniserve_engine_gateway::EngineGateway::new(client), engine)
}

/// Normalize a scripted engine batch to the canonical generation-event contract.
pub fn canonical_engine_outputs(
    request: &uniserve_engine_gateway::protocol::EngineCoreRequest,
    mut batch: uniserve_engine_gateway::protocol::EngineCoreOutputs,
) -> uniserve_engine_gateway::protocol::EngineCoreOutputs {
    use uniserve_engine_gateway::protocol::generation::{GenerationFinish, GenerationOutput};
    use uniserve_engine_gateway::protocol::{
        EngineCoreEvent, EngineCoreEventType, EngineCoreFinishReason, EngineCoreOutput,
    };

    let mut outputs = Vec::with_capacity(batch.outputs.len().saturating_add(2));
    let has_scheduled = batch.outputs.iter().any(|output| {
        output.events.as_ref().is_some_and(|events| {
            events
                .iter()
                .any(|event| event.r#type == EngineCoreEventType::Scheduled)
        })
    });
    if !has_scheduled {
        outputs.push(EngineCoreOutput {
            request_id: request.request_id.clone(),
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
        });
    }

    let mut completion_tokens = 0_u64;
    let mut finished = false;
    for mut output in batch.outputs.drain(..) {
        if let Some(prompt_logprobs) = output.new_prompt_logprobs_tensors.take() {
            outputs.push(EngineCoreOutput {
                request_id: request.request_id.clone(),
                new_prompt_logprobs_tensors: Some(prompt_logprobs),
                ..Default::default()
            });
        }
        completion_tokens = completion_tokens.saturating_add(output.new_token_ids.len() as u64);
        if let Some(finish_reason) = output.finish_reason {
            finished = true;
            let has_typed_finish = output
                .generation
                .as_ref()
                .and_then(|generation| generation.finish.as_ref())
                .is_some();
            let hidden_eos = finish_reason == EngineCoreFinishReason::Stop
                && output.new_token_ids.is_empty()
                && output.stop_reason.is_none()
                && !has_typed_finish;
            if hidden_eos {
                completion_tokens = completion_tokens.saturating_add(1);
            }
            let reason = match finish_reason {
                EngineCoreFinishReason::Stop if hidden_eos => "eos",
                EngineCoreFinishReason::Stop => "stop",
                EngineCoreFinishReason::Length => "max_tokens",
                EngineCoreFinishReason::Abort | EngineCoreFinishReason::Aborted => "aborted",
                EngineCoreFinishReason::Error => "error",
                EngineCoreFinishReason::Repetition => "repetition",
                EngineCoreFinishReason::Cancelled => "cancelled",
            };
            let message = (finish_reason == EngineCoreFinishReason::Error)
                .then(|| "Internal server error".to_string());
            output
                .generation
                .get_or_insert_with(GenerationOutput::default)
                .finish
                .get_or_insert_with(|| GenerationFinish {
                    reason: reason.to_string(),
                    prompt_tokens: request.generation.prompt_token_ids().len() as u64,
                    completion_tokens,
                    images: 0,
                    message,
                });
        }
        outputs.push(output);
    }

    batch.outputs = outputs;
    if finished {
        batch.finished_requests = Some(BTreeSet::from([request.request_id.clone()]));
    }
    batch
}

pub async fn collect_events(
    stream: uniserve_serving::ServeEventStream,
) -> anyhow::Result<Vec<uniserve_serving::ServeEvent>> {
    stream.try_collect().await.map_err(Into::into)
}

pub fn assert_terminal_success(events: &[uniserve_serving::ServeEvent]) -> anyhow::Result<()> {
    use uniserve_serving::{FinishStatus, ServeEvent};

    let terminals = events
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
        .collect::<Vec<_>>();
    anyhow::ensure!(
        terminals.len() == 1,
        "expected exactly one terminal event, found {}",
        terminals.len()
    );
    match terminals[0] {
        ServeEvent::Finished {
            reason: FinishStatus::Abort | FinishStatus::Error,
            ..
        } => anyhow::bail!("request finished with a non-success status"),
        ServeEvent::Finished { .. } => Ok(()),
        other => anyhow::bail!("request ended without successful completion: {other:?}"),
    }
}

/// A minimal ChatML template used by the fixture chat renderer. It is only rich
/// enough to round-trip system/user/assistant turns; production templates are
/// loaded from model files.
const FIXTURE_CHAT_TEMPLATE: &str = concat!(
    "{% for message in messages %}",
    "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n",
    "{% endfor %}",
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
);

/// Deterministic byte tokenizer for fixtures: it byte-encodes/decodes text and
/// maps a small set of ChatML/vision/reasoning control tokens to fixed ids.
#[derive(Debug)]
struct FixtureByteTokenizer;

impl uniserve_model_profile::tokenizer::Tokenizer for FixtureByteTokenizer {
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
        let bytes = token_ids
            .iter()
            .map(|token_id| *token_id as u8)
            .collect::<Vec<_>>();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
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

/// Build a text-only [`ResolvedModel`](uniserve_serving::ResolvedModel) fixture
/// backed by the deterministic byte tokenizer and a minimal ChatML renderer.
#[allow(clippy::expect_used)]
pub fn resolved_model_fixture(model_name: &str) -> uniserve_serving::ResolvedModel {
    let profile = mock_model_profile(model_name);
    let tokenizer: uniserve_model_profile::tokenizer::DynTokenizer =
        std::sync::Arc::new(FixtureByteTokenizer);
    let renderer = uniserve_serving::chat::HfChatRenderer::new(
        Some(FIXTURE_CHAT_TEMPLATE.to_string()),
        std::collections::HashMap::new(),
        uniserve_serving::chat::ChatTemplateContentFormatOption::Auto,
    )
    .expect("build fixture chat renderer");
    let capabilities = uniserve_core::GenerationRuntimeCapabilities {
        supports_understanding: true,
        ..Default::default()
    };
    uniserve_serving::ResolvedModel::resolve(profile, tokenizer, renderer, capabilities, 4096)
        .expect("resolve fixture model")
}

/// Build a [`ServingRuntimeFixture`] from a resolved model and an engine
/// gateway (typically produced by [`resolved_model_fixture`] and
/// [`mock_engine_gateway`]).
pub fn serving_runtime_fixture(
    model: uniserve_serving::ResolvedModel,
    gateway: uniserve_engine_gateway::EngineGateway,
) -> ServingRuntimeFixture {
    let engine_control = gateway.app_control();
    let runtime = uniserve_serving::ServingRuntime::new(model, gateway);
    ServingRuntimeFixture {
        runtime,
        engine_control,
    }
}

pub struct ServingRuntimeFixture {
    runtime: uniserve_serving::ServingRuntime,
    engine_control: uniserve_engine_gateway::EngineAppControl,
}

impl ServingRuntimeFixture {
    pub fn into_parts(
        self,
    ) -> (
        uniserve_serving::ServingRuntime,
        uniserve_engine_gateway::EngineAppControl,
    ) {
        (self.runtime, self.engine_control)
    }
}

impl Deref for ServingRuntimeFixture {
    type Target = uniserve_serving::ServingRuntime;

    fn deref(&self) -> &Self::Target {
        &self.runtime
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PngInfo {
    pub width: u32,
    pub height: u32,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeImageContract {
    pub image_id: u32,
    pub begin_width: u32,
    pub begin_height: u32,
    pub done_width: u32,
    pub done_height: u32,
    pub bytes: u64,
    pub sha256: String,
}

pub fn decode_b64_png(pixels_png_b64: &str) -> anyhow::Result<Vec<u8>> {
    base64::engine::general_purpose::STANDARD
        .decode(pixels_png_b64.as_bytes())
        .map_err(Into::into)
}

pub fn png_info(bytes: &[u8]) -> anyhow::Result<PngInfo> {
    let decoder = png::Decoder::new(Cursor::new(bytes));
    let reader = decoder.read_info()?;
    let info = reader.info();
    let sha256 = Sha256::digest(bytes);
    Ok(PngInfo {
        width: info.width,
        height: info.height,
        bytes: bytes.len() as u64,
        sha256: format!("{sha256:x}"),
    })
}

pub fn b64_png_info(pixels_png_b64: &str) -> anyhow::Result<PngInfo> {
    let bytes = decode_b64_png(pixels_png_b64)?;
    png_info(&bytes)
}

pub fn assert_png_dimensions(
    pixels_png_b64: &str,
    expected_width: u32,
    expected_height: u32,
) -> anyhow::Result<PngInfo> {
    let info = b64_png_info(pixels_png_b64)?;
    anyhow::ensure!(
        info.width == expected_width && info.height == expected_height,
        "PNG dimensions were {}x{}, expected {}x{}",
        info.width,
        info.height,
        expected_width,
        expected_height
    );
    Ok(info)
}

pub fn synthetic_png_b64(width: u32, height: u32) -> anyhow::Result<String> {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        let row_bytes = width as usize * 3;
        let mut image = vec![0_u8; row_bytes * height as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let idx = y * row_bytes + x * 3;
                image[idx] = ((x * 255) / (width.max(1) as usize)) as u8;
                image[idx + 1] = ((y * 255) / (height.max(1) as usize)) as u8;
                image[idx + 2] = 128;
            }
        }
        writer.write_image_data(&image)?;
    }
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}

pub fn native_image_contract(events: &[GenEvent]) -> anyhow::Result<NativeImageContract> {
    let begin = events
        .iter()
        .enumerate()
        .find_map(|(index, event)| match event {
            GenEvent::ImageBegin {
                image_id,
                height,
                width,
                ..
            } => Some((index, *image_id, *width, *height)),
            _ => None,
        });
    let commit = events
        .iter()
        .enumerate()
        .find_map(|(index, event)| match event {
            GenEvent::ImageCommit { image_id } => Some((index, *image_id)),
            _ => None,
        });
    let done = events
        .iter()
        .enumerate()
        .find_map(|(index, event)| match event {
            GenEvent::ImageDone {
                image_id,
                height,
                width,
                bytes,
                sha256,
                ..
            } => Some((index, *image_id, *width, *height, *bytes, sha256.clone())),
            _ => None,
        });
    let (begin_index, begin_id, begin_width, begin_height) =
        begin.ok_or_else(|| anyhow::anyhow!("missing ImageBegin event"))?;
    let (commit_index, commit_id) =
        commit.ok_or_else(|| anyhow::anyhow!("missing ImageCommit event"))?;
    let (done_index, done_id, done_width, done_height, bytes, sha256) =
        done.ok_or_else(|| anyhow::anyhow!("missing ImageDone event"))?;
    anyhow::ensure!(
        begin_id == commit_id && commit_id == done_id,
        "image lifecycle ids did not match: begin={begin_id}, commit={commit_id}, done={done_id}"
    );
    anyhow::ensure!(
        begin_index < commit_index && commit_index < done_index,
        "image lifecycle order must be begin, commit, done"
    );
    Ok(NativeImageContract {
        image_id: begin_id,
        begin_width,
        begin_height,
        done_width,
        done_height,
        bytes,
        sha256,
    })
}

pub fn image_done_json_metadata(event_json: &Value) -> anyhow::Result<PngInfo> {
    anyhow::ensure!(
        event_json.get("type").and_then(Value::as_str) == Some("image_done"),
        "expected image_done event JSON"
    );
    Ok(PngInfo {
        width: required_u32(event_json, "width")?,
        height: required_u32(event_json, "height")?,
        bytes: event_json
            .get("bytes")
            .and_then(Value::as_u64)
            .ok_or_else(|| anyhow::anyhow!("missing image_done bytes"))?,
        sha256: event_json
            .get("sha256")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("missing image_done sha256"))?
            .to_string(),
    })
}

fn required_u32(value: &Value, key: &str) -> anyhow::Result<u32> {
    let raw = value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow::anyhow!("missing {key}"))?;
    u32::try_from(raw).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniserve_serving::{CandidateId, ServeEvent};

    #[test]
    fn synthetic_png_reports_actual_dimensions_and_hash() {
        let b64 = synthetic_png_b64(19, 11).expect("synthetic png");
        let info = assert_png_dimensions(&b64, 19, 11).expect("png dimensions");
        assert!(info.bytes > 0);
        assert_eq!(info.sha256.len(), 64);
    }

    #[test]
    fn native_contract_pairs_begin_and_done_metadata() {
        let events = vec![
            GenEvent::ImageBegin {
                image_id: 7,
                height: 11,
                width: 19,
                steps: 2,
            },
            GenEvent::ImageCommit { image_id: 7 },
            GenEvent::ImageDone {
                image_id: 7,
                height: 11,
                width: 19,
                bytes: 123,
                sha256: "a".repeat(64),
                pixels_png_b64: String::new(),
                public_commit: None,
            },
        ];
        let contract = native_image_contract(&events).expect("native image contract");
        assert_eq!((contract.done_width, contract.done_height), (19, 11));
    }

    #[tokio::test]
    async fn semantic_and_mock_fixtures_have_stable_runtime_identity() {
        let request = generate_input_fixture("request-1", "hello");
        assert_eq!(request.request_id.as_ref(), "request-1");
        let profile = mock_model_profile("fixture-model");
        assert_eq!(profile.family_id(), "qwen3");
        let (gateway, _engine) = mock_engine_gateway("fixture-model");
        assert_eq!(gateway.snapshot().model_name, "fixture-model");

        let model = resolved_model_fixture("fixture-model");
        assert_eq!(model.served_model_name(), "fixture-model");
        let fixture = serving_runtime_fixture(model, gateway);
        assert_eq!(fixture.served_model_name(), "fixture-model");
        let (_runtime, _control) = fixture.into_parts();
    }

    #[test]
    fn terminal_assertion_rejects_failure_and_duplicate_terminals() {
        let success = ServeEvent::Finished {
            candidate_id: CandidateId::PRIMARY,
            reason: uniserve_serving::FinishStatus::Length,
            finish_detail: None,
        };
        assert_terminal_success(std::slice::from_ref(&success)).expect("successful terminal");
        assert!(assert_terminal_success(&[success.clone(), success]).is_err());
        assert!(
            assert_terminal_success(&[ServeEvent::Failed {
                request_id: "request-1".into(),
                message: "failure".into(),
            }])
            .is_err()
        );
    }
}
