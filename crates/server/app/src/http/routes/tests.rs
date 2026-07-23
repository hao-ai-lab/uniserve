// Route tests should use `Service::call` rather than `ServiceExt::oneshot`.
// `oneshot` consumes the router and can drop `AppState` before a streaming
// response body is fully drained, which closes the mock engine connection too
// early and causes flaky `closed unexpectedly` failures.

use std::collections::BTreeSet;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use bytes::Bytes;
use serde_json::json;
use serial_test::serial;
use tower::{Service as _, ServiceExt as _};
use uniserve_core::ContextSegment;
use uniserve_engine_gateway::EngineGateway;
use uniserve_engine_gateway::transport::protocol::generation::{
    GenerationFinish, GenerationOutput, WireImageEvent,
};
use uniserve_engine_gateway::transport::protocol::logprobs::{
    Logprobs, MaybeWireLogprobs, PositionLogprobs, TokenLogprob,
};
use uniserve_engine_gateway::transport::protocol::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest, StopReason,
};
use uniserve_engine_gateway::transport::test_utils::spawn_mock_engine_task;
use uniserve_engine_gateway::transport::{EngineCoreClient, GenerationConstraint, MockEngine};
use uniserve_model_profile::dialect::resolve_generation_dialect_for_model;
use uniserve_observability::METRICS;
use uniserve_serving::chat::{
    ChatBackend, ChatContent, ChatContentPart, ChatMessage, ChatRenderer, ChatRequest,
    ChatTextBackend, DefaultChatOutputProcessor, DynChatOutputProcessor, DynChatRenderer,
    NewChatOutputProcessorOptions, ParserSelection,
};
use uniserve_serving::text::tokenizer::{DynTokenizer, Tokenizer};
use uniserve_serving::text::{Prompt, TextBackend};

use super::{build_router, build_router_with_dev_mode, build_router_with_dev_mode_and_lora};
use crate::AppState;

const TEST_PNG_B64: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WlMZrAAAAAASUVORK5CYII=";

fn request_output(
    request_id: &str,
    new_token_ids: Vec<u32>,
    finish_reason: Option<EngineCoreFinishReason>,
) -> EngineCoreOutput {
    request_output_with_stop_reason(request_id, new_token_ids, finish_reason, None)
}

fn request_output_with_stop_reason(
    request_id: &str,
    new_token_ids: Vec<u32>,
    finish_reason: Option<EngineCoreFinishReason>,
    stop_reason: Option<StopReason>,
) -> EngineCoreOutput {
    EngineCoreOutput {
        request_id: request_id.to_string(),
        new_token_ids,
        new_logprobs: None,
        new_prompt_logprobs_tensors: None,
        pooling_output: None,
        finish_reason,
        stop_reason,
        events: None,
        kv_transfer_params: None,
        trace_headers: None,
        prefill_stats: None,
        routed_experts: None,
        num_nans_in_logits: 0,
        generation: None,
    }
}

fn request_output_with_logprobs(
    request_id: &str,
    new_token_ids: Vec<u32>,
    finish_reason: Option<EngineCoreFinishReason>,
    stop_reason: Option<StopReason>,
    new_logprobs: Option<Logprobs>,
    new_prompt_logprobs_tensors: Option<Logprobs>,
) -> EngineCoreOutput {
    EngineCoreOutput {
        request_id: request_id.to_string(),
        new_token_ids,
        new_logprobs: new_logprobs.map(MaybeWireLogprobs::Direct),
        new_prompt_logprobs_tensors: new_prompt_logprobs_tensors.map(MaybeWireLogprobs::Direct),
        pooling_output: None,
        finish_reason,
        stop_reason,
        events: None,
        kv_transfer_params: None,
        trace_headers: None,
        prefill_stats: None,
        routed_experts: None,
        num_nans_in_logits: 0,
        generation: None,
    }
}

fn request_output_with_logprobs_and_kv(
    request_id: &str,
    new_token_ids: Vec<u32>,
    finish_reason: Option<EngineCoreFinishReason>,
    stop_reason: Option<StopReason>,
    new_logprobs: Option<Logprobs>,
    new_prompt_logprobs_tensors: Option<Logprobs>,
    kv_transfer_params: Option<serde_json::Value>,
) -> EngineCoreOutput {
    EngineCoreOutput {
        request_id: request_id.to_string(),
        new_token_ids,
        new_logprobs: new_logprobs.map(MaybeWireLogprobs::Direct),
        new_prompt_logprobs_tensors: new_prompt_logprobs_tensors.map(MaybeWireLogprobs::Direct),
        pooling_output: None,
        finish_reason,
        stop_reason,
        events: None,
        kv_transfer_params,
        trace_headers: None,
        prefill_stats: None,
        routed_experts: None,
        num_nans_in_logits: 0,
        generation: None,
    }
}

fn native_image_output(request_id: &str, image: WireImageEvent) -> EngineCoreOutput {
    EngineCoreOutput {
        request_id: request_id.to_string(),
        generation: Some(GenerationOutput {
            image: Some(image),
            finish: None,
        }),
        ..Default::default()
    }
}

fn native_finish_output(
    request_id: &str,
    reason: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
    images: u64,
) -> EngineCoreOutput {
    EngineCoreOutput {
        request_id: request_id.to_string(),
        finish_reason: Some(EngineCoreFinishReason::Stop),
        generation: Some(GenerationOutput {
            image: None,
            finish: Some(GenerationFinish {
                reason: reason.to_string(),
                prompt_tokens,
                completion_tokens,
                images,
                message: None,
            }),
        }),
        ..Default::default()
    }
}

fn bytes_to_token_ids(bytes: &[u8]) -> Vec<u32> {
    bytes.iter().map(|byte| u32::from(*byte)).collect()
}

fn default_stream_output_specs() -> Vec<(Vec<u32>, Option<EngineCoreFinishReason>)> {
    vec![
        (vec![b'h' as u32], None),
        (vec![b'i' as u32], None),
        (Vec::new(), Some(EngineCoreFinishReason::Stop)),
    ]
}

fn assert_adapter_a_lora_request(request: &EngineCoreRequest) {
    assert_eq!(request.generation.lora_id, Some(1));
}

fn sse_data_payloads(text: &str) -> Vec<&str> {
    text.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect()
}

/// Decode the non-sentinel SSE payloads of a streamed response into JSON.
fn sse_json_chunks(text: &str) -> Vec<serde_json::Value> {
    sse_data_payloads(text)
        .into_iter()
        .filter(|payload| *payload != "[DONE]")
        .map(|payload| serde_json::from_str(payload).expect("sse chunk json"))
        .collect()
}

/// Concatenate the streamed chat-completion text content across SSE chunks.
fn streamed_chat_content(text: &str) -> String {
    sse_json_chunks(text)
        .iter()
        .filter_map(|chunk| {
            chunk["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_owned)
        })
        .collect()
}

/// Concatenate streamed reasoning content independently of engine chunk boundaries.
fn streamed_reasoning_content(text: &str) -> String {
    sse_json_chunks(text)
        .iter()
        .filter_map(|chunk| {
            chunk["choices"][0]["delta"]["reasoning_content"]
                .as_str()
                .map(str::to_owned)
        })
        .collect()
}

/// Concatenate streamed tool arguments independently of engine chunk boundaries.
fn streamed_tool_arguments(text: &str) -> String {
    sse_json_chunks(text)
        .iter()
        .filter_map(|chunk| {
            chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .map(str::to_owned)
        })
        .collect()
}

/// Concatenate the streamed text-completion text across SSE chunks.
fn streamed_completion_text(text: &str) -> String {
    sse_json_chunks(text)
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["text"].as_str().map(str::to_owned))
        .collect()
}

/// Find the single finish_reason emitted across the streamed SSE chunks.
fn streamed_finish_reason(text: &str) -> Option<String> {
    sse_json_chunks(text).iter().find_map(|chunk| {
        chunk["choices"][0]["finish_reason"]
            .as_str()
            .map(str::to_owned)
    })
}

type TestFuture<'a> = Pin<Box<dyn Future<Output = ()> + Send + 'a>>;

fn boxed_test_future<'a>(future: impl Future<Output = ()> + Send + 'a) -> TestFuture<'a> {
    Box::pin(future)
}

struct MockEngineTask {
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    join_handle: Option<tokio::task::JoinHandle<()>>,
}

impl MockEngineTask {
    fn new(
        (shutdown_tx, join_handle): (
            tokio::sync::oneshot::Sender<()>,
            tokio::task::JoinHandle<()>,
        ),
    ) -> Self {
        Self {
            shutdown_tx: Some(shutdown_tx),
            join_handle: Some(join_handle),
        }
    }

    async fn finish(self) {
        self.await.expect("mock engine task");
    }

    fn abort(&self) {
        if let Some(join_handle) = &self.join_handle {
            join_handle.abort();
        }
    }

    async fn abort_and_join(mut self) {
        if let Some(join_handle) = self.join_handle.take() {
            join_handle.abort();
            let _ = join_handle.await;
        }
    }
}

impl Future for MockEngineTask {
    type Output = Result<(), tokio::task::JoinError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
        match self.join_handle.as_mut() {
            Some(join_handle) => Pin::new(join_handle).poll(cx),
            None => Poll::Ready(Ok(())),
        }
    }
}

impl Drop for MockEngineTask {
    fn drop(&mut self) {
        if let Some(join_handle) = &self.join_handle {
            join_handle.abort();
        }
    }
}

fn engine_outputs_for_request(
    request: &EngineCoreRequest,
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
) -> EngineCoreOutputs {
    uniserve_testkit::canonical_engine_outputs(
        request,
        EngineCoreOutputs {
            outputs: output_specs
                .into_iter()
                .map(|(token_ids, finish_reason)| {
                    request_output(&request.request_id, token_ids, finish_reason)
                })
                .collect(),
            ..Default::default()
        },
    )
}

fn send_canonical_outputs(
    mock: &MockEngine,
    request: &EngineCoreRequest,
    outputs: EngineCoreOutputs,
) {
    mock.send_outputs(uniserve_testkit::canonical_engine_outputs(request, outputs));
}

fn test_gateway(client: EngineCoreClient) -> EngineGateway {
    EngineGateway::new(client)
}

fn test_app_state(
    served_model_names: Vec<String>,
    fixture: uniserve_testkit::ServingRuntimeFixture,
) -> AppState {
    let (runtime, engine_control) = fixture.into_parts();
    AppState::new(served_model_names, runtime, engine_control)
}

fn sample_logprobs_for_token(token_id: u32, alternate_token_id: u32) -> Logprobs {
    Logprobs {
        positions: vec![PositionLogprobs {
            entries: vec![
                TokenLogprob {
                    token_id,
                    logprob: -0.1,
                    rank: 1,
                },
                TokenLogprob {
                    token_id: alternate_token_id,
                    logprob: -0.2,
                    rank: 1,
                },
            ],
        }],
    }
}

fn sample_logprobs_for_tokens(token_ids: &[u32]) -> Logprobs {
    Logprobs {
        positions: token_ids
            .iter()
            .map(|&token_id| PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id,
                        logprob: -0.1,
                        rank: 1,
                    },
                    TokenLogprob {
                        token_id: token_id.saturating_add(1),
                        logprob: -0.2,
                        rank: 2,
                    },
                ],
            })
            .collect(),
    }
}

fn prompt_logprobs_for_hello() -> Logprobs {
    Logprobs {
        positions: vec![
            PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id: b'e' as u32,
                        logprob: -0.3,
                        rank: 1,
                    },
                    TokenLogprob {
                        token_id: b'a' as u32,
                        logprob: -0.5,
                        rank: 1,
                    },
                ],
            },
            PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id: b'l' as u32,
                        logprob: -0.4,
                        rank: 1,
                    },
                    TokenLogprob {
                        token_id: b'r' as u32,
                        logprob: -0.6,
                        rank: 1,
                    },
                ],
            },
            PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id: b'l' as u32,
                        logprob: -0.45,
                        rank: 1,
                    },
                    TokenLogprob {
                        token_id: b'i' as u32,
                        logprob: -0.65,
                        rank: 1,
                    },
                ],
            },
            PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id: b'o' as u32,
                        logprob: -0.5,
                        rank: 1,
                    },
                    TokenLogprob {
                        token_id: b'u' as u32,
                        logprob: -0.7,
                        rank: 1,
                    },
                ],
            },
        ],
    }
}

fn prompt_logprobs_for_tokens(token_ids: &[u32]) -> Logprobs {
    Logprobs {
        positions: token_ids
            .iter()
            .skip(1)
            .map(|&token_id| PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id,
                        logprob: -0.3,
                        rank: 1,
                    },
                    TokenLogprob {
                        token_id: token_id.saturating_add(1),
                        logprob: -0.5,
                        rank: 2,
                    },
                ],
            })
            .collect(),
    }
}

#[derive(Clone)]
struct FakeChatBackend {
    model_id: String,
}

#[derive(Debug)]
struct FakeChatTokenizer;

impl Tokenizer for FakeChatTokenizer {
    fn encode(
        &self,
        text: &str,
        _add_special_tokens: bool,
    ) -> uniserve_serving::text::tokenizer::Result<Vec<u32>> {
        let mut token_ids = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            if let Some(stripped) = rest.strip_prefix("<image>") {
                token_ids.push(999);
                rest = stripped;
                continue;
            }
            if let Some(stripped) = rest.strip_prefix("<img>") {
                token_ids.push(151670);
                rest = stripped;
                continue;
            }
            if let Some(stripped) = rest.strip_prefix("</img>") {
                token_ids.push(151671);
                rest = stripped;
                continue;
            }

            let ch = rest.chars().next().expect("rest is not empty");
            let mut buf = [0; 4];
            token_ids.extend(ch.encode_utf8(&mut buf).bytes().map(u32::from));
            rest = &rest[ch.len_utf8()..];
        }
        Ok(token_ids)
    }

    fn decode(
        &self,
        token_ids: &[u32],
        _skip_special_tokens: bool,
    ) -> uniserve_serving::text::tokenizer::Result<String> {
        Ok(
            String::from_utf8_lossy(&token_ids.iter().map(|id| *id as u8).collect::<Vec<_>>())
                .into_owned(),
        )
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match token {
            "<image>" => Some(999),
            "<|image_pad|>" => Some(151655),
            "<|im_start|>" => Some(151644),
            "<|im_end|>" => Some(151645),
            "<img>" => Some(151670),
            "</img>" => Some(151671),
            "<think>" => Some(0xF001),
            "</think>" => Some(0xF002),
            "<|START_THINKING|>" => Some(0xF003),
            "<|END_THINKING|>" => Some(0xF004),
            "◁think▷" => Some(0xF005),
            "◁/think▷" => Some(0xF006),
            _ => None,
        }
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        match id {
            999 => Some("<image>".to_string()),
            151655 => Some("<|image_pad|>".to_string()),
            151644 => Some("<|im_start|>".to_string()),
            151645 => Some("<|im_end|>".to_string()),
            151670 => Some("<img>".to_string()),
            151671 => Some("</img>".to_string()),
            0xF001 => Some("<think>".to_string()),
            0xF002 => Some("</think>".to_string()),
            0xF003 => Some("<|START_THINKING|>".to_string()),
            0xF004 => Some("<|END_THINKING|>".to_string()),
            0xF005 => Some("◁think▷".to_string()),
            0xF006 => Some("◁/think▷".to_string()),
            _ => None,
        }
    }
}

impl FakeChatBackend {
    fn new() -> Self {
        Self {
            model_id: "test-model".to_string(),
        }
    }

    fn with_model_id(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
        }
    }
}

impl fmt::Debug for FakeChatBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FakeChatBackend")
            .field("model_id", &self.model_id)
            .finish_non_exhaustive()
    }
}

impl TextBackend for FakeChatBackend {
    fn tokenizer(&self) -> DynTokenizer {
        Arc::new(FakeChatTokenizer)
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

impl ChatBackend for FakeChatBackend {
    fn chat_renderer(&self) -> DynChatRenderer {
        Arc::new(self.clone())
    }

    fn new_chat_output_processor(
        &self,
        request: &mut ChatRequest,
        options: NewChatOutputProcessorOptions<'_>,
    ) -> uniserve_serving::chat::Result<DynChatOutputProcessor> {
        Ok(Box::new(DefaultChatOutputProcessor::new(
            request,
            &self.model_id,
            self.tokenizer(),
            options.tool_call_parser,
            options.uniserve_reasoning_parser,
        )?))
    }
}

impl ChatRenderer for FakeChatBackend {
    fn render(
        &self,
        request: &ChatRequest,
    ) -> uniserve_serving::chat::template::Result<uniserve_serving::chat::RenderedPrompt> {
        let mut prompt = String::new();
        for message in &request.messages {
            prompt.push_str(message.role().as_str());
            prompt.push_str(": ");
            prompt.push_str(&render_fake_message_content(message)?);
            prompt.push('\n');
        }
        if request.chat_options.add_generation_prompt() {
            prompt.push_str("assistant:");
        }
        Ok(uniserve_serving::chat::RenderedPrompt {
            prompt: Prompt::Text(prompt),
        })
    }
}

fn render_fake_message_content(
    message: &ChatMessage,
) -> uniserve_serving::chat::template::Result<String> {
    match message {
        ChatMessage::System { content }
        | ChatMessage::Developer { content, .. }
        | ChatMessage::User { content }
        | ChatMessage::ToolResponse { content, .. } => render_fake_content(content),
        ChatMessage::Assistant { .. } => Ok(message.text_content()?),
    }
}

fn render_fake_content(content: &ChatContent) -> uniserve_serving::chat::template::Result<String> {
    Ok(match content {
        ChatContent::Text(text) => text.clone(),
        ChatContent::Parts(parts) => {
            let mut out = String::new();
            for part in parts {
                match part {
                    ChatContentPart::Text { text } => out.push_str(text),
                    ChatContentPart::ImageUrl { .. } => out.push_str("<image>"),
                }
            }
            out
        }
    })
}

#[derive(Clone, Debug)]
struct FailingDecodeChatBackend;

#[derive(Debug)]
struct FailingDecodeTokenizer;

impl Tokenizer for FailingDecodeTokenizer {
    fn encode(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> uniserve_serving::text::tokenizer::Result<Vec<u32>> {
        FakeChatTokenizer.encode(text, add_special_tokens)
    }

    fn decode(
        &self,
        token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> uniserve_serving::text::tokenizer::Result<String> {
        if token_ids.contains(&(b'i' as u32)) {
            return Err(uniserve_serving::text::tokenizer::TokenizerError(
                "forced decode failure for streaming test".to_string(),
            ));
        }

        FakeChatTokenizer.decode(token_ids, skip_special_tokens)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        FakeChatTokenizer.token_to_id(token)
    }
}

impl TextBackend for FailingDecodeChatBackend {
    fn tokenizer(&self) -> DynTokenizer {
        Arc::new(FailingDecodeTokenizer)
    }

    fn model_id(&self) -> &str {
        "test-model"
    }
}

impl ChatBackend for FailingDecodeChatBackend {
    fn chat_renderer(&self) -> DynChatRenderer {
        Arc::new(self.clone())
    }

    fn new_chat_output_processor(
        &self,
        _request: &mut ChatRequest,
        _options: NewChatOutputProcessorOptions<'_>,
    ) -> uniserve_serving::chat::Result<DynChatOutputProcessor> {
        Ok(Box::new(DefaultChatOutputProcessor::plain_text_only()))
    }
}

impl ChatRenderer for FailingDecodeChatBackend {
    fn render(
        &self,
        request: &ChatRequest,
    ) -> uniserve_serving::chat::template::Result<uniserve_serving::chat::RenderedPrompt> {
        FakeChatBackend::new().render(request)
    }
}

async fn test_models_with_engine_outputs_and_backend_inner(
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
    expected_prompt_token_ids: Option<Vec<u32>>,
    backend: Arc<dyn ChatTextBackend>,
) -> (uniserve_testkit::ServingRuntimeFixture, MockEngineTask) {
    let (client, mock) = EngineCoreClient::connect_mock("test-model");
    let engine_task = MockEngineTask::new(spawn_mock_engine_task(mock, move |mut mock| {
        Box::pin(async move {
            let request = mock.recv_request().await;
            if let Some(expected) = expected_prompt_token_ids {
                assert_eq!(request.generation.prompt_token_ids(), expected);
            }
            mock.send_outputs(engine_outputs_for_request(&request, output_specs));
        })
    }));

    (
        uniserve_testkit::serving_runtime_from_shared_backend(test_gateway(client), backend),
        engine_task,
    )
}

async fn test_models_with_engine_outputs_and_backend(
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
    backend: Arc<dyn ChatTextBackend>,
) -> (uniserve_testkit::ServingRuntimeFixture, MockEngineTask) {
    test_models_with_engine_outputs_and_backend_inner(output_specs, None, backend).await
}

async fn test_chat_with_engine_outputs(
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
) -> (uniserve_testkit::ServingRuntimeFixture, MockEngineTask) {
    test_models_with_engine_outputs_and_backend(output_specs, Arc::new(FakeChatBackend::new()))
        .await
}

async fn test_app() -> axum::Router {
    test_app_with_dev_mode(false).await
}

#[tokio::test]
async fn chat_plan_endpoint_returns_redacted_execution_plan_without_submission() {
    let mut app = test_app().await;
    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions/plan")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "messages": [{"role": "user", "content": "private prompt"}],
                        "max_completion_tokens": 7
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read response");
    let plan: serde_json::Value = serde_json::from_slice(&body).expect("plan json");
    assert_eq!(plan["generation"]["max_tokens"], 7);
    assert!(plan["profile_id"].is_string());
    assert!(!String::from_utf8_lossy(&body).contains("private prompt"));
}

async fn test_app_with_dev_mode(dev_mode_enabled: bool) -> axum::Router {
    let (chat, _engine_task) = test_models_with_engine_outputs_and_backend(
        default_stream_output_specs(),
        Arc::new(FakeChatBackend::new()),
    )
    .await;
    build_router_with_dev_mode(
        Arc::new(test_app_state(
            vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
            chat,
        )),
        dev_mode_enabled,
    )
}

async fn test_app_with_request_id_headers() -> (axum::Router, MockEngineTask) {
    let (chat, engine_task) = test_models_with_engine_outputs_and_backend(
        default_stream_output_specs(),
        Arc::new(FakeChatBackend::new()),
    )
    .await;
    let app = build_router(Arc::new(
        test_app_state(vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()], chat)
            .with_request_id_headers(true),
    ));
    (app, engine_task)
}

async fn test_health_app_with_engine_script<F>(
    script: F,
) -> (axum::Router, Arc<AppState>, MockEngineTask)
where
    F: FnOnce(MockEngine) -> TestFuture<'static> + Send + 'static,
{
    let (client, mock) = EngineCoreClient::connect_mock("test-model");
    let engine_task = MockEngineTask::new(spawn_mock_engine_task(mock, script));

    let chat = uniserve_testkit::serving_runtime_from_shared_backend(
        test_gateway(client),
        Arc::new(FakeChatBackend::new()),
    );
    let state = Arc::new(test_app_state(
        vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
        chat,
    ));
    (build_router(Arc::clone(&state)), state, engine_task)
}

async fn test_admin_app_with_engine_script<F>(script: F) -> (axum::Router, MockEngineTask)
where
    F: FnOnce(MockEngine) -> TestFuture<'static> + Send + 'static,
{
    let (client, mock) = EngineCoreClient::connect_mock("test-model");
    let engine_task = MockEngineTask::new(spawn_mock_engine_task(mock, script));

    let chat = uniserve_testkit::serving_runtime_from_shared_backend(
        test_gateway(client),
        Arc::new(FakeChatBackend::new()),
    );
    (
        build_router_with_dev_mode_and_lora(
            Arc::new(test_app_state(
                vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
                chat,
            )),
            true,
            true,
        ),
        engine_task,
    )
}

async fn test_app_with_native_engine_script<F>(script: F) -> (axum::Router, MockEngineTask)
where
    F: FnOnce(MockEngine) -> TestFuture<'static> + Send + 'static,
{
    let (client, mock) = EngineCoreClient::connect_mock("test-model");
    let engine_task = MockEngineTask::new(spawn_mock_engine_task(mock, script));
    let chat = uniserve_testkit::serving_runtime_from_shared_backend(
        test_gateway(client),
        Arc::new(FakeChatBackend::with_model_id("SenseNova-U1")),
    )
    .with_tool_call_parser(ParserSelection::Explicit("qwen3_xml".to_string()));
    let profile = resolve_generation_dialect_for_model("SenseNova-U1", &FakeChatTokenizer)
        .expect("profile resolution")
        .expect("SenseNova profile");
    (
        build_router(Arc::new(
            test_app_state(vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()], chat)
                .with_generation_dialect(profile),
        )),
        engine_task,
    )
}

async fn test_app_with_engine_handle() -> (axum::Router, MockEngineTask) {
    test_app_with_stream_output_specs(default_stream_output_specs()).await
}

async fn test_app_with_stream_output_specs(
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
) -> (axum::Router, MockEngineTask) {
    let (chat, engine_task) =
        test_models_with_engine_outputs_and_backend(output_specs, Arc::new(FakeChatBackend::new()))
            .await;
    (
        build_router(Arc::new(test_app_state(
            vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
            chat,
        ))),
        engine_task,
    )
}

async fn test_app_with_backend_and_stream_output_specs(
    backend: Arc<dyn ChatTextBackend>,
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
) -> (axum::Router, MockEngineTask) {
    let (chat, engine_task) =
        test_models_with_engine_outputs_and_backend(output_specs, backend).await;
    (
        build_router(Arc::new(test_app_state(
            vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
            chat,
        ))),
        engine_task,
    )
}

async fn test_app_with_backend_and_engine_request_check<F>(
    backend: Arc<dyn ChatTextBackend>,
    check_request: F,
) -> (axum::Router, MockEngineTask)
where
    F: FnOnce(&EngineCoreRequest) + Send + 'static,
{
    let (client, mock) = EngineCoreClient::connect_mock("test-model");
    let engine_task = MockEngineTask::new(spawn_mock_engine_task(mock, move |mut mock| {
        Box::pin(async move {
            let request = mock.recv_request().await;
            check_request(&request);
            mock.send_outputs(engine_outputs_for_request(
                &request,
                default_stream_output_specs(),
            ));
        })
    }));

    let chat = uniserve_testkit::serving_runtime_from_shared_backend(test_gateway(client), backend);
    (
        build_router(Arc::new(test_app_state(
            vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
            chat,
        ))),
        engine_task,
    )
}

async fn test_chat_with_engine_handle() -> (uniserve_testkit::ServingRuntimeFixture, MockEngineTask)
{
    test_chat_with_engine_outputs(default_stream_output_specs()).await
}

async fn server_load(app: &axum::Router) -> u64 {
    let response = app
        .clone()
        .call(
            Request::builder()
                .method("GET")
                .uri("/load")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let value: serde_json::Value = serde_json::from_slice(&body).expect("json body");
    value["server_load"].as_u64().expect("server_load")
}

async fn health_status(app: &axum::Router) -> (StatusCode, Bytes) {
    let response = app
        .clone()
        .call(
            Request::builder()
                .method("GET")
                .uri("/health")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    (status, body)
}

async fn health_response(app: &axum::Router, request_id: Option<&str>) -> axum::response::Response {
    let mut builder = Request::builder().method("GET").uri("/health");
    if let Some(request_id) = request_id {
        builder = builder.header("X-Request-Id", request_id);
    }

    app.clone()
        .call(builder.body(Body::empty()).expect("build request"))
        .await
        .expect("call app")
}

fn metric_value(rendered: &str, metric: &str, labels: Option<&str>) -> Option<f64> {
    rendered.lines().find_map(|line| {
        let rest = line.strip_prefix(metric)?;

        match labels {
            Some(labels) => {
                let (encoded_labels, value) = rest.split_once("} ")?;
                if !encoded_labels.starts_with('{') {
                    return None;
                }
                let expected_parts = labels.split(',');
                if expected_parts
                    .into_iter()
                    .all(|part| encoded_labels.contains(part))
                {
                    value.parse::<f64>().ok()
                } else {
                    None
                }
            }
            None => rest
                .strip_prefix(' ')
                .and_then(|value| value.parse::<f64>().ok()),
        }
    })
}

fn metric_delta(
    rendered_before: &str,
    rendered_after: &str,
    metric: &str,
    labels: Option<&str>,
) -> f64 {
    metric_value(rendered_after, metric, labels).unwrap_or(0.0)
        - metric_value(rendered_before, metric, labels).unwrap_or(0.0)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn list_models_returns_configured_model() {
    let mut app = test_app().await;
    let response = app
        .call(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["data"][0]["id"], "Qwen/Qwen1.5-0.5B-Chat");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn api_key_protects_serving_routes_but_not_operational_probes() {
    let (chat, engine_task) = test_chat_with_engine_handle().await;
    let mut app = build_router(Arc::new(
        test_app_state(vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()], chat)
            .with_api_key(Some("public-key".to_string())),
    ));

    let response = app
        .call(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    let response = app
        .call(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .call(
            Request::builder()
                .uri("/v1/models")
                .header("authorization", "Bearer public-key")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(response.status(), StatusCode::OK);

    drop(app);
    engine_task.abort_and_join().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn admin_api_key_protects_management_routes_when_configured() {
    let (chat, engine_task) = test_chat_with_engine_handle().await;
    let app = build_router(Arc::new(
        test_app_state(vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()], chat)
            .with_api_key(Some("public-key".to_string()))
            .with_admin_api_key(Some("admin-key".to_string()))
            .with_server_dev_mode(true),
    ));

    let response = app
        .clone()
        .call(
            Request::builder()
                .uri("/server_info")
                .header("authorization", "Bearer public-key")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    let response = app
        .clone()
        .call(
            Request::builder()
                .uri("/server_info")
                .header("authorization", "Bearer admin-key")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    engine_task.abort_and_join().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn configured_admission_limit_sheds_tracked_requests() {
    let (chat, engine_task) = test_chat_with_engine_handle().await;
    let state = Arc::new(
        test_app_state(vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()], chat)
            .with_max_concurrent_requests(Some(1)),
    );
    state.increment_server_load();
    let mut app = build_router(Arc::clone(&state));

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    state.decrement_server_load();
    drop(app);
    engine_task.abort_and_join().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn configured_request_timeout_returns_gateway_timeout() {
    let (client, mock) = EngineCoreClient::connect_mock("test-model");
    let engine_task = MockEngineTask::new(spawn_mock_engine_task(mock, |mut mock| {
        boxed_test_future(async move {
            let _request = mock.recv_request().await;
            std::future::pending::<()>().await;
        })
    }));
    let chat = uniserve_testkit::serving_runtime_from_shared_backend(
        test_gateway(client),
        Arc::new(FakeChatBackend::new()),
    );
    let mut app = build_router(Arc::new(
        test_app_state(vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()], chat)
            .with_request_timeout(Some(Duration::from_millis(10))),
    ));

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": false,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    drop(app);
    engine_task.abort_and_join().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn request_id_header_is_absent_by_default() {
    let app = test_app().await;
    let response = health_response(&app, None).await;

    assert_eq!(response.status(), StatusCode::OK);
    assert!(!response.headers().contains_key("x-request-id"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn request_id_header_generates_uuid_hex_when_enabled() {
    let (app, _engine_task) = test_app_with_request_id_headers().await;
    let response = health_response(&app, None).await;

    assert_eq!(response.status(), StatusCode::OK);
    let request_id = response
        .headers()
        .get("x-request-id")
        .expect("x-request-id header")
        .to_str()
        .expect("header is ascii");
    assert_eq!(request_id.len(), 32);
    assert!(request_id.chars().all(|ch| ch.is_ascii_hexdigit()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn request_id_header_echoes_incoming_header_when_enabled() {
    let (app, _engine_task) = test_app_with_request_id_headers().await;
    let response = health_response(&app, Some("req-123")).await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers().get("x-request-id").unwrap(), "req-123");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn version_returns_engine_uniserve_version() {
    let mut app = test_app().await;
    let response = app
        .call(
            Request::builder()
                .uri("/version")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(
        json,
        json!({
            "version": "mock",
            "rust_frontend_version": env!("CARGO_PKG_VERSION"),
        })
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn server_info_endpoint_is_dev_mode_only() {
    let mut app = test_app().await;
    let response = app
        .call(
            Request::builder()
                .uri("/server_info")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn load_lora_adapter_registers_model_and_forwards_lora_request() {
    let (mut app, engine_task) = test_admin_app_with_engine_script(|mut mock| {
        boxed_test_future(async move {
            // `add_lora`/`remove_lora` are answered in-process by the mock client;
            // only the three generate requests reach the mock engine. Each must
            // carry the resolved `adapter-a` LoRA request.
            let request = mock.recv_request().await;
            assert_adapter_a_lora_request(&request);
            mock.send_outputs(engine_outputs_for_request(
                &request,
                default_stream_output_specs(),
            ));

            let request = mock.recv_request().await;
            assert_adapter_a_lora_request(&request);
            mock.send_outputs(engine_outputs_for_request(
                &request,
                default_stream_output_specs(),
            ));

            let request = mock.recv_request().await;
            assert_eq!(request.generation.prompt_token_ids(), vec![11, 22]);
            assert_adapter_a_lora_request(&request);
            mock.send_outputs(engine_outputs_for_request(
                &request,
                default_stream_output_specs(),
            ));
        })
    })
    .await;

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/load_lora_adapter")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "lora_name": "adapter-a",
                        "lora_path": "org/adapter-a"
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(response.status(), StatusCode::OK);

    let models = app
        .call(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");
    let body = to_bytes(models.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["data"][1]["id"], "adapter-a");

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "adapter-a",
                        "prompt": "hello",
                        "max_tokens": 2
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "adapter-a",
                        "stream": false,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/inference/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "adapter-a",
                        "token_ids": [11, 22],
                        "stream": false,
                        "sampling_params": {
                            "max_tokens": 2
                        }
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/unload_lora_adapter")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "lora_name": "adapter-a",
                        "lora_int_id": 1
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(response.status(), StatusCode::OK);

    let models = app
        .call(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");
    let body = to_bytes(models.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["data"].as_array().expect("model data").len(), 1);

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "adapter-a",
                        "prompt": "hello",
                        "max_tokens": 2
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    drop(app);
    engine_task.finish().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn server_info_endpoint_returns_not_found_without_snapshot() {
    let mut app = test_app_with_dev_mode(true).await;
    let response = app
        .call(
            Request::builder()
                .uri("/server_info")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn unload_lora_adapter_rejects_mismatched_lora_int_id() {
    // `add_lora` is answered in-process by the mock client; the unload mismatch is
    // rejected app-side, so the engine never sees a generate request.
    let (mut app, engine_task) =
        test_admin_app_with_engine_script(|_mock| boxed_test_future(async move {})).await;

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/load_lora_adapter")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "lora_name": "adapter-a",
                        "lora_path": "org/adapter-a"
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/unload_lora_adapter")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "lora_name": "adapter-a",
                        "lora_int_id": 99
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let models = app
        .call(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");
    let body = to_bytes(models.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["data"][1]["id"], "adapter-a");

    drop(app);
    engine_task.finish().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn load_lora_adapter_rejects_base_model_name_collision() {
    let (mut app, engine_task) =
        test_admin_app_with_engine_script(|_mock| boxed_test_future(async move {})).await;

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/load_lora_adapter")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "lora_name": "Qwen/Qwen1.5-0.5B-Chat",
                        "lora_path": "org/adapter-a"
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    drop(app);
    engine_task.finish().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn http_metrics_record_list_models_requests() {
    let mut app = test_app().await;
    let before = METRICS.render().unwrap();

    let response = app
        .call(
            Request::builder()
                .method("GET")
                .uri("/v1/models")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    // HTTP metrics are recorded when the response body is fully sent (the
    // body-end guard in `track_http_metrics`), so drive the body to completion
    // before reading the post-request snapshot.
    let _ = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("drain body");

    let after = METRICS.render().unwrap();
    assert_eq!(
        metric_delta(
            &before,
            &after,
            "http_requests_total",
            Some("method=\"GET\",status=\"2xx\",handler=\"/v1/models\""),
        ),
        1.0
    );
    assert_eq!(
        metric_delta(
            &before,
            &after,
            "http_request_duration_seconds_count",
            Some("method=\"GET\",handler=\"/v1/models\""),
        ),
        1.0
    );
    assert_eq!(
        metric_delta(
            &before,
            &after,
            "http_request_duration_highr_seconds_count",
            None,
        ),
        1.0
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn wrong_model_returns_not_found() {
    let mut app = test_app().await;
    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "wrong-model",
                        "stream": true,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn invalid_request_returns_openai_error() {
    let mut app = test_app().await;
    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": false,
                        "stream_options": {"include_usage": true},
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn chat_semantic_context_rejection_remains_an_invalid_request() {
    let mut app = test_app().await;
    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "messages": [{"role": "user", "content": "x".repeat(40000)}],
                        "max_completion_tokens": 1
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let error: serde_json::Value = serde_json::from_slice(&body).expect("error json");
    assert_eq!(error["error"]["type"], "invalid_request_error");
    assert!(
        error["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("maximum context length"))
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_stream_chat_returns_json_response() {
    let (app, engine_task) = test_app_with_engine_handle().await;
    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": false,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("application/json"))
    );

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    assert_eq!(json["choices"][0]["message"]["content"], "hi");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(json["usage"]["prompt_tokens"], 22);
    assert_eq!(json["usage"]["completion_tokens"], 3);
    assert_eq!(json["usage"]["total_tokens"], 25);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_stream_chat_includes_logprobs_and_prompt_logprobs() {
    let (client, mock) = EngineCoreClient::connect_mock("test-model");

    let engine_task = MockEngineTask::new(spawn_mock_engine_task(mock, |mut mock| {
        boxed_test_future(async move {
            let request = mock.recv_request().await;
            let prompt_token_ids = request.generation.prompt_token_ids();

            send_canonical_outputs(
                &mock,
                &request,
                EngineCoreOutputs {
                    engine_index: 0,
                    outputs: vec![request_output_with_logprobs(
                        &request.request_id,
                        bytes_to_token_ids(b"hi"),
                        Some(EngineCoreFinishReason::Stop),
                        None,
                        Some(sample_logprobs_for_tokens(&bytes_to_token_ids(b"hi"))),
                        Some(prompt_logprobs_for_tokens(&prompt_token_ids)),
                    )],
                    scheduler_stats: None,
                    timestamp: 0.0,
                    utility_output: None,
                    finished_requests: None,
                    wave_complete: None,
                    start_wave: None,
                },
            );
        })
    }));

    let chat = uniserve_testkit::serving_runtime_from_shared_backend(
        test_gateway(client),
        Arc::new(FakeChatBackend::new()),
    );
    let mut app = build_router(Arc::new(test_app_state(
        vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
        chat,
    )));

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": false,
                        "logprobs": true,
                        "prompt_logprobs": 1,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    assert_eq!(
        json["choices"][0]["logprobs"]["content"][0]["token"],
        json!("h")
    );
    assert_eq!(
        json["choices"][0]["logprobs"]["content"][1]["token"],
        json!("i")
    );
    assert_eq!(json["prompt_logprobs"][0], serde_json::Value::Null);
    assert!(json["prompt_logprobs"][1].is_object());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn happy_path_returns_sse_stream() {
    let (app, engine_task) = test_app_with_engine_handle().await;
    let before = METRICS.render().unwrap();
    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": true,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let after = METRICS.render().unwrap();

    assert!(text.contains("\"role\":\"assistant\""), "{text}");
    assert!(text.starts_with("data: "), "{text}");
    assert_eq!(
        metric_delta(
            &before,
            &after,
            "http_requests_total",
            Some("method=\"POST\",status=\"2xx\",handler=\"/v1/chat/completions\""),
        ),
        1.0
    );
    assert_eq!(
        metric_delta(
            &before,
            &after,
            "http_request_duration_seconds_count",
            Some("method=\"POST\",handler=\"/v1/chat/completions\""),
        ),
        1.0
    );
    assert_eq!(
        metric_delta(
            &before,
            &after,
            "http_request_duration_highr_seconds_count",
            None,
        ),
        1.0
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn chat_completions_streams_default_image_deltas() {
    let (mut app, engine_task) = test_app_with_native_engine_script(|mut mock| {
        boxed_test_future(async move {
            let request = mock.recv_request().await;
            let generation = &request.generation;
            assert_eq!(generation.constraint, GenerationConstraint::Default);
            assert_eq!(generation.image.width, 2048);
            assert_eq!(generation.image.height, 1152);
            assert_eq!(generation.image.steps, 7);
            assert_eq!(generation.image.cfg_text_scale, 4.5);
            assert_eq!(generation.image.cfg_img_scale, 1.25);
            assert_eq!(generation.image.cfg_renorm_type, "none");
            assert_eq!(generation.image.timestep_shift, 3.0);
            assert_eq!(generation.image.seed, Some(123));
            assert_eq!(generation.image.max_images, 2);
            assert_eq!(generation.max_und_tokens, 16);
            assert!(generation.prompt_token_count() > 0);

            send_canonical_outputs(
                &mock,
                &request,
                EngineCoreOutputs {
                    engine_index: 0,
                    outputs: vec![
                        request_output(&request.request_id, bytes_to_token_ids(b"hi"), None),
                        native_image_output(
                            &request.request_id,
                            WireImageEvent::Begin {
                                image_id: 0,
                                height: 1152,
                                width: 2048,
                                steps: 7,
                            },
                        ),
                        native_image_output(
                            &request.request_id,
                            WireImageEvent::Step {
                                image_id: 0,
                                step: 1,
                            },
                        ),
                        native_image_output(
                            &request.request_id,
                            WireImageEvent::Done {
                                image_id: 0,
                                height: 1152,
                                width: 2048,
                                bytes: 3,
                                sha256: "sha".to_string(),
                                png_b64: "QUJD".to_string(),
                            },
                        ),
                        request_output(&request.request_id, bytes_to_token_ids(b"!"), None),
                        native_finish_output(&request.request_id, "stop", 17, 3, 1),
                    ],
                    scheduler_stats: None,
                    timestamp: 0.0,
                    utility_output: None,
                    finished_requests: None,
                    wave_complete: None,
                    start_wave: None,
                },
            );
        })
    })
    .await;

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": true,
                        "stream_options": {"include_usage": true},
                        "modalities": ["text", "image"],
                        "messages": [
                            {"role": "system", "content": "system"},
                            {"role": "user", "content": "draw a compact travel scene"}
                        ],
                        "max_completion_tokens": 16,
                        "temperature": 0.0,
                        "image_config": {
                            "resolution": "16:9",
                            "steps": 7,
                            "seed": 123,
                            "guidance_scale": 4.5,
                            "image_guidance_scale": 1.25,
                            "cfg_norm": "none",
                            "num_images": 2
                        }
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");

    assert_eq!(streamed_chat_content(&text), "hi!");
    let chunks = sse_json_chunks(&text);
    let image_urls: Vec<String> = chunks
        .iter()
        .filter_map(|chunk| chunk["choices"].get(0))
        .filter_map(|choice| choice["delta"]["images"].as_array())
        .flat_map(|images| images.iter())
        .filter_map(|image| image["image_url"]["url"].as_str().map(str::to_owned))
        .collect();
    assert_eq!(image_urls, vec!["data:image/png;base64,QUJD"]);
    assert_eq!(streamed_finish_reason(&text), Some("stop".to_string()));
    let usage = chunks
        .iter()
        .find(|chunk| chunk["usage"]["prompt_tokens"] == json!(17))
        .expect("usage chunk");
    assert_eq!(usage["usage"]["image_count"], 1);
    assert_eq!(usage["usage"]["image_steps"], 1);
    assert!(!chunks.iter().any(|chunk| {
        matches!(
            chunk["type"].as_str(),
            Some("image_begin" | "image_step" | "image_done")
        )
    }));
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn public_generate_route_is_not_mounted_but_chat_and_images_are_available() {
    let (mut app, engine_task) = test_app_with_native_engine_script(|mut mock| {
        boxed_test_future(async move {
            for request_index in 0..2 {
                let request = mock.recv_request().await;
                assert_eq!(request.generation.constraint, GenerationConstraint::GenOnly);
                if request_index == 1 {
                    assert_eq!(request.request_id, "img-image-route-test");
                }
                send_canonical_outputs(
                    &mock,
                    &request,
                    EngineCoreOutputs {
                        engine_index: 0,
                        outputs: vec![
                            native_image_output(
                                &request.request_id,
                                WireImageEvent::Done {
                                    image_id: 0,
                                    height: 1152,
                                    width: 2048,
                                    bytes: 3,
                                    sha256: "sha".to_string(),
                                    png_b64: "QUJD".to_string(),
                                },
                            ),
                            native_finish_output(&request.request_id, "image_done", 7, 0, 1),
                        ],
                        scheduler_stats: None,
                        timestamp: 0.0,
                        utility_output: None,
                        finished_requests: None,
                        wave_complete: None,
                        start_wave: None,
                    },
                );
            }
        })
    })
    .await;

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "modalities": ["image"],
                        "messages": [{"role": "user", "content": "draw"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(
        json["choices"][0]["message"]["images"][0]["image_url"]["url"],
        "data:image/png;base64,QUJD"
    );

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/images/generations")
                .header("content-type", "application/json")
                .header("x-request-id", "image-route-test")
                .body(Body::from(json!({"prompt": "draw"}).to_string()))
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["data"][0]["b64_json"], "QUJD");

    engine_task.await.expect("mock engine task");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn images_generation_maps_runtime_failure_to_http_error() {
    let (mut app, engine_task) = test_app_with_native_engine_script(|mut mock| {
        boxed_test_future(async move {
            let request = mock.recv_request().await;
            send_canonical_outputs(
                &mock,
                &request,
                EngineCoreOutputs {
                    engine_index: 0,
                    outputs: vec![native_finish_output(&request.request_id, "error", 7, 0, 0)],
                    scheduler_stats: None,
                    timestamp: 0.0,
                    utility_output: None,
                    finished_requests: None,
                    wave_complete: None,
                    start_wave: None,
                },
            );
        })
    })
    .await;

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/images/generations")
                .header("content-type", "application/json")
                .body(Body::from(json!({"prompt": "draw"}).to_string()))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let payload: serde_json::Value = serde_json::from_slice(&body).expect("error JSON");
    assert_eq!(payload["error"]["type"], "server_error");
    assert!(payload.get("data").is_none());
    engine_task.await.expect("mock engine task");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn images_generation_validates_model_and_size_before_submission() {
    let (mut app, engine_task) =
        test_app_with_native_engine_script(|_mock| boxed_test_future(async move {})).await;

    let unknown_model = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/images/generations")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model": "missing-model", "prompt": "draw"}).to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(unknown_model.status(), StatusCode::NOT_FOUND);
    let body = to_bytes(unknown_model.into_body(), usize::MAX)
        .await
        .expect("read body");
    let payload: serde_json::Value = serde_json::from_slice(&body).expect("error JSON");
    assert_eq!(payload["error"]["code"], "model_not_found");

    let invalid_size = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/images/generations")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"prompt": "draw", "size": "1024-by-1024"}).to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(invalid_size.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(invalid_size.into_body(), usize::MAX)
        .await
        .expect("read body");
    let payload: serde_json::Value = serde_json::from_slice(&body).expect("error JSON");
    assert_eq!(payload["error"]["param"], "size");

    engine_task.await.expect("mock engine task");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn images_generation_plan_preserves_quality_controls_without_submission() {
    let (mut app, engine_task) =
        test_app_with_native_engine_script(|_mock| boxed_test_future(async move {})).await;

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/images/generations/plan")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "prompt": "private image prompt",
                        "size": "2048x1152",
                        "steps": 12,
                        "num_inference_steps": 12,
                        "seed": 7,
                        "guidance_scale": 4.0,
                        "image_guidance_scale": 1.25,
                        "cfg_norm": "none",
                        "cfg_interval": [0.1, 0.9],
                        "timestep_shift": 3.0
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let plan: serde_json::Value = serde_json::from_slice(&body).expect("plan json");
    assert_eq!(plan["generation"]["constraint"], "gen_only");
    assert_eq!(plan["generation"]["image"]["width"], 2048);
    assert_eq!(plan["generation"]["image"]["height"], 1152);
    assert_eq!(plan["generation"]["image"]["steps"], 12);
    assert_eq!(plan["generation"]["image"]["cfg_text_scale"], 4.0);
    assert_eq!(plan["generation"]["image"]["cfg_img_scale"], 1.25);
    assert_eq!(plan["generation"]["image"]["cfg_renorm_type"], "none");
    assert_eq!(
        plan["generation"]["image"]["cfg_interval"],
        json!([0.1, 0.9])
    );
    assert_eq!(plan["generation"]["image"]["timestep_shift"], 3.0);
    assert!(!String::from_utf8_lossy(&body).contains("private image prompt"));
    engine_task.await.expect("mock engine task");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn chat_completions_rejects_unsupported_image_config_output_type() {
    let (mut app, engine_task) =
        test_app_with_native_engine_script(|_mock| boxed_test_future(async move {})).await;

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": true,
                        "modalities": ["text", "image"],
                        "messages": [{"role": "user", "content": "draw"}],
                        "image_config": {
                            "width": 2048,
                            "height": 1152,
                            "image_type": "jpeg"
                        }
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(
        json["error"]["message"],
        "image_type must be png for image chat completions"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn chat_completions_routes_image_input_to_native_und_only() {
    let (mut app, engine_task) = test_app_with_native_engine_script(|mut mock| {
        boxed_test_future(async move {
            let request = mock.recv_request().await;
            assert_eq!(request.generation.constraint, GenerationConstraint::UndOnly);
            let images: Vec<_> = request
                .generation
                .context
                .iter()
                .filter_map(|segment| match segment {
                    ContextSegment::Image { image, .. } => Some(image),
                    ContextSegment::UndTokens { .. } => None,
                })
                .collect();
            assert_eq!(images.len(), 1);
            assert_eq!(images[0].b64, TEST_PNG_B64);

            send_canonical_outputs(
                &mock,
                &request,
                EngineCoreOutputs {
                    engine_index: 0,
                    outputs: vec![
                        request_output(&request.request_id, bytes_to_token_ids(b"hi"), None),
                        native_finish_output(&request.request_id, "eos", 9, 2, 0),
                    ],
                    scheduler_stats: None,
                    timestamp: 0.0,
                    utility_output: None,
                    finished_requests: None,
                    wave_complete: None,
                    start_wave: None,
                },
            );
        })
    })
    .await;

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": true,
                        "messages": [
                            {"role":"system","content":"Use the available evidence."},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Inspect this image."},
                                    {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{TEST_PNG_B64}")}}
                                ]
                            },
                            {"role":"assistant","content":"Checking metadata.","tool_calls":[{
                                "id":"call-1","type":"function",
                                "function":{"name":"lookup","arguments":"{\"id\":1}"}
                            }]},
                            {"role":"tool","tool_call_id":"call-1","content":"metadata-result"},
                            {"role":"user","content":"Describe the result."}
                        ],
                        "tools":[{"type":"function","function":{
                            "name":"lookup","description":"Lookup metadata",
                            "parameters":{"type":"object","properties":{"id":{"type":"integer"}}}
                        }}],
                        "tool_choice":"auto",
                        "max_completion_tokens": 16
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    assert_eq!(status, StatusCode::OK, "{text}");
    engine_task.await.expect("mock engine task");
    assert_eq!(streamed_chat_content(&text), "hi");
    assert_eq!(streamed_finish_reason(&text), Some("stop".to_string()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn native_generate_routes_ordered_multimodal_context_to_runtime_events() {
    let (mut app, engine_task) = test_app_with_native_engine_script(|mut mock| {
        boxed_test_future(async move {
            let request = mock.recv_request().await;
            assert_eq!(request.generation.constraint, GenerationConstraint::UndOnly);
            assert_eq!(
                request
                    .generation
                    .context
                    .iter()
                    .filter(|segment| matches!(segment, ContextSegment::Image { .. }))
                    .count(),
                1
            );
            send_canonical_outputs(
                &mock,
                &request,
                EngineCoreOutputs {
                    engine_index: 0,
                    outputs: vec![
                        request_output(&request.request_id, bytes_to_token_ids(b"hi"), None),
                        native_finish_output(&request.request_id, "eos", 9, 2, 0),
                    ],
                    scheduler_stats: None,
                    timestamp: 0.0,
                    utility_output: None,
                    finished_requests: None,
                    wave_complete: None,
                    start_wave: None,
                },
            );
        })
    })
    .await;

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/inference/v1/native/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "profile_id": "sensenova-u1",
                        "constraint": "und_only",
                        "context": [
                            {"type": "text", "role": "user", "text": "Describe this image."},
                            {"type": "image", "b64": TEST_PNG_B64}
                        ],
                        "max_tokens": 16
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read native stream");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    engine_task.await.expect("mock engine task");
    assert!(text.contains(r#""type":"text""#), "{text}");
    let visible = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|event| event["type"] == "text")
        .filter_map(|event| event["text"].as_str().map(str::to_string))
        .collect::<String>();
    assert_eq!(visible, "hi");
    assert!(text.contains(r#""type":"finished""#), "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn chat_completions_maps_resolution_to_profile_bucket() {
    let (mut app, engine_task) = test_app_with_native_engine_script(|mut mock| {
        boxed_test_future(async move {
            let request = mock.recv_request().await;
            assert_eq!(request.generation.constraint, GenerationConstraint::GenOnly);
            assert_eq!(request.generation.image.width, 2048);
            assert_eq!(request.generation.image.height, 1152);

            send_canonical_outputs(
                &mock,
                &request,
                EngineCoreOutputs {
                    engine_index: 0,
                    outputs: vec![
                        native_image_output(
                            &request.request_id,
                            WireImageEvent::Done {
                                image_id: 0,
                                height: 1152,
                                width: 2048,
                                bytes: 3,
                                sha256: "sha".to_string(),
                                png_b64: "QUJD".to_string(),
                            },
                        ),
                        native_finish_output(&request.request_id, "image_done", 7, 0, 1),
                    ],
                    scheduler_stats: None,
                    timestamp: 0.0,
                    utility_output: None,
                    finished_requests: None,
                    wave_complete: None,
                    start_wave: None,
                },
            );
        })
    })
    .await;

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "modalities": ["image"],
                        "messages": [{"role": "user", "content": "draw"}],
                        "image_config": {"resolution": "1.5K", "steps": 7}
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    let images = json["choices"][0]["message"]["images"]
        .as_array()
        .expect("images array");
    assert_eq!(images.len(), 1);
    assert_eq!(images[0]["image_url"]["url"], "data:image/png;base64,QUJD");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn http_metrics_exclude_metrics_route() {
    let mut app = test_app().await;
    let before = METRICS.render().unwrap();

    let response = app
        .call(
            Request::builder()
                .method("GET")
                .uri("/metrics")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let after = METRICS.render().unwrap();
    assert_eq!(
        metric_delta(
            &before,
            &after,
            "http_request_duration_highr_seconds_count",
            None,
        ),
        0.0
    );
    assert_eq!(
        metric_value(
            &after,
            "http_requests_total",
            Some("method=\"GET\",status=\"2xx\",handler=\"/metrics\""),
        ),
        None
    );
    assert_eq!(
        metric_value(
            &after,
            "http_request_duration_seconds_count",
            Some("method=\"GET\",handler=\"/metrics\""),
        ),
        None
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn http_metrics_group_error_statuses() {
    let mut app = test_app().await;
    let before = METRICS.render().unwrap();

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": false,
                        "stream_options": {"include_usage": true},
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    // Metrics are recorded at body completion; drain the body before snapshot.
    let _ = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("drain body");

    let after = METRICS.render().unwrap();
    assert_eq!(
        metric_delta(
            &before,
            &after,
            "http_requests_total",
            Some("method=\"POST\",status=\"4xx\",handler=\"/v1/chat/completions\""),
        ),
        1.0
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn load_endpoint_tracks_chat_stream_lifecycle() {
    let (app, engine_task) = test_app_with_engine_handle().await;

    assert_eq!(server_load(&app).await, 0);

    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": true,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(server_load(&app).await, 1);

    let _body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");

    assert_eq!(server_load(&app).await, 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn health_endpoint_returns_ok_with_empty_body_when_client_is_healthy() {
    let (app, _state, engine_task) =
        test_health_app_with_engine_script(|_mock| boxed_test_future(async move {})).await;

    let (status, body) = health_status(&app).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.is_empty(), "expected empty body, got {:?}", body);

    engine_task.await.expect("mock engine task");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn load_endpoint_resets_when_stream_response_is_dropped() {
    let (app, engine_task) = test_app_with_engine_handle().await;

    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "prompt": "hello",
                        "stream": true
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(server_load(&app).await, 1);

    drop(response);
    tokio::task::yield_now().await;
    engine_task.await.expect("mock engine task");

    assert_eq!(server_load(&app).await, 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn stream_error_is_returned_as_openai_error_sse() {
    let (app, engine_task) = test_app_with_backend_and_stream_output_specs(
        Arc::new(FailingDecodeChatBackend),
        default_stream_output_specs(),
    )
    .await;
    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": true,
                        "stream_options": {"include_usage": true},
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");

    assert!(text.contains("\"role\":\"assistant\""), "{text}");
    assert!(text.contains("\"type\":\"server_error\""), "{text}");
    assert!(
        text.contains("forced decode failure for streaming test"),
        "{text}"
    );
    assert!(!text.contains("\"usage\":"), "{text}");
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn invalid_terminal_finish_reason_is_returned_as_openai_error_sse() {
    let (app, engine_task) =
        test_app_with_stream_output_specs(vec![(vec![], Some(EngineCoreFinishReason::Error))])
            .await;
    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": true,
                        "stream_options": {"include_usage": true},
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");

    assert!(text.contains("\"role\":\"assistant\""), "{text}");
    assert!(text.contains("\"type\":\"server_error\""), "{text}");
    assert!(text.contains("Internal server error"), "{text}");
    assert!(!text.contains("\"usage\":"), "{text}");
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn include_usage_adds_final_usage_chunk_before_done() {
    let (app, engine_task) = test_app_with_stream_output_specs(default_stream_output_specs()).await;
    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": true,
                        "stream_options": {"include_usage": true},
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");

    let payloads = sse_data_payloads(&text);
    let finish_index = payloads
        .iter()
        .position(|payload| payload.contains("\"finish_reason\":\"stop\""))
        .expect("finish chunk");
    let usage_index = payloads
        .iter()
        .position(|payload| payload.contains("\"usage\":"))
        .expect("usage chunk");
    let done_index = payloads
        .iter()
        .position(|payload| *payload == "[DONE]")
        .expect("done sentinel");

    assert!(finish_index < usage_index, "{text}");
    assert!(usage_index < done_index, "{text}");

    let usage_chunk: serde_json::Value =
        serde_json::from_str(payloads[usage_index]).expect("usage chunk json");
    assert_eq!(usage_chunk["choices"], json!([]));
    assert_eq!(usage_chunk["usage"]["prompt_tokens"], 22);
    assert_eq!(usage_chunk["usage"]["completion_tokens"], 3);
    assert_eq!(usage_chunk["usage"]["total_tokens"], 25);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn stream_without_include_usage_keeps_existing_shape() {
    let (app, engine_task) = test_app_with_stream_output_specs(default_stream_output_specs()).await;
    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": true,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");

    assert!(!text.contains("\"usage\":"), "{text}");
    assert!(text.contains("\"finish_reason\":\"stop\""), "{text}");
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn completions_invalid_request_returns_openai_error() {
    let mut app = test_app().await;
    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "prompt": "hello",
                        "stream": false,
                        "stream_options": {"include_usage": true}
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_stream_completions_return_json_response() {
    let (app, engine_task) = test_app_with_engine_handle().await;
    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "prompt": "hello",
                        "stream": false
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("application/json"))
    );

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    assert_eq!(json["object"], "text_completion");
    assert_eq!(json["choices"][0]["text"], "hi");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(json["usage"]["completion_tokens"], 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_stream_completions_echo_prepends_prompt_text() {
    let (app, engine_task) = test_app_with_engine_handle().await;
    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "prompt": "hello",
                        "echo": true,
                        "stream": false
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    assert_eq!(json["choices"][0]["text"], "hellohi");
    assert_eq!(json["usage"]["prompt_tokens"], 5);
    assert_eq!(json["usage"]["completion_tokens"], 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_stream_completions_include_logprobs() {
    let (client, mock) = EngineCoreClient::connect_mock("test-model");

    let engine_task = MockEngineTask::new(spawn_mock_engine_task(mock, |mut mock| {
        boxed_test_future(async move {
            let request = mock.recv_request().await;
            send_canonical_outputs(
                &mock,
                &request,
                EngineCoreOutputs {
                    engine_index: 0,
                    outputs: vec![
                        request_output_with_logprobs(
                            &request.request_id,
                            vec![b'h' as u32],
                            None,
                            None,
                            Some(sample_logprobs_for_token(b'h' as u32, b'H' as u32)),
                            None,
                        ),
                        request_output_with_logprobs(
                            &request.request_id,
                            vec![b'i' as u32],
                            Some(EngineCoreFinishReason::Stop),
                            None,
                            Some(sample_logprobs_for_token(b'i' as u32, b'I' as u32)),
                            None,
                        ),
                    ],
                    scheduler_stats: None,
                    timestamp: 0.0,
                    utility_output: None,
                    finished_requests: Some(BTreeSet::from([request.request_id.clone()])),
                    wave_complete: None,
                    start_wave: None,
                },
            );
        })
    }));

    let chat = uniserve_testkit::serving_runtime_from_shared_backend(
        test_gateway(client),
        Arc::new(FakeChatBackend::new()),
    );
    let mut app = build_router(Arc::new(test_app_state(
        vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
        chat,
    )));

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "prompt": "hello",
                        "stream": false,
                        "logprobs": 1
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    assert_eq!(json["choices"][0]["logprobs"]["tokens"], json!(["h", "i"]));
    assert_eq!(
        json["choices"][0]["logprobs"]["token_logprobs"],
        json!([-0.1, -0.1])
    );
    assert_eq!(json["choices"][0]["logprobs"]["text_offset"], json!([0, 1]));
    assert_eq!(
        json["choices"][0]["logprobs"]["top_logprobs"][0],
        json!({"h": -0.1, "H": -0.2})
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_stream_completions_include_prompt_logprobs() {
    let (client, mock) = EngineCoreClient::connect_mock("test-model");

    let engine_task = MockEngineTask::new(spawn_mock_engine_task(mock, |mut mock| {
        boxed_test_future(async move {
            let request = mock.recv_request().await;
            send_canonical_outputs(
                &mock,
                &request,
                EngineCoreOutputs {
                    engine_index: 0,
                    outputs: vec![request_output_with_logprobs(
                        &request.request_id,
                        vec![b'h' as u32, b'i' as u32, b'!' as u32],
                        Some(EngineCoreFinishReason::Stop),
                        None,
                        Some(Logprobs {
                            positions: vec![
                                sample_logprobs_for_token(b'h' as u32, b'H' as u32).positions[0]
                                    .clone(),
                                sample_logprobs_for_token(b'i' as u32, b'I' as u32).positions[0]
                                    .clone(),
                                sample_logprobs_for_token(b'!' as u32, b'?' as u32).positions[0]
                                    .clone(),
                            ],
                        }),
                        Some(prompt_logprobs_for_hello()),
                    )],
                    scheduler_stats: None,
                    timestamp: 0.0,
                    utility_output: None,
                    finished_requests: None,
                    wave_complete: None,
                    start_wave: None,
                },
            );
        })
    }));

    let chat = uniserve_testkit::serving_runtime_from_shared_backend(
        test_gateway(client),
        Arc::new(FakeChatBackend::new()),
    );
    let mut app = build_router(Arc::new(test_app_state(
        vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
        chat,
    )));

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "prompt": "hello",
                        "stream": false,
                        "echo": true,
                        "logprobs": 1
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    assert_eq!(json["choices"][0]["text"], "hellohi!");
    assert_eq!(
        json["choices"][0]["logprobs"]["tokens"],
        json!(["h", "e", "l", "l", "o", "h", "i", "!"])
    );
    assert_eq!(
        json["choices"][0]["logprobs"]["text_offset"],
        json!([0, 1, 2, 3, 4, 5, 6, 7])
    );
    assert_eq!(
        json["choices"][0]["logprobs"]["token_logprobs"],
        json!([null, -0.3, -0.4, -0.45, -0.5, -0.1, -0.1, -0.1])
    );
    assert_eq!(
        json["choices"][0]["prompt_logprobs"][0],
        serde_json::Value::Null
    );
    assert_eq!(
        json["choices"][0]["prompt_logprobs"][1],
        json!({"a": -0.5, "e": -0.3})
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn chat_completions_header_request_id_takes_precedence() {
    let (client, mock) = EngineCoreClient::connect_mock("test-model");

    let engine_task = MockEngineTask::new(spawn_mock_engine_task(mock, |mut mock| {
        boxed_test_future(async move {
            let request = mock.recv_request().await;
            assert_eq!(request.request_id, "chatcmpl-header-req");

            mock.send_outputs(engine_outputs_for_request(
                &request,
                default_stream_output_specs(),
            ));
        })
    }));

    let chat = uniserve_testkit::serving_runtime_from_shared_backend(
        EngineGateway::new(client),
        Arc::new(FakeChatBackend::new()),
    );
    let mut app = build_router(Arc::new(test_app_state(
        vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
        chat,
    )));

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("X-Request-Id", "header-req")
                .body(Body::from(
                    json!({
                        "request_id": "body-req",
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    assert_eq!(json["id"], "chatcmpl-header-req");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_stream_raw_generate_returns_token_output_envelope() {
    let (client, mock) = EngineCoreClient::connect_mock("test-model");

    let engine_task = MockEngineTask::new(spawn_mock_engine_task(mock, |mut mock| {
        boxed_test_future(async move {
            let request = mock.recv_request().await;
            assert_eq!(request.generation.prompt_token_ids(), vec![11, 22]);
            assert_eq!(request.request_id, "raw-req");

            send_canonical_outputs(
                &mock,
                &request,
                EngineCoreOutputs {
                    engine_index: 0,
                    outputs: vec![
                        request_output_with_logprobs(
                            &request.request_id,
                            vec![33],
                            None,
                            None,
                            Some(sample_logprobs_for_token(33, 34)),
                            Some(prompt_logprobs_for_tokens(&[11, 22])),
                        ),
                        request_output_with_logprobs_and_kv(
                            &request.request_id,
                            vec![44],
                            Some(EngineCoreFinishReason::Stop),
                            None,
                            Some(sample_logprobs_for_token(44, 45)),
                            None,
                            Some(json!({"connector": "x"})),
                        ),
                    ],
                    scheduler_stats: None,
                    timestamp: 0.0,
                    utility_output: None,
                    finished_requests: None,
                    wave_complete: None,
                    start_wave: None,
                },
            );
        })
    }));

    let chat = uniserve_testkit::serving_runtime_from_shared_backend(
        EngineGateway::new(client),
        Arc::new(FakeChatBackend::new()),
    );
    let mut app = build_router(Arc::new(test_app_state(
        vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
        chat,
    )));

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/inference/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "request_id": "raw-req",
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "token_ids": [11, 22],
                        "stream": false,
                        "sampling_params": {
                            "max_tokens": 2,
                            "logprobs": 1,
                            "prompt_logprobs": 1
                        }
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    assert_eq!(json["request_id"], "raw-req");
    assert_eq!(json["choices"][0]["index"], 0);
    assert_eq!(json["choices"][0]["token_ids"], json!([33, 44]));
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(
        json["choices"][0]["logprobs"]["content"][0]["token"],
        "token_id:33"
    );
    assert_eq!(
        json["choices"][0]["logprobs"]["content"][1]["top_logprobs"][0]["token"],
        "token_id:44"
    );
    assert_eq!(json["prompt_logprobs"][0], serde_json::Value::Null);
    assert_eq!(
        json["prompt_logprobs"][1]["22"]["decoded_token"],
        "token_id:22"
    );
    assert_eq!(json["kv_transfer_params"], json!({"connector": "x"}));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn stream_raw_generate_returns_sse_chunks_and_usage() {
    let (client, mock) = EngineCoreClient::connect_mock("test-model");

    let engine_task = MockEngineTask::new(spawn_mock_engine_task(mock, |mut mock| {
        boxed_test_future(async move {
            let request = mock.recv_request().await;
            assert_eq!(request.generation.prompt_token_ids(), vec![11, 22]);

            send_canonical_outputs(
                &mock,
                &request,
                EngineCoreOutputs {
                    engine_index: 0,
                    outputs: vec![
                        request_output_with_logprobs(
                            &request.request_id,
                            vec![33],
                            None,
                            None,
                            Some(sample_logprobs_for_token(33, 34)),
                            None,
                        ),
                        request_output_with_logprobs(
                            &request.request_id,
                            vec![44],
                            Some(EngineCoreFinishReason::Stop),
                            None,
                            Some(sample_logprobs_for_token(44, 45)),
                            None,
                        ),
                    ],
                    scheduler_stats: None,
                    timestamp: 0.0,
                    utility_output: None,
                    finished_requests: None,
                    wave_complete: None,
                    start_wave: None,
                },
            );
        })
    }));

    let chat = uniserve_testkit::serving_runtime_from_shared_backend(
        EngineGateway::new(client),
        Arc::new(FakeChatBackend::new()),
    );
    let mut app = build_router(Arc::new(test_app_state(
        vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
        chat,
    )));

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/inference/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "request_id": "raw-stream",
                        "token_ids": [11, 22],
                        "stream": true,
                        "stream_options": {
                            "include_usage": true,
                            "continuous_usage_stats": true
                        },
                        "sampling_params": {
                            "max_tokens": 2,
                            "logprobs": 1
                        }
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let payloads = sse_data_payloads(&text);
    assert_eq!(payloads.len(), 5, "{text}");

    let first: serde_json::Value = serde_json::from_str(payloads[0]).expect("first chunk json");
    assert_eq!(first["request_id"], "raw-stream");
    assert_eq!(first["choices"][0]["index"], 0);
    assert_eq!(first["choices"][0]["token_ids"], json!([33]));
    assert_eq!(
        first["choices"][0]["logprobs"]["content"][0]["token"],
        "token_id:33"
    );
    assert_eq!(first["usage"]["prompt_tokens"], 2);
    assert_eq!(first["usage"]["completion_tokens"], 1);

    let second: serde_json::Value = serde_json::from_str(payloads[1]).expect("second chunk json");
    assert_eq!(second["choices"][0]["token_ids"], json!([44]));
    assert!(second["choices"][0].get("finish_reason").is_none());
    assert_eq!(second["usage"]["completion_tokens"], 2);

    let finish: serde_json::Value = serde_json::from_str(payloads[2]).expect("finish chunk json");
    assert_eq!(finish["choices"][0]["token_ids"], json!([]));
    assert_eq!(finish["choices"][0]["finish_reason"], "stop");
    assert_eq!(finish["usage"]["completion_tokens"], 2);

    let usage: serde_json::Value = serde_json::from_str(payloads[3]).expect("usage chunk json");
    assert_eq!(usage["choices"], json!([]));
    assert_eq!(usage["usage"]["prompt_tokens"], 2);
    assert_eq!(usage["usage"]["completion_tokens"], 2);
    assert_eq!(usage["usage"]["total_tokens"], 4);
    assert_eq!(payloads[4], "[DONE]");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn stream_raw_generate_emits_final_usage_without_continuous_usage() {
    let (mut app, engine_task) = test_app_with_stream_output_specs(vec![
        (vec![33], None),
        (vec![44], Some(EngineCoreFinishReason::Stop)),
    ])
    .await;

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/inference/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "request_id": "raw-stream-final-usage",
                        "token_ids": [11, 22],
                        "stream": true,
                        "stream_options": {
                            "include_usage": true
                        },
                        "sampling_params": {
                            "max_tokens": 2
                        }
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let payloads = sse_data_payloads(&text);
    assert_eq!(payloads.len(), 5, "{text}");

    let first: serde_json::Value = serde_json::from_str(payloads[0]).expect("first chunk json");
    assert_eq!(first["choices"][0]["token_ids"], json!([33]));
    assert!(first.get("usage").is_none());

    let second: serde_json::Value = serde_json::from_str(payloads[1]).expect("second chunk json");
    assert_eq!(second["choices"][0]["token_ids"], json!([44]));
    assert!(second["choices"][0].get("finish_reason").is_none());
    assert!(second.get("usage").is_none());

    let finish: serde_json::Value = serde_json::from_str(payloads[2]).expect("finish chunk json");
    assert_eq!(finish["choices"][0]["token_ids"], json!([]));
    assert_eq!(finish["choices"][0]["finish_reason"], "stop");
    assert!(finish.get("usage").is_none());

    let usage: serde_json::Value = serde_json::from_str(payloads[3]).expect("usage chunk json");
    assert_eq!(usage["choices"], json!([]));
    assert_eq!(usage["usage"]["prompt_tokens"], 2);
    assert_eq!(usage["usage"]["completion_tokens"], 2);
    assert_eq!(usage["usage"]["total_tokens"], 4);
    assert_eq!(payloads[4], "[DONE]");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn stream_raw_generate_emits_empty_finish_chunk() {
    let (mut app, engine_task) = test_app_with_stream_output_specs(vec![
        (vec![33], None),
        (vec![], Some(EngineCoreFinishReason::Stop)),
    ])
    .await;

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/inference/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "request_id": "raw-stream-empty-finish",
                        "token_ids": [11, 22],
                        "stream": true,
                        "sampling_params": {
                            "max_tokens": 2
                        }
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let payloads = sse_data_payloads(&text);
    assert_eq!(payloads.len(), 3, "{text}");

    let first: serde_json::Value = serde_json::from_str(payloads[0]).expect("first chunk json");
    assert_eq!(first["choices"][0]["token_ids"], json!([33]));
    assert!(first["choices"][0].get("finish_reason").is_none());

    let second: serde_json::Value = serde_json::from_str(payloads[1]).expect("second chunk json");
    assert_eq!(second["choices"][0]["token_ids"], json!([]));
    assert_eq!(second["choices"][0]["finish_reason"], "stop");
    assert_eq!(payloads[2], "[DONE]");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn stream_raw_generate_error_finish_returns_sse_error() {
    let (mut app, engine_task) =
        test_app_with_stream_output_specs(vec![(vec![], Some(EngineCoreFinishReason::Error))])
            .await;

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/inference/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "request_id": "raw-stream-error",
                        "token_ids": [11, 22],
                        "stream": true,
                        "stream_options": {
                            "include_usage": true
                        },
                        "sampling_params": {
                            "max_tokens": 2
                        }
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");

    assert!(text.contains("\"type\":\"server_error\""), "{text}");
    assert!(text.contains("Internal server error"), "{text}");
    assert!(!text.contains("\"finish_reason\":\"error\""), "{text}");
    assert!(!text.contains("\"usage\":"), "{text}");
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn raw_generate_rejects_empty_token_ids() {
    let mut app = test_app().await;

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/inference/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "token_ids": [],
                        "sampling_params": {}
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["error"]["param"], "token_ids");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn raw_generate_rejects_wrong_model() {
    let mut app = test_app().await;

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/inference/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "wrong-model",
                        "token_ids": [11, 22],
                        "sampling_params": {}
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn completions_happy_path_returns_sse_stream() {
    let (app, engine_task) = test_app_with_engine_handle().await;
    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "prompt": "hello",
                        "stream": true,
                        "stream_options": {"include_usage": true}
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let payloads = sse_data_payloads(&text);
    let usage_index = payloads
        .iter()
        .position(|payload| payload.contains("\"usage\":"))
        .expect("usage chunk");
    let done_index = payloads
        .iter()
        .position(|payload| *payload == "[DONE]")
        .expect("done sentinel");

    assert!(
        payloads
            .iter()
            .any(|payload| payload.contains("\"text\":\"h\"")),
        "{text}"
    );
    assert!(
        payloads
            .iter()
            .any(|payload| payload.contains("\"finish_reason\":\"stop\"")),
        "{text}"
    );
    assert!(usage_index < done_index, "{text}");

    let usage_chunk: serde_json::Value =
        serde_json::from_str(payloads[usage_index]).expect("usage chunk json");
    assert_eq!(usage_chunk["choices"], json!([]));
    assert_eq!(usage_chunk["usage"]["completion_tokens"], 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn completions_echo_stream_emits_separate_prompt_chunk() {
    let (app, engine_task) = test_app_with_engine_handle().await;
    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "prompt": "hello",
                        "echo": true,
                        "stream": true,
                        "stream_options": {"include_usage": true}
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let payloads = sse_data_payloads(&text);
    let hello_index = payloads
        .iter()
        .position(|payload| payload.contains("\"text\":\"hello\""))
        .expect("prompt echo chunk");
    let h_index = payloads
        .iter()
        .position(|payload| payload.contains("\"text\":\"h\""))
        .expect("first generation chunk");

    assert!(hello_index < h_index, "{text}");
    assert!(
        payloads
            .iter()
            .any(|payload| payload.contains("\"text\":\"i\"")),
        "{text}"
    );

    let usage_chunk: serde_json::Value = serde_json::from_str(
        payloads
            .iter()
            .find(|payload| payload.contains("\"usage\":"))
            .expect("usage chunk"),
    )
    .expect("usage chunk json");
    assert_eq!(usage_chunk["usage"]["prompt_tokens"], 5);
    assert_eq!(usage_chunk["usage"]["completion_tokens"], 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn reasoning_blocks_are_mapped_to_reasoning_sse_chunks() {
    let (app, engine_task) = test_app_with_backend_and_stream_output_specs(
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B")),
        vec![
            (bytes_to_token_ids(b"<think>"), None),
            (bytes_to_token_ids(b"think "), None),
            (bytes_to_token_ids(b"more</think>"), None),
            (
                bytes_to_token_ids(b"answer"),
                Some(EngineCoreFinishReason::Length),
            ),
        ],
    )
    .await;

    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": true,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");

    assert_eq!(streamed_reasoning_content(&text), "think more");
    assert_eq!(streamed_chat_content(&text), "answer");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn include_reasoning_false_suppresses_reasoning_in_non_stream_chat() {
    let (app, engine_task) = test_app_with_backend_and_stream_output_specs(
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B")),
        vec![
            (bytes_to_token_ids(b"<think>think</think>"), None),
            (
                bytes_to_token_ids(b"answer"),
                Some(EngineCoreFinishReason::Length),
            ),
        ],
    )
    .await;

    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": false,
                        "include_reasoning": false,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let json: serde_json::Value = serde_json::from_str(&text).expect("decode json");

    assert_eq!(json["choices"][0]["message"]["content"], "answer");
    assert!(
        json["choices"][0]["message"]
            .as_object()
            .is_some_and(|message| !message.contains_key("reasoning_content")),
        "{text}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn include_reasoning_false_suppresses_non_stream_output_metadata() {
    let (client, mock) = EngineCoreClient::connect_mock("test-model");

    let engine_task = MockEngineTask::new(spawn_mock_engine_task(mock, |mut mock| {
        boxed_test_future(async move {
            let request = mock.recv_request().await;
            let reasoning_token_ids = bytes_to_token_ids(b"<think>think</think>");
            let answer_token_ids = bytes_to_token_ids(b"answer");

            send_canonical_outputs(
                &mock,
                &request,
                EngineCoreOutputs {
                    engine_index: 0,
                    outputs: vec![
                        request_output_with_logprobs(
                            &request.request_id,
                            reasoning_token_ids.clone(),
                            None,
                            None,
                            Some(sample_logprobs_for_tokens(&reasoning_token_ids)),
                            None,
                        ),
                        request_output_with_logprobs(
                            &request.request_id,
                            answer_token_ids.clone(),
                            Some(EngineCoreFinishReason::Length),
                            None,
                            Some(sample_logprobs_for_tokens(&answer_token_ids)),
                            None,
                        ),
                    ],
                    scheduler_stats: None,
                    timestamp: 0.0,
                    utility_output: None,
                    finished_requests: None,
                    wave_complete: None,
                    start_wave: None,
                },
            );
        })
    }));

    let chat = uniserve_testkit::serving_runtime_from_shared_backend(
        test_gateway(client),
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B")),
    );
    let mut app = build_router(Arc::new(test_app_state(
        vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
        chat,
    )));

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": false,
                        "include_reasoning": false,
                        "logprobs": true,
                        "return_token_ids": true,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let json: serde_json::Value = serde_json::from_str(&text).expect("decode json");
    let choice = json["choices"][0].as_object().expect("choice object");

    assert_eq!(json["choices"][0]["message"]["content"], "answer");
    assert!(
        json["choices"][0]["message"]
            .as_object()
            .is_some_and(|message| !message.contains_key("reasoning_content")),
        "{text}"
    );
    assert!(!choice.contains_key("logprobs"), "{text}");
    assert!(!choice.contains_key("token_ids"), "{text}");
    assert!(json["prompt_token_ids"].is_array(), "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn tool_calls_are_mapped_to_tool_call_sse_chunks() {
    let (app, engine_task) = test_app_with_backend_and_stream_output_specs(
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B")),
        vec![
            (bytes_to_token_ids(b"<think>Need tool.</think>"), None),
            (
                bytes_to_token_ids(b"<tool_call>\n{\"name\":\"get_weather\", "),
                None,
            ),
            (
                bytes_to_token_ids(b"\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>"),
                Some(EngineCoreFinishReason::Stop),
            ),
        ],
    )
    .await;

    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": true,
                        "messages": [{"role": "user", "content": "hello"}],
                        "tools": [{
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}}
                                }
                            }
                        }]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");

    assert!(text.contains("\"tool_calls\":"), "{text}");
    assert!(text.contains("\"name\":\"get_weather\""), "{text}");
    assert_eq!(streamed_tool_arguments(&text), r#"{"city":"Paris"}"#);
    assert!(text.contains("\"finish_reason\":\"tool_calls\""), "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn tool_call_sse_chunks_can_carry_logprobs() {
    let (client, mock) = EngineCoreClient::connect_mock("test-model");

    let engine_task = MockEngineTask::new(spawn_mock_engine_task(mock, |mut mock| {
        Box::pin(async move {
            let request = mock.recv_request().await;

            send_canonical_outputs(
                &mock,
                &request,
                EngineCoreOutputs {
                    engine_index: 0,
                    outputs: vec![request_output_with_logprobs(
                        &request.request_id,
                        bytes_to_token_ids(b"<think>Need tool.</think>"),
                        None,
                        None,
                        Some(sample_logprobs_for_tokens(&bytes_to_token_ids(
                            b"<think>Need tool.</think>",
                        ))),
                        None,
                    )],
                    scheduler_stats: None,
                    timestamp: 0.0,
                    utility_output: None,
                    finished_requests: None,
                    wave_complete: None,
                    start_wave: None,
                },
            );
            send_canonical_outputs(
                &mock,
                &request,
                EngineCoreOutputs {
                    engine_index: 0,
                    outputs: vec![request_output_with_logprobs(
                        &request.request_id,
                        bytes_to_token_ids(b"<tool_call>\n{\"name\":\"get_weather\", "),
                        None,
                        None,
                        Some(sample_logprobs_for_tokens(&bytes_to_token_ids(
                            b"<tool_call>\n{\"name\":\"get_weather\", ",
                        ))),
                        None,
                    )],
                    scheduler_stats: None,
                    timestamp: 0.0,
                    utility_output: None,
                    finished_requests: None,
                    wave_complete: None,
                    start_wave: None,
                },
            );
            send_canonical_outputs(
                &mock,
                &request,
                EngineCoreOutputs {
                    engine_index: 0,
                    outputs: vec![request_output_with_logprobs(
                        &request.request_id,
                        bytes_to_token_ids(b"\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>"),
                        Some(EngineCoreFinishReason::Stop),
                        None,
                        Some(sample_logprobs_for_tokens(&bytes_to_token_ids(
                            b"\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>",
                        ))),
                        None,
                    )],
                    scheduler_stats: None,
                    timestamp: 0.0,
                    utility_output: None,
                    finished_requests: Some(BTreeSet::from([request.request_id.clone()])),
                    wave_complete: None,
                    start_wave: None,
                },
            );
        })
    }));

    let chat = uniserve_testkit::serving_runtime_from_shared_backend(
        test_gateway(client),
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B")),
    );
    let app = build_router(Arc::new(test_app_state(
        vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
        chat,
    )));

    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": true,
                        "logprobs": true,
                        "messages": [{"role": "user", "content": "hello"}],
                        "tools": [{
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}}
                                }
                            }
                        }]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.finish().await;
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");

    assert!(text.contains("\"tool_calls\":"), "{text}");
    assert!(text.contains("\"logprobs\":{\"content\":"), "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn streaming_chat_prompt_logprobs_are_rejected() {
    let (app, engine_task) = test_app_with_engine_handle().await;
    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": true,
                        "prompt_logprobs": 1,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    engine_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn reset_prefix_cache_route_rejects_unconfigured_external_reset() {
    // `reset_prefix_cache` is answered in-process by the mock client.
    let (app, engine_task) =
        test_admin_app_with_engine_script(|_mock| boxed_test_future(async move {})).await;

    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/reset_prefix_cache?reset_running_requests=true&reset_external=true")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert!(String::from_utf8_lossy(&body).contains("no external prefix-cache connector"));
    engine_task.await.expect("mock engine task");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn reset_mm_cache_route_sends_expected_utility_call() {
    // `reset_mm_cache` is answered in-process by the mock client.
    let (app, engine_task) =
        test_admin_app_with_engine_script(|_mock| boxed_test_future(async move {})).await;

    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/reset_mm_cache")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    assert!(body.is_empty());
    engine_task.await.expect("mock engine task");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn reset_encoder_cache_route_sends_expected_utility_call() {
    // `reset_encoder_cache` is answered in-process by the mock client.
    let (app, engine_task) =
        test_admin_app_with_engine_script(|_mock| boxed_test_future(async move {})).await;

    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/reset_encoder_cache")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    assert!(body.is_empty());
    engine_task.await.expect("mock engine task");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn sleep_route_uses_python_compatible_default_query_values() {
    // `sleep` is answered in-process by the mock client.
    let (app, engine_task) =
        test_admin_app_with_engine_script(|_mock| boxed_test_future(async move {})).await;

    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/sleep")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    assert!(body.is_empty());
    engine_task.await.expect("mock engine task");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn wake_up_route_without_tags_sends_none() {
    // `wake_up` is answered in-process by the mock client.
    let (app, engine_task) =
        test_admin_app_with_engine_script(|_mock| boxed_test_future(async move {})).await;

    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/wake_up")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    assert!(body.is_empty());
    engine_task.await.expect("mock engine task");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn admin_routes_are_hidden_when_dev_mode_is_disabled() {
    let (chat, engine_task) = test_chat_with_engine_handle().await;
    let app = build_router_with_dev_mode(
        Arc::new(test_app_state(
            vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
            chat,
        )),
        false,
    );

    for (method, uri) in [
        ("GET", "/is_sleeping"),
        ("POST", "/sleep"),
        ("POST", "/wake_up"),
        ("POST", "/collective_rpc"),
        ("POST", "/reset_prefix_cache"),
        ("POST", "/reset_mm_cache"),
        ("POST", "/reset_encoder_cache"),
    ] {
        let response = app
            .clone()
            .call(
                Request::builder()
                    .method(method)
                    .uri(uri)
                    .body(Body::empty())
                    .expect("build request"),
            )
            .await
            .expect("call app");

        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{method} {uri}");
    }

    engine_task.abort_and_join().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn streaming_chat_concatenated_content_and_stop_finish_reason() {
    // The mock engine emits "h", "i", then "!" with a terminal Stop. The "!" token
    // is the EOS-driven stop token and is suppressed from the visible output, so the
    // streamed chat content concatenates to "hi" and the single finish_reason is
    // "stop".
    let (app, engine_task) = test_app_with_engine_handle().await;
    let mut app = app;
    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": true,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");

    assert_eq!(streamed_chat_content(&text), "hi");
    assert_eq!(streamed_finish_reason(&text).as_deref(), Some("stop"));
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn chat_request_sampling_fields_are_mapped_into_engine_request() {
    // prepare_chat_request field mapping: temperature/top_p/top_k/seed/max_tokens
    // from the OpenAI chat body must reach the engine-core request unchanged.
    let (mut app, engine_task) = test_app_with_backend_and_engine_request_check(
        Arc::new(FakeChatBackend::new()),
        |request| {
            let sampling = &request.generation.sampling;
            assert!(
                (sampling.temperature - 0.5).abs() < 1e-6,
                "temperature was {}",
                sampling.temperature
            );
            assert!(
                (sampling.top_p - 0.8).abs() < 1e-6,
                "top_p was {}",
                sampling.top_p
            );
            assert_eq!(sampling.top_k, 7);
            assert_eq!(sampling.seed, Some(99));
            assert_eq!(request.generation.max_und_tokens, 16);
        },
    )
    .await;

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": false,
                        "temperature": 0.5,
                        "top_p": 0.8,
                        "top_k": 7,
                        "seed": 99,
                        "max_completion_tokens": 16,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let _ = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
}

// ========================= Stop string tests =========================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_stream_completions_stop_string_excluded_from_output() {
    // Engine generates "say world" but stop string "wor" truncates output to "say
    // ".
    let output_specs = vec![
        (bytes_to_token_ids(b"say"), None),
        (
            bytes_to_token_ids(b" world"),
            Some(EngineCoreFinishReason::Length),
        ),
    ];
    let (app, engine_task) = test_app_with_stream_output_specs(output_specs).await;

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "prompt": "hello",
                        "stream": false,
                        "stop": ["wor"]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    assert_eq!(json["choices"][0]["text"], "say ");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(json["choices"][0]["stop_reason"], "wor");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_stream_completions_stop_string_included_in_output() {
    // Same tokens but include_stop_str_in_output=true includes the stop string in
    // the output.
    let output_specs = vec![
        (bytes_to_token_ids(b"say"), None),
        (
            bytes_to_token_ids(b" world"),
            Some(EngineCoreFinishReason::Length),
        ),
    ];
    let (app, engine_task) = test_app_with_stream_output_specs(output_specs).await;

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "prompt": "hello",
                        "stream": false,
                        "stop": ["wor"],
                        "include_stop_str_in_output": true
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    assert_eq!(json["choices"][0]["text"], "say wor");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(json["choices"][0]["stop_reason"], "wor");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_stream_completions_no_stop_string_match_preserves_original_finish_reason() {
    // Stop string "xyz" does not appear in "hi!" so the original finish reason is
    // preserved.
    let (app, engine_task) = test_app_with_engine_handle().await;

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "prompt": "hello",
                        "stream": false,
                        "stop": ["xyz"]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    // Default output is "hi" (stop token '!' suppressed), finish_reason remains
    // "stop" from EOS.
    assert_eq!(json["choices"][0]["text"], "hi");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    // No text stop string matched — stop_reason should be absent.
    assert!(json["choices"][0]["stop_reason"].is_null());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_stream_completions_stop_string_array_matches_first_occurrence() {
    // Multiple stop strings: "rl" appears in "world" but " wo" appears earlier.
    let output_specs = vec![(
        bytes_to_token_ids(b"say world"),
        Some(EngineCoreFinishReason::Length),
    )];
    let (app, engine_task) = test_app_with_stream_output_specs(output_specs).await;

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "prompt": "hello",
                        "stream": false,
                        "stop": [" wo", "rl"]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    // " wo" is detected first (at byte 3), so output is truncated to "say".
    assert_eq!(json["choices"][0]["text"], "say");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(json["choices"][0]["stop_reason"], " wo");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn completions_empty_stop_string_returns_validation_error() {
    let (app, _engine_task) = test_app_with_engine_handle().await;

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "prompt": "hello",
                        "stream": false,
                        "stop": [""]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

// ============== Streaming completions stop-string trimming ==============

/// Drive a streaming `/v1/completions` request with the given engine output
/// specs and stop options, returning the full SSE body text. Uses
/// `Service::call` (not `oneshot`) so the streaming body drains before the
/// router/state is dropped.
async fn stream_completion_with_stop(
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
    stop: &str,
    include_stop_str_in_output: bool,
) -> String {
    let (mut app, engine_task) = test_app_with_stream_output_specs(output_specs).await;

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "prompt": "hello",
                        "stream": true,
                        "stop": [stop],
                        "include_stop_str_in_output": include_stop_str_in_output
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    String::from_utf8(body.to_vec()).expect("utf8 body")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn stream_completions_stop_string_excluded_single_chunk() {
    // Engine emits "say world" in one decode step; stop "wor" truncates the
    // streamed text to "say " with finish_reason "stop".
    let text = stream_completion_with_stop(
        vec![(
            bytes_to_token_ids(b"say world"),
            Some(EngineCoreFinishReason::Length),
        )],
        "wor",
        false,
    )
    .await;

    assert_eq!(streamed_completion_text(&text), "say ");
    assert_eq!(streamed_finish_reason(&text).as_deref(), Some("stop"));
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn stream_completions_stop_string_included_single_chunk() {
    // Same single-chunk emission, but include_stop_str_in_output keeps "wor".
    let text = stream_completion_with_stop(
        vec![(
            bytes_to_token_ids(b"say world"),
            Some(EngineCoreFinishReason::Length),
        )],
        "wor",
        true,
    )
    .await;

    assert_eq!(streamed_completion_text(&text), "say wor");
    assert_eq!(streamed_finish_reason(&text).as_deref(), Some("stop"));
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn stream_completions_stop_string_excluded_split_across_chunks() {
    // Stop "wor" straddles two decode steps: "say wo" then "rld". Trimming must
    // span the chunk boundary, yielding "say " without dropping the SSE body.
    let text = stream_completion_with_stop(
        vec![
            (bytes_to_token_ids(b"say wo"), None),
            (
                bytes_to_token_ids(b"rld"),
                Some(EngineCoreFinishReason::Length),
            ),
        ],
        "wor",
        false,
    )
    .await;

    assert_eq!(streamed_completion_text(&text), "say ");
    assert_eq!(streamed_finish_reason(&text).as_deref(), Some("stop"));
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn stream_completions_stop_string_included_split_across_chunks() {
    // Same cross-boundary emission, but include_stop_str_in_output keeps "wor".
    let text = stream_completion_with_stop(
        vec![
            (bytes_to_token_ids(b"say wo"), None),
            (
                bytes_to_token_ids(b"rld"),
                Some(EngineCoreFinishReason::Length),
            ),
        ],
        "wor",
        true,
    )
    .await;

    assert_eq!(streamed_completion_text(&text), "say wor");
    assert_eq!(streamed_finish_reason(&text).as_deref(), Some("stop"));
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
}
