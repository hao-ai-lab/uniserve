#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeSet;
use std::fmt;
use std::sync::{Arc, Barrier};
use std::time::Duration;

use futures::StreamExt as _;
use tokio::time::timeout;
use uniserve_engine_gateway::EngineGateway;
use uniserve_engine_gateway::transport::EngineCoreClient;
use uniserve_engine_gateway::transport::protocol::generation::GenerationFinish;
use uniserve_engine_gateway::transport::protocol::logprobs::{
    Logprobs, MaybeWireLogprobs, PositionLogprobs, TokenLogprob,
};
use uniserve_engine_gateway::transport::protocol::{
    EngineCoreEvent, EngineCoreEventType, EngineCoreFinishReason, EngineCoreOutput,
    EngineCoreOutputs, StopReason,
};
use uniserve_engine_gateway::transport::test_utils::spawn_mock_engine_task;
use uniserve_serving::chat::{
    AssistantBlockKind, AssistantContentBlock, ChatBackend, ChatMessage, ChatRenderer, ChatRequest,
    ChatRole, ChatTextBackend, ChatTool, ChatToolChoice, DefaultChatOutputProcessor,
    DynChatOutputProcessor, DynChatRenderer, GenerationPromptMode, NewChatOutputProcessorOptions,
    ParserSelection, RenderedPrompt, SamplingParams,
};
use uniserve_serving::text::tokenizer::{DynTokenizer, Tokenizer};
use uniserve_serving::text::{
    DecodedLogprobs, DecodedPositionLogprobs, DecodedTokenLogprob, Prompt, TextBackend,
};
use uniserve_serving::{
    CandidateId, FinishStatus, RequestMetadata, ServeError, ServeEvent, ServeRequest,
    ServingRuntime,
};

const SPECIAL_STOP_TOKEN_ID: u32 = 256;

fn request_output(
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
        public_commit: None,
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
        new_logprobs: new_logprobs
            .map(normalize_decoded_logprobs)
            .map(MaybeWireLogprobs::Direct),
        new_prompt_logprobs_tensors: new_prompt_logprobs_tensors
            .map(normalize_decoded_logprobs)
            .map(MaybeWireLogprobs::Direct),
        pooling_output: None,
        finish_reason,
        stop_reason,
        events: None,
        public_commit: None,
        kv_transfer_params: None,
        trace_headers: None,
        prefill_stats: None,
        routed_experts: None,
        num_nans_in_logits: 0,
        generation: None,
    }
}

fn scheduling_output(request_id: &str) -> EngineCoreOutput {
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

fn canonicalize_outputs(
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

/// Apply the rank normalization defined by the decoded logprobs wire payload
/// (`PositionLogprobs::from_decoded_row`): the sampled
/// token (index 0) keeps its rank, while every alternative candidate takes its
/// 1-based position index as its rank. The out-of-process ZMQ mock got this for
/// free via msgpack encode/decode; the in-process mock hands `Direct` logprobs
/// straight through, so the scripting helper applies the protocol rule here.
fn normalize_decoded_logprobs(mut logprobs: Logprobs) -> Logprobs {
    for position in &mut logprobs.positions {
        for (index, entry) in position.entries.iter_mut().enumerate() {
            if index != 0 {
                entry.rank = index as u32;
            }
        }
    }
    logprobs
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
                    rank: 2,
                },
            ],
        }],
    }
}

fn prompt_logprobs_for_tokens(prompt_token_ids: &[u32]) -> Logprobs {
    Logprobs {
        positions: prompt_token_ids
            .iter()
            .copied()
            .skip(1)
            .map(|token_id| PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id,
                        logprob: -0.3,
                        rank: 1,
                    },
                    TokenLogprob {
                        token_id: token_id ^ 1,
                        logprob: -0.4,
                        rank: 2,
                    },
                ],
            })
            .collect(),
    }
}

fn bytes_to_token_ids(bytes: &[u8]) -> Vec<u32> {
    bytes.iter().map(|byte| u32::from(*byte)).collect()
}

fn serving_runtime_from_mock_client(
    client: EngineCoreClient,
    backend: Arc<dyn ChatTextBackend>,
) -> ServingRuntime {
    ServingRuntime::from_shared_backend(EngineGateway::new(client), backend)
}

#[derive(Clone)]
struct FakeChatBackend {
    has_template: bool,
    model_id: String,
}

#[derive(Debug)]
struct FakeChatTokenizer;

impl Tokenizer for FakeChatTokenizer {
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
        skip_special_tokens: bool,
    ) -> uniserve_model_profile::tokenizer::Result<String> {
        let bytes = token_ids
            .iter()
            .filter_map(|id| {
                if skip_special_tokens && *id == SPECIAL_STOP_TOKEN_ID {
                    None
                } else {
                    Some(*id as u8)
                }
            })
            .collect::<Vec<_>>();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match token {
            "<think>" => Some(0xF001),
            "</think>" => Some(0xF002),
            "<|START_THINKING|>" => Some(0xF003),
            "<|END_THINKING|>" => Some(0xF004),
            "◁think▷" => Some(0xF005),
            "◁/think▷" => Some(0xF006),
            "<|im_start|>" => Some(0xF007),
            "<|im_end|>" => Some(0xF008),
            "<|vision_start|>" => Some(0xF009),
            "<|vision_end|>" => Some(0xF00A),
            _ => None,
        }
    }
}

impl fmt::Debug for FakeChatBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FakeChatBackend").finish_non_exhaustive()
    }
}

impl FakeChatBackend {
    fn new() -> Self {
        Self {
            has_template: true,
            model_id: "test-model".to_string(),
        }
    }

    fn without_template() -> Self {
        Self {
            has_template: false,
            model_id: "test-model".to_string(),
        }
    }

    fn with_model_id(model_id: impl Into<String>) -> Self {
        Self {
            has_template: true,
            model_id: model_id.into(),
        }
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
    ) -> uniserve_serving::chat::template::Result<RenderedPrompt> {
        if !self.has_template {
            return Err(uniserve_serving::chat::template::Error::MissingChatTemplate);
        }

        let mut prompt = String::new();
        for message in &request.messages {
            prompt.push_str(message.role().as_str());
            prompt.push_str(": ");
            prompt.push_str(&message.text_content()?);
            prompt.push('\n');
        }
        if request.chat_options.add_generation_prompt() {
            prompt.push_str("assistant:");
        }

        Ok(RenderedPrompt {
            prompt: Prompt::Text(prompt),
        })
    }
}

#[derive(Clone)]
struct BlockingChatBackend {
    inner: FakeChatBackend,
    entered: Arc<Barrier>,
    release: Arc<Barrier>,
}

impl fmt::Debug for BlockingChatBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BlockingChatBackend")
            .finish_non_exhaustive()
    }
}

impl TextBackend for BlockingChatBackend {
    fn tokenizer(&self) -> DynTokenizer {
        self.inner.tokenizer()
    }

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
}

impl ChatBackend for BlockingChatBackend {
    fn chat_renderer(&self) -> DynChatRenderer {
        Arc::new(self.clone())
    }

    fn new_chat_output_processor(
        &self,
        request: &mut ChatRequest,
        options: NewChatOutputProcessorOptions<'_>,
    ) -> uniserve_serving::chat::Result<DynChatOutputProcessor> {
        self.inner.new_chat_output_processor(request, options)
    }
}

impl ChatRenderer for BlockingChatBackend {
    fn render(
        &self,
        request: &ChatRequest,
    ) -> uniserve_serving::chat::template::Result<RenderedPrompt> {
        self.entered.wait();
        self.release.wait();
        self.inner.render(request)
    }
}

#[derive(Clone)]
struct BlockingTokenizerBackend {
    inner: FakeChatBackend,
    entered: Arc<Barrier>,
    release: Arc<Barrier>,
}

impl fmt::Debug for BlockingTokenizerBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BlockingTokenizerBackend")
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct BlockingTokenizer {
    entered: Arc<Barrier>,
    release: Arc<Barrier>,
}

impl Tokenizer for BlockingTokenizer {
    fn encode(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> uniserve_model_profile::tokenizer::Result<Vec<u32>> {
        self.entered.wait();
        self.release.wait();
        FakeChatTokenizer.encode(text, add_special_tokens)
    }

    fn decode(
        &self,
        token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> uniserve_model_profile::tokenizer::Result<String> {
        FakeChatTokenizer.decode(token_ids, skip_special_tokens)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        FakeChatTokenizer.token_to_id(token)
    }
}

impl TextBackend for BlockingTokenizerBackend {
    fn tokenizer(&self) -> DynTokenizer {
        Arc::new(BlockingTokenizer {
            entered: Arc::clone(&self.entered),
            release: Arc::clone(&self.release),
        })
    }

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
}

impl ChatBackend for BlockingTokenizerBackend {
    fn chat_renderer(&self) -> DynChatRenderer {
        self.inner.chat_renderer()
    }

    fn new_chat_output_processor(
        &self,
        request: &mut ChatRequest,
        options: NewChatOutputProcessorOptions<'_>,
    ) -> uniserve_serving::chat::Result<DynChatOutputProcessor> {
        self.inner.new_chat_output_processor(request, options)
    }
}

#[derive(Clone, Debug)]
struct FailingDecodeBackend {
    inner: FakeChatBackend,
}

#[derive(Debug)]
struct FailingDecodeTokenizer;

impl Tokenizer for FailingDecodeTokenizer {
    fn encode(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> uniserve_model_profile::tokenizer::Result<Vec<u32>> {
        FakeChatTokenizer.encode(text, add_special_tokens)
    }

    fn decode(
        &self,
        token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> uniserve_model_profile::tokenizer::Result<String> {
        if token_ids.contains(&(b'i' as u32)) {
            return Err(uniserve_model_profile::tokenizer::TokenizerError(
                "decode failed".to_string(),
            ));
        }
        FakeChatTokenizer.decode(token_ids, skip_special_tokens)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        FakeChatTokenizer.token_to_id(token)
    }
}

impl TextBackend for FailingDecodeBackend {
    fn tokenizer(&self) -> DynTokenizer {
        Arc::new(FailingDecodeTokenizer)
    }

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
}

impl ChatBackend for FailingDecodeBackend {
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

impl ChatRenderer for FailingDecodeBackend {
    fn render(
        &self,
        request: &ChatRequest,
    ) -> uniserve_serving::chat::template::Result<RenderedPrompt> {
        self.inner.render(request)
    }
}

/// Skip `LogprobsDelta` events that carry only token_ids (no logprobs),
/// returning the next semantically interesting event.
async fn next_semantic<S>(stream: &mut S) -> Option<Result<ServeEvent, ServeError>>
where
    S: futures::Stream<Item = Result<ServeEvent, ServeError>> + Unpin,
{
    loop {
        match stream.next().await {
            Some(Ok(ServeEvent::TextDelta {
                text,
                logprobs: None,
                ..
            })) if text.is_empty() => continue,
            Some(Ok(ServeEvent::Usage { .. })) => continue,
            Some(Ok(ServeEvent::Scheduled {
                request_id,
                resources,
                ..
            })) => {
                assert!(!request_id.is_empty());
                assert!(resources.expected_kv_tokens > 0);
                continue;
            }
            other => return other,
        }
    }
}

fn canonical_chat_request(request: ChatRequest) -> ServeRequest {
    ServeRequest::from_chat_request(request, RequestMetadata::default()).unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_during_compilation_prevents_engine_submission() {
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let backend: Arc<dyn ChatTextBackend> = Arc::new(BlockingChatBackend {
        inner: FakeChatBackend::new(),
        entered: Arc::clone(&entered),
        release: Arc::clone(&release),
    });
    let (client, mut mock) = EngineCoreClient::connect_mock("test-model");
    let runtime = Arc::new(serving_runtime_from_mock_client(client, backend));
    let mut request = ChatRequest::for_test();
    request.request_id = "compile-cancel".to_string();
    let serve_request = canonical_chat_request(request);
    let serving_runtime = Arc::clone(&runtime);
    let mut serve_task = tokio::spawn(async move { serving_runtime.serve(serve_request).await });

    tokio::task::spawn_blocking(move || entered.wait())
        .await
        .expect("wait for renderer");
    runtime
        .cancel("compile-cancel".to_string())
        .await
        .expect("cancel compiling request");
    let serve_result = timeout(Duration::from_secs(1), &mut serve_task).await;
    tokio::task::spawn_blocking(move || release.wait())
        .await
        .expect("release renderer");
    let mut stream = serve_result
        .expect("serve must observe compile cancellation")
        .expect("serve task")
        .expect("cancelled stream");

    assert!(matches!(
        stream.next().await,
        Some(Ok(ServeEvent::Cancelled { request_id })) if request_id.as_ref() == "compile-cancel"
    ));
    assert!(
        timeout(Duration::from_millis(100), mock.recv())
            .await
            .is_err()
    );
}

#[tokio::test(flavor = "current_thread")]
async fn raw_compilation_does_not_block_the_async_executor() {
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let backend: Arc<dyn ChatTextBackend> = Arc::new(BlockingTokenizerBackend {
        inner: FakeChatBackend::new(),
        entered: Arc::clone(&entered),
        release: Arc::clone(&release),
    });
    let (client, _mock) = EngineCoreClient::connect_mock("test-model");
    let runtime = Arc::new(serving_runtime_from_mock_client(client, backend));
    let compile_runtime = Arc::clone(&runtime);
    let compile_task = tokio::spawn(async move {
        compile_runtime
            .compile_async(ServeRequest::text("blocking-tokenizer", "hello"))
            .await
    });

    tokio::task::spawn_blocking(move || entered.wait())
        .await
        .expect("wait for tokenizer");
    let heartbeat = tokio::spawn(async {
        tokio::task::yield_now().await;
        7
    });
    assert_eq!(
        timeout(Duration::from_secs(1), heartbeat)
            .await
            .expect("executor heartbeat must not stall")
            .expect("heartbeat task"),
        7
    );

    tokio::task::spawn_blocking(move || release.wait())
        .await
        .expect("release tokenizer");
    compile_task
        .await
        .expect("compile task")
        .expect("compile request");
}

async fn serve_chat(
    runtime: &ServingRuntime,
    request: ChatRequest,
) -> Result<uniserve_serving::ServeEventStream, ServeError> {
    runtime.serve(canonical_chat_request(request)).await
}

#[derive(Debug)]
struct TerminalEvents {
    prompt_tokens: u32,
    visible_output_tokens: u32,
    internal_tokens: u32,
    reason: FinishStatus,
}

async fn next_terminal(stream: &mut uniserve_serving::ServeEventStream) -> TerminalEvents {
    let mut usage = None;
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            ServeEvent::Usage {
                prompt_tokens,
                visible_output_tokens,
                internal_tokens,
                ..
            } => usage = Some((prompt_tokens, visible_output_tokens, internal_tokens)),
            ServeEvent::Finished { reason, .. } => {
                let (prompt_tokens, visible_output_tokens, internal_tokens) =
                    usage.expect("usage before terminal");
                return TerminalEvents {
                    prompt_tokens,
                    visible_output_tokens,
                    internal_tokens,
                    reason,
                };
            }
            other => panic!("unexpected event before terminal: {other:?}"),
        }
    }
    panic!("stream ended without terminal events")
}

async fn collect_events(mut stream: uniserve_serving::ServeEventStream) -> Vec<ServeEvent> {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.unwrap());
    }
    events
}

fn visible_text(events: &[ServeEvent]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            ServeEvent::TextDelta { text, .. } if !text.is_empty() => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

fn reasoning_text(events: &[ServeEvent]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            ServeEvent::ReasoningDelta { text, .. } if !text.is_empty() => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

fn sample_request(request_id: &str) -> ChatRequest {
    ChatRequest {
        messages: vec![
            ChatMessage::text(ChatRole::System, "You are terse."),
            ChatMessage::text(ChatRole::User, "Say hi"),
        ],
        sampling_params: SamplingParams {
            max_tokens: Some(8),
            ..Default::default()
        },
        request_id: request_id.to_string(),
        ..ChatRequest::for_test()
    }
}

fn sample_tool_request(request_id: &str) -> ChatRequest {
    let mut request = sample_request(request_id);
    request.sampling_params.stop_token_ids = Some(vec![SPECIAL_STOP_TOKEN_ID]);
    request.tools = vec![ChatTool {
        name: "get_weather".to_string(),
        description: Some("Get weather".to_string()),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }),
        strict: None,
    }];
    request.tool_choice = ChatToolChoice::Auto;
    request
}

#[tokio::test]
async fn serving_plan_inspection_records_profile_policy_resources_and_origins() {
    let (client, _mock) = EngineCoreClient::connect_mock("test-model");
    let backend: Arc<dyn ChatTextBackend> = Arc::new(FakeChatBackend::new());
    let runtime = serving_runtime_from_mock_client(client, backend);

    let plan = runtime
        .compile_async(canonical_chat_request(sample_request("plan-chat")))
        .await
        .unwrap();
    let inspection = plan.inspect();

    assert_eq!(inspection.request_id.as_ref(), "plan-chat");
    assert_eq!(
        inspection.generation.candidate_ids,
        vec![CandidateId::PRIMARY]
    );
    assert_eq!(
        inspection.generation.constraint,
        uniserve_core::GenerationConstraint::UndOnly
    );
    assert_eq!(
        inspection.resources.prompt_tokens,
        inspection.context_prompt_tokens()
    );
    assert_eq!(inspection.resources.adapter_slots, 0);
    assert!(inspection.cache.read_enabled);
    assert!(inspection.cache.write_enabled);
    assert_eq!(inspection.rendered_segments.len(), 2);
    assert!(matches!(
        inspection.rendered_segments[0].origin,
        uniserve_serving::RenderedSegmentOrigin::ChatMessage {
            role: uniserve_serving::ContextRole::System,
            ..
        }
    ));
    assert_eq!(inspection.output.reasoning_parser, "auto");
    assert_eq!(inspection.output.tool_parser, "auto");
}

#[tokio::test]
async fn multimodal_plan_inspection_records_rendered_scaffold_and_image_placement() {
    let (client, _mock) = EngineCoreClient::connect_mock("test-model");
    let backend: Arc<dyn ChatTextBackend> = Arc::new(FakeChatBackend::new());
    let mut dialect = uniserve_model_profile::dialect::resolve_generation_dialect_for_model(
        "bagel",
        &FakeChatTokenizer,
    )
    .unwrap()
    .unwrap();
    dialect.image_ingest.step_kv_tokens = vec![
        uniserve_core::ImageKvEffect::Exact { tokens: 8 },
        uniserve_core::ImageKvEffect::Exact { tokens: 8 },
    ];
    dialect
        .generation_policy
        .feedback
        .as_mut()
        .unwrap()
        .ingest
        .step_kv_tokens = vec![uniserve_core::ImageKvEffect::Exact { tokens: 8 }];
    let runtime =
        serving_runtime_from_mock_client(client, backend).with_generation_dialect(dialect);
    let mut request = ServeRequest::chat(
        "multimodal-plan",
        vec![ChatMessage::user(
            uniserve_serving::chat::ChatContent::Parts(vec![
                uniserve_serving::chat::ChatContentPart::Text {
                    text: "Continue from this image".to_string(),
                },
                uniserve_serving::chat::ChatContentPart::ImageUrl {
                    image_url: "data:image/png;base64,aW1hZ2U=".to_string(),
                    detail: None,
                    uuid: None,
                },
            ]),
        )],
    );
    request.modalities.input_image = true;
    request.modalities.output_image = true;
    request.generation.constraint = uniserve_core::GenerationConstraint::Default;
    request.generation.max_tokens = Some(8);
    request.generation.image.max_images = Some(1);

    let plan = runtime.compile_async(request).await.unwrap();
    let inspection = plan.inspect();
    assert!(matches!(
        inspection
            .rendered_segments
            .first()
            .map(|segment| &segment.origin),
        Some(uniserve_serving::RenderedSegmentOrigin::ProfileSystem)
    ));
    assert!(inspection.rendered_segments.iter().any(|segment| matches!(
        segment.origin,
        uniserve_serving::RenderedSegmentOrigin::ChatMessage {
            index: 0,
            role: uniserve_serving::ContextRole::User,
        }
    )));
    assert!(inspection.rendered_segments.iter().any(|segment| matches!(
        segment,
        uniserve_serving::RenderedSegmentInspection {
            origin: uniserve_serving::RenderedSegmentOrigin::Image { index: 0 },
            token_start: Some(_),
            ..
        }
    )));
    assert!(matches!(
        inspection.context,
        uniserve_serving::PlannedContextInspection::Chat {
            message_count: 2,
            multimodal: true,
            ..
        }
    ));
}

#[tokio::test]
async fn plan_cache_inspection_matches_lowered_prompt_scoring_policy() {
    let (client, _mock) = EngineCoreClient::connect_mock("test-model");
    let runtime = serving_runtime_from_mock_client(client, Arc::new(FakeChatBackend::new()));
    let mut chat = sample_request("prompt-cache-plan");
    chat.sampling_params.prompt_logprobs = Some(1);
    let mut request = canonical_chat_request(chat);
    request.cache.namespace = Some("ab".to_string());
    request.cache.salt = Some("c".to_string());

    let first = runtime.compile_async(request.clone()).await.unwrap();
    let second = runtime.compile_async(request).await.unwrap();
    assert!(!first.inspect().cache.read_enabled);
    assert!(first.inspect().cache.write_enabled);
    assert_eq!(
        first.inspect().cache.key_fingerprint,
        second.inspect().cache.key_fingerprint
    );

    let mut distinct = canonical_chat_request(sample_request("distinct-cache-plan"));
    distinct.cache.namespace = Some("a".to_string());
    distinct.cache.salt = Some("bc".to_string());
    let distinct = runtime.compile_async(distinct).await.unwrap();
    assert_ne!(
        first.inspect().cache.key_fingerprint,
        distinct.inspect().cache.key_fingerprint
    );
}

#[tokio::test]
async fn serving_plan_inspection_is_deterministic_for_identical_inputs() {
    let (client, _mock) = EngineCoreClient::connect_mock("test-model");
    let runtime = serving_runtime_from_mock_client(client, Arc::new(FakeChatBackend::new()));
    let request = canonical_chat_request(sample_request("deterministic-plan"));

    let first = runtime.compile_async(request.clone()).await.unwrap();
    let second = runtime.compile_async(request).await.unwrap();

    assert_eq!(first.inspect(), second.inspect());
}

#[test]
fn synchronous_compile_supports_chat_plan_inspection_without_an_async_runtime() {
    let setup_runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (client, _mock) = {
        let _guard = setup_runtime.enter();
        EngineCoreClient::connect_mock("test-model")
    };
    drop(setup_runtime);
    let backend: Arc<dyn ChatTextBackend> = Arc::new(FakeChatBackend::new());
    let runtime = serving_runtime_from_mock_client(client, backend);

    let plan = runtime
        .compile(canonical_chat_request(sample_request("sync-plan-chat")))
        .unwrap();

    assert_eq!(plan.inspect().request_id.as_ref(), "sync-plan-chat");
    assert!(matches!(
        plan.inspect().context,
        uniserve_serving::PlannedContextInspection::Chat { .. }
    ));
}

#[tokio::test]
async fn synchronous_compile_is_safe_inside_an_async_host() {
    let (client, _mock) = EngineCoreClient::connect_mock("test-model");
    let backend: Arc<dyn ChatTextBackend> = Arc::new(FakeChatBackend::new());
    let runtime = serving_runtime_from_mock_client(client, backend);

    let plan = runtime
        .compile(canonical_chat_request(sample_request("nested-sync-plan")))
        .unwrap();

    assert_eq!(plan.inspect().request_id.as_ref(), "nested-sync-plan");
}

#[tokio::test]
async fn execution_plans_are_runtime_local() {
    let (client_a, _mock_a) = EngineCoreClient::connect_mock("test-model");
    let (client_b, _mock_b) = EngineCoreClient::connect_mock("test-model");
    let runtime_a = serving_runtime_from_mock_client(client_a, Arc::new(FakeChatBackend::new()));
    let runtime_b = serving_runtime_from_mock_client(client_b, Arc::new(FakeChatBackend::new()));
    let plan = runtime_a
        .compile_async(canonical_chat_request(sample_request("foreign-plan")))
        .await
        .unwrap();

    let error = match Box::pin(runtime_b.execute(plan)).await {
        Ok(_) => panic!("foreign plan must be rejected before execution"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        ServeError::ForeignExecutionPlan { request_id } if request_id.as_ref() == "foreign-plan"
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_streams_text_events() {
    let (client, mock) = EngineCoreClient::connect_mock("test-model");
    let (shutdown_tx, engine_task) = spawn_mock_engine_task(mock, |mut mock| {
        Box::pin(async move {
            let request = mock.recv_request().await;
            assert_eq!(request.request_id, "chat-1");
            assert_eq!(
                String::from_utf8(
                    request
                        .generation
                        .prompt_token_ids()
                        .into_iter()
                        .map(|id| id as u8)
                        .collect()
                )
                .unwrap(),
                "system: You are terse.\nuser: Say hi\nassistant:"
            );
            mock.send_outputs(canonicalize_outputs(
                request.generation.prompt_token_ids().len(),
                EngineCoreOutputs {
                    outputs: vec![
                        scheduling_output("chat-1"),
                        request_output("chat-1", vec![b'H' as u32], None, None),
                        request_output("chat-1", vec![b'i' as u32], None, None),
                        request_output(
                            "chat-1",
                            Vec::new(),
                            Some(EngineCoreFinishReason::Stop),
                            Some(StopReason::TokenId(b'!' as u32)),
                        ),
                    ],
                    finished_requests: Some(BTreeSet::from(["chat-1".to_string()])),
                    ..Default::default()
                },
            ));
        })
    });

    let backend: Arc<dyn ChatTextBackend> = Arc::new(FakeChatBackend::new());
    let runtime = serving_runtime_from_mock_client(client, backend);

    let mut request = sample_request("chat-1");
    request.sampling_params.stop_token_ids = Some(vec![b'!' as u32]);
    let mut stream = serve_chat(&runtime, request).await.unwrap();

    match next_semantic(&mut stream).await.unwrap().unwrap() {
        ServeEvent::Accepted {
            prompt_token_ids,
            prompt_logprobs: None,
            ..
        } => {
            assert_eq!(
                prompt_token_ids.len(),
                "system: You are terse.\nuser: Say hi\nassistant:".len()
            );
            assert!(!prompt_token_ids.is_empty());
        }
        other => panic!("expected Start, got {other:?}"),
    }
    assert!(matches!(
        stream.next().await,
        Some(Ok(ServeEvent::Scheduled {
            request_id,
            resources,
            ..
        })) if request_id.as_ref() == "chat-1" && resources.expected_kv_tokens > 0
    ));
    assert_eq!(
        next_semantic(&mut stream).await.unwrap().unwrap(),
        ServeEvent::OutputBlockStart {
            candidate_id: CandidateId::PRIMARY,
            index: 0,
            kind: AssistantBlockKind::Text
        }
    );
    assert_eq!(
        next_semantic(&mut stream).await.unwrap().unwrap(),
        ServeEvent::TextDelta {
            candidate_id: CandidateId::PRIMARY,
            text: "H".to_string(),
            token_ids: Vec::new(),
            logprobs: None
        }
    );
    assert_eq!(
        next_semantic(&mut stream).await.unwrap().unwrap(),
        ServeEvent::TextDelta {
            candidate_id: CandidateId::PRIMARY,
            text: "i".to_string(),
            token_ids: Vec::new(),
            logprobs: None
        }
    );
    assert_eq!(
        next_semantic(&mut stream).await.unwrap().unwrap(),
        ServeEvent::OutputBlockEnd {
            candidate_id: CandidateId::PRIMARY,
            index: 0,
            block: AssistantContentBlock::Text {
                text: "Hi".to_string(),
            }
        }
    );

    let terminal = next_terminal(&mut stream).await;
    assert_eq!(
        terminal.prompt_tokens as usize,
        "system: You are terse.\nuser: Say hi\nassistant:".len()
    );
    assert_eq!(terminal.visible_output_tokens, 2);
    assert_eq!(terminal.internal_tokens, 1);
    assert!(matches!(terminal.reason, FinishStatus::Stop { .. }));
    assert!(next_semantic(&mut stream).await.is_none());

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_stream_waits_for_complete_utf8_before_emitting() {
    let (client, mock) = EngineCoreClient::connect_mock("test-model");
    let (shutdown_tx, engine_task) = spawn_mock_engine_task(mock, |mut mock| {
        Box::pin(async move {
            let request = mock.recv_request().await;
            mock.send_outputs(canonicalize_outputs(
                request.generation.prompt_token_ids().len(),
                EngineCoreOutputs {
                    outputs: vec![
                        scheduling_output("chat-utf8"),
                        request_output("chat-utf8", bytes_to_token_ids(&[0xe4]), None, None),
                        request_output("chat-utf8", bytes_to_token_ids(&[0xbd, 0xa0]), None, None),
                        request_output(
                            "chat-utf8",
                            Vec::new(),
                            Some(EngineCoreFinishReason::Stop),
                            Some(StopReason::TokenId(b'!' as u32)),
                        ),
                    ],
                    finished_requests: Some(BTreeSet::from(["chat-utf8".to_string()])),
                    ..Default::default()
                },
            ));
        })
    });

    let backend: Arc<dyn ChatTextBackend> = Arc::new(FakeChatBackend::new());
    let runtime = serving_runtime_from_mock_client(client, backend);

    let mut request = sample_request("chat-utf8");
    request.sampling_params.stop_token_ids = Some(vec![b'!' as u32]);
    let mut stream = serve_chat(&runtime, request).await.unwrap();

    assert!(matches!(
        next_semantic(&mut stream).await,
        Some(Ok(ServeEvent::Accepted {
            prompt_logprobs: None,
            ..
        }))
    ));
    assert_eq!(
        next_semantic(&mut stream).await.unwrap().unwrap(),
        ServeEvent::OutputBlockStart {
            candidate_id: CandidateId::PRIMARY,
            index: 0,
            kind: AssistantBlockKind::Text
        }
    );
    assert_eq!(
        next_semantic(&mut stream).await.unwrap().unwrap(),
        ServeEvent::TextDelta {
            candidate_id: CandidateId::PRIMARY,
            text: "你".to_string(),
            token_ids: Vec::new(),
            logprobs: None
        }
    );
    assert_eq!(
        next_semantic(&mut stream).await.unwrap().unwrap(),
        ServeEvent::OutputBlockEnd {
            candidate_id: CandidateId::PRIMARY,
            index: 0,
            block: AssistantContentBlock::Text {
                text: "你".to_string(),
            }
        }
    );

    let terminal = next_terminal(&mut stream).await;
    assert_eq!(terminal.visible_output_tokens, 3);
    assert_eq!(terminal.internal_tokens, 1);

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_stream_flushes_held_text_on_finish() {
    let (client, mock) = EngineCoreClient::connect_mock("test-model");
    let (shutdown_tx, engine_task) = spawn_mock_engine_task(mock, |mut mock| {
        Box::pin(async move {
            let request = mock.recv_request().await;
            mock.send_outputs(canonicalize_outputs(
                request.generation.prompt_token_ids().len(),
                EngineCoreOutputs {
                    outputs: vec![
                        scheduling_output("chat-final-flush"),
                        request_output(
                            "chat-final-flush",
                            bytes_to_token_ids(b"ok st"),
                            Some(EngineCoreFinishReason::Length),
                            None,
                        ),
                    ],
                    finished_requests: Some(BTreeSet::from(["chat-final-flush".to_string()])),
                    ..Default::default()
                },
            ));
        })
    });

    let backend: Arc<dyn ChatTextBackend> = Arc::new(FakeChatBackend::new());
    let runtime = serving_runtime_from_mock_client(client, backend);

    let events = collect_events(
        serve_chat(&runtime, sample_request("chat-final-flush"))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(visible_text(&events), "ok st");
    assert!(events.iter().any(|event| matches!(
        event,
        ServeEvent::OutputBlockEnd {
            block: AssistantContentBlock::Text { text },
            ..
        } if text == "ok st"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ServeEvent::Usage {
            visible_output_tokens: 5,
            ..
        }
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ServeEvent::Finished {
            reason: FinishStatus::Length,
            ..
        }
    )));

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    runtime.shutdown().await.unwrap();
}

#[test]
fn chat_request_rejects_conflicting_generation_modes() {
    let mut request = sample_request("chat-2");
    request.chat_options.generation_prompt_mode = GenerationPromptMode::ContinueFinalAssistant;
    let error = request.validate().unwrap_err();

    assert!(matches!(
        error,
        uniserve_serving::chat::protocol::Error::ContinueFinalAssistantWithoutFinalAssistant
    ));
}

#[test]
fn chat_request_accepts_continue_final_assistant_mode_with_final_assistant() {
    let mut request = sample_request("chat-2b");
    request.messages = vec![ChatMessage::assistant_text("hello")];
    request.chat_options.generation_prompt_mode = GenerationPromptMode::ContinueFinalAssistant;

    request.validate().unwrap();
}

#[test]
fn backend_requires_a_template() {
    let request = sample_request("chat-3");
    let backend = FakeChatBackend::without_template();
    let error = backend.chat_renderer().render(&request).unwrap_err();
    assert!(matches!(
        error,
        uniserve_serving::chat::template::Error::MissingChatTemplate
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_stream_reports_decode_failure_as_error_event() {
    let (client, mock) = EngineCoreClient::connect_mock("test-model");
    let (shutdown_tx, engine_task) = spawn_mock_engine_task(mock, |mut mock| {
        Box::pin(async move {
            let request = mock.recv_request().await;
            mock.send_outputs(canonicalize_outputs(
                request.generation.prompt_token_ids().len(),
                EngineCoreOutputs {
                    outputs: vec![
                        scheduling_output("chat-4"),
                        request_output("chat-4", vec![b'i' as u32], None, None),
                    ],
                    ..Default::default()
                },
            ));
        })
    });

    let backend: Arc<dyn ChatTextBackend> = Arc::new(FailingDecodeBackend {
        inner: FakeChatBackend::new(),
    });
    let runtime = serving_runtime_from_mock_client(client, backend);

    let mut stream = serve_chat(&runtime, sample_request("chat-4"))
        .await
        .unwrap();
    assert!(matches!(
        next_semantic(&mut stream).await,
        Some(Ok(ServeEvent::Accepted {
            prompt_logprobs: None,
            ..
        }))
    ));

    match timeout(Duration::from_secs(2), next_semantic(&mut stream))
        .await
        .unwrap()
    {
        Some(Err(ServeError::Chat(uniserve_serving::chat::Error::Text(
            uniserve_serving::text::Error::Tokenizer(message),
        )))) => {
            assert_eq!(message, "decode failed");
        }
        other => panic!("unexpected event after close: {other:?}"),
    }

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_stream_preserves_terminal_stop_token_when_requested() {
    let (client, mock) = EngineCoreClient::connect_mock("test-model");
    let (shutdown_tx, engine_task) = spawn_mock_engine_task(mock, |mut mock| {
        Box::pin(async move {
            let request = mock.recv_request().await;
            mock.send_outputs(canonicalize_outputs(
                request.generation.prompt_token_ids().len(),
                EngineCoreOutputs {
                    outputs: vec![
                        scheduling_output("chat-include-stop"),
                        request_output(
                            "chat-include-stop",
                            vec![b'H' as u32, b'i' as u32, b'!' as u32],
                            Some(EngineCoreFinishReason::Stop),
                            Some(StopReason::TokenId(b'!' as u32)),
                        ),
                    ],
                    finished_requests: Some(BTreeSet::from(["chat-include-stop".to_string()])),
                    ..Default::default()
                },
            ));
        })
    });

    let backend: Arc<dyn ChatTextBackend> = Arc::new(FakeChatBackend::new());
    let runtime = serving_runtime_from_mock_client(client, backend);

    let mut request = sample_request("chat-include-stop");
    request.sampling_params.stop_token_ids = Some(vec![b'!' as u32]);
    request.decode_options.include_stop_str_in_output = true;
    let events = collect_events(serve_chat(&runtime, request).await.unwrap()).await;
    assert_eq!(visible_text(&events), "Hi!");
    assert!(events.iter().any(|event| matches!(
        event,
        ServeEvent::OutputBlockEnd {
            block: AssistantContentBlock::Text { text },
            ..
        } if text == "Hi!"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ServeEvent::Usage {
            visible_output_tokens: 3,
            ..
        }
    )));

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_stream_separates_reasoning_blocks_automatically() {
    let (client, mock) = EngineCoreClient::connect_mock("test-model");
    let (shutdown_tx, engine_task) = spawn_mock_engine_task(mock, |mut mock| {
        Box::pin(async move {
            let request = mock.recv_request().await;
            mock.send_outputs(canonicalize_outputs(
                request.generation.prompt_token_ids().len(),
                EngineCoreOutputs {
                    outputs: vec![
                        scheduling_output("chat-reasoning"),
                        request_output(
                            "chat-reasoning",
                            bytes_to_token_ids(b"<think>"),
                            None,
                            None,
                        ),
                        request_output(
                            "chat-reasoning",
                            bytes_to_token_ids(b"reason "),
                            None,
                            None,
                        ),
                        request_output(
                            "chat-reasoning",
                            bytes_to_token_ids(b"more</think>"),
                            None,
                            None,
                        ),
                        request_output(
                            "chat-reasoning",
                            bytes_to_token_ids(b"answer"),
                            Some(EngineCoreFinishReason::Length),
                            None,
                        ),
                    ],
                    finished_requests: Some(BTreeSet::from(["chat-reasoning".to_string()])),
                    ..Default::default()
                },
            ));
        })
    });

    let backend: Arc<dyn ChatTextBackend> =
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B"));
    let runtime = serving_runtime_from_mock_client(client, backend);

    let events = collect_events(
        serve_chat(&runtime, sample_request("chat-reasoning"))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(reasoning_text(&events), "reason more");
    assert_eq!(visible_text(&events), "answer");
    assert!(events.iter().any(|event| matches!(
        event,
        ServeEvent::OutputBlockEnd {
            block: AssistantContentBlock::Reasoning { text },
            ..
        } if text == "reason more"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ServeEvent::OutputBlockEnd {
            block: AssistantContentBlock::Text { text },
            ..
        } if text == "answer"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ServeEvent::Finished {
            reason: FinishStatus::Length,
            ..
        }
    )));

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_collectors_return_structured_message_and_visible_text() {
    let (client, mock) = EngineCoreClient::connect_mock("test-model");
    let (shutdown_tx, engine_task) = spawn_mock_engine_task(mock, |mut mock| {
        Box::pin(async move {
            let request = mock.recv_request().await;
            mock.send_outputs(canonicalize_outputs(
                request.generation.prompt_token_ids().len(),
                EngineCoreOutputs {
                    outputs: vec![
                        scheduling_output("chat-collect"),
                        request_output(
                            "chat-collect",
                            bytes_to_token_ids(b"<think>inner</think>outer"),
                            Some(EngineCoreFinishReason::Length),
                            None,
                        ),
                    ],
                    finished_requests: Some(BTreeSet::from(["chat-collect".to_string()])),
                    ..Default::default()
                },
            ));
        })
    });

    let backend: Arc<dyn ChatTextBackend> =
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B"));
    let runtime = serving_runtime_from_mock_client(client, backend);

    let events = collect_events(
        serve_chat(&runtime, sample_request("chat-collect"))
            .await
            .unwrap(),
    )
    .await;
    assert!(events.iter().any(|event| matches!(
        event,
        ServeEvent::OutputBlockEnd {
            block: AssistantContentBlock::Reasoning { text },
            ..
        } if text == "inner"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ServeEvent::OutputBlockEnd {
            block: AssistantContentBlock::Text { text },
            ..
        } if text == "outer"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ServeEvent::Usage {
            prompt_tokens,
            visible_output_tokens,
            ..
        } if *prompt_tokens as usize == "system: You are terse.\nuser: Say hi\nassistant:".len()
            && *visible_output_tokens as usize == "<think>inner</think>outer".len()
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ServeEvent::Finished {
            reason: FinishStatus::Length,
            ..
        }
    )));

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_explicitly_disables_reasoning_parser() {
    let (client, mock) = EngineCoreClient::connect_mock("test-model");
    let (shutdown_tx, engine_task) = spawn_mock_engine_task(mock, |mut mock| {
        Box::pin(async move {
            let request = mock.recv_request().await;
            mock.send_outputs(canonicalize_outputs(
                request.generation.prompt_token_ids().len(),
                EngineCoreOutputs {
                    outputs: vec![
                        scheduling_output("chat-reasoning-disabled"),
                        request_output(
                            "chat-reasoning-disabled",
                            bytes_to_token_ids(b"<think>"),
                            None,
                            None,
                        ),
                        request_output(
                            "chat-reasoning-disabled",
                            bytes_to_token_ids(b"reason "),
                            None,
                            None,
                        ),
                        request_output(
                            "chat-reasoning-disabled",
                            bytes_to_token_ids(b"more</think>"),
                            None,
                            None,
                        ),
                        request_output(
                            "chat-reasoning-disabled",
                            bytes_to_token_ids(b"answer"),
                            Some(EngineCoreFinishReason::Length),
                            None,
                        ),
                    ],
                    finished_requests: Some(BTreeSet::from(
                        ["chat-reasoning-disabled".to_string()],
                    )),
                    ..Default::default()
                },
            ));
        })
    });

    let backend: Arc<dyn ChatTextBackend> =
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B"));
    let runtime = serving_runtime_from_mock_client(client, backend)
        .with_reasoning_parser(ParserSelection::None);

    let events = collect_events(
        serve_chat(&runtime, sample_request("chat-reasoning-disabled"))
            .await
            .unwrap(),
    )
    .await;
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, ServeEvent::ReasoningDelta { .. }))
    );
    let visible_text = events
        .iter()
        .filter_map(|event| match event {
            ServeEvent::TextDelta { text, .. } if !text.is_empty() => Some(text.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert_eq!(visible_text, "<think>reason more</think>answer");
    assert!(events.iter().any(|event| matches!(
        event,
        ServeEvent::Finished {
            reason: FinishStatus::Length,
            ..
        }
    )));

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_stream_parses_tool_calls_automatically() {
    let (client, mock) = EngineCoreClient::connect_mock("test-model");
    let (shutdown_tx, engine_task) = spawn_mock_engine_task(mock, |mut mock| {
        Box::pin(async move {
            let request = mock.recv_request().await;
            mock.send_outputs(canonicalize_outputs(
                request.generation.prompt_token_ids().len(),
                EngineCoreOutputs {
                    outputs: vec![
                        scheduling_output("chat-tool"),
                        request_output(
                            "chat-tool",
                            bytes_to_token_ids(b"<think>Need tool.</think>"),
                            None,
                            None,
                        ),
                        request_output(
                            "chat-tool",
                            bytes_to_token_ids(b"<tool_call>\n{\"name\":\"get_weather\", "),
                            None,
                            None,
                        ),
                        request_output(
                            "chat-tool",
                            bytes_to_token_ids(
                                b"\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>",
                            ),
                            Some(EngineCoreFinishReason::Stop),
                            Some(StopReason::TokenId(SPECIAL_STOP_TOKEN_ID)),
                        ),
                    ],
                    finished_requests: Some(BTreeSet::from(["chat-tool".to_string()])),
                    ..Default::default()
                },
            ));
        })
    });

    let backend: Arc<dyn ChatTextBackend> =
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B"));
    let runtime = serving_runtime_from_mock_client(client, backend);
    let mut stream = serve_chat(&runtime, sample_tool_request("chat-tool"))
        .await
        .unwrap();

    let mut saw_tool_start = false;
    let mut streamed_tool_args = String::new();
    let mut saw_tool_end = false;

    while let Some(event) = stream.next().await {
        match event.unwrap() {
            ServeEvent::Accepted { .. }
            | ServeEvent::Scheduled { .. }
            | ServeEvent::TextDelta { .. }
            | ServeEvent::ReasoningDelta { .. }
            | ServeEvent::OutputBlockStart { .. }
            | ServeEvent::OutputBlockEnd { .. }
            | ServeEvent::Usage { .. } => {}
            ServeEvent::ToolCallStart { name, .. } => {
                saw_tool_start = true;
                assert_eq!(name, "get_weather");
            }
            ServeEvent::ToolCallArgumentsDelta { delta, .. } => {
                streamed_tool_args.push_str(&delta);
            }
            ServeEvent::ToolCallEnd {
                name, arguments, ..
            } => {
                saw_tool_end = true;
                assert_eq!(name, "get_weather");
                assert_eq!(arguments, r#"{"city":"Paris"}"#);
            }
            ServeEvent::Finished { reason, .. } => {
                assert!(matches!(reason, FinishStatus::Stop { .. }));
                break;
            }
            other => panic!("unexpected canonical event: {other:?}"),
        }
    }

    assert!(saw_tool_start);
    assert_eq!(streamed_tool_args, r#"{"city":"Paris"}"#);
    assert!(saw_tool_end);

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_collect_message_preserves_tool_call_arguments_in_final_only_mode() {
    let (client, mock) = EngineCoreClient::connect_mock("test-model");
    let (shutdown_tx, engine_task) = spawn_mock_engine_task(mock, |mut mock| {
        Box::pin(async move {
            let request = mock.recv_request().await;
            mock.send_outputs(canonicalize_outputs(
                request.generation.prompt_token_ids().len(),
                EngineCoreOutputs {
                    outputs: vec![
                        scheduling_output("chat-final-only-tool"),
                        request_output(
                            "chat-final-only-tool",
                            bytes_to_token_ids(b"<think>Need tool.</think>"),
                            None,
                            None,
                        ),
                        request_output(
                            "chat-final-only-tool",
                            bytes_to_token_ids(b"<tool_call>\n{\"name\":\"get_weather\", "),
                            None,
                            None,
                        ),
                        request_output(
                            "chat-final-only-tool",
                            bytes_to_token_ids(
                                b"\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>",
                            ),
                            Some(EngineCoreFinishReason::Stop),
                            Some(StopReason::TokenId(SPECIAL_STOP_TOKEN_ID)),
                        ),
                    ],
                    finished_requests: Some(BTreeSet::from(["chat-final-only-tool".to_string()])),
                    ..Default::default()
                },
            ));
        })
    });

    let backend: Arc<dyn ChatTextBackend> =
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B"));
    let runtime = serving_runtime_from_mock_client(client, backend);
    let mut request = sample_tool_request("chat-final-only-tool");
    request.intermediate = false;

    let events = collect_events(serve_chat(&runtime, request).await.unwrap()).await;
    assert!(events.iter().any(|event| matches!(
        event,
        ServeEvent::ToolCallEnd {
            name,
            arguments,
            ..
        } if name == "get_weather" && arguments == r#"{"city":"Paris"}"#
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ServeEvent::Finished {
            reason: FinishStatus::Stop { .. },
            ..
        }
    )));

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_stream_and_collect_preserve_prompt_and_sample_logprobs() {
    let (client, mock) = EngineCoreClient::connect_mock("test-model");
    let (shutdown_tx, engine_task) = spawn_mock_engine_task(mock, |mut mock| {
        Box::pin(async move {
            for _ in 0..2 {
                let request = mock.recv_request().await;
                let prompt_token_ids = request.generation.prompt_token_ids();
                mock.send_outputs(canonicalize_outputs(
                    prompt_token_ids.len(),
                    EngineCoreOutputs {
                        outputs: vec![
                            scheduling_output(&request.request_id),
                            request_output_with_logprobs(
                                &request.request_id,
                                Vec::new(),
                                None,
                                None,
                                None,
                                Some(prompt_logprobs_for_tokens(&prompt_token_ids)),
                            ),
                            request_output_with_logprobs(
                                &request.request_id,
                                vec![b'H' as u32],
                                None,
                                None,
                                Some(sample_logprobs_for_token(b'H' as u32, b'h' as u32)),
                                None,
                            ),
                            request_output_with_logprobs(
                                &request.request_id,
                                vec![b'i' as u32],
                                Some(EngineCoreFinishReason::Length),
                                None,
                                Some(sample_logprobs_for_token(b'i' as u32, b'I' as u32)),
                                None,
                            ),
                        ],
                        finished_requests: Some(BTreeSet::from([request.request_id])),
                        ..Default::default()
                    },
                ));
            }
        })
    });

    let backend: Arc<dyn ChatTextBackend> = Arc::new(FakeChatBackend::new());
    let runtime = serving_runtime_from_mock_client(client, backend);

    let mut request = sample_request("chat-logprobs");
    request.sampling_params.logprobs = Some(1);
    request.sampling_params.prompt_logprobs = Some(1);

    let mut stream = serve_chat(&runtime, request.clone()).await.unwrap();
    match next_semantic(&mut stream).await.unwrap().unwrap() {
        ServeEvent::Accepted {
            prompt_token_ids,
            prompt_logprobs,
            ..
        } => {
            assert_eq!(
                prompt_token_ids.len(),
                "system: You are terse.\nuser: Say hi\nassistant:".len()
            );
            assert!(!prompt_token_ids.is_empty());
            let prompt_logprobs = prompt_logprobs.expect("complete prompt logprobs");
            assert_eq!(prompt_logprobs.first_token_id, b's' as u32);
            assert_eq!(prompt_logprobs.first_token, "s");
            assert_eq!(
                prompt_logprobs.scored_positions.len(),
                prompt_token_ids.len() - 1
            );
            assert_eq!(
                prompt_logprobs.scored_positions[0].entries[0],
                DecodedTokenLogprob {
                    token_id: b'y' as u32,
                    token: "y".to_string(),
                    logprob: -0.3,
                    rank: 1,
                }
            );
        }
        other => panic!("expected Start, got {other:?}"),
    }
    assert_eq!(
        next_semantic(&mut stream).await.unwrap().unwrap(),
        ServeEvent::OutputBlockStart {
            candidate_id: CandidateId::PRIMARY,
            index: 0,
            kind: AssistantBlockKind::Text
        }
    );
    assert_eq!(
        next_semantic(&mut stream).await.unwrap().unwrap(),
        ServeEvent::TextDelta {
            candidate_id: CandidateId::PRIMARY,
            text: "H".to_string(),
            token_ids: Vec::new(),
            logprobs: None
        }
    );
    assert_eq!(
        next_semantic(&mut stream).await.unwrap().unwrap(),
        ServeEvent::TextDelta {
            candidate_id: CandidateId::PRIMARY,
            text: String::new(),
            logprobs: Some(DecodedLogprobs {
                positions: vec![DecodedPositionLogprobs {
                    entries: vec![
                        DecodedTokenLogprob {
                            token_id: b'H' as u32,
                            token: "H".to_string(),
                            logprob: -0.1,
                            rank: 1,
                        },
                        DecodedTokenLogprob {
                            token_id: b'h' as u32,
                            token: "h".to_string(),
                            logprob: -0.2,
                            rank: 1,
                        },
                    ],
                }],
            }),
            token_ids: vec![b'H' as u32],
        }
    );
    while !matches!(
        next_semantic(&mut stream).await,
        Some(Ok(ServeEvent::Finished { .. }))
    ) {}

    request.request_id = "chat-logprobs-collect".to_string();
    let events = collect_events(serve_chat(&runtime, request).await.unwrap()).await;
    assert_eq!(visible_text(&events), "Hi");
    let prompt_logprobs = events.iter().find_map(|event| match event {
        ServeEvent::Accepted {
            prompt_logprobs, ..
        } => prompt_logprobs.clone(),
        _ => None,
    });
    let prompt_logprobs = prompt_logprobs.expect("complete collected prompt logprobs");
    assert_eq!(prompt_logprobs.first_token_id, b's' as u32);
    assert_eq!(
        prompt_logprobs.scored_positions.len(),
        "system: You are terse.\nuser: Say hi\nassistant:".len() - 1
    );
    assert_eq!(
        prompt_logprobs.scored_positions[0].entries[0].token_id,
        b'y' as u32
    );
    let sample_positions = events
        .iter()
        .filter_map(|event| match event {
            ServeEvent::TextDelta {
                logprobs: Some(logprobs),
                ..
            } => Some(logprobs.positions.clone()),
            _ => None,
        })
        .flatten()
        .collect::<Vec<_>>();
    assert_eq!(
        sample_positions,
        vec![
            DecodedPositionLogprobs {
                entries: vec![
                    DecodedTokenLogprob {
                        token_id: b'H' as u32,
                        token: "H".to_string(),
                        logprob: -0.1,
                        rank: 1,
                    },
                    DecodedTokenLogprob {
                        token_id: b'h' as u32,
                        token: "h".to_string(),
                        logprob: -0.2,
                        rank: 1,
                    },
                ],
            },
            DecodedPositionLogprobs {
                entries: vec![
                    DecodedTokenLogprob {
                        token_id: b'i' as u32,
                        token: "i".to_string(),
                        logprob: -0.1,
                        rank: 1,
                    },
                    DecodedTokenLogprob {
                        token_id: b'I' as u32,
                        token: "I".to_string(),
                        logprob: -0.2,
                        rank: 1,
                    },
                ],
            },
        ]
    );

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_rejects_unknown_tool_parser_before_engine_request() {
    let (client, mock) = EngineCoreClient::connect_mock("test-model");
    let (shutdown_tx, engine_task) = spawn_mock_engine_task(mock, |mut mock| {
        Box::pin(async move {
            assert!(
                timeout(Duration::from_millis(100), mock.recv())
                    .await
                    .is_err(),
                "chat request should fail before any engine request is sent"
            );
        })
    });

    let backend: Arc<dyn ChatTextBackend> = Arc::new(FakeChatBackend::new());
    let runtime = serving_runtime_from_mock_client(client, backend).with_tool_call_parser(
        ParserSelection::Explicit("definitely_missing_tool_parser".into()),
    );
    let error = match serve_chat(&runtime, sample_tool_request("chat-tool-no-model")).await {
        Ok(_) => panic!("unknown explicit tool parser should fail"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        ServeError::Chat(uniserve_serving::chat::Error::ParserUnavailableByName { name, .. })
        if name == "definitely_missing_tool_parser"
    ));

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_rejects_unknown_reasoning_parser_before_engine_request() {
    let (client, mock) = EngineCoreClient::connect_mock("test-model");
    let (shutdown_tx, engine_task) = spawn_mock_engine_task(mock, |mut mock| {
        Box::pin(async move {
            assert!(
                timeout(Duration::from_millis(100), mock.recv())
                    .await
                    .is_err(),
                "chat request should fail before any engine request is sent"
            );
        })
    });

    let backend: Arc<dyn ChatTextBackend> = Arc::new(FakeChatBackend::new());
    let runtime = serving_runtime_from_mock_client(client, backend).with_reasoning_parser(
        ParserSelection::Explicit("definitely_missing_reasoning_parser".into()),
    );
    let error = match serve_chat(&runtime, sample_request("chat-reasoning-no-model")).await {
        Ok(_) => panic!("unknown explicit reasoning parser should fail"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        ServeError::Chat(uniserve_serving::chat::Error::ParserUnavailableByName { name, .. })
        if name == "definitely_missing_reasoning_parser"
    ));

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_rejects_tool_requests_when_tool_parser_is_disabled() {
    let (client, mock) = EngineCoreClient::connect_mock("test-model");
    let (shutdown_tx, engine_task) = spawn_mock_engine_task(mock, |mut mock| {
        Box::pin(async move {
            assert!(
                timeout(Duration::from_millis(100), mock.recv())
                    .await
                    .is_err(),
                "chat request should fail before any engine request is sent"
            );
        })
    });

    let backend: Arc<dyn ChatTextBackend> = Arc::new(FakeChatBackend::new());
    let runtime = serving_runtime_from_mock_client(client, backend)
        .with_tool_call_parser(ParserSelection::None);
    let error = match serve_chat(&runtime, sample_tool_request("chat-tool-parser-disabled")).await {
        Ok(_) => panic!("tool requests should fail when tool parsing is disabled"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        ServeError::Chat(uniserve_serving::chat::Error::ParserDisabled { kind: "tool" })
    ));

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    runtime.shutdown().await.unwrap();
}
