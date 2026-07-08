//! Canonical semantic serving runtime.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::collections::HashMap;
use std::pin::Pin;

use asynk_strim_attr::{TryYielder, try_stream};
use futures::{Stream, StreamExt as _, pin_mut};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uniserve_chat::{
    AssistantBlockKind, AssistantContentBlock, ChatEvent, ChatMessage, ChatOptions, ChatRequest,
    ChatTool, ChatToolChoice, PreparedChatRequest,
};
use uniserve_engine_client::protocol::StructuredOutputsParams;
use uniserve_engine_client::protocol::lora::LoraRequest;
use uniserve_engine_client::{
    GenEvent, NativeEventStream, NativeFinishReason, NativeGenerateRequest,
};
use uniserve_llm::FinishReason;
use uniserve_model_profile::ModelProfile;
use uniserve_text::{
    DecodedLogprobs, DecodedPromptLogprobs, DecodedTextEvent, PreparedTextRequest, Prompt,
    SamplingParams, TextDecodeOptions, TextRequest,
};

pub type ServeRequestId = String;
pub type CandidateId = u32;
pub type ServeEventStream = Pin<Box<dyn Stream<Item = Result<ServeEvent>> + Send>>;

pub type Result<T> = std::result::Result<T, ServeError>;

/// Runtime-local serving errors.
#[derive(Debug, Error)]
pub enum ServeError {
    #[error("request `{request_id}` asks for {requested} outputs; only one output is supported")]
    UnsupportedOutputCount { request_id: String, requested: u32 },
    #[error("request `{request_id}` contains unsupported runtime extension `{key}`")]
    UnsupportedRuntimeExtension { request_id: String, key: String },
    #[error("text runtime error")]
    Text(#[from] uniserve_text::Error),
    #[error("chat runtime error")]
    Chat(#[from] uniserve_chat::Error),
    #[error("engine runtime error: {0}")]
    Engine(String),
}

/// Semantic request accepted by [`ServingRuntime`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServeRequest {
    pub request_id: ServeRequestId,
    pub model_context: ModelContext,
    pub generation: GenerationPolicy,
    pub modalities: ModalityPolicy,
    pub adapter: AdapterSelection,
    pub cache: CachePolicy,
    pub scheduling: SchedulingPolicy,
    pub metadata: RequestMetadata,
}

impl ServeRequest {
    pub fn text(request_id: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            model_context: ModelContext::RawPrompt(prompt.into()),
            generation: GenerationPolicy::default(),
            modalities: ModalityPolicy::default(),
            adapter: AdapterSelection::Base,
            cache: CachePolicy::default(),
            scheduling: SchedulingPolicy::default(),
            metadata: RequestMetadata::default(),
        }
    }

    pub fn chat(request_id: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        Self {
            request_id: request_id.into(),
            model_context: ModelContext::Chat {
                messages,
                chat_options: ChatOptions::default(),
                tools: Vec::new(),
                tool_choice: ChatToolChoice::None,
                documents: None,
            },
            generation: GenerationPolicy::default(),
            modalities: ModalityPolicy::default(),
            adapter: AdapterSelection::Base,
            cache: CachePolicy::default(),
            scheduling: SchedulingPolicy::default(),
            metadata: RequestMetadata::default(),
        }
    }

    pub fn from_text_request(text: TextRequest, metadata: RequestMetadata) -> Result<Self> {
        let TextRequest {
            request_id,
            prompt,
            mm_features,
            sampling_params,
            decode_options,
            intermediate,
            priority,
            cache_salt,
            add_special_tokens,
            data_parallel_rank,
            lora_request,
        } = text;
        let mut generation =
            generation_from_sampling(&request_id, sampling_params, decode_options)?;
        generation.intermediate = intermediate;
        generation.add_special_tokens = add_special_tokens;
        let model_context = match prompt {
            Prompt::Text(text) => ModelContext::RawPrompt(text),
            Prompt::TokenIds(token_ids) => ModelContext::TokenIds(token_ids),
        };

        Ok(Self {
            request_id,
            model_context,
            generation,
            modalities: ModalityPolicy {
                input_image: mm_features
                    .as_ref()
                    .is_some_and(|features| !features.is_empty()),
                ..ModalityPolicy::default()
            },
            adapter: adapter_from_lora(lora_request),
            cache: CachePolicy {
                salt: cache_salt,
                ..CachePolicy::default()
            },
            scheduling: SchedulingPolicy {
                priority,
                data_parallel_rank,
                ..SchedulingPolicy::default()
            },
            metadata,
        })
    }

    pub fn from_chat_request(chat: ChatRequest, metadata: RequestMetadata) -> Result<Self> {
        let ChatRequest {
            request_id,
            messages,
            sampling_params,
            chat_options,
            tools,
            tool_choice,
            decode_options,
            intermediate,
            priority,
            documents,
            cache_salt,
            add_special_tokens,
            data_parallel_rank,
            lora_request,
        } = chat;
        let mut generation =
            generation_from_sampling(&request_id, sampling_params, decode_options)?;
        generation.intermediate = intermediate;
        generation.add_special_tokens = add_special_tokens;

        Ok(Self {
            request_id,
            model_context: ModelContext::Chat {
                messages,
                chat_options,
                tools,
                tool_choice,
                documents,
            },
            generation,
            modalities: ModalityPolicy::default(),
            adapter: adapter_from_lora(lora_request),
            cache: CachePolicy {
                salt: cache_salt,
                ..CachePolicy::default()
            },
            scheduling: SchedulingPolicy {
                priority,
                data_parallel_rank,
                ..SchedulingPolicy::default()
            },
            metadata,
        })
    }
}

/// Ordered semantic input context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelContext {
    RawPrompt(String),
    TokenIds(Vec<u32>),
    Chat {
        messages: Vec<ChatMessage>,
        chat_options: ChatOptions,
        tools: Vec<ChatTool>,
        tool_choice: ChatToolChoice,
        documents: Option<Vec<serde_json::Value>>,
    },
}

/// Semantic generation controls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationPolicy {
    pub num_outputs: u32,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub seed: Option<i64>,
    pub max_tokens: Option<u32>,
    pub min_tokens: Option<u32>,
    pub logprobs: Option<i32>,
    pub prompt_logprobs: Option<i32>,
    pub min_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub stop_token_ids: Vec<u32>,
    pub stop_strings: Vec<String>,
    pub ignore_eos: bool,
    pub logit_bias: Option<HashMap<u32, f32>>,
    pub allowed_token_ids: Option<Vec<u32>>,
    pub bad_words: Vec<String>,
    pub logprob_token_ids: Option<Vec<u32>>,
    pub structured_output: Option<StructuredOutputIntent>,
    pub kv_transfer_params: Option<serde_json::Value>,
    pub skip_special_tokens: bool,
    pub include_stop_string_in_output: bool,
    pub output_detail: OutputDetail,
    pub intermediate: bool,
    pub add_special_tokens: bool,
}

impl Default for GenerationPolicy {
    fn default() -> Self {
        Self {
            num_outputs: 1,
            temperature: None,
            top_p: None,
            top_k: None,
            seed: None,
            max_tokens: None,
            min_tokens: None,
            logprobs: None,
            prompt_logprobs: None,
            min_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            repetition_penalty: None,
            stop_token_ids: Vec::new(),
            stop_strings: Vec::new(),
            ignore_eos: false,
            logit_bias: None,
            allowed_token_ids: None,
            bad_words: Vec::new(),
            logprob_token_ids: None,
            structured_output: None,
            kv_transfer_params: None,
            skip_special_tokens: true,
            include_stop_string_in_output: false,
            output_detail: OutputDetail::default(),
            intermediate: true,
            add_special_tokens: false,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputDetail {
    #[default]
    VisibleText,
    Tokens,
    Logprobs,
}

/// Semantic structured-output intent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StructuredOutputIntent {
    JsonObject {
        disable_any_whitespace: bool,
        whitespace_pattern: Option<String>,
    },
    JsonSchema {
        schema: serde_json::Value,
        disable_any_whitespace: bool,
        disable_additional_properties: bool,
        whitespace_pattern: Option<String>,
    },
    Regex(String),
    Choice(Vec<String>),
    Grammar(String),
    StructuralTag(String),
}

/// Requested modality capabilities.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModalityPolicy {
    pub input_text: bool,
    pub input_image: bool,
    pub output_text: bool,
    pub output_image: bool,
}

impl Default for ModalityPolicy {
    fn default() -> Self {
        Self {
            input_text: true,
            input_image: false,
            output_text: true,
            output_image: false,
        }
    }
}

/// Router-resolved adapter selection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdapterSelection {
    Base,
    Adapter {
        name: String,
        internal_id: u64,
        path: String,
        load_inplace: bool,
        is_3d_lora_weight: bool,
    },
}

impl Default for AdapterSelection {
    fn default() -> Self {
        Self::Base
    }
}

/// Semantic cache policy.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachePolicy {
    pub namespace: Option<String>,
    pub salt: Option<String>,
    pub bypass_read: bool,
    pub no_store: bool,
    pub replayable: bool,
}

/// Scheduler admission hints.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchedulingPolicy {
    pub priority: i32,
    pub data_parallel_rank: Option<u32>,
    pub deadline_ms: Option<u64>,
    pub trace_context: HashMap<String, String>,
}

/// Typed request metadata preserved for inspection and metrics.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestMetadata {
    pub tenant: Option<String>,
    pub route: Option<String>,
    pub protocol_adapter: Option<String>,
}

/// Runtime-local pure execution plan.
#[derive(Debug)]
pub struct ExecutionPlan {
    request_id: ServeRequestId,
    profile: ModelProfile,
    capability_snapshot: RuntimeCapabilitySnapshot,
    branch: PlannedRequest,
    inspection: PlanInspection,
}

impl ExecutionPlan {
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    pub fn inspect(&self) -> &PlanInspection {
        &self.inspection
    }

    pub fn profile(&self) -> &ModelProfile {
        &self.profile
    }

    pub fn capability_snapshot(&self) -> &RuntimeCapabilitySnapshot {
        &self.capability_snapshot
    }
}

#[derive(Debug)]
enum PlannedRequest {
    Text(PreparedTextRequest),
    Chat(PreparedChatRequest),
}

/// Sanitized plan inspection form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanInspection {
    pub request_id: ServeRequestId,
    pub profile_id: String,
    pub dialect_id: String,
    pub tokenizer_fingerprint: String,
    pub capability_snapshot: RuntimeCapabilitySnapshot,
    pub context: PlannedContextInspection,
    pub generation: GenerationPlanInspection,
    pub cache: CachePolicy,
    pub adapter: AdapterSelection,
    pub scheduling: SchedulingPolicy,
    pub output: OutputProcessingPlan,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeCapabilitySnapshot {
    pub supports_text: bool,
    pub supports_chat: bool,
    pub supports_multimodal_input: bool,
}

impl From<&ModelProfile> for RuntimeCapabilitySnapshot {
    fn from(profile: &ModelProfile) -> Self {
        Self {
            supports_text: profile.supports_text,
            supports_chat: profile.supports_chat,
            supports_multimodal_input: profile.supports_multimodal_input,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlannedContextInspection {
    Text {
        prompt_token_count: usize,
        pre_tokenized: bool,
    },
    Chat {
        message_count: usize,
        prompt_token_count: usize,
        multimodal: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationPlanInspection {
    pub num_outputs: u32,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub stop_token_count: usize,
    pub stop_string_count: usize,
    pub structured_output: Option<StructuredOutputIntent>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputProcessingPlan {
    pub visible_text: bool,
    pub reasoning: bool,
    pub tools: bool,
    pub logprobs: bool,
    pub token_ids: bool,
}

/// Single-model semantic runtime.
pub struct ServingRuntime {
    profile: ModelProfile,
    chat: uniserve_chat::ChatLlm,
}

impl ServingRuntime {
    pub fn new(profile: ModelProfile, chat: uniserve_chat::ChatLlm) -> Self {
        Self { profile, chat }
    }

    pub fn from_chat_runtime(chat: uniserve_chat::ChatLlm) -> Self {
        let profile = ModelProfile::from_chat_runtime(&chat);
        Self::new(profile, chat)
    }

    pub fn profile(&self) -> &ModelProfile {
        &self.profile
    }

    pub fn chat(&self) -> &uniserve_chat::ChatLlm {
        &self.chat
    }

    pub fn compile(&self, request: ServeRequest) -> Result<ExecutionPlan> {
        futures::executor::block_on(self.compile_async(request))
    }

    pub async fn compile_async(&self, request: ServeRequest) -> Result<ExecutionPlan> {
        if request.generation.num_outputs != 1 {
            return Err(ServeError::UnsupportedOutputCount {
                request_id: request.request_id,
                requested: request.generation.num_outputs,
            });
        }

        let capability_snapshot = RuntimeCapabilitySnapshot::from(&self.profile);
        let generation_inspection = GenerationPlanInspection {
            num_outputs: request.generation.num_outputs,
            max_tokens: request.generation.max_tokens,
            temperature: request.generation.temperature,
            top_p: request.generation.top_p,
            top_k: request.generation.top_k,
            stop_token_count: request.generation.stop_token_ids.len(),
            stop_string_count: request.generation.stop_strings.len(),
            structured_output: request.generation.structured_output.clone(),
        };
        let output = OutputProcessingPlan {
            visible_text: request.modalities.output_text,
            reasoning: matches!(&request.model_context, ModelContext::Chat { .. }),
            tools: matches!(&request.model_context, ModelContext::Chat { tools, .. } if !tools.is_empty()),
            logprobs: request.generation.logprobs.is_some()
                || request.generation.prompt_logprobs.is_some(),
            token_ids: matches!(request.generation.output_detail, OutputDetail::Tokens),
        };

        let (branch, context) = match request.model_context.clone() {
            ModelContext::RawPrompt(prompt) => {
                let text_request = self.to_text_request(&request, Prompt::Text(prompt));
                let prepared = self.chat.text().compile(text_request)?;
                let prompt_token_count = prepared.generate_request.prompt_token_ids.len();
                (
                    PlannedRequest::Text(prepared),
                    PlannedContextInspection::Text {
                        prompt_token_count,
                        pre_tokenized: false,
                    },
                )
            }
            ModelContext::TokenIds(token_ids) => {
                let text_request = self.to_text_request(&request, Prompt::TokenIds(token_ids));
                let prepared = self.chat.text().compile(text_request)?;
                let prompt_token_count = prepared.generate_request.prompt_token_ids.len();
                (
                    PlannedRequest::Text(prepared),
                    PlannedContextInspection::Text {
                        prompt_token_count,
                        pre_tokenized: true,
                    },
                )
            }
            ModelContext::Chat {
                messages,
                chat_options,
                tools,
                tool_choice,
                documents,
            } => {
                let chat_request = self.to_chat_request(
                    &request,
                    messages,
                    chat_options,
                    tools,
                    tool_choice,
                    documents,
                );
                let prepared = self.chat.compile(chat_request).await?;
                let prompt_token_count = prepared
                    .prepared_text_request
                    .generate_request
                    .prompt_token_ids
                    .len();
                let multimodal = prepared.chat_request.has_multimodal();
                let message_count = prepared.chat_request.messages.len();
                (
                    PlannedRequest::Chat(prepared),
                    PlannedContextInspection::Chat {
                        message_count,
                        prompt_token_count,
                        multimodal,
                    },
                )
            }
        };

        let inspection = PlanInspection {
            request_id: request.request_id.clone(),
            profile_id: self.profile.profile_id.clone(),
            dialect_id: self.profile.dialect_id.clone(),
            tokenizer_fingerprint: self.profile.tokenizer_fingerprint.clone(),
            capability_snapshot: capability_snapshot.clone(),
            context,
            generation: generation_inspection,
            cache: request.cache,
            adapter: request.adapter,
            scheduling: request.scheduling,
            output,
        };

        Ok(ExecutionPlan {
            request_id: request.request_id,
            profile: self.profile.clone(),
            capability_snapshot,
            branch,
            inspection,
        })
    }

    pub async fn serve(&self, request: ServeRequest) -> Result<ServeEventStream> {
        let plan = self.compile_async(request).await?;
        self.execute(plan).await
    }

    pub async fn serve_native(
        &self,
        request_id: ServeRequestId,
        request: NativeGenerateRequest,
    ) -> Result<ServeEventStream> {
        let stream = self
            .chat
            .uniserve_engine_client()
            .generate_native(request)
            .await
            .map_err(|error| ServeError::Engine(error.to_string()))?;
        Ok(Box::pin(native_event_stream(request_id, stream)))
    }

    pub async fn execute(&self, plan: ExecutionPlan) -> Result<ServeEventStream> {
        match plan.branch {
            PlannedRequest::Text(prepared) => {
                let request_id = prepared.text_request.request_id.clone();
                let stream = self.chat.text().generate_prepared(prepared).await?;
                Ok(Box::pin(text_event_stream(request_id, stream)))
            }
            PlannedRequest::Chat(prepared) => {
                let request_id = prepared.chat_request.request_id.clone();
                let stream = self.chat.chat_prepared(prepared).await?;
                Ok(Box::pin(chat_event_stream(request_id, stream)))
            }
        }
    }

    pub async fn cancel(
        &self,
        request_id: ServeRequestId,
    ) -> std::result::Result<(), ServeControlError> {
        self.chat
            .uniserve_engine_client()
            .abort(&[request_id])
            .await
            .map_err(|message| ServeControlError::Engine(message.to_string()))
    }

    pub async fn abort(
        &self,
        request_id: ServeRequestId,
        reason: AbortReason,
    ) -> std::result::Result<(), ServeControlError> {
        let _ = reason;
        self.cancel(request_id).await
    }

    pub async fn shutdown(self) -> std::result::Result<(), uniserve_chat::Error> {
        self.chat.shutdown().await
    }

    fn to_text_request(&self, request: &ServeRequest, prompt: Prompt) -> TextRequest {
        TextRequest {
            request_id: request.request_id.clone(),
            prompt,
            mm_features: None,
            sampling_params: sampling_params(&request.generation, &request.cache),
            decode_options: decode_options(&request.generation),
            intermediate: request.generation.intermediate,
            priority: request.scheduling.priority,
            cache_salt: request.cache.salt.clone(),
            add_special_tokens: request.generation.add_special_tokens,
            data_parallel_rank: request.scheduling.data_parallel_rank,
            lora_request: lora_request(&request.adapter),
        }
    }

    fn to_chat_request(
        &self,
        request: &ServeRequest,
        messages: Vec<ChatMessage>,
        chat_options: ChatOptions,
        tools: Vec<ChatTool>,
        tool_choice: ChatToolChoice,
        documents: Option<Vec<serde_json::Value>>,
    ) -> ChatRequest {
        ChatRequest {
            request_id: request.request_id.clone(),
            messages,
            sampling_params: sampling_params(&request.generation, &request.cache),
            chat_options,
            tools,
            tool_choice,
            decode_options: decode_options(&request.generation),
            intermediate: request.generation.intermediate,
            priority: request.scheduling.priority,
            documents,
            cache_salt: request.cache.salt.clone(),
            add_special_tokens: request.generation.add_special_tokens,
            data_parallel_rank: request.scheduling.data_parallel_rank,
            lora_request: lora_request(&request.adapter),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbortReason {
    OwnerCancelled,
    Admin,
    RuntimeShutdown,
}

#[derive(Debug, Error)]
pub enum ServeControlError {
    #[error("engine control failed: {0}")]
    Engine(String),
}

/// Protocol-neutral runtime event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ServeEvent {
    Accepted {
        request_id: ServeRequestId,
        prompt_token_count: usize,
        prompt_token_ids: Vec<u32>,
        prompt_logprobs: Option<DecodedPromptLogprobs>,
    },
    Scheduled {
        request_id: ServeRequestId,
        queued_at: Option<f64>,
        scheduled_at: Option<f64>,
    },
    TextDelta {
        candidate_id: CandidateId,
        text: String,
        token_ids: Vec<u32>,
        logprobs: Option<DecodedLogprobs>,
    },
    InternalTextDelta {
        candidate_id: CandidateId,
        text: String,
    },
    ReasoningDelta {
        candidate_id: CandidateId,
        text: String,
    },
    OutputBlockStart {
        candidate_id: CandidateId,
        index: usize,
        kind: AssistantBlockKind,
    },
    OutputBlockEnd {
        candidate_id: CandidateId,
        index: usize,
        block: AssistantContentBlock,
    },
    ToolCallStart {
        candidate_id: CandidateId,
        index: usize,
        id: String,
        name: String,
    },
    ToolCallArgumentsDelta {
        candidate_id: CandidateId,
        index: usize,
        delta: String,
    },
    ToolCallEnd {
        candidate_id: CandidateId,
        index: usize,
        id: String,
        name: String,
        arguments: String,
    },
    ImageBegin {
        candidate_id: CandidateId,
        image_id: String,
        width: Option<u32>,
        height: Option<u32>,
        steps: Option<u32>,
    },
    ImageStep {
        candidate_id: CandidateId,
        image_id: String,
        step: u32,
    },
    TokenLogprobs {
        candidate_id: CandidateId,
        token_id: u32,
        top: Vec<(u32, f32)>,
    },
    ImageCommit {
        candidate_id: CandidateId,
        image_id: String,
    },
    ImageDone {
        candidate_id: CandidateId,
        image_id: String,
        width: Option<u32>,
        height: Option<u32>,
        bytes: Option<u64>,
        sha256: Option<String>,
        pixels_png_b64: Option<String>,
    },
    Usage {
        prompt_tokens: u32,
        visible_output_tokens: u32,
        internal_tokens: u32,
        image_count: u32,
        image_steps: u32,
    },
    Finished {
        candidate_id: CandidateId,
        reason: FinishStatus,
        finish_detail: Option<String>,
        kv_transfer_params: Option<serde_json::Value>,
    },
    Rejected {
        request_id: ServeRequestId,
        message: String,
    },
    Cancelled {
        request_id: ServeRequestId,
    },
    Aborted {
        request_id: ServeRequestId,
    },
    Failed {
        request_id: ServeRequestId,
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishStatus {
    Stop {
        stop_reason: Option<serde_json::Value>,
    },
    Length,
    Abort,
    Error,
    Repetition,
}

impl From<&FinishReason> for FinishStatus {
    fn from(reason: &FinishReason) -> Self {
        match reason {
            FinishReason::Stop(stop_reason) => Self::Stop {
                stop_reason: stop_reason.as_ref().map(|value| {
                    serde_json::to_value(value)
                        .unwrap_or_else(|_| serde_json::Value::String(format!("{value:?}")))
                }),
            },
            FinishReason::Length => Self::Length,
            FinishReason::Abort => Self::Abort,
            FinishReason::Error => Self::Error,
            FinishReason::Repetition => Self::Repetition,
        }
    }
}

#[try_stream]
async fn text_event_stream(
    request_id: String,
    stream: impl Stream<Item = uniserve_text::Result<DecodedTextEvent>> + Send,
    mut y: TryYielder<ServeEvent, ServeError>,
) -> Result<()> {
    pin_mut!(stream);
    while let Some(next) = stream.next().await {
        match next? {
            DecodedTextEvent::Start {
                prompt_token_ids,
                prompt_logprobs,
            } => {
                y.yield_ok(ServeEvent::Accepted {
                    request_id: request_id.clone(),
                    prompt_token_count: prompt_token_ids.len(),
                    prompt_token_ids: prompt_token_ids.to_vec(),
                    prompt_logprobs,
                })
                .await;
            }
            DecodedTextEvent::TextDelta {
                delta,
                token_ids,
                logprobs,
                finished,
                ..
            } => {
                if !delta.is_empty() || !token_ids.is_empty() || finished.is_some() {
                    y.yield_ok(ServeEvent::TextDelta {
                        candidate_id: 0,
                        text: delta,
                        token_ids,
                        logprobs,
                    })
                    .await;
                }
                if let Some(finished) = finished {
                    y.yield_ok(ServeEvent::Usage {
                        prompt_tokens: finished.prompt_token_count as u32,
                        visible_output_tokens: finished.output_token_count as u32,
                        internal_tokens: 0,
                        image_count: 0,
                        image_steps: 0,
                    })
                    .await;
                    y.yield_ok(ServeEvent::Finished {
                        candidate_id: 0,
                        reason: FinishStatus::from(&finished.finish_reason),
                        finish_detail: None,
                        kv_transfer_params: finished.kv_transfer_params,
                    })
                    .await;
                }
            }
        }
    }

    Ok(())
}

#[try_stream]
async fn chat_event_stream(
    request_id: String,
    stream: impl Stream<Item = uniserve_chat::Result<ChatEvent>> + Send,
    mut y: TryYielder<ServeEvent, ServeError>,
) -> Result<()> {
    pin_mut!(stream);
    while let Some(next) = stream.next().await {
        match next? {
            ChatEvent::Start {
                prompt_token_ids,
                prompt_logprobs,
            } => {
                y.yield_ok(ServeEvent::Accepted {
                    request_id: request_id.clone(),
                    prompt_token_count: prompt_token_ids.len(),
                    prompt_token_ids: prompt_token_ids.to_vec(),
                    prompt_logprobs,
                })
                .await;
            }
            ChatEvent::BlockDelta { kind, delta, .. } => match kind {
                uniserve_chat::AssistantBlockKind::Text => {
                    if !delta.is_empty() {
                        y.yield_ok(ServeEvent::TextDelta {
                            candidate_id: 0,
                            text: delta,
                            token_ids: Vec::new(),
                            logprobs: None,
                        })
                        .await;
                    }
                }
                uniserve_chat::AssistantBlockKind::Reasoning => {
                    if !delta.is_empty() {
                        y.yield_ok(ServeEvent::ReasoningDelta {
                            candidate_id: 0,
                            text: delta,
                        })
                        .await;
                    }
                }
                uniserve_chat::AssistantBlockKind::ToolCall => {}
            },
            ChatEvent::LogprobsDelta {
                token_ids,
                logprobs,
            } => {
                if !token_ids.is_empty() || logprobs.is_some() {
                    y.yield_ok(ServeEvent::TextDelta {
                        candidate_id: 0,
                        text: String::new(),
                        token_ids,
                        logprobs,
                    })
                    .await;
                }
            }
            ChatEvent::ToolCallStart { index, id, name } => {
                y.yield_ok(ServeEvent::ToolCallStart {
                    candidate_id: 0,
                    index,
                    id,
                    name,
                })
                .await;
            }
            ChatEvent::ToolCallArgumentsDelta { index, delta } => {
                y.yield_ok(ServeEvent::ToolCallArgumentsDelta {
                    candidate_id: 0,
                    index,
                    delta,
                })
                .await;
            }
            ChatEvent::ToolCallEnd { index, call } => {
                y.yield_ok(ServeEvent::ToolCallEnd {
                    candidate_id: 0,
                    index,
                    id: call.id,
                    name: call.name,
                    arguments: call.arguments,
                })
                .await;
            }
            ChatEvent::Done {
                prompt_token_count,
                output_token_count,
                finish_reason,
                kv_transfer_params,
                ..
            } => {
                y.yield_ok(ServeEvent::Usage {
                    prompt_tokens: prompt_token_count as u32,
                    visible_output_tokens: output_token_count as u32,
                    internal_tokens: 0,
                    image_count: 0,
                    image_steps: 0,
                })
                .await;
                y.yield_ok(ServeEvent::Finished {
                    candidate_id: 0,
                    reason: FinishStatus::from(&finish_reason),
                    finish_detail: None,
                    kv_transfer_params,
                })
                .await;
            }
            ChatEvent::BlockStart { index, kind } => {
                y.yield_ok(ServeEvent::OutputBlockStart {
                    candidate_id: 0,
                    index,
                    kind,
                })
                .await;
            }
            ChatEvent::BlockEnd { index, block } => {
                y.yield_ok(ServeEvent::OutputBlockEnd {
                    candidate_id: 0,
                    index,
                    block,
                })
                .await;
            }
        }
    }

    Ok(())
}

#[try_stream]
async fn native_event_stream(
    request_id: String,
    mut stream: NativeEventStream,
    mut y: TryYielder<ServeEvent, ServeError>,
) -> Result<()> {
    while let Some(event) = stream.next().await {
        match event {
            GenEvent::Scheduled {
                queued_at,
                scheduled_at,
            } => {
                y.yield_ok(ServeEvent::Scheduled {
                    request_id: request_id.clone(),
                    queued_at: Some(queued_at),
                    scheduled_at: Some(scheduled_at),
                })
                .await;
            }
            GenEvent::TextToken { id, .. } => {
                y.yield_ok(ServeEvent::TextDelta {
                    candidate_id: 0,
                    text: String::new(),
                    token_ids: vec![id],
                    logprobs: None,
                })
                .await;
            }
            GenEvent::TokenLogprobs { id, top } => {
                y.yield_ok(ServeEvent::TokenLogprobs {
                    candidate_id: 0,
                    token_id: id,
                    top,
                })
                .await;
            }
            GenEvent::ImageBegin {
                image_id,
                height,
                width,
                steps,
            } => {
                y.yield_ok(ServeEvent::ImageBegin {
                    candidate_id: 0,
                    image_id: image_id.to_string(),
                    width: Some(width),
                    height: Some(height),
                    steps: Some(steps as u32),
                })
                .await;
            }
            GenEvent::ImageStep { image_id, step } => {
                y.yield_ok(ServeEvent::ImageStep {
                    candidate_id: 0,
                    image_id: image_id.to_string(),
                    step: step as u32,
                })
                .await;
            }
            GenEvent::ImageDone {
                image_id,
                height,
                width,
                bytes,
                sha256,
                pixels_png_b64,
            } => {
                y.yield_ok(ServeEvent::ImageDone {
                    candidate_id: 0,
                    image_id: image_id.to_string(),
                    width: Some(width),
                    height: Some(height),
                    bytes: Some(bytes),
                    sha256: Some(sha256),
                    pixels_png_b64: Some(pixels_png_b64),
                })
                .await;
            }
            GenEvent::Finished {
                reason,
                stop_reason,
                prompt_tokens,
                completion_tokens,
                images,
            } => {
                let finish_detail = native_finish_detail(&reason).to_string();
                y.yield_ok(ServeEvent::Usage {
                    prompt_tokens: prompt_tokens as u32,
                    visible_output_tokens: completion_tokens as u32,
                    internal_tokens: 0,
                    image_count: images as u32,
                    image_steps: 0,
                })
                .await;
                y.yield_ok(ServeEvent::Finished {
                    candidate_id: 0,
                    reason: native_finish_status(reason, stop_reason),
                    finish_detail: Some(finish_detail),
                    kv_transfer_params: None,
                })
                .await;
            }
            GenEvent::Rejected { message } => {
                y.yield_ok(ServeEvent::Rejected {
                    request_id: request_id.clone(),
                    message,
                })
                .await;
            }
            GenEvent::Error { message } => {
                y.yield_ok(ServeEvent::Failed {
                    request_id: request_id.clone(),
                    message,
                })
                .await;
            }
        }
    }

    Ok(())
}

fn native_finish_status(reason: NativeFinishReason, stop_reason: Option<String>) -> FinishStatus {
    match reason {
        NativeFinishReason::Eos | NativeFinishReason::Stop | NativeFinishReason::ImageDone => {
            FinishStatus::Stop {
                stop_reason: stop_reason.map(serde_json::Value::String),
            }
        }
        NativeFinishReason::MaxTokens => FinishStatus::Length,
        NativeFinishReason::Cancelled | NativeFinishReason::Aborted => FinishStatus::Abort,
        NativeFinishReason::Error => FinishStatus::Error,
    }
}

fn native_finish_detail(reason: &NativeFinishReason) -> &'static str {
    match reason {
        NativeFinishReason::Eos => "eos",
        NativeFinishReason::MaxTokens => "max_tokens",
        NativeFinishReason::Stop => "stop",
        NativeFinishReason::ImageDone => "image_done",
        NativeFinishReason::Cancelled => "cancelled",
        NativeFinishReason::Aborted => "aborted",
        NativeFinishReason::Error => "error",
    }
}

fn sampling_params(generation: &GenerationPolicy, cache: &CachePolicy) -> SamplingParams {
    SamplingParams {
        temperature: generation.temperature,
        top_p: generation.top_p,
        top_k: generation.top_k,
        seed: generation.seed,
        max_tokens: generation.max_tokens,
        min_tokens: generation.min_tokens,
        logprobs: generation.logprobs,
        prompt_logprobs: generation.prompt_logprobs,
        min_p: generation.min_p,
        frequency_penalty: generation.frequency_penalty,
        presence_penalty: generation.presence_penalty,
        repetition_penalty: generation.repetition_penalty,
        stop_token_ids: (!generation.stop_token_ids.is_empty())
            .then(|| generation.stop_token_ids.clone()),
        ignore_eos: generation.ignore_eos,
        logit_bias: generation.logit_bias.clone(),
        allowed_token_ids: generation.allowed_token_ids.clone(),
        bad_words: (!generation.bad_words.is_empty()).then(|| generation.bad_words.clone()),
        logprob_token_ids: generation.logprob_token_ids.clone(),
        structured_outputs: generation
            .structured_output
            .as_ref()
            .map(structured_output_params),
        skip_reading_prefix_cache: Some(cache.bypass_read),
        uniserve_xargs: generation
            .kv_transfer_params
            .as_ref()
            .map(|value| HashMap::from([("kv_transfer_params".to_string(), value.clone())])),
    }
}

fn decode_options(generation: &GenerationPolicy) -> TextDecodeOptions {
    TextDecodeOptions {
        skip_special_tokens: generation.skip_special_tokens,
        include_stop_str_in_output: generation.include_stop_string_in_output,
        stop_strings: (!generation.stop_strings.is_empty())
            .then(|| generation.stop_strings.clone()),
        min_tokens: generation.min_tokens.unwrap_or(0),
    }
}

fn lora_request(adapter: &AdapterSelection) -> Option<LoraRequest> {
    match adapter {
        AdapterSelection::Base => None,
        AdapterSelection::Adapter {
            name,
            internal_id,
            path,
            load_inplace,
            is_3d_lora_weight,
        } => Some(LoraRequest::new(
            name.clone(),
            *internal_id,
            path.clone(),
            *load_inplace,
            *is_3d_lora_weight,
        )),
    }
}

fn structured_output_params(intent: &StructuredOutputIntent) -> StructuredOutputsParams {
    match intent {
        StructuredOutputIntent::JsonObject {
            disable_any_whitespace,
            whitespace_pattern,
        } => StructuredOutputsParams {
            json_object: Some(true),
            disable_any_whitespace: *disable_any_whitespace,
            whitespace_pattern: whitespace_pattern.clone(),
            ..StructuredOutputsParams::default()
        },
        StructuredOutputIntent::JsonSchema {
            schema,
            disable_any_whitespace,
            disable_additional_properties,
            whitespace_pattern,
        } => StructuredOutputsParams {
            json: Some(schema.clone()),
            disable_any_whitespace: *disable_any_whitespace,
            disable_additional_properties: *disable_additional_properties,
            whitespace_pattern: whitespace_pattern.clone(),
            ..StructuredOutputsParams::default()
        },
        StructuredOutputIntent::Regex(regex) => StructuredOutputsParams {
            regex: Some(regex.clone()),
            ..StructuredOutputsParams::default()
        },
        StructuredOutputIntent::Choice(choice) => StructuredOutputsParams {
            choice: Some(choice.clone()),
            ..StructuredOutputsParams::default()
        },
        StructuredOutputIntent::Grammar(grammar) => StructuredOutputsParams {
            grammar: Some(grammar.clone()),
            ..StructuredOutputsParams::default()
        },
        StructuredOutputIntent::StructuralTag(tag) => StructuredOutputsParams {
            structural_tag: Some(tag.clone()),
            ..StructuredOutputsParams::default()
        },
    }
}

fn generation_from_sampling(
    request_id: &str,
    sampling: SamplingParams,
    decode: TextDecodeOptions,
) -> Result<GenerationPolicy> {
    let SamplingParams {
        temperature,
        top_p,
        top_k,
        seed,
        max_tokens,
        min_tokens,
        logprobs,
        prompt_logprobs,
        min_p,
        frequency_penalty,
        presence_penalty,
        repetition_penalty,
        stop_token_ids,
        ignore_eos,
        logit_bias,
        allowed_token_ids,
        bad_words,
        logprob_token_ids,
        structured_outputs,
        skip_reading_prefix_cache: _,
        uniserve_xargs,
    } = sampling;

    let kv_transfer_params = match uniserve_xargs {
        None => None,
        Some(mut xargs) => {
            let kv_transfer_params = xargs.remove("kv_transfer_params");
            if let Some(key) = xargs.into_keys().next() {
                return Err(ServeError::UnsupportedRuntimeExtension {
                    request_id: request_id.to_string(),
                    key,
                });
            }
            kv_transfer_params
        }
    };

    Ok(GenerationPolicy {
        num_outputs: 1,
        temperature,
        top_p,
        top_k,
        seed,
        max_tokens,
        min_tokens,
        logprobs,
        prompt_logprobs,
        min_p,
        frequency_penalty,
        presence_penalty,
        repetition_penalty,
        stop_token_ids: stop_token_ids.unwrap_or_default(),
        stop_strings: decode.stop_strings.unwrap_or_default(),
        ignore_eos,
        logit_bias,
        allowed_token_ids,
        bad_words: bad_words.unwrap_or_default(),
        logprob_token_ids,
        structured_output: structured_outputs
            .as_ref()
            .and_then(structured_output_intent),
        kv_transfer_params,
        skip_special_tokens: decode.skip_special_tokens,
        include_stop_string_in_output: decode.include_stop_str_in_output,
        output_detail: OutputDetail::default(),
        intermediate: true,
        add_special_tokens: false,
    })
}

fn structured_output_intent(params: &StructuredOutputsParams) -> Option<StructuredOutputIntent> {
    if let Some(schema) = &params.json {
        return Some(StructuredOutputIntent::JsonSchema {
            schema: schema.clone(),
            disable_any_whitespace: params.disable_any_whitespace,
            disable_additional_properties: params.disable_additional_properties,
            whitespace_pattern: params.whitespace_pattern.clone(),
        });
    }
    if let Some(regex) = &params.regex {
        return Some(StructuredOutputIntent::Regex(regex.clone()));
    }
    if let Some(choice) = &params.choice {
        return Some(StructuredOutputIntent::Choice(choice.clone()));
    }
    if let Some(grammar) = &params.grammar {
        return Some(StructuredOutputIntent::Grammar(grammar.clone()));
    }
    if params.json_object == Some(true) {
        return Some(StructuredOutputIntent::JsonObject {
            disable_any_whitespace: params.disable_any_whitespace,
            whitespace_pattern: params.whitespace_pattern.clone(),
        });
    }
    if let Some(tag) = &params.structural_tag {
        return Some(StructuredOutputIntent::StructuralTag(tag.clone()));
    }
    None
}

fn adapter_from_lora(lora: Option<LoraRequest>) -> AdapterSelection {
    match lora {
        None => AdapterSelection::Base,
        Some(lora) => AdapterSelection::Adapter {
            name: lora.lora_name,
            internal_id: lora.lora_int_id,
            path: lora.lora_path,
            load_inplace: lora.load_inplace,
            is_3d_lora_weight: lora.is_3d_lora_weight,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_request_inspection_redacts_prompt_text() {
        let profile = ModelProfile::text_only("test-profile");
        let inspection = PlanInspection {
            request_id: "req".to_string(),
            profile_id: profile.profile_id,
            dialect_id: profile.dialect_id,
            tokenizer_fingerprint: profile.tokenizer_fingerprint,
            capability_snapshot: RuntimeCapabilitySnapshot {
                supports_text: true,
                supports_chat: false,
                supports_multimodal_input: false,
            },
            context: PlannedContextInspection::Text {
                prompt_token_count: 3,
                pre_tokenized: false,
            },
            generation: GenerationPlanInspection {
                num_outputs: 1,
                max_tokens: Some(8),
                temperature: Some(0.7),
                top_p: None,
                top_k: None,
                stop_token_count: 0,
                stop_string_count: 0,
                structured_output: None,
            },
            cache: CachePolicy::default(),
            adapter: AdapterSelection::Base,
            scheduling: SchedulingPolicy::default(),
            output: OutputProcessingPlan {
                visible_text: true,
                reasoning: false,
                tools: false,
                logprobs: false,
                token_ids: false,
            },
        };

        let rendered = serde_json::to_string(&inspection).unwrap();
        assert!(!rendered.contains("secret prompt"));
        assert!(rendered.contains("prompt_token_count"));
    }
}
