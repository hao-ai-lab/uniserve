//! Canonical semantic serving runtime.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod chat;
mod dialect_generation;
pub mod text;

use std::borrow::Borrow;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll};
use std::time::Instant;

use self::chat::{
    AssistantBlockKind, AssistantContentBlock, ChatContent, ChatContentPart, ChatEvent,
    ChatMessage, ChatOptions, ChatRequest, ChatTool, ChatToolChoice, GenerationPromptMode,
    PreparedChatRequest,
};
use self::text::{
    DecodedLogprobs, DecodedPromptLogprobs, DecodedTextEvent, FinishReason, PreparedTextRequest,
    Prompt, SamplingParams, StopReason, TextDecodeOptions, TextRequest,
};
use asynk_strim_attr::{TryYielder, try_stream};
use base64::Engine as _;
use futures::{Stream, StreamExt as _, pin_mut};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use thiserror::Error;
use tokio::sync::{Notify, mpsc};
use uniserve_core::{
    GenerationBehaviorDescriptor, GenerationCapabilityNeeds, GenerationConstraint,
    GenerationRequest, GenerationRuntimeCapabilities,
};
use uniserve_engine_gateway::transport::{GenEvent, GenerationEventStream, GenerationFinishReason};
use uniserve_engine_gateway::{EngineGateway, EngineGatewaySnapshot, GenerationSubmission};
pub use uniserve_engine_gateway::{PublicCommit, PublicModality, SemanticRoot};
use uniserve_model_profile::ModelProfile;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ServeRequestId(String);

impl ServeRequestId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn into_inner(self) -> String {
        self.0
    }
}

impl fmt::Display for ServeRequestId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl Deref for ServeRequestId {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<str> for ServeRequestId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Borrow<str> for ServeRequestId {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl From<String> for ServeRequestId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ServeRequestId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl From<ServeRequestId> for String {
    fn from(value: ServeRequestId) -> Self {
        value.0
    }
}

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct CandidateId(u32);

impl CandidateId {
    pub const PRIMARY: Self = Self(0);

    pub const fn get(self) -> u32 {
        self.0
    }
}

impl From<u32> for CandidateId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<CandidateId> for u32 {
    fn from(value: CandidateId) -> Self {
        value.0
    }
}
pub type ServeEventStream = Pin<Box<dyn Stream<Item = Result<ServeEvent>> + Send>>;

pub type Result<T> = std::result::Result<T, ServeError>;

static NEXT_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);

/// Runtime-local serving errors.
#[derive(Debug, Error)]
pub enum ServeError {
    #[error("request `{request_id}` asks for {requested} outputs; only one output is supported")]
    UnsupportedOutputCount {
        request_id: ServeRequestId,
        requested: u32,
    },
    #[error("request `{request_id}` requires unsupported capability `{capability}`")]
    UnsupportedCapability {
        request_id: ServeRequestId,
        capability: &'static str,
    },
    #[error(
        "request `{request_id}` tokenizer fingerprint `{provided}` does not match profile `{expected}`"
    )]
    TokenizerMismatch {
        request_id: ServeRequestId,
        provided: String,
        expected: String,
    },
    #[error(
        "request `{request_id}` has {prompt_tokens} prompt tokens, exceeding the {max_tokens}-token profile limit"
    )]
    ContextLengthExceeded {
        request_id: ServeRequestId,
        prompt_tokens: usize,
        max_tokens: u32,
    },
    #[error(
        "request `{request_id}` requires {required_tokens} KV tokens, exceeding the {max_tokens}-token runtime limit"
    )]
    ContextCapacityExceeded {
        request_id: ServeRequestId,
        required_tokens: u64,
        max_tokens: u32,
    },
    #[error("execution plan `{request_id}` belongs to another serving runtime")]
    ForeignExecutionPlan { request_id: ServeRequestId },
    #[error("request `{request_id}` is already active")]
    DuplicateRequestId { request_id: ServeRequestId },
    #[error("text runtime error")]
    Text(#[from] crate::text::Error),
    #[error("chat runtime error")]
    Chat(#[from] crate::chat::Error),
    #[error("engine runtime error: {0}")]
    Engine(String),
    #[error("request `{request_id}` cannot be compiled: {message}")]
    Compilation {
        request_id: ServeRequestId,
        message: String,
    },
    #[error("request `{request_id}` output processing failed: {message}")]
    OutputProcessing {
        request_id: ServeRequestId,
        message: String,
    },
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
    pub fn text(request_id: impl Into<ServeRequestId>, prompt: impl Into<String>) -> Self {
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

    pub fn chat(request_id: impl Into<ServeRequestId>, messages: Vec<ChatMessage>) -> Self {
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
            sampling_params,
            decode_options,
            intermediate,
            priority,
            cache_salt,
            add_special_tokens,
            data_parallel_rank,
            trace_context,
            adapter,
        } = text;
        let write_prefix_cache = sampling_params.write_prefix_cache.unwrap_or(true);
        let mut generation =
            generation_from_sampling(&request_id, sampling_params, decode_options)?;
        generation.intermediate = intermediate;
        generation.add_special_tokens = add_special_tokens;
        let model_context = match prompt {
            Prompt::Text(text) => ModelContext::RawPrompt(text),
            Prompt::TokenIds(token_ids) => ModelContext::TokenIds {
                token_ids,
                tokenizer: TokenizerReference::RuntimeProfile,
            },
        };

        Ok(Self {
            request_id: request_id.into(),
            model_context,
            generation,
            modalities: ModalityPolicy::default(),
            adapter,
            cache: CachePolicy {
                salt: cache_salt,
                no_store: !write_prefix_cache,
                ..CachePolicy::default()
            },
            scheduling: SchedulingPolicy {
                priority,
                data_parallel_rank,
                trace_context,
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
            trace_context,
            adapter,
        } = chat;
        let write_prefix_cache = sampling_params.write_prefix_cache.unwrap_or(true);
        let mut generation =
            generation_from_sampling(&request_id, sampling_params, decode_options)?;
        generation.intermediate = intermediate;
        generation.add_special_tokens = add_special_tokens;

        let input_image = messages.iter().any(ChatMessage::has_multimodal);
        Ok(Self {
            request_id: request_id.into(),
            model_context: ModelContext::Chat {
                messages,
                chat_options,
                tools,
                tool_choice,
                documents,
            },
            generation,
            modalities: ModalityPolicy {
                input_image,
                ..ModalityPolicy::default()
            },
            adapter,
            cache: CachePolicy {
                salt: cache_salt,
                no_store: !write_prefix_cache,
                ..CachePolicy::default()
            },
            scheduling: SchedulingPolicy {
                priority,
                data_parallel_rank,
                trace_context,
                ..SchedulingPolicy::default()
            },
            metadata,
        })
    }
}

fn validate_tokenizer_references(request: &ServeRequest, expected: &str) -> Result<()> {
    let validate = |provided: &str| {
        if provided == expected {
            Ok(())
        } else {
            Err(ServeError::TokenizerMismatch {
                request_id: request.request_id.clone(),
                provided: provided.to_string(),
                expected: expected.to_string(),
            })
        }
    };
    match &request.model_context {
        ModelContext::TokenIds {
            tokenizer: TokenizerReference::Fingerprint(provided),
            ..
        } => validate(provided),
        ModelContext::Segments(segments) => {
            for segment in segments {
                if let ContextSegment::TokenIds {
                    tokenizer_fingerprint,
                    ..
                } = segment
                {
                    validate(tokenizer_fingerprint)?;
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn replace_chat_images(
    request_id: &str,
    messages: &mut [ChatMessage],
    marker_text: Option<&str>,
) -> Result<(Vec<ImageInput>, Vec<String>)> {
    let request_fingerprint = format!("{:x}", Sha256::digest(request_id.as_bytes()));
    let mut images = Vec::new();
    let mut placeholders = Vec::new();
    for message in messages {
        let content = match message {
            ChatMessage::System { content }
            | ChatMessage::Developer { content, .. }
            | ChatMessage::User { content }
            | ChatMessage::ToolResponse { content, .. } => content,
            ChatMessage::Assistant { .. } => continue,
        };
        let ChatContent::Parts(parts) = content else {
            continue;
        };
        for part in parts {
            let ChatContentPart::ImageUrl { image_url, .. } = part else {
                continue;
            };
            let placeholder = marker_text.map(str::to_string).unwrap_or_else(|| {
                format!(
                    "[UNISERVE_IMAGE_SLOT_{}_{}]",
                    &request_fingerprint[..16],
                    images.len()
                )
            });
            images.push(ImageInput {
                b64: data_image_payload(request_id, image_url)?,
                placement: None,
            });
            placeholders.push(placeholder.clone());
            *part = ChatContentPart::Text { text: placeholder };
        }
    }
    Ok((images, placeholders))
}

fn data_image_payload(request_id: &str, url: &str) -> Result<String> {
    let (metadata, payload) = url.split_once(',').ok_or_else(|| ServeError::Compilation {
        request_id: request_id.into(),
        message: "image chat requires a data:image/*;base64 URL".to_string(),
    })?;
    if !metadata.starts_with("data:image/") || !metadata.ends_with(";base64") || payload.is_empty()
    {
        return Err(ServeError::Compilation {
            request_id: request_id.into(),
            message: "image chat requires a data:image/*;base64 URL".to_string(),
        });
    }
    base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(|error| ServeError::Compilation {
            request_id: request_id.into(),
            message: format!("image chat contains invalid base64 data: {error}"),
        })?;
    Ok(payload.to_string())
}

fn tokenize_rendered_chat_with_images(
    request_id: &str,
    prompt: Prompt,
    tokenizer: &crate::text::tokenizer::DynTokenizer,
    add_special_tokens: bool,
    markers_in_prompt: bool,
    placeholders: &[String],
    mut images: Vec<ImageInput>,
) -> Result<(Vec<u32>, Vec<ImageInput>)> {
    match prompt {
        Prompt::TokenIds(token_ids) if markers_in_prompt => Ok((token_ids, images)),
        Prompt::TokenIds(_) => Err(ServeError::Compilation {
            request_id: request_id.into(),
            message: "image placement requires a text-rendering chat profile".to_string(),
        }),
        Prompt::Text(rendered) if markers_in_prompt => tokenizer
            .encode(&rendered, add_special_tokens)
            .map(|token_ids| (token_ids, images))
            .map_err(|error| ServeError::Compilation {
                request_id: request_id.into(),
                message: format!("image chat tokenization failed: {error}"),
            }),
        Prompt::Text(rendered) => {
            let mut clean = String::with_capacity(rendered.len());
            let mut cursor = 0;
            let mut image_offsets = Vec::with_capacity(placeholders.len());
            for placeholder in placeholders {
                if rendered.matches(placeholder).count() != 1 {
                    return Err(ServeError::Compilation {
                        request_id: request_id.into(),
                        message: "chat rendering did not preserve one unique slot per input image"
                            .to_string(),
                    });
                }
                let relative = rendered[cursor..].find(placeholder).ok_or_else(|| {
                    ServeError::Compilation {
                        request_id: request_id.into(),
                        message: "chat rendering changed input-image order".to_string(),
                    }
                })?;
                let start = cursor + relative;
                clean.push_str(&rendered[cursor..start]);
                image_offsets.push(clean.len());
                cursor = start + placeholder.len();
            }
            clean.push_str(&rendered[cursor..]);
            let token_ids = tokenizer
                .encode(&clean, add_special_tokens)
                .map_err(|error| ServeError::Compilation {
                    request_id: request_id.into(),
                    message: format!("image chat tokenization failed: {error}"),
                })?;
            for (image, byte_offset) in images.iter_mut().zip(image_offsets) {
                let prefix_tokens = tokenizer
                    .encode(&clean[..byte_offset], add_special_tokens)
                    .map_err(|error| ServeError::Compilation {
                        request_id: request_id.into(),
                        message: format!("image placement tokenization failed: {error}"),
                    })?;
                image.placement =
                    Some(
                        prefix_tokens
                            .len()
                            .try_into()
                            .map_err(|_| ServeError::Compilation {
                                request_id: request_id.into(),
                                message: "image placement exceeds the supported token range"
                                    .to_string(),
                            })?,
                    );
            }
            Ok((token_ids, images))
        }
    }
}

fn semantic_image_count(context: &ModelContext) -> usize {
    match context {
        ModelContext::Segments(segments) => segments
            .iter()
            .filter(|segment| matches!(segment, ContextSegment::Image(_)))
            .count(),
        ModelContext::Chat { messages, .. } => messages
            .iter()
            .map(|message| match message {
                ChatMessage::System { content }
                | ChatMessage::Developer { content, .. }
                | ChatMessage::User { content }
                | ChatMessage::ToolResponse { content, .. } => match content {
                    ChatContent::Text(_) => 0,
                    ChatContent::Parts(parts) => parts
                        .iter()
                        .filter(|part| matches!(part, ChatContentPart::ImageUrl { .. }))
                        .count(),
                },
                ChatMessage::Assistant { .. } => 0,
            })
            .sum(),
        ModelContext::RawPrompt(_) | ModelContext::TokenIds { .. } => 0,
    }
}

fn inspect_rendered_segments(context: &ModelContext) -> Vec<RenderedSegmentInspection> {
    match context {
        ModelContext::RawPrompt(_) => vec![RenderedSegmentInspection {
            origin: RenderedSegmentOrigin::RawPrompt,
            token_start: Some(0),
            token_end: None,
        }],
        ModelContext::TokenIds { token_ids, .. } => vec![RenderedSegmentInspection {
            origin: RenderedSegmentOrigin::Pretokenized,
            token_start: Some(0),
            token_end: Some(token_ids.len()),
        }],
        ModelContext::Chat { messages, .. } => messages
            .iter()
            .enumerate()
            .map(|(index, message)| RenderedSegmentInspection {
                origin: RenderedSegmentOrigin::ChatMessage {
                    index,
                    role: context_role(message.role()),
                },
                token_start: None,
                token_end: None,
            })
            .collect(),
        ModelContext::Segments(segments) => segments
            .iter()
            .enumerate()
            .map(|(index, segment)| RenderedSegmentInspection {
                origin: match segment {
                    ContextSegment::Text { role, .. } => {
                        RenderedSegmentOrigin::SemanticText { index, role: *role }
                    }
                    ContextSegment::TokenIds { .. } => RenderedSegmentOrigin::Pretokenized,
                    ContextSegment::Image(_) => RenderedSegmentOrigin::Image { index },
                },
                token_start: None,
                token_end: None,
            })
            .collect(),
    }
}

fn context_role(role: crate::chat::ChatRole) -> ContextRole {
    match role {
        crate::chat::ChatRole::System => ContextRole::System,
        crate::chat::ChatRole::Developer => ContextRole::Developer,
        crate::chat::ChatRole::User => ContextRole::User,
        crate::chat::ChatRole::Assistant => ContextRole::Assistant,
        crate::chat::ChatRole::ToolResponse => ContextRole::Tool,
    }
}

fn planned_prompt_tokens(context: &PlannedContextInspection) -> usize {
    match context {
        PlannedContextInspection::Text {
            prompt_token_count, ..
        }
        | PlannedContextInspection::Chat {
            prompt_token_count, ..
        }
        | PlannedContextInspection::Segments {
            prompt_token_count, ..
        } => *prompt_token_count,
    }
}

fn planned_generation_request(branch: &PlannedRequest) -> &GenerationRequest {
    match branch {
        PlannedRequest::Text(prepared) => &prepared.submission.request,
        PlannedRequest::Chat(prepared) => &prepared.prepared_text_request.submission.request,
        PlannedRequest::DialectGeneration(prepared) => &prepared.generation,
    }
}

fn planned_max_tokens(branch: &PlannedRequest) -> u32 {
    match branch {
        PlannedRequest::Text(prepared) => prepared
            .submission
            .request
            .max_und_tokens
            .min(u32::MAX as usize) as u32,
        PlannedRequest::Chat(prepared) => prepared
            .prepared_text_request
            .submission
            .request
            .max_und_tokens
            .min(u32::MAX as usize) as u32,
        PlannedRequest::DialectGeneration(prepared) => {
            prepared.generation.max_und_tokens.min(u32::MAX as usize) as u32
        }
    }
}

fn apply_compiled_generation_inspection(
    inspection: &mut GenerationPlanInspection,
    branch: &PlannedRequest,
) {
    inspection.constraint = planned_generation_request(branch).constraint;
    match branch {
        PlannedRequest::Text(prepared) => {
            apply_core_sampling_inspection(inspection, &prepared.submission.request.sampling);
        }
        PlannedRequest::Chat(prepared) => {
            apply_core_sampling_inspection(
                inspection,
                &prepared.prepared_text_request.submission.request.sampling,
            );
        }
        PlannedRequest::DialectGeneration(prepared) => {
            let request = &prepared.generation;
            let sampling = &request.sampling;
            inspection.temperature = Some(sampling.temperature);
            inspection.top_p = Some(sampling.top_p);
            inspection.top_k = Some(sampling.top_k);
            inspection.seed = sampling.seed.and_then(|seed| i64::try_from(seed).ok());
            inspection.min_tokens = sampling.min_tokens.min(u32::MAX as usize) as u32;
            inspection.min_p = sampling.min_p;
            inspection.frequency_penalty = sampling.frequency_penalty;
            inspection.presence_penalty = sampling.presence_penalty;
            inspection.repetition_penalty = sampling.repetition_penalty;
            inspection.ignore_eos = sampling.ignore_eos;
            inspection.logit_bias_count = sampling.logit_bias.len();
            inspection.allowed_token_count =
                sampling.allowed_token_ids.as_ref().map_or(0, Vec::len);
            inspection.bad_word_count = sampling.bad_words_ids.len();
            inspection.image = Some(ImageGenerationPlanInspection::from(&request.image));
        }
    }
}

fn apply_core_sampling_inspection(
    inspection: &mut GenerationPlanInspection,
    sampling: &uniserve_core::SamplingParams,
) {
    inspection.temperature = Some(sampling.temperature);
    inspection.top_p = Some(sampling.top_p);
    inspection.top_k = Some(sampling.top_k);
    inspection.seed = sampling.seed.and_then(|seed| i64::try_from(seed).ok());
    inspection.min_tokens = sampling.min_tokens.min(u32::MAX as usize) as u32;
    inspection.min_p = sampling.min_p;
    inspection.frequency_penalty = sampling.frequency_penalty;
    inspection.presence_penalty = sampling.presence_penalty;
    inspection.repetition_penalty = sampling.repetition_penalty;
    inspection.ignore_eos = sampling.ignore_eos;
    inspection.logit_bias_count = sampling.logit_bias.len();
    inspection.allowed_token_count = sampling.allowed_token_ids.as_ref().map_or(0, Vec::len);
    inspection.bad_word_count = sampling.bad_words_ids.len();
    inspection.image = None;
}

fn compiled_choice_token_counts(branch: &PlannedRequest) -> Vec<usize> {
    let grammar = match branch {
        PlannedRequest::Text(prepared) => prepared.submission.request.grammar.as_ref(),
        PlannedRequest::Chat(prepared) => prepared
            .prepared_text_request
            .submission
            .request
            .grammar
            .as_ref(),
        PlannedRequest::DialectGeneration(prepared) => prepared.generation.grammar.as_ref(),
    };
    match grammar {
        Some(uniserve_core::GrammarSpec::Choice { token_sequences }) => {
            token_sequences.iter().map(Vec::len).collect()
        }
        Some(uniserve_core::GrammarSpec::Compiled { .. }) | None => Vec::new(),
    }
}

fn compile_cache_policy(
    profile: &ModelProfile,
    cache: &CachePolicy,
    generation: &uniserve_core::GenerationCachePolicyDescriptor,
    encoder_pin_count: usize,
) -> CompiledCachePolicy {
    let mut hasher = Sha256::new();
    hasher.update(profile.identity.config_fingerprint.as_bytes());
    if let Some(material) = cache_isolation_material(cache) {
        hasher.update(material.as_bytes());
    }
    CompiledCachePolicy {
        namespace: cache.namespace.clone(),
        key_fingerprint: format!("{:x}", hasher.finalize()),
        read_enabled: generation.read,
        write_enabled: generation.write,
        replayable: cache.replayable,
        encoder_pin_count,
    }
}

fn cache_isolation_material(cache: &CachePolicy) -> Option<String> {
    (cache.namespace.is_some() || cache.salt.is_some()).then(|| {
        let namespace = cache.namespace.as_deref().unwrap_or_default();
        let salt = cache.salt.as_deref().unwrap_or_default();
        format!("{}:{namespace}{}:{salt}", namespace.len(), salt.len())
    })
}

pub(crate) fn cache_isolation_key(cache: &CachePolicy) -> Option<u64> {
    cache_isolation_material(cache).map(|material| {
        material
            .as_bytes()
            .iter()
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
                (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
            })
    })
}

fn resource_bounds(
    request: &ServeRequest,
    branch: &PlannedRequest,
    prompt_tokens: usize,
    grammar_required: bool,
    max_context_tokens: u32,
) -> ResourceBounds {
    let generation = planned_generation_request(branch);
    let context_steps = generation.context.iter().flat_map(|segment| match segment {
        uniserve_core::ContextSegment::Image { ingest, .. } => ingest.steps.clone(),
        uniserve_core::ContextSegment::UndTokens { .. } => Vec::new(),
    });
    let needs = generation
        .behavior
        .capability_needs(&generation.policy, context_steps);
    let mut required_features = [
        (needs.understanding, "understanding"),
        (needs.vision_encode, "vision_encode"),
        (needs.latent_encode, "latent_encode"),
        (needs.image_generation, "image_generation"),
    ]
    .into_iter()
    .filter(|&(needed, _)| needed)
    .map(|(_, feature)| feature.to_string())
    .collect::<Vec<_>>();
    if grammar_required {
        required_features.push("grammar_mask".to_string());
    }
    if !matches!(request.adapter, AdapterSelection::Base) {
        required_features.push("adapter".to_string());
    }
    let expected_kv_tokens = generation.resources.max_kv_tokens as u64;
    let image_count = generation.context_image_count();
    let image_latent_units = generation.resources.max_image_latent_units;
    let scratch_units = generation.resources.max_scratch_units;
    let host_scratch_tokens = generation.resources.max_host_scratch_tokens;
    let encoder_cache_pins = generation.resources.encoder_cache_keys.len();
    let replayable =
        request.cache.replayable && !generation.resources.generated_feedback_makes_non_replayable;
    ResourceBounds {
        max_context_tokens: Some(max_context_tokens),
        prompt_tokens,
        expected_kv_tokens,
        image_count,
        image_latent_units,
        scratch_units,
        host_scratch_tokens,
        encoder_cache_pins,
        grammar_states: usize::from(grammar_required),
        adapter_slots: usize::from(!matches!(request.adapter, AdapterSelection::Base)),
        replayable,
        required_features,
    }
}

/// Ordered semantic input context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelContext {
    RawPrompt(String),
    TokenIds {
        token_ids: Vec<u32>,
        tokenizer: TokenizerReference,
    },
    Chat {
        messages: Vec<ChatMessage>,
        chat_options: ChatOptions,
        tools: Vec<ChatTool>,
        tool_choice: ChatToolChoice,
        documents: Option<Vec<serde_json::Value>>,
    },
    Segments(Vec<ContextSegment>),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenizerReference {
    RuntimeProfile,
    Fingerprint(String),
}

/// One ordered semantic context segment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ContextSegment {
    Text {
        role: ContextRole,
        text: String,
    },
    TokenIds {
        token_ids: Vec<u32>,
        tokenizer_fingerprint: String,
    },
    Image(ImageInput),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContextRole {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageInput {
    pub b64: String,
    pub placement: Option<u32>,
}

/// Semantic generation controls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationPolicy {
    pub constraint: GenerationConstraint,
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
    pub skip_special_tokens: bool,
    pub include_stop_string_in_output: bool,
    pub output_detail: OutputDetail,
    pub intermediate: bool,
    pub add_special_tokens: bool,
    pub image: ImageGenerationPolicy,
}

impl Default for GenerationPolicy {
    fn default() -> Self {
        Self {
            constraint: GenerationConstraint::Default,
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
            skip_special_tokens: true,
            include_stop_string_in_output: false,
            output_detail: OutputDetail::default(),
            intermediate: true,
            add_special_tokens: false,
            image: ImageGenerationPolicy::default(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ImageGenerationPolicy {
    pub resolution: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub steps: Option<u16>,
    pub cfg_text_scale: Option<f32>,
    pub cfg_img_scale: Option<f32>,
    pub cfg_interval: Option<[f32; 2]>,
    pub cfg_renorm_type: Option<String>,
    pub cfg_renorm_min: Option<f32>,
    pub timestep_shift: Option<f32>,
    pub seed: Option<u64>,
    pub negative_prompt: Option<String>,
    pub max_images: Option<u16>,
    pub prompts: Vec<String>,
    pub retain_images: Option<bool>,
    pub image_bias: Option<f32>,
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AdapterSelection {
    #[default]
    Base,
    Adapter {
        name: String,
        internal_id: u64,
        path: String,
        load_inplace: bool,
        is_3d_lora_weight: bool,
    },
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
    runtime_id: u64,
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
    Text(Box<PreparedTextRequest>),
    Chat(Box<PreparedChatRequest>),
    DialectGeneration(Box<PreparedDialectRequest>),
}

#[derive(Debug)]
struct PreparedDialectRequest {
    generation: GenerationRequest,
    chat_output: Option<ChatRequest>,
}

/// Sanitized plan inspection form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanInspection {
    pub runtime_id: u64,
    pub request_id: ServeRequestId,
    pub profile_id: String,
    pub dialect_id: String,
    pub tokenizer_fingerprint: String,
    pub config_fingerprint: String,
    pub capability_snapshot: RuntimeCapabilitySnapshot,
    pub context: PlannedContextInspection,
    pub rendered_segments: Vec<RenderedSegmentInspection>,
    pub render: RenderPlanInspection,
    pub generation: GenerationPlanInspection,
    pub dialect: Option<DialectPlanInspection>,
    pub cache: CompiledCachePolicy,
    pub adapter: AdapterSelection,
    pub scheduling: SchedulingPolicy,
    pub resources: ResourceBounds,
    pub output: OutputProcessingPlan,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RenderPlanInspection {
    pub renderer_id: String,
    pub chat_template_override_fingerprint: Option<String>,
}

impl PlanInspection {
    pub fn context_prompt_tokens(&self) -> usize {
        planned_prompt_tokens(&self.context)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeCapabilitySnapshot {
    pub supports_text: bool,
    pub supports_chat: bool,
    pub supports_multimodal_input: bool,
    pub supports_image_output: bool,
    pub supports_structured_output: bool,
    pub supports_logprobs: bool,
    pub supports_prefix_cache: bool,
    pub supports_adapters: bool,
    pub engine_count: usize,
    pub max_model_tokens: u32,
    pub model_dtype: uniserve_core::ModelDtype,
    pub generation_runtime: GenerationRuntimeCapabilities,
}

impl RuntimeCapabilitySnapshot {
    fn resolve(profile: &ModelProfile, gateway: &EngineGatewaySnapshot) -> Self {
        Self {
            supports_text: profile.modalities.text_input && profile.modalities.text_output,
            supports_chat: profile.modalities.chat_input,
            supports_multimodal_input: profile.modalities.image_input,
            supports_image_output: profile.modalities.image_output,
            supports_structured_output: profile.features.structured_output,
            supports_logprobs: profile.features.logprobs,
            supports_prefix_cache: profile.features.prefix_cache,
            supports_adapters: profile.features.adapters,
            engine_count: gateway.engine_count,
            max_model_tokens: gateway.max_model_len,
            model_dtype: gateway.model_dtype,
            generation_runtime: gateway.generation_capabilities.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RenderedSegmentInspection {
    pub origin: RenderedSegmentOrigin,
    pub token_start: Option<usize>,
    pub token_end: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenderedSegmentOrigin {
    RawPrompt,
    Pretokenized,
    ChatMessage { index: usize, role: ContextRole },
    ProfileSystem,
    ProfileAssistantPrefix,
    SemanticText { index: usize, role: ContextRole },
    Image { index: usize },
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
    Segments {
        segment_count: usize,
        prompt_token_count: usize,
        image_count: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationPlanInspection {
    pub constraint: GenerationConstraint,
    pub num_outputs: u32,
    pub candidate_ids: Vec<CandidateId>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub seed: Option<i64>,
    pub min_tokens: u32,
    pub min_p: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub repetition_penalty: f32,
    pub ignore_eos: bool,
    pub logit_bias_count: usize,
    pub allowed_token_count: usize,
    pub bad_word_count: usize,
    pub stop_token_count: usize,
    pub stop_string_count: usize,
    pub structured_output: Option<StructuredOutputIntent>,
    pub compiled_choice_token_counts: Vec<usize>,
    pub grammar_state_required: bool,
    pub image: Option<ImageGenerationPlanInspection>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageGenerationPlanInspection {
    pub width: u32,
    pub height: u32,
    pub steps: u16,
    pub cfg_text_scale: f32,
    pub cfg_img_scale: f32,
    pub cfg_renorm_type: String,
    pub cfg_renorm_min: f32,
    pub cfg_interval: (f32, f32),
    pub timestep_shift: f32,
    pub seed: Option<u64>,
    pub max_images: u16,
    pub image_prompt_count: usize,
    pub retain_images: bool,
}

impl From<&uniserve_core::ImageParams> for ImageGenerationPlanInspection {
    fn from(image: &uniserve_core::ImageParams) -> Self {
        Self {
            width: image.width,
            height: image.height,
            steps: image.steps,
            cfg_text_scale: image.cfg_text_scale,
            cfg_img_scale: image.cfg_img_scale,
            cfg_renorm_type: image.cfg_renorm_type.clone(),
            cfg_renorm_min: image.cfg_renorm_min,
            cfg_interval: image.cfg_interval,
            timestep_shift: image.timestep_shift,
            seed: image.seed,
            max_images: image.max_images,
            image_prompt_count: image.image_prompts.len(),
            retain_images: image.retain_images,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DialectPlanInspection {
    pub id: String,
    pub image_ingest: uniserve_core::ImageIngestRecipe,
    pub generation_policy: uniserve_core::GenerationPolicyDescriptor,
    pub image_defaults: uniserve_model_profile::dialect::ImageGenerationDefaults,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledCachePolicy {
    pub namespace: Option<String>,
    pub key_fingerprint: String,
    pub read_enabled: bool,
    pub write_enabled: bool,
    pub replayable: bool,
    pub encoder_pin_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceBounds {
    pub max_context_tokens: Option<u32>,
    pub prompt_tokens: usize,
    pub expected_kv_tokens: u64,
    pub image_count: usize,
    pub image_latent_units: u64,
    pub scratch_units: u64,
    pub host_scratch_tokens: u64,
    pub encoder_cache_pins: usize,
    pub grammar_states: usize,
    pub adapter_slots: usize,
    pub replayable: bool,
    pub required_features: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputProcessingPlan {
    pub visible_text: bool,
    pub reasoning: bool,
    pub tools: bool,
    pub logprobs: bool,
    pub token_ids: bool,
    pub skip_special_tokens: bool,
    pub include_stop_string: bool,
    pub reasoning_parser: String,
    pub tool_parser: String,
    pub hidden_text_visible: bool,
}

/// Single-model semantic runtime.
pub struct ServingRuntime {
    runtime_id: u64,
    profile: ModelProfile,
    gateway: EngineGateway,
    chat: crate::chat::ChatRuntime,
    metrics: Arc<RuntimeLifecycleMetrics>,
    requests: Arc<RuntimeRequestRegistry>,
    compilations: Arc<CompilationTracker>,
}

#[derive(Default)]
struct CompilationTracker {
    active: AtomicUsize,
    changed: Notify,
}

impl CompilationTracker {
    fn begin(self: &Arc<Self>) -> ActiveCompilation {
        self.active.fetch_add(1, Ordering::AcqRel);
        ActiveCompilation {
            tracker: Arc::clone(self),
        }
    }

    async fn drain(&self) {
        loop {
            let changed = self.changed.notified();
            if self.active.load(Ordering::Acquire) == 0 {
                return;
            }
            changed.await;
        }
    }
}

struct ActiveCompilation {
    tracker: Arc<CompilationTracker>,
}

impl Drop for ActiveCompilation {
    fn drop(&mut self) {
        if self.tracker.active.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.tracker.changed.notify_waiters();
        }
    }
}

fn validate_request_capabilities_for_profile(
    profile: &ModelProfile,
    request: &ServeRequest,
) -> Result<()> {
    let reject = |capability| ServeError::UnsupportedCapability {
        request_id: request.request_id.clone(),
        capability,
    };
    if request.modalities.input_text && !profile.modalities.text_input {
        return Err(reject("text_input"));
    }
    if matches!(request.model_context, ModelContext::Chat { .. }) && !profile.modalities.chat_input
    {
        return Err(reject("chat_input"));
    }
    if request.modalities.input_image && !profile.modalities.image_input {
        return Err(reject("image_input"));
    }
    if request.modalities.output_text && !profile.modalities.text_output {
        return Err(reject("text_output"));
    }
    if request.modalities.output_image && !profile.modalities.image_output {
        return Err(reject("image_output"));
    }
    if request.generation.structured_output.is_some() && !profile.features.structured_output {
        return Err(reject("structured_output"));
    }
    if matches!(
        &request.model_context,
        ModelContext::Chat { chat_options, .. } if chat_options.chat_template.is_some()
    ) && !profile.render.request_chat_template_override_allowed
    {
        return Err(reject("request_chat_template_override"));
    }
    if (request.generation.logprobs.is_some() || request.generation.prompt_logprobs.is_some())
        && !profile.features.logprobs
    {
        return Err(reject("logprobs"));
    }
    if (!matches!(request.adapter, AdapterSelection::Base)) && !profile.features.adapters {
        return Err(reject("adapter"));
    }
    if request.cache.bypass_read && !profile.features.prefix_cache {
        return Err(reject("prefix_cache"));
    }
    if request.scheduling.deadline_ms.is_some() {
        return Err(reject("scheduling_deadline"));
    }
    if request.modalities.output_image
        && request.generation.constraint == GenerationConstraint::UndOnly
    {
        return Err(reject("image_output_with_und_only_constraint"));
    }
    if request.modalities.input_image
        && !request.modalities.output_image
        && request.generation.constraint != GenerationConstraint::UndOnly
    {
        return Err(reject(
            "image_input_text_output_requires_und_only_constraint",
        ));
    }
    if request.modalities.input_image && semantic_image_count(&request.model_context) == 0 {
        return Err(reject("declared_image_input_without_image_segment"));
    }
    validate_tokenizer_references(request, profile.tokenizer_fingerprint())?;
    Ok(())
}

impl ServingRuntime {
    pub fn new(
        profile: ModelProfile,
        gateway: EngineGateway,
        text_backend: crate::text::DynTextBackend,
        chat_backend: crate::chat::DynChatBackend,
    ) -> Self {
        let gateway_max_model_tokens = gateway.snapshot().max_model_len;
        let max_model_tokens = profile
            .context_limits
            .max_model_tokens
            .map_or(gateway_max_model_tokens, |profile_limit| {
                profile_limit.min(gateway_max_model_tokens)
            });
        let text = crate::text::TextRuntime::new(&gateway, text_backend)
            .with_max_model_len(max_model_tokens);
        let chat = crate::chat::ChatRuntime::new(text, chat_backend);
        Self {
            runtime_id: NEXT_RUNTIME_ID.fetch_add(1, Ordering::Relaxed),
            profile,
            gateway,
            chat,
            metrics: Arc::new(RuntimeLifecycleMetrics::default()),
            requests: Arc::new(RuntimeRequestRegistry::default()),
            compilations: Arc::new(CompilationTracker::default()),
        }
    }

    pub fn from_backends(
        gateway: EngineGateway,
        text_backend: crate::text::DynTextBackend,
        chat_backend: crate::chat::DynChatBackend,
    ) -> Self {
        let profile = ModelProfile::loaded(text_backend.model_id());
        Self::new(profile, gateway, text_backend, chat_backend)
    }

    #[allow(
        clippy::clone_on_ref_ptr,
        reason = "clone performs an Arc trait-object upcast from ChatTextBackend to TextBackend"
    )]
    pub fn from_shared_backend(
        gateway: EngineGateway,
        backend: crate::chat::DynChatTextBackend,
    ) -> Self {
        let text_backend: crate::text::DynTextBackend = backend.clone();
        Self::from_backends(gateway, text_backend, backend)
    }

    pub fn profile(&self) -> &ModelProfile {
        &self.profile
    }

    pub fn with_generation_dialect(
        mut self,
        dialect: uniserve_model_profile::dialect::GenerationDialectProfile,
    ) -> Self {
        self.profile = self.profile.with_generation_dialect(dialect);
        self
    }

    pub fn with_tool_call_parser(mut self, selection: crate::chat::ParserSelection) -> Self {
        self.profile.parsers.tools = selection.to_string();
        self.chat = self.chat.with_tool_call_parser(selection);
        self
    }

    pub fn with_reasoning_parser(mut self, selection: crate::chat::ParserSelection) -> Self {
        self.profile.parsers.reasoning = selection.to_string();
        self.chat = self.chat.with_reasoning_parser(selection);
        self
    }

    pub fn with_max_model_len(mut self, max_model_len: u32) -> Self {
        self.profile.context_limits.max_model_tokens = Some(max_model_len);
        self.chat = self.chat.with_text_max_model_len(max_model_len);
        self
    }

    pub fn tokenizer(&self) -> crate::text::tokenizer::DynTokenizer {
        self.chat.text().tokenizer()
    }

    pub fn metrics_snapshot(&self) -> RuntimeMetricsSnapshot {
        let mut snapshot = self.metrics.snapshot();
        snapshot.active = self.requests.active_count() as u64;
        snapshot
    }

    pub fn request_stats(&self, request_id: &str) -> Option<RequestStatsSnapshot> {
        self.requests.stats(request_id)
    }

    pub async fn drain_request(&self, request_id: &str) -> Option<RequestStatsSnapshot> {
        self.requests.drain_request(request_id).await
    }

    pub async fn drain(&self) {
        self.requests.drain().await;
    }

    pub fn compile(&self, request: ServeRequest) -> Result<ExecutionPlan> {
        if tokio::runtime::Handle::try_current().is_ok() {
            let request_id = request.request_id.clone();
            return std::thread::scope(|scope| {
                scope
                    .spawn(move || self.compile_with_owned_runtime(request))
                    .join()
                    .unwrap_or_else(|_| {
                        Err(ServeError::Compilation {
                            request_id,
                            message: "synchronous compilation worker panicked".to_string(),
                        })
                    })
            });
        }
        self.compile_with_owned_runtime(request)
    }

    fn compile_with_owned_runtime(&self, request: ServeRequest) -> Result<ExecutionPlan> {
        let request_id = request.request_id.clone();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| ServeError::Compilation {
                request_id,
                message: format!("failed to initialize compilation runtime: {error}"),
            })?;
        runtime.block_on(self.compile_on_worker(request))
    }

    pub async fn compile_async(&self, request: ServeRequest) -> Result<ExecutionPlan> {
        let request_id = request.request_id.clone();
        let runtime = self.clone_for_compilation();
        let active = self.compilations.begin();
        tokio::task::spawn_blocking(move || {
            let _active = active;
            {
                let runtime = runtime;
                runtime.compile_with_owned_runtime(request)
            }
        })
        .await
        .map_err(|error| ServeError::Compilation {
            request_id,
            message: format!("compilation worker failed: {error}"),
        })?
    }

    async fn compile_on_worker(&self, request: ServeRequest) -> Result<ExecutionPlan> {
        if request.generation.num_outputs != 1 {
            return Err(ServeError::UnsupportedOutputCount {
                request_id: request.request_id,
                requested: request.generation.num_outputs,
            });
        }

        let capability_snapshot =
            RuntimeCapabilitySnapshot::resolve(&self.profile, &self.gateway.snapshot());
        self.validate_request_capabilities(&request, &capability_snapshot)?;
        let mut generation_inspection = GenerationPlanInspection {
            constraint: request.generation.constraint,
            num_outputs: request.generation.num_outputs,
            candidate_ids: vec![CandidateId::PRIMARY],
            max_tokens: request.generation.max_tokens,
            temperature: request.generation.temperature,
            top_p: request.generation.top_p,
            top_k: request.generation.top_k,
            seed: request.generation.seed,
            min_tokens: request.generation.min_tokens.unwrap_or(0),
            min_p: request.generation.min_p.unwrap_or(0.0),
            frequency_penalty: request.generation.frequency_penalty.unwrap_or(0.0),
            presence_penalty: request.generation.presence_penalty.unwrap_or(0.0),
            repetition_penalty: request.generation.repetition_penalty.unwrap_or(1.0),
            ignore_eos: request.generation.ignore_eos,
            logit_bias_count: request
                .generation
                .logit_bias
                .as_ref()
                .map_or(0, HashMap::len),
            allowed_token_count: request
                .generation
                .allowed_token_ids
                .as_ref()
                .map_or(0, Vec::len),
            bad_word_count: request.generation.bad_words.len(),
            stop_token_count: request.generation.stop_token_ids.len(),
            stop_string_count: request.generation.stop_strings.len(),
            structured_output: request.generation.structured_output.clone(),
            compiled_choice_token_counts: Vec::new(),
            grammar_state_required: request.generation.structured_output.is_some(),
            image: None,
        };
        let mut output = OutputProcessingPlan {
            visible_text: request.modalities.output_text,
            reasoning: matches!(&request.model_context, ModelContext::Chat { .. }),
            tools: matches!(&request.model_context, ModelContext::Chat { tools, .. } if !tools.is_empty()),
            logprobs: request.generation.logprobs.is_some()
                || request.generation.prompt_logprobs.is_some(),
            token_ids: matches!(request.generation.output_detail, OutputDetail::Tokens),
            skip_special_tokens: request.generation.skip_special_tokens,
            include_stop_string: request.generation.include_stop_string_in_output,
            reasoning_parser: self.profile.parsers.reasoning.clone(),
            tool_parser: self.profile.parsers.tools.clone(),
            hidden_text_visible: self.profile.parsers.expose_hidden_content,
        };
        let mut rendered_segments = inspect_rendered_segments(&request.model_context);
        let render = RenderPlanInspection {
            renderer_id: self.profile.render.renderer_id.clone(),
            chat_template_override_fingerprint: match &request.model_context {
                ModelContext::Chat { chat_options, .. } => chat_options
                    .chat_template
                    .as_ref()
                    .map(|template| format!("{:x}", Sha256::digest(template.as_bytes()))),
                _ => None,
            },
        };

        let (branch, context) = match request.model_context.clone() {
            ModelContext::RawPrompt(prompt) => {
                if request.modalities.output_image {
                    self.compile_dialect_branch(&request, &capability_snapshot)?
                } else {
                    let text_request = self.to_text_request(&request, Prompt::Text(prompt));
                    let prepared = self.chat.text().compile(text_request)?;
                    let prompt_token_count = prepared.submission.request.prompt_token_count();
                    (
                        PlannedRequest::Text(Box::new(prepared)),
                        PlannedContextInspection::Text {
                            prompt_token_count,
                            pre_tokenized: false,
                        },
                    )
                }
            }
            ModelContext::TokenIds {
                token_ids,
                tokenizer,
            } => {
                if let TokenizerReference::Fingerprint(provided) = tokenizer
                    && provided != self.profile.tokenizer_fingerprint()
                {
                    return Err(ServeError::TokenizerMismatch {
                        request_id: request.request_id.clone(),
                        provided,
                        expected: self.profile.tokenizer_fingerprint().to_string(),
                    });
                }
                let text_request = self.to_text_request(&request, Prompt::TokenIds(token_ids));
                let prepared = self.chat.text().compile(text_request)?;
                let prompt_token_count = prepared.submission.request.prompt_token_count();
                (
                    PlannedRequest::Text(Box::new(prepared)),
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
                if request.modalities.input_image || request.modalities.output_image {
                    let (branch, context, actual_rendered_segments) = self
                        .compile_multimodal_chat_branch(
                            &request,
                            messages,
                            chat_options,
                            tools,
                            tool_choice,
                            documents,
                            &capability_snapshot,
                        )
                        .await?;
                    rendered_segments = actual_rendered_segments;
                    (branch, context)
                } else {
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
                        .submission
                        .request
                        .prompt_token_count();
                    let multimodal = prepared.chat_request.has_multimodal();
                    let message_count = prepared.chat_request.messages.len();
                    (
                        PlannedRequest::Chat(Box::new(prepared)),
                        PlannedContextInspection::Chat {
                            message_count,
                            prompt_token_count,
                            multimodal,
                        },
                    )
                }
            }
            ModelContext::Segments(_) => {
                self.compile_dialect_branch(&request, &capability_snapshot)?
            }
        };

        output.skip_special_tokens = match &branch {
            PlannedRequest::Text(prepared) => {
                prepared.text_request.decode_options.skip_special_tokens
            }
            PlannedRequest::Chat(prepared) => {
                prepared.chat_request.decode_options.skip_special_tokens
            }
            PlannedRequest::DialectGeneration(prepared) => prepared
                .chat_output
                .as_ref()
                .map_or(request.generation.skip_special_tokens, |chat| {
                    chat.decode_options.skip_special_tokens
                }),
        };

        generation_inspection.max_tokens = Some(planned_max_tokens(&branch));
        apply_compiled_generation_inspection(&mut generation_inspection, &branch);
        generation_inspection.compiled_choice_token_counts = compiled_choice_token_counts(&branch);
        let prompt_tokens = planned_prompt_tokens(&context);
        let max_tokens = self.effective_max_model_tokens(&capability_snapshot);
        if prompt_tokens >= max_tokens as usize {
            return Err(ServeError::ContextLengthExceeded {
                request_id: request.request_id,
                prompt_tokens,
                max_tokens,
            });
        }
        let encoder_pin_count = match &branch {
            PlannedRequest::DialectGeneration(prepared) => {
                prepared.generation.resources.encoder_cache_keys.len()
            }
            PlannedRequest::Text(_) | PlannedRequest::Chat(_) => 0,
        };
        let cache = compile_cache_policy(
            &self.profile,
            &request.cache,
            &planned_generation_request(&branch).cache,
            encoder_pin_count,
        );
        let resources = resource_bounds(
            &request,
            &branch,
            prompt_tokens,
            generation_inspection.grammar_state_required,
            max_tokens,
        );
        if resources.expected_kv_tokens > u64::from(max_tokens) {
            return Err(ServeError::ContextCapacityExceeded {
                request_id: request.request_id,
                required_tokens: resources.expected_kv_tokens,
                max_tokens,
            });
        }
        let dialect =
            self.profile
                .generation_dialect
                .as_ref()
                .map(|dialect| DialectPlanInspection {
                    id: dialect.id.clone(),
                    image_ingest: dialect.image_ingest.clone(),
                    generation_policy: dialect.generation_policy.clone(),
                    image_defaults: dialect.image_defaults.clone(),
                });

        let inspection = PlanInspection {
            runtime_id: self.runtime_id,
            request_id: request.request_id.clone(),
            profile_id: self.profile.profile_id().to_string(),
            dialect_id: self.profile.dialect_id().to_string(),
            tokenizer_fingerprint: self.profile.tokenizer_fingerprint().to_string(),
            config_fingerprint: self.profile.identity.config_fingerprint.clone(),
            capability_snapshot: capability_snapshot.clone(),
            context,
            rendered_segments,
            render,
            generation: generation_inspection,
            dialect,
            cache,
            adapter: request.adapter,
            scheduling: request.scheduling,
            resources,
            output,
        };

        Ok(ExecutionPlan {
            runtime_id: self.runtime_id,
            request_id: request.request_id,
            profile: self.profile.clone(),
            capability_snapshot,
            branch,
            inspection,
        })
    }

    pub async fn serve(&self, request: ServeRequest) -> Result<ServeEventStream> {
        let request_id = request.request_id.clone();
        let compile_started = Instant::now();
        if !self.requests.register(
            request_id.clone(),
            self.profile.profile_id().to_string(),
            self.profile.dialect_id().to_string(),
            0,
            RequestLifecycleState::Compiling,
        ) {
            self.metrics.rejected.fetch_add(1, Ordering::Relaxed);
            return Err(ServeError::DuplicateRequestId { request_id });
        }
        let plan_result = tokio::select! {
            terminal = self.requests.wait_for_control(&request_id) => {
                return Ok(self.control_event_stream(request_id, terminal));
            }
            plan = self.compile_async(request) => plan,
        };
        let plan = match plan_result {
            Ok(plan) => plan,
            Err(error) => {
                self.metrics.rejected.fetch_add(1, Ordering::Relaxed);
                self.requests.complete(
                    &request_id,
                    RequestLifecycleState::Rejected,
                    compile_started.elapsed().as_micros() as u64,
                );
                return Err(error);
            }
        };
        let compile_duration_us = compile_started.elapsed().as_micros() as u64;
        self.requests
            .mark_submitting(&request_id, compile_duration_us);
        Box::pin(self.execute_inner(plan, true, compile_duration_us)).await
    }

    pub async fn execute(&self, plan: ExecutionPlan) -> Result<ServeEventStream> {
        Box::pin(self.execute_inner(plan, false, 0)).await
    }

    async fn execute_inner(
        &self,
        plan: ExecutionPlan,
        registered: bool,
        compile_duration_us: u64,
    ) -> Result<ServeEventStream> {
        if plan.runtime_id != self.runtime_id {
            self.metrics.rejected.fetch_add(1, Ordering::Relaxed);
            self.requests
                .record_terminal(RequestStatsSnapshot::terminal(
                    plan.request_id.clone(),
                    plan.inspection.profile_id.clone(),
                    plan.inspection.dialect_id.clone(),
                    RequestLifecycleState::Rejected,
                    compile_duration_us,
                ));
            return Err(ServeError::ForeignExecutionPlan {
                request_id: plan.request_id,
            });
        }
        let request_id = plan.request_id.clone();
        if !registered
            && !self.requests.register(
                request_id.clone(),
                plan.inspection.profile_id.clone(),
                plan.inspection.dialect_id.clone(),
                compile_duration_us,
                RequestLifecycleState::Submitting,
            )
        {
            self.metrics.rejected.fetch_add(1, Ordering::Relaxed);
            return Err(ServeError::DuplicateRequestId { request_id });
        }
        if let Some(terminal) = self.requests.control_terminal(&request_id) {
            return Ok(self.control_event_stream(request_id, terminal));
        }
        let event_context = EventContext {
            profile_id: plan.inspection.profile_id.clone(),
            dialect_id: plan.inspection.dialect_id.clone(),
            compile_duration_us,
            cache: plan.inspection.cache.clone(),
            resources: plan.inspection.resources.clone(),
            skip_special_tokens: plan.inspection.output.skip_special_tokens,
            metrics: Arc::clone(&self.metrics),
        };
        let stream_result: Result<ServeEventStream> = match plan.branch {
            PlannedRequest::Text(prepared) => {
                let request_id = prepared.text_request.request_id.clone();
                match self
                    .chat
                    .text()
                    .generate_prepared(&self.gateway, *prepared)
                    .await
                {
                    Ok(stream) => Ok(Box::pin(text_event_stream(
                        request_id.into(),
                        event_context,
                        stream,
                    )) as ServeEventStream),
                    Err(error) => Err(error.into()),
                }
            }
            PlannedRequest::Chat(prepared) => {
                let request_id = prepared.chat_request.request_id.clone();
                match Box::pin(self.chat.chat_prepared(&self.gateway, *prepared)).await {
                    Ok(stream) => Ok(Box::pin(chat_event_stream(
                        request_id.into(),
                        event_context,
                        stream,
                    )) as ServeEventStream),
                    Err(error) => Err(error.into()),
                }
            }
            PlannedRequest::DialectGeneration(prepared) => {
                let request_id = request_id.clone();
                let PreparedDialectRequest {
                    generation: request,
                    mut chat_output,
                } = *prepared;
                let prompt_token_ids = request.prompt_token_ids();
                let tokenizer = self.chat.text().tokenizer();
                let event_tokenizer = Arc::clone(&tokenizer);
                let prompt_logprobs_requested = request.sampling.prompt_logprobs_requested();
                let generated_logprobs_requested = request.sampling.generated_logprobs_requested();
                let decode_options = TextDecodeOptions {
                    skip_special_tokens: plan.inspection.output.skip_special_tokens,
                    include_stop_str_in_output: plan.inspection.output.include_stop_string,
                    stop_strings: (!request.stop_strings.is_empty())
                        .then(|| request.stop_strings.clone()),
                    min_tokens: request.sampling.min_tokens.min(u32::MAX as usize) as u32,
                };
                let processors = (|| -> Result<_> {
                    let chat_processor = chat_output
                        .as_mut()
                        .map(|chat_request| self.chat.new_output_processor(chat_request))
                        .transpose()?;
                    let dialect = plan.profile.generation_dialect.as_ref().ok_or_else(|| {
                        ServeError::Compilation {
                            request_id: request_id.clone(),
                            message: "the execution plan has no generation dialect".to_string(),
                        }
                    })?;
                    let dialect_processor = dialect_generation::DialectStreamProcessor::new(
                        tokenizer,
                        dialect,
                        &prompt_token_ids,
                        chat_processor.is_none(),
                    )
                    .map_err(|error| ServeError::Compilation {
                        request_id: request_id.clone(),
                        message: error.to_string(),
                    })?;
                    Ok((dialect_processor, chat_processor))
                })();
                match processors {
                    Err(error) => Err(error),
                    Ok((processor, chat_processor)) => {
                        let mut submission =
                            GenerationSubmission::new(request_id.to_string(), request);
                        submission.data_parallel_rank =
                            plan.inspection.scheduling.data_parallel_rank;
                        submission.trace_headers =
                            (!plan.inspection.scheduling.trace_context.is_empty()).then(|| {
                                plan.inspection
                                    .scheduling
                                    .trace_context
                                    .iter()
                                    .map(|(key, value)| (key.clone(), value.clone()))
                                    .collect()
                            });
                        match self.gateway.submit_generation(submission).await {
                            Ok(stream) => Ok(Box::pin(dialect_event_stream(
                                request_id,
                                event_context,
                                prompt_token_ids,
                                event_tokenizer,
                                prompt_logprobs_requested,
                                generated_logprobs_requested,
                                decode_options,
                                processor,
                                chat_processor,
                                stream,
                            )) as ServeEventStream),
                            Err(error) => Err(ServeError::Engine(error.to_string())),
                        }
                    }
                }
            }
        };
        if let Some(terminal) = self.requests.control_terminal(&request_id) {
            if let Ok(stream) = stream_result {
                drop(stream);
            }
            self.apply_engine_control(&request_id, terminal).await;
            return Ok(self.control_event_stream(request_id, terminal));
        }
        let stream = match stream_result {
            Ok(stream) => stream,
            Err(error) => {
                self.metrics.failed.fetch_add(1, Ordering::Relaxed);
                self.requests.complete(
                    &request_id,
                    RequestLifecycleState::Failed,
                    compile_duration_us,
                );
                return Err(error);
            }
        };
        if !self.requests.accept(&request_id) {
            let terminal = self
                .requests
                .control_terminal(&request_id)
                .unwrap_or(LifecycleTerminal::Cancelled);
            drop(stream);
            self.apply_engine_control(&request_id, terminal).await;
            return Ok(self.control_event_stream(request_id, terminal));
        }
        self.metrics.accepted.fetch_add(1, Ordering::Relaxed);
        let stream = Box::pin(control_aware_event_stream(
            request_id.clone(),
            Arc::clone(&self.requests),
            stream,
        )) as ServeEventStream;
        Ok(Box::pin(LifecycleTrackedStream::new(
            request_id,
            stream,
            Arc::clone(&self.metrics),
            Arc::clone(&self.requests),
        )))
    }

    fn control_event_stream(
        &self,
        request_id: ServeRequestId,
        terminal: LifecycleTerminal,
    ) -> ServeEventStream {
        let event = match terminal {
            LifecycleTerminal::Aborted => ServeEvent::Aborted {
                request_id: request_id.clone(),
            },
            LifecycleTerminal::Cancelled
            | LifecycleTerminal::Finished
            | LifecycleTerminal::Rejected
            | LifecycleTerminal::Failed => ServeEvent::Cancelled {
                request_id: request_id.clone(),
            },
        };
        let inner = Box::pin(futures::stream::iter([Ok(event)])) as ServeEventStream;
        Box::pin(LifecycleTrackedStream::new(
            request_id,
            inner,
            Arc::clone(&self.metrics),
            Arc::clone(&self.requests),
        ))
    }

    pub async fn cancel(
        &self,
        request_id: impl Into<ServeRequestId>,
    ) -> std::result::Result<(), ServeControlError> {
        let request_id = request_id.into();
        let engine_may_own_request = self
            .requests
            .mark_control(&request_id, RequestLifecycleState::Cancelling);
        if engine_may_own_request {
            self.gateway
                .cancel(&request_id)
                .await
                .map_err(|message| ServeControlError::Engine(message.to_string()))?;
        }
        Ok(())
    }

    pub async fn abort(
        &self,
        request_id: impl Into<ServeRequestId>,
        reason: AbortReason,
    ) -> std::result::Result<(), ServeControlError> {
        let request_id = request_id.into();
        let engine_may_own_request = self
            .requests
            .mark_control(&request_id, RequestLifecycleState::Aborting);
        if engine_may_own_request {
            self.gateway
                .abort(&request_id)
                .await
                .map_err(|message| ServeControlError::Engine(message.to_string()))?;
        }
        let _ = reason;
        Ok(())
    }

    async fn apply_engine_control(&self, request_id: &str, terminal: LifecycleTerminal) {
        let result = match terminal {
            LifecycleTerminal::Aborted => self.gateway.abort(request_id).await,
            LifecycleTerminal::Cancelled
            | LifecycleTerminal::Finished
            | LifecycleTerminal::Rejected
            | LifecycleTerminal::Failed => self.gateway.cancel(request_id).await,
        };
        if let Err(error) = result {
            tracing::warn!(%request_id, %error, "failed to apply pending request control after submission");
        }
    }

    pub async fn shutdown(self) -> std::result::Result<(), uniserve_engine_gateway::Error> {
        self.drain().await;
        self.compilations.drain().await;
        self.gateway.shutdown().await
    }

    fn clone_for_compilation(&self) -> Self {
        Self {
            runtime_id: self.runtime_id,
            profile: self.profile.clone(),
            gateway: self.gateway.clone(),
            chat: self.chat.clone(),
            metrics: Arc::clone(&self.metrics),
            requests: Arc::clone(&self.requests),
            compilations: Arc::clone(&self.compilations),
        }
    }

    fn validate_request_capabilities(
        &self,
        request: &ServeRequest,
        capabilities: &RuntimeCapabilitySnapshot,
    ) -> Result<()> {
        validate_request_capabilities_for_profile(&self.profile, request)?;
        let reject = |capability| ServeError::UnsupportedCapability {
            request_id: request.request_id.clone(),
            capability,
        };
        let needs = if let Some(dialect) = &self.profile.generation_dialect {
            let behavior = GenerationBehaviorDescriptor::resolve(
                request.generation.constraint,
                &dialect.generation_policy,
            );
            let context_steps = if request.modalities.input_image {
                dialect.image_ingest.steps.clone()
            } else {
                Vec::new()
            };
            behavior.capability_needs(&dialect.generation_policy, context_steps)
        } else {
            GenerationCapabilityNeeds {
                understanding: true,
                ..Default::default()
            }
        };
        if let Err(capability) = capabilities.generation_runtime.covers(&needs) {
            return Err(reject(capability));
        }
        Ok(())
    }

    async fn compile_multimodal_chat_branch(
        &self,
        request: &ServeRequest,
        mut messages: Vec<ChatMessage>,
        mut chat_options: ChatOptions,
        tools: Vec<ChatTool>,
        tool_choice: ChatToolChoice,
        documents: Option<Vec<serde_json::Value>>,
        capabilities: &RuntimeCapabilitySnapshot,
    ) -> Result<(
        PlannedRequest,
        PlannedContextInspection,
        Vec<RenderedSegmentInspection>,
    )> {
        let dialect =
            self.profile
                .generation_dialect
                .as_ref()
                .ok_or_else(|| ServeError::Compilation {
                    request_id: request.request_id.clone(),
                    message: "the resolved model profile has no generation dialect".to_string(),
                })?;
        let mut rendered_segments = messages
            .iter()
            .enumerate()
            .map(|(index, message)| RenderedSegmentInspection {
                origin: RenderedSegmentOrigin::ChatMessage {
                    index,
                    role: context_role(message.role()),
                },
                token_start: None,
                token_end: None,
            })
            .collect::<Vec<_>>();
        let image_count = semantic_image_count(&request.model_context);
        let prompt_kind = uniserve_model_profile::dialect::PromptKind::for_request(
            request.generation.constraint,
            image_count > 0,
        );
        let scaffold = dialect.chat_prompt_scaffold(prompt_kind);
        if !messages
            .iter()
            .any(|message| matches!(message, ChatMessage::System { .. }))
            && let Some(system) = scaffold.default_system.filter(|value| !value.is_empty())
        {
            messages.insert(0, ChatMessage::system(system));
            rendered_segments.insert(
                0,
                RenderedSegmentInspection {
                    origin: RenderedSegmentOrigin::ProfileSystem,
                    token_start: None,
                    token_end: None,
                },
            );
        }
        if !scaffold.assistant_prefix.is_empty()
            && chat_options.generation_prompt_mode == GenerationPromptMode::StartNewAssistant
        {
            messages.push(ChatMessage::assistant_text(scaffold.assistant_prefix));
            rendered_segments.push(RenderedSegmentInspection {
                origin: RenderedSegmentOrigin::ProfileAssistantPrefix,
                token_start: None,
                token_end: None,
            });
            chat_options.generation_prompt_mode = GenerationPromptMode::ContinueFinalAssistant;
        }
        let message_count = messages.len();

        let marker_text = dialect.context_markers_in_prompt().then(|| {
            format!(
                "{}{}",
                dialect.controls.start_of_image_text, dialect.controls.end_of_image_text
            )
        });
        if marker_text.as_ref().is_some_and(String::is_empty) {
            return Err(ServeError::Compilation {
                request_id: request.request_id.clone(),
                message: "the generation profile has no context-image marker text".to_string(),
            });
        }
        let (images, placeholders) =
            replace_chat_images(&request.request_id, &mut messages, marker_text.as_deref())?;
        let chat_request = self.to_chat_request(
            request,
            messages,
            chat_options,
            tools,
            tool_choice,
            documents,
        );
        let (chat_request, rendered) = self.chat.render(chat_request).await?;
        let (prompt_token_ids, images) = tokenize_rendered_chat_with_images(
            &request.request_id,
            rendered.prompt,
            &self.chat.text().tokenizer(),
            request.generation.add_special_tokens,
            dialect.context_markers_in_prompt(),
            &placeholders,
            images,
        )?;
        let prompt_token_count = prompt_token_ids.len();
        let mut segments = Vec::with_capacity(images.len() + 1);
        segments.push(ContextSegment::TokenIds {
            token_ids: prompt_token_ids,
            tokenizer_fingerprint: self.profile.tokenizer_fingerprint().to_string(),
        });
        segments.extend(images.into_iter().map(ContextSegment::Image));
        let mut dialect_request = request.clone();
        dialect_request.model_context = ModelContext::Segments(segments);
        dialect_request.generation.skip_special_tokens =
            chat_request.decode_options.skip_special_tokens;
        let (mut branch, _) = self.compile_dialect_branch(&dialect_request, capabilities)?;
        let PlannedRequest::DialectGeneration(prepared) = &mut branch else {
            unreachable!("multimodal chat must compile through the generation dialect")
        };
        prepared.chat_output = Some(chat_request);
        let mut image_index = 0;
        for segment in &prepared.generation.context {
            if let uniserve_core::ContextSegment::Image { image, .. } = segment {
                let token_start = match image.placement {
                    uniserve_core::SegmentPlacement::AtToken { position } => {
                        Some(position as usize)
                    }
                    uniserve_core::SegmentPlacement::Append => Some(prompt_token_count),
                };
                rendered_segments.push(RenderedSegmentInspection {
                    origin: RenderedSegmentOrigin::Image { index: image_index },
                    token_start,
                    token_end: token_start,
                });
                image_index += 1;
            }
        }
        Ok((
            branch,
            PlannedContextInspection::Chat {
                message_count,
                prompt_token_count,
                multimodal: true,
            },
            rendered_segments,
        ))
    }

    fn compile_dialect_branch(
        &self,
        request: &ServeRequest,
        capabilities: &RuntimeCapabilitySnapshot,
    ) -> Result<(PlannedRequest, PlannedContextInspection)> {
        let dialect =
            self.profile
                .generation_dialect
                .as_ref()
                .ok_or_else(|| ServeError::Compilation {
                    request_id: request.request_id.clone(),
                    message: "the resolved model profile has no generation dialect".to_string(),
                })?;
        let generation_request = dialect_generation::compile_generation_request(
            request,
            self.chat.text().tokenizer(),
            dialect,
            &capabilities.generation_runtime,
            self.profile
                .context_limits
                .max_output_tokens
                .or(self.profile.generation_defaults.max_output_tokens),
            self.effective_max_model_tokens(capabilities),
        )
        .map_err(|error| ServeError::Compilation {
            request_id: request.request_id.clone(),
            message: error.message().to_string(),
        })?;
        let (segment_count, image_count) = match &request.model_context {
            ModelContext::Segments(segments) => (
                segments.len(),
                segments
                    .iter()
                    .filter(|segment| matches!(segment, ContextSegment::Image(_)))
                    .count(),
            ),
            _ => (1, 0),
        };
        let prompt_token_count = generation_request.prompt_token_count();
        Ok((
            PlannedRequest::DialectGeneration(Box::new(PreparedDialectRequest {
                generation: generation_request,
                chat_output: None,
            })),
            PlannedContextInspection::Segments {
                segment_count,
                prompt_token_count,
                image_count,
            },
        ))
    }

    fn effective_max_model_tokens(&self, capabilities: &RuntimeCapabilitySnapshot) -> u32 {
        self.profile
            .context_limits
            .max_model_tokens
            .map_or(capabilities.max_model_tokens, |profile_limit| {
                profile_limit.min(capabilities.max_model_tokens)
            })
    }

    fn to_text_request(&self, request: &ServeRequest, prompt: Prompt) -> TextRequest {
        TextRequest {
            request_id: request.request_id.to_string(),
            prompt,
            sampling_params: sampling_params(&request.generation, &request.cache),
            decode_options: decode_options(&request.generation),
            intermediate: request.generation.intermediate,
            priority: request.scheduling.priority,
            cache_salt: cache_isolation_material(&request.cache),
            add_special_tokens: request.generation.add_special_tokens,
            data_parallel_rank: request.scheduling.data_parallel_rank,
            trace_context: request.scheduling.trace_context.clone(),
            adapter: request.adapter.clone(),
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
            request_id: request.request_id.to_string(),
            messages,
            sampling_params: sampling_params(&request.generation, &request.cache),
            chat_options,
            tools,
            tool_choice,
            decode_options: decode_options(&request.generation),
            intermediate: request.generation.intermediate,
            priority: request.scheduling.priority,
            documents,
            cache_salt: cache_isolation_material(&request.cache),
            add_special_tokens: request.generation.add_special_tokens,
            data_parallel_rank: request.scheduling.data_parallel_rank,
            trace_context: request.scheduling.trace_context.clone(),
            adapter: request.adapter.clone(),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestLifecycleState {
    Compiling,
    Submitting,
    Accepted,
    Scheduled,
    Streaming,
    Cancelling,
    Aborting,
    Finished,
    Rejected,
    Cancelled,
    Aborted,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestStatsSnapshot {
    pub request_id: ServeRequestId,
    pub profile_id: String,
    pub dialect_id: String,
    pub state: RequestLifecycleState,
    pub prompt_tokens: u32,
    pub visible_output_tokens: u32,
    pub internal_tokens: u32,
    pub image_count: u32,
    pub image_steps: u32,
    pub cache: CacheAccounting,
    pub resources: ResourceAccounting,
    pub timings: RuntimeTimings,
}

impl RequestStatsSnapshot {
    fn submitting(
        request_id: ServeRequestId,
        profile_id: String,
        dialect_id: String,
        compile_us: u64,
    ) -> Self {
        Self {
            request_id,
            profile_id,
            dialect_id,
            state: RequestLifecycleState::Submitting,
            prompt_tokens: 0,
            visible_output_tokens: 0,
            internal_tokens: 0,
            image_count: 0,
            image_steps: 0,
            cache: CacheAccounting::default(),
            resources: ResourceAccounting::default(),
            timings: RuntimeTimings {
                compile_us,
                ..RuntimeTimings::default()
            },
        }
    }

    fn terminal(
        request_id: ServeRequestId,
        profile_id: String,
        dialect_id: String,
        state: RequestLifecycleState,
        total_us: u64,
    ) -> Self {
        let mut snapshot = Self::submitting(request_id, profile_id, dialect_id, total_us);
        snapshot.state = state;
        snapshot.timings.total_us = total_us;
        snapshot
    }
}

#[derive(Default)]
struct RequestRegistryState {
    active: HashMap<ServeRequestId, RequestStatsSnapshot>,
    completed: HashMap<ServeRequestId, RequestStatsSnapshot>,
    completed_order: VecDeque<ServeRequestId>,
}

struct RuntimeRequestRegistry {
    state: Mutex<RequestRegistryState>,
    changed: Notify,
    completed_retention: usize,
}

impl Default for RuntimeRequestRegistry {
    fn default() -> Self {
        Self {
            state: Mutex::new(RequestRegistryState::default()),
            changed: Notify::new(),
            completed_retention: 1024,
        }
    }
}

impl RuntimeRequestRegistry {
    fn lock(&self) -> std::sync::MutexGuard<'_, RequestRegistryState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn register(
        &self,
        request_id: ServeRequestId,
        profile_id: String,
        dialect_id: String,
        compile_us: u64,
        initial_state: RequestLifecycleState,
    ) -> bool {
        let mut state = self.lock();
        if state.active.contains_key(&request_id) {
            return false;
        }
        state.completed.remove(&request_id);
        state.completed_order.retain(|id| id != &request_id);
        let mut stats = RequestStatsSnapshot::submitting(
            request_id.clone(),
            profile_id,
            dialect_id,
            compile_us,
        );
        stats.state = initial_state;
        state.active.insert(request_id.clone(), stats);
        true
    }

    fn mark_submitting(&self, request_id: &str, compile_us: u64) {
        if let Some(stats) = self.lock().active.get_mut(request_id) {
            if !matches!(
                stats.state,
                RequestLifecycleState::Cancelling | RequestLifecycleState::Aborting
            ) {
                stats.state = RequestLifecycleState::Submitting;
            }
            stats.timings.compile_us = compile_us;
        }
    }

    fn accept(&self, request_id: &str) -> bool {
        let mut registry = self.lock();
        let Some(stats) = registry.active.get_mut(request_id) else {
            return false;
        };
        if matches!(
            stats.state,
            RequestLifecycleState::Cancelling | RequestLifecycleState::Aborting
        ) {
            return false;
        }
        stats.state = RequestLifecycleState::Accepted;
        true
    }

    fn mark_control(&self, request_id: &str, state: RequestLifecycleState) -> bool {
        let (engine_may_own_request, changed) = {
            let mut registry = self.lock();
            let Some(stats) = registry.active.get_mut(request_id) else {
                return false;
            };
            let previous = stats.state;
            let next = match (previous, state) {
                (RequestLifecycleState::Aborting, _) => RequestLifecycleState::Aborting,
                (RequestLifecycleState::Cancelling, RequestLifecycleState::Aborting) => {
                    RequestLifecycleState::Aborting
                }
                (RequestLifecycleState::Cancelling, _) => RequestLifecycleState::Cancelling,
                (_, requested) => requested,
            };
            let changed = next != previous;
            stats.state = next;
            (
                previous != RequestLifecycleState::Compiling && changed,
                changed,
            )
        };
        if changed {
            self.changed.notify_waiters();
        }
        engine_may_own_request
    }

    fn control_terminal(&self, request_id: &str) -> Option<LifecycleTerminal> {
        match self.lock().active.get(request_id).map(|stats| stats.state) {
            Some(RequestLifecycleState::Cancelling) => Some(LifecycleTerminal::Cancelled),
            Some(RequestLifecycleState::Aborting) => Some(LifecycleTerminal::Aborted),
            _ => None,
        }
    }

    async fn wait_for_control(&self, request_id: &str) -> LifecycleTerminal {
        loop {
            let changed = self.changed.notified();
            if let Some(terminal) = self.control_terminal(request_id) {
                return terminal;
            }
            changed.await;
        }
    }

    fn drop_terminal(&self, request_id: &str) -> LifecycleTerminal {
        match self.lock().active.get(request_id).map(|stats| stats.state) {
            Some(RequestLifecycleState::Aborting) => LifecycleTerminal::Aborted,
            _ => LifecycleTerminal::Cancelled,
        }
    }

    fn observe(
        &self,
        request_id: &str,
        event: &ServeEvent,
        elapsed_us: u64,
    ) -> Option<LifecycleTerminal> {
        let mut state = self.lock();
        let stats = state.active.get_mut(request_id)?;
        match stats.state {
            RequestLifecycleState::Cancelling => return Some(LifecycleTerminal::Cancelled),
            RequestLifecycleState::Aborting => return Some(LifecycleTerminal::Aborted),
            _ => {}
        }
        match event {
            ServeEvent::Accepted {
                compile_duration_us,
                prompt_token_count,
                ..
            } => {
                stats.state = RequestLifecycleState::Accepted;
                stats.prompt_tokens = (*prompt_token_count).min(u32::MAX as usize) as u32;
                stats.timings.compile_us = *compile_duration_us;
            }
            ServeEvent::Scheduled {
                queued_at,
                scheduled_at,
                cache,
                resources,
                ..
            } => {
                stats.state = RequestLifecycleState::Scheduled;
                stats.cache = cache.clone();
                stats.resources = resources.clone();
                stats.timings.queue_us =
                    (*queued_at).zip(*scheduled_at).map(|(queued, scheduled)| {
                        ((scheduled - queued).max(0.0) * 1_000_000.0) as u64
                    });
            }
            ServeEvent::PublicCommit { .. } => {}
            ServeEvent::TextDelta { token_ids, .. } => {
                stats.state = RequestLifecycleState::Streaming;
                stats.visible_output_tokens = stats
                    .visible_output_tokens
                    .saturating_add(token_ids.len().min(u32::MAX as usize) as u32);
                stats
                    .timings
                    .first_visible_output_us
                    .get_or_insert(elapsed_us);
            }
            ServeEvent::InternalTextDelta { .. } => {
                stats.state = RequestLifecycleState::Streaming;
                stats.internal_tokens = stats.internal_tokens.saturating_add(1);
            }
            ServeEvent::ReasoningDelta { .. }
            | ServeEvent::OutputBlockStart { .. }
            | ServeEvent::OutputBlockEnd { .. }
            | ServeEvent::ToolCallStart { .. }
            | ServeEvent::ToolCallArgumentsDelta { .. }
            | ServeEvent::ToolCallEnd { .. } => {
                stats.state = RequestLifecycleState::Streaming;
            }
            ServeEvent::ImageBegin {
                elapsed_us: event_elapsed,
                ..
            }
            | ServeEvent::ImageCommit {
                elapsed_us: event_elapsed,
                ..
            } => {
                stats.state = RequestLifecycleState::Streaming;
                stats
                    .timings
                    .first_visible_output_us
                    .get_or_insert(*event_elapsed);
            }
            ServeEvent::ImageStep { .. } => {
                stats.state = RequestLifecycleState::Streaming;
                stats.image_steps = stats.image_steps.saturating_add(1);
            }
            ServeEvent::ImageDone { .. } => {
                stats.state = RequestLifecycleState::Streaming;
                stats.image_count = stats.image_count.saturating_add(1);
            }
            ServeEvent::Usage {
                prompt_tokens,
                visible_output_tokens,
                internal_tokens,
                image_count,
                image_steps,
                cache,
                resources,
                timings,
            } => {
                stats.prompt_tokens = *prompt_tokens;
                stats.visible_output_tokens = *visible_output_tokens;
                stats.internal_tokens = *internal_tokens;
                stats.image_count = *image_count;
                stats.image_steps = *image_steps;
                stats.cache = cache.clone();
                stats.resources = resources.clone();
                stats.timings = timings.clone();
            }
            ServeEvent::Finished { .. }
            | ServeEvent::Rejected { .. }
            | ServeEvent::Cancelled { .. }
            | ServeEvent::Aborted { .. }
            | ServeEvent::Failed { .. } => {}
        }
        None
    }

    fn complete(
        &self,
        request_id: &str,
        terminal: RequestLifecycleState,
        elapsed_us: u64,
    ) -> Option<RequestLifecycleState> {
        let mut state = self.lock();
        let mut stats = state.active.remove(request_id)?;
        let actual_terminal = match stats.state {
            RequestLifecycleState::Cancelling => RequestLifecycleState::Cancelled,
            RequestLifecycleState::Aborting => RequestLifecycleState::Aborted,
            _ => terminal,
        };
        stats.state = actual_terminal;
        stats.timings.total_us = stats.timings.total_us.max(elapsed_us);
        Self::insert_completed(&mut state, stats, self.completed_retention);
        drop(state);
        self.changed.notify_waiters();
        Some(actual_terminal)
    }

    fn record_terminal(&self, stats: RequestStatsSnapshot) {
        let mut state = self.lock();
        Self::insert_completed(&mut state, stats, self.completed_retention);
        drop(state);
        self.changed.notify_waiters();
    }

    fn insert_completed(
        state: &mut RequestRegistryState,
        stats: RequestStatsSnapshot,
        retention: usize,
    ) {
        let request_id = stats.request_id.clone();
        state.completed_order.retain(|id| id != &request_id);
        state.completed.insert(request_id.clone(), stats);
        state.completed_order.push_back(request_id);
        while state.completed_order.len() > retention {
            if let Some(expired) = state.completed_order.pop_front() {
                state.completed.remove(&expired);
            }
        }
    }

    fn stats(&self, request_id: &str) -> Option<RequestStatsSnapshot> {
        let state = self.lock();
        state
            .active
            .get(request_id)
            .or_else(|| state.completed.get(request_id))
            .cloned()
    }

    fn active_count(&self) -> usize {
        self.lock().active.len()
    }

    async fn drain_request(&self, request_id: &str) -> Option<RequestStatsSnapshot> {
        loop {
            let changed = self.changed.notified();
            if !self.lock().active.contains_key(request_id) {
                return self.stats(request_id);
            }
            changed.await;
        }
    }

    async fn drain(&self) {
        loop {
            let changed = self.changed.notified();
            if self.active_count() == 0 {
                return;
            }
            changed.await;
        }
    }
}

#[derive(Debug, Clone)]
struct EventContext {
    profile_id: String,
    dialect_id: String,
    compile_duration_us: u64,
    cache: CompiledCachePolicy,
    resources: ResourceBounds,
    skip_special_tokens: bool,
    metrics: Arc<RuntimeLifecycleMetrics>,
}

#[derive(Debug, Default)]
struct RuntimeLifecycleMetrics {
    accepted: AtomicU64,
    scheduled: AtomicU64,
    finished: AtomicU64,
    rejected: AtomicU64,
    cancelled: AtomicU64,
    aborted: AtomicU64,
    failed: AtomicU64,
}

impl RuntimeLifecycleMetrics {
    fn snapshot(&self) -> RuntimeMetricsSnapshot {
        RuntimeMetricsSnapshot {
            active: 0,
            accepted: self.accepted.load(Ordering::Relaxed),
            scheduled: self.scheduled.load(Ordering::Relaxed),
            finished: self.finished.load(Ordering::Relaxed),
            rejected: self.rejected.load(Ordering::Relaxed),
            cancelled: self.cancelled.load(Ordering::Relaxed),
            aborted: self.aborted.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeMetricsSnapshot {
    pub active: u64,
    pub accepted: u64,
    pub scheduled: u64,
    pub finished: u64,
    pub rejected: u64,
    pub cancelled: u64,
    pub aborted: u64,
    pub failed: u64,
}

struct LifecycleGuard {
    request_id: ServeRequestId,
    metrics: Arc<RuntimeLifecycleMetrics>,
    requests: Arc<RuntimeRequestRegistry>,
    started: Instant,
    terminal: bool,
}

struct LifecycleTrackedStream {
    inner: ServeEventStream,
    lifecycle: LifecycleGuard,
}

impl LifecycleTrackedStream {
    fn new(
        request_id: ServeRequestId,
        inner: ServeEventStream,
        metrics: Arc<RuntimeLifecycleMetrics>,
        requests: Arc<RuntimeRequestRegistry>,
    ) -> Self {
        Self {
            inner,
            lifecycle: LifecycleGuard::new(request_id, metrics, requests),
        }
    }
}

impl Stream for LifecycleTrackedStream {
    type Item = Result<ServeEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(mut event))) => {
                let elapsed_us = self.lifecycle.started.elapsed().as_micros() as u64;
                if let Some(control) =
                    self.lifecycle
                        .requests
                        .observe(&self.lifecycle.request_id, &event, elapsed_us)
                {
                    let terminal = self.lifecycle.terminal(control, elapsed_us);
                    event = control_terminal_event(&self.lifecycle.request_id, terminal);
                    return Poll::Ready(Some(Ok(event)));
                }
                let terminal = match &event {
                    ServeEvent::Finished { .. } => Some(LifecycleTerminal::Finished),
                    ServeEvent::Rejected { .. } => Some(LifecycleTerminal::Rejected),
                    ServeEvent::Cancelled { .. } => Some(LifecycleTerminal::Cancelled),
                    ServeEvent::Aborted { .. } => Some(LifecycleTerminal::Aborted),
                    ServeEvent::Failed { .. } => Some(LifecycleTerminal::Failed),
                    _ => None,
                };
                if let Some(terminal) = terminal {
                    let actual = self.lifecycle.terminal(terminal, elapsed_us);
                    if actual != terminal {
                        event = control_terminal_event(&self.lifecycle.request_id, actual);
                    }
                }
                Poll::Ready(Some(Ok(event)))
            }
            Poll::Ready(Some(Err(error))) => {
                let elapsed_us = self.lifecycle.started.elapsed().as_micros() as u64;
                let actual = self
                    .lifecycle
                    .terminal(LifecycleTerminal::Failed, elapsed_us);
                if actual == LifecycleTerminal::Failed {
                    Poll::Ready(Some(Err(error)))
                } else {
                    Poll::Ready(Some(Ok(control_terminal_event(
                        &self.lifecycle.request_id,
                        actual,
                    ))))
                }
            }
            Poll::Ready(None) => {
                if !self.lifecycle.terminal {
                    let elapsed_us = self.lifecycle.started.elapsed().as_micros() as u64;
                    let actual = self
                        .lifecycle
                        .terminal(LifecycleTerminal::Failed, elapsed_us);
                    if actual != LifecycleTerminal::Failed {
                        return Poll::Ready(Some(Ok(control_terminal_event(
                            &self.lifecycle.request_id,
                            actual,
                        ))));
                    }
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl LifecycleGuard {
    fn new(
        request_id: ServeRequestId,
        metrics: Arc<RuntimeLifecycleMetrics>,
        requests: Arc<RuntimeRequestRegistry>,
    ) -> Self {
        Self {
            request_id,
            metrics,
            requests,
            started: Instant::now(),
            terminal: false,
        }
    }

    fn terminal(&mut self, kind: LifecycleTerminal, elapsed_us: u64) -> LifecycleTerminal {
        if self.terminal {
            return kind;
        }
        let actual = self
            .requests
            .complete(&self.request_id, kind.into(), elapsed_us)
            .and_then(lifecycle_terminal_for_state)
            .unwrap_or(kind);
        let counter = match actual {
            LifecycleTerminal::Finished => &self.metrics.finished,
            LifecycleTerminal::Rejected => &self.metrics.rejected,
            LifecycleTerminal::Cancelled => &self.metrics.cancelled,
            LifecycleTerminal::Aborted => &self.metrics.aborted,
            LifecycleTerminal::Failed => &self.metrics.failed,
        };
        counter.fetch_add(1, Ordering::Relaxed);
        self.terminal = true;
        actual
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LifecycleTerminal {
    Finished,
    Rejected,
    Cancelled,
    Aborted,
    Failed,
}

fn lifecycle_terminal_for_state(state: RequestLifecycleState) -> Option<LifecycleTerminal> {
    match state {
        RequestLifecycleState::Finished => Some(LifecycleTerminal::Finished),
        RequestLifecycleState::Rejected => Some(LifecycleTerminal::Rejected),
        RequestLifecycleState::Cancelled | RequestLifecycleState::Cancelling => {
            Some(LifecycleTerminal::Cancelled)
        }
        RequestLifecycleState::Aborted | RequestLifecycleState::Aborting => {
            Some(LifecycleTerminal::Aborted)
        }
        RequestLifecycleState::Failed => Some(LifecycleTerminal::Failed),
        RequestLifecycleState::Compiling
        | RequestLifecycleState::Submitting
        | RequestLifecycleState::Accepted
        | RequestLifecycleState::Scheduled
        | RequestLifecycleState::Streaming => None,
    }
}

fn control_terminal_event(request_id: &ServeRequestId, terminal: LifecycleTerminal) -> ServeEvent {
    match terminal {
        LifecycleTerminal::Aborted => ServeEvent::Aborted {
            request_id: request_id.clone(),
        },
        LifecycleTerminal::Cancelled => ServeEvent::Cancelled {
            request_id: request_id.clone(),
        },
        LifecycleTerminal::Finished | LifecycleTerminal::Rejected | LifecycleTerminal::Failed => {
            unreachable!("only cancellation and abort can replace an engine terminal")
        }
    }
}

impl From<LifecycleTerminal> for RequestLifecycleState {
    fn from(value: LifecycleTerminal) -> Self {
        match value {
            LifecycleTerminal::Finished => Self::Finished,
            LifecycleTerminal::Rejected => Self::Rejected,
            LifecycleTerminal::Cancelled => Self::Cancelled,
            LifecycleTerminal::Aborted => Self::Aborted,
            LifecycleTerminal::Failed => Self::Failed,
        }
    }
}

impl Drop for LifecycleGuard {
    fn drop(&mut self) {
        if !self.terminal {
            let elapsed_us = self.started.elapsed().as_micros() as u64;
            let terminal = self.requests.drop_terminal(&self.request_id);
            let _ = self.terminal(terminal, elapsed_us);
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheAccounting {
    pub read_enabled: bool,
    pub write_enabled: bool,
    pub encoder_pin_count: usize,
    pub transfer: Option<serde_json::Value>,
}

impl From<&CompiledCachePolicy> for CacheAccounting {
    fn from(cache: &CompiledCachePolicy) -> Self {
        Self {
            read_enabled: cache.read_enabled,
            write_enabled: cache.write_enabled,
            encoder_pin_count: cache.encoder_pin_count,
            transfer: None,
        }
    }
}

fn queue_duration_us(queued_at: Option<f64>, scheduled_at: Option<f64>) -> Option<u64> {
    let seconds = scheduled_at? - queued_at?;
    (seconds.is_finite() && seconds >= 0.0).then_some((seconds * 1_000_000.0) as u64)
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceAccounting {
    pub expected_kv_tokens: u64,
    pub image_latent_units: u64,
    pub scratch_units: u64,
    pub host_scratch_tokens: u64,
    pub encoder_cache_pins: usize,
    pub grammar_states: usize,
    pub adapter_slots: usize,
    pub replayable: bool,
}

impl From<&ResourceBounds> for ResourceAccounting {
    fn from(resources: &ResourceBounds) -> Self {
        Self {
            expected_kv_tokens: resources.expected_kv_tokens,
            image_latent_units: resources.image_latent_units,
            scratch_units: resources.scratch_units,
            host_scratch_tokens: resources.host_scratch_tokens,
            encoder_cache_pins: resources.encoder_cache_pins,
            grammar_states: resources.grammar_states,
            adapter_slots: resources.adapter_slots,
            replayable: resources.replayable,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeTimings {
    pub compile_us: u64,
    pub queue_us: Option<u64>,
    pub first_visible_output_us: Option<u64>,
    pub total_us: u64,
}

/// Protocol-neutral runtime event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ServeEvent {
    Accepted {
        request_id: ServeRequestId,
        profile_id: String,
        dialect_id: String,
        compile_duration_us: u64,
        prompt_token_count: usize,
        prompt_token_ids: Vec<u32>,
        prompt_logprobs: Option<DecodedPromptLogprobs>,
    },
    Scheduled {
        request_id: ServeRequestId,
        queued_at: Option<f64>,
        scheduled_at: Option<f64>,
        cache: CacheAccounting,
        resources: ResourceAccounting,
    },
    PublicCommit {
        commit: PublicCommit,
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
        elapsed_us: u64,
    },
    ImageStep {
        candidate_id: CandidateId,
        image_id: String,
        step: u32,
        elapsed_us: u64,
    },
    ImageCommit {
        candidate_id: CandidateId,
        image_id: String,
        elapsed_us: u64,
    },
    ImageDone {
        candidate_id: CandidateId,
        image_id: String,
        width: Option<u32>,
        height: Option<u32>,
        bytes: Option<u64>,
        sha256: Option<String>,
        pixels_png_b64: Option<String>,
        elapsed_us: u64,
    },
    Usage {
        prompt_tokens: u32,
        visible_output_tokens: u32,
        internal_tokens: u32,
        image_count: u32,
        image_steps: u32,
        cache: CacheAccounting,
        resources: ResourceAccounting,
        timings: RuntimeTimings,
    },
    Finished {
        candidate_id: CandidateId,
        reason: FinishStatus,
        finish_detail: Option<String>,
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
    Stop { cause: Option<StopCause> },
    Length,
    Abort,
    Error,
    Repetition,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum StopCause {
    Eos,
    TokenId(u32),
    Text(String),
}

impl From<&FinishReason> for FinishStatus {
    fn from(reason: &FinishReason) -> Self {
        match reason {
            FinishReason::Stop(stop_reason) => Self::Stop {
                cause: stop_reason.as_ref().map(|value| match value {
                    StopReason::TokenId(id) => StopCause::TokenId(*id),
                    StopReason::Text(text) => StopCause::Text(text.clone()),
                }),
            },
            FinishReason::Length => Self::Length,
            FinishReason::Abort => Self::Abort,
            FinishReason::Cancelled => Self::Abort,
            FinishReason::Aborted => Self::Abort,
            FinishReason::Error => Self::Error,
            FinishReason::Repetition => Self::Repetition,
        }
    }
}

#[try_stream]
async fn control_aware_event_stream(
    request_id: ServeRequestId,
    requests: Arc<RuntimeRequestRegistry>,
    mut stream: ServeEventStream,
    mut y: TryYielder<ServeEvent, ServeError>,
) -> Result<()> {
    loop {
        if let Some(terminal) = requests.control_terminal(&request_id) {
            y.yield_ok(control_terminal_event(&request_id, terminal))
                .await;
            return Ok(());
        }
        tokio::select! {
            next = stream.next() => match next {
                Some(Ok(event)) => {
                    if let Some(terminal) = requests.control_terminal(&request_id) {
                        y.yield_ok(control_terminal_event(&request_id, terminal)).await;
                        return Ok(());
                    }
                    let terminal = matches!(
                        event,
                        ServeEvent::Finished { .. }
                            | ServeEvent::Rejected { .. }
                            | ServeEvent::Cancelled { .. }
                            | ServeEvent::Aborted { .. }
                            | ServeEvent::Failed { .. }
                    );
                    y.yield_ok(event).await;
                    if terminal {
                        return Ok(());
                    }
                }
                Some(Err(error)) => return Err(error),
                None => return Ok(()),
            },
            terminal = requests.wait_for_control(&request_id) => {
                y.yield_ok(control_terminal_event(&request_id, terminal)).await;
                return Ok(());
            }
        }
    }
}

#[try_stream]
async fn text_event_stream(
    request_id: ServeRequestId,
    event_context: EventContext,
    stream: impl Stream<Item = crate::text::Result<DecodedTextEvent>> + Send,
    mut y: TryYielder<ServeEvent, ServeError>,
) -> Result<()> {
    let started = Instant::now();
    let mut first_visible_output_us = None;
    let mut queue_us = None;
    pin_mut!(stream);
    while let Some(next) = stream.next().await {
        let event = match next {
            Ok(event) => event,
            Err(error) => {
                return Err(error.into());
            }
        };
        match event {
            DecodedTextEvent::Start {
                prompt_token_ids,
                prompt_logprobs,
                queued_at,
                scheduled_at,
            } => {
                queue_us = queue_duration_us(queued_at, scheduled_at);
                y.yield_ok(ServeEvent::Accepted {
                    request_id: request_id.clone(),
                    profile_id: event_context.profile_id.clone(),
                    dialect_id: event_context.dialect_id.clone(),
                    compile_duration_us: event_context.compile_duration_us,
                    prompt_token_count: prompt_token_ids.len(),
                    prompt_token_ids: prompt_token_ids.to_vec(),
                    prompt_logprobs,
                })
                .await;
                y.yield_ok(ServeEvent::Scheduled {
                    request_id: request_id.clone(),
                    queued_at,
                    scheduled_at,
                    cache: CacheAccounting::from(&event_context.cache),
                    resources: ResourceAccounting::from(&event_context.resources),
                })
                .await;
                event_context
                    .metrics
                    .scheduled
                    .fetch_add(1, Ordering::Relaxed);
            }
            DecodedTextEvent::TextDelta {
                delta,
                token_ids,
                logprobs,
                public_commit,
                finished,
            } => {
                if !delta.is_empty() && first_visible_output_us.is_none() {
                    first_visible_output_us = Some(started.elapsed().as_micros() as u64);
                }
                if !delta.is_empty()
                    && let Some(commit) = public_commit
                {
                    y.yield_ok(ServeEvent::PublicCommit { commit }).await;
                }
                if !delta.is_empty() || !token_ids.is_empty() || finished.is_some() {
                    y.yield_ok(ServeEvent::TextDelta {
                        candidate_id: CandidateId::PRIMARY,
                        text: delta,
                        token_ids,
                        logprobs,
                    })
                    .await;
                }
                if let Some(finished) = finished {
                    let mut cache = CacheAccounting::from(&event_context.cache);
                    cache.transfer = finished.kv_transfer_params.clone();
                    let visible_output_tokens = finished
                        .output_token_count
                        .saturating_sub(finished.internal_token_count);
                    y.yield_ok(ServeEvent::Usage {
                        prompt_tokens: finished.prompt_token_count as u32,
                        visible_output_tokens: visible_output_tokens as u32,
                        internal_tokens: finished.internal_token_count as u32,
                        image_count: 0,
                        image_steps: 0,
                        cache,
                        resources: ResourceAccounting::from(&event_context.resources),
                        timings: RuntimeTimings {
                            compile_us: event_context.compile_duration_us,
                            queue_us,
                            first_visible_output_us,
                            total_us: started.elapsed().as_micros() as u64,
                        },
                    })
                    .await;
                    match finished.finish_reason {
                        FinishReason::Cancelled => {
                            y.yield_ok(ServeEvent::Cancelled {
                                request_id: request_id.clone(),
                            })
                            .await;
                        }
                        FinishReason::Abort | FinishReason::Aborted => {
                            y.yield_ok(ServeEvent::Aborted {
                                request_id: request_id.clone(),
                            })
                            .await;
                        }
                        FinishReason::Error => {
                            y.yield_ok(ServeEvent::Failed {
                                request_id: request_id.clone(),
                                message: "engine execution failed".to_string(),
                            })
                            .await;
                        }
                        reason => {
                            y.yield_ok(ServeEvent::Finished {
                                candidate_id: CandidateId::PRIMARY,
                                reason: FinishStatus::from(&reason),
                                finish_detail: None,
                            })
                            .await;
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

#[try_stream]
async fn chat_event_stream(
    request_id: ServeRequestId,
    event_context: EventContext,
    stream: impl Stream<Item = crate::chat::Result<ChatEvent>> + Send,
    mut y: TryYielder<ServeEvent, ServeError>,
) -> Result<()> {
    let started = Instant::now();
    let mut first_visible_output_us = None;
    let mut queue_us = None;
    pin_mut!(stream);
    while let Some(next) = stream.next().await {
        let event = match next {
            Ok(event) => event,
            Err(error) => {
                return Err(error.into());
            }
        };
        match event {
            ChatEvent::Start {
                prompt_token_ids,
                prompt_logprobs,
                queued_at,
                scheduled_at,
            } => {
                queue_us = queue_duration_us(queued_at, scheduled_at);
                y.yield_ok(ServeEvent::Accepted {
                    request_id: request_id.clone(),
                    profile_id: event_context.profile_id.clone(),
                    dialect_id: event_context.dialect_id.clone(),
                    compile_duration_us: event_context.compile_duration_us,
                    prompt_token_count: prompt_token_ids.len(),
                    prompt_token_ids: prompt_token_ids.to_vec(),
                    prompt_logprobs,
                })
                .await;
                event_context
                    .metrics
                    .scheduled
                    .fetch_add(1, Ordering::Relaxed);
                y.yield_ok(ServeEvent::Scheduled {
                    request_id: request_id.clone(),
                    queued_at,
                    scheduled_at,
                    cache: CacheAccounting::from(&event_context.cache),
                    resources: ResourceAccounting::from(&event_context.resources),
                })
                .await;
            }
            ChatEvent::BlockDelta { kind, delta, .. } => match kind {
                crate::chat::AssistantBlockKind::Text => {
                    if !delta.is_empty() {
                        first_visible_output_us
                            .get_or_insert_with(|| started.elapsed().as_micros() as u64);
                        y.yield_ok(ServeEvent::TextDelta {
                            candidate_id: CandidateId::PRIMARY,
                            text: delta,
                            token_ids: Vec::new(),
                            logprobs: None,
                        })
                        .await;
                    }
                }
                crate::chat::AssistantBlockKind::Reasoning => {
                    if !delta.is_empty() {
                        first_visible_output_us
                            .get_or_insert_with(|| started.elapsed().as_micros() as u64);
                        y.yield_ok(ServeEvent::ReasoningDelta {
                            candidate_id: CandidateId::PRIMARY,
                            text: delta,
                        })
                        .await;
                    }
                }
                crate::chat::AssistantBlockKind::ToolCall => {}
            },
            ChatEvent::LogprobsDelta {
                token_ids,
                logprobs,
            } => {
                if !token_ids.is_empty() || logprobs.is_some() {
                    y.yield_ok(ServeEvent::TextDelta {
                        candidate_id: CandidateId::PRIMARY,
                        text: String::new(),
                        token_ids,
                        logprobs,
                    })
                    .await;
                }
            }
            ChatEvent::ToolCallStart { index, id, name } => {
                y.yield_ok(ServeEvent::ToolCallStart {
                    candidate_id: CandidateId::PRIMARY,
                    index,
                    id,
                    name,
                })
                .await;
            }
            ChatEvent::ToolCallArgumentsDelta { index, delta } => {
                y.yield_ok(ServeEvent::ToolCallArgumentsDelta {
                    candidate_id: CandidateId::PRIMARY,
                    index,
                    delta,
                })
                .await;
            }
            ChatEvent::ToolCallEnd { index, call } => {
                y.yield_ok(ServeEvent::ToolCallEnd {
                    candidate_id: CandidateId::PRIMARY,
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
                visible_output_token_count,
                internal_token_count,
                finish_reason,
                kv_transfer_params,
                ..
            } => {
                debug_assert_eq!(
                    output_token_count,
                    visible_output_token_count.saturating_add(internal_token_count)
                );
                let mut cache = CacheAccounting::from(&event_context.cache);
                cache.transfer = kv_transfer_params;
                y.yield_ok(ServeEvent::Usage {
                    prompt_tokens: prompt_token_count as u32,
                    visible_output_tokens: visible_output_token_count as u32,
                    internal_tokens: internal_token_count as u32,
                    image_count: 0,
                    image_steps: 0,
                    cache,
                    resources: ResourceAccounting::from(&event_context.resources),
                    timings: RuntimeTimings {
                        compile_us: event_context.compile_duration_us,
                        queue_us,
                        first_visible_output_us,
                        total_us: started.elapsed().as_micros() as u64,
                    },
                })
                .await;
                match finish_reason {
                    FinishReason::Cancelled => {
                        y.yield_ok(ServeEvent::Cancelled {
                            request_id: request_id.clone(),
                        })
                        .await;
                    }
                    FinishReason::Abort | FinishReason::Aborted => {
                        y.yield_ok(ServeEvent::Aborted {
                            request_id: request_id.clone(),
                        })
                        .await;
                    }
                    FinishReason::Error => {
                        y.yield_ok(ServeEvent::Failed {
                            request_id: request_id.clone(),
                            message: "engine execution failed".to_string(),
                        })
                        .await;
                    }
                    reason => {
                        y.yield_ok(ServeEvent::Finished {
                            candidate_id: CandidateId::PRIMARY,
                            reason: FinishStatus::from(&reason),
                            finish_detail: None,
                        })
                        .await;
                    }
                }
            }
            ChatEvent::BlockStart { index, kind } => {
                y.yield_ok(ServeEvent::OutputBlockStart {
                    candidate_id: CandidateId::PRIMARY,
                    index,
                    kind,
                })
                .await;
            }
            ChatEvent::BlockEnd { index, block } => {
                y.yield_ok(ServeEvent::OutputBlockEnd {
                    candidate_id: CandidateId::PRIMARY,
                    index,
                    block,
                })
                .await;
            }
        }
    }

    Ok(())
}

enum ChatDecodedInput {
    Event(Box<DecodedTextEvent>),
    Barrier(Arc<AtomicBool>),
}

struct ChatDecodedInputStream {
    receiver: mpsc::Receiver<ChatDecodedInput>,
}

impl Stream for ChatDecodedInputStream {
    type Item = crate::chat::output::Result<DecodedTextEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match self.receiver.poll_recv(cx) {
                Poll::Ready(Some(ChatDecodedInput::Event(event))) => {
                    return Poll::Ready(Some(Ok(*event)));
                }
                Poll::Ready(Some(ChatDecodedInput::Barrier(reached))) => {
                    reached.store(true, Ordering::Release);
                }
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

struct ChatOutputBridge {
    sender: mpsc::Sender<ChatDecodedInput>,
    output: crate::chat::output::DynChatEventStream,
}

impl ChatOutputBridge {
    fn new(processor: crate::chat::DynChatOutputProcessor) -> crate::chat::output::Result<Self> {
        let (sender, receiver) = mpsc::channel(2);
        let decoded = Box::pin(ChatDecodedInputStream { receiver });
        let output = processor.process(decoded)?;
        Ok(Self { sender, output })
    }

    async fn push(
        &mut self,
        request_id: &ServeRequestId,
        event: DecodedTextEvent,
    ) -> Result<Vec<ChatEvent>> {
        self.sender
            .try_send(ChatDecodedInput::Event(Box::new(event)))
            .map_err(|_| ServeError::OutputProcessing {
                request_id: request_id.clone(),
                message: "chat output processor closed before receiving decoded text".to_string(),
            })?;
        let reached = Arc::new(AtomicBool::new(false));
        self.sender
            .try_send(ChatDecodedInput::Barrier(Arc::clone(&reached)))
            .map_err(|_| ServeError::OutputProcessing {
                request_id: request_id.clone(),
                message: "chat output processor closed before its synchronization barrier"
                    .to_string(),
            })?;

        let mut events = Vec::new();
        futures::future::poll_fn(|cx| {
            loop {
                match self.output.as_mut().poll_next(cx) {
                    Poll::Ready(Some(Ok(event))) => events.push(event),
                    Poll::Ready(Some(Err(error))) => {
                        return Poll::Ready(Err(ServeError::OutputProcessing {
                            request_id: request_id.clone(),
                            message: error.to_string(),
                        }));
                    }
                    Poll::Ready(None) => {
                        return Poll::Ready(Err(ServeError::OutputProcessing {
                            request_id: request_id.clone(),
                            message: "chat output processor closed before terminal output"
                                .to_string(),
                        }));
                    }
                    Poll::Pending if reached.load(Ordering::Acquire) => {
                        return Poll::Ready(Ok(std::mem::take(&mut events)));
                    }
                    Poll::Pending => return Poll::Pending,
                }
            }
        })
        .await
    }
}

struct ChatDone {
    prompt_token_count: usize,
    output_token_count: usize,
    visible_output_token_count: usize,
    internal_token_count: usize,
    finish_reason: FinishReason,
    kv_transfer_params: Option<serde_json::Value>,
}

enum MappedChatEvent {
    Ignore,
    Event(ServeEvent),
    Done(ChatDone),
}

fn map_chat_event(event: ChatEvent) -> MappedChatEvent {
    match event {
        ChatEvent::Start { .. } => MappedChatEvent::Ignore,
        ChatEvent::BlockStart { index, kind } => {
            MappedChatEvent::Event(ServeEvent::OutputBlockStart {
                candidate_id: CandidateId::PRIMARY,
                index,
                kind,
            })
        }
        ChatEvent::BlockDelta { kind, delta, .. } => match kind {
            AssistantBlockKind::Text => MappedChatEvent::Event(ServeEvent::TextDelta {
                candidate_id: CandidateId::PRIMARY,
                text: delta,
                token_ids: Vec::new(),
                logprobs: None,
            }),
            AssistantBlockKind::Reasoning => MappedChatEvent::Event(ServeEvent::ReasoningDelta {
                candidate_id: CandidateId::PRIMARY,
                text: delta,
            }),
            AssistantBlockKind::ToolCall => MappedChatEvent::Ignore,
        },
        ChatEvent::LogprobsDelta {
            token_ids,
            logprobs,
        } => MappedChatEvent::Event(ServeEvent::TextDelta {
            candidate_id: CandidateId::PRIMARY,
            text: String::new(),
            token_ids,
            logprobs,
        }),
        ChatEvent::BlockEnd { index, block } => {
            MappedChatEvent::Event(ServeEvent::OutputBlockEnd {
                candidate_id: CandidateId::PRIMARY,
                index,
                block,
            })
        }
        ChatEvent::ToolCallStart { index, id, name } => {
            MappedChatEvent::Event(ServeEvent::ToolCallStart {
                candidate_id: CandidateId::PRIMARY,
                index,
                id,
                name,
            })
        }
        ChatEvent::ToolCallArgumentsDelta { index, delta } => {
            MappedChatEvent::Event(ServeEvent::ToolCallArgumentsDelta {
                candidate_id: CandidateId::PRIMARY,
                index,
                delta,
            })
        }
        ChatEvent::ToolCallEnd { index, call } => MappedChatEvent::Event(ServeEvent::ToolCallEnd {
            candidate_id: CandidateId::PRIMARY,
            index,
            id: call.id,
            name: call.name,
            arguments: call.arguments,
        }),
        ChatEvent::Done {
            prompt_token_count,
            output_token_count,
            visible_output_token_count,
            internal_token_count,
            finish_reason,
            kv_transfer_params,
            ..
        } => MappedChatEvent::Done(ChatDone {
            prompt_token_count,
            output_token_count,
            visible_output_token_count,
            internal_token_count,
            finish_reason,
            kv_transfer_params,
        }),
    }
}

async fn emit_dialect_text_update(
    request_id: &ServeRequestId,
    processor: &mut dialect_generation::DialectStreamProcessor,
    mut chat_bridge: Option<&mut ChatOutputBridge>,
    text: String,
    token_ids: Vec<u32>,
    logprobs: Option<DecodedLogprobs>,
    mut public_commit: Option<PublicCommit>,
    finished: Option<crate::text::Finished>,
    started: &Instant,
    first_visible_output_us: &mut Option<u64>,
    y: &mut TryYielder<ServeEvent, ServeError>,
) -> Result<Option<ChatDone>> {
    let delta = processor.push(&text);
    if let Some(bridge) = chat_bridge.as_mut() {
        debug_assert!(delta.reasoning.is_empty());
        let events = bridge
            .push(
                request_id,
                DecodedTextEvent::TextDelta {
                    delta: delta.visible,
                    token_ids,
                    logprobs,
                    public_commit: public_commit.clone(),
                    finished,
                },
            )
            .await?;
        let mut done = None;
        for event in events {
            match map_chat_event(event) {
                MappedChatEvent::Ignore => {}
                MappedChatEvent::Event(event) => {
                    let visible = matches!(
                        &event,
                        ServeEvent::TextDelta { text, .. } if !text.is_empty()
                    ) || matches!(&event, ServeEvent::ReasoningDelta { text, .. } if !text.is_empty())
                        || matches!(&event, ServeEvent::ToolCallStart { .. });
                    if visible {
                        first_visible_output_us
                            .get_or_insert_with(|| started.elapsed().as_micros() as u64);
                        if let Some(commit) = public_commit.take() {
                            y.yield_ok(ServeEvent::PublicCommit { commit }).await;
                        }
                    }
                    y.yield_ok(event).await;
                }
                MappedChatEvent::Done(next) => {
                    if done.replace(next).is_some() {
                        return Err(ServeError::OutputProcessing {
                            request_id: request_id.clone(),
                            message: "chat output processor emitted multiple terminal events"
                                .to_string(),
                        });
                    }
                }
            }
        }
        return Ok(done);
    }

    if !delta.reasoning.is_empty() {
        first_visible_output_us.get_or_insert_with(|| started.elapsed().as_micros() as u64);
        if let Some(commit) = public_commit.take() {
            y.yield_ok(ServeEvent::PublicCommit { commit }).await;
        }
        y.yield_ok(ServeEvent::ReasoningDelta {
            candidate_id: CandidateId::PRIMARY,
            text: delta.reasoning,
        })
        .await;
    }
    if !delta.visible.is_empty() {
        first_visible_output_us.get_or_insert_with(|| started.elapsed().as_micros() as u64);
        if let Some(commit) = public_commit.take() {
            y.yield_ok(ServeEvent::PublicCommit { commit }).await;
        }
    }
    if !delta.visible.is_empty()
        || !token_ids.is_empty()
        || logprobs.is_some()
        || finished.is_some()
    {
        y.yield_ok(ServeEvent::TextDelta {
            candidate_id: CandidateId::PRIMARY,
            text: delta.visible,
            token_ids,
            logprobs,
        })
        .await;
    }
    Ok(finished.map(|finished| ChatDone {
        prompt_token_count: finished.prompt_token_count,
        output_token_count: finished.output_token_count,
        visible_output_token_count: finished
            .output_token_count
            .saturating_sub(finished.internal_token_count),
        internal_token_count: finished.internal_token_count,
        finish_reason: finished.finish_reason,
        kv_transfer_params: finished.kv_transfer_params,
    }))
}

async fn emit_dialect_terminal(
    request_id: &ServeRequestId,
    event_context: &EventContext,
    started: &Instant,
    queue_us: Option<u64>,
    first_visible_output_us: Option<u64>,
    image_count: u32,
    image_steps: u32,
    finish_detail: Option<String>,
    done: ChatDone,
    y: &mut TryYielder<ServeEvent, ServeError>,
) {
    debug_assert_eq!(
        done.output_token_count,
        done.visible_output_token_count
            .saturating_add(done.internal_token_count)
    );
    let mut cache = CacheAccounting::from(&event_context.cache);
    cache.transfer = done.kv_transfer_params;
    y.yield_ok(ServeEvent::Usage {
        prompt_tokens: done.prompt_token_count.min(u32::MAX as usize) as u32,
        visible_output_tokens: done.visible_output_token_count.min(u32::MAX as usize) as u32,
        internal_tokens: done.internal_token_count.min(u32::MAX as usize) as u32,
        image_count,
        image_steps,
        cache,
        resources: ResourceAccounting::from(&event_context.resources),
        timings: RuntimeTimings {
            compile_us: event_context.compile_duration_us,
            queue_us,
            first_visible_output_us,
            total_us: started.elapsed().as_micros() as u64,
        },
    })
    .await;
    match done.finish_reason {
        FinishReason::Cancelled => {
            y.yield_ok(ServeEvent::Cancelled {
                request_id: request_id.clone(),
            })
            .await;
        }
        FinishReason::Abort | FinishReason::Aborted => {
            y.yield_ok(ServeEvent::Aborted {
                request_id: request_id.clone(),
            })
            .await;
        }
        FinishReason::Error => {
            y.yield_ok(ServeEvent::Failed {
                request_id: request_id.clone(),
                message: finish_detail.unwrap_or_else(|| "engine execution failed".to_string()),
            })
            .await;
        }
        reason => {
            y.yield_ok(ServeEvent::Finished {
                candidate_id: CandidateId::PRIMARY,
                reason: FinishStatus::from(&reason),
                finish_detail,
            })
            .await;
        }
    }
}

#[try_stream]
async fn dialect_event_stream(
    request_id: ServeRequestId,
    event_context: EventContext,
    prompt_token_ids: Vec<u32>,
    tokenizer: uniserve_model_profile::tokenizer::DynTokenizer,
    prompt_logprobs_requested: bool,
    generated_logprobs_requested: bool,
    mut decode_options: TextDecodeOptions,
    mut processor: dialect_generation::DialectStreamProcessor,
    chat_processor: Option<crate::chat::DynChatOutputProcessor>,
    mut stream: GenerationEventStream,
    mut y: TryYielder<ServeEvent, ServeError>,
) -> Result<()> {
    let started = Instant::now();
    let mut first_visible_output_us = None;
    let mut queue_us = None;
    let mut emitted_output_tokens = 0_u32;
    let mut image_count = 0_u32;
    let mut image_steps = 0_u32;
    let expected_prompt_positions = prompt_token_ids.len().saturating_sub(1);
    let mut prompt_positions = Vec::new();
    let mut accepted = false;
    let mut pending_scheduled = None;
    let mut pending_token = None;
    let mut last_public_commit = None;
    let mut pending_image_events = Vec::new();
    let mut chat_bridge = chat_processor
        .map(ChatOutputBridge::new)
        .transpose()
        .map_err(|error| ServeError::OutputProcessing {
            request_id: request_id.clone(),
            message: error.to_string(),
        })?;
    let mut decoder = tokenizer.create_decode_stream(
        &prompt_token_ids,
        decode_options.skip_special_tokens,
        crate::text::output::stop_string_holdback_bytes(&decode_options),
    );
    macro_rules! emit_accepted {
        ($prompt_logprobs:expr) => {{
            let prompt_logprobs: Option<DecodedPromptLogprobs> = $prompt_logprobs;
            y.yield_ok(ServeEvent::Accepted {
                request_id: request_id.clone(),
                profile_id: event_context.profile_id.clone(),
                dialect_id: event_context.dialect_id.clone(),
                compile_duration_us: event_context.compile_duration_us,
                prompt_token_count: prompt_token_ids.len(),
                prompt_token_ids: prompt_token_ids.clone(),
                prompt_logprobs: prompt_logprobs.clone(),
            })
            .await;
            if let Some(bridge) = chat_bridge.as_mut() {
                let events = bridge
                    .push(
                        &request_id,
                        DecodedTextEvent::Start {
                            prompt_token_ids: Arc::from(prompt_token_ids.clone()),
                            prompt_logprobs,
                            queued_at: None,
                            scheduled_at: None,
                        },
                    )
                    .await?;
                for event in events {
                    if !matches!(map_chat_event(event), MappedChatEvent::Ignore) {
                        return Err(ServeError::OutputProcessing {
                            request_id: request_id.clone(),
                            message: "chat output processor emitted output while initializing"
                                .to_string(),
                        });
                    }
                }
            }
            accepted = true;
            if let Some((queued_at, scheduled_at)) = pending_scheduled.take() {
                y.yield_ok(ServeEvent::Scheduled {
                    request_id: request_id.clone(),
                    queued_at: Some(queued_at),
                    scheduled_at: Some(scheduled_at),
                    cache: CacheAccounting::from(&event_context.cache),
                    resources: ResourceAccounting::from(&event_context.resources),
                })
                .await;
            }
        }};
    }
    macro_rules! flush_pending_images {
        () => {{
            for event in pending_image_events.drain(..) {
                y.yield_ok(event).await;
            }
        }};
    }
    macro_rules! consume_token {
        ($id:expr, $logprobs:expr, $public_commit:expr) => {{
            let id = $id;
            let logprobs: Option<DecodedLogprobs> = $logprobs;
            let public_commit: Option<PublicCommit> = $public_commit;
            if public_commit.is_some() {
                last_public_commit = public_commit.clone();
            }
            flush_pending_images!();
            emitted_output_tokens = emitted_output_tokens.saturating_add(1);
            let new_bytes =
                decoder
                    .push_token(id)
                    .map_err(|error| ServeError::OutputProcessing {
                        request_id: request_id.clone(),
                        message: error.to_string(),
                    })?;
            let matched_stop = if emitted_output_tokens > decode_options.min_tokens {
                decode_options.stop_strings.as_ref().and_then(|stops| {
                    crate::text::output::matches_stop_string(stops, decoder.output(), new_bytes)
                })
            } else {
                None
            };
            let (text, stop_string) = if let Some((index, offset)) = matched_stop {
                let stop_string = decode_options
                    .stop_strings
                    .as_mut()
                    .ok_or_else(|| ServeError::OutputProcessing {
                        request_id: request_id.clone(),
                        message: "stop-string match lost its configured stop set".to_string(),
                    })?
                    .swap_remove(index);
                let truncate_to = if decode_options.include_stop_str_in_output {
                    offset + stop_string.len()
                } else {
                    offset
                };
                let (last_chunk, _) = decoder.flush(Some(truncate_to)).map_err(|error| {
                    ServeError::OutputProcessing {
                        request_id: request_id.clone(),
                        message: error.to_string(),
                    }
                })?;
                (last_chunk.unwrap_or_default(), Some(stop_string))
            } else {
                (decoder.next_chunk().unwrap_or_default(), None)
            };
            let finished = stop_string
                .as_ref()
                .map(|stop_string| crate::text::Finished {
                    prompt_token_count: prompt_token_ids.len(),
                    output_token_count: emitted_output_tokens as usize,
                    internal_token_count: 0,
                    finish_reason: FinishReason::Stop(Some(StopReason::Text(stop_string.clone()))),
                    kv_transfer_params: None,
                });
            if stop_string.is_none() {
                stream.acknowledge_text_prefix();
            } else {
                stream.cancel_at_consumed_prefix(
                    uniserve_engine_gateway::transport::StreamCancelCause::StopStringMatched,
                );
            }
            let done = emit_dialect_text_update(
                &request_id,
                &mut processor,
                chat_bridge.as_mut(),
                text,
                vec![id],
                logprobs,
                public_commit,
                finished,
                &started,
                &mut first_visible_output_us,
                &mut y,
            )
            .await?;
            if stop_string.is_some() {
                let done = done.ok_or_else(|| ServeError::OutputProcessing {
                    request_id: request_id.clone(),
                    message: "output processor omitted the stop-string terminal event".to_string(),
                })?;
                emit_dialect_terminal(
                    &request_id,
                    &event_context,
                    &started,
                    queue_us,
                    first_visible_output_us,
                    image_count,
                    image_steps,
                    Some("stop".to_string()),
                    done,
                    &mut y,
                )
                .await;
                return Ok(());
            }
        }};
    }
    macro_rules! ensure_output_ready {
        ($kind:literal) => {{
            if !accepted || pending_token.is_some() {
                return Err(ServeError::OutputProcessing {
                    request_id: request_id.clone(),
                    message: concat!($kind, " arrived before output metadata was complete")
                        .to_string(),
                });
            }
        }};
    }
    if !prompt_logprobs_requested || expected_prompt_positions == 0 {
        let prompt_logprobs =
            if prompt_logprobs_requested {
                let first_token_id = prompt_token_ids.first().copied().ok_or_else(|| {
                    ServeError::OutputProcessing {
                        request_id: request_id.clone(),
                        message: "prompt logprobs require a non-empty tokenized prompt".to_string(),
                    }
                })?;
                let first_token = tokenizer
                    .decode(&[first_token_id], event_context.skip_special_tokens)
                    .map_err(|error| ServeError::OutputProcessing {
                        request_id: request_id.clone(),
                        message: error.to_string(),
                    })?;
                Some(DecodedPromptLogprobs {
                    first_token_id,
                    first_token,
                    scored_positions: Vec::new(),
                })
            } else {
                None
            };
        emit_accepted!(prompt_logprobs);
    }
    while let Some(event) = stream.next().await {
        match event {
            GenEvent::Scheduled {
                queued_at,
                scheduled_at,
            } => {
                queue_us = Some(((scheduled_at - queued_at).max(0.0) * 1_000_000.0) as u64);
                if accepted {
                    y.yield_ok(ServeEvent::Scheduled {
                        request_id: request_id.clone(),
                        queued_at: Some(queued_at),
                        scheduled_at: Some(scheduled_at),
                        cache: CacheAccounting::from(&event_context.cache),
                        resources: ResourceAccounting::from(&event_context.resources),
                    })
                    .await;
                } else {
                    pending_scheduled = Some((queued_at, scheduled_at));
                }
                event_context
                    .metrics
                    .scheduled
                    .fetch_add(1, Ordering::Relaxed);
            }
            GenEvent::PromptLogprobs { positions } => {
                prompt_positions.extend(positions);
                if prompt_positions.len() > expected_prompt_positions {
                    return Err(ServeError::OutputProcessing {
                        request_id: request_id.clone(),
                        message: "engine returned more prompt logprob positions than requested"
                            .to_string(),
                    });
                }
                if !accepted && prompt_positions.len() == expected_prompt_positions {
                    let positions = std::mem::take(&mut prompt_positions);
                    let decoded = crate::text::output::decode_prompt_logprobs(
                        &request_id,
                        tokenizer.as_ref(),
                        &prompt_token_ids,
                        &positions,
                        event_context.skip_special_tokens,
                    )
                    .map_err(|error| ServeError::OutputProcessing {
                        request_id: request_id.clone(),
                        message: error.to_string(),
                    })?;
                    emit_accepted!(Some(decoded));
                }
            }
            GenEvent::TextToken {
                id, public_commit, ..
            } => {
                if !accepted {
                    return Err(ServeError::OutputProcessing {
                        request_id: request_id.clone(),
                        message: "engine began generation before prompt logprobs were complete"
                            .to_string(),
                    });
                }
                if pending_token.is_some() {
                    return Err(ServeError::OutputProcessing {
                        request_id: request_id.clone(),
                        message: "engine emitted a new token before resolving prior logprobs"
                            .to_string(),
                    });
                }
                if generated_logprobs_requested {
                    pending_token = Some((id, public_commit));
                } else {
                    consume_token!(id, None, public_commit);
                }
            }
            GenEvent::TokenLogprobs { id, candidates } => {
                let (pending, public_commit) =
                    pending_token
                        .take()
                        .ok_or_else(|| ServeError::OutputProcessing {
                            request_id: request_id.clone(),
                            message: "engine returned token logprobs without a pending token"
                                .to_string(),
                        })?;
                if pending != id || candidates.first().is_none_or(|entry| entry.token_id != id) {
                    return Err(ServeError::OutputProcessing {
                        request_id: request_id.clone(),
                        message: "token logprobs do not match the emitted token".to_string(),
                    });
                }
                let logprobs = crate::text::output::decode_logprobs(
                    tokenizer.as_ref(),
                    &[
                        uniserve_engine_gateway::generation::GenerationPositionLogprobs {
                            entries: candidates,
                        },
                    ],
                    event_context.skip_special_tokens,
                )
                .map_err(|error| ServeError::OutputProcessing {
                    request_id: request_id.clone(),
                    message: error.to_string(),
                })?;
                consume_token!(id, Some(logprobs), public_commit);
            }
            GenEvent::ImageBegin {
                image_id,
                height,
                width,
                steps,
            } => {
                ensure_output_ready!("image-begin event");
                first_visible_output_us.get_or_insert_with(|| started.elapsed().as_micros() as u64);
                y.yield_ok(ServeEvent::ImageBegin {
                    candidate_id: CandidateId::PRIMARY,
                    image_id: image_id.to_string(),
                    width: Some(width),
                    height: Some(height),
                    steps: Some(steps as u32),
                    elapsed_us: started.elapsed().as_micros() as u64,
                })
                .await;
            }
            GenEvent::ImageStep { image_id, step } => {
                ensure_output_ready!("image-step event");
                image_steps = image_steps.saturating_add(1);
                y.yield_ok(ServeEvent::ImageStep {
                    candidate_id: CandidateId::PRIMARY,
                    image_id: image_id.to_string(),
                    step: step as u32,
                    elapsed_us: started.elapsed().as_micros() as u64,
                })
                .await;
            }
            GenEvent::ImageCommit { image_id } => {
                ensure_output_ready!("image-commit event");
                pending_image_events.push(ServeEvent::ImageCommit {
                    candidate_id: CandidateId::PRIMARY,
                    image_id: image_id.to_string(),
                    elapsed_us: started.elapsed().as_micros() as u64,
                });
            }
            GenEvent::ImageDone {
                image_id,
                height,
                width,
                bytes,
                sha256,
                pixels_png_b64,
                public_commit,
            } => {
                ensure_output_ready!("image-done event");
                image_count = image_count.saturating_add(1);
                if let Some(commit) = public_commit {
                    pending_image_events.push(ServeEvent::PublicCommit { commit });
                }
                pending_image_events.push(ServeEvent::ImageDone {
                    candidate_id: CandidateId::PRIMARY,
                    image_id: image_id.to_string(),
                    width: Some(width),
                    height: Some(height),
                    bytes: Some(bytes),
                    sha256: Some(sha256),
                    pixels_png_b64: Some(pixels_png_b64),
                    elapsed_us: started.elapsed().as_micros() as u64,
                });
            }
            GenEvent::Finished {
                reason,
                stop_reason,
                prompt_tokens,
                completion_tokens,
                images,
                kv_transfer_params,
            } => {
                ensure_output_ready!("terminal event");
                flush_pending_images!();
                let (last_chunk, _) =
                    decoder
                        .flush(None)
                        .map_err(|error| ServeError::OutputProcessing {
                            request_id: request_id.clone(),
                            message: error.to_string(),
                        })?;
                let finish_detail = generation_finish_detail(&reason).to_string();
                let finished = crate::text::Finished {
                    prompt_token_count: prompt_tokens,
                    output_token_count: completion_tokens,
                    internal_token_count: completion_tokens
                        .saturating_sub(emitted_output_tokens as usize),
                    finish_reason: generation_text_finish_reason(reason, stop_reason),
                    kv_transfer_params,
                };
                let done = emit_dialect_text_update(
                    &request_id,
                    &mut processor,
                    chat_bridge.as_mut(),
                    last_chunk.unwrap_or_default(),
                    Vec::new(),
                    None,
                    last_public_commit,
                    Some(finished),
                    &started,
                    &mut first_visible_output_us,
                    &mut y,
                )
                .await?
                .ok_or_else(|| ServeError::OutputProcessing {
                    request_id: request_id.clone(),
                    message: "output processor omitted the engine terminal event".to_string(),
                })?;
                image_count = image_count.max(images.min(u32::MAX as usize) as u32);
                emit_dialect_terminal(
                    &request_id,
                    &event_context,
                    &started,
                    queue_us,
                    first_visible_output_us,
                    image_count,
                    image_steps,
                    Some(finish_detail),
                    done,
                    &mut y,
                )
                .await;
                return Ok(());
            }
            GenEvent::Rejected { message } => {
                flush_pending_images!();
                y.yield_ok(ServeEvent::Rejected {
                    request_id: request_id.clone(),
                    message,
                })
                .await;
                return Ok(());
            }
            GenEvent::Error { message } => {
                flush_pending_images!();
                y.yield_ok(ServeEvent::Failed {
                    request_id: request_id.clone(),
                    message,
                })
                .await;
                return Ok(());
            }
        }
    }

    flush_pending_images!();
    Err(ServeError::OutputProcessing {
        request_id,
        message: "engine stream closed before a terminal event".to_string(),
    })
}

fn generation_text_finish_reason(
    reason: GenerationFinishReason,
    stop_reason: Option<String>,
) -> FinishReason {
    match reason {
        GenerationFinishReason::Eos | GenerationFinishReason::ImageDone => FinishReason::Stop(None),
        GenerationFinishReason::Stop => FinishReason::Stop(stop_reason.map(|reason| {
            reason
                .strip_prefix("token:")
                .and_then(|id| id.parse().ok())
                .map_or_else(|| StopReason::Text(reason), StopReason::TokenId)
        })),
        GenerationFinishReason::MaxTokens => FinishReason::Length,
        GenerationFinishReason::Cancelled => FinishReason::Cancelled,
        GenerationFinishReason::Aborted => FinishReason::Aborted,
        GenerationFinishReason::Repetition => FinishReason::Repetition,
        GenerationFinishReason::Error => FinishReason::Error,
    }
}

fn generation_finish_detail(reason: &GenerationFinishReason) -> &'static str {
    match reason {
        GenerationFinishReason::Eos => "eos",
        GenerationFinishReason::MaxTokens => "max_tokens",
        GenerationFinishReason::Stop => "stop",
        GenerationFinishReason::ImageDone => "image_done",
        GenerationFinishReason::Cancelled => "cancelled",
        GenerationFinishReason::Aborted => "aborted",
        GenerationFinishReason::Repetition => "repetition",
        GenerationFinishReason::Error => "error",
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
        structured_outputs: generation.structured_output.clone(),
        skip_reading_prefix_cache: Some(cache.bypass_read),
        write_prefix_cache: Some(!cache.no_store),
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

fn generation_from_sampling(
    _request_id: &str,
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
        write_prefix_cache: _,
    } = sampling;

    Ok(GenerationPolicy {
        constraint: GenerationConstraint::UndOnly,
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
        structured_output: structured_outputs,
        skip_special_tokens: decode.skip_special_tokens,
        include_stop_string_in_output: decode.include_stop_str_in_output,
        output_detail: OutputDetail::default(),
        intermediate: true,
        add_special_tokens: false,
        image: ImageGenerationPolicy::default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct ByteTokenizer;

    impl uniserve_model_profile::tokenizer::Tokenizer for ByteTokenizer {
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
            Ok(String::from_utf8_lossy(
                &token_ids
                    .iter()
                    .map(|token_id| *token_id as u8)
                    .collect::<Vec<_>>(),
            )
            .into_owned())
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

    fn dialect_event_context() -> EventContext {
        EventContext {
            profile_id: "profile".to_string(),
            dialect_id: "bagel".to_string(),
            compile_duration_us: 7,
            cache: CompiledCachePolicy {
                namespace: None,
                key_fingerprint: "cache".to_string(),
                read_enabled: true,
                write_enabled: true,
                replayable: true,
                encoder_pin_count: 0,
            },
            resources: ResourceBounds {
                max_context_tokens: Some(32),
                prompt_tokens: 1,
                expected_kv_tokens: 2,
                image_count: 0,
                image_latent_units: 0,
                scratch_units: 0,
                host_scratch_tokens: 0,
                encoder_cache_pins: 0,
                grammar_states: 0,
                adapter_slots: 0,
                replayable: true,
                required_features: vec!["prefill_und".to_string(), "decode_und".to_string()],
            },
            skip_special_tokens: false,
            metrics: Arc::new(RuntimeLifecycleMetrics::default()),
        }
    }

    fn public_commit(event_seq: u64, modality: PublicModality) -> PublicCommit {
        PublicCommit {
            event_seq,
            modality,
            committed_at: event_seq as f64,
            semantic_root: SemanticRoot {
                producer_op_id: event_seq,
                point_index: event_seq as u32,
                semantic_digest: format!("{event_seq:064x}"),
            },
        }
    }

    fn bagel_processor(
        tokenizer: uniserve_model_profile::tokenizer::DynTokenizer,
    ) -> dialect_generation::DialectStreamProcessor {
        bagel_processor_with_reasoning(tokenizer, true)
    }

    fn bagel_processor_with_reasoning(
        tokenizer: uniserve_model_profile::tokenizer::DynTokenizer,
        profile_reasoning: bool,
    ) -> dialect_generation::DialectStreamProcessor {
        let dialect = uniserve_model_profile::dialect::resolve_generation_dialect_for_model(
            "bagel",
            tokenizer.as_ref(),
        )
        .expect("resolve dialect")
        .expect("BAGEL dialect");
        dialect_generation::DialectStreamProcessor::new(
            tokenizer,
            &dialect,
            &[b'p' as u32],
            profile_reasoning,
        )
        .expect("stream processor")
    }

    #[tokio::test]
    async fn dialect_stream_attaches_ranked_logprobs_to_text_delta() {
        let tokenizer: uniserve_model_profile::tokenizer::DynTokenizer = Arc::new(ByteTokenizer);
        let processor = bagel_processor(Arc::clone(&tokenizer));
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        tx.try_send(GenEvent::Scheduled {
            queued_at: 1.0,
            scheduled_at: 2.0,
        })
        .unwrap();
        tx.try_send(GenEvent::TextToken {
            id: b'a' as u32,
            logprob: Some(-0.25),
            public_commit: None,
        })
        .unwrap();
        tx.try_send(GenEvent::TokenLogprobs {
            id: b'a' as u32,
            candidates: vec![uniserve_engine_gateway::GenerationTokenLogprob {
                token_id: b'a' as u32,
                logprob: -0.25,
                rank: 3,
            }],
        })
        .unwrap();
        tx.try_send(GenEvent::Finished {
            reason: GenerationFinishReason::MaxTokens,
            stop_reason: None,
            prompt_tokens: 1,
            completion_tokens: 1,
            images: 0,
            kv_transfer_params: None,
        })
        .unwrap();
        drop(tx);

        let events = dialect_event_stream(
            "req".into(),
            dialect_event_context(),
            vec![b'p' as u32],
            tokenizer,
            false,
            true,
            TextDecodeOptions::default(),
            processor,
            None,
            GenerationEventStream::new(rx),
        )
        .collect::<Vec<_>>()
        .await;
        let delta = events
            .iter()
            .find_map(|event| match event {
                Ok(ServeEvent::TextDelta {
                    text,
                    token_ids,
                    logprobs,
                    ..
                }) => Some((text, token_ids, logprobs.as_ref())),
                _ => None,
            })
            .expect("text delta");
        assert_eq!(delta.0, "a");
        assert_eq!(delta.1, &[b'a' as u32]);
        let candidate = &delta.2.expect("decoded logprobs").positions[0].entries[0];
        assert_eq!(candidate.token_id, b'a' as u32);
        assert_eq!(candidate.rank, 3);
        assert!(events.iter().all(Result::is_ok));
        assert!(matches!(
            events.last(),
            Some(Ok(ServeEvent::Finished {
                reason: FinishStatus::Length,
                ..
            }))
        ));
    }

    #[tokio::test]
    async fn dialect_stream_publishes_an_image_with_its_next_text_token() {
        let tokenizer: uniserve_model_profile::tokenizer::DynTokenizer = Arc::new(ByteTokenizer);
        let processor = bagel_processor(Arc::clone(&tokenizer));
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        tx.try_send(GenEvent::Scheduled {
            queued_at: 1.0,
            scheduled_at: 2.0,
        })
        .unwrap();

        let events = dialect_event_stream(
            "feedback-output".into(),
            dialect_event_context(),
            vec![b'p' as u32],
            tokenizer,
            false,
            false,
            TextDecodeOptions::default(),
            processor,
            None,
            GenerationEventStream::new(rx),
        );
        tokio::pin!(events);
        assert!(matches!(
            events.next().await,
            Some(Ok(ServeEvent::Accepted { .. }))
        ));
        assert!(matches!(
            events.next().await,
            Some(Ok(ServeEvent::Scheduled { .. }))
        ));

        tx.send(GenEvent::TextToken {
            id: b'a' as u32,
            logprob: None,
            public_commit: Some(public_commit(1, PublicModality::Text)),
        })
        .await
        .unwrap();
        assert!(matches!(
            events.next().await,
            Some(Ok(ServeEvent::PublicCommit {
                commit: PublicCommit { event_seq: 1, .. }
            }))
        ));
        assert!(matches!(
            events.next().await,
            Some(Ok(ServeEvent::TextDelta { text, .. })) if text == "a"
        ));

        tx.send(GenEvent::ImageCommit { image_id: 0 })
            .await
            .unwrap();
        tx.send(GenEvent::ImageDone {
            image_id: 0,
            height: 1,
            width: 1,
            bytes: 3,
            sha256: "image".to_string(),
            pixels_png_b64: "cG5n".to_string(),
            public_commit: Some(public_commit(3, PublicModality::Image)),
        })
        .await
        .unwrap();
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(20), events.next())
                .await
                .is_err(),
            "the image became public before a continuation token arrived"
        );

        tx.send(GenEvent::TextToken {
            id: b'b' as u32,
            logprob: None,
            public_commit: Some(public_commit(4, PublicModality::Text)),
        })
        .await
        .unwrap();
        assert!(matches!(
            events.next().await,
            Some(Ok(ServeEvent::ImageCommit { .. }))
        ));
        assert!(matches!(
            events.next().await,
            Some(Ok(ServeEvent::PublicCommit {
                commit: PublicCommit { event_seq: 3, .. }
            }))
        ));
        assert!(matches!(
            events.next().await,
            Some(Ok(ServeEvent::ImageDone { .. }))
        ));
        assert!(matches!(
            events.next().await,
            Some(Ok(ServeEvent::PublicCommit {
                commit: PublicCommit { event_seq: 4, .. }
            }))
        ));
        assert!(matches!(
            events.next().await,
            Some(Ok(ServeEvent::TextDelta { text, .. })) if text == "b"
        ));

        tx.send(GenEvent::Finished {
            reason: GenerationFinishReason::Eos,
            stop_reason: None,
            prompt_tokens: 1,
            completion_tokens: 2,
            images: 1,
            kv_transfer_params: None,
        })
        .await
        .unwrap();
        drop(tx);
        let terminal = events.collect::<Vec<_>>().await;
        assert!(matches!(
            terminal.last(),
            Some(Ok(ServeEvent::Finished {
                reason: FinishStatus::Stop { .. },
                ..
            }))
        ));
    }

    #[tokio::test]
    async fn dialect_stream_rejects_terminal_output_with_pending_logprobs() {
        let tokenizer: uniserve_model_profile::tokenizer::DynTokenizer = Arc::new(ByteTokenizer);
        let processor = bagel_processor(Arc::clone(&tokenizer));
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        tx.try_send(GenEvent::Scheduled {
            queued_at: 1.0,
            scheduled_at: 2.0,
        })
        .unwrap();
        tx.try_send(GenEvent::TextToken {
            id: b'a' as u32,
            logprob: Some(-0.25),
            public_commit: None,
        })
        .unwrap();
        tx.try_send(GenEvent::Finished {
            reason: GenerationFinishReason::MaxTokens,
            stop_reason: None,
            prompt_tokens: 1,
            completion_tokens: 1,
            images: 0,
            kv_transfer_params: None,
        })
        .unwrap();
        drop(tx);

        let events = dialect_event_stream(
            "req".into(),
            dialect_event_context(),
            vec![b'p' as u32],
            tokenizer,
            false,
            true,
            TextDecodeOptions::default(),
            processor,
            None,
            GenerationEventStream::new(rx),
        )
        .collect::<Vec<_>>()
        .await;
        assert!(matches!(
            events.last(),
            Some(Err(ServeError::OutputProcessing { message, .. }))
                if message.contains("before output metadata was complete")
        ));
    }

    #[tokio::test]
    async fn dialect_stream_truncates_at_runtime_stop_strings() {
        let tokenizer: uniserve_model_profile::tokenizer::DynTokenizer = Arc::new(ByteTokenizer);
        let processor = bagel_processor(Arc::clone(&tokenizer));
        let (tx, rx) = tokio::sync::mpsc::channel(32);
        tx.try_send(GenEvent::Scheduled {
            queued_at: 1.0,
            scheduled_at: 2.0,
        })
        .unwrap();
        for &byte in b"abSTOPignored" {
            tx.try_send(GenEvent::TextToken {
                id: u32::from(byte),
                logprob: None,
                public_commit: None,
            })
            .unwrap();
        }
        drop(tx);

        let events = dialect_event_stream(
            "stop-string".into(),
            dialect_event_context(),
            vec![b'p' as u32],
            tokenizer,
            false,
            false,
            TextDecodeOptions {
                skip_special_tokens: false,
                include_stop_str_in_output: false,
                stop_strings: Some(vec!["STOP".to_string()]),
                min_tokens: 0,
            },
            processor,
            None,
            GenerationEventStream::new(rx),
        )
        .collect::<Vec<_>>()
        .await;

        let visible = events
            .iter()
            .filter_map(|event| match event {
                Ok(ServeEvent::TextDelta { text, .. }) => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(visible, "ab");
        assert!(events.iter().any(|event| matches!(
            event,
            Ok(ServeEvent::Usage {
                visible_output_tokens: 6,
                ..
            })
        )));
        assert!(matches!(
            events.last(),
            Some(Ok(ServeEvent::Finished {
                reason: FinishStatus::Stop {
                    cause: Some(StopCause::Text(stop)),
                },
                ..
            })) if stop == "STOP"
        ));
    }

    #[tokio::test]
    async fn dialect_chat_stream_emits_reasoning_and_tool_semantics() {
        let tokenizer: uniserve_model_profile::tokenizer::DynTokenizer = Arc::new(ByteTokenizer);
        let processor = bagel_processor_with_reasoning(Arc::clone(&tokenizer), false);
        let mut chat_request = ChatRequest::for_test();
        chat_request.tools = vec![ChatTool {
            name: "get_weather".to_string(),
            description: Some("Get weather".to_string()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
            }),
            strict: None,
        }];
        chat_request.tool_choice = ChatToolChoice::Auto;
        let chat_processor = crate::chat::DefaultChatOutputProcessor::new(
            &mut chat_request,
            "qwen-test",
            Arc::clone(&tokenizer),
            &crate::chat::ParserSelection::Explicit("qwen3_xml".to_string()),
            &crate::chat::ParserSelection::Explicit("qwen3".to_string()),
        )
        .expect("chat output processor");
        let output = concat!(
            "<think>check weather</think>",
            "<tool_call>\n",
            "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}\n",
            "</tool_call>"
        );
        let (tx, rx) = tokio::sync::mpsc::channel(output.len() + 2);
        tx.try_send(GenEvent::Scheduled {
            queued_at: 1.0,
            scheduled_at: 2.0,
        })
        .unwrap();
        for byte in output.bytes() {
            tx.try_send(GenEvent::TextToken {
                id: u32::from(byte),
                logprob: None,
                public_commit: None,
            })
            .unwrap();
        }
        tx.try_send(GenEvent::Finished {
            reason: GenerationFinishReason::Eos,
            stop_reason: None,
            prompt_tokens: 1,
            completion_tokens: output.len(),
            images: 0,
            kv_transfer_params: None,
        })
        .unwrap();
        drop(tx);

        let events = dialect_event_stream(
            "structured-chat".into(),
            dialect_event_context(),
            vec![b'p' as u32],
            tokenizer,
            false,
            false,
            TextDecodeOptions {
                skip_special_tokens: false,
                ..TextDecodeOptions::default()
            },
            processor,
            Some(Box::new(chat_processor)),
            GenerationEventStream::new(rx),
        )
        .collect::<Vec<_>>()
        .await;

        let reasoning = events
            .iter()
            .filter_map(|event| match event {
                Ok(ServeEvent::ReasoningDelta { text, .. }) => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        let arguments = events
            .iter()
            .filter_map(|event| match event {
                Ok(ServeEvent::ToolCallArgumentsDelta { delta, .. }) => Some(delta.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(reasoning, "check weather");
        assert!(events.iter().any(|event| matches!(
            event,
            Ok(ServeEvent::ToolCallStart { name, .. }) if name == "get_weather"
        )));
        assert_eq!(arguments, r#"{"city":"Paris"}"#);
        assert!(events.iter().any(|event| matches!(
            event,
            Ok(ServeEvent::ToolCallEnd { name, arguments, .. })
                if name == "get_weather" && arguments == r#"{"city":"Paris"}"#
        )));
        assert!(matches!(
            events.last(),
            Some(Ok(ServeEvent::Finished {
                reason: FinishStatus::Stop { .. },
                ..
            }))
        ));
    }

    fn image_chat_request() -> ServeRequest {
        let mut request = ServeRequest::chat(
            "image-chat",
            vec![
                ChatMessage::system("policy"),
                ChatMessage::user(ChatContent::Parts(vec![
                    ChatContentPart::Text {
                        text: "describe".to_string(),
                    },
                    ChatContentPart::ImageUrl {
                        image_url: "data:image/png;base64,aW1hZ2U=".to_string(),
                        detail: None,
                        uuid: None,
                    },
                ])),
            ],
        );
        request.modalities.input_image = true;
        request.generation.constraint = GenerationConstraint::UndOnly;
        request
    }

    #[test]
    fn image_chat_replaces_images_without_flattening_message_order() {
        let request = image_chat_request();
        assert_eq!(semantic_image_count(&request.model_context), 1);
        let request_id = request.request_id.clone();
        let ModelContext::Chat { mut messages, .. } = request.model_context else {
            panic!("chat context");
        };
        let (images, placeholders) =
            replace_chat_images(&request_id, &mut messages, Some("<img></img>"))
                .expect("semantic image context");

        assert!(matches!(&messages[0], ChatMessage::System { .. }));
        assert!(matches!(&messages[1], ChatMessage::User { .. }));
        let ChatMessage::User {
            content: ChatContent::Parts(parts),
        } = &messages[1]
        else {
            panic!("user parts");
        };
        assert!(matches!(
            &parts[0],
            ChatContentPart::Text { text } if text == "describe"
        ));
        assert!(matches!(
            &parts[1],
            ChatContentPart::Text { text } if text == "<img></img>"
        ));
        assert_eq!(images[0].b64, "aW1hZ2U=");
        assert_eq!(placeholders, vec!["<img></img>"]);
    }

    #[test]
    fn image_chat_requires_executable_profile_capability() {
        let request = image_chat_request();
        let profile = ModelProfile::loaded("text-profile");
        assert!(matches!(
            validate_request_capabilities_for_profile(&profile, &request),
            Err(ServeError::UnsupportedCapability {
                capability: "image_input",
                ..
            })
        ));
    }

    #[test]
    fn chat_template_override_requires_explicit_profile_permission() {
        let mut request = ServeRequest::chat("req", vec![ChatMessage::user("hello")]);
        let ModelContext::Chat { chat_options, .. } = &mut request.model_context else {
            panic!("expected chat context");
        };
        chat_options.chat_template = Some("{{ messages }}".to_string());
        let mut profile = ModelProfile::loaded("test-profile");

        assert!(matches!(
            validate_request_capabilities_for_profile(&profile, &request),
            Err(ServeError::UnsupportedCapability {
                capability: "request_chat_template_override",
                ..
            })
        ));

        profile.render.request_chat_template_override_allowed = true;
        assert!(validate_request_capabilities_for_profile(&profile, &request).is_ok());
    }

    #[test]
    fn lifecycle_tracking_counts_unpolled_drop_as_cancellation() {
        let metrics = Arc::new(RuntimeLifecycleMetrics::default());
        let requests = Arc::new(RuntimeRequestRegistry::default());
        assert!(requests.register(
            "req".into(),
            "profile".into(),
            "dialect".into(),
            1,
            RequestLifecycleState::Submitting,
        ));
        let stream: ServeEventStream = Box::pin(futures::stream::pending());
        drop(LifecycleTrackedStream::new(
            "req".into(),
            stream,
            Arc::clone(&metrics),
            Arc::clone(&requests),
        ));

        assert_eq!(metrics.snapshot().cancelled, 1);
        assert_eq!(metrics.snapshot().failed, 0);
        assert_eq!(
            requests.stats("req").map(|stats| stats.state),
            Some(RequestLifecycleState::Cancelled)
        );
    }

    #[test]
    fn lifecycle_tracking_counts_terminal_event_exactly_once() {
        let metrics = Arc::new(RuntimeLifecycleMetrics::default());
        let requests = Arc::new(RuntimeRequestRegistry::default());
        assert!(requests.register(
            "req".into(),
            "profile".into(),
            "dialect".into(),
            1,
            RequestLifecycleState::Submitting,
        ));
        let stream: ServeEventStream =
            Box::pin(futures::stream::iter([Ok(ServeEvent::Finished {
                candidate_id: CandidateId::PRIMARY,
                reason: FinishStatus::Stop { cause: None },
                finish_detail: None,
            })]));
        let mut tracked = LifecycleTrackedStream::new(
            "req".into(),
            stream,
            Arc::clone(&metrics),
            Arc::clone(&requests),
        );
        let event = futures::executor::block_on(tracked.next());
        assert!(matches!(event, Some(Ok(ServeEvent::Finished { .. }))));
        drop(tracked);

        assert_eq!(metrics.snapshot().finished, 1);
        assert_eq!(metrics.snapshot().cancelled, 0);
        assert_eq!(
            requests.stats("req").map(|stats| stats.state),
            Some(RequestLifecycleState::Finished)
        );
    }

    #[test]
    fn lifecycle_tracking_preserves_abort_when_stream_is_dropped() {
        let metrics = Arc::new(RuntimeLifecycleMetrics::default());
        let requests = Arc::new(RuntimeRequestRegistry::default());
        assert!(requests.register(
            "req".into(),
            "profile".into(),
            "dialect".into(),
            1,
            RequestLifecycleState::Submitting,
        ));
        requests.mark_control("req", RequestLifecycleState::Aborting);
        let stream: ServeEventStream = Box::pin(futures::stream::pending());

        drop(LifecycleTrackedStream::new(
            "req".into(),
            stream,
            Arc::clone(&metrics),
            Arc::clone(&requests),
        ));

        assert_eq!(metrics.snapshot().aborted, 1);
        assert_eq!(metrics.snapshot().cancelled, 0);
        assert_eq!(
            requests.stats("req").map(|stats| stats.state),
            Some(RequestLifecycleState::Aborted)
        );
    }

    #[test]
    fn admission_control_cannot_be_overwritten_by_acceptance() {
        let requests = RuntimeRequestRegistry::default();
        assert!(requests.register(
            "req".into(),
            "profile".into(),
            "dialect".into(),
            1,
            RequestLifecycleState::Submitting,
        ));

        assert!(requests.mark_control("req", RequestLifecycleState::Cancelling));
        assert!(!requests.accept("req"));
        assert_eq!(
            requests.stats("req").map(|stats| stats.state),
            Some(RequestLifecycleState::Cancelling)
        );
    }

    #[test]
    fn control_state_cannot_be_overwritten_by_submission_or_stream_events() {
        let requests = RuntimeRequestRegistry::default();
        assert!(requests.register(
            "req".into(),
            "profile".into(),
            "dialect".into(),
            1,
            RequestLifecycleState::Compiling,
        ));

        assert!(!requests.mark_control("req", RequestLifecycleState::Cancelling));
        requests.mark_submitting("req", 7);
        assert_eq!(
            requests.observe(
                "req",
                &ServeEvent::TextDelta {
                    candidate_id: CandidateId::PRIMARY,
                    text: "buffered".into(),
                    token_ids: vec![1],
                    logprobs: None,
                },
                9,
            ),
            Some(LifecycleTerminal::Cancelled)
        );
        assert_eq!(
            requests.stats("req").map(|stats| stats.state),
            Some(RequestLifecycleState::Cancelling)
        );
    }

    #[test]
    fn abort_control_cannot_be_downgraded_to_cancel() {
        let requests = RuntimeRequestRegistry::default();
        assert!(requests.register(
            "req".into(),
            "profile".into(),
            "dialect".into(),
            1,
            RequestLifecycleState::Accepted,
        ));

        assert!(requests.mark_control("req", RequestLifecycleState::Aborting));
        assert!(!requests.mark_control("req", RequestLifecycleState::Cancelling));
        assert_eq!(
            requests.control_terminal("req"),
            Some(LifecycleTerminal::Aborted)
        );
    }

    #[tokio::test]
    async fn active_stream_emits_runtime_control_terminal() {
        let requests = Arc::new(RuntimeRequestRegistry::default());
        assert!(requests.register(
            "req".into(),
            "profile".into(),
            "dialect".into(),
            1,
            RequestLifecycleState::Accepted,
        ));
        let source = Box::pin(futures::stream::pending()) as ServeEventStream;
        let mut stream = Box::pin(control_aware_event_stream(
            "req".into(),
            Arc::clone(&requests),
            source,
        ));

        assert!(requests.mark_control("req", RequestLifecycleState::Aborting));
        assert!(matches!(
            stream.next().await,
            Some(Ok(ServeEvent::Aborted { request_id })) if request_id.as_ref() == "req"
        ));
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn buffered_engine_terminal_cannot_beat_runtime_cancel() {
        let requests = Arc::new(RuntimeRequestRegistry::default());
        assert!(requests.register(
            "req".into(),
            "profile".into(),
            "dialect".into(),
            1,
            RequestLifecycleState::Accepted,
        ));
        let source = Box::pin(futures::stream::iter([Ok(ServeEvent::Finished {
            candidate_id: CandidateId::PRIMARY,
            reason: FinishStatus::Stop { cause: None },
            finish_detail: None,
        })])) as ServeEventStream;
        let mut stream = Box::pin(control_aware_event_stream(
            "req".into(),
            Arc::clone(&requests),
            source,
        ));

        assert!(requests.mark_control("req", RequestLifecycleState::Cancelling));
        assert!(matches!(
            stream.next().await,
            Some(Ok(ServeEvent::Cancelled { request_id })) if request_id.as_ref() == "req"
        ));
        assert!(stream.next().await.is_none());
    }

    #[test]
    fn lifecycle_tracking_reclassifies_buffered_finish_after_abort() {
        let metrics = Arc::new(RuntimeLifecycleMetrics::default());
        let requests = Arc::new(RuntimeRequestRegistry::default());
        assert!(requests.register(
            "req".into(),
            "profile".into(),
            "dialect".into(),
            1,
            RequestLifecycleState::Accepted,
        ));
        let stream: ServeEventStream =
            Box::pin(futures::stream::iter([Ok(ServeEvent::Finished {
                candidate_id: CandidateId::PRIMARY,
                reason: FinishStatus::Stop { cause: None },
                finish_detail: None,
            })]));
        let mut tracked = LifecycleTrackedStream::new(
            "req".into(),
            stream,
            Arc::clone(&metrics),
            Arc::clone(&requests),
        );

        assert!(requests.mark_control("req", RequestLifecycleState::Aborting));
        assert!(matches!(
            futures::executor::block_on(tracked.next()),
            Some(Ok(ServeEvent::Aborted { request_id })) if request_id.as_ref() == "req"
        ));
        assert_eq!(metrics.snapshot().aborted, 1);
        assert_eq!(metrics.snapshot().finished, 0);
        assert_eq!(
            requests.stats("req").map(|stats| stats.state),
            Some(RequestLifecycleState::Aborted)
        );
    }

    #[tokio::test]
    async fn request_drain_waits_for_terminal_state() {
        let requests = Arc::new(RuntimeRequestRegistry::default());
        assert!(requests.register(
            "req".into(),
            "profile".into(),
            "dialect".into(),
            1,
            RequestLifecycleState::Submitting,
        ));
        let waiter = {
            let requests = Arc::clone(&requests);
            tokio::spawn(async move { requests.drain_request("req").await })
        };
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished());
        assert_eq!(
            requests.complete("req", RequestLifecycleState::Finished, 9),
            Some(RequestLifecycleState::Finished)
        );
        let stats = waiter.await.expect("drain task").expect("request stats");
        assert_eq!(stats.state, RequestLifecycleState::Finished);
        assert_eq!(stats.timings.total_us, 9);
    }

    #[test]
    fn text_request_inspection_redacts_prompt_text() {
        let profile = ModelProfile::text_only("test-profile");
        let inspection = PlanInspection {
            runtime_id: 1,
            request_id: "req".into(),
            profile_id: profile.profile_id().to_string(),
            dialect_id: profile.dialect_id().to_string(),
            tokenizer_fingerprint: profile.tokenizer_fingerprint().to_string(),
            config_fingerprint: profile.identity.config_fingerprint.clone(),
            capability_snapshot: RuntimeCapabilitySnapshot {
                supports_text: true,
                supports_chat: false,
                supports_multimodal_input: false,
                supports_image_output: false,
                supports_structured_output: true,
                supports_logprobs: true,
                supports_prefix_cache: true,
                supports_adapters: true,
                engine_count: 1,
                max_model_tokens: 32_768,
                model_dtype: uniserve_core::ModelDtype::BFloat16,
                generation_runtime: GenerationRuntimeCapabilities::default(),
            },
            context: PlannedContextInspection::Text {
                prompt_token_count: 3,
                pre_tokenized: false,
            },
            rendered_segments: vec![RenderedSegmentInspection {
                origin: RenderedSegmentOrigin::RawPrompt,
                token_start: Some(0),
                token_end: Some(3),
            }],
            render: RenderPlanInspection {
                renderer_id: profile.render.renderer_id.clone(),
                chat_template_override_fingerprint: None,
            },
            generation: GenerationPlanInspection {
                constraint: GenerationConstraint::UndOnly,
                num_outputs: 1,
                candidate_ids: vec![CandidateId::PRIMARY],
                max_tokens: Some(8),
                temperature: Some(0.7),
                top_p: None,
                top_k: None,
                seed: None,
                min_tokens: 0,
                min_p: 0.0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                repetition_penalty: 1.0,
                ignore_eos: false,
                logit_bias_count: 0,
                allowed_token_count: 0,
                bad_word_count: 0,
                stop_token_count: 0,
                stop_string_count: 0,
                structured_output: None,
                compiled_choice_token_counts: Vec::new(),
                grammar_state_required: false,
                image: None,
            },
            dialect: None,
            cache: CompiledCachePolicy {
                namespace: None,
                key_fingerprint: "cache".to_string(),
                read_enabled: true,
                write_enabled: true,
                replayable: false,
                encoder_pin_count: 0,
            },
            adapter: AdapterSelection::Base,
            scheduling: SchedulingPolicy::default(),
            resources: ResourceBounds {
                max_context_tokens: None,
                prompt_tokens: 3,
                expected_kv_tokens: 11,
                image_count: 0,
                image_latent_units: 0,
                scratch_units: 0,
                host_scratch_tokens: 0,
                encoder_cache_pins: 0,
                grammar_states: 0,
                adapter_slots: 0,
                replayable: false,
                required_features: vec!["und".to_string()],
            },
            output: OutputProcessingPlan {
                visible_text: true,
                reasoning: false,
                tools: false,
                logprobs: false,
                token_ids: false,
                skip_special_tokens: true,
                include_stop_string: false,
                reasoning_parser: "none".to_string(),
                tool_parser: "none".to_string(),
                hidden_text_visible: false,
            },
        };

        let rendered = serde_json::to_string(&inspection).unwrap();
        assert!(!rendered.contains("secret prompt"));
        assert!(rendered.contains("prompt_token_count"));
    }
}
