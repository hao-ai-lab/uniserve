//! Closed, load-bound model resolution and the sole model-owned `tokenize`
//! arrow.
//!
//! [`ResolvedModel`] is the closed value the server resolves at load time. It
//! owns the only transition from [`GenerateReqInput`] to
//! [`TokenizedGenerateReqInput`] via the inherent [`ResolvedModel::tokenize`]
//! method. There is no name registry, family router, plugin factory, or
//! trait-object tower in front of it.

use std::collections::BTreeSet;

use uniserve_core::{
    ContextSegment as CoreContextSegment, GenerationBehaviorDescriptor,
    GenerationCachePolicyDescriptor, GenerationCapabilityNeeds, GenerationConstraint,
    GenerationPolicyDescriptor, GenerationRequest, GenerationResourceBounds,
    GenerationRuntimeCapabilities, ImageParams, RequestId, SamplingParams as EngineSamplingParams,
    UndVisibility,
};
use uniserve_model_profile::omni::bagel::BagelProfile;
use uniserve_model_profile::omni::sensenova::SenseNovaProfile;
use uniserve_model_profile::tokenizer::DynTokenizer;
use uniserve_model_profile::{CommonModelProfile, ModelIdentity, ModelProfile};

use crate::chat::{ChatRequest, HfChatRenderer, Qwen3ChatOutputProcessor};
use crate::input::{
    GenerateReqInput, ModelEventIdentity, OutputContract, OutputProcessorPolicy, PromptInput,
    SubmissionMetadata, TokenizedGenerateReqInput,
};
use crate::text::{SamplingHints, TextDecodeOptions, resolve_max_tokens};
use crate::{CacheAccounting, ResourceAccounting, Result, ServeError, cache_isolation_key};

/// Public endpoint admitted by one resolved model description.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ServedEndpoint {
    ChatCompletions,
    ImageGenerations,
}

/// Public input or output modality admitted by one resolved description.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ServedModality {
    Text,
    Image,
}

/// Public behavior whose semantics are owned by the resolved description and
/// the shared response path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ServedFeature {
    Streaming,
    Usage,
    Logprobs,
    Reasoning,
    ToolCalling,
    RepeatedInterleave,
}

/// Sampling control admitted by every configured sampler route.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ServedSamplingControl {
    Greedy,
    Temperature,
    TopK,
    TopP,
    MinP,
    RepetitionPenalty,
    FrequencyPenalty,
    PresencePenalty,
    LogitBias,
    AllowedTokenIds,
    BadWords,
    MinTokens,
    Logprobs,
    StopTokenIds,
    Eos,
    StopStrings,
}

impl ServedSamplingControl {
    pub const ALL: [Self; 16] = [
        Self::Greedy,
        Self::Temperature,
        Self::TopK,
        Self::TopP,
        Self::MinP,
        Self::RepetitionPenalty,
        Self::FrequencyPenalty,
        Self::PresencePenalty,
        Self::LogitBias,
        Self::AllowedTokenIds,
        Self::BadWords,
        Self::MinTokens,
        Self::Logprobs,
        Self::StopTokenIds,
        Self::Eos,
        Self::StopStrings,
    ];
}

/// Exact public route declaration for one load-bound model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServedModelCapabilities {
    pub endpoints: Vec<ServedEndpoint>,
    pub input_modalities: Vec<ServedModality>,
    pub output_modalities: Vec<ServedModality>,
    pub features: Vec<ServedFeature>,
    pub sampling_controls: Vec<ServedSamplingControl>,
}

/// The closed load-bound model owner.
pub enum ResolvedModel {
    Qwen3(Qwen3Desc),
    SenseNova(SenseNovaDesc),
    Bagel(BagelDesc),
}

/// Text chat description: HF tokenization + chat template + fixed Qwen3 parser
/// policy.
pub struct Qwen3Desc {
    identity: ModelIdentity,
    tokenizer: DynTokenizer,
    renderer: HfChatRenderer,
    hints: SamplingHints,
    capabilities: GenerationRuntimeCapabilities,
    logprobs_supported: bool,
    parse_reasoning: bool,
}

/// SenseNova omni description: image input, text output, image output, and
/// repeated interleave through description-owned framing/ingest/output filter.
pub struct SenseNovaDesc {
    identity: ModelIdentity,
    tokenizer: DynTokenizer,
    renderer: HfChatRenderer,
    preprocessing: SenseNovaProfile,
    capabilities: GenerationRuntimeCapabilities,
    default_max_output_tokens: Option<u32>,
    max_model_tokens: u32,
}

/// BAGEL omni description: image input, text output, and image output.
pub struct BagelDesc {
    identity: ModelIdentity,
    tokenizer: DynTokenizer,
    renderer: HfChatRenderer,
    preprocessing: BagelProfile,
    capabilities: GenerationRuntimeCapabilities,
    default_max_output_tokens: Option<u32>,
    max_model_tokens: u32,
}

impl ResolvedModel {
    /// Fallible, exhaustive resolution over the closed configured set.
    ///
    /// The typed description selects the variant; required description-owned
    /// assets are checked before construction.
    pub fn resolve(
        profile: ModelProfile,
        tokenizer: DynTokenizer,
        renderer: HfChatRenderer,
        capabilities: GenerationRuntimeCapabilities,
        max_model_tokens: u32,
        parse_reasoning: bool,
    ) -> Result<Self> {
        match profile {
            ModelProfile::Qwen3(profile) => {
                validate_runtime_capabilities(
                    &profile.common.identity,
                    &capabilities,
                    GenerationCapabilityNeeds {
                        understanding: true,
                        ..GenerationCapabilityNeeds::default()
                    },
                )?;
                let hints = sampling_hints(&profile.common, max_model_tokens);
                Ok(Self::Qwen3(Qwen3Desc {
                    identity: profile.common.identity,
                    tokenizer,
                    renderer,
                    hints,
                    capabilities,
                    logprobs_supported: true,
                    parse_reasoning,
                }))
            }
            ModelProfile::SenseNova(profile) => {
                validate_runtime_capabilities(
                    &profile.common.identity,
                    &capabilities,
                    configured_omni_needs(
                        &profile.preprocessing.generation_policy,
                        &profile.preprocessing.image_ingest,
                    ),
                )?;
                let default_max_output_tokens = profile
                    .common
                    .context_limits
                    .max_output_tokens
                    .or(profile.common.generation_defaults.max_output_tokens);
                Ok(Self::SenseNova(SenseNovaDesc {
                    identity: profile.common.identity,
                    tokenizer,
                    renderer,
                    preprocessing: profile.preprocessing,
                    capabilities,
                    default_max_output_tokens,
                    max_model_tokens,
                }))
            }
            ModelProfile::Bagel(profile) => {
                validate_runtime_capabilities(
                    &profile.common.identity,
                    &capabilities,
                    configured_omni_needs(
                        &profile.preprocessing.generation_policy,
                        &profile.preprocessing.image_ingest,
                    ),
                )?;
                let default_max_output_tokens = profile
                    .common
                    .context_limits
                    .max_output_tokens
                    .or(profile.common.generation_defaults.max_output_tokens);
                Ok(Self::Bagel(BagelDesc {
                    identity: profile.common.identity,
                    tokenizer,
                    renderer,
                    preprocessing: profile.preprocessing,
                    capabilities,
                    default_max_output_tokens,
                    max_model_tokens,
                }))
            }
        }
    }

    /// Served-model identity used by `/v1/models` and event provenance.
    pub fn served_identity(&self) -> &ModelIdentity {
        match self {
            Self::Qwen3(d) => &d.identity,
            Self::SenseNova(d) => &d.identity,
            Self::Bagel(d) => &d.identity,
        }
    }

    /// Friendly served-model name.
    pub fn served_model_name(&self) -> &str {
        &self.served_identity().model_id
    }

    /// Event identity stamped onto `Accepted`.
    pub fn event_identity(&self) -> ModelEventIdentity {
        let identity = self.served_identity();
        ModelEventIdentity {
            profile_id: identity.profile_id.clone(),
            description_id: identity.description_id.clone(),
        }
    }

    /// Tokenizer bound into the resolved description.
    pub fn tokenizer(&self) -> DynTokenizer {
        match self {
            Self::Qwen3(d) => std::sync::Arc::clone(&d.tokenizer),
            Self::SenseNova(d) => std::sync::Arc::clone(&d.tokenizer),
            Self::Bagel(d) => std::sync::Arc::clone(&d.tokenizer),
        }
    }

    /// True when the description supports image output.
    pub fn supports_image_output(&self) -> bool {
        !matches!(self, Self::Qwen3(_))
    }

    /// True when the description supports image input.
    pub fn supports_image_input(&self) -> bool {
        !matches!(self, Self::Qwen3(_))
    }

    /// Exact route capabilities exposed by model discovery and enforced by
    /// request admission.
    pub fn served_capabilities(&self) -> ServedModelCapabilities {
        let mut endpoints = vec![ServedEndpoint::ChatCompletions];
        let mut input_modalities = vec![ServedModality::Text];
        let mut output_modalities = vec![ServedModality::Text];
        let mut features = vec![
            ServedFeature::Streaming,
            ServedFeature::Usage,
            ServedFeature::Logprobs,
        ];
        match self {
            Self::Qwen3(_) => {
                features.push(ServedFeature::Reasoning);
                features.push(ServedFeature::ToolCalling);
            }
            Self::SenseNova(_) => {
                endpoints.push(ServedEndpoint::ImageGenerations);
                input_modalities.push(ServedModality::Image);
                output_modalities.push(ServedModality::Image);
                features.push(ServedFeature::Reasoning);
                features.push(ServedFeature::RepeatedInterleave);
            }
            Self::Bagel(_) => {
                endpoints.push(ServedEndpoint::ImageGenerations);
                input_modalities.push(ServedModality::Image);
                output_modalities.push(ServedModality::Image);
            }
        }
        ServedModelCapabilities {
            endpoints,
            input_modalities,
            output_modalities,
            features,
            sampling_controls: ServedSamplingControl::ALL.to_vec(),
        }
    }

    /// Deterministic capability admission or rejection from the resolved route.
    pub fn validate_request(&self, request: &GenerateReqInput) -> Result<()> {
        let reject = |capability: &'static str| ServeError::UnsupportedCapability {
            request_id: request.request_id.clone(),
            capability,
        };
        let has_input_image = request.has_input_image();
        if has_input_image && !self.supports_image_input() {
            return Err(reject("image_input"));
        }
        if request.modalities.output_image && !self.supports_image_output() {
            return Err(reject("image_output"));
        }
        if !request.modalities.output_text && !request.modalities.output_image {
            return Err(reject("no_output_modality"));
        }
        let declared = self.served_capabilities();
        if request.uses_tools() && !declared.features.contains(&ServedFeature::ToolCalling) {
            return Err(reject("tool_calling"));
        }
        if request.requests_reasoning() && !declared.features.contains(&ServedFeature::Reasoning) {
            return Err(reject("reasoning"));
        }
        let (capabilities, needs) = match self {
            Self::Qwen3(d) => (
                &d.capabilities,
                GenerationCapabilityNeeds {
                    understanding: true,
                    ..Default::default()
                },
            ),
            Self::SenseNova(d) => (
                &d.capabilities,
                omni_capability_needs(
                    &d.preprocessing.generation_policy,
                    &d.preprocessing.image_ingest,
                    request,
                ),
            ),
            Self::Bagel(d) => (
                &d.capabilities,
                omni_capability_needs(
                    &d.preprocessing.generation_policy,
                    &d.preprocessing.image_ingest,
                    request,
                ),
            ),
        };
        if let Err(capability) = capabilities.covers(&needs) {
            return Err(reject(capability));
        }
        Ok(())
    }

    /// The sole model-owned arrow from [`GenerateReqInput`] to
    /// [`TokenizedGenerateReqInput`]. Inherent (not `From`/`TryFrom`/`Into`).
    pub fn tokenize(&self, request: GenerateReqInput) -> Result<TokenizedGenerateReqInput> {
        match self {
            Self::Qwen3(d) => d.tokenize(request),
            Self::SenseNova(d) => crate::omni::tokenize_sensenova(
                &d.preprocessing,
                std::sync::Arc::clone(&d.tokenizer),
                &d.renderer,
                &d.capabilities,
                d.default_max_output_tokens,
                d.max_model_tokens,
                self.event_identity(),
                request,
            ),
            Self::Bagel(d) => crate::omni::tokenize_bagel(
                &d.preprocessing,
                std::sync::Arc::clone(&d.tokenizer),
                &d.renderer,
                &d.capabilities,
                d.default_max_output_tokens,
                d.max_model_tokens,
                self.event_identity(),
                request,
            ),
        }
    }
}

fn configured_omni_needs(
    policy: &GenerationPolicyDescriptor,
    image_ingest: &uniserve_core::ImageIngestRecipe,
) -> GenerationCapabilityNeeds {
    GenerationBehaviorDescriptor::resolve(GenerationConstraint::Default, policy)
        .capability_needs(policy, image_ingest.steps.iter().copied())
}

fn validate_runtime_capabilities(
    identity: &ModelIdentity,
    capabilities: &GenerationRuntimeCapabilities,
    needs: GenerationCapabilityNeeds,
) -> Result<()> {
    capabilities.covers(&needs).map_err(|capability| {
        ServeError::ModelResolution(format!(
            "configured model description `{}` requires worker capability `{capability}`",
            identity.description_id
        ))
    })
}

fn omni_capability_needs(
    policy: &GenerationPolicyDescriptor,
    image_ingest: &uniserve_core::ImageIngestRecipe,
    request: &GenerateReqInput,
) -> GenerationCapabilityNeeds {
    let has_input_image = request.has_input_image();
    let constraint = crate::omni::generation_constraint(request);
    let behavior = GenerationBehaviorDescriptor::resolve(constraint, policy);
    let context_steps = if has_input_image {
        image_ingest.steps.clone()
    } else {
        Vec::new()
    };
    behavior.capability_needs(policy, context_steps)
}

fn sampling_hints(profile: &CommonModelProfile, max_model_tokens: u32) -> SamplingHints {
    let primary = profile.stop_tokens.primary_eos_token_id;
    let mut extra: BTreeSet<u32> = profile.stop_tokens.eos_token_ids.clone();
    if let Some(primary) = primary {
        extra.remove(&primary);
    }
    SamplingHints {
        primary_eos_token_id: primary,
        extra_eos_token_ids: extra,
        default_temperature: profile.generation_defaults.temperature,
        default_top_p: profile.generation_defaults.top_p,
        default_top_k: profile.generation_defaults.top_k,
        default_min_p: profile.generation_defaults.min_p,
        default_repetition_penalty: profile.generation_defaults.repetition_penalty,
        default_max_tokens: profile.generation_defaults.max_output_tokens,
        max_model_len: Some(max_model_tokens),
    }
}

impl Qwen3Desc {
    fn tokenize(&self, request: GenerateReqInput) -> Result<TokenizedGenerateReqInput> {
        let request_id = request.request_id.clone();
        self.tokenize_inner(request)
            .map_err(|message| ServeError::Tokenize {
                request_id,
                message,
            })
    }

    fn tokenize_inner(
        &self,
        request: GenerateReqInput,
    ) -> std::result::Result<TokenizedGenerateReqInput, String> {
        let (prompt_token_ids, output_processor, skip_special_tokens) = match &request.prompt {
            PromptInput::Text(text) => {
                let ids = self
                    .tokenizer
                    .encode(text, false)
                    .map_err(|error| error.to_string())?;
                (
                    ids,
                    OutputProcessorPolicy::None,
                    request.decode.skip_special_tokens,
                )
            }
            PromptInput::Chat {
                messages,
                tools,
                tool_choice,
                reasoning_effort,
            } => {
                let mut chat_request = ChatRequest {
                    messages: messages.clone(),
                    chat_options: crate::chat::ChatOptions {
                        generation_prompt_mode:
                            crate::chat::GenerationPromptMode::StartNewAssistant,
                        reasoning_effort: *reasoning_effort,
                    },
                    tools: tools.clone(),
                    tool_choice: *tool_choice,
                    decode_options: TextDecodeOptions {
                        skip_special_tokens: request.decode.skip_special_tokens,
                        include_stop_str_in_output: request.decode.include_stop_string_in_output,
                        stop_strings: (!request.stop.stop_strings.is_empty())
                            .then(|| request.stop.stop_strings.clone()),
                        min_tokens: request.sampling.min_tokens.unwrap_or(0),
                    },
                };
                chat_request.validate().map_err(|error| error.to_string())?;
                // Build the processor once to apply parser-driven request
                // adjustments (e.g. disabling special-token skipping).
                let _processor = Qwen3ChatOutputProcessor::new(
                    &mut chat_request,
                    std::sync::Arc::clone(&self.tokenizer),
                    self.parse_reasoning,
                )
                .map_err(|error| error.to_string())?;
                let rendered_text = self
                    .renderer
                    .render(&chat_request)
                    .map_err(|error| error.to_string())?;
                let ids = self
                    .tokenizer
                    .encode(&rendered_text, false)
                    .map_err(|error| error.to_string())?;
                let skip = chat_request.decode_options.skip_special_tokens;
                (
                    ids,
                    OutputProcessorPolicy::Qwen3 {
                        request: Box::new(chat_request),
                        parse_reasoning: self.parse_reasoning,
                    },
                    skip,
                )
            }
        };

        let prompt_len = prompt_token_ids.len() as u32;
        let lowered = self.lower_sampling(&request, prompt_len)?;

        let constraint = GenerationConstraint::UndOnly;
        let mut policy = GenerationPolicyDescriptor::default();
        policy.termination.emit_stop_token = request.decode.include_stop_string_in_output;
        let isolation_key = cache_isolation_key(
            request.cache.namespace.as_deref(),
            request.cache.salt.as_deref(),
        );
        let cache = GenerationCachePolicyDescriptor {
            read: !request.cache.bypass_read && !lowered.sampling.prompt_logprobs_requested(),
            write: !request.cache.no_store,
            isolation_key,
        };
        let prompt_logprobs_requested = lowered.sampling.prompt_logprobs_requested();
        let generated_logprobs_requested = lowered.sampling.generated_logprobs_requested();
        let max_und_tokens = lowered.max_tokens as usize;
        let resources = GenerationResourceBounds {
            context_tokens: prompt_len as usize,
            max_kv_tokens: (prompt_len as usize).saturating_add(max_und_tokens),
            ..GenerationResourceBounds::default()
        };
        let generation = GenerationRequest {
            request_id: RequestId(stable_hash(request.request_id.as_ref())),
            context: vec![CoreContextSegment::UndTokens {
                token_ids: prompt_token_ids.clone(),
                visibility: UndVisibility::Internal,
            }],
            negative_context: Vec::new(),
            constraint,
            behavior: GenerationBehaviorDescriptor::resolve(constraint, &policy),
            sampling: lowered.sampling,
            image: ImageParams::default(),
            max_und_tokens,
            stop_strings: request.stop.stop_strings.clone(),
            stop_token_ids: lowered.stop_token_ids,
            priority: request.scheduling.priority,
            cache: cache.clone(),
            policy,
            resources: resources.clone(),
        };
        generation.validate().map_err(|error| error.to_string())?;

        let cache_accounting = CacheAccounting {
            read_enabled: cache.read,
            write_enabled: cache.write,
            encoder_pin_count: 0,
        };
        let resource_accounting = ResourceAccounting {
            expected_kv_tokens: resources.max_kv_tokens as u64,
            image_latent_units: 0,
            encoder_cache_pins: 0,
            replayable: true,
        };

        let decode = TextDecodeOptions {
            skip_special_tokens,
            include_stop_str_in_output: request.decode.include_stop_string_in_output,
            stop_strings: (!request.stop.stop_strings.is_empty())
                .then(|| request.stop.stop_strings.clone()),
            min_tokens: request.sampling.min_tokens.unwrap_or(0),
        };

        Ok(TokenizedGenerateReqInput {
            request_id: request.request_id,
            request: generation,
            prompt_token_ids,
            decode,
            emit_token_ids: matches!(
                request.output,
                OutputContract::Tokens | OutputContract::Logprobs
            ),
            prompt_logprobs_requested,
            generated_logprobs_requested,
            skip_special_tokens,
            output_processor,
            submission: SubmissionMetadata {
                trace_headers: (!request.scheduling.trace_context.is_empty())
                    .then(|| request.scheduling.trace_context.clone()),
            },
            identity: ModelEventIdentity {
                profile_id: self.identity.profile_id.clone(),
                description_id: self.identity.description_id.clone(),
            },
            cache: cache_accounting,
            resources: resource_accounting,
        })
    }

    fn lower_sampling(
        &self,
        request: &GenerateReqInput,
        prompt_len: u32,
    ) -> std::result::Result<LoweredSampling, String> {
        let sampling = &request.sampling;
        let stop = &request.stop;
        let hints = &self.hints;

        let temperature = sampling
            .temperature
            .or(hints.default_temperature)
            .unwrap_or(1.0);
        let top_p = sampling.top_p.or(hints.default_top_p).unwrap_or(1.0);
        let top_k = sampling.top_k.or(hints.default_top_k).unwrap_or(0);
        let min_p = sampling.min_p.or(hints.default_min_p).unwrap_or(0.0);
        let repetition_penalty = sampling
            .repetition_penalty
            .or(hints.default_repetition_penalty)
            .unwrap_or(1.0);
        let max_tokens = resolve_max_tokens(
            sampling.max_tokens,
            hints.default_max_tokens,
            hints.max_model_len,
            prompt_len,
        )
        .map_err(|error| error.to_string())?;
        let min_tokens = sampling.min_tokens.unwrap_or(0);
        let frequency_penalty = sampling.frequency_penalty.unwrap_or(0.0);
        let presence_penalty = sampling.presence_penalty.unwrap_or(0.0);

        let mut stop_token_ids = stop.stop_token_ids.clone();
        if !sampling.ignore_eos {
            for token_id in &hints.extra_eos_token_ids {
                if !stop_token_ids.contains(token_id) {
                    stop_token_ids.push(*token_id);
                }
            }
        }

        for (field, value) in [
            ("logprobs", stop.logprobs),
            ("prompt_logprobs", stop.prompt_logprobs),
        ] {
            if let Some(value) = value
                && value < -1
            {
                return Err(format!("{field} must be non-negative or -1, got {value}"));
            }
        }
        if min_tokens > max_tokens {
            return Err(format!(
                "min_tokens ({min_tokens}) exceeds max_tokens ({max_tokens})"
            ));
        }

        let bad_words_ids = tokenize_bad_words(&stop.bad_words, self.tokenizer.as_ref())?;

        let mut canonical_logit_bias: Vec<(u32, f32)> = stop
            .logit_bias
            .as_ref()
            .map(|biases| biases.iter().map(|(&token, &bias)| (token, bias)).collect())
            .unwrap_or_default();
        canonical_logit_bias.sort_by_key(|(token, _)| *token);

        let core = EngineSamplingParams {
            temperature,
            top_k,
            top_p,
            ignore_eos: sampling.ignore_eos,
            seed: sampling.seed.map(|value| value as u64),
            min_p,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            logit_bias: canonical_logit_bias,
            min_tokens: min_tokens as usize,
            return_logprobs: stop.logprobs.is_some() || stop.logprob_token_ids.is_some(),
            n_logprobs: match stop.logprobs {
                Some(-1) => u32::MAX,
                Some(value) => value as u32,
                None => 0,
            },
            return_prompt_logprobs: stop.prompt_logprobs.is_some(),
            n_prompt_logprobs: match stop.prompt_logprobs {
                Some(-1) => u32::MAX,
                Some(value) => value as u32,
                None => 0,
            },
            logprob_token_ids: stop.logprob_token_ids.clone().unwrap_or_default(),
            bad_words_ids: bad_words_ids.unwrap_or_default(),
            allowed_token_ids: stop.allowed_token_ids.clone(),
            typical_p: 1.0,
            forced_token_ids: Vec::new(),
        };
        core.validate().map_err(|error| error.to_string())?;

        // Logprob feature gate.
        if (stop.logprobs.is_some() || stop.prompt_logprobs.is_some()) && !self.logprobs_supported {
            return Err("this model does not support logprobs".to_string());
        }

        Ok(LoweredSampling {
            sampling: core,
            max_tokens,
            stop_token_ids,
        })
    }
}

struct LoweredSampling {
    sampling: EngineSamplingParams,
    max_tokens: u32,
    stop_token_ids: Vec<u32>,
}

/// Convert bad-word strings into token-ID sequences, encoding each word both
/// with and without a leading space (prefix-space convention) and deduping.
fn tokenize_bad_words(
    bad_words: &[String],
    tokenizer: &uniserve_model_profile::tokenizer::HuggingFaceTokenizer,
) -> std::result::Result<Option<Vec<Vec<u32>>>, String> {
    if bad_words.is_empty() {
        return Ok(None);
    }
    let mut all_token_ids = Vec::new();
    for bad_word in bad_words {
        let without_space = tokenizer
            .encode(bad_word, false)
            .map_err(|e| e.to_string())?;
        let with_space = tokenizer
            .encode(&format!(" {}", bad_word.trim_start()), false)
            .map_err(|e| e.to_string())?;
        let keep_with_space = !with_space.is_empty()
            && (without_space.is_empty()
                || (with_space[0] != without_space[0] && with_space.len() == without_space.len()));
        if !without_space.is_empty() {
            all_token_ids.push(without_space);
        }
        if keep_with_space {
            all_token_ids.push(with_space);
        }
    }
    Ok((!all_token_ids.is_empty()).then_some(all_token_ids))
}

fn stable_hash(value: &str) -> u64 {
    value
        .as_bytes()
        .iter()
        .fold(0xcbf29ce484222325, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        })
}
