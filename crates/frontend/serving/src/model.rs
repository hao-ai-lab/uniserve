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
    GenerationRuntimeCapabilities, ImageParams, RequestId, UndVisibility,
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
use crate::sampling::{SamplingDefaults, SamplingFallbacks, lower_sampling};
use crate::text::{TextDecodeOptions, resolve_max_tokens};
use crate::{CacheAccounting, ResourceAccounting, Result, ServeError, cache_isolation_key};

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
    sampling: SamplingDefaults,
    capabilities: GenerationRuntimeCapabilities,
}

/// SenseNova omni description: image input, text output, image output, and
/// repeated interleave through description-owned framing/ingest/output filter.
pub struct SenseNovaDesc {
    identity: ModelIdentity,
    tokenizer: DynTokenizer,
    renderer: HfChatRenderer,
    preprocessing: SenseNovaProfile,
    capabilities: GenerationRuntimeCapabilities,
    sampling: SamplingDefaults,
}

/// BAGEL omni description: image input, text output, and image output.
pub struct BagelDesc {
    identity: ModelIdentity,
    tokenizer: DynTokenizer,
    renderer: HfChatRenderer,
    preprocessing: BagelProfile,
    capabilities: GenerationRuntimeCapabilities,
    sampling: SamplingDefaults,
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
    ) -> Result<Self> {
        match profile {
            ModelProfile::Qwen3(profile) => {
                let sampling = sampling_defaults(
                    &profile.common,
                    max_model_tokens,
                    SamplingFallbacks {
                        temperature: 1.0,
                        top_p: 1.0,
                        top_k: 0,
                    },
                );
                Ok(Self::Qwen3(Qwen3Desc {
                    identity: profile.common.identity,
                    tokenizer,
                    renderer,
                    sampling,
                    capabilities,
                }))
            }
            ModelProfile::SenseNova(profile) => {
                let sampling = sampling_defaults(
                    &profile.common,
                    max_model_tokens,
                    SamplingFallbacks {
                        temperature: 0.0,
                        top_p: 1.0,
                        top_k: 0,
                    },
                );
                Ok(Self::SenseNova(SenseNovaDesc {
                    identity: profile.common.identity,
                    tokenizer,
                    renderer,
                    preprocessing: profile.preprocessing,
                    capabilities,
                    sampling,
                }))
            }
            ModelProfile::Bagel(profile) => {
                let sampling = sampling_defaults(
                    &profile.common,
                    max_model_tokens,
                    SamplingFallbacks {
                        temperature: 0.0,
                        top_p: 1.0,
                        top_k: 0,
                    },
                );
                Ok(Self::Bagel(BagelDesc {
                    identity: profile.common.identity,
                    tokenizer,
                    renderer,
                    preprocessing: profile.preprocessing,
                    capabilities,
                    sampling,
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
                &d.sampling,
                self.event_identity(),
                request,
            ),
            Self::Bagel(d) => crate::omni::tokenize_bagel(
                &d.preprocessing,
                std::sync::Arc::clone(&d.tokenizer),
                &d.renderer,
                &d.capabilities,
                &d.sampling,
                self.event_identity(),
                request,
            ),
        }
    }
}

fn omni_capability_needs(
    policy: &GenerationPolicyDescriptor,
    image_ingest: &uniserve_core::ImageIngestRecipe,
    request: &GenerateReqInput,
) -> GenerationCapabilityNeeds {
    let has_input_image = request.has_input_image();
    let constraint = if request.modalities.output_image && !request.modalities.output_text {
        GenerationConstraint::GenOnly
    } else if has_input_image && !request.modalities.output_image {
        GenerationConstraint::UndOnly
    } else {
        GenerationConstraint::Default
    };
    let behavior = GenerationBehaviorDescriptor::resolve(constraint, policy);
    let context_steps = if has_input_image {
        image_ingest.steps.clone()
    } else {
        Vec::new()
    };
    behavior.capability_needs(policy, context_steps)
}

fn sampling_defaults(
    profile: &CommonModelProfile,
    max_model_tokens: u32,
    fallbacks: SamplingFallbacks,
) -> SamplingDefaults {
    let primary = profile.stop_tokens.primary_eos_token_id;
    let mut extra: BTreeSet<u32> = profile.stop_tokens.eos_token_ids.clone();
    if let Some(primary) = primary {
        extra.remove(&primary);
    }
    SamplingDefaults {
        additional_eos_token_ids: extra,
        temperature: profile.generation_defaults.temperature,
        top_p: profile.generation_defaults.top_p,
        top_k: profile.generation_defaults.top_k,
        min_p: profile.generation_defaults.min_p,
        repetition_penalty: profile.generation_defaults.repetition_penalty,
        max_tokens: profile
            .context_limits
            .max_output_tokens
            .or(profile.generation_defaults.max_output_tokens),
        max_model_tokens,
        fallbacks,
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
                    OutputProcessorPolicy::Qwen3(Box::new(chat_request)),
                    skip,
                )
            }
        };

        let prompt_len = prompt_token_ids.len() as u32;
        let lowered = lower_sampling(self.tokenizer.as_ref(), &request, &self.sampling)?;
        let max_tokens = resolve_max_tokens(
            request.sampling.max_tokens,
            self.sampling.max_tokens,
            Some(self.sampling.max_model_tokens),
            prompt_len,
        )
        .map_err(|error| error.to_string())?;
        if request.sampling.min_tokens.unwrap_or(0) > max_tokens {
            return Err(format!(
                "min_tokens ({}) exceeds max_tokens ({max_tokens})",
                request.sampling.min_tokens.unwrap_or(0)
            ));
        }

        let constraint = GenerationConstraint::UndOnly;
        let mut policy = GenerationPolicyDescriptor::default();
        policy.termination.emit_stop_token = request.decode.include_stop_string_in_output;
        let isolation_key = cache_isolation_key(
            request.cache.namespace.as_deref(),
            request.cache.salt.as_deref(),
        );
        let cache = GenerationCachePolicyDescriptor {
            read: !request.cache.bypass_read && !lowered.params.prompt_logprobs_requested(),
            write: !request.cache.no_store,
            isolation_key,
        };
        let prompt_logprobs_requested = lowered.params.prompt_logprobs_requested();
        let generated_logprobs_requested = lowered.params.generated_logprobs_requested();
        let max_und_tokens = max_tokens as usize;
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
            sampling: lowered.params,
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
            scratch_units: 0,
            host_scratch_tokens: 0,
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
}

fn stable_hash(value: &str) -> u64 {
    value
        .as_bytes()
        .iter()
        .fold(0xcbf29ce484222325, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        })
}
