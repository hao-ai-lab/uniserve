//! Multimodal prompt preprocessing for SenseNova and Bagel profiles.

mod output;

use std::io::Cursor;
use thiserror::Error;

use crate::profile::omni::bagel::{BagelProfile, CONTEXT_SYSTEM_PROMPT};
use crate::profile::omni::resolution::{ResolutionPolicy, resolve_resolution};
use crate::profile::omni::sensenova::SenseNovaProfile;
use base64::Engine as _;
use uniserve_core::{
    ContextSegment as CoreContextSegment, GenerationBehaviorDescriptor,
    GenerationCachePolicyDescriptor, GenerationConstraint, GenerationLimits,
    GenerationPolicyDescriptor, GenerationRequest, GenerationResourceBounds, ImageIngestRecipe,
    ImageParams, ImageSegment as CoreImageSegment, RequestId, SamplingParams, SegmentPlacement,
    UndVisibility,
};

use crate::serving::chat::{
    ChatContent, ChatContentPart, ChatMessage, ChatRequest, GenerationPromptMode, HfChatRenderer,
};
use crate::serving::input::{
    GenerateReqInput, ModalitySelection, ModelEventIdentity, OutputProcessorPolicy, PromptInput,
    TokenizedGenerateReqInput,
};
use crate::serving::text::TextDecodeOptions;
use crate::serving::text::tokenizer::DynTokenizer;
use crate::serving::{
    CacheAccounting, ResourceAccounting, Result, ServeError, cache_isolation_key,
};

pub(crate) use output::{SenseNovaOutputProcessor, SenseNovaTextDelta};

type OmniResult<T> = std::result::Result<T, OmniError>;

#[derive(Debug, Error)]
/// Failure while preprocessing a multimodal generation request.
pub enum OmniError {
    #[error(transparent)]
    Tokenizer(#[from] crate::profile::tokenizer::TokenizerError),
    #[error(transparent)]
    Chat(#[from] crate::serving::chat::Error),
    #[error(transparent)]
    Resolution(#[from] crate::profile::omni::resolution::ResolutionError),
    #[error(transparent)]
    Generation(#[from] uniserve_core::GenerationRequestError),
    #[error(transparent)]
    Sampling(#[from] uniserve_core::SamplingParamsError),
    #[error(transparent)]
    Assets(#[from] crate::profile::assets::Error),
    #[error("{0}")]
    Invalid(String),
}

impl From<String> for OmniError {
    /// Converts the source value into this type.
    fn from(message: String) -> Self {
        Self::Invalid(message)
    }
}

const DEFAULT_STEPS: u16 = 50;
const DEFAULT_TEMPERATURE: f32 = 0.0;
const DEFAULT_TOP_P: f32 = 1.0;
const DEFAULT_TOP_K: u32 = 0;

mod context_image_defaults {
    /// Default text classifier-free guidance scale.
    pub(super) const CFG_TEXT_SCALE: f32 = 4.0;
    /// Default image classifier-free guidance scale.
    pub(super) const CFG_IMG_SCALE: f32 = 2.0;
    /// Default lower bound for guidance renormalization.
    pub(super) const CFG_RENORM_MIN: f32 = 0.0;
    /// Default denoising interval for classifier-free guidance.
    pub(super) const CFG_INTERVAL: (f32, f32) = (0.0, 1.0);
    /// Default square output dimension in pixels.
    pub(super) const RESOLUTION: u32 = 512;
}

/// Profile and request values shared by multimodal tokenizers.
pub(super) struct RuntimeBinding<'a> {
    pub(super) tokenizer: DynTokenizer,
    pub(super) renderer: &'a HfChatRenderer,
    pub(super) limits: &'a GenerationLimits,
    pub(super) default_max_output_tokens: Option<u32>,
    pub(super) max_model_tokens: u32,
    pub(super) identity: ModelEventIdentity,
}

#[derive(Debug, Clone)]
struct LoweredInput {
    prompt_ids: Vec<u32>,
    negative_prompt_ids: Vec<u32>,
    sampling: SamplingParams,
    image: ImageParams,
    constraint: GenerationConstraint,
    images: Vec<RenderedImage>,
    stop_token_ids: Vec<u32>,
}

#[derive(Debug, Clone)]
struct PositionedImageInput {
    b64: String,
}

#[derive(Debug, Clone)]
struct RenderedImage {
    hash: u64,
    position: u32,
    b64: String,
}

/// Tokenizes and lowers a request using the SenseNova profile.
pub(super) fn tokenize_sensenova(
    profile: &SenseNovaProfile,
    binding: RuntimeBinding<'_>,
    request: GenerateReqInput,
) -> Result<TokenizedGenerateReqInput> {
    let request_id = request.request_id.clone();
    let result = (|| {
        let lowered = lower_sensenova(profile, &binding.tokenizer, binding.renderer, &request)?;
        let context = build_sensenova_context(profile, &lowered)?;
        let policy = profile
            .generation_policy_for_dimensions(lowered.image.width, lowered.image.height)
            .map_err(|error| error.to_string())?;
        finish_tokenized(
            &binding,
            &policy,
            request,
            lowered,
            context,
            OutputProcessorPolicy::SenseNova(profile.output_filter.clone()),
        )
    })();
    result.map_err(|source| ServeError::Tokenize {
        request_id,
        source: crate::serving::TokenizeError::Omni(source),
    })
}

/// Tokenizes and lowers a request using the Bagel profile.
pub(super) fn tokenize_bagel(
    profile: &BagelProfile,
    binding: RuntimeBinding<'_>,
    request: GenerateReqInput,
) -> Result<TokenizedGenerateReqInput> {
    let request_id = request.request_id.clone();
    let result = (|| {
        let lowered = lower_bagel(profile, &binding.tokenizer, binding.renderer, &request)?;
        let context = build_bagel_context(profile, &lowered)?;
        let policy = profile
            .generation_policy_for_dimensions(lowered.image.width, lowered.image.height)
            .map_err(|error| error.to_string())?;
        finish_tokenized(
            &binding,
            &policy,
            request,
            lowered,
            context,
            OutputProcessorPolicy::Bagel,
        )
    })();
    result.map_err(|source| ServeError::Tokenize {
        request_id,
        source: crate::serving::TokenizeError::Omni(source),
    })
}

/// Lowers a SenseNova request into token, image, sampling, and termination inputs.
fn lower_sensenova(
    profile: &SenseNovaProfile,
    tokenizer: &DynTokenizer,
    renderer: &HfChatRenderer,
    request: &GenerateReqInput,
) -> OmniResult<LoweredInput> {
    let constraint = generation_constraint(request);
    let (prompt_ids, images) = sensenova_prompt(profile, tokenizer, renderer, request, constraint)?;
    validate_prompt(&prompt_ids)?;
    let negative_prompt_ids = profile
        .render_negative_prompt_ids(tokenizer, request.negative_text.as_deref().unwrap_or(""))
        .map_err(|error| error.to_string())?;
    Ok(LoweredInput {
        prompt_ids,
        negative_prompt_ids,
        sampling: resolve_sampling(request)?,
        image: resolve_image_params(
            &profile.image_defaults,
            &profile.resolution_policy,
            request,
            constraint,
        )?,
        constraint,
        images,
        stop_token_ids: request.stop.stop_token_ids.clone(),
    })
}

/// Lowers a Bagel request into token, image, sampling, and termination inputs.
fn lower_bagel(
    profile: &BagelProfile,
    tokenizer: &DynTokenizer,
    renderer: &HfChatRenderer,
    request: &GenerateReqInput,
) -> OmniResult<LoweredInput> {
    let constraint = generation_constraint(request);
    let (prompt_ids, images, context_image_mode) =
        bagel_prompt(profile, tokenizer, renderer, request, constraint)?;
    validate_prompt(&prompt_ids)?;
    let negative_prompt_ids = if context_image_mode {
        prompt_ids.clone()
    } else {
        profile
            .render_negative_prompt_ids(tokenizer, request.negative_text.as_deref().unwrap_or(""))
            .map_err(|error| error.to_string())?
    };
    let image = if context_image_mode {
        bagel_context_image_params(profile, request)?
    } else {
        resolve_image_params(
            &profile.image_defaults,
            &profile.resolution_policy,
            request,
            constraint,
        )?
    };
    Ok(LoweredInput {
        prompt_ids,
        negative_prompt_ids,
        sampling: resolve_sampling(request)?,
        image,
        constraint,
        images,
        stop_token_ids: request.stop.stop_token_ids.clone(),
    })
}

/// Renders a SenseNova text or chat prompt and resolves positioned input images.
fn sensenova_prompt(
    profile: &SenseNovaProfile,
    tokenizer: &DynTokenizer,
    renderer: &HfChatRenderer,
    request: &GenerateReqInput,
    constraint: GenerationConstraint,
) -> OmniResult<(Vec<u32>, Vec<RenderedImage>)> {
    match &request.prompt {
        PromptInput::Text(prompt) => {
            let images = top_level_images(request);
            if images.is_empty() {
                let prompt_ids = profile
                    .render_prompt_ids(tokenizer, constraint, prompt, None, None)
                    .map_err(|error| error.to_string())?;
                Ok((prompt_ids, Vec::new()))
            } else {
                let placeholders = image_placeholders(request.request_id.as_ref(), images.len());
                let prompt = prompt_with_image_slots(prompt, &placeholders);
                let rendered = profile.render_prompt_text(constraint, &prompt, None, None);
                tokenize_sensenova_with_slots(tokenizer, &rendered, &placeholders, images, profile)
            }
        }
        PromptInput::Chat { .. } => {
            render_sensenova_chat(profile, tokenizer, renderer, request, constraint)
        }
    }
}

/// Renders a Bagel prompt and selects its context-image conditioning mode.
fn bagel_prompt(
    profile: &BagelProfile,
    tokenizer: &DynTokenizer,
    renderer: &HfChatRenderer,
    request: &GenerateReqInput,
    constraint: GenerationConstraint,
) -> OmniResult<(Vec<u32>, Vec<RenderedImage>, bool)> {
    match &request.prompt {
        PromptInput::Chat { .. } => {
            let (prompt_ids, images) = render_bagel_chat(tokenizer, renderer, request, constraint)?;
            Ok((prompt_ids, images, false))
        }
        PromptInput::Text(prompt)
            if constraint == GenerationConstraint::UndOnly && !request.images.is_empty() =>
        {
            let mut prompt_ids = profile
                .wrap_context_text(tokenizer, CONTEXT_SYSTEM_PROMPT)
                .map_err(|error| error.to_string())?;
            let image_position = prompt_ids.len() as u32;
            prompt_ids.extend(
                profile
                    .wrap_context_text(tokenizer, prompt)
                    .map_err(|error| error.to_string())?,
            );
            let images = top_level_images(request)
                .into_iter()
                .map(|image| rendered_image(&image, image_position))
                .collect();
            Ok((prompt_ids, images, true))
        }
        PromptInput::Text(prompt) => {
            let images = top_level_images(request);
            let prompt_ids = profile
                .render_prompt_ids(
                    tokenizer,
                    constraint,
                    !images.is_empty(),
                    prompt,
                    None,
                    None,
                )
                .map_err(|error| error.to_string())?;
            let position = prompt_ids.len() as u32;
            let images = images
                .into_iter()
                .map(|image| rendered_image(&image, position))
                .collect();
            Ok((prompt_ids, images, false))
        }
    }
}

/// Renders structured SenseNova chat while preserving one placeholder per input image.
fn render_sensenova_chat(
    profile: &SenseNovaProfile,
    tokenizer: &DynTokenizer,
    renderer: &HfChatRenderer,
    request: &GenerateReqInput,
    constraint: GenerationConstraint,
) -> OmniResult<(Vec<u32>, Vec<RenderedImage>)> {
    let PromptInput::Chat {
        messages,
        tools,
        tool_choice,
        reasoning_effort,
    } = &request.prompt
    else {
        unreachable!("caller matched chat prompt")
    };
    let mut messages = messages.clone();
    if !messages
        .iter()
        .any(|message| matches!(message, ChatMessage::System { .. }))
        && let Some(system) = SenseNovaProfile::default_system_prompt(constraint)
    {
        messages.insert(0, ChatMessage::system(system));
    }
    let mut generation_prompt_mode = GenerationPromptMode::StartNewAssistant;
    let assistant_prefix = SenseNovaProfile::assistant_prefix(constraint);
    if !assistant_prefix.is_empty() {
        messages.push(ChatMessage::assistant_text(assistant_prefix));
        generation_prompt_mode = GenerationPromptMode::ContinueFinalAssistant;
    }
    let (images, placeholders) = replace_chat_images(request.request_id.as_ref(), &mut messages)?;
    let rendered = renderer
        .render(&ChatRequest {
            messages,
            chat_options: crate::serving::chat::ChatOptions {
                generation_prompt_mode,
                reasoning_effort: *reasoning_effort,
            },
            tools: tools.clone(),
            tool_choice: *tool_choice,
            decode_options: TextDecodeOptions::default(),
        })
        .map_err(|error| error.to_string())?;
    tokenize_sensenova_with_slots(tokenizer, &rendered, &placeholders, images, profile)
}

/// Replaces SenseNova image slots and computes their positions in token space.
fn tokenize_sensenova_with_slots(
    tokenizer: &DynTokenizer,
    rendered: &str,
    placeholders: &[String],
    images: Vec<PositionedImageInput>,
    profile: &SenseNovaProfile,
) -> OmniResult<(Vec<u32>, Vec<RenderedImage>)> {
    let marker = format!(
        "{}{}",
        profile.controls.start_of_image_text, profile.controls.end_of_image_text
    );
    let (clean, byte_offsets) = replace_rendered_slots(rendered, placeholders, &marker)?;
    let prompt_ids = tokenizer
        .encode(&clean, false)
        .map_err(|error| format!("SenseNova prompt tokenization failed: {error}"))?;
    let images = images
        .into_iter()
        .zip(byte_offsets)
        .map(|(image, byte_offset)| -> OmniResult<RenderedImage> {
            let prefix = tokenizer
                .encode(&clean[..byte_offset], false)
                .map_err(|error| {
                    format!("SenseNova image placement tokenization failed: {error}")
                })?;
            let position = prefix
                .len()
                .checked_sub(1)
                .ok_or_else(|| "SenseNova image marker encodes to no tokens".to_string())?;
            if prefix[position] != profile.controls.end_of_image {
                return Err(OmniError::Invalid(
                    "SenseNova image marker does not end at its configured token".to_string(),
                ));
            }
            let position = position
                .try_into()
                .map_err(|_| "SenseNova image placement exceeds the token range".to_string())?;
            Ok(rendered_image(&image, position))
        })
        .collect::<OmniResult<Vec<_>>>()?;
    Ok((prompt_ids, images))
}

/// Renders structured Bagel chat while preserving one placeholder per input image.
fn render_bagel_chat(
    tokenizer: &DynTokenizer,
    renderer: &HfChatRenderer,
    request: &GenerateReqInput,
    constraint: GenerationConstraint,
) -> OmniResult<(Vec<u32>, Vec<RenderedImage>)> {
    let PromptInput::Chat {
        messages,
        tools,
        tool_choice,
        reasoning_effort,
    } = &request.prompt
    else {
        unreachable!("caller matched chat prompt")
    };
    let mut messages = messages.clone();
    if !messages
        .iter()
        .any(|message| matches!(message, ChatMessage::System { .. }))
        && let Some(system) = BagelProfile::default_system_prompt(constraint)
    {
        messages.insert(0, ChatMessage::system(system));
    }
    let (images, placeholders) = replace_chat_images(request.request_id.as_ref(), &mut messages)?;
    let rendered = renderer
        .render(&ChatRequest {
            messages,
            chat_options: crate::serving::chat::ChatOptions {
                generation_prompt_mode: GenerationPromptMode::StartNewAssistant,
                reasoning_effort: *reasoning_effort,
            },
            tools: tools.clone(),
            tool_choice: *tool_choice,
            decode_options: TextDecodeOptions::default(),
        })
        .map_err(|error| error.to_string())?;
    tokenize_bagel_with_slots(tokenizer, &rendered, &placeholders, images)
}

/// Removes Bagel image slots and computes their positions in token space.
fn tokenize_bagel_with_slots(
    tokenizer: &DynTokenizer,
    rendered: &str,
    placeholders: &[String],
    images: Vec<PositionedImageInput>,
) -> OmniResult<(Vec<u32>, Vec<RenderedImage>)> {
    let (clean, byte_offsets) = replace_rendered_slots(rendered, placeholders, "")?;
    let prompt_ids = tokenizer
        .encode(&clean, false)
        .map_err(|error| format!("Bagel chat tokenization failed: {error}"))?;
    let images = images
        .into_iter()
        .zip(byte_offsets)
        .map(|(image, byte_offset)| {
            let position = tokenizer
                .encode(&clean[..byte_offset], false)
                .map_err(|error| format!("Bagel image placement tokenization failed: {error}"))?
                .len()
                .try_into()
                .map_err(|_| "Bagel image placement exceeds the token range".to_string())?;
            Ok(rendered_image(&image, position))
        })
        .collect::<OmniResult<Vec<_>>>()?;
    Ok((prompt_ids, images))
}

/// Validates resolved multimodal state and assembles the canonical engine request.
fn finish_tokenized(
    binding: &RuntimeBinding<'_>,
    policy: &GenerationPolicyDescriptor,
    request: GenerateReqInput,
    mut lowered: LoweredInput,
    mut context: Vec<CoreContextSegment>,
    output_processor: OutputProcessorPolicy,
) -> OmniResult<TokenizedGenerateReqInput> {
    // Resolve the autoregressive budget only for requests whose behavior can
    // enter text decoding.
    let prompt_tokens = u32::try_from(lowered.prompt_ids.len())
        .map_err(|_| "generation prompt exceeds the supported token count".to_string())?;
    let behavior = GenerationBehaviorDescriptor::resolve(lowered.constraint, policy);
    let mut max_tokens = if behavior.und_decode {
        crate::serving::text::resolve_max_tokens(
            request.sampling.max_tokens,
            binding.default_max_output_tokens,
            Some(binding.max_model_tokens),
            prompt_tokens,
        )
        .map_err(|error| error.to_string())? as usize
    } else {
        0
    };

    // Sampling choices determine logprob delivery and whether prefix-cache
    // reads remain semantically valid for this request.
    apply_request_sampling(&binding.tokenizer, &request, &mut lowered.sampling)?;
    let prompt_logprobs_requested = lowered.sampling.prompt_logprobs_requested();
    let generated_logprobs_requested = lowered.sampling.generated_logprobs_requested();
    let cache = GenerationCachePolicyDescriptor {
        read: !request.cache.bypass_read && !prompt_logprobs_requested,
        write: !request.cache.no_store,
        isolation_key: cache_isolation_key(
            request.cache.namespace.as_deref(),
            request.cache.salt.as_deref(),
        ),
    };
    for segment in &mut context {
        if let CoreContextSegment::Image { image, .. } = segment {
            image.hash = isolated_cache_key(image.hash, cache.isolation_key);
        }
    }

    // Negative conditioning is internal context and contributes to resource
    // bounds only when the selected generation behavior consumes it.
    let negative_context = (!lowered.negative_prompt_ids.is_empty())
        .then(|| CoreContextSegment::UndTokens {
            token_ids: lowered.negative_prompt_ids.clone(),
            visibility: UndVisibility::Internal,
        })
        .into_iter()
        .collect::<Vec<_>>();

    // Compile conservative capacity before admission. Text output may shrink to
    // fit the model context, but non-text resource requirements remain fixed.
    let mut resources =
        GenerationResourceBounds::conservative(uniserve_core::GenerationResources {
            context: &context,
            negative_context: &negative_context,
            behavior: &behavior,
            policy,
            image: &lowered.image,
            max_und_tokens: max_tokens,
            cache: &cache,
            limits: binding.limits,
        })
        .map_err(|error| error.to_string())?;
    if resources.max_kv_tokens > binding.max_model_tokens as usize && behavior.und_decode {
        let excess = resources.max_kv_tokens - binding.max_model_tokens as usize;
        max_tokens = max_tokens
            .checked_sub(excess)
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                format!(
                    "generation context requires at least {} KV tokens, exceeding the {}-token runtime limit",
                    resources.max_kv_tokens.saturating_sub(max_tokens),
                    binding.max_model_tokens
                )
            })?;
        resources = GenerationResourceBounds::conservative(uniserve_core::GenerationResources {
            context: &context,
            negative_context: &negative_context,
            behavior: &behavior,
            policy,
            image: &lowered.image,
            max_und_tokens: max_tokens,
            cache: &cache,
            limits: binding.limits,
        })
        .map_err(|error| error.to_string())?;
    }

    if resources.max_kv_tokens > binding.max_model_tokens as usize {
        return Err(format!(
            "generation requires {} KV tokens, exceeding the {}-token runtime limit",
            resources.max_kv_tokens, binding.max_model_tokens
        )
        .into());
    }

    // Public accounting captures the admitted cache and resource contract before
    // ownership moves into the core generation request.
    let cache_accounting = CacheAccounting {
        read_enabled: cache.read,
        write_enabled: cache.write,
        encoder_pin_count: resources.encoder_cache_keys.len(),
    };
    let resource_accounting = ResourceAccounting {
        expected_kv_tokens: resources.max_kv_tokens as u64,
        image_latent_units: resources.max_image_latent_units,
        encoder_cache_pins: resources.encoder_cache_keys.len(),
        replayable: !resources.generated_feedback_makes_non_replayable,
    };

    let generation = GenerationRequest {
        request_id: RequestId(fnv1a(request.request_id.as_bytes())),
        context,
        negative_context,
        constraint: lowered.constraint,
        behavior,
        sampling: lowered.sampling,
        image: lowered.image,
        max_und_tokens: max_tokens,
        stop_strings: request.stop.stop_strings.clone(),
        stop_token_ids: lowered.stop_token_ids,
        priority: request.scheduling.priority,
        cache,
        policy: policy.clone(),
        resources,
    };
    generation.validate().map_err(|error| error.to_string())?;

    // Derive the flattened prompt only from the validated canonical context.
    let prompt_token_ids = generation.prompt_token_ids();
    Ok(TokenizedGenerateReqInput {
        request_id: request.request_id,
        request: generation,
        tokenizer: std::sync::Arc::clone(&binding.tokenizer),
        prompt_token_ids,
        decode: TextDecodeOptions {
            skip_special_tokens: request.decode.skip_special_tokens,
            include_stop_str_in_output: request.decode.include_stop_string_in_output,
            stop_strings: (!request.stop.stop_strings.is_empty())
                .then(|| request.stop.stop_strings.clone()),
            min_tokens: request.sampling.min_tokens.unwrap_or(0),
        },
        emit_token_ids: matches!(
            request.output,
            crate::serving::input::OutputDetail::Tokens
                | crate::serving::input::OutputDetail::Logprobs
        ),
        prompt_logprobs_requested,
        generated_logprobs_requested,
        skip_special_tokens: request.decode.skip_special_tokens,
        output_processor,
        identity: binding.identity.clone(),
        cache: cache_accounting,
        resources: resource_accounting,
    })
}

/// Applies request-specific penalties, masks, and bad-word tokens to profile sampling.
fn apply_request_sampling(
    tokenizer: &DynTokenizer,
    request: &GenerateReqInput,
    sampling: &mut SamplingParams,
) -> OmniResult<()> {
    sampling.ignore_eos = request.sampling.ignore_eos;
    sampling.min_tokens = request.sampling.min_tokens.unwrap_or(0) as usize;
    sampling.min_p = request.sampling.min_p.unwrap_or(0.0);
    sampling.frequency_penalty = request.sampling.frequency_penalty.unwrap_or(0.0);
    sampling.presence_penalty = request.sampling.presence_penalty.unwrap_or(0.0);
    sampling.repetition_penalty = request.sampling.repetition_penalty.unwrap_or(1.0);
    if let Some(request_bias) = &request.stop.logit_bias {
        let mut merged = std::collections::BTreeMap::new();
        for (token_id, bias) in sampling.logit_bias.drain(..) {
            merged.insert(token_id, bias);
        }
        for (&token_id, &bias) in request_bias {
            *merged.entry(token_id).or_insert(0.0) += bias;
        }
        sampling.logit_bias = merged.into_iter().collect();
    }
    sampling.return_logprobs =
        request.stop.logprobs.is_some() || request.stop.logprob_token_ids.is_some();
    sampling.n_logprobs = request
        .stop
        .logprobs
        .map_or(0, |count| if count < 0 { u32::MAX } else { count as u32 });
    sampling.return_prompt_logprobs = request.stop.prompt_logprobs.is_some();
    sampling.n_prompt_logprobs = request
        .stop
        .prompt_logprobs
        .map_or(0, |count| if count < 0 { u32::MAX } else { count as u32 });
    sampling.logprob_token_ids = request.stop.logprob_token_ids.clone().unwrap_or_default();
    sampling.allowed_token_ids = request.stop.allowed_token_ids.clone();
    sampling.bad_words_ids = request
        .stop
        .bad_words
        .iter()
        .map(|word| tokenizer.encode(word, false).map_err(OmniError::from))
        .collect::<OmniResult<Vec<_>>>()?;
    Ok(())
}

/// Resolves and validates baseline multimodal sampling parameters.
fn resolve_sampling(request: &GenerateReqInput) -> OmniResult<SamplingParams> {
    let temperature = finite(
        request.sampling.temperature.unwrap_or(DEFAULT_TEMPERATURE),
        "temperature",
    )?;
    if temperature < 0.0 {
        return Err(OmniError::Invalid(
            "temperature must be non-negative".to_string(),
        ));
    }
    let top_p = finite(request.sampling.top_p.unwrap_or(DEFAULT_TOP_P), "top_p")?;
    if !(0.0..=1.0).contains(&top_p) || top_p == 0.0 {
        return Err(OmniError::Invalid("top_p must be in (0, 1]".to_string()));
    }
    Ok(SamplingParams {
        temperature,
        top_p,
        top_k: request.sampling.top_k.unwrap_or(DEFAULT_TOP_K),
        seed: request
            .image_gen
            .as_ref()
            .and_then(|image| image.seed)
            .or_else(|| {
                request
                    .sampling
                    .seed
                    .and_then(|value| value.try_into().ok())
            }),
        ..SamplingParams::default()
    })
}

/// Merges profile defaults with request image controls and validates generation geometry.
fn resolve_image_params(
    defaults: &crate::profile::omni::ImageGenerationDefaults,
    resolution_policy: &ResolutionPolicy,
    request: &GenerateReqInput,
    constraint: GenerationConstraint,
) -> OmniResult<ImageParams> {
    let image = request.image_gen.clone().unwrap_or_default();
    let resolution = resolve_resolution(
        resolution_policy,
        image.resolution.or(Some(defaults.resolution)),
        image.width,
        image.height,
    )
    .map_err(|error| error.to_string())?;
    let steps = image.steps.unwrap_or(defaults.steps);
    if steps == 0 {
        return Err(OmniError::Invalid(
            "image.steps must be positive".to_string(),
        ));
    }
    let cfg_interval = image
        .cfg_interval
        .map(|interval| (interval[0], interval[1]))
        .unwrap_or(defaults.cfg_interval);
    validate_cfg_interval(cfg_interval)?;
    let max_images = image.max_images.unwrap_or(defaults.max_images);
    validate_max_images(max_images, defaults.max_images_limit)?;
    Ok(ImageParams {
        steps,
        cfg_text_scale: finite(
            image.cfg_text_scale.unwrap_or(defaults.cfg_text_scale),
            "image.cfg_text_scale",
        )?,
        cfg_img_scale: finite(
            image.cfg_img_scale.unwrap_or(defaults.cfg_img_scale),
            "image.cfg_img_scale",
        )?,
        cfg_renorm_type: image
            .cfg_renorm_type
            .unwrap_or_else(|| defaults.cfg_renorm_type.clone()),
        cfg_renorm_min: finite(
            image.cfg_renorm_min.unwrap_or(defaults.cfg_renorm_min),
            "image.cfg_renorm_min",
        )?,
        cfg_interval,
        timestep_shift: finite(
            image.timestep_shift.unwrap_or(defaults.timestep_shift),
            "image.timestep_shift",
        )?,
        height: resolution.height,
        width: resolution.width,
        seed: image
            .seed
            .or_else(|| {
                request
                    .sampling
                    .seed
                    .and_then(|value| value.try_into().ok())
            })
            .or(defaults.seed),
        negative_prompt: request.negative_text.clone().unwrap_or_default(),
        max_images,
        image_prompts: image.prompts,
        retain_images: image
            .retain_images
            .unwrap_or(constraint != GenerationConstraint::GenOnly),
    })
}

/// Resolves Bagel image parameters for understanding requests with context images.
fn bagel_context_image_params(
    profile: &BagelProfile,
    request: &GenerateReqInput,
) -> OmniResult<ImageParams> {
    let image = request.image_gen.clone().unwrap_or_default();
    let steps = image.steps.unwrap_or(DEFAULT_STEPS);
    if steps == 0 {
        return Err(OmniError::Invalid(
            "image.steps must be positive".to_string(),
        ));
    }
    let max_images = image.max_images.unwrap_or(2);
    validate_max_images(max_images, profile.image_defaults.max_images_limit)?;
    Ok(ImageParams {
        steps,
        cfg_text_scale: context_image_defaults::CFG_TEXT_SCALE,
        cfg_img_scale: context_image_defaults::CFG_IMG_SCALE,
        cfg_renorm_type: uniserve_core::CfgRenorm::TextChannel,
        cfg_renorm_min: context_image_defaults::CFG_RENORM_MIN,
        cfg_interval: context_image_defaults::CFG_INTERVAL,
        timestep_shift: image
            .timestep_shift
            .unwrap_or(profile.image_defaults.timestep_shift),
        height: context_image_defaults::RESOLUTION,
        width: context_image_defaults::RESOLUTION,
        seed: image
            .seed
            .or_else(|| {
                request
                    .sampling
                    .seed
                    .and_then(|value| value.try_into().ok())
            })
            .or(Some(0)),
        negative_prompt: String::new(),
        max_images,
        image_prompts: image.prompts,
        retain_images: image.retain_images.unwrap_or(true),
    })
}

/// Builds the sensenova context.
fn build_sensenova_context(
    profile: &SenseNovaProfile,
    lowered: &LoweredInput,
) -> OmniResult<Vec<CoreContextSegment>> {
    let image_count = lowered.images.len();
    let ingests = lowered
        .images
        .iter()
        .map(|image| -> OmniResult<ImageIngestRecipe> {
            let (width, height) = image_dimensions(&image.b64)?;
            profile
                .image_ingest_for_dimensions(width, height, image_count)
                .map_err(OmniError::from)
        })
        .collect::<OmniResult<Vec<_>>>()?;
    assemble_context(&lowered.prompt_ids, &lowered.images, ingests)
}

/// Builds the bagel context.
fn build_bagel_context(
    profile: &BagelProfile,
    lowered: &LoweredInput,
) -> OmniResult<Vec<CoreContextSegment>> {
    let image_count = lowered.images.len();
    let ingests = lowered
        .images
        .iter()
        .map(|image| -> OmniResult<ImageIngestRecipe> {
            let (width, height) = image_dimensions(&image.b64)?;
            profile
                .image_ingest_for_dimensions(width, height, image_count)
                .map_err(OmniError::from)
        })
        .collect::<OmniResult<Vec<_>>>()?;
    assemble_context(&lowered.prompt_ids, &lowered.images, ingests)
}

/// Interleaves token spans and positioned image segments into canonical context order.
fn assemble_context(
    prompt_ids: &[u32],
    images: &[RenderedImage],
    ingests: Vec<ImageIngestRecipe>,
) -> OmniResult<Vec<CoreContextSegment>> {
    if images.len() != ingests.len() {
        return Err(OmniError::Invalid(
            "image ingest declarations do not match input images".to_string(),
        ));
    }
    let mut indexed = images.iter().cloned().zip(ingests).collect::<Vec<_>>();
    indexed.sort_by_key(|(image, _)| image.position);
    let mut segments = Vec::with_capacity(indexed.len().saturating_mul(2).saturating_add(1));
    let mut token_cursor = 0usize;
    for (image, ingest) in indexed {
        let position = (image.position as usize).min(prompt_ids.len());
        if position > token_cursor {
            segments.push(CoreContextSegment::UndTokens {
                token_ids: prompt_ids[token_cursor..position].to_vec(),
                visibility: UndVisibility::Internal,
            });
        }
        segments.push(CoreContextSegment::Image {
            image: CoreImageSegment {
                hash: image.hash,
                b64: image.b64,
                placement: SegmentPlacement::AtToken {
                    position: image.position,
                },
            },
            ingest,
        });
        token_cursor = position;
    }
    if token_cursor < prompt_ids.len() {
        segments.push(CoreContextSegment::UndTokens {
            token_ids: prompt_ids[token_cursor..].to_vec(),
            visibility: UndVisibility::Internal,
        });
    }
    if segments.is_empty() {
        segments.push(CoreContextSegment::UndTokens {
            token_ids: prompt_ids.to_vec(),
            visibility: UndVisibility::Internal,
        });
    }
    Ok(segments)
}

/// Resolves the generation constraint implied by requested modalities.
pub(crate) fn generation_constraint(request: &GenerateReqInput) -> GenerationConstraint {
    match request.modalities {
        ModalitySelection::Text => GenerationConstraint::UndOnly,
        ModalitySelection::Image => GenerationConstraint::GenOnly,
        ModalitySelection::TextAndImage => GenerationConstraint::Default,
    }
}

/// Returns the images attached at the top request level.
fn top_level_images(request: &GenerateReqInput) -> Vec<PositionedImageInput> {
    request
        .images
        .iter()
        .map(|image| PositionedImageInput {
            b64: image.b64.clone(),
        })
        .collect()
}

/// Replaces chat image parts with unique template placeholders and retains their payloads.
fn replace_chat_images(
    request_id: &str,
    messages: &mut [ChatMessage],
) -> OmniResult<(Vec<PositionedImageInput>, Vec<String>)> {
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
            let placeholder = image_placeholder(request_id, images.len());
            images.push(PositionedImageInput {
                b64: data_image_payload(image_url)?,
            });
            placeholders.push(placeholder.clone());
            *part = ChatContentPart::Text { text: placeholder };
        }
    }
    Ok((images, placeholders))
}

/// Returns the placeholder for one image slot.
fn image_placeholder(request_id: &str, index: usize) -> String {
    format!("[IMAGE_SLOT_{:016x}_{index}]", fnv1a(request_id.as_bytes()))
}

/// Builds placeholders for the requested image slots.
fn image_placeholders(request_id: &str, count: usize) -> Vec<String> {
    (0..count)
        .map(|index| image_placeholder(request_id, index))
        .collect()
}

/// Replaces each rendered image placeholder exactly once and returns its byte offset.
fn replace_rendered_slots(
    rendered: &str,
    placeholders: &[String],
    replacement: &str,
) -> OmniResult<(String, Vec<usize>)> {
    let mut clean = String::with_capacity(rendered.len());
    let mut cursor = 0;
    let mut byte_offsets = Vec::with_capacity(placeholders.len());
    for placeholder in placeholders {
        if rendered.matches(placeholder).count() != 1 {
            return Err(OmniError::Invalid(
                "chat rendering must preserve one slot per input image".to_string(),
            ));
        }
        let relative = rendered[cursor..]
            .find(placeholder)
            .ok_or_else(|| "chat rendering changed input-image order".to_string())?;
        let start = cursor + relative;
        clean.push_str(&rendered[cursor..start]);
        clean.push_str(replacement);
        byte_offsets.push(clean.len());
        cursor = start + placeholder.len();
    }
    clean.push_str(&rendered[cursor..]);
    Ok((clean, byte_offsets))
}

/// Decodes an inline image data URL.
fn data_image_payload(url: &str) -> OmniResult<String> {
    let (metadata, payload) = url
        .split_once(',')
        .ok_or_else(|| "image chat requires a data:image/*;base64 URL".to_string())?;
    if !metadata.starts_with("data:image/") || !metadata.ends_with(";base64") || payload.is_empty()
    {
        return Err(OmniError::Invalid(
            "image chat requires a data:image/*;base64 URL".to_string(),
        ));
    }
    base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(|error| format!("image chat contains invalid base64 data: {error}"))?;
    Ok(payload.to_string())
}

/// Renders an image input for the model prompt.
fn rendered_image(image: &PositionedImageInput, position: u32) -> RenderedImage {
    RenderedImage {
        hash: fnv1a(image.b64.as_bytes()),
        position,
        b64: image.b64.clone(),
    }
}

/// Builds a prompt containing the required image slots.
fn prompt_with_image_slots(prompt: &str, placeholders: &[String]) -> String {
    let mut output = String::new();
    if placeholders.len() == 1 {
        output.push_str(&placeholders[0]);
        output.push('\n');
    } else {
        for (index, placeholder) in placeholders.iter().enumerate() {
            output.push_str(&format!("Image-{}:{placeholder}\n", index + 1));
        }
    }
    output.push_str(prompt);
    output
}

/// Returns the decoded image dimensions.
fn image_dimensions(b64: &str) -> OmniResult<(u32, u32)> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|error| format!("invalid input image base64: {error}"))?;
    Ok(image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|error| format!("invalid input image data: {error}"))?
        .into_dimensions()
        .map_err(|error| format!("invalid input image data: {error}"))?)
}

/// Validates the prompt.
fn validate_prompt(prompt_ids: &[u32]) -> OmniResult<()> {
    if prompt_ids.is_empty() {
        Err(OmniError::Invalid(
            "generation requires a non-empty prompt".to_string(),
        ))
    } else {
        Ok(())
    }
}

/// Validates the max images.
fn validate_max_images(value: u16, limit: u16) -> OmniResult<()> {
    if value == 0 {
        return Err(OmniError::Invalid(
            "image.max_images must be positive".to_string(),
        ));
    }
    if value > limit {
        return Err(format!("image.max_images exceeds model limit {limit}").into());
    }
    Ok(())
}

/// Validates the CFG interval.
fn validate_cfg_interval(value: (f32, f32)) -> OmniResult<()> {
    let (lo, hi) = value;
    if !lo.is_finite() || !hi.is_finite() || lo > hi {
        return Err(OmniError::Invalid(
            "image.cfg_interval must be a finite ordered pair".to_string(),
        ));
    }
    Ok(())
}

/// Returns the value when it is finite.
fn finite(value: f32, name: &str) -> OmniResult<f32> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("{name} must be finite").into())
    }
}

/// Builds a cache key isolated by request namespace and salt.
fn isolated_cache_key(content_key: u64, isolation_key: Option<u64>) -> u64 {
    let Some(isolation_key) = isolation_key else {
        return content_key;
    };
    let mut bytes = [0_u8; 16];
    bytes[..8].copy_from_slice(&content_key.to_le_bytes());
    bytes[8..].copy_from_slice(&isolation_key.to_le_bytes());
    fnv1a(&bytes)
}

/// Computes a stable FNV-1a hash.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
