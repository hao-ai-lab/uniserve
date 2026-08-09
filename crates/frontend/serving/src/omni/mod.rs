//! Description-owned preprocessing for the configured SenseNova and Bagel routes.

mod output;

use std::io::Cursor;

use base64::Engine as _;
use uniserve_core::{
    ContextSegment as CoreContextSegment, GenerationBehaviorDescriptor,
    GenerationCachePolicyDescriptor, GenerationConstraint, GenerationPolicyDescriptor,
    GenerationRequest, GenerationResourceBounds, GenerationRuntimeCapabilities, ImageIngestRecipe,
    ImageParams, ImageSegment as CoreImageSegment, RequestId, SegmentPlacement, UndVisibility,
};
use uniserve_model_profile::omni::bagel::{BagelProfile, CONTEXT_SYSTEM_PROMPT};
use uniserve_model_profile::omni::resolution::{ResolutionPolicy, resolve_resolution};
use uniserve_model_profile::omni::sensenova::SenseNovaProfile;

use crate::chat::{
    ChatContent, ChatContentPart, ChatMessage, ChatRequest, GenerationPromptMode, HfChatRenderer,
};
use crate::input::{
    GenerateReqInput, ModelEventIdentity, OutputProcessorPolicy, PromptInput, SubmissionMetadata,
    TokenizedGenerateReqInput,
};
use crate::sampling::{SamplingDefaults, lower_sampling};
use crate::text::TextDecodeOptions;
use crate::text::tokenizer::DynTokenizer;
use crate::{CacheAccounting, ResourceAccounting, Result, ServeError, cache_isolation_key};

pub(crate) use output::{SenseNovaOutputProcessor, SenseNovaTextDelta};

const DEFAULT_STEPS: u16 = 50;

mod context_image_defaults {
    pub(super) const CFG_TEXT_SCALE: f32 = 4.0;
    pub(super) const CFG_IMG_SCALE: f32 = 2.0;
    pub(super) const CFG_RENORM_TYPE: &str = "text_channel";
    pub(super) const CFG_RENORM_MIN: f32 = 0.0;
    pub(super) const CFG_INTERVAL: (f32, f32) = (0.0, 1.0);
    pub(super) const RESOLUTION: u32 = 512;
}

struct RuntimeBinding<'a> {
    tokenizer: DynTokenizer,
    capabilities: &'a GenerationRuntimeCapabilities,
    policy: &'a GenerationPolicyDescriptor,
    sampling: &'a SamplingDefaults,
    identity: ModelEventIdentity,
}

#[derive(Debug, Clone)]
struct LoweredInput {
    prompt_ids: Vec<u32>,
    negative_prompt_ids: Vec<u32>,
    image: ImageParams,
    constraint: GenerationConstraint,
    images: Vec<RenderedImage>,
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

pub(crate) fn tokenize_sensenova(
    profile: &SenseNovaProfile,
    tokenizer: DynTokenizer,
    renderer: &HfChatRenderer,
    capabilities: &GenerationRuntimeCapabilities,
    sampling: &SamplingDefaults,
    identity: ModelEventIdentity,
    request: GenerateReqInput,
) -> Result<TokenizedGenerateReqInput> {
    let request_id = request.request_id.clone();
    let binding = RuntimeBinding {
        tokenizer,
        capabilities,
        policy: &profile.generation_policy,
        sampling,
        identity,
    };
    let result = (|| {
        let lowered = lower_sensenova(profile, &binding.tokenizer, renderer, &request)?;
        let context = build_sensenova_context(profile, &lowered)?;
        finish_tokenized(
            &binding,
            request,
            lowered,
            context,
            OutputProcessorPolicy::SenseNova(profile.output_filter.clone()),
        )
    })();
    result.map_err(|message| ServeError::Tokenize {
        request_id,
        message,
    })
}

pub(crate) fn tokenize_bagel(
    profile: &BagelProfile,
    tokenizer: DynTokenizer,
    renderer: &HfChatRenderer,
    capabilities: &GenerationRuntimeCapabilities,
    sampling: &SamplingDefaults,
    identity: ModelEventIdentity,
    request: GenerateReqInput,
) -> Result<TokenizedGenerateReqInput> {
    let request_id = request.request_id.clone();
    let binding = RuntimeBinding {
        tokenizer,
        capabilities,
        policy: &profile.generation_policy,
        sampling,
        identity,
    };
    let result = (|| {
        let lowered = lower_bagel(profile, &binding.tokenizer, renderer, &request)?;
        let context = build_bagel_context(profile, &lowered)?;
        finish_tokenized(
            &binding,
            request,
            lowered,
            context,
            OutputProcessorPolicy::Bagel,
        )
    })();
    result.map_err(|message| ServeError::Tokenize {
        request_id,
        message,
    })
}

fn lower_sensenova(
    profile: &SenseNovaProfile,
    tokenizer: &DynTokenizer,
    renderer: &HfChatRenderer,
    request: &GenerateReqInput,
) -> std::result::Result<LoweredInput, String> {
    let constraint = derive_constraint(request);
    let (prompt_ids, images) = sensenova_prompt(profile, tokenizer, renderer, request, constraint)?;
    validate_prompt(&prompt_ids)?;
    let negative_prompt_ids = profile
        .render_negative_prompt_ids(tokenizer, request.negative_text.as_deref().unwrap_or(""))
        .map_err(|error| error.to_string())?;
    Ok(LoweredInput {
        prompt_ids,
        negative_prompt_ids,
        image: resolve_image_params(
            &profile.image_defaults,
            &profile.resolution_policy,
            request,
            constraint,
        )?,
        constraint,
        images,
    })
}

fn lower_bagel(
    profile: &BagelProfile,
    tokenizer: &DynTokenizer,
    renderer: &HfChatRenderer,
    request: &GenerateReqInput,
) -> std::result::Result<LoweredInput, String> {
    let constraint = derive_constraint(request);
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
        image,
        constraint,
        images,
    })
}

fn sensenova_prompt(
    profile: &SenseNovaProfile,
    tokenizer: &DynTokenizer,
    renderer: &HfChatRenderer,
    request: &GenerateReqInput,
    constraint: GenerationConstraint,
) -> std::result::Result<(Vec<u32>, Vec<RenderedImage>), String> {
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

fn bagel_prompt(
    profile: &BagelProfile,
    tokenizer: &DynTokenizer,
    renderer: &HfChatRenderer,
    request: &GenerateReqInput,
    constraint: GenerationConstraint,
) -> std::result::Result<(Vec<u32>, Vec<RenderedImage>, bool), String> {
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

fn render_sensenova_chat(
    profile: &SenseNovaProfile,
    tokenizer: &DynTokenizer,
    renderer: &HfChatRenderer,
    request: &GenerateReqInput,
    constraint: GenerationConstraint,
) -> std::result::Result<(Vec<u32>, Vec<RenderedImage>), String> {
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
            chat_options: crate::chat::ChatOptions {
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

fn tokenize_sensenova_with_slots(
    tokenizer: &DynTokenizer,
    rendered: &str,
    placeholders: &[String],
    images: Vec<PositionedImageInput>,
    profile: &SenseNovaProfile,
) -> std::result::Result<(Vec<u32>, Vec<RenderedImage>), String> {
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
        .map(|(image, byte_offset)| {
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
                return Err(
                    "SenseNova image marker does not end at its configured token".to_string(),
                );
            }
            let position = position
                .try_into()
                .map_err(|_| "SenseNova image placement exceeds the token range".to_string())?;
            Ok(rendered_image(&image, position))
        })
        .collect::<std::result::Result<Vec<_>, String>>()?;
    Ok((prompt_ids, images))
}

fn render_bagel_chat(
    tokenizer: &DynTokenizer,
    renderer: &HfChatRenderer,
    request: &GenerateReqInput,
    constraint: GenerationConstraint,
) -> std::result::Result<(Vec<u32>, Vec<RenderedImage>), String> {
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
            chat_options: crate::chat::ChatOptions {
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

fn tokenize_bagel_with_slots(
    tokenizer: &DynTokenizer,
    rendered: &str,
    placeholders: &[String],
    images: Vec<PositionedImageInput>,
) -> std::result::Result<(Vec<u32>, Vec<RenderedImage>), String> {
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
        .collect::<std::result::Result<Vec<_>, String>>()?;
    Ok((prompt_ids, images))
}

fn finish_tokenized(
    binding: &RuntimeBinding<'_>,
    request: GenerateReqInput,
    lowered: LoweredInput,
    mut context: Vec<CoreContextSegment>,
    output_processor: OutputProcessorPolicy,
) -> std::result::Result<TokenizedGenerateReqInput, String> {
    let prompt_tokens = u32::try_from(lowered.prompt_ids.len())
        .map_err(|_| "generation prompt exceeds the supported token count".to_string())?;
    let sampling = lower_sampling(binding.tokenizer.as_ref(), &request, binding.sampling)?;
    let behavior = GenerationBehaviorDescriptor::resolve(lowered.constraint, binding.policy);
    let mut max_tokens = if behavior.und_decode {
        crate::text::resolve_max_tokens(
            request.sampling.max_tokens,
            binding.sampling.max_tokens,
            Some(binding.sampling.max_model_tokens),
            prompt_tokens,
        )
        .map_err(|error| error.to_string())? as usize
    } else {
        0
    };

    let prompt_logprobs_requested = sampling.params.prompt_logprobs_requested();
    let generated_logprobs_requested = sampling.params.generated_logprobs_requested();
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
    let negative_context = (!lowered.negative_prompt_ids.is_empty())
        .then(|| CoreContextSegment::UndTokens {
            token_ids: lowered.negative_prompt_ids.clone(),
            visibility: UndVisibility::Internal,
        })
        .into_iter()
        .collect::<Vec<_>>();
    let mut resources = GenerationResourceBounds::conservative(
        &context,
        &negative_context,
        &behavior,
        binding.policy,
        &lowered.image,
        max_tokens,
        &cache,
        binding.capabilities,
    )
    .map_err(|error| error.to_string())?;
    if resources.max_kv_tokens > binding.sampling.max_model_tokens as usize && behavior.und_decode {
        let excess = resources.max_kv_tokens - binding.sampling.max_model_tokens as usize;
        max_tokens = max_tokens
            .checked_sub(excess)
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                format!(
                    "generation context requires at least {} KV tokens, exceeding the {}-token runtime limit",
                    resources.max_kv_tokens.saturating_sub(max_tokens),
                    binding.sampling.max_model_tokens
                )
            })?;
        resources = GenerationResourceBounds::conservative(
            &context,
            &negative_context,
            &behavior,
            binding.policy,
            &lowered.image,
            max_tokens,
            &cache,
            binding.capabilities,
        )
        .map_err(|error| error.to_string())?;
    }
    if resources.max_kv_tokens > binding.sampling.max_model_tokens as usize {
        return Err(format!(
            "generation requires {} KV tokens, exceeding the {}-token runtime limit",
            resources.max_kv_tokens, binding.sampling.max_model_tokens
        ));
    }
    if sampling.params.min_tokens > max_tokens {
        return Err(format!(
            "min_tokens ({}) exceeds max_tokens ({max_tokens})",
            sampling.params.min_tokens
        ));
    }

    let cache_accounting = CacheAccounting {
        read_enabled: cache.read,
        write_enabled: cache.write,
        encoder_pin_count: resources.encoder_cache_keys.len(),
    };
    let resource_accounting = ResourceAccounting {
        expected_kv_tokens: resources.max_kv_tokens as u64,
        image_latent_units: resources.max_image_latent_units,
        scratch_units: resources.max_scratch_units,
        host_scratch_tokens: resources.max_host_scratch_tokens,
        encoder_cache_pins: resources.encoder_cache_keys.len(),
        replayable: !resources.generated_feedback_makes_non_replayable,
    };
    let generation = GenerationRequest {
        request_id: RequestId(fnv1a(request.request_id.as_bytes())),
        context,
        negative_context,
        constraint: lowered.constraint,
        behavior,
        sampling: sampling.params,
        image: lowered.image,
        max_und_tokens: max_tokens,
        stop_strings: request.stop.stop_strings.clone(),
        stop_token_ids: sampling.stop_token_ids,
        priority: request.scheduling.priority,
        cache,
        policy: binding.policy.clone(),
        resources,
    };
    generation.validate().map_err(|error| error.to_string())?;
    let prompt_token_ids = generation.prompt_token_ids();
    Ok(TokenizedGenerateReqInput {
        request_id: request.request_id,
        request: generation,
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
            crate::input::OutputContract::Tokens | crate::input::OutputContract::Logprobs
        ),
        prompt_logprobs_requested,
        generated_logprobs_requested,
        skip_special_tokens: request.decode.skip_special_tokens,
        output_processor,
        submission: SubmissionMetadata {
            trace_headers: (!request.scheduling.trace_context.is_empty())
                .then(|| request.scheduling.trace_context.clone()),
        },
        identity: binding.identity.clone(),
        cache: cache_accounting,
        resources: resource_accounting,
    })
}

fn resolve_image_params(
    defaults: &uniserve_model_profile::omni::ImageGenerationDefaults,
    resolution_policy: &ResolutionPolicy,
    request: &GenerateReqInput,
    constraint: GenerationConstraint,
) -> std::result::Result<ImageParams, String> {
    let image = request.image_gen.clone().unwrap_or_default();
    let resolution = resolve_resolution(
        resolution_policy,
        image
            .resolution
            .as_deref()
            .or(Some(defaults.resolution.as_str())),
        image.width,
        image.height,
    )?;
    let steps = image.steps.unwrap_or(defaults.steps);
    if steps == 0 {
        return Err("image.steps must be positive".to_string());
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

fn bagel_context_image_params(
    profile: &BagelProfile,
    request: &GenerateReqInput,
) -> std::result::Result<ImageParams, String> {
    let image = request.image_gen.clone().unwrap_or_default();
    let steps = image.steps.unwrap_or(DEFAULT_STEPS);
    if steps == 0 {
        return Err("image.steps must be positive".to_string());
    }
    let max_images = image.max_images.unwrap_or(2);
    validate_max_images(max_images, profile.image_defaults.max_images_limit)?;
    Ok(ImageParams {
        steps,
        cfg_text_scale: context_image_defaults::CFG_TEXT_SCALE,
        cfg_img_scale: context_image_defaults::CFG_IMG_SCALE,
        cfg_renorm_type: context_image_defaults::CFG_RENORM_TYPE.to_string(),
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

fn build_sensenova_context(
    profile: &SenseNovaProfile,
    lowered: &LoweredInput,
) -> std::result::Result<Vec<CoreContextSegment>, String> {
    let image_count = lowered.images.len();
    let ingests = lowered
        .images
        .iter()
        .map(|image| {
            let (width, height) = image_dimensions(&image.b64)?;
            profile
                .image_ingest_for_dimensions(width, height, image_count)
                .map_err(|error| error.to_string())
        })
        .collect::<std::result::Result<Vec<_>, String>>()?;
    assemble_context(&lowered.prompt_ids, &lowered.images, ingests)
}

fn build_bagel_context(
    profile: &BagelProfile,
    lowered: &LoweredInput,
) -> std::result::Result<Vec<CoreContextSegment>, String> {
    let image_count = lowered.images.len();
    let ingests = lowered
        .images
        .iter()
        .map(|image| {
            let (width, height) = image_dimensions(&image.b64)?;
            profile
                .image_ingest_for_dimensions(width, height, image_count)
                .map_err(|error| error.to_string())
        })
        .collect::<std::result::Result<Vec<_>, String>>()?;
    assemble_context(&lowered.prompt_ids, &lowered.images, ingests)
}

fn assemble_context(
    prompt_ids: &[u32],
    images: &[RenderedImage],
    ingests: Vec<ImageIngestRecipe>,
) -> std::result::Result<Vec<CoreContextSegment>, String> {
    if images.len() != ingests.len() {
        return Err("image ingest declarations do not match input images".to_string());
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

fn derive_constraint(request: &GenerateReqInput) -> GenerationConstraint {
    if request.modalities.output_image && !request.modalities.output_text {
        GenerationConstraint::GenOnly
    } else if request.has_input_image() && !request.modalities.output_image {
        GenerationConstraint::UndOnly
    } else {
        GenerationConstraint::Default
    }
}

fn top_level_images(request: &GenerateReqInput) -> Vec<PositionedImageInput> {
    request
        .images
        .iter()
        .map(|image| PositionedImageInput {
            b64: image.b64.clone(),
        })
        .collect()
}

fn replace_chat_images(
    request_id: &str,
    messages: &mut [ChatMessage],
) -> std::result::Result<(Vec<PositionedImageInput>, Vec<String>), String> {
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

fn image_placeholder(request_id: &str, index: usize) -> String {
    format!("[IMAGE_SLOT_{:016x}_{index}]", fnv1a(request_id.as_bytes()))
}

fn image_placeholders(request_id: &str, count: usize) -> Vec<String> {
    (0..count)
        .map(|index| image_placeholder(request_id, index))
        .collect()
}

fn replace_rendered_slots(
    rendered: &str,
    placeholders: &[String],
    replacement: &str,
) -> std::result::Result<(String, Vec<usize>), String> {
    let mut clean = String::with_capacity(rendered.len());
    let mut cursor = 0;
    let mut byte_offsets = Vec::with_capacity(placeholders.len());
    for placeholder in placeholders {
        if rendered.matches(placeholder).count() != 1 {
            return Err("chat rendering must preserve one slot per input image".to_string());
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

fn data_image_payload(url: &str) -> std::result::Result<String, String> {
    let (metadata, payload) = url
        .split_once(',')
        .ok_or_else(|| "image chat requires a data:image/*;base64 URL".to_string())?;
    if !metadata.starts_with("data:image/") || !metadata.ends_with(";base64") || payload.is_empty()
    {
        return Err("image chat requires a data:image/*;base64 URL".to_string());
    }
    base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(|error| format!("image chat contains invalid base64 data: {error}"))?;
    Ok(payload.to_string())
}

fn rendered_image(image: &PositionedImageInput, position: u32) -> RenderedImage {
    RenderedImage {
        hash: fnv1a(image.b64.as_bytes()),
        position,
        b64: image.b64.clone(),
    }
}

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

fn image_dimensions(b64: &str) -> std::result::Result<(u32, u32), String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|error| format!("invalid input image base64: {error}"))?;
    image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|error| format!("invalid input image data: {error}"))?
        .into_dimensions()
        .map_err(|error| format!("invalid input image data: {error}"))
}

fn validate_prompt(prompt_ids: &[u32]) -> std::result::Result<(), String> {
    if prompt_ids.is_empty() {
        Err("generation requires a non-empty prompt".to_string())
    } else {
        Ok(())
    }
}

fn validate_max_images(value: u16, limit: u16) -> std::result::Result<(), String> {
    if value == 0 {
        return Err("image.max_images must be positive".to_string());
    }
    if value > limit {
        return Err(format!("image.max_images exceeds model limit {limit}"));
    }
    Ok(())
}

fn validate_cfg_interval(value: (f32, f32)) -> std::result::Result<(), String> {
    let (lo, hi) = value;
    if !lo.is_finite() || !hi.is_finite() || lo > hi {
        return Err("image.cfg_interval must be a finite ordered pair".to_string());
    }
    Ok(())
}

fn finite(value: f32, name: &str) -> std::result::Result<f32, String> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("{name} must be finite"))
    }
}

fn isolated_cache_key(content_key: u64, isolation_key: Option<u64>) -> u64 {
    let Some(isolation_key) = isolation_key else {
        return content_key;
    };
    let mut bytes = [0_u8; 16];
    bytes[..8].copy_from_slice(&content_key.to_le_bytes());
    bytes[8..].copy_from_slice(&isolation_key.to_le_bytes());
    fnv1a(&bytes)
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
