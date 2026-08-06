//! SenseNova/Bagel model-private framing, image ingest, and generation
//! lowering.
//!
//! This module owns everything between a [`GenerateReqInput`] admitted for an
//! omni description and the single [`TokenizedGenerateReqInput`] submitted to
//! the engine: prompt framing, image placement/ingest, negative-prompt
//! encoding, image-control normalization, resource declaration, and the
//! description-owned output filter.

use std::io::Cursor;

use base64::Engine as _;
use uniserve_core::{
    ContextSegment as CoreContextSegment, GenerationBehaviorDescriptor,
    GenerationCachePolicyDescriptor, GenerationConstraint, GenerationRequest,
    GenerationResourceBounds, GenerationRuntimeCapabilities, ImageParams,
    ImageSegment as CoreImageSegment, RequestId, SamplingParams as EngineSamplingParams,
    SegmentPlacement, UndVisibility,
};
use uniserve_model_profile::dialect::{
    GenerationDialectProfile, PromptContext, PromptKind, resolution::resolve_resolution,
};

use crate::chat::{
    ChatContent, ChatContentPart, ChatMessage, ChatRequest, GenerationPromptMode, HfChatRenderer,
};
use crate::input::{
    GenerateReqInput, ImageGenControls, ModelEventIdentity, OutputProcessorPolicy, PromptInput,
    SubmissionMetadata, TokenizedGenerateReqInput,
};
use crate::text::TextDecodeOptions;
use crate::text::tokenizer::DynTokenizer;
use crate::{CacheAccounting, ResourceAccounting, Result, ServeError, cache_isolation_key};

#[path = "../dialect_generation/output.rs"]
pub(crate) mod output;

/// Default number of diffusion sampling steps for image generation.
const DEFAULT_STEPS: u16 = 50;
/// Default sampling temperature (0.0 = greedy / deterministic).
const DEFAULT_TEMPERATURE: f32 = 0.0;
/// Default nucleus-sampling probability mass (1.0 = disabled).
const DEFAULT_TOP_P: f32 = 1.0;
/// Default top-k cutoff (0 = disabled).
const DEFAULT_TOP_K: u32 = 0;

/// CFG and resolution constants for text output with image context.
mod context_image_defaults {
    pub(super) const CFG_TEXT_SCALE: f32 = 4.0;
    pub(super) const CFG_IMG_SCALE: f32 = 2.0;
    pub(super) const CFG_RENORM_TYPE: &str = "text_channel";
    pub(super) const CFG_RENORM_MIN: f32 = 0.0;
    pub(super) const CFG_INTERVAL: (f32, f32) = (0.0, 1.0);
    pub(super) const RESOLUTION: u32 = 512;
}

/// Cache bounds needed to build the generation cache descriptor and image
/// isolation key.
struct CacheDescriptorInput {
    bypass_read: bool,
    no_store: bool,
    isolation_key: Option<u64>,
}

/// Read-only context handed to omni tokenization.
pub(crate) struct OmniContext<'a> {
    pub dialect: &'a GenerationDialectProfile,
    pub tokenizer: DynTokenizer,
    pub renderer: &'a HfChatRenderer,
    pub capabilities: &'a GenerationRuntimeCapabilities,
    pub default_max_output_tokens: Option<u32>,
    pub max_model_tokens: u32,
    pub identity: ModelEventIdentity,
}

/// Lower one omni-admitted [`GenerateReqInput`] into the single tokenized value
/// submitted to the engine gateway.
pub(crate) fn tokenize_omni(
    ctx: &OmniContext<'_>,
    request: GenerateReqInput,
) -> Result<TokenizedGenerateReqInput> {
    let request_id = request.request_id.clone();
    build_tokenized(ctx, request).map_err(|message| ServeError::Tokenize {
        request_id,
        message,
    })
}

fn build_tokenized(
    ctx: &OmniContext<'_>,
    request: GenerateReqInput,
) -> std::result::Result<TokenizedGenerateReqInput, String> {
    let has_input_image = request.has_input_image();
    let constraint = derive_constraint(&request, has_input_image);

    let input = CompileInput::from_request(ctx, &request, constraint)?;
    let mut lowered =
        GenerationRequestCompiler::new(std::sync::Arc::clone(&ctx.tokenizer), ctx.dialect)
            .build(&input)?;

    let prompt_tokens = u32::try_from(lowered.prompt_ids.len())
        .map_err(|_| "generation prompt exceeds the supported token count".to_string())?;
    let policy = ctx.dialect.generation_policy.clone();
    let behavior = GenerationBehaviorDescriptor::resolve(lowered.constraint, &policy);
    let mut max_tokens = if behavior.und_decode {
        crate::text::resolve_max_tokens(
            request.sampling.max_tokens,
            ctx.default_max_output_tokens,
            Some(ctx.max_model_tokens),
            prompt_tokens,
        )
        .map_err(|error| error.to_string())? as usize
    } else {
        0
    };

    lowered.sampling.ignore_eos = request.sampling.ignore_eos;
    lowered.sampling.min_tokens = request.sampling.min_tokens.unwrap_or(0) as usize;
    lowered.sampling.min_p = request.sampling.min_p.unwrap_or(0.0);
    lowered.sampling.frequency_penalty = request.sampling.frequency_penalty.unwrap_or(0.0);
    lowered.sampling.presence_penalty = request.sampling.presence_penalty.unwrap_or(0.0);
    lowered.sampling.repetition_penalty = request.sampling.repetition_penalty.unwrap_or(1.0);
    if let Some(request_bias) = &request.stop.logit_bias {
        let mut merged_bias = std::collections::BTreeMap::new();
        for (token_id, bias) in lowered.sampling.logit_bias.drain(..) {
            merged_bias.insert(token_id, bias);
        }
        for (&token_id, &bias) in request_bias {
            *merged_bias.entry(token_id).or_insert(0.0) += bias;
        }
        lowered.sampling.logit_bias = merged_bias.into_iter().collect();
    }
    lowered.sampling.return_logprobs =
        request.stop.logprobs.is_some() || request.stop.logprob_token_ids.is_some();
    lowered.sampling.n_logprobs = request
        .stop
        .logprobs
        .map_or(0, |count| if count < 0 { u32::MAX } else { count as u32 });
    lowered.sampling.return_prompt_logprobs = request.stop.prompt_logprobs.is_some();
    lowered.sampling.n_prompt_logprobs = request
        .stop
        .prompt_logprobs
        .map_or(0, |count| if count < 0 { u32::MAX } else { count as u32 });
    lowered.sampling.logprob_token_ids = request.stop.logprob_token_ids.clone().unwrap_or_default();
    lowered.sampling.allowed_token_ids = request.stop.allowed_token_ids.clone();
    lowered.sampling.bad_words_ids = request
        .stop
        .bad_words
        .iter()
        .map(|word| {
            ctx.tokenizer
                .encode(word, false)
                .map_err(|error| error.to_string())
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let prompt_logprobs_requested = lowered.sampling.prompt_logprobs_requested();
    let generated_logprobs_requested = lowered.sampling.generated_logprobs_requested();

    let cache_input = CacheDescriptorInput {
        bypass_read: request.cache.bypass_read,
        no_store: request.cache.no_store,
        isolation_key: cache_isolation_key(
            request.cache.namespace.as_deref(),
            request.cache.salt.as_deref(),
        ),
    };
    let cache = generation_cache_policy(&cache_input, prompt_logprobs_requested);
    let mut context = build_context_segments(&lowered, ctx.dialect)?;
    for segment in &mut context {
        if let CoreContextSegment::Image { image, .. } = segment {
            image.hash = isolated_cache_key(image.hash, cache.isolation_key);
        }
    }
    let negative_context: Vec<CoreContextSegment> = (!lowered.neg_prompt_ids.is_empty())
        .then(|| CoreContextSegment::UndTokens {
            token_ids: lowered.neg_prompt_ids.clone(),
            visibility: UndVisibility::Internal,
        })
        .into_iter()
        .collect();
    let image = lowered.image;
    let mut resources = GenerationResourceBounds::conservative(
        &context,
        &negative_context,
        &behavior,
        &policy,
        &image,
        max_tokens,
        &cache,
        ctx.capabilities,
    )
    .map_err(|error| error.to_string())?;
    if resources.max_kv_tokens > ctx.max_model_tokens as usize && behavior.und_decode {
        let excess = resources.max_kv_tokens - ctx.max_model_tokens as usize;
        max_tokens = max_tokens
            .checked_sub(excess)
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                format!(
                    "generation context requires at least {} KV tokens, exceeding the {}-token runtime limit",
                    resources.max_kv_tokens.saturating_sub(max_tokens),
                    ctx.max_model_tokens
                )
            })?;
        resources = GenerationResourceBounds::conservative(
            &context,
            &negative_context,
            &behavior,
            &policy,
            &image,
            max_tokens,
            &cache,
            ctx.capabilities,
        )
        .map_err(|error| error.to_string())?;
    }
    if resources.max_kv_tokens > ctx.max_model_tokens as usize {
        return Err(format!(
            "generation requires {} KV tokens, exceeding the {}-token runtime limit",
            resources.max_kv_tokens, ctx.max_model_tokens
        ));
    }

    let cache_accounting = CacheAccounting {
        read_enabled: cache.read,
        write_enabled: cache.write,
        encoder_pin_count: resources.encoder_cache_keys.len(),
        transfer: None,
    };
    let resource_accounting = resource_accounting_from(&resources);

    let generation = GenerationRequest {
        request_id: RequestId(fnv1a(request.request_id.as_bytes())),
        context,
        negative_context,
        constraint: lowered.constraint,
        behavior,
        sampling: lowered.sampling,
        image,
        max_und_tokens: max_tokens,
        stop_strings: request.stop.stop_strings.clone(),
        stop_token_ids: lowered.stop_token_ids,
        priority: request.scheduling.priority,
        cache,
        policy,
        resources,
    };
    generation.validate().map_err(|error| error.to_string())?;

    let prompt_token_ids = generation.prompt_token_ids();
    let decode = TextDecodeOptions {
        skip_special_tokens: request.decode.skip_special_tokens,
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
            crate::input::OutputContract::Tokens | crate::input::OutputContract::Logprobs
        ),
        prompt_logprobs_requested,
        generated_logprobs_requested,
        skip_special_tokens: request.decode.skip_special_tokens,
        output_processor: OutputProcessorPolicy::Dialect(ctx.dialect.output_filter.clone()),
        submission: SubmissionMetadata {
            trace_headers: (!request.scheduling.trace_context.is_empty())
                .then(|| request.scheduling.trace_context.clone()),
        },
        identity: ctx.identity.clone(),
        cache: cache_accounting,
        resources: resource_accounting,
    })
}

fn resource_accounting_from(resources: &GenerationResourceBounds) -> ResourceAccounting {
    ResourceAccounting {
        expected_kv_tokens: resources.max_kv_tokens as u64,
        image_latent_units: resources.max_image_latent_units,
        scratch_units: resources.max_scratch_units,
        host_scratch_tokens: resources.max_host_scratch_tokens,
        encoder_cache_pins: resources.encoder_cache_keys.len(),
        replayable: !resources.generated_feedback_makes_non_replayable,
    }
}

fn derive_constraint(request: &GenerateReqInput, has_input_image: bool) -> GenerationConstraint {
    if request.modalities.output_image && !request.modalities.output_text {
        GenerationConstraint::GenOnly
    } else if has_input_image && !request.modalities.output_image {
        GenerationConstraint::UndOnly
    } else {
        GenerationConstraint::Default
    }
}

#[derive(Debug, Clone, Default)]
struct CompileInput {
    prompt: String,
    prompt_ids: Option<Vec<u32>>,
    constraint: Option<GenerationConstraint>,
    system_prompt: Option<String>,
    assistant_prefix: Option<String>,
    negative_prompt: Option<String>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    seed: Option<u64>,
    stop_token_ids: Vec<u32>,
    image: Option<ImageGenControls>,
    input_images: Vec<PositionedImageInput>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PositionedImageInput {
    b64: String,
    placement: Option<u32>,
}

impl CompileInput {
    fn from_request(
        ctx: &OmniContext<'_>,
        request: &GenerateReqInput,
        constraint: GenerationConstraint,
    ) -> std::result::Result<Self, String> {
        let mut input = Self {
            constraint: Some(constraint),
            temperature: request.sampling.temperature,
            top_p: request.sampling.top_p,
            top_k: request.sampling.top_k,
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
            stop_token_ids: request.stop.stop_token_ids.clone(),
            negative_prompt: request.negative_text.clone(),
            image: request.image_gen.clone(),
            ..Self::default()
        };

        match &request.prompt {
            PromptInput::Text(text) => {
                input.prompt = text.clone();
                input.input_images = request
                    .images
                    .iter()
                    .map(|image| PositionedImageInput {
                        b64: image.b64.clone(),
                        placement: None,
                    })
                    .collect();
            }
            PromptInput::Chat { .. } => {
                let (prompt_ids, images) = render_chat_to_tokens(ctx, request, constraint)?;
                input.prompt_ids = Some(prompt_ids);
                input.input_images = images;
            }
        }
        Ok(input)
    }

    fn image(&self) -> ImageGenControls {
        self.image.clone().unwrap_or_default()
    }

    fn negative_prompt(&self) -> String {
        self.negative_prompt.clone().unwrap_or_default()
    }

    fn input_images(&self) -> Vec<PositionedImageInput> {
        self.input_images.clone()
    }

    fn prompt_context(&self) -> PromptContext {
        PromptContext {
            prompt: self.prompt.clone(),
            system_prompt: self.system_prompt.clone(),
            assistant_prefix: self.assistant_prefix.clone(),
        }
    }
}

/// Render a chat prompt (with dialect scaffold and image markers) into prompt
/// token IDs plus positioned input images.
fn render_chat_to_tokens(
    ctx: &OmniContext<'_>,
    request: &GenerateReqInput,
    constraint: GenerationConstraint,
) -> std::result::Result<(Vec<u32>, Vec<PositionedImageInput>), String> {
    let PromptInput::Chat {
        messages,
        tools,
        tool_choice,
        reasoning_effort,
    } = &request.prompt
    else {
        return Err("omni chat rendering requires a chat prompt".to_string());
    };
    let mut messages = messages.clone();
    let mut chat_options = crate::chat::ChatOptions {
        generation_prompt_mode: GenerationPromptMode::StartNewAssistant,
        reasoning_effort: *reasoning_effort,
    };

    let image_count = semantic_image_count(&messages);
    let prompt_kind = PromptKind::for_request(constraint, image_count > 0);
    let scaffold = ctx.dialect.chat_prompt_scaffold(prompt_kind);
    if !messages
        .iter()
        .any(|message| matches!(message, ChatMessage::System { .. }))
        && let Some(system) = scaffold.default_system.filter(|value| !value.is_empty())
    {
        messages.insert(0, ChatMessage::system(system));
    }
    if !scaffold.assistant_prefix.is_empty()
        && chat_options.generation_prompt_mode == GenerationPromptMode::StartNewAssistant
    {
        messages.push(ChatMessage::assistant_text(scaffold.assistant_prefix));
        chat_options.generation_prompt_mode = GenerationPromptMode::ContinueFinalAssistant;
    }

    let marker_text = ctx.dialect.context_markers_in_prompt().then(|| {
        format!(
            "{}{}",
            ctx.dialect.controls.start_of_image_text, ctx.dialect.controls.end_of_image_text
        )
    });
    if marker_text.as_ref().is_some_and(String::is_empty) {
        return Err("the generation profile has no context-image marker text".to_string());
    }
    let (images, placeholders) = replace_chat_images(
        request.request_id.as_ref(),
        &mut messages,
        marker_text.as_deref(),
    )?;

    let chat_request = ChatRequest {
        messages,
        chat_options,
        tools: tools.clone(),
        tool_choice: *tool_choice,
        decode_options: TextDecodeOptions::default(),
    };
    let rendered_text = ctx
        .renderer
        .render(&chat_request)
        .map_err(|error| error.to_string())?;
    tokenize_rendered_chat_with_images(
        &ctx.tokenizer,
        &rendered_text,
        ctx.dialect.context_markers_in_prompt(),
        &placeholders,
        images,
    )
}

fn semantic_image_count(messages: &[ChatMessage]) -> usize {
    messages
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
        .sum()
}

fn replace_chat_images(
    request_id: &str,
    messages: &mut [ChatMessage],
    marker_text: Option<&str>,
) -> std::result::Result<(Vec<PositionedImageInput>, Vec<String>), String> {
    let request_fingerprint = format!("{:016x}", fnv1a(request_id.as_bytes()));
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
                    request_fingerprint,
                    images.len()
                )
            });
            images.push(PositionedImageInput {
                b64: data_image_payload(image_url)?,
                placement: None,
            });
            placeholders.push(placeholder.clone());
            *part = ChatContentPart::Text { text: placeholder };
        }
    }
    Ok((images, placeholders))
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

fn tokenize_rendered_chat_with_images(
    tokenizer: &DynTokenizer,
    rendered: &str,
    markers_in_prompt: bool,
    placeholders: &[String],
    mut images: Vec<PositionedImageInput>,
) -> std::result::Result<(Vec<u32>, Vec<PositionedImageInput>), String> {
    if markers_in_prompt {
        let token_ids = tokenizer
            .encode(rendered, false)
            .map_err(|error| format!("image chat tokenization failed: {error}"))?;
        return Ok((token_ids, images));
    }
    let mut clean = String::with_capacity(rendered.len());
    let mut cursor = 0;
    let mut image_offsets = Vec::with_capacity(placeholders.len());
    for placeholder in placeholders {
        if rendered.matches(placeholder).count() != 1 {
            return Err(
                "chat rendering did not preserve one unique slot per input image".to_string(),
            );
        }
        let relative = rendered[cursor..]
            .find(placeholder)
            .ok_or_else(|| "chat rendering changed input-image order".to_string())?;
        let start = cursor + relative;
        clean.push_str(&rendered[cursor..start]);
        image_offsets.push(clean.len());
        cursor = start + placeholder.len();
    }
    clean.push_str(&rendered[cursor..]);
    let token_ids = tokenizer
        .encode(&clean, false)
        .map_err(|error| format!("image chat tokenization failed: {error}"))?;
    for (image, byte_offset) in images.iter_mut().zip(image_offsets) {
        let prefix_tokens = tokenizer
            .encode(&clean[..byte_offset], false)
            .map_err(|error| format!("image placement tokenization failed: {error}"))?;
        image.placement = Some(
            prefix_tokens
                .len()
                .try_into()
                .map_err(|_| "image placement exceeds the supported token range".to_string())?,
        );
    }
    Ok((token_ids, images))
}

#[derive(Debug, Clone)]
struct LoweredGenerationInput {
    prompt_ids: Vec<u32>,
    neg_prompt_ids: Vec<u32>,
    sampling: EngineSamplingParams,
    image: ImageParams,
    constraint: GenerationConstraint,
    mm_items: Vec<RenderedImage>,
    stop_token_ids: Vec<u32>,
}

#[derive(Debug, Clone)]
struct RenderedImage {
    hash: u64,
    position: u32,
    b64: String,
}

struct GenerationRequestCompiler<'a> {
    tokenizer: DynTokenizer,
    profile: &'a GenerationDialectProfile,
}

impl<'a> GenerationRequestCompiler<'a> {
    fn new(tokenizer: DynTokenizer, profile: &'a GenerationDialectProfile) -> Self {
        Self { tokenizer, profile }
    }

    fn build(&self, body: &CompileInput) -> std::result::Result<LoweredGenerationInput, String> {
        let constraint = body.constraint.unwrap_or_default();
        let input_images = body.input_images();
        let path = RequestPath::new(constraint, !input_images.is_empty());
        validate_prompt(body)?;
        if !self.profile.supports_constraint(constraint) {
            return Err(format!(
                "constraint {} is not supported by this model profile",
                constraint.as_str()
            ));
        }
        if path.uses_feedback_ingest() {
            return self.build_und_with_images(body, constraint, &input_images);
        }

        let image = self.resolve_image_params(body, constraint)?;
        let negative_prompt = body.negative_prompt();
        let (prompt_ids, mm_items) = self.build_context_prompt(body, path.prompt, &input_images)?;
        let neg_prompt_ids = self
            .profile
            .build_negative_prompt_ids(&self.tokenizer, &negative_prompt)
            .map_err(|error| error.to_string())?;
        let sampling = self.resolve_sampling(body)?;

        Ok(LoweredGenerationInput {
            prompt_ids,
            neg_prompt_ids,
            sampling,
            image,
            constraint,
            mm_items,
            stop_token_ids: body.stop_token_ids.clone(),
        })
    }

    fn build_und_with_images(
        &self,
        body: &CompileInput,
        constraint: GenerationConstraint,
        input_images: &[PositionedImageInput],
    ) -> std::result::Result<LoweredGenerationInput, String> {
        if self.profile.context_markers_in_prompt() || body.prompt_ids.is_some() {
            return self.build_und_with_marked_images(body, constraint, input_images);
        }
        if input_images.is_empty() {
            return Err("und_only requests with context images require image data".to_string());
        }
        let first_image = input_images.first().ok_or_else(|| {
            "und_only requests with context images require image data".to_string()
        })?;
        let system = body
            .system_prompt
            .as_deref()
            .unwrap_or_else(|| self.profile.context_system_prompt());
        let mut sys_ids = self
            .profile
            .wrap_context_text(&self.tokenizer, system)
            .map_err(|error| error.to_string())?;
        let question_ids = self
            .profile
            .wrap_context_text(&self.tokenizer, &body.prompt)
            .map_err(|error| error.to_string())?;
        let image_position = first_image.placement.unwrap_or(sys_ids.len() as u32);
        sys_ids.extend_from_slice(&question_ids);

        let image_body = body.image();
        let steps = image_body.steps.unwrap_or(DEFAULT_STEPS);
        if steps == 0 {
            return Err("image.steps must be positive".to_string());
        }
        let max_images = image_body.max_images.unwrap_or(2);
        self.validate_max_images(max_images)?;
        let seed = image_body.seed.or(body.seed).or(Some(0));

        let image = ImageParams {
            steps,
            cfg_text_scale: context_image_defaults::CFG_TEXT_SCALE,
            cfg_img_scale: context_image_defaults::CFG_IMG_SCALE,
            cfg_renorm_type: context_image_defaults::CFG_RENORM_TYPE.into(),
            cfg_renorm_min: context_image_defaults::CFG_RENORM_MIN,
            cfg_interval: context_image_defaults::CFG_INTERVAL,
            timestep_shift: image_body
                .timestep_shift
                .unwrap_or(self.profile.image_defaults.timestep_shift),
            height: context_image_defaults::RESOLUTION,
            width: context_image_defaults::RESOLUTION,
            seed,
            negative_prompt: String::new(),
            max_images,
            image_prompts: image_body.prompts,
            retain_images: image_body.retain_images.unwrap_or(true),
        };
        Ok(LoweredGenerationInput {
            neg_prompt_ids: sys_ids.clone(),
            prompt_ids: sys_ids,
            sampling: self.resolve_sampling(body)?,
            image,
            constraint,
            mm_items: input_images
                .iter()
                .map(|image| mm_item(image, image.placement.unwrap_or(image_position)))
                .collect(),
            stop_token_ids: body.stop_token_ids.clone(),
        })
    }

    fn build_und_with_marked_images(
        &self,
        body: &CompileInput,
        constraint: GenerationConstraint,
        input_images: &[PositionedImageInput],
    ) -> std::result::Result<LoweredGenerationInput, String> {
        if input_images.is_empty() {
            return Err("und_only requests with context images require image data".to_string());
        }
        let (prompt_ids, mm_items) =
            self.build_context_prompt(body, PromptKind::UndWithImages, input_images)?;
        let negative_prompt = body.negative_prompt();
        Ok(LoweredGenerationInput {
            neg_prompt_ids: self
                .profile
                .build_negative_prompt_ids(&self.tokenizer, &negative_prompt)
                .map_err(|error| error.to_string())?,
            prompt_ids,
            sampling: self.resolve_sampling(body)?,
            image: self.resolve_image_params(body, constraint)?,
            constraint,
            mm_items,
            stop_token_ids: body.stop_token_ids.clone(),
        })
    }

    fn build_context_prompt(
        &self,
        body: &CompileInput,
        kind: PromptKind,
        input_images: &[PositionedImageInput],
    ) -> std::result::Result<(Vec<u32>, Vec<RenderedImage>), String> {
        if let Some(prompt_ids) = &body.prompt_ids {
            if input_images.is_empty() {
                return Ok((prompt_ids.clone(), Vec::new()));
            }
            if self.profile.context_markers_in_prompt() {
                let mut marker_positions = prompt_ids
                    .iter()
                    .enumerate()
                    .filter(|&(_, &token)| token == self.profile.controls.end_of_image)
                    .map(|(index, _)| index as u32);
                let mut mm_items = Vec::with_capacity(input_images.len());
                for image in input_images {
                    let position = marker_positions.next().ok_or_else(|| {
                        "pre-tokenized context lost an input-image marker".to_string()
                    })?;
                    mm_items.push(mm_item(image, position));
                }
                if marker_positions.next().is_some() {
                    return Err(
                        "pre-tokenized context contains more image markers than image segments"
                            .to_string(),
                    );
                }
                return Ok((prompt_ids.clone(), mm_items));
            }
            let fallback_position = prompt_ids.len() as u32;
            let mm_items = input_images
                .iter()
                .map(|image| mm_item(image, image.placement.unwrap_or(fallback_position)))
                .collect();
            return Ok((prompt_ids.clone(), mm_items));
        }
        if input_images.is_empty() {
            let prompt_ids = self
                .profile
                .build_prompt_ids(&self.tokenizer, &body.prompt_context(), kind)
                .map_err(|error| error.to_string())?;
            return Ok((prompt_ids, Vec::new()));
        }
        let controls = &self.profile.controls;
        if self.profile.context_markers_in_prompt() {
            if controls.start_of_image_text.is_empty() || controls.end_of_image_text.is_empty() {
                return Err(
                    "model profile declares no image marker tokens for context images".to_string(),
                );
            }
            let user_text = prompt_with_image_markers(
                &body.prompt,
                input_images.len(),
                &controls.start_of_image_text,
                &controls.end_of_image_text,
            );
            let prompt_ids = self
                .profile
                .build_prompt_ids_with_text(
                    &self.tokenizer,
                    &body.prompt_context(),
                    kind,
                    &user_text,
                )
                .map_err(|error| error.to_string())?;
            let mut marker_positions = prompt_ids
                .iter()
                .enumerate()
                .filter(|&(_, &token)| token == controls.end_of_image)
                .map(|(index, _)| index as u32);
            let mut mm_items = Vec::with_capacity(input_images.len());
            for image in input_images {
                let position = marker_positions
                    .next()
                    .ok_or_else(|| "context image prompt lost its image markers".to_string())?;
                mm_items.push(mm_item(image, position));
            }
            return Ok((prompt_ids, mm_items));
        }
        let prompt_ids = self
            .profile
            .build_prompt_ids(&self.tokenizer, &body.prompt_context(), kind)
            .map_err(|error| error.to_string())?;
        let fallback_position = prompt_ids.len() as u32;
        let mm_items = input_images
            .iter()
            .map(|image| mm_item(image, image.placement.unwrap_or(fallback_position)))
            .collect();
        Ok((prompt_ids, mm_items))
    }

    fn resolve_image_params(
        &self,
        body: &CompileInput,
        constraint: GenerationConstraint,
    ) -> std::result::Result<ImageParams, String> {
        let image_body = body.image();
        let defaults = &self.profile.image_defaults;
        let resolution = resolve_resolution(
            &self.profile.resolution_policy,
            image_body
                .resolution
                .as_deref()
                .or(Some(defaults.resolution.as_str())),
            image_body.width,
            image_body.height,
        )?;
        let steps = image_body.steps.unwrap_or(defaults.steps);
        if steps == 0 {
            return Err("image.steps must be positive".to_string());
        }
        let cfg_interval = image_body
            .cfg_interval
            .map(|interval| (interval[0], interval[1]))
            .unwrap_or(defaults.cfg_interval);
        validate_cfg_interval(cfg_interval)?;

        let max_images = image_body.max_images.unwrap_or(defaults.max_images);
        self.validate_max_images(max_images)?;
        let negative_prompt = body.negative_prompt();
        Ok(ImageParams {
            steps,
            cfg_text_scale: finite_or(
                image_body.cfg_text_scale.unwrap_or(defaults.cfg_text_scale),
                "image.cfg_text_scale",
            )?,
            cfg_img_scale: finite_or(
                image_body.cfg_img_scale.unwrap_or(defaults.cfg_img_scale),
                "image.cfg_img_scale",
            )?,
            cfg_renorm_type: image_body
                .cfg_renorm_type
                .clone()
                .unwrap_or_else(|| defaults.cfg_renorm_type.clone()),
            cfg_renorm_min: finite_or(
                image_body.cfg_renorm_min.unwrap_or(defaults.cfg_renorm_min),
                "image.cfg_renorm_min",
            )?,
            cfg_interval,
            timestep_shift: finite_or(
                image_body.timestep_shift.unwrap_or(defaults.timestep_shift),
                "image.timestep_shift",
            )?,
            height: resolution.height,
            width: resolution.width,
            seed: image_body.seed.or(body.seed).or(defaults.seed),
            negative_prompt,
            max_images,
            image_prompts: image_body.prompts,
            retain_images: image_body
                .retain_images
                .unwrap_or(constraint != GenerationConstraint::GenOnly),
        })
    }

    fn resolve_sampling(
        &self,
        body: &CompileInput,
    ) -> std::result::Result<EngineSamplingParams, String> {
        let temperature = finite_or(
            body.temperature.unwrap_or(DEFAULT_TEMPERATURE),
            "temperature",
        )?;
        if temperature < 0.0 {
            return Err("temperature must be non-negative".to_string());
        }
        let top_p = finite_or(body.top_p.unwrap_or(DEFAULT_TOP_P), "top_p")?;
        if !(0.0..=1.0).contains(&top_p) || top_p == 0.0 {
            return Err("top_p must be in (0, 1]".to_string());
        }
        Ok(EngineSamplingParams {
            temperature,
            top_p,
            top_k: body.top_k.unwrap_or(DEFAULT_TOP_K),
            seed: body.seed,
            ..EngineSamplingParams::default()
        })
    }

    fn validate_max_images(&self, value: u16) -> std::result::Result<(), String> {
        if value == 0 {
            return Err("image.max_images must be positive".to_string());
        }
        if value > self.profile.image_defaults.max_images_limit {
            return Err(format!(
                "image.max_images exceeds profile limit {}",
                self.profile.image_defaults.max_images_limit
            ));
        }
        Ok(())
    }
}

fn generation_cache_policy(
    cache: &CacheDescriptorInput,
    prompt_logprobs_requested: bool,
) -> GenerationCachePolicyDescriptor {
    GenerationCachePolicyDescriptor {
        read: !cache.bypass_read && !prompt_logprobs_requested,
        write: !cache.no_store,
        isolation_key: cache.isolation_key,
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

fn build_context_segments(
    lowered: &LoweredGenerationInput,
    profile: &GenerationDialectProfile,
) -> std::result::Result<Vec<CoreContextSegment>, String> {
    let mut images = lowered.mm_items.clone();
    images.sort_by_key(|image| image.position);
    let image_count = images.len();
    let mut segments = Vec::with_capacity(images.len().saturating_mul(2).saturating_add(1));
    let mut token_cursor = 0usize;
    for image in images {
        let position = (image.position as usize).min(lowered.prompt_ids.len());
        if position > token_cursor {
            segments.push(CoreContextSegment::UndTokens {
                token_ids: lowered.prompt_ids[token_cursor..position].to_vec(),
                visibility: UndVisibility::Internal,
            });
        }
        let ingest = if profile.image_ingest_requires_dimensions() {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(&image.b64)
                .map_err(|error| format!("invalid input image base64: {error}"))?;
            let dimensions = image::ImageReader::new(Cursor::new(bytes))
                .with_guessed_format()
                .map_err(|error| format!("invalid input image data: {error}"))?
                .into_dimensions()
                .map_err(|error| format!("invalid input image data: {error}"))?;
            profile
                .image_ingest_for_dimensions(dimensions.0, dimensions.1, image_count)
                .map_err(|error| error.to_string())?
        } else {
            profile.image_ingest.clone()
        };
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
    if token_cursor < lowered.prompt_ids.len() {
        segments.push(CoreContextSegment::UndTokens {
            token_ids: lowered.prompt_ids[token_cursor..].to_vec(),
            visibility: UndVisibility::Internal,
        });
    }
    if segments.is_empty() {
        segments.push(CoreContextSegment::UndTokens {
            token_ids: lowered.prompt_ids.clone(),
            visibility: UndVisibility::Internal,
        });
    }
    Ok(segments)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RequestPath {
    prompt: PromptKind,
    feedback_ingest: bool,
}

impl RequestPath {
    fn new(constraint: GenerationConstraint, has_input_images: bool) -> Self {
        match (constraint, has_input_images) {
            (GenerationConstraint::Default, _) => Self {
                prompt: PromptKind::Default,
                feedback_ingest: false,
            },
            (GenerationConstraint::UndOnly, true) => Self {
                prompt: PromptKind::UndWithImages,
                feedback_ingest: true,
            },
            (GenerationConstraint::UndOnly, false) => Self {
                prompt: PromptKind::Und,
                feedback_ingest: false,
            },
            (GenerationConstraint::GenOnly, _) => Self {
                prompt: PromptKind::Gen,
                feedback_ingest: false,
            },
        }
    }

    fn uses_feedback_ingest(self) -> bool {
        self.feedback_ingest
    }
}

fn validate_prompt(body: &CompileInput) -> std::result::Result<(), String> {
    if body.prompt.trim().is_empty()
        && body
            .prompt_ids
            .as_ref()
            .is_none_or(|token_ids| token_ids.is_empty())
    {
        return Err("generation requires a non-empty prompt".to_string());
    }
    Ok(())
}

fn prompt_with_image_markers(
    prompt: &str,
    image_count: usize,
    start_marker: &str,
    end_marker: &str,
) -> String {
    let mut user_text = String::new();
    if image_count == 1 {
        user_text.push_str(start_marker);
        user_text.push_str(end_marker);
        user_text.push('\n');
    } else {
        for index in 0..image_count {
            user_text.push_str(&format!(
                "Image-{}:{}{}\n",
                index + 1,
                start_marker,
                end_marker
            ));
        }
    }
    user_text.push_str(prompt);
    user_text
}

fn mm_item(image: &PositionedImageInput, position: u32) -> RenderedImage {
    RenderedImage {
        hash: fnv1a(image.b64.as_bytes()),
        position,
        b64: image.b64.clone(),
    }
}

fn validate_cfg_interval(value: (f32, f32)) -> std::result::Result<(), String> {
    let (lo, hi) = value;
    if !lo.is_finite() || !hi.is_finite() || lo > hi {
        return Err("image.cfg_interval must be a finite ordered pair".to_string());
    }
    Ok(())
}

fn finite_or(value: f32, name: &str) -> std::result::Result<f32, String> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("{name} must be finite"))
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
