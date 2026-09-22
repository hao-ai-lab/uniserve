//! Multimodal prompt preprocessing for SenseNova and Bagel profiles.

mod output;

use std::io::Cursor;
use thiserror::Error;

use crate::profile::omni::bagel::{BagelProfile, CONTEXT_SYSTEM_PROMPT};
use crate::profile::omni::resolution::{ResolutionPolicy, resolve_resolution};
use crate::profile::omni::sensenova::SenseNovaProfile;
use base64::Engine as _;
use uniserve_core::{
    GenerationConstraint, GenerationRequest, ImageEncoderInput, ImageInput, ImageParams,
    MultimodalInputs, SamplingParams,
};

use crate::serving::chat::{
    ChatContent, ChatContentPart, ChatMessage, ChatRequest, GenerationPromptMode, HfChatRenderer,
};
use crate::serving::input::{ModalitySelection, OutputProcessorPolicy, PromptInput};
use crate::serving::text::tokenizer::DynTokenizer;

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

#[derive(Debug, Clone)]
struct RenderedImage {
    hash: u64,
    position: u32,
    b64: String,
}

/// Fills the SenseNova prompt, image inputs, and image-generation parameters.
#[allow(clippy::too_many_arguments)]
pub(super) fn preprocess_sensenova(
    profile: &SenseNovaProfile,
    processor: &crate::serving::InputProcessor,
    request_id: &crate::serving::ServeRequestId,
    prompt: PromptInput,
    images: Vec<crate::serving::ImageInput>,
    negative_text: Option<String>,
    image_gen: Option<crate::serving::ImageGenControls>,
    generation: &mut GenerationRequest,
) -> OmniResult<OutputProcessorPolicy> {
    let (prompt_token_ids, images) = sensenova_prompt(
        profile,
        &processor.tokenizer,
        processor
            .renderer
            .as_ref()
            .expect("text models require a chat renderer"),
        request_id,
        prompt,
        images.into_iter().map(|image| image.b64).collect(),
        generation.constraint,
    )?;
    validate_prompt(&prompt_token_ids)?;
    let negative_prompt_token_ids = profile
        .render_negative_prompt_ids(&processor.tokenizer, negative_text.as_deref().unwrap_or(""))
        .map_err(|error| error.to_string())?;
    let image = resolve_image_params(
        &profile.image_defaults,
        &profile.resolution_policy,
        image_gen.unwrap_or_default(),
        generation.sampling.seed,
        negative_text.unwrap_or_default(),
        generation.constraint,
    )?;
    let multimodal_inputs = prepare_image_inputs(
        prompt_token_ids.len(),
        images,
        profile.image_num_positions,
        |width, height, count| {
            profile
                .image_encoders_for_dimensions(width, height, count)
                .map_err(OmniError::from)
        },
    )?;
    let policy = profile
        .image_generation_for_dimensions(image.width, image.height)
        .map_err(|error| error.to_string())?;
    generation.prompt_token_ids = prompt_token_ids;
    generation.negative_prompt_token_ids = negative_prompt_token_ids;
    generation.multimodal_inputs = multimodal_inputs;
    generation.image = image;

    generation.image_generation = policy;
    Ok(OutputProcessorPolicy::SenseNova(
        profile.output_filter.clone(),
    ))
}

/// Renders Bagel input with the model's context-image conditioning rules.
#[allow(clippy::too_many_arguments)]
pub(super) fn preprocess_bagel(
    profile: &BagelProfile,
    processor: &crate::serving::InputProcessor,
    request_id: &crate::serving::ServeRequestId,
    prompt: PromptInput,
    images: Vec<crate::serving::ImageInput>,
    negative_text: Option<String>,
    image_gen: Option<crate::serving::ImageGenControls>,
    generation: &mut GenerationRequest,
) -> OmniResult<OutputProcessorPolicy> {
    let (prompt_token_ids, images, context_image_mode) = bagel_prompt(
        profile,
        &processor.tokenizer,
        processor
            .renderer
            .as_ref()
            .expect("text models require a chat renderer"),
        request_id,
        prompt,
        images.into_iter().map(|image| image.b64).collect(),
        generation.constraint,
    )?;
    validate_prompt(&prompt_token_ids)?;
    let negative_prompt_token_ids = if context_image_mode {
        prompt_token_ids.clone()
    } else {
        profile
            .render_negative_prompt_ids(
                &processor.tokenizer,
                negative_text.as_deref().unwrap_or(""),
            )
            .map_err(|error| error.to_string())?
    };
    let image = if context_image_mode {
        bagel_context_image_params(
            profile,
            image_gen.unwrap_or_default(),
            generation.sampling.seed,
        )?
    } else {
        resolve_image_params(
            &profile.image_defaults,
            &profile.resolution_policy,
            image_gen.unwrap_or_default(),
            generation.sampling.seed,
            negative_text.unwrap_or_default(),
            generation.constraint,
        )?
    };
    let multimodal_inputs = prepare_image_inputs(
        prompt_token_ids.len(),
        images,
        profile.image_num_positions,
        |width, height, count| {
            profile
                .image_encoders_for_dimensions(width, height, count)
                .map_err(OmniError::from)
        },
    )?;
    let policy = profile
        .image_generation_for_dimensions(image.width, image.height)
        .map_err(|error| error.to_string())?;
    generation.prompt_token_ids = prompt_token_ids;
    generation.negative_prompt_token_ids = negative_prompt_token_ids;
    generation.multimodal_inputs = multimodal_inputs;
    generation.image = image;

    generation.image_generation = policy;
    Ok(OutputProcessorPolicy::None)
}

/// Renders a SenseNova text or chat prompt and resolves positioned input images.
fn sensenova_prompt(
    profile: &SenseNovaProfile,
    tokenizer: &DynTokenizer,
    renderer: &HfChatRenderer,
    request_id: &str,
    prompt: PromptInput,
    images: Vec<String>,
    constraint: GenerationConstraint,
) -> OmniResult<(Vec<u32>, Vec<RenderedImage>)> {
    match prompt {
        PromptInput::Text(prompt) => {
            if images.is_empty() {
                let prompt_ids = profile
                    .render_prompt_ids(tokenizer, constraint, &prompt, None, None)
                    .map_err(|error| error.to_string())?;
                Ok((prompt_ids, Vec::new()))
            } else {
                let placeholders = image_placeholders(request_id, images.len());
                let prompt = prompt_with_image_slots(&prompt, &placeholders);
                let rendered = profile.render_prompt_text(constraint, &prompt, None, None);
                preprocess_sensenova_with_slots(
                    tokenizer,
                    &rendered,
                    &placeholders,
                    images,
                    profile,
                )
            }
        }
        PromptInput::Chat(chat) => {
            render_sensenova_chat(profile, tokenizer, renderer, request_id, chat, constraint)
        }
    }
}

/// Renders a Bagel prompt and selects its context-image conditioning mode.
fn bagel_prompt(
    profile: &BagelProfile,
    tokenizer: &DynTokenizer,
    renderer: &HfChatRenderer,
    request_id: &str,
    prompt: PromptInput,
    images: Vec<String>,
    constraint: GenerationConstraint,
) -> OmniResult<(Vec<u32>, Vec<RenderedImage>, bool)> {
    match prompt {
        PromptInput::Chat(chat) => {
            let (prompt_ids, images) =
                render_bagel_chat(tokenizer, renderer, request_id, chat, constraint)?;
            Ok((prompt_ids, images, false))
        }
        PromptInput::Text(prompt)
            if constraint == GenerationConstraint::UndOnly && !images.is_empty() =>
        {
            let mut prompt_ids = profile
                .wrap_context_text(tokenizer, CONTEXT_SYSTEM_PROMPT)
                .map_err(|error| error.to_string())?;
            let image_position = prompt_ids.len() as u32;
            prompt_ids.extend(
                profile
                    .wrap_context_text(tokenizer, &prompt)
                    .map_err(|error| error.to_string())?,
            );
            let images = images
                .into_iter()
                .map(|image| rendered_image(image, image_position))
                .collect();
            Ok((prompt_ids, images, true))
        }
        PromptInput::Text(prompt) => {
            let prompt_ids = profile
                .render_prompt_ids(
                    tokenizer,
                    constraint,
                    !images.is_empty(),
                    &prompt,
                    None,
                    None,
                )
                .map_err(|error| error.to_string())?;
            let position = prompt_ids.len() as u32;
            let images = images
                .into_iter()
                .map(|image| rendered_image(image, position))
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
    request_id: &str,
    mut chat: ChatRequest,
    constraint: GenerationConstraint,
) -> OmniResult<(Vec<u32>, Vec<RenderedImage>)> {
    if !chat
        .messages
        .iter()
        .any(|message| matches!(message, ChatMessage::System { .. }))
        && let Some(system) = SenseNovaProfile::default_system_prompt(constraint)
    {
        chat.messages.insert(0, ChatMessage::system(system));
    }
    let mut generation_prompt_mode = GenerationPromptMode::StartNewAssistant;
    let assistant_prefix = SenseNovaProfile::assistant_prefix(constraint);
    if !assistant_prefix.is_empty() {
        chat.messages
            .push(ChatMessage::assistant_text(assistant_prefix));
        generation_prompt_mode = GenerationPromptMode::ContinueFinalAssistant;
    }
    let (images, placeholders) = replace_chat_images(request_id, &mut chat.messages)?;
    chat.chat_options.generation_prompt_mode = generation_prompt_mode;
    let rendered = renderer.render(&chat).map_err(|error| error.to_string())?;
    preprocess_sensenova_with_slots(tokenizer, &rendered, &placeholders, images, profile)
}

/// Replaces SenseNova image slots and computes their positions in token space.
fn preprocess_sensenova_with_slots(
    tokenizer: &DynTokenizer,
    rendered: &str,
    placeholders: &[String],
    images: Vec<String>,
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
                .map_err(|error| format!("SenseNova image params tokenization failed: {error}"))?;
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
                .map_err(|_| "SenseNova image params exceeds the token range".to_string())?;
            Ok(rendered_image(image, position))
        })
        .collect::<OmniResult<Vec<_>>>()?;
    Ok((prompt_ids, images))
}

/// Renders structured Bagel chat while preserving one placeholder per input image.
fn render_bagel_chat(
    tokenizer: &DynTokenizer,
    renderer: &HfChatRenderer,
    request_id: &str,
    mut chat: ChatRequest,
    constraint: GenerationConstraint,
) -> OmniResult<(Vec<u32>, Vec<RenderedImage>)> {
    if !chat
        .messages
        .iter()
        .any(|message| matches!(message, ChatMessage::System { .. }))
        && let Some(system) = BagelProfile::default_system_prompt(constraint)
    {
        chat.messages.insert(0, ChatMessage::system(system));
    }
    let (images, placeholders) = replace_chat_images(request_id, &mut chat.messages)?;
    chat.chat_options.generation_prompt_mode = GenerationPromptMode::StartNewAssistant;
    let rendered = renderer.render(&chat).map_err(|error| error.to_string())?;
    tokenize_bagel_with_slots(tokenizer, &rendered, &placeholders, images)
}

/// Removes Bagel image slots and computes their positions in token space.
fn tokenize_bagel_with_slots(
    tokenizer: &DynTokenizer,
    rendered: &str,
    placeholders: &[String],
    images: Vec<String>,
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
                .map_err(|error| format!("Bagel image params tokenization failed: {error}"))?
                .len()
                .try_into()
                .map_err(|_| "Bagel image params exceeds the token range".to_string())?;
            Ok(rendered_image(image, position))
        })
        .collect::<OmniResult<Vec<_>>>()?;
    Ok((prompt_ids, images))
}

/// Resolves sampling and checks multimodal requirements against loaded model limits.
pub(super) fn prepare_generation_resources(
    processor: &crate::serving::InputProcessor,
    controls: &crate::serving::SamplingConfig,
    stop: &crate::serving::StopConfig,
    generation: &mut GenerationRequest,
) -> OmniResult<()> {
    generation.sampling = SamplingParams {
        temperature: DEFAULT_TEMPERATURE,
        top_p: DEFAULT_TOP_P,
        top_k: DEFAULT_TOP_K,
        seed: generation.sampling.seed,
        ..SamplingParams::default()
    };
    // Resolve the autoregressive budget only for requests whose behavior can
    // enter text decoding.
    let prompt_tokens = u32::try_from(generation.prompt_token_ids.len())
        .map_err(|_| "generation prompt exceeds the supported token count".to_string())?;
    generation.max_und_tokens = if generation.decodes_text() {
        crate::serving::text::resolve_max_tokens(
            controls.max_tokens,
            processor.config.sampling_defaults.max_output_tokens,
            Some(processor.config.max_model_tokens()),
            prompt_tokens,
        )
        .map_err(|error| error.to_string())? as usize
    } else {
        0
    };

    // Sampling choices determine logprob delivery and whether prefix-cache
    // reads remain semantically valid for this request.
    super::sampling::apply_sampling(
        &processor.tokenizer,
        controls,
        stop,
        &mut generation.sampling,
    )
    .map_err(|error| error.to_string())?;
    let prompt_logprobs_requested = generation.sampling.prompt_logprobs_requested();
    generation.cache.read &= !prompt_logprobs_requested;
    for image in &mut generation.multimodal_inputs.images {
        image.hash = isolated_cache_key(image.hash, generation.cache.isolation_key);
    }

    // Resolve conservative capacity before admission. Text output may shrink to
    // fit the model context, but non-text resource requirements remain fixed.
    generation
        .validate_resources(&processor.limits)
        .map_err(|error| error.to_string())?;
    let mut max_kv_tokens = generation
        .max_kv_tokens(&processor.limits)
        .map_err(|error| error.to_string())?;
    if max_kv_tokens > processor.config.max_model_tokens() as usize && generation.decodes_text() {
        let excess = max_kv_tokens - processor.config.max_model_tokens() as usize;
        generation.max_und_tokens = generation.max_und_tokens
            .checked_sub(excess)
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                format!(
                    "generation context requires at least {} KV tokens, exceeding the {}-token runtime limit",
                    max_kv_tokens.saturating_sub(generation.max_und_tokens),
                    processor.config.max_model_tokens()
                )
            })?;
        max_kv_tokens = generation
            .max_kv_tokens(&processor.limits)
            .map_err(|error| error.to_string())?;
    }

    if max_kv_tokens > processor.config.max_model_tokens() as usize {
        return Err(format!(
            "generation requires {} KV tokens, exceeding the {}-token runtime limit",
            max_kv_tokens,
            processor.config.max_model_tokens()
        )
        .into());
    }

    generation.validate().map_err(|error| error.to_string())?;
    Ok(())
}

/// Merges profile defaults with request image controls and validates generation geometry.
fn resolve_image_params(
    defaults: &crate::profile::omni::ImageGenerationDefaults,
    resolution_policy: &ResolutionPolicy,
    image: crate::serving::ImageGenControls,
    sampling_seed: Option<u64>,
    negative_prompt: String,
    constraint: GenerationConstraint,
) -> OmniResult<ImageParams> {
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
        seed: sampling_seed.or(defaults.seed),
        negative_prompt,
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
    image: crate::serving::ImageGenControls,
    sampling_seed: Option<u64>,
) -> OmniResult<ImageParams> {
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
        seed: sampling_seed.or(Some(0)),
        negative_prompt: String::new(),
        max_images,
        image_prompts: image.prompts,
        retain_images: image.retain_images.unwrap_or(true),
    })
}

/// Resolves each image's encoder inputs at its final prompt position.
fn prepare_image_inputs(
    num_prompt_tokens: usize,
    images: Vec<RenderedImage>,
    num_positions: u32,
    mut encoders_for_dimensions: impl FnMut(u32, u32, usize) -> OmniResult<Vec<ImageEncoderInput>>,
) -> OmniResult<MultimodalInputs> {
    let image_count = images.len();
    let mut images = images
        .into_iter()
        .map(|image| {
            if image.position as usize > num_prompt_tokens {
                return Err(OmniError::Invalid(
                    "image position exceeds the tokenized prompt".into(),
                ));
            }
            let (width, height) = image_dimensions(&image.b64)?;
            let encoders = encoders_for_dimensions(width, height, image_count)?;
            Ok(ImageInput {
                hash: image.hash,
                b64: image.b64,
                position: image.position,
                num_positions,
                encoders,
            })
        })
        .collect::<OmniResult<Vec<_>>>()?;
    images.sort_by_key(|image| image.position);
    Ok(MultimodalInputs { images })
}

/// Resolves the generation constraint implied by requested modalities.
pub(crate) fn generation_constraint(modalities: ModalitySelection) -> GenerationConstraint {
    match modalities {
        ModalitySelection::Text => GenerationConstraint::UndOnly,
        ModalitySelection::Image => GenerationConstraint::GenOnly,
        ModalitySelection::TextAndImage => GenerationConstraint::Default,
    }
}

/// Replaces chat image parts with unique template placeholders and retains their payloads.
fn replace_chat_images(
    request_id: &str,
    messages: &mut [ChatMessage],
) -> OmniResult<(Vec<String>, Vec<String>)> {
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
            images.push(data_image_payload(image_url)?);
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
fn rendered_image(b64: String, position: u32) -> RenderedImage {
    RenderedImage {
        hash: fnv1a(b64.as_bytes()),
        position,
        b64,
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
