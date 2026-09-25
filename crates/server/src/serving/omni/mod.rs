//! Multimodal prompt preprocessing for SenseNova and Bagel profiles.
//!
//! `InputProcessor::preprocess_generation` in `serving::model` calls
//! [`preprocess_sensenova`] or [`preprocess_bagel`] for the loaded profile.
//! Each fills the model computation inputs of the `GenerationRequest`
//! (`prompt_token_ids`, `negative_prompt_token_ids`, `multimodal_inputs`,
//! `image`, and `image_generation`) and returns the output processor policy.
//! [`prepare_generation_resources`] then resolves sampling, cache policy, and
//! the KV budget for both profiles.
//!
//! Chat images, and images attached to a SenseNova text prompt, are carried
//! through prompt rendering as request-scoped text placeholders. After the
//! chat template or prompt layout renders, each placeholder is replaced
//! (SenseNova) or removed (Bagel), and its byte offset in the cleaned text is
//! converted into a prompt-token position by tokenizing the text before it.
//! Bagel text prompts place images at token positions fixed by the prompt
//! layout instead (see `bagel_prompt`). Either way the position becomes
//! `ImageInput::position`, the prompt-token boundary at which the image's
//! encoder output enters the context.

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
    /// Wraps a free-form message as [`OmniError::Invalid`].
    ///
    /// Preprocessing helpers flatten profile errors with
    /// `.map_err(|error| error.to_string())?`, which relies on this conversion.
    fn from(message: String) -> Self {
        Self::Invalid(message)
    }
}

/// Denoising steps for Bagel context-image mode when the request sets none.
const DEFAULT_STEPS: u16 = 50;
// Omni sampler defaults applied by `prepare_generation_resources` before
// request controls. A zero temperature selects greedy decoding.
const DEFAULT_TEMPERATURE: f32 = 0.0;
const DEFAULT_TOP_P: f32 = 1.0;
const DEFAULT_TOP_K: u32 = 0;

/// Fixed image parameters for Bagel context-image mode.
///
/// `bagel_context_image_params` applies these without consulting request
/// controls.
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

/// Input image whose prompt position is resolved but whose encoder inputs are not.
#[derive(Debug, Clone)]
struct RenderedImage {
    /// FNV-1a hash of `b64`. SenseNova preprocessing mixes in the image's
    /// pixel bound, and `prepare_generation_resources` later mixes in the
    /// request's cache isolation key when it has one.
    hash: u64,
    /// Prompt-token position with the meaning of `ImageInput::position`.
    position: u32,
    /// Base64 image payload, without any data-URL header.
    b64: String,
}

/// Fills the SenseNova prompt, image inputs, and image-generation parameters.
///
/// On success the request's prompt, negative prompt, multimodal inputs, image
/// parameters, and image-generation policy are replaced; on error `generation`
/// is left unmodified. The returned policy selects the SenseNova reasoning and
/// visible-wrapper output filter.
///
/// # Errors
///
/// Fails, among other conditions, when the processor has no chat template
/// (checked for text prompts too), rendering or tokenization fails, rendering
/// loses or reorders an image slot, the prompt is empty, image controls are
/// invalid, an image payload cannot be decoded, or the profile cannot size the
/// image encoders or the requested output resolution.
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
            .ok_or(crate::serving::chat::Error::MissingChatTemplate)?,
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
    let mut multimodal_inputs = prepare_image_inputs(
        prompt_token_ids.len(),
        images,
        profile.image_num_positions,
        |width, height, count| {
            profile
                .image_encoders_for_dimensions(width, height, count)
                .map_err(OmniError::from)
        },
    )?;
    // An input image's encoder product depends on its pixel bound, which the
    // request's input image count sets, so the bound joins the image's
    // encoder-cache identity.
    let max_pixels = SenseNovaProfile::input_image_max_pixels(multimodal_inputs.images.len());
    for image in &mut multimodal_inputs.images {
        image.hash = combine_cache_keys(image.hash, max_pixels);
    }
    let policy = profile
        .image_generation_for_dimensions(image.width, image.height)
        .map_err(|error| error.to_string())?;

    // Every fallible step is complete; publish the prepared inputs together.
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
///
/// A plain-text understanding-only request with input images uses context-image
/// mode (see `bagel_prompt`): its negative prompt is a copy of the positive
/// prompt and its image parameters come from `bagel_context_image_params`
/// instead of `resolve_image_params`. Other requests resolve the negative
/// prompt and image parameters as [`preprocess_sensenova`] does. On error
/// `generation` is left unmodified. The returned policy is always
/// `OutputProcessorPolicy::None`, so Bagel text is emitted without output
/// filtering.
///
/// # Errors
///
/// Fails under the same kinds of conditions as [`preprocess_sensenova`].
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
            .ok_or(crate::serving::chat::Error::MissingChatTemplate)?,
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
///
/// A text prompt with images gets one placeholder line per image prepended
/// (see `prompt_with_image_slots`); chat images stay where their content parts
/// appear.
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
///
/// The returned flag is `true` only for a text prompt under
/// `GenerationConstraint::UndOnly` with input images. That layout is
/// `wrap_context_text(CONTEXT_SYSTEM_PROMPT)`, then every image, then
/// `wrap_context_text(prompt)`. Other text prompts place every image after the
/// whole rendered prompt; chat prompts place each image where its content
/// part appears.
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
///
/// Inserts `SenseNovaProfile::default_system_prompt` when the request has no
/// system message and the constraint defines one. A non-empty assistant
/// prefix for the constraint is appended as a final assistant message that
/// the template continues instead of opening a new assistant turn.
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
///
/// Each placeholder becomes the profile's start-of-image and end-of-image
/// marker text. An image's position is the index of its end-of-image token,
/// so the image fills the gap between the two markers. The position is
/// found by tokenizing the text up to the end of the marker and requiring its
/// last token to be `controls.end_of_image`, which fails if the marker text
/// does not encode to that token.
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
///
/// An image's position is the token count of the cleaned text before its
/// slot. This assumes the prefix encodes to a prefix of the full prompt's
/// tokens; unlike the SenseNova path, no marker token checks it.
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
///
/// Runs after [`preprocess_sensenova`] or [`preprocess_bagel`] has filled the
/// prompt and image inputs. It replaces `generation.sampling`, sets
/// `max_und_tokens`, may clear `cache.read`, and rewrites each input image's
/// `hash` into its cache-isolated form, so it must run exactly once per
/// request. On error `generation` may be partially updated; the caller
/// discards it.
///
/// # Errors
///
/// Fails when the prompt length does not fit in `u32`, the prompt fills the
/// model context while the request decodes text, sampling or stop controls
/// are invalid, `GenerationRequest::validate_resources` rejects the request
/// against the loaded limits, the KV requirement exceeds
/// `ModelConfig::max_model_tokens` even after shrinking the text budget, or
/// the final request fails `GenerationRequest::validate`.
pub(super) fn prepare_generation_resources(
    processor: &crate::serving::InputProcessor,
    controls: &crate::serving::SamplingConfig,
    stop: &crate::serving::StopConfig,
    generation: &mut GenerationRequest,
) -> OmniResult<()> {
    // Omni profiles start from these fixed sampler defaults rather than the
    // sampler values in the checkpoint's `sampling_defaults`; only its
    // `max_output_tokens` applies, below. The seed already resolved by
    // `preprocess_generation` (image seed first, then text seed) is kept.
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
    // Encoder-cache identity folds in the request's cache isolation key, so
    // the same image in different cache partitions maps to different entries.
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
    // Shrink the text budget by the excess once; at least one text token must
    // remain. The check after this block covers requests that do not decode
    // text and so have no budget to shrink.
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
///
/// Request fields override profile defaults field by field. The sampling seed
/// (already resolved from the image and text seeds) overrides the profile
/// seed. Generated images are retained by default unless the request is
/// image-only (`GenerationConstraint::GenOnly`).
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
        cfg_renorm_type: image.cfg_renorm_type.unwrap_or(defaults.cfg_renorm_type),
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
///
/// Guidance, renormalization, and resolution are fixed (see
/// `context_image_defaults`) and cannot be overridden by the request; of the
/// image controls only `steps`, `max_images`, `timestep_shift`, `prompts`, and
/// `retain_images` are honored. The seed is the sampling seed, which already
/// includes any image seed, or zero when neither is set.
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
    // The fixed canvas still passes the profile's per-axis latent bound.
    let resolution = resolve_resolution(
        &profile.resolution_policy,
        None,
        Some(context_image_defaults::RESOLUTION),
        Some(context_image_defaults::RESOLUTION),
    )
    .map_err(|error| error.to_string())?;
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
        height: resolution.height,
        width: resolution.width,
        seed: sampling_seed.or(Some(0)),
        negative_prompt: String::new(),
        max_images,
        image_prompts: image.prompts,
        retain_images: image.retain_images.unwrap_or(true),
    })
}

/// Resolves each image's encoder inputs at its final prompt position.
///
/// `encoders_for_dimensions` receives the decoded image width and height in
/// pixels and the request's total image count. The output is ordered by
/// prompt position.
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
    // The sort is stable, so images sharing a position keep their input order
    // as `ImageInput::position` requires.
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
///
/// Returns the base64 payloads and their placeholders in message order.
/// Assistant messages are skipped, so images in assistant content are not
/// collected.
///
/// # Errors
///
/// Fails when an image part is not a `data:image/*;base64` URL with valid
/// base64 content (see `data_image_payload`).
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
///
/// The placeholder embeds a hash of the request identifier and the slot index.
/// `replace_rendered_slots` rejects a rendered prompt in which any placeholder
/// does not occur exactly once, including when request text repeats it.
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
///
/// Placeholders must appear in `placeholders` order. Each returned offset
/// indexes the cleaned text just past that placeholder's `replacement`.
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

/// Validates an inline image data URL and returns its base64 payload.
///
/// The payload is decoded only to validate it; the still-encoded text is
/// returned.
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

/// Pairs an image payload with its prompt position and content hash.
fn rendered_image(b64: String, position: u32) -> RenderedImage {
    RenderedImage {
        hash: fnv1a(b64.as_bytes()),
        position,
        b64,
    }
}

/// Prepends the image slots to a SenseNova text prompt, one per line.
///
/// A single slot is bare; multiple slots are labeled `Image-1:`, `Image-2:`,
/// and so on.
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

/// Returns the `(width, height)` in pixels of a base64-encoded image.
///
/// Only the image header is read; the pixels are not decoded.
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

/// Rejects an empty prompt token sequence.
fn validate_prompt(prompt_ids: &[u32]) -> OmniResult<()> {
    if prompt_ids.is_empty() {
        Err(OmniError::Invalid(
            "generation requires a non-empty prompt".to_string(),
        ))
    } else {
        Ok(())
    }
}

/// Requires `1 <= value <= limit` for `image.max_images`.
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

/// Requires a finite `(lo, hi)` guidance interval with `lo <= hi`.
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
///
/// Returns `content_key` unchanged when the request has no isolation key, so
/// unisolated requests share encoder-cache entries for identical image
/// payloads.
fn isolated_cache_key(content_key: u64, isolation_key: Option<u64>) -> u64 {
    isolation_key.map_or(content_key, |isolation_key| {
        combine_cache_keys(content_key, isolation_key)
    })
}

/// Mixes a value that changes an encoder product into its cache key.
fn combine_cache_keys(key: u64, value: u64) -> u64 {
    let mut bytes = [0_u8; 16];
    bytes[..8].copy_from_slice(&key.to_le_bytes());
    bytes[8..].copy_from_slice(&value.to_le_bytes());
    fnv1a(&bytes)
}

/// Computes a 64-bit FNV-1a hash that is stable across processes and builds.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
