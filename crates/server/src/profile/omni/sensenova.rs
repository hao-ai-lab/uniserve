//! SenseNova model parameters, prompt layout, and image configuration.
//!
//! Input images and generated-image feedback both use the ViT encode step,
//! whose grid is 32 pixels per token (the default patch size 16 at downsample
//! ratio 0.5 in `uniserve_models/sensenova_u1`). Output canvases are limited
//! to the fixed resolution buckets in `resolution_policy`.

use serde::{Deserialize, Serialize};
use uniserve_core::{
    FeedbackNextToken, FeedbackSource, GenerationConstraint, GenerationFeatures, GenerationLimits,
    ImageEncoderInput, ImageGenerationConfig, ImageIngestStep, ImageTrigger, ModelDtype,
};

use super::resolution::{ResolutionBucket, ResolutionName, ResolutionPolicy};
use super::{
    DelimitedTextPolicy, GenerationControls, ImageGenerationDefaults, OutputFilterPolicy, chatml,
    encode, model_dtype_bytes, pixel_bound_tokens, required_token, required_token_id,
};
use crate::profile::assets;
use crate::profile::tokenizer::HuggingFaceTokenizer;

/// System instruction for `GenerationConstraint::Default`.
const DEFAULT_SYSTEM_PROMPT: &str = "You are a multimodal assistant capable of reasoning with both text and images. You support two modes:\n\nThink Mode: When reasoning is needed, you MUST start with a <think></think> block and place all reasoning inside it. You MUST interleave text with generated images using tags like <image1>, <image2>. Images can ONLY be generated between <think> and </think>, and may be referenced in the final answer.\n\nNon-Think Mode: When no reasoning is needed, directly provide the answer without reasoning. Do not use tags like <image1>, <image2>; present any images naturally alongside the text.\n\nAfter the think block, always provide a concise, user-facing final answer. The answer may include text, images, or both. Match the user's language in both reasoning and the final answer.";
/// System instruction for `GenerationConstraint::GenOnly` and for the
/// classifier-free guidance context.
const IMAGE_SYSTEM_PROMPT: &str = "You are an image generation and editing assistant that accurately understands and executes user intent.\n\nYou support two modes:\n\n1. Think Mode:\nIf the task requires reasoning, you MUST start with a <think></think> block. Put all reasoning inside the block using plain text. DO NOT include any image tags. Keep it reasonable and directly useful for producing the final image.\n\n2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final image.\n\nTask Types:\n\nA. Text-to-Image Generation:\n- Generate a high-quality image based on the user's description.\n- Ensure visual clarity, semantic consistency, and completeness.\n- DO NOT introduce elements that contradict or override the user's intent.\n\nB. Image Editing:\n- Use the provided image(s) as input or reference for modification or transformation.\n- The result can be an edited image or a new image based on the reference(s).\n- Preserve all unspecified attributes unless explicitly changed.\n\nGeneral Rules:\n- For any visible text in the image, follow the language specified for the rendered text in the user's description, not the language of the prompt. If no language is specified, use the user's input language.";
/// Assistant prefix for `GenOnly`: ends the prompt with an empty reasoning
/// block and the start-of-image token.
const IMAGE_ASSISTANT_PREFIX: &str = "<think>\n\n</think>\n\n<img>";
/// Assistant prefix of the classifier-free guidance context.
const NEGATIVE_ASSISTANT_PREFIX: &str = "<img>";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Loaded SenseNova profile and tokenizer assets.
pub struct SenseNovaProfile {
    /// Special tokens that delimit multimodal generation.
    pub controls: GenerationControls,
    /// Default image-generation parameters.
    pub image_defaults: ImageGenerationDefaults,
    /// Supported image dimensions and default resolution.
    pub resolution_policy: ResolutionPolicy,
    /// Filters applied to decoded assistant output.
    pub output_filter: OutputFilterPolicy,
    /// Ordered encoders and their loaded KV requirements for input images.
    pub image_encoders: Vec<ImageEncoderInput>,
    /// Logical positions contributed by one input image, independently of KV tokens.
    pub image_num_positions: u32,
    /// Image trigger and feedback encoder requirements.
    pub image_generation: ImageGenerationConfig,
}

impl SenseNovaProfile {
    /// Stable profile identifier.
    pub const ID: &'static str = "sensenova";
    /// Pixel-to-latent downsampling factor.
    pub const LATENT_DOWNSAMPLE: u32 = 32;
    /// Maximum reusable encoder products tracked by the profile.
    pub const ENCODER_CACHE_ENTRIES: usize = 256;

    /// Returns bounded runtime capabilities for the selected dtype.
    ///
    /// These are profile ceilings. The engine scheduler's
    /// `resolve_generation_limits` clamps the latent, grid-token, feature-byte,
    /// and encoder-cache limits to the capacities the loaded worker reports and
    /// drops features whose calls the worker does not support.
    pub fn runtime_limits(model_dtype: ModelDtype) -> GenerationLimits {
        let dtype_bytes = model_dtype_bytes(model_dtype);
        // `GenerationRequest::image_latent_bytes` divides the latent feature
        // bytes by `max_vae_grid_tokens` to get the bytes per latent unit:
        // a 3-channel 32x32 pixel patch.
        GenerationLimits {
            features: GenerationFeatures::UNDERSTANDING
                | GenerationFeatures::VISION_ENCODE
                | GenerationFeatures::IMAGE_GENERATION,
            max_latent_units: 4_096,
            latent_downsample: Self::LATENT_DOWNSAMPLE,
            max_vae_grid_tokens: 4_096,
            max_vit_grid_tokens: 4_900,
            max_latent_feature_bytes: 4_096 * 3 * 32 * 32 * dtype_bytes,
            max_vision_feature_bytes: 4_900 * 4_096 * dtype_bytes,
            commit_marker_tokens: 0,
            max_cfg_branches: 3,
            encoder_cache_entries: Self::ENCODER_CACHE_ENTRIES as u32,
        }
    }

    /// Resolves required control tokens from the loaded tokenizer.
    ///
    /// Encoder token counts stay unset here; the `*_for_dimensions` methods
    /// return per-image copies with the counts set. Fails when the tokenizer
    /// lacks any of the ChatML or image delimiter tokens.
    pub fn resolve(tokenizer: &HuggingFaceTokenizer) -> assets::Result<Self> {
        let (start_of_image, start_of_image_text) =
            required_token(tokenizer, "<img>", "SenseNova start-of-image")?;
        let (end_of_image, end_of_image_text) =
            required_token(tokenizer, "</img>", "SenseNova end-of-image")?;
        let controls = GenerationControls {
            bos: required_token_id(tokenizer, "<|im_start|>", "SenseNova beginning-of-sequence")?,
            eos: required_token_id(tokenizer, "<|im_end|>", "SenseNova end-of-sequence")?,
            start_of_image,
            end_of_image,
            start_of_image_text,
            end_of_image_text,
        };
        let image_encoders = vec![ImageEncoderInput {
            encoder: ImageIngestStep::VitEncode,
            num_kv_tokens: None,
            max_kv_tokens: None,
        }];
        let image_num_positions = 1;
        let image_generation = ImageGenerationConfig {
            trigger: ImageTrigger::Token {
                token_id: controls.start_of_image,
            },
            requires_text_for_image: false,
            feedback_source: Some(FeedbackSource::DeviceProduct),
            feedback_next_token: FeedbackNextToken::Token {
                token_id: controls.end_of_image,
            },
            num_feedback_positions: 2,
            feedback_encoders: vec![ImageEncoderInput {
                encoder: ImageIngestStep::VitEncode,
                num_kv_tokens: None,
                max_kv_tokens: None,
            }],
            sample_feedback_continuation: true,
        };
        let resolution_policy = resolution_policy();
        Ok(Self {
            controls,
            image_defaults: ImageGenerationDefaults {
                resolution: ResolutionName::Landscape16x9,
                steps: 50,
                cfg_text_scale: 4.0,
                cfg_img_scale: 1.0,
                cfg_renorm_type: uniserve_core::CfgRenorm::None,
                cfg_renorm_min: 0.0,
                cfg_interval: (0.0, 1.0),
                timestep_shift: 3.0,
                seed: Some(42),
                max_images: 4,
                max_images_limit: 10,
            },
            resolution_policy,
            output_filter: OutputFilterPolicy {
                reasoning: Some(DelimitedTextPolicy {
                    start: "<think>".to_string(),
                    end: "</think>".to_string(),
                }),
                visible_wrappers: vec![DelimitedTextPolicy {
                    start: "<answer>".to_string(),
                    end: "</answer>".to_string(),
                }],
            },
            image_encoders,
            image_num_positions,
            image_generation,
        })
    }

    /// Returns the default system instruction for a generation constraint.
    pub fn default_system_prompt(constraint: GenerationConstraint) -> Option<&'static str> {
        match constraint {
            GenerationConstraint::Default => Some(DEFAULT_SYSTEM_PROMPT),
            GenerationConstraint::GenOnly => Some(IMAGE_SYSTEM_PROMPT),
            GenerationConstraint::UndOnly => None,
        }
    }

    /// Returns the assistant prefix required by a generation constraint.
    pub fn assistant_prefix(constraint: GenerationConstraint) -> &'static str {
        match constraint {
            GenerationConstraint::GenOnly => IMAGE_ASSISTANT_PREFIX,
            GenerationConstraint::Default | GenerationConstraint::UndOnly => "",
        }
    }

    /// Renders and encodes the positive prompt for one request.
    pub fn render_prompt_ids(
        &self,
        tokenizer: &HuggingFaceTokenizer,
        constraint: GenerationConstraint,
        prompt: &str,
        system_prompt: Option<&str>,
        assistant_prefix: Option<&str>,
    ) -> crate::profile::tokenizer::Result<Vec<u32>> {
        encode(
            tokenizer,
            &self.render_prompt_text(constraint, prompt, system_prompt, assistant_prefix),
        )
    }

    /// Renders positive prompt text before tokenization.
    ///
    /// Explicit `system_prompt` and `assistant_prefix` values replace the
    /// constraint's defaults; `UndOnly` has no default system instruction.
    pub fn render_prompt_text(
        &self,
        constraint: GenerationConstraint,
        prompt: &str,
        system_prompt: Option<&str>,
        assistant_prefix: Option<&str>,
    ) -> String {
        let system = system_prompt.or_else(|| Self::default_system_prompt(constraint));
        let assistant = assistant_prefix.unwrap_or_else(|| Self::assistant_prefix(constraint));
        chatml(system, prompt, assistant)
    }

    /// Renders and encodes classifier-free guidance context.
    ///
    /// Unlike Bagel, an empty prompt still renders the framed ChatML turn with
    /// the image system instruction.
    pub fn render_negative_prompt_ids(
        &self,
        tokenizer: &HuggingFaceTokenizer,
        prompt: &str,
    ) -> crate::profile::tokenizer::Result<Vec<u32>> {
        encode(
            tokenizer,
            &chatml(Some(IMAGE_SYSTEM_PROMPT), prompt, NEGATIVE_ASSISTANT_PREFIX),
        )
    }

    /// Builds the encoder inputs for an input image of the given dimensions.
    ///
    /// The image is fitted to a 32-aligned grid with an area bounded by
    /// 512x512 and 2048x2048 pixels and adds no marker tokens. Fails for a
    /// zero dimension or an aspect ratio above 200. The image count does not
    /// affect the result.
    pub fn image_encoders_for_dimensions(
        &self,
        width: u32,
        height: u32,
        _image_count: usize,
    ) -> assets::Result<Vec<ImageEncoderInput>> {
        let tokens = pixel_bound_tokens(width, height, 32, 262_144, 4_194_304, 32, 0)?;
        Ok(vec![ImageEncoderInput {
            encoder: ImageIngestStep::VitEncode,
            num_kv_tokens: Some(tokens),
            max_kv_tokens: None,
        }])
    }

    /// Resolves the generated-image feedback KV contribution for the requested canvas.
    ///
    /// Uses the same grid as an input image plus one marker token. Fails for
    /// a zero dimension or an aspect ratio above 200, or when the profile's
    /// feedback configuration has no feedback source or is not a single ViT
    /// encoder.
    pub fn image_generation_for_dimensions(
        &self,
        width: u32,
        height: u32,
    ) -> assets::Result<ImageGenerationConfig> {
        let tokens = pixel_bound_tokens(width, height, 32, 262_144, 4_194_304, 32, 1)?;
        let mut policy = self.image_generation.clone();
        if policy.feedback_source.is_none()
            || policy.feedback_encoders.len() != 1
            || policy.feedback_encoders[0].encoder != ImageIngestStep::VitEncode
        {
            return Err(assets::Error::invalid(
                "image feedback encoders do not match the model processor",
            ));
        }
        policy.feedback_encoders[0].num_kv_tokens = Some(tokens);
        policy.feedback_encoders[0].max_kv_tokens = None;
        Ok(policy)
    }
}

/// Returns the fixed SenseNova image-resolution bucket policy.
///
/// Only these exact dimensions are accepted (`allow_custom` is false).
fn resolution_policy() -> ResolutionPolicy {
    let buckets = [
        (ResolutionName::Square, 1536, 1536),
        (ResolutionName::Landscape16x9, 2048, 1152),
        (ResolutionName::OnePointFiveK, 2048, 1152),
        (ResolutionName::Portrait9x16, 1152, 2048),
        (ResolutionName::Landscape3x2, 1888, 1248),
        (ResolutionName::Portrait2x3, 1248, 1888),
        (ResolutionName::Landscape4x3, 1760, 1312),
        (ResolutionName::Portrait3x4, 1312, 1760),
        (ResolutionName::Portrait1x2, 1088, 2144),
        (ResolutionName::Landscape2x1, 2144, 1088),
        (ResolutionName::Portrait1x3, 864, 2592),
        (ResolutionName::Landscape3x1, 2592, 864),
    ]
    .into_iter()
    .map(|(name, width, height)| ResolutionBucket {
        name,
        width,
        height,
    })
    .collect::<Vec<_>>();
    // Index 1 is the 16:9 bucket, matching `image_defaults.resolution` in
    // `SenseNovaProfile::resolve`; reordering the list changes the default.
    ResolutionPolicy {
        default: buckets[1].clone(),
        buckets,
        allow_custom: false,
    }
}
