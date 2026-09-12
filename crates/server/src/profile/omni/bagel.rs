//! Bagel model parameters, prompt layout, and image configuration.

use serde::{Deserialize, Serialize};
use uniserve_core::{
    FeedbackNextToken, FeedbackSource, GenerationConstraint, GenerationFeatures, GenerationLimits,
    ImageEncoderInput, ImageGenerationConfig, ImageIngestStep, ImageTrigger, ModelDtype,
};

use super::resolution::{ResolutionBucket, ResolutionName, ResolutionPolicy};
use super::{
    GenerationControls, ImageGenerationDefaults, OutputFilterPolicy, encode, model_dtype_bytes,
    required_token, required_token_id, stride_resize_tokens,
};
use crate::profile::assets;
use crate::profile::tokenizer::HuggingFaceTokenizer;

const DEFAULT_SYSTEM_PROMPT: &str = "You should first think about the planning process in the mind and then generate the image. \n     The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here";
/// System instruction that establishes Bagel reasoning and answer delimiters.
pub const CONTEXT_SYSTEM_PROMPT: &str = "\nLet's think step by step to answer the question. For text-based thinking, enclose the process within <think> </think>, e.g. <think> thinking process here </think>. For visual thinking, enclose the content within <image_start> </image_end>, e.g. <image_start> thinking image here </image_end>. Finally conclude with the final answer wrapped in <answer></answer> tags, i.e.<answer> answer here </answer>.\n";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Loaded Bagel profile and tokenizer assets.
pub struct BagelProfile {
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

impl BagelProfile {
    /// Stable profile identifier.
    pub const ID: &'static str = "bagel";
    /// Pixel-to-latent downsampling factor.
    pub const LATENT_DOWNSAMPLE: u32 = 16;
    /// Maximum reusable encoder products tracked by the profile.
    pub const ENCODER_CACHE_ENTRIES: usize = 256;

    /// Returns bounded runtime capabilities for the selected dtype.
    pub fn runtime_limits(model_dtype: ModelDtype) -> GenerationLimits {
        let dtype_bytes = model_dtype_bytes(model_dtype);
        GenerationLimits {
            features: GenerationFeatures::UNDERSTANDING
                | GenerationFeatures::VISION_ENCODE
                | GenerationFeatures::LATENT_ENCODE
                | GenerationFeatures::IMAGE_GENERATION,
            max_latent_units: 1_024,
            latent_downsample: Self::LATENT_DOWNSAMPLE,
            max_vae_grid_tokens: 1_026,
            max_vit_grid_tokens: 4_902,
            max_latent_feature_bytes: 1_026 * 16 * 2 * 2 * dtype_bytes,
            max_vision_feature_bytes: 4_902 * 3_584 * dtype_bytes,
            commit_marker_tokens: 2,
            max_cfg_branches: 3,
            encoder_cache_entries: Self::ENCODER_CACHE_ENTRIES as u32,
        }
    }

    /// Resolves required control tokens from the loaded tokenizer.
    pub fn resolve(tokenizer: &HuggingFaceTokenizer) -> assets::Result<Self> {
        let (start_of_image, start_of_image_text) =
            required_token(tokenizer, "<|vision_start|>", "Bagel start-of-image")?;
        let (end_of_image, end_of_image_text) =
            required_token(tokenizer, "<|vision_end|>", "Bagel end-of-image")?;
        let controls = GenerationControls {
            bos: required_token_id(tokenizer, "<|im_start|>", "Bagel beginning-of-sequence")?,
            eos: required_token_id(tokenizer, "<|im_end|>", "Bagel end-of-sequence")?,
            start_of_image,
            end_of_image,
            start_of_image_text,
            end_of_image_text,
        };
        let image_encoders = vec![
            ImageEncoderInput {
                encoder: ImageIngestStep::VaeEncode,
                num_kv_tokens: None,
                max_kv_tokens: None,
            },
            ImageEncoderInput {
                encoder: ImageIngestStep::VitEncode,
                num_kv_tokens: None,
                max_kv_tokens: None,
            },
        ];
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
                encoder: ImageIngestStep::VaeEncode,
                num_kv_tokens: None,
                max_kv_tokens: None,
            }],
            sample_feedback_continuation: false,
        };
        let default_resolution = ResolutionBucket {
            name: ResolutionName::Square,
            width: 512,
            height: 512,
        };
        Ok(Self {
            controls,
            image_defaults: ImageGenerationDefaults {
                resolution: ResolutionName::Square,
                steps: 50,
                cfg_text_scale: 4.0,
                cfg_img_scale: 1.0,
                cfg_renorm_type: uniserve_core::CfgRenorm::Global,
                cfg_renorm_min: 0.0,
                cfg_interval: (0.0, 1.0),
                timestep_shift: 1.0,
                seed: None,
                max_images: 1,
                max_images_limit: 16,
            },
            resolution_policy: ResolutionPolicy {
                default: default_resolution.clone(),
                buckets: vec![default_resolution],
                allow_custom: true,
            },
            output_filter: OutputFilterPolicy {
                reasoning: None,
                visible_wrappers: Vec::new(),
            },
            image_encoders,
            image_num_positions,
            image_generation,
        })
    }

    /// Returns the default system instruction for a generation constraint.
    pub fn default_system_prompt(constraint: GenerationConstraint) -> Option<&'static str> {
        matches!(constraint, GenerationConstraint::Default).then_some(DEFAULT_SYSTEM_PROMPT)
    }

    /// Renders and encodes the positive prompt for one request.
    pub fn render_prompt_ids(
        &self,
        tokenizer: &HuggingFaceTokenizer,
        constraint: GenerationConstraint,
        has_images: bool,
        prompt: &str,
        system_prompt: Option<&str>,
        assistant_prefix: Option<&str>,
    ) -> crate::profile::tokenizer::Result<Vec<u32>> {
        match (constraint, has_images) {
            (GenerationConstraint::Default, _) => encode(
                tokenizer,
                &format!(
                    "<|im_start|>{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}",
                    system_prompt.unwrap_or(DEFAULT_SYSTEM_PROMPT),
                    prompt,
                    assistant_prefix.unwrap_or("")
                ),
            ),
            (GenerationConstraint::UndOnly, false) => encode(
                tokenizer,
                &format!(
                    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}",
                    prompt,
                    assistant_prefix.unwrap_or("")
                ),
            ),
            (GenerationConstraint::UndOnly, true) => encode(tokenizer, prompt),
            (GenerationConstraint::GenOnly, _) => self.wrap_context_text(tokenizer, prompt),
        }
    }

    /// Renders and encodes classifier-free guidance context.
    pub fn render_negative_prompt_ids(
        &self,
        tokenizer: &HuggingFaceTokenizer,
        prompt: &str,
    ) -> crate::profile::tokenizer::Result<Vec<u32>> {
        if prompt.is_empty() {
            return Ok(Vec::new());
        }
        self.wrap_context_text(tokenizer, prompt)
    }

    /// Wraps plain context text in the profile's required delimiters.
    pub fn wrap_context_text(
        &self,
        tokenizer: &HuggingFaceTokenizer,
        text: &str,
    ) -> crate::profile::tokenizer::Result<Vec<u32>> {
        let mut ids = vec![self.controls.bos];
        ids.extend(encode(tokenizer, text)?);
        ids.push(self.controls.eos);
        Ok(ids)
    }

    /// Builds the encoder inputs for an input image of the given dimensions.
    pub fn image_encoders_for_dimensions(
        &self,
        width: u32,
        height: u32,
        _image_count: usize,
    ) -> assets::Result<Vec<ImageEncoderInput>> {
        let vae_tokens = stride_resize_tokens(width, height, &[(1024, 512, 16, 1_806_336)], 16, 2)?;
        let vit_tokens = stride_resize_tokens(
            width,
            height,
            &[(1024, 512, 16, 1_806_336), (980, 224, 14, 1_806_336)],
            14,
            2,
        )?;
        Ok(vec![
            ImageEncoderInput {
                encoder: ImageIngestStep::VaeEncode,
                num_kv_tokens: Some(vae_tokens),
                max_kv_tokens: None,
            },
            ImageEncoderInput {
                encoder: ImageIngestStep::VitEncode,
                num_kv_tokens: Some(vit_tokens),
                max_kv_tokens: None,
            },
        ])
    }

    /// Resolves the generated-image feedback KV contribution for the requested canvas.
    pub fn image_generation_for_dimensions(
        &self,
        width: u32,
        height: u32,
    ) -> assets::Result<ImageGenerationConfig> {
        let tokens = stride_resize_tokens(width, height, &[(1024, 512, 16, 1_806_336)], 16, 2)?;
        let mut policy = self.image_generation.clone();
        if policy.feedback_source.is_none()
            || policy.feedback_encoders.len() != 1
            || policy.feedback_encoders[0].encoder != ImageIngestStep::VaeEncode
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
