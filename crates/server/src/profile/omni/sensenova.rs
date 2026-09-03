use serde::{Deserialize, Serialize};
use uniserve_core::{
    FeedbackNextToken, FeedbackSource, GenOnlyStartPolicyDescriptor, GeneratedImageFeedbackRecipe,
    GenerationConstraint, GenerationFeatures, GenerationLimits, GenerationPolicyDescriptor,
    ImageIngestRecipe, ImageIngestStep, ImageKvEffect, Modality, ModelDtype,
    TriggerPolicyDescriptor,
};

use super::resolution::{ResolutionBucket, ResolutionName, ResolutionPolicy};
use super::{
    DelimitedTextPolicy, GenerationControls, ImageGenerationDefaults, OutputFilterPolicy, chatml,
    encode, model_dtype_bytes, pixel_bound_tokens, required_token, required_token_id,
};
use crate::profile::assets;
use crate::profile::tokenizer::HuggingFaceTokenizer;

const DEFAULT_SYSTEM_PROMPT: &str = "You are a multimodal assistant capable of reasoning with both text and images. You support two modes:\n\nThink Mode: When reasoning is needed, you MUST start with a <think></think> block and place all reasoning inside it. You MUST interleave text with generated images using tags like <image1>, <image2>. Images can ONLY be generated between <think> and </think>, and may be referenced in the final answer.\n\nNon-Think Mode: When no reasoning is needed, directly provide the answer without reasoning. Do not use tags like <image1>, <image2>; present any images naturally alongside the text.\n\nAfter the think block, always provide a concise, user-facing final answer. The answer may include text, images, or both. Match the user's language in both reasoning and the final answer.";
const IMAGE_SYSTEM_PROMPT: &str = "You are an image generation and editing assistant that accurately understands and executes user intent.\n\nYou support two modes:\n\n1. Think Mode:\nIf the task requires reasoning, you MUST start with a <think></think> block. Put all reasoning inside the block using plain text. DO NOT include any image tags. Keep it reasonable and directly useful for producing the final image.\n\n2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final image.\n\nTask Types:\n\nA. Text-to-Image Generation:\n- Generate a high-quality image based on the user's description.\n- Ensure visual clarity, semantic consistency, and completeness.\n- DO NOT introduce elements that contradict or override the user's intent.\n\nB. Image Editing:\n- Use the provided image(s) as input or reference for modification or transformation.\n- The result can be an edited image or a new image based on the reference(s).\n- Preserve all unspecified attributes unless explicitly changed.\n\nGeneral Rules:\n- For any visible text in the image, follow the language specified for the rendered text in the user's description, not the language of the prompt. If no language is specified, use the user's input language.";
const IMAGE_ASSISTANT_PREFIX: &str = "<think>\n\n</think>\n\n<img>";
const NEGATIVE_ASSISTANT_PREFIX: &str = "<img>";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SenseNovaProfile {
    pub controls: GenerationControls,
    pub image_defaults: ImageGenerationDefaults,
    pub resolution_policy: ResolutionPolicy,
    pub output_filter: OutputFilterPolicy,
    pub image_ingest: ImageIngestRecipe,
    pub generation_policy: GenerationPolicyDescriptor,
}

impl SenseNovaProfile {
    pub const ID: &'static str = "sensenova";
    pub const LATENT_DOWNSAMPLE: u32 = 32;
    pub const ENCODER_CACHE_ENTRIES: usize = 256;

    pub fn runtime_limits(model_dtype: ModelDtype) -> GenerationLimits {
        let dtype_bytes = model_dtype_bytes(model_dtype);
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
        let image_ingest = ImageIngestRecipe {
            steps: vec![ImageIngestStep::VitEncode],
            logical_positions: 1,
            step_kv_tokens: vec![ImageKvEffect::WorkerDefined],
            modality: Modality::Und,
        };
        let generation_policy = GenerationPolicyDescriptor {
            trigger: TriggerPolicyDescriptor::Token {
                token_id: controls.start_of_image,
            },
            gen_only_start: GenOnlyStartPolicyDescriptor::Immediate,
            feedback: Some(GeneratedImageFeedbackRecipe {
                source: FeedbackSource::DeviceProduct,
                next_und_token: FeedbackNextToken::Token {
                    token_id: controls.end_of_image,
                },
                ingest: ImageIngestRecipe {
                    steps: vec![ImageIngestStep::VitEncode],
                    logical_positions: 2,
                    step_kv_tokens: vec![ImageKvEffect::WorkerDefined],
                    modality: Modality::Und,
                },
                sample_continuation: true,
            }),
            ..GenerationPolicyDescriptor::default()
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
            image_ingest,
            generation_policy,
        })
    }

    pub fn default_system_prompt(constraint: GenerationConstraint) -> Option<&'static str> {
        match constraint {
            GenerationConstraint::Default => Some(DEFAULT_SYSTEM_PROMPT),
            GenerationConstraint::GenOnly => Some(IMAGE_SYSTEM_PROMPT),
            GenerationConstraint::UndOnly => None,
        }
    }

    pub fn assistant_prefix(constraint: GenerationConstraint) -> &'static str {
        match constraint {
            GenerationConstraint::GenOnly => IMAGE_ASSISTANT_PREFIX,
            GenerationConstraint::Default | GenerationConstraint::UndOnly => "",
        }
    }

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

    pub fn image_ingest_for_dimensions(
        &self,
        width: u32,
        height: u32,
        _image_count: usize,
    ) -> assets::Result<ImageIngestRecipe> {
        let tokens = pixel_bound_tokens(width, height, 32, 262_144, 4_194_304, 32, 0)?;
        let mut ingest = self.image_ingest.clone();
        ingest.step_kv_tokens[0] = ImageKvEffect::Exact { tokens };
        Ok(ingest)
    }

    pub fn generation_policy_for_dimensions(
        &self,
        width: u32,
        height: u32,
    ) -> assets::Result<GenerationPolicyDescriptor> {
        let tokens = pixel_bound_tokens(width, height, 32, 262_144, 4_194_304, 32, 1)?;
        let mut policy = self.generation_policy.clone();
        let feedback = policy
            .feedback
            .as_mut()
            .ok_or_else(|| assets::Error::invalid("SenseNova generation policy has no feedback"))?;
        if feedback.ingest.steps.as_slice() != [ImageIngestStep::VitEncode]
            || feedback.ingest.step_kv_tokens.len() != 1
        {
            return Err(assets::Error::invalid(
                "SenseNova feedback recipe does not match its image processor",
            ));
        }
        feedback.ingest.step_kv_tokens[0] = ImageKvEffect::Exact { tokens };
        Ok(policy)
    }
}

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
    ResolutionPolicy {
        default: buckets[1].clone(),
        buckets,
        allow_custom: false,
    }
}
