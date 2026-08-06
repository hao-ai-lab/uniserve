use serde::{Deserialize, Serialize};
use uniserve_core::{
    FeedbackNextToken, FeedbackSource, GenOnlyStartPolicyDescriptor, GeneratedImageFeedbackRecipe,
    GenerationConstraint, GenerationPolicyDescriptor, ImageIngestRecipe, ImageIngestStep,
    ImageKvEffect, Modality, TriggerPolicyDescriptor,
};

use super::resolution::{ResolutionBucket, ResolutionPolicy};
use super::{
    DelimitedTextPolicy, GenerationControls, ImageGenerationDefaults, OutputFilterPolicy, chatml,
    encode, pixel_bound_tokens, required_token, required_token_id,
};
use crate::assets;
use crate::tokenizer::HuggingFaceTokenizer;

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
                resolution: "16:9".to_string(),
                steps: 50,
                cfg_text_scale: 4.0,
                cfg_img_scale: 1.0,
                cfg_renorm_type: "none".to_string(),
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
    ) -> crate::tokenizer::Result<Vec<u32>> {
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
    ) -> crate::tokenizer::Result<Vec<u32>> {
        if prompt.is_empty() {
            return Ok(Vec::new());
        }
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
}

fn resolution_policy() -> ResolutionPolicy {
    let buckets = [
        ("1:1", 1536, 1536),
        ("16:9", 2048, 1152),
        ("1.5K", 2048, 1152),
        ("9:16", 1152, 2048),
        ("3:2", 1888, 1248),
        ("2:3", 1248, 1888),
        ("4:3", 1760, 1312),
        ("3:4", 1312, 1760),
        ("1:2", 1088, 2144),
        ("2:1", 2144, 1088),
        ("1:3", 864, 2592),
        ("3:1", 2592, 864),
    ]
    .into_iter()
    .map(|(name, width, height)| ResolutionBucket {
        name: name.to_string(),
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
