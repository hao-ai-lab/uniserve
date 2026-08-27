use serde::{Deserialize, Serialize};
use uniserve_core::{
    FeedbackNextToken, FeedbackSource, GenOnlyStartPolicyDescriptor, GeneratedImageFeedbackRecipe,
    GenerationConstraint, GenerationPolicyDescriptor, ImageIngestRecipe, ImageIngestStep,
    ImageKvEffect, Modality, TriggerPolicyDescriptor,
};

use super::resolution::{ResolutionBucket, ResolutionName, ResolutionPolicy};
use super::{
    GenerationControls, ImageGenerationDefaults, OutputFilterPolicy, encode, required_token,
    required_token_id, stride_resize_tokens,
};
use crate::profile::assets;
use crate::profile::tokenizer::HuggingFaceTokenizer;

const DEFAULT_SYSTEM_PROMPT: &str = "You should first think about the planning process in the mind and then generate the image. \n     The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here";
pub const CONTEXT_SYSTEM_PROMPT: &str = "\nLet's think step by step to answer the question. For text-based thinking, enclose the process within <think> </think>, e.g. <think> thinking process here </think>. For visual thinking, enclose the content within <image_start> </image_end>, e.g. <image_start> thinking image here </image_end>. Finally conclude with the final answer wrapped in <answer></answer> tags, i.e.<answer> answer here </answer>.\n";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BagelProfile {
    pub controls: GenerationControls,
    pub image_defaults: ImageGenerationDefaults,
    pub resolution_policy: ResolutionPolicy,
    pub output_filter: OutputFilterPolicy,
    pub image_ingest: ImageIngestRecipe,
    pub generation_policy: GenerationPolicyDescriptor,
}

impl BagelProfile {
    pub const ID: &'static str = "bagel";

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
        let image_ingest = ImageIngestRecipe {
            steps: vec![ImageIngestStep::VaeEncode, ImageIngestStep::VitEncode],
            logical_positions: 1,
            step_kv_tokens: vec![ImageKvEffect::WorkerDefined, ImageKvEffect::WorkerDefined],
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
                    steps: vec![ImageIngestStep::VaeEncode],
                    logical_positions: 2,
                    step_kv_tokens: vec![ImageKvEffect::WorkerDefined],
                    modality: Modality::Und,
                },
                sample_continuation: false,
            }),
            ..GenerationPolicyDescriptor::default()
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
            image_ingest,
            generation_policy,
        })
    }

    pub fn default_system_prompt(constraint: GenerationConstraint) -> Option<&'static str> {
        matches!(constraint, GenerationConstraint::Default).then_some(DEFAULT_SYSTEM_PROMPT)
    }

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

    pub fn image_ingest_for_dimensions(
        &self,
        width: u32,
        height: u32,
        _image_count: usize,
    ) -> assets::Result<ImageIngestRecipe> {
        let vae_tokens = stride_resize_tokens(width, height, &[(1024, 512, 16, 1_806_336)], 16, 2)?;
        let vit_tokens = stride_resize_tokens(
            width,
            height,
            &[(1024, 512, 16, 1_806_336), (980, 224, 14, 1_806_336)],
            14,
            2,
        )?;
        let mut ingest = self.image_ingest.clone();
        ingest.step_kv_tokens = vec![
            ImageKvEffect::Exact { tokens: vae_tokens },
            ImageKvEffect::Exact { tokens: vit_tokens },
        ];
        Ok(ingest)
    }

    pub fn generation_policy_for_dimensions(
        &self,
        width: u32,
        height: u32,
    ) -> assets::Result<GenerationPolicyDescriptor> {
        let tokens = stride_resize_tokens(width, height, &[(1024, 512, 16, 1_806_336)], 16, 2)?;
        let mut policy = self.generation_policy.clone();
        let feedback = policy
            .feedback
            .as_mut()
            .ok_or_else(|| assets::Error::invalid("Bagel generation policy has no feedback"))?;
        if feedback.ingest.steps.as_slice() != [ImageIngestStep::VaeEncode]
            || feedback.ingest.step_kv_tokens.len() != 1
        {
            return Err(assets::Error::invalid(
                "Bagel feedback recipe does not match its image processor",
            ));
        }
        feedback.ingest.step_kv_tokens[0] = ImageKvEffect::Exact { tokens };
        Ok(policy)
    }
}
