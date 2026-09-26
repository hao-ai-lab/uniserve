//! DiffusionGemma checkpoint facts consumed by the server.
//!
//! A `DiffusionGemmaForBlockDiffusion` checkpoint (`model_type =
//! "diffusion_gemma"`) is a Gemma-4 encoder-decoder that denoises a fixed-length
//! token canvas over a prompt prefix. [`DiffusionGemmaProfile::resolve`]
//! normalizes the checkpoint metadata the server needs once at startup: the
//! canvas length and its control tokens, the Gemma-4 image patch budget that
//! sizes image placeholders, and the block-diffusion sampler defaults of
//! `generation_config.json`. The text context limit lives under the root
//! configuration's `text_config` and is read by `assets::ModelConfig`.
//!
//! Both published precisions (BF16 and the ModelOpt NVFP4 variant) share this
//! metadata; the NVFP4 checkpoint differs only by a `quantization_config`,
//! which the profile records without changing any serving behavior.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::profile::assets::{self, GenerationConfig, read_json};
use crate::profile::omni::required_token_id;
use crate::profile::tokenizer::HuggingFaceTokenizer;

/// Serving facts of one DiffusionGemma checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiffusionGemmaProfile {
    /// Tokens in one denoising canvas (`canvas_length`): the block length of
    /// block-diffusion generation and the full System One readout canvas.
    pub canvas_length: u32,
    /// Vocabulary identities of the canvas and image control tokens.
    pub tokens: ControlTokens,
    /// Gemma-4 image patch arithmetic that sizes image placeholders.
    pub images: PatchBudget,
    /// Block-diffusion sampler defaults from `generation_config.json`.
    pub denoising: DenoisingDefaults,
    /// Whether the checkpoint declares a `quantization_config` (the ModelOpt
    /// NVFP4 variant quantizes only the MoE experts); serving is identical.
    pub quantized: bool,
}

/// Control-token identities of a DiffusionGemma vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlTokens {
    /// `<mask>`: an undecided canvas position.
    pub mask: u32,
    /// `<pad>`: canvas filler after the end of the model turn.
    pub pad: u32,
    /// `<turn|>`: closes a chat turn, and ends the scaffold of a readout canvas.
    pub turn_end: u32,
    /// `<|image|>` (`image_token_id`): one image soft-token placeholder.
    pub image: u32,
    /// `<|image>` (`boi_token_id`): opens an image's soft-token run.
    pub image_start: u32,
    /// `<image|>` (`eoi_token_id`): closes an image's soft-token run.
    pub image_end: u32,
}

/// Gemma-4 aspect-ratio-preserving image resize budget.
///
/// An image is resized so that its side lengths are multiples of
/// `patch_size * pooling_kernel_size` pixels and it yields at most
/// `max_soft_tokens * pooling_kernel_size²` patches; each
/// `pooling_kernel_size²` patches pool into one soft token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatchBudget {
    /// Square patch side in pixels (`vision_config.patch_size`).
    pub patch_size: u32,
    /// Square pooling window side in patches (`vision_config.pooling_kernel_size`).
    pub pooling_kernel_size: u32,
    /// Soft tokens of the largest image (`vision_soft_tokens_per_image`).
    pub max_soft_tokens: u32,
}

impl PatchBudget {
    /// Returns the number of soft tokens an image of `width` x `height` pixels
    /// contributes, exactly as the Hugging Face `Gemma4ImageProcessor` resizes
    /// it (`get_aspect_ratio_preserving_size`).
    ///
    /// The image is scaled to the patch budget's pixel area, each side is
    /// rounded down to a multiple of the pooled patch side, and a side that
    /// rounds to zero becomes one pooled patch side while the other follows the
    /// integer aspect ratio, bounded by the longest admissible side. The float
    /// arithmetic follows the reference operation for operation, since
    /// IEEE-754 division and square root round identically in both.
    ///
    /// # Errors
    ///
    /// Returns a message for an image with a zero side, which has no aspect
    /// ratio, or when the resize would exceed the patch budget (impossible for
    /// positive sides, kept as the reference's own guard).
    pub fn soft_tokens(&self, width: u32, height: u32) -> Result<u32, String> {
        if width == 0 || height == 0 {
            return Err(format!("image of {width}x{height} pixels has no area"));
        }
        let pooling = u64::from(self.pooling_kernel_size);
        let patch = u64::from(self.patch_size);
        let max_patches = u64::from(self.max_soft_tokens) * pooling * pooling;
        let side_multiple = pooling * patch;
        let target_pixels = (max_patches * patch * patch) as f64;
        let (height, width) = (f64::from(height), f64::from(width));

        let factor = (target_pixels / (height * width)).sqrt();
        let mut target_height =
            ((factor * height) / side_multiple as f64).floor() as u64 * side_multiple;
        let mut target_width =
            ((factor * width) / side_multiple as f64).floor() as u64 * side_multiple;

        // One side rounds to zero only for a very elongated image; that side
        // keeps one pooled patch and the other follows the aspect ratio.
        let max_side = (max_patches / (pooling * pooling)) * side_multiple;
        if target_height == 0 && target_width == 0 {
            return Err("image resizes to zero pixels".to_owned());
        } else if target_height == 0 {
            target_height = side_multiple;
            target_width = ((width / height).floor() as u64 * side_multiple).min(max_side);
        } else if target_width == 0 {
            target_width = side_multiple;
            target_height = ((height / width).floor() as u64 * side_multiple).min(max_side);
        }
        if (target_height * target_width) as f64 > target_pixels {
            return Err(format!(
                "image resizes to {target_width}x{target_height} pixels, beyond {max_patches} patches"
            ));
        }

        let patches = (target_height / patch) * (target_width / patch);
        u32::try_from(patches / (pooling * pooling))
            .map_err(|_| "image soft-token count overflows".to_owned())
    }
}

/// Block-diffusion sampler defaults of a DiffusionGemma checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DenoisingDefaults {
    /// Maximum denoising steps per canvas (`max_denoising_steps`).
    pub max_denoising_steps: u32,
    /// Entropy budget of the tokens accepted in one step
    /// (`sampler_config.entropy_bound`), in nats.
    pub entropy_bound: f32,
    /// Sampling temperature at the first step (`t_min`).
    pub t_min: f32,
    /// Sampling temperature at the last step (`t_max`).
    pub t_max: f32,
    /// Stopping threshold on the canvas's residual uncertainty
    /// (`confidence_threshold`).
    pub confidence_threshold: f32,
    /// Consecutive unchanged steps that end denoising early
    /// (`stability_threshold`).
    pub stability_threshold: u32,
}

/// The root `config.json` fields a DiffusionGemma profile reads.
#[derive(Deserialize)]
struct CheckpointConfig {
    canvas_length: u32,
    image_token_id: u32,
    boi_token_id: u32,
    eoi_token_id: u32,
    vision_soft_tokens_per_image: u32,
    vision_config: VisionConfig,
    #[serde(default)]
    quantization_config: Option<serde_json::Value>,
}

/// The `vision_config` fields that size image patches.
#[derive(Deserialize)]
struct VisionConfig {
    patch_size: u32,
    pooling_kernel_size: u32,
}

impl DiffusionGemmaProfile {
    /// Resolves the profile from the checkpoint's root configuration, its
    /// generation defaults, and its tokenizer.
    ///
    /// The canvas control tokens are resolved by their text (`<mask>`,
    /// `<pad>`, `<turn|>`); the image token identities come from the root
    /// configuration.
    ///
    /// # Errors
    ///
    /// Fails when `config_path` is absent or unreadable, when it lacks a
    /// required field, when a control token is missing from the vocabulary, or
    /// when `generation_config.json` lacks a sampler default.
    pub fn resolve(
        config_path: Option<&Path>,
        generation: &GenerationConfig,
        tokenizer: &HuggingFaceTokenizer,
    ) -> assets::Result<Self> {
        let config_path = config_path.ok_or_else(|| {
            assets::Error::invalid("a DiffusionGemma checkpoint requires its root config.json")
        })?;
        let config: CheckpointConfig = read_json(config_path)?;
        let tokens = ControlTokens {
            mask: required_token_id(tokenizer, "<mask>", "DiffusionGemma canvas mask")?,
            pad: required_token_id(tokenizer, "<pad>", "DiffusionGemma canvas padding")?,
            turn_end: required_token_id(tokenizer, "<turn|>", "DiffusionGemma end-of-turn")?,
            image: config.image_token_id,
            image_start: config.boi_token_id,
            image_end: config.eoi_token_id,
        };
        let sampler = generation
            .sampler_config
            .as_ref()
            .ok_or(assets::Error::MissingField {
                field: "sampler_config",
            })?;
        let denoising = DenoisingDefaults {
            max_denoising_steps: required(generation.max_denoising_steps, "max_denoising_steps")?,
            entropy_bound: required(sampler.entropy_bound, "sampler_config.entropy_bound")?,
            t_min: required(generation.t_min, "t_min")?,
            t_max: required(generation.t_max, "t_max")?,
            confidence_threshold: required(
                generation.confidence_threshold,
                "confidence_threshold",
            )?,
            stability_threshold: required(generation.stability_threshold, "stability_threshold")?,
        };

        Ok(Self {
            canvas_length: config.canvas_length,
            tokens,
            images: PatchBudget {
                patch_size: config.vision_config.patch_size,
                pooling_kernel_size: config.vision_config.pooling_kernel_size,
                max_soft_tokens: config.vision_soft_tokens_per_image,
            },
            denoising,
            quantized: config.quantization_config.is_some(),
        })
    }
}

/// Returns a required generation-config value or names the missing field.
fn required<T>(value: Option<T>, field: &'static str) -> assets::Result<T> {
    value.ok_or(assets::Error::MissingField { field })
}

#[cfg(test)]
mod tests {
    use super::PatchBudget;

    /// Soft-token counts match the Hugging Face Gemma-4 processor for square,
    /// landscape, portrait, tiny, and extremely elongated images. The cases
    /// were produced by `Gemma4Processor._get_num_multimodal_tokens` and
    /// spot-checked against the image processor on decoded images.
    #[test]
    fn image_soft_tokens_follow_the_gemma4_processor() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            patch_size: u32,
            pooling_kernel_size: u32,
            max_soft_tokens: u32,
            cases: Vec<Case>,
        }
        #[derive(serde::Deserialize)]
        struct Case {
            width: u32,
            height: u32,
            soft_tokens: u32,
        }
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../../tests/python/fixtures/diffusion_gemma_image_tokens.json"
        ))
        .unwrap();
        let budget = PatchBudget {
            patch_size: fixture.patch_size,
            pooling_kernel_size: fixture.pooling_kernel_size,
            max_soft_tokens: fixture.max_soft_tokens,
        };

        for case in fixture.cases {
            assert_eq!(
                budget.soft_tokens(case.width, case.height),
                Ok(case.soft_tokens),
                "{}x{}",
                case.width,
                case.height
            );
        }
    }

    #[test]
    fn an_image_without_area_has_no_soft_tokens() {
        let budget = PatchBudget {
            patch_size: 16,
            pooling_kernel_size: 3,
            max_soft_tokens: 280,
        };

        assert!(budget.soft_tokens(0, 10).is_err());
        assert!(budget.soft_tokens(10, 0).is_err());
    }
}
