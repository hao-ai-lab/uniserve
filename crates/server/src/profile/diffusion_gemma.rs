//! DiffusionGemma checkpoint facts consumed by the server.
//!
//! A `DiffusionGemmaForBlockDiffusion` checkpoint (`model_type =
//! "diffusion_gemma"`) is a Gemma-4 encoder-decoder that denoises a fixed-length
//! token canvas over a prompt prefix. [`DiffusionGemmaProfile::resolve`]
//! normalizes the checkpoint metadata the server needs once at startup: the
//! canvas length and its control tokens, the Gemma-4 image patch budget that
//! sizes image placeholders, and the block-diffusion sampler defaults of
//! `generation_config.json`, which a server may override
//! ([`DenoisingOverrides`]). The text context limit lives under the root
//! configuration's `text_config` and is read by `assets::ModelConfig`.
//! [`ControlTokens::expand_images`] expands a rendered prompt's image tokens
//! as the Gemma-4 processor does.
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
///
/// The sampling temperature falls linearly from `t_max` at a canvas's first
/// step to `t_min` at its last (Transformers'
/// `LinearTemperatureScheduleLogitsProcessor`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DenoisingDefaults {
    /// Maximum denoising steps per canvas (`max_denoising_steps`).
    pub max_denoising_steps: u32,
    /// Entropy budget of the tokens accepted in one step
    /// (`sampler_config.entropy_bound`), in nats.
    pub entropy_bound: f32,
    /// Sampling temperature at the last step (`t_min`).
    pub t_min: f32,
    /// Sampling temperature at the first step (`t_max`).
    pub t_max: f32,
    /// Stopping threshold on the canvas's residual uncertainty
    /// (`confidence_threshold`).
    pub confidence_threshold: f32,
    /// Consecutive unchanged steps that end denoising early
    /// (`stability_threshold`); zero ends it on confidence alone.
    pub stability_threshold: u32,
}

impl DenoisingDefaults {
    /// Returns these defaults with every value `overrides` names replaced.
    ///
    /// # Errors
    ///
    /// Returns a message naming the first resulting value outside the
    /// domain the block-diffusion sampler accepts
    /// (`uniserve.diffusion.canvas.CanvasSampling`): a positive step limit,
    /// a positive entropy bound and confidence threshold, and temperatures
    /// with `0 <= t_min < t_max`, all finite. A zero stability threshold
    /// stops a canvas on confidence alone.
    pub fn with_overrides(self, overrides: &DenoisingOverrides) -> Result<Self, String> {
        let value = Self {
            max_denoising_steps: overrides
                .max_denoising_steps
                .unwrap_or(self.max_denoising_steps),
            entropy_bound: overrides.entropy_bound.unwrap_or(self.entropy_bound),
            t_min: overrides.t_min.unwrap_or(self.t_min),
            t_max: overrides.t_max.unwrap_or(self.t_max),
            confidence_threshold: overrides
                .confidence_threshold
                .unwrap_or(self.confidence_threshold),
            stability_threshold: overrides
                .stability_threshold
                .unwrap_or(self.stability_threshold),
        };
        let positive = |number: f32| number.is_finite() && number > 0.0;
        let invalid = if value.max_denoising_steps == 0 {
            Some("max_denoising_steps must be positive")
        } else if !positive(value.entropy_bound) {
            Some("entropy_bound must be positive and finite")
        } else if !(value.t_min.is_finite() && value.t_min >= 0.0) {
            Some("t_min must be finite and non-negative")
        } else if !(value.t_max.is_finite() && value.t_max > value.t_min) {
            Some("t_max must be finite and above t_min")
        } else if !positive(value.confidence_threshold) {
            Some("confidence_threshold must be positive and finite")
        } else {
            None
        };
        match invalid {
            Some(message) => Err(message.to_owned()),
            None => Ok(value),
        }
    }
}

/// Server-level replacements for the checkpoint's block-diffusion sampler
/// defaults, read from a JSON object whose keys name `generation_config.json`
/// fields. Requests carry no such fields, so these settings apply to every
/// request the server generates.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenoisingOverrides {
    /// Replaces `max_denoising_steps`.
    pub max_denoising_steps: Option<u32>,
    /// Replaces `sampler_config.entropy_bound`.
    pub entropy_bound: Option<f32>,
    /// Replaces `t_min`.
    pub t_min: Option<f32>,
    /// Replaces `t_max`.
    pub t_max: Option<f32>,
    /// Replaces `confidence_threshold`.
    pub confidence_threshold: Option<f32>,
    /// Replaces `stability_threshold`.
    pub stability_threshold: Option<u32>,
}

/// Where one image's soft tokens sit in a prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImagePlacement {
    /// Index of the image among the request's input images.
    pub source: usize,
    /// Prompt position of the first soft token, just after `<|image>`.
    pub offset: u32,
    /// Number of consecutive soft-token placeholders.
    pub soft_tokens: u32,
}

/// A rendered prompt whose image tokens do not match its input images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ImageTokenMismatch {
    /// The request's text encodes image tokens of its own.
    #[error("the request text encodes {0} image placeholder token(s) beyond its images")]
    Extra(usize),
    /// The template rendered fewer image tokens than the request has images.
    #[error("the prompt renders {rendered} image token(s) for {images} image(s)")]
    Missing {
        /// Image tokens in the rendered prompt.
        rendered: usize,
        /// Input images of the request.
        images: usize,
    },
}

impl ControlTokens {
    /// Expands a rendered prompt's image tokens as the Gemma-4 processor
    /// does, and reports where each image's soft tokens stand.
    ///
    /// The chat template writes one `<|image|>` per image, in input order;
    /// each becomes `<|image>`, `soft_tokens[i]` copies of `<|image|>`, and
    /// `<image|>`. All other tokens are kept.
    ///
    /// # Errors
    ///
    /// Returns [`ImageTokenMismatch`] unless `rendered` holds exactly one
    /// image token per entry of `soft_tokens`.
    pub fn expand_images(
        &self,
        rendered: &[u32],
        soft_tokens: &[u32],
    ) -> Result<(Vec<u32>, Vec<ImagePlacement>), ImageTokenMismatch> {
        let placeholders = rendered
            .iter()
            .filter(|token| **token == self.image)
            .count();
        if placeholders > soft_tokens.len() {
            return Err(ImageTokenMismatch::Extra(placeholders - soft_tokens.len()));
        }
        if placeholders < soft_tokens.len() {
            return Err(ImageTokenMismatch::Missing {
                rendered: placeholders,
                images: soft_tokens.len(),
            });
        }

        let expanded: usize = soft_tokens.iter().map(|tokens| *tokens as usize + 1).sum();
        let mut token_ids = Vec::with_capacity(rendered.len() + expanded);
        let mut placements = Vec::with_capacity(soft_tokens.len());
        for &token in rendered {
            if token != self.image {
                token_ids.push(token);
                continue;
            }
            let source = placements.len();
            let count = soft_tokens[source];
            token_ids.push(self.image_start);
            let offset = u32::try_from(token_ids.len()).unwrap_or(u32::MAX);
            token_ids.extend(std::iter::repeat_n(self.image, count as usize));
            token_ids.push(self.image_end);
            placements.push(ImagePlacement {
                source,
                offset,
                soft_tokens: count,
            });
        }
        Ok((token_ids, placements))
    }
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
    use super::{DenoisingDefaults, DenoisingOverrides, PatchBudget};

    const CHECKPOINT: DenoisingDefaults = DenoisingDefaults {
        max_denoising_steps: 48,
        entropy_bound: 0.1,
        t_min: 0.4,
        t_max: 0.8,
        confidence_threshold: 0.005,
        stability_threshold: 1,
    };

    /// A server override replaces exactly the values it names, read from a
    /// JSON object keyed by the `generation_config.json` field names.
    #[test]
    fn a_denoising_override_replaces_the_values_it_names() {
        let overrides: DenoisingOverrides =
            serde_json::from_str(r#"{"max_denoising_steps": 32, "t_max": 1.0}"#).unwrap();
        assert_eq!(
            CHECKPOINT.with_overrides(&overrides),
            Ok(DenoisingDefaults {
                max_denoising_steps: 32,
                t_max: 1.0,
                ..CHECKPOINT
            })
        );
        assert_eq!(
            CHECKPOINT.with_overrides(&DenoisingOverrides::default()),
            Ok(CHECKPOINT)
        );
    }

    /// Unknown keys and out-of-range values are refused, naming the field.
    #[test]
    fn an_invalid_denoising_override_is_refused() {
        assert!(serde_json::from_str::<DenoisingOverrides>(r#"{"temperature": 0.5}"#).is_err());
        for (json, field) in [
            (r#"{"max_denoising_steps": 0}"#, "max_denoising_steps"),
            (r#"{"entropy_bound": 0.0}"#, "entropy_bound"),
            (r#"{"t_min": -0.1}"#, "t_min"),
            (r#"{"t_max": 0.4}"#, "t_max"),
            (r#"{"confidence_threshold": 0.0}"#, "confidence_threshold"),
        ] {
            let overrides: DenoisingOverrides = serde_json::from_str(json).unwrap();
            let message = CHECKPOINT.with_overrides(&overrides).unwrap_err();
            assert!(message.starts_with(field), "{message}");
        }
    }

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
