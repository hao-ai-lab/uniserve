//! Shared contracts and model-specific profiles for multimodal generation.

pub mod bagel;
pub mod resolution;
pub mod sensenova;

use serde::{Deserialize, Serialize};

use crate::profile::assets::{self, Error as AssetError};
use crate::profile::tokenizer::HuggingFaceTokenizer;

/// Returns the storage width in bytes for a model dtype.
pub(super) const fn model_dtype_bytes(dtype: uniserve_core::ModelDtype) -> u64 {
    match dtype {
        uniserve_core::ModelDtype::Float16 | uniserve_core::ModelDtype::BFloat16 => 2,
        uniserve_core::ModelDtype::Float32 => 4,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Model-token controls for switching between understanding and generation.
pub struct GenerationControls {
    /// Beginning-of-sequence token identifier.
    pub bos: u32,
    /// End-of-sequence token identifier.
    pub eos: u32,
    /// Token identifier that opens generated image content.
    pub start_of_image: u32,
    /// Token identifier that closes generated image content.
    pub end_of_image: u32,
    /// Tokenizer text corresponding to [`Self::start_of_image`].
    pub start_of_image_text: String,
    /// Tokenizer text corresponding to [`Self::end_of_image`].
    pub end_of_image_text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Profile-provided defaults for output image generation.
pub struct ImageGenerationDefaults {
    /// Default named output resolution.
    pub resolution: resolution::ResolutionName,
    /// Default diffusion step count.
    pub steps: u16,
    /// Default classifier-free guidance scale for text conditioning.
    pub cfg_text_scale: f32,
    /// Default classifier-free guidance scale for image conditioning.
    pub cfg_img_scale: f32,
    /// Default guidance renormalization algorithm.
    pub cfg_renorm_type: uniserve_core::CfgRenorm,
    /// Minimum scale at which guidance renormalization applies.
    pub cfg_renorm_min: f32,
    /// Fractional diffusion interval over which guidance applies.
    pub cfg_interval: (f32, f32),
    /// Default diffusion timestep shift.
    pub timestep_shift: f32,
    /// Optional deterministic sampling seed.
    pub seed: Option<u64>,
    /// Default maximum number of images generated for one request.
    pub max_images: u16,
    /// Hard profile limit on generated images per request.
    pub max_images_limit: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Start and end delimiters for a filtered text section.
pub struct DelimitedTextPolicy {
    /// Text delimiter that opens the section.
    pub start: String,
    /// Text delimiter that closes the section.
    pub end: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Model-selected filters applied to generated assistant text.
pub struct OutputFilterPolicy {
    /// Optional section whose contents are exposed as reasoning.
    pub reasoning: Option<DelimitedTextPolicy>,
    /// Sections whose delimiters are removed while their contents remain visible.
    pub visible_wrappers: Vec<DelimitedTextPolicy>,
}

/// Resolves a required special-token string from tokenizer metadata.
pub(super) fn required_token(
    tokenizer: &HuggingFaceTokenizer,
    token: &str,
    role: &str,
) -> assets::Result<(u32, String)> {
    tokenizer
        .token_to_id(token)
        .map(|id| (id, token.to_string()))
        .ok_or_else(|| {
            AssetError::invalid(format!(
                "configured {role} control token {token:?} is missing from the tokenizer"
            ))
        })
}

/// Resolves the unique token identifier for a required token string.
pub(super) fn required_token_id(
    tokenizer: &HuggingFaceTokenizer,
    token: &str,
    role: &str,
) -> assets::Result<u32> {
    required_token(tokenizer, token, role).map(|(id, _)| id)
}

/// Encodes prompt text without automatically adding special tokens.
pub(super) fn encode(
    tokenizer: &HuggingFaceTokenizer,
    text: &str,
) -> crate::profile::tokenizer::Result<Vec<u32>> {
    tokenizer.encode(text, false)
}

/// Renders a minimal ChatML prompt with an open assistant turn.
pub(super) fn chatml(system: Option<&str>, user: &str, assistant_suffix: &str) -> String {
    let mut output = String::new();
    if let Some(system) = system {
        output.push_str("<|im_start|>system\n");
        output.push_str(system);
        output.push_str("<|im_end|>\n");
    }
    output.push_str("<|im_start|>user\n");
    output.push_str(user);
    output.push_str("<|im_end|>\n<|im_start|>assistant\n");
    output.push_str(assistant_suffix);
    output
}

/// Computes visual token count after stride-aligned resizing.
pub(super) fn stride_resize_tokens(
    width: u32,
    height: u32,
    transforms: &[(u32, u32, u32, u64)],
    token_stride: u32,
    marker_tokens: u32,
) -> assets::Result<u32> {
    let mut dimensions = (width, height);
    for &(max_side, min_side, stride, max_pixels) in transforms {
        dimensions = stride_resize(
            dimensions.0,
            dimensions.1,
            max_side,
            min_side,
            stride,
            max_pixels,
        )?;
    }
    dimensions_to_tokens(dimensions, token_stride, marker_tokens)
}

/// Computes visual token count under pixel-area and aspect-ratio bounds.
pub(super) fn pixel_bound_tokens(
    width: u32,
    height: u32,
    factor: u32,
    min_pixels: u64,
    max_pixels: u64,
    token_stride: u32,
    marker_tokens: u32,
) -> assets::Result<u32> {
    let dimensions = pixel_bound_resize(width, height, factor, min_pixels, max_pixels)?;
    dimensions_to_tokens(dimensions, token_stride, marker_tokens)
}

/// Converts image dimensions into model token counts.
fn dimensions_to_tokens(
    (width, height): (u32, u32),
    token_stride: u32,
    marker_tokens: u32,
) -> assets::Result<u32> {
    if token_stride == 0
        || !width.is_multiple_of(token_stride)
        || !height.is_multiple_of(token_stride)
    {
        return Err(AssetError::invalid(
            "image dimensions are outside the configured token stride",
        ));
    }
    let tokens = u64::from(width / token_stride)
        .saturating_mul(u64::from(height / token_stride))
        .saturating_add(u64::from(marker_tokens));
    u32::try_from(tokens)
        .map_err(|_| AssetError::invalid("image token count exceeds the engine range"))
}

/// Resizes dimensions within side and pixel limits while preserving stride alignment.
fn stride_resize(
    width: u32,
    height: u32,
    max_side: u32,
    min_side: u32,
    stride: u32,
    max_pixels: u64,
) -> assets::Result<(u32, u32)> {
    if width == 0 || height == 0 || max_side == 0 || min_side == 0 || stride == 0 || max_pixels == 0
    {
        return Err(AssetError::invalid(
            "configured stride-resize geometry requires positive values",
        ));
    }
    let mut scale = (f64::from(max_side) / f64::from(width.max(height))).min(1.0);
    scale = scale.max(f64::from(min_side) / f64::from(width.min(height)));
    let mut resized = scale_to_stride(width, height, scale, stride);
    if u64::from(resized.0).saturating_mul(u64::from(resized.1)) > max_pixels {
        scale = max_pixels as f64 / (f64::from(resized.0) * f64::from(resized.1));
        resized = scale_to_stride(resized.0, resized.1, scale, stride);
    }
    if resized.0.max(resized.1) > max_side {
        scale = f64::from(max_side) / f64::from(resized.0.max(resized.1));
        resized = scale_to_stride(resized.0, resized.1, scale, stride);
    }
    Ok(resized)
}

/// Scales both dimensions and rounds them to positive stride multiples.
fn scale_to_stride(width: u32, height: u32, scale: f64, stride: u32) -> (u32, u32) {
    let scale_one = |value: u32| {
        let scaled = (f64::from(value) * scale).round_ties_even();
        let aligned = (scaled / f64::from(stride)).round_ties_even() * f64::from(stride);
        stride.max(aligned.max(f64::from(stride)) as u32)
    };
    (scale_one(width), scale_one(height))
}

/// Resizes dimensions to factor-aligned values within configured pixel-area bounds.
fn pixel_bound_resize(
    width: u32,
    height: u32,
    factor: u32,
    min_pixels: u64,
    max_pixels: u64,
) -> assets::Result<(u32, u32)> {
    if width == 0 || height == 0 || factor == 0 || min_pixels == 0 || max_pixels < min_pixels {
        return Err(AssetError::invalid(
            "configured pixel-bound geometry is invalid",
        ));
    }
    let aspect = f64::from(width.max(height)) / f64::from(width.min(height));
    if aspect > 200.0 {
        return Err(AssetError::invalid(
            "input image aspect ratio must not exceed 200",
        ));
    }
    let round_factor = |value: u32| {
        factor.max(
            ((f64::from(value) / f64::from(factor)).round_ties_even() as u32)
                .saturating_mul(factor),
        )
    };
    let mut resized_h = round_factor(height);
    let mut resized_w = round_factor(width);
    let pixels = u64::from(resized_h).saturating_mul(u64::from(resized_w));
    if pixels > max_pixels {
        let beta = (f64::from(height) * f64::from(width) / max_pixels as f64).sqrt();
        resized_h = factor.max(
            ((f64::from(height) / beta / f64::from(factor)).floor() as u32).saturating_mul(factor),
        );
        resized_w = factor.max(
            ((f64::from(width) / beta / f64::from(factor)).floor() as u32).saturating_mul(factor),
        );
    } else if pixels < min_pixels {
        let beta = (min_pixels as f64 / (f64::from(height) * f64::from(width))).sqrt();
        resized_h =
            ((f64::from(height) * beta / f64::from(factor)).ceil() as u32).saturating_mul(factor);
        resized_w =
            ((f64::from(width) * beta / f64::from(factor)).ceil() as u32).saturating_mul(factor);
    }
    Ok((resized_w, resized_h))
}
