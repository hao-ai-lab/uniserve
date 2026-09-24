//! OpenAI-compatible image-generation request and response schemas.

use serde::{Deserialize, Serialize};
use validator::Validate;

use super::common::Normalizable;

/// Configured OpenAI-compatible image-generation request for
/// `/v1/images/generations`.
///
/// Serde enforces the field set and types. The value checks (non-blank
/// prompt, `n == 1`, served model name, positive `steps`, well-formed `size`)
/// run in `InputProcessor::preprocess_image_request` when the request is
/// lowered.
#[derive(Debug, Clone, Deserialize, PartialEq, Validate)]
#[serde(deny_unknown_fields)]
pub struct ImageGenerationRequest {
    /// Text prompt describing the requested image.
    pub prompt: String,
    /// Served model name; the served-model check is skipped when omitted.
    #[serde(default)]
    pub model: Option<String>,
    /// Number of images to generate; lowering accepts only `1`.
    #[serde(default = "default_image_count")]
    pub n: u16,
    /// Output size as `WIDTHxHEIGHT` in pixels. When omitted, the profile's
    /// resolution policy chooses both dimensions.
    #[serde(default)]
    pub size: Option<String>,
    /// Number of denoising steps per image.
    #[serde(default)]
    pub steps: Option<u16>,
    /// Image-generation random seed.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Text prompt describing content to suppress.
    #[serde(default)]
    pub negative_prompt: Option<String>,
    /// Text classifier-free-guidance scale.
    #[serde(default)]
    pub guidance_scale: Option<f32>,
    /// Image classifier-free-guidance scale.
    #[serde(default)]
    pub image_guidance_scale: Option<f32>,
    /// Classifier-free-guidance renormalization policy.
    #[serde(default)]
    pub cfg_norm: Option<uniserve_core::CfgRenorm>,
    /// Fractional denoising interval in which guidance is active.
    #[serde(default)]
    pub cfg_interval: Option<[f32; 2]>,
    /// Diffusion scheduler timestep shift.
    #[serde(default)]
    pub timestep_shift: Option<f32>,
}

impl Normalizable for ImageGenerationRequest {}

/// Returns the default number of requested images.
const fn default_image_count() -> u16 {
    1
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
/// Completed OpenAI-compatible image-generation response.
pub struct ImageGenerationResponse {
    /// Response creation timestamp in Unix seconds.
    pub created: u64,
    /// Generated images in the order their `ImageDone` events arrived.
    pub data: Vec<GeneratedImageData>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
/// Base64-encoded generated image and optional revised prompt.
pub struct GeneratedImageData {
    /// Base64-encoded PNG image bytes.
    pub b64_json: String,
    /// Model-revised prompt, when produced.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
    /// Image height in pixels.
    pub height: u32,
    /// Image width in pixels.
    pub width: u32,
    /// Encoded image length in bytes.
    pub bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_rejects_unknown_or_untyped_fields() {
        let unknown =
            serde_json::from_str::<ImageGenerationRequest>(r#"{"prompt":"draw","width":1024}"#)
                .expect_err("unknown image controls must not bypass adapter validation");
        assert!(unknown.to_string().contains("unknown field `width`"));

        serde_json::from_str::<ImageGenerationRequest>(r#"{"prompt":"draw","steps":70000}"#)
            .expect_err("steps must fit the typed image-step range");
    }
}
