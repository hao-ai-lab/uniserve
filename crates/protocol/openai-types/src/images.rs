use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::Normalizable;

/// OpenAI-compatible image generation request plus typed UniServe controls.
#[derive(Debug, Clone, Deserialize, PartialEq, Validate)]
#[serde(deny_unknown_fields)]
pub struct ImageGenerationRequest {
    pub prompt: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default = "default_image_count")]
    pub n: u16,
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub quality: Option<String>,
    #[serde(default)]
    pub style: Option<String>,
    #[serde(default)]
    pub response_format: Option<ImageResponseFormat>,
    #[serde(default)]
    pub output_format: Option<ImageOutputFormat>,
    #[serde(default)]
    pub background: Option<String>,
    #[serde(default)]
    pub moderation: Option<String>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub request_id: Option<String>,
    #[serde(default)]
    pub steps: Option<u16>,
    #[serde(default)]
    pub num_inference_steps: Option<u16>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default)]
    pub guidance_scale: Option<f32>,
    #[serde(default)]
    pub image_guidance_scale: Option<f32>,
    #[serde(default)]
    pub cfg_norm: Option<String>,
    #[serde(default)]
    pub cfg_interval: Option<[f32; 2]>,
    #[serde(default)]
    pub timestep_shift: Option<f32>,
}

impl Normalizable for ImageGenerationRequest {}

const fn default_image_count() -> u16 {
    1
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ImageResponseFormat {
    B64Json,
    Url,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ImageOutputFormat {
    Png,
    Jpeg,
    Webp,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ImageGenerationResponse {
    pub created: u64,
    pub data: Vec<GeneratedImageData>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct GeneratedImageData {
    pub b64_json: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
    pub height: u32,
    pub width: u32,
    pub bytes: u64,
    pub sha256: String,
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

        let invalid_steps =
            serde_json::from_str::<ImageGenerationRequest>(r#"{"prompt":"draw","steps":70000}"#)
                .expect_err("steps must fit the typed image-step range");
        assert!(invalid_steps.to_string().contains("u16"));
    }
}
