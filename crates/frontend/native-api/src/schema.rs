use serde::Deserialize;
use uniserve_engine_client::GenerationConstraint;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct NativeGenerateBody {
    #[serde(default)]
    pub prompt: String,
    #[serde(default)]
    pub constraint: Option<GenerationConstraint>,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub assistant_prefix: Option<String>,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop_token_ids: Vec<u32>,
    #[serde(default)]
    pub image_bias: Option<f32>,
    #[serde(default)]
    pub image: Option<NativeImageBody>,
    #[serde(default)]
    pub input_images: Vec<NativeInputImage>,
    #[serde(default)]
    pub input_image_b64: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct NativeImageBody {
    #[serde(default)]
    pub resolution: Option<String>,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    #[serde(default)]
    pub steps: Option<u16>,
    #[serde(default)]
    pub cfg_text_scale: Option<f32>,
    #[serde(default)]
    pub cfg_img_scale: Option<f32>,
    #[serde(default)]
    pub cfg_interval: Option<[f32; 2]>,
    #[serde(default)]
    pub cfg_renorm_type: Option<String>,
    #[serde(default)]
    pub cfg_renorm_min: Option<f32>,
    #[serde(default)]
    pub timestep_shift: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default)]
    pub max_images: Option<u16>,
    #[serde(default, alias = "image_prompts")]
    pub prompts: Vec<String>,
    #[serde(default)]
    pub retain_images: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct NativeInputImage {
    pub b64: String,
    #[serde(default)]
    pub position: Option<u32>,
    #[serde(default)]
    pub num_tokens: Option<u32>,
}

impl NativeGenerateBody {
    pub fn image(&self) -> NativeImageBody {
        self.image.clone().unwrap_or_default()
    }

    pub fn negative_prompt(&self) -> String {
        self.negative_prompt
            .clone()
            .or_else(|| {
                self.image
                    .as_ref()
                    .and_then(|image| image.negative_prompt.clone())
            })
            .unwrap_or_default()
    }

    pub fn input_images(&self) -> Vec<NativeInputImage> {
        let mut images = self.input_images.clone();
        if let Some(b64) = &self.input_image_b64
            && images.is_empty()
        {
            images.push(NativeInputImage {
                b64: b64.clone(),
                position: None,
                num_tokens: None,
            });
        }
        images
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_rejects_unknown_generation_selector() {
        let error = serde_json::from_str::<NativeGenerateBody>(
            r#"{"prompt":"draw a city","mode":"image"}"#,
        )
        .expect_err("unknown generation selectors must be rejected");

        assert!(error.to_string().contains("unknown field `mode`"));
    }
}
