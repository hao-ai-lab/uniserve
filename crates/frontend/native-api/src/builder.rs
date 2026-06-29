use uniserve_engine_client::{
    EngineSamplingParams, GenMode, ImageParams, MmItem, NativeGenerateRequest,
};
use uniserve_text::tokenizer::DynTokenizer;

use super::defaults;
use super::profiles::{IU_SYSTEM_PROMPT, NativeModelProfile};
use super::resolution::resolve_resolution;
use super::schema::NativeGenerateBody;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildError {
    message: String,
}

impl BuildError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

/// CFG and resolution constants for understanding (InterleaveUnd) mode.

/// Understanding mode uses a fixed, model-agnostic visual-reasoning configuration
/// that intentionally differs from the per-profile image-generation defaults
/// (`NativeModelProfile::image_defaults`). Naming them here keeps the values
/// discoverable and prevents silent drift between call sites.
mod understanding_defaults {
 /// Text-guidance CFG scale for understanding-mode visual reasoning.
    pub(super) const CFG_TEXT_SCALE: f32 = 4.0;
 /// Image-guidance CFG scale for understanding-mode visual reasoning.
    pub(super) const CFG_IMG_SCALE: f32 = 2.0;
 /// CFG renorm strategy for understanding-mode visual reasoning.
    pub(super) const CFG_RENORM_TYPE: &str = "text_channel";
 /// CFG renorm floor for understanding-mode visual reasoning.
    pub(super) const CFG_RENORM_MIN: f32 = 0.0;
 /// CFG interval (lo, hi) for understanding-mode visual reasoning.
    pub(super) const CFG_INTERVAL: (f32, f32) = (0.0, 1.0);
 /// Square latent resolution used for understanding-mode thinking images.
    pub(super) const RESOLUTION: u32 = 512;
}

pub struct NativeRequestBuilder<'a> {
    tokenizer: DynTokenizer,
    profile: &'a NativeModelProfile,
}

impl<'a> NativeRequestBuilder<'a> {
    pub fn new(tokenizer: DynTokenizer, profile: &'a NativeModelProfile) -> Self {
        Self { tokenizer, profile }
    }

    pub fn build(&self, body: &NativeGenerateBody) -> Result<NativeGenerateRequest, BuildError> {
        let mode_name = body
            .mode
            .as_deref()
            .unwrap_or(self.profile.default_mode_name());
        let mode = parse_mode(mode_name)?;
        validate_prompt(body, mode)?;
        if !self.profile.supports_mode(mode) {
            return Err(BuildError::new(format!(
                "mode {} is not supported by this model profile",
                mode_name
            )));
        }
        if mode == GenMode::InterleaveUnd {
            return self.build_understanding(body);
        }

        let image = self.resolve_image_params(body)?;
        let negative_prompt = body.negative_prompt();
        let prompt_ids = self.profile.build_prompt_ids(&self.tokenizer, body, mode);
        let neg_prompt_ids = self
            .profile
            .build_negative_prompt_ids(&self.tokenizer, &negative_prompt);
        let sampling = self.resolve_sampling(body, mode)?;

        Ok(NativeGenerateRequest {
            prompt_ids,
            neg_prompt_ids,
            sampling,
            image,
            mode,
            max_tokens: body.max_tokens.unwrap_or(defaults::DEFAULT_MAX_TOKENS),
            mm_items: Vec::new(),
            stop_token_ids: body.stop_token_ids.clone(),
        })
    }

    fn build_understanding(
        &self,
        body: &NativeGenerateBody,
    ) -> Result<NativeGenerateRequest, BuildError> {
        let input_images = body.input_images();
        let first_image = input_images
            .first()
            .ok_or_else(|| BuildError::new("understand mode requires an input image"))?;
        let system = body.system_prompt.as_deref().unwrap_or(IU_SYSTEM_PROMPT);
        let mut sys_ids = self
            .profile
            .wrap_understanding_text(&self.tokenizer, system);
        let question_ids = self
            .profile
            .wrap_understanding_text(&self.tokenizer, &body.prompt);
        let image_position = first_image.position.unwrap_or(sys_ids.len() as u32);
        sys_ids.extend_from_slice(&question_ids);

        let image_body = body.image();
        let steps = image_body.steps.unwrap_or(defaults::DEFAULT_STEPS);
        if steps == 0 {
            return Err(BuildError::new("image.steps must be positive"));
        }
        let max_images = image_body.max_images.unwrap_or(2);
        self.validate_max_images(max_images)?;
        let seed = image_body.seed.or(body.seed).or(Some(0));

        let image = ImageParams {
            steps,
            cfg_text_scale: understanding_defaults::CFG_TEXT_SCALE,
            cfg_img_scale: understanding_defaults::CFG_IMG_SCALE,
            cfg_renorm_type: understanding_defaults::CFG_RENORM_TYPE.into(),
            cfg_renorm_min: understanding_defaults::CFG_RENORM_MIN,
            cfg_interval: understanding_defaults::CFG_INTERVAL,
            timestep_shift: image_body
                .timestep_shift
                .unwrap_or(self.profile.image_defaults.timestep_shift),
            height: understanding_defaults::RESOLUTION,
            width: understanding_defaults::RESOLUTION,
            seed,
            negative_prompt: String::new(),
            max_images,
            image_prompts: image_body.prompts,
            retain_images: image_body.retain_images.unwrap_or(true),
        };
        Ok(NativeGenerateRequest {
            neg_prompt_ids: sys_ids.clone(),
            prompt_ids: sys_ids,
            sampling: self.resolve_sampling(body, GenMode::InterleaveUnd)?,
            image,
            mode: GenMode::InterleaveUnd,
            max_tokens: body.max_tokens.unwrap_or(defaults::DEFAULT_MAX_TOKENS),
            mm_items: vec![MmItem {
                hash: fnv1a(first_image.b64.as_bytes()),
                position: image_position,
                num_tokens: first_image.num_tokens.unwrap_or(0),
                b64: first_image.b64.clone(),
            }],
            stop_token_ids: body.stop_token_ids.clone(),
        })
    }

    fn resolve_image_params(&self, body: &NativeGenerateBody) -> Result<ImageParams, BuildError> {
        let image_body = body.image();
        let defaults = &self.profile.image_defaults;
        let resolution = resolve_resolution(
            self.profile.resolution_policy,
            image_body
                .resolution
                .as_deref()
                .or(Some(defaults.resolution)),
            image_body.width,
            image_body.height,
        )
        .map_err(BuildError::new)?;
        let steps = image_body.steps.unwrap_or(defaults.steps);
        if steps == 0 {
            return Err(BuildError::new("image.steps must be positive"));
        }
        let cfg_interval = image_body
            .cfg_interval
            .map(|interval| (interval[0], interval[1]))
            .unwrap_or(defaults.cfg_interval);
        validate_cfg_interval(cfg_interval)?;

        let max_images = image_body.max_images.unwrap_or(defaults.max_images);
        self.validate_max_images(max_images)?;
        let negative_prompt = body.negative_prompt();
        Ok(ImageParams {
            steps,
            cfg_text_scale: finite_or(
                image_body.cfg_text_scale.unwrap_or(defaults.cfg_text_scale),
                "image.cfg_text_scale",
            )?,
            cfg_img_scale: finite_or(
                image_body.cfg_img_scale.unwrap_or(defaults.cfg_img_scale),
                "image.cfg_img_scale",
            )?,
            cfg_renorm_type: image_body
                .cfg_renorm_type
                .clone()
                .unwrap_or_else(|| defaults.cfg_renorm_type.into()),
            cfg_renorm_min: finite_or(
                image_body.cfg_renorm_min.unwrap_or(defaults.cfg_renorm_min),
                "image.cfg_renorm_min",
            )?,
            cfg_interval,
            timestep_shift: finite_or(
                image_body.timestep_shift.unwrap_or(defaults.timestep_shift),
                "image.timestep_shift",
            )?,
            height: resolution.height,
            width: resolution.width,
            seed: image_body.seed.or(body.seed).or(defaults.seed),
            negative_prompt,
            max_images,
            image_prompts: image_body.prompts,
            retain_images: image_body.retain_images.unwrap_or(true),
        })
    }

    fn resolve_sampling(
        &self,
        body: &NativeGenerateBody,
        mode: GenMode,
    ) -> Result<EngineSamplingParams, BuildError> {
        let temperature = finite_or(
            body.temperature.unwrap_or(defaults::DEFAULT_TEMPERATURE),
            "temperature",
        )?;
        if temperature < 0.0 {
            return Err(BuildError::new("temperature must be non-negative"));
        }
        let top_p = finite_or(body.top_p.unwrap_or(defaults::DEFAULT_TOP_P), "top_p")?;
        if !(0.0..=1.0).contains(&top_p) || top_p == 0.0 {
            return Err(BuildError::new("top_p must be in (0, 1]"));
        }
        let image_bias = body.image_bias.unwrap_or(0.0);
        let logit_bias = if mode == GenMode::AutoInterleave
            && image_bias != 0.0
            && self.profile.controls.start_of_image != 0
        {
            vec![(self.profile.controls.start_of_image, image_bias)]
        } else {
            Vec::new()
        };
        Ok(EngineSamplingParams {
            temperature,
            top_p,
            top_k: body.top_k.unwrap_or(defaults::DEFAULT_TOP_K),
            seed: body.seed,
            logit_bias,
            ..EngineSamplingParams::default()
        })
    }

    fn validate_max_images(&self, value: u16) -> Result<(), BuildError> {
        if value == 0 {
            return Err(BuildError::new("image.max_images must be positive"));
        }
        if value > self.profile.image_defaults.max_images_limit {
            return Err(BuildError::new(format!(
                "image.max_images exceeds profile limit {}",
                self.profile.image_defaults.max_images_limit
            )));
        }
        Ok(())
    }
}

pub fn parse_mode(value: &str) -> Result<GenMode, BuildError> {
    match value {
        "text" => Ok(GenMode::Text),
        "image" => Ok(GenMode::Image),
        "auto" | "auto_interleave" | "interleave" => Ok(GenMode::AutoInterleave),
        "understand" | "interleave_und" | "understanding" => Ok(GenMode::InterleaveUnd),
        other => Err(BuildError::new(format!("unsupported mode: {other}"))),
    }
}

fn validate_prompt(body: &NativeGenerateBody, mode: GenMode) -> Result<(), BuildError> {
    if body.prompt.trim().is_empty() {
        return Err(BuildError::new(format!(
            "{} mode requires a non-empty prompt",
            match mode {
                GenMode::Text => "text",
                GenMode::Image => "image",
                GenMode::AutoInterleave => "interleave",
                GenMode::InterleaveUnd => "understand",
            }
        )));
    }
    Ok(())
}

fn validate_cfg_interval(value: (f32, f32)) -> Result<(), BuildError> {
    let (lo, hi) = value;
    if !lo.is_finite() || !hi.is_finite() || lo < 0.0 || hi > 1.0 || lo > hi {
        return Err(BuildError::new(
            "image.cfg_interval must satisfy 0 <= lo <= hi <= 1",
        ));
    }
    Ok(())
}

fn finite_or(value: f32, name: &str) -> Result<f32, BuildError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(BuildError::new(format!("{name} must be finite")))
    }
}

pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use uniserve_text::tokenizer::{DynTokenizer, Tokenizer};

    use super::*;
    use crate::profiles::{NativeModelFamily, resolve_native_profile};

    #[derive(Debug)]
    struct SenseNovaTokenizer;

    impl Tokenizer for SenseNovaTokenizer {
        fn encode(
            &self,
            text: &str,
            _add_special_tokens: bool,
        ) -> uniserve_text::tokenizer::Result<Vec<u32>> {
            Ok(text.bytes().map(u32::from).collect())
        }

        fn decode(
            &self,
            token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> uniserve_text::tokenizer::Result<String> {
            Ok(
                String::from_utf8_lossy(&token_ids.iter().map(|id| *id as u8).collect::<Vec<_>>())
                    .into_owned(),
            )
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            match token {
                "<img>" => Some(151670),
                "</img>" => Some(151671),
                "<IMG_CONTEXT>" => Some(151672),
                "<|im_start|>" => Some(151644),
                "<|im_end|>" => Some(151645),
                _ => None,
            }
        }
    }

    #[test]
    fn sensenova_defaults_match_official_path() {
        let tok: DynTokenizer = Arc::new(SenseNovaTokenizer);
        let profile = resolve_native_profile(&*tok);
        assert_eq!(profile.family, NativeModelFamily::SenseNovaU1);
        let body = NativeGenerateBody {
            prompt: "Generate a travel guide covering Sonoma, Sequoia, Tahoe, and the Golden Gate."
                .into(),
            ..Default::default()
        };
        let request = NativeRequestBuilder::new(tok, &profile)
            .build(&body)
            .unwrap();
        assert_eq!(request.mode, GenMode::AutoInterleave);
        assert_eq!(request.image.width, 2048);
        assert_eq!(request.image.height, 1152);
        assert_eq!(request.image.steps, 50);
        assert_eq!(request.image.cfg_text_scale, 4.0);
        assert_eq!(request.image.cfg_img_scale, 1.0);
        assert_eq!(request.image.timestep_shift, 3.0);
        assert_eq!(request.image.seed, Some(42));
        assert_eq!(request.image.max_images, 4);
        assert!(request.image.retain_images);
        assert!(request.image.image_prompts.is_empty());
        assert_eq!(request.max_tokens, 32768);
    }

    #[test]
    fn unsupported_sensenova_preview_resolution_is_rejected() {
        let tok: DynTokenizer = Arc::new(SenseNovaTokenizer);
        let profile = resolve_native_profile(&*tok);
        let body = NativeGenerateBody {
            prompt: "paint".into(),
            mode: Some("image".into()),
            image: Some(super::super::schema::NativeImageBody {
                resolution: Some("preview".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(
            NativeRequestBuilder::new(tok, &profile)
                .build(&body)
                .is_err()
        );
    }

    #[test]
    fn client_image_values_override_profile_defaults() {
        let tok: DynTokenizer = Arc::new(SenseNovaTokenizer);
        let profile = resolve_native_profile(&*tok);
        let body = NativeGenerateBody {
            prompt: "paint".into(),
            mode: Some("image".into()),
            image: Some(super::super::schema::NativeImageBody {
                resolution: Some("1:1".into()),
                steps: Some(12),
                cfg_text_scale: Some(3.0),
                max_images: Some(2),
                seed: Some(7),
                ..Default::default()
            }),
            ..Default::default()
        };
        let request = NativeRequestBuilder::new(tok, &profile)
            .build(&body)
            .unwrap();
        assert_eq!((request.image.width, request.image.height), (1536, 1536));
        assert_eq!(request.image.steps, 12);
        assert_eq!(request.image.cfg_text_scale, 3.0);
        assert_eq!(request.image.max_images, 2);
        assert_eq!(request.image.seed, Some(7));
        assert!(request.image.retain_images);
        assert!(request.image.image_prompts.is_empty());
    }

    #[test]
    fn explicit_image_prompts_are_preserved() {
        let tok: DynTokenizer = Arc::new(SenseNovaTokenizer);
        let profile = resolve_native_profile(&*tok);
        let body = NativeGenerateBody {
            prompt: "Generate a travel guide".into(),
            mode: Some("interleave".into()),
            image: Some(super::super::schema::NativeImageBody {
                prompts: vec!["custom visual prompt".into()],
                ..Default::default()
            }),
            ..Default::default()
        };
        let request = NativeRequestBuilder::new(tok, &profile)
            .build(&body)
            .unwrap();

        assert_eq!(request.image.image_prompts, vec!["custom visual prompt"]);
    }

    #[test]
    fn invalid_interval_is_rejected() {
        let tok: DynTokenizer = Arc::new(SenseNovaTokenizer);
        let profile = resolve_native_profile(&*tok);
        let body = NativeGenerateBody {
            prompt: "paint".into(),
            mode: Some("image".into()),
            image: Some(super::super::schema::NativeImageBody {
                cfg_interval: Some([0.8, 0.2]),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(
            NativeRequestBuilder::new(tok, &profile)
                .build(&body)
                .is_err()
        );
    }
}
