use uniserve_engine_client::{
    EngineSamplingParams, GenerationConstraint, ImageParams, MmItem, NativeGenerateRequest,
};
use uniserve_text::tokenizer::DynTokenizer;

use super::defaults;
use super::profiles::{NativeModelProfile, NativePromptKind};
use super::resolution::resolve_resolution;
use super::schema::{NativeGenerateBody, NativeInputImage};

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

/// CFG and resolution constants for text output with image context.
///
/// Context-image requests use a fixed, model-agnostic visual-reasoning
/// configuration that intentionally differs from the per-profile image
/// generation defaults (`NativeModelProfile::image_defaults`).
mod context_image_defaults {
    /// Text-guidance CFG scale for visual reasoning.
    pub(super) const CFG_TEXT_SCALE: f32 = 4.0;
    /// Image-guidance CFG scale for visual reasoning.
    pub(super) const CFG_IMG_SCALE: f32 = 2.0;
    /// CFG renorm strategy for visual reasoning.
    pub(super) const CFG_RENORM_TYPE: &str = "text_channel";
    /// CFG renorm floor for visual reasoning.
    pub(super) const CFG_RENORM_MIN: f32 = 0.0;
    /// CFG interval (lo, hi) for visual reasoning.
    pub(super) const CFG_INTERVAL: (f32, f32) = (0.0, 1.0);
    /// Square latent resolution used for visual reasoning images.
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
        let constraint = body.constraint.unwrap_or_default();
        let input_images = body.input_images();
        let path = RequestPath::new(constraint, !input_images.is_empty());
        validate_prompt(body)?;
        if !self.profile.supports_constraint(constraint) {
            return Err(BuildError::new(format!(
                "constraint {} is not supported by this model profile",
                constraint.as_str()
            )));
        }
        if path.uses_feedback_ingest() {
            return self.build_und_with_images(body, constraint, &input_images);
        }

        let image = self.resolve_image_params(body, constraint)?;
        let negative_prompt = body.negative_prompt();
        let (prompt_ids, mm_items) = self.build_context_prompt(body, path.prompt, &input_images)?;
        let neg_prompt_ids = self
            .profile
            .build_negative_prompt_ids(&self.tokenizer, &negative_prompt);
        let sampling = self.resolve_sampling(body, constraint)?;

        Ok(NativeGenerateRequest {
            prompt_ids,
            neg_prompt_ids,
            sampling,
            image,
            constraint,
            max_tokens: body.max_tokens.unwrap_or(defaults::DEFAULT_MAX_TOKENS),
            mm_items,
            stop_token_ids: body.stop_token_ids.clone(),
        })
    }

    fn build_und_with_images(
        &self,
        body: &NativeGenerateBody,
        constraint: GenerationConstraint,
        input_images: &[NativeInputImage],
    ) -> Result<NativeGenerateRequest, BuildError> {
        if self.profile.context_markers_in_prompt() {
            return self.build_und_with_marked_images(body, constraint, input_images);
        }
        let first_image = input_images.first().ok_or_else(|| {
            BuildError::new("und_only requests with context images require image data")
        })?;
        let system = body
            .system_prompt
            .as_deref()
            .unwrap_or_else(|| self.profile.context_system_prompt());
        let mut sys_ids = self.profile.wrap_context_text(&self.tokenizer, system);
        let question_ids = self
            .profile
            .wrap_context_text(&self.tokenizer, &body.prompt);
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
            cfg_text_scale: context_image_defaults::CFG_TEXT_SCALE,
            cfg_img_scale: context_image_defaults::CFG_IMG_SCALE,
            cfg_renorm_type: context_image_defaults::CFG_RENORM_TYPE.into(),
            cfg_renorm_min: context_image_defaults::CFG_RENORM_MIN,
            cfg_interval: context_image_defaults::CFG_INTERVAL,
            timestep_shift: image_body
                .timestep_shift
                .unwrap_or(self.profile.image_defaults.timestep_shift),
            height: context_image_defaults::RESOLUTION,
            width: context_image_defaults::RESOLUTION,
            seed,
            negative_prompt: String::new(),
            max_images,
            image_prompts: image_body.prompts,
            retain_images: image_body.retain_images.unwrap_or(true),
        };
        Ok(NativeGenerateRequest {
            neg_prompt_ids: sys_ids.clone(),
            prompt_ids: sys_ids,
            sampling: self.resolve_sampling(body, constraint)?,
            image,
            constraint,
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

    fn build_und_with_marked_images(
        &self,
        body: &NativeGenerateBody,
        constraint: GenerationConstraint,
        input_images: &[NativeInputImage],
    ) -> Result<NativeGenerateRequest, BuildError> {
        if input_images.is_empty() {
            return Err(BuildError::new(
                "und_only requests with context images require image data",
            ));
        }
        let (prompt_ids, mm_items) =
            self.build_context_prompt(body, NativePromptKind::UndWithImages, input_images)?;
        let negative_prompt = body.negative_prompt();
        Ok(NativeGenerateRequest {
            neg_prompt_ids: self
                .profile
                .build_negative_prompt_ids(&self.tokenizer, &negative_prompt),
            prompt_ids,
            sampling: self.resolve_sampling(body, constraint)?,
            image: self.resolve_image_params(body, constraint)?,
            constraint,
            max_tokens: body.max_tokens.unwrap_or(defaults::DEFAULT_MAX_TOKENS),
            mm_items,
            stop_token_ids: body.stop_token_ids.clone(),
        })
    }

    fn build_context_prompt(
        &self,
        body: &NativeGenerateBody,
        kind: NativePromptKind,
        input_images: &[NativeInputImage],
    ) -> Result<(Vec<u32>, Vec<MmItem>), BuildError> {
        if input_images.is_empty() {
            return Ok((
                self.profile.build_prompt_ids(&self.tokenizer, body, kind),
                Vec::new(),
            ));
        }
        let controls = &self.profile.controls;
        if self.profile.context_markers_in_prompt() {
            if controls.start_of_image_text.is_empty() || controls.end_of_image_text.is_empty() {
                return Err(BuildError::new(
                    "model profile declares no image marker tokens for context images",
                ));
            }
            let user_text = prompt_with_image_markers(
                &body.prompt,
                input_images.len(),
                &controls.start_of_image_text,
                &controls.end_of_image_text,
            );
            let prompt_ids =
                self.profile
                    .build_prompt_ids_with_text(&self.tokenizer, body, kind, &user_text);
            let mut marker_positions = prompt_ids
                .iter()
                .enumerate()
                .filter(|&(_, &token)| token == controls.end_of_image)
                .map(|(index, _)| index as u32);
            let mut mm_items = Vec::with_capacity(input_images.len());
            for image in input_images {
                let position = marker_positions.next().ok_or_else(|| {
                    BuildError::new("context image prompt lost its image markers")
                })?;
                mm_items.push(mm_item(image, position));
            }
            return Ok((prompt_ids, mm_items));
        }
        let prompt_ids = self.profile.build_prompt_ids(&self.tokenizer, body, kind);
        let fallback_position = prompt_ids.len() as u32;
        let mm_items = input_images
            .iter()
            .map(|image| mm_item(image, image.position.unwrap_or(fallback_position)))
            .collect();
        Ok((prompt_ids, mm_items))
    }

    fn resolve_image_params(
        &self,
        body: &NativeGenerateBody,
        constraint: GenerationConstraint,
    ) -> Result<ImageParams, BuildError> {
        let image_body = body.image();
        let defaults = &self.profile.image_defaults;
        let resolution = resolve_resolution(
            &self.profile.resolution_policy,
            image_body
                .resolution
                .as_deref()
                .or(Some(defaults.resolution.as_str())),
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
                .unwrap_or_else(|| defaults.cfg_renorm_type.clone()),
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
            retain_images: image_body
                .retain_images
                .unwrap_or(constraint != GenerationConstraint::GenOnly),
        })
    }

    fn resolve_sampling(
        &self,
        body: &NativeGenerateBody,
        constraint: GenerationConstraint,
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
        let logit_bias = if constraint == GenerationConstraint::Default
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RequestPath {
    prompt: NativePromptKind,
    feedback_ingest: bool,
}

impl RequestPath {
    fn new(constraint: GenerationConstraint, has_input_images: bool) -> Self {
        match (constraint, has_input_images) {
            (GenerationConstraint::Default, _) => Self {
                prompt: NativePromptKind::Default,
                feedback_ingest: false,
            },
            (GenerationConstraint::UndOnly, true) => Self {
                prompt: NativePromptKind::UndWithImages,
                feedback_ingest: true,
            },
            (GenerationConstraint::UndOnly, false) => Self {
                prompt: NativePromptKind::Und,
                feedback_ingest: false,
            },
            (GenerationConstraint::GenOnly, _) => Self {
                prompt: NativePromptKind::Gen,
                feedback_ingest: false,
            },
        }
    }

    fn uses_feedback_ingest(self) -> bool {
        self.feedback_ingest
    }
}

fn validate_prompt(body: &NativeGenerateBody) -> Result<(), BuildError> {
    if body.prompt.trim().is_empty() {
        return Err(BuildError::new("generation requires a non-empty prompt"));
    }
    Ok(())
}

fn prompt_with_image_markers(
    prompt: &str,
    image_count: usize,
    start_marker: &str,
    end_marker: &str,
) -> String {
    let mut user_text = String::new();
    if image_count == 1 {
        user_text.push_str(start_marker);
        user_text.push_str(end_marker);
        user_text.push('\n');
    } else {
        for index in 0..image_count {
            user_text.push_str(&format!(
                "Image-{}:{}{}\n",
                index + 1,
                start_marker,
                end_marker
            ));
        }
    }
    user_text.push_str(prompt);
    user_text
}

fn mm_item(image: &NativeInputImage, position: u32) -> MmItem {
    MmItem {
        hash: fnv1a(image.b64.as_bytes()),
        position,
        num_tokens: image.num_tokens.unwrap_or(0),
        b64: image.b64.clone(),
    }
}

fn validate_cfg_interval(value: (f32, f32)) -> Result<(), BuildError> {
    let (lo, hi) = value;
    if !lo.is_finite() || !hi.is_finite() || lo > hi {
        return Err(BuildError::new(
            "image.cfg_interval must be a finite ordered pair",
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
    use crate::profiles::resolve_native_profile_for_model;

    #[derive(Debug)]
    struct SenseNovaTokenizer;

    impl Tokenizer for SenseNovaTokenizer {
        fn encode(
            &self,
            text: &str,
            _add_special_tokens: bool,
        ) -> uniserve_text::tokenizer::Result<Vec<u32>> {
            // Added tokens encode atomically (as the real tokenizer does);
            // everything else byte-encodes.
            const MARKERS: [(&str, u32); 2] = [("<img>", 151670), ("</img>", 151671)];
            let mut ids = Vec::new();
            let mut rest = text;
            'outer: while !rest.is_empty() {
                for (marker, id) in MARKERS {
                    if let Some(stripped) = rest.strip_prefix(marker) {
                        ids.push(id);
                        rest = stripped;
                        continue 'outer;
                    }
                }
                let mut chars = rest.chars();
                let ch = chars.next().expect("non-empty");
                let mut buf = [0u8; 4];
                ids.extend(ch.encode_utf8(&mut buf).bytes().map(u32::from));
                rest = chars.as_str();
            }
            Ok(ids)
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
                "<|im_start|>" => Some(151644),
                "<|im_end|>" => Some(151645),
                _ => None,
            }
        }
    }

    #[test]
    fn sensenova_defaults_match_official_path() {
        let tok: DynTokenizer = Arc::new(SenseNovaTokenizer);
        let profile = resolve_native_profile_for_model("sensenova-u1", &*tok);
        assert_eq!(profile.id, "sensenova-u1");
        let body = NativeGenerateBody {
            prompt: "Generate a travel guide covering Sonoma, Sequoia, Tahoe, and the Golden Gate."
                .into(),
            ..Default::default()
        };
        let request = NativeRequestBuilder::new(tok, &profile)
            .build(&body)
            .unwrap();
        assert_eq!(request.constraint, GenerationConstraint::Default);
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
        let rendered = request
            .prompt_ids
            .iter()
            .filter_map(|id| u8::try_from(*id).ok())
            .map(char::from)
            .collect::<String>();
        assert!(rendered.contains("MUST interleave text with generated images"));
        assert!(!rendered.contains("MAY place generated images"));
    }

    #[test]
    fn sensenova_context_images_build_marked_mm_items() {
        let tok: DynTokenizer = Arc::new(SenseNovaTokenizer);
        let profile = resolve_native_profile_for_model("sensenova-u1", &*tok);
        let body = NativeGenerateBody {
            prompt: "Describe this image in detail.".into(),
            constraint: Some(GenerationConstraint::UndOnly),
            input_image_b64: Some("aGVsbG8=".into()),
            ..Default::default()
        };
        let request = NativeRequestBuilder::new(tok, &profile)
            .build(&body)
            .unwrap();
        assert_eq!(request.constraint, GenerationConstraint::UndOnly);
        assert_eq!(request.mm_items.len(), 1);
        let position = request.mm_items[0].position as usize;
        // The encode gap sits exactly between the in-prompt markers.
        assert_eq!(
            request.prompt_ids[position], 151671,
            "position is the </img> token"
        );
        assert_eq!(
            request.prompt_ids[position - 1],
            151670,
            "preceded by <img>"
        );
        // No CFG branch is needed for a text-only answer.
        assert!(request.neg_prompt_ids.is_empty());
        // Context-image requests still use the profile resolution bucket.
        assert_eq!(request.image.width, 2048);
        assert_eq!(request.image.height, 1152);
    }

    #[test]
    fn sensenova_und_only_multi_image_positions_are_ordered() {
        let tok: DynTokenizer = Arc::new(SenseNovaTokenizer);
        let profile = resolve_native_profile_for_model("sensenova-u1", &*tok);
        let body = NativeGenerateBody {
            prompt: "Compare the two images.".into(),
            constraint: Some(GenerationConstraint::UndOnly),
            input_images: vec![
                super::super::schema::NativeInputImage {
                    b64: "aQ==".into(),
                    position: None,
                    num_tokens: None,
                },
                super::super::schema::NativeInputImage {
                    b64: "ag==".into(),
                    position: None,
                    num_tokens: None,
                },
            ],
            ..Default::default()
        };
        let request = NativeRequestBuilder::new(tok, &profile)
            .build(&body)
            .unwrap();
        assert_eq!(request.mm_items.len(), 2);
        let first = request.mm_items[0].position as usize;
        let second = request.mm_items[1].position as usize;
        assert!(first < second);
        assert_eq!(request.prompt_ids[first], 151671);
        assert_eq!(request.prompt_ids[second], 151671);
        assert_ne!(request.mm_items[0].hash, request.mm_items[1].hash);
    }

    #[test]
    fn unsupported_sensenova_preview_resolution_is_rejected() {
        let tok: DynTokenizer = Arc::new(SenseNovaTokenizer);
        let profile = resolve_native_profile_for_model("sensenova-u1", &*tok);
        let body = NativeGenerateBody {
            prompt: "paint".into(),
            constraint: Some(GenerationConstraint::GenOnly),
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
        let profile = resolve_native_profile_for_model("sensenova-u1", &*tok);
        let body = NativeGenerateBody {
            prompt: "paint".into(),
            constraint: Some(GenerationConstraint::GenOnly),
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
        assert!(!request.image.retain_images);
        assert!(request.image.image_prompts.is_empty());
    }

    #[test]
    fn gen_only_respects_explicit_retain_images() {
        let tok: DynTokenizer = Arc::new(SenseNovaTokenizer);
        let profile = resolve_native_profile_for_model("sensenova-u1", &*tok);
        let body = NativeGenerateBody {
            prompt: "paint".into(),
            constraint: Some(GenerationConstraint::GenOnly),
            image: Some(super::super::schema::NativeImageBody {
                retain_images: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        };
        let request = NativeRequestBuilder::new(tok, &profile)
            .build(&body)
            .unwrap();
        assert!(request.image.retain_images);
    }

    #[test]
    fn explicit_image_prompts_are_preserved() {
        let tok: DynTokenizer = Arc::new(SenseNovaTokenizer);
        let profile = resolve_native_profile_for_model("sensenova-u1", &*tok);
        let body = NativeGenerateBody {
            prompt: "Generate a travel guide".into(),
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
        let profile = resolve_native_profile_for_model("sensenova-u1", &*tok);
        let body = NativeGenerateBody {
            prompt: "paint".into(),
            constraint: Some(GenerationConstraint::GenOnly),
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

    #[test]
    fn cfg_interval_can_cover_the_full_timestep_domain() {
        let tok: DynTokenizer = Arc::new(SenseNovaTokenizer);
        let profile = resolve_native_profile_for_model("sensenova-u1", &*tok);
        let body = NativeGenerateBody {
            prompt: "paint".into(),
            constraint: Some(GenerationConstraint::GenOnly),
            image: Some(super::super::schema::NativeImageBody {
                cfg_interval: Some([-1.0, 2.0]),
                ..Default::default()
            }),
            ..Default::default()
        };
        let request = NativeRequestBuilder::new(tok, &profile)
            .build(&body)
            .unwrap();

        assert_eq!(request.image.cfg_interval, (-1.0, 2.0));
    }
}
