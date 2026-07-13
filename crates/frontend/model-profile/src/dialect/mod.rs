use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uniserve_core::{
    GenOnlyStartPolicyDescriptor, GeneratedImageFeedbackRecipe, GenerationConstraint,
    GenerationPolicyDescriptor, ImageIngestRecipe, ImageIngestStep, ImageKvEffect,
    TerminationPolicyDescriptor, TriggerPolicyDescriptor, VisibilityPolicyDescriptor,
};

use self::resolution::{ResolutionBucket, ResolutionPolicy};
use crate::assets::{self, Error as AssetError};
use crate::tokenizer::{DynTokenizer, Tokenizer};

pub mod resolution;

const BAGEL_PROFILE_JSON: &str = include_str!("../../profiles/bagel.json");
const SENSENOVA_PROFILE_JSON: &str = include_str!("../../profiles/sensenova-u1.json");
const THINKMORPH_PROFILE_JSON: &str = include_str!("../../profiles/thinkmorph.json");

/// Protocol-neutral prompt inputs consumed by a generation dialect renderer.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptContext {
    pub prompt: String,
    pub system_prompt: Option<String>,
    pub assistant_prefix: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationControls {
    pub bos: u32,
    pub eos: u32,
    pub start_of_image: u32,
    pub end_of_image: u32,
    pub image_start_ids: Vec<u32>,
    /// Tokenizer-resolved literal text of the image begin/end markers, for
    /// profiles that carry input-image markers inside the prompt stream.
    pub start_of_image_text: String,
    pub end_of_image_text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageGenerationDefaults {
    pub resolution: String,
    pub steps: u16,
    pub cfg_text_scale: f32,
    pub cfg_img_scale: f32,
    pub cfg_renorm_type: String,
    pub cfg_renorm_min: f32,
    pub cfg_interval: (f32, f32),
    pub timestep_shift: f32,
    pub seed: Option<u64>,
    pub max_images: u16,
    pub max_images_limit: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DelimitedTextPolicy {
    pub start: String,
    pub end: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputFilterPolicy {
    pub reasoning: Option<DelimitedTextPolicy>,
    pub visible_wrappers: Vec<DelimitedTextPolicy>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum PromptRecipe {
    Chatml {
        default_system: Option<String>,
        default_assistant_prefix: String,
    },
    BagelText,
    BagelImage,
    BagelDefault {
        default_system: String,
    },
    Raw,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct PromptRecipes {
    default: PromptRecipe,
    und: PromptRecipe,
    und_with_images: PromptRecipe,
    r#gen: PromptRecipe,
    negative: PromptRecipe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromptKind {
    Default,
    Und,
    UndWithImages,
    Gen,
}

impl PromptKind {
    pub fn for_request(constraint: GenerationConstraint, has_input_images: bool) -> Self {
        match constraint {
            GenerationConstraint::Default => Self::Default,
            GenerationConstraint::UndOnly if has_input_images => Self::UndWithImages,
            GenerationConstraint::UndOnly => Self::Und,
            GenerationConstraint::GenOnly => Self::Gen,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatPromptScaffold {
    pub default_system: Option<String>,
    pub assistant_prefix: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationDialectProfile {
    pub id: String,
    pub controls: GenerationControls,
    pub image_defaults: ImageGenerationDefaults,
    pub resolution_policy: ResolutionPolicy,
    pub output_filter: OutputFilterPolicy,
    pub image_ingest: ImageIngestRecipe,
    pub generation_policy: GenerationPolicyDescriptor,
    image_ingest_estimators: Vec<ImageKvEstimator>,
    supported_constraints: Vec<GenerationConstraint>,
    prompts: PromptRecipes,
    context_system_prompt: String,
    context_markers_in_prompt: bool,
}

impl GenerationDialectProfile {
    pub fn supports_constraint(&self, constraint: GenerationConstraint) -> bool {
        self.supported_constraints.contains(&constraint)
    }

    pub fn context_system_prompt(&self) -> &str {
        &self.context_system_prompt
    }

    pub fn image_ingest_requires_dimensions(&self) -> bool {
        self.image_ingest
            .step_kv_tokens
            .contains(&ImageKvEffect::WorkerDefined)
    }

    pub fn image_ingest_for_dimensions(
        &self,
        width: u32,
        height: u32,
        image_count: usize,
    ) -> assets::Result<ImageIngestRecipe> {
        let mut recipe = self.image_ingest.clone();
        if !self.image_ingest_requires_dimensions() {
            return Ok(recipe);
        }
        if width == 0 || height == 0 {
            return Err(AssetError::message(
                "input image dimensions must be positive",
            ));
        }
        if self.image_ingest_estimators.len() != recipe.steps.len() {
            return Err(AssetError::message(format!(
                "generation profile {:?} has no exact image-KV estimator for every ingest step",
                self.id
            )));
        }
        for (index, effect) in recipe.step_kv_tokens.iter_mut().enumerate() {
            if *effect == ImageKvEffect::WorkerDefined {
                let tokens =
                    self.image_ingest_estimators[index].tokens(width, height, image_count)?;
                *effect = ImageKvEffect::Exact { tokens };
            }
        }
        Ok(recipe)
    }

    /// Whether input images ride as markers inside the prompt token stream.
    pub fn context_markers_in_prompt(&self) -> bool {
        self.context_markers_in_prompt
    }

    pub fn chat_prompt_scaffold(&self, kind: PromptKind) -> ChatPromptScaffold {
        let recipe = match kind {
            PromptKind::Default => &self.prompts.default,
            PromptKind::Und => &self.prompts.und,
            PromptKind::UndWithImages => &self.prompts.und_with_images,
            PromptKind::Gen => &self.prompts.r#gen,
        };
        match recipe {
            PromptRecipe::Chatml {
                default_system,
                default_assistant_prefix,
            } => ChatPromptScaffold {
                default_system: default_system.clone(),
                assistant_prefix: default_assistant_prefix.clone(),
            },
            PromptRecipe::BagelText => ChatPromptScaffold {
                default_system: None,
                assistant_prefix: String::new(),
            },
            PromptRecipe::BagelDefault { default_system } => ChatPromptScaffold {
                default_system: Some(default_system.clone()),
                assistant_prefix: String::new(),
            },
            PromptRecipe::BagelImage | PromptRecipe::Raw => ChatPromptScaffold {
                default_system: None,
                assistant_prefix: String::new(),
            },
        }
    }

    pub fn build_prompt_ids(
        &self,
        tok: &DynTokenizer,
        body: &PromptContext,
        kind: PromptKind,
    ) -> crate::tokenizer::Result<Vec<u32>> {
        self.build_prompt_ids_with_text(tok, body, kind, &body.prompt)
    }

    pub fn build_prompt_ids_with_text(
        &self,
        tok: &DynTokenizer,
        body: &PromptContext,
        kind: PromptKind,
        text: &str,
    ) -> crate::tokenizer::Result<Vec<u32>> {
        let recipe = match kind {
            PromptKind::Default => &self.prompts.default,
            PromptKind::Und => &self.prompts.und,
            PromptKind::UndWithImages => &self.prompts.und_with_images,
            PromptKind::Gen => &self.prompts.r#gen,
        };
        render_prompt(tok, &self.controls, recipe, body, text)
    }

    pub fn build_negative_prompt_ids(
        &self,
        tok: &DynTokenizer,
        negative_prompt: &str,
    ) -> crate::tokenizer::Result<Vec<u32>> {
        if negative_prompt.is_empty() {
            return Ok(Vec::new());
        }
        let body = PromptContext {
            prompt: negative_prompt.to_string(),
            ..Default::default()
        };
        render_prompt(
            tok,
            &self.controls,
            &self.prompts.negative,
            &body,
            negative_prompt,
        )
    }

    pub fn wrap_context_text(
        &self,
        tok: &DynTokenizer,
        text: &str,
    ) -> crate::tokenizer::Result<Vec<u32>> {
        let mut ids = vec![self.controls.bos];
        ids.extend(encode(tok, text)?);
        ids.push(self.controls.eos);
        Ok(ids)
    }
}

/// Resolve the selected generation dialect for a model. Models without a
/// repository manifest or recognized family have no image-generation contract.
pub fn resolve_generation_dialect_for_model(
    model_ref: &str,
    tokenizer: &dyn Tokenizer,
) -> assets::Result<Option<GenerationDialectProfile>> {
    if let Some(profile) = profile_from_model_manifest(model_ref, tokenizer)? {
        return Ok(Some(profile));
    }
    profile_key_from_model(model_ref)?
        .map(|key| profile_from_key(key, tokenizer))
        .transpose()
}

fn profile_from_key(
    key: &str,
    tokenizer: &dyn Tokenizer,
) -> assets::Result<GenerationDialectProfile> {
    let json = match key {
        "sensenova-u1" => SENSENOVA_PROFILE_JSON,
        "thinkmorph" => THINKMORPH_PROFILE_JSON,
        "bagel" => BAGEL_PROFILE_JSON,
        other => {
            return Err(AssetError::message(format!(
                "unknown generation profile {other:?}"
            )));
        }
    };
    profile_from_manifest(parse_builtin_manifest(json)?, tokenizer)
}

fn parse_builtin_manifest(json: &str) -> assets::Result<ProfileManifest> {
    serde_json::from_str(json).map_err(|error| {
        AssetError::message(format!("invalid built-in generation profile: {error}"))
    })
}

fn profile_from_model_manifest(
    model_ref: &str,
    tokenizer: &dyn Tokenizer,
) -> assets::Result<Option<GenerationDialectProfile>> {
    let path = Path::new(model_ref);
    if !path.is_dir() {
        return Ok(None);
    }
    let manifest_path = path.join("uniserve_profile.json");
    if !manifest_path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(&manifest_path).map_err(|error| {
        AssetError::message(format!(
            "failed to read generation profile {}: {error}",
            manifest_path.display()
        ))
    })?;
    let manifest = serde_json::from_str::<ProfileManifest>(&text).map_err(|error| {
        AssetError::message(format!(
            "invalid generation profile {}: {error}",
            manifest_path.display()
        ))
    })?;
    profile_from_manifest(manifest, tokenizer).map(Some)
}

fn profile_from_manifest(
    manifest: ProfileManifest,
    tokenizer: &dyn Tokenizer,
) -> assets::Result<GenerationDialectProfile> {
    let controls = controls_from_manifest(&manifest.control_tokens, tokenizer)?;
    let id = manifest.id;
    if id.trim().is_empty() {
        return Err(AssetError::message(
            "generation profile id must not be empty",
        ));
    }
    let ImageIngestManifest {
        steps,
        logical_positions,
        step_kv_tokens,
        step_estimators,
    } = manifest.image_ingest;
    let image_ingest = ImageIngestRecipe {
        steps,
        logical_positions,
        step_kv_tokens,
        modality: uniserve_core::Modality::Und,
    };
    if image_ingest.steps.is_empty() {
        return Err(AssetError::message(format!(
            "generation profile {id:?} declares an empty image ingest recipe"
        )));
    }
    if image_ingest.steps.len() != image_ingest.step_kv_tokens.len() {
        return Err(AssetError::message(format!(
            "generation profile {id:?} must declare one KV effect per image ingest step"
        )));
    }
    if image_ingest
        .step_kv_tokens
        .contains(&ImageKvEffect::WorkerDefined)
        && step_estimators.len() != image_ingest.steps.len()
    {
        return Err(AssetError::message(format!(
            "generation profile {id:?} must declare one exact KV estimator per image ingest step"
        )));
    }
    let GenerationPolicyManifest {
        trigger,
        gen_only_start,
        visibility,
        termination,
        feedback,
    } = manifest.generation_policy;
    let trigger = trigger.resolve(&controls)?;
    let generation_policy = GenerationPolicyDescriptor {
        trigger,
        gen_only_start,
        visibility,
        termination,
        feedback,
    };
    let supported_constraints = manifest
        .supported_constraints
        .iter()
        .map(|constraint| parse_profile_constraint(constraint))
        .collect::<assets::Result<Vec<_>>>()?;
    if supported_constraints.is_empty() {
        return Err(AssetError::message(format!(
            "generation profile {id:?} declares no supported constraints"
        )));
    }
    let mut distinct_constraints = supported_constraints.clone();
    distinct_constraints.sort_by_key(|constraint| constraint.as_str());
    distinct_constraints.dedup();
    if distinct_constraints.len() != supported_constraints.len() {
        return Err(AssetError::message(format!(
            "generation profile {id:?} repeats a supported constraint"
        )));
    }
    if supported_constraints.contains(&GenerationConstraint::Default)
        && !matches!(generation_policy.trigger, TriggerPolicyDescriptor::Disabled)
        && generation_policy.termination.gen_commit_continues_default
        && generation_policy.feedback.as_ref().is_none_or(|feedback| {
            matches!(
                feedback.writeback,
                uniserve_core::FeedbackWriteback::Disabled
            )
        })
    {
        return Err(AssetError::message(format!(
            "generation profile {id:?} enables default Gen continuation without feedback"
        )));
    }
    if let Some(GeneratedImageFeedbackRecipe {
        writeback: uniserve_core::FeedbackWriteback::Reingest { ingest },
        ..
    }) = generation_policy.feedback.as_ref()
        && ingest.steps.is_empty()
    {
        return Err(AssetError::message(format!(
            "generation profile {id:?} declares an empty feedback ingest recipe"
        )));
    }
    let image_defaults: ImageGenerationDefaults = manifest.image_defaults.into();
    validate_image_defaults(&id, &image_defaults)?;
    let resolution_policy: ResolutionPolicy = manifest.resolution.try_into()?;
    if !resolution_policy
        .buckets
        .iter()
        .any(|bucket| bucket.name.eq_ignore_ascii_case(&image_defaults.resolution))
    {
        return Err(AssetError::message(format!(
            "generation profile {id:?} image default resolution {:?} has no matching bucket",
            image_defaults.resolution
        )));
    }
    let output_filter = manifest.output_filter.try_into()?;
    Ok(GenerationDialectProfile {
        id,
        controls,
        image_defaults,
        resolution_policy,
        output_filter,
        image_ingest,
        generation_policy,
        image_ingest_estimators: step_estimators,
        supported_constraints,
        prompts: manifest.prompts.try_into()?,
        context_system_prompt: manifest.context_system_prompt,
        context_markers_in_prompt: manifest.context_images.markers_in_prompt,
    })
}

fn controls_from_manifest(
    spec: &ControlTokenManifest,
    tokenizer: &dyn Tokenizer,
) -> assets::Result<GenerationControls> {
    let (start_of_image, start_of_image_text) =
        required_token(tokenizer, &spec.start_of_image, "start_of_image")?;
    let (end_of_image, end_of_image_text) =
        required_token(tokenizer, &spec.end_of_image, "end_of_image")?;
    let image_start_ids = spec
        .image_start_text
        .as_ref()
        .map(|text| {
            tokenizer.encode(text, false).map_err(|error| {
                AssetError::message(format!(
                    "failed to encode generation profile image_start_text: {error}"
                ))
            })
        })
        .transpose()?
        .unwrap_or_default();
    Ok(GenerationControls {
        bos: required_token_id(tokenizer, &spec.bos, "bos")?,
        eos: required_token_id(tokenizer, &spec.eos, "eos")?,
        start_of_image,
        end_of_image,
        image_start_ids,
        start_of_image_text,
        end_of_image_text,
    })
}

fn required_token(
    tokenizer: &dyn Tokenizer,
    candidates: &[String],
    role: &str,
) -> assets::Result<(u32, String)> {
    candidates
        .iter()
        .find_map(|token| tokenizer.token_to_id(token).map(|id| (id, token.clone())))
        .ok_or_else(|| {
            AssetError::message(format!(
                "generation profile {role} control token is missing from the tokenizer"
            ))
        })
}

fn required_token_id(
    tokenizer: &dyn Tokenizer,
    candidates: &[String],
    role: &str,
) -> assets::Result<u32> {
    required_token(tokenizer, candidates, role).map(|(id, _)| id)
}

fn profile_key_from_model(model_ref: &str) -> assets::Result<Option<&'static str>> {
    let path = Path::new(model_ref);
    if path.is_dir() {
        let config_path = path.join("config.json");
        if config_path.exists() {
            let text = fs::read_to_string(&config_path).map_err(|error| {
                AssetError::message(format!(
                    "failed to read model config {}: {error}",
                    config_path.display()
                ))
            })?;
            let config = serde_json::from_str::<Value>(&text).map_err(|error| {
                AssetError::message(format!(
                    "invalid model config {}: {error}",
                    config_path.display()
                ))
            })?;
            if let Some(key) = profile_key_from_config(&config) {
                return Ok(Some(key));
            }
        }
    }
    let lower = model_ref.to_ascii_lowercase();
    if lower.contains("sensenova") || lower.contains("neo_chat") || lower.contains("neo-unify") {
        Ok(Some("sensenova-u1"))
    } else if lower.contains("thinkmorph") || lower.contains("think-morph") {
        Ok(Some("thinkmorph"))
    } else if lower.contains("bagel") {
        Ok(Some("bagel"))
    } else {
        Ok(None)
    }
}

fn profile_key_from_config(config: &Value) -> Option<&'static str> {
    let model_type = config
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase();
    let architectures: Vec<String> = config
        .get("architectures")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_ascii_lowercase)
        .collect();
    if model_type == "neo_chat"
        || model_type == "neo_unify"
        || architectures.iter().any(|arch| {
            matches!(
                arch.as_str(),
                "neochatmodel" | "neo_chat" | "neo-unify" | "neo_unify"
            )
        })
    {
        return Some("sensenova-u1");
    }
    if model_type.contains("thinkmorph")
        || architectures
            .iter()
            .any(|arch| arch.contains("thinkmorph") || arch.contains("think_morph"))
    {
        return Some("thinkmorph");
    }
    if model_type.contains("bagel") || architectures.iter().any(|arch| arch.contains("bagel")) {
        return Some("bagel");
    }
    None
}

fn render_prompt(
    tok: &DynTokenizer,
    controls: &GenerationControls,
    recipe: &PromptRecipe,
    body: &PromptContext,
    prompt: &str,
) -> crate::tokenizer::Result<Vec<u32>> {
    match recipe {
        PromptRecipe::Chatml {
            default_system,
            default_assistant_prefix,
        } => {
            let system = body.system_prompt.as_deref().or(default_system.as_deref());
            let assistant_prefix = body
                .assistant_prefix
                .as_deref()
                .unwrap_or(default_assistant_prefix);
            encode(tok, &chatml(system, prompt, assistant_prefix))
        }
        PromptRecipe::BagelText => encode(
            tok,
            &format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}",
                prompt,
                body.assistant_prefix.as_deref().unwrap_or("")
            ),
        ),
        PromptRecipe::BagelImage => {
            let mut ids = vec![controls.bos];
            ids.extend(encode(tok, prompt)?);
            ids.push(controls.eos);
            Ok(ids)
        }
        PromptRecipe::BagelDefault { default_system } => encode(
            tok,
            &format!(
                "<|im_start|>{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}",
                body.system_prompt.as_deref().unwrap_or(default_system),
                prompt,
                body.assistant_prefix.as_deref().unwrap_or("")
            ),
        ),
        PromptRecipe::Raw => encode(tok, prompt),
    }
}

fn encode(tok: &DynTokenizer, text: &str) -> crate::tokenizer::Result<Vec<u32>> {
    tok.encode(text, false)
}

fn chatml(system: Option<&str>, user: &str, assistant_suffix: &str) -> String {
    let mut out = String::new();
    if let Some(system) = system {
        out.push_str("<|im_start|>system\n");
        out.push_str(system);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>user\n");
    out.push_str(user);
    out.push_str("<|im_end|>\n<|im_start|>assistant\n");
    out.push_str(assistant_suffix);
    out
}

fn parse_profile_constraint(value: &str) -> assets::Result<GenerationConstraint> {
    value.parse().map_err(|error| {
        AssetError::message(format!(
            "invalid generation profile constraint {value:?}: {error}"
        ))
    })
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileManifest {
    id: String,
    control_tokens: ControlTokenManifest,
    supported_constraints: Vec<String>,
    image_defaults: ImageDefaultsManifest,
    resolution: ResolutionManifest,
    output_filter: OutputFilterManifest,
    prompts: PromptManifestSet,
    context_system_prompt: String,
    image_ingest: ImageIngestManifest,
    generation_policy: GenerationPolicyManifest,
    #[serde(default)]
    context_images: ContextImageManifest,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ImageIngestManifest {
    steps: Vec<ImageIngestStep>,
    logical_positions: u32,
    step_kv_tokens: Vec<ImageKvEffect>,
    #[serde(default)]
    step_estimators: Vec<ImageKvEstimator>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
enum ImageKvEstimator {
    ResizeChain {
        transforms: Vec<StrideResize>,
        token_stride: u32,
        marker_tokens: u32,
    },
    PixelBounds {
        factor: u32,
        min_pixels: u64,
        max_pixels: u64,
        #[serde(default)]
        shared_pixel_budget: Option<u64>,
        token_stride: u32,
        marker_tokens: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StrideResize {
    max_side: u32,
    min_side: u32,
    stride: u32,
    max_pixels: u64,
}

impl ImageKvEstimator {
    fn tokens(&self, width: u32, height: u32, image_count: usize) -> assets::Result<u32> {
        let (width, height, token_stride, marker_tokens) = match self {
            Self::ResizeChain {
                transforms,
                token_stride,
                marker_tokens,
            } => {
                let mut dimensions = (width, height);
                for transform in transforms {
                    dimensions = transform.resize(dimensions.0, dimensions.1)?;
                }
                (dimensions.0, dimensions.1, *token_stride, *marker_tokens)
            }
            Self::PixelBounds {
                factor,
                min_pixels,
                max_pixels,
                shared_pixel_budget,
                token_stride,
                marker_tokens,
            } => {
                let max_pixels = shared_pixel_budget.map_or(*max_pixels, |budget| {
                    (*max_pixels).min(budget / image_count.max(1) as u64)
                });
                let dimensions =
                    pixel_bound_resize(width, height, *factor, *min_pixels, max_pixels)?;
                (dimensions.0, dimensions.1, *token_stride, *marker_tokens)
            }
        };
        if token_stride == 0
            || !width.is_multiple_of(token_stride)
            || !height.is_multiple_of(token_stride)
        {
            return Err(AssetError::message(
                "image-KV estimator produced dimensions outside its token stride",
            ));
        }
        let tokens = u64::from(width / token_stride)
            .saturating_mul(u64::from(height / token_stride))
            .saturating_add(u64::from(marker_tokens));
        u32::try_from(tokens)
            .map_err(|_| AssetError::message("image-KV token count exceeds the engine range"))
    }
}

impl StrideResize {
    fn resize(self, width: u32, height: u32) -> assets::Result<(u32, u32)> {
        if self.max_side == 0 || self.min_side == 0 || self.stride == 0 || self.max_pixels == 0 {
            return Err(AssetError::message(
                "stride-resize geometry must be positive",
            ));
        }
        let mut scale = (f64::from(self.max_side) / f64::from(width.max(height))).min(1.0);
        scale = scale.max(f64::from(self.min_side) / f64::from(width.min(height)));
        let mut resized = scale_to_stride(width, height, scale, self.stride);
        if u64::from(resized.0).saturating_mul(u64::from(resized.1)) > self.max_pixels {
            scale = self.max_pixels as f64 / (f64::from(resized.0) * f64::from(resized.1));
            resized = scale_to_stride(resized.0, resized.1, scale, self.stride);
        }
        if resized.0.max(resized.1) > self.max_side {
            scale = f64::from(self.max_side) / f64::from(resized.0.max(resized.1));
            resized = scale_to_stride(resized.0, resized.1, scale, self.stride);
        }
        Ok(resized)
    }
}

fn scale_to_stride(width: u32, height: u32, scale: f64, stride: u32) -> (u32, u32) {
    let scale_one = |value: u32| {
        let scaled = (f64::from(value) * scale).round_ties_even();
        let aligned = (scaled / f64::from(stride)).round_ties_even() * f64::from(stride);
        stride.max(aligned.max(f64::from(stride)) as u32)
    };
    (scale_one(width), scale_one(height))
}

fn pixel_bound_resize(
    width: u32,
    height: u32,
    factor: u32,
    min_pixels: u64,
    max_pixels: u64,
) -> assets::Result<(u32, u32)> {
    if factor == 0 || min_pixels == 0 || max_pixels < min_pixels {
        return Err(AssetError::message(
            "pixel-bound resize geometry is invalid",
        ));
    }
    let aspect = f64::from(width.max(height)) / f64::from(width.min(height));
    if aspect > 200.0 {
        return Err(AssetError::message(
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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenerationPolicyManifest {
    trigger: TriggerManifest,
    #[serde(default)]
    gen_only_start: GenOnlyStartPolicyDescriptor,
    #[serde(default)]
    visibility: VisibilityPolicyDescriptor,
    #[serde(default)]
    termination: TerminationPolicyDescriptor,
    feedback: Option<GeneratedImageFeedbackRecipe>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
enum TriggerManifest {
    Disabled,
    StartOfImage,
    ImageStartSuffix,
    RoundCloseThenImageStartSuffix,
}

impl TriggerManifest {
    fn resolve(self, controls: &GenerationControls) -> assets::Result<TriggerPolicyDescriptor> {
        Ok(match self {
            Self::Disabled => TriggerPolicyDescriptor::Disabled,
            Self::StartOfImage => TriggerPolicyDescriptor::Token {
                token_id: controls.start_of_image,
            },
            Self::ImageStartSuffix => {
                if controls.image_start_ids.is_empty() {
                    return Err(AssetError::message(
                        "generation profile image-start suffix encodes to no tokens",
                    ));
                }
                TriggerPolicyDescriptor::Suffix {
                    token_ids: controls.image_start_ids.clone(),
                }
            }
            Self::RoundCloseThenImageStartSuffix => {
                if controls.image_start_ids.is_empty() {
                    return Err(AssetError::message(
                        "generation profile round-close image-start suffix encodes to no tokens",
                    ));
                }
                TriggerPolicyDescriptor::RoundCloseThenSuffix {
                    close_token_ids: vec![controls.eos],
                    trigger_token_ids: controls.image_start_ids.clone(),
                }
            }
        })
    }
}

/// Input-image request-construction policy.
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct ContextImageManifest {
    /// When true the image begin/end markers are ordinary prompt tokens and
    /// the encode op fills the gap between them (one shared temporal RoPE
    /// index per image). When false the worker emits the markers during encode.
    #[serde(default)]
    markers_in_prompt: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ControlTokenManifest {
    bos: Vec<String>,
    eos: Vec<String>,
    start_of_image: Vec<String>,
    end_of_image: Vec<String>,
    #[serde(default)]
    image_start_text: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ImageDefaultsManifest {
    resolution: String,
    steps: u16,
    cfg_text_scale: f32,
    cfg_img_scale: f32,
    cfg_renorm_type: String,
    cfg_renorm_min: f32,
    cfg_interval: [f32; 2],
    timestep_shift: f32,
    seed: Option<u64>,
    max_images: u16,
    max_images_limit: u16,
}

impl From<ImageDefaultsManifest> for ImageGenerationDefaults {
    fn from(value: ImageDefaultsManifest) -> Self {
        Self {
            resolution: value.resolution,
            steps: value.steps,
            cfg_text_scale: value.cfg_text_scale,
            cfg_img_scale: value.cfg_img_scale,
            cfg_renorm_type: value.cfg_renorm_type,
            cfg_renorm_min: value.cfg_renorm_min,
            cfg_interval: (value.cfg_interval[0], value.cfg_interval[1]),
            timestep_shift: value.timestep_shift,
            seed: value.seed,
            max_images: value.max_images,
            max_images_limit: value.max_images_limit,
        }
    }
}

fn validate_image_defaults(id: &str, defaults: &ImageGenerationDefaults) -> assets::Result<()> {
    if defaults.steps == 0 {
        return Err(AssetError::message(format!(
            "generation profile {id:?} image steps must be positive"
        )));
    }
    if defaults.max_images == 0 || defaults.max_images > defaults.max_images_limit {
        return Err(AssetError::message(format!(
            "generation profile {id:?} image count defaults are invalid"
        )));
    }
    let finite = [
        defaults.cfg_text_scale,
        defaults.cfg_img_scale,
        defaults.cfg_renorm_min,
        defaults.cfg_interval.0,
        defaults.cfg_interval.1,
        defaults.timestep_shift,
    ];
    if finite.iter().any(|value| !value.is_finite())
        || defaults.cfg_interval.0 > defaults.cfg_interval.1
    {
        return Err(AssetError::message(format!(
            "generation profile {id:?} image defaults contain invalid numeric values"
        )));
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResolutionManifest {
    default: String,
    allow_custom: bool,
    buckets: Vec<ResolutionBucketManifest>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResolutionBucketManifest {
    name: String,
    width: u32,
    height: u32,
}

impl From<ResolutionBucketManifest> for ResolutionBucket {
    fn from(value: ResolutionBucketManifest) -> Self {
        Self {
            name: value.name,
            width: value.width,
            height: value.height,
        }
    }
}

impl TryFrom<ResolutionManifest> for ResolutionPolicy {
    type Error = AssetError;

    fn try_from(value: ResolutionManifest) -> Result<Self, Self::Error> {
        let buckets: Vec<ResolutionBucket> = value.buckets.into_iter().map(Into::into).collect();
        if buckets.is_empty() {
            return Err(AssetError::message(
                "generation profile must declare at least one resolution bucket",
            ));
        }
        if buckets
            .iter()
            .any(|bucket| bucket.name.trim().is_empty() || bucket.width == 0 || bucket.height == 0)
        {
            return Err(AssetError::message(
                "generation profile resolution buckets require a name and positive dimensions",
            ));
        }
        let mut names = buckets
            .iter()
            .map(|bucket| bucket.name.to_ascii_lowercase())
            .collect::<Vec<_>>();
        names.sort_unstable();
        names.dedup();
        if names.len() != buckets.len() {
            return Err(AssetError::message(
                "generation profile resolution bucket names must be unique",
            ));
        }
        let default = buckets
            .iter()
            .find(|bucket| bucket.name.eq_ignore_ascii_case(&value.default))
            .cloned()
            .ok_or_else(|| {
                AssetError::message(format!(
                    "generation profile default resolution {:?} has no matching bucket",
                    value.default
                ))
            })?;
        Ok(Self {
            default,
            buckets,
            allow_custom: value.allow_custom,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OutputFilterManifest {
    reasoning: Option<DelimitedTextManifest>,
    #[serde(default)]
    visible_wrappers: Vec<DelimitedTextManifest>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DelimitedTextManifest {
    start: String,
    end: String,
}

impl From<DelimitedTextManifest> for DelimitedTextPolicy {
    fn from(value: DelimitedTextManifest) -> Self {
        Self {
            start: value.start,
            end: value.end,
        }
    }
}

impl TryFrom<OutputFilterManifest> for OutputFilterPolicy {
    type Error = AssetError;

    fn try_from(value: OutputFilterManifest) -> Result<Self, Self::Error> {
        let policy = Self {
            reasoning: value.reasoning.map(Into::into),
            visible_wrappers: value.visible_wrappers.into_iter().map(Into::into).collect(),
        };
        for (name, delimiter) in policy
            .reasoning
            .iter()
            .map(|delimiter| ("reasoning", delimiter))
            .chain(
                policy
                    .visible_wrappers
                    .iter()
                    .map(|delimiter| ("visible wrapper", delimiter)),
            )
        {
            if delimiter.start.is_empty() || delimiter.end.is_empty() {
                return Err(AssetError::message(format!(
                    "generation profile output-filter {name} delimiters must not be empty"
                )));
            }
        }
        Ok(policy)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptManifestSet {
    default: PromptRecipeManifest,
    und: PromptRecipeManifest,
    und_with_images: PromptRecipeManifest,
    r#gen: PromptRecipeManifest,
    negative: PromptRecipeManifest,
}

impl TryFrom<PromptManifestSet> for PromptRecipes {
    type Error = AssetError;

    fn try_from(value: PromptManifestSet) -> Result<Self, Self::Error> {
        Ok(Self {
            default: value.default.try_into()?,
            und: value.und.try_into()?,
            und_with_images: value.und_with_images.try_into()?,
            r#gen: value.r#gen.try_into()?,
            negative: value.negative.try_into()?,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptRecipeManifest {
    kind: PromptRecipeKind,
    #[serde(default)]
    default_system: Option<String>,
    #[serde(default)]
    default_assistant_prefix: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PromptRecipeKind {
    Chatml,
    BagelText,
    BagelImage,
    BagelDefault,
    Raw,
}

impl TryFrom<PromptRecipeManifest> for PromptRecipe {
    type Error = AssetError;

    fn try_from(value: PromptRecipeManifest) -> Result<Self, Self::Error> {
        Ok(match value.kind {
            PromptRecipeKind::Chatml => PromptRecipe::Chatml {
                default_system: value.default_system,
                default_assistant_prefix: value.default_assistant_prefix,
            },
            PromptRecipeKind::BagelText => PromptRecipe::BagelText,
            PromptRecipeKind::BagelImage => PromptRecipe::BagelImage,
            PromptRecipeKind::BagelDefault => PromptRecipe::BagelDefault {
                default_system: value.default_system.ok_or_else(|| {
                    AssetError::message("bagel_default prompt requires default_system")
                })?,
            },
            PromptRecipeKind::Raw => PromptRecipe::Raw,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::tokenizer::{DynTokenizer, Tokenizer};

    use super::*;

    #[derive(Debug)]
    struct ByteTokenizer;

    impl Tokenizer for ByteTokenizer {
        fn encode(
            &self,
            text: &str,
            _add_special_tokens: bool,
        ) -> crate::tokenizer::Result<Vec<u32>> {
            Ok(text.bytes().map(u32::from).collect())
        }

        fn decode(
            &self,
            token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> crate::tokenizer::Result<String> {
            Ok(
                String::from_utf8_lossy(&token_ids.iter().map(|id| *id as u8).collect::<Vec<_>>())
                    .into_owned(),
            )
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            match token {
                "<img>" => Some(10),
                "</img>" => Some(11),
                "<|im_start|>" => Some(13),
                "<|im_end|>" => Some(14),
                _ => None,
            }
        }
    }

    fn temp_model_config(config: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "uniserve-generation-profile-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), config).unwrap();
        dir
    }

    #[test]
    fn resolves_sensenova_profile_from_model_metadata() {
        let tok = ByteTokenizer;
        let dir = temp_model_config(r#"{"architectures":["NEOChatModel"]}"#);
        let profile = resolve_generation_dialect_for_model(dir.to_str().unwrap(), &tok)
            .expect("profile resolution")
            .expect("SenseNova profile");
        assert_eq!(profile.id, "sensenova-u1");
        assert_eq!(profile.resolution_policy.default.width, 2048);
        assert_eq!(profile.resolution_policy.default.height, 1152);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn loads_checkpoint_profile_manifest_before_metadata_fallback() {
        let tok = ByteTokenizer;
        let dir = temp_model_config(r#"{"architectures":["NEOChatModel"]}"#);
        let mut manifest: serde_json::Value = serde_json::from_str(BAGEL_PROFILE_JSON).unwrap();
        manifest["id"] = serde_json::Value::String("custom-profile".into());
        std::fs::write(
            dir.join("uniserve_profile.json"),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();

        let profile = resolve_generation_dialect_for_model(dir.to_str().unwrap(), &tok)
            .expect("profile resolution")
            .expect("checkpoint profile");
        assert_eq!(profile.id, "custom-profile");
        assert!(profile.supports_constraint(GenerationConstraint::UndOnly));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn malformed_checkpoint_profile_does_not_fall_back_to_model_metadata() {
        let tok = ByteTokenizer;
        let dir = temp_model_config(r#"{"architectures":["NEOChatModel"]}"#);
        std::fs::write(dir.join("uniserve_profile.json"), "{").unwrap();

        let error = resolve_generation_dialect_for_model(dir.to_str().unwrap(), &tok)
            .expect_err("malformed checkpoint profile must fail at the repository boundary");
        assert!(error.to_string().contains("invalid generation profile"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn checkpoint_profile_requires_resolvable_control_tokens() {
        let tok = ByteTokenizer;
        let dir = temp_model_config(r#"{"architectures":["NEOChatModel"]}"#);
        let mut manifest: serde_json::Value = serde_json::from_str(BAGEL_PROFILE_JSON).unwrap();
        manifest["control_tokens"]["start_of_image"] = serde_json::json!(["<missing>"]);
        std::fs::write(
            dir.join("uniserve_profile.json"),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();

        let error = resolve_generation_dialect_for_model(dir.to_str().unwrap(), &tok)
            .expect_err("missing control token must reject the profile");
        assert!(error.to_string().contains("start_of_image control token"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn checkpoint_profile_rejects_empty_output_delimiters() {
        let tok = ByteTokenizer;
        let dir = temp_model_config(r#"{"architectures":["NEOChatModel"]}"#);
        let mut manifest: serde_json::Value = serde_json::from_str(BAGEL_PROFILE_JSON).unwrap();
        manifest["output_filter"]["visible_wrappers"] = serde_json::json!([{
            "start": "",
            "end": "</answer>"
        }]);
        std::fs::write(
            dir.join("uniserve_profile.json"),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();

        let error = resolve_generation_dialect_for_model(dir.to_str().unwrap(), &tok)
            .expect_err("empty output delimiter must reject the profile");
        assert!(error.to_string().contains("delimiters must not be empty"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn sensenova_image_prompt_uses_profile_generation_prompt() {
        let tok: DynTokenizer = Arc::new(ByteTokenizer);
        let profile = profile_from_key("sensenova-u1", &*tok).expect("SenseNova profile");
        let body = PromptContext {
            prompt: "paint a lake".into(),
            ..Default::default()
        };
        let ids = profile
            .build_prompt_ids(&tok, &body, PromptKind::Gen)
            .expect("render prompt");
        let text = tok.decode(&ids, false).unwrap();
        assert!(text.contains("image generation and editing assistant"));
        assert!(text.ends_with("<think>\n\n</think>\n\n<img>"));
    }

    #[test]
    fn manifests_resolve_ingest_and_generation_policies() {
        let tok = ByteTokenizer;
        let sensenova = profile_from_key("sensenova-u1", &tok).expect("SenseNova profile");
        assert_eq!(
            sensenova.image_ingest.steps,
            vec![ImageIngestStep::VitEncode]
        );
        assert_eq!(
            sensenova.generation_policy.trigger,
            TriggerPolicyDescriptor::Token { token_id: 10 }
        );
        assert_eq!(
            sensenova.generation_policy.gen_only_start,
            GenOnlyStartPolicyDescriptor::Immediate
        );
        assert_eq!(
            sensenova
                .generation_policy
                .feedback
                .as_ref()
                .map(|feedback| feedback.commit),
            Some(uniserve_core::CommitRecipe::CommitGenThenWriteback)
        );

        let thinkmorph = profile_from_key("thinkmorph", &tok).expect("ThinkMorph profile");
        assert_eq!(
            thinkmorph.image_ingest.steps,
            vec![ImageIngestStep::VaeEncode, ImageIngestStep::VitEncode]
        );
        assert!(matches!(
            thinkmorph.generation_policy.trigger,
            TriggerPolicyDescriptor::RoundCloseThenSuffix { .. }
        ));
        assert_eq!(
            thinkmorph.generation_policy.gen_only_start,
            GenOnlyStartPolicyDescriptor::Immediate
        );
        assert_eq!(
            thinkmorph
                .generation_policy
                .feedback
                .as_ref()
                .map(|feedback| feedback.next_und_token),
            Some(uniserve_core::FeedbackNextToken::Bos)
        );
        assert_eq!(
            thinkmorph
                .generation_policy
                .feedback
                .as_ref()
                .map(|feedback| feedback.commit),
            Some(uniserve_core::CommitRecipe::CommitGen)
        );

        let bagel = profile_from_key("bagel", &tok).expect("BAGEL profile");
        assert_eq!(
            bagel
                .generation_policy
                .feedback
                .as_ref()
                .map(|feedback| feedback.commit),
            Some(uniserve_core::CommitRecipe::CommitGen)
        );
    }

    #[test]
    fn profiles_resolve_exact_input_image_kv_from_dimensions() {
        let tok = ByteTokenizer;
        let bagel = profile_from_key("bagel", &tok).expect("BAGEL profile");
        for (width, height, vae_tokens, vit_tokens) in [
            (512, 512, 1_026, 1_371),
            (640, 480, 1_378, 1_815),
            (1_920, 1_080, 2_306, 2_732),
            (300, 1_200, 1_026, 1_262),
            (2_048, 2_048, 4_098, 4_902),
            (224, 224, 1_026, 1_371),
            (321, 517, 1_666, 2_185),
        ] {
            let bagel_ingest = bagel
                .image_ingest_for_dimensions(width, height, 1)
                .expect("BAGEL image KV");
            assert_eq!(
                bagel_ingest.step_kv_tokens,
                vec![
                    ImageKvEffect::Exact { tokens: vae_tokens },
                    ImageKvEffect::Exact { tokens: vit_tokens },
                ],
                "BAGEL image geometry {width}x{height}"
            );
        }

        let sensenova = profile_from_key("sensenova-u1", &tok).expect("SenseNova profile");
        let sensenova_ingest = sensenova
            .image_ingest_for_dimensions(512, 512, 1)
            .expect("SenseNova image KV");
        assert_eq!(
            sensenova_ingest.step_kv_tokens,
            vec![ImageKvEffect::Exact { tokens: 256 }]
        );
    }
}
