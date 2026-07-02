use std::fs;
use std::path::Path;

use serde::Deserialize;
use serde_json::Value;
use uniserve_engine_client::GenMode;
use uniserve_text::tokenizer::DynTokenizer;

use super::resolution::{ResolutionBucket, ResolutionPolicy};
use super::schema::NativeGenerateBody;

const BAGEL_PROFILE_JSON: &str = include_str!("../profiles/bagel.json");
const SENSENOVA_PROFILE_JSON: &str = include_str!("../profiles/sensenova-u1.json");

#[derive(Debug, Clone, Default)]
pub struct NativeControls {
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

#[derive(Debug, Clone)]
pub struct NativeImageDefaults {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeDelimitedText {
    pub start: String,
    pub end: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeOutputFilter {
    pub reasoning: Option<NativeDelimitedText>,
    pub visible_wrappers: Vec<NativeDelimitedText>,
}

#[derive(Debug, Clone)]
enum PromptRecipe {
    Chatml {
        default_system: Option<String>,
        default_assistant_prefix: String,
    },
    BagelText,
    BagelImage,
    BagelAutoInterleave {
        default_system: String,
    },
    Raw,
}

#[derive(Debug, Clone)]
struct NativePromptRecipes {
    text: PromptRecipe,
    image: PromptRecipe,
    auto_interleave: PromptRecipe,
    understand: PromptRecipe,
    negative: PromptRecipe,
}

#[derive(Debug, Clone)]
pub struct NativeModelProfile {
    pub id: String,
    pub controls: NativeControls,
    pub image_defaults: NativeImageDefaults,
    pub resolution_policy: ResolutionPolicy,
    pub output_filter: NativeOutputFilter,
    default_mode: GenMode,
    supported_modes: Vec<GenMode>,
    prompts: NativePromptRecipes,
    understanding_system_prompt: String,
    understanding_markers_in_prompt: bool,
}

impl Default for NativeModelProfile {
    fn default() -> Self {
        profile_from_manifest(
            serde_json::from_str(BAGEL_PROFILE_JSON).expect("built-in Bagel profile is valid JSON"),
            None,
        )
    }
}

impl NativeModelProfile {
    pub fn default_mode_name(&self) -> &'static str {
        mode_name(self.default_mode)
    }

    pub fn supports_mode(&self, mode: GenMode) -> bool {
        self.supported_modes.contains(&mode)
    }

    pub fn understanding_system_prompt(&self) -> &str {
        &self.understanding_system_prompt
    }

    /// Whether understanding-mode input images ride as markers inside the
    /// prompt token stream (the encode op fills the gap between them).
    pub fn understanding_markers_in_prompt(&self) -> bool {
        self.understanding_markers_in_prompt
    }

    /// Render the understanding-mode prompt through the profile's `understand`
    /// recipe (used by marker-in-prompt profiles; the legacy path wraps text
    /// with bos/eos directly).
    pub fn build_understanding_prompt_ids(
        &self,
        tok: &DynTokenizer,
        body: &NativeGenerateBody,
        user_text: &str,
    ) -> Vec<u32> {
        render_prompt(tok, &self.controls, &self.prompts.understand, body, user_text)
    }

    pub fn build_prompt_ids(
        &self,
        tok: &DynTokenizer,
        body: &NativeGenerateBody,
        mode: GenMode,
    ) -> Vec<u32> {
        let recipe = match mode {
            GenMode::Text => &self.prompts.text,
            GenMode::Image => &self.prompts.image,
            GenMode::AutoInterleave => &self.prompts.auto_interleave,
            GenMode::InterleaveUnd => &self.prompts.understand,
        };
        render_prompt(tok, &self.controls, recipe, body, &body.prompt)
    }

    pub fn build_negative_prompt_ids(&self, tok: &DynTokenizer, negative_prompt: &str) -> Vec<u32> {
        if negative_prompt.is_empty() {
            return Vec::new();
        }
        let body = NativeGenerateBody {
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

    pub fn wrap_understanding_text(&self, tok: &DynTokenizer, text: &str) -> Vec<u32> {
        let mut ids = vec![self.controls.bos];
        ids.extend(encode(tok, text));
        ids.push(self.controls.eos);
        ids
    }
}

pub fn resolve_native_profile(
    tokenizer: &dyn uniserve_text::tokenizer::Tokenizer,
) -> NativeModelProfile {
    profile_from_key("bagel", tokenizer)
}

pub fn resolve_native_profile_for_model(
    model_ref: &str,
    tokenizer: &dyn uniserve_text::tokenizer::Tokenizer,
) -> NativeModelProfile {
    if let Some(profile) = profile_from_model_manifest(model_ref, tokenizer) {
        return profile;
    }
    let key = profile_key_from_model(model_ref).unwrap_or("bagel");
    profile_from_key(key, tokenizer)
}

fn profile_from_key(
    key: &str,
    tokenizer: &dyn uniserve_text::tokenizer::Tokenizer,
) -> NativeModelProfile {
    let json = if key == "sensenova-u1" {
        SENSENOVA_PROFILE_JSON
    } else {
        BAGEL_PROFILE_JSON
    };
    profile_from_manifest(
        serde_json::from_str(json).expect("built-in native model profile is valid JSON"),
        Some(tokenizer),
    )
}

fn profile_from_model_manifest(
    model_ref: &str,
    tokenizer: &dyn uniserve_text::tokenizer::Tokenizer,
) -> Option<NativeModelProfile> {
    let path = Path::new(model_ref);
    if !path.is_dir() {
        return None;
    }
    let text = fs::read_to_string(path.join("uniserve_profile.json")).ok()?;
    let manifest = serde_json::from_str::<ProfileManifest>(&text).ok()?;
    Some(profile_from_manifest(manifest, Some(tokenizer)))
}

fn profile_from_manifest(
    manifest: ProfileManifest,
    tokenizer: Option<&dyn uniserve_text::tokenizer::Tokenizer>,
) -> NativeModelProfile {
    let controls = tokenizer
        .map(|tok| controls_from_manifest(&manifest.control_tokens, tok))
        .unwrap_or_default();
    NativeModelProfile {
        id: manifest.id,
        controls,
        image_defaults: manifest.image_defaults.into(),
        resolution_policy: manifest.resolution.into(),
        output_filter: manifest.output_filter.into(),
        default_mode: parse_profile_mode(&manifest.default_mode),
        supported_modes: manifest
            .supported_modes
            .iter()
            .map(|mode| parse_profile_mode(mode))
            .collect(),
        prompts: manifest.prompts.into(),
        understanding_system_prompt: manifest.understanding_system_prompt,
        understanding_markers_in_prompt: manifest.understanding.markers_in_prompt,
    }
}

fn controls_from_manifest(
    spec: &ControlTokenManifest,
    tokenizer: &dyn uniserve_text::tokenizer::Tokenizer,
) -> NativeControls {
    let (start_of_image, start_of_image_text) = first_token(tokenizer, &spec.start_of_image);
    let (end_of_image, end_of_image_text) = first_token(tokenizer, &spec.end_of_image);
    NativeControls {
        bos: first_token_id(tokenizer, &spec.bos),
        eos: first_token_id(tokenizer, &spec.eos),
        start_of_image,
        end_of_image,
        image_start_ids: spec
            .image_start_text
            .as_ref()
            .and_then(|text| tokenizer.encode(text, false).ok())
            .unwrap_or_default(),
        start_of_image_text,
        end_of_image_text,
    }
}

fn first_token(
    tokenizer: &dyn uniserve_text::tokenizer::Tokenizer,
    candidates: &[String],
) -> (u32, String) {
    candidates
        .iter()
        .find_map(|token| tokenizer.token_to_id(token).map(|id| (id, token.clone())))
        .unwrap_or((0, String::new()))
}

fn first_token_id(
    tokenizer: &dyn uniserve_text::tokenizer::Tokenizer,
    candidates: &[String],
) -> u32 {
    candidates
        .iter()
        .find_map(|token| tokenizer.token_to_id(token))
        .unwrap_or(0)
}

fn profile_key_from_model(model_ref: &str) -> Option<&'static str> {
    let path = Path::new(model_ref);
    if path.is_dir()
        && let Ok(text) = fs::read_to_string(path.join("config.json"))
        && let Ok(config) = serde_json::from_str::<Value>(&text)
    {
        return profile_key_from_config(&config);
    }
    let lower = model_ref.to_ascii_lowercase();
    if lower.contains("sensenova") || lower.contains("neo_chat") || lower.contains("neo-unify") {
        Some("sensenova-u1")
    } else if lower.contains("bagel") {
        Some("bagel")
    } else {
        None
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
    if model_type.contains("bagel") || architectures.iter().any(|arch| arch.contains("bagel")) {
        return Some("bagel");
    }
    None
}

fn render_prompt(
    tok: &DynTokenizer,
    controls: &NativeControls,
    recipe: &PromptRecipe,
    body: &NativeGenerateBody,
    prompt: &str,
) -> Vec<u32> {
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
            ids.extend(encode(tok, prompt));
            ids.push(controls.eos);
            ids
        }
        PromptRecipe::BagelAutoInterleave { default_system } => encode(
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

fn encode(tok: &DynTokenizer, text: &str) -> Vec<u32> {
    tok.encode(text, false).unwrap_or_default()
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

fn mode_name(mode: GenMode) -> &'static str {
    match mode {
        GenMode::Text => "text",
        GenMode::Image => "image",
        GenMode::AutoInterleave => "interleave",
        GenMode::InterleaveUnd => "understand",
    }
}

fn parse_profile_mode(value: &str) -> GenMode {
    match value {
        "text" => GenMode::Text,
        "image" => GenMode::Image,
        "auto" | "auto_interleave" | "interleave" => GenMode::AutoInterleave,
        "understand" | "interleave_und" | "understanding" => GenMode::InterleaveUnd,
        other => panic!("unknown native profile mode {other:?}"),
    }
}

#[derive(Debug, Deserialize)]
struct ProfileManifest {
    id: String,
    control_tokens: ControlTokenManifest,
    default_mode: String,
    supported_modes: Vec<String>,
    image_defaults: ImageDefaultsManifest,
    resolution: ResolutionManifest,
    output_filter: OutputFilterManifest,
    prompts: PromptManifestSet,
    understanding_system_prompt: String,
    #[serde(default)]
    understanding: UnderstandingManifest,
}

/// Understanding-mode (i2t) request-construction policy.
#[derive(Debug, Deserialize, Default)]
struct UnderstandingManifest {
    /// When true the image begin/end markers are ordinary prompt tokens and
    /// the encode op fills the gap between them (one shared temporal RoPE
    /// index per image). When false (legacy default) the worker emits the
    /// markers itself during the encode op.
    #[serde(default)]
    markers_in_prompt: bool,
}

#[derive(Debug, Deserialize)]
struct ControlTokenManifest {
    bos: Vec<String>,
    eos: Vec<String>,
    start_of_image: Vec<String>,
    end_of_image: Vec<String>,
    #[serde(default)]
    image_start_text: Option<String>,
}

#[derive(Debug, Deserialize)]
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

impl From<ImageDefaultsManifest> for NativeImageDefaults {
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

#[derive(Debug, Deserialize)]
struct ResolutionManifest {
    default: String,
    allow_custom: bool,
    buckets: Vec<ResolutionBucketManifest>,
}

#[derive(Debug, Deserialize)]
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

impl From<ResolutionManifest> for ResolutionPolicy {
    fn from(value: ResolutionManifest) -> Self {
        let buckets: Vec<ResolutionBucket> = value.buckets.into_iter().map(Into::into).collect();
        let default = buckets
            .iter()
            .find(|bucket| bucket.name.eq_ignore_ascii_case(&value.default))
            .cloned()
            .unwrap_or_else(|| {
                buckets
                    .first()
                    .cloned()
                    .expect("native profile must declare at least one resolution bucket")
            });
        Self {
            default,
            buckets,
            allow_custom: value.allow_custom,
        }
    }
}

#[derive(Debug, Deserialize)]
struct OutputFilterManifest {
    reasoning: Option<DelimitedTextManifest>,
    #[serde(default)]
    visible_wrappers: Vec<DelimitedTextManifest>,
}

#[derive(Debug, Deserialize)]
struct DelimitedTextManifest {
    start: String,
    end: String,
}

impl From<DelimitedTextManifest> for NativeDelimitedText {
    fn from(value: DelimitedTextManifest) -> Self {
        Self {
            start: value.start,
            end: value.end,
        }
    }
}

impl From<OutputFilterManifest> for NativeOutputFilter {
    fn from(value: OutputFilterManifest) -> Self {
        Self {
            reasoning: value.reasoning.map(Into::into),
            visible_wrappers: value.visible_wrappers.into_iter().map(Into::into).collect(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct PromptManifestSet {
    text: PromptRecipeManifest,
    image: PromptRecipeManifest,
    auto_interleave: PromptRecipeManifest,
    understand: PromptRecipeManifest,
    negative: PromptRecipeManifest,
}

impl From<PromptManifestSet> for NativePromptRecipes {
    fn from(value: PromptManifestSet) -> Self {
        Self {
            text: value.text.into(),
            image: value.image.into(),
            auto_interleave: value.auto_interleave.into(),
            understand: value.understand.into(),
            negative: value.negative.into(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct PromptRecipeManifest {
    kind: String,
    #[serde(default)]
    default_system: Option<String>,
    #[serde(default)]
    default_assistant_prefix: String,
}

impl From<PromptRecipeManifest> for PromptRecipe {
    fn from(value: PromptRecipeManifest) -> Self {
        match value.kind.as_str() {
            "chatml" => PromptRecipe::Chatml {
                default_system: value.default_system,
                default_assistant_prefix: value.default_assistant_prefix,
            },
            "bagel_text" => PromptRecipe::BagelText,
            "bagel_image" => PromptRecipe::BagelImage,
            "bagel_auto_interleave" => PromptRecipe::BagelAutoInterleave {
                default_system: value
                    .default_system
                    .expect("bagel_auto_interleave prompt requires default_system"),
            },
            "raw" => PromptRecipe::Raw,
            other => panic!("unknown native prompt recipe {other:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use uniserve_text::tokenizer::{DynTokenizer, Tokenizer};

    use super::*;

    #[derive(Debug)]
    struct ByteTokenizer;

    impl Tokenizer for ByteTokenizer {
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
            "uniserve-native-profile-{}-{}",
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
        let profile = resolve_native_profile_for_model(dir.to_str().unwrap(), &tok);
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

        let profile = resolve_native_profile_for_model(dir.to_str().unwrap(), &tok);
        assert_eq!(profile.id, "custom-profile");
        assert_eq!(profile.default_mode_name(), "text");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn sensenova_image_prompt_uses_profile_generation_prompt() {
        let tok: DynTokenizer = Arc::new(ByteTokenizer);
        let profile = profile_from_key("sensenova-u1", &*tok);
        let body = NativeGenerateBody {
            prompt: "paint a lake".into(),
            mode: Some("image".into()),
            ..Default::default()
        };
        let ids = profile.build_prompt_ids(&tok, &body, GenMode::Image);
        let text = tok.decode(&ids, false).unwrap();
        assert!(text.contains("image generation and editing assistant"));
        assert!(text.ends_with("<think>\n\n</think>\n\n<img>"));
    }
}
