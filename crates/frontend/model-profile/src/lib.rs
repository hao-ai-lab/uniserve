//! Resolved model behavior snapshots used by the serving runtime.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use crate::assets::{
    GenerationConfig, HfTokenizerConfig, ModelConfig, ResolvedModelFiles, TokenizerSource,
    load_generation_config, load_model_config, load_tokenizer_config,
};
use crate::dialect::{GenerationDialectProfile, resolve_generation_dialect_for_model};
use crate::tokenizer::Tokenizer;

pub mod assets;
pub mod dialect;
pub mod reasoning;
pub mod tokenizer;
pub mod tools;

/// Deployment-owned profile inputs. Repository metadata is resolved first and
/// these typed values are then applied as the highest deployment precedence.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileDeploymentConfig {
    pub renderer_id: Option<String>,
    pub chat_template_override: Option<String>,
    pub allow_request_chat_template_override: bool,
    pub language_model_only: bool,
    pub max_model_tokens: Option<u32>,
    pub tool_parser: Option<String>,
    pub reasoning_parser: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelIdentity {
    pub model_id: String,
    pub profile_id: String,
    pub family_id: String,
    pub dialect_id: String,
    pub config_fingerprint: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerKind {
    HuggingFace,
    Tiktoken,
    Tekken,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerDescriptor {
    pub kind: TokenizerKind,
    pub source: String,
    pub fingerprint: String,
    pub config_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GenerationDefaultsDescriptor {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub max_output_tokens: Option<u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextLimits {
    pub max_model_tokens: Option<u32>,
    pub max_output_tokens: Option<u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StopTokenPolicy {
    pub primary_eos_token_id: Option<u32>,
    pub eos_token_ids: BTreeSet<u32>,
    pub eos_aliases: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModalityCapabilities {
    pub text_input: bool,
    pub chat_input: bool,
    pub image_input: bool,
    pub audio_input: bool,
    pub video_input: bool,
    pub text_output: bool,
    pub image_output: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RenderPolicyDescriptor {
    pub renderer_id: String,
    pub template_source: TemplateSource,
    pub template_fingerprint: Option<String>,
    pub system_prompt_policy: String,
    pub tool_rendering_policy: String,
    pub image_marker_policy: String,
    pub request_chat_template_override_allowed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "source")]
pub enum TemplateSource {
    BuiltIn,
    ModelRepository,
    Deployment,
    None,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParserPolicyDescriptor {
    pub reasoning: String,
    pub tools: String,
    pub harmony: bool,
    pub expose_hidden_content: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureCapabilities {
    pub structured_output: bool,
    pub grammar: bool,
    pub logprobs: bool,
    pub prefix_cache: bool,
    pub encoder_cache: bool,
    pub adapters: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeRequirements {
    pub requires_image_encoder: bool,
    pub requires_image_latents: bool,
    pub requires_grammar_masking: bool,
    pub requires_adapter_slots: bool,
}

/// Stable, fully resolved identity, policy, capability, and dialect snapshot
/// for one single-model runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelProfile {
    pub identity: ModelIdentity,
    pub tokenizer: TokenizerDescriptor,
    pub generation_defaults: GenerationDefaultsDescriptor,
    pub context_limits: ContextLimits,
    pub stop_tokens: StopTokenPolicy,
    pub modalities: ModalityCapabilities,
    pub render: RenderPolicyDescriptor,
    pub parsers: ParserPolicyDescriptor,
    pub features: FeatureCapabilities,
    pub runtime_requirements: RuntimeRequirements,
    pub generation_dialect: Option<GenerationDialectProfile>,
}

impl ModelProfile {
    /// Resolve one profile using deterministic precedence: built-in family
    /// defaults, model repository files, then typed deployment configuration.
    pub fn resolve(
        model_id: &str,
        files: &ResolvedModelFiles,
        deployment: &ProfileDeploymentConfig,
        tokenizer: &dyn Tokenizer,
    ) -> assets::Result<Self> {
        let model_config = load_model_config(files.config_path.as_deref())?;
        let generation_config = load_generation_config(files.generation_config_path.as_deref())?;
        let tokenizer_config = load_tokenizer_config(files.tokenizer_config_path.as_deref())?;
        let generation_dialect = resolve_generation_dialect_for_model(model_id, tokenizer)?;
        let has_image_ingest = generation_dialect.is_some();
        let config_fingerprint = profile_fingerprint(files, deployment)?;
        let family_id = resolve_family_id(model_id, &model_config);
        let profile_id = format!("{family_id}:{}", &config_fingerprint[..16]);
        let tokenizer_descriptor = tokenizer_descriptor(files)?;
        let stop_tokens = stop_token_policy(&tokenizer_config, &generation_config, tokenizer);
        let repository_max_tokens = model_config.max_position_embeddings();
        let max_model_tokens = deployment.max_model_tokens.or(repository_max_tokens);
        let image_input = has_image_ingest && !deployment.language_model_only;
        let image_output = !deployment.language_model_only
            && generation_dialect.as_ref().is_some_and(|dialect| {
                dialect.supports_constraint(uniserve_core::GenerationConstraint::GenOnly)
            });
        let model_type = model_config.model_type().unwrap_or_default();
        let dialect_id = generation_dialect
            .as_ref()
            .map_or_else(|| family_id.clone(), |dialect| dialect.id.clone());

        Ok(Self {
            identity: ModelIdentity {
                model_id: model_id.to_string(),
                profile_id,
                family_id,
                dialect_id,
                config_fingerprint,
            },
            tokenizer: tokenizer_descriptor,
            generation_defaults: generation_defaults(&generation_config),
            context_limits: ContextLimits {
                max_model_tokens,
                max_output_tokens: generation_config.max_new_tokens,
            },
            stop_tokens,
            modalities: ModalityCapabilities {
                text_input: true,
                chat_input: true,
                image_input,
                text_output: true,
                image_output,
                ..Default::default()
            },
            render: render_policy(files, deployment)?,
            parsers: ParserPolicyDescriptor {
                reasoning: deployment
                    .reasoning_parser
                    .clone()
                    .unwrap_or_else(|| "auto".to_string()),
                tools: deployment
                    .tool_parser
                    .clone()
                    .unwrap_or_else(|| "auto".to_string()),
                harmony: model_type == "gpt_oss",
                expose_hidden_content: false,
            },
            features: FeatureCapabilities {
                structured_output: true,
                grammar: true,
                logprobs: true,
                prefix_cache: true,
                encoder_cache: image_input,
                adapters: true,
            },
            runtime_requirements: RuntimeRequirements {
                requires_image_encoder: image_input,
                requires_image_latents: image_output,
                requires_grammar_masking: false,
                requires_adapter_slots: false,
            },
            generation_dialect,
        })
    }

    /// Build a profile visible from an already loaded backend when repository
    /// descriptors are unavailable, primarily for embedders and focused tests.
    pub fn loaded(model_id: impl Into<String>) -> Self {
        let model_id = model_id.into();
        fallback_profile(model_id, true)
    }

    pub fn text_only(profile_id: impl Into<String>) -> Self {
        fallback_profile(profile_id.into(), false)
    }

    pub fn profile_id(&self) -> &str {
        &self.identity.profile_id
    }

    pub fn family_id(&self) -> &str {
        &self.identity.family_id
    }

    pub fn dialect_id(&self) -> &str {
        &self.identity.dialect_id
    }

    pub fn tokenizer_fingerprint(&self) -> &str {
        &self.tokenizer.fingerprint
    }

    pub fn with_generation_dialect(mut self, dialect: GenerationDialectProfile) -> Self {
        self.identity.dialect_id = dialect.id.clone();
        self.modalities.image_input = true;
        self.modalities.image_output =
            dialect.supports_constraint(uniserve_core::GenerationConstraint::GenOnly);
        self.features.encoder_cache = true;
        self.runtime_requirements.requires_image_encoder = true;
        self.runtime_requirements.requires_image_latents = self.modalities.image_output;
        self.generation_dialect = Some(dialect);
        self
    }

    pub fn with_max_model_tokens(mut self, max_model_tokens: u32) -> Self {
        self.context_limits.max_model_tokens = Some(max_model_tokens);
        self
    }

    pub fn with_parser_selections(
        mut self,
        tool_parser: impl Into<String>,
        reasoning_parser: impl Into<String>,
    ) -> Self {
        self.parsers.tools = tool_parser.into();
        self.parsers.reasoning = reasoning_parser.into();
        self
    }
}

fn fallback_profile(model_id: String, supports_chat: bool) -> ModelProfile {
    let fingerprint = digest_bytes(model_id.as_bytes());
    ModelProfile {
        identity: ModelIdentity {
            profile_id: format!("embedded:{}", &fingerprint[..16]),
            family_id: model_id.clone(),
            dialect_id: if supports_chat {
                "chat-template"
            } else {
                "text"
            }
            .to_string(),
            model_id: model_id.clone(),
            config_fingerprint: fingerprint.clone(),
        },
        tokenizer: TokenizerDescriptor {
            kind: TokenizerKind::HuggingFace,
            source: "embedded".to_string(),
            fingerprint,
            config_fingerprint: None,
        },
        generation_defaults: GenerationDefaultsDescriptor::default(),
        context_limits: ContextLimits::default(),
        stop_tokens: StopTokenPolicy::default(),
        modalities: ModalityCapabilities {
            text_input: true,
            chat_input: supports_chat,
            image_input: false,
            text_output: true,
            ..Default::default()
        },
        render: RenderPolicyDescriptor {
            renderer_id: if supports_chat { "embedded" } else { "raw" }.to_string(),
            template_source: if supports_chat {
                TemplateSource::BuiltIn
            } else {
                TemplateSource::None
            },
            template_fingerprint: None,
            system_prompt_policy: "request".to_string(),
            tool_rendering_policy: "renderer".to_string(),
            image_marker_policy: "renderer".to_string(),
            request_chat_template_override_allowed: false,
        },
        parsers: ParserPolicyDescriptor {
            reasoning: "auto".to_string(),
            tools: "auto".to_string(),
            harmony: false,
            expose_hidden_content: false,
        },
        features: FeatureCapabilities {
            structured_output: true,
            grammar: true,
            logprobs: true,
            prefix_cache: true,
            encoder_cache: false,
            adapters: true,
        },
        runtime_requirements: RuntimeRequirements {
            requires_image_encoder: false,
            ..Default::default()
        },
        generation_dialect: None,
    }
}

fn generation_defaults(config: &GenerationConfig) -> GenerationDefaultsDescriptor {
    GenerationDefaultsDescriptor {
        temperature: config.temperature,
        top_p: config.top_p,
        top_k: config.top_k,
        min_p: config.min_p,
        repetition_penalty: config.repetition_penalty,
        max_output_tokens: config.max_new_tokens,
    }
}

fn resolve_family_id(model_id: &str, config: &ModelConfig) -> String {
    config
        .model_type()
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| model_id.to_ascii_lowercase().replace(['/', ' '], "-"))
}

fn tokenizer_descriptor(files: &ResolvedModelFiles) -> assets::Result<TokenizerDescriptor> {
    let (kind, source) = match &files.tokenizer {
        TokenizerSource::HuggingFace(path) => (TokenizerKind::HuggingFace, path),
        TokenizerSource::Tiktoken(path) => (TokenizerKind::Tiktoken, path),
        TokenizerSource::Tekken(path) => (TokenizerKind::Tekken, path),
    };
    Ok(TokenizerDescriptor {
        kind,
        source: source.display().to_string(),
        fingerprint: digest_file(source)?,
        config_fingerprint: files
            .tokenizer_config_path
            .as_deref()
            .map(digest_file)
            .transpose()?,
    })
}

fn stop_token_policy(
    tokenizer_config: &HfTokenizerConfig,
    generation_config: &GenerationConfig,
    tokenizer: &dyn Tokenizer,
) -> StopTokenPolicy {
    let primary = tokenizer_config
        .special_tokens
        .eos_token
        .as_ref()
        .and_then(|token| tokenizer.token_to_id(token.as_str()));
    let mut ids = generation_config
        .eos_token_id
        .clone()
        .map(|ids| ids.into_set())
        .unwrap_or_default();
    if let Some(primary) = primary {
        ids.insert(primary);
    }
    StopTokenPolicy {
        primary_eos_token_id: primary,
        eos_token_ids: ids,
        eos_aliases: tokenizer_config
            .special_tokens
            .eos_token
            .as_ref()
            .map(|token| vec![token.as_str().to_string()])
            .unwrap_or_default(),
    }
}

fn render_policy(
    files: &ResolvedModelFiles,
    deployment: &ProfileDeploymentConfig,
) -> assets::Result<RenderPolicyDescriptor> {
    let (template_source, template_fingerprint) =
        if let Some(template) = &deployment.chat_template_override {
            (
                TemplateSource::Deployment,
                Some(digest_bytes(template.as_bytes())),
            )
        } else if let Some(path) = &files.chat_template_path {
            (TemplateSource::ModelRepository, Some(digest_file(path)?))
        } else {
            (TemplateSource::BuiltIn, None)
        };
    Ok(RenderPolicyDescriptor {
        renderer_id: deployment
            .renderer_id
            .clone()
            .unwrap_or_else(|| "auto".to_string()),
        template_source,
        template_fingerprint,
        system_prompt_policy: "profile_then_request".to_string(),
        tool_rendering_policy: "renderer".to_string(),
        image_marker_policy: "dialect".to_string(),
        request_chat_template_override_allowed: deployment.allow_request_chat_template_override,
    })
}

fn profile_fingerprint(
    files: &ResolvedModelFiles,
    deployment: &ProfileDeploymentConfig,
) -> assets::Result<String> {
    let mut hasher = Sha256::new();
    for path in [
        files.config_path.as_deref(),
        files.generation_config_path.as_deref(),
        files.tokenizer_config_path.as_deref(),
        files.preprocessor_config_path.as_deref(),
        files.chat_template_path.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        hasher.update(
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default(),
        );
        hasher.update(fs::read(path).map_err(|error| {
            assets::Error::message(format!(
                "failed to fingerprint '{}': {error}",
                path.display()
            ))
        })?);
    }
    hasher.update(serde_json::to_vec(deployment).map_err(|error| {
        assets::Error::message(format!("failed to fingerprint deployment profile: {error}"))
    })?);
    Ok(format!("{:x}", hasher.finalize()))
}

fn digest_file(path: &Path) -> assets::Result<String> {
    let bytes = fs::read(path).map_err(|error| {
        assets::Error::message(format!(
            "failed to fingerprint '{}': {error}",
            path.display()
        ))
    })?;
    Ok(digest_bytes(&bytes))
}

fn digest_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::*;

    #[derive(Debug)]
    struct TestTokenizer;

    impl Tokenizer for TestTokenizer {
        fn encode(&self, text: &str, _add_special_tokens: bool) -> tokenizer::Result<Vec<u32>> {
            Ok(text.bytes().map(u32::from).collect())
        }

        fn decode(
            &self,
            token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> tokenizer::Result<String> {
            Ok(token_ids.iter().map(|id| *id as u8 as char).collect())
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            match token {
                "<eos>" => Some(2),
                "<|im_start|>" => Some(13),
                "<|im_end|>" => Some(14),
                "<img>" => Some(15),
                "</img>" => Some(16),
                _ => None,
            }
        }
    }

    fn resolved_files() -> (tempfile::TempDir, ResolvedModelFiles) {
        let dir = tempdir().unwrap();
        let write = |name: &str, value: &str| {
            let path = dir.path().join(name);
            fs::write(&path, value).unwrap();
            path
        };
        let tokenizer = write("tokenizer.json", "{}");
        let tokenizer_config = write("tokenizer_config.json", r#"{"eos_token":"<eos>"}"#);
        let generation = write(
            "generation_config.json",
            r#"{"temperature":0.7,"max_new_tokens":64,"eos_token_id":[2,3]}"#,
        );
        let config = write(
            "config.json",
            r#"{"model_type":"repo-family","max_position_embeddings":8192}"#,
        );
        let files = ResolvedModelFiles {
            tokenizer: TokenizerSource::HuggingFace(tokenizer),
            tokenizer_config_path: Some(tokenizer_config),
            generation_config_path: Some(generation),
            preprocessor_config_path: None,
            chat_template_path: None,
            config_path: Some(config),
        };
        (dir, files)
    }

    #[test]
    fn repository_metadata_and_deployment_overrides_resolve_deterministically() {
        let (_dir, files) = resolved_files();
        let deployment = ProfileDeploymentConfig {
            renderer_id: Some("deepseek-v4".to_string()),
            max_model_tokens: Some(4096),
            tool_parser: Some("qwen3".to_string()),
            ..Default::default()
        };
        let first =
            ModelProfile::resolve("org/model", &files, &deployment, &TestTokenizer).unwrap();
        let second =
            ModelProfile::resolve("org/model", &files, &deployment, &TestTokenizer).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.family_id(), "repo-family");
        assert_eq!(first.context_limits.max_model_tokens, Some(4096));
        assert_eq!(first.generation_defaults.temperature, Some(0.7));
        assert_eq!(first.stop_tokens.primary_eos_token_id, Some(2));
        assert_eq!(first.stop_tokens.eos_token_ids, BTreeSet::from([2, 3]));
        assert_eq!(first.render.renderer_id, "deepseek-v4");
        assert_eq!(first.parsers.tools, "qwen3");
        assert!(!first.modalities.image_input);
    }

    #[test]
    fn language_only_disables_image_dialect_capabilities() {
        let (_dir, files) = resolved_files();
        fs::write(
            files.config_path.as_ref().unwrap(),
            r#"{"model_type":"neo_chat","max_position_embeddings":8192}"#,
        )
        .unwrap();
        let profile = ModelProfile::resolve(
            "local/SenseNova-U1",
            &files,
            &ProfileDeploymentConfig {
                language_model_only: true,
                ..Default::default()
            },
            &TestTokenizer,
        )
        .unwrap();

        assert!(!profile.modalities.image_input);
        assert!(!profile.modalities.image_output);
        assert!(!profile.features.encoder_cache);
        assert!(!profile.runtime_requirements.requires_image_encoder);
        assert!(!profile.runtime_requirements.requires_image_latents);
    }

    #[test]
    fn image_ingest_dialect_advertises_image_capabilities() {
        let (_dir, files) = resolved_files();
        fs::write(
            files.config_path.as_ref().unwrap(),
            r#"{"model_type":"neo_chat","max_position_embeddings":8192}"#,
        )
        .unwrap();

        let profile = ModelProfile::resolve(
            "local/SenseNova-U1",
            &files,
            &ProfileDeploymentConfig::default(),
            &TestTokenizer,
        )
        .unwrap();

        assert_eq!(profile.dialect_id(), "sensenova-u1");
        assert!(profile.modalities.image_input);
        assert!(profile.features.encoder_cache);
        assert!(profile.runtime_requirements.requires_image_encoder);
    }

    #[test]
    fn text_profile_does_not_receive_a_generation_dialect_fallback() {
        let (_dir, files) = resolved_files();
        let profile = ModelProfile::resolve(
            "org/text-model",
            &files,
            &ProfileDeploymentConfig::default(),
            &TestTokenizer,
        )
        .unwrap();

        assert!(!profile.modalities.image_input);
        assert!(!profile.modalities.image_output);
        assert!(!profile.features.encoder_cache);
        assert!(!profile.runtime_requirements.requires_image_encoder);
        assert!(!profile.runtime_requirements.requires_image_latents);
        assert_eq!(profile.dialect_id(), "repo-family");
        assert!(profile.generation_dialect.is_none());
    }
}
