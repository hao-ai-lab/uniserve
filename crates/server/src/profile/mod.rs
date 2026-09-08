//! Model-profile selection and load-time serving capabilities.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use crate::profile::assets::{
    GenerationConfig, HfTokenizerConfig, ResolvedModelFiles, load_generation_config,
    load_model_config, load_tokenizer_config,
};
use crate::profile::omni::bagel::BagelProfile;
use crate::profile::omni::sensenova::SenseNovaProfile;
use crate::profile::tokenizer::HuggingFaceTokenizer;

pub mod assets;
pub mod omni;
pub mod reasoning;
pub mod tokenizer;
pub mod tools;

/// The complete configured model-description set selected at server startup.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelDescription {
    /// Qwen3 text-generation profile.
    #[default]
    Qwen3,
    /// SenseNova multimodal-generation profile.
    #[serde(rename = "sensenova")]
    SenseNova,
    /// Bagel multimodal-generation profile.
    Bagel,
    /// MiniMax H3 video-generation profile.
    MiniMaxH3,
}

impl ModelDescription {
    /// Returns the stable profile identifier.
    pub const fn id(self) -> &'static str {
        match self {
            Self::Qwen3 => "qwen3",
            Self::SenseNova => "sensenova",
            Self::Bagel => "bagel",
            Self::MiniMaxH3 => "minimax_h3",
        }
    }

    /// Returns the model-family identifier.
    const fn model_type(self) -> &'static str {
        match self {
            Self::Qwen3 => "qwen3",
            Self::SenseNova => "neo_chat",
            Self::Bagel => "bagel",
            Self::MiniMaxH3 => "minimax_h3",
        }
    }
}

impl std::str::FromStr for ModelDescription {
    type Err = ModelDescriptionParseError;

    /// Parses the value from its string representation.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "qwen3" => Ok(Self::Qwen3),
            "sensenova" => Ok(Self::SenseNova),
            "bagel" => Ok(Self::Bagel),
            "minimax-h3" | "minimax_h3" => Ok(Self::MiniMaxH3),
            _ => Err(ModelDescriptionParseError(value.to_owned())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unsupported model description {0:?}")]
/// Error returned for an unsupported model-description name.
pub struct ModelDescriptionParseError(String);

/// configuration-owned profile inputs applied after repository metadata.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileOverrides {
    /// Optional template text that replaces the repository-provided chat template.
    pub chat_template_override: Option<String>,
    /// Optional configuration ceiling on the complete model context.
    pub max_model_tokens: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Model name, revision, and numeric precision resolved at load time.
pub struct ModelIdentity {
    /// Model name exposed through serving APIs.
    pub served_name: String,
    /// Profile family that defines this model's serving behavior.
    pub description: ModelDescription,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
/// Model-provided defaults for sampling and generation limits.
pub struct GenerationDefaultsDescriptor {
    /// Default sampling temperature.
    pub temperature: Option<f32>,
    /// Default nucleus-sampling probability mass.
    pub top_p: Option<f32>,
    /// Default top-k candidate limit.
    pub top_k: Option<u32>,
    /// Default minimum relative token probability.
    pub min_p: Option<f32>,
    /// Default repetition penalty.
    pub repetition_penalty: Option<f32>,
    /// Default maximum number of generated tokens.
    pub max_output_tokens: Option<u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
/// Maximum context and generated-token lengths supported by a profile.
pub struct ContextLimits {
    /// Maximum combined input and output token count.
    pub max_model_tokens: Option<u32>,
    /// Maximum generated-token count.
    pub max_output_tokens: Option<u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
/// End-of-sequence and stop-token handling supplied by a profile.
pub struct StopTokenPolicy {
    /// Canonical end-of-sequence token identifier.
    pub primary_eos_token_id: Option<u32>,
    /// Token identifiers that terminate generation.
    pub eos_token_ids: BTreeSet<u32>,
    /// Tokenizer strings recognized as end-of-sequence aliases.
    pub eos_aliases: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Model capabilities and policies shared across serving families.
pub struct CommonModelProfile {
    /// Stable identity exposed to clients and runtime components.
    pub identity: ModelIdentity,
    /// Model-provided generation defaults.
    pub generation_defaults: GenerationDefaultsDescriptor,
    /// Context and output length limits.
    pub context_limits: ContextLimits,
    /// End-of-sequence recognition policy.
    pub stop_tokens: StopTokenPolicy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Fully resolved SenseNova model profile.
pub(crate) struct SenseNovaModelProfile {
    pub common: CommonModelProfile,
    pub preprocessing: SenseNovaProfile,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Fully resolved Bagel model profile.
pub(crate) struct BagelModelProfile {
    pub common: CommonModelProfile,
    pub preprocessing: BagelProfile,
}

/// The closed resolved profile value consumed by the serving model description.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) enum ModelProfile {
    Qwen3(CommonModelProfile),
    SenseNova(SenseNovaModelProfile),
    Bagel(BagelModelProfile),
    MiniMaxH3(CommonModelProfile),
}

impl ModelProfile {
    /// Constructs a MiniMax H3 description with model defaults.
    pub(crate) fn minimax_h3(model_id: &str) -> Self {
        Self::MiniMaxH3(CommonModelProfile {
            identity: ModelIdentity {
                served_name: model_id.to_string(),
                description: ModelDescription::MiniMaxH3,
            },
            generation_defaults: GenerationDefaultsDescriptor::default(),
            context_limits: ContextLimits::default(),
            stop_tokens: StopTokenPolicy::default(),
        })
    }

    /// Resolves model assets and profile-specific serving contracts.
    pub(crate) fn resolve(
        description: ModelDescription,
        model_id: &str,
        files: &ResolvedModelFiles,
        configuration: &ProfileOverrides,
        tokenizer: &HuggingFaceTokenizer,
    ) -> assets::Result<Self> {
        let model_config = load_model_config(files.config_path.as_deref())?;
        let actual_model_type = model_config
            .model_type()
            .ok_or(assets::Error::MissingField {
                field: "model_type",
            })?;
        if actual_model_type != description.model_type() {
            return Err(assets::Error::ModelTypeMismatch {
                expected: description.model_type(),
                actual: actual_model_type.to_owned(),
            });
        }
        let generation_config = load_generation_config(files.generation_config_path.as_deref())?;
        let tokenizer_config = load_tokenizer_config(files.tokenizer_config_path.as_deref())?;
        let common = CommonModelProfile {
            identity: ModelIdentity {
                served_name: model_id.to_string(),
                description,
            },
            generation_defaults: generation_defaults(&generation_config),
            context_limits: ContextLimits {
                max_model_tokens: configuration
                    .max_model_tokens
                    .or(model_config.max_position_embeddings()),
                max_output_tokens: generation_config.max_new_tokens,
            },
            stop_tokens: stop_token_policy(&tokenizer_config, &generation_config, tokenizer),
        };
        match description {
            ModelDescription::Qwen3 => Ok(Self::Qwen3(common)),
            ModelDescription::SenseNova => Ok(Self::SenseNova(SenseNovaModelProfile {
                common,
                preprocessing: SenseNovaProfile::resolve(tokenizer)?,
            })),
            ModelDescription::Bagel => Ok(Self::Bagel(BagelModelProfile {
                common,
                preprocessing: BagelProfile::resolve(tokenizer)?,
            })),
            ModelDescription::MiniMaxH3 => Ok(Self::minimax_h3(model_id)),
        }
    }

    /// Returns the common capabilities of this resolved profile.
    pub(crate) fn common(&self) -> &CommonModelProfile {
        match self {
            Self::Qwen3(profile) => profile,
            Self::SenseNova(profile) => &profile.common,
            Self::Bagel(profile) => &profile.common,
            Self::MiniMaxH3(profile) => profile,
        }
    }

    /// Returns mutable access to common profile capabilities.
    pub(crate) fn common_mut(&mut self) -> &mut CommonModelProfile {
        match self {
            Self::Qwen3(profile) => profile,
            Self::SenseNova(profile) => &mut profile.common,
            Self::Bagel(profile) => &mut profile.common,
            Self::MiniMaxH3(profile) => profile,
        }
    }
}

/// Extracts serving generation defaults from repository configuration.
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

/// Combines tokenizer and generation metadata into one canonical termination policy.
fn stop_token_policy(
    tokenizer_config: &HfTokenizerConfig,
    generation_config: &GenerationConfig,
    tokenizer: &HuggingFaceTokenizer,
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

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;
    use tokenizers::models::bpe::{BPE, Vocab};
    use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};
    use uniserve_core::{GenerationConstraint, ImageIngestStep, ImageKvEffect};

    use super::{ModelDescription, ModelProfile, ProfileOverrides};
    use crate::profile::assets::ResolvedModelFiles;
    use crate::profile::tokenizer::HuggingFaceTokenizer;

    const SPECIAL_TOKENS: &[&str] = &[
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<img>",
        "</img>",
        "<think>",
        "</think>",
        "<answer>",
        "</answer>",
    ];

    fn configured_files(model_type: &str) -> (tempfile::TempDir, ResolvedModelFiles) {
        let directory = tempdir().expect("create model directory");
        let mut vocab = Vocab::from_iter([("<unk>".to_string(), 0_u32)]);
        for codepoint in 1_u32..=127 {
            vocab.insert(char::from_u32(codepoint).unwrap().to_string(), codepoint);
        }
        let model = BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .unk_token("<unk>".to_string())
            .build()
            .expect("build tokenizer model");
        let mut tokenizer = TokenizerBuilder::new(model);
        tokenizer.add_special_tokens(
            &SPECIAL_TOKENS
                .iter()
                .map(|token| AddedToken::from(*token, true))
                .collect::<Vec<_>>(),
        );
        let tokenizer_path = directory.path().join("tokenizer.json");
        tokenizer
            .save(&tokenizer_path, false)
            .expect("save tokenizer");
        let config_path = directory.path().join("config.json");
        fs::write(
            &config_path,
            format!(
                r#"{{"model_type":"{model_type}","max_position_embeddings":4096,"num_attention_heads":8}}"#
            ),
        )
        .expect("write model config");
        let tokenizer_config_path = directory.path().join("tokenizer_config.json");
        fs::write(&tokenizer_config_path, r#"{"eos_token":"<|im_end|>"}"#)
            .expect("write tokenizer config");
        let generation_config_path = directory.path().join("generation_config.json");
        fs::write(
            &generation_config_path,
            r#"{"eos_token_id":2,"temperature":0.6,"top_p":0.95,"top_k":20,"max_new_tokens":512}"#,
        )
        .expect("write generation config");
        let files = ResolvedModelFiles {
            tokenizer_path,
            tokenizer_config_path: Some(tokenizer_config_path),
            generation_config_path: Some(generation_config_path),
            preprocessor_config_path: None,
            chat_template_path: None,
            config_path: Some(config_path),
        };
        (directory, files)
    }

    #[test]
    fn configured_descriptions_resolve_their_serving_behavior() {
        for (description, model_type) in [
            (ModelDescription::Qwen3, "qwen3"),
            (ModelDescription::SenseNova, "neo_chat"),
            (ModelDescription::Bagel, "bagel"),
        ] {
            let (_directory, files) = configured_files(model_type);
            let tokenizer = HuggingFaceTokenizer::new(&files.tokenizer_path).unwrap();
            let profile = ModelProfile::resolve(
                description,
                description.id(),
                &files,
                &ProfileOverrides::default(),
                &tokenizer,
            )
            .unwrap();
            assert_eq!(profile.common().identity.description, description);
            assert_eq!(profile.common().context_limits.max_model_tokens, Some(4096));
            assert_eq!(
                profile.common().generation_defaults.max_output_tokens,
                Some(512)
            );
            match profile {
                ModelProfile::Qwen3(_) => assert_eq!(description, ModelDescription::Qwen3),
                ModelProfile::SenseNova(profile) => {
                    assert_eq!(description, ModelDescription::SenseNova);
                    assert_eq!(
                        profile.preprocessing.image_defaults.resolution,
                        crate::profile::omni::resolution::ResolutionName::Landscape16x9
                    );
                    assert_eq!(
                        profile.preprocessing.image_ingest.steps,
                        vec![ImageIngestStep::VitEncode]
                    );
                    assert!(profile.preprocessing.generation_policy.feedback.is_some());
                }
                ModelProfile::Bagel(profile) => {
                    assert_eq!(description, ModelDescription::Bagel);
                    assert!(profile.preprocessing.resolution_policy.allow_custom);
                    assert_eq!(
                        profile.preprocessing.image_ingest.steps,
                        vec![ImageIngestStep::VaeEncode, ImageIngestStep::VitEncode]
                    );
                    assert!(profile.preprocessing.generation_policy.feedback.is_some());
                }
                ModelProfile::MiniMaxH3(_) => {
                    assert_eq!(description, ModelDescription::MiniMaxH3);
                }
            }
        }
    }

    #[test]
    fn model_description_must_match_repository_model_type() {
        let (_directory, files) = configured_files("bagel");
        let tokenizer = HuggingFaceTokenizer::new(&files.tokenizer_path).unwrap();
        let error = ModelProfile::resolve(
            ModelDescription::SenseNova,
            "configured-model",
            &files,
            &ProfileOverrides::default(),
            &tokenizer,
        )
        .unwrap_err();
        assert!(error.to_string().contains("requires model_type"));
    }

    #[test]
    fn omni_descriptions_define_prompt_framing_and_image_geometry() {
        let (_directory, files) = configured_files("neo_chat");
        let tokenizer = HuggingFaceTokenizer::new(&files.tokenizer_path).unwrap();
        let ModelProfile::SenseNova(profile) = ModelProfile::resolve(
            ModelDescription::SenseNova,
            "sensenova",
            &files,
            &ProfileOverrides::default(),
            &tokenizer,
        )
        .unwrap() else {
            unreachable!()
        };
        let prompt = profile
            .preprocessing
            .render_prompt_ids(
                &tokenizer,
                GenerationConstraint::GenOnly,
                "draw a lighthouse",
                None,
                None,
            )
            .unwrap();
        let rendered = tokenizer.decode(&prompt, false).unwrap();
        assert!(rendered.starts_with("<|im_start|>system\nYou are an image generation"));
        assert!(rendered.contains("You support two modes:"));
        assert!(rendered.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n<img>"));
        let negative = profile
            .preprocessing
            .render_negative_prompt_ids(&tokenizer, "")
            .unwrap();
        let rendered_negative = tokenizer.decode(&negative, false).unwrap();
        assert!(rendered_negative.starts_with("<|im_start|>system\nYou are an image generation"));
        assert!(rendered_negative.ends_with("<|im_start|>assistant\n<img>"));
        let ingest = profile
            .preprocessing
            .image_ingest_for_dimensions(2048, 1152, 1)
            .unwrap();
        assert_eq!(
            ingest.step_kv_tokens,
            vec![ImageKvEffect::Exact { tokens: 2304 }]
        );
        let policy = profile
            .preprocessing
            .generation_policy_for_dimensions(2048, 1152)
            .unwrap();
        assert_eq!(
            policy.feedback.unwrap().ingest.step_kv_tokens,
            vec![ImageKvEffect::Exact { tokens: 2305 }]
        );

        let (_directory, files) = configured_files("bagel");
        let tokenizer = HuggingFaceTokenizer::new(&files.tokenizer_path).unwrap();
        let ModelProfile::Bagel(profile) = ModelProfile::resolve(
            ModelDescription::Bagel,
            "bagel",
            &files,
            &ProfileOverrides::default(),
            &tokenizer,
        )
        .unwrap() else {
            unreachable!()
        };
        let prompt = profile
            .preprocessing
            .render_prompt_ids(
                &tokenizer,
                GenerationConstraint::Default,
                false,
                "draw a lighthouse",
                None,
                None,
            )
            .unwrap();
        let rendered = tokenizer.decode(&prompt, false).unwrap();
        assert!(rendered.starts_with("<|im_start|>You should first think"));
        assert!(rendered.ends_with("<|im_start|>assistant\n"));
        let ingest = profile
            .preprocessing
            .image_ingest_for_dimensions(1024, 512, 1)
            .unwrap();
        assert_eq!(
            ingest.step_kv_tokens,
            vec![
                ImageKvEffect::Exact { tokens: 2050 },
                ImageKvEffect::Exact { tokens: 2452 },
            ]
        );
        let policy = profile
            .preprocessing
            .generation_policy_for_dimensions(512, 512)
            .unwrap();
        assert_eq!(
            policy.feedback.unwrap().ingest.step_kv_tokens,
            vec![ImageKvEffect::Exact { tokens: 1026 }]
        );
    }
}
