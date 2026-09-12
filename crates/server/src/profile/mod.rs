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

/// Model-provided sampling defaults. `None` remains distinct from an explicit zero.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SamplingDefaults {
    /// Temperature used when the request omits it; zero selects greedy sampling.
    pub temperature: Option<f32>,
    /// Default nucleus probability mass.
    pub top_p: Option<f32>,
    /// Default candidate limit; zero disables top-k filtering.
    pub top_k: Option<u32>,
    /// Default minimum probability relative to the most likely token.
    pub min_p: Option<f32>,
    /// Default multiplicative penalty for repeated tokens.
    pub repetition_penalty: Option<f32>,
    /// Checkpoint ceiling on generated token count.
    pub max_output_tokens: Option<u32>,
}

/// Model-specific numerical and prompt settings; the selected variant owns its facts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelParameters {
    /// Qwen3 chat framing and structured text output.
    Qwen3,
    /// SenseNova image, prompt, and generation settings.
    SenseNova(SenseNovaProfile),
    /// Bagel image, prompt, and generation settings.
    Bagel(BagelProfile),
    /// Fast H3 checkpoint and duration limits.
    MiniMaxH3 {
        /// Maximum requested duration in seconds, before frame alignment.
        max_video_seconds: f64,
        /// Fixed number of denoising predictions in the checkpoint contract.
        num_inference_steps: u32,
    },
}

/// Immutable model facts shared by startup, request preprocessing, and discovery.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name exposed by discovery and generation responses.
    pub served_name: String,
    /// Loaded model-specific settings, including the family identity.
    pub parameters: ModelParameters,
    /// Checkpoint defaults applied only to omitted request fields.
    pub sampling_defaults: SamplingDefaults,
    /// Combined input/output token ceiling from metadata and configuration.
    pub max_model_tokens: Option<u32>,
    /// Canonical tokenizer EOS, placed first when starting the engine.
    pub primary_eos_token_id: Option<u32>,
    /// Complete EOS set resolved from tokenizer and generation metadata.
    pub eos_token_ids: BTreeSet<u32>,
}

impl ModelConfig {
    /// Resolves vocabulary, checkpoint defaults, and model-specific settings once.
    pub fn from_files(
        description: ModelDescription,
        model_id: &str,
        files: &ResolvedModelFiles,
        max_model_tokens: Option<u32>,
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
        let (primary_eos_token_id, eos_token_ids) =
            stop_token_ids(&tokenizer_config, &generation_config, tokenizer);
        let parameters = match description {
            ModelDescription::Qwen3 => ModelParameters::Qwen3,
            ModelDescription::SenseNova => {
                ModelParameters::SenseNova(SenseNovaProfile::resolve(tokenizer)?)
            }
            ModelDescription::Bagel => ModelParameters::Bagel(BagelProfile::resolve(tokenizer)?),
            ModelDescription::MiniMaxH3 => ModelParameters::MiniMaxH3 {
                max_video_seconds: 15.0,
                num_inference_steps: 4,
            },
        };
        Ok(Self {
            served_name: model_id.to_owned(),
            parameters,
            sampling_defaults: generation_defaults(&generation_config),
            max_model_tokens: max_model_tokens.or(model_config.max_position_embeddings()),
            primary_eos_token_id,
            eos_token_ids,
        })
    }

    /// Stable public model-family identifier derived from the loaded settings.
    pub const fn description(&self) -> ModelDescription {
        match &self.parameters {
            ModelParameters::Qwen3 => ModelDescription::Qwen3,
            ModelParameters::SenseNova(_) => ModelDescription::SenseNova,
            ModelParameters::Bagel(_) => ModelDescription::Bagel,
            ModelParameters::MiniMaxH3 { .. } => ModelDescription::MiniMaxH3,
        }
    }
}

/// Extracts serving generation defaults from repository configuration.
fn generation_defaults(config: &GenerationConfig) -> SamplingDefaults {
    SamplingDefaults {
        temperature: config.temperature,
        top_p: config.top_p,
        top_k: config.top_k,
        min_p: config.min_p,
        repetition_penalty: config.repetition_penalty,
        max_output_tokens: config.max_new_tokens,
    }
}

/// Combines tokenizer and generation metadata into one canonical termination policy.
fn stop_token_ids(
    tokenizer_config: &HfTokenizerConfig,
    generation_config: &GenerationConfig,
    tokenizer: &HuggingFaceTokenizer,
) -> (Option<u32>, BTreeSet<u32>) {
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
    (primary, ids)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;
    use tokenizers::models::bpe::{BPE, Vocab};
    use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};
    use uniserve_core::{GenerationConstraint, ImageIngestStep};

    use super::{ModelConfig, ModelDescription, ModelParameters};
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
            let profile =
                ModelConfig::from_files(description, description.id(), &files, None, &tokenizer)
                    .unwrap();
            assert_eq!(profile.description(), description);
            assert_eq!(profile.max_model_tokens, Some(4096));
            assert_eq!(profile.sampling_defaults.max_output_tokens, Some(512));
            match profile.parameters {
                ModelParameters::Qwen3 => assert_eq!(description, ModelDescription::Qwen3),
                ModelParameters::SenseNova(profile) => {
                    assert_eq!(description, ModelDescription::SenseNova);
                    assert_eq!(
                        profile.image_defaults.resolution,
                        crate::profile::omni::resolution::ResolutionName::Landscape16x9
                    );
                    assert_eq!(
                        profile
                            .image_encoders
                            .iter()
                            .map(|input| input.encoder)
                            .collect::<Vec<_>>(),
                        vec![ImageIngestStep::VitEncode]
                    );
                    assert!(profile.image_generation.feedback_source.is_some());
                }
                ModelParameters::Bagel(profile) => {
                    assert_eq!(description, ModelDescription::Bagel);
                    assert!(profile.resolution_policy.allow_custom);
                    assert_eq!(
                        profile
                            .image_encoders
                            .iter()
                            .map(|input| input.encoder)
                            .collect::<Vec<_>>(),
                        vec![ImageIngestStep::VaeEncode, ImageIngestStep::VitEncode]
                    );
                    assert!(profile.image_generation.feedback_source.is_some());
                }
                ModelParameters::MiniMaxH3 { .. } => {
                    assert_eq!(description, ModelDescription::MiniMaxH3);
                }
            }
        }
    }

    #[test]
    fn model_description_must_match_repository_model_type() {
        let (_directory, files) = configured_files("bagel");
        let tokenizer = HuggingFaceTokenizer::new(&files.tokenizer_path).unwrap();
        let error = ModelConfig::from_files(
            ModelDescription::SenseNova,
            "configured-model",
            &files,
            None,
            &tokenizer,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            crate::profile::assets::Error::ModelTypeMismatch { expected: "neo_chat", actual } if actual == "bagel"
        ));
    }

    #[test]
    fn omni_descriptions_define_prompt_framing_and_image_geometry() {
        let (_directory, files) = configured_files("neo_chat");
        let tokenizer = HuggingFaceTokenizer::new(&files.tokenizer_path).unwrap();
        let config = ModelConfig::from_files(
            ModelDescription::SenseNova,
            "sensenova",
            &files,
            None,
            &tokenizer,
        )
        .unwrap();
        let ModelParameters::SenseNova(profile) = config.parameters else {
            unreachable!()
        };
        let prompt = profile
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
        let negative = profile.render_negative_prompt_ids(&tokenizer, "").unwrap();
        let rendered_negative = tokenizer.decode(&negative, false).unwrap();
        assert!(rendered_negative.starts_with("<|im_start|>system\nYou are an image generation"));
        assert!(rendered_negative.ends_with("<|im_start|>assistant\n<img>"));
        let ingest = profile
            .image_encoders_for_dimensions(2048, 1152, 1)
            .unwrap();
        assert_eq!(
            ingest
                .iter()
                .map(|input| input.num_kv_tokens)
                .collect::<Vec<_>>(),
            vec![Some(2304)]
        );
        let policy = profile.image_generation_for_dimensions(2048, 1152).unwrap();
        assert_eq!(
            policy
                .feedback_encoders
                .iter()
                .map(|input| input.num_kv_tokens)
                .collect::<Vec<_>>(),
            vec![Some(2305)]
        );

        let (_directory, files) = configured_files("bagel");
        let tokenizer = HuggingFaceTokenizer::new(&files.tokenizer_path).unwrap();
        let config =
            ModelConfig::from_files(ModelDescription::Bagel, "bagel", &files, None, &tokenizer)
                .unwrap();
        let ModelParameters::Bagel(profile) = config.parameters else {
            unreachable!()
        };
        let prompt = profile
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
        let ingest = profile.image_encoders_for_dimensions(1024, 512, 1).unwrap();
        assert_eq!(
            ingest
                .iter()
                .map(|input| input.num_kv_tokens)
                .collect::<Vec<_>>(),
            vec![Some(2050), Some(2452),]
        );
        let policy = profile.image_generation_for_dimensions(512, 512).unwrap();
        assert_eq!(
            policy
                .feedback_encoders
                .iter()
                .map(|input| input.num_kv_tokens)
                .collect::<Vec<_>>(),
            vec![Some(1026)]
        );
    }
}
