//! Model-profile selection and load-time serving capabilities.
//!
//! The server identifies the served model family from the checkpoint itself:
//! the `model_type` of its root architecture config (see
//! `ResolvedModelFiles::config_path`) or, for a diffusers pipeline, the
//! `_class_name` of its root index (see [`assets::pipeline_index`]). The
//! result is one [`ModelConfig`], built at startup by `ModelConfig::load` in
//! `serving::model`. Engine startup reads it, and `InputProcessor::new` then
//! binds the worker's capabilities onto it before request preprocessing and
//! discovery share it immutably.
//!
//! Submodules own checkpoint asset resolution ([`assets`]), the multimodal
//! family profiles ([`omni`]), streaming reasoning and tool-call parsers
//! ([`reasoning`], [`tools`]), and the tokenizer ([`tokenizer`]).

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
    /// Every served model family.
    ///
    /// `from_model_type` and `from_pipeline_class` search only this list, so a
    /// variant missing from it compiles but is never selected from checkpoint
    /// metadata.
    const ALL: [Self; 4] = [Self::Qwen3, Self::SenseNova, Self::Bagel, Self::MiniMaxH3];

    /// Returns the stable profile identifier.
    pub const fn id(self) -> &'static str {
        match self {
            Self::Qwen3 => "qwen3",
            Self::SenseNova => "sensenova",
            Self::Bagel => "bagel",
            Self::MiniMaxH3 => "minimax_h3",
        }
    }

    /// Returns the `model_type` a root `config.json` declares for this family,
    /// or `None` for a family that ships as a diffusers pipeline.
    const fn model_type(self) -> Option<&'static str> {
        match self {
            Self::Qwen3 => Some("qwen3"),
            Self::SenseNova => Some("neo_chat"),
            Self::Bagel => Some("bagel"),
            Self::MiniMaxH3 => None,
        }
    }

    /// Returns the pipeline class a family that ships as a diffusers pipeline
    /// declares in its root index.
    const fn pipeline_class(self) -> Option<&'static str> {
        match self {
            Self::MiniMaxH3 => Some("MiniMaxH3ModularPipeline"),
            Self::Qwen3 | Self::SenseNova | Self::Bagel => None,
        }
    }

    /// Resolves the served profile from a checkpoint's `model_type` field.
    ///
    /// The checkpoint configuration is the only source of model identity.
    /// `None` means UniServe serves no family with this `model_type` from a
    /// root configuration; pipeline families never match here.
    pub fn from_model_type(model_type: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|description| description.model_type() == Some(model_type))
    }

    /// Resolves the served profile from a pipeline checkpoint's `_class_name`.
    pub fn from_pipeline_class(class_name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|description| description.pipeline_class() == Some(class_name))
    }
}

impl std::str::FromStr for ModelDescription {
    type Err = ModelDescriptionParseError;

    /// Parses a profile identifier as returned by [`ModelDescription::id`],
    /// also accepting the hyphenated spelling `minimax-h3`.
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
    /// Default generation length from `max_new_tokens`, used only when the
    /// request omits `max_tokens`; an explicit request value replaces it.
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
        ///
        /// [`ModelConfig::from_pipeline`] leaves it zero; `InputProcessor::new`
        /// binds the count the worker advertises in its startup handshake.
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
    ///
    /// `InputProcessor::new` replaces it with
    /// `WorkerCapabilities::max_model_tokens`, which `build_state` in the crate
    /// root computes as the smaller of this value and the started engine's
    /// `max_model_len`.
    pub max_model_tokens: Option<u32>,
    /// Tokenizer-config `eos_token` resolved through the vocabulary.
    ///
    /// `special_token_ids` in the crate root places it first in the engine's
    /// EOS list unless the profile's `GenerationControls` carry a nonzero
    /// `eos`, which takes that place instead.
    pub primary_eos_token_id: Option<u32>,
    /// Complete EOS set resolved from tokenizer and generation metadata.
    pub eos_token_ids: BTreeSet<u32>,
}

impl ModelConfig {
    /// Resolves vocabulary, checkpoint defaults, and model-specific settings once.
    ///
    /// The family comes from the `model_type` of `files.config_path`; a
    /// configured `max_model_tokens` takes precedence over the checkpoint's
    /// `max_position_embeddings`.
    ///
    /// # Errors
    ///
    /// Fails when a present metadata file cannot be read or parsed, when
    /// `model_type` is missing or names no family served from a root
    /// configuration, or when a multimodal profile cannot resolve its control
    /// tokens from `tokenizer`.
    pub fn from_files(
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
        let description =
            ModelDescription::from_model_type(actual_model_type).ok_or_else(|| {
                assets::Error::UnsupportedModelType {
                    actual: actual_model_type.to_owned(),
                }
            })?;

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
            // `from_model_type` resolves only families described by a root
            // configuration; a pipeline family resolves through `from_pipeline`.
            ModelDescription::MiniMaxH3 => {
                return Err(assets::Error::UnsupportedModelType {
                    actual: actual_model_type.to_owned(),
                });
            }
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

    /// Resolves a family that ships as a diffusers pipeline.
    ///
    /// A pipeline checkpoint has no root generation or tokenizer metadata, so
    /// the result carries no sampling defaults and no EOS tokens. Its
    /// denoising step count belongs to the loaded numerical plan, which the
    /// worker handshake binds. `max_model_tokens` overrides the family's prompt
    /// bound.
    ///
    /// # Errors
    ///
    /// Returns [`assets::Error::UnsupportedPipeline`] for a family that does
    /// not ship as a pipeline.
    pub fn from_pipeline(
        model_id: &str,
        description: ModelDescription,
        max_video_seconds: f64,
        max_model_tokens: Option<u32>,
    ) -> assets::Result<Self> {
        let (parameters, default_max_model_tokens) = match description {
            // MiniMax H3's text encoder serves prompts of up to 16,384 tokens.
            ModelDescription::MiniMaxH3 => (
                ModelParameters::MiniMaxH3 {
                    max_video_seconds,
                    num_inference_steps: 0,
                },
                16_384,
            ),
            ModelDescription::Qwen3 | ModelDescription::SenseNova | ModelDescription::Bagel => {
                return Err(assets::Error::UnsupportedPipeline {
                    class_name: description.id().to_owned(),
                });
            }
        };

        Ok(Self {
            served_name: model_id.to_owned(),
            parameters,
            sampling_defaults: SamplingDefaults::default(),
            max_model_tokens: Some(max_model_tokens.unwrap_or(default_max_model_tokens)),
            primary_eos_token_id: None,
            eos_token_ids: BTreeSet::new(),
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
///
/// Returns the tokenizer-config `eos_token` as the primary id and the union of
/// the generation-config `eos_token_id` values with that primary. An
/// `eos_token` absent from the vocabulary yields no primary rather than an
/// error.
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

    /// Writes a minimal checkpoint for `model_type`: a character-level BPE
    /// tokenizer over ASCII plus [`SPECIAL_TOKENS`], and root, tokenizer, and
    /// generation configs. Dropping the returned directory deletes the files.
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
                ModelConfig::from_files(description.id(), &files, None, &tokenizer).unwrap();
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
    fn a_pipeline_family_resolves_from_its_pipeline_class() {
        assert_eq!(
            ModelDescription::from_pipeline_class("MiniMaxH3ModularPipeline"),
            Some(ModelDescription::MiniMaxH3)
        );
        assert_eq!(ModelDescription::from_model_type("minimax_h3"), None);

        let profile =
            ModelConfig::from_pipeline("h3", ModelDescription::MiniMaxH3, 15.0, None).unwrap();
        assert_eq!(profile.description(), ModelDescription::MiniMaxH3);
        assert_eq!(profile.max_model_tokens, Some(16_384));
    }

    #[test]
    fn unsupported_repository_model_type_is_rejected() {
        let (_directory, files) = configured_files("llama");
        let tokenizer = HuggingFaceTokenizer::new(&files.tokenizer_path).unwrap();
        let error =
            ModelConfig::from_files("configured-model", &files, None, &tokenizer).unwrap_err();
        assert!(matches!(
            error,
            crate::profile::assets::Error::UnsupportedModelType { actual } if actual == "llama"
        ));
    }

    #[test]
    fn omni_descriptions_define_prompt_framing_and_image_geometry() {
        let (_directory, files) = configured_files("neo_chat");
        let tokenizer = HuggingFaceTokenizer::new(&files.tokenizer_path).unwrap();
        let config = ModelConfig::from_files("sensenova", &files, None, &tokenizer).unwrap();
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

        // 2048x1152 is already 32-aligned and inside the pixel bounds, so it
        // yields a 64x36 grid; generated-image feedback adds one marker token.
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
        let config = ModelConfig::from_files("bagel", &files, None, &tokenizer).unwrap();
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

        // VAE: 1024x512 at stride 16 is a 64x32 grid. ViT: the VAE canvas is
        // resized to 980x490 at stride 14, a 70x35 grid. Both add two markers.
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
