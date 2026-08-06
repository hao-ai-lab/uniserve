//! Closed model-profile resolution for the configured serving set.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::collections::BTreeSet;
use std::fs;

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use crate::assets::{
    GenerationConfig, HfTokenizerConfig, ResolvedModelFiles, load_generation_config,
    load_model_config, load_tokenizer_config,
};
use crate::omni::bagel::BagelProfile;
use crate::omni::sensenova::SenseNovaProfile;
use crate::tokenizer::HuggingFaceTokenizer;

pub mod assets;
pub mod omni;
pub mod reasoning;
pub mod tokenizer;
pub mod tools;

/// The complete configured model-description set selected at server startup.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelDescription {
    #[default]
    Qwen3,
    #[serde(rename = "sensenova")]
    SenseNova,
    Bagel,
}

impl ModelDescription {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Qwen3 => "qwen3",
            Self::SenseNova => "sensenova",
            Self::Bagel => "bagel",
        }
    }

    const fn model_type(self) -> &'static str {
        match self {
            Self::Qwen3 => "qwen3",
            Self::SenseNova => "neo_chat",
            Self::Bagel => "bagel",
        }
    }
}

/// Deployment-owned profile inputs applied after repository metadata.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileDeploymentConfig {
    pub chat_template_override: Option<String>,
    pub max_model_tokens: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelIdentity {
    pub model_id: String,
    pub profile_id: String,
    pub description_id: String,
    pub config_fingerprint: String,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommonModelProfile {
    pub identity: ModelIdentity,
    pub generation_defaults: GenerationDefaultsDescriptor,
    pub context_limits: ContextLimits,
    pub stop_tokens: StopTokenPolicy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen3ModelProfile {
    pub common: CommonModelProfile,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SenseNovaModelProfile {
    pub common: CommonModelProfile,
    pub preprocessing: SenseNovaProfile,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BagelModelProfile {
    pub common: CommonModelProfile,
    pub preprocessing: BagelProfile,
}

/// The closed resolved profile value consumed by the serving model description.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelProfile {
    Qwen3(Qwen3ModelProfile),
    SenseNova(SenseNovaModelProfile),
    Bagel(BagelModelProfile),
}

impl ModelProfile {
    pub fn resolve(
        description: ModelDescription,
        model_id: &str,
        files: &ResolvedModelFiles,
        deployment: &ProfileDeploymentConfig,
        tokenizer: &HuggingFaceTokenizer,
    ) -> assets::Result<Self> {
        let model_config = load_model_config(files.config_path.as_deref())?;
        let actual_model_type = model_config
            .model_type()
            .ok_or_else(|| assets::Error::message("configured model config has no model_type"))?;
        if actual_model_type != description.model_type() {
            return Err(assets::Error::message(format!(
                "model description {} requires model_type {:?}, found {:?}",
                description.id(),
                description.model_type(),
                actual_model_type
            )));
        }
        let generation_config = load_generation_config(files.generation_config_path.as_deref())?;
        let tokenizer_config = load_tokenizer_config(files.tokenizer_config_path.as_deref())?;
        let config_fingerprint = profile_fingerprint(files, deployment)?;
        let description_id = description.id().to_string();
        let common = CommonModelProfile {
            identity: ModelIdentity {
                model_id: model_id.to_string(),
                profile_id: format!("{description_id}:{}", &config_fingerprint[..16]),
                description_id,
                config_fingerprint,
            },
            generation_defaults: generation_defaults(&generation_config),
            context_limits: ContextLimits {
                max_model_tokens: deployment
                    .max_model_tokens
                    .or(model_config.max_position_embeddings()),
                max_output_tokens: generation_config.max_new_tokens,
            },
            stop_tokens: stop_token_policy(&tokenizer_config, &generation_config, tokenizer),
        };
        match description {
            ModelDescription::Qwen3 => Ok(Self::Qwen3(Qwen3ModelProfile { common })),
            ModelDescription::SenseNova => Ok(Self::SenseNova(SenseNovaModelProfile {
                common,
                preprocessing: SenseNovaProfile::resolve(tokenizer)?,
            })),
            ModelDescription::Bagel => Ok(Self::Bagel(BagelModelProfile {
                common,
                preprocessing: BagelProfile::resolve(tokenizer)?,
            })),
        }
    }

    pub fn common(&self) -> &CommonModelProfile {
        match self {
            Self::Qwen3(profile) => &profile.common,
            Self::SenseNova(profile) => &profile.common,
            Self::Bagel(profile) => &profile.common,
        }
    }

    pub fn common_mut(&mut self) -> &mut CommonModelProfile {
        match self {
            Self::Qwen3(profile) => &mut profile.common,
            Self::SenseNova(profile) => &mut profile.common,
            Self::Bagel(profile) => &mut profile.common,
        }
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

fn profile_fingerprint(
    files: &ResolvedModelFiles,
    deployment: &ProfileDeploymentConfig,
) -> assets::Result<String> {
    let mut hasher = Sha256::new();
    let paths = std::iter::once(files.tokenizer_path.as_path()).chain(
        [
            files.config_path.as_deref(),
            files.generation_config_path.as_deref(),
            files.tokenizer_config_path.as_deref(),
            files.preprocessor_config_path.as_deref(),
            files.chat_template_path.as_deref(),
        ]
        .into_iter()
        .flatten(),
    );
    for path in paths {
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

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;
    use tokenizers::models::bpe::{BPE, Vocab};
    use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};
    use uniserve_core::{GenerationConstraint, ImageIngestStep, ImageKvEffect};

    use super::{ModelDescription, ModelProfile, ProfileDeploymentConfig};
    use crate::assets::ResolvedModelFiles;
    use crate::tokenizer::HuggingFaceTokenizer;

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
    fn configured_descriptions_resolve_their_serving_contracts() {
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
                &ProfileDeploymentConfig::default(),
                &tokenizer,
            )
            .unwrap();
            assert_eq!(profile.common().identity.description_id, description.id());
            assert_eq!(profile.common().context_limits.max_model_tokens, Some(4096));
            assert_eq!(
                profile.common().generation_defaults.max_output_tokens,
                Some(512)
            );
            match profile {
                ModelProfile::Qwen3(_) => assert_eq!(description, ModelDescription::Qwen3),
                ModelProfile::SenseNova(profile) => {
                    assert_eq!(description, ModelDescription::SenseNova);
                    assert_eq!(profile.preprocessing.image_defaults.resolution, "16:9");
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
            &ProfileDeploymentConfig::default(),
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
            &ProfileDeploymentConfig::default(),
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
        let ingest = profile
            .preprocessing
            .image_ingest_for_dimensions(2048, 1152, 1)
            .unwrap();
        assert_eq!(
            ingest.step_kv_tokens,
            vec![ImageKvEffect::Exact { tokens: 2304 }]
        );

        let (_directory, files) = configured_files("bagel");
        let tokenizer = HuggingFaceTokenizer::new(&files.tokenizer_path).unwrap();
        let ModelProfile::Bagel(profile) = ModelProfile::resolve(
            ModelDescription::Bagel,
            "bagel",
            &files,
            &ProfileDeploymentConfig::default(),
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
    }
}
