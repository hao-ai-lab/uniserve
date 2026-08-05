//! Resolved model behavior snapshots used by the serving runtime.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::collections::BTreeSet;
use std::fs;

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use crate::assets::{
    GenerationConfig, HfTokenizerConfig, ResolvedModelFiles, load_generation_config,
    load_model_config, load_tokenizer_config,
};
use crate::dialect::{GenerationDialectProfile, resolve_generation_dialect};
use crate::tokenizer::Tokenizer;

pub mod assets;
pub mod dialect;
pub mod reasoning;
pub mod tokenizer;
pub mod tools;

/// The complete configured model-description set.
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
    pub const fn family_id(self) -> &'static str {
        match self {
            Self::Qwen3 => "qwen3",
            Self::SenseNova => "sensenova",
            Self::Bagel => "bagel",
        }
    }
}

/// Deployment-owned profile inputs. Repository metadata is resolved first and
/// these typed values are then applied as the highest deployment precedence.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileDeploymentConfig {
    pub chat_template_override: Option<String>,
    pub max_model_tokens: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelIdentity {
    pub model_id: String,
    pub profile_id: String,
    pub family_id: String,
    pub dialect_id: String,
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

/// Stable, fully resolved identity, policy, and dialect snapshot for one
/// single-model runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelProfile {
    pub description: ModelDescription,
    pub identity: ModelIdentity,
    pub generation_defaults: GenerationDefaultsDescriptor,
    pub context_limits: ContextLimits,
    pub stop_tokens: StopTokenPolicy,
    pub generation_dialect: Option<GenerationDialectProfile>,
}

impl ModelProfile {
    /// Resolve one profile using deterministic precedence: built-in family
    /// defaults, model repository files, then typed deployment configuration.
    pub fn resolve(
        description: ModelDescription,
        model_id: &str,
        files: &ResolvedModelFiles,
        deployment: &ProfileDeploymentConfig,
        tokenizer: &dyn Tokenizer,
    ) -> assets::Result<Self> {
        let model_config = load_model_config(files.config_path.as_deref())?;
        let generation_config = load_generation_config(files.generation_config_path.as_deref())?;
        let tokenizer_config = load_tokenizer_config(files.tokenizer_config_path.as_deref())?;
        let generation_dialect = resolve_generation_dialect(description, tokenizer)?;
        let config_fingerprint = profile_fingerprint(files, deployment)?;
        let family_id = description.family_id().to_string();
        let profile_id = format!("{family_id}:{}", &config_fingerprint[..16]);
        let stop_tokens = stop_token_policy(&tokenizer_config, &generation_config, tokenizer);
        let repository_max_tokens = model_config.max_position_embeddings();
        let max_model_tokens = deployment.max_model_tokens.or(repository_max_tokens);
        let dialect_id = generation_dialect
            .as_ref()
            .map_or_else(|| family_id.clone(), |dialect| dialect.id.clone());

        Ok(Self {
            description,
            identity: ModelIdentity {
                model_id: model_id.to_string(),
                profile_id,
                family_id,
                dialect_id,
                config_fingerprint,
            },
            generation_defaults: generation_defaults(&generation_config),
            context_limits: ContextLimits {
                max_model_tokens,
                max_output_tokens: generation_config.max_new_tokens,
            },
            stop_tokens,
            generation_dialect,
        })
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
