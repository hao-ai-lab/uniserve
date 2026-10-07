//! Deserialized tokenizer and processor metadata used during profile resolution.
//!
//! Each loader treats an absent file (a `None` path) as an empty document and
//! fills every field with its serde default, so a missing optional file never
//! fails resolution. Fields a present file omits also take their defaults,
//! and unknown JSON fields are ignored.

use std::collections::BTreeSet;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::profile::assets::error::{Error, Result};

/// Tokenizer metadata consumed by configured chat rendering and stop handling.
///
/// Read from `tokenizer_config.json` by `ModelConfig::from_files` (for the
/// primary EOS) and by `HfChatRenderer::load` (for the template and the
/// special tokens exposed to it).
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct HfTokenizerConfig {
    /// Named tokenizer special tokens.
    #[serde(flatten)]
    pub special_tokens: HfSpecialTokens,
    /// Embedded default chat template, when configured.
    pub chat_template: Option<String>,
}

/// A named Hugging Face special token represented as text or an object.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum NamedSpecialToken {
    /// Token represented directly as text.
    Text(String),
    /// Token represented by an object containing its text.
    WithContent {
        /// Token text.
        content: String,
    },
}

impl Serialize for NamedSpecialToken {
    /// Serializes either form as its plain token text, which is the value a
    /// chat template sees for `bos_token` and the other special tokens.
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl From<NamedSpecialToken> for String {
    /// Extracts the token text from either form.
    fn from(value: NamedSpecialToken) -> Self {
        match value {
            NamedSpecialToken::Text(string) => string,
            NamedSpecialToken::WithContent { content } => content,
        }
    }
}

impl NamedSpecialToken {
    /// Returns the token text.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Text(value) | Self::WithContent { content: value } => value,
        }
    }
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
/// Special-token strings resolved from tokenizer metadata.
///
/// Serialization omits unset tokens, so a chat template sees them as
/// undefined rather than null.
pub struct HfSpecialTokens {
    /// Beginning-of-sequence token.
    pub bos_token: Option<NamedSpecialToken>,
    /// End-of-sequence token.
    pub eos_token: Option<NamedSpecialToken>,
    /// Unknown-token marker.
    pub unk_token: Option<NamedSpecialToken>,
    /// Padding token.
    pub pad_token: Option<NamedSpecialToken>,
}

impl HfSpecialTokens {
    /// Returns whether no special-token value is configured.
    pub fn is_empty(&self) -> bool {
        self.bos_token.is_none()
            && self.eos_token.is_none()
            && self.unk_token.is_none()
            && self.pad_token.is_none()
    }
}

/// Model metadata used to bind a root-configured checkpoint to its description.
///
/// This is the raw view of the file `ResolvedModelFiles::config_path` selects,
/// distinct from the resolved `crate::profile::ModelConfig`. Composite
/// checkpoints nest their language model under `llm_config` (the SenseNova
/// and Bagel layouts) or `text_config` (DiffusionGemma); the root
/// `model_type` still identifies the family.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    model_type: Option<String>,
    max_position_embeddings: Option<u32>,
    /// Nested language-model section of a SenseNova or Bagel checkpoint.
    llm_config: Option<Box<ModelConfig>>,
    /// Nested language-model section of a DiffusionGemma checkpoint.
    text_config: Option<Box<ModelConfig>>,
}

/// Generation defaults consumed by configured request lowering.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct GenerationConfig {
    /// One or more token identifiers that terminate generation.
    pub eos_token_id: Option<OneOrManyTokenIds>,
    /// Default sampling temperature.
    pub temperature: Option<f32>,
    /// Default nucleus-sampling probability mass.
    pub top_p: Option<f32>,
    /// Default candidate-token limit.
    pub top_k: Option<u32>,
    /// Default minimum relative token probability.
    pub min_p: Option<f32>,
    /// Default repetition penalty.
    pub repetition_penalty: Option<f32>,
    /// Default maximum number of generated tokens.
    pub max_new_tokens: Option<u32>,
    /// Block-diffusion denoising steps per canvas.
    pub max_denoising_steps: Option<u32>,
    /// Block-diffusion sampler settings.
    pub sampler_config: Option<DiffusionSamplerConfig>,
    /// Block-diffusion sampling temperature at the first step.
    pub t_min: Option<f32>,
    /// Block-diffusion sampling temperature at the last step.
    pub t_max: Option<f32>,
    /// Block-diffusion stopping threshold on residual canvas uncertainty.
    pub confidence_threshold: Option<f32>,
    /// Consecutive unchanged block-diffusion steps that end denoising early.
    pub stability_threshold: Option<u32>,
}

/// The `sampler_config` section of a block-diffusion `generation_config.json`.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct DiffusionSamplerConfig {
    /// Entropy budget, in nats, of the tokens one denoising step accepts.
    pub entropy_bound: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
/// Token-id field accepted as one identifier or a list, as
/// `generation_config.json` writes `eos_token_id`.
pub enum OneOrManyTokenIds {
    /// One token identifier.
    One(u32),
    /// Ordered token identifier list.
    Many(Vec<u32>),
}

impl OneOrManyTokenIds {
    /// Converts one-or-many identifiers into a deduplicated ordered set.
    pub fn into_set(self) -> BTreeSet<u32> {
        match self {
            Self::One(id) => BTreeSet::from([id]),
            Self::Many(ids) => ids.into_iter().collect(),
        }
    }
}

impl ModelConfig {
    /// Returns the nested `llm_config` or `text_config` section when present,
    /// otherwise the root configuration.
    fn language_model(&self) -> &Self {
        self.llm_config
            .as_deref()
            .or(self.text_config.as_deref())
            .unwrap_or(self)
    }

    /// Returns the configured model-type discriminator.
    pub fn model_type(&self) -> Option<&str> {
        self.model_type.as_deref()
    }

    /// Returns the language model's declared maximum position count.
    ///
    /// Reads only the nested language-model section when it exists, without
    /// falling back to a root-level value.
    pub fn max_position_embeddings(&self) -> Option<u32> {
        self.language_model().max_position_embeddings
    }
}

/// Loads tokenizer metadata, returning defaults when no file is configured.
pub fn load_tokenizer_config(path: Option<&Path>) -> Result<HfTokenizerConfig> {
    read_json_file(path)
}

/// Loads generation defaults, returning defaults when no file is configured.
pub fn load_generation_config(path: Option<&Path>) -> Result<GenerationConfig> {
    read_json_file(path)
}

/// Loads model metadata, returning defaults when no file is configured.
pub fn load_model_config(path: Option<&Path>) -> Result<ModelConfig> {
    read_json_file(path)
}

/// Deserializes the JSON file at `path`, or returns `T::default()` for `None`.
///
/// A present but unreadable or malformed file is an error.
fn read_json_file<T>(path: Option<&Path>) -> Result<T>
where
    T: for<'de> Deserialize<'de> + Default,
{
    let Some(path) = path else {
        return Ok(T::default());
    };
    read_json(path)
}

/// Deserializes the JSON file at `path`.
///
/// # Errors
///
/// Returns [`Error::Io`] for an unreadable file and [`Error::Json`] for
/// content that does not deserialize into `T`, including a missing required
/// field.
pub fn read_json<T>(path: &Path) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let content = std::fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&content).map_err(|source| Error::Json {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::ModelConfig;

    /// A flat Qwen3 layout and the nested SenseNova, Bagel, and DiffusionGemma
    /// layouts: the family comes from the root `model_type` and the context
    /// limit from the language-model section.
    #[test]
    fn configured_model_layouts_expose_context_limits() {
        for (source, model_type, max_tokens) in [
            (
                r#"{"model_type":"qwen3","num_attention_heads":64,"max_position_embeddings":40960}"#,
                "qwen3",
                40960,
            ),
            (
                r#"{"model_type":"neo_chat","llm_config":{"model_type":"qwen3","num_attention_heads":32,"max_position_embeddings":262144}}"#,
                "neo_chat",
                262144,
            ),
            (
                r#"{"model_type":"bagel","llm_config":{"model_type":"qwen2","num_attention_heads":28,"max_position_embeddings":32768}}"#,
                "bagel",
                32768,
            ),
            (
                r#"{"model_type":"diffusion_gemma","canvas_length":256,"text_config":{"model_type":"diffusion_gemma_text","max_position_embeddings":262144}}"#,
                "diffusion_gemma",
                262144,
            ),
        ] {
            let config: ModelConfig = serde_json::from_str(source).unwrap();
            assert_eq!(config.model_type(), Some(model_type));
            assert_eq!(config.max_position_embeddings(), Some(max_tokens));
        }
    }
}
