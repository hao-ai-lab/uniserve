//! Root component index of a diffusers pipeline checkpoint.
//!
//! A pipeline checkpoint names its pipeline class and component folders in a
//! root index instead of a root `config.json`. The classic layout publishes
//! `model_index.json`, where each component lives in the folder named by its
//! key; the modular layout publishes `modular_model_index.json`, where a
//! component's specification may name its folder explicitly.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::Value;

use crate::profile::assets::error::{Error, Result};
use crate::profile::assets::model_files::resolve_model_file;

/// Index files consulted in order: the classic pipeline index, then the modular one.
const PIPELINE_INDEX_FILES: [&str; 2] = ["model_index.json", "modular_model_index.json"];

/// Pipeline class and component folders declared by a checkpoint's root index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineIndex {
    /// Pipeline class the index declares in `_class_name`.
    pub class_name: String,
    /// Component name mapped to the checkpoint folder holding its files.
    components: BTreeMap<String, String>,
}

impl PipelineIndex {
    /// Parses a root index.
    ///
    /// Keys starting with `_` are index metadata. Every other key whose value
    /// is a component specification (`[library, class, options?]`) names a
    /// component; its folder is the `subfolder` option when present and the
    /// key otherwise.
    pub fn parse(content: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(content)
            .map_err(|error| Error::invalid(format!("pipeline index is not JSON: {error}")))?;
        let index = value
            .as_object()
            .ok_or_else(|| Error::invalid("pipeline index must be a JSON object"))?;
        let class_name = index
            .get("_class_name")
            .and_then(Value::as_str)
            .ok_or(Error::MissingField {
                field: "_class_name",
            })?
            .to_owned();
        let components = index
            .iter()
            .filter(|(name, specification)| !name.starts_with('_') && specification.is_array())
            .map(|(name, specification)| {
                let folder = specification
                    .get(2)
                    .and_then(|options| options.get("subfolder"))
                    .and_then(Value::as_str)
                    .unwrap_or(name);
                (name.clone(), folder.to_owned())
            })
            .collect();
        Ok(Self {
            class_name,
            components,
        })
    }

    /// Returns the checkpoint-relative path of `filename` inside `component`'s folder.
    pub fn component_file(&self, component: &str, filename: &str) -> Result<String> {
        let folder = self.components.get(component).ok_or_else(|| {
            Error::invalid(format!(
                "pipeline {} declares no {component} component",
                self.class_name
            ))
        })?;
        Ok(format!("{folder}/{filename}"))
    }
}

/// Resolves the root index of a diffusers pipeline checkpoint.
///
/// A repository that publishes neither index file is not a pipeline
/// checkpoint, so a resolution failure yields `None` rather than an error.
pub async fn resolve_pipeline_index(model_id: &str) -> Result<Option<PipelineIndex>> {
    for filename in PIPELINE_INDEX_FILES {
        if let Ok(path) = resolve_model_file(model_id, filename).await {
            return read_pipeline_index(&path).map(Some);
        }
    }
    Ok(None)
}

/// Reads and parses one resolved index file.
fn read_pipeline_index(path: &Path) -> Result<PipelineIndex> {
    let content = std::fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    PipelineIndex::parse(&content)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_modular_component_resolves_to_its_declared_subfolder() {
        let index = PipelineIndex::parse(
            r#"{
                "_class_name": "MiniMaxH3ModularPipeline",
                "_diffusers_version": "0.36.0",
                "tokenizer": ["transformers", "Qwen2TokenizerFast",
                              {"subfolder": "text_tokenizer", "revision": null}]
            }"#,
        )
        .unwrap();

        assert_eq!(index.class_name, "MiniMaxH3ModularPipeline");
        assert_eq!(
            index.component_file("tokenizer", "tokenizer.json").unwrap(),
            "text_tokenizer/tokenizer.json"
        );
    }

    #[test]
    fn a_classic_component_resolves_to_its_key() {
        let index = PipelineIndex::parse(
            r#"{"_class_name": "FluxPipeline", "tokenizer": ["transformers", "CLIPTokenizer"]}"#,
        )
        .unwrap();

        assert_eq!(
            index.component_file("tokenizer", "tokenizer.json").unwrap(),
            "tokenizer/tokenizer.json"
        );
        assert!(index.component_file("vae", "config.json").is_err());
    }

    #[tokio::test]
    async fn a_directory_without_an_index_is_not_a_pipeline() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("config.json"), "{}").unwrap();

        let resolved = resolve_pipeline_index(directory.path().to_str().unwrap())
            .await
            .unwrap();

        assert_eq!(resolved, None);
    }
}
