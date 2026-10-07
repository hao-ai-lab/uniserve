//! Root component index of a diffusers pipeline checkpoint.
//!
//! A pipeline checkpoint names its pipeline class and component folders in a
//! root index instead of a root `config.json`. The classic layout publishes
//! `model_index.json`, where each component lives in the folder named by its
//! key; the modular layout publishes `modular_model_index.json`, where a
//! component's specification may name its folder explicitly.
//!
//! Server startup (`ModelConfig::load` in `serving::model`) consults this
//! index, through `PipelineCheckpoint`, before any other asset: its presence
//! marks a checkpoint as a pipeline, and `_class_name` selects the family.
//! The Dynamo worker binary uses the same index to accept only MiniMax H3
//! checkpoints.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::Value;

use crate::profile::assets::error::{Error, Result};
use crate::profile::assets::model_files::ModelSource;

/// Index files consulted in order: the classic pipeline index, then the modular one.
///
/// The first file that resolves wins, even when both are published.
const PIPELINE_INDEX_FILES: [&str; 2] = ["model_index.json", "modular_model_index.json"];

/// Pipeline class and component folders declared by a checkpoint's root index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineIndex {
    /// Pipeline class the index declares in `_class_name`.
    pub class_name: String,
    /// Component name mapped to the checkpoint-relative folder holding its
    /// files.
    components: BTreeMap<String, String>,
}

impl PipelineIndex {
    /// Parses a root index.
    ///
    /// Keys starting with `_` are index metadata. Every other key whose value
    /// is an array (a component specification `[library, class, options?]`)
    /// names a component; its folder is the string `subfolder` option when
    /// present and the key otherwise. Keys with non-array values are ignored.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] when `content` is not JSON or not a JSON
    /// object, and [`Error::MissingField`] when `_class_name` is absent or not
    /// a string.
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
    ///
    /// The result is suitable for `resolve_model_file`. An undeclared
    /// component is [`Error::Invalid`].
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
/// A checkpoint that publishes neither index file is not a pipeline
/// checkpoint: [`Error::MissingFile`] for both files yields `None`. Any other
/// resolution failure, such as a refused, rate-limited or unreachable Hub
/// request, is returned, since it says nothing about whether the index
/// exists. An index that resolves but cannot be read or parsed is an error.
pub async fn resolve_pipeline_index(model_id: &str) -> Result<Option<PipelineIndex>> {
    pipeline_index(&ModelSource::from_model_id(model_id)?).await
}

/// Resolves the root index from an already classified model source.
pub(super) async fn pipeline_index(source: &ModelSource) -> Result<Option<PipelineIndex>> {
    for filename in PIPELINE_INDEX_FILES {
        match source.file(filename).await {
            Ok(path) => return read_pipeline_index(&path).map(Some),
            Err(Error::MissingFile { .. }) => continue,
            Err(error) => return Err(error),
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
    use axum::http::StatusCode;

    use super::*;
    use crate::profile::assets::hub_stub::HubStub;

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

    const REPO: &str = "org/pipeline";
    const MODULAR_INDEX: &str = r#"{"_class_name": "MiniMaxH3ModularPipeline"}"#;

    /// A Hub repository that publishes an index is a pipeline, and one that
    /// publishes neither index file is not.
    #[tokio::test]
    async fn a_hub_repository_is_a_pipeline_when_it_publishes_an_index() {
        let root = tempfile::tempdir().unwrap();
        let pipeline = HubStub::with_files(&[("modular_model_index.json", MODULAR_INDEX)])
            .serve(root.path(), REPO)
            .await;
        let plain = HubStub::with_files(&[("config.json", r#"{"model_type":"qwen3"}"#)])
            .serve(root.path(), "org/plain")
            .await;

        let index = pipeline_index(&pipeline).await.unwrap().unwrap();

        assert_eq!(index.class_name, "MiniMaxH3ModularPipeline");
        assert_eq!(pipeline_index(&plain).await.unwrap(), None);
    }

    /// A download the Hub refuses is reported as a remote failure rather than
    /// taken as a checkpoint without an index.
    #[tokio::test]
    async fn a_refused_index_download_is_a_remote_error() {
        let root = tempfile::tempdir().unwrap();
        let source = HubStub {
            download_status: Some(StatusCode::UNAUTHORIZED),
            ..HubStub::with_files(&[("modular_model_index.json", MODULAR_INDEX)])
        }
        .serve(root.path(), REPO)
        .await;

        let error = pipeline_index(&source).await.unwrap_err();

        assert!(matches!(error, Error::Remote { .. }), "{error:?}");
    }
}
