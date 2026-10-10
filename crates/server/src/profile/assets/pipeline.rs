//! Component files of a diffusers pipeline checkpoint.
//!
//! A pipeline checkpoint holds its components in the folders its root index
//! names ([`PipelineIndex`]). A FastH3 export may omit some of them; its
//! `fastvideo_inference.json` then pins, in `base_model_revision`
//! (`hf://<repository>@<revision>`), the base revision whose folders supply
//! them. [`PipelineCheckpoint`] reads each component file from the
//! checkpoint, or from the Hub snapshot of that revision when the checkpoint
//! lacks it, as the worker's loader (`uniserve_models.loading`) reads them.

use std::fmt;
use std::path::PathBuf;

use hf_hub::{Repo, RepoType};
use serde_json::Value;

use crate::profile::assets::error::{Error, Result};
use crate::profile::assets::model_files::{HubClient, ModelSource, fetch_file};
use crate::profile::assets::pipeline_index::{PipelineIndex, pipeline_index};

/// Inference contract a FastVideo export publishes at its root.
const INFERENCE_CONTRACT: &str = "fastvideo_inference.json";

/// A Hugging Face repository pinned at one commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseRevision {
    /// Repository id, `<owner>/<name>`.
    pub repository: String,
    /// Commit of the repository the export draws from.
    pub revision: String,
}

impl BaseRevision {
    /// Parses `hf://<repository>@<revision>`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] when `value` is not of that form with an
    /// `<owner>/<name>` repository and a non-empty revision.
    pub fn parse(value: &str) -> Result<Self> {
        let malformed = || {
            Error::invalid(format!(
                "a FastH3 export's base_model_revision must be \
                 hf://<repository>@<revision>, got {value:?}"
            ))
        };
        let (repository, revision) = value
            .strip_prefix("hf://")
            .and_then(|pinned| pinned.rsplit_once('@'))
            .ok_or_else(malformed)?;
        let well_formed = repository.split_once('/').is_some_and(|(owner, name)| {
            !owner.is_empty() && !name.is_empty() && !name.contains('/')
        });
        if !well_formed || revision.is_empty() {
            return Err(malformed());
        }
        Ok(Self {
            repository: repository.to_owned(),
            revision: revision.to_owned(),
        })
    }
}

impl fmt::Display for BaseRevision {
    /// Formats the pin as `<repository>@<revision>`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}@{}", self.repository, self.revision)
    }
}

/// Returns `filename` from the Hub cache's snapshot of `pin`, downloading it
/// at that commit when the snapshot lacks it.
///
/// `huggingface_hub` fills a commit's snapshot without recording a ref for
/// the commit, which `hf_hub`'s own cache lookup requires, so a file the
/// snapshot holds is read from it directly.
///
/// # Errors
///
/// Returns [`Error::MissingFile`] when the Hub does not publish `filename` at
/// `pin`, and [`Error::Remote`] when a Hub request fails.
async fn pinned_file(client: &HubClient, pin: &BaseRevision, filename: &str) -> Result<PathBuf> {
    let repo = Repo::with_revision(
        pin.repository.clone(),
        RepoType::Model,
        pin.revision.clone(),
    );
    let cached = client
        .cache
        .repo(repo.clone())
        .pointer_path(&pin.revision)
        .join(filename);
    if cached.is_file() {
        return Ok(cached);
    }
    fetch_file(&client.api.repo(repo), &pin.to_string(), filename).await
}

/// Reads the base revision a FastH3 export pins, or `None` for a checkpoint
/// that publishes no inference contract or pins no base.
async fn pinned_base(root: &ModelSource) -> Result<Option<BaseRevision>> {
    let path = match root.file(INFERENCE_CONTRACT).await {
        Ok(path) => path,
        Err(Error::MissingFile { .. }) => return Ok(None),
        Err(error) => return Err(error),
    };
    let content = std::fs::read_to_string(&path).map_err(|source| Error::Io {
        path: path.clone(),
        source,
    })?;
    let contract: Value =
        serde_json::from_str(&content).map_err(|source| Error::Json { path, source })?;
    match contract.get("base_model_revision") {
        None => Ok(None),
        Some(pin) => pin
            .as_str()
            .ok_or_else(|| {
                Error::invalid(format!(
                    "{} states a base_model_revision that is not a string",
                    root.id()
                ))
            })
            .and_then(BaseRevision::parse)
            .map(Some),
    }
}

/// A diffusers pipeline checkpoint with the source of each component's files.
pub struct PipelineCheckpoint {
    /// The checkpoint's root index.
    index: PipelineIndex,
    /// The checkpoint itself.
    root: ModelSource,
    /// The base revision that supplies the folders a FastH3 export omits,
    /// with the client its snapshot is read through; `None` for a
    /// checkpoint that pins none.
    base: Option<(HubClient, BaseRevision)>,
}

impl PipelineCheckpoint {
    /// Resolves the pipeline checkpoint `model_id` names.
    ///
    /// `model_id` is a local directory or a Hub repository, as for
    /// [`resolve_model_file`](super::resolve_model_file). Returns `None` for
    /// a checkpoint that publishes no pipeline index.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] when the export's pin is malformed, and the
    /// errors of [`resolve_pipeline_index`](super::resolve_pipeline_index)
    /// for the root index and the inference contract.
    pub async fn resolve(model_id: &str) -> Result<Option<Self>> {
        let root = ModelSource::from_model_id(model_id)?;
        Self::locate(root, || HubClient::from_env(model_id)).await
    }

    /// Resolves the checkpoint `root` holds; `hub` builds the client its
    /// base revision is read through.
    async fn locate(
        root: ModelSource,
        hub: impl FnOnce() -> Result<HubClient>,
    ) -> Result<Option<Self>> {
        let Some(index) = pipeline_index(&root).await? else {
            return Ok(None);
        };
        let base = match pinned_base(&root).await? {
            Some(pin) => Some((hub()?, pin)),
            None => None,
        };
        Ok(Some(Self { index, root, base }))
    }

    /// The pipeline class the root index declares.
    pub fn class_name(&self) -> &str {
        &self.index.class_name
    }

    /// Returns `filename` inside `component`'s folder, read from the
    /// checkpoint, or from its base revision when the checkpoint lacks it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] for a component the index does not declare,
    /// and the errors of [`resolve_model_file`](super::resolve_model_file)
    /// for the file, including [`Error::MissingFile`] when neither source
    /// holds it.
    pub async fn component_file(&self, component: &str, filename: &str) -> Result<PathBuf> {
        let path = self.index.component_file(component, filename)?;
        match (self.root.file(&path).await, &self.base) {
            (Err(Error::MissingFile { .. }), Some((client, pin))) => {
                pinned_file(client, pin, &path).await
            }
            (file, _) => file,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use tempfile::{TempDir, tempdir};

    use super::*;
    use crate::profile::assets::hub_stub::{HubStub, seed_snapshot};

    const BASE: &str = "org/base";
    const PIN: &str = "9bfb6693f2cf6de171db46d1aa586f67d773a1da";

    /// Index of a FastH3 export: every MiniMax-H3 component, of which the
    /// export holds only `transformer_ref`.
    const INDEX: &str = r#"{
        "_class_name": "MiniMaxH3ModularPipeline",
        "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration", {"subfolder": "text_encoder"}],
        "tokenizer": ["transformers", "Qwen2TokenizerFast", {"subfolder": "tokenizer"}],
        "processor": ["transformers", "Qwen3VLProcessor", {"subfolder": "processor"}],
        "vae": ["diffusers", "AutoencoderKLMiniMaxH3", {"subfolder": "vae"}],
        "audio_vae": ["diffusers", "AutoencoderKLMiniMaxH3Audio", {"subfolder": "audio_vae"}],
        "transformer_ref": ["diffusers", "MiniMaxH3Transformer3DModel", {"subfolder": "transformer_ref"}]
    }"#;

    fn write(path: &Path, content: &str) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, content).unwrap();
    }

    /// A FastH3 OmniRef-style export pinning `BASE` at `PIN`.
    fn export() -> TempDir {
        let root = tempdir().unwrap();
        write(&root.path().join("modular_model_index.json"), INDEX);
        write(
            &root.path().join(INFERENCE_CONTRACT),
            &format!(r#"{{"model_type": "ref2va", "base_model_revision": "hf://{BASE}@{PIN}"}}"#),
        );
        write(
            &root.path().join("transformer_ref/config.json"),
            "export DiT",
        );
        root
    }

    fn read(path: PathBuf) -> String {
        fs::read_to_string(path).unwrap()
    }

    /// The folders the export omits come from the Hub cache's snapshot of
    /// the pinned commit: a cached file is read without a Hub request (the
    /// stub does not publish it), and an uncached one is downloaded into
    /// that snapshot. The export's own DiT is read from the export.
    #[tokio::test]
    async fn an_export_reads_the_folders_it_omits_at_the_pinned_revision() {
        let (export, cache) = (export(), tempdir().unwrap());
        let repository = seed_snapshot(
            cache.path(),
            BASE,
            PIN,
            &[("tokenizer/tokenizer.json", "cached")],
        );
        let client = HubStub {
            commit: Some(PIN),
            ..HubStub::with_files(&[("processor/config.json", "downloaded")])
        }
        .client(cache.path())
        .await;

        let checkpoint =
            PipelineCheckpoint::locate(ModelSource::Local(export.path().to_path_buf()), || {
                Ok(client)
            })
            .await
            .unwrap()
            .unwrap();

        assert_eq!(checkpoint.class_name(), "MiniMaxH3ModularPipeline");
        let snapshot = repository.join("snapshots").join(PIN);
        let tokenizer = checkpoint
            .component_file("tokenizer", "tokenizer.json")
            .await
            .unwrap();
        assert_eq!(tokenizer, snapshot.join("tokenizer/tokenizer.json"));
        assert_eq!(read(tokenizer), "cached");
        let processor = checkpoint
            .component_file("processor", "config.json")
            .await
            .unwrap();
        assert!(processor.starts_with(&snapshot), "{}", processor.display());
        assert_eq!(read(processor), "downloaded");
        assert_eq!(
            read(
                checkpoint
                    .component_file("transformer_ref", "config.json")
                    .await
                    .unwrap()
            ),
            "export DiT"
        );
    }

    #[test]
    fn a_base_pin_names_one_repository_at_one_revision() {
        assert_eq!(
            BaseRevision::parse(&format!("hf://{BASE}@{PIN}")).unwrap(),
            BaseRevision {
                repository: BASE.to_owned(),
                revision: PIN.to_owned()
            }
        );
        for malformed in [
            "org/base@main",
            "hf://org/base",
            "hf://base@main",
            "hf://org/base/extra@main",
            "hf://org/base@",
            "hf:///base@main",
        ] {
            assert!(BaseRevision::parse(malformed).is_err(), "{malformed}");
        }
    }
}
