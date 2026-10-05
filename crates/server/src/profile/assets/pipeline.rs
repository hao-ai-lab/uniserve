//! Component files of a diffusers pipeline checkpoint.
//!
//! A pipeline checkpoint holds its components in the folders its root index
//! names ([`PipelineIndex`]). A FastVideo component export (FastH3 OmniRef)
//! holds only one DiT partition and its schedulers: its
//! `fastvideo_inference.json` marks it with `pdd_steps` and pins, in
//! `base_model_revision` (`hf://<repository>@<revision>`), the diffusers root
//! whose folders supply its other components, [`BASE_COMPONENTS`].
//! [`PipelineCheckpoint`] reads each component file from the checkpoint or
//! from that base, which it resolves by the rules the worker's loader
//! (`uniserve_models.loading`) applies:
//!
//! * a configured local directory supplies the base components in place;
//! * otherwise the base is the Hub cache's snapshot of the pinned commit,
//!   and a file the snapshot lacks is downloaded into it at that commit.

use std::fmt;
use std::path::{Path, PathBuf};

use hf_hub::{Repo, RepoType};
use serde_json::Value;

use crate::profile::assets::error::{Error, Result};
use crate::profile::assets::model_files::{HubClient, ModelSource, fetch_file, local_file};
use crate::profile::assets::pipeline_index::{PipelineIndex, pipeline_index};

/// Checkpoint folders a component export draws from its pinned base: every
/// component but its own DiT partition and schedulers.
pub const BASE_COMPONENTS: [&str; 5] =
    ["audio_vae", "processor", "text_encoder", "tokenizer", "vae"];

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
                "a component export's base_model_revision must be \
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

/// Where a component export's base components are read.
enum BaseCheckpoint {
    /// A caller-supplied directory holding the base components.
    Local(PathBuf),
    /// The Hub cache's snapshot of the pinned revision.
    Hub {
        /// Client and cache the snapshot is read and filled through.
        client: HubClient,
        /// The pinned revision.
        pin: BaseRevision,
    },
}

impl BaseCheckpoint {
    /// Resolves a local directory supplying the base components.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] when `root` is not a directory.
    fn local(root: &Path) -> Result<Self> {
        if !root.is_dir() {
            return Err(Error::invalid(format!(
                "base checkpoint {} is not a directory",
                root.display()
            )));
        }
        Ok(Self::Local(root.to_path_buf()))
    }

    /// Returns one checkpoint-relative file of the base.
    async fn file(&self, filename: &str) -> Result<PathBuf> {
        match self {
            Self::Local(root) => local_file(root, filename),
            Self::Hub { client, pin } => pinned_file(client, pin, filename).await,
        }
    }
}

/// Returns `filename` from the Hub cache's snapshot of `pin`, downloading it
/// at that commit when the snapshot lacks it.
///
/// A commit's snapshot is immutable, so a file it holds is read without a
/// Hub request; `huggingface_hub` fills such snapshots without recording a
/// ref for the commit, which `hf_hub`'s own cache lookup requires.
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
    let snapshot = client.cache.repo(repo.clone()).pointer_path(&pin.revision);
    let cached = snapshot.join(filename);
    if cached.is_file() {
        return Ok(cached);
    }

    fetch_file(&client.api.repo(repo), &pin.to_string(), filename).await
}

/// Reads the base a component export pins.
///
/// Returns `None` for a checkpoint that publishes no inference contract or
/// whose contract is not a component export's (it declares no `pdd_steps`).
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
    if contract.get("pdd_steps").is_none() {
        return Ok(None);
    }
    let pin = contract
        .get("base_model_revision")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            Error::invalid(format!(
                "component export {} declares no base_model_revision in {INFERENCE_CONTRACT}",
                root.id()
            ))
        })?;
    BaseRevision::parse(pin).map(Some)
}

/// A diffusers pipeline checkpoint with the source of each component's files.
pub struct PipelineCheckpoint {
    /// The checkpoint's root index.
    index: PipelineIndex,
    /// The checkpoint itself.
    root: ModelSource,
    /// The base a component export draws [`BASE_COMPONENTS`] from; `None`
    /// for a checkpoint that holds every component.
    base: Option<BaseCheckpoint>,
}

impl PipelineCheckpoint {
    /// Resolves the pipeline checkpoint `model_id` names.
    ///
    /// `model_id` is a local directory or a Hub repository, as for
    /// [`resolve_model_file`](super::resolve_model_file). `base_model` names
    /// a local directory supplying the base components; without it the base
    /// comes from the declared Hub revision. Returns `None` for a checkpoint
    /// that publishes no pipeline index.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] when `base_model` is given for a checkpoint
    /// that pins no base, when the export's pin is malformed, or when the
    /// local base is not a directory; and the errors of
    /// [`resolve_pipeline_index`](super::resolve_pipeline_index) for the root
    /// index and the inference contract.
    pub async fn resolve(model_id: &str, base_model: Option<&Path>) -> Result<Option<Self>> {
        let root = ModelSource::from_model_id(model_id)?;
        Self::locate(root, base_model, || HubClient::from_env(model_id)).await
    }

    /// Resolves the checkpoint `root` holds; `hub` builds the client a base
    /// without a local copy is read through.
    async fn locate(
        root: ModelSource,
        base_model: Option<&Path>,
        hub: impl FnOnce() -> Result<HubClient>,
    ) -> Result<Option<Self>> {
        let takes_no_base = |root: &ModelSource, base: &Path| {
            Error::invalid(format!(
                "checkpoint {} pins no base checkpoint, so it takes no base ({})",
                root.id(),
                base.display()
            ))
        };
        let Some(index) = pipeline_index(&root).await? else {
            return match base_model {
                Some(base) => Err(takes_no_base(&root, base)),
                None => Ok(None),
            };
        };
        let base = match (pinned_base(&root).await?, base_model) {
            (None, None) => None,
            (None, Some(base)) => return Err(takes_no_base(&root, base)),
            (Some(_), Some(base)) => Some(BaseCheckpoint::local(base)?),
            (Some(pin), None) => Some(BaseCheckpoint::Hub {
                client: hub()?,
                pin,
            }),
        };
        Ok(Some(Self { index, root, base }))
    }

    /// The pipeline class the root index declares.
    pub fn class_name(&self) -> &str {
        &self.index.class_name
    }

    /// Returns `filename` inside `component`'s folder, read from the base
    /// when the checkpoint is a component export and the folder is one of
    /// [`BASE_COMPONENTS`], and from the checkpoint otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] for a component the index does not declare,
    /// and the errors of [`resolve_model_file`](super::resolve_model_file)
    /// for the file, including [`Error::MissingFile`] when its source does
    /// not hold it.
    pub async fn component_file(&self, component: &str, filename: &str) -> Result<PathBuf> {
        let path = self.index.component_file(component, filename)?;
        let folder = path.split('/').next().unwrap_or_default();
        match &self.base {
            Some(base) if BASE_COMPONENTS.contains(&folder) => base.file(&path).await,
            _ => self.root.file(&path).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::{TempDir, tempdir};

    use super::*;
    use crate::profile::assets::hub_stub::{HubStub, seed_snapshot};

    const BASE: &str = "org/base";
    const PIN: &str = "9bfb6693f2cf6de171db46d1aa586f67d773a1da";

    /// Index of a component export: every MiniMax-H3 component, of which the
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

    /// A FastH3 OmniRef-style component export pinning `BASE` at `PIN`.
    fn export() -> TempDir {
        let root = tempdir().unwrap();
        write(&root.path().join("modular_model_index.json"), INDEX);
        write(
            &root.path().join(INFERENCE_CONTRACT),
            &format!(r#"{{"pdd_steps": 32, "base_model_revision": "hf://{BASE}@{PIN}"}}"#),
        );
        write(
            &root.path().join("transformer_ref/config.json"),
            "export DiT",
        );
        root
    }

    /// Local component files, independent of how the directory was populated.
    fn base_copy() -> TempDir {
        let root = tempdir().unwrap();
        for component in BASE_COMPONENTS {
            let name = format!("{component}/config.json");
            write(&root.path().join(&name), &format!("base {component}"));
        }
        write(
            &root.path().join("tokenizer/tokenizer.json"),
            "base tokenizer",
        );
        root
    }

    fn read(path: PathBuf) -> String {
        fs::read_to_string(path).unwrap()
    }

    /// Read local base components and the export's own DiT from their
    /// respective directories.
    #[tokio::test]
    async fn a_component_export_reads_its_base_components_from_a_local_copy() {
        let (export, base) = (export(), base_copy());

        let checkpoint =
            PipelineCheckpoint::resolve(export.path().to_str().unwrap(), Some(base.path()))
                .await
                .unwrap()
                .unwrap();

        assert_eq!(checkpoint.class_name(), "MiniMaxH3ModularPipeline");
        assert_eq!(
            checkpoint
                .component_file("tokenizer", "tokenizer.json")
                .await
                .unwrap(),
            base.path().join("tokenizer/tokenizer.json")
        );
        assert_eq!(
            read(
                checkpoint
                    .component_file("processor", "config.json")
                    .await
                    .unwrap()
            ),
            "base processor"
        );
        assert_eq!(
            read(
                checkpoint
                    .component_file("transformer_ref", "config.json")
                    .await
                    .unwrap()
            ),
            "export DiT"
        );
        assert!(matches!(
            checkpoint.component_file("processor", "absent.json").await,
            Err(Error::MissingFile { .. })
        ));
    }

    /// Without a local copy, the base components come from the Hub cache's
    /// snapshot of the pinned commit: a cached file is read without a Hub
    /// request (the stub does not publish it), and an uncached one is
    /// downloaded into that snapshot.
    #[tokio::test]
    async fn a_component_export_reads_its_base_from_the_hub_cache_at_the_pinned_revision() {
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

        let checkpoint = PipelineCheckpoint::locate(
            ModelSource::Local(export.path().to_path_buf()),
            None,
            || Ok(client),
        )
        .await
        .unwrap()
        .unwrap();

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

    /// A base is refused for a checkpoint that pins none: a pipeline that
    /// holds every component, and a checkpoint that is no pipeline.
    #[tokio::test]
    async fn a_base_for_a_checkpoint_that_pins_none_is_refused() {
        let base = base_copy();
        let pipeline = tempdir().unwrap();
        write(&pipeline.path().join("model_index.json"), INDEX);
        let text = tempdir().unwrap();
        write(
            &text.path().join("config.json"),
            r#"{"model_type":"qwen3"}"#,
        );

        for checkpoint in [&pipeline, &text] {
            let resolved =
                PipelineCheckpoint::resolve(checkpoint.path().to_str().unwrap(), Some(base.path()))
                    .await;
            assert!(
                matches!(&resolved, Err(Error::Invalid(message)) if message.contains("pins no base"))
            );
        }
        assert!(
            PipelineCheckpoint::resolve(pipeline.path().to_str().unwrap(), None)
                .await
                .unwrap()
                .is_some()
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
