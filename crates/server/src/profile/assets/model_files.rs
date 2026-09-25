//! Local and Hugging Face Hub model-file resolution.
//!
//! A model id that names an existing directory is read in place. Any other id
//! is a Hub repository: the Hub decides which files it publishes, through its
//! file listing or a 404 answer for one file, and each published file is read
//! from the local Hub cache (under `HF_HOME` when set) when cached and
//! downloaded otherwise. A failed Hub request is reported as an error, never
//! taken as evidence that a file is absent. `ResolvedModelFiles` requires only
//! `tokenizer.json`; every other file is optional and reported as `None` when
//! absent.

use std::path::{Path, PathBuf};

use hf_hub::api::tokio::{Api, ApiBuilder, ApiError, ApiRepo};
use thiserror_ext::AsReport as _;

use crate::profile::assets::error::{Error, Result};

/// Environment variable whose non-empty value overrides the token that
/// `hf_hub` reads from the cache's token file.
const HF_TOKEN_ENV: &str = "HF_TOKEN";

/// Concrete files resolved for one configured Hugging Face model.
#[derive(Debug, Clone)]
pub struct ResolvedModelFiles {
    /// Required tokenizer definition.
    pub tokenizer_path: PathBuf,
    /// Optional tokenizer metadata.
    pub tokenizer_config_path: Option<PathBuf>,
    /// Optional generation defaults.
    pub generation_config_path: Option<PathBuf>,
    /// Optional media preprocessor metadata.
    pub preprocessor_config_path: Option<PathBuf>,
    /// Optional standalone chat template.
    pub chat_template_path: Option<PathBuf>,
    /// Optional model architecture metadata: `config.json` when it declares
    /// `model_type` or `architectures`, otherwise `llm_config.json` when
    /// present, otherwise `config.json` as found.
    pub config_path: Option<PathBuf>,
}

impl ResolvedModelFiles {
    /// Resolves configured model files from a local directory, the local Hub cache, or the Hub.
    ///
    /// For a Hub repository the file listing decides which optional files
    /// exist, so the listing is fetched even when every file is cached, and a
    /// listed file the cache lacks is downloaded.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MissingFile`] when the directory or the repository
    /// listing has no `tokenizer.json`, and [`Error::Remote`] when a Hub
    /// request, including the listing, fails.
    pub async fn new(model_id: &str) -> Result<Self> {
        ModelSource::from_model_id(model_id)?.model_files().await
    }
}

/// Resolves one required file from a local model directory, the local Hub cache, or the Hub.
///
/// `filename` is checkpoint-relative and may name a subfolder. A file the
/// directory or the repository does not have is [`Error::MissingFile`]; for
/// an uncached Hub file that is the Hub's 404 answer. Every other Hub
/// failure, such as a refused, rate-limited or unreachable request, is
/// [`Error::Remote`], so callers never mistake it for an absent file.
pub async fn resolve_model_file(model_id: &str, filename: &str) -> Result<PathBuf> {
    ModelSource::from_model_id(model_id)?.file(filename).await
}

/// Where a configured model id's files are read from.
pub(crate) enum ModelSource {
    /// An existing local checkpoint directory, read in place.
    Local(PathBuf),
    /// A Hub repository reached through `api`.
    Hub {
        /// Hub client; its cache supplies cached files and receives
        /// downloads.
        api: Api,
        /// Repository identifier.
        repo_id: String,
    },
}

impl ModelSource {
    /// Classifies a configured model id.
    ///
    /// An existing directory is local; any other id names a Hub repository,
    /// reached with a client and cache configured from the environment
    /// (`HF_HOME`, `HF_ENDPOINT` and `HF_TOKEN`).
    pub(crate) fn from_model_id(model_id: &str) -> Result<Self> {
        let local = Path::new(model_id);
        if local.is_dir() {
            return Ok(Self::Local(local.to_path_buf()));
        }
        Ok(Self::Hub {
            api: build_api(model_id)?,
            repo_id: model_id.to_owned(),
        })
    }

    /// Resolves one checkpoint-relative file; see [`resolve_model_file`].
    pub(crate) async fn file(&self, filename: &str) -> Result<PathBuf> {
        match self {
            Self::Local(directory) => {
                local_file_if_exists(directory, filename).ok_or_else(|| Error::MissingFile {
                    model: directory.display().to_string(),
                    file: filename.to_owned(),
                })
            }
            Self::Hub { api, repo_id } => {
                fetch_file(&api.model(repo_id.clone()), repo_id, filename).await
            }
        }
    }

    /// Resolves the tokenizer and metadata file set; see [`ResolvedModelFiles::new`].
    pub(crate) async fn model_files(&self) -> Result<ResolvedModelFiles> {
        match self {
            Self::Local(directory) => resolve_local_model_files(directory),
            Self::Hub { api, repo_id } => resolve_remote_model_files(api, repo_id).await,
        }
    }
}

/// Resolves the model files present in a local checkpoint directory.
fn resolve_local_model_files(model_dir: &Path) -> Result<ResolvedModelFiles> {
    let tokenizer_path =
        local_file_if_exists(model_dir, "tokenizer.json").ok_or_else(|| Error::MissingFile {
            model: model_dir.display().to_string(),
            file: "tokenizer.json".to_owned(),
        })?;
    Ok(ResolvedModelFiles {
        tokenizer_path,
        tokenizer_config_path: local_file_if_exists(model_dir, "tokenizer_config.json"),
        generation_config_path: local_file_if_exists(model_dir, "generation_config.json"),
        preprocessor_config_path: local_file_if_exists(model_dir, "preprocessor_config.json"),
        chat_template_path: discover_chat_template_in_dir(model_dir),
        config_path: resolve_local_config_path(model_dir),
    })
}

/// Resolves the required and optional files a Hub repository's listing advertises.
///
/// Each listed file is taken from the cache when cached and downloaded
/// otherwise, so a cache that holds only some of the files is completed.
async fn resolve_remote_model_files(api: &Api, model_id: &str) -> Result<ResolvedModelFiles> {
    let repo = api.model(model_id.to_string());
    let info = repo.info().await.map_err(|error| Error::Remote {
        model: model_id.to_owned(),
        message: error.as_report().to_string(),
    })?;
    let siblings = info
        .siblings
        .iter()
        .map(|sibling| sibling.rfilename.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    if !siblings.contains("tokenizer.json") {
        return Err(Error::MissingFile {
            model: model_id.to_owned(),
            file: "tokenizer.json".to_owned(),
        });
    }
    let tokenizer_path = fetch_file(&repo, model_id, "tokenizer.json").await?;
    let tokenizer_config_path =
        download_if_present(&repo, model_id, &siblings, "tokenizer_config.json").await?;
    let generation_config_path =
        download_if_present(&repo, model_id, &siblings, "generation_config.json").await?;
    let preprocessor_config_path =
        download_if_present(&repo, model_id, &siblings, "preprocessor_config.json").await?;
    let chat_template_path = match select_chat_template_sibling(&siblings) {
        Some(name) => Some(fetch_file(&repo, model_id, name).await?),
        None => None,
    };
    let config_path = match download_if_present(&repo, model_id, &siblings, "config.json").await? {
        Some(path) if config_json_is_usable(&path) => Some(path),
        other => download_if_present(&repo, model_id, &siblings, "llm_config.json")
            .await?
            .or(other),
    };
    Ok(ResolvedModelFiles {
        tokenizer_path,
        tokenizer_config_path,
        generation_config_path,
        preprocessor_config_path,
        chat_template_path,
        config_path,
    })
}

/// Downloads an optional model file when the repository provides it.
async fn download_if_present(
    repo: &ApiRepo,
    model_id: &str,
    siblings: &std::collections::BTreeSet<&str>,
    filename: &str,
) -> Result<Option<PathBuf>> {
    match siblings.contains(filename) {
        true => fetch_file(repo, model_id, filename).await.map(Some),
        false => Ok(None),
    }
}

/// Returns a Hub file's cached copy, downloading it when it is not cached.
///
/// A 404 answer, the Hub's response for a file the repository does not
/// publish, is [`Error::MissingFile`]; every other failure is
/// [`Error::Remote`].
async fn fetch_file(repo: &ApiRepo, model_id: &str, filename: &str) -> Result<PathBuf> {
    repo.get(filename).await.map_err(|error| match &error {
        ApiError::RequestError(request)
            if request
                .status()
                .is_some_and(|status| status.as_u16() == 404) =>
        {
            Error::MissingFile {
                model: model_id.to_owned(),
                file: filename.to_owned(),
            }
        }
        _ => Error::Remote {
            model: model_id.to_owned(),
            message: format!("failed to download '{filename}': {}", error.as_report()),
        },
    })
}

/// Builds a Hub API client with download progress enabled, authenticated by
/// a non-empty `HF_TOKEN` and otherwise by the cache's token file, if any.
fn build_api(model_id: &str) -> Result<Api> {
    let mut builder = ApiBuilder::from_env().with_progress(true);
    if let Ok(token) = std::env::var(HF_TOKEN_ENV)
        && !token.is_empty()
    {
        builder = builder.with_token(Some(token));
    }
    builder.build().map_err(|error| Error::Remote {
        model: model_id.to_owned(),
        message: error.to_report_string(),
    })
}

/// Returns a local model file when it exists.
fn local_file_if_exists(dir: &Path, filename: &str) -> Option<PathBuf> {
    let path = dir.join(filename);
    path.is_file().then_some(path)
}

/// Returns whether a configuration file is readable JSON that declares
/// `model_type` or `architectures`.
fn config_json_is_usable(path: &Path) -> bool {
    let Ok(content) = std::fs::read_to_string(path) else {
        return false;
    };
    match serde_json::from_str::<serde_json::Value>(&content) {
        Ok(value) => value.get("model_type").is_some() || value.get("architectures").is_some(),
        Err(_) => false,
    }
}

/// Selects the architecture config in a local directory with the same
/// precedence as [`ResolvedModelFiles::config_path`] documents.
fn resolve_local_config_path(dir: &Path) -> Option<PathBuf> {
    if let Some(path) = local_file_if_exists(dir, "config.json")
        && config_json_is_usable(&path)
    {
        return Some(path);
    }
    local_file_if_exists(dir, "llm_config.json")
        .or_else(|| local_file_if_exists(dir, "config.json"))
}

/// Selects the standalone chat template from a repository listing.
///
/// `chat_template.json` wins over `chat_template.jinja`; otherwise the first
/// top-level `.jinja` file in lexicographic order is used. Templates in
/// subfolders are never selected.
fn select_chat_template_sibling<'a>(
    siblings: &std::collections::BTreeSet<&'a str>,
) -> Option<&'a str> {
    if siblings.contains("chat_template.json") {
        return Some("chat_template.json");
    }
    if siblings.contains("chat_template.jinja") {
        return Some("chat_template.jinja");
    }
    siblings
        .iter()
        .copied()
        .find(|name| name.ends_with(".jinja") && !name.contains('/'))
}

/// Discovers a standalone chat template in a local checkpoint directory,
/// with the same named-file precedence as [`select_chat_template_sibling`].
fn discover_chat_template_in_dir(dir: &Path) -> Option<PathBuf> {
    for filename in ["chat_template.json", "chat_template.jinja"] {
        let path = dir.join(filename);
        if path.is_file() {
            return Some(path);
        }
    }
    // Directory iteration order is platform-defined, so with several
    // top-level `.jinja` files the choice among them is unspecified.
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| {
            path.is_file()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.ends_with(".jinja"))
        })
}

#[cfg(test)]
mod tests {
    use std::fs;

    use axum::http::StatusCode;
    use tempfile::tempdir;

    use super::{ResolvedModelFiles, select_chat_template_sibling};
    use crate::profile::assets::Error;
    use crate::profile::assets::hub_stub::{HubStub, seed_cache};

    const REPO: &str = "org/model";

    /// Files a Hub text-model repository publishes.
    const PUBLISHED: [(&str, &str); 4] = [
        ("tokenizer.json", "{}"),
        ("tokenizer_config.json", r#"{"eos_token":"<|im_end|>"}"#),
        ("generation_config.json", r#"{"temperature":0.6}"#),
        ("config.json", r#"{"model_type":"qwen3"}"#),
    ];

    fn siblings(names: &[&'static str]) -> std::collections::BTreeSet<&'static str> {
        names.iter().copied().collect()
    }

    #[test]
    fn chat_template_selection_is_deterministic() {
        assert_eq!(
            select_chat_template_sibling(&siblings(&[
                "chat_template.json",
                "chat_template.jinja",
                "tokenizer.json",
            ])),
            Some("chat_template.json")
        );
        assert_eq!(
            select_chat_template_sibling(&siblings(&["chat_template.jinja", "tokenizer.json"])),
            Some("chat_template.jinja")
        );
        assert_eq!(
            select_chat_template_sibling(&siblings(&[
                "additional_chat_templates/tool_use.jinja",
                "tokenizer.json",
            ])),
            None
        );
    }

    #[tokio::test]
    async fn local_model_resolves_hugging_face_tokenizer() {
        let dir = tempdir().expect("create temp dir");
        fs::write(dir.path().join("tokenizer.json"), "{}").expect("write tokenizer");
        fs::write(dir.path().join("config.json"), r#"{"model_type":"qwen3"}"#)
            .expect("write config");

        let files = ResolvedModelFiles::new(dir.path().to_str().expect("utf8 path"))
            .await
            .expect("resolve local model files");

        assert_eq!(files.tokenizer_path, dir.path().join("tokenizer.json"));
        assert_eq!(files.config_path, Some(dir.path().join("config.json")));
    }

    /// A Hub cache that holds `tokenizer.json` but not the other metadata
    /// files resolves the repository's full file set: the listed files the
    /// cache lacks are downloaded rather than reported absent.
    #[tokio::test]
    async fn a_partial_hub_cache_resolves_every_published_file() {
        let root = tempdir().unwrap();
        seed_cache(root.path(), REPO, &PUBLISHED[..1]);
        let source = HubStub::with_files(&PUBLISHED)
            .serve(root.path(), REPO)
            .await;

        let files = source.model_files().await.unwrap();

        let read = |path: Option<std::path::PathBuf>| fs::read_to_string(path.unwrap()).unwrap();
        assert_eq!(read(files.tokenizer_config_path), PUBLISHED[1].1);
        assert_eq!(read(files.generation_config_path), PUBLISHED[2].1);
        assert_eq!(read(files.config_path), PUBLISHED[3].1);
        assert_eq!(files.preprocessor_config_path, None);
        assert_eq!(files.chat_template_path, None);
    }

    /// A repository listing the Hub refuses is reported as a remote failure
    /// even when the cache holds `tokenizer.json`.
    #[tokio::test]
    async fn a_refused_repository_listing_is_a_remote_error() {
        let root = tempdir().unwrap();
        seed_cache(root.path(), REPO, &PUBLISHED[..1]);
        let source = HubStub {
            listing_status: Some(StatusCode::UNAUTHORIZED),
            ..HubStub::with_files(&PUBLISHED)
        }
        .serve(root.path(), REPO)
        .await;

        let error = source.model_files().await.unwrap_err();

        assert!(matches!(error, Error::Remote { .. }), "{error:?}");
    }
}
