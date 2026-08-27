use std::path::{Path, PathBuf};

use hf_hub::Cache;
use hf_hub::api::tokio::{Api, ApiBuilder, ApiRepo};
use thiserror_ext::AsReport as _;

use crate::profile::assets::error::{Error, Result};

const HF_TOKEN_ENV: &str = "HF_TOKEN";

/// Concrete files resolved for one configured Hugging Face model.
#[derive(Debug, Clone)]
pub struct ResolvedModelFiles {
    pub tokenizer_path: PathBuf,
    pub tokenizer_config_path: Option<PathBuf>,
    pub generation_config_path: Option<PathBuf>,
    pub preprocessor_config_path: Option<PathBuf>,
    pub chat_template_path: Option<PathBuf>,
    pub config_path: Option<PathBuf>,
}

impl ResolvedModelFiles {
    /// Resolve configured model files from a local directory, the local Hub cache, or the Hub.
    pub async fn new(model_id: &str) -> Result<Self> {
        if Path::new(model_id).is_dir() {
            return resolve_local_model_files(Path::new(model_id));
        }
        if let Some(files) = resolve_cached_model_files(model_id)? {
            return Ok(files);
        }
        resolve_remote_model_files(model_id).await
    }
}

fn resolve_local_model_files(model_dir: &Path) -> Result<ResolvedModelFiles> {
    let tokenizer_path =
        local_file_if_exists(model_dir, "tokenizer.json").ok_or_else(|| Error::MissingFile {
            model: model_dir.display().to_string(),
            file: "tokenizer.json",
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

async fn resolve_remote_model_files(model_id: &str) -> Result<ResolvedModelFiles> {
    let api = build_api(model_id)?;
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
            file: "tokenizer.json",
        });
    }
    let tokenizer_path = download_known_file(&repo, model_id, "tokenizer.json").await?;
    let tokenizer_config_path =
        download_if_present(&repo, model_id, &siblings, "tokenizer_config.json").await?;
    let generation_config_path =
        download_if_present(&repo, model_id, &siblings, "generation_config.json").await?;
    let preprocessor_config_path =
        download_if_present(&repo, model_id, &siblings, "preprocessor_config.json").await?;
    let chat_template_path = match select_chat_template_sibling(&siblings) {
        Some(name) => Some(download_known_file(&repo, model_id, name).await?),
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

fn resolve_cached_model_files(model_id: &str) -> Result<Option<ResolvedModelFiles>> {
    let cache_repo = Cache::from_env().model(model_id.to_string());
    let Some(tokenizer_path) = cache_repo.get("tokenizer.json") else {
        return Ok(None);
    };
    let model_dir = tokenizer_path
        .parent()
        .ok_or_else(|| {
            Error::invalid("resolved tokenizer file has no parent directory".to_string())
        })?
        .to_path_buf();
    let config_path = match cache_repo.get("config.json") {
        Some(path) if config_json_is_usable(&path) => Some(path),
        other => cache_repo.get("llm_config.json").or(other),
    };
    Ok(Some(ResolvedModelFiles {
        tokenizer_path,
        tokenizer_config_path: cache_repo.get("tokenizer_config.json"),
        generation_config_path: cache_repo.get("generation_config.json"),
        preprocessor_config_path: cache_repo.get("preprocessor_config.json"),
        chat_template_path: discover_chat_template_in_dir(&model_dir),
        config_path,
    }))
}

async fn download_if_present(
    repo: &ApiRepo,
    model_id: &str,
    siblings: &std::collections::BTreeSet<&str>,
    filename: &str,
) -> Result<Option<PathBuf>> {
    match siblings.contains(filename) {
        true => download_known_file(repo, model_id, filename)
            .await
            .map(Some),
        false => Ok(None),
    }
}

async fn download_known_file(repo: &ApiRepo, model_id: &str, filename: &str) -> Result<PathBuf> {
    repo.get(filename).await.map_err(|error| Error::Remote {
        model: model_id.to_owned(),
        message: format!("failed to download '{filename}': {}", error.as_report()),
    })
}

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

fn local_file_if_exists(dir: &Path, filename: &str) -> Option<PathBuf> {
    let path = dir.join(filename);
    path.is_file().then_some(path)
}

fn config_json_is_usable(path: &Path) -> bool {
    let Ok(content) = std::fs::read_to_string(path) else {
        return false;
    };
    match serde_json::from_str::<serde_json::Value>(&content) {
        Ok(value) => value.get("model_type").is_some() || value.get("architectures").is_some(),
        Err(_) => false,
    }
}

fn resolve_local_config_path(dir: &Path) -> Option<PathBuf> {
    if let Some(path) = local_file_if_exists(dir, "config.json")
        && config_json_is_usable(&path)
    {
        return Some(path);
    }
    local_file_if_exists(dir, "llm_config.json")
        .or_else(|| local_file_if_exists(dir, "config.json"))
}

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

fn discover_chat_template_in_dir(dir: &Path) -> Option<PathBuf> {
    for filename in ["chat_template.json", "chat_template.jinja"] {
        let path = dir.join(filename);
        if path.is_file() {
            return Some(path);
        }
    }
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

    use tempfile::tempdir;

    use super::{ResolvedModelFiles, select_chat_template_sibling};

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
}
