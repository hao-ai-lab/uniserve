use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use axum::extract::State;
use serde::Deserialize;
use thiserror_ext::AsReport;
use uniserve_openai_types::Normalizable;
use validator::Validate;

use crate::error::ApiError;
use crate::routes::openai::utils::validated_json::ValidatedJson;
use uniserve_server_app::AppState;
use uniserve_server_app::{LoadLoraError, UnloadLoraError};

const RUNTIME_LORA_ALLOWED_PATH_PREFIXES_ENV: &str = "UNISERVE_RUNTIME_LORA_ALLOWED_PATH_PREFIXES";

#[derive(Debug, Deserialize, Validate)]
pub(crate) struct LoadLoraAdapterRequest {
    lora_name: String,
    lora_path: String,
    #[serde(default)]
    load_inplace: bool,
    #[serde(default)]
    is_3d_lora_weight: bool,
}

impl Normalizable for LoadLoraAdapterRequest {}

#[derive(Debug, Deserialize, Validate)]
pub(crate) struct UnloadLoraAdapterRequest {
    lora_name: String,
    #[serde(default)]
    lora_int_id: Option<u64>,
}

impl Normalizable for UnloadLoraAdapterRequest {}

fn runtime_lora_allowed_path_prefixes() -> Option<Vec<PathBuf>> {
    uniserve_config::env_paths(RUNTIME_LORA_ALLOWED_PATH_PREFIXES_ENV)
}

fn looks_like_local_lora_path(lora_path: &str) -> bool {
    let path = Path::new(lora_path);
    path.is_absolute()
        || lora_path.starts_with('~')
        || lora_path.starts_with('.')
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir))
}

fn validate_lora_path_access(
    lora_path: &str,
    allowed_prefixes: Option<&[PathBuf]>,
) -> Result<Option<String>, ApiError> {
    let path = Path::new(lora_path);
    if !looks_like_local_lora_path(lora_path) && !path.exists() {
        return Ok(None);
    }

    let Some(allowed_prefixes) = allowed_prefixes else {
        return Err(ApiError::invalid_request(
            format!(
                "Local LoRA adapter paths require {RUNTIME_LORA_ALLOWED_PATH_PREFIXES_ENV} to be configured."
            ),
            Some("lora_path"),
        ));
    };

    if !path.is_absolute() {
        return Err(ApiError::invalid_request(
            format!(
                "Local LoRA adapter paths must be absolute and under one of the prefixes configured by {RUNTIME_LORA_ALLOWED_PATH_PREFIXES_ENV}."
            ),
            Some("lora_path"),
        ));
    }

    let canonical_path = path.canonicalize().map_err(|_| {
        ApiError::invalid_request(
            "Local LoRA adapter path must exist and be accessible.".to_string(),
            Some("lora_path"),
        )
    })?;
    let canonical_prefixes = allowed_prefixes
        .iter()
        .map(|prefix| {
            prefix.canonicalize().map_err(|_| {
                ApiError::server_error(format!(
                    "configured {RUNTIME_LORA_ALLOWED_PATH_PREFIXES_ENV} path prefix must exist and be accessible"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    if !canonical_prefixes
        .iter()
        .any(|prefix| canonical_path.starts_with(prefix))
    {
        return Err(ApiError::invalid_request(
            "Local LoRA adapter path is outside the configured allowed prefixes.".to_string(),
            Some("lora_path"),
        ));
    }

    Ok(Some(canonical_path.to_string_lossy().into_owned()))
}

/// Dynamically load one LoRA adapter and expose it as an OpenAI model id.
pub(super) async fn load_lora_adapter(
    State(state): State<Arc<AppState>>,
    ValidatedJson(request): ValidatedJson<LoadLoraAdapterRequest>,
) -> Result<String, ApiError> {
    if request.lora_name.is_empty() || request.lora_path.is_empty() {
        return Err(ApiError::invalid_request(
            "Both 'lora_name' and 'lora_path' must be provided.".to_string(),
            None,
        ));
    }
    // `is_3d_lora_weight` is a phantom contract field: it is collected here and
    // carried over the wire, but no backend honors it. The worker's
    // MergeOnLoadLoRA always interprets adapters as 2D A@B deltas, so a request
    // with `is_3d_lora_weight=true` would otherwise produce a silently wrong
    // merge. Reject it loudly instead of accepting a flag we cannot deliver.
    if request.is_3d_lora_weight {
        return Err(ApiError::invalid_request(
            "'is_3d_lora_weight' is not supported: the LoRA backend only merges 2D adapter weights."
                .to_string(),
            Some("is_3d_lora_weight"),
        ));
    }
    let allowed_prefixes = runtime_lora_allowed_path_prefixes();
    let lora_path = validate_lora_path_access(&request.lora_path, allowed_prefixes.as_deref())?
        .unwrap_or(request.lora_path);

    let lora_name = request.lora_name;
    state
        .load_lora(
            lora_name.clone(),
            lora_path,
            request.load_inplace,
            request.is_3d_lora_weight,
        )
        .await
        .map_err(|error| match error {
            LoadLoraError::AlreadyLoaded { lora_name } => ApiError::invalid_request(
                format!(
                    "The lora adapter '{lora_name}' has already been loaded. If you want to load the adapter in place, set 'load_inplace' to true."
                ),
                Some("lora_name"),
            ),
            LoadLoraError::BaseModelName { lora_name } => ApiError::invalid_request(
                format!("The lora adapter name '{lora_name}' conflicts with a served base model."),
                Some("lora_name"),
            ),
            LoadLoraError::Engine(error) => ApiError::server_error(format!(
                "failed to load LoRA adapter '{lora_name}': {}",
                error.to_report_string()
            )),
            LoadLoraError::NotLoaded { lora_name } => ApiError::server_error(format!(
                "failed to load LoRA adapter '{lora_name}': engine rejected the adapter"
            )),
            LoadLoraError::IdSpaceExhausted => ApiError::server_error(
                "failed to load LoRA adapter: the adapter id space (u32) is exhausted.".to_string(),
            ),
        })?;

    Ok(format!(
        "Success: LoRA adapter '{lora_name}' added successfully."
    ))
}

/// Remove one LoRA adapter from the engine and frontend registry.
pub(super) async fn unload_lora_adapter(
    State(state): State<Arc<AppState>>,
    ValidatedJson(request): ValidatedJson<UnloadLoraAdapterRequest>,
) -> Result<String, ApiError> {
    if request.lora_name.is_empty() {
        return Err(ApiError::invalid_request(
            "'lora_name' needs to be provided to unload a LoRA adapter.".to_string(),
            Some("lora_name"),
        ));
    }

    let lora_request = state
        .unload_lora(&request.lora_name, request.lora_int_id)
        .await
        .map_err(|error| match error {
            UnloadLoraError::NotFound { lora_name } => ApiError::model_not_found(lora_name),
            UnloadLoraError::IntIdMismatch {
                lora_name,
                expected,
                actual,
            } => ApiError::invalid_request(
                format!(
                    "The requested lora_int_id {actual} does not match loaded adapter '{lora_name}' with id {expected}."
                ),
                Some("lora_int_id"),
            ),
            UnloadLoraError::Engine(error) => ApiError::server_error(format!(
                "failed to unload LoRA adapter '{}': {}",
                request.lora_name,
                error.to_report_string()
            )),
            UnloadLoraError::NotRemoved {
                lora_name,
                lora_int_id,
            } => ApiError::server_error(format!(
                "failed to unload LoRA adapter '{lora_name}' with id {lora_int_id}"
            )),
        })?;

    Ok(format!(
        "Success: LoRA adapter '{}' removed successfully.",
        lora_request.lora_name
    ))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::validate_lora_path_access;

    fn temp_lora_dir(test_name: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "uniserve-lora-{test_name}-{}-{suffix}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create temp lora dir");
        path
    }

    #[test]
    fn lora_path_allows_hf_repo_ids_without_prefixes() {
        assert_eq!(
            validate_lora_path_access("org/adapter-a", None).expect("hf repo id should be allowed"),
            None
        );
    }

    #[test]
    fn lora_path_rejects_local_paths_without_prefixes() {
        assert!(validate_lora_path_access("/tmp/adapter-a", None).is_err());
        assert!(validate_lora_path_access("./adapter-a", None).is_err());
        assert!(validate_lora_path_access("~/adapter-a", None).is_err());
        assert!(validate_lora_path_access("subdir/../../../etc/sensitive", None).is_err());
    }

    #[test]
    fn lora_path_rejects_existing_bare_relative_paths_without_prefixes() {
        let root =
            PathBuf::from("target").join(format!("uniserve-lora-relative-{}", std::process::id()));
        let adapter = root.join("adapter-a");
        fs::create_dir_all(&adapter).expect("create relative adapter dir");

        assert!(
            validate_lora_path_access(adapter.to_str().expect("utf-8 temp path"), None).is_err()
        );

        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn lora_path_allows_absolute_paths_under_configured_prefixes() {
        let root = temp_lora_dir("allowed-prefix");
        let allowed = root.join("allowed");
        let adapter = allowed.join("adapter-a");
        fs::create_dir_all(&adapter).expect("create adapter dir");

        let prefixes = [allowed];
        let resolved =
            validate_lora_path_access(adapter.to_str().expect("utf-8 temp path"), Some(&prefixes))
                .expect("path under configured prefix should be allowed");
        assert_eq!(
            resolved.as_deref(),
            Some(
                adapter
                    .canonicalize()
                    .expect("canonical adapter")
                    .to_str()
                    .expect("utf-8 temp path")
            )
        );

        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn lora_path_rejects_parent_escape_from_configured_prefixes() {
        let root = temp_lora_dir("parent-escape");
        let allowed = root.join("allowed");
        let private_adapter = root.join("private").join("adapter-a");
        fs::create_dir_all(&allowed).expect("create allowed dir");
        fs::create_dir_all(&private_adapter).expect("create private adapter dir");

        let escaped = allowed.join("../private/adapter-a");
        let prefixes = [allowed];
        assert!(
            validate_lora_path_access(escaped.to_str().expect("utf-8 temp path"), Some(&prefixes))
                .is_err()
        );

        fs::remove_dir_all(root).ok();
    }
}
