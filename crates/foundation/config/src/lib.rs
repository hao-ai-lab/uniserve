//! Shared configuration loading and validation helpers.

#![deny(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
use std::env;
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use thiserror::Error;

/// Result alias for shared configuration helpers.
pub type Result<T> = std::result::Result<T, ConfigError>;

/// Error taxonomy for configuration sources.
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("environment variable `{key}` has invalid value `{value}`: {message}")]
    InvalidEnv {
        key: String,
        value: String,
        message: String,
    },
    #[error("failed to read config file `{path}`: {source}")]
    ReadFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse config file `{path}` as JSON: {source}")]
    ParseJson {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

/// Read and deserialize one JSON config file.
/// This performs only *structural* validation: the file must exist, be valid
/// UTF-8 JSON, and match the shape of `T` (returning [`ConfigError::ReadFile`]
/// or [`ConfigError::ParseJson`] respectively). It does **not** perform any
/// schema or semantic validation beyond what `T`'s `Deserialize` impl enforces
/// (e.g. value ranges, cross-field invariants). Callers needing such checks
/// must validate the returned value themselves.
pub fn read_json_file<T>(path: impl AsRef<Path>) -> Result<T>
where
    T: DeserializeOwned,
{
    let path = path.as_ref();
    let content = std::fs::read_to_string(path).map_err(|source| ConfigError::ReadFile {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&content).map_err(|source| ConfigError::ParseJson {
        path: path.to_path_buf(),
        source,
    })
}

/// Parse an optional boolean environment flag.
/// The only accepted values are `true` and `false`.
pub fn env_bool(key: &str) -> Result<Option<bool>> {
    let Some(raw) = env::var_os(key) else {
        return Ok(None);
    };
    let value = raw.to_string_lossy().trim().to_ascii_lowercase();
    match value.as_str() {
        "true" => Ok(Some(true)),
        "false" => Ok(Some(false)),
        _ => Err(ConfigError::InvalidEnv {
            key: key.to_string(),
            value: raw.to_string_lossy().into_owned(),
            message: "expected true or false".to_string(),
        }),
    }
}

/// Split a path-list environment variable using platform path separators.
pub fn env_paths(key: &str) -> Option<Vec<PathBuf>> {
    let paths: Vec<_> = env::split_paths(&env::var_os(key)?)
        .filter(|path| !path.as_os_str().is_empty())
        .collect();
    (!paths.is_empty()).then_some(paths)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use serde::Deserialize;
    use tempfile::tempdir;

    use super::{ConfigError, read_json_file};

    #[derive(Debug, Deserialize, PartialEq, Eq)]
    struct Fixture {
        value: u32,
    }

    #[test]
    fn read_json_file_parses_typed_config() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.json");
        fs::write(&path, r#"{"value":7}"#).unwrap();

        assert_eq!(
            read_json_file::<Fixture>(&path).unwrap(),
            Fixture { value: 7 }
        );
    }

    #[test]
    fn read_json_file_reports_missing_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("does-not-exist.json");

        let err = read_json_file::<Fixture>(&path).unwrap_err();
        match err {
            ConfigError::ReadFile { path: reported, .. } => assert_eq!(reported, path),
            other => panic!("expected ReadFile, got {other:?}"),
        }
    }

    #[test]
    fn read_json_file_reports_malformed_json() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.json");
        fs::write(&path, "{ not valid json").unwrap();

        let err = read_json_file::<Fixture>(&path).unwrap_err();
        assert!(matches!(err, ConfigError::ParseJson { .. }));
    }

    #[test]
    fn read_json_file_does_not_validate_semantics() {
        // `read_json_file` only enforces `T`'s structure; out-of-range values
        // (e.g. a negative count for a field a caller treats as a positive
        // limit) deserialize fine. Semantic validation is the caller's job.
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.json");
        fs::write(&path, r#"{"value":0}"#).unwrap();

        assert_eq!(
            read_json_file::<Fixture>(&path).unwrap(),
            Fixture { value: 0 }
        );
    }
}
