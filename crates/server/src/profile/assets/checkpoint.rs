//! Tensor shapes read from a safetensors checkpoint header.
//!
//! A safetensors file starts with an 8-byte little-endian header length and
//! that many bytes of JSON mapping each tensor name to its dtype, shape and
//! data offsets. Reading only the header gives tensor shapes without loading
//! weights: a local file is read in place, and a Hub file through two HTTP
//! range requests, so a multi-gigabyte shard is never downloaded.

use std::io::Read as _;
use std::path::Path;

use hf_hub::api::tokio::Api;
use serde_json::Value;

use crate::profile::assets::error::{Error, Result};
use crate::profile::assets::model_files::ModelSource;

/// Largest header the safetensors format permits, in bytes.
const MAX_HEADER_BYTES: u64 = 100_000_000;

/// Returns the shape of `tensor` in the first of `filenames` the checkpoint
/// publishes.
///
/// `filenames` are checkpoint-relative and tried in order, as a model
/// loader's explicit primary-checkpoint names are.
///
/// # Errors
///
/// Returns [`Error::MissingFile`] when the checkpoint publishes none of
/// `filenames`, [`Error::Remote`] when a Hub request fails, [`Error::Io`]
/// when a local file cannot be read, and [`Error::Invalid`] when the header
/// is malformed or does not declare `tensor` with an integer shape.
pub async fn resolve_tensor_shape(
    model_id: &str,
    filenames: &[&str],
    tensor: &str,
) -> Result<Vec<u64>> {
    tensor_shape(&ModelSource::from_model_id(model_id)?, filenames, tensor).await
}

/// Resolves a tensor shape from an already classified model source.
async fn tensor_shape(source: &ModelSource, filenames: &[&str], tensor: &str) -> Result<Vec<u64>> {
    for filename in filenames {
        let header = match header(source, filename).await {
            Ok(header) => header,
            Err(Error::MissingFile { .. }) => continue,
            Err(error) => return Err(error),
        };
        return header
            .get(tensor)
            .and_then(|entry| entry.get("shape"))
            .and_then(Value::as_array)
            .and_then(|shape| shape.iter().map(Value::as_u64).collect::<Option<Vec<_>>>())
            .ok_or_else(|| {
                Error::invalid(format!(
                    "checkpoint file `{filename}` declares no integer shape for tensor `{tensor}`"
                ))
            });
    }
    Err(Error::MissingFile {
        model: match source {
            ModelSource::Local(directory) => directory.display().to_string(),
            ModelSource::Hub { repo_id, .. } => repo_id.clone(),
        },
        file: filenames.join(" or "),
    })
}

/// Reads and parses the JSON header of one safetensors file.
async fn header(source: &ModelSource, filename: &str) -> Result<Value> {
    let bytes = match source {
        ModelSource::Local(directory) => local_header(&directory.join(filename))?,
        ModelSource::Hub { api, repo_id } => hub_header(api, repo_id, filename).await?,
    };
    serde_json::from_slice(&bytes).map_err(|error| {
        Error::invalid(format!(
            "checkpoint file `{filename}` has a malformed safetensors header: {error}"
        ))
    })
}

/// Returns a header length in bytes after checking the format's bound.
fn header_length(prefix: [u8; 8], filename: &str) -> Result<u64> {
    let length = u64::from_le_bytes(prefix);
    if length > MAX_HEADER_BYTES {
        return Err(Error::invalid(format!(
            "checkpoint file `{filename}` declares a {length}-byte safetensors header"
        )));
    }
    Ok(length)
}

/// Reads the header bytes of a local safetensors file.
///
/// A file that does not exist is [`Error::MissingFile`].
fn local_header(path: &Path) -> Result<Vec<u8>> {
    let io_error = |source| Error::Io {
        path: path.to_path_buf(),
        source,
    };
    let mut file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(Error::MissingFile {
                model: path
                    .parent()
                    .map(|parent| parent.display().to_string())
                    .unwrap_or_default(),
                file: path
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            });
        }
        Err(error) => return Err(io_error(error)),
    };

    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix).map_err(io_error)?;
    let length = header_length(prefix, &path.display().to_string())?;
    let mut header = Vec::new();
    file.take(length)
        .read_to_end(&mut header)
        .map_err(io_error)?;
    if header.len() as u64 != length {
        return Err(Error::invalid(format!(
            "checkpoint file `{}` ends inside its safetensors header",
            path.display()
        )));
    }
    Ok(header)
}

/// Reads the header bytes of a Hub safetensors file with two range requests:
/// the 8-byte length prefix, then the header itself.
async fn hub_header(api: &Api, repo_id: &str, filename: &str) -> Result<Vec<u8>> {
    let prefix = hub_range(api, repo_id, filename, 0, 8).await?;
    let prefix: [u8; 8] = prefix.as_slice().try_into().map_err(|_| {
        Error::invalid(format!(
            "checkpoint file `{filename}` is shorter than a safetensors header"
        ))
    })?;
    let length = header_length(prefix, filename)?;
    hub_range(api, repo_id, filename, 8, length).await
}

/// Fetches `length` bytes of a Hub file starting at byte `start`.
///
/// The Hub's 404 answer is [`Error::MissingFile`]. Any other failure, and an
/// answer that is not the requested partial content, is [`Error::Remote`]:
/// the whole file is never read.
async fn hub_range(
    api: &Api,
    repo_id: &str,
    filename: &str,
    start: u64,
    length: u64,
) -> Result<Vec<u8>> {
    let remote = |message: String| Error::Remote {
        model: repo_id.to_owned(),
        message: format!("failed to read the header of '{filename}': {message}"),
    };
    if length == 0 {
        return Ok(Vec::new());
    }

    let url = api.model(repo_id.to_owned()).url(filename);
    let response = api
        .client()
        .get(url)
        .header("Range", format!("bytes={start}-{}", start + length - 1))
        .send()
        .await
        .map_err(|error| remote(error.to_string()))?;
    match response.status().as_u16() {
        // Partial Content: the server honored the byte range.
        206 => {}
        404 => {
            return Err(Error::MissingFile {
                model: repo_id.to_owned(),
                file: filename.to_owned(),
            });
        }
        status => return Err(remote(format!("unexpected HTTP status {status}"))),
    }

    let bytes = response
        .bytes()
        .await
        .map_err(|error| remote(error.to_string()))?;
    if bytes.len() as u64 != length {
        return Err(remote(format!(
            "expected {length} bytes, received {}",
            bytes.len()
        )));
    }
    Ok(bytes.to_vec())
}

/// Serializes a safetensors file holding zero-filled BF16 tensors of the
/// given shapes, for tests that need a checkpoint header.
#[cfg(test)]
pub(crate) fn bf16_safetensors(tensors: &[(&str, &[u64])]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut offset = 0_u64;
    for (name, shape) in tensors {
        let bytes = shape.iter().product::<u64>() * 2;
        header.insert(
            (*name).to_owned(),
            serde_json::json!({
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [offset, offset + bytes],
            }),
        );
        offset += bytes;
    }

    let header = serde_json::to_vec(&header).unwrap();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header);
    file.resize(file.len() + offset as usize, 0);
    file
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use tempfile::tempdir;

    use super::{bf16_safetensors, tensor_shape};
    use crate::profile::assets::Error;
    use crate::profile::assets::hub_stub::HubStub;
    use crate::profile::assets::model_files::ModelSource;

    const PRIMARY: [&str; 2] = ["ema.safetensors", "model.safetensors"];
    const TABLE: &str = "latent_pos_embed.pos_embed";

    /// A local checkpoint's shape comes from the first listed file present.
    #[tokio::test]
    async fn a_local_header_declares_the_tensor_shape() {
        let directory = tempdir().unwrap();
        std::fs::write(
            directory.path().join("model.safetensors"),
            bf16_safetensors(&[("other", &[3]), (TABLE, &[16, 8])]),
        )
        .unwrap();
        let source = ModelSource::Local(directory.path().to_path_buf());

        assert_eq!(
            tensor_shape(&source, &PRIMARY, TABLE).await.unwrap(),
            [16, 8]
        );
        assert!(matches!(
            tensor_shape(&source, &PRIMARY, "absent").await,
            Err(Error::Invalid(_))
        ));
        assert!(matches!(
            tensor_shape(&source, &["absent.safetensors"], TABLE).await,
            Err(Error::MissingFile { .. })
        ));
    }

    /// A Hub checkpoint's header is read with byte ranges, and a refused
    /// request is a remote failure rather than a missing file.
    #[tokio::test]
    async fn a_hub_header_is_read_without_the_tensor_data() {
        let root = tempdir().unwrap();
        let file = bf16_safetensors(&[(TABLE, &[4096, 4])]);
        let published = HubStub::with_files(&[("ema.safetensors", file.as_slice())]);
        let refused = HubStub {
            download_status: Some(StatusCode::FORBIDDEN),
            ..published.clone()
        };

        let source = published.serve(root.path(), "org/model").await;
        assert_eq!(
            tensor_shape(&source, &PRIMARY, TABLE).await.unwrap(),
            [4096, 4]
        );

        let source = refused.serve(root.path(), "org/model").await;
        assert!(matches!(
            tensor_shape(&source, &PRIMARY, TABLE).await,
            Err(Error::Remote { .. })
        ));
    }
}
