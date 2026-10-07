//! Loopback stand-in for the Hugging Face Hub endpoints the asset resolvers use.
//!
//! The stub serves a repository listing (`/api/models/{repo}/revision/main`)
//! and whole or byte-range file downloads (`/{repo}/resolve/{revision}/{file}`)
//! of one commit's files, answering an unpublished file with 404 as the Hub
//! does, or answers either kind of request with a fixed failure status. Tests
//! drive the real Hub client against it, so only the network boundary is
//! replaced.

use std::collections::BTreeMap;
use std::path::Path;

use axum::Router;
use axum::http::{HeaderMap, HeaderName, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use hf_hub::Cache;
use hf_hub::api::tokio::ApiBuilder;

use crate::profile::assets::model_files::{HubClient, ModelSource};

/// Commit the stub reports for every file unless told otherwise, and the
/// snapshot [`seed_cache`] writes, so downloads land beside seeded files.
const COMMIT: &str = "5f3b0c1e";

/// Repository contents and failure modes served by the stub.
#[derive(Clone, Default)]
pub(super) struct HubStub {
    /// Published file contents by repository-relative path.
    pub(super) files: BTreeMap<String, Vec<u8>>,
    /// Status answered for the repository listing in place of the listing.
    pub(super) listing_status: Option<StatusCode>,
    /// Status answered for every file download in place of its content.
    pub(super) download_status: Option<StatusCode>,
    /// Commit reported for every file in place of [`COMMIT`], whatever
    /// revision the request names.
    pub(super) commit: Option<&'static str>,
}

impl HubStub {
    /// Publishes `files` as `(path, content)` pairs; contents must be
    /// non-empty.
    pub(super) fn with_files<C: AsRef<[u8]>>(files: &[(&str, C)]) -> Self {
        Self {
            files: files
                .iter()
                .map(|(name, content)| ((*name).to_owned(), content.as_ref().to_vec()))
                .collect(),
            ..Self::default()
        }
    }

    /// Serves the stub on a loopback port for the rest of the test.
    ///
    /// Returns a Hub source for `repo_id` whose client talks to the stub and
    /// whose cache lives under `cache_root`, the directory [`seed_cache`]
    /// writes.
    pub(super) async fn serve(self, cache_root: &Path, repo_id: &str) -> ModelSource {
        ModelSource::Hub {
            api: self.client(cache_root).await.api,
            repo_id: repo_id.to_owned(),
        }
    }

    /// Serves the stub on a loopback port for the rest of the test and
    /// returns a client that talks to it, with its cache under `cache_root`.
    pub(super) async fn client(self, cache_root: &Path) -> HubClient {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let router = Router::new().fallback(move |uri: Uri, headers: HeaderMap| {
            let stub = self.clone();
            async move { stub.answer(&uri, &headers) }
        });
        tokio::spawn(async move { axum::serve(listener, router).await });

        // A cache under the test directory also keeps the client from reading
        // a token file outside it.
        let cache = Cache::new(cache_root.join("hub"));
        let api = ApiBuilder::from_cache(cache.clone())
            .with_endpoint(endpoint)
            .with_progress(false)
            .build()
            .unwrap();
        HubClient { api, cache }
    }

    /// The commit the stub reports.
    fn commit(&self) -> &'static str {
        self.commit.unwrap_or(COMMIT)
    }

    /// Answers one request by its path and `Range` header.
    fn answer(&self, uri: &Uri, headers: &HeaderMap) -> Response {
        let path = uri.path();
        if path.starts_with("/api/models/") {
            if let Some(status) = self.listing_status {
                return status.into_response();
            }
            let siblings = self
                .files
                .keys()
                .map(|name| serde_json::json!({ "rfilename": name }))
                .collect::<Vec<_>>();
            return axum::Json(serde_json::json!({ "siblings": siblings, "sha": self.commit() }))
                .into_response();
        }

        let Some((_, filename)) = path
            .split_once("/resolve/")
            .and_then(|(_, rest)| rest.split_once('/'))
        else {
            return StatusCode::NOT_FOUND.into_response();
        };
        if let Some(status) = self.download_status {
            return status.into_response();
        }
        let Some(content) = self.files.get(filename) else {
            return StatusCode::NOT_FOUND.into_response();
        };

        // A `bytes=start-end` range is answered as partial content with the
        // end clamped to the file, as HTTP servers do; the Hub client probes
        // metadata with `bytes=0-0` and requests past the end when it
        // downloads. A request without a range receives the whole file.
        let last = content.len() - 1;
        let (status, first, end) = match headers
            .get(header::RANGE)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("bytes="))
            .and_then(|value| value.split_once('-'))
            .and_then(|(start, end)| {
                Some((start.parse::<usize>().ok()?, end.parse::<usize>().ok()?))
            }) {
            Some((start, end)) => (StatusCode::PARTIAL_CONTENT, start, end.min(last)),
            None => (StatusCode::OK, 0, last),
        };

        // The client sizes the download from Content-Range, names the cached
        // blob after the ETag and files it under the reported commit.
        (
            status,
            [
                (header::ETAG, format!("\"{}\"", filename.replace('/', "-"))),
                (
                    header::CONTENT_RANGE,
                    format!("bytes {first}-{end}/{}", content.len()),
                ),
                (
                    HeaderName::from_static("x-repo-commit"),
                    self.commit().to_owned(),
                ),
            ],
            content[first..=end].to_vec(),
        )
            .into_response()
    }
}

/// Writes `files` into the Hub cache under `cache_root` as the snapshot of
/// `repo_id` that `refs/main` names, the layout an earlier download leaves.
pub(super) fn seed_cache(cache_root: &Path, repo_id: &str, files: &[(&str, &str)]) {
    let repository = seed_snapshot(cache_root, repo_id, COMMIT, files);
    std::fs::create_dir_all(repository.join("refs")).unwrap();
    std::fs::write(repository.join("refs").join("main"), COMMIT).unwrap();
}

/// Writes `files` into the Hub cache under `cache_root` as the snapshot of
/// `repo_id` at `commit`, with no ref naming it, the layout
/// `huggingface_hub` leaves when it downloads at a commit revision. Returns
/// the repository's cache directory.
pub(super) fn seed_snapshot(
    cache_root: &Path,
    repo_id: &str,
    commit: &str,
    files: &[(&str, &str)],
) -> std::path::PathBuf {
    let repository = cache_root
        .join("hub")
        .join(format!("models--{}", repo_id.replace('/', "--")));
    let snapshot = repository.join("snapshots").join(commit);
    for (name, content) in files {
        let path = snapshot.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }
    repository
}
