//! Condition media sources: resolving `conditions[i].uri` into bytes under
//! the server's media policy.
//!
//! - `data:<type>;base64,<payload>` is always accepted. Its size is already
//!   bounded by the request body limit; the decoded media is bounded like any
//!   other source.
//! - `file://` resolves only below the configured media directory
//!   (`--media-directory`) and is refused when none is configured. A path
//!   outside the directory is refused before the filesystem is consulted, so
//!   requests cannot learn which files exist elsewhere. The path is then
//!   canonicalized, so symbolic links that lead outside the directory are
//!   refused, and the canonical path is opened without following a final
//!   symbolic link, so a link swapped in after the check is refused as well.
//! - `http://` and `https://` are fetched by the server unless remote media is
//!   disabled (`--remote-media`). A fetch has a connect timeout, a total
//!   timeout covering the body, a redirect limit and the size caps below; a
//!   redirect may not leave HTTP(S) or downgrade HTTPS to HTTP.
//!
//! Server-side request forgery: unless the policy allows private addresses,
//! every address the server connects to must be globally routable. Host
//! names are resolved by a resolver that drops loopback, private, link-local,
//! shared (CGNAT), multicast, documentation, benchmarking and reserved
//! addresses (IPv4-mapped, NAT64 and 6to4 IPv6 addresses by their embedded
//! IPv4 address), so the check applies to the address actually connected to
//! and DNS rebinding cannot bypass it; literal addresses in a URL or redirect
//! are checked the same way. Proxies from the environment are not used, so
//! no intermediary resolves on the server's behalf. Ports are not
//! restricted.
//!
//! Each condition is capped by its media type (image 30 MiB, video 50 MiB,
//! audio 15 MiB) and a request's media together by a total cap; a
//! `Content-Length` above the cap is refused before the body is read and a
//! body is abandoned as soon as it exceeds the cap.

use std::io::Read as _;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::os::unix::fs::OpenOptionsExt as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use base64::Engine as _;
use base64::engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig};
use bytes::{Bytes, BytesMut};
use reqwest::Url;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use reqwest::redirect;
use thiserror_ext::AsReport as _;

use super::VideoInputError;
use super::plan::ConditionType;

const MIB: u64 = 1 << 20;
/// Default cap of one image condition.
pub const IMAGE_BYTES: u64 = 30 * MIB;
/// Default cap of one video condition.
pub const VIDEO_BYTES: u64 = 50 * MIB;
/// Default cap of one audio condition.
pub const AUDIO_BYTES: u64 = 15 * MIB;

/// Size caps on fetched media, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaLimits {
    /// Cap of one image condition.
    pub image_bytes: u64,
    /// Cap of one video or video_audio condition.
    pub video_bytes: u64,
    /// Cap of one audio condition.
    pub audio_bytes: u64,
    /// Cap of all media of one request (`--max-request-bytes`).
    pub total_bytes: u64,
}

impl MediaLimits {
    /// The default per-type caps under a request total of `total_bytes`.
    pub const fn new(total_bytes: u64) -> Self {
        Self {
            image_bytes: IMAGE_BYTES,
            video_bytes: VIDEO_BYTES,
            audio_bytes: AUDIO_BYTES,
            total_bytes,
        }
    }

    const fn cap(&self, condition_type: ConditionType) -> u64 {
        match condition_type {
            ConditionType::Image => self.image_bytes,
            ConditionType::Video | ConditionType::VideoAudio => self.video_bytes,
            ConditionType::Audio => self.audio_bytes,
        }
    }
}

/// How the server fetches `http(s)://` media.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemoteMediaPolicy {
    /// Whether remote media is fetched at all (`--remote-media`).
    pub enabled: bool,
    /// Longest time to establish a connection.
    pub connect_timeout: Duration,
    /// Longest time for one fetch, body included.
    pub total_timeout: Duration,
    /// Most redirects one fetch follows.
    pub max_redirects: usize,
    /// Whether loopback, private and other non-global addresses may be
    /// fetched; only for deployments whose media servers live on such
    /// networks.
    pub allow_private_addresses: bool,
}

impl Default for RemoteMediaPolicy {
    /// Enabled, with a 10-second connect timeout, a 60-second total timeout,
    /// at most 5 redirects and public addresses only.
    fn default() -> Self {
        Self {
            enabled: true,
            connect_timeout: Duration::from_secs(10),
            total_timeout: Duration::from_secs(60),
            max_redirects: 5,
            allow_private_addresses: false,
        }
    }
}

/// The server's media policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaPolicy {
    /// The directory `file://` URIs resolve under (`--media-directory`);
    /// `None` refuses `file://`.
    pub media_directory: Option<PathBuf>,
    /// How remote media is fetched.
    pub remote: RemoteMediaPolicy,
    /// Size caps.
    pub limits: MediaLimits,
}

/// The bytes of one condition's media.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchedMedia {
    /// The media bytes; cloning shares them.
    pub bytes: Bytes,
    /// The media type the source declared (`data:` type or `Content-Type`),
    /// lowercased without parameters. Advisory only: the probe identifies
    /// the format from the bytes.
    pub declared_type: Option<String>,
}

impl FetchedMedia {
    /// The media bytes.
    pub fn bytes(&self) -> &Bytes {
        &self.bytes
    }
}

/// The bytes one request may still fetch.
struct Budget(AtomicU64);

impl Budget {
    /// Reserves `bytes`; `false` when the request total would be exceeded.
    fn take(&self, bytes: u64) -> bool {
        self.0
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |remaining| {
                remaining.checked_sub(bytes)
            })
            .is_ok()
    }
}

/// The media directory as configured and as canonicalized; a `file://`
/// path may name it either way.
#[derive(Debug, Clone)]
struct MediaRoot {
    configured: PathBuf,
    canonical: PathBuf,
}

/// Resolves condition URIs under a [`MediaPolicy`]; shared by all requests.
pub struct MediaFetcher {
    media_root: Option<MediaRoot>,
    remote: RemoteMediaPolicy,
    limits: MediaLimits,
    client: Option<reqwest::Client>,
}

impl MediaFetcher {
    /// Prepares the policy: resolves the media directory and builds the HTTP
    /// client when remote media is enabled.
    ///
    /// # Errors
    ///
    /// Fails when the media directory cannot be resolved or the HTTP client
    /// cannot be built.
    pub fn new(policy: MediaPolicy) -> anyhow::Result<Self> {
        let media_root = policy
            .media_directory
            .map(|directory| -> anyhow::Result<MediaRoot> {
                let failed =
                    |error| anyhow::anyhow!("media directory {}: {error}", directory.display());
                Ok(MediaRoot {
                    configured: std::path::absolute(&directory).map_err(failed)?,
                    canonical: std::fs::canonicalize(&directory).map_err(failed)?,
                })
            })
            .transpose()?;
        let client = policy
            .remote
            .enabled
            .then(|| http_client(&policy.remote))
            .transpose()?;
        Ok(Self {
            media_root,
            remote: policy.remote,
            limits: policy.limits,
            client,
        })
    }

    /// Fetches the media of every condition, concurrently, under one
    /// request total.
    ///
    /// `sources` holds each condition's type and URI in request order.
    ///
    /// # Errors
    ///
    /// Returns [`VideoInputError::Invalid`] naming the first failing
    /// `conditions[i]` to finish.
    pub async fn fetch_all(
        &self,
        sources: &[(ConditionType, &str)],
    ) -> Result<Vec<FetchedMedia>, VideoInputError> {
        let budget = Budget(AtomicU64::new(self.limits.total_bytes));
        futures::future::try_join_all(
            sources
                .iter()
                .enumerate()
                .map(|(index, &(kind, uri))| self.fetch(index, kind, uri, &budget)),
        )
        .await
    }

    async fn fetch(
        &self,
        index: usize,
        condition_type: ConditionType,
        uri: &str,
        budget: &Budget,
    ) -> Result<FetchedMedia, VideoInputError> {
        let reject = |message: String| VideoInputError::condition(index, message);
        let cap = self.limits.cap(condition_type);
        let scheme = uri
            .split_once(':')
            .map(|(scheme, _)| scheme.to_ascii_lowercase());
        let fetched = match scheme.as_deref() {
            // Remote bodies reserve their bytes against the total as they
            // arrive; local media once read.
            Some("http" | "https") => {
                return self.fetch_remote(uri, cap, budget).await.map_err(reject);
            }
            Some("data") => decode_data_uri(uri, cap),
            Some("file") => self.read_file(uri, cap).await,
            _ => Err("uri must be a data:, file://, http:// or https:// URI".to_owned()),
        }
        .map_err(reject)?;
        if !budget.take(fetched.bytes.len() as u64) {
            return Err(reject(self.total_exceeded()));
        }
        Ok(fetched)
    }

    fn total_exceeded(&self) -> String {
        format!(
            "the request's media exceed {} bytes in total",
            self.limits.total_bytes
        )
    }

    async fn read_file(&self, uri: &str, cap: u64) -> Result<FetchedMedia, String> {
        let Some(root) = self.media_root.clone() else {
            return Err("file:// media is disabled: the server has no media directory".to_owned());
        };
        let url = Url::parse(uri).map_err(|error| format!("malformed uri: {error}"))?;
        let path = url
            .to_file_path()
            .map_err(|()| "a file:// uri must name a local absolute path".to_owned())?;
        tokio::task::spawn_blocking(move || read_media_file(&root, &path, cap))
            .await
            .map_err(|error| format!("reading the file failed: {error}"))?
    }

    async fn fetch_remote(
        &self,
        uri: &str,
        cap: u64,
        budget: &Budget,
    ) -> Result<FetchedMedia, String> {
        let Some(client) = &self.client else {
            return Err("remote media is disabled on this server".to_owned());
        };
        let url = Url::parse(uri).map_err(|error| format!("malformed uri: {error}"))?;
        check_remote_url(&url, self.remote.allow_private_addresses)?;

        let failed =
            |error: reqwest::Error| format!("fetching {uri} failed: {}", error.as_report());
        let mut response = client.get(url).send().await.map_err(failed)?;
        let status = response.status();
        if !status.is_success() {
            return Err(format!(
                "fetching {uri} failed: the server answered {status}"
            ));
        }
        if response.content_length().is_some_and(|length| length > cap) {
            return Err(format!("the media exceeds the {cap}-byte limit"));
        }
        let declared_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .and_then(media_type);

        let mut body = BytesMut::new();
        while let Some(chunk) = response.chunk().await.map_err(failed)? {
            let length = chunk.len() as u64;
            if body.len() as u64 + length > cap {
                return Err(format!("the media exceeds the {cap}-byte limit"));
            }
            if !budget.take(length) {
                return Err(self.total_exceeded());
            }
            body.extend_from_slice(&chunk);
        }
        Ok(FetchedMedia {
            bytes: body.freeze(),
            declared_type,
        })
    }
}

/// The essence of a media type: lowercased, parameters dropped.
fn media_type(value: &str) -> Option<String> {
    let essence = value.split(';').next()?.trim().to_ascii_lowercase();
    (!essence.is_empty()).then_some(essence)
}

/// Decodes a base64 `data:` URI of at most `cap` decoded bytes.
fn decode_data_uri(uri: &str, cap: u64) -> Result<FetchedMedia, String> {
    let (header, payload) = uri
        .get(5..)
        .and_then(|rest| rest.split_once(','))
        .ok_or_else(|| "a data: uri needs a comma before its payload".to_owned())?;
    let mut parameters = header.split(';');
    let declared_type = parameters.next().and_then(media_type);
    if !parameters.any(|parameter| parameter.trim().eq_ignore_ascii_case("base64")) {
        return Err("a data: uri must be base64-encoded".to_owned());
    }
    if base64::decoded_len_estimate(payload.len()) as u64 > cap + 3 {
        return Err(format!("the media exceeds the {cap}-byte limit"));
    }
    // Line breaks are tolerated; the payload is copied only when present.
    let compact: std::borrow::Cow<'_, str> =
        if payload.bytes().any(|byte| byte.is_ascii_whitespace()) {
            payload.split_ascii_whitespace().collect::<String>().into()
        } else {
            payload.into()
        };
    const ENGINE: GeneralPurpose = GeneralPurpose::new(
        &base64::alphabet::STANDARD,
        GeneralPurposeConfig::new().with_decode_padding_mode(DecodePaddingMode::Indifferent),
    );
    let bytes = ENGINE
        .decode(compact.as_bytes())
        .map_err(|error| format!("the data: uri payload is not base64: {error}"))?;
    if bytes.is_empty() {
        return Err("the data: uri payload is empty".to_owned());
    }
    if bytes.len() as u64 > cap {
        return Err(format!("the media exceeds the {cap}-byte limit"));
    }
    Ok(FetchedMedia {
        bytes: Bytes::from(bytes),
        declared_type,
    })
}

/// Reads a regular file below the media directory of at most `cap` bytes.
///
/// `path` is absolute with its `.` and `..` segments resolved, as URL
/// parsing leaves a `file://` path.
fn read_media_file(root: &MediaRoot, path: &Path, cap: u64) -> Result<FetchedMedia, String> {
    let shown = path.display();
    let outside = || format!("{shown} lies outside the media directory");
    if !path.starts_with(&root.configured) && !path.starts_with(&root.canonical) {
        return Err(outside());
    }
    // Canonicalization resolves every symbolic link, so a path that leaves
    // the media directory through a link ends outside its canonical form.
    let canonical = std::fs::canonicalize(path).map_err(|_| format!("{shown} does not exist"))?;
    if !canonical.starts_with(&root.canonical) {
        return Err(outside());
    }
    // Refuse a final symbolic link swapped in after canonicalization.
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&canonical)
        .map_err(|error| format!("{shown} cannot be opened: {error}"))?;
    let metadata = file
        .metadata()
        .map_err(|error| format!("{shown} cannot be read: {error}"))?;
    if !metadata.is_file() {
        return Err(format!("{shown} is not a regular file"));
    }
    if metadata.len() > cap {
        return Err(format!("the media exceeds the {cap}-byte limit"));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    (&mut file)
        .take(cap + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("{shown} cannot be read: {error}"))?;
    if bytes.len() as u64 > cap {
        return Err(format!("the media exceeds the {cap}-byte limit"));
    }
    if bytes.is_empty() {
        return Err(format!("{shown} is empty"));
    }
    Ok(FetchedMedia {
        bytes: Bytes::from(bytes),
        declared_type: None,
    })
}

/// Checks a URL the server is about to fetch: HTTP(S), with a host, and,
/// unless private addresses are allowed, no non-global literal address.
fn check_remote_url(url: &Url, allow_private: bool) -> Result<(), String> {
    if !matches!(url.scheme(), "http" | "https") {
        return Err(format!("{url} is not an http:// or https:// URL"));
    }
    let host = url
        .host_str()
        .filter(|host| !host.is_empty())
        .ok_or_else(|| format!("{url} has no host"))?;
    // URL parsing normalizes literal addresses (IPv6 in brackets), so a
    // literal parses here whichever form the request wrote it in.
    let literal = host
        .trim_start_matches('[')
        .trim_end_matches(']')
        .parse::<IpAddr>();
    if let Ok(address) = literal
        && !allow_private
        && !is_public(address)
    {
        return Err(format!("{url} names a non-public address"));
    }
    Ok(())
}

fn http_client(policy: &RemoteMediaPolicy) -> anyhow::Result<reqwest::Client> {
    let allow_private = policy.allow_private_addresses;
    let max_redirects = policy.max_redirects;
    let redirects = redirect::Policy::custom(move |attempt| {
        if attempt.previous().len() > max_redirects {
            return attempt.error(format!("more than {max_redirects} redirects"));
        }
        let downgrade = attempt
            .previous()
            .last()
            .is_some_and(|previous| previous.scheme() == "https")
            && attempt.url().scheme() == "http";
        if downgrade {
            return attempt.error("a redirect from https to http");
        }
        match check_remote_url(attempt.url(), allow_private) {
            Ok(()) => attempt.follow(),
            Err(error) => attempt.error(format!("a redirect to {error}")),
        }
    });
    let mut builder = reqwest::Client::builder()
        .connect_timeout(policy.connect_timeout)
        .timeout(policy.total_timeout)
        .redirect(redirects)
        .no_proxy()
        .user_agent(concat!("uniserve/", env!("CARGO_PKG_VERSION")));
    if !allow_private {
        builder = builder.dns_resolver(Arc::new(PublicResolver));
    }
    Ok(builder.build()?)
}

/// Resolves host names to their globally routable addresses only.
struct PublicResolver;

impl Resolve for PublicResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let host = name.as_str().to_owned();
        Box::pin(async move {
            let resolved: Vec<SocketAddr> = tokio::net::lookup_host((host.as_str(), 0))
                .await?
                .filter(|address| is_public(address.ip()))
                .collect();
            if resolved.is_empty() {
                return Err(format!("{host} resolves to no public address").into());
            }
            let addresses: Addrs = Box::new(resolved.into_iter());
            Ok(addresses)
        })
    }
}

/// Whether an address is globally routable.
fn is_public(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => is_public_v4(address),
        IpAddr::V6(address) => is_public_v6(address),
    }
}

fn is_public_v4(address: Ipv4Addr) -> bool {
    let [a, b, c, _] = address.octets();
    let non_public = a == 0 // "this network"
        || a == 10
        || a == 127
        || (a == 100 && (64..128).contains(&b)) // shared address space (CGNAT)
        || (a == 169 && b == 254)
        || (a == 172 && (16..32).contains(&b))
        || (a == 192 && b == 0 && c == 0) // IETF protocol assignments
        || (a == 192 && b == 0 && c == 2) // documentation
        || (a == 192 && b == 88 && c == 99) // 6to4 relay anycast
        || (a == 192 && b == 168)
        || (a == 198 && (b == 18 || b == 19)) // benchmarking
        || (a == 198 && b == 51 && c == 100) // documentation
        || (a == 203 && b == 0 && c == 113) // documentation
        || a >= 224; // multicast, reserved and broadcast
    !non_public
}

fn is_public_v6(address: Ipv6Addr) -> bool {
    if let Some(mapped) = address.to_ipv4_mapped() {
        return is_public_v4(mapped);
    }
    let segments = address.segments();
    let embedded = |high: u16, low: u16| {
        Ipv4Addr::new((high >> 8) as u8, high as u8, (low >> 8) as u8, low as u8)
    };
    match segments {
        // NAT64 well-known prefix: judged by the embedded address.
        [0x64, 0xff9b, 0, 0, 0, 0, high, low] => return is_public_v4(embedded(high, low)),
        // 6to4: judged by the embedded address.
        [0x2002, high, low, ..] => return is_public_v4(embedded(high, low)),
        _ => {}
    }
    let [first, second, ..] = segments;
    let non_public = address.is_unspecified()
        || address.is_loopback()
        || segments[..6] == [0; 6] // IPv4-compatible (deprecated)
        || (first == 0x64 && second == 0xff9b) // local-use NAT64
        || (first == 0x100 && segments[1..4] == [0; 3]) // discard
        || (first == 0x2001 && second < 0x200) // IETF protocol assignments, Teredo
        || (first == 0x2001 && second == 0xdb8) // documentation
        || (first & 0xfe00) == 0xfc00 // unique local
        || (first & 0xffc0) == 0xfe80 // link-local
        || (first & 0xffc0) == 0xfec0 // site-local (deprecated)
        || (first & 0xff00) == 0xff00; // multicast
    !non_public
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, SocketAddr};
    use std::path::Path;
    use std::time::Duration;

    use axum::Router;
    use axum::body::Body;
    use axum::http::{StatusCode, header};
    use axum::response::{IntoResponse, Redirect};
    use axum::routing::get;
    use base64::Engine as _;

    use super::super::RequestField;
    use super::super::plan::ConditionType;
    use super::{MediaFetcher, MediaLimits, MediaPolicy, RemoteMediaPolicy, is_public};

    fn limits() -> MediaLimits {
        MediaLimits {
            image_bytes: 64,
            video_bytes: 128,
            audio_bytes: 32,
            total_bytes: 160,
        }
    }

    fn media_fetcher(media_directory: Option<&Path>, remote: RemoteMediaPolicy) -> MediaFetcher {
        MediaFetcher::new(MediaPolicy {
            media_directory: media_directory.map(Path::to_path_buf),
            remote,
            limits: limits(),
        })
        .unwrap()
    }

    fn local_remote() -> RemoteMediaPolicy {
        RemoteMediaPolicy {
            allow_private_addresses: true,
            connect_timeout: Duration::from_secs(5),
            total_timeout: Duration::from_secs(2),
            ..RemoteMediaPolicy::default()
        }
    }

    async fn fetch_one(
        fetcher: &MediaFetcher,
        kind: ConditionType,
        uri: &str,
    ) -> Result<Vec<u8>, String> {
        fetcher
            .fetch_all(&[(kind, uri)])
            .await
            .map(|mut media| media.remove(0).bytes.to_vec())
            .map_err(|error| {
                assert_eq!(error.field(), Some(RequestField::Condition(0)), "{error}");
                error.to_string()
            })
    }

    fn data_uri(kind: &str, bytes: &[u8]) -> String {
        format!(
            "data:{kind};base64,{}",
            base64::engine::general_purpose::STANDARD.encode(bytes)
        )
    }

    /// Base64 `data:` URIs decode under the per-type cap; other encodings
    /// and malformed payloads are refused.
    #[tokio::test]
    async fn data_uris_decode_under_the_cap() {
        let fetcher = media_fetcher(None, RemoteMediaPolicy::default());
        let media = fetcher
            .fetch_all(&[(ConditionType::Image, &data_uri("Image/PNG", b"pixels"))])
            .await
            .unwrap();
        assert_eq!(media[0].bytes.as_ref(), b"pixels");
        assert_eq!(media[0].declared_type.as_deref(), Some("image/png"));

        let wrapped = "data:audio/wav;base64,cGl4\nZWxz";
        let decoded = fetch_one(&fetcher, ConditionType::Audio, wrapped).await;
        assert_eq!(decoded.unwrap(), b"pixels");

        for uri in [
            "data:image/png,pixels".to_owned(),
            "data:image/png;base64,***".to_owned(),
            "data:image/png;base64,".to_owned(),
            "data:image/png;base64".to_owned(),
            data_uri("image/png", &[7; 65]),
        ] {
            let refused = fetch_one(&fetcher, ConditionType::Image, &uri).await;
            assert!(refused.is_err(), "{uri}");
        }
        // Each type has its own cap.
        let audio = data_uri("audio/wav", &[7; 33]);
        assert!(
            fetch_one(&fetcher, ConditionType::Audio, &audio)
                .await
                .is_err()
        );
        let video = data_uri("video/mp4", &[7; 128]);
        assert!(
            fetch_one(&fetcher, ConditionType::Video, &video)
                .await
                .is_ok()
        );
    }

    /// The media of a request share one total cap.
    #[tokio::test]
    async fn a_request_shares_one_total() {
        let fetcher = media_fetcher(None, RemoteMediaPolicy::default());
        let video = data_uri("video/mp4", &[1; 100]);
        let image = data_uri("image/png", &[2; 64]);
        let error = fetcher
            .fetch_all(&[
                (ConditionType::Video, &video),
                (ConditionType::Image, &image),
            ])
            .await
            .unwrap_err();
        assert!(error.to_string().contains("in total"), "{error}");
    }

    /// `file://` resolves only below the media directory: traversal and
    /// symbolic links out of it are refused, links within it are followed.
    #[tokio::test]
    async fn files_resolve_below_the_media_directory() {
        let outside = tempfile::tempdir().unwrap();
        let secret = outside.path().join("secret.wav");
        std::fs::write(&secret, b"secret").unwrap();
        let root = tempfile::tempdir().unwrap();
        let media = root.path().join("media");
        std::fs::create_dir(&media).unwrap();
        std::fs::write(media.join("voice.wav"), b"voice").unwrap();
        std::fs::write(media.join("large.wav"), [0; 33]).unwrap();
        std::os::unix::fs::symlink(&secret, media.join("escape.wav")).unwrap();
        std::os::unix::fs::symlink(media.join("voice.wav"), media.join("alias.wav")).unwrap();

        let fetcher = media_fetcher(Some(&media), RemoteMediaPolicy::default());
        let uri = |path: &Path| format!("file://{}", path.display());
        let audio = ConditionType::Audio;
        assert_eq!(
            fetch_one(&fetcher, audio, &uri(&media.join("voice.wav")))
                .await
                .unwrap(),
            b"voice"
        );
        assert_eq!(
            fetch_one(&fetcher, audio, &uri(&media.join("alias.wav")))
                .await
                .unwrap(),
            b"voice"
        );
        for refused in [
            uri(&media.join("escape.wav")),
            format!("file://{}/../../{}", media.display(), secret.display()),
            uri(&secret),
            uri(&media.join("missing.wav")),
            uri(&media),
            uri(&media.join("large.wav")),
            format!("file://example.com{}", media.join("voice.wav").display()),
        ] {
            assert!(
                fetch_one(&fetcher, audio, &refused).await.is_err(),
                "{refused}"
            );
        }
        // Whether a file exists outside the directory is not revealed.
        for outside_path in [secret.clone(), outside.path().join("missing.wav")] {
            let error = fetch_one(&fetcher, audio, &uri(&outside_path))
                .await
                .unwrap_err();
            assert!(error.contains("outside the media directory"), "{error}");
        }
        // The directory may be named through a link to it.
        let linked = root.path().join("linked");
        std::os::unix::fs::symlink(&media, &linked).unwrap();
        let through_link = media_fetcher(Some(&linked), RemoteMediaPolicy::default());
        for path in [linked.join("voice.wav"), media.join("voice.wav")] {
            let read = fetch_one(&through_link, audio, &uri(&path)).await;
            assert_eq!(read.unwrap(), b"voice");
        }

        let without_directory = media_fetcher(None, RemoteMediaPolicy::default());
        let error = fetch_one(&without_directory, audio, &uri(&media.join("voice.wav")))
            .await
            .unwrap_err();
        assert!(error.contains("disabled"), "{error}");
        assert!(
            fetch_one(&without_directory, audio, "ftp://example.com/a.wav")
                .await
                .is_err()
        );
        assert!(
            fetch_one(&without_directory, audio, "voice.wav")
                .await
                .is_err()
        );
    }

    async fn serve(router: Router) -> SocketAddr {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        address
    }

    fn media_server() -> Router {
        let stream = || {
            let chunks =
                (0..6).map(|_| Ok::<_, std::io::Error>(bytes::Bytes::from_static(&[9; 16])));
            Body::from_stream(futures::stream::iter(chunks))
        };
        Router::new()
            .route(
                "/image.png",
                get(|| async {
                    (
                        [(header::CONTENT_TYPE, "image/png; charset=binary")],
                        vec![1_u8; 48],
                    )
                }),
            )
            .route("/large.png", get(|| async { vec![1_u8; 65] }))
            .route("/streamed.png", get(move || async move { stream() }))
            .route("/missing.png", get(|| async { StatusCode::NOT_FOUND }))
            .route(
                "/slow.png",
                get(|| async {
                    tokio::time::sleep(Duration::from_secs(10)).await;
                    vec![1_u8; 4].into_response()
                }),
            )
            .route(
                "/hop/{count}",
                get(
                    |axum::extract::Path(count): axum::extract::Path<u32>| async move {
                        if count == 0 {
                            Redirect::temporary("/image.png").into_response()
                        } else {
                            Redirect::temporary(&format!("/hop/{}", count - 1)).into_response()
                        }
                    },
                ),
            )
            .route(
                "/to-file",
                get(|| async { Redirect::temporary("file:///etc/passwd") }),
            )
    }

    /// Remote media is fetched under the size, status, redirect and time
    /// limits.
    #[tokio::test]
    async fn remote_media_follows_the_policy() {
        let address = serve(media_server()).await;
        let url = |path: &str| format!("http://{address}{path}");
        let fetcher = media_fetcher(None, local_remote());
        let image = ConditionType::Image;

        let media = fetcher
            .fetch_all(&[(image, &url("/image.png"))])
            .await
            .unwrap();
        assert_eq!(media[0].bytes.len(), 48);
        assert_eq!(media[0].declared_type.as_deref(), Some("image/png"));
        assert_eq!(
            fetch_one(&fetcher, image, &url("/hop/4"))
                .await
                .unwrap()
                .len(),
            48
        );

        for (path, reason) in [
            ("/large.png", "limit"),
            ("/streamed.png", "limit"),
            ("/missing.png", "404"),
            ("/hop/5", "redirect"),
            // A redirect out of HTTP(S) is not followed.
            ("/to-file", "307"),
            ("/slow.png", "failed"),
        ] {
            let error = fetch_one(&fetcher, image, &url(path)).await.unwrap_err();
            assert!(error.contains(reason), "{path}: {error}");
        }
        // The streamed body fits the larger video cap.
        assert_eq!(
            fetch_one(&fetcher, ConditionType::Video, &url("/streamed.png"))
                .await
                .unwrap()
                .len(),
            96
        );

        let disabled = media_fetcher(
            None,
            RemoteMediaPolicy {
                enabled: false,
                ..RemoteMediaPolicy::default()
            },
        );
        let error = fetch_one(&disabled, image, &url("/image.png"))
            .await
            .unwrap_err();
        assert!(error.contains("disabled"), "{error}");
    }

    /// Without the private-address allowance, loopback is refused whether
    /// named by address or by host name.
    #[tokio::test]
    async fn private_addresses_are_refused_by_default() {
        let address = serve(media_server()).await;
        let fetcher = media_fetcher(None, RemoteMediaPolicy::default());
        for uri in [
            format!("http://{address}/image.png"),
            format!("http://localhost:{}/image.png", address.port()),
            format!("http://[::1]:{}/image.png", address.port()),
        ] {
            assert!(
                fetch_one(&fetcher, ConditionType::Image, &uri)
                    .await
                    .is_err(),
                "{uri}"
            );
        }
    }

    /// Only globally routable addresses are public.
    #[test]
    fn public_addresses() {
        for address in [
            "8.8.8.8",
            "1.1.1.1",
            "100.63.255.255",
            "100.128.0.0",
            "172.32.0.1",
            "2606:4700:4700::1111",
            "2002:0808:0808::1",
            "64:ff9b::808:808",
            "::ffff:8.8.8.8",
        ] {
            assert!(is_public(address.parse::<IpAddr>().unwrap()), "{address}");
        }
        for address in [
            "0.0.0.0",
            "10.1.2.3",
            "100.64.0.1",
            "127.0.0.1",
            "169.254.169.254",
            "172.16.0.1",
            "192.0.0.8",
            "192.0.2.1",
            "192.168.1.1",
            "198.18.0.1",
            "198.51.100.1",
            "203.0.113.1",
            "224.0.0.1",
            "255.255.255.255",
            "::",
            "::1",
            "::127.0.0.1",
            "::ffff:127.0.0.1",
            "::ffff:10.0.0.1",
            "64:ff9b::a00:1",
            "64:ff9b:1::1",
            "2001:db8::1",
            "2001::1",
            "2002:0a00:0001::1",
            "fc00::1",
            "fd12:3456::1",
            "fe80::1",
            "fec0::1",
            "ff02::1",
        ] {
            assert!(!is_public(address.parse::<IpAddr>().unwrap()), "{address}");
        }
    }
}
