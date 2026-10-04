//! Image reference resolution under address, redirect, time, size, and type limits.
//!
//! A `data:image/*;base64` URL is decoded locally. An `http(s)` URL is fetched
//! with a shared reqwest client, and these rules apply to the initial URL and
//! to every redirect hop:
//!
//! - Only `http` and `https` are followed, for at most
//!   [`ImageFetcher::MAX_REDIRECTS`] redirects.
//! - Unless [`ImageFetchPolicy::allow_private`] is set, every destination
//!   address must be public ([`is_public_address`]). A literal address in the
//!   URL is checked before the request. A host name is checked inside the
//!   client's DNS resolver, and the client connects only to the addresses that
//!   resolver returned, so a host cannot pass the check with one DNS answer
//!   and connect with another.
//! - The whole fetch, redirects and body included, completes within
//!   [`ImageFetchPolicy::timeout`].
//! - The image may not exceed [`ImageFetchPolicy::max_bytes`]. A declared
//!   `Content-Length` over the limit fails before the body is read, and the
//!   streamed byte count enforces the limit whatever the headers declare.
//! - The response must declare an `image/*` `Content-Type` or begin with a
//!   recognized image magic number, and its header must decode.
//!
//! Environment proxy settings are ignored: a proxy resolves the host itself,
//! which would bypass the address check.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use bytes::{Bytes, BytesMut};
use reqwest::header::{ACCEPT, CONTENT_TYPE, LOCATION};
use reqwest::{StatusCode, Url};
use serde::Serialize;
use thiserror::Error;
use tokio::sync::Semaphore;

use super::ImageInput;

/// Limits applied to every image reference a request supplies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ImageFetchPolicy {
    /// Time allowed for one http(s) fetch: connecting, every redirect hop,
    /// and reading the complete body. Waiting for a free fetch slot (see
    /// [`ImageFetcher::MAX_CONCURRENT_FETCHES`]) is not counted.
    pub timeout: Duration,
    /// Largest accepted encoded image, in bytes, for fetched bodies and for
    /// the decoded payload of data URLs alike.
    pub max_bytes: u64,
    /// Whether http(s) fetches may connect to addresses [`is_public_address`]
    /// rejects. Scheme, redirect, time, size, and type limits still apply.
    pub allow_private: bool,
}

impl ImageFetchPolicy {
    /// Default [`Self::timeout`].
    pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(20);
    /// Default [`Self::max_bytes`] (20 MB).
    pub const DEFAULT_MAX_BYTES: u64 = 20_000_000;
}

impl Default for ImageFetchPolicy {
    /// Returns the default time and size limits with public destinations only.
    fn default() -> Self {
        Self {
            timeout: Self::DEFAULT_TIMEOUT,
            max_bytes: Self::DEFAULT_MAX_BYTES,
            allow_private: false,
        }
    }
}

/// Failure to turn one image reference into an [`ImageInput`].
///
/// Messages describe the failure in terms of the reference the client sent.
/// They never name a redirect target or a resolved address, which may belong
/// to a network the client cannot otherwise observe.
#[derive(Debug, Error)]
pub enum ImageFetchError {
    /// The reference is neither an http(s) URL nor an image data URL.
    #[error("image URL must be an http(s) URL or a data:image/*;base64 URL")]
    UnsupportedScheme,
    /// The reference cannot be parsed as an absolute URL.
    #[error("image URL is not a valid absolute URL: {0}")]
    InvalidUrl(url::ParseError),
    /// A `data:` reference is not of the form `data:image/<subtype>;base64,<payload>`.
    #[error("image data URL must have the form data:image/<subtype>;base64,<payload>")]
    InvalidDataUrl,
    /// A data URL payload is not valid standard base64.
    #[error("image data URL contains invalid base64 data: {0}")]
    InvalidBase64(base64::DecodeError),
    /// The URL's host is, or resolves to, an address outside public unicast space.
    #[error("image URL host resolves to a non-public address")]
    NonPublicAddress,
    /// A redirect leads to an address outside public unicast space.
    #[error("image URL redirects to a non-public address")]
    NonPublicRedirect,
    /// A redirect leads to a scheme other than http or https.
    #[error("image URL redirects to a scheme other than http or https")]
    UnsupportedRedirect,
    /// A redirect response carries no usable `Location` header.
    #[error("image URL returned a redirect without a valid Location header")]
    InvalidRedirect,
    /// The URL redirects more times than [`ImageFetcher::MAX_REDIRECTS`].
    #[error("image URL redirects more than {limit} times")]
    TooManyRedirects {
        /// Redirects allowed per fetch.
        limit: usize,
    },
    /// The final response status is not a success.
    #[error("image URL returned HTTP status {status}")]
    Status {
        /// Response status code.
        status: u16,
    },
    /// The image exceeds [`ImageFetchPolicy::max_bytes`].
    #[error("image exceeds the {limit}-byte size limit")]
    TooLarge {
        /// Configured size limit in bytes.
        limit: u64,
    },
    /// The response neither declares an image type nor starts with an image magic number.
    #[error(
        "image URL response is not an image: its Content-Type is not image/* and its bytes match no image format"
    )]
    NotImage,
    /// The bytes do not form a readable image of a supported format.
    #[error("image data cannot be decoded: {0}")]
    Undecodable(image::ImageError),
    /// The fetch did not complete within [`ImageFetchPolicy::timeout`].
    #[error("image URL fetch did not complete within {timeout:?}")]
    Timeout {
        /// Configured fetch time limit.
        timeout: Duration,
    },
    /// Connecting, TLS, or reading the response failed.
    #[error("image URL could not be fetched: {reason}")]
    Transport {
        /// Innermost cause reported by the HTTP client.
        reason: String,
    },
    /// The blocking task that decodes image bytes did not complete.
    #[error("image decoding task failed")]
    Task(#[source] tokio::task::JoinError),
}

/// Failure of one reference in an ordered list resolved by [`ImageFetcher::fetch_all`].
#[derive(Debug, Error)]
#[error("input image {}: {source}", .index + 1)]
pub struct ImageListError {
    /// Zero-based position of the failing reference in the input list.
    pub index: usize,
    /// Why the reference could not be resolved.
    #[source]
    pub source: ImageFetchError,
}

/// Resolves image references into validated [`ImageInput`] values.
///
/// One fetcher serves the whole server: it owns the HTTP client (and its
/// connection pool) and the slots that bound concurrent downloads. Cloning is
/// cheap and shares both.
#[derive(Debug, Clone)]
pub struct ImageFetcher {
    client: reqwest::Client,
    policy: ImageFetchPolicy,
    slots: Arc<Semaphore>,
}

impl ImageFetcher {
    /// Redirects followed before a fetch fails.
    pub const MAX_REDIRECTS: usize = 3;

    /// Largest number of http(s) downloads in progress at once across the server.
    ///
    /// Each download buffers at most `max_bytes`, so the limit also bounds the
    /// memory in-flight downloads hold (1.28 GB at the 20 MB default). Further
    /// fetches wait for a slot; data URLs never take one.
    pub const MAX_CONCURRENT_FETCHES: usize = 64;

    /// Builds a fetcher enforcing `policy`.
    ///
    /// # Errors
    ///
    /// Fails when the HTTP client cannot be initialized, for example when the
    /// TLS backend cannot load its configuration.
    pub fn new(policy: ImageFetchPolicy) -> anyhow::Result<Self> {
        let client = reqwest::Client::builder()
            // Redirects are followed by `download` so each hop is validated
            // before it is requested.
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .dns_resolver(Arc::new(AddressResolver {
                allow_private: policy.allow_private,
            }))
            .user_agent(concat!("uniserve/", env!("CARGO_PKG_VERSION")))
            .build()?;
        Ok(Self {
            client,
            policy,
            slots: Arc::new(Semaphore::new(Self::MAX_CONCURRENT_FETCHES)),
        })
    }

    /// Resolves one image reference: a `data:image/*;base64` URL or an http(s) URL.
    ///
    /// Network I/O runs on the async runtime; base64 decoding, header
    /// decoding, and content hashing run on the blocking pool. Dropping the
    /// returned future cancels an in-progress download.
    ///
    /// # Errors
    ///
    /// Returns the [`ImageFetchError`] describing the first rule the
    /// reference violates (see the module documentation).
    pub async fn fetch(&self, reference: String) -> Result<ImageInput, ImageFetchError> {
        if has_prefix_ignore_case(&reference, "data:") {
            let max_bytes = self.policy.max_bytes;
            return tokio::task::spawn_blocking(move || decode_data_url(&reference, max_bytes))
                .await
                .map_err(ImageFetchError::Task)?;
        }

        let url = Url::parse(&reference).map_err(ImageFetchError::InvalidUrl)?;
        // A reference that can never be fetched fails before it waits for a slot.
        self.check_destination(&url, false)?;

        // The semaphore is never closed, so acquisition only waits.
        let Ok(slot) = self.slots.acquire().await else {
            unreachable!("the image fetch semaphore is never closed");
        };
        let timeout = self.policy.timeout;
        let (body, declares_image) = tokio::time::timeout(timeout, self.download(url))
            .await
            .map_err(|_| ImageFetchError::Timeout { timeout })??;
        drop(slot);

        tokio::task::spawn_blocking(move || accept_body(body, declares_image))
            .await
            .map_err(ImageFetchError::Task)?
    }

    /// Resolves an ordered list of references concurrently, preserving order.
    ///
    /// The first failure cancels the remaining fetches and is reported with
    /// the index of its reference.
    ///
    /// # Errors
    ///
    /// Returns an [`ImageListError`] for the first reference that fails.
    pub async fn fetch_all(
        &self,
        references: Vec<String>,
    ) -> Result<Vec<ImageInput>, ImageListError> {
        futures::future::try_join_all(references.into_iter().enumerate().map(
            |(index, reference)| async move {
                self.fetch(reference)
                    .await
                    .map_err(|source| ImageListError { index, source })
            },
        ))
        .await
    }

    /// Checks the scheme of a URL about to be requested and, when the policy
    /// requires public destinations, its literal address.
    ///
    /// Host names are checked by [`AddressResolver`] when the client connects,
    /// against the exact addresses the connection uses.
    fn check_destination(&self, url: &Url, redirected: bool) -> Result<(), ImageFetchError> {
        if !matches!(url.scheme(), "http" | "https") {
            return Err(if redirected {
                ImageFetchError::UnsupportedRedirect
            } else {
                ImageFetchError::UnsupportedScheme
            });
        }

        if self.policy.allow_private {
            return Ok(());
        }
        let literal = match url.host() {
            Some(url::Host::Ipv4(address)) => Some(IpAddr::V4(address)),
            Some(url::Host::Ipv6(address)) => Some(IpAddr::V6(address)),
            Some(url::Host::Domain(_)) | None => None,
        };
        match literal {
            Some(address) if !is_public_address(address) => Err(non_public(redirected)),
            _ => Ok(()),
        }
    }

    /// Requests `url`, following validated redirects, and reads the final body.
    ///
    /// Returns the body and whether the response declared an `image/*` type.
    /// The caller has already checked `url` itself.
    async fn download(&self, mut url: Url) -> Result<(Bytes, bool), ImageFetchError> {
        let mut redirects = 0;
        loop {
            let redirected = redirects > 0;
            let response = self
                .client
                .get(url.clone())
                .header(ACCEPT, "image/*")
                .send()
                .await
                .map_err(|error| transport_error(error, redirected))?;

            let status = response.status();
            if is_followed_redirect(status) {
                if redirects == Self::MAX_REDIRECTS {
                    return Err(ImageFetchError::TooManyRedirects {
                        limit: Self::MAX_REDIRECTS,
                    });
                }
                let location = response
                    .headers()
                    .get(LOCATION)
                    .and_then(|value| value.to_str().ok())
                    .ok_or(ImageFetchError::InvalidRedirect)?;
                // A relative `Location` resolves against the URL that returned it.
                let target = url
                    .join(location)
                    .map_err(|_| ImageFetchError::InvalidRedirect)?;
                self.check_destination(&target, true)?;

                url = target;
                redirects += 1;
                continue;
            }
            if !status.is_success() {
                return Err(ImageFetchError::Status {
                    status: status.as_u16(),
                });
            }
            return self.read_body(response, redirected).await;
        }
    }

    /// Reads a success response body within the size limit.
    async fn read_body(
        &self,
        mut response: reqwest::Response,
        redirected: bool,
    ) -> Result<(Bytes, bool), ImageFetchError> {
        let limit = self.policy.max_bytes;
        // `content_length` is the length the HTTP framing will deliver; a
        // chunked body has none and is bounded only by the streamed count.
        let declared_length = response.content_length();
        if declared_length.is_some_and(|length| length > limit) {
            return Err(ImageFetchError::TooLarge { limit });
        }
        let declares_image = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(is_image_media_type);

        // The preallocation never exceeds `limit`, which was checked above.
        let mut body = BytesMut::with_capacity(declared_length.unwrap_or(0) as usize);
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|error| transport_error(error, redirected))?
        {
            if (body.len() + chunk.len()) as u64 > limit {
                return Err(ImageFetchError::TooLarge { limit });
            }
            body.extend_from_slice(&chunk);
        }
        Ok((body.freeze(), declares_image))
    }
}

/// Returns whether a fetch may connect to `address` under the public-only policy.
///
/// Rejected: unspecified, loopback, private-use (RFC 1918), shared (CGNAT),
/// link-local (which holds the 169.254.169.254 cloud metadata service),
/// unique-local and site-local IPv6 (which hold IPv6 metadata services such
/// as `fd00:ec2::254`), documentation, benchmarking, IETF protocol
/// assignment, discard-only, multicast, reserved, and broadcast ranges. IPv6
/// forms that carry an IPv4 destination (IPv4-mapped, IPv4-compatible,
/// IPv4-translated, NAT64, and 6to4) are judged by that IPv4 address.
pub fn is_public_address(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => is_public_ipv4(address),
        IpAddr::V6(address) => is_public_ipv6(address),
    }
}

/// IPv4 ranges outside public unicast space, as `(network, prefix length)`.
const NON_PUBLIC_IPV4: &[(Ipv4Addr, u8)] = &[
    // "This network", including the unspecified address 0.0.0.0, which
    // Linux connects to the local host.
    (Ipv4Addr::new(0, 0, 0, 0), 8),
    (Ipv4Addr::new(10, 0, 0, 0), 8),
    // Shared address space for carrier-grade NAT, which also holds the
    // 100.100.100.200 cloud metadata service.
    (Ipv4Addr::new(100, 64, 0, 0), 10),
    (Ipv4Addr::new(127, 0, 0, 0), 8),
    // Link-local, which holds the 169.254.169.254 and 169.254.170.2 cloud
    // metadata services.
    (Ipv4Addr::new(169, 254, 0, 0), 16),
    (Ipv4Addr::new(172, 16, 0, 0), 12),
    // IETF protocol assignments.
    (Ipv4Addr::new(192, 0, 0, 0), 24),
    // Documentation (TEST-NET-1).
    (Ipv4Addr::new(192, 0, 2, 0), 24),
    // Deprecated 6to4 relay anycast.
    (Ipv4Addr::new(192, 88, 99, 0), 24),
    (Ipv4Addr::new(192, 168, 0, 0), 16),
    // Benchmarking.
    (Ipv4Addr::new(198, 18, 0, 0), 15),
    // Documentation (TEST-NET-2 and TEST-NET-3).
    (Ipv4Addr::new(198, 51, 100, 0), 24),
    (Ipv4Addr::new(203, 0, 113, 0), 24),
    // Multicast.
    (Ipv4Addr::new(224, 0, 0, 0), 4),
    // Reserved, including the limited broadcast address 255.255.255.255.
    (Ipv4Addr::new(240, 0, 0, 0), 4),
];

/// IPv6 ranges outside public unicast space, as `(network, prefix length)`.
///
/// Ranges that embed an IPv4 destination are handled by `embedded_ipv4`
/// before this table is consulted.
const NON_PUBLIC_IPV6: &[(Ipv6Addr, u8)] = &[
    // Unspecified and loopback.
    (Ipv6Addr::UNSPECIFIED, 128),
    (Ipv6Addr::LOCALHOST, 128),
    // Local-use NAT64.
    (Ipv6Addr::new(0x64, 0xff9b, 1, 0, 0, 0, 0, 0), 48),
    // Discard-only.
    (Ipv6Addr::new(0x100, 0, 0, 0, 0, 0, 0, 0), 64),
    // IETF protocol assignments, including Teredo.
    (Ipv6Addr::new(0x2001, 0, 0, 0, 0, 0, 0, 0), 23),
    // Documentation.
    (Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 0), 32),
    (Ipv6Addr::new(0x3fff, 0, 0, 0, 0, 0, 0, 0), 20),
    // Segment-routing SIDs.
    (Ipv6Addr::new(0x5f00, 0, 0, 0, 0, 0, 0, 0), 16),
    // Unique-local.
    (Ipv6Addr::new(0xfc00, 0, 0, 0, 0, 0, 0, 0), 7),
    // Link-local and deprecated site-local.
    (Ipv6Addr::new(0xfe80, 0, 0, 0, 0, 0, 0, 0), 10),
    (Ipv6Addr::new(0xfec0, 0, 0, 0, 0, 0, 0, 0), 10),
    // Multicast.
    (Ipv6Addr::new(0xff00, 0, 0, 0, 0, 0, 0, 0), 8),
];

/// Returns whether an IPv4 address lies outside every [`NON_PUBLIC_IPV4`] range.
fn is_public_ipv4(address: Ipv4Addr) -> bool {
    let bits = u32::from(address);
    !NON_PUBLIC_IPV4.iter().any(|(network, length)| {
        let mask = u32::MAX.checked_shl(32 - u32::from(*length)).unwrap_or(0);
        bits & mask == u32::from(*network) & mask
    })
}

/// Returns whether an IPv6 address is public, judging embedded IPv4
/// destinations by their IPv4 address.
fn is_public_ipv6(address: Ipv6Addr) -> bool {
    if let Some(ipv4) = embedded_ipv4(address) {
        return is_public_ipv4(ipv4);
    }
    let bits = u128::from(address);
    !NON_PUBLIC_IPV6.iter().any(|(network, length)| {
        let mask = u128::MAX.checked_shl(128 - u32::from(*length)).unwrap_or(0);
        bits & mask == u128::from(*network) & mask
    })
}

/// Returns the IPv4 destination an IPv6 address carries, if any.
///
/// Covers IPv4-mapped (`::ffff:0:0/96`), IPv4-compatible (`::/96`, which also
/// holds `::` and `::1`), IPv4-translated (`::ffff:0:0:0/96`), NAT64
/// (`64:ff9b::/96`), and 6to4 (`2002::/16`, IPv4 in bits 16..48).
fn embedded_ipv4(address: Ipv6Addr) -> Option<Ipv4Addr> {
    let bits = u128::from(address);

    // The four /96 forms carry the IPv4 address in their low 32 bits; the
    // match is on the 96-bit prefix above it.
    const IPV4_COMPATIBLE: u128 = 0;
    const IPV4_MAPPED: u128 = 0xffff;
    const IPV4_TRANSLATED: u128 = 0xffff_0000;
    const NAT64: u128 = 0x0064_ff9b_0000_0000_0000_0000;
    if matches!(
        bits >> 32,
        IPV4_COMPATIBLE | IPV4_MAPPED | IPV4_TRANSLATED | NAT64
    ) {
        return Some(Ipv4Addr::from(bits as u32));
    }

    // 6to4 carries it in the 32 bits after the 16-bit `2002` prefix.
    if bits >> 112 == 0x2002 {
        return Some(Ipv4Addr::from((bits >> 80) as u32));
    }
    None
}

/// DNS resolution that admits only addresses the fetch policy allows.
///
/// The client connects to exactly the addresses this resolver returns, so the
/// address check and the connection share one resolution. A host with any
/// non-public address is refused outright rather than filtered, so a DNS
/// answer that mixes public and internal addresses cannot select either.
struct AddressResolver {
    allow_private: bool,
}

impl reqwest::dns::Resolve for AddressResolver {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let allow_private = self.allow_private;
        Box::pin(async move {
            // The client replaces the port on every returned address.
            let addresses = tokio::net::lookup_host((name.as_str(), 0))
                .await?
                .collect::<Vec<SocketAddr>>();
            if !allow_private
                && addresses
                    .iter()
                    .any(|address| !is_public_address(address.ip()))
            {
                return Err(Box::new(NonPublicHost) as Box<dyn std::error::Error + Send + Sync>);
            }
            Ok(Box::new(addresses.into_iter()) as reqwest::dns::Addrs)
        })
    }
}

/// Resolver refusal, recovered from the client's connect error chain by
/// `transport_error`.
#[derive(Debug, Error)]
#[error("host resolves to a non-public address")]
struct NonPublicHost;

/// Returns the non-public destination error for the initial URL or a redirect.
fn non_public(redirected: bool) -> ImageFetchError {
    if redirected {
        ImageFetchError::NonPublicRedirect
    } else {
        ImageFetchError::NonPublicAddress
    }
}

/// Maps an HTTP client error without exposing the URL it was requesting.
///
/// A resolver refusal becomes the non-public destination error; any other
/// failure is described by its innermost cause (for example "Connection
/// refused" or a TLS certificate error), which names no URL or address.
fn transport_error(error: reqwest::Error, redirected: bool) -> ImageFetchError {
    let error = error.without_url();
    let mut cause: &(dyn std::error::Error + 'static) = &error;
    loop {
        if cause.is::<NonPublicHost>() {
            return non_public(redirected);
        }
        match cause.source() {
            Some(next) => cause = next,
            None => break,
        }
    }
    ImageFetchError::Transport {
        reason: cause.to_string(),
    }
}

/// Returns whether a status is a redirect `download` follows with a GET.
fn is_followed_redirect(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::MOVED_PERMANENTLY
            | StatusCode::FOUND
            | StatusCode::SEE_OTHER
            | StatusCode::TEMPORARY_REDIRECT
            | StatusCode::PERMANENT_REDIRECT
    )
}

/// Returns whether a `Content-Type` value names an `image/<subtype>` media type.
fn is_image_media_type(value: &str) -> bool {
    let essence = value.split(';').next().unwrap_or_default().trim();
    essence
        .split_once('/')
        .is_some_and(|(kind, subtype)| kind.eq_ignore_ascii_case("image") && !subtype.is_empty())
}

/// Validates a fetched body as an image.
fn accept_body(body: Bytes, declares_image: bool) -> Result<ImageInput, ImageFetchError> {
    if !declares_image && image::guess_format(&body).is_err() {
        return Err(ImageFetchError::NotImage);
    }
    ImageInput::from_bytes(body)
}

/// Decodes and validates a `data:image/<subtype>[;parameters];base64,<payload>` URL.
///
/// The scheme, media type, and `base64` marker match case-insensitively. The
/// payload must be standard padded base64 without whitespace.
fn decode_data_url(reference: &str, max_bytes: u64) -> Result<ImageInput, ImageFetchError> {
    let (metadata, payload) = reference
        .split_once(',')
        .ok_or(ImageFetchError::InvalidDataUrl)?;
    let mut parameters = metadata["data:".len()..].split(';');
    let media_type = parameters.next().unwrap_or_default();
    let is_base64 = parameters
        .next_back()
        .is_some_and(|marker| marker.eq_ignore_ascii_case("base64"));
    if !is_image_media_type(media_type) || !is_base64 || payload.is_empty() {
        return Err(ImageFetchError::InvalidDataUrl);
    }

    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(ImageFetchError::InvalidBase64)?;
    if bytes.len() as u64 > max_bytes {
        return Err(ImageFetchError::TooLarge { limit: max_bytes });
    }
    ImageInput::from_bytes(bytes)
}

/// Returns whether `value` starts with the ASCII `prefix`, ignoring case.
fn has_prefix_ignore_case(value: &str, prefix: &str) -> bool {
    value
        .get(..prefix.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use image::ImageFormat;
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
    use tokio::net::{TcpListener, TcpStream};

    use super::*;

    /// Encodes a black `width` x `height` RGB image in `format`.
    fn encoded_image(format: ImageFormat, width: u32, height: u32) -> Vec<u8> {
        let mut bytes = Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(width, height)
            .write_to(&mut bytes, format)
            .unwrap();
        bytes.into_inner()
    }

    /// Returns the image data URL carrying `bytes`.
    fn data_url(bytes: &[u8]) -> String {
        format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(bytes)
        )
    }

    /// Default limits with loopback destinations allowed, which the local
    /// test server needs.
    fn loopback_policy() -> ImageFetchPolicy {
        ImageFetchPolicy {
            allow_private: true,
            ..ImageFetchPolicy::default()
        }
    }

    /// One scripted answer of [`TestServer`].
    enum Reply {
        /// Writes the bytes, then closes the connection.
        Close(Vec<u8>),
        /// Writes the bytes, then holds the connection open without sending more.
        Hold(Vec<u8>),
    }

    /// Complete HTTP/1.1 response with a `Content-Length` body.
    fn response(status: &str, headers: &[(&str, &str)], body: &[u8]) -> Vec<u8> {
        let mut head = format!(
            "HTTP/1.1 {status}\r\nConnection: close\r\nContent-Length: {}\r\n",
            body.len()
        );
        for (name, value) in headers {
            head.push_str(&format!("{name}: {value}\r\n"));
        }
        head.push_str("\r\n");

        let mut bytes = head.into_bytes();
        bytes.extend_from_slice(body);
        bytes
    }

    /// Complete `200 OK` response carrying `body` as `content_type`.
    fn ok(content_type: &str, body: &[u8]) -> Reply {
        Reply::Close(response("200 OK", &[("Content-Type", content_type)], body))
    }

    /// Redirect response with the given status line and `Location`.
    fn redirect(status: &str, location: &str) -> Reply {
        Reply::Close(response(status, &[("Location", location)], b""))
    }

    /// HTTP/1.1 server on an ephemeral 127.0.0.1 port. It answers each
    /// request path with a scripted reply and counts accepted connections.
    struct TestServer {
        address: SocketAddr,
        connections: Arc<AtomicUsize>,
    }

    impl TestServer {
        /// Starts serving on the current runtime; `route` maps a request path
        /// to its reply.
        async fn start(route: impl Fn(&str) -> Reply + Send + Sync + 'static) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let connections = Arc::new(AtomicUsize::new(0));
            let accepted = Arc::clone(&connections);
            let route = Arc::new(route);

            tokio::spawn(async move {
                while let Ok((mut socket, _)) = listener.accept().await {
                    accepted.fetch_add(1, Ordering::SeqCst);
                    let route = Arc::clone(&route);
                    tokio::spawn(async move {
                        let path = request_path(&mut socket).await;
                        match route(&path) {
                            Reply::Close(bytes) => {
                                let _ = socket.write_all(&bytes).await;
                                let _ = socket.shutdown().await;
                            }
                            Reply::Hold(bytes) => {
                                let _ = socket.write_all(&bytes).await;
                                std::future::pending::<()>().await;
                            }
                        }
                    });
                }
            });
            Self {
                address,
                connections,
            }
        }

        /// Returns the http URL of `path` on this server.
        fn url(&self, path: &str) -> String {
            format!("http://{}{path}", self.address)
        }

        /// Number of connections accepted so far.
        fn connections(&self) -> usize {
            self.connections.load(Ordering::SeqCst)
        }
    }

    /// Reads a request head and returns its request-target path.
    async fn request_path(socket: &mut TcpStream) -> String {
        let mut head = Vec::new();
        let mut buffer = [0_u8; 1024];
        while !head.windows(4).any(|window| window == b"\r\n\r\n") {
            match socket.read(&mut buffer).await {
                Ok(0) | Err(_) => break,
                Ok(read) => head.extend_from_slice(&buffer[..read]),
            }
        }
        String::from_utf8_lossy(&head)
            .split_whitespace()
            .nth(1)
            .unwrap_or("/")
            .to_owned()
    }

    /// Each row is an address and whether the public-only policy lets a fetch
    /// connect to it.
    #[test]
    fn public_addresses_are_distinguished_from_internal_ones() {
        let cases = [
            // Public unicast, including IPv6 forms that carry a public IPv4
            // destination, and the neighbors of internal ranges.
            ("8.8.8.8", true),
            ("93.184.216.34", true),
            ("100.63.255.255", true),
            ("100.128.0.0", true),
            ("172.15.255.255", true),
            ("172.32.0.0", true),
            ("2606:4700:4700::1111", true),
            ("2a00:1450:4001:80b::200e", true),
            ("::ffff:8.8.8.8", true),
            ("64:ff9b::808:808", true),
            ("2002:808:808::1", true),
            // Unspecified, "this network", and loopback.
            ("0.0.0.0", false),
            ("0.1.2.3", false),
            ("127.0.0.1", false),
            ("127.255.255.254", false),
            ("::", false),
            ("::1", false),
            // Private use and shared address space.
            ("10.0.0.1", false),
            ("172.16.0.1", false),
            ("172.31.255.255", false),
            ("192.168.1.1", false),
            ("100.64.0.1", false),
            // Link-local, unique-local, site-local, and the cloud metadata
            // services within them.
            ("169.254.0.1", false),
            ("169.254.169.254", false),
            ("169.254.170.2", false),
            ("100.100.100.200", false),
            ("fe80::1", false),
            ("fc00::1", false),
            ("fd00:ec2::254", false),
            ("fec0::1", false),
            // Protocol assignments, documentation, and benchmarking.
            ("192.0.0.8", false),
            ("192.0.2.1", false),
            ("198.18.0.1", false),
            ("198.19.255.255", false),
            ("198.51.100.1", false),
            ("203.0.113.1", false),
            ("2001::1", false),
            ("2001:db8::1", false),
            ("3fff::1", false),
            ("64:ff9b:1::1", false),
            ("100::1", false),
            // Multicast, reserved, and broadcast.
            ("224.0.0.1", false),
            ("239.255.255.250", false),
            ("240.0.0.1", false),
            ("255.255.255.255", false),
            ("ff02::1", false),
            // IPv6 forms that carry an internal IPv4 destination.
            ("::ffff:127.0.0.1", false),
            ("::ffff:169.254.169.254", false),
            ("::ffff:10.0.0.1", false),
            ("::127.0.0.1", false),
            ("::ffff:0:192.168.0.1", false),
            ("64:ff9b::a00:1", false),
            ("2002:7f00:1::", false),
            ("2002:a9fe:a9fe::1", false),
        ];
        for (address, public) in cases {
            let parsed = address.parse::<IpAddr>().unwrap();
            assert_eq!(is_public_address(parsed), public, "{address}");
        }
    }

    /// A data URL resolves to its decoded bytes and header dimensions, and
    /// the same bytes fetched over http resolve to an equal image with the
    /// same content hash, while different bytes hash differently.
    #[tokio::test]
    async fn fetched_and_inline_images_resolve_identically() {
        let png = encoded_image(ImageFormat::Png, 3, 2);
        let body = png.clone();
        let server = TestServer::start(move |_| ok("image/png", &body)).await;
        let fetcher = ImageFetcher::new(loopback_policy()).unwrap();

        let inline = fetcher.fetch(data_url(&png)).await.unwrap();
        assert_eq!(inline.bytes().as_ref(), png.as_slice());
        assert_eq!((inline.width(), inline.height()), (3, 2));

        let fetched = fetcher.fetch(server.url("/image.png")).await.unwrap();
        assert_eq!(fetched, inline);

        let other = encoded_image(ImageFormat::Png, 2, 3);
        let other = fetcher.fetch(data_url(&other)).await.unwrap();
        assert_ne!(other.hash(), inline.hash());
    }

    /// `fetch_all` returns images in reference order and names the index of a
    /// reference that fails.
    #[tokio::test]
    async fn image_lists_keep_their_order_and_name_the_failing_entry() {
        let wide = encoded_image(ImageFormat::Png, 4, 1);
        let tall = encoded_image(ImageFormat::Png, 1, 4);
        let server = TestServer::start(move |_| ok("image/png", &tall)).await;
        let fetcher = ImageFetcher::new(loopback_policy()).unwrap();

        let images = fetcher
            .fetch_all(vec![server.url("/tall.png"), data_url(&wide)])
            .await
            .unwrap();
        let dimensions = images
            .iter()
            .map(|image| (image.width(), image.height()))
            .collect::<Vec<_>>();
        assert_eq!(dimensions, [(1, 4), (4, 1)]);

        let error = fetcher
            .fetch_all(vec![data_url(&wide), "ftp://example.com/x.png".to_owned()])
            .await
            .unwrap_err();
        assert_eq!(error.index, 1);
        assert!(matches!(error.source, ImageFetchError::UnsupportedScheme));
    }

    /// A body served without an image `Content-Type` is accepted when its
    /// magic number identifies PNG, JPEG, GIF, WebP, or BMP, and its header
    /// dimensions are read.
    #[tokio::test]
    async fn images_are_recognized_by_magic_number() {
        let formats = [
            ("/png", ImageFormat::Png),
            ("/jpeg", ImageFormat::Jpeg),
            ("/gif", ImageFormat::Gif),
            ("/webp", ImageFormat::WebP),
            ("/bmp", ImageFormat::Bmp),
        ];
        let bodies = formats
            .iter()
            .map(|(path, format)| (path.to_string(), encoded_image(*format, 5, 3)))
            .collect::<HashMap<_, _>>();
        let server =
            TestServer::start(move |path| ok("application/octet-stream", &bodies[path])).await;
        let fetcher = ImageFetcher::new(loopback_policy()).unwrap();

        for (path, _) in formats {
            let image = fetcher.fetch(server.url(path)).await.unwrap();
            assert_eq!((image.width(), image.height()), (5, 3), "{path}");
        }
    }

    /// Responses that carry no usable image are refused: an unsuccessful
    /// status, a body neither declared nor recognizable as an image, and a
    /// declared image whose bytes do not decode.
    #[tokio::test]
    async fn responses_without_an_image_are_refused() {
        let server = TestServer::start(|path| match path {
            "/missing" => Reply::Close(response("404 Not Found", &[], b"")),
            "/page" => ok("text/html; charset=utf-8", b"<html><body>cat</body></html>"),
            "/unlabeled" => Reply::Close(response("200 OK", &[], b"plain text")),
            _ => ok("image/png", b"these bytes are not a PNG file"),
        })
        .await;
        let fetcher = ImageFetcher::new(loopback_policy()).unwrap();

        assert!(matches!(
            fetcher.fetch(server.url("/missing")).await,
            Err(ImageFetchError::Status { status: 404 })
        ));
        for path in ["/page", "/unlabeled"] {
            assert!(
                matches!(
                    fetcher.fetch(server.url(path)).await,
                    Err(ImageFetchError::NotImage)
                ),
                "{path}"
            );
        }
        assert!(matches!(
            fetcher.fetch(server.url("/broken")).await,
            Err(ImageFetchError::Undecodable(_))
        ));
    }

    /// The size limit admits an image of exactly the limit and refuses one
    /// byte more, whether the response declares its length, is delimited by
    /// connection close, or declares a `Content-Length` its chunked framing
    /// overrides.
    #[tokio::test]
    async fn images_over_the_size_limit_are_refused() {
        let png = encoded_image(ImageFormat::Png, 8, 8);
        let body = png.clone();
        let server = TestServer::start(move |path| {
            let head = match path {
                "/declared" => return ok("image/png", &body),
                "/close-delimited" => {
                    "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nConnection: close\r\n\r\n"
                        .to_owned()
                }
                _ => format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: 16\r\n\
                     Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n{:x}\r\n",
                    body.len()
                ),
            };
            let mut bytes = head.into_bytes();
            bytes.extend_from_slice(&body);
            if path != "/close-delimited" {
                bytes.extend_from_slice(b"\r\n0\r\n\r\n");
            }
            Reply::Close(bytes)
        })
        .await;
        let paths = ["/declared", "/close-delimited", "/chunked"];

        let exact = ImageFetcher::new(ImageFetchPolicy {
            max_bytes: png.len() as u64,
            ..loopback_policy()
        })
        .unwrap();
        for path in paths {
            let image = exact.fetch(server.url(path)).await.unwrap();
            assert_eq!(image.bytes().as_ref(), png.as_slice(), "{path}");
        }

        let limit = png.len() as u64 - 1;
        let smaller = ImageFetcher::new(ImageFetchPolicy {
            max_bytes: limit,
            ..loopback_policy()
        })
        .unwrap();
        for path in paths {
            let result = smaller.fetch(server.url(path)).await;
            assert!(
                matches!(result, Err(ImageFetchError::TooLarge { limit: refused }) if refused == limit),
                "{path}: {result:?}"
            );
        }
    }

    /// A declared length over the limit is refused as soon as the response
    /// head arrives. The server never sends the body, so a fetch that waited
    /// for it would end at the time limit instead.
    #[tokio::test]
    async fn declared_oversize_is_refused_before_the_body_arrives() {
        let server = TestServer::start(|_| {
            Reply::Hold(
                b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: 1000000\r\n\r\n"
                    .to_vec(),
            )
        })
        .await;
        let fetcher = ImageFetcher::new(ImageFetchPolicy {
            max_bytes: 1000,
            ..loopback_policy()
        })
        .unwrap();

        assert!(matches!(
            fetcher.fetch(server.url("/image.png")).await,
            Err(ImageFetchError::TooLarge { limit: 1000 })
        ));
    }

    /// A fetch that stalls before the response head or in the middle of the
    /// body fails at the time limit.
    #[tokio::test]
    async fn stalled_fetches_end_at_the_time_limit() {
        let server = TestServer::start(|path| match path {
            "/silent" => Reply::Hold(Vec::new()),
            _ => Reply::Hold(
                b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: 100\r\n\r\n\x89PNG"
                    .to_vec(),
            ),
        })
        .await;
        let timeout = Duration::from_millis(200);
        let fetcher = ImageFetcher::new(ImageFetchPolicy {
            timeout,
            ..loopback_policy()
        })
        .unwrap();

        for path in ["/silent", "/partial"] {
            let result = fetcher.fetch(server.url(path)).await;
            assert!(
                matches!(result, Err(ImageFetchError::Timeout { timeout: limit }) if limit == timeout),
                "{path}: {result:?}"
            );
        }
    }

    /// Only http(s) URLs and well-formed image data URLs are accepted. Other
    /// schemes, relative references, malformed data URLs, and data URLs over
    /// the size limit are refused.
    #[tokio::test]
    async fn unsupported_references_are_refused() {
        let fetcher = ImageFetcher::new(ImageFetchPolicy {
            max_bytes: 256,
            ..ImageFetchPolicy::default()
        })
        .unwrap();
        let fetch = |reference: &str| fetcher.fetch(reference.to_owned());

        for reference in [
            "ftp://example.com/image.png",
            "file:///etc/passwd",
            "gopher://example.com/image",
            "javascript:alert(1)",
            "blob:https://example.com/image",
        ] {
            assert!(
                matches!(
                    fetch(reference).await,
                    Err(ImageFetchError::UnsupportedScheme)
                ),
                "{reference}"
            );
        }
        assert!(matches!(
            fetch("images/cat.png").await,
            Err(ImageFetchError::InvalidUrl(_))
        ));

        for reference in [
            "data:text/plain;base64,aGVsbG8=",
            "data:image/png,rawbytes",
            "data:image/;base64,AAAA",
            "data:image/png;base64,",
            "data:image/png;base64",
        ] {
            assert!(
                matches!(fetch(reference).await, Err(ImageFetchError::InvalidDataUrl)),
                "{reference}"
            );
        }
        assert!(matches!(
            fetch("data:image/png;base64,not*base64").await,
            Err(ImageFetchError::InvalidBase64(_))
        ));

        // An uncompressed 16x16 BMP is 822 bytes, over the 256-byte limit.
        let large = encoded_image(ImageFormat::Bmp, 16, 16);
        assert!(matches!(
            fetch(&data_url(&large)).await,
            Err(ImageFetchError::TooLarge { limit: 256 })
        ));
    }

    /// Redirects, relative or absolute, are followed for up to three hops of
    /// any followed status. Every hop is validated again: a fourth redirect,
    /// a redirect to a scheme other than http(s), and a redirect without a
    /// location are refused.
    #[tokio::test]
    async fn redirects_are_limited_and_revalidated() {
        let png = encoded_image(ImageFormat::Png, 2, 2);
        let server = TestServer::start(move |path| match path {
            "/image.png" => ok("image/png", &png),
            "/hop1" => redirect("302 Found", "/image.png"),
            "/hop2" => redirect("301 Moved Permanently", "hop1"),
            "/hop3" => redirect("307 Temporary Redirect", "/hop2"),
            "/hop4" => redirect("308 Permanent Redirect", "/hop3"),
            "/to-ftp" => redirect("302 Found", "ftp://example.com/image.png"),
            "/to-file" => redirect("303 See Other", "file:///etc/passwd"),
            _ => Reply::Close(response("302 Found", &[], b"")),
        })
        .await;
        let fetcher = ImageFetcher::new(loopback_policy()).unwrap();

        let image = fetcher.fetch(server.url("/hop3")).await.unwrap();
        assert_eq!((image.width(), image.height()), (2, 2));

        assert!(matches!(
            fetcher.fetch(server.url("/hop4")).await,
            Err(ImageFetchError::TooManyRedirects { limit: 3 })
        ));
        for path in ["/to-ftp", "/to-file"] {
            assert!(
                matches!(
                    fetcher.fetch(server.url(path)).await,
                    Err(ImageFetchError::UnsupportedRedirect)
                ),
                "{path}"
            );
        }
        assert!(matches!(
            fetcher.fetch(server.url("/no-location")).await,
            Err(ImageFetchError::InvalidRedirect)
        ));
    }

    /// A connection failure on a redirect hop is reported without naming the
    /// redirect target, which the client never sent.
    #[tokio::test]
    async fn transport_failures_do_not_reveal_redirect_targets() {
        // Keep the destination reserved while refusing every response. Other
        // tests can bind ephemeral ports without receiving this redirect.
        let target_server = TestServer::start(|_| Reply::Close(Vec::new())).await;
        let target = target_server.url("/internal-target");
        let server = TestServer::start(move |_| redirect("302 Found", &target)).await;
        let fetcher = ImageFetcher::new(loopback_policy()).unwrap();

        let error = fetcher.fetch(server.url("/image.png")).await.unwrap_err();
        let message = error.to_string();
        assert!(
            matches!(error, ImageFetchError::Transport { .. }),
            "{message}"
        );
        assert!(
            !message.contains(&target_server.address.port().to_string()),
            "{message}"
        );
        assert!(!message.contains("internal-target"), "{message}");
    }

    /// Under the default policy, a URL whose host is or resolves to a
    /// loopback address is refused before any connection, however the host
    /// is spelled; allowing private destinations reaches the same server.
    #[tokio::test]
    async fn non_public_hosts_are_refused_before_connecting() {
        let png = encoded_image(ImageFormat::Png, 1, 1);
        let server = TestServer::start(move |_| ok("image/png", &png)).await;
        let port = server.address.port();
        let fetcher = ImageFetcher::new(ImageFetchPolicy::default()).unwrap();

        for host in [
            "127.0.0.1",
            "127.1",
            "2130706433",
            "0x7f.0.0.1",
            "0.0.0.0",
            "[::1]",
            "[::ffff:127.0.0.1]",
            "localhost",
        ] {
            let result = fetcher
                .fetch(format!("http://{host}:{port}/image.png"))
                .await;
            assert!(
                matches!(result, Err(ImageFetchError::NonPublicAddress)),
                "{host}: {result:?}"
            );
        }
        assert_eq!(server.connections(), 0);

        ImageFetcher::new(loopback_policy())
            .unwrap()
            .fetch(server.url("/image.png"))
            .await
            .unwrap();
        assert_eq!(server.connections(), 1);
    }
}
