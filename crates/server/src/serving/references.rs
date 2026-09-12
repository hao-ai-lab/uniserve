//! Ordered reference-media request descriptors. Media bytes are decoded only at admission.

use base64::Engine as _;
use serde::Deserialize;
use validator::ValidateUrl;

/// Largest encoded reference and aggregate bundle admitted by the HTTP contract.
pub const MAX_REFERENCE_BYTES: usize = 32 * 1024 * 1024;
pub const MAX_REFERENCE_BUNDLE_BYTES: usize = 96 * 1024 * 1024;

/// Semantic modality, independent of a container's optional soundtrack.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceKind {
    Image,
    Video,
    Audio,
}

/// Task annotation; image references do not imply exact FL2VA endpoint anchors.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceTask {
    Reference,
    FirstFrame,
    FirstLastFrame,
    ContinueScene,
    ContinueShot,
}

/// Position of this source within the conditioning task.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceRole {
    Reference,
    FirstFrame,
    LastFrame,
    Preceding,
}

/// Exactly one source representation. Base64 is raw standard base64, not a data URI.
#[derive(Clone, Deserialize, PartialEq, Eq)]
#[serde(
    tag = "type",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum ReferenceSource {
    Url(String),
    Base64(String),
}

// Reference URLs may include signed query strings; payloads and URLs must not enter logs.
impl std::fmt::Debug for ReferenceSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Url(_) => "Url(<redacted>)",
            Self::Base64(_) => "Base64(<redacted>)",
        })
    }
}

/// One source in caller order. Video soundtrack selection is explicit.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct VideoReference {
    #[serde(rename = "type")]
    pub kind: ReferenceKind,
    pub task: ReferenceTask,
    pub role: ReferenceRole,
    pub source: ReferenceSource,
    /// Required for video, forbidden for other modalities.
    pub include_audio: Option<bool>,
}

/// Validate descriptor bounds and role consistency before fetching or decoding any media.
/// Network address policy and decoded-media geometry remain admission obligations.
pub fn validate_references(references: &[VideoReference]) -> Result<(), &'static str> {
    if references.is_empty() {
        return Ok(());
    }
    if references.len() > 12 {
        return Err("references permits at most 12 ordered sources");
    }
    let mut counts = [0usize; 3];
    let mut bytes = 0usize;
    for reference in references {
        let index = match reference.kind {
            ReferenceKind::Image => 0,
            ReferenceKind::Video => 1,
            ReferenceKind::Audio => 2,
        };
        counts[index] += 1;
        if (reference.kind == ReferenceKind::Video) != reference.include_audio.is_some() {
            return Err("include_audio is required for video and forbidden for image/audio");
        }
        let valid_role = match reference.task {
            ReferenceTask::Reference => reference.role == ReferenceRole::Reference,
            ReferenceTask::FirstFrame => {
                reference.kind == ReferenceKind::Image
                    && reference.role == ReferenceRole::FirstFrame
            }
            ReferenceTask::FirstLastFrame => {
                reference.kind == ReferenceKind::Image
                    && matches!(
                        reference.role,
                        ReferenceRole::FirstFrame | ReferenceRole::LastFrame
                    )
            }
            ReferenceTask::ContinueScene | ReferenceTask::ContinueShot => {
                reference.kind == ReferenceKind::Video && reference.role == ReferenceRole::Preceding
            }
        };
        if !valid_role {
            return Err("reference modality, task and role disagree");
        }
        match &reference.source {
            ReferenceSource::Url(url) => {
                // This is syntax admission only, not SSRF authorization. The fetcher must
                // resolve and validate every destination and enforce streaming byte limits.
                if url.len() > 8192
                    || !url.validate_url()
                    || !(url.starts_with("https://") || url.starts_with("http://"))
                    || url.chars().any(char::is_whitespace)
                {
                    return Err("reference URL must be bounded HTTP or HTTPS without whitespace");
                }
            }
            ReferenceSource::Base64(encoded) => {
                if encoded.is_empty() || encoded.len() > MAX_REFERENCE_BYTES.div_ceil(3) * 4 {
                    return Err("reference base64 source exceeds its encoded byte bound");
                }
                let decoded = base64::engine::general_purpose::STANDARD
                    .decode(encoded)
                    .map_err(|_| "reference source contains invalid base64")?;
                if decoded.len() > MAX_REFERENCE_BYTES {
                    return Err("reference source exceeds its byte bound");
                }
                bytes += decoded.len();
            }
        }
    }
    if counts[0] > 9 || counts[1] > 3 || counts[2] > 3 || counts[0] + counts[1] == 0 {
        return Err("references permits 9 images, 3 videos, 3 audio and requires a visual source");
    }
    if bytes > MAX_REFERENCE_BUNDLE_BYTES {
        return Err("reference bundle exceeds its byte bound");
    }
    Ok(())
}
