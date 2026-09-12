//! Bounded inline image-reference admission. Payloads never enter diagnostics.

use base64::Engine as _;
use serde::Deserialize;
use std::io::Cursor;
use uniserve_core::ImageReference;

/// Maximum compressed source size; decoded storage is bounded independently.
pub const MAX_REFERENCE_BYTES: usize = 32 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceKind {
    Image,
    Video,
    Audio,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceTask {
    Reference,
    FirstFrame,
    FirstLastFrame,
    ContinueScene,
    ContinueShot,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceRole {
    Reference,
    FirstFrame,
    LastFrame,
    Preceding,
}

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

impl std::fmt::Debug for ReferenceSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ReferenceSource(<redacted>)")
    }
}

/// One ordered source. Only an inline image with reference task/role is supported.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct VideoReference {
    #[serde(rename = "type")]
    pub kind: ReferenceKind,
    pub task: ReferenceTask,
    pub role: ReferenceRole,
    pub source: ReferenceSource,
    pub include_audio: Option<bool>,
}

/// Validate the supported descriptor before model admission or image decoding.
pub fn validate_references(references: &[VideoReference]) -> Result<(), &'static str> {
    if references.len() > 1 {
        return Err("references permits at most 1 image");
    }
    for reference in references {
        if reference.kind != ReferenceKind::Image {
            return Err("references supports only image sources");
        }
        if reference.task != ReferenceTask::Reference || reference.role != ReferenceRole::Reference
        {
            return Err("references requires task=reference and role=reference");
        }
        if reference.include_audio.is_some() {
            return Err("references image forbids include_audio");
        }
        let ReferenceSource::Base64(encoded) = &reference.source else {
            return Err("references requires inline base64 PNG/JPEG; URLs are forbidden");
        };
        if encoded.is_empty() || encoded.len() > MAX_REFERENCE_BYTES.div_ceil(3) * 4 {
            return Err("references base64 exceeds the 32 MiB source bound");
        }
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|_| "references requires valid standard base64")?;
        if bytes.len() > MAX_REFERENCE_BYTES {
            return Err("references exceeds the 32 MiB source bound");
        }
    }
    Ok(())
}

/// Decode only after capability admission, checking geometry before raster allocation.
pub fn admit_references(
    references: &[VideoReference],
    capable: bool,
) -> Result<Option<ImageReference>, &'static str> {
    if references.is_empty() {
        return Ok(None);
    }
    if !capable {
        return Err("references requires a model contract declaring max=1, kinds=[image]");
    }
    validate_references(references)?;
    let ReferenceSource::Base64(encoded) = &references[0].source else {
        unreachable!()
    };
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|_| "references requires valid standard base64")?;
    let format = image::guess_format(&bytes).map_err(|_| "references requires PNG or JPEG")?;
    if !matches!(format, image::ImageFormat::Png | image::ImageFormat::Jpeg) {
        return Err("references requires PNG or JPEG");
    }
    let (width, height) = image::ImageReader::with_format(Cursor::new(&bytes), format)
        .into_dimensions()
        .map_err(|_| "references image header is invalid")?;
    if width == 0
        || height == 0
        || width > 4096
        || height > 4096
        || width % 32 != 0
        || height % 32 != 0
    {
        return Err("references image dimensions must be multiples of 32 in 32..=4096");
    }
    let mut reader = image::ImageReader::with_format(Cursor::new(&bytes), format);
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(4096);
    limits.max_image_height = Some(4096);
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    let pixels = reader
        .decode()
        .map_err(|_| "references image is invalid or exceeds decode bounds")?
        .to_rgb8()
        .into_raw();
    Ok(Some(ImageReference {
        width,
        height,
        pixels,
    }))
}
