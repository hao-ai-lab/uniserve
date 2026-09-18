//! Validation and metadata extraction for PNG artifacts returned by workers.

use std::fmt::Write as _;
use std::io::Cursor;

use base64::Engine as _;
use sha2::{Digest as _, Sha256};

#[derive(Debug, Clone, PartialEq, Eq)]
/// Verified dimensions, size, and digest of a PNG artifact.
pub(crate) struct ImageArtifactMetadata {
    pub(crate) height: u32,
    pub(crate) width: u32,
    pub(crate) bytes: u64,
    pub(crate) sha256: String,
}

/// Parses image dimensions from a base64 PNG's IHDR header, decoding only the
/// base64 prefix. The response path validates artifact dimensions per
/// final image call; decoding the entire multi-megabyte frame there costs hundreds
/// of milliseconds per image, while the header carries the dimensions in the
/// first 24 bytes.
pub(crate) fn png_artifact_dims_b64(pixels_png_b64: &str) -> Option<(u32, u32)> {
    let prefix = &pixels_png_b64.as_bytes()[..pixels_png_b64.len().min(44) & !3];
    let head = base64::engine::general_purpose::STANDARD
        .decode(prefix)
        .ok()?;
    if head.len() < 24 || &head[..8] != b"\x89PNG\r\n\x1a\n" || &head[12..16] != b"IHDR" {
        return None;
    }
    let width = u32::from_be_bytes([head[16], head[17], head[18], head[19]]);
    let height = u32::from_be_bytes([head[20], head[21], head[22], head[23]]);
    (width != 0 && height != 0).then_some((height, width))
}

/// Verifies PNG framing and declared metadata, then returns response metadata.
pub(crate) fn validate_png_artifact(
    pixels_png_b64: &str,
    expected_hw: Option<(u32, u32)>,
) -> Option<ImageArtifactMetadata> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(pixels_png_b64.as_bytes())
        .ok()?;
    let decoder = png::Decoder::new(Cursor::new(bytes.as_slice()));
    let mut reader = decoder.read_info().ok()?;
    let (width, height) = (reader.info().width, reader.info().height);
    if width == 0 || height == 0 || expected_hw.is_some_and(|expected| expected != (height, width))
    {
        return None;
    }
    let mut decoded = vec![0; reader.output_buffer_size()?];
    let output = reader.next_frame(&mut decoded).ok()?;
    if (output.height, output.width) != (height, width) {
        return None;
    }
    let mut sha256 = String::with_capacity(64);
    for byte in Sha256::digest(&bytes) {
        write!(&mut sha256, "{byte:02x}").expect("writing to a String is infallible");
    }
    Some(ImageArtifactMetadata {
        height,
        width,
        bytes: bytes.len() as u64,
        sha256,
    })
}
