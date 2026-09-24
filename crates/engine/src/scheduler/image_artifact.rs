//! Validation and metadata extraction for PNG artifacts returned by workers.
//!
//! Workers deliver a generated image as a base64-encoded PNG. Result
//! validation (`generation::validate_generation_result`) reads only the IHDR
//! header to check the requested dimensions; the output path
//! (`image_done_event`) decodes the whole image once before publishing it.

use std::io::Cursor;

use base64::Engine as _;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Verified dimensions and encoded size of a PNG image.
pub(crate) struct PngInfo {
    pub(crate) height: u32,
    pub(crate) width: u32,
    /// Size of the decoded PNG file in bytes, not of its base64 text.
    pub(crate) bytes: u64,
}

/// Parses `(height, width)` from a base64 PNG's IHDR header, decoding only
/// the base64 prefix.
///
/// Result validation checks artifact dimensions for every image-decoding call;
/// decoding a multi-megabyte frame there is far more expensive than reading
/// the header, which carries the dimensions in its first 24 bytes. The frame
/// itself is not checked. Returns `None` when the prefix is not valid base64,
/// is shorter than the header, lacks the PNG signature, does not start with
/// an IHDR chunk, or declares a zero dimension.
pub(crate) fn png_artifact_dims_b64(pixels_png_b64: &str) -> Option<(u32, u32)> {
    // Signature (8 bytes), chunk length (4), `IHDR` (4), width (4), and height
    // (4) fill the first 24 bytes; 44 base64 characters decode to 33 bytes.
    // Rounding down to a multiple of four keeps the prefix on a base64 group
    // boundary so it decodes without padding.
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
///
/// Fully decodes the first frame. `expected_hw`, when set, is the required
/// `(height, width)`. Returns `None` when the payload is not valid base64, the
/// PNG header cannot be read, a dimension is zero, the dimensions differ from
/// `expected_hw`, the output buffer size cannot be computed, the frame fails
/// to decode, or the decoded frame's dimensions differ from the header's.
pub(crate) fn validate_png_artifact(
    pixels_png_b64: &str,
    expected_hw: Option<(u32, u32)>,
) -> Option<PngInfo> {
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
    Some(PngInfo {
        height,
        width,
        bytes: bytes.len() as u64,
    })
}
