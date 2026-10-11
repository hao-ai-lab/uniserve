//! Metadata extraction for PNG artifacts returned by workers.
//!
//! Workers deliver a generated image as a base64-encoded PNG and own its
//! encoding. The output path (`image_done_event`) reads the artifact's size
//! and dimensions from the encoded text without decoding the image.

use base64::Engine as _;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Dimensions and encoded size of a PNG image.
pub(crate) struct PngInfo {
    pub(crate) height: u32,
    pub(crate) width: u32,
    /// Size of the decoded PNG file in bytes, not of its base64 text.
    pub(crate) bytes: u64,
}

/// Reads a base64 PNG's dimensions from its IHDR header and its decoded size
/// from the base64 length, decoding only the base64 prefix.
///
/// The header carries the dimensions in its first 24 bytes, so a
/// multi-megabyte frame is not decoded. Returns `None` when the prefix is not
/// valid base64, is shorter than the header, lacks the PNG signature, does
/// not start with an IHDR chunk, or declares a zero dimension.
pub(crate) fn png_artifact_info(pixels_png_b64: &str) -> Option<PngInfo> {
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
    if width == 0 || height == 0 {
        return None;
    }

    // Four base64 characters carry three bytes; padding characters carry none.
    let padding = pixels_png_b64
        .bytes()
        .rev()
        .take(2)
        .filter(|&byte| byte == b'=')
        .count();
    let bytes = (pixels_png_b64.len() / 4 * 3).saturating_sub(padding) as u64;
    Some(PngInfo {
        height,
        width,
        bytes,
    })
}
