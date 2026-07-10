use std::io::Cursor;

use base64::Engine as _;
use sha2::{Digest as _, Sha256};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ImageArtifactMetadata {
    pub(crate) height: u32,
    pub(crate) width: u32,
    pub(crate) bytes: u64,
    pub(crate) sha256: String,
}

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
    let sha256 = Sha256::digest(&bytes);
    Some(ImageArtifactMetadata {
        height,
        width,
        bytes: bytes.len() as u64,
        sha256: format!("{sha256:x}"),
    })
}
