//! Validated encoded input images.

use std::io::Cursor;

use bytes::Bytes;
use sha2::{Digest as _, Sha256};

use super::ImageFetchError;

/// One encoded input image with the facts preprocessing needs before pixels are decoded.
///
/// The bytes are the complete image file exactly as received (PNG, JPEG, GIF,
/// WebP, or BMP). The dimensions come from the file header alone, so
/// construction never decodes pixels; the worker decodes them. Every value is
/// derived from `bytes` at construction, so the fields cannot disagree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageInput {
    bytes: Bytes,
    width: u32,
    height: u32,
    hash: u64,
}

impl ImageInput {
    /// Validates an encoded image file and reads its header dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`ImageFetchError::Undecodable`] when the bytes match no
    /// supported image format or their header cannot be read.
    pub fn from_bytes(bytes: impl Into<Bytes>) -> Result<Self, ImageFetchError> {
        let bytes = bytes.into();

        // The format comes from the magic number, not from any declared media
        // type, so a mislabeled file is read as what it actually is.
        let (width, height) = image::ImageReader::new(Cursor::new(&bytes[..]))
            .with_guessed_format()
            .map_err(|error| ImageFetchError::Undecodable(image::ImageError::IoError(error)))?
            .into_dimensions()
            .map_err(ImageFetchError::Undecodable)?;

        let hash = content_hash(&bytes);
        Ok(Self {
            bytes,
            width,
            height,
            hash,
        })
    }

    /// Encoded image file bytes.
    pub fn bytes(&self) -> &Bytes {
        &self.bytes
    }

    /// Width in pixels, from the file header.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels, from the file header.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Stable content identity of the encoded bytes.
    ///
    /// Equal bytes always give equal hashes, whether the image arrived inline
    /// or from a URL, so the value serves as the encoder-cache identity
    /// `uniserve_core::ImageInput::hash` starts from.
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

/// Returns the first eight bytes, little-endian, of the SHA-256 digest of `bytes`.
///
/// The encoder cache is shared across requests without an isolation key, so a
/// collision would serve one client's image features for another client's
/// image. A cryptographic digest keeps a crafted second image from reaching
/// the same 64-bit identity, which a non-cryptographic hash does not.
fn content_hash(bytes: &[u8]) -> u64 {
    let digest = Sha256::digest(bytes);
    let mut prefix = [0_u8; 8];
    prefix.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(prefix)
}
