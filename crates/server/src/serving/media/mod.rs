//! Input media acquisition shared by every request path that accepts images.
//!
//! A request names an image by reference: an inline `data:image/*;base64`
//! URL or an `http(s)` URL. [`ImageFetcher`] resolves references into
//! [`ImageInput`] values (the encoded file bytes, their header dimensions, and
//! a content hash) before model preprocessing runs, so network I/O stays on
//! the async runtime and never enters the blocking preprocessing pool. Chat
//! `image_url` parts and programmatic prompts reach model preprocessing in
//! this same form, whatever their source.

mod fetch;
mod image;

pub use fetch::{
    ImageFetchError, ImageFetchPolicy, ImageFetcher, ImageListError, is_public_address,
};
pub use image::ImageInput;
