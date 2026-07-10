//! UniServe native wire schema and semantic conversion.

mod convert;
pub mod events;
pub mod schema;

pub use convert::{NativeAdapterError, NativeRequestResolution, into_serve_request};
pub use schema::{NativeContextRole, NativeContextSegment, NativeGenerateBody, NativeImageBody};
