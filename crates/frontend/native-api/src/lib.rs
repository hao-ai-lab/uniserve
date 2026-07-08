//! UniServe native image/interleaved generation API conversion.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
pub mod builder;
pub mod defaults;
pub mod events;
pub mod profiles;
pub mod resolution;
pub mod schema;

pub use builder::NativeRequestBuilder;
pub use profiles::{
    NativeDelimitedText, NativeModelProfile, NativeOutputFilter, mode_name, resolve_native_profile,
    resolve_native_profile_for_model,
};
pub use schema::{NativeGenerateBody, NativeImageBody};
