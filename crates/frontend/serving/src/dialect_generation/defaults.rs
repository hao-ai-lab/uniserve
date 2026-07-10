//! Shared default values for dialect generation sampling and image parameters.
//!
//! Model-specific image defaults live in generation profile manifests. This module
//! only contains shared request defaults that are not model profile data.

/// Default number of diffusion sampling steps for image generation.
pub(super) const DEFAULT_STEPS: u16 = 50;
/// Default sampling temperature (0.0 = greedy / deterministic).
pub(super) const DEFAULT_TEMPERATURE: f32 = 0.0;
/// Default nucleus-sampling probability mass (1.0 = disabled).
pub(super) const DEFAULT_TOP_P: f32 = 1.0;
/// Default top-k cutoff (0 = disabled, consider the full vocabulary).
pub(super) const DEFAULT_TOP_K: u32 = 0;
