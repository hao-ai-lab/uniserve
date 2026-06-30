//! Shared default values for native generation sampling / image parameters.
//!
//! Model-specific image defaults live in native profile manifests. This module
//! only contains shared request defaults that are not model profile data.

// --- Shared sampling / generation defaults ------------------------------------

/// Default upper bound on generated tokens when a request omits `max_tokens`.
pub const DEFAULT_MAX_TOKENS: usize = 32768;
/// Default number of diffusion sampling steps for image generation.
pub const DEFAULT_STEPS: u16 = 50;
/// Default classifier-free-guidance scale applied to the text conditioning.
pub const DEFAULT_CFG_TEXT_SCALE: f32 = 4.0;
/// Default classifier-free-guidance scale applied to the image conditioning.
pub const DEFAULT_CFG_IMG_SCALE: f32 = 1.0;
/// Default minimum value for CFG renormalization.
pub const DEFAULT_CFG_RENORM_MIN: f32 = 0.0;
/// Default `(start, end)` fraction of the schedule over which CFG is applied.
pub const DEFAULT_CFG_INTERVAL: (f32, f32) = (0.0, 1.0);
/// Default number of images produced per request.
pub const DEFAULT_MAX_IMAGES: u16 = 1;
/// Default sampling temperature (0.0 = greedy / deterministic).
pub const DEFAULT_TEMPERATURE: f32 = 0.0;
/// Default nucleus-sampling probability mass (1.0 = disabled).
pub const DEFAULT_TOP_P: f32 = 1.0;
/// Default top-k cutoff (0 = disabled, consider the full vocabulary).
pub const DEFAULT_TOP_K: u32 = 0;
