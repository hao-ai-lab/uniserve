//! Default values for native generation sampling / image parameters.
//!
//! These constants are the single source of truth for every "magic number"
//! used when a request omits a field; consumers (`builder.rs`, `profiles.rs`)
//! reference them by name rather than re-typing literals, so a value only ever
//! needs to change here. Grouped by scope: shared text/image defaults first,
//! then the per-model (Bagel, SenseNova) overrides.

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

// --- Bagel model defaults -----------------------------------------------------

/// Default output width (pixels) for the Bagel image model.
pub const BAGEL_DEFAULT_WIDTH: u32 = 512;
/// Default output height (pixels) for the Bagel image model.
pub const BAGEL_DEFAULT_HEIGHT: u32 = 512;
/// Default diffusion timestep-shift factor for Bagel.
pub const BAGEL_DEFAULT_TIMESTEP_SHIFT: f32 = 1.0;
/// Default CFG renormalization strategy for Bagel.
pub const BAGEL_DEFAULT_CFG_RENORM_TYPE: &str = "global";
/// Maximum number of images Bagel will accept per request.
pub const BAGEL_MAX_IMAGES: u16 = 16;

// --- SenseNova model defaults -------------------------------------------------

/// Default aspect-ratio preset for the SenseNova image model.
pub const SENSENOVA_DEFAULT_RESOLUTION: &str = "16:9";
/// Default output width (pixels) for SenseNova.
pub const SENSENOVA_DEFAULT_WIDTH: u32 = 2048;
/// Default output height (pixels) for SenseNova.
pub const SENSENOVA_DEFAULT_HEIGHT: u32 = 1152;
/// Default RNG seed used when a SenseNova request omits `seed`.
pub const SENSENOVA_DEFAULT_SEED: u64 = 42;
/// Default diffusion timestep-shift factor for SenseNova.
pub const SENSENOVA_DEFAULT_TIMESTEP_SHIFT: f32 = 3.0;
/// Default CFG renormalization strategy for SenseNova.
pub const SENSENOVA_DEFAULT_CFG_RENORM_TYPE: &str = "none";
/// Maximum number of images SenseNova will accept per request.
pub const SENSENOVA_MAX_IMAGES: u16 = 10;
