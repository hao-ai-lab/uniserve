//! OpenAI-compatible video-generation request schema.

use serde::Deserialize;
use validator::Validate;

use super::common::Normalizable;

/// Returns the default video duration.
fn default_video_seconds() -> f64 {
    5.0
}

/// MiniMax H3 T2VA request. The configuration owns the canvas and capacity while
/// duration is resolved into immutable media geometry at admission.
#[derive(Debug, Clone, Deserialize, PartialEq, Validate)]
#[serde(deny_unknown_fields)]
pub struct VideoGenerationRequest {
    /// Served model name.
    pub model: String,
    /// Text prompt describing the requested video.
    pub prompt: String,
    /// Video-generation random seed.
    #[serde(default)]
    pub seed: u64,
    /// Requested duration in seconds.
    #[serde(default = "default_video_seconds")]
    pub seconds: f64,
    /// Scheduler grid points, including the terminal point; defaults to the checkpoint recipe.
    #[serde(default)]
    #[validate(range(min = 2, max = 1000))]
    pub steps: Option<u32>,
}

impl Normalizable for VideoGenerationRequest {}
