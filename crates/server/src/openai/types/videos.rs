use serde::Deserialize;
use validator::Validate;

use super::common::Normalizable;

fn default_video_seconds() -> f64 {
    5.0
}

/// MiniMax H3 T2VA request. The deployment owns the canvas and capacity while
/// duration is resolved into immutable media geometry at admission.
#[derive(Debug, Clone, Deserialize, PartialEq, Validate)]
#[serde(deny_unknown_fields)]
pub struct VideoGenerationRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub seed: u64,
    #[serde(default = "default_video_seconds")]
    pub seconds: f64,
}

impl Normalizable for VideoGenerationRequest {}
