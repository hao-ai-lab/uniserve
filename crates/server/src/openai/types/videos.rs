use serde::Deserialize;
use validator::Validate;

use super::common::Normalizable;

/// Fixed MiniMax H3 T2VA request. The deployment profile owns shape, duration,
/// sampling ladder, codecs, and output count, so they are not request fields.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Validate)]
#[serde(deny_unknown_fields)]
pub struct VideoGenerationRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub seed: u64,
}

impl Normalizable for VideoGenerationRequest {}
