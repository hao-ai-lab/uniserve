//! OpenAI-compatible video-generation request schema.

use serde::Deserialize;
use validator::Validate;

use super::common::Normalizable;

/// MiniMax H3 text-to-video-and-audio request for the `/v1/videos` routes.
///
/// The request carries no canvas, frame-rate, or step controls.
/// `InputProcessor::video_sampling` resolves `seconds` into an aligned frame
/// count at 24 fps, takes the step count from the model configuration, and
/// rejects durations the model cannot serve.
///
/// The HTTP `VideoBody` extractor deserializes it from a JSON body or from
/// multipart fields, and `deny_unknown_fields` rejects any other field; the
/// Dynamo worker constructs it from its own request type.
#[derive(Debug, Clone, Deserialize, PartialEq, Validate)]
#[serde(deny_unknown_fields)]
pub struct VideoGenerationRequest {
    /// Served model name.
    pub model: String,
    /// Text prompt describing the requested video.
    pub prompt: String,
    /// Video-generation random seed; `0` when omitted.
    #[serde(default)]
    pub seed: u64,
    /// Requested duration in seconds. When omitted, `video_sampling` uses the
    /// lesser of 5 seconds and the model's configured maximum.
    #[serde(default)]
    pub seconds: Option<f64>,
}

impl Normalizable for VideoGenerationRequest {}
