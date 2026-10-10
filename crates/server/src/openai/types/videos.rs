//! MiniMax-H3 video-generation request schema.
//!
//! The body is the official MiniMax / SGLang request form: a `task`, its
//! ordered `conditions` and the requested `target`. The same names arrive as
//! JSON or as multipart text fields; the HTTP `VideoBody` extractor parses
//! both into [`VideoGenerationRequest`], and `deny_unknown_fields` rejects
//! every field not listed here, at every level.

use serde::Deserialize;
use validator::Validate;

use super::common::Normalizable;
use uniserve_core::VideoTask;

use crate::serving::video::plan::{ConditionRole, ConditionType};

/// The seed a request without one uses, as the official reference does.
pub const DEFAULT_VIDEO_SEED: u64 = 42;

/// One video, audio or image condition of a request, in request order.
///
/// The order is semantic: the conditioner presents references in it and
/// their ordinal labels name them.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VideoCondition {
    /// `image`, `video`, `video_audio` (a video that must have a soundtrack)
    /// or `audio`.
    #[serde(rename = "type")]
    pub condition_type: ConditionType,
    /// `data:`, `file://` or `http(s)://` location of the media.
    pub uri: String,
    /// `keyframe` (an image anchoring a generated frame) or `reference`.
    pub role: ConditionRole,
    /// The generated frame a keyframe anchors: 0 or -1.
    #[serde(default)]
    pub frame_index: Option<i64>,
    /// Offset into a video reference, seconds; its soundtrack follows it.
    #[serde(default)]
    pub start_time_seconds: Option<f64>,
}

/// The requested output; the canvas and the duration come only from here.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VideoTarget {
    /// Short edge of the canvas in pixels; only 768 is served.
    pub short_edge: u32,
    /// `auto` or `W:H`.
    pub aspect_ratio: String,
    /// Duration in seconds; a reference-conditioned request whose only
    /// audio-bearing reference sets its duration may omit it.
    #[serde(default)]
    pub duration_seconds: Option<f64>,
}

/// MiniMax-H3 video-and-audio generation request for the `/v1/videos`
/// routes. The served checkpoint fixes the schedule, and every request
/// generates one video.
#[derive(Debug, Clone, Deserialize, PartialEq, Validate)]
#[serde(deny_unknown_fields)]
pub struct VideoGenerationRequest {
    /// Served model name.
    pub model: String,
    /// Prompt; the conditioner presents it after the conditions.
    pub prompt: String,
    /// `t2va`, `fl2va` or `ref2va`; one the served denoiser serves.
    pub task: VideoTask,
    /// The task's conditions, in request order.
    #[serde(default)]
    pub conditions: Vec<VideoCondition>,
    /// The requested output.
    pub target: VideoTarget,
    /// Video-generation random seed.
    #[serde(default = "default_seed")]
    pub seed: u64,
}

const fn default_seed() -> u64 {
    DEFAULT_VIDEO_SEED
}

impl Normalizable for VideoGenerationRequest {}
