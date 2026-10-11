//! MiniMax-H3 request planning.
//!
//! [`plan_request`] resolves everything a request implies before any pixel is
//! decoded: the generated canvas and frame counts, how every condition is
//! prepared (keyframe fit, reference image size, 24 fps reference clip,
//! 32 kHz soundtrack), what the Qwen3-VL conditioner sees of it (vision
//! grids, 2 fps samples and block timestamps), and the units the worker
//! encodes each condition in. [`check_request`] applies the subset of rules
//! that needs no media, so a malformed request is rejected before its media
//! are fetched.
//!
//! The arithmetic reproduces the diffusers MiniMax-H3 pipeline and the
//! checkpoint's Qwen3-VL processor operation for operation, in `f64` where
//! they compute in Python floats, so that every rounding lands on the same
//! side; it is tested against
//! `tests/python/fixtures/minimax_h3_plan.json`, generated from that
//! pipeline. Rules the reference leaves open are fixed here:
//!
//! - `start_time_seconds` skips the first `floor(s * 24 + 0.5)` frames of a
//!   reference video's 24 fps timeline and the first `floor(s * rate + 0.5)`
//!   samples of its soundtrack;
//! - a reference video needs at least 22 frames (one `17n + 5` VAE window)
//!   on its 24 fps timeline after the start offset;
//! - `ref2va` keyframes are fitted to the target canvas like `fl2va` ones
//!   (the first stretched, a second cover-cropped) and do not enter the
//!   conditioner;
//! - without `target.duration_seconds`, a `ref2va` request whose references
//!   carry exactly one soundtrack lasts as long as that soundtrack after its
//!   start offset.

use std::path::Path;

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use uniserve_core::{
    Canvas, ConditionMedia, ConditionVision, ImageFit, MediaLocator, VideoCondition, VideoTask,
};

use super::probe::{AudioFacts, ImageFacts, MediaFacts, VideoFacts};
use super::{RequestField, VideoInputError};
use crate::profile::video::{VideoRaster, VideoResolution};
use crate::serving::{VIDEO_FPS, video_frame_count};

/// Short edge of the adapt_shape_v1 canvas rule, in pixels.
pub const CANVAS_SHORT_EDGE: u32 = 768;
/// Area cap of the canvas rule, in pixels.
pub const CANVAS_MAX_PIXELS: u32 = 768 * 1344;
/// What every canvas side is a multiple of: the VAE's 16x spatial
/// compression times the 2x2 denoiser patch.
pub const CANVAS_MULTIPLE: u32 = 32;
/// Narrowest accepted width-to-height ratio, 1:4.
pub const MIN_ASPECT_RATIO: f64 = 0.25;
/// Widest accepted width-to-height ratio, 4:1.
pub const MAX_ASPECT_RATIO: f64 = 4.0;
/// The aspect ratios `t2va` and `ref2va` accept besides `auto`.
pub const NAMED_ASPECT_RATIOS: [(u32, u32); 6] =
    [(21, 9), (16, 9), (4, 3), (1, 1), (3, 4), (9, 16)];
/// The ratio `auto` means for `t2va` and `ref2va`.
const DEFAULT_ASPECT_RATIO: (u32, u32) = (16, 9);

/// Short edge the released checkpoint encodes an image reference at.
pub const REFERENCE_IMAGE_SHORT_EDGE: u32 = 2048;

/// Pixel frames per video VAE window, and the overlap a window carries.
const VAE_FRAMES_PER_CHUNK: u32 = 17;
const VAE_LATENTS_PER_CHUNK: u32 = 5;
/// Fewest 24 fps frames a reference video must keep: one VAE window.
pub const MIN_REFERENCE_FRAMES: u32 = VAE_FRAMES_PER_CHUNK + VAE_LATENTS_PER_CHUNK;

/// The audio VAE's sample rate and hop: 40 latents per second.
pub const AUDIO_SAMPLE_RATE: u32 = 32_000;
/// Samples per audio latent.
pub const AUDIO_HOP: u32 = 800;
/// Stereo channels, packed channel-major.
const AUDIO_CHANNELS: u32 = 2;

/// The rate the conditioner samples a reference video at.
const VIDEO_SAMPLE_FPS: f64 = 2.0;

/// Image references one `ref2va` request may carry, as documented for the
/// released checkpoint.
pub const MAX_IMAGE_REFERENCES: usize = 9;
/// Video references, with or without soundtrack, one request may carry.
pub const MAX_VIDEO_REFERENCES: usize = 3;
/// Audio references one request may carry, not counting video soundtracks.
pub const MAX_AUDIO_REFERENCES: usize = 3;
/// References of all types one request may carry; keyframes not counted.
pub const MAX_REFERENCES: usize = 12;

/// The media type of a condition, by its request name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditionType {
    /// A JPEG, PNG or WEBP image.
    Image,
    /// A video; its soundtrack is a condition too when it has one.
    Video,
    /// A video that must have a soundtrack.
    VideoAudio,
    /// A WAV or MP3 audio file.
    Audio,
}

/// Whether a condition anchors a generated frame or is a reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditionRole {
    /// Anchors the first or last generated frame.
    Keyframe,
    /// Conditions the generation as a whole.
    Reference,
}

/// The generated frame a keyframe anchors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FramePosition {
    /// `frame_index` 0.
    First,
    /// `frame_index` -1.
    Last,
}

/// Denoiser rows of one latent frame of `canvas`: one per 32x32 pixel block.
pub const fn rows_per_frame(canvas: Canvas) -> u32 {
    (canvas.width / CANVAS_MULTIPLE) * (canvas.height / CANVAS_MULTIPLE)
}

/// Resolves a display aspect into a canvas with the adapt_shape_v1 rule.
///
/// Only the ratio of the arguments matters. The short edge starts at 768
/// pixels, an area above `768 * 1344` scales both sides down, and each side
/// rounds half to even to a multiple of 32, so the final area may end up
/// slightly above the cap.
///
/// # Errors
///
/// Returns a message when the ratio is not finite and positive or lies
/// outside 1:4 to 4:1.
pub fn canvas(aspect_width: f64, aspect_height: f64) -> Result<Canvas, String> {
    if !(aspect_width > 0.0 && aspect_height > 0.0) {
        return Err(format!(
            "aspect must be positive, got {aspect_width}:{aspect_height}"
        ));
    }
    let ratio = aspect_width / aspect_height;
    if !(MIN_ASPECT_RATIO..=MAX_ASPECT_RATIO).contains(&ratio) {
        return Err(format!(
            "aspect {aspect_width}:{aspect_height} lies outside 1:4 to 4:1"
        ));
    }
    let short_edge = f64::from(CANVAS_SHORT_EDGE);
    let (mut width, mut height) = if ratio >= 1.0 {
        (short_edge * ratio, short_edge)
    } else {
        (short_edge, short_edge / ratio)
    };
    let area = width * height;
    let max_pixels = f64::from(CANVAS_MAX_PIXELS);
    if area > max_pixels {
        // `powf(0.5)` rather than `sqrt` reproduces the reference's `** 0.5`.
        let scale = (max_pixels / area).powf(0.5);
        width *= scale;
        height *= scale;
    }
    Ok(Canvas {
        width: nearest_multiple(width),
        height: nearest_multiple(height),
    })
}

/// Rounds a side half to even to the nearest multiple of 32, at least 32.
fn nearest_multiple(value: f64) -> u32 {
    let multiple = f64::from(CANVAS_MULTIPLE);
    // Sides stay far below `u32::MAX`, so the conversion is exact.
    ((value / multiple).round_ties_even() * multiple).max(multiple) as u32
}

/// The requested output, as the request states it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Target<'a> {
    /// `target.short_edge`: 768, the canvas rule's, or 480 where the
    /// deployment serves a FastH3 export's 480p buckets.
    pub short_edge: u32,
    /// `target.aspect_ratio`: `auto` or `W:H`.
    pub aspect_ratio: &'a str,
    /// `target.duration_seconds`, when given.
    pub duration_seconds: Option<f64>,
}

/// One condition's request fields other than its URI.
#[derive(Debug, Clone, PartialEq)]
pub struct ConditionSpec {
    /// `conditions[i].type`.
    pub condition_type: ConditionType,
    /// `conditions[i].role`.
    pub role: ConditionRole,
    /// `conditions[i].frame_index`: 0 or -1 for a keyframe.
    pub frame_index: Option<i64>,
    /// `conditions[i].start_time_seconds`, for video references.
    pub start_seconds: Option<f64>,
}

/// How a deployment sizes the references of a `ref2va` request: the options
/// of the MiniMax H3 video processor (`--mm-processor-kwargs`).
///
/// A reference is only ever reduced by default: an image whose short edge
/// exceeds `image_short_edge` scales down to it, and a video with more pixels
/// than the canvas of its aspect scales down to that canvas, while smaller
/// references keep their own size. The released pipeline also enlarges
/// smaller ones to those sizes, which `enlarge` selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReferenceConfig {
    /// Short edge, in pixels, an image reference is encoded at, at most: a
    /// multiple of 32 up to [`REFERENCE_IMAGE_SHORT_EDGE`].
    #[serde(rename = "reference_image_short_edge")]
    pub image_short_edge: u32,
    /// Whether a reference smaller than its encoding size is enlarged to it:
    /// an image to `image_short_edge` and a video to the canvas of its
    /// aspect.
    #[serde(rename = "enlarge_references")]
    pub enlarge: bool,
}

impl Default for ReferenceConfig {
    /// The released checkpoint's short edge, without enlarging.
    fn default() -> Self {
        Self {
            image_short_edge: REFERENCE_IMAGE_SHORT_EDGE,
            enlarge: false,
        }
    }
}

impl ReferenceConfig {
    /// The released pipeline's sizing: every reference at its encoding size,
    /// enlarged when smaller.
    pub const RELEASED: Self = Self {
        image_short_edge: REFERENCE_IMAGE_SHORT_EDGE,
        enlarge: true,
    };

    /// Reads the config from the processor options a deployment states
    /// (`--mm-processor-kwargs`): `reference_image_short_edge` and
    /// `enlarge_references`, each defaulting when absent.
    ///
    /// # Errors
    ///
    /// Returns a message for an option this processor does not take, a value
    /// of the wrong type, or a short edge [`ReferenceConfig::validate`]
    /// refuses.
    pub fn from_kwargs(
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<Self, String> {
        let config = Self::deserialize(serde_json::Value::Object(kwargs.clone()))
            .map_err(|error| error.to_string())?;
        config.validate()?;
        Ok(config)
    }

    /// Checks that the short edge is a multiple of 32 from 32 pixels to the
    /// released checkpoint's.
    ///
    /// # Errors
    ///
    /// Returns a message naming the short edge.
    pub fn validate(&self) -> Result<(), String> {
        let edge = self.image_short_edge;
        if !edge.is_multiple_of(CANVAS_MULTIPLE)
            || !(CANVAS_MULTIPLE..=REFERENCE_IMAGE_SHORT_EDGE).contains(&edge)
        {
            return Err(format!(
                "a reference image short edge must be a multiple of {CANVAS_MULTIPLE} from \
                 {CANVAS_MULTIPLE} to {REFERENCE_IMAGE_SHORT_EDGE} pixels, got {edge}"
            ));
        }
        Ok(())
    }
}

/// What the deployment serves, narrowing the model's own rules.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanLimits {
    /// The tasks of the placed denoiser.
    pub tasks: Vec<VideoTask>,
    /// Longest requested duration served (`--max-video-seconds`).
    pub max_video_seconds: f64,
    /// The only canvases the deployment serves: those its workers prepare
    /// and declare in their handshake.
    pub canvases: Vec<Canvas>,
    /// How the deployment sizes references.
    pub references: ReferenceConfig,
}

impl PlanLimits {
    /// The `target.short_edge` values the deployment serves, in the order of
    /// its canvases: each served canvas's training-bucket class, or the
    /// canvas rule's 768 for a canvas no bucket has.
    pub fn short_edges(&self) -> Vec<u32> {
        let mut edges = Vec::new();
        for canvas in &self.canvases {
            let edge = VideoRaster::find(canvas.width, canvas.height)
                .map_or(CANVAS_SHORT_EDGE, |bucket| bucket.resolution.short_edge());
            if !edges.contains(&edge) {
                edges.push(edge);
            }
        }
        edges
    }
}

/// A Qwen3-VL `(t, h, w)` patch grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct VisionGrid {
    /// Temporal blocks; 1 for an image.
    pub t: u32,
    /// Patch rows.
    pub h: u32,
    /// Patch columns.
    pub w: u32,
}

/// The Qwen3-VL processor geometry: patching and pixel budgets.
///
/// Read from the checkpoint's `processor/preprocessor_config.json` and
/// `processor/video_preprocessor_config.json` with
/// [`ProcessorConfig::read`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessorConfig {
    /// Pixels per patch side.
    pub patch_size: u32,
    /// Frames merged into one temporal patch.
    pub temporal_patch_size: u32,
    /// Patches per merged vision token side.
    pub merge_size: u32,
    /// Lower pixel budget of an image.
    pub image_min_pixels: u64,
    /// Upper pixel budget of an image.
    pub image_max_pixels: u64,
    /// Lower budget of a video: frames times pixels.
    pub video_min_pixels: u64,
    /// Upper budget of a video: frames times pixels.
    pub video_max_pixels: u64,
}

#[derive(Deserialize)]
struct ProcessorFile {
    patch_size: u32,
    temporal_patch_size: u32,
    merge_size: u32,
    size: ProcessorSize,
}

#[derive(Deserialize)]
struct ProcessorSize {
    shortest_edge: u64,
    longest_edge: u64,
}

impl ProcessorConfig {
    /// Reads a checkpoint's image processor config
    /// (`processor/preprocessor_config.json`) and video processor config
    /// (`processor/video_preprocessor_config.json`).
    ///
    /// # Errors
    ///
    /// Fails when a file cannot be read or parsed, or when the two configs
    /// disagree on the patch geometry.
    pub fn read(image: &Path, video: &Path) -> anyhow::Result<Self> {
        let read = |path: &Path| -> anyhow::Result<ProcessorFile> {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?;
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
        };
        let image = read(image)?;
        let video = read(video)?;
        anyhow::ensure!(
            (
                image.patch_size,
                image.temporal_patch_size,
                image.merge_size
            ) == (
                video.patch_size,
                video.temporal_patch_size,
                video.merge_size
            ),
            "the image and video processors disagree on the patch geometry"
        );
        Ok(Self {
            patch_size: image.patch_size,
            temporal_patch_size: image.temporal_patch_size,
            merge_size: image.merge_size,
            image_min_pixels: image.size.shortest_edge,
            image_max_pixels: image.size.longest_edge,
            video_min_pixels: video.size.shortest_edge,
            video_max_pixels: video.size.longest_edge,
        })
    }

    /// What a resized side is a multiple of: one merged token.
    const fn factor(&self) -> u32 {
        self.patch_size * self.merge_size
    }

    /// The patch grid the image processor resizes an image to.
    ///
    /// This is Qwen2-VL's `smart_resize`: each side rounds half to even to
    /// the factor, and a size outside the pixel budget is rescaled with floor
    /// (above) or ceil (below) rounding.
    ///
    /// # Errors
    ///
    /// Returns a message for an aspect beyond 200:1.
    pub fn image_grid(&self, height: u32, width: u32) -> Result<VisionGrid, String> {
        check_vision_aspect(height, width)?;
        let factor = f64::from(self.factor());
        let (height, width) = (f64::from(height), f64::from(width));
        let mut resized_height = (height / factor).round_ties_even() * factor;
        let mut resized_width = (width / factor).round_ties_even() * factor;
        if resized_height * resized_width > self.image_max_pixels as f64 {
            let beta = (height * width / self.image_max_pixels as f64).sqrt();
            resized_height = factor.max((height / beta / factor).floor() * factor);
            resized_width = factor.max((width / beta / factor).floor() * factor);
        } else if resized_height * resized_width < self.image_min_pixels as f64 {
            let beta = (self.image_min_pixels as f64 / (height * width)).sqrt();
            resized_height = (height * beta / factor).ceil() * factor;
            resized_width = (width * beta / factor).ceil() * factor;
        }
        Ok(self.grid(1, resized_height, resized_width))
    }

    /// The patch grid of `frames` sampled video frames.
    ///
    /// This is Qwen3-VL's video `smart_resize`: the budget covers all frames,
    /// which are padded to a whole temporal patch by repeating the last one.
    ///
    /// # Errors
    ///
    /// Returns a message for frames smaller than one merged token or an
    /// aspect beyond 200:1.
    pub fn video_grid(&self, frames: u32, height: u32, width: u32) -> Result<VisionGrid, String> {
        let factor = self.factor();
        if height < factor || width < factor {
            return Err(format!("video frames of {width}x{height} are too small"));
        }
        check_vision_aspect(height, width)?;
        let temporal = self.temporal_patch_size;
        let padded_frames = frames.div_ceil(temporal) * temporal;
        let factor = f64::from(factor);
        let pixels = f64::from(frames) * f64::from(height) * f64::from(width);
        let (height, width) = (f64::from(height), f64::from(width));
        let mut resized_height = (height / factor).round_ties_even() * factor;
        let mut resized_width = (width / factor).round_ties_even() * factor;
        let budget = f64::from(padded_frames) * resized_height * resized_width;
        if budget > self.video_max_pixels as f64 {
            let beta = (pixels / self.video_max_pixels as f64).sqrt();
            resized_height = factor.max((height / beta / factor).floor() * factor);
            resized_width = factor.max((width / beta / factor).floor() * factor);
        } else if budget < self.video_min_pixels as f64 {
            let beta = (self.video_min_pixels as f64 / pixels).sqrt();
            resized_height = (height * beta / factor).ceil() * factor;
            resized_width = (width * beta / factor).ceil() * factor;
        }
        Ok(self.grid(padded_frames / temporal, resized_height, resized_width))
    }

    /// Vision tokens of one temporal block of `grid`.
    pub const fn block_tokens(&self, grid: VisionGrid) -> u32 {
        grid.h * grid.w / (self.merge_size * self.merge_size)
    }

    fn grid(&self, t: u32, resized_height: f64, resized_width: f64) -> VisionGrid {
        // Resized sides are positive multiples of the factor below 2^32.
        VisionGrid {
            t,
            h: resized_height as u32 / self.patch_size,
            w: resized_width as u32 / self.patch_size,
        }
    }
}

fn check_vision_aspect(height: u32, width: u32) -> Result<(), String> {
    let (long, short) = (height.max(width), height.min(width));
    if f64::from(long) / f64::from(short) > 200.0 {
        return Err(format!("an aspect of {width}x{height} exceeds 200:1"));
    }
    Ok(())
}

/// An aspect-preserving resize followed by a centred crop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CoverCrop {
    /// Width the image is resized to before cropping.
    pub width: u32,
    /// Height the image is resized to before cropping.
    pub height: u32,
    /// Left edge of the crop in the resized image.
    pub left: u32,
    /// Top edge of the crop in the resized image.
    pub top: u32,
}

/// Returns the LANCZOS resize and centred crop of a `width` x `height` image
/// that cover `canvas`.
///
/// The scale is the larger of the two side ratios; each resized side rounds
/// half to even and never falls below the canvas, and the crop is centred
/// with floor division.
pub fn cover_crop(width: u32, height: u32, canvas: Canvas) -> CoverCrop {
    let scale = (f64::from(canvas.width) / f64::from(width))
        .max(f64::from(canvas.height) / f64::from(height));
    let resized_width = canvas
        .width
        .max((f64::from(width) * scale).round_ties_even() as u32);
    let resized_height = canvas
        .height
        .max((f64::from(height) * scale).round_ties_even() as u32);
    CoverCrop {
        width: resized_width,
        height: resized_height,
        left: (resized_width - canvas.width) / 2,
        top: (resized_height - canvas.height) / 2,
    }
}

/// How a keyframe is put on the target canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct KeyframeFit {
    /// The generated frame the keyframe anchors.
    pub position: FramePosition,
    /// `None` when the keyframe is stretched onto the canvas (the first
    /// keyframe of a request); the crop otherwise.
    pub cover_crop: Option<CoverCrop>,
}

/// The samples of a soundtrack the request conditions on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AudioClip {
    /// The native sample rate.
    pub sample_rate: u32,
    /// Native samples skipped at the start.
    pub start_sample: u64,
    /// Native samples kept after the start offset.
    pub source_samples: u64,
    /// Samples after resampling to 32 kHz.
    pub samples: u64,
    /// Audio latents per channel; the clip packs `2 * latents` channel-major
    /// rows.
    pub latents: u32,
}

impl AudioClip {
    /// Denoiser rows of the clip: one per latent per stereo channel.
    pub const fn rows(&self) -> u32 {
        AUDIO_CHANNELS * self.latents
    }
}

/// The frames of a reference video the request conditions on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct VideoClip {
    /// The canvas of the video's own display aspect the frames are put on.
    pub canvas: Canvas,
    /// Frames of the 24 fps timeline skipped at the start.
    pub start_frame: u32,
    /// 24 fps frames kept after the start offset, at most the generated
    /// frame count; the conditioner samples these.
    pub frames: u32,
    /// The leading `17n + 5` frames the VAE encodes.
    pub vae_frames: u32,
    /// The video's soundtrack, when it has one.
    pub soundtrack: Option<AudioClip>,
}

/// The conditioner's view of one image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ImageVision {
    /// The patch grid; `t` is 1.
    pub grid: VisionGrid,
    /// Vision tokens of the image.
    pub tokens: u32,
}

/// The conditioner's view of one reference video.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VideoVision {
    /// The patch grid; `t` counts vision blocks.
    pub grid: VisionGrid,
    /// Vision tokens of each block.
    pub block_tokens: u32,
    /// The 24 fps frames sampled at 2 fps.
    pub frame_indices: Vec<u32>,
    /// The timestamp label of each block, in seconds.
    pub block_timestamps: Vec<f64>,
}

/// What the conditioner reads of a condition.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Vision {
    /// One image block.
    Image(ImageVision),
    /// Timestamped video blocks.
    Video(VideoVision),
}

/// How a condition's media is prepared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Prepared {
    /// A keyframe fitted to the target canvas.
    Keyframe(KeyframeFit),
    /// An image reference resized (LANCZOS) to this size.
    Image(Canvas),
    /// A reference video clip and its soundtrack.
    Video(VideoClip),
    /// An audio reference.
    Audio(AudioClip),
}

/// How one condition is prepared and what it contributes.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConditionPlan {
    /// Position of the condition in the request.
    pub index: usize,
    /// The preparation of its media.
    pub prepared: Prepared,
    /// The conditioner's view, for conditions it reads.
    pub vision: Option<Vision>,
    /// Denoiser audio condition rows.
    pub audio_rows: u32,
}

impl ConditionPlan {
    /// Denoiser video rows of each unit the video encoder encodes, in unit
    /// order.
    ///
    /// `canvas` is the generated canvas. A keyframe is one unit of one latent
    /// frame of it. A reference image is `image_bands` units,
    /// bands of its rows of 32x32-pixel patches dealt as evenly as possible
    /// in order: with `g` patch rows, band `b` holds rows `b * g / bands` to
    /// `(b + 1) * g / bands`, rounded down, which is the video encoder's
    /// `row_bands` partition. `image_bands` is clamped to one through `g`.
    /// A video's `17n + 5` encoded frames are `n + 1` units of up to 17
    /// frames: each complete one yields 5 latent frames and the last 2, the
    /// video encoder dropping the leading 3 of its padded final window. An
    /// audio reference has none.
    pub fn latent_units(&self, canvas: Canvas, image_bands: u32) -> Vec<u32> {
        match &self.prepared {
            Prepared::Keyframe(_) => vec![rows_per_frame(canvas)],
            Prepared::Image(size) => {
                let patch_rows = size.height / CANVAS_MULTIPLE;
                let rows_per_patch_row = size.width / CANVAS_MULTIPLE;
                let bands = image_bands.clamp(1, patch_rows.max(1));
                (0..bands)
                    .map(|band| {
                        let start = band * patch_rows / bands;
                        let stop = (band + 1) * patch_rows / bands;
                        (stop - start) * rows_per_patch_row
                    })
                    .collect()
            }
            Prepared::Video(clip) => {
                let windows = (clip.vae_frames - VAE_LATENTS_PER_CHUNK) / VAE_FRAMES_PER_CHUNK;
                let rows_per_frame = rows_per_frame(clip.canvas);
                let mut units = vec![VAE_LATENTS_PER_CHUNK * rows_per_frame; windows as usize];
                units.push(
                    (latent_frames(clip.vae_frames) - windows * VAE_LATENTS_PER_CHUNK)
                        * rows_per_frame,
                );
                units
            }
            Prepared::Audio(_) => Vec::new(),
        }
    }

    /// Describes the condition for the engine and the workers: its role, the
    /// published media at `source`, what the media reader decodes it into,
    /// what the conditioner reads of it, and the denoiser rows it encodes to,
    /// a reference image in `image_bands` units ([`Self::latent_units`]).
    /// `canvas` is the generated canvas, which a keyframe is fitted to.
    pub fn describe(
        &self,
        canvas: Canvas,
        source: MediaLocator,
        image_bands: u32,
    ) -> VideoCondition {
        let audio = |clip: &AudioClip| uniserve_core::AudioClip {
            sample_rate: clip.sample_rate,
            start_sample: clip.start_sample,
            source_samples: clip.source_samples,
            samples: clip.samples,
        };
        let (role, media) = match &self.prepared {
            Prepared::Keyframe(fit) => {
                let role = match fit.position {
                    FramePosition::First => uniserve_core::ConditionRole::FirstFrame,
                    FramePosition::Last => uniserve_core::ConditionRole::LastFrame,
                };
                let size = canvas;
                // A stretched keyframe is resized straight onto the canvas; a
                // cover-cropped one past it, then centred.
                let fit = match fit.cover_crop {
                    None => ImageFit {
                        resized: size,
                        left: 0,
                        top: 0,
                        size,
                    },
                    Some(crop) => ImageFit {
                        resized: Canvas {
                            width: crop.width,
                            height: crop.height,
                        },
                        left: crop.left,
                        top: crop.top,
                        size,
                    },
                };
                (role, ConditionMedia::Image(fit))
            }
            Prepared::Image(size) => (
                uniserve_core::ConditionRole::Reference,
                ConditionMedia::Image(ImageFit {
                    resized: *size,
                    left: 0,
                    top: 0,
                    size: *size,
                }),
            ),
            Prepared::Video(clip) => (
                uniserve_core::ConditionRole::Reference,
                ConditionMedia::Video {
                    clip: uniserve_core::VideoClip {
                        canvas: clip.canvas,
                        start_frame: clip.start_frame,
                        frames: clip.frames,
                        vae_frames: clip.vae_frames,
                    },
                    soundtrack: clip.soundtrack.as_ref().map(audio),
                },
            ),
            Prepared::Audio(clip) => (
                uniserve_core::ConditionRole::Reference,
                ConditionMedia::Audio(audio(clip)),
            ),
        };
        let vision = self.vision.as_ref().map(|vision| match vision {
            Vision::Image(image) => ConditionVision {
                grid: core_grid(image.grid),
                tokens: image.tokens,
                frame_indices: Vec::new(),
            },
            Vision::Video(video) => ConditionVision {
                grid: core_grid(video.grid),
                tokens: video.grid.t * video.block_tokens,
                frame_indices: video.frame_indices.clone(),
            },
        });
        VideoCondition {
            role,
            source,
            media,
            vision,
            latent_units: self.latent_units(canvas, image_bands),
            audio_rows: self.audio_rows,
        }
    }
}

const fn core_grid(grid: VisionGrid) -> uniserve_core::VisionGrid {
    uniserve_core::VisionGrid {
        t: grid.t,
        h: grid.h,
        w: grid.w,
    }
}

/// The resolved size of a request and of every condition.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RequestPlan {
    /// The request's task.
    pub task: VideoTask,
    /// The generated canvas.
    pub canvas: Canvas,
    /// The requested duration in seconds: `target.duration_seconds`, or the
    /// soundtrack's length after its start offset when that sets it.
    pub duration_seconds: f64,
    /// Generated frames, of the form `17n + 5`.
    pub num_frames: u32,
    /// One plan per condition, in request order.
    pub conditions: Vec<ConditionPlan>,
}

impl RequestPlan {
    /// The units each condition's reference image is encoded in when one
    /// latent encoding round covers `lane` units, one entry per condition
    /// (other conditions' entries are unread).
    ///
    /// A lone reference image is the largest visual condition and is a
    /// single rank's work as one unit; its bands spread its encoder tiles
    /// over the ranks of one round. Fewer images than the lane divide it
    /// evenly, at least one unit each. More images than the lane encode
    /// whole, a lane's worth per round, except that when every visual unit
    /// is an image, the `k` images of a final partial round take `lane / k`
    /// bands each whenever `k` divides the lane, so that round is full
    /// rather than leaving ranks idle behind its whole images.
    pub fn image_bands(&self, lane: u32) -> Vec<u32> {
        let is_image = |plan: &ConditionPlan| matches!(plan.prepared, Prepared::Image(_));
        let images = self.conditions.iter().filter(|plan| is_image(plan)).count() as u32;
        let mut bands = vec![(lane / images.max(1)).max(1); self.conditions.len()];
        let only_images = self
            .conditions
            .iter()
            .all(|plan| is_image(plan) || matches!(plan.prepared, Prepared::Audio(_)));
        let partial = images % lane.max(1);
        if only_images && images > lane && partial > 0 && lane.is_multiple_of(partial) {
            // The last `partial` images, in request order, form the final
            // round.
            let mut remaining = partial;
            for (entry, plan) in bands.iter_mut().zip(&self.conditions).rev() {
                if remaining == 0 {
                    break;
                }
                if is_image(plan) {
                    *entry = lane / partial;
                    remaining -= 1;
                }
            }
        }
        bands
    }
}

/// The video latent frames of a `17n + 5` frame count.
const fn latent_frames(frames: u32) -> u32 {
    (frames - VAE_LATENTS_PER_CHUNK) / VAE_FRAMES_PER_CHUNK * VAE_LATENTS_PER_CHUNK + 2
}

/// Returns the size an image reference is encoded at.
///
/// An image whose short edge exceeds `config.image_short_edge` scales down
/// to it; a smaller one keeps its size, or scales up to that edge under
/// `config.enlarge`. Each side then rounds half to even to a multiple of 32.
/// Unlike the canvas rule there is no area cap.
///
/// # Errors
///
/// Returns a message when the image lies outside 1:4 to 4:1.
pub fn reference_image_size(
    width: u32,
    height: u32,
    config: &ReferenceConfig,
) -> Result<Canvas, String> {
    if u64::from(width) > 4 * u64::from(height) || u64::from(height) > 4 * u64::from(width) {
        return Err(format!(
            "a reference image must lie within 1:4 and 4:1, got {width}x{height}"
        ));
    }
    let mut scale = f64::from(config.image_short_edge) / f64::from(width.min(height));
    if !config.enlarge {
        scale = scale.min(1.0);
    }
    Ok(Canvas {
        width: nearest_multiple(f64::from(width) * scale),
        height: nearest_multiple(f64::from(height) * scale),
    })
}

/// Samples a 24 fps clip at 2 fps and labels its vision blocks.
///
/// Every twelfth frame is sampled. The conditioner merges the samples in
/// groups of `temporal_patch`, repeating the last one to fill a group, and
/// labels a group with the mean of its timestamps.
///
/// # Errors
///
/// Returns a message when fewer frames are sampled than one group holds.
pub fn sample_video_frames(
    frames: u32,
    temporal_patch: u32,
) -> Result<(Vec<u32>, Vec<f64>), String> {
    let stride = f64::from(VIDEO_FPS) / VIDEO_SAMPLE_FPS;
    let mut indices: Vec<u32> = Vec::new();
    let mut cursor = 0.0_f64;
    loop {
        // Python's `round` rounds half to even.
        let index = cursor.round_ties_even();
        if index >= f64::from(frames) {
            break;
        }
        let index = index as u32;
        if indices.last().is_none_or(|&last| index > last) {
            indices.push(index);
        }
        cursor += stride;
    }
    let temporal_patch = temporal_patch as usize;
    if indices.len() < temporal_patch {
        return Err(format!(
            "a reference video must sample at least {temporal_patch} frames at \
             {VIDEO_SAMPLE_FPS} fps, got {}",
            indices.len()
        ));
    }
    let mut timestamps: Vec<f64> = (0..indices.len())
        .map(|index| index as f64 / VIDEO_SAMPLE_FPS)
        .collect();
    let last = timestamps[timestamps.len() - 1];
    timestamps.resize(
        indices.len().div_ceil(temporal_patch) * temporal_patch,
        last,
    );
    let blocks = timestamps
        .chunks(temporal_patch)
        .map(|group| (group[0] + group[temporal_patch - 1]) / 2.0)
        .collect();
    Ok((indices, blocks))
}

/// Units of a `rate` timeline skipped by a start offset of `seconds`.
fn start_offset(seconds: f64, rate: f64) -> u64 {
    // Start offsets are validated finite and non-negative.
    (seconds * rate + 0.5).floor() as u64
}

/// Sizes a soundtrack: start offset, truncation, 32 kHz resampling, latents.
///
/// The first `floor(s * rate + 0.5)` samples are skipped and the rest is
/// truncated at the native rate to the generated duration, as the reference
/// does before it resamples to 32 kHz.
///
/// # Errors
///
/// Returns a message when the offset lies past the end of the soundtrack.
pub fn audio_clip(
    media: &AudioFacts,
    start_seconds: f64,
    frames: u32,
) -> Result<AudioClip, String> {
    let rate = media.sample_rate;
    let start_sample = start_offset(start_seconds, f64::from(rate));
    if start_sample >= media.samples {
        return Err("start_time_seconds lies past the end of the audio track".to_owned());
    }
    let available = media.samples - start_sample;
    // The reference truncates to `int(frames / 24 * rate)` samples.
    let limit = (f64::from(frames) / f64::from(VIDEO_FPS) * f64::from(rate)).trunc() as u64;
    let source = available.min(limit);
    let samples = if rate == AUDIO_SAMPLE_RATE {
        source
    } else {
        // torchaudio's resampler returns ceil(new * length / orig) samples
        // for the gcd-reduced rates.
        let divisor = gcd(u64::from(rate), u64::from(AUDIO_SAMPLE_RATE));
        let new = u64::from(AUDIO_SAMPLE_RATE) / divisor;
        let orig = u64::from(rate) / divisor;
        (source * new).div_ceil(orig)
    };
    Ok(AudioClip {
        sample_rate: rate,
        start_sample,
        source_samples: source,
        samples,
        // The audio VAE pads the waveform to a whole hop.
        latents: u32::try_from(samples.div_ceil(u64::from(AUDIO_HOP)))
            .map_err(|_| "the audio track is too long".to_owned())?,
    })
}

const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

/// Sizes a reference video and the conditioner's view of it.
///
/// The video is resampled to 24 fps (each frame held until the slot of the
/// next), its first `floor(s * 24 + 0.5)` frames are skipped and the rest is
/// truncated to the generated frame count, on the canvas of its own display
/// aspect. A video with fewer pixels than that canvas keeps its own size,
/// each side rounded half to even to a multiple of 32, unless
/// `config.enlarge` scales it up to the canvas. Its soundtrack is offset by
/// the same start and sized like an audio reference.
///
/// # Errors
///
/// Returns a message when the display aspect lies outside 1:4 to 4:1, fewer
/// than 22 frames remain after the offset, or the offset lies past the end of
/// the soundtrack.
pub fn video_reference(
    media: &VideoFacts,
    start_seconds: f64,
    frames: u32,
    vision: &ProcessorConfig,
    config: &ReferenceConfig,
) -> Result<(VideoClip, VideoVision), String> {
    // Display sizes stay far below 2^53, so the conversions are exact.
    let mut size = canvas(media.display_width as f64, media.display_height as f64)
        .map_err(|error| format!("the video's {error}"))?;
    let pixels = |width: u32, height: u32| u64::from(width) * u64::from(height);
    if !config.enlarge && pixels(media.width, media.height) < pixels(size.width, size.height) {
        size = Canvas {
            width: nearest_multiple(f64::from(media.width)),
            height: nearest_multiple(f64::from(media.height)),
        };
    }
    let frame_rate =
        f64::from(media.frame_rate.numerator) / f64::from(media.frame_rate.denominator);
    let fps = f64::from(VIDEO_FPS);
    let timeline = if frame_rate == fps {
        media.frames
    } else {
        // The timeline ends at the 24 fps slot the stream's end rounds to.
        (media.frames as f64 * (fps / frame_rate) + 0.5).floor() as u64
    };
    let start_frame = start_offset(start_seconds, fps);
    let available = timeline.saturating_sub(start_frame);
    if available < u64::from(MIN_REFERENCE_FRAMES) {
        return Err(format!(
            "a reference video needs at least {MIN_REFERENCE_FRAMES} frames at {VIDEO_FPS} fps \
             after start_time_seconds, got {available}"
        ));
    }
    let kept = u32::try_from(available).map_or(frames, |available| available.min(frames));
    // The VAE encodes the leading complete `17n + 5` frames.
    let vae_frames = ((kept - VAE_LATENTS_PER_CHUNK) / VAE_FRAMES_PER_CHUNK).max(1)
        * VAE_FRAMES_PER_CHUNK
        + VAE_LATENTS_PER_CHUNK;
    let soundtrack = media
        .soundtrack
        .as_ref()
        .map(|track| audio_clip(track, start_seconds, frames))
        .transpose()?;

    let (frame_indices, block_timestamps) = sample_video_frames(kept, vision.temporal_patch_size)?;
    let sampled = u32::try_from(frame_indices.len()).unwrap_or(u32::MAX);
    let grid = vision.video_grid(sampled, size.height, size.width)?;
    let clip = VideoClip {
        canvas: size,
        // Offsets past the timeline were rejected above.
        start_frame: u32::try_from(start_frame).unwrap_or(u32::MAX),
        frames: kept,
        vae_frames,
        soundtrack,
    };
    let seen = VideoVision {
        grid,
        block_tokens: vision.block_tokens(grid),
        frame_indices,
        block_timestamps,
    };
    Ok((clip, seen))
}

/// Parses `W:H` into positive integers written in canonical decimal form;
/// `None` when malformed.
fn parse_aspect_ratio(value: &str) -> Option<(u32, u32)> {
    let (width, height) = value.split_once(':')?;
    let side = |text: &str| -> Option<u32> {
        let parsed: u32 = text.parse().ok()?;
        (parsed > 0 && parsed.to_string() == text).then_some(parsed)
    };
    Some((side(width)?, side(height)?))
}

/// The aspect ratio a target asks for: a named ratio for `t2va` and
/// `ref2va` (`auto` being 16:9), any ratio within 1:4 to 4:1 for `fl2va`,
/// and `None` for `fl2va` with `auto`, whose canvas follows the first
/// keyframe.
fn requested_ratio(
    task: VideoTask,
    aspect_ratio: &str,
) -> Result<Option<(u32, u32)>, VideoInputError> {
    if aspect_ratio == "auto" {
        return Ok((task != VideoTask::Fl2va).then_some(DEFAULT_ASPECT_RATIO));
    }
    let ratio = parse_aspect_ratio(aspect_ratio);
    let (accepted, allowed) = match task {
        VideoTask::Fl2va => (
            ratio.is_some_and(|(width, height)| {
                (MIN_ASPECT_RATIO..=MAX_ASPECT_RATIO)
                    .contains(&(f64::from(width) / f64::from(height)))
            }),
            "auto or W:H within 1:4 to 4:1".to_owned(),
        ),
        VideoTask::T2va | VideoTask::Ref2va => {
            let named: Vec<String> = NAMED_ASPECT_RATIOS
                .iter()
                .map(|(width, height)| format!("{width}:{height}"))
                .collect();
            (
                ratio.is_some_and(|ratio| NAMED_ASPECT_RATIOS.contains(&ratio)),
                format!("auto or one of {}", named.join(", ")),
            )
        }
    };
    match ratio {
        Some(ratio) if accepted => Ok(Some(ratio)),
        _ => Err(VideoInputError::invalid(
            RequestField::TargetAspectRatio,
            format!("must be {allowed}, got {aspect_ratio:?}"),
        )),
    }
}

/// The canvas a target names without consulting media; `None` for `fl2va`
/// with `auto`.
///
/// At the canvas rule's 768-pixel short edge a ratio resolves through
/// [`canvas`]. Another short edge names the FastH3 training bucket of that
/// resolution class, whose sides the rule does not produce: 480p 21:9 is
/// 992x416.
fn named_canvas(task: VideoTask, target: &Target<'_>) -> Result<Option<Canvas>, VideoInputError> {
    let Some((width, height)) = requested_ratio(task, target.aspect_ratio)? else {
        return Ok(None);
    };
    if target.short_edge == CANVAS_SHORT_EDGE {
        return canvas(f64::from(width), f64::from(height))
            .map(Some)
            .map_err(|error| VideoInputError::invalid(RequestField::TargetAspectRatio, error));
    }
    VideoResolution::from_short_edge(target.short_edge)
        .and_then(|resolution| VideoRaster::named(resolution, &format!("{width}:{height}")))
        .map(|bucket| {
            Some(Canvas {
                width: bucket.width,
                height: bucket.height,
            })
        })
        .ok_or_else(|| {
            VideoInputError::invalid(
                RequestField::TargetAspectRatio,
                format!(
                    "{width}:{height} has no canvas at a {}-pixel short edge",
                    target.short_edge
                ),
            )
        })
}

/// Every named aspect ratio the deployment serves at each of its short edges,
/// with the canvas it resolves to, short edge by short edge in
/// [`NAMED_ASPECT_RATIOS`] order.
pub fn named_canvases(limits: &PlanLimits) -> Vec<(u32, String, Canvas)> {
    let mut served = Vec::new();
    for short_edge in limits.short_edges() {
        for (width, height) in NAMED_ASPECT_RATIOS {
            let aspect_ratio = format!("{width}:{height}");
            let target = Target {
                short_edge,
                aspect_ratio: &aspect_ratio,
                duration_seconds: None,
            };
            if let Ok(Some(size)) = named_canvas(VideoTask::T2va, &target)
                && check_served(size, limits).is_ok()
            {
                served.push((short_edge, aspect_ratio, size));
            }
        }
    }
    served
}

fn check_served(size: Canvas, limits: &PlanLimits) -> Result<(), VideoInputError> {
    if limits.canvases.contains(&size) {
        return Ok(());
    }
    Err(VideoInputError::invalid(
        RequestField::TargetAspectRatio,
        format!(
            "resolves to {}x{}, which this deployment does not serve",
            size.width, size.height
        ),
    ))
}

/// Applies every rule that needs no media, in a fixed order: the served task,
/// the target's short edge and aspect ratio, each condition's fields, the
/// task's condition counts and keyframe order, then the duration.
///
/// Calling this before [`plan_request`] rejects a malformed request before
/// its media are fetched; planning resolves the duration and canvas from the
/// probed media.
///
/// # Errors
///
/// Returns [`VideoInputError::Invalid`] naming the first field at fault.
pub fn check_request(
    task: VideoTask,
    target: &Target<'_>,
    conditions: &[ConditionSpec],
    limits: &PlanLimits,
) -> Result<(), VideoInputError> {
    if !limits.tasks.contains(&task) {
        let served: Vec<&str> = limits.tasks.iter().map(|task| task.as_str()).collect();
        return Err(VideoInputError::invalid(
            RequestField::Task,
            format!(
                "{} is not served; this deployment serves {}",
                task.as_str(),
                served.join(", ")
            ),
        ));
    }
    let short_edges = limits.short_edges();
    if !short_edges.contains(&target.short_edge) {
        let served: Vec<String> = short_edges.iter().map(u32::to_string).collect();
        return Err(VideoInputError::invalid(
            RequestField::TargetShortEdge,
            format!("must be {}, got {}", served.join(" or "), target.short_edge),
        ));
    }

    if let Some(size) = named_canvas(task, target)? {
        check_served(size, limits)?;
    }

    check_conditions(task, conditions)?;

    match target.duration_seconds {
        Some(seconds) => {
            video_frame_count(seconds, limits.max_video_seconds)
                .map_err(|error| VideoInputError::invalid(RequestField::TargetDuration, error))?;
        }
        None => {
            let may_follow_audio = task == VideoTask::Ref2va
                && conditions
                    .iter()
                    .any(|condition| condition.condition_type != ConditionType::Image);
            if !may_follow_audio {
                return Err(VideoInputError::invalid(
                    RequestField::TargetDuration,
                    "is required unless a ref2va request has exactly one reference with audio",
                ));
            }
        }
    }
    Ok(())
}

/// Checks each condition's fields, then the task's counts and keyframe
/// order.
fn check_conditions(task: VideoTask, conditions: &[ConditionSpec]) -> Result<(), VideoInputError> {
    if task == VideoTask::T2va {
        if !conditions.is_empty() {
            return Err(VideoInputError::invalid(
                RequestField::Conditions,
                "t2va takes no conditions",
            ));
        }
        return Ok(());
    }

    for (index, condition) in conditions.iter().enumerate() {
        let reject = |message: String| Err(VideoInputError::condition(index, message));
        match condition.role {
            ConditionRole::Keyframe => {
                if condition.condition_type != ConditionType::Image {
                    return reject("a keyframe must be an image".to_owned());
                }
                match condition.frame_index {
                    None => return reject("a keyframe needs frame_index".to_owned()),
                    Some(0 | -1) => {}
                    Some(other) => {
                        return reject(format!("frame_index must be 0 or -1, got {other}"));
                    }
                }
            }
            ConditionRole::Reference => {
                if task == VideoTask::Fl2va {
                    return reject("fl2va takes keyframes only".to_owned());
                }
                if condition.frame_index.is_some() {
                    return reject("only a keyframe takes frame_index".to_owned());
                }
            }
        }
        if let Some(start) = condition.start_seconds {
            let video = matches!(
                condition.condition_type,
                ConditionType::Video | ConditionType::VideoAudio
            );
            if condition.role == ConditionRole::Keyframe || !video {
                return reject("start_time_seconds applies to video references only".to_owned());
            }
            if !start.is_finite() || start < 0.0 {
                return reject("start_time_seconds must be finite and >= 0".to_owned());
            }
        }
    }

    let signature: Vec<i64> = conditions
        .iter()
        .filter(|condition| condition.role == ConditionRole::Keyframe)
        .filter_map(|condition| condition.frame_index)
        .collect();
    let keyframes_valid = matches!(signature.as_slice(), [0] | [-1] | [0, -1]);
    if (task == VideoTask::Fl2va || !signature.is_empty()) && !keyframes_valid {
        return Err(VideoInputError::invalid(
            RequestField::Conditions,
            format!("keyframe frame_index values must be [0], [-1] or [0, -1], got {signature:?}"),
        ));
    }

    if task == VideoTask::Ref2va {
        let references = || {
            conditions
                .iter()
                .filter(|condition| condition.role == ConditionRole::Reference)
        };
        let count = |kind: fn(ConditionType) -> bool| {
            references()
                .filter(|condition| kind(condition.condition_type))
                .count()
        };
        let images = count(|kind| kind == ConditionType::Image);
        let audios = count(|kind| kind == ConditionType::Audio);
        let videos = count(|kind| matches!(kind, ConditionType::Video | ConditionType::VideoAudio));
        for (what, found, limit) in [
            ("image references", images, MAX_IMAGE_REFERENCES),
            ("video references", videos, MAX_VIDEO_REFERENCES),
            ("audio references", audios, MAX_AUDIO_REFERENCES),
            ("references", references().count(), MAX_REFERENCES),
        ] {
            if found > limit {
                return Err(VideoInputError::invalid(
                    RequestField::Conditions,
                    format!("at most {limit} {what}, got {found}"),
                ));
            }
        }
        if images + videos == 0 {
            return Err(VideoInputError::invalid(
                RequestField::Conditions,
                "ref2va needs at least one image or video reference",
            ));
        }
    }
    Ok(())
}

/// Checks that the probed media fits the condition's type.
fn check_media(
    index: usize,
    condition: &ConditionSpec,
    media: &MediaFacts,
) -> Result<(), VideoInputError> {
    let fits = matches!(
        (condition.condition_type, media),
        (ConditionType::Image, MediaFacts::Image(_))
            | (
                ConditionType::Video | ConditionType::VideoAudio,
                MediaFacts::Video(_)
            )
            | (ConditionType::Audio, MediaFacts::Audio(_))
    );
    if !fits {
        return Err(VideoInputError::condition(
            index,
            "the media does not match the condition type",
        ));
    }
    if let (ConditionType::VideoAudio, MediaFacts::Video(video)) = (condition.condition_type, media)
        && video.soundtrack.is_none()
    {
        return Err(VideoInputError::condition(
            index,
            "a video_audio reference needs an audio track",
        ));
    }
    Ok(())
}

/// The soundtrack a reference conditions on, if any.
fn soundtrack<'a>(condition: &ConditionSpec, media: &'a MediaFacts) -> Option<&'a AudioFacts> {
    if condition.role == ConditionRole::Keyframe {
        return None;
    }
    match media {
        MediaFacts::Audio(audio) => Some(audio),
        MediaFacts::Video(video) => video.soundtrack.as_ref(),
        MediaFacts::Image(_) => None,
    }
}

/// The generated frame count, from the target or from the one soundtrack.
fn resolve_frames(
    target: &Target<'_>,
    soundtracks: &[(usize, &AudioFacts, f64)],
    limits: &PlanLimits,
) -> Result<(f64, u32), VideoInputError> {
    if let Some(seconds) = target.duration_seconds {
        return video_frame_count(seconds, limits.max_video_seconds)
            .map(|frames| (seconds, frames))
            .map_err(|error| VideoInputError::invalid(RequestField::TargetDuration, error));
    }
    let [(index, track, start_seconds)] = soundtracks else {
        return Err(VideoInputError::invalid(
            RequestField::TargetDuration,
            format!(
                "is required unless exactly one reference has audio, got {}",
                soundtracks.len()
            ),
        ));
    };
    let rate = f64::from(track.sample_rate);
    let start_sample = start_offset(*start_seconds, rate);
    if start_sample >= track.samples {
        return Err(VideoInputError::condition(
            *index,
            "start_time_seconds lies past the end of the audio track",
        ));
    }
    let seconds = (track.samples - start_sample) as f64 / rate;
    video_frame_count(seconds, limits.max_video_seconds)
        .map(|frames| (seconds, frames))
        .map_err(|error| {
            VideoInputError::condition(
                *index,
                format!("the audio track sets the duration, but {error}"),
            )
        })
}

/// The generated canvas: from the ratio, or from the first keyframe.
fn target_canvas(
    task: VideoTask,
    target: &Target<'_>,
    media: &[MediaFacts],
) -> Result<Canvas, VideoInputError> {
    if let Some(size) = named_canvas(task, target)? {
        return Ok(size);
    }
    // fl2va with `auto`: the conditions are keyframes and the first one sets
    // the canvas through the canvas rule, at its 768-pixel short edge.
    if target.short_edge != CANVAS_SHORT_EDGE {
        return Err(VideoInputError::invalid(
            RequestField::TargetShortEdge,
            format!(
                "must be {CANVAS_SHORT_EDGE} for an auto aspect ratio, got {}",
                target.short_edge
            ),
        ));
    }
    let Some(MediaFacts::Image(first)) = media.first() else {
        return Err(VideoInputError::condition(
            0,
            "the first keyframe is not an image",
        ));
    };
    canvas(f64::from(first.width), f64::from(first.height)).map_err(|error| {
        VideoInputError::condition(
            0,
            format!("the first keyframe sets the canvas, but its {error}"),
        )
    })
}

fn plan_keyframe(
    index: usize,
    condition: &ConditionSpec,
    image: &ImageFacts,
    size: Canvas,
    order: usize,
    task: VideoTask,
    vision: &ProcessorConfig,
) -> Result<ConditionPlan, VideoInputError> {
    let position = if condition.frame_index == Some(0) {
        FramePosition::First
    } else {
        FramePosition::Last
    };
    // The first keyframe anchors the canvas and is stretched onto it; a
    // second one follows and is cover-cropped.
    let crop = (order > 0).then(|| cover_crop(image.width, image.height, size));
    let seen = if task == VideoTask::Fl2va {
        let grid = vision
            .image_grid(size.height, size.width)
            .map_err(|error| VideoInputError::condition(index, error))?;
        Some(Vision::Image(ImageVision {
            grid,
            tokens: vision.block_tokens(grid),
        }))
    } else {
        None
    };
    Ok(ConditionPlan {
        index,
        prepared: Prepared::Keyframe(KeyframeFit {
            position,
            cover_crop: crop,
        }),
        vision: seen,
        audio_rows: 0,
    })
}

fn plan_reference(
    index: usize,
    condition: &ConditionSpec,
    media: &MediaFacts,
    frames: u32,
    vision: &ProcessorConfig,
    config: &ReferenceConfig,
) -> Result<ConditionPlan, VideoInputError> {
    let reject = |error: String| VideoInputError::condition(index, error);
    let plan = |prepared, seen, audio_rows| ConditionPlan {
        index,
        prepared,
        vision: seen,
        audio_rows,
    };
    match media {
        MediaFacts::Image(image) => {
            let size = reference_image_size(image.width, image.height, config).map_err(reject)?;
            let grid = vision.image_grid(size.height, size.width).map_err(reject)?;
            let seen = Vision::Image(ImageVision {
                grid,
                tokens: vision.block_tokens(grid),
            });
            Ok(plan(Prepared::Image(size), Some(seen), 0))
        }
        MediaFacts::Audio(audio) => {
            let clip = audio_clip(audio, 0.0, frames).map_err(reject)?;
            Ok(plan(Prepared::Audio(clip), None, clip.rows()))
        }
        MediaFacts::Video(video) => {
            let start = condition.start_seconds.unwrap_or(0.0);
            let (clip, seen) =
                video_reference(video, start, frames, vision, config).map_err(reject)?;
            let audio_rows = clip.soundtrack.map_or(0, |track| track.rows());
            Ok(plan(
                Prepared::Video(clip),
                Some(Vision::Video(seen)),
                audio_rows,
            ))
        }
    }
}

/// Resolves every size a request implies from the probed media.
///
/// `media` holds the probed facts of each condition, in request order. The
/// caller applies the request-only rules of [`check_request`] first, which
/// need no media and so run before it is fetched.
///
/// # Errors
///
/// Returns [`VideoInputError::Invalid`] naming the first field at fault, and
/// [`VideoInputError::Internal`] when `media` and `conditions` differ in
/// length.
pub fn plan_request(
    task: VideoTask,
    target: &Target<'_>,
    conditions: &[ConditionSpec],
    media: &[MediaFacts],
    vision: &ProcessorConfig,
    limits: &PlanLimits,
) -> Result<RequestPlan, VideoInputError> {
    if media.len() != conditions.len() {
        return Err(VideoInputError::internal(format!(
            "{} conditions were planned with {} probed media",
            conditions.len(),
            media.len()
        )));
    }

    let mut soundtracks = Vec::new();
    for (index, (condition, facts)) in conditions.iter().zip(media).enumerate() {
        check_media(index, condition, facts)?;
        if let Some(track) = soundtrack(condition, facts) {
            soundtracks.push((index, track, condition.start_seconds.unwrap_or(0.0)));
        }
    }
    let (duration_seconds, frames) = resolve_frames(target, &soundtracks, limits)?;

    let size = target_canvas(task, target, media)?;
    check_served(size, limits)?;

    let mut plans = Vec::with_capacity(conditions.len());
    let mut keyframes = 0;
    for (index, (condition, facts)) in conditions.iter().zip(media).enumerate() {
        let plan = match (condition.role, facts) {
            (ConditionRole::Keyframe, MediaFacts::Image(image)) => {
                keyframes += 1;
                plan_keyframe(index, condition, image, size, keyframes - 1, task, vision)?
            }
            (ConditionRole::Keyframe, _) => {
                return Err(VideoInputError::condition(
                    index,
                    "a keyframe must be an image",
                ));
            }
            (ConditionRole::Reference, _) => {
                plan_reference(index, condition, facts, frames, vision, &limits.references)?
            }
        };
        plans.push(plan);
    }
    Ok(RequestPlan {
        task,
        canvas: size,
        duration_seconds,
        num_frames: frames,
        conditions: plans,
    })
}

#[cfg(test)]
pub(super) mod tests {
    use serde::Deserialize;
    use serde_json::{Value, json};

    use super::super::RequestField;
    use super::super::presentation::present;
    use super::super::presentation::tests::character_tokenizer;
    use super::super::probe::{AudioFacts, FrameRate, ImageFacts, MediaFacts, VideoFacts};
    use super::{
        AudioClip, Canvas, ConditionPlan, ConditionRole, ConditionSpec, ConditionType,
        FramePosition, KeyframeFit, PlanLimits, Prepared, ProcessorConfig, ReferenceConfig,
        RequestPlan, Target, VideoTask, Vision, audio_clip, canvas, check_request, cover_crop,
        latent_frames, plan_request, reference_image_size, rows_per_frame, video_reference,
    };

    /// The shared planning vectors, generated from the diffusers reference.
    pub(in crate::serving::video) fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../../../tests/python/fixtures/minimax_h3_plan.json"
        ))
        .unwrap()
    }

    pub(in crate::serving::video) fn vision(fixture: &Value) -> ProcessorConfig {
        ProcessorConfig::deserialize(&fixture["vision"]).unwrap()
    }

    /// The canvases of the vectors' accepted requests: what a deployment that
    /// prepares them all serves.
    pub(in crate::serving::video) fn served_canvases() -> Vec<Canvas> {
        let mut canvases = Vec::new();
        for case in fixture()["requests"].as_array().unwrap() {
            if let Some(expected) = case.get("expected") {
                let size = size(&expected["canvas"]);
                if !canvases.contains(&size) {
                    canvases.push(size);
                }
            }
        }
        canvases
    }

    /// Limits that serve every task and duration the model does, at the
    /// canvases of the vectors.
    pub(in crate::serving::video) fn limits() -> PlanLimits {
        PlanLimits {
            tasks: vec![VideoTask::T2va, VideoTask::Fl2va, VideoTask::Ref2va],
            max_video_seconds: 15.0,
            canvases: served_canvases(),
            // The fixture's cases follow the released pipeline.
            references: ReferenceConfig::RELEASED,
        }
    }

    fn pair(value: &Value) -> (u32, u32) {
        (
            value[0].as_u64().unwrap() as u32,
            value[1].as_u64().unwrap() as u32,
        )
    }

    fn size(value: &Value) -> Canvas {
        let (width, height) = pair(value);
        Canvas { width, height }
    }

    fn audio_facts(value: &Value) -> AudioFacts {
        AudioFacts {
            sample_rate: value["sample_rate"].as_u64().unwrap() as u32,
            samples: value["samples"].as_u64().unwrap(),
        }
    }

    fn video_facts(value: &Value) -> VideoFacts {
        let (numerator, denominator) = pair(&value["frame_rate"]);
        // The fixture states a video's displayed size as its aspect.
        let (width, height) = pair(&value["display"]);
        VideoFacts {
            display_width: value["display"][0].as_u64().unwrap(),
            display_height: value["display"][1].as_u64().unwrap(),
            width,
            height,
            frame_rate: FrameRate {
                numerator,
                denominator,
            },
            frames: value["frames"].as_u64().unwrap(),
            soundtrack: value
                .get("soundtrack")
                .filter(|track| !track.is_null())
                .map(audio_facts),
        }
    }

    fn media(value: &Value) -> MediaFacts {
        if value.get("width").is_some() {
            MediaFacts::Image(ImageFacts {
                width: value["width"].as_u64().unwrap() as u32,
                height: value["height"].as_u64().unwrap() as u32,
            })
        } else if value.get("display").is_some() {
            MediaFacts::Video(video_facts(value))
        } else {
            MediaFacts::Audio(audio_facts(value))
        }
    }

    /// The task, target, conditions and media of a vectors request.
    pub(in crate::serving::video) struct Request {
        pub(in crate::serving::video) task: VideoTask,
        pub(in crate::serving::video) aspect_ratio: String,
        pub(in crate::serving::video) target: (u32, Option<f64>),
        pub(in crate::serving::video) conditions: Vec<ConditionSpec>,
        pub(in crate::serving::video) media: Vec<MediaFacts>,
    }

    impl Request {
        pub(in crate::serving::video) fn parse(case: &Value) -> Self {
            let target = &case["target"];
            let conditions = case["conditions"].as_array().unwrap();
            Self {
                task: VideoTask::deserialize(&case["task"]).unwrap(),
                aspect_ratio: target["aspect_ratio"].as_str().unwrap().to_owned(),
                target: (
                    target["short_edge"].as_u64().unwrap() as u32,
                    target.get("duration_seconds").and_then(Value::as_f64),
                ),
                conditions: conditions
                    .iter()
                    .map(|entry| ConditionSpec {
                        condition_type: ConditionType::deserialize(&entry["type"]).unwrap(),
                        role: ConditionRole::deserialize(&entry["role"]).unwrap(),
                        frame_index: entry.get("frame_index").and_then(Value::as_i64),
                        start_seconds: entry.get("start_time_seconds").and_then(Value::as_f64),
                    })
                    .collect(),
                media: conditions
                    .iter()
                    .map(|entry| media(&entry["media"]))
                    .collect(),
            }
        }

        pub(in crate::serving::video) fn target(&self) -> Target<'_> {
            Target {
                short_edge: self.target.0,
                aspect_ratio: &self.aspect_ratio,
                duration_seconds: self.target.1,
            }
        }

        pub(in crate::serving::video) fn plan(
            &self,
            vision: &ProcessorConfig,
        ) -> Result<RequestPlan, super::super::VideoInputError> {
            check_request(self.task, &self.target(), &self.conditions, &limits())?;
            plan_request(
                self.task,
                &self.target(),
                &self.conditions,
                &self.media,
                vision,
                &limits(),
            )
        }
    }

    fn audio_json(clip: &AudioClip) -> Value {
        json!({
            "start_sample": clip.start_sample,
            "source_samples": clip.source_samples,
            "samples": clip.samples,
            "latents": clip.latents,
        })
    }

    /// A condition plan in the vectors' layout; `canvas` is the generated
    /// canvas. A condition's video rows are its encoder units' rows.
    fn condition_json(plan: &ConditionPlan, canvas: Canvas) -> Value {
        let vision = match &plan.vision {
            None => Value::Null,
            Some(Vision::Image(seen)) => json!({
                "grid": [seen.grid.t, seen.grid.h, seen.grid.w],
                "tokens": seen.tokens,
            }),
            Some(Vision::Video(seen)) => json!({
                "grid": [seen.grid.t, seen.grid.h, seen.grid.w],
                "block_tokens": seen.block_tokens,
                "frame_indices": seen.frame_indices,
                "block_timestamps": seen.block_timestamps,
            }),
        };
        let mut entry = json!({
            "index": plan.index,
            "video_rows": plan.latent_units(canvas, 1).iter().sum::<u32>(),
            "audio_rows": plan.audio_rows,
            "vision": vision,
        });
        let fields = match plan.prepared {
            Prepared::Keyframe(fit) => json!({
                "kind": "keyframe",
                "position": fit.position,
                "cover_crop": fit.cover_crop,
            }),
            Prepared::Image(size) => json!({
                "kind": "image",
                "resize": [size.width, size.height],
            }),
            Prepared::Audio(clip) => json!({"kind": "audio", "clip": audio_json(&clip)}),
            Prepared::Video(clip) => json!({
                "kind": "video",
                "canvas": [clip.canvas.width, clip.canvas.height],
                "start_frame": clip.start_frame,
                "clip_frames": clip.frames,
                "vae_frames": clip.vae_frames,
                "latent_frames": latent_frames(clip.vae_frames),
                "soundtrack": clip.soundtrack.as_ref().map(audio_json),
            }),
        };
        let entry_fields = entry.as_object_mut().unwrap();
        for (key, value) in fields.as_object().unwrap() {
            entry_fields.insert(key.clone(), value.clone());
        }
        entry
    }

    /// The canvas rule matches the reference's on named, probed and
    /// out-of-range aspects.
    #[test]
    fn canvases_match_the_reference() {
        let fixture = fixture();
        for case in fixture["canvas_cases"].as_array().unwrap() {
            let (width, height) = pair(&case["aspect"]);
            let resolved = canvas(f64::from(width), f64::from(height));
            if case.get("error").is_some() {
                assert!(resolved.is_err(), "{case}");
            } else {
                assert_eq!(resolved.unwrap(), size(&case["canvas"]), "{case}");
            }
        }
    }

    /// Durations align up to `17n + 5` frames within the served range, with
    /// the video latent count the reference derives from them.
    #[test]
    fn durations_match_the_reference() {
        let fixture = fixture();
        let vision = vision(&fixture);
        for case in fixture["duration_cases"].as_array().unwrap() {
            let seconds = case["seconds"].as_f64().unwrap();
            let target = Target {
                short_edge: 768,
                aspect_ratio: "auto",
                duration_seconds: Some(seconds),
            };
            let plan = plan_request(VideoTask::T2va, &target, &[], &[], &vision, &limits());
            if case.get("error").is_some() {
                let field = plan.unwrap_err().field();
                assert_eq!(field, Some(RequestField::TargetDuration), "{case}");
                continue;
            }
            let plan = plan.unwrap();
            assert_eq!(u64::from(plan.num_frames), case["num_frames"], "{case}");
            assert_eq!(
                u64::from(latent_frames(plan.num_frames)),
                case["latent_frames"],
                "{case}"
            );
        }
    }

    /// Under the released sizing, image references resize to a 2048-pixel
    /// short edge and enter the conditioner at the processor's grid.
    #[test]
    fn reference_images_match_the_reference() {
        let fixture = fixture();
        let vision = vision(&fixture);
        for case in fixture["reference_image_cases"].as_array().unwrap() {
            let (width, height) = pair(&case["size"]);
            let resized = reference_image_size(width, height, &ReferenceConfig::RELEASED);
            if case.get("error").is_some() {
                assert!(resized.is_err(), "{case}");
                continue;
            }
            let resized = resized.unwrap();
            assert_eq!(resized, size(&case["resize"]), "{case}");
            let grid = vision.image_grid(resized.height, resized.width).unwrap();
            assert_eq!(
                json!([grid.t, grid.h, grid.w]),
                case["vision_grid"],
                "{case}"
            );
            assert_eq!(
                u64::from(vision.block_tokens(grid)),
                case["vision_tokens"],
                "{case}"
            );
            assert_eq!(u64::from(rows_per_frame(resized)), case["rows"], "{case}");
        }
    }

    /// A following keyframe's cover crop reproduces the reference's resize
    /// and crop box.
    #[test]
    fn keyframe_crops_match_the_reference() {
        let fixture = fixture();
        for case in fixture["keyframe_cases"].as_array().unwrap() {
            let (width, height) = pair(&case["size"]);
            let crop = cover_crop(width, height, size(&case["canvas"]));
            assert_eq!(json!(crop), case["cover_crop"], "{case}");
        }
    }

    /// A video's encoded frames split into the video encoder's 17-frame
    /// windows: each complete window yields five latent frames of rows and
    /// the padded last one two, which together are the condition's rows. A
    /// still image is one unit.
    #[test]
    fn latent_units_partition_the_condition_rows() {
        let canvas = Canvas {
            width: 1344,
            height: 768,
        };
        let video = ConditionPlan {
            index: 0,
            prepared: Prepared::Video(super::VideoClip {
                canvas,
                start_frame: 0,
                frames: 124,
                vae_frames: 124,
                soundtrack: None,
            }),
            vision: None,
            audio_rows: 0,
        };
        let units = video.latent_units(canvas, 4);
        assert_eq!(units, [vec![5 * 1008; 7], vec![2 * 1008]].concat());
        assert_eq!(units.iter().sum::<u32>(), 37 * 1008);

        let image = ConditionPlan {
            prepared: Prepared::Image(canvas),
            ..video
        };
        assert_eq!(image.latent_units(canvas, 1), [1008]);
    }

    /// A reference image's bands are runs of whole 32-pixel patch rows dealt
    /// as evenly as possible, together its rows; a keyframe stays one unit.
    #[test]
    fn reference_image_bands_partition_its_patch_rows() {
        // A 16:9 reference at its 2048-pixel short edge: 64 patch rows of
        // 114 patches.
        let size = Canvas {
            width: 3648,
            height: 2048,
        };
        let image = ConditionPlan {
            index: 0,
            prepared: Prepared::Image(size),
            vision: None,
            audio_rows: 0,
        };
        assert_eq!(image.latent_units(size, 4), [16 * 114; 4]);
        assert_eq!(image.latent_units(size, 3), [21 * 114, 21 * 114, 22 * 114]);
        assert_eq!(image.latent_units(size, 0), [64 * 114]);
        assert_eq!(image.latent_units(size, 100), [114; 64]);

        let keyframe = ConditionPlan {
            prepared: Prepared::Keyframe(KeyframeFit {
                position: FramePosition::First,
                cover_crop: None,
            }),
            ..image.clone()
        };
        assert_eq!(keyframe.latent_units(size, 4), [rows_per_frame(size)]);
    }

    /// A reference is only ever reduced unless the deployment enlarges: an
    /// image above the short edge and a video above the canvas of its aspect
    /// scale down as the released pipeline scales them, and smaller ones keep
    /// their own size on the 32-pixel grid.
    #[test]
    fn references_are_enlarged_only_on_request() {
        let sized = |width, height| Canvas { width, height };
        let unenlarged = ReferenceConfig::default();
        let image = |width, height, config: &ReferenceConfig| {
            reference_image_size(width, height, config).unwrap()
        };
        // 1080 rounds to 34 patches; 768 is already on the grid.
        assert_eq!(image(1920, 1080, &unenlarged), sized(1920, 1088));
        assert_eq!(image(768, 768, &unenlarged), sized(768, 768));
        assert_eq!(
            image(768, 768, &ReferenceConfig::RELEASED),
            sized(2048, 2048)
        );
        // A 2304-pixel short edge exceeds 2048 and is reduced either way.
        for config in [&unenlarged, &ReferenceConfig::RELEASED] {
            assert_eq!(image(4096, 2304, config), sized(3648, 2048));
        }
        // A smaller short edge reduces further and still never enlarges.
        let half = ReferenceConfig {
            image_short_edge: 1024,
            ..unenlarged
        };
        assert_eq!(image(2560, 1440, &half), sized(1824, 1024));
        assert_eq!(image(768, 768, &half), sized(768, 768));
        // The 1:4 to 4:1 bound holds at any size.
        assert!(reference_image_size(4100, 1000, &unenlarged).is_err());

        let fixture = fixture();
        let vision = vision(&fixture);
        let video = |width: u32, height: u32, config: &ReferenceConfig| {
            let facts = VideoFacts {
                display_width: u64::from(width),
                display_height: u64::from(height),
                width,
                height,
                frame_rate: FrameRate {
                    numerator: 24,
                    denominator: 1,
                },
                frames: 48,
                soundtrack: None,
            };
            video_reference(&facts, 0.0, 124, &vision, config)
                .unwrap()
                .0
                .canvas
        };
        // 720p holds fewer pixels than the 16:9 canvas: 720 is 22.5 patches
        // and rounds half to even. 1080p is reduced to the canvas.
        assert_eq!(video(1280, 720, &unenlarged), sized(1280, 704));
        assert_eq!(video(1920, 1080, &unenlarged), sized(1344, 768));
        assert_eq!(
            video(1280, 720, &ReferenceConfig::RELEASED),
            sized(1344, 768)
        );
    }

    /// A reference config reads the processor options a deployment states,
    /// absent ones keeping their defaults, and refuses an option the
    /// processor does not take; its short edge lies on the 32-pixel grid up
    /// to the released checkpoint's.
    #[test]
    fn a_reference_config_reads_processor_options() {
        let read = |options: Value| ReferenceConfig::from_kwargs(options.as_object().unwrap());
        assert_eq!(read(json!({})), Ok(ReferenceConfig::default()));
        assert_eq!(
            read(json!({"enlarge_references": true})),
            Ok(ReferenceConfig::RELEASED)
        );
        assert_eq!(
            read(json!({"reference_image_short_edge": 1024})),
            Ok(ReferenceConfig {
                image_short_edge: 1024,
                enlarge: false,
            })
        );
        assert!(read(json!({"num_crops": 4})).is_err());
        assert!(read(json!({"enlarge_references": "yes"})).is_err());
        assert!(read(json!({"reference_image_short_edge": 1000})).is_err());

        for (edge, valid) in [
            (2048, true),
            (32, true),
            (0, false),
            (1000, false),
            (4096, false),
        ] {
            let config = ReferenceConfig {
                image_short_edge: edge,
                enlarge: false,
            };
            assert_eq!(config.validate().is_ok(), valid, "{edge}");
        }
    }

    /// Reference videos resample to 24 fps, honour the start offset, keep a
    /// whole VAE window, and sample the reference's 2 fps vision blocks.
    #[test]
    fn reference_videos_match_the_reference() {
        let fixture = fixture();
        let vision = vision(&fixture);
        for case in fixture["video_reference_cases"].as_array().unwrap() {
            let facts = video_facts(case);
            let start = case["start_seconds"].as_f64().unwrap();
            let frames = case["num_frames"].as_u64().unwrap() as u32;
            let planned =
                video_reference(&facts, start, frames, &vision, &ReferenceConfig::RELEASED);
            if case.get("error").is_some() {
                assert!(planned.is_err(), "{case}");
                continue;
            }
            let (clip, seen) = planned.unwrap();
            assert_eq!(clip.canvas, size(&case["canvas"]), "{case}");
            assert_eq!(u64::from(clip.start_frame), case["start_frame"], "{case}");
            assert_eq!(u64::from(clip.frames), case["clip_frames"], "{case}");
            assert_eq!(u64::from(clip.vae_frames), case["vae_frames"], "{case}");
            let latents = latent_frames(clip.vae_frames);
            assert_eq!(u64::from(latents), case["latent_frames"], "{case}");
            assert_eq!(json!(seen.frame_indices), case["frame_indices"], "{case}");
            assert_eq!(
                json!(seen.block_timestamps),
                case["block_timestamps"],
                "{case}"
            );
            assert_eq!(
                json!([seen.grid.t, seen.grid.h, seen.grid.w]),
                case["vision_grid"]
            );
            assert_eq!(u64::from(seen.block_tokens), case["block_tokens"], "{case}");
            let rows = latents * rows_per_frame(clip.canvas);
            assert_eq!(u64::from(rows), case["rows"], "{case}");
        }
    }

    /// Soundtracks truncate at their native rate, resample to 32 kHz and pad
    /// to whole 800-sample latents.
    #[test]
    fn audio_clips_match_the_reference() {
        let fixture = fixture();
        for case in fixture["audio_cases"].as_array().unwrap() {
            let clip = audio_clip(
                &audio_facts(case),
                case["start_seconds"].as_f64().unwrap(),
                case["num_frames"].as_u64().unwrap() as u32,
            )
            .unwrap();
            assert_eq!(audio_json(&clip), case["clip"], "{case}");
        }
    }

    /// Whole requests plan to the reference's canvas, frame counts, condition
    /// preparation, vision grids and rows.
    #[test]
    fn request_plans_match_the_reference() {
        let fixture = fixture();
        let vision = vision(&fixture);
        for case in fixture["requests"].as_array().unwrap() {
            let Some(expected) = case.get("expected") else {
                continue;
            };
            let name = &case["name"];
            let plan = Request::parse(case).plan(&vision).unwrap();
            assert_eq!(plan.canvas, size(&expected["canvas"]), "{name}");
            assert_eq!(u64::from(plan.num_frames), expected["num_frames"], "{name}");
            let conditions: Vec<Value> = plan
                .conditions
                .iter()
                .map(|condition| condition_json(condition, plan.canvas))
                .collect();
            assert_eq!(Value::from(conditions), expected["conditions"], "{name}");
        }
    }

    /// Each rejected request names the field the serving contract blames,
    /// whether planning or the presentation rejects it.
    #[test]
    fn rejected_requests_name_the_field() {
        let fixture = fixture();
        let vision = vision(&fixture);
        let tokenizer = character_tokenizer();
        for case in fixture["requests"].as_array().unwrap() {
            let Some(error) = case.get("error") else {
                continue;
            };
            let rejection = match Request::parse(case).plan(&vision) {
                Ok(plan) => present(&tokenizer, &plan, case["prompt"].as_str().unwrap())
                    .expect_err("the request was accepted"),
                Err(rejection) => rejection,
            };
            assert_eq!(
                rejection.field().unwrap().to_string(),
                error["field"],
                "{}",
                case["name"]
            );
        }
    }

    const FIVE_SECONDS: Target<'static> = Target {
        short_edge: 768,
        aspect_ratio: "auto",
        duration_seconds: Some(5.0),
    };

    /// A deployment serves only the canvases its workers prepare; others are
    /// rejected on the aspect ratio before any media is needed.
    #[test]
    fn prepared_canvases_restrict_the_target() {
        let fixture = fixture();
        let vision = vision(&fixture);
        let limits = PlanLimits {
            canvases: vec![Canvas {
                width: 1344,
                height: 768,
            }],
            ..limits()
        };
        let plan =
            plan_request(VideoTask::T2va, &FIVE_SECONDS, &[], &[], &vision, &limits).unwrap();
        assert_eq!((plan.canvas.width, plan.canvas.height), (1344, 768));

        let portrait = Target {
            aspect_ratio: "9:16",
            ..FIVE_SECONDS
        };
        let error = super::check_request(VideoTask::T2va, &portrait, &[], &limits).unwrap_err();
        assert_eq!(error.field(), Some(RequestField::TargetAspectRatio));
    }

    /// A deployment serving 480p buckets accepts their short edge: a named
    /// ratio resolves to the training bucket rather than a rescaled canvas
    /// rule, and a short edge or bucket the deployment did not prepare is
    /// refused before any media is needed.
    #[test]
    fn served_buckets_resolve_each_short_edge() {
        let fixture = fixture();
        let vision = vision(&fixture);
        let canvas = |width, height| Canvas { width, height };
        let served = PlanLimits {
            canvases: vec![canvas(1344, 768), canvas(992, 416), canvas(832, 480)],
            ..limits()
        };
        assert_eq!(served.short_edges(), vec![768, 480]);
        for (short_edge, aspect_ratio, expected) in [
            (768, "16:9", canvas(1344, 768)),
            (480, "21:9", canvas(992, 416)),
            // `auto` is 16:9 for t2va.
            (480, "auto", canvas(832, 480)),
        ] {
            let target = Target {
                short_edge,
                aspect_ratio,
                ..FIVE_SECONDS
            };
            let plan = plan_request(VideoTask::T2va, &target, &[], &[], &vision, &served).unwrap();
            assert_eq!(plan.canvas, expected, "{short_edge} {aspect_ratio}");
        }
        for (short_edge, aspect_ratio, field) in [
            (720, "16:9", RequestField::TargetShortEdge),
            (768, "21:9", RequestField::TargetAspectRatio),
            (480, "1:1", RequestField::TargetAspectRatio),
        ] {
            let target = Target {
                short_edge,
                aspect_ratio,
                ..FIVE_SECONDS
            };
            let error = super::check_request(VideoTask::T2va, &target, &[], &served).unwrap_err();
            assert_eq!(error.field(), Some(field), "{short_edge} {aspect_ratio}");
        }

        // A deployment that prepares only 768p canvases serves no other
        // short edge.
        let target = Target {
            short_edge: 480,
            aspect_ratio: "16:9",
            ..FIVE_SECONDS
        };
        let error = super::check_request(VideoTask::T2va, &target, &[], &limits()).unwrap_err();
        assert_eq!(error.field(), Some(RequestField::TargetShortEdge));
    }

    /// The served maximum duration narrows the model's own range.
    #[test]
    fn max_video_seconds_bounds_the_duration() {
        let ten_seconds = Target {
            duration_seconds: Some(10.0),
            ..FIVE_SECONDS
        };
        let limits = PlanLimits {
            max_video_seconds: 8.0,
            ..limits()
        };
        let error = super::check_request(VideoTask::T2va, &ten_seconds, &[], &limits).unwrap_err();
        assert_eq!(error.field(), Some(RequestField::TargetDuration));
    }

    /// A task outside the placed denoiser's set is rejected first.
    #[test]
    fn only_served_tasks_are_accepted() {
        let limits = PlanLimits {
            tasks: vec![VideoTask::Ref2va],
            ..limits()
        };
        let error = super::check_request(VideoTask::T2va, &FIVE_SECONDS, &[], &limits).unwrap_err();
        assert_eq!(error.field(), Some(RequestField::Task));
    }

    /// Aspect ratios are accepted only in canonical `W:H` form.
    #[test]
    fn aspect_ratios_must_be_canonical() {
        for (ratio, accepted) in [
            ("16:9", true),
            ("7:4", true),
            ("016:9", false),
            ("+16:9", false),
            ("16:9:1", false),
            ("16/9", false),
            ("0:9", false),
            ("", false),
        ] {
            let target = Target {
                short_edge: 768,
                aspect_ratio: ratio,
                duration_seconds: Some(5.0),
            };
            let keyframe = ConditionSpec {
                condition_type: ConditionType::Image,
                role: ConditionRole::Keyframe,
                frame_index: Some(0),
                start_seconds: None,
            };
            let checked = super::check_request(VideoTask::Fl2va, &target, &[keyframe], &limits());
            assert_eq!(checked.is_ok(), accepted, "{ratio:?}");
        }
    }
}
