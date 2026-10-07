//! The conditions of a video request, as the workers read and encode them.
//!
//! The server plans every condition before admission (`serving::video::plan`
//! in `uniserve-server`): how its media is prepared, what the conditioner
//! reads of it, and the denoiser rows it adds. A [`VideoCondition`] carries
//! that plan to the engine and on to the workers: the shared-memory object
//! holding the fetched bytes, the exact media the worker's media reader
//! decodes them into, the conditioner's view, and the rows the latent
//! encoders produce. The engine sizes the products of the calls that read
//! and encode conditions from these values alone; it never decodes media.

use serde::{Deserialize, Serialize};

use crate::Canvas;

/// How a condition relates to the generated video.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditionRole {
    /// A keyframe anchoring the first generated frame.
    FirstFrame,
    /// A keyframe anchoring the last generated frame.
    LastFrame,
    /// A reference conditioning the generation as a whole.
    Reference,
}

impl ConditionRole {
    /// Whether the condition anchors a generated frame.
    pub const fn is_keyframe(self) -> bool {
        matches!(self, Self::FirstFrame | Self::LastFrame)
    }
}

/// Where a condition's fetched media bytes are: a POSIX shared-memory object
/// on the host that runs the server and the media reader.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MediaLocator {
    /// Object name without its leading `/`.
    pub name: String,
    /// Published byte count.
    pub bytes: u64,
}

/// A decoded still image put on a raster: resized with LANCZOS to `resized`,
/// then the `size` window whose top-left corner is (`left`, `top`) kept.
///
/// A stretched image is resized straight to its raster (`resized == size`,
/// no offset); a cover-cropped keyframe is resized past the canvas and
/// centred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ImageFit {
    /// Size the decoded image is resized to.
    pub resized: Canvas,
    /// Left edge of the kept window in the resized image.
    pub left: u32,
    /// Top edge of the kept window in the resized image.
    pub top: u32,
    /// The kept raster, which the video encoder encodes.
    pub size: Canvas,
}

/// The frames of a reference video the media reader produces.
///
/// The video is resampled to 24 fps and scaled onto `canvas` in one decoding
/// pass; its first `start_frame` frames are skipped and the next `frames`
/// kept. The conditioner samples the kept frames; the video encoder encodes
/// their leading `vae_frames`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VideoClip {
    /// The canvas of the video's own display aspect.
    pub canvas: Canvas,
    /// Frames of the 24 fps timeline skipped at the start.
    pub start_frame: u32,
    /// Frames of the 24 fps timeline kept after the start offset.
    pub frames: u32,
    /// The leading kept frames the video encoder encodes.
    pub vae_frames: u32,
}

/// The samples of a soundtrack or audio reference the media reader produces.
///
/// The track is decoded at its native rate, its first `start_sample` samples
/// skipped and the next `source_samples` kept, then resampled once to the
/// model's audio rate, giving `samples` stereo samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AudioClip {
    /// The track's native sample rate.
    pub sample_rate: u32,
    /// Native samples skipped at the start.
    pub start_sample: u64,
    /// Native samples kept after the start offset.
    pub source_samples: u64,
    /// Samples after resampling to the model's audio rate.
    pub samples: u64,
}

/// What the media reader decodes a condition's bytes into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditionMedia {
    /// A keyframe or an image reference: one frame.
    Image(ImageFit),
    /// A reference video and, when the request conditions on it, its
    /// soundtrack.
    Video {
        /// The frames.
        clip: VideoClip,
        /// The soundtrack.
        soundtrack: Option<AudioClip>,
    },
    /// An audio reference.
    Audio(AudioClip),
}

impl ConditionMedia {
    /// The audio track the condition carries, if any.
    pub const fn audio(&self) -> Option<&AudioClip> {
        match self {
            Self::Image(_) => None,
            Self::Video { soundtrack, .. } => soundtrack.as_ref(),
            Self::Audio(clip) => Some(clip),
        }
    }

    /// The frame count and raster of the pixels the video encoder encodes,
    /// if the condition has any.
    pub const fn pixels(&self) -> Option<(u32, Canvas)> {
        match self {
            Self::Image(fit) => Some((1, fit.size)),
            Self::Video { clip, .. } => Some((clip.vae_frames, clip.canvas)),
            Self::Audio(_) => None,
        }
    }
}

/// A vision patch grid: `t` temporal patches of `h` by `w` spatial patches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VisionGrid {
    /// Temporal patches; 1 for an image.
    pub t: u32,
    /// Patch rows.
    pub h: u32,
    /// Patch columns.
    pub w: u32,
}

impl VisionGrid {
    /// Patches of the grid.
    pub const fn patches(&self) -> u64 {
        self.t as u64 * self.h as u64 * self.w as u64
    }
}

/// What the conditioner reads of a visual condition.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConditionVision {
    /// The patch grid the condition's frames are resized to.
    pub grid: VisionGrid,
    /// Vision tokens the condition occupies in the presentation, which is
    /// the count of its vision placeholders.
    pub tokens: u32,
    /// The kept frames of a reference video the conditioner samples, in
    /// order; empty for an image.
    pub frame_indices: Vec<u32>,
}

/// One condition of a video request.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VideoCondition {
    /// Keyframe anchor or reference.
    pub role: ConditionRole,
    /// The fetched media bytes.
    pub source: MediaLocator,
    /// What the media reader decodes them into.
    pub media: ConditionMedia,
    /// What the conditioner reads, for a condition it reads.
    pub vision: Option<ConditionVision>,
    /// Denoiser rows each temporal unit of the condition's pixels encodes
    /// to, in unit order: one unit for a still image, one per video encoder
    /// window for a video. Empty for an audio reference.
    pub latent_units: Vec<u32>,
    /// Denoiser audio rows of the condition's audio track; zero without one.
    pub audio_rows: u32,
}

impl VideoCondition {
    /// Denoiser video rows of the condition.
    pub fn video_rows(&self) -> u64 {
        self.latent_units.iter().map(|rows| u64::from(*rows)).sum()
    }

    /// Bytes of the RGB24 pixels the video encoder encodes.
    pub fn pixel_bytes(&self) -> u64 {
        self.media.pixels().map_or(0, |(frames, canvas)| {
            u64::from(frames) * u64::from(canvas.width) * u64::from(canvas.height) * 3
        })
    }

    /// Checks the condition's internal consistency: a keyframe is one still
    /// image the conditioner may read; a condition with pixels encodes to
    /// positive rows in at least one unit and one without has none; a
    /// condition with an audio track encodes to positive audio rows and one
    /// without has none; a video's conditioner view samples kept frames in
    /// increasing order and an image's samples none; every extent is
    /// positive.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.source.name.is_empty() || self.source.name.contains('/') || self.source.bytes == 0 {
            return Err("a condition's media locator is invalid");
        }
        if self.role.is_keyframe() && !matches!(self.media, ConditionMedia::Image(_)) {
            return Err("a keyframe is one still image");
        }
        let canvases_positive = |canvas: Canvas| canvas.width > 0 && canvas.height > 0;
        match &self.media {
            ConditionMedia::Image(fit) => {
                if !canvases_positive(fit.size)
                    || !canvases_positive(fit.resized)
                    || u64::from(fit.left) + u64::from(fit.size.width)
                        > u64::from(fit.resized.width)
                    || u64::from(fit.top) + u64::from(fit.size.height)
                        > u64::from(fit.resized.height)
                {
                    return Err("an image fit keeps a window outside its resized image");
                }
            }
            ConditionMedia::Video { clip, .. } => {
                if !canvases_positive(clip.canvas)
                    || clip.vae_frames == 0
                    || clip.vae_frames > clip.frames
                {
                    return Err("a video clip encodes frames it does not keep");
                }
            }
            ConditionMedia::Audio(_) => {}
        }
        if let Some(track) = self.media.audio()
            && (track.sample_rate == 0 || track.source_samples == 0 || track.samples == 0)
        {
            return Err("an audio track keeps no samples");
        }
        let has_pixels = self.media.pixels().is_some();
        if has_pixels == self.latent_units.is_empty() || self.latent_units.contains(&0) {
            return Err("a condition's latent units disagree with its pixels");
        }
        if self.media.audio().is_some() != (self.audio_rows > 0) {
            return Err("a condition's audio rows disagree with its audio track");
        }
        if let Some(vision) = &self.vision {
            let video = match &self.media {
                ConditionMedia::Video { clip, .. } => Some(clip),
                ConditionMedia::Image(_) => None,
                ConditionMedia::Audio(_) => {
                    return Err("the conditioner reads no audio reference");
                }
            };
            if vision.grid.patches() == 0 || vision.tokens == 0 {
                return Err("a conditioner view covers no patches");
            }
            match video {
                None if !vision.frame_indices.is_empty() || vision.grid.t != 1 => {
                    return Err("an image's conditioner view samples one frame");
                }
                Some(clip)
                    if vision.frame_indices.is_empty()
                        || !vision
                            .frame_indices
                            .windows(2)
                            .all(|pair| pair[0] < pair[1])
                        || vision
                            .frame_indices
                            .last()
                            .is_some_and(|last| *last >= clip.frames) =>
                {
                    return Err("a video's conditioner view samples frames it does not keep");
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(role: ConditionRole) -> VideoCondition {
        let canvas = Canvas {
            width: 64,
            height: 32,
        };
        VideoCondition {
            role,
            source: MediaLocator {
                name: "uniserve-media-1".to_owned(),
                bytes: 10,
            },
            media: ConditionMedia::Image(ImageFit {
                resized: canvas,
                left: 0,
                top: 0,
                size: canvas,
            }),
            vision: Some(ConditionVision {
                grid: VisionGrid { t: 1, h: 2, w: 4 },
                tokens: 2,
                frame_indices: Vec::new(),
            }),
            latent_units: vec![2],
            audio_rows: 0,
        }
    }

    /// A still image keyframe or reference is consistent, and its sizes
    /// follow from its raster and units.
    #[test]
    fn an_image_condition_is_consistent() {
        let condition = image(ConditionRole::FirstFrame);
        assert_eq!(condition.validate(), Ok(()));
        assert_eq!(condition.video_rows(), 2);
        assert_eq!(condition.pixel_bytes(), 64 * 32 * 3);
    }

    /// Inconsistent conditions are refused: a keyframe that is not an image,
    /// rows without pixels, and a video view sampling frames past the clip.
    #[test]
    fn inconsistent_conditions_are_refused() {
        let mut audio = image(ConditionRole::FirstFrame);
        audio.media = ConditionMedia::Audio(AudioClip {
            sample_rate: 48_000,
            start_sample: 0,
            source_samples: 48_000,
            samples: 32_000,
        });
        assert!(audio.validate().is_err());

        audio.role = ConditionRole::Reference;
        audio.vision = None;
        assert!(audio.validate().is_err(), "rows without pixels");
        audio.latent_units.clear();
        audio.audio_rows = 80;
        assert_eq!(audio.validate(), Ok(()));

        let mut video = image(ConditionRole::Reference);
        video.media = ConditionMedia::Video {
            clip: VideoClip {
                canvas: Canvas {
                    width: 64,
                    height: 32,
                },
                start_frame: 0,
                frames: 24,
                vae_frames: 22,
            },
            soundtrack: None,
        };
        video.vision = Some(ConditionVision {
            grid: VisionGrid { t: 1, h: 2, w: 4 },
            tokens: 2,
            frame_indices: vec![0, 12, 24],
        });
        assert!(video.validate().is_err(), "frame 24 is not kept");
    }
}
