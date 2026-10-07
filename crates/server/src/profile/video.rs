//! MiniMax H3 output rasters: the trained resolution and aspect-ratio buckets.
//!
//! A deployment provisions the cartesian product of its configured
//! resolutions and aspect ratios; the worker prepares every product raster at
//! startup and a request selects one of them. Every bucket side is a multiple
//! of 32 pixels, one latent token (VAE stride 16, patch 2).

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::profile::omni::resolution::ResolutionName;

/// Short-side resolution class of an H3 output raster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VideoResolution {
    /// 768-pixel class, the checkpoint's native training resolution.
    #[serde(rename = "768p")]
    P768,
    /// 480-pixel class.
    #[serde(rename = "480p")]
    P480,
}

impl VideoResolution {
    /// Every resolution class the checkpoint was trained on.
    pub const ALL: [Self; 2] = [Self::P768, Self::P480];

    /// Returns the public name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::P768 => "768p",
            Self::P480 => "480p",
        }
    }
}

impl fmt::Display for VideoResolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for VideoResolution {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|resolution| resolution.as_str() == value)
            .ok_or_else(|| format!("unknown video resolution {value:?}; expected 768p or 480p"))
    }
}

/// Aspect ratios of the checkpoint's training buckets, landscape first.
pub const VIDEO_ASPECT_RATIOS: [ResolutionName; 6] = [
    ResolutionName::Landscape21x9,
    ResolutionName::Landscape16x9,
    ResolutionName::Landscape4x3,
    ResolutionName::Square,
    ResolutionName::Portrait3x4,
    ResolutionName::Portrait9x16,
];

/// One trained output raster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VideoRaster {
    /// Resolution class.
    pub resolution: VideoResolution,
    /// Aspect ratio.
    pub aspect_ratio: ResolutionName,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

impl VideoRaster {
    /// Returns the training bucket for a resolution and aspect ratio, or
    /// `None` when the checkpoint has no such bucket.
    pub const fn new(resolution: VideoResolution, aspect_ratio: ResolutionName) -> Option<Self> {
        let (width, height) = match (resolution, aspect_ratio) {
            (VideoResolution::P768, ResolutionName::Landscape21x9) => (1536, 672),
            (VideoResolution::P768, ResolutionName::Landscape16x9) => (1344, 768),
            (VideoResolution::P768, ResolutionName::Landscape4x3) => (1024, 768),
            (VideoResolution::P768, ResolutionName::Square) => (768, 768),
            (VideoResolution::P768, ResolutionName::Portrait3x4) => (768, 1024),
            (VideoResolution::P768, ResolutionName::Portrait9x16) => (768, 1344),
            (VideoResolution::P480, ResolutionName::Landscape21x9) => (992, 416),
            (VideoResolution::P480, ResolutionName::Landscape16x9) => (832, 480),
            (VideoResolution::P480, ResolutionName::Landscape4x3) => (640, 480),
            (VideoResolution::P480, ResolutionName::Square) => (480, 480),
            (VideoResolution::P480, ResolutionName::Portrait3x4) => (480, 640),
            (VideoResolution::P480, ResolutionName::Portrait9x16) => (480, 832),
            _ => return None,
        };
        Some(Self {
            resolution,
            aspect_ratio,
            width,
            height,
        })
    }
}

/// The rasters a deployment serves: every configured resolution crossed with
/// every configured aspect ratio, resolution-major.
///
/// The first resolution and the first aspect ratio are the request defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoRasters {
    resolutions: Vec<VideoResolution>,
    aspect_ratios: Vec<ResolutionName>,
}

impl Default for VideoRasters {
    /// 768p landscape and portrait, the rasters every graph-captured
    /// deployment can hold.
    fn default() -> Self {
        Self {
            resolutions: vec![VideoResolution::P768],
            aspect_ratios: vec![ResolutionName::Landscape16x9, ResolutionName::Portrait9x16],
        }
    }
}

impl VideoRasters {
    /// Validates a deployment's raster selection.
    ///
    /// # Errors
    ///
    /// Returns a message when either list is empty, repeats an entry, or names
    /// an aspect ratio with no training bucket.
    pub fn new(
        resolutions: &[VideoResolution],
        aspect_ratios: &[ResolutionName],
    ) -> Result<Self, String> {
        if resolutions.is_empty() || aspect_ratios.is_empty() {
            return Err("video resolutions and aspect ratios must not be empty".to_owned());
        }
        for (index, resolution) in resolutions.iter().enumerate() {
            if resolutions[..index].contains(resolution) {
                return Err(format!("video resolution {resolution} is repeated"));
            }
        }
        for (index, aspect_ratio) in aspect_ratios.iter().enumerate() {
            if !VIDEO_ASPECT_RATIOS.contains(aspect_ratio) {
                return Err(format!(
                    "video aspect ratio {} has no trained bucket; expected one of {}",
                    aspect_ratio.as_str(),
                    join(VIDEO_ASPECT_RATIOS.iter().map(|name| name.as_str()))
                ));
            }
            if aspect_ratios[..index].contains(aspect_ratio) {
                return Err(format!(
                    "video aspect ratio {} is repeated",
                    aspect_ratio.as_str()
                ));
            }
        }
        Ok(Self {
            resolutions: resolutions.to_vec(),
            aspect_ratios: aspect_ratios.to_vec(),
        })
    }

    /// Configured resolutions, the default first.
    pub fn resolutions(&self) -> &[VideoResolution] {
        &self.resolutions
    }

    /// Configured aspect ratios, the default first.
    pub fn aspect_ratios(&self) -> &[ResolutionName] {
        &self.aspect_ratios
    }

    /// Every served raster, resolution-major.
    pub fn rasters(&self) -> impl Iterator<Item = VideoRaster> + '_ {
        self.resolutions.iter().flat_map(|&resolution| {
            self.aspect_ratios
                .iter()
                .filter_map(move |&aspect_ratio| VideoRaster::new(resolution, aspect_ratio))
        })
    }

    /// Selects a served raster; an omitted field takes its configured default.
    ///
    /// # Errors
    ///
    /// Returns a message naming the served values when the deployment does
    /// not provision the requested resolution or aspect ratio.
    pub fn select(
        &self,
        resolution: Option<VideoResolution>,
        aspect_ratio: Option<ResolutionName>,
    ) -> Result<VideoRaster, String> {
        let resolution = resolution.unwrap_or(self.resolutions[0]);
        let aspect_ratio = aspect_ratio.unwrap_or(self.aspect_ratios[0]);
        if !self.resolutions.contains(&resolution) {
            return Err(format!(
                "resolution must be one of {}",
                join(self.resolutions.iter().map(|value| value.as_str()))
            ));
        }
        if !self.aspect_ratios.contains(&aspect_ratio) {
            return Err(format!(
                "aspect_ratio must be one of {}",
                join(self.aspect_ratios.iter().map(|value| value.as_str()))
            ));
        }
        VideoRaster::new(resolution, aspect_ratio).ok_or_else(|| {
            format!(
                "no trained bucket for {resolution} {}",
                aspect_ratio.as_str()
            )
        })
    }

    /// Returns the served raster with exactly these pixel dimensions.
    pub fn find_size(&self, width: u32, height: u32) -> Option<VideoRaster> {
        self.rasters()
            .find(|raster| raster.width == width && raster.height == height)
    }

    /// Worker `--video-frame-sizes` value: comma-separated `HEIGHTxWIDTH`.
    pub fn worker_frame_sizes(&self) -> String {
        self.rasters()
            .map(|raster| format!("{}x{}", raster.height, raster.width))
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Joins names into a human-readable list.
fn join<S: AsRef<str>>(values: impl Iterator<Item = S>) -> String {
    values
        .map(|value| value.as_ref().to_owned())
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_bucket_is_token_aligned_and_mirrored() {
        for resolution in VideoResolution::ALL {
            for aspect_ratio in VIDEO_ASPECT_RATIOS {
                let raster = VideoRaster::new(resolution, aspect_ratio).unwrap();
                assert_eq!(raster.width % 32, 0);
                assert_eq!(raster.height % 32, 0);
            }
            for (landscape, portrait) in [
                (ResolutionName::Landscape16x9, ResolutionName::Portrait9x16),
                (ResolutionName::Landscape4x3, ResolutionName::Portrait3x4),
            ] {
                let landscape = VideoRaster::new(resolution, landscape).unwrap();
                let portrait = VideoRaster::new(resolution, portrait).unwrap();
                assert_eq!(
                    (landscape.width, landscape.height),
                    (portrait.height, portrait.width)
                );
            }
        }
        assert!(VideoRaster::new(VideoResolution::P768, ResolutionName::Landscape3x2).is_none());
    }

    #[test]
    fn selection_defaults_to_the_first_configured_values() {
        let rasters = VideoRasters::new(
            &[VideoResolution::P480, VideoResolution::P768],
            &[ResolutionName::Portrait9x16, ResolutionName::Square],
        )
        .unwrap();
        let default = rasters.select(None, None).unwrap();
        assert_eq!((default.width, default.height), (480, 832));
        let square = rasters
            .select(Some(VideoResolution::P768), Some(ResolutionName::Square))
            .unwrap();
        assert_eq!((square.width, square.height), (768, 768));
        assert!(
            rasters
                .select(None, Some(ResolutionName::Landscape16x9))
                .is_err()
        );
        assert_eq!(
            rasters.worker_frame_sizes(),
            "832x480,480x480,1344x768,768x768"
        );
        assert_eq!(rasters.find_size(768, 768), Some(square));
        assert_eq!(rasters.find_size(1344, 768), None);
    }

    #[test]
    fn invalid_selections_are_rejected() {
        let p768 = [VideoResolution::P768];
        assert!(VideoRasters::new(&[], &[ResolutionName::Square]).is_err());
        assert!(VideoRasters::new(&p768, &[]).is_err());
        assert!(VideoRasters::new(&p768, &[ResolutionName::Landscape3x2]).is_err());
        assert!(
            VideoRasters::new(&p768, &[ResolutionName::Square, ResolutionName::Square]).is_err()
        );
        assert!(
            VideoRasters::new(
                &[VideoResolution::P768, VideoResolution::P768],
                &[ResolutionName::Square]
            )
            .is_err()
        );
        assert_eq!(
            "480p".parse::<VideoResolution>().unwrap(),
            VideoResolution::P480
        );
        assert!("720p".parse::<VideoResolution>().is_err());
    }
}
