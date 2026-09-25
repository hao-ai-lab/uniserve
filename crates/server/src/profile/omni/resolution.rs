//! Named output resolutions and aspect-ratio bucket selection.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
/// Named output canvas accepted by multimodal profiles.
pub enum ResolutionName {
    /// Square 1:1 canvas.
    #[default]
    #[serde(rename = "1:1")]
    Square,
    /// Landscape 16:9 canvas.
    #[serde(rename = "16:9")]
    Landscape16x9,
    /// Model-specific 1.5K canvas preset.
    #[serde(rename = "1.5K")]
    OnePointFiveK,
    /// Portrait 9:16 canvas.
    #[serde(rename = "9:16")]
    Portrait9x16,
    /// Landscape 3:2 canvas.
    #[serde(rename = "3:2")]
    Landscape3x2,
    /// Portrait 2:3 canvas.
    #[serde(rename = "2:3")]
    Portrait2x3,
    /// Landscape 4:3 canvas.
    #[serde(rename = "4:3")]
    Landscape4x3,
    /// Portrait 3:4 canvas.
    #[serde(rename = "3:4")]
    Portrait3x4,
    /// Portrait 1:2 canvas.
    #[serde(rename = "1:2")]
    Portrait1x2,
    /// Landscape 2:1 canvas.
    #[serde(rename = "2:1")]
    Landscape2x1,
    /// Portrait 1:3 canvas.
    #[serde(rename = "1:3")]
    Portrait1x3,
    /// Landscape 3:1 canvas.
    #[serde(rename = "3:1")]
    Landscape3x1,
}

impl ResolutionName {
    /// Returns the stable wire name for this resolution.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Square => "1:1",
            Self::Landscape16x9 => "16:9",
            Self::OnePointFiveK => "1.5K",
            Self::Portrait9x16 => "9:16",
            Self::Landscape3x2 => "3:2",
            Self::Portrait2x3 => "2:3",
            Self::Landscape4x3 => "4:3",
            Self::Portrait3x4 => "3:4",
            Self::Portrait1x2 => "1:2",
            Self::Landscape2x1 => "2:1",
            Self::Portrait1x3 => "1:3",
            Self::Landscape3x1 => "3:1",
        }
    }
}

impl std::str::FromStr for ResolutionName {
    type Err = ResolutionError;

    /// Parses a wire name as returned by [`ResolutionName::as_str`], also
    /// accepting the lowercase spelling `1.5k`.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "1:1" => Ok(Self::Square),
            "16:9" => Ok(Self::Landscape16x9),
            "1.5K" | "1.5k" => Ok(Self::OnePointFiveK),
            "9:16" => Ok(Self::Portrait9x16),
            "3:2" => Ok(Self::Landscape3x2),
            "2:3" => Ok(Self::Portrait2x3),
            "4:3" => Ok(Self::Landscape4x3),
            "3:4" => Ok(Self::Portrait3x4),
            "1:2" => Ok(Self::Portrait1x2),
            "2:1" => Ok(Self::Landscape2x1),
            "1:3" => Ok(Self::Portrait1x3),
            "3:1" => Ok(Self::Landscape3x1),
            _ => Err(ResolutionError::UnsupportedName(value.to_owned())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// One aspect-ratio bucket and its output dimensions.
pub struct ResolutionBucket {
    /// Stable name of the resolution preset.
    pub name: ResolutionName,
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Bucket set and default canvas used for resolution selection.
pub struct ResolutionPolicy {
    /// Resolution selected when the request provides no dimensions or name.
    ///
    /// [`resolve_resolution`] reads only its `name` and looks that name up in
    /// `buckets`, so a bucket of the same name must exist; that bucket's
    /// dimensions are the ones returned.
    pub default: ResolutionBucket,
    /// Named resolutions accepted by the profile; also the only explicit
    /// dimensions accepted when `allow_custom` is false.
    pub buckets: Vec<ResolutionBucket>,
    /// Whether arbitrary positive pixel dimensions are accepted.
    pub allow_custom: bool,
    /// Longest accepted side of the resolved canvas in pixels, when the
    /// model bounds each axis; `None` leaves the sides unbounded here.
    pub max_side: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Selected output dimensions.
pub struct ResolvedResolution {
    /// Selected output width in pixels.
    pub width: u32,
    /// Selected output height in pixels.
    pub height: u32,
}

/// Resolves explicit dimensions or a named bucket under `policy`.
///
/// Explicit dimensions take precedence: when either `width` or `height` is
/// given, `requested` is ignored, both must be present and positive, and they
/// must match a bucket exactly unless the policy allows custom dimensions.
/// Otherwise the named bucket, or the policy default, must exist in
/// `buckets`. Either way, no side of the result may exceed
/// `policy.max_side`.
pub fn resolve_resolution(
    policy: &ResolutionPolicy,
    requested: Option<ResolutionName>,
    width: Option<u32>,
    height: Option<u32>,
) -> Result<ResolvedResolution, ResolutionError> {
    let resolved = if width.is_some() || height.is_some() {
        let width = width.ok_or(ResolutionError::MissingWidth)?;
        let height = height.ok_or(ResolutionError::MissingHeight)?;
        if width == 0 || height == 0 {
            return Err(ResolutionError::NonPositiveDimensions);
        }
        if !policy.allow_custom
            && !policy
                .buckets
                .iter()
                .any(|bucket| bucket.width == width && bucket.height == height)
        {
            return Err(ResolutionError::UnsupportedDimensions { width, height });
        }
        ResolvedResolution { width, height }
    } else {
        let requested = requested.unwrap_or(policy.default.name);
        let bucket = policy
            .buckets
            .iter()
            .find(|bucket| bucket.name == requested)
            .ok_or_else(|| ResolutionError::UnsupportedName(requested.as_str().to_owned()))?;
        ResolvedResolution {
            width: bucket.width,
            height: bucket.height,
        }
    };

    if let Some(max_side) = policy.max_side
        && resolved.width.max(resolved.height) > max_side
    {
        return Err(ResolutionError::SideExceedsLimit {
            width: resolved.width,
            height: resolved.height,
            max_side,
        });
    }
    Ok(resolved)
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
/// Invalid or unsupported output-resolution selection.
pub enum ResolutionError {
    /// An explicit height was provided without a width.
    #[error("custom image dimensions require width")]
    MissingWidth,
    /// An explicit width was provided without a height.
    #[error("custom image dimensions require height")]
    MissingHeight,
    /// At least one explicit dimension is zero.
    #[error("image dimensions must be positive")]
    NonPositiveDimensions,
    /// Explicit dimensions do not match an allowed bucket.
    #[error("unsupported image dimensions {width}x{height}")]
    UnsupportedDimensions {
        /// Requested width in pixels.
        width: u32,
        /// Requested height in pixels.
        height: u32,
    },
    /// A named resolution is not present in the profile policy.
    #[error("unsupported image resolution {0:?}")]
    UnsupportedName(String),
    /// A side of the resolved canvas exceeds the model's per-axis limit.
    #[error("image dimensions {width}x{height} exceed the model's {max_side}-pixel side limit")]
    SideExceedsLimit {
        /// Resolved width in pixels.
        width: u32,
        /// Resolved height in pixels.
        height: u32,
        /// Longest accepted side in pixels.
        max_side: u32,
    },
}

#[cfg(test)]
mod tests {
    use super::{
        ResolutionBucket, ResolutionError, ResolutionName, ResolutionPolicy, resolve_resolution,
    };

    fn policy(allow_custom: bool) -> ResolutionPolicy {
        let default = ResolutionBucket {
            name: ResolutionName::Square,
            width: 512,
            height: 512,
        };
        ResolutionPolicy {
            default: default.clone(),
            buckets: vec![default],
            allow_custom,
            max_side: None,
        }
    }

    /// A per-axis limit rejects custom dimensions with either side above it
    /// and accepts a side equal to it.
    #[test]
    fn side_limit_bounds_each_axis() {
        let bounded = ResolutionPolicy {
            max_side: Some(1024),
            ..policy(true)
        };
        for (width, height) in [(1040, 16), (16, 1040)] {
            assert_eq!(
                resolve_resolution(&bounded, None, Some(width), Some(height)),
                Err(ResolutionError::SideExceedsLimit {
                    width,
                    height,
                    max_side: 1024
                })
            );
        }
        let edge = resolve_resolution(&bounded, None, Some(1024), Some(16)).unwrap();
        assert_eq!((edge.width, edge.height), (1024, 16));
    }

    /// A fixed policy accepts its bucket by name or by exact dimensions and
    /// rejects others; a custom policy accepts any positive dimensions.
    #[test]
    fn configured_dimensions() {
        let fixed = policy(false);
        let named = resolve_resolution(&fixed, Some(ResolutionName::Square), None, None).unwrap();
        assert_eq!((named.width, named.height), (512, 512));
        let explicit =
            resolve_resolution(&fixed, Some(ResolutionName::Square), Some(512), Some(512)).unwrap();
        assert_eq!((explicit.width, explicit.height), (512, 512));
        assert!(resolve_resolution(&fixed, None, Some(768), Some(512)).is_err());

        let custom = resolve_resolution(&policy(true), None, Some(768), Some(512)).unwrap();
        assert_eq!((custom.width, custom.height), (768, 512));
    }
}
