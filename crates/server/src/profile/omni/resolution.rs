use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResolutionName {
    #[default]
    #[serde(rename = "1:1")]
    Square,
    #[serde(rename = "16:9")]
    Landscape16x9,
    #[serde(rename = "1.5K")]
    OnePointFiveK,
    #[serde(rename = "9:16")]
    Portrait9x16,
    #[serde(rename = "3:2")]
    Landscape3x2,
    #[serde(rename = "2:3")]
    Portrait2x3,
    #[serde(rename = "4:3")]
    Landscape4x3,
    #[serde(rename = "3:4")]
    Portrait3x4,
    #[serde(rename = "1:2")]
    Portrait1x2,
    #[serde(rename = "2:1")]
    Landscape2x1,
    #[serde(rename = "1:3")]
    Portrait1x3,
    #[serde(rename = "3:1")]
    Landscape3x1,
}

impl ResolutionName {
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
pub struct ResolutionBucket {
    pub name: ResolutionName,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolutionPolicy {
    pub default: ResolutionBucket,
    pub buckets: Vec<ResolutionBucket>,
    pub allow_custom: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedResolution {
    pub width: u32,
    pub height: u32,
}

pub fn resolve_resolution(
    policy: &ResolutionPolicy,
    requested: Option<ResolutionName>,
    width: Option<u32>,
    height: Option<u32>,
) -> Result<ResolvedResolution, ResolutionError> {
    if width.is_some() || height.is_some() {
        let width = width.ok_or(ResolutionError::MissingWidth)?;
        let height = height.ok_or(ResolutionError::MissingHeight)?;
        if width == 0 || height == 0 {
            return Err(ResolutionError::NonPositiveDimensions);
        }
        if policy.allow_custom
            || policy
                .buckets
                .iter()
                .any(|bucket| bucket.width == width && bucket.height == height)
        {
            return Ok(ResolvedResolution { width, height });
        }
        return Err(ResolutionError::UnsupportedDimensions { width, height });
    }

    let requested = requested.unwrap_or(policy.default.name);
    let bucket = policy
        .buckets
        .iter()
        .find(|bucket| bucket.name == requested)
        .ok_or_else(|| ResolutionError::UnsupportedName(requested.as_str().to_owned()))?;
    Ok(ResolvedResolution {
        width: bucket.width,
        height: bucket.height,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ResolutionError {
    #[error("custom image dimensions require width")]
    MissingWidth,
    #[error("custom image dimensions require height")]
    MissingHeight,
    #[error("image dimensions must be positive")]
    NonPositiveDimensions,
    #[error("unsupported image dimensions {width}x{height}")]
    UnsupportedDimensions { width: u32, height: u32 },
    #[error("unsupported image resolution {0:?}")]
    UnsupportedName(String),
}

#[cfg(test)]
mod tests {
    use super::{ResolutionBucket, ResolutionName, ResolutionPolicy, resolve_resolution};

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
        }
    }

    #[test]
    fn configured_dimension_contract() {
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
