use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolutionBucket {
    pub name: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolutionPolicy {
    pub default: ResolutionBucket,
    pub buckets: Vec<ResolutionBucket>,
    pub allow_custom: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedResolution {
    pub width: u32,
    pub height: u32,
}

pub fn resolve_resolution(
    policy: &ResolutionPolicy,
    resolution: Option<&str>,
    width: Option<u32>,
    height: Option<u32>,
) -> Result<ResolvedResolution, String> {
    let bucket = match resolution {
        Some(value) => find_bucket(&policy.buckets, value)
            .ok_or_else(|| format!("unsupported resolution: {value}"))?,
        None => policy.default.clone(),
    };
    let width = width.unwrap_or(bucket.width);
    let height = height.unwrap_or(bucket.height);
    if width == 0 || height == 0 {
        return Err("width and height must be positive".into());
    }
    if policy.allow_custom {
        return Ok(ResolvedResolution { width, height });
    }
    if policy
        .buckets
        .iter()
        .any(|b| b.width == width && b.height == height)
    {
        return Ok(ResolvedResolution { width, height });
    }
    Err(format!("unsupported dimensions: {width}x{height}"))
}

fn find_bucket(buckets: &[ResolutionBucket], value: &str) -> Option<ResolutionBucket> {
    let normalized = value.trim().to_ascii_lowercase().replace('×', "x");
    buckets
        .iter()
        .find(|bucket| {
            bucket.name.eq_ignore_ascii_case(&normalized)
                || normalized == format!("{}x{}", bucket.width, bucket.height)
        })
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sensenova_policy() -> ResolutionPolicy {
        ResolutionPolicy {
            default: ResolutionBucket {
                name: "16:9".into(),
                width: 2048,
                height: 1152,
            },
            buckets: vec![ResolutionBucket {
                name: "16:9".into(),
                width: 2048,
                height: 1152,
            }],
            allow_custom: false,
        }
    }

    #[test]
    fn maps_sensenova_aliases() {
        let policy = sensenova_policy();
        assert_eq!(
            resolve_resolution(&policy, Some("16:9"), None, None).unwrap(),
            ResolvedResolution {
                width: 2048,
                height: 1152
            }
        );
        assert_eq!(
            resolve_resolution(&policy, Some("2048x1152"), None, None).unwrap(),
            ResolvedResolution {
                width: 2048,
                height: 1152
            }
        );
    }

    #[test]
    fn unsupported_dimensions_are_rejected() {
        let policy = sensenova_policy();
        assert!(resolve_resolution(&policy, Some("16:9"), Some(512), Some(512)).is_err());
    }
}
