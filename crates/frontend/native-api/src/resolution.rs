#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolutionBucket {
    pub name: &'static str,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct ResolutionPolicy {
    pub default: ResolutionBucket,
    pub buckets: &'static [ResolutionBucket],
    pub allow_custom: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedResolution {
    pub width: u32,
    pub height: u32,
}

pub const BAGEL_BUCKETS: &[ResolutionBucket] = &[ResolutionBucket {
    name: "1:1",
    width: 512,
    height: 512,
}];

pub const SENSENOVA_BUCKETS: &[ResolutionBucket] = &[
    ResolutionBucket {
        name: "1:1",
        width: 1536,
        height: 1536,
    },
    ResolutionBucket {
        name: "16:9",
        width: 2048,
        height: 1152,
    },
    ResolutionBucket {
        name: "9:16",
        width: 1152,
        height: 2048,
    },
    ResolutionBucket {
        name: "3:2",
        width: 1888,
        height: 1248,
    },
    ResolutionBucket {
        name: "2:3",
        width: 1248,
        height: 1888,
    },
    ResolutionBucket {
        name: "4:3",
        width: 1760,
        height: 1312,
    },
    ResolutionBucket {
        name: "3:4",
        width: 1312,
        height: 1760,
    },
    ResolutionBucket {
        name: "1:2",
        width: 1088,
        height: 2144,
    },
    ResolutionBucket {
        name: "2:1",
        width: 2144,
        height: 1088,
    },
    ResolutionBucket {
        name: "1:3",
        width: 864,
        height: 2592,
    },
    ResolutionBucket {
        name: "3:1",
        width: 2592,
        height: 864,
    },
];

pub fn resolve_resolution(
    policy: ResolutionPolicy,
    resolution: Option<&str>,
    width: Option<u32>,
    height: Option<u32>,
) -> Result<ResolvedResolution, String> {
    let bucket = match resolution {
        Some(value) => find_bucket(policy.buckets, value)
            .ok_or_else(|| format!("unsupported resolution: {value}"))?,
        None => policy.default,
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
    buckets.iter().copied().find(|bucket| {
        bucket.name.eq_ignore_ascii_case(&normalized)
            || normalized == format!("{}x{}", bucket.width, bucket.height)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_sensenova_aliases() {
        let policy = ResolutionPolicy {
            default: SENSENOVA_BUCKETS[1],
            buckets: SENSENOVA_BUCKETS,
            allow_custom: false,
        };
        assert_eq!(
            resolve_resolution(policy, Some("16:9"), None, None).unwrap(),
            ResolvedResolution {
                width: 2048,
                height: 1152
            }
        );
        assert_eq!(
            resolve_resolution(policy, Some("2048x1152"), None, None).unwrap(),
            ResolvedResolution {
                width: 2048,
                height: 1152
            }
        );
    }

    #[test]
    fn unsupported_dimensions_are_rejected() {
        let policy = ResolutionPolicy {
            default: SENSENOVA_BUCKETS[1],
            buckets: SENSENOVA_BUCKETS,
            allow_custom: false,
        };
        assert!(resolve_resolution(policy, Some("16:9"), Some(512), Some(512)).is_err());
    }
}
