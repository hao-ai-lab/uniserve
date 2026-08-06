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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedResolution {
    pub width: u32,
    pub height: u32,
}

pub fn resolve_resolution(
    policy: &ResolutionPolicy,
    requested: Option<&str>,
    width: Option<u32>,
    height: Option<u32>,
) -> Result<ResolvedResolution, String> {
    if width.is_some() || height.is_some() {
        let width = width.ok_or_else(|| "custom image dimensions require width".to_string())?;
        let height = height.ok_or_else(|| "custom image dimensions require height".to_string())?;
        if width == 0 || height == 0 {
            return Err("image dimensions must be positive".to_string());
        }
        if policy.allow_custom
            || policy
                .buckets
                .iter()
                .any(|bucket| bucket.width == width && bucket.height == height)
        {
            return Ok(ResolvedResolution { width, height });
        }
        return Err(format!("unsupported image dimensions {width}x{height}"));
    }

    let requested = requested.unwrap_or(&policy.default.name);
    let bucket = policy
        .buckets
        .iter()
        .find(|bucket| bucket.name.eq_ignore_ascii_case(requested))
        .ok_or_else(|| format!("unsupported image resolution {requested:?}"))?;
    Ok(ResolvedResolution {
        width: bucket.width,
        height: bucket.height,
    })
}

#[cfg(test)]
mod tests {
    use super::{ResolutionBucket, ResolutionPolicy, resolve_resolution};

    fn policy(allow_custom: bool) -> ResolutionPolicy {
        let default = ResolutionBucket {
            name: "1:1".to_string(),
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
        let named = resolve_resolution(&fixed, Some("1:1"), None, None).unwrap();
        assert_eq!((named.width, named.height), (512, 512));
        let explicit = resolve_resolution(&fixed, Some("1:1"), Some(512), Some(512)).unwrap();
        assert_eq!((explicit.width, explicit.height), (512, 512));
        assert!(resolve_resolution(&fixed, None, Some(768), Some(512)).is_err());

        let custom = resolve_resolution(&policy(true), None, Some(768), Some(512)).unwrap();
        assert_eq!((custom.width, custom.height), (768, 512));
    }
}
