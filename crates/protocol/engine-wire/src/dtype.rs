use serde::{Deserialize, Serialize};

/// Effective model dtype reported by the engine after config resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelDtype {
    #[serde(rename = "float16")]
    Float16,
    #[serde(rename = "bfloat16")]
    BFloat16,
    #[serde(rename = "float32")]
    Float32,
}

impl ModelDtype {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Float16 => "float16",
            Self::BFloat16 => "bfloat16",
            Self::Float32 => "float32",
        }
    }

 /// Map a worker-reported `kv_dtype` string into the protocol enum.
 ///
 /// The worker emits short aliases (`bf16`/`fp16`/`fp32`, plus quantized
 /// spellings like `fp8_e4m3`) that the canonical serde representation (the
 /// long `bfloat16`/`float16`/`float32`) cannot parse, so this is the single
 /// place that understands both vocabularies. Unknown or quantized dtypes fall
 /// back to `BFloat16`.
 ///
 /// Note this is the *KV-cache* dtype; it is reused as the reported *model*
 /// dtype because no separate weight-dtype channel exists on the caps wire yet.
    pub fn from_kv_str(kv_dtype: &str) -> Self {
        match kv_dtype {
            "fp16" | "float16" | "f16" => Self::Float16,
            "fp32" | "float32" | "f32" => Self::Float32,
 // "bf16"/"bfloat16" and any unknown/quantized future dtype.
            _ => Self::BFloat16,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ModelDtype;

    #[test]
    fn serde_uses_protocol_dtype_strings() {
        assert_eq!(
            serde_json::to_value(ModelDtype::Float16).unwrap(),
            serde_json::json!("float16")
        );
        assert_eq!(
            serde_json::from_value::<ModelDtype>(serde_json::json!("bfloat16")).unwrap(),
            ModelDtype::BFloat16
        );
        assert_eq!(ModelDtype::Float32.as_str(), "float32");
    }

    #[test]
    fn from_kv_str_maps_short_and_long_aliases() {
        for s in ["fp16", "float16", "f16"] {
            assert_eq!(ModelDtype::from_kv_str(s), ModelDtype::Float16, "{s}");
        }
        for s in ["fp32", "float32", "f32"] {
            assert_eq!(ModelDtype::from_kv_str(s), ModelDtype::Float32, "{s}");
        }
        for s in ["bf16", "bfloat16"] {
            assert_eq!(ModelDtype::from_kv_str(s), ModelDtype::BFloat16, "{s}");
        }
 // Quantized / unknown KV dtypes fall back to BFloat16.
        assert_eq!(ModelDtype::from_kv_str("fp8_e4m3"), ModelDtype::BFloat16);
        assert_eq!(ModelDtype::from_kv_str(""), ModelDtype::BFloat16);
    }
}
