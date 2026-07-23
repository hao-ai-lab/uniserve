use serde::{Deserialize, Serialize};
use serde_json::Value;

/// JSON schema specification nested inside a `json_schema` response format.
/// Mirrors the `JsonSchemaResponseFormat` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonSchemaFormat {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// The actual JSON schema object.
    pub schema: Value,
    #[serde(default)]
    pub strict: Option<bool>,
}

/// Supported `response_format` types for chat and completion requests.
/// This is our own definition (rather than the `openai-protocol` crate's) so
/// that we can support the `structural_tag` variant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema {
        json_schema: JsonSchemaFormat,
    },
    /// Structural-tag format. Conversion layers serialize the entire object,
    /// including the `type` field, into the engine structured-output payload.

    /// The payload is captured as a catch-all map so compatible structural-tag
    /// shapes are preserved without needing separate typed structs for each
    /// external dialect.

    /// Note: this is intentionally an opaque pass-through. Schema validation of
    /// the structural-tag grammar is deferred to the engine, which compiles the
    /// serialized payload into a constraint; validating dialect-specific shapes
    /// here would require typed structs per dialect and is out of scope for this
    /// protocol DTO.
    StructuralTag {
        #[serde(flatten)]
        extra: serde_json::Map<String, Value>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `{"type": "text"}` deserializes into the unit `Text` variant and
    /// round-trips back to the same tagged JSON.
    #[test]
    fn text_format_round_trips() {
        let json = serde_json::json!({"type": "text"});
        let format: ResponseFormat = serde_json::from_value(json.clone()).unwrap();

        assert_eq!(format, ResponseFormat::Text);
        assert_eq!(serde_json::to_value(&format).unwrap(), json);
    }

    /// `json_object` uses snake_case tag renaming.
    #[test]
    fn json_object_format_round_trips() {
        let json = serde_json::json!({"type": "json_object"});
        let format: ResponseFormat = serde_json::from_value(json.clone()).unwrap();

        assert_eq!(format, ResponseFormat::JsonObject);
        assert_eq!(serde_json::to_value(&format).unwrap(), json);
    }

    /// A `json_schema` format carries the nested `JsonSchemaFormat` and
    /// round-trips through serde preserving name/strict/schema.
    #[test]
    fn json_schema_format_round_trips() {
        let json = serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {"type": "object"},
                "strict": true,
            },
        });

        let format: ResponseFormat = serde_json::from_value(json.clone()).unwrap();
        match &format {
            ResponseFormat::JsonSchema { json_schema } => {
                assert_eq!(json_schema.name, "person");
                assert_eq!(json_schema.strict, Some(true));
                assert_eq!(json_schema.schema, serde_json::json!({"type": "object"}));
            }
            other => panic!("expected JsonSchema, got {other:?}"),
        }

        assert_eq!(serde_json::to_value(&format).unwrap(), json);
    }

    /// The `structural_tag` variant captures the entire payload (including the
    /// `type` tag) in its catch-all `extra` map and re-serializes it intact.
    #[test]
    fn structural_tag_captures_extra_payload_in_catch_all() {
        let json = serde_json::json!({
            "type": "structural_tag",
            "structures": [{"begin": "<a>", "end": "</a>"}],
            "triggers": ["<a>"],
        });

        let format: ResponseFormat = serde_json::from_value(json.clone()).unwrap();
        match &format {
            ResponseFormat::StructuralTag { extra } => {
                // Dialect-specific fields are preserved opaquely in the catch-all.
                assert_eq!(
                    extra.get("structures"),
                    Some(&serde_json::json!([{"begin": "<a>", "end": "</a>"}]))
                );
                assert_eq!(extra.get("triggers"), Some(&serde_json::json!(["<a>"])));
            }
            other => panic!("expected StructuralTag, got {other:?}"),
        }

        // The whole payload, including the type tag, round-trips unchanged.
        assert_eq!(serde_json::to_value(&format).unwrap(), json);
    }
}
