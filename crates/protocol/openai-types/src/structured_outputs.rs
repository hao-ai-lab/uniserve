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
    #[serde(alias = "json_schema")]
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
