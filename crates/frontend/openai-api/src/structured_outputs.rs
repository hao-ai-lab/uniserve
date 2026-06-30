use crate::error::ApiError;
use uniserve_engine_client::protocol::StructuredOutputsParams;
use uniserve_openai_types::ResponseFormat;

/// Convert an explicit `structured_outputs` JSON blob into
/// [`StructuredOutputsParams`].
fn deserialize_structured_outputs(
    raw: &serde_json::Value,
) -> Result<StructuredOutputsParams, ApiError> {
    serde_json::from_value(raw.clone()).map_err(|e| {
        ApiError::invalid_request(
            format!("invalid structured_outputs: {e}"),
            Some("structured_outputs"),
        )
    })
}

/// Convert a typed [`ResponseFormat`] and/or raw `structured_outputs` blob into
/// engine [`StructuredOutputsParams`].

pub fn convert_from_response_format(
    response_format: Option<&ResponseFormat>,
    structured_outputs: &Option<serde_json::Value>,
) -> Result<Option<StructuredOutputsParams>, ApiError> {
    if let Some(raw) = structured_outputs {
        return Ok(Some(deserialize_structured_outputs(raw)?));
    }

    let Some(fmt) = response_format else {
        return Ok(None);
    };
    match fmt {
        ResponseFormat::Text => Ok(None),
        ResponseFormat::JsonObject => Ok(Some(StructuredOutputsParams {
            json_object: Some(true),
            ..Default::default()
        })),
        ResponseFormat::JsonSchema { json_schema } => Ok(Some(StructuredOutputsParams {
            json: Some(json_schema.schema.clone()),
            ..Default::default()
        })),
        ResponseFormat::StructuralTag { .. } => {
            // The engine expects the complete response_format object,
            // including the `type` field, as the structural-tag payload.
            let tag_json = serde_json::to_string(fmt).map_err(|e| {
                ApiError::invalid_request(
                    format!("failed to serialize structural_tag: {e}"),
                    Some("response_format"),
                )
            })?;
            Ok(Some(StructuredOutputsParams {
                structural_tag: Some(tag_json),
                ..Default::default()
            }))
        }
    }
}

/// Convert raw `response_format` and/or `structured_outputs` JSON blobs into
/// engine [`StructuredOutputsParams`].

/// Used by the completions endpoint which keeps both fields as opaque
/// `serde_json::Value`.
pub fn convert_from_response_format_value(
    response_format: &Option<serde_json::Value>,
    structured_outputs: &Option<serde_json::Value>,
) -> Result<Option<StructuredOutputsParams>, ApiError> {
    if let Some(raw) = structured_outputs {
        return Ok(Some(deserialize_structured_outputs(raw)?));
    }

    let Some(raw) = response_format else {
        return Ok(None);
    };

    // Deserialize into our typed enum and delegate.
    let fmt: ResponseFormat = serde_json::from_value(raw.clone()).map_err(|e| {
        ApiError::invalid_request(
            format!("invalid response_format: {e}"),
            Some("response_format"),
        )
    })?;
    convert_from_response_format(Some(&fmt), &None)
}
