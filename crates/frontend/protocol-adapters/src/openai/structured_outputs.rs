use crate::openai::error::ApiError;
use serde::Deserialize;
use uniserve_openai_types::ResponseFormat;
use uniserve_serving::StructuredOutputIntent;

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct StructuredOutputsFields {
    json: Option<serde_json::Value>,
    regex: Option<String>,
    choice: Option<Vec<String>>,
    grammar: Option<String>,
    json_object: Option<bool>,
    disable_any_whitespace: bool,
    disable_additional_properties: bool,
    whitespace_pattern: Option<String>,
    structural_tag: Option<String>,
}

/// Convert the external `structured_outputs` field object into one semantic constraint.
fn deserialize_structured_outputs(
    raw: &serde_json::Value,
) -> Result<StructuredOutputIntent, ApiError> {
    let fields: StructuredOutputsFields = serde_json::from_value(raw.clone()).map_err(|e| {
        ApiError::invalid_request(
            format!("invalid structured_outputs: {e}"),
            Some("structured_outputs"),
        )
    })?;

    let active_constraints = usize::from(fields.json.is_some())
        + usize::from(fields.regex.is_some())
        + usize::from(fields.choice.is_some())
        + usize::from(fields.grammar.is_some())
        + usize::from(fields.json_object == Some(true))
        + usize::from(fields.structural_tag.is_some());
    if active_constraints != 1 {
        return Err(ApiError::invalid_request(
            "`structured_outputs` must select exactly one constraint.".to_string(),
            Some("structured_outputs"),
        ));
    }

    if let Some(schema) = fields.json {
        return Ok(StructuredOutputIntent::JsonSchema {
            schema,
            disable_any_whitespace: fields.disable_any_whitespace,
            disable_additional_properties: fields.disable_additional_properties,
            whitespace_pattern: fields.whitespace_pattern,
        });
    }
    if fields.json_object == Some(true) {
        if fields.disable_additional_properties {
            return Err(ApiError::invalid_request(
                "`disable_additional_properties` requires a JSON schema constraint.".to_string(),
                Some("structured_outputs"),
            ));
        }
        return Ok(StructuredOutputIntent::JsonObject {
            disable_any_whitespace: fields.disable_any_whitespace,
            whitespace_pattern: fields.whitespace_pattern,
        });
    }
    if fields.disable_any_whitespace
        || fields.disable_additional_properties
        || fields.whitespace_pattern.is_some()
    {
        return Err(ApiError::invalid_request(
            "JSON whitespace and property controls require a JSON constraint.".to_string(),
            Some("structured_outputs"),
        ));
    }
    if let Some(regex) = fields.regex {
        return Ok(StructuredOutputIntent::Regex(regex));
    }
    if let Some(choice) = fields.choice {
        return Ok(StructuredOutputIntent::Choice(choice));
    }
    if let Some(grammar) = fields.grammar {
        return Ok(StructuredOutputIntent::Grammar(grammar));
    }
    if let Some(structural_tag) = fields.structural_tag {
        return Ok(StructuredOutputIntent::StructuralTag(structural_tag));
    }
    unreachable!("exactly one structured-output constraint was validated")
}

/// Convert a typed [`ResponseFormat`] or explicit constraint field object.
pub fn convert_from_response_format(
    response_format: Option<&ResponseFormat>,
    structured_outputs: &Option<serde_json::Value>,
) -> Result<Option<StructuredOutputIntent>, ApiError> {
    if let Some(raw) = structured_outputs {
        return Ok(Some(deserialize_structured_outputs(raw)?));
    }

    let Some(fmt) = response_format else {
        return Ok(None);
    };
    match fmt {
        ResponseFormat::Text => Ok(None),
        ResponseFormat::JsonObject => Ok(Some(StructuredOutputIntent::JsonObject {
            disable_any_whitespace: false,
            whitespace_pattern: None,
        })),
        ResponseFormat::JsonSchema { json_schema } => {
            Ok(Some(StructuredOutputIntent::JsonSchema {
                schema: json_schema.schema.clone(),
                disable_any_whitespace: false,
                disable_additional_properties: false,
                whitespace_pattern: None,
            }))
        }
        ResponseFormat::StructuralTag { .. } => {
            let tag_json = serde_json::to_string(fmt).map_err(|e| {
                ApiError::invalid_request(
                    format!("failed to serialize structural_tag: {e}"),
                    Some("response_format"),
                )
            })?;
            Ok(Some(StructuredOutputIntent::StructuralTag(tag_json)))
        }
    }
}

/// Convert raw completion-protocol fields into a semantic constraint.
pub fn convert_from_response_format_value(
    response_format: &Option<serde_json::Value>,
    structured_outputs: &Option<serde_json::Value>,
) -> Result<Option<StructuredOutputIntent>, ApiError> {
    if let Some(raw) = structured_outputs {
        return Ok(Some(deserialize_structured_outputs(raw)?));
    }

    let Some(raw) = response_format else {
        return Ok(None);
    };

    let fmt: ResponseFormat = serde_json::from_value(raw.clone()).map_err(|e| {
        ApiError::invalid_request(
            format!("invalid response_format: {e}"),
            Some("response_format"),
        )
    })?;
    convert_from_response_format(Some(&fmt), &None)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn explicit_json_schema_preserves_semantic_controls() {
        let raw = Some(json!({
            "json": {"type": "object"},
            "disable_any_whitespace": true,
            "disable_additional_properties": true
        }));

        assert_eq!(
            convert_from_response_format(None, &raw).unwrap(),
            Some(StructuredOutputIntent::JsonSchema {
                schema: json!({"type": "object"}),
                disable_any_whitespace: true,
                disable_additional_properties: true,
                whitespace_pattern: None,
            })
        );
    }

    #[test]
    fn explicit_constraints_reject_ambiguous_or_misowned_fields() {
        let ambiguous = Some(json!({"regex": "a+", "grammar": "root ::= \"a\""}));
        assert!(convert_from_response_format(None, &ambiguous).is_err());

        let non_json_control = Some(json!({
            "regex": "a+",
            "disable_any_whitespace": true
        }));
        assert!(convert_from_response_format(None, &non_json_control).is_err());
    }

    #[test]
    fn explicit_constraint_takes_precedence_over_response_format() {
        let raw = Some(json!({"choice": ["yes", "no"]}));
        assert_eq!(
            convert_from_response_format(Some(&ResponseFormat::JsonObject), &raw).unwrap(),
            Some(StructuredOutputIntent::Choice(vec![
                "yes".to_string(),
                "no".to_string()
            ]))
        );
    }
}
