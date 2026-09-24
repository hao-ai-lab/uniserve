//! Hugging Face-compatible JSON serialization filter for chat templates.
//!
//! `build_environment` registers [`hf_tojson_filter`] as `tojson`, replacing
//! MiniJinja's built-in filter of that name.

use minijinja::value::{Kwargs, ViaDeserialize};
use minijinja::{Error as MinijinjaError, ErrorKind, Value};
use serde::Deserialize;
use serde_json::Value as JsonValue;
use serde_json_fmt::{JsonFormat, JsonSyntaxError};
use thiserror_ext::AsReport;

/// Serializes a template value using Hugging Face `tojson` semantics.
///
/// Hugging Face templates rely on Python `json.dumps` semantics:
/// - no HTML escaping
/// - extra kwargs such as `ensure_ascii`, `separators`, and `sort_keys`
/// - Python-style `indent` handling
///
/// `ensure_ascii` and `sort_keys` default to `false`. Object keys keep their
/// insertion order unless `sort_keys` is set. The result is marked safe, so
/// auto-escaping does not alter it.
///
/// # Errors
///
/// Fails on an unknown or mistyped kwarg, an `indent` or separator string that
/// `serde_json_fmt` rejects, or a value that cannot be serialized as JSON.
pub(super) fn hf_tojson_filter(
    ViaDeserialize(value): ViaDeserialize<JsonValue>,
    kwargs: Kwargs,
) -> std::result::Result<Value, MinijinjaError> {
    let ensure_ascii = kwargs.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false);
    let indent = parse_indent(
        kwargs
            .get::<Option<ViaDeserialize<IndentArg>>>("indent")?
            .map(|value| value.0),
    );
    let separators = parse_separators(
        kwargs
            .get::<Option<ViaDeserialize<SeparatorsArg>>>("separators")?
            .map(|value| value.0),
        indent.is_some(),
    );
    let sort_keys = kwargs.get::<Option<bool>>("sort_keys")?.unwrap_or(false);

    kwargs.assert_all_used()?;

    let json_str = {
        let value_to_serialize = if sort_keys {
            &sort_json_keys(&value)
        } else {
            &value
        };

        build_json_format(indent, separators.0, separators.1, ensure_ascii)?
            .format_to_string(value_to_serialize)
            .map_err(|e| {
                MinijinjaError::new(
                    ErrorKind::InvalidOperation,
                    format!("Failed to serialize JSON: {}", e.as_report()),
                )
            })?
    };

    Ok(Value::from_safe_string(json_str))
}

#[derive(Deserialize)]
#[serde(untagged)]
enum IndentArg {
    // Python `json.dumps` accepts bool, int, and string indentation styles.
    // A bool acts as the int 0 or 1, as it does in Python.
    Bool(bool),
    Integer(i64),
    String(String),
}

/// Converts a `tojson` `indent` argument into the per-level indent string.
///
/// `None` selects compact single-line output. `Some("")`, which an `indent`
/// of zero, a negative number, `false`, or an empty string yields, still
/// breaks lines but does not indent them, matching Python.
fn parse_indent(value: Option<IndentArg>) -> Option<String> {
    match value? {
        IndentArg::Bool(indent) => Some(if indent {
            " ".to_owned()
        } else {
            String::new()
        }),
        IndentArg::Integer(indent) => Some(if indent > 0 {
            " ".repeat(indent as usize)
        } else {
            String::new()
        }),
        IndentArg::String(indent) => Some(indent),
    }
}

#[derive(Deserialize)]
struct SeparatorsArg((String, String));

/// Returns the `(item, key)` separators for `tojson`.
///
/// Without an explicit `separators` argument, the defaults follow Python: the
/// item separator drops its trailing space when `pretty` (an `indent` was
/// given), so indented output has no trailing whitespace.
fn parse_separators(value: Option<SeparatorsArg>, pretty: bool) -> (String, String) {
    let Some(SeparatorsArg((item_separator, key_separator))) = value else {
        let default_item_separator = if pretty { "," } else { ", " };
        let default_key_separator = ": ";

        return (
            default_item_separator.to_owned(),
            default_key_separator.to_owned(),
        );
    };

    (item_separator, key_separator)
}

/// Builds the `serde_json_fmt` formatter for the resolved `tojson` options.
fn build_json_format(
    indent: Option<String>,
    item_separator: String,
    key_separator: String,
    ensure_ascii: bool,
) -> std::result::Result<JsonFormat, MinijinjaError> {
    JsonFormat::new()
        .indent(indent)
        .map_err(map_json_syntax_error("indent"))?
        .comma(item_separator)
        .map_err(map_json_syntax_error("separators (item)"))?
        .colon(key_separator)
        .map_err(map_json_syntax_error("separators (key)"))
        .map(|format| format.ascii(ensure_ascii))
}

/// Returns a mapper that reports an invalid `field` value as a MiniJinja
/// `InvalidOperation` error.
fn map_json_syntax_error(
    field: &'static str,
) -> impl FnOnce(JsonSyntaxError) -> MinijinjaError + Copy {
    move |error| {
        MinijinjaError::new(
            ErrorKind::InvalidOperation,
            format!("invalid {field} value for tojson: {error}"),
        )
    }
}

/// Sorts every object key in a JSON value recursively.
///
/// A rebuilt map is required: `serde_json` is built with `preserve_order`, so
/// its `Map` keeps insertion order rather than sorting keys.
fn sort_json_keys(value: &JsonValue) -> JsonValue {
    match value {
        JsonValue::Object(map) => {
            let mut sorted: serde_json::Map<String, JsonValue> = serde_json::Map::new();
            let mut keys: Vec<_> = map.keys().collect();
            keys.sort();
            for key in keys {
                sorted.insert(key.clone(), sort_json_keys(&map[key]));
            }
            JsonValue::Object(sorted)
        }
        JsonValue::Array(arr) => JsonValue::Array(arr.iter().map(sort_json_keys).collect()),
        _ => value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use minijinja::Environment;
    use serde_json::json;

    use super::hf_tojson_filter;

    fn render(template: &str, payload: serde_json::Value) -> String {
        let mut env = Environment::new();
        env.add_filter("tojson", hf_tojson_filter);
        env.render_str(template, json!({ "payload": payload }))
            .unwrap()
    }

    #[test]
    fn tojson_does_not_html_escape_like_minijinja_builtin() {
        let rendered = render("{{ payload|tojson }}", json!("<tag>&'"));
        assert_eq!(rendered, "\"<tag>&'\"");
    }

    #[test]
    fn tojson_supports_sort_keys_recursively() {
        let rendered = render(
            "{{ payload|tojson(sort_keys=true) }}",
            json!({
                "z": {"b": 1, "a": 2},
                "a": 0
            }),
        );

        assert_eq!(rendered, "{\"a\": 0, \"z\": {\"a\": 2, \"b\": 1}}");
    }

    #[test]
    fn tojson_supports_indent() {
        let rendered = render("{{ payload|tojson(indent=2) }}", json!([1, 2]));

        assert_eq!(rendered, "[\n  1,\n  2\n]");
    }

    #[test]
    fn tojson_supports_ensure_ascii_false() {
        let rendered = render("{{ payload|tojson(ensure_ascii=false) }}", json!("中文"));
        assert_eq!(rendered, "\"中文\"");
    }

    #[test]
    fn tojson_supports_ensure_ascii_true() {
        let rendered = render("{{ payload|tojson(ensure_ascii=true) }}", json!("中文"));
        assert_eq!(rendered, "\"\\u4e2d\\u6587\"");
    }

    #[test]
    fn tojson_supports_separators() {
        let rendered = render(
            "{{ payload|tojson(separators=[',', ':']) }}",
            json!({
                "x": [1, 2]
            }),
        );

        assert_eq!(rendered, "{\"x\":[1,2]}");
    }

    #[test]
    fn tojson_uses_standard_serde_json_number_spelling() {
        let payload = serde_json::from_str(r#"{"x":2,"y":1.00}"#).unwrap();
        let rendered = render("{{ payload|tojson }}", payload);

        // The source spelling `1.00` is not preserved: serde_json's
        // `arbitrary_precision` feature, which would keep it, serializes
        // numbers in a private tagged form that MiniJinja renders literally.
        assert_eq!(rendered, "{\"x\": 2, \"y\": 1.0}");
    }

    #[test]
    fn serialized_json_numbers_do_not_leak_serde_private_representation() {
        let payload: serde_json::Value = serde_json::from_str(r#"{"x":2,"y":1.00}"#).unwrap();
        let rendered = render("{{ payload }}", payload);

        // Guards against enabling serde_json's `arbitrary_precision` feature:
        // with it, templates would print serde's private tagged number form.
        assert_eq!(rendered, r#"{"x": 2, "y": 1.0}"#);
    }

    #[test]
    fn tojson_supports_negative_indent_as_newline_only() {
        let rendered = render("{{ payload|tojson(indent=-1) }}", json!([1, 2]));
        assert_eq!(rendered, "[\n1,\n2\n]");
    }

    #[test]
    fn tojson_supports_string_indent() {
        let rendered = render("{{ payload|tojson(indent='  ') }}", json!([1, 2]));
        assert_eq!(rendered, "[\n  1,\n  2\n]");
    }

    #[test]
    fn tojson_supports_boolean_indent() {
        let rendered_true = render("{{ payload|tojson(indent=true) }}", json!([1, 2]));
        assert_eq!(rendered_true, "[\n 1,\n 2\n]");

        let rendered_false = render("{{ payload|tojson(indent=false) }}", json!([1, 2]));
        assert_eq!(rendered_false, "[\n1,\n2\n]");
    }

    #[test]
    fn tojson_combines_indent_sort_keys_separators_and_ensure_ascii() {
        let rendered = render(
            "{{ payload|tojson(ensure_ascii=true, sort_keys=true, separators=[',', ':'], indent='  ') }}",
            json!({
                "b": "<中>",
                "a": [1, 2]
            }),
        );

        assert_eq!(
            rendered,
            "{\n  \"a\":[\n    1,\n    2\n  ],\n  \"b\":\"<\\u4e2d>\"\n}"
        );
    }
}
