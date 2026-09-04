//! MiniJinja object wrappers that preserve JSON map order and value semantics.

use std::sync::Arc;

use indexmap::IndexMap;
use minijinja::value::{Enumerator, Object, ObjectExt, ObjectRepr};
use minijinja::{Error as TemplateError, ErrorKind as TemplateErrorKind, State};
use serde::Serialize;
use serde_json::Value as JsonValue;

/// A wrapper around `minijinja::Value` that can be constructed with `to_template_value` and used
/// as a value in the chat template.
#[derive(Debug, Serialize)]
#[serde(transparent)]
pub(super) struct TemplateValue(minijinja::Value);

/// Wraps a JSON value while preserving ordered-object template semantics.
pub(super) fn to_template_value(value: JsonValue) -> TemplateValue {
    TemplateValue(match value {
        JsonValue::Array(values) => values
            .into_iter()
            .map(to_template_value)
            .map(|value| value.0)
            .collect::<minijinja::Value>(),
        JsonValue::Object(values) => minijinja::Value::from_object(TemplateMap(
            values
                .into_iter()
                .map(|(key, value)| (key, to_template_value(value).0))
                .collect(),
        )),
        // For primitive values, directly convert them to `minijinja::Value` using `from_serialize`.
        value => minijinja::Value::from_serialize(value),
    })
}

/// A custom map type that always returns `UnknownMethod` for method calls, so that pycompat can
/// always handle dict methods through the unknown-method callback.
///
/// Use `IndexMap` to preserve the original key order when iterating.
///
/// MiniJinja's default map can resolve a same-named field before Python dict methods. HF templates
/// commonly call `dict.items`, which would fail if the map had an `items` field.
/// See issue: https://github.com/mitsuhiko/minijinja/issues/903
#[derive(Debug)]
struct TemplateMap(IndexMap<String, minijinja::Value>);

impl Object for TemplateMap {
    /// Returns the template representation of the value.
    fn repr(self: &Arc<Self>) -> ObjectRepr {
        ObjectRepr::Map
    }

    /// Returns an indexed child value.
    fn get_value(self: &Arc<Self>, key: &minijinja::Value) -> Option<minijinja::Value> {
        self.0.get(key.as_str()?).cloned()
    }

    /// Returns a named child value.
    fn get_value_by_str(self: &Arc<Self>, key: &str) -> Option<minijinja::Value> {
        self.0.get(key).cloned()
    }

    /// Returns an iterator over the value.
    fn enumerate(self: &Arc<Self>) -> Enumerator {
        self.mapped_rev_enumerator(|this| {
            Box::new(
                this.0
                    .keys()
                    .map(|key| minijinja::Value::from(key.as_str())),
            )
        })
    }

    /// Returns the number of enumerable elements.
    fn enumerator_len(self: &Arc<Self>) -> Option<usize> {
        Some(self.0.len())
    }

    /// Returns the result of invoking a supported template method.
    fn call_method(
        self: &Arc<Self>,
        _state: &State<'_, '_>,
        _method: &str,
        _args: &[minijinja::Value],
    ) -> std::result::Result<minijinja::Value, TemplateError> {
        // Always return `UnknownMethod` for method calls,
        // so that pycompat can handle dict methods through the unknown-method callback.
        Err(TemplateError::from(TemplateErrorKind::UnknownMethod))
    }
}
