//! Chat-template source loading, Jinja environment construction, and
//! compiled-template rendering.
//!
//! [`load_chat_template`] and [`resolve_chat_template`] turn template files
//! and configured values into template source. `CompiledChatTemplate` pairs
//! one compiled MiniJinja environment with the content format resolved for it
//! (detection lives in `format`), and `TemplateContext` is the set of global
//! variables a render sees.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::profile::assets::HfSpecialTokens;
use minijinja::Environment;
use serde::{Deserialize, Serialize};
use serde_json::{self};

use super::error::TemplateError;
use super::format::{
    ChatTemplateContentFormat, ChatTemplateContentFormatOption, detect_chat_template_content_format,
};
use super::tojson::hf_tojson_filter;
use crate::serving::chat::ReasoningEffort;
use crate::serving::chat::template::renderer::hf::{TemplateMessage, TemplateTool};

type Result<T> = std::result::Result<T, TemplateError>;

/// Builds a pre-configured environment with the given template string.
///
/// The template is registered as `"chat"` and compiled immediately, so syntax
/// errors are returned here.
fn build_environment(template: String) -> Result<Environment<'static>> {
    let mut env = Environment::new();

    // Hugging Face chat templates are written for Jinja environments with
    // `trim_blocks` and `lstrip_blocks` enabled; their whitespace output
    // depends on both.
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);

    env.add_template_owned("chat".to_owned(), template)?;

    // Python string and dict methods (`startswith`, `items`, ...) resolve
    // through pycompat. `TemplateMap` defers every method call to this
    // callback. The `tojson` registration replaces MiniJinja's built-in filter
    // with the Python `json.dumps`-compatible one.
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_filter("tojson", hf_tojson_filter);

    Ok(env)
}

#[serde_with::skip_serializing_none]
#[derive(Default, Serialize)]
/// Values installed into the Jinja environment for one render.
///
/// Serializes as one flat map in field order: the flattened special tokens and
/// template kwargs become top-level variables, `None` fields are omitted (so
/// templates see them as undefined), and on a key collision the later entry
/// wins.
pub(super) struct TemplateContext<'a> {
    pub(super) messages: &'a [TemplateMessage],
    pub(super) add_generation_prompt: bool,
    pub(super) continue_final_message: bool,
    pub(super) tools: Option<&'a [TemplateTool]>,
    pub(super) documents: Option<&'a [serde_json::Value]>,
    #[serde(flatten)]
    pub(super) special_tokens: Option<&'a HfSpecialTokens>,
    #[serde(flatten)]
    pub(super) template_kwargs: Option<&'a HashMap<String, serde_json::Value>>,
    // Declared after `template_kwargs` so that a request's `reasoning_effort`
    // overrides a default kwarg of the same name; when the request sets none,
    // the field is omitted and the kwarg stays visible.
    pub(super) reasoning_effort: Option<ReasoningEffort>,
}

/// Loads chat template from a file (`.jinja` or `.json` containing Jinja).
///
/// A `.json` file must hold either a JSON string or an object with a string
/// `chat_template` field; its template is returned verbatim. Any other file is
/// read as Jinja source, trimmed, and has each literal backslash-`n` sequence
/// replaced with a newline, which also rewrites a `\n` that the template
/// meant literally. Never returns `Ok(None)`.
///
/// # Errors
///
/// Returns `TemplateError::ReadTemplateFile`, `ParseTemplateJson`, or
/// `InvalidTemplateJson` when the file cannot be read or its JSON form is
/// invalid.
pub fn load_chat_template(template_path: &Path) -> Result<Option<String>> {
    let content = fs::read_to_string(template_path).map_err(TemplateError::ReadTemplateFile)?;

    if template_path.extension().is_some_and(|ext| ext == "json") {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum ChatTemplateFile {
            String(String),
            Object { chat_template: String },
        }

        let json_value =
            serde_json::from_str(&content).map_err(TemplateError::ParseTemplateJson)?;
        let json_template =
            serde_json::from_value(json_value).map_err(|_| TemplateError::InvalidTemplateJson)?;

        return Ok(Some(match json_template {
            ChatTemplateFile::String(template) => template,
            ChatTemplateFile::Object { chat_template } => chat_template,
        }));
    }

    let template = content.trim().replace("\\n", "\n");
    Ok(Some(template))
}

/// Resolves a configured chat template value into a template string.
///
/// An existing path is loaded with [`load_chat_template`]. Otherwise a value
/// containing `{`, `}`, or a newline is taken as inline Jinja source, used
/// as-is; anything else is treated as a mistyped path and rejected with
/// `TemplateError::MissingTemplatePath`.
pub fn resolve_chat_template(chat_template: &str) -> Result<String> {
    let path = Path::new(chat_template);
    if path.exists() {
        return load_chat_template(path).map(|template| template.unwrap_or_default());
    }

    const JINJA_CHARS: [char; 3] = ['{', '}', '\n'];
    if chat_template.chars().any(|c| JINJA_CHARS.contains(&c)) {
        return Ok(chat_template.to_string());
    }

    Err(TemplateError::MissingTemplatePath)
}

/// One compiled chat template with its Jinja environment and resolved content
/// format.
pub(super) struct CompiledChatTemplate {
    /// Fully configured environment holding the template under the name
    /// `"chat"`.
    env: Environment<'static>,
    content_format: ChatTemplateContentFormat,
}

impl CompiledChatTemplate {
    /// Compiles the given chat template string into a [`CompiledChatTemplate`].
    pub(super) fn new(
        template: String,
        content_format: ChatTemplateContentFormatOption,
    ) -> Result<Self> {
        let content_format = match content_format {
            ChatTemplateContentFormatOption::Auto => detect_chat_template_content_format(&template),
            ChatTemplateContentFormatOption::String => ChatTemplateContentFormat::String,
            ChatTemplateContentFormatOption::OpenAi => ChatTemplateContentFormat::OpenAi,
        };
        let env = build_environment(template)?;
        Ok(Self {
            env,
            content_format,
        })
    }

    /// Applies the compiled template to the given context and returns the
    /// rendered prompt.
    pub(super) fn apply(&self, ctx: TemplateContext<'_>) -> Result<String> {
        let tmpl = self.env.get_template("chat")?;
        tmpl.render(ctx).map_err(TemplateError::from)
    }

    /// Returns the resolved message-content shape expected by the template.
    pub(super) fn content_format(&self) -> ChatTemplateContentFormat {
        self.content_format
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::*;

    #[test]
    fn test_chat_template_state_valid_template() {
        let template = CompiledChatTemplate::new(
            "{{ messages }}".to_string(),
            ChatTemplateContentFormatOption::Auto,
        )
        .unwrap();
        let result = template.apply(TemplateContext::default()).unwrap();
        assert_eq!(result, "[]");
    }

    #[test]
    fn test_chat_template_state_invalid_template() {
        let result = CompiledChatTemplate::new(
            "{% invalid".to_string(),
            ChatTemplateContentFormatOption::Auto,
        );
        assert!(result.is_err());
        let err = result.err().unwrap().to_string();
        assert!(
            err.contains("failed to render jinja template"),
            "Error should explain parse failure, got: {err}"
        );
    }

    #[test]
    fn test_special_tokens_undefined_when_not_provided() {
        let template = "{% if bos_token is defined %}{{ bos_token }}{% endif %}hello";
        let template =
            CompiledChatTemplate::new(template.to_string(), ChatTemplateContentFormatOption::Auto)
                .unwrap();

        let result = template.apply(TemplateContext::default()).unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_load_chat_template_from_file_jinja() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("chat_template.jinja");
        fs::write(&path, "{{ messages }}").unwrap();

        let template = load_chat_template(&path).unwrap();

        assert_eq!(template.as_deref(), Some("{{ messages }}"));
    }

    #[test]
    fn test_resolve_chat_template_from_inline_literal() {
        let template = resolve_chat_template("{{ messages }}").unwrap();

        assert_eq!(template, "{{ messages }}");
    }

    #[test]
    fn test_resolve_chat_template_from_existing_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("chat_template.jinja");
        fs::write(&path, "{{ messages }}").unwrap();

        let template = resolve_chat_template(path.to_str().unwrap()).unwrap();

        assert_eq!(template, "{{ messages }}");
    }

    #[test]
    fn test_resolve_chat_template_rejects_missing_path_like_value() {
        let error = resolve_chat_template("missing_template.jinja").unwrap_err();

        assert!(matches!(error, TemplateError::MissingTemplatePath));
    }

    #[test]
    fn test_load_chat_template_from_file_json_string() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("chat_template.json");
        fs::write(&path, "\"{{ messages }}\"").unwrap();

        let template = load_chat_template(&path).unwrap();

        assert_eq!(template.as_deref(), Some("{{ messages }}"));
    }

    #[test]
    fn test_load_chat_template_from_file_json_object() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("chat_template.json");
        fs::write(&path, r#"{"chat_template":"{{ messages }}"}"#).unwrap();

        let template = load_chat_template(&path).unwrap();

        assert_eq!(template.as_deref(), Some("{{ messages }}"));
    }
}
