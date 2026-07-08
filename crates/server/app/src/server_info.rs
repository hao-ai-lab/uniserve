use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::config::Config;

const SENSITIVE_UNISERVE_ENV_PATTERNS: &[&str] =
    &["KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "AUTH"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerInfoConfigFormat {
    Text,
    Json,
}

/// Snapshot returned by `/server_info`.
#[derive(Debug, Clone)]
pub struct ServerInfoSnapshot {
    uniserve_config_text: String,
    uniserve_config_json: Value,
    uniserve_env: BTreeMap<String, String>,
    system_env: BTreeMap<String, String>,
}

impl ServerInfoSnapshot {
    /// Capture the runtime configuration fields available to the Rust frontend.
    pub fn from_config(config: &Config) -> Self {
        let uniserve_config_json = serde_json::to_value(config).unwrap_or_else(|error| {
            json!({
                "error": format!("failed to serialize server config: {error}")
            })
        });

        Self {
            uniserve_config_text: render_config_text(&uniserve_config_json),
            uniserve_config_json,
            uniserve_env: collect_uniserve_env(),
            system_env: collect_system_env(),
        }
    }

    pub fn response(&self, config_format: ServerInfoConfigFormat) -> Value {
        let uniserve_config = match config_format {
            ServerInfoConfigFormat::Text => Value::String(self.uniserve_config_text.clone()),
            ServerInfoConfigFormat::Json => self.uniserve_config_json.clone(),
        };

        json!({
            "uniserve_config": uniserve_config,
            "uniserve_env": self.uniserve_env.clone(),
            "system_env": self.system_env.clone(),
        })
    }
}

fn render_config_text(config: &Value) -> String {
    match config {
        Value::Object(fields) => fields
            .iter()
            .map(|(key, value)| format!("{key}={}", render_config_text_value(value)))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => render_config_text_value(config),
    }
}

fn render_config_text_value(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::String(value) => value.clone(),
        _ => value.to_string(),
    }
}

fn collect_uniserve_env() -> BTreeMap<String, String> {
    std::env::vars()
        .filter(|(key, _)| is_public_uniserve_env_key(key))
        .collect()
}

fn is_public_uniserve_env_key(key: &str) -> bool {
    let key = key.to_ascii_uppercase();
    key.starts_with("UNISERVE_")
        && !SENSITIVE_UNISERVE_ENV_PATTERNS
            .iter()
            .any(|pattern| key.contains(pattern))
}

fn collect_system_env() -> BTreeMap<String, String> {
    BTreeMap::from([
        ("arch".to_string(), std::env::consts::ARCH.to_string()),
        ("family".to_string(), std::env::consts::FAMILY.to_string()),
        ("os".to_string(), std::env::consts::OS.to_string()),
    ])
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use serde_json::{Value, json};

    use super::{is_public_uniserve_env_key, render_config_text};

    #[test]
    fn render_config_text_formats_config_snapshot() {
        let rendered = render_config_text(&json!({
            "model": "test-model",
            "served_model_name": ["served-model"],
            "chat_template": null,
            "enable_log_requests": true,
        }));
        let lines = rendered.lines().collect::<BTreeSet<_>>();

        assert_eq!(
            lines,
            BTreeSet::from([
                "chat_template=None",
                "enable_log_requests=true",
                "model=test-model",
                "served_model_name=[\"served-model\"]",
            ])
        );
        assert_eq!(
            render_config_text(&Value::String("inline".to_string())),
            "inline"
        );
        assert_eq!(render_config_text(&Value::Null), "None");
    }

    #[test]
    fn server_info_env_filter_excludes_sensitive_uniserve_keys() {
        for key in [
            "UNISERVE_API_KEY",
            "UNISERVE_AUTH_TOKEN",
            "UNISERVE_SECRET",
            "UNISERVE_PASSWORD",
            "UNISERVE_CREDENTIAL_FILE",
            "uniserve_token",
        ] {
            assert!(!is_public_uniserve_env_key(key), "{key}");
        }
    }

    #[test]
    fn server_info_env_filter_includes_public_uniserve_keys() {
        assert!(is_public_uniserve_env_key("UNISERVE_USE_MODELSCOPE"));
        assert!(!is_public_uniserve_env_key("OTHER_ENV"));
    }
}
