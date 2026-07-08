//! Resolved model behavior snapshots used by the serving runtime.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use serde::{Deserialize, Serialize};
use uniserve_chat::ChatLlm;

/// Stable identity and capability snapshot for one single-model runtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelProfile {
    pub profile_id: String,
    pub family_id: String,
    pub dialect_id: String,
    pub tokenizer_fingerprint: String,
    pub supports_text: bool,
    pub supports_chat: bool,
    pub supports_multimodal_input: bool,
}

impl ModelProfile {
    /// Build the runtime profile visible from an already loaded chat/text stack.
    pub fn from_chat_runtime(chat: &ChatLlm) -> Self {
        let model_id = chat.model_id().to_string();
        Self {
            profile_id: model_id.clone(),
            family_id: model_id.clone(),
            dialect_id: "chat-template".to_string(),
            tokenizer_fingerprint: format!("tokenizer:{model_id}"),
            supports_text: true,
            supports_chat: true,
            supports_multimodal_input: chat.has_multimodal_backend(),
        }
    }

    /// Build a text-only profile fixture for tests and gateway-only runtimes.
    pub fn text_only(profile_id: impl Into<String>) -> Self {
        let profile_id = profile_id.into();
        Self {
            family_id: profile_id.clone(),
            dialect_id: "text".to_string(),
            tokenizer_fingerprint: format!("tokenizer:{profile_id}"),
            profile_id,
            supports_text: true,
            supports_chat: false,
            supports_multimodal_input: false,
        }
    }
}
