use std::sync::Arc;

use crate::chat::template::ChatTemplateLoadOptions;
use crate::text::backend::hf::HfTextBackend;
use crate::text::tokenizer::DynTokenizer;
use crate::text::{DynTextBackend, TextBackend as _};
use tracing::info;
use uniserve_model_profile::assets::{ResolvedModelFiles, load_model_config};
use uniserve_model_profile::{ModelProfile, ProfileDeploymentConfig};

use crate::chat::backend::{
    ChatBackend, DynChatBackend, LoadModelBackendsOptions, LoadedModelBackends,
    NewChatOutputProcessorOptions,
};
use crate::chat::error::Result;
use crate::chat::output::{
    DefaultChatOutputProcessor, HarmonyChatOutputProcessor, validate_harmony_parser_overrides,
};
use crate::chat::renderer::hf::HfChatRenderer;
use crate::chat::renderer::{DeepSeekV4ChatRenderer, DeepSeekV32ChatRenderer, DynChatRenderer};
use crate::chat::request::ChatRequest;
use crate::chat::{DynChatOutputProcessor, RendererSelection};

/// [`ChatBackend`] implementation built on Hugging Face model files.
pub struct HfChatBackend {
    model_id: String,
    model_type: String,
    tokenizer: DynTokenizer,
    chat_renderer: DynChatRenderer,
}

impl HfChatBackend {
    /// Load the chat backend from resolved Hugging Face model files.
    pub fn from_resolved_model_files(
        files: ResolvedModelFiles,
        model_id: String,
        options: LoadModelBackendsOptions,
        tokenizer: DynTokenizer,
    ) -> Result<Self> {
        let model_config = load_model_config(files.config_path.as_deref())?;
        let model_type = model_config.model_type().unwrap_or_default();
        let renderer = options.renderer.resolve(model_type);
        let chat_renderer: DynChatRenderer = match renderer {
            RendererSelection::Auto => unreachable!("renderer auto should be resolved above"),
            RendererSelection::Hf => Arc::new(HfChatRenderer::load(
                &files,
                ChatTemplateLoadOptions {
                    chat_template_content_format: options.chat_template_content_format,
                    chat_template: options.chat_template,
                    default_chat_template_kwargs: options.default_chat_template_kwargs,
                },
                None,
            )?),
            RendererSelection::DeepSeekV32 => Arc::new(DeepSeekV32ChatRenderer::new()),
            RendererSelection::DeepSeekV4 => Arc::new(DeepSeekV4ChatRenderer::new()),
        };

        info!(
            model_id,
            model_type,
            %renderer,
            "loaded chat backend with Hugging Face model files"
        );

        Ok(Self {
            model_id,
            model_type: model_type.to_string(),
            tokenizer,
            chat_renderer,
        })
    }
}

impl ChatBackend for HfChatBackend {
    fn chat_renderer(&self) -> DynChatRenderer {
        Arc::clone(&self.chat_renderer)
    }

    fn new_chat_output_processor(
        &self,
        request: &mut ChatRequest,
        options: NewChatOutputProcessorOptions<'_>,
    ) -> Result<DynChatOutputProcessor> {
        if self.model_type == "gpt_oss" {
            validate_harmony_parser_overrides(
                options.tool_call_parser,
                options.uniserve_reasoning_parser,
            )?;
            return Ok(Box::new(HarmonyChatOutputProcessor::new(request)?));
        }

        Ok(Box::new(DefaultChatOutputProcessor::new(
            request,
            &self.model_id,
            Arc::clone(&self.tokenizer),
            options.tool_call_parser,
            options.uniserve_reasoning_parser,
        )?))
    }
}

/// Load the Hugging Face text and chat backends for the given model id.
pub(super) async fn load_model_backends(
    model_id: &str,
    options: LoadModelBackendsOptions,
) -> Result<LoadedModelBackends> {
    let files = ResolvedModelFiles::new(model_id).await?;
    let profile_deployment = ProfileDeploymentConfig {
        renderer_id: Some(options.renderer.to_string()),
        chat_template_override: options.chat_template.clone(),
        language_model_only: options.language_model_only,
        ..Default::default()
    };
    let profile_files = files.clone();
    let text_backend =
        HfTextBackend::from_resolved_model_files(files.clone(), model_id.to_string())?;
    let tokenizer = text_backend.tokenizer();
    let text_backend: DynTextBackend = Arc::new(text_backend);

    let chat_backend = HfChatBackend::from_resolved_model_files(
        files,
        model_id.to_string(),
        options,
        Arc::clone(&tokenizer),
    )?;
    let profile =
        ModelProfile::resolve(model_id, &profile_files, &profile_deployment, &*tokenizer)?;
    let chat_backend: DynChatBackend = Arc::new(chat_backend);

    Ok(LoadedModelBackends {
        profile,
        text_backend,
        chat_backend,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Arc;

    use crate::text::tokenizer::{DynTokenizer, Tokenizer};
    use tempfile::tempdir;
    use uniserve_model_profile::assets::{ResolvedModelFiles, TokenizerSource};

    use super::HfChatBackend;
    use crate::chat::RendererSelection;
    use crate::chat::backend::{ChatBackend, LoadModelBackendsOptions};
    use crate::chat::request::{ChatContent, ChatMessage, ChatRequest};

    fn request_with_user_text(text: &str) -> ChatRequest {
        ChatRequest {
            request_id: "renderer-selection-test".to_string(),
            messages: vec![ChatMessage::User {
                content: ChatContent::Text(text.to_string()),
            }],
            ..ChatRequest::for_test()
        }
    }

    fn write_json(path: &std::path::Path, content: &str) {
        std::fs::write(path, content).unwrap();
    }

    fn resolved_files(config_json: &str, tokenizer_config_json: &str) -> ResolvedModelFiles {
        let dir = tempdir().unwrap();
        let root = dir.keep();
        let config_path = root.join("config.json");
        let tokenizer_config_path = root.join("tokenizer_config.json");
        write_json(&config_path, config_json);
        write_json(&tokenizer_config_path, tokenizer_config_json);

        ResolvedModelFiles {
            tokenizer: TokenizerSource::HuggingFace(PathBuf::from("/tmp/unused-tokenizer.json")),
            tokenizer_config_path: Some(tokenizer_config_path),
            generation_config_path: None,
            preprocessor_config_path: None,
            chat_template_path: None,
            config_path: Some(config_path),
        }
    }

    struct TestTokenizer;

    impl Tokenizer for TestTokenizer {
        fn encode(
            &self,
            _text: &str,
            _add_special_tokens: bool,
        ) -> crate::text::tokenizer::Result<Vec<u32>> {
            Ok(Vec::new())
        }

        fn decode(
            &self,
            _token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> crate::text::tokenizer::Result<String> {
            Ok(String::new())
        }

        fn token_to_id(&self, _token: &str) -> Option<u32> {
            None
        }
    }

    fn test_tokenizer() -> DynTokenizer {
        Arc::new(TestTokenizer)
    }

    fn render_prompt(
        renderer: RendererSelection,
        config_json: &str,
        tokenizer_config_json: &str,
    ) -> String {
        let backend = HfChatBackend::from_resolved_model_files(
            resolved_files(config_json, tokenizer_config_json),
            "test-model".to_string(),
            LoadModelBackendsOptions {
                renderer,
                language_model_only: false,
                chat_template_content_format: Default::default(),
                chat_template: None,
                default_chat_template_kwargs: HashMap::new(),
            },
            test_tokenizer(),
        )
        .unwrap();

        backend
            .chat_renderer()
            .render(&request_with_user_text("hello"))
            .unwrap()
            .prompt
            .into_text()
            .expect("renderer should return text prompt")
    }

    #[test]
    fn auto_uses_deepseek_renderer_for_deepseek_v32_model_type() {
        let prompt = render_prompt(
            RendererSelection::Auto,
            r#"{"model_type":"deepseek_v32"}"#,
            r#"{}"#,
        );

        assert_eq!(
            prompt,
            "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>"
        );
    }

    #[test]
    fn auto_uses_hf_renderer_for_other_model_types() {
        let prompt = render_prompt(
            RendererSelection::Auto,
            r#"{"model_type":"qwen2"}"#,
            r#"{"chat_template":"{{ messages[0].content }}"}"#,
        );

        assert_eq!(prompt, "hello");
    }

    #[test]
    fn explicit_deepseek_renderer_overrides_generic_model_type() {
        let prompt = render_prompt(
            RendererSelection::DeepSeekV32,
            r#"{"model_type":"qwen2"}"#,
            r#"{"chat_template":"{{ messages[0].content }}"}"#,
        );

        assert_eq!(
            prompt,
            "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>"
        );
    }

    #[test]
    fn explicit_hf_renderer_overrides_deepseek_v32_model_type() {
        let prompt = render_prompt(
            RendererSelection::Hf,
            r#"{"model_type":"deepseek_v32"}"#,
            r#"{"chat_template":"{{ messages[0].content }}"}"#,
        );

        assert_eq!(prompt, "hello");
    }

    #[test]
    fn auto_uses_nested_text_config_model_type() {
        let prompt = render_prompt(
            RendererSelection::Auto,
            r#"{"text_config":{"model_type":"deepseek_v32","num_attention_heads":32}}"#,
            r#"{}"#,
        );

        assert_eq!(
            prompt,
            "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>"
        );
    }
}
