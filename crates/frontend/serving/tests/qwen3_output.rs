use std::sync::Arc;

use futures::stream;
use tempfile::tempdir;
use tokenizers::models::bpe::BPE;
use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};
use uniserve_model_profile::tokenizer::{DynTokenizer, HuggingFaceTokenizer};
use uniserve_serving::chat::{
    AssistantMessageExt as _, ChatEventStream, ChatRequest, ChatTool, ChatToolChoice,
    Qwen3ChatOutputProcessor,
};
use uniserve_serving::text::{DecodedTextEvent, FinishReason, Finished};

fn qwen_tokenizer() -> Result<DynTokenizer, Box<dyn std::error::Error + Send + Sync>> {
    let model = BPE::builder()
        .vocab_and_merges([("<unk>".to_string(), 0), ("a".to_string(), 1)], Vec::new())
        .unk_token("<unk>".to_string())
        .build()?;
    let mut tokenizer = TokenizerBuilder::new(model);
    tokenizer.add_special_tokens(&[
        AddedToken::from("<think>", true),
        AddedToken::from("</think>", true),
        AddedToken::from("<|im_end|>", true),
    ]);
    let directory = tempdir()?;
    let path = directory.path().join("tokenizer.json");
    tokenizer.save(&path, false)?;
    Ok(Arc::new(HuggingFaceTokenizer::new(&path)?))
}

#[tokio::test]
async fn qwen3_processor_emits_reasoning_text_and_tool_calls()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let tokenizer = qwen_tokenizer()?;
    let prompt_end = tokenizer
        .token_to_id("<|im_end|>")
        .ok_or_else(|| std::io::Error::other("configured prompt-end token is unavailable"))?;
    let mut request = ChatRequest::for_test();
    request.tools = vec![ChatTool {
        name: "lookup".to_string(),
        description: None,
        parameters: serde_json::json!({
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"]
        }),
        strict: None,
    }];
    request.tool_choice = ChatToolChoice::Auto;
    let processor = Qwen3ChatOutputProcessor::new(&mut request, tokenizer, true)?;
    let decoded = stream::iter([
        Ok(DecodedTextEvent::Start {
            queued_at: None,
            scheduled_at: None,
            prompt_token_ids: vec![prompt_end].into(),
            prompt_logprobs: None,
        }),
        Ok(DecodedTextEvent::TextDelta {
            delta: concat!(
                "<think>hidden</think>before",
                "<tool_call>\n",
                r#"{"name":"lookup","arguments":{"q":"x"}}"#,
                "\n</tool_call>",
                "between",
                "<tool_call>\n",
                r#"{"name":"lookup","arguments":{"q":"y"}}"#,
                "\n</tool_call>",
                "after"
            )
            .to_string(),
            token_ids: Vec::new(),
            logprobs: None,
            public_commit: None,
            finished: Some(Finished {
                prompt_token_count: 1,
                output_token_count: 1,
                internal_token_count: 0,
                finish_reason: FinishReason::stop_eos(),
            }),
        }),
    ]);
    let output = processor.process(Box::pin(decoded))?;

    let collected = ChatEventStream::new("qwen3-output".to_string(), output)
        .collect_message()
        .await?;
    assert_eq!(collected.message.reasoning().as_deref(), Some("hidden"));
    assert_eq!(collected.message.text(), "beforebetweenafter");
    let calls = collected.message.tool_calls().collect::<Vec<_>>();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].name, "lookup");
    assert_eq!(calls[0].arguments, r#"{"q":"x"}"#);
    assert_eq!(calls[1].name, "lookup");
    assert_eq!(calls[1].arguments, r#"{"q":"y"}"#);
    Ok(())
}

/// With reasoning parsing disabled, `<think>` delimiters stream verbatim as
/// assistant content instead of opening a reasoning block.
#[tokio::test]
async fn disabled_reasoning_parsing_streams_delimiters_as_content()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let tokenizer = qwen_tokenizer()?;
    let mut request = ChatRequest::for_test();
    let processor = Qwen3ChatOutputProcessor::new(&mut request, tokenizer, false)?;
    let decoded = stream::iter([
        Ok(DecodedTextEvent::Start {
            queued_at: None,
            scheduled_at: None,
            prompt_token_ids: Vec::new().into(),
            prompt_logprobs: None,
        }),
        Ok(DecodedTextEvent::TextDelta {
            delta: "<think>".to_string(),
            token_ids: Vec::new(),
            logprobs: None,
            public_commit: None,
            finished: None,
        }),
        Ok(DecodedTextEvent::TextDelta {
            delta: "hi</think>after".to_string(),
            token_ids: Vec::new(),
            logprobs: None,
            public_commit: None,
            finished: Some(Finished {
                prompt_token_count: 0,
                output_token_count: 2,
                internal_token_count: 0,
                finish_reason: FinishReason::stop_eos(),
            }),
        }),
    ]);
    let output = processor.process(Box::pin(decoded))?;
    let collected = ChatEventStream::new("qwen3-raw-output".to_string(), output)
        .collect_message()
        .await?;
    assert_eq!(collected.message.reasoning(), None);
    assert_eq!(collected.message.text(), "<think>hi</think>after");
    Ok(())
}
