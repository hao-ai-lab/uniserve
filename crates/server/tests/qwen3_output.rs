//! Qwen3 reasoning and tool-call output behavior across stream boundaries.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use futures::{StreamExt, stream};
use tempfile::tempdir;
use tokenizers::models::bpe::BPE;
use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};
use uniserve_server::profile::tokenizer::{DynTokenizer, HuggingFaceTokenizer};
use uniserve_server::serving::chat::output::AssistantEvent;
use uniserve_server::serving::chat::{
    AssistantBlockKind, ChatRequest, ChatToolChoice, Qwen3ChatOutputProcessor, Tool,
};
use uniserve_server::serving::text::{DecodedTextEvent, FinishReason, Finished};

/// Builds a tokenizer whose only special tokens are the reasoning delimiters
/// that `Qwen3ReasoningParser` requires and the `<|im_end|>` turn delimiter.
///
/// The tokenizer is loaded into memory, so the temporary directory holding
/// its JSON file can be dropped on return.
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

/// A single decoded delta containing a reasoning block, text, and two XML tool
/// calls separates into reasoning deltas, visible text with the tool-call
/// markup removed, and one call per `<tool_call>` block in order.
#[tokio::test]
async fn qwen3_processor_emits_reasoning_text_and_tool_calls()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let tokenizer = qwen_tokenizer()?;
    let prompt_end = tokenizer
        .token_to_id("<|im_end|>")
        .ok_or_else(|| std::io::Error::other("configured prompt-end token is unavailable"))?;
    let mut request = ChatRequest::for_test();
    request.tools = vec![Tool {
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
    // Tool parsing requires `Auto` with declared tools; the final argument
    // enables reasoning parsing.
    let processor = Qwen3ChatOutputProcessor::new(&mut request, tokenizer, true)?;

    // The prompt ends with `<|im_end|>` and has no reasoning delimiter after
    // it, so generation starts outside a reasoning section and the `<think>`
    // in the output opens one.
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
            finished: Some(Finished {
                prompt_token_count: 1,
                output_token_count: 1,
                internal_token_count: 0,
                finish_reason: FinishReason::stop_eos(),
            }),
        }),
    ]);

    let output = processor.parse(decoded);
    futures::pin_mut!(output);
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut calls = Vec::<(String, String)>::new();
    let mut finished = false;
    while let Some(event) = output.next().await {
        match event? {
            AssistantEvent::TextDelta {
                kind: AssistantBlockKind::Text,
                delta,
            } => text.push_str(&delta),
            AssistantEvent::TextDelta {
                kind: AssistantBlockKind::Reasoning,
                delta,
            } => reasoning.push_str(&delta),
            AssistantEvent::ToolCallStart { name, .. } => calls.push((name, String::new())),
            AssistantEvent::ToolCallArgumentsDelta { delta } => {
                calls
                    .last_mut()
                    .expect("tool arguments follow a named call")
                    .1
                    .push_str(&delta);
            }
            AssistantEvent::Done {
                output_token_count,
                finish_reason,
                ..
            } => {
                assert_eq!(output_token_count, 1);
                assert_eq!(finish_reason, FinishReason::stop_eos());
                finished = true;
            }
            _ => {}
        }
    }

    assert!(finished);
    assert_eq!(reasoning, "hidden");
    assert_eq!(text, "beforebetweenafter");
    assert_eq!(
        calls,
        vec![
            ("lookup".to_string(), r#"{"q":"x"}"#.to_string()),
            ("lookup".to_string(), r#"{"q":"y"}"#.to_string()),
        ]
    );
    Ok(())
}

/// Assistant output in stream order, with adjacent text deltas merged and
/// each call's argument deltas joined, so the comparison does not depend on
/// how deltas are split.
#[derive(Debug, PartialEq, Eq)]
enum OrderedOutput {
    Text(String),
    Call { name: String, arguments: String },
}

/// Text and tool calls keep their stream order when one decoded delta spans
/// call boundaries, both when the delta starts outside a call and when it
/// starts inside a call's arguments.
#[tokio::test]
async fn qwen3_processor_keeps_text_and_tool_call_order_within_a_delta()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut request = ChatRequest::for_test();
    request.tools = vec![Tool {
        name: "lookup".to_string(),
        description: None,
        parameters: serde_json::json!({"type": "object"}),
        strict: None,
    }];
    request.tool_choice = ChatToolChoice::Auto;
    let processor = Qwen3ChatOutputProcessor::new(&mut request, qwen_tokenizer()?, false)?;

    let text_delta = |delta: &str, finished: Option<Finished>| {
        Ok(DecodedTextEvent::TextDelta {
            delta: delta.to_string(),
            token_ids: Vec::new(),
            logprobs: None,
            finished,
        })
    };
    // The first delta starts outside any call and ends inside the second
    // call's arguments; the second delta starts inside those arguments.
    let decoded = stream::iter([
        Ok(DecodedTextEvent::Start {
            queued_at: None,
            scheduled_at: None,
            prompt_token_ids: Vec::new().into(),
            prompt_logprobs: None,
        }),
        text_delta(
            concat!(
                "before",
                "<tool_call>\n",
                r#"{"name":"lookup","arguments":{"q":"x"}}"#,
                "\n</tool_call>",
                "between",
                "<tool_call>\n",
                r#"{"name":"lookup","arguments":{"q":"#,
            ),
            None,
        ),
        text_delta(
            concat!(
                r#""y"}}"#,
                "\n</tool_call>",
                "after",
                "<tool_call>\n",
                r#"{"name":"lookup","arguments":{"q":"z"}}"#,
                "\n</tool_call>",
            ),
            Some(Finished {
                prompt_token_count: 0,
                output_token_count: 2,
                internal_token_count: 0,
                finish_reason: FinishReason::stop_eos(),
            }),
        ),
    ]);

    let output = processor.parse(decoded);
    futures::pin_mut!(output);
    let mut ordered = Vec::<OrderedOutput>::new();
    let mut finished = false;
    while let Some(event) = output.next().await {
        match event? {
            AssistantEvent::TextDelta { kind, delta } => {
                assert_eq!(kind, AssistantBlockKind::Text);
                if let Some(OrderedOutput::Text(text)) = ordered.last_mut() {
                    text.push_str(&delta);
                } else {
                    ordered.push(OrderedOutput::Text(delta));
                }
            }
            AssistantEvent::ToolCallStart { name, .. } => ordered.push(OrderedOutput::Call {
                name,
                arguments: String::new(),
            }),
            AssistantEvent::ToolCallArgumentsDelta { delta } => {
                let Some(OrderedOutput::Call { arguments, .. }) = ordered.last_mut() else {
                    panic!("arguments delta {delta:?} does not follow its tool call");
                };
                arguments.push_str(&delta);
            }
            AssistantEvent::Done { .. } => finished = true,
            _ => {}
        }
    }

    let call = |q: &str| OrderedOutput::Call {
        name: "lookup".to_string(),
        arguments: format!(r#"{{"q":"{q}"}}"#),
    };
    assert!(finished);
    assert_eq!(
        ordered,
        vec![
            OrderedOutput::Text("before".to_string()),
            call("x"),
            OrderedOutput::Text("between".to_string()),
            call("y"),
            OrderedOutput::Text("after".to_string()),
            call("z"),
        ]
    );
    Ok(())
}

/// With reasoning parsing disabled, `<think>` delimiters stream verbatim as
/// assistant content instead of opening a reasoning block, including when the
/// delimiters arrive in separate deltas.
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
            finished: None,
        }),
        Ok(DecodedTextEvent::TextDelta {
            delta: "hi</think>after".to_string(),
            token_ids: Vec::new(),
            logprobs: None,
            finished: Some(Finished {
                prompt_token_count: 0,
                output_token_count: 2,
                internal_token_count: 0,
                finish_reason: FinishReason::stop_eos(),
            }),
        }),
    ]);

    let output = processor.parse(decoded);
    futures::pin_mut!(output);
    let mut text = String::new();
    let mut finished = false;
    while let Some(event) = output.next().await {
        match event? {
            AssistantEvent::TextDelta { kind, delta } => {
                assert_eq!(kind, AssistantBlockKind::Text);
                text.push_str(&delta);
            }
            AssistantEvent::Done {
                output_token_count,
                finish_reason,
                ..
            } => {
                assert_eq!(output_token_count, 2);
                assert_eq!(finish_reason, FinishReason::stop_eos());
                finished = true;
            }
            _ => {}
        }
    }

    assert!(finished);
    assert_eq!(text, "<think>hi</think>after");
    Ok(())
}
