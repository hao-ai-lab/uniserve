//! Chat funnel coverage retargeted to the S02 `GenerateReqInput` -> `generate`
//! path. Broad renderer/parser unit coverage lives in the serving crate's own
//! module tests; this file exercises the end-to-end chat funnel.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use common::{
    assert_terminal_success, build_runtime, canonicalize_outputs, chat_input, drive, resolve_qwen3,
    scheduling_output, token_output,
};
use uniserve_engine_gateway::transport::protocol::{
    EngineCoreFinishReason, EngineCoreOutputs, StopReason,
};
use uniserve_serving::ServeEvent;
use uniserve_serving::chat::{ChatContent, ChatMessage};

fn script_reply(
    request_id: &'static str,
    bytes: &'static str,
) -> impl FnOnce(uniserve_engine_gateway::transport::protocol::EngineCoreRequest) -> EngineCoreOutputs
{
    move |request| {
        let prompt_len = request.generation.prompt_token_ids().len();
        let mut outputs = vec![scheduling_output(request_id)];
        for byte in bytes.bytes() {
            outputs.push(token_output(request_id, vec![byte as u32], None, None));
        }
        outputs.push(token_output(
            request_id,
            Vec::new(),
            Some(EngineCoreFinishReason::Stop),
            Some(StopReason::TokenId(0)),
        ));
        canonicalize_outputs(
            prompt_len,
            EngineCoreOutputs {
                outputs,
                ..Default::default()
            },
        )
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_funnel_renders_system_and_user_into_the_prompt() {
    let (runtime, mock) = build_runtime(resolve_qwen3());
    let messages = vec![
        ChatMessage::system("You are terse."),
        ChatMessage::user("Say hi"),
    ];
    let events = drive(
        &runtime,
        mock,
        chat_input("c-render", messages),
        script_reply("c-render", "Hi"),
    )
    .await;

    // The rendered prompt (echoed on Accepted) carries both the system and user text.
    let prompt = events
        .iter()
        .find_map(|event| match event {
            ServeEvent::Accepted {
                prompt_token_ids, ..
            } => Some(
                String::from_utf8(prompt_token_ids.iter().map(|id| *id as u8).collect()).unwrap(),
            ),
            _ => None,
        })
        .expect("accepted event");
    assert!(prompt.contains("You are terse."), "prompt: {prompt}");
    assert!(prompt.contains("Say hi"), "prompt: {prompt}");

    let text: String = events
        .iter()
        .filter_map(|event| match event {
            ServeEvent::TextDelta { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "Hi");
    assert_terminal_success(&events);
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_funnel_separates_reasoning_from_visible_answer() {
    let (runtime, mock) = build_runtime(resolve_qwen3());
    let messages = vec![ChatMessage::user("think then answer")];
    let events = drive(
        &runtime,
        mock,
        chat_input("c-reasoning", messages),
        script_reply("c-reasoning", "<think>secret plan</think>Final answer."),
    )
    .await;

    let reasoning: String = events
        .iter()
        .filter_map(|event| match event {
            ServeEvent::ReasoningDelta { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect();
    let visible: String = events
        .iter()
        .filter_map(|event| match event {
            ServeEvent::TextDelta { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(reasoning, "secret plan");
    assert_eq!(visible, "Final answer.");
    assert_terminal_success(&events);
    runtime.shutdown().await.unwrap();
}

#[test]
fn chat_message_content_helpers_are_available() {
    // Guard the retained chat protocol surface used by wire lowering.
    let message = ChatMessage::user("hello");
    assert!(matches!(
        message,
        ChatMessage::User { content: ChatContent::Text(text) } if text == "hello"
    ));
}
