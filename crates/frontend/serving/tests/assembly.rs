//! Stream-assembly oracle: `generate()` over a scripted engine yields ordered
//! `ServeEvent` sequences with exactly one gateway submission.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use common::{
    assert_terminal_success, build_runtime, canonicalize_outputs, chat_input, drive, resolve_qwen3,
    scheduling_output, text_input, token_output,
};
use uniserve_engine_gateway::transport::protocol::{
    EngineCoreFinishReason, EngineCoreOutputs, StopReason,
};
use uniserve_serving::{CandidateId, FinishStatus, OutputContract, ServeEvent};

fn kinds(events: &[ServeEvent]) -> Vec<&'static str> {
    events
        .iter()
        .map(|event| match event {
            ServeEvent::Accepted { .. } => "accepted",
            ServeEvent::Scheduled { .. } => "scheduled",
            ServeEvent::PublicCommit { .. } => "public_commit",
            ServeEvent::TextDelta { .. } => "text_delta",
            ServeEvent::InternalTextDelta { .. } => "internal_text_delta",
            ServeEvent::ReasoningDelta { .. } => "reasoning_delta",
            ServeEvent::OutputBlockStart { .. } => "block_start",
            ServeEvent::OutputBlockEnd { .. } => "block_end",
            ServeEvent::ToolCallStart { .. } => "tool_call_start",
            ServeEvent::ToolCallArgumentsDelta { .. } => "tool_call_args",
            ServeEvent::ToolCallEnd { .. } => "tool_call_end",
            ServeEvent::ImageBegin { .. } => "image_begin",
            ServeEvent::ImageStep { .. } => "image_step",
            ServeEvent::ImageCommit { .. } => "image_commit",
            ServeEvent::ImageDone { .. } => "image_done",
            ServeEvent::Usage { .. } => "usage",
            ServeEvent::Finished { .. } => "finished",
            ServeEvent::Rejected { .. } => "rejected",
            ServeEvent::Cancelled { .. } => "cancelled",
            ServeEvent::Aborted { .. } => "aborted",
            ServeEvent::Failed { .. } => "failed",
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn text_stream_orders_accepted_scheduled_delta_usage_finished() {
    let (runtime, mock) = build_runtime(resolve_qwen3());
    let events = drive(&runtime, mock, text_input("t-order", "hello"), |request| {
        let prompt_len = request.generation.prompt_token_ids().len();
        canonicalize_outputs(
            prompt_len,
            EngineCoreOutputs {
                outputs: vec![
                    scheduling_output("t-order"),
                    token_output("t-order", vec![b'H' as u32], None, None),
                    token_output("t-order", vec![b'i' as u32], None, None),
                    token_output(
                        "t-order",
                        Vec::new(),
                        Some(EngineCoreFinishReason::Length),
                        None,
                    ),
                ],
                ..Default::default()
            },
        )
    })
    .await;

    let order = kinds(&events);
    assert_eq!(order.first(), Some(&"accepted"));
    assert!(order.contains(&"scheduled"));
    let first_delta = order
        .iter()
        .position(|k| *k == "text_delta")
        .expect("text delta");
    let usage = order.iter().position(|k| *k == "usage").expect("usage");
    let finished = order
        .iter()
        .position(|k| *k == "finished")
        .expect("finished");
    assert!(
        first_delta < usage && usage < finished,
        "ordering: {order:?}"
    );
    let text: String = events
        .iter()
        .filter_map(|event| match event {
            ServeEvent::TextDelta { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "Hi");
    assert!(matches!(
        events.last(),
        Some(ServeEvent::Finished {
            reason: FinishStatus::Length,
            ..
        })
    ));
    assert_terminal_success(&events);
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn text_emits_token_ids_only_when_output_contract_requests_them() {
    let (runtime, mock) = build_runtime(resolve_qwen3());
    let mut request = text_input("t-tokens", "hi");
    request.output = OutputContract::Tokens;
    let events = drive(&runtime, mock, request, |request| {
        let prompt_len = request.generation.prompt_token_ids().len();
        canonicalize_outputs(
            prompt_len,
            EngineCoreOutputs {
                outputs: vec![
                    scheduling_output("t-tokens"),
                    token_output("t-tokens", vec![b'A' as u32], None, None),
                    token_output(
                        "t-tokens",
                        Vec::new(),
                        Some(EngineCoreFinishReason::Length),
                        None,
                    ),
                ],
                ..Default::default()
            },
        )
    })
    .await;

    let delta = events
        .iter()
        .find_map(|event| match event {
            ServeEvent::TextDelta { token_ids, .. } if !token_ids.is_empty() => {
                Some(token_ids.clone())
            }
            _ => None,
        })
        .expect("token ids present under Tokens contract");
    assert_eq!(delta, vec![b'A' as u32]);
    assert_terminal_success(&events);
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_stream_surfaces_output_blocks() {
    let (runtime, mock) = build_runtime(resolve_qwen3());
    let request = chat_input(
        "c-blocks",
        vec![uniserve_serving::chat::ChatMessage::user("Say hi")],
    );
    let events = drive(&runtime, mock, request, |request| {
        let prompt_len = request.generation.prompt_token_ids().len();
        canonicalize_outputs(
            prompt_len,
            EngineCoreOutputs {
                outputs: vec![
                    scheduling_output("c-blocks"),
                    token_output("c-blocks", vec![b'H' as u32], None, None),
                    token_output("c-blocks", vec![b'i' as u32], None, None),
                    token_output(
                        "c-blocks",
                        Vec::new(),
                        Some(EngineCoreFinishReason::Stop),
                        Some(StopReason::TokenId(b'!' as u32)),
                    ),
                ],
                ..Default::default()
            },
        )
    })
    .await;

    let order = kinds(&events);
    assert_eq!(order.first(), Some(&"accepted"));
    assert!(order.contains(&"block_start"), "chat blocks: {order:?}");
    assert!(order.contains(&"block_end"), "chat blocks: {order:?}");
    let text: String = events
        .iter()
        .filter_map(|event| match event {
            ServeEvent::TextDelta { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "Hi");
    assert!(matches!(
        events.last(),
        Some(ServeEvent::Finished {
            candidate_id: CandidateId::PRIMARY,
            reason: FinishStatus::Stop { .. },
            ..
        })
    ));
    assert_terminal_success(&events);
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stop_token_finish_maps_to_stop_status() {
    let (runtime, mock) = build_runtime(resolve_qwen3());
    let events = drive(&runtime, mock, text_input("t-stop", "hello"), |request| {
        let prompt_len = request.generation.prompt_token_ids().len();
        canonicalize_outputs(
            prompt_len,
            EngineCoreOutputs {
                outputs: vec![
                    scheduling_output("t-stop"),
                    token_output("t-stop", vec![b'x' as u32], None, None),
                    token_output(
                        "t-stop",
                        Vec::new(),
                        Some(EngineCoreFinishReason::Stop),
                        None,
                    ),
                ],
                ..Default::default()
            },
        )
    })
    .await;
    assert!(matches!(
        events.last(),
        Some(ServeEvent::Finished {
            reason: FinishStatus::Stop { .. },
            ..
        })
    ));
    assert_terminal_success(&events);
    runtime.shutdown().await.unwrap();
}
