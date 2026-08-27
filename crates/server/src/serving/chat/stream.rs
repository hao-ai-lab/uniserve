use std::sync::Arc;

use crate::serving::text::{DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs};
use futures::{Stream, StreamExt as _, pin_mut};

use crate::serving::chat::FinishReason;
use crate::serving::chat::error::{Error, Result};
use crate::serving::chat::{AssistantContentBlock, AssistantMessage, ChatEvent};

/// Final structured assistant message plus terminal stream metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectedAssistantMessage {
    pub message: AssistantMessage,
    pub prompt_token_count: usize,
    pub prompt_token_ids: Arc<[u32]>,
    pub prompt_logprobs: Option<DecodedPromptLogprobs>,
    pub logprobs: Option<DecodedLogprobs>,
    pub token_ids: Vec<u32>,
    pub output_token_count: usize,
    pub visible_output_token_count: usize,
    pub internal_token_count: usize,
    pub finish_reason: FinishReason,
}

impl CollectedAssistantMessage {
    pub async fn collect(
        request_id: impl Into<String>,
        stream: impl Stream<Item = Result<ChatEvent>> + Send,
    ) -> Result<Self> {
        let request_id = request_id.into();
        pin_mut!(stream);
        let mut message = AssistantMessage::default();
        let mut prompt_logprobs = None;
        let mut prompt_token_ids: Arc<[u32]> = Arc::from([]);
        let mut logprob_positions: Vec<DecodedPositionLogprobs> = Vec::new();
        let mut token_ids: Vec<u32> = Vec::new();
        while let Some(event) = stream.next().await.transpose()? {
            match event {
                ChatEvent::Start {
                    prompt_logprobs: start_prompt_logprobs,
                    prompt_token_ids: start_prompt_token_ids,
                    ..
                } => {
                    prompt_logprobs = start_prompt_logprobs;
                    prompt_token_ids = start_prompt_token_ids;
                }
                ChatEvent::BlockEnd { block, .. } => message.push_block(block),
                ChatEvent::LogprobsDelta {
                    logprobs,
                    token_ids: delta_ids,
                } => {
                    if let Some(logprobs) = logprobs {
                        logprob_positions.extend(logprobs.positions);
                    }
                    token_ids.extend(delta_ids);
                }
                ChatEvent::Done {
                    message: done,
                    prompt_token_count,
                    output_token_count,
                    visible_output_token_count,
                    internal_token_count,
                    finish_reason,
                } => {
                    return Ok(CollectedAssistantMessage {
                        message: done,
                        prompt_token_count,
                        prompt_token_ids,
                        prompt_logprobs,
                        logprobs: (!logprob_positions.is_empty()).then_some(DecodedLogprobs {
                            positions: logprob_positions,
                        }),
                        token_ids,
                        output_token_count,
                        visible_output_token_count,
                        internal_token_count,
                        finish_reason,
                    });
                }
                ChatEvent::ToolCallEnd { call, .. } => {
                    message.push_block(AssistantContentBlock::ToolCall(call));
                }
                ChatEvent::BlockStart { .. }
                | ChatEvent::BlockDelta { .. }
                | ChatEvent::PublicCommit { .. }
                | ChatEvent::ToolCallStart { .. }
                | ChatEvent::ToolCallArgumentsDelta { .. } => {}
            }
        }

        Err(Error::StreamClosedBeforeTerminalOutput { request_id })
    }
}

#[cfg(test)]
mod tests {

    use crate::serving::chat::FinishReason;
    use crate::serving::text::{
        DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs, DecodedTokenLogprob,
    };
    use futures::stream;

    use super::CollectedAssistantMessage;
    use crate::serving::chat::ChatEvent;
    use crate::serving::chat::error::Error;

    #[tokio::test]
    async fn collect_message_requires_terminal_done_event() {
        let stream = stream::iter([Ok(ChatEvent::Start {
            queued_at: None,
            scheduled_at: None,
            prompt_token_ids: vec![].into(),
            prompt_logprobs: None,
        })]);

        let error = CollectedAssistantMessage::collect("chat-missing-done", stream)
            .await
            .expect_err("missing done");
        assert!(matches!(
            error,
            Error::StreamClosedBeforeTerminalOutput { request_id }
            if request_id == "chat-missing-done"
        ));
    }

    #[tokio::test]
    async fn collect_message_retains_prompt_and_sample_logprobs() {
        let stream = stream::iter(vec![
            Ok(ChatEvent::Start {
                queued_at: None,
                scheduled_at: None,
                prompt_token_ids: vec![10, 11].into(),
                prompt_logprobs: Some(DecodedPromptLogprobs {
                    first_token_id: 0,
                    first_token: "o".to_string(),
                    scored_positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "p".to_string(),
                            logprob: -0.1,
                            rank: 1,
                        }],
                    }],
                }),
            }),
            Ok(ChatEvent::LogprobsDelta {
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "a".to_string(),
                            logprob: -0.2,
                            rank: 1,
                        }],
                    }],
                }),
                token_ids: vec![],
            }),
            Ok(ChatEvent::Done {
                message: Default::default(),
                prompt_token_count: 2,
                output_token_count: 1,
                visible_output_token_count: 1,
                internal_token_count: 0,
                finish_reason: FinishReason::stop_eos(),
            }),
        ]);

        let collected = CollectedAssistantMessage::collect("chat-logprobs", stream)
            .await
            .unwrap();
        assert_eq!(
            collected,
            CollectedAssistantMessage {
                message: Default::default(),
                prompt_token_count: 2,
                visible_output_token_count: 1,
                internal_token_count: 0,
                prompt_token_ids: vec![10, 11].into(),
                prompt_logprobs: Some(DecodedPromptLogprobs {
                    first_token_id: 0,
                    first_token: "o".to_string(),
                    scored_positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "p".to_string(),
                            logprob: -0.1,
                            rank: 1,
                        }],
                    }],
                }),
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "a".to_string(),
                            logprob: -0.2,
                            rank: 1,
                        }],
                    }],
                }),
                token_ids: vec![],
                output_token_count: 1,
                finish_reason: FinishReason::stop_eos(),
            }
        );
    }
}
