use std::sync::Arc;

use uuid::Uuid;

use crate::serving::chat::AssistantBlockKind;
use crate::serving::chat::output::FinishReason;
use crate::serving::text::output::{DecodedLogprobs, DecodedPromptLogprobs};
use uniserve_core::PublicCommit;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ReasoningEvent {
    Start {
        prompt_token_ids: Arc<[u32]>,
        prompt_logprobs: Option<DecodedPromptLogprobs>,
        queued_at: Option<f64>,
        scheduled_at: Option<f64>,
    },
    TextDelta {
        kind: AssistantBlockKind,
        delta: String,
    },
    SampleDelta {
        logprobs: Option<DecodedLogprobs>,
        token_ids: Vec<u32>,
    },
    PublicCommit(PublicCommit),
    Done {
        prompt_token_count: usize,
        output_token_count: usize,
        internal_token_count: usize,
        finish_reason: FinishReason,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AssistantEvent {
    Start {
        prompt_token_ids: Arc<[u32]>,
        prompt_logprobs: Option<DecodedPromptLogprobs>,
        queued_at: Option<f64>,
        scheduled_at: Option<f64>,
    },
    TextDelta {
        kind: AssistantBlockKind,
        delta: String,
    },
    SampleDelta {
        logprobs: Option<DecodedLogprobs>,
        token_ids: Vec<u32>,
    },
    PublicCommit(PublicCommit),
    ToolCallStart {
        id: String,
        name: String,
    },
    ToolCallArgumentsDelta {
        delta: String,
    },
    Done {
        prompt_token_count: usize,
        output_token_count: usize,
        internal_token_count: usize,
        finish_reason: FinishReason,
    },
}

impl From<ReasoningEvent> for AssistantEvent {
    fn from(event: ReasoningEvent) -> Self {
        match event {
            ReasoningEvent::Start {
                prompt_token_ids,
                prompt_logprobs,
                queued_at,
                scheduled_at,
            } => Self::Start {
                prompt_token_ids,
                prompt_logprobs,
                queued_at,
                scheduled_at,
            },
            ReasoningEvent::TextDelta { kind, delta } => Self::TextDelta { kind, delta },
            ReasoningEvent::SampleDelta {
                logprobs,
                token_ids,
            } => Self::SampleDelta {
                logprobs,
                token_ids,
            },
            ReasoningEvent::PublicCommit(commit) => Self::PublicCommit(commit),
            ReasoningEvent::Done {
                prompt_token_count,
                output_token_count,
                internal_token_count,
                finish_reason,
            } => Self::Done {
                prompt_token_count,
                output_token_count,
                internal_token_count,
                finish_reason,
            },
        }
    }
}

pub(crate) fn generate_tool_call_id() -> String {
    format!("call_{}", &Uuid::new_v4().simple().to_string()[..24])
}
