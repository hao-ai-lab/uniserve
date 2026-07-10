use serde_json::{Value, json};
use uniserve_serving::{FinishStatus, ServeEvent};

pub fn is_terminal(event: &ServeEvent) -> bool {
    matches!(
        event,
        ServeEvent::Finished { .. }
            | ServeEvent::Rejected { .. }
            | ServeEvent::Cancelled { .. }
            | ServeEvent::Aborted { .. }
            | ServeEvent::Failed { .. }
    )
}

pub fn event_json(event: &ServeEvent) -> Value {
    match event {
        ServeEvent::Accepted {
            request_id,
            profile_id,
            dialect_id,
            compile_duration_us,
            prompt_token_count,
            prompt_token_ids,
            prompt_logprobs,
        } => json!({
            "type":"accepted",
            "request_id":request_id,
            "profile_id":profile_id,
            "dialect_id":dialect_id,
            "compile_duration_us":compile_duration_us,
            "prompt_tokens":prompt_token_count,
            "prompt_token_ids":prompt_token_ids,
            "prompt_logprobs":prompt_logprobs
        }),
        ServeEvent::Scheduled {
            request_id,
            queued_at,
            scheduled_at,
            cache,
            resources,
        } => json!({
            "type":"scheduled",
            "request_id":request_id,
            "queued_at":queued_at,
            "scheduled_at":scheduled_at,
            "cache":cache,
            "resources":resources
        }),
        ServeEvent::TextDelta {
            candidate_id,
            text,
            token_ids,
            logprobs,
        } => json!({
            "type":"text",
            "candidate_id":candidate_id,
            "text":text,
            "token_ids":token_ids,
            "logprobs":logprobs
        }),
        ServeEvent::InternalTextDelta { candidate_id, text } => {
            json!({"type":"internal_text","candidate_id":candidate_id,"text":text})
        }
        ServeEvent::ReasoningDelta { candidate_id, text } => {
            json!({"type":"reasoning","candidate_id":candidate_id,"text":text})
        }
        ServeEvent::OutputBlockStart {
            candidate_id,
            index,
            kind,
        } => json!({
            "type":"output_block_start",
            "candidate_id":candidate_id,
            "index":index,
            "kind":kind
        }),
        ServeEvent::OutputBlockEnd {
            candidate_id,
            index,
            block,
        } => json!({
            "type":"output_block_end",
            "candidate_id":candidate_id,
            "index":index,
            "block":block
        }),
        ServeEvent::ToolCallStart {
            candidate_id,
            index,
            id,
            name,
        } => json!({
            "type":"tool_call_start",
            "candidate_id":candidate_id,
            "index":index,
            "id":id,
            "name":name
        }),
        ServeEvent::ToolCallArgumentsDelta {
            candidate_id,
            index,
            delta,
        } => json!({
            "type":"tool_call_arguments_delta",
            "candidate_id":candidate_id,
            "index":index,
            "delta":delta
        }),
        ServeEvent::ToolCallEnd {
            candidate_id,
            index,
            id,
            name,
            arguments,
        } => json!({
            "type":"tool_call_end",
            "candidate_id":candidate_id,
            "index":index,
            "id":id,
            "name":name,
            "arguments":arguments
        }),
        ServeEvent::ImageBegin {
            candidate_id,
            image_id,
            height,
            width,
            steps,
            elapsed_us,
        } => {
            json!({"type":"image_begin","candidate_id":candidate_id,"image_id":image_id,"height":height,"width":width,"steps":steps,"elapsed_us":elapsed_us})
        }
        ServeEvent::ImageStep {
            candidate_id,
            image_id,
            step,
            elapsed_us,
        } => {
            json!({"type":"image_step","candidate_id":candidate_id,"image_id":image_id,"step":step,"elapsed_us":elapsed_us})
        }
        ServeEvent::ImageCommit {
            candidate_id,
            image_id,
            elapsed_us,
        } => {
            json!({"type":"image_commit","candidate_id":candidate_id,"image_id":image_id,"elapsed_us":elapsed_us})
        }
        ServeEvent::ImageDone {
            candidate_id,
            image_id,
            height,
            width,
            bytes,
            sha256,
            pixels_png_b64,
            elapsed_us,
        } => json!({
            "type":"image_done",
            "candidate_id":candidate_id,
            "image_id":image_id,
            "height":height,
            "width":width,
            "bytes":bytes,
            "sha256":sha256,
            "pixels_png_b64":pixels_png_b64,
            "elapsed_us":elapsed_us
        }),
        ServeEvent::Usage {
            prompt_tokens,
            visible_output_tokens,
            internal_tokens,
            image_count,
            image_steps,
            cache,
            resources,
            timings,
        } => json!({
            "type":"usage",
            "prompt_tokens":prompt_tokens,
            "visible_output_tokens":visible_output_tokens,
            "internal_tokens":internal_tokens,
            "images":image_count,
            "image_steps":image_steps,
            "cache":cache,
            "resources":resources,
            "timings":timings
        }),
        ServeEvent::Finished {
            candidate_id,
            reason,
            finish_detail,
        } => json!({
            "type":"finished",
            "candidate_id":candidate_id,
            "reason":finish_reason(reason),
            "stop_cause":match reason { FinishStatus::Stop { cause } => cause, _ => &None },
            "detail":finish_detail
        }),
        ServeEvent::Rejected { message, .. } => json!({"type":"rejected","message":message}),
        ServeEvent::Cancelled { request_id } => json!({"type":"cancelled","request_id":request_id}),
        ServeEvent::Aborted { request_id } => json!({"type":"aborted","request_id":request_id}),
        ServeEvent::Failed {
            request_id,
            message,
        } => json!({"type":"error","request_id":request_id,"message":message}),
    }
}

fn finish_reason(reason: &FinishStatus) -> &'static str {
    match reason {
        FinishStatus::Stop { .. } => "stop",
        FinishStatus::Length => "length",
        FinishStatus::Abort => "abort",
        FinishStatus::Error => "error",
        FinishStatus::Repetition => "repetition",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniserve_serving::CandidateId;

    #[test]
    fn image_commit_json_preserves_runtime_timing() {
        let value = event_json(&ServeEvent::ImageCommit {
            candidate_id: CandidateId::PRIMARY,
            image_id: "3".to_string(),
            elapsed_us: 42,
        });

        assert_eq!(value["type"], "image_commit");
        assert_eq!(value["image_id"], "3");
        assert_eq!(value["elapsed_us"], 42);
    }

    #[test]
    fn tool_call_json_preserves_structured_identity_and_arguments() {
        let value = event_json(&ServeEvent::ToolCallEnd {
            candidate_id: CandidateId::PRIMARY,
            index: 2,
            id: "call-7".to_string(),
            name: "lookup".to_string(),
            arguments: r#"{"id":7}"#.to_string(),
        });

        assert_eq!(value["type"], "tool_call_end");
        assert_eq!(value["candidate_id"], 0);
        assert_eq!(value["index"], 2);
        assert_eq!(value["id"], "call-7");
        assert_eq!(value["name"], "lookup");
        assert_eq!(value["arguments"], r#"{"id":7}"#);
    }
}
