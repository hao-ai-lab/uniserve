use uniserve_server::serving::{FinishStatus, GenerateReqInput, SamplingConfig, ServeEvent};

pub fn text_fixture(request_id: &str, prompt: &str, max_tokens: u32) -> GenerateReqInput {
    let mut request = GenerateReqInput::text(request_id, prompt);
    request.sampling = SamplingConfig {
        max_tokens: Some(max_tokens),
        temperature: Some(0.0),
        ..SamplingConfig::default()
    };
    request
}

pub fn visible_text(events: &[ServeEvent]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            ServeEvent::TextDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

pub fn has_terminal_success(events: &[ServeEvent]) -> bool {
    let mut terminals = events.iter().filter(|event| {
        matches!(
            event,
            ServeEvent::Finished { .. }
                | ServeEvent::Rejected { .. }
                | ServeEvent::Cancelled { .. }
                | ServeEvent::Aborted { .. }
                | ServeEvent::Failed { .. }
        )
    });
    matches!(
        (terminals.next(), terminals.next()),
        (
            Some(ServeEvent::Finished {
                reason: FinishStatus::Stop { .. } | FinishStatus::Length | FinishStatus::Repetition,
                ..
            }),
            None
        )
    )
}

#[cfg(test)]
mod tests {
    use uniserve_server::serving::{CandidateId, FinishStatus};

    use super::*;

    #[test]
    fn semantic_fixture_uses_runtime_request_shape() {
        let request = text_fixture("bench-1", "hello", 8);
        assert_eq!(request.sampling.max_tokens, Some(8));
        assert_eq!(request.sampling.temperature, Some(0.0));
        assert_eq!(request.request_id.as_ref(), "bench-1");
    }

    #[test]
    fn event_helpers_collect_visible_text_and_terminal() {
        let events = vec![
            ServeEvent::TextDelta {
                candidate_id: CandidateId::PRIMARY,
                text: "he".to_string(),
                token_ids: vec![1],
                logprobs: None,
            },
            ServeEvent::TextDelta {
                candidate_id: CandidateId::PRIMARY,
                text: "llo".to_string(),
                token_ids: vec![2],
                logprobs: None,
            },
            ServeEvent::Finished {
                candidate_id: CandidateId::PRIMARY,
                reason: FinishStatus::Stop { cause: None },
                finish_detail: None,
            },
        ];
        assert_eq!(visible_text(&events), "hello");
        assert!(has_terminal_success(&events));
    }
}
