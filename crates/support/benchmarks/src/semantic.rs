use uniserve_serving::{
    CachePolicy, ExecutionPlan, GenerationPolicy, PlanInspection, ServeEvent, ServeRequest,
};

pub fn text_fixture(request_id: &str, prompt: &str, max_tokens: u32) -> ServeRequest {
    let mut request = ServeRequest::text(request_id, prompt);
    request.generation = GenerationPolicy {
        max_tokens: Some(max_tokens),
        temperature: Some(0.0),
        ..GenerationPolicy::default()
    };
    request.cache = CachePolicy {
        replayable: true,
        ..CachePolicy::default()
    };
    request
}

pub fn plan_snapshot(plan: &ExecutionPlan) -> &PlanInspection {
    plan.inspect()
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
    events
        .iter()
        .any(|event| matches!(event, ServeEvent::Finished { .. }))
}

#[cfg(test)]
mod tests {
    use uniserve_serving::FinishStatus;

    use super::*;

    #[test]
    fn semantic_fixture_uses_runtime_request_shape() {
        let request = text_fixture("bench-1", "hello", 8);
        assert_eq!(request.generation.max_tokens, Some(8));
        assert!(request.cache.replayable);
    }

    #[test]
    fn event_helpers_collect_visible_text_and_terminal() {
        let events = vec![
            ServeEvent::TextDelta {
                candidate_id: 0,
                text: "he".to_string(),
                token_ids: vec![1],
                logprobs: None,
            },
            ServeEvent::TextDelta {
                candidate_id: 0,
                text: "llo".to_string(),
                token_ids: vec![2],
                logprobs: None,
            },
            ServeEvent::Finished {
                candidate_id: 0,
                reason: FinishStatus::Stop { stop_reason: None },
                finish_detail: None,
                kv_transfer_params: None,
            },
        ];
        assert_eq!(visible_text(&events), "hello");
        assert!(has_terminal_success(&events));
    }
}
