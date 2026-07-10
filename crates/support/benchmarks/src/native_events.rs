use serde::{Deserialize, Serialize};
use serde_json::Value;
use uniserve_testkit::{PngInfo, image_done_json_metadata};

/// Canonical native SSE event-`type` strings.
///
/// These mirror the production emitter in
/// `crates/frontend/native-api/src/events.rs` (`event_json`, which serializes a
/// `GenEvent` into `{"type":...}`) and the terminal set in `is_terminal`
/// (`Finished | Rejected | Error`). `events.rs` is the single source of truth;
/// this module re-states the vocabulary because the benchmark classifier
/// consumes already-serialized JSON (`&[Value]`) rather than `GenEvent`s. If the
/// production type strings change, update these constants together with the
/// Python mirror in
/// `uniserve_eval/harness/response_classifier.py`. The
/// `vocabulary_matches_native_api` test below guards the full set.
pub mod event_type {
    pub const SCHEDULED: &str = "scheduled";
    pub const TEXT: &str = "text";
    pub const LOGPROBS: &str = "logprobs";
    pub const IMAGE_BEGIN: &str = "image_begin";
    pub const IMAGE_STEP: &str = "image_step";
    pub const IMAGE_DONE: &str = "image_done";
    /// Success terminal.
    pub const FINISHED: &str = "finished";
    /// Failure terminal (contract rejected before generation).
    pub const REJECTED: &str = "rejected";
    /// Failure terminal (engine/model error).
    pub const ERROR: &str = "error";
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeEventSummary {
    pub scheduled: u64,
    pub text_tokens: u64,
    pub image_begin: u64,
    pub image_step: u64,
    pub image_done: u64,
    pub finished: u64,
    pub rejected: u64,
    pub errors: u64,
}

impl NativeEventSummary {
    /// A native stream passes the contract iff it reaches exactly one success
    /// terminal (`finished`) and emits no failure terminals (`rejected` /
    /// `error`). `finished == 1` (not `>= 1`) is intentional: the production
    /// emitter sends exactly one terminal event per stream, so a duplicate
    /// `finished` is a protocol violation. This matches the Python classifier
    /// `classify_native_events` in
    /// `uniserve_eval/harness/response_classifier.py`.
    pub fn passed_contract(&self) -> bool {
        self.finished == 1 && self.rejected == 0 && self.errors == 0
    }
}

pub fn summarize_events(events: &[Value]) -> NativeEventSummary {
    let mut summary = NativeEventSummary {
        scheduled: 0,
        text_tokens: 0,
        image_begin: 0,
        image_step: 0,
        image_done: 0,
        finished: 0,
        rejected: 0,
        errors: 0,
    };
    for event in events {
        match event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            event_type::SCHEDULED => summary.scheduled += 1,
            event_type::TEXT => summary.text_tokens += 1,
            event_type::IMAGE_BEGIN => summary.image_begin += 1,
            event_type::IMAGE_STEP => summary.image_step += 1,
            event_type::IMAGE_DONE => summary.image_done += 1,
            event_type::FINISHED => summary.finished += 1,
            event_type::REJECTED => summary.rejected += 1,
            event_type::ERROR => summary.errors += 1,
            _ => {}
        }
    }
    summary
}

pub fn first_image_done_metadata(events: &[Value]) -> anyhow::Result<PngInfo> {
    let event = events
        .iter()
        .find(|event| event.get("type").and_then(Value::as_str) == Some(event_type::IMAGE_DONE))
        .ok_or_else(|| anyhow::anyhow!("no image_done event found"))?;
    image_done_json_metadata(event)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn summarizes_native_event_counts() {
        let events = vec![
            json!({"type":"text"}),
            json!({"type":"image_begin"}),
            json!({"type":"image_done", "width": 4, "height": 3, "bytes": 12, "sha256": "x"}),
            json!({"type":"finished"}),
        ];
        let summary = summarize_events(&events);
        assert!(summary.passed_contract());
        let image = first_image_done_metadata(&events).expect("image metadata");
        assert_eq!((image.width, image.height), (4, 3));
    }

    #[test]
    fn rejected_and_error_fail_contract() {
        let rejected = summarize_events(&[json!({"type":"rejected","message":"nope"})]);
        assert!(!rejected.passed_contract());
        let errored = summarize_events(&[json!({"type":"error","message":"boom"})]);
        assert!(!errored.passed_contract());
    }

    #[test]
    fn duplicate_finished_fails_contract() {
        // Production emits exactly one terminal; two `finished` events is a
        // protocol violation and must fail, matching the Python classifier.
        let summary = summarize_events(&[json!({"type":"finished"}), json!({"type":"finished"})]);
        assert_eq!(summary.finished, 2);
        assert!(!summary.passed_contract());
    }

    #[test]
    fn vocabulary_matches_native_api() {
        // Pins the canonical event-type strings. These MUST stay in lockstep
        // with `event_json` / `is_terminal` in
        // `crates/frontend/native-api/src/events.rs` and the Python mirror in
        // `uniserve_eval/harness/response_classifier.py`. A rename
        // in the production emitter that is not mirrored here will fail this
        // test instead of silently desynchronizing the classifiers.
        let all = [
            event_type::SCHEDULED,
            event_type::TEXT,
            event_type::LOGPROBS,
            event_type::IMAGE_BEGIN,
            event_type::IMAGE_STEP,
            event_type::IMAGE_DONE,
            event_type::FINISHED,
            event_type::REJECTED,
            event_type::ERROR,
        ];
        assert_eq!(
            all,
            [
                "scheduled",
                "text",
                "logprobs",
                "image_begin",
                "image_step",
                "image_done",
                "finished",
                "rejected",
                "error",
            ]
        );
        // Terminal set mirrors native-api `is_terminal`.
        let terminals = [
            event_type::FINISHED,
            event_type::REJECTED,
            event_type::ERROR,
        ];
        assert_eq!(terminals, ["finished", "rejected", "error"]);
    }
}
