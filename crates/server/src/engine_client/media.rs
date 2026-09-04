//! Frontend metadata attached to terminal media submissions.

use std::collections::BTreeMap;
use uniserve_core::MediaGeometry;

/// Frontend-owned metadata and the compact fixed-profile media input.
#[derive(Debug, Clone)]
pub struct MediaSubmission {
    /// External request identifier used by serving lifecycle tracking.
    pub external_request_id: String,
    /// Tokenized media prompt.
    pub prompt_token_ids: Vec<u32>,
    /// Request-level random seed.
    pub seed: u64,
    /// Scheduler priority, with larger values taking precedence when configured.
    pub priority: i32,
    /// Fixed frame and decode-work geometry.
    pub geometry: MediaGeometry,
    /// Caller-supplied arrival timestamp, when available.
    pub arrival_time: Option<f64>,
    /// Explicit data-parallel rank assignment, when requested.
    pub data_parallel_rank: Option<u32>,
    /// Distributed trace headers propagated to the engine.
    pub trace_headers: Option<BTreeMap<String, String>>,
}

impl MediaSubmission {
    /// Constructs a media submission with explicit prompt and geometry.
    pub fn new(
        external_request_id: impl Into<String>,
        prompt_token_ids: Vec<u32>,
        seed: u64,
        geometry: MediaGeometry,
    ) -> Self {
        Self {
            external_request_id: external_request_id.into(),
            prompt_token_ids,
            seed,
            priority: 0,
            geometry,
            arrival_time: None,
            data_parallel_rank: None,
            trace_headers: None,
        }
    }
}
