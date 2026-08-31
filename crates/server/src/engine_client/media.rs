use std::collections::BTreeMap;
use uniserve_core::MediaPlan;

/// Frontend-owned metadata and the compact fixed-profile media input.
#[derive(Debug, Clone)]
pub struct MediaSubmission {
    pub external_request_id: String,
    pub prompt_token_ids: Vec<u32>,
    pub seed: u64,
    pub priority: i32,
    pub output_path: String,
    pub plan: MediaPlan,
    pub arrival_time: Option<f64>,
    pub data_parallel_rank: Option<u32>,
    pub trace_headers: Option<BTreeMap<String, String>>,
}

impl MediaSubmission {
    pub fn new(
        external_request_id: impl Into<String>,
        prompt_token_ids: Vec<u32>,
        seed: u64,
        output_path: impl Into<String>,
        plan: MediaPlan,
    ) -> Self {
        Self {
            external_request_id: external_request_id.into(),
            prompt_token_ids,
            seed,
            priority: 0,
            output_path: output_path.into(),
            plan,
            arrival_time: None,
            data_parallel_rank: None,
            trace_headers: None,
        }
    }
}
