use std::collections::BTreeMap;

/// Frontend-owned metadata and the compact fixed-profile media input.
#[derive(Debug, Clone)]
pub struct MediaSubmission {
    pub external_request_id: String,
    pub prompt: String,
    pub seed: u64,
    pub priority: i32,
    pub output_path: String,
    pub arrival_time: Option<f64>,
    pub data_parallel_rank: Option<u32>,
    pub trace_headers: Option<BTreeMap<String, String>>,
}

impl MediaSubmission {
    pub fn new(
        external_request_id: impl Into<String>,
        prompt: impl Into<String>,
        seed: u64,
        output_path: impl Into<String>,
    ) -> Self {
        Self {
            external_request_id: external_request_id.into(),
            prompt: prompt.into(),
            seed,
            priority: 0,
            output_path: output_path.into(),
            arrival_time: None,
            data_parallel_rank: None,
            trace_headers: None,
        }
    }
}
