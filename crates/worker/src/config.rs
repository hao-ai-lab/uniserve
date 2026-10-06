//! Immutable rank execution settings, independent of the numerical backend.

use uniserve_worker_ipc::{CanvasSampling, LaneConfig};

use crate::{Error, Result};

/// Settings resolved before model loading and physical capacity fitting.
/// Numerical backend options remain with their backend, outside this value.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerConfig {
    /// Execution device; numerical binding resolves an unspecified CUDA index.
    pub device: String,
    pub rank: usize,
    pub world_size: usize,
    pub role: String,
    pub expert_exchange: String,
    pub expert_microbatches: usize,

    /// Tokens per KV page in the group with the widest token rows; unset until binding.
    pub block_size: Option<usize>,
    pub kv_token_capacity: Option<usize>,
    pub attention_backend: Option<String>,
    pub max_batch_calls: usize,
    pub max_batch_tokens: usize,
    pub max_sequence_tokens: usize,

    /// Maximum admitted video duration, in seconds.
    pub max_video_seconds: f64,
    /// Maximum condition rows reserved for a video request; zero omits them.
    pub max_condition_rows: usize,
    /// Caller-selected executable for decoding reference media.
    pub ffmpeg: String,
    pub min_video_seconds: Option<f64>,
    /// Prompt-token capacities admitted by standalone denoiser layouts.
    pub video_text_capacities: Vec<usize>,
    pub deployment_components: Vec<String>,

    /// Sampling shared by all generating canvases of this deployment.
    pub canvas_sampling: Option<CanvasSampling>,
    pub max_request_pool_size: usize,
    pub encoder_cache_entries: usize,
    pub generation_device: Option<String>,
    pub min_request_pool_size: usize,
    /// Device-storage grant after loaded weights and fixed overhead, in bytes.
    pub pool_storage_bytes: Option<usize>,
    pub model_dtype: String,
    pub kv_cache_dtype: Option<String>,
    /// Fraction of total device storage available to this process.
    pub kv_storage_fraction: f64,
    pub lanes: Vec<LaneConfig>,

    pub graph_policy: String,
    pub decode_graph_batch_sizes: Vec<usize>,
    pub prefill_cuda_graph: bool,
    pub prefill_outputs: bool,
    pub prefill_graph_token_sizes: Vec<usize>,
    pub flow_cuda_graph: bool,
    pub flow_graph_batch_sizes: Vec<usize>,
    /// Image capture sizes in (height, width) pixels.
    pub flow_graph_shapes: Vec<(usize, usize)>,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            device: "cpu".into(),
            rank: 0,
            world_size: 1,
            role: "model".into(),
            expert_exchange: "alltoall".into(),
            expert_microbatches: 1,
            block_size: None,
            kv_token_capacity: None,
            attention_backend: None,
            max_batch_calls: 1024,
            max_batch_tokens: 8192,
            max_sequence_tokens: 16384,
            max_video_seconds: 15.0,
            max_condition_rows: 0,
            ffmpeg: "ffmpeg".into(),
            min_video_seconds: None,
            video_text_capacities: vec![],
            deployment_components: vec![],
            canvas_sampling: None,
            max_request_pool_size: 128,
            encoder_cache_entries: 256,
            generation_device: None,
            min_request_pool_size: 1,
            pool_storage_bytes: None,
            model_dtype: "bfloat16".into(),
            kv_cache_dtype: None,
            kv_storage_fraction: 0.7,
            lanes: vec![],
            graph_policy: "auto".into(),
            decode_graph_batch_sizes: default_decode_sizes(),
            prefill_cuda_graph: true,
            prefill_outputs: true,
            prefill_graph_token_sizes: default_prefill_sizes(),
            flow_cuda_graph: true,
            flow_graph_batch_sizes: vec![1, 2, 3, 4],
            flow_graph_shapes: vec![(1152, 2048), (2048, 1152)],
        }
    }
}

/// Default decode buckets: dense small batches followed by strides of eight.
pub fn default_decode_sizes() -> Vec<usize> {
    (1..=32).chain((40..=128).step_by(8)).collect()
}

/// Default prefill buckets keep 64-token spacing through 4096, then 512.
pub fn default_prefill_sizes() -> Vec<usize> {
    (4..=32)
        .step_by(4)
        .chain((48..=256).step_by(16))
        .chain((288..=512).step_by(32))
        .chain((576..=4096).step_by(64))
        .chain((4608..=16384).step_by(512))
        .collect()
}

/// A prefill bucket reserves at least one inert padding sequence.
pub const PREFILL_ROW_BUCKETS: &[usize] = &[8, 16, 32];

/// Additional KV pages that the widest default graph padding may occupy.
pub fn graph_padding_block_count(block_size: usize) -> usize {
    let decode = default_decode_sizes().into_iter().max().unwrap_or(1) - 1;
    let prefill = default_prefill_sizes()
        .into_iter()
        .scan(0, |previous, size| {
            let gap = size - *previous;
            *previous = size;
            Some(gap)
        })
        .max()
        .unwrap_or(0);
    decode.max(prefill).div_ceil(block_size.max(1))
}

impl WorkerConfig {
    /// Distinct execution devices whose resident storage belongs to this rank.
    pub fn devices(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.device.as_str()).chain(
            self.generation_device
                .as_deref()
                .filter(|device| *device != self.device),
        )
    }

    /// Check process placement and physical resource bounds before allocation.
    pub fn validate(&self) -> Result<()> {
        let invalid = |message: &str| Error::Invalid(message.into());
        if !matches!(self.graph_policy.as_str(), "off" | "auto" | "full") {
            return Err(invalid("graph policy must be off, auto, or full"));
        }

        if !matches!(self.role.as_str(), "model" | "experts") {
            return Err(invalid("worker role must be model or experts"));
        }

        if !(1..=4).contains(&self.expert_microbatches) {
            return Err(invalid("expert microbatches must be from one to four"));
        }

        if self.device.is_empty() {
            return Err(invalid("worker device must be named"));
        }

        if self.world_size == 0 || self.rank >= self.world_size {
            return Err(invalid("worker configuration process rank is invalid"));
        }

        if self.min_request_pool_size == 0
            || self.min_request_pool_size > self.max_request_pool_size
        {
            return Err(invalid("worker request slot bounds are invalid"));
        }

        if self.block_size == Some(0)
            || [
                self.max_batch_calls,
                self.max_batch_tokens,
                self.max_sequence_tokens,
                self.max_request_pool_size,
                self.encoder_cache_entries,
            ]
            .contains(&0)
        {
            return Err(invalid("worker configuration capacities must be positive"));
        }

        if !self.max_video_seconds.is_finite() || self.max_video_seconds <= 0.0 {
            return Err(invalid(
                "video duration capacity must be finite and positive",
            ));
        }

        if self.ffmpeg.is_empty() {
            return Err(invalid("the media reader needs an ffmpeg path"));
        }

        if self.min_video_seconds.is_some_and(|seconds| {
            !seconds.is_finite() || seconds <= 0.0 || seconds > self.max_video_seconds
        }) {
            return Err(invalid(
                "shortest video duration must lie within the capacity",
            ));
        }

        if self.video_text_capacities.contains(&0) {
            return Err(invalid("video text capacities must be positive"));
        }

        let mut lane_ids = std::collections::HashSet::new();
        let mut operations = std::collections::HashSet::new();
        for lane in &self.lanes {
            if !lane_ids.insert(&lane.lane_id) {
                return Err(invalid("lane ids must be unique"));
            }
            if lane.call_kinds.iter().any(|kind| !operations.insert(*kind)) {
                return Err(invalid("call kinds must have one execution lane binding"));
            }
        }

        if !(self.kv_storage_fraction > 0.0 && self.kv_storage_fraction <= 1.0) {
            return Err(invalid(
                "worker configuration KV storage fraction must be in (0, 1]",
            ));
        }

        if !matches!(
            self.model_dtype.as_str(),
            "float16" | "bfloat16" | "float32"
        ) {
            return Err(invalid("worker model dtype is unsupported"));
        }

        if self.kv_cache_dtype.as_deref().is_some_and(|dtype| {
            !matches!(dtype, "float16" | "bfloat16" | "float32" | "float8_e4m3fn")
        }) {
            return Err(invalid("worker KV dtype is unsupported"));
        }

        Ok(())
    }
}
