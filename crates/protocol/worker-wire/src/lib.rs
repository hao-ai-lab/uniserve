//! Worker op-batch wire types,
//! mirrored on the Python side through the `uniserve_worker.ipc` PyO3 bridge.
//! Serialized on the host-worker boundary as FlatBuffers tables with closed
//! enums. Wire fields are descriptors and scalars only — never KV pages, hidden
//! states, or latents.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use uniserve_core::{
    BlockId, CfgParams, ImageParams, KvCacheGroupSpec, Modality, RankInfo, RequestId,
    SamplingParams,
};

pub mod flat;
pub mod resources;
#[allow(warnings)]
pub mod schema {
    include!(concat!(env!("OUT_DIR"), "/flatbuffers/mod.rs"));
}
pub use resources::{
    LeasePolicy, ResourceClass, ResourceEvent, ResourceEventKind, ResourceHandle, ResourceLease,
    ResourcePressure,
};

/// Forward-op taxonomy.

/// There is no standalone `VaeDecode` op: image decode runs via `CommitGen`
/// (`commit_gen` -> `decode_image`). The wire enum, FlatBuffers schema, and
/// Python `contracts.OP_KINDS` / `mode_for_op` vocabularies stay aligned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpKind {
    PrefillUnd,
    DecodeUnd,
    TargetVerifyUnd,
    DenoiseGen,
    CommitGen,
    CommitWriteback,
    VaeEncode,
    VitEncode,
    /// Sampler-stage op: turn a `Logits` handle into a sampled token. Produced by
    /// the StageRouter when a Sampler pool is split off; never emitted by the base
    /// scheduler, which samples inside the decode worker by default.
    Sample,
    /// PostProcess-stage op: encode one finished image/video frame. Produced by
    /// the StageRouter when a PostProcess pool is split off; never
    /// emitted by the base scheduler.
    EncodeFrame,
}

/// Source for text input token ids on a [`ForwardOp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenSource {
    Wire,
    LastSampled,
}

impl Default for TokenSource {
    fn default() -> Self {
        Self::Wire
    }
}

/// Static per-request state that crosses the wire **once**, when the scheduler
/// first dispatches work for a request (the `NewRequestData` half of the
/// stateful-diff contract). The worker keeps a per-request control record built
/// from this; subsequent [`ForwardOp`]s carry only deltas (new block ids, new
/// tokens, per-step masks). All fields are descriptors — never tensors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewRequestData {
    pub req_id: RequestId,
    /// Static sampling parameters (per-step masks stay on the op).
    pub sampling: Option<SamplingParams>,
    /// Image diffusion parameters (dimensions may be refined worker-side from
    /// its own VAE-resize of an input image).
    pub image: Option<ImageParams>,
    /// CFG text-unconditional / image precontext prompt.
    pub neg_token_ids: Option<Vec<u32>>,
    /// LoRA adapter applied to this request's ops (worker-resident).
    pub lora_id: Option<u32>,
    /// Initial logical KV block allocation.
    pub block_ids: Vec<BlockId>,
    /// KV-cache group the block ids live in.
    pub group_id: u32,
}

impl NewRequestData {
    pub fn new(req_id: RequestId) -> Self {
        Self {
            req_id,
            sampling: None,
            image: None,
            neg_token_ids: None,
            lora_id: None,
            block_ids: Vec::new(),
            group_id: 0,
        }
    }
}

/// A single unit of GPU work under the stateful-diff
/// contract: per-request statics live in the worker record seeded by
/// [`NewRequestData`]; the op carries only this step's deltas and dynamic view
/// descriptors — never tensors, and not the full block list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardOp {
    pub req_id: RequestId,
    pub kind: OpKind,
    pub modality: Modality,
    /// Logical KV blocks allocated since the last op for this request (the
    /// worker appends them to its per-request block map).
    pub new_block_ids: Vec<BlockId>,
    pub pos_range: (u32, u32),
    pub token_ids: Option<Vec<u32>>,
    /// Where text input ids are read from. `Wire` uses `token_ids`; `LastSampled`
    /// tells the worker to feed the last sampled device token for this
    /// request, while `token_ids` still describes the logical one-token shape.
    #[serde(default)]
    pub token_source: TokenSource,
    pub timestep_idx: Option<u16>,
    pub cond_pos: Option<u32>,
    /// Model-neutral classifier-free-guidance descriptor. Geometry such as
    /// branch KV spans is computed worker-side from this plus request state.
    pub cfg: Option<CfgParams>,
    pub image_in: Option<u64>, // StagedImageId (Vit/Vae encode handle)
    /// Optional per-image text conditioning prompt for a generated image. This
    /// is a small descriptor string; the worker tokenizes it into a temporary
    /// branch KV cache and does not ship hidden states over the wire.
    pub image_prompt: Option<String>,
    // ---- context-image encode input. All descriptors/small-input. ----
    /// Input-image bytes (base64 PNG/JPEG) for a Vit/Vae encode op. Mirror of the
    /// output `image_png_b64` "small result" — bytes in, not KV.
    pub image_b64: Option<String>,
    // ---- which KV-cache group these block ids live in (0 == the single
    // full-attention group BAGEL uses today). ----
    pub group_id: u32,
    // ---- logits-processor descriptors. All are small id lists / scalars,
    // never tensors — the worker masks/penalizes with them in the canonical order. ----
    /// Allowed-token whitelist: if `Some`, every other logit is masked to -inf.
    pub allowed_tokens: Option<Vec<u32>>,
    /// Tokens to suppress (mask to -inf): EOS while under `min_tokens`,
    /// completed bad-words, etc. — enforced in the control plane, applied by the worker.
    pub suppress_tokens: Option<Vec<u32>>,
    /// Bounded window of recently generated token ids for penalty application
    /// (the host owns generation, so the worker stays stateless — risk 4).
    pub recent_tokens: Option<Vec<u32>>,
    // ---- multimodal encode. ----
    /// Content hash of the staged image this op encodes (encoder-cache key).
    pub mm_hash: Option<u64>,
    /// Reserved contract field: draft tokens for worker-side verify-and-accept
    /// speculative decoding. No drafter exists yet — this is the wire seam only.
    pub spec_token_ids: Option<Vec<u32>>,
    /// Number of sequential denoise timesteps to run inside this op. This is a
    /// scalar scheduler hint; the worker still returns one result with
    /// `num_steps_done` set to the cumulative timestep cursor.
    pub denoise_step_count: Option<u16>,
    /// Number of sequential greedy text decode tokens to run inside this op.
    /// This is a scalar scheduler hint; the worker still returns one result
    /// with `sampled_token_ids` carrying every committed sampled token.
    pub decode_token_count: Option<u16>,
    /// Token ids that force a worker-side decode burst to stop immediately
    /// after sampling. Small id list only; typically EOS/stop/image triggers.
    pub decode_stop_token_ids: Option<Vec<u32>>,
    /// True when a stop-token hit finishes the request and any additional
    /// speculative KV rows can be discarded with that request.
    #[serde(default)]
    pub decode_stop_terminal: bool,
    // ---- op-lifecycle id so the host correlates this op's result with the
    // submitted op. Scalar, never a tensor. ----
    pub op_id: Option<u64>,
    /// Sampler stage: handle to the logits a `sample` op consumes. Scalar id routed
    /// by the StageRouter; the data-plane locator (if any) is fetched worker-side
    /// and never crosses this control-plane field as a tensor.
    pub logits_handle: Option<u64>,
    /// Data-plane locator the consumer worker resolves to fetch this op's input
    /// tensor. base64 of the opaque `Locator` bytes — a small descriptor
    /// (opaque descriptor bytes), routed verbatim by the host.
    pub locator: Option<String>,
}

impl Default for ForwardOp {
    fn default() -> Self {
        Self {
            req_id: RequestId(0),
            kind: OpKind::PrefillUnd,
            modality: Modality::Und,
            new_block_ids: Vec::new(),
            pos_range: (0, 0),
            token_ids: None,
            token_source: TokenSource::Wire,
            timestep_idx: None,
            cond_pos: None,
            cfg: None,
            image_in: None,
            image_prompt: None,
            group_id: 0,
            allowed_tokens: None,
            suppress_tokens: None,
            recent_tokens: None,
            mm_hash: None,
            spec_token_ids: None,
            denoise_step_count: None,
            decode_token_count: None,
            decode_stop_token_ids: None,
            decode_stop_terminal: false,
            image_b64: None,
            op_id: None,
            logits_handle: None,
            locator: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardBatch {
    pub step_id: u64,
    /// Requests first dispatched in this batch: their static state crosses
    /// here, once; the worker seeds a per-request record before running `ops`.
    pub new_reqs: Vec<NewRequestData>,
    pub ops: Vec<ForwardOp>,
}

/// Small per-sequence result. `image_png_b64` is the
/// finished image bytes (a "small result") — output, not KV.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SeqResult {
    pub req_id: RequestId,
    pub sampled_token_id: Option<u32>,
    pub denoise_done: bool,
    pub num_steps_done: Option<u16>,
    pub image_png_b64: Option<String>,
    pub image_hw: Option<(u32, u32)>,
    // ---- logprobs (scalars + id-scalar pairs, never logits tensors). ----
    /// Logprob of the sampled token (the `gather_logprobs` analog).
    pub sampled_logprob: Option<f32>,
    /// Top-`n_logprobs` `(token_id, logprob)` pairs for this step.
    pub top_logprobs: Option<Vec<(u32, f32)>>,
    /// All token ids sampled by a sequential text decode burst. Small id list;
    /// `sampled_token_id` remains the last token for scalar consumers.
    pub sampled_token_ids: Option<Vec<u32>>,
    // ---- opaque worker-side handle to the encoder output produced by a
    // VitEncode/VaeEncode op. The embedding itself never returns to the host. ----
    pub encoder_handle: Option<u64>,
    /// Number of KV positions an encode/commit op appended (the host
    /// advances its KV-length mirror by this; the worker is authoritative).
    pub num_tokens: Option<u32>,
    /// Reserved contract field: how many of the op's `spec_token_ids` the worker
    /// accepted. Consumed by scheduler accounting when a drafter is wired in.
    pub num_accepted_tokens: Option<u32>,
    /// echo of the op's `op_id` for result↔op
    /// correlation in the lifecycle trace.
    pub op_id: Option<u64>,
    /// Sampler stage: handle to the logits this op produced, when sampling is
    /// peeled into a separate Sampler pool. Scalar id; routing key.
    pub logits_handle: Option<u64>,
    /// Data-plane locator for the tensor this op produced (published logits /
    /// embedding). base64 of the opaque `Locator`
    /// bytes; the host routes it to the consumer op verbatim, never parsing it.
    pub locator: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardResult {
    pub step_id: u64,
    pub per_seq: Vec<SeqResult>,
    /// worker compute time for this batch in microseconds (the worker
    /// MetricsService measurement, promoted onto the wire). A scalar — never a
    /// tensor.
    pub worker_exec_us: Option<u64>,
    /// Worker-local forward/kernel counters for this batch. These are safe
    /// scalar counters only; tensors and payload data stay worker-resident.
    #[serde(default)]
    pub forward_stats: Option<WorkerForwardStats>,
}

/// Worker-local per-forward counters carried on `ForwardResult` and in
/// `get_metrics` snapshots. Field names mirror the Python metrics service so
/// they can be bridged without bespoke translation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkerForwardStats {
    #[serde(default)]
    pub mode_counts: BTreeMap<String, u64>,
    #[serde(default)]
    pub mode_tokens: BTreeMap<String, u64>,
    #[serde(default)]
    pub mode_us: BTreeMap<String, u64>,
    #[serde(default)]
    pub component_us: BTreeMap<String, u64>,
    #[serde(default)]
    pub attention_launches: u64,
    #[serde(default)]
    pub attention_us: u64,
    #[serde(default)]
    pub attention_backend_counts: BTreeMap<String, u64>,
    #[serde(default)]
    pub cuda_graph_captures: u64,
    #[serde(default)]
    pub cuda_graph_replays: u64,
    #[serde(default)]
    pub cuda_graph_misses: u64,
    #[serde(default)]
    pub cuda_graph_fallbacks: u64,
    #[serde(default)]
    pub cuda_graph_unpadded_tokens: u64,
    #[serde(default)]
    pub cuda_graph_padded_tokens: u64,
    #[serde(default)]
    pub cuda_graph_runtime_mode_counts: BTreeMap<String, u64>,
    #[serde(default)]
    pub text_decode_token_relay_hits: u64,
    #[serde(default)]
    pub text_decode_token_relay_misses: u64,
    #[serde(default)]
    pub text_decode_position_relay_hits: u64,
    #[serde(default)]
    pub text_decode_position_relay_misses: u64,
    #[serde(default)]
    pub flashinfer_decode_plan_calls: u64,
    #[serde(default)]
    pub flashinfer_decode_plan_reuses: u64,
    #[serde(default)]
    pub flashinfer_decode_plan_rows: u64,
    #[serde(default)]
    pub flashinfer_decode_plan_indices: u64,
    #[serde(default)]
    pub flashinfer_decode_graph_plan_calls: u64,
    #[serde(default)]
    pub flashinfer_decode_graph_plan_reuses: u64,
    #[serde(default)]
    pub spec_verify_rows: u64,
    #[serde(default)]
    pub spec_verify_draft_tokens: u64,
    #[serde(default)]
    pub spec_verify_accepted_tokens: u64,
    #[serde(default)]
    pub spec_verify_rejected_tokens: u64,
    #[serde(default)]
    pub spec_verify_committed_tokens: u64,
    #[serde(default)]
    pub spec_verify_path_counts: BTreeMap<String, u64>,
}

/// Adapter (LoRA) residency mode a worker supports. Makes the otherwise-
/// ambiguous `lora_id` surface a declared contract: the scheduler plans adapter
/// routing only against what the worker promises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AdapterMode {
    /// No LoRA support (SenseNova today).
    #[default]
    None,
    /// One merged adapter applies engine-wide; only one resident at a time
    /// (Bagel today: merge-on-load, `lora_id` selects but cannot mix).
    EngineWide,
    /// Per-request adapter routing (reserved; not implemented).
    PerRequest,
    /// Per-batch multi-adapter routing (reserved; not implemented).
    MultiAdapter,
}

/// Execution / batching constraints a worker declares so the scheduler assembles
/// only batches the worker can run. All scalars — never tensors.
///
/// The scheduler owns lane formation; workers advertise scalar limits here
/// rather than a separate mixed-op capability flag.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConstraints {
    /// Max ops the worker accepts in one `ForwardBatch` (0 == host default).
    pub max_batch_ops: u32,
}
impl Default for ExecutionConstraints {
    fn default() -> Self {
        Self { max_batch_ops: 0 }
    }
}

/// Worker-reported capabilities: supported controls, adapter mode, execution
/// constraints — the scheduler plans only against declared capabilities.
/// All additions are layout metadata / scalars — never tensors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCaps {
    pub block_size: u32,
    pub num_blocks: u32,
    pub num_layers: u32,
    pub scratch_capacity_tokens: u64,
    pub supported_ops: Vec<String>,
    pub max_latent_size: u32,
    pub latent_downsample: u32,
    #[serde(default)]
    pub max_vae_grid_tokens: u32,
    #[serde(default)]
    pub max_vit_grid_tokens: u32,
    #[serde(default = "default_commit_marker_tokens")]
    pub commit_marker_tokens: u32,
    #[serde(default = "default_gen_rope_advance")]
    pub gen_rope_advance: u32,
    #[serde(default = "default_max_cfg_branches")]
    pub max_cfg_branches: u32,
    pub bytes_per_token: u64,
    // ---- KV-cache groups (hybrid layouts). Empty == one implicit full
    // group spanning [1, num_blocks) — the BAGEL default — so the field is a
    // no-op until a hybrid model reports groups. ----
    pub groups: Vec<KvCacheGroupSpec>,
    // ---- KV dtype / attention backend / quantization the worker chose;
    // the host consumes these to size its logical pool, it selects nothing. ----
    pub kv_dtype: String,
    pub attention_backend: String,
    pub quantization: Option<String>,
    pub rank: RankInfo,
    // ---- how many op-batches the host may keep in flight against this
    // worker (the batch-queue depth). Default 1 == today's synchronous behavior. ----
    pub pipeline_depth: u32,
    // ---- encoder-output cache budget (number of cached encoder handles). ----
    pub encoder_cache_budget: u32,
    // ---- scheduler/runtime consume only declared capabilities. ----
    /// Control kinds this worker actually implements (subset of
    /// copy_blocks/load_lora/unload_lora/free_encoder/reset_prefix_cache/sleep/
    /// wake_up). A control absent here is rejected before execution rather than
    /// silently no-op'd.
    pub supported_controls: Vec<String>,
    /// LoRA residency mode.
    pub adapter_mode: AdapterMode,
    /// Batching/grouping constraints for execution planning.
    pub execution_constraints: ExecutionConstraints,
    /// Resource classes this worker accounts for — the host
    /// issues leases and asserts invariants only for declared classes.
    pub resource_classes: Vec<ResourceClass>,
}
fn default_kv_dtype() -> String {
    "bf16".into()
}
fn default_attention_backend() -> String {
    "flashinfer".into()
}
fn default_pipeline_depth() -> u32 {
    1
}
fn default_commit_marker_tokens() -> u32 {
    2
}
fn default_gen_rope_advance() -> u32 {
    2
}
fn default_max_cfg_branches() -> u32 {
    3
}
impl Default for EngineCaps {
    fn default() -> Self {
        Self {
            block_size: 64,
            num_blocks: 4096,
            num_layers: 28,
            scratch_capacity_tokens: 1 << 20,
            supported_ops: vec![
                "prefill_und".into(),
                "decode_und".into(),
                "denoise_gen".into(),
                "commit_gen".into(),
            ],
            max_latent_size: 64,
            latent_downsample: 16,
            max_vae_grid_tokens: 64,
            max_vit_grid_tokens: 0,
            commit_marker_tokens: default_commit_marker_tokens(),
            gen_rope_advance: default_gen_rope_advance(),
            max_cfg_branches: default_max_cfg_branches(),
            bytes_per_token: 57344,
            groups: Vec::new(),
            kv_dtype: default_kv_dtype(),
            attention_backend: default_attention_backend(),
            quantization: None,
            rank: RankInfo::default(),
            pipeline_depth: default_pipeline_depth(),
            encoder_cache_budget: 0,
            supported_controls: Vec::new(),
            adapter_mode: AdapterMode::None,
            execution_constraints: ExecutionConstraints::default(),
            resource_classes: Vec::new(),
        }
    }
}

/// Host -> worker request discriminant. Fieldless serde-snake_case enum: each
/// variant serializes to the exact wire string the Python worker reads from
/// `WorkerRequest.kind`, so the discriminant is exhaustive and unmisspellable on
/// the Rust side while the bytes crossing the IPC boundary are unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestKind {
    GetCaps,
    Execute,
    DropRequest,
    Shutdown,
    CopyBlocks,
    LoadLora,
    UnloadLora,
    FreeEncoder,
    ResetPrefixCache,
    Sleep,
    WakeUp,
    GetMetrics,
    GetPressure,
}

impl RequestKind {
    /// Wire string carried in the `kind` field across the IPC boundary, matching
    /// the serde-snake_case serialization byte-for-byte.
    pub fn as_wire_str(self) -> &'static str {
        match self {
            Self::GetCaps => "get_caps",
            Self::Execute => "execute",
            Self::DropRequest => "drop_request",
            Self::Shutdown => "shutdown",
            Self::CopyBlocks => "copy_blocks",
            Self::LoadLora => "load_lora",
            Self::UnloadLora => "unload_lora",
            Self::FreeEncoder => "free_encoder",
            Self::ResetPrefixCache => "reset_prefix_cache",
            Self::Sleep => "sleep",
            Self::WakeUp => "wake_up",
            Self::GetMetrics => "get_metrics",
            Self::GetPressure => "get_pressure",
        }
    }

    /// Parse a wire string back into the discriminant. Used for the
    /// control-method strings on `EngineCaps.supported_controls`, which share the
    /// request-kind vocabulary.
    pub fn from_wire_str(name: &str) -> Option<Self> {
        Some(match name {
            "get_caps" => Self::GetCaps,
            "execute" => Self::Execute,
            "drop_request" => Self::DropRequest,
            "shutdown" => Self::Shutdown,
            "copy_blocks" => Self::CopyBlocks,
            "load_lora" => Self::LoadLora,
            "unload_lora" => Self::UnloadLora,
            "free_encoder" => Self::FreeEncoder,
            "reset_prefix_cache" => Self::ResetPrefixCache,
            "sleep" => Self::Sleep,
            "wake_up" => Self::WakeUp,
            "get_metrics" => Self::GetMetrics,
            "get_pressure" => Self::GetPressure,
            _ => return None,
        })
    }
}

/// IPC envelope, host -> worker. Control ops (`copy_blocks`,
/// `reset_prefix_cache`, `free_encoder`, `load_lora`/`unload_lora`) carry only
/// ids, hashes, paths and commands — never KV.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRequest {
    pub kind: RequestKind,
    /// Correlation id for control calls: the worker echoes it on the matching
    /// response so acks route by id (and fan-outs join per rank) instead of by
    /// queue position.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch: Option<ForwardBatch>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub req_id: Option<RequestId>,
    /// `(src_block_id, dst_block_id)` pairs for prefix-cache physical reuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub copies: Option<Vec<(BlockId, BlockId)>>,
    /// LoRA adapter id for load/unload control ops.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_id: Option<u32>,
    /// filesystem path / repo id for a LoRA adapter to load.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_path: Option<String>,
    /// encoder-output handles whose physical storage the worker may reclaim.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub free_handles: Option<Vec<u64>>,
}
impl WorkerRequest {
    fn bare(kind: RequestKind) -> Self {
        Self {
            kind,
            call_id: None,
            batch: None,
            req_id: None,
            copies: None,
            lora_id: None,
            lora_path: None,
            free_handles: None,
        }
    }
    pub fn get_caps() -> Self {
        Self::bare(RequestKind::GetCaps)
    }
    pub fn execute(batch: ForwardBatch) -> Self {
        Self {
            batch: Some(batch),
            ..Self::bare(RequestKind::Execute)
        }
    }
    pub fn drop_request(req_id: RequestId) -> Self {
        Self {
            req_id: Some(req_id),
            ..Self::bare(RequestKind::DropRequest)
        }
    }
    pub fn shutdown() -> Self {
        Self::bare(RequestKind::Shutdown)
    }
    pub fn copy_blocks(copies: Vec<(BlockId, BlockId)>) -> Self {
        Self {
            copies: Some(copies),
            ..Self::bare(RequestKind::CopyBlocks)
        }
    }
    pub fn load_lora(lora_id: u32, lora_path: String) -> Self {
        Self {
            lora_id: Some(lora_id),
            lora_path: Some(lora_path),
            ..Self::bare(RequestKind::LoadLora)
        }
    }
    pub fn unload_lora(lora_id: u32) -> Self {
        Self {
            lora_id: Some(lora_id),
            ..Self::bare(RequestKind::UnloadLora)
        }
    }
    pub fn free_encoder(handles: Vec<u64>) -> Self {
        Self {
            free_handles: Some(handles),
            ..Self::bare(RequestKind::FreeEncoder)
        }
    }
    pub fn reset_prefix_cache() -> Self {
        Self::bare(RequestKind::ResetPrefixCache)
    }
    pub fn sleep() -> Self {
        Self::bare(RequestKind::Sleep)
    }
    pub fn wake_up() -> Self {
        Self::bare(RequestKind::WakeUp)
    }
    pub fn get_metrics() -> Self {
        Self::bare(RequestKind::GetMetrics)
    }
    pub fn get_pressure() -> Self {
        Self::bare(RequestKind::GetPressure)
    }
}

/// Worker-local metrics snapshot. All values are scalar counters or string-keyed
/// scalar maps; no tensors or payload data cross the wire.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkerMetrics {
    pub executes: u64,
    pub ops_total: u64,
    pub exec_us_total: u64,
    pub last_exec_us: u64,
    pub op_kind_counts: BTreeMap<String, u64>,
    pub op_kind_us: BTreeMap<String, u64>,
    pub control_ok: BTreeMap<String, u64>,
    pub control_err: BTreeMap<String, u64>,
    pub error_counts: BTreeMap<String, u64>,
    pub cuda_graph_captures: u64,
    pub cuda_graph_replays: u64,
    pub cuda_graph_misses: u64,
    pub cuda_graph_fallbacks: u64,
    pub cuda_graph_unpadded_tokens: u64,
    pub cuda_graph_padded_tokens: u64,
    pub cuda_graph_runtime_mode_counts: BTreeMap<String, u64>,
    #[serde(default)]
    pub forward: Option<WorkerForwardStats>,
}

/// IPC envelope, worker -> host.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerResponse {
    pub kind: String, // "caps" | "result" | "ok" | "error" | "metrics" | "pressure"
    /// Echo of the request's `call_id` for control-call correlation.
    pub call_id: Option<u64>,
    pub caps: Option<EngineCaps>,
    pub result: Option<ForwardResult>,
    pub metrics: Option<WorkerMetrics>,
    pub pressure: Option<Vec<ResourcePressure>>,
    pub message: Option<String>,
    // ---- typed error taxonomy. On an
    // "error" response these classify the failure so the host can decide
    // abort/retry/drop vs tear-down instead of treating every error as fatal.
    // All scalars / short strings — never tensors. ----
    /// Stable error class (see Python `runtime.errors.ErrorCode`).
    pub code: Option<String>,
    /// Whether retrying the same op could succeed.
    pub retryable: Option<bool>,
    /// Whether the worker can no longer serve (tear it down) vs a single
    /// request/op failure (drop just that request).
    pub fatal: Option<bool>,
}

impl WorkerResponse {
    /// `true` when an `"error"` response should tear the worker down.
    pub fn is_fatal_error(&self) -> bool {
        self.kind == "error" && self.fatal.unwrap_or(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_op_envelopes() {
        let cb = WorkerRequest::copy_blocks(vec![(BlockId(1), BlockId(2))]);
        assert_eq!(cb.kind, RequestKind::CopyBlocks);
        let back = flat::decode_request(&flat::encode_request(&cb).unwrap()).unwrap();
        assert_eq!(back.copies.unwrap(), vec![(BlockId(1), BlockId(2))]);
        let ll = WorkerRequest::load_lora(3, "/tmp/adapter".into());
        assert_eq!(ll.kind, RequestKind::LoadLora);
        assert_eq!(ll.lora_id, Some(3));
    }

    #[test]
    fn request_kind_serializes_to_unchanged_wire_strings() {
        // The Python worker reads `WorkerRequest.kind` as a plain string via the
        // pythonize/serde path; pin every variant's serialized form so the enum
        // discriminant stays byte-identical to the legacy `kind: String` values.
        let cases = [
            (RequestKind::GetCaps, "get_caps"),
            (RequestKind::Execute, "execute"),
            (RequestKind::DropRequest, "drop_request"),
            (RequestKind::Shutdown, "shutdown"),
            (RequestKind::CopyBlocks, "copy_blocks"),
            (RequestKind::LoadLora, "load_lora"),
            (RequestKind::UnloadLora, "unload_lora"),
            (RequestKind::FreeEncoder, "free_encoder"),
            (RequestKind::ResetPrefixCache, "reset_prefix_cache"),
            (RequestKind::Sleep, "sleep"),
            (RequestKind::WakeUp, "wake_up"),
            (RequestKind::GetMetrics, "get_metrics"),
            (RequestKind::GetPressure, "get_pressure"),
        ];
        for (kind, wire) in cases {
            assert_eq!(serde_json::to_value(kind).unwrap(), serde_json::json!(wire));
            assert_eq!(kind.as_wire_str(), wire);
            assert_eq!(RequestKind::from_wire_str(wire), Some(kind));
            assert_eq!(
                serde_json::from_value::<RequestKind>(serde_json::json!(wire)).unwrap(),
                kind
            );
        }
        // The whole envelope still serializes with a flat top-level `kind` string,
        // exactly as the Python worker dispatches on (`req.get("kind")`).
        let req = WorkerRequest::execute(ForwardBatch {
            step_id: 1,
            new_reqs: Vec::new(),
            ops: Vec::new(),
        });
        let value = serde_json::to_value(&req).unwrap();
        assert_eq!(value["kind"], serde_json::json!("execute"));
    }

    #[test]
    fn flatbuffer_request_response_roundtrip() {
        let mut req = WorkerRequest::execute(ForwardBatch {
            step_id: 11,
            new_reqs: vec![NewRequestData {
                sampling: Some(SamplingParams {
                    temperature: 0.7,
                    n_logprobs: 3,
                    ..Default::default()
                }),
                image: Some(ImageParams {
                    width: 2048,
                    height: 1152,
                    ..Default::default()
                }),
                neg_token_ids: Some(vec![1, 2]),
                lora_id: Some(7),
                block_ids: vec![BlockId(9), BlockId(10)],
                group_id: 2,
                ..NewRequestData::new(RequestId(5))
            }],
            ops: vec![ForwardOp {
                req_id: RequestId(5),
                kind: OpKind::DenoiseGen,
                modality: Modality::Gen,
                new_block_ids: vec![BlockId(11)],
                pos_range: (128, 256),
                token_ids: Some(vec![101, 102]),
                token_source: TokenSource::LastSampled,
                timestep_idx: Some(7),
                cond_pos: Some(9),
                cfg: Some(CfgParams {
                    branch_count: 3,
                    text_scale: 4.0,
                    img_scale: 1.0,
                    renorm_type: "global".into(),
                    renorm_min: 0.0,
                    interval: (0.0, 1.0),
                }),
                image_in: Some(99),
                image_prompt: Some("clean scenic destination photograph".into()),
                image_b64: Some("AAAA".into()),
                group_id: 2,
                allowed_tokens: Some(vec![1]),
                suppress_tokens: Some(vec![2]),
                recent_tokens: Some(vec![3]),
                mm_hash: Some(0xABCD),
                spec_token_ids: Some(vec![5]),
                denoise_step_count: Some(2),
                decode_token_count: Some(4),
                decode_stop_token_ids: Some(vec![9, 10]),
                decode_stop_terminal: true,
                op_id: Some(44),
                logits_handle: Some(0xBEEF),
                locator: Some("bG9jYXRvcg==".into()),
            }],
        });
        req.call_id = Some(77);
        let back = flat::decode_request(&flat::encode_request(&req).unwrap()).unwrap();
        assert_eq!(back.call_id, Some(77));
        let batch = back.batch.unwrap();
        assert_eq!(batch.step_id, 11);
        assert_eq!(batch.new_reqs[0].image.as_ref().unwrap().width, 2048);
        assert_eq!(batch.new_reqs[0].block_ids, vec![BlockId(9), BlockId(10)]);
        assert_eq!(batch.ops[0].kind, OpKind::DenoiseGen);
        assert_eq!(batch.ops[0].token_source, TokenSource::LastSampled);
        assert_eq!(
            batch.ops[0].image_prompt.as_deref(),
            Some("clean scenic destination photograph")
        );
        assert_eq!(batch.ops[0].op_id, Some(44));
        assert_eq!(batch.ops[0].denoise_step_count, Some(2));
        assert_eq!(batch.ops[0].decode_token_count, Some(4));
        assert_eq!(batch.ops[0].decode_stop_token_ids, Some(vec![9, 10]));
        assert!(batch.ops[0].decode_stop_terminal);
        assert_eq!(batch.ops[0].logits_handle, Some(0xBEEF));
        assert_eq!(batch.ops[0].locator.as_deref(), Some("bG9jYXRvcg=="));
    }

    #[test]
    fn flatbuffer_response_variants_roundtrip() {
        let mut metrics = WorkerMetrics {
            executes: 1,
            ops_total: 2,
            exec_us_total: 30,
            last_exec_us: 20,
            ..Default::default()
        };
        metrics.op_kind_counts.insert("prefill_und".into(), 1);
        metrics.cuda_graph_replays = 3;
        metrics.cuda_graph_padded_tokens = 2;
        metrics
            .cuda_graph_runtime_mode_counts
            .insert("decode".into(), 3);
        let metrics_resp = WorkerResponse {
            kind: "metrics".into(),
            call_id: Some(9),
            caps: None,
            result: None,
            metrics: Some(metrics),
            pressure: Some(vec![ResourcePressure {
                class: ResourceClass::KvBlock,
                total: 4,
                used: 2,
                evictable: 0,
                free: 2,
            }]),
            message: None,
            code: None,
            retryable: None,
            fatal: None,
        };
        let decoded =
            flat::decode_response(&flat::encode_response(&metrics_resp).unwrap()).unwrap();
        assert_eq!(decoded.call_id, Some(9));
        let decoded_metrics = decoded.metrics.unwrap();
        assert_eq!(decoded_metrics.executes, 1);
        assert_eq!(decoded_metrics.cuda_graph_replays, 3);
        assert_eq!(decoded_metrics.cuda_graph_padded_tokens, 2);
        assert_eq!(
            decoded_metrics.cuda_graph_runtime_mode_counts.get("decode"),
            Some(&3),
        );
        assert_eq!(decoded.pressure.unwrap()[0].free, 2);

        let nonfatal = WorkerResponse {
            kind: "error".into(),
            call_id: None,
            caps: None,
            result: None,
            metrics: None,
            pressure: None,
            message: Some("bad op".into()),
            code: Some("UnsupportedOperation".into()),
            retryable: Some(false),
            fatal: Some(false),
        };
        let decoded = flat::decode_response(&flat::encode_response(&nonfatal).unwrap()).unwrap();
        assert!(!decoded.is_fatal_error());
    }

    #[test]
    fn flatbuffer_native_image_result_roundtrip() {
        let resp = WorkerResponse {
            kind: "result".into(),
            call_id: Some(8),
            caps: None,
            result: Some(ForwardResult {
                step_id: 11,
                per_seq: vec![SeqResult {
                    req_id: RequestId(5),
                    image_hw: Some((2048, 1152)),
                    image_png_b64: Some("AAAA".into()),
                    sampled_token_id: Some(42),
                    sampled_token_ids: Some(vec![40, 41, 42]),
                    op_id: Some(44),
                    logits_handle: Some(0x1234),
                    locator: Some("c2VxbG9j".into()),
                    ..Default::default()
                }],
                worker_exec_us: Some(123),
                forward_stats: Some(WorkerForwardStats {
                    component_us: BTreeMap::from([("text_model_forward".into(), 77)]),
                    flashinfer_decode_plan_calls: 2,
                    spec_verify_accepted_tokens: 3,
                    ..Default::default()
                }),
            }),
            metrics: None,
            pressure: None,
            message: None,
            code: None,
            retryable: None,
            fatal: None,
        };
        let back = flat::decode_response(&flat::encode_response(&resp).unwrap()).unwrap();
        assert_eq!(back.call_id, Some(8));
        let result = back.result.unwrap();
        assert_eq!(result.worker_exec_us, Some(123));
        let forward_stats = result.forward_stats.unwrap();
        assert_eq!(
            forward_stats.component_us.get("text_model_forward"),
            Some(&77)
        );
        assert_eq!(forward_stats.flashinfer_decode_plan_calls, 2);
        assert_eq!(forward_stats.spec_verify_accepted_tokens, 3);
        assert_eq!(result.per_seq[0].image_hw, Some((2048, 1152)));
        assert_eq!(result.per_seq[0].sampled_token_id, Some(42));
        assert_eq!(result.per_seq[0].sampled_token_ids, Some(vec![40, 41, 42]));
        assert_eq!(result.per_seq[0].op_id, Some(44));
        assert_eq!(result.per_seq[0].logits_handle, Some(0x1234));
        assert_eq!(result.per_seq[0].locator.as_deref(), Some("c2VxbG9j"));
    }

    /// the serde structs and the FlatBuffers schema are two independent
    /// source-of-truth definitions kept in sync only by the hand-written mapping
    /// in `flat.rs`. The compiler catches a *missing* field (the `*T` struct
    /// literals are exhaustive), but it cannot catch a swapped/mis-paired scalar
    /// within a correctly-listed tuple field, because the serde side carries
    /// `(u32, u32)` / `(f32, f32)` tuples while the FB side splits them into
    /// `_lo`/`_hi`, `_w`/`_h` scalar pairs. This test pins the pair *ordering*
    /// by round-tripping deliberately asymmetric values: any `pos_lo<->pos_hi`,
    /// `image_hw_w<->image_hw_h`, `interval_lo<->interval_hi`, or
    /// `cfg_interval_lo<->cfg_interval_hi` transposition in `flat.rs` flips the
    /// observed tuple and fails here instead of silently mis-mapping on the wire.
    #[test]
    fn flatbuffer_scalar_pair_ordering_is_preserved() {
        // Asymmetric values so a `.0`/`.1` swap cannot round-trip equal.
        let req = WorkerRequest::execute(ForwardBatch {
            step_id: 1,
            new_reqs: vec![NewRequestData {
                image: Some(ImageParams {
                    // `cfg_interval` splits into cfg_interval_lo/cfg_interval_hi.
                    cfg_interval: (0.125, 0.875),
                    width: 100,
                    height: 200,
                    ..Default::default()
                }),
                ..NewRequestData::new(RequestId(1))
            }],
            ops: vec![ForwardOp {
                req_id: RequestId(1),
                // `pos_range` splits into pos_lo/pos_hi.
                pos_range: (3, 7),
                cfg: Some(CfgParams {
                    branch_count: 1,
                    text_scale: 1.0,
                    img_scale: 1.0,
                    renorm_type: "global".into(),
                    renorm_min: 0.0,
                    // `interval` splits into interval_lo/interval_hi.
                    interval: (0.25, 0.75),
                }),
                ..ForwardOp::default()
            }],
        });
        let back = flat::decode_request(&flat::encode_request(&req).unwrap()).unwrap();
        let batch = back.batch.unwrap();
        assert_eq!(batch.ops[0].pos_range, (3, 7), "pos_lo/pos_hi transposed");
        assert_eq!(
            batch.ops[0].cfg.as_ref().unwrap().interval,
            (0.25, 0.75),
            "interval_lo/interval_hi transposed"
        );
        let image = batch.new_reqs[0].image.as_ref().unwrap();
        assert_eq!(
            image.cfg_interval,
            (0.125, 0.875),
            "cfg_interval_lo/cfg_interval_hi transposed"
        );
        assert_eq!(
            (image.width, image.height),
            (100, 200),
            "image width/height transposed"
        );

        // `SeqResult.image_hw` splits into image_hw_w/image_hw_h on the FB side.
        let resp = WorkerResponse {
            kind: "result".into(),
            call_id: None,
            caps: None,
            result: Some(ForwardResult {
                step_id: 1,
                per_seq: vec![SeqResult {
                    req_id: RequestId(1),
                    image_hw: Some((640, 480)),
                    ..Default::default()
                }],
                worker_exec_us: None,
                forward_stats: None,
            }),
            metrics: None,
            pressure: None,
            message: None,
            code: None,
            retryable: None,
            fatal: None,
        };
        let back = flat::decode_response(&flat::encode_response(&resp).unwrap()).unwrap();
        assert_eq!(
            back.result.unwrap().per_seq[0].image_hw,
            Some((640, 480)),
            "image_hw_w/image_hw_h transposed"
        );
    }

    /// Drift guard for the canonical op-kind vocabulary
    /// (`crates/protocol/vocab/op_kinds.toml`): the Rust `OpKind` enum must equal
    /// the schema exactly — same set of wire strings (serde snake_case), and the
    /// FlatBuffers declaration order (ubyte ordinals, a wire contract) preserved.
    /// Adding an op kind requires updating the schema AND every language, or
    /// this fails.
    #[test]
    fn op_kind_vocabulary_matches_canonical_schema() {
        const SCHEMA: &str = include_str!("../../vocab/op_kinds.toml");
        let schema: toml::Value = toml::from_str(SCHEMA).unwrap();

        // Every Rust `OpKind` variant, enumerated via an exhaustive match so a
        // new variant fails to compile here until it is added to this list and
        // the schema.
        fn wire_str(k: OpKind) -> &'static str {
            match k {
                OpKind::PrefillUnd => "prefill_und",
                OpKind::DecodeUnd => "decode_und",
                OpKind::TargetVerifyUnd => "target_verify_und",
                OpKind::DenoiseGen => "denoise_gen",
                OpKind::CommitGen => "commit_gen",
                OpKind::CommitWriteback => "commit_writeback",
                OpKind::VaeEncode => "vae_encode",
                OpKind::VitEncode => "vit_encode",
                OpKind::Sample => "sample",
                OpKind::EncodeFrame => "encode_frame",
            }
        }
        let rust_variants = [
            OpKind::PrefillUnd,
            OpKind::DecodeUnd,
            OpKind::TargetVerifyUnd,
            OpKind::DenoiseGen,
            OpKind::CommitGen,
            OpKind::CommitWriteback,
            OpKind::VaeEncode,
            OpKind::VitEncode,
            OpKind::Sample,
            OpKind::EncodeFrame,
        ];

        // The `wire_str` helper above must agree with serde snake_case (the form
        // that crosses the IPC boundary) for every variant.
        for k in rust_variants {
            assert_eq!(
                serde_json::to_value(k).unwrap(),
                serde_json::json!(wire_str(k)),
                "OpKind::{k:?} serde form drifted from its mapped wire string"
            );
        }

        let rust_wire: std::collections::BTreeSet<String> = rust_variants
            .iter()
            .map(|k| wire_str(*k).to_string())
            .collect();

        // `[[op]]` table: wire strings + Rust variant identifiers.
        let ops = schema["op"].as_array().expect("schema [[op]] array");
        let schema_wire: std::collections::BTreeSet<String> = ops
            .iter()
            .map(|o| o["wire"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            rust_wire, schema_wire,
            "OpKind wire set drifted from op_kinds.toml"
        );

        // The schema's `rust` column must name the variant whose wire string is
        // its `wire` column, so the human-readable Rust identifiers stay pinned.
        let variant_for_wire: std::collections::BTreeMap<&str, &str> = rust_variants
            .iter()
            .map(|k| (wire_str(*k), variant_ident(*k)))
            .collect();
        fn variant_ident(k: OpKind) -> &'static str {
            match k {
                OpKind::PrefillUnd => "PrefillUnd",
                OpKind::DecodeUnd => "DecodeUnd",
                OpKind::TargetVerifyUnd => "TargetVerifyUnd",
                OpKind::DenoiseGen => "DenoiseGen",
                OpKind::CommitGen => "CommitGen",
                OpKind::CommitWriteback => "CommitWriteback",
                OpKind::VaeEncode => "VaeEncode",
                OpKind::VitEncode => "VitEncode",
                OpKind::Sample => "Sample",
                OpKind::EncodeFrame => "EncodeFrame",
            }
        }
        for o in ops {
            let wire = o["wire"].as_str().unwrap();
            let rust = o["rust"].as_str().unwrap();
            assert_eq!(
                variant_for_wire.get(wire),
                Some(&rust),
                "op_kinds.toml `rust` column for {wire:?} drifted"
            );
        }

        // FlatBuffers declaration order: the schema's `flatbuffers_order` list
        // must match the `enum OpKind` body in worker.fbs verbatim (ordinals are
        // a wire contract). worker.fbs lives one crate over; skip if absent
        // (Python-only checkout).
        let order: Vec<String> = schema["flatbuffers_order"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        // The order list must cover exactly the same variant set (as Rust idents).
        let order_set: std::collections::BTreeSet<&str> =
            order.iter().map(String::as_str).collect();
        let ident_set: std::collections::BTreeSet<&str> =
            rust_variants.iter().map(|k| variant_ident(*k)).collect();
        assert_eq!(
            order_set, ident_set,
            "flatbuffers_order set drifted from OpKind variants"
        );

        let fbs_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../worker-ipc-core/schema/worker.fbs");
        if let Ok(fbs) = std::fs::read_to_string(&fbs_path) {
            let start = fbs.find("enum OpKind").expect("enum OpKind in worker.fbs");
            let open = fbs[start..].find('{').unwrap() + start;
            let close = fbs[open..].find('}').unwrap() + open;
            let body = &fbs[open + 1..close];
            let members: Vec<String> = body
                .split(',')
                .map(|m| m.split('=').next().unwrap().trim().to_string())
                .filter(|m| !m.is_empty())
                .collect();
            assert_eq!(
                members, order,
                "worker.fbs OpKind member order drifted from op_kinds.toml flatbuffers_order"
            );
        }
    }

    /// Compile-time tripwire: exhaustive destructuring of hot wire payloads
    /// (`ForwardOp`, `SeqResult`, `NewRequestData`) with no `..` rest pattern,
    /// so adding a field breaks compilation and forces an explicit decision
    /// about whether it may carry tensor data.
    #[test]
    fn wire_payload_fields_are_descriptors_only() {
        // The bindings are intentionally unused; the destructuring itself is the
        // assertion (it must enumerate every field — no `..`).
        let ForwardOp {
            req_id: _,
            kind: _,
            modality: _,
            new_block_ids: _,         // logical block ids (scalars), never KV bytes
            pos_range: _,             // (u32, u32) descriptor
            token_ids: _,             // input token ids — small id list, not logits
            token_source: _,          // enum selector
            timestep_idx: _,          // scalar
            cond_pos: _,              // scalar
            cfg: _,                   // CfgParams descriptor (scalars only)
            image_in: _,              // StagedImageId handle (scalar)
            image_prompt: _,          // short prompt string descriptor
            image_b64: _,             // input image bytes (b64), output-style small result
            group_id: _,              // scalar
            allowed_tokens: _,        // id list
            suppress_tokens: _,       // id list
            recent_tokens: _,         // bounded id list
            mm_hash: _,               // content hash (scalar)
            spec_token_ids: _,        // draft token ids (scalars)
            denoise_step_count: _,    // scalar burst count for sequential denoise
            decode_token_count: _,    // scalar burst count for sequential text decode
            decode_stop_token_ids: _, // id list of scalar burst stop tokens
            decode_stop_terminal: _,  // whether stop ends the request
            op_id: _,                 // lifecycle id (scalar)
            logits_handle: _,         // opaque logits handle (scalar); logits stay off-wire
            locator: _, // base64 data-plane locator (small descriptor); tensor stays off-wire
        } = ForwardOp::default();

        let SeqResult {
            req_id: _,
            sampled_token_id: _,    // sampled token id (scalar)
            denoise_done: _,        // bool
            num_steps_done: _,      // scalar
            image_png_b64: _,       // finished image bytes (small result, not KV)
            image_hw: _,            // (u32, u32) descriptor
            sampled_logprob: _,     // scalar
            top_logprobs: _,        // (id, logprob) scalar pairs, never a logits tensor
            sampled_token_ids: _,   // sampled token id list, never logits
            encoder_handle: _,      // opaque handle (scalar); embedding stays worker-side
            num_tokens: _,          // scalar
            num_accepted_tokens: _, // scalar
            op_id: _,               // lifecycle id (scalar)
            logits_handle: _,       // opaque logits handle (scalar); logits stay off-wire
            locator: _, // base64 data-plane locator (small descriptor); tensor stays off-wire
        } = SeqResult::default();

        let NewRequestData {
            req_id: _,
            sampling: _,      // SamplingParams descriptor (scalars / small id lists)
            image: _,         // ImageParams descriptor (scalars)
            neg_token_ids: _, // id list
            lora_id: _,       // scalar handle
            block_ids: _,     // logical block ids (scalars), never KV bytes
            group_id: _,      // scalar
        } = NewRequestData::new(RequestId(0));
    }
}
