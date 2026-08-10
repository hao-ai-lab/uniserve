//! Shared IDs, value parameters, and pure helpers.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
use std::sync::{Arc, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

pub mod generation;
pub mod philox;
pub mod product_blob;
pub mod sampling;
pub use generation::{
    ContextSegment, FeedbackNextToken, FeedbackSource, GenOnlyStartPolicyDescriptor,
    GeneratedImageFeedbackRecipe, GenerationBehaviorDescriptor, GenerationCachePolicyDescriptor,
    GenerationCapabilityNeeds, GenerationConstraint, GenerationConstraintParseError,
    GenerationPolicyDescriptor, GenerationRequest, GenerationRequestError,
    GenerationResourceBounds, GenerationResourceError, GenerationRuntimeCapabilities,
    ImageIngestRecipe, ImageIngestStep, ImageKvEffect, ImageSegment, SegmentPlacement,
    TerminationPolicyDescriptor, TriggerPolicyDescriptor, UndTokenAction, UndVisibility,
    VisibilityPolicyDescriptor, denoise_scratch_tokens, encoder_cache_key,
};
pub use sampling::{SampleOutput, apply_sampling, score_token_logprobs, try_apply_sampling_counts};

/// A cloneable, thread-safe wake the command ingress fires after enqueuing a
/// command, so a parked event-driven executor wakes immediately instead of
/// after its safety-net timeout.
///
/// Defined in the foundation crate so the executor seam (which mints the real
/// waker over an iceoryx2 notifier) and the engine API (which holds it on the
/// command front door and fires it on every send) can share the type without a
/// cross-crate dependency. Polling executors hand out [`CommandWaker::noop`]:
/// they observe commands through their own timed wait, so firing it is a no-op.
#[derive(Clone, Default)]
pub struct CommandWaker(Option<Arc<dyn Fn() + Send + Sync>>);

impl CommandWaker {
    /// A waker that does nothing (the polling path).
    pub fn noop() -> Self {
        Self(None)
    }

    /// Wrap a wake closure (e.g. fire an iceoryx2 notifier).
    pub fn new(wake: impl Fn() + Send + Sync + 'static) -> Self {
        Self(Some(Arc::new(wake)))
    }

    /// Fire the wake. A no-op for the polling variant.
    pub fn wake(&self) {
        if let Some(f) = &self.0 {
            f();
        }
    }

    /// Whether this is the no-op (polling) waker.
    pub fn is_noop(&self) -> bool {
        self.0.is_none()
    }
}

impl std::fmt::Debug for CommandWaker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommandWaker")
            .field("active", &self.0.is_some())
            .finish()
    }
}

/// Current wall-clock time in fractional seconds since the Unix epoch.
///
/// Shared by the frontend, engine client, scheduler, and engine process for
/// latency metrics and wire timestamps. Never panics: a clock set before the
/// epoch (or stepped backward) clamps to `0.0` rather than unwrapping the
/// `Result`.
pub fn now_unix_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

/// Current wall-clock time in whole seconds since the Unix epoch.
///
/// Integer-seconds companion to [`now_unix_secs`] for callers that want a `u64`.
/// Shares the same panic-free epoch source so the two cannot diverge.
pub fn now_unix_secs_u64() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Process-local monotonic time in fractional seconds from a stable epoch.
/// Differences between values remain valid across wall-clock adjustments.
pub fn now_monotonic_secs() -> f64 {
    EPOCH.get_or_init(Instant::now).elapsed().as_secs_f64()
}

/// Process-local monotonic time in whole microseconds from the same epoch as
/// [`now_monotonic_secs`]. Used for lifecycle phase stamps, whose differences
/// stay valid across wall-clock adjustments.
pub fn now_monotonic_us() -> u64 {
    EPOCH.get_or_init(Instant::now).elapsed().as_micros() as u64
}

static EPOCH: OnceLock<Instant> = OnceLock::new();

/// Logical KV block id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct BlockId(pub u32);

/// Request id.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize,
)]
pub struct RequestId(pub u64);

/// Trace id for end-to-end lifecycle reconstruction. One
/// per request unless the frontend correlates several; defaults to the request id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct TraceId(pub u64);

/// Op id: a single op within a program/request. `(request, seq)`
/// flattened to a u64 on the wire so the host correlates op result ↔ submitted op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct OpId(pub u64);

/// The two modality branches of the MoT model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Und, // understanding / text
    Gen, // generation / image latents
}

/// Effective model dtype reported after runtime configuration resolves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelDtype {
    #[serde(rename = "float16")]
    Float16,
    #[serde(rename = "bfloat16")]
    BFloat16,
    #[serde(rename = "float32")]
    Float32,
}

impl ModelDtype {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Float16 => "float16",
            Self::BFloat16 => "bfloat16",
            Self::Float32 => "float32",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "float16" => Some(Self::Float16),
            "bfloat16" => Some(Self::BFloat16),
            "float32" => Some(Self::Float32),
            _ => None,
        }
    }
}

/// Text sampling parameters.
///
/// Worker-side math (temperature, top_k, top_p, min_p, penalties, logit_bias)
/// operates on the logits tensor inside the worker; control-flow floors
/// (`min_tokens`, `ignore_eos`) are enforced on the host. Fields are scalars or
/// small id/weight lists — never tensors — so they can cross the worker boundary
/// without carrying device state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub ignore_eos: bool,
    pub seed: Option<u64>,
    /// Minimum-probability cutoff relative to the top token.
    #[serde(default)]
    pub min_p: f32,
    /// Multiplicative repetition penalty (1.0 == no-op).
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    /// Additive frequency penalty over the running output (0.0 == no-op).
    #[serde(default)]
    pub frequency_penalty: f32,
    /// Additive presence penalty over the running output (0.0 == no-op).
    #[serde(default)]
    pub presence_penalty: f32,
    /// Per-token logit bias, `(token_id, bias)` pairs (empty == no-op).
    #[serde(default)]
    pub logit_bias: Vec<(u32, f32)>,
    /// Floor on generated tokens before EOS/stop may fire.
    #[serde(default)]
    pub min_tokens: usize,
    /// Whether to return the sampled token's log probability.
    #[serde(default)]
    pub return_logprobs: bool,
    /// How many highest-probability alternatives to return per generated step.
    #[serde(default)]
    pub n_logprobs: u32,
    /// Whether to score prompt positions during prefill.
    #[serde(default)]
    pub return_prompt_logprobs: bool,
    /// How many highest-probability alternatives to return per scored prompt position.
    #[serde(default)]
    pub n_prompt_logprobs: u32,
    /// Token IDs whose log probabilities are returned at every scored position.
    #[serde(default)]
    pub logprob_token_ids: Vec<u32>,
    /// Bad-word token sequences: generation may not complete any of these.
    /// Enforced host-side by masking the completing token.
    #[serde(default)]
    pub bad_words_ids: Vec<Vec<u32>>,
    /// If set, only these token ids may be sampled (whitelist mask).
    #[serde(default)]
    pub allowed_token_ids: Option<Vec<u32>>,
    /// Typical-sampling mass cutoff over the locally-typical set (1.0 == no-op).
    #[serde(default = "default_typical_p")]
    pub typical_p: f32,
    /// Forced-decoding schedule: point `i` of an operation's span is forced to
    /// `forced_token_ids[i]` when present, overriding stochastic selection.
    #[serde(default)]
    pub forced_token_ids: Vec<u32>,
}
fn default_repetition_penalty() -> f32 {
    1.0
}
fn default_typical_p() -> f32 {
    1.0
}

/// Why [`SamplingParams`] were rejected before worker execution.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SamplingParamsError {
    #[error("{field} must be finite, got {got}")]
    NonFinite { field: &'static str, got: f32 },
    #[error("temperature must be non-negative, got {got}")]
    NegativeTemperature { got: f32 },
    #[error("top_p must be in (0, 1], got {got}")]
    TopP { got: f32 },
    #[error("min_p must be in [0, 1], got {got}")]
    MinP { got: f32 },
    #[error("repetition_penalty must be positive, got {got}")]
    RepetitionPenalty { got: f32 },
    #[error("allowed_token_ids must contain at least one token when present")]
    EmptyAllowedTokenIds,
    #[error("bad_words_ids[{index}] must contain at least one token")]
    EmptyBadWord { index: usize },
}

impl SamplingParams {
    pub fn generated_logprobs_requested(&self) -> bool {
        self.return_logprobs || self.n_logprobs > 0 || !self.logprob_token_ids.is_empty()
    }

    pub fn prompt_logprobs_requested(&self) -> bool {
        self.return_prompt_logprobs || self.n_prompt_logprobs > 0
    }

    /// Validate sampling math inputs before they reach a worker or simulator.
    pub fn validate(&self) -> Result<(), SamplingParamsError> {
        for (field, value) in [
            ("temperature", self.temperature),
            ("top_p", self.top_p),
            ("min_p", self.min_p),
            ("repetition_penalty", self.repetition_penalty),
            ("frequency_penalty", self.frequency_penalty),
            ("presence_penalty", self.presence_penalty),
        ] {
            if !value.is_finite() {
                return Err(SamplingParamsError::NonFinite { field, got: value });
            }
        }
        for &(_, bias) in &self.logit_bias {
            if !bias.is_finite() {
                return Err(SamplingParamsError::NonFinite {
                    field: "logit_bias",
                    got: bias,
                });
            }
        }
        if self.temperature < 0.0 {
            return Err(SamplingParamsError::NegativeTemperature {
                got: self.temperature,
            });
        }
        if !(0.0 < self.top_p && self.top_p <= 1.0) {
            return Err(SamplingParamsError::TopP { got: self.top_p });
        }
        if !(0.0..=1.0).contains(&self.min_p) {
            return Err(SamplingParamsError::MinP { got: self.min_p });
        }
        if self.repetition_penalty <= 0.0 {
            return Err(SamplingParamsError::RepetitionPenalty {
                got: self.repetition_penalty,
            });
        }
        if self.allowed_token_ids.as_ref().is_some_and(Vec::is_empty) {
            return Err(SamplingParamsError::EmptyAllowedTokenIds);
        }
        if let Some(index) = self.bad_words_ids.iter().position(Vec::is_empty) {
            return Err(SamplingParamsError::EmptyBadWord { index });
        }
        Ok(())
    }
}

impl Default for SamplingParams {
    /// Default `temperature` is `0.0`, which `sampling::apply_sampling` treats
    /// as greedy (argmax). The production frontend resolves an unset user
    /// temperature to `1.0` during lowering; a `SamplingParams` reaching the
    /// worker has always passed through that path. This default is for direct
    /// constructors (tests, sim, wire fallbacks) only.
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            ignore_eos: false,
            seed: None,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            logit_bias: Vec::new(),
            min_tokens: 0,
            return_logprobs: false,
            n_logprobs: 0,
            return_prompt_logprobs: false,
            n_prompt_logprobs: 0,
            logprob_token_ids: Vec::new(),
            bad_words_ids: Vec::new(),
            allowed_token_ids: None,
            typical_p: 1.0,
            forced_token_ids: Vec::new(),
        }
    }
}

/// Image (diffusion) parameters. `max_images` makes the admission budget finite for requests that can produce images.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageParams {
    pub steps: u16,
    pub cfg_text_scale: f32,
    pub cfg_img_scale: f32,
    pub cfg_renorm_type: String,
    pub cfg_renorm_min: f32,
    pub cfg_interval: (f32, f32),
    #[serde(default = "default_timestep_shift")]
    pub timestep_shift: f32,
    pub height: u32,
    pub width: u32,
    pub seed: Option<u64>,
    pub negative_prompt: String,
    pub max_images: u16,
    #[serde(default)]
    pub image_prompts: Vec<String>,
    #[serde(default = "default_retain_images")]
    pub retain_images: bool,
}

impl ImageParams {
    /// Number of active classifier-free-guidance branches implied by the two
    /// guidance axes. This is the authoritative scratch branch-slot bound.
    pub fn cfg_branch_count(&self) -> u8 {
        let text_off = scale_approx(self.cfg_text_scale, 1.0);
        let image_off = scale_approx(self.cfg_img_scale, 1.0);
        if text_off && image_off {
            1
        } else if text_off || image_off {
            2
        } else {
            3
        }
    }
}

fn scale_approx(left: f32, right: f32) -> bool {
    (left - right).abs() <= 1.0e-6_f32 * left.abs().max(right.abs()).max(1.0)
}
fn default_timestep_shift() -> f32 {
    1.0
}
fn default_retain_images() -> bool {
    true
}
/// Why an [`ImageParams`] was rejected by [`ImageParams::validate`].
/// Each variant names the offending field and the bound it violated so the
/// frontend can surface a precise 4xx instead of letting unbounded diffusion
/// work reach the worker.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ImageParamsError {
    #[error("steps must be in 1..={max}, got {got}")]
    Steps { got: u16, max: u16 },
    #[error("height must be a non-zero multiple of {multiple} in {min}..={max}, got {got}")]
    Height {
        got: u32,
        min: u32,
        max: u32,
        multiple: u32,
    },
    #[error("width must be a non-zero multiple of {multiple} in {min}..={max}, got {got}")]
    Width {
        got: u32,
        min: u32,
        max: u32,
        multiple: u32,
    },
    #[error("{field} must be finite and in 0.0..={max}, got {got}")]
    CfgScale {
        field: &'static str,
        got: f32,
        max: f32,
    },
    #[error("max_images must be in 1..={max}, got {got}")]
    MaxImages { got: u16, max: u16 },
    #[error("{field} must be finite, got {got}")]
    NonFinite { field: &'static str, got: f32 },
    #[error("cfg_interval must be an ordered pair, got ({lo}, {hi})")]
    CfgIntervalOrder { lo: f32, hi: f32 },
    #[error("cfg_renorm_type must not be empty")]
    EmptyCfgRenormType,
}

impl ImageParams {
    /// Upper bound on diffusion steps.
    pub const MAX_STEPS: u16 = 1000;
    /// Pixel-dimension bounds. `height`/`width` must be non-zero, within
    /// `[MIN_DIM, MAX_DIM]`, and a multiple of `DIM_MULTIPLE`.
    pub const MIN_DIM: u32 = 16;
    pub const MAX_DIM: u32 = 4096;
    pub const DIM_MULTIPLE: u32 = 16;
    /// Upper bound on any single CFG scale.
    pub const MAX_CFG_SCALE: f32 = 100.0;
    /// Upper bound on `max_images` per request.
    pub const MAX_IMAGES: u16 = 256;

    /// Validate diffusion parameters before they reach the worker.
    pub fn validate(&self) -> Result<(), ImageParamsError> {
        if self.steps == 0 || self.steps > Self::MAX_STEPS {
            return Err(ImageParamsError::Steps {
                got: self.steps,
                max: Self::MAX_STEPS,
            });
        }
        let check_dim = |got: u32| -> bool {
            got != 0
                && (Self::MIN_DIM..=Self::MAX_DIM).contains(&got)
                && got.is_multiple_of(Self::DIM_MULTIPLE)
        };
        if !check_dim(self.height) {
            return Err(ImageParamsError::Height {
                got: self.height,
                min: Self::MIN_DIM,
                max: Self::MAX_DIM,
                multiple: Self::DIM_MULTIPLE,
            });
        }
        if !check_dim(self.width) {
            return Err(ImageParamsError::Width {
                got: self.width,
                min: Self::MIN_DIM,
                max: Self::MAX_DIM,
                multiple: Self::DIM_MULTIPLE,
            });
        }
        for (field, scale) in [
            ("cfg_text_scale", self.cfg_text_scale),
            ("cfg_img_scale", self.cfg_img_scale),
        ] {
            if !scale.is_finite() || !(0.0..=Self::MAX_CFG_SCALE).contains(&scale) {
                return Err(ImageParamsError::CfgScale {
                    field,
                    got: scale,
                    max: Self::MAX_CFG_SCALE,
                });
            }
        }
        for (field, value) in [
            ("cfg_renorm_min", self.cfg_renorm_min),
            ("cfg_interval.0", self.cfg_interval.0),
            ("cfg_interval.1", self.cfg_interval.1),
            ("timestep_shift", self.timestep_shift),
        ] {
            if !value.is_finite() {
                return Err(ImageParamsError::NonFinite { field, got: value });
            }
        }
        if self.cfg_interval.0 > self.cfg_interval.1 {
            return Err(ImageParamsError::CfgIntervalOrder {
                lo: self.cfg_interval.0,
                hi: self.cfg_interval.1,
            });
        }
        if self.cfg_renorm_type.trim().is_empty() {
            return Err(ImageParamsError::EmptyCfgRenormType);
        }
        if self.max_images == 0 || self.max_images > Self::MAX_IMAGES {
            return Err(ImageParamsError::MaxImages {
                got: self.max_images,
                max: Self::MAX_IMAGES,
            });
        }
        Ok(())
    }
}

impl Default for ImageParams {
    fn default() -> Self {
        Self {
            steps: 50,
            cfg_text_scale: 4.0,
            cfg_img_scale: 1.0,
            cfg_renorm_type: "global".into(),
            cfg_renorm_min: 0.0,
            cfg_interval: (0.0, 1.0),
            timestep_shift: 1.0,
            height: 512,
            width: 512,
            seed: None,
            negative_prompt: String::new(),
            max_images: 1,
            image_prompts: Vec::new(),
            retain_images: true,
        }
    }
}

/// CFG parameters carried on gen ops.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfgParams {
    pub branch_count: u8, // 1..=3
    pub text_scale: f32,
    pub img_scale: f32,
    pub renorm_type: String,
    pub renorm_min: f32,
    #[serde(default)]
    pub interval: (f32, f32),
}

/// Physical KV layout descriptor:
/// one page-first buffer addressed by `base + block_id*page_stride + layer*layer_stride`.
/// Computed on the host for admission/sizing; the worker owns the actual buffer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct KvLayout {
    pub num_blocks: u32,
    pub block_size: u32, // tokens per block (page)
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub dtype_bytes: u32,
}
impl KvLayout {
    pub fn bytes_per_token(&self) -> u64 {
        // k + v, all layers
        2 * self.num_kv_heads as u64
            * self.head_dim as u64
            * self.num_layers as u64
            * self.dtype_bytes as u64
    }
    pub fn total_bytes(&self) -> u64 {
        self.num_blocks as u64 * self.block_size as u64 * self.bytes_per_token()
    }
}

/// Attention kind for a KV-cache group.
/// A model with mixed attention (e.g. full + sliding-window) maps to multiple
/// groups, each with its own physical page subspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KvGroupKind {
    /// Full attention: every block is retained for the request's lifetime.
    #[default]
    Full,
    /// Sliding-window attention with `sink` always-kept prefix tokens;
    /// blocks outside the window are eviction candidates once they fall out of range.
    SlidingWindow { window: u32, sink: u32 },
}

/// One KV-cache group: a physical page subspace with its own layout and kind.
/// Reported by the worker at handshake.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCacheGroupSpec {
    pub group_id: u32,
    /// First physical page id owned by this group (groups partition the id space).
    pub block_offset: u32,
    /// Number of physical pages in this group's subspace.
    pub num_blocks: u32,
    #[serde(default)]
    pub kind: KvGroupKind,
}

/// Rank and parallelism topology descriptor reported by a worker. The host fans
/// descriptors to ranks and joins small results; cross-rank KV movement lives
/// inside the worker tier, not on the control-plane wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankInfo {
    pub tp_rank: u32,
    pub tp_size: u32,
    pub pp_rank: u32,
    pub pp_size: u32,
    pub dp_rank: u32,
    pub dp_size: u32,
}
impl Default for RankInfo {
    fn default() -> Self {
        Self {
            tp_rank: 0,
            tp_size: 1,
            pp_rank: 0,
            pp_size: 1,
            dp_rank: 0,
            dp_size: 1,
        }
    }
}

/// Prefix-cache block-hash algorithm. Pluggable, seeded from config; the
/// default is the fast non-cryptographic FNV-1a-with-seed mixer. Sha256 is an
/// option for environments that want a cryptographic digest. Both produce a
/// `u64` slot for `BlockHashToBlock`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum HashAlgo {
    #[default]
    Fnv1a,
    Sha256,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Core default temperature is intentionally greedy (0.0).
    #[test]
    fn sampling_default_is_greedy() {
        let sp = SamplingParams::default();
        assert_eq!(
            sp.temperature, 0.0,
            "core default must stay greedy (0.0); unset->1.0 is resolved in lowering, not here"
        );
        assert_eq!(sp.top_p, 1.0);
        assert_eq!(sp.repetition_penalty, 1.0);
        assert_eq!(sp.top_k, 0);
    }

    #[test]
    fn image_params_default_is_valid() {
        assert!(ImageParams::default().validate().is_ok());
    }

    // ----------------------------------------------------------------------
    // now_unix_secs / now_unix_secs_u64 share one panic-free epoch source. The
    // production functions read the wall clock, which is not injectable, so the
    // exact conversion invariants are tested as a pure `SystemTime -> (f64, u64)`
    // mapping with an injectable instant that mirrors the production expression
    // (`duration_since(UNIX_EPOCH).unwrap_or_default()` then `.as_secs_f64()` /
    // `.as_secs()`). The real functions get a lighter panic-free smoke check.
    // ----------------------------------------------------------------------

    /// Pure mirror of the two helpers with the clock injected as a `SystemTime`.
    /// Returns `(fractional_secs, whole_secs)`; pre-epoch instants clamp to zero
    /// exactly as `unwrap_or_default()` does in production.
    fn epoch_conversion(clock: SystemTime) -> (f64, u64) {
        let d = clock.duration_since(UNIX_EPOCH).unwrap_or_default();
        (d.as_secs_f64(), d.as_secs())
    }

    #[test]
    fn epoch_conversion_u64_is_floor_of_f64() {
        // A fractional post-epoch instant: whole seconds equal floor of the
        // fractional seconds.
        let clock = UNIX_EPOCH + std::time::Duration::from_millis(1_234_750); // 1234.75s
        let (frac, whole) = epoch_conversion(clock);
        assert!((frac - 1234.75).abs() < 1e-6, "fractional secs preserved");
        assert_eq!(whole, 1234, "whole secs == floor(fractional)");
        assert_eq!(
            whole,
            frac.floor() as u64,
            "u64 helper == floor(f64 helper)"
        );
    }

    #[test]
    fn epoch_conversion_at_epoch_is_zero() {
        let (frac, whole) = epoch_conversion(UNIX_EPOCH);
        assert_eq!(frac, 0.0);
        assert_eq!(whole, 0);
    }

    #[test]
    fn epoch_conversion_clamps_pre_epoch_to_zero() {
        // A clock set before the epoch makes `duration_since` error; the
        // panic-free source clamps to a zero duration => 0.0 / 0, never panics.
        let clock = UNIX_EPOCH - std::time::Duration::from_secs(5);
        let (frac, whole) = epoch_conversion(clock);
        assert_eq!(frac, 0.0, "pre-epoch clamps fractional to 0.0");
        assert_eq!(whole, 0, "pre-epoch clamps whole to 0");
    }

    #[test]
    fn real_clock_helpers_are_panic_free_and_non_negative() {
        // Smoke check that both production helpers run without panicking on the
        // live clock and return non-negative values from the shared source.
        let f = now_unix_secs();
        let u = now_unix_secs_u64();
        assert!(f.is_finite() && f >= 0.0, "fractional secs sane: {f}");
        // u is u64 so inherently >= 0; assert it is in the same era as the float
        // (within one second, allowing for the two separate clock reads).
        let diff = (f.floor() as i128 - u as i128).abs();
        assert!(diff <= 1, "integer and fractional helpers agree within 1s");
    }

    #[test]
    fn image_params_validate_rejects_out_of_bounds() {
        let bad_steps = ImageParams {
            steps: 0,
            ..Default::default()
        };
        assert!(matches!(
            bad_steps.validate(),
            Err(ImageParamsError::Steps { .. })
        ));

        let too_many_steps = ImageParams {
            steps: ImageParams::MAX_STEPS + 1,
            ..Default::default()
        };
        assert!(matches!(
            too_many_steps.validate(),
            Err(ImageParamsError::Steps { .. })
        ));

        let zero_height = ImageParams {
            height: 0,
            ..Default::default()
        };
        assert!(matches!(
            zero_height.validate(),
            Err(ImageParamsError::Height { .. })
        ));

        let unaligned_width = ImageParams {
            width: 513, // not a multiple of DIM_MULTIPLE
            ..Default::default()
        };
        assert!(matches!(
            unaligned_width.validate(),
            Err(ImageParamsError::Width { .. })
        ));

        let huge_dim = ImageParams {
            width: ImageParams::MAX_DIM + ImageParams::DIM_MULTIPLE,
            ..Default::default()
        };
        assert!(matches!(
            huge_dim.validate(),
            Err(ImageParamsError::Width { .. })
        ));

        let nan_scale = ImageParams {
            cfg_text_scale: f32::NAN,
            ..Default::default()
        };
        assert!(matches!(
            nan_scale.validate(),
            Err(ImageParamsError::CfgScale { .. })
        ));

        let neg_scale = ImageParams {
            cfg_img_scale: -1.0,
            ..Default::default()
        };
        assert!(matches!(
            neg_scale.validate(),
            Err(ImageParamsError::CfgScale { .. })
        ));

        let zero_images = ImageParams {
            max_images: 0,
            ..Default::default()
        };
        assert!(matches!(
            zero_images.validate(),
            Err(ImageParamsError::MaxImages { .. })
        ));
    }

    #[test]
    fn model_dtype_uses_canonical_strings() {
        assert_eq!(
            serde_json::to_value(ModelDtype::Float16).unwrap(),
            serde_json::json!("float16")
        );
        assert_eq!(ModelDtype::parse("bfloat16"), Some(ModelDtype::BFloat16));
        assert_eq!(ModelDtype::parse("float32"), Some(ModelDtype::Float32));
    }
}
