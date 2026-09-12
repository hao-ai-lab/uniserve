//! Foundational request values, identifiers, clocks, and sampling primitives.
//!
//! The crate contains transport-independent contracts shared by the server,
//! scheduler, worker protocol, and simulation runtime.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod parallel;
pub use parallel::{
    ComponentDistribution, EntryConfig, ParallelConfig, ParallelConfigError, SequenceParallel,
};

use std::sync::{Arc, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Serializable statistics and observer-facing wire values.
pub mod codec;
mod events;
/// Multimodal generation descriptors, resource bounds, and validation.
pub mod generation;
/// Counter-based Philox random-number generation.
pub mod philox;
/// Compact binary payloads carried by product values.
pub mod product_blob;
/// Deterministic token sampling and log-probability scoring.
pub mod sampling;
pub use codec::stats::WorkerForwardStats;
pub use events::{
    ArtifactEvent, ArtifactHandle, DiffusionRequest, DiffusionRequestError, Event, FinishReason,
    ImageReference, MediaGeometry, MediaKind, PositionLogprobs, Request, RuntimeFamily, StopReason,
    TokenLogprob,
};
pub use generation::{
    ContextSegment, FeedbackNextToken, FeedbackSource, GenOnlyStartPolicyDescriptor,
    GeneratedImageFeedbackRecipe, GenerationBehaviorDescriptor, GenerationCachePolicyDescriptor,
    GenerationConstraint, GenerationConstraintParseError, GenerationFeatures, GenerationLimits,
    GenerationPolicyDescriptor, GenerationRequest, GenerationRequestError,
    GenerationResourceBounds, GenerationResourceError, GenerationResources, ImageIngestRecipe,
    ImageIngestStep, ImageKvEffect, ImageSegment, SegmentPosition, TerminationPolicyDescriptor,
    TriggerPolicyDescriptor, UndTokenAction, UndVisibility, VisibilityPolicyDescriptor,
    encoder_cache_key,
};
pub use sampling::{SampleOutput, score_token_logprobs, try_apply_sampling_counts};

/// Thread-safe notification used to wake an engine after command enqueue.
pub trait Wake: Send + Sync {
    /// Signals the waiting engine owner.
    fn wake(&self);
}

impl<F> Wake for F
where
    F: Fn() + Send + Sync,
{
    /// Invokes the closure as a wake notification.
    fn wake(&self) {
        self()
    }
}

/// Cloneable command wake backed by a live notifier or a no-op value.
#[derive(Clone, Default)]
pub enum CommandWaker {
    /// Performs no notification.
    #[default]
    Noop,
    /// Delegates notification to a shared [`Wake`] implementation.
    Live(Arc<dyn Wake>),
}

impl CommandWaker {
    /// Returns a waker that performs no notification.
    pub fn noop() -> Self {
        Self::Noop
    }

    /// Wraps a thread-safe wake closure.
    pub fn new(wake: impl Fn() + Send + Sync + 'static) -> Self {
        Self::Live(Arc::new(wake))
    }

    /// Signals the live notifier, or returns immediately for [`Self::Noop`].
    pub fn wake(&self) {
        if let Self::Live(waker) = self {
            waker.wake();
        }
    }

    /// Returns whether this waker performs no notification.
    pub fn is_noop(&self) -> bool {
        matches!(self, Self::Noop)
    }
}

impl std::fmt::Debug for CommandWaker {
    /// Formats the waker by variant without exposing the live closure.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("CommandWaker")
            .field(&if self.is_noop() { "Noop" } else { "Live" })
            .finish()
    }
}

/// Returns wall-clock time in fractional seconds since the Unix epoch.
///
/// Times before the epoch clamp to zero.
pub fn now_unix_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

/// Returns wall-clock time in whole seconds since the Unix epoch.
///
/// Times before the epoch clamp to zero.
pub fn now_unix_secs_u64() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Returns process-local monotonic time in fractional seconds from a stable epoch.
///
/// Differences between values remain valid across wall-clock adjustments.
pub fn now_monotonic_secs() -> f64 {
    EPOCH.get_or_init(Instant::now).elapsed().as_secs_f64()
}

/// Returns process-local monotonic time in whole microseconds from the epoch of
/// [`now_monotonic_secs`]. Used for lifecycle phase stamps, whose differences
/// stay valid across wall-clock adjustments.
pub fn now_monotonic_us() -> u64 {
    EPOCH.get_or_init(Instant::now).elapsed().as_micros() as u64
}

static EPOCH: OnceLock<Instant> = OnceLock::new();

/// Logical KV block id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct BlockId(pub u32);

/// Request identity within one engine authority.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize,
)]
pub struct RequestId(pub u64);

/// Correlation identity for end-to-end lifecycle reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct TraceId(pub u64);

/// Operation identity within a request program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct OpId(pub u64);

/// Understanding and generation branches of a multimodal model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    /// Understanding or text branch.
    Und,
    /// Image-generation or latent branch.
    Gen,
}

/// Effective model dtype reported after runtime configuration resolves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelDtype {
    /// IEEE 754 half precision.
    #[serde(rename = "float16")]
    Float16,
    /// Brain floating-point half precision.
    #[serde(rename = "bfloat16")]
    BFloat16,
    /// IEEE 754 single precision.
    #[serde(rename = "float32")]
    Float32,
}

impl ModelDtype {
    /// Returns the stable wire name for this dtype.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Float16 => "float16",
            Self::BFloat16 => "bfloat16",
            Self::Float32 => "float32",
        }
    }

    /// Parses a supported wire name without allocating an error.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "float16" => Some(Self::Float16),
            "bfloat16" => Some(Self::BFloat16),
            "float32" => Some(Self::Float32),
            _ => None,
        }
    }
}

impl std::str::FromStr for ModelDtype {
    type Err = ModelDtypeParseError;

    /// Parses a stable model-dtype wire name.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value).ok_or_else(|| ModelDtypeParseError(value.to_owned()))
    }
}

impl std::fmt::Display for ModelDtype {
    /// Writes the stable model-dtype wire name.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Error returned for an unsupported model dtype string.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unsupported model dtype {0:?}")]
pub struct ModelDtypeParseError(String);

/// Storage dtype used by paged key/value cache tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvCacheDtype {
    /// IEEE 754 half precision.
    #[serde(rename = "float16")]
    Float16,
    /// Brain floating-point half precision.
    #[serde(rename = "bfloat16")]
    BFloat16,
    /// IEEE 754 single precision.
    #[serde(rename = "float32")]
    Float32,
    /// Finite-only E4M3 8-bit floating point.
    #[serde(rename = "float8_e4m3fn")]
    Float8E4m3Fn,
}

impl KvCacheDtype {
    /// Returns the stable wire name for this dtype.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Float16 => "float16",
            Self::BFloat16 => "bfloat16",
            Self::Float32 => "float32",
            Self::Float8E4m3Fn => "float8_e4m3fn",
        }
    }
}

impl std::str::FromStr for KvCacheDtype {
    type Err = KvCacheDtypeParseError;

    /// Parses a stable KV-cache dtype wire name.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "float16" => Ok(Self::Float16),
            "bfloat16" => Ok(Self::BFloat16),
            "float32" => Ok(Self::Float32),
            "float8_e4m3fn" => Ok(Self::Float8E4m3Fn),
            _ => Err(KvCacheDtypeParseError(value.to_owned())),
        }
    }
}

impl std::fmt::Display for KvCacheDtype {
    /// Writes the stable KV-cache dtype wire name.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Error returned for an unsupported KV-cache dtype string.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unsupported KV-cache dtype {0:?}")]
pub struct KvCacheDtypeParseError(String);

/// Text sampling parameters.
///
/// Worker-side math (temperature, top_k, top_p, min_p, penalties, logit_bias)
/// operates on the logits tensor inside the worker; control-flow floors
/// (`min_tokens`, `ignore_eos`) are enforced on the host. Fields are scalars or
/// small id/weight lists — never tensors — so they can cross the worker boundary
/// without carrying device state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Softmax temperature; zero selects greedy decoding.
    pub temperature: f32,
    /// Maximum number of highest-logit candidates; zero disables the cutoff.
    pub top_k: u32,
    /// Cumulative probability mass retained by nucleus sampling.
    pub top_p: f32,
    /// Whether EOS tokens remain eligible after the minimum-token floor.
    pub ignore_eos: bool,
    /// Optional deterministic random seed.
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

/// Returns the neutral multiplicative repetition penalty.
fn default_repetition_penalty() -> f32 {
    1.0
}

/// Returns the typical-sampling cutoff that retains the full distribution.
fn default_typical_p() -> f32 {
    1.0
}

/// Validation failure returned by [`SamplingParams::validate`].
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SamplingParamsError {
    /// A floating-point parameter is NaN or infinite.
    #[error("{field} must be finite, got {got}")]
    NonFinite {
        /// Name of the invalid parameter.
        field: &'static str,
        /// Supplied non-finite value.
        got: f32,
    },
    /// Temperature is below zero.
    #[error("temperature must be non-negative, got {got}")]
    NegativeTemperature {
        /// Supplied temperature.
        got: f32,
    },
    /// Nucleus probability lies outside `(0, 1]`.
    #[error("top_p must be in (0, 1], got {got}")]
    TopP {
        /// Supplied probability.
        got: f32,
    },
    /// Minimum relative probability lies outside `[0, 1]`.
    #[error("min_p must be in [0, 1], got {got}")]
    MinP {
        /// Supplied probability.
        got: f32,
    },
    /// Repetition penalty is not positive.
    #[error("repetition_penalty must be positive, got {got}")]
    RepetitionPenalty {
        /// Supplied penalty.
        got: f32,
    },
    /// An explicit allowed-token set is empty.
    #[error("allowed_token_ids must contain at least one token when present")]
    EmptyAllowedTokenIds,
    /// A bad-word sequence is empty.
    #[error("bad_words_ids[{index}] must contain at least one token")]
    EmptyBadWord {
        /// Index of the empty sequence.
        index: usize,
    },
}

impl SamplingParams {
    /// Returns whether generated-token logprobs are enabled.
    pub fn generated_logprobs_requested(&self) -> bool {
        self.return_logprobs || self.n_logprobs > 0 || !self.logprob_token_ids.is_empty()
    }

    /// Returns whether prompt-token logprobs are enabled.
    pub fn prompt_logprobs_requested(&self) -> bool {
        self.return_prompt_logprobs || self.n_prompt_logprobs > 0
    }

    /// Validates sampling math inputs before they reach a worker or simulator.
    pub fn validate(&self) -> Result<(), SamplingParamsError> {
        // Reject non-finite scalar math inputs before applying their individual
        // range constraints.
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

        // Bias values participate directly in logit arithmetic and follow the
        // same finite-value contract as scalar processors.
        for &(_, bias) in &self.logit_bias {
            if !bias.is_finite() {
                return Err(SamplingParamsError::NonFinite {
                    field: "logit_bias",
                    got: bias,
                });
            }
        }

        // Enforce the mathematical domains of each sampling transform.
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

        // Empty vocabulary constraints cannot express a valid sampling set.
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
    /// Defaults direct construction to greedy sampling.
    ///
    /// Request lowering assigns `1.0` when an API caller omits temperature, so
    /// the zero value applies only when code constructs these parameters directly.
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

/// Image-generation parameters bounded for scheduler admission.
///
/// `max_images` limits the total media work and feedback resources reachable by
/// one request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageParams {
    /// Number of denoising steps.
    pub steps: u16,
    /// Classifier-free guidance scale for text conditioning.
    pub cfg_text_scale: f32,
    /// Classifier-free guidance scale for image conditioning.
    pub cfg_img_scale: f32,
    /// Guidance renormalization strategy.
    pub cfg_renorm_type: CfgRenorm,
    /// Lower bound applied by guidance renormalization.
    pub cfg_renorm_min: f32,
    /// Inclusive normalized step interval where guidance is active.
    pub cfg_interval: (f32, f32),
    /// Shift applied to the diffusion timestep schedule.
    #[serde(default = "default_timestep_shift")]
    pub timestep_shift: f32,
    /// Output height in pixels.
    pub height: u32,
    /// Output width in pixels.
    pub width: u32,
    /// Optional deterministic diffusion seed.
    pub seed: Option<u64>,
    /// Negative text conditioning prompt.
    pub negative_prompt: String,
    /// Maximum images this request may generate.
    pub max_images: u16,
    /// Optional per-image prompt overrides.
    #[serde(default)]
    pub image_prompts: Vec<String>,
    /// Whether completed image bytes remain available to the caller.
    #[serde(default = "default_retain_images")]
    pub retain_images: bool,
}

impl ImageParams {
    /// Returns the classifier-free-guidance branch count implied by both axes.
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

/// Returns whether two guidance scales are equal within relative tolerance.
fn scale_approx(left: f32, right: f32) -> bool {
    (left - right).abs() <= 1.0e-6_f32 * left.abs().max(right.abs()).max(1.0)
}

/// Returns the neutral diffusion timestep shift.
fn default_timestep_shift() -> f32 {
    1.0
}

/// Returns the default policy that retains materialized image bytes.
fn default_retain_images() -> bool {
    true
}

/// Validation failure returned by [`ImageParams::validate`].
/// Each variant names the offending field and the bound it violated so the
/// caller can report a precise input error before scheduling diffusion work.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ImageParamsError {
    /// Denoising step count lies outside its supported range.
    #[error("steps must be in 1..={max}, got {got}")]
    Steps {
        /// Supplied step count.
        got: u16,
        /// Maximum supported step count.
        max: u16,
    },
    /// Image height violates range or alignment requirements.
    #[error("height must be a non-zero multiple of {multiple} in {min}..={max}, got {got}")]
    Height {
        /// Supplied height.
        got: u32,
        /// Minimum supported height.
        min: u32,
        /// Maximum supported height.
        max: u32,
        /// Required height alignment.
        multiple: u32,
    },
    /// Image width violates range or alignment requirements.
    #[error("width must be a non-zero multiple of {multiple} in {min}..={max}, got {got}")]
    Width {
        /// Supplied width.
        got: u32,
        /// Minimum supported width.
        min: u32,
        /// Maximum supported width.
        max: u32,
        /// Required width alignment.
        multiple: u32,
    },
    /// A guidance scale is non-finite or exceeds its supported range.
    #[error("{field} must be finite and in 0.0..={max}, got {got}")]
    CfgScale {
        /// Name of the invalid scale.
        field: &'static str,
        /// Supplied scale.
        got: f32,
        /// Maximum supported scale.
        max: f32,
    },
    /// Per-request image count lies outside its supported range.
    #[error("max_images must be in 1..={max}, got {got}")]
    MaxImages {
        /// Supplied image count.
        got: u16,
        /// Maximum supported image count.
        max: u16,
    },
    /// A floating-point parameter is NaN or infinite.
    #[error("{field} must be finite, got {got}")]
    NonFinite {
        /// Name of the invalid parameter.
        field: &'static str,
        /// Supplied non-finite value.
        got: f32,
    },
    /// Guidance interval endpoints are descending.
    #[error("cfg_interval must be an ordered pair, got ({lo}, {hi})")]
    CfgIntervalOrder {
        /// Supplied lower endpoint.
        lo: f32,
        /// Supplied upper endpoint.
        hi: f32,
    },
}

/// Classifier-free guidance renormalization strategy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CfgRenorm {
    /// Leaves classifier-free guidance unnormalized.
    None,
    /// Renormalizes the combined guidance prediction globally.
    #[default]
    Global,
    /// Renormalizes against the text-conditioned prediction.
    TextChannel,
}

impl CfgRenorm {
    /// Returns the stable wire name for this strategy.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Global => "global",
            Self::TextChannel => "text_channel",
        }
    }
}

impl std::str::FromStr for CfgRenorm {
    type Err = CfgRenormParseError;

    /// Parses a stable guidance-renormalization wire name.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "none" => Ok(Self::None),
            "global" => Ok(Self::Global),
            "text_channel" => Ok(Self::TextChannel),
            _ => Err(CfgRenormParseError(value.to_owned())),
        }
    }
}

impl std::fmt::Display for CfgRenorm {
    /// Writes the stable guidance-renormalization wire name.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Error returned for an unsupported guidance renormalization string.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unsupported CFG renormalization mode {0:?}")]
pub struct CfgRenormParseError(String);

impl ImageParams {
    /// Upper bound on diffusion steps.
    pub const MAX_STEPS: u16 = 1000;
    /// Pixel-dimension bounds. `height`/`width` must be non-zero, within
    /// `[MIN_DIM, MAX_DIM]`, and a multiple of `DIM_MULTIPLE`.
    pub const MIN_DIM: u32 = 16;
    /// Maximum supported image dimension in pixels.
    pub const MAX_DIM: u32 = 4096;
    /// Required divisibility for each image dimension.
    pub const DIM_MULTIPLE: u32 = 16;
    /// Upper bound on any single CFG scale.
    pub const MAX_CFG_SCALE: f32 = 100.0;
    /// Upper bound on `max_images` per request.
    pub const MAX_IMAGES: u16 = 256;

    /// Validates diffusion parameters before they reach the worker.
    pub fn validate(&self) -> Result<(), ImageParamsError> {
        // Bound discrete work before validating shape- and guidance-dependent
        // values.
        if self.steps == 0 || self.steps > Self::MAX_STEPS {
            return Err(ImageParamsError::Steps {
                got: self.steps,
                max: Self::MAX_STEPS,
            });
        }

        // Both dimensions obey the same positive, bounded, aligned grid
        // contract.
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

        // Each guidance axis accepts finite non-negative scaling up to the
        // runtime-independent request bound.
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

        // Remaining continuous parameters must be finite before interval order
        // is evaluated.
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
    /// Returns a bounded single-image diffusion configuration.
    fn default() -> Self {
        Self {
            steps: 50,
            cfg_text_scale: 4.0,
            cfg_img_scale: 1.0,
            cfg_renorm_type: CfgRenorm::Global,
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

/// Classifier-free-guidance configuration attached to image-generation operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfgParams {
    /// Number of active guidance branches in `1..=3`.
    pub branch_count: u8,
    /// Text-conditioning guidance scale.
    pub text_scale: f32,
    /// Image-conditioning guidance scale.
    pub img_scale: f32,
    /// Stable renormalization strategy name.
    pub renorm_type: String,
    /// Minimum renormalization factor.
    pub renorm_min: f32,
    /// Inclusive normalized denoising interval where guidance is active.
    #[serde(default)]
    pub interval: (f32, f32),
}

/// Physical layout of a page-first key/value cache buffer.
///
/// Elements are addressed by `base + block_id * page_stride + layer * layer_stride`.
/// The descriptor supports admission and sizing without transferring ownership
/// of the backing buffer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct KvLayout {
    /// Number of physical KV pages.
    pub num_blocks: u32,
    /// Tokens stored in each physical page.
    pub block_size: u32,
    /// Transformer layers represented in the cache.
    pub num_layers: u32,
    /// KV heads stored per layer.
    pub num_kv_heads: u32,
    /// Elements stored per head.
    pub head_dim: u32,
    /// Storage bytes for each KV element.
    pub dtype_bytes: u32,
}

impl KvLayout {
    /// Returns storage consumed by one token across all layers and heads.
    pub fn bytes_per_token(&self) -> u64 {
        // Account for both key and value tensors at every layer.
        2 * self.num_kv_heads as u64
            * self.head_dim as u64
            * self.num_layers as u64
            * self.dtype_bytes as u64
    }

    /// Returns storage consumed by the complete KV layout.
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
    SlidingWindow {
        /// Number of recent tokens retained for attention.
        window: u32,
        /// Number of prefix tokens retained outside the window.
        sink: u32,
    },
}

/// One positional KV-cache group reported by the worker at handshake.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCacheGroup {
    /// Number of physical pages in this group's subspace.
    pub num_blocks: u32,
    /// Attention retention policy for the group.
    #[serde(default)]
    pub kind: KvGroupKind,
}

/// Algorithm used to derive prefix-cache block keys.
///
/// FNV-1a provides a seeded non-cryptographic mixer, while SHA-256 provides a
/// cryptographic digest truncated to the cache's 64-bit key space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum HashAlgo {
    /// Seeded 64-bit FNV-1a hashing.
    #[default]
    Fnv1a,
    /// SHA-256 truncated to the cache's 64-bit key space.
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

    /// Converts an injected clock to fractional and whole Unix seconds.
    ///
    /// Instants before the epoch clamp to zero.
    fn epoch_conversion(clock: SystemTime) -> (f64, u64) {
        let d = clock.duration_since(UNIX_EPOCH).unwrap_or_default();
        (d.as_secs_f64(), d.as_secs())
    }

    #[test]
    fn epoch_conversion_u64_is_floor_of_f64() {
        // Whole seconds are the floor of the fractional representation for an
        // instant 1,234.75 seconds after the epoch.
        let clock = UNIX_EPOCH + std::time::Duration::from_millis(1_234_750);
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
        // `duration_since` rejects pre-epoch values, and the public clock
        // helpers map that error to a zero duration.
        let clock = UNIX_EPOCH - std::time::Duration::from_secs(5);
        let (frac, whole) = epoch_conversion(clock);
        assert_eq!(frac, 0.0, "pre-epoch clamps fractional to 0.0");
        assert_eq!(whole, 0, "pre-epoch clamps whole to 0");
    }

    #[test]
    fn real_clock_helpers_are_panic_free_and_non_negative() {
        // Both clock views use the same non-negative epoch conversion.
        let f = now_unix_secs();
        let u = now_unix_secs_u64();
        assert!(f.is_finite() && f >= 0.0, "fractional secs sane: {f}");
        // Separate clock reads may cross one whole-second boundary.
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
