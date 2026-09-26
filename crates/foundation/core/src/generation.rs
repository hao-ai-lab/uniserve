//! Tokenized generation requests, image configuration, and capacity calculations.
//!
//! Requests own their input data and effective parameters. The engine owns
//! mutable computation progress, output delivery, and physical allocations.
//!
//! "Und" names the autoregressive understanding branch that decodes text, and
//! "Gen" names the diffusion branch that generates images. The server's model
//! preprocessing builds a [`GenerationRequest`] and sizes it with the capacity
//! helpers here against the engine's resolved limits before submission. The
//! engine's admission (`Scheduler::enqueue`) runs
//! [`GenerationRequest::validate`], checks
//! [`ImageGenerationConfig::required_features`] with
//! [`GenerationLimits::covers`], and then runs
//! [`GenerationRequest::validate_resources`] and
//! [`GenerationRequest::max_kv_tokens`] against the loaded limits.

use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::{ImageParams, ImageParamsError, RequestId, SamplingParams, SamplingParamsError};

/// Output constraint applied to the default generation paradigm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationConstraint {
    /// Both Und and Gen branches are enabled when the model description supports them.
    #[default]
    Default,
    /// Only the Und branch may produce user-visible output.
    UndOnly,
    /// Only the Gen branch may produce user-visible output. Und tokens may still
    /// be generated internally when the model description requires them for control.
    GenOnly,
}

impl GenerationConstraint {
    /// Returns the stable snake-case wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::UndOnly => "und_only",
            Self::GenOnly => "gen_only",
        }
    }
}

impl FromStr for GenerationConstraint {
    type Err = GenerationConstraintParseError;

    /// Parses a stable generation-constraint wire name.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "default" => Ok(Self::Default),
            "und_only" => Ok(Self::UndOnly),
            "gen_only" => Ok(Self::GenOnly),
            other => Err(GenerationConstraintParseError {
                value: other.to_string(),
            }),
        }
    }
}

/// A rejected [`GenerationConstraint`] string.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unsupported generation constraint: {value}")]
pub struct GenerationConstraintParseError {
    /// Unsupported wire value.
    pub value: String,
}

/// Positioned media consumed alongside the already-tokenized positive prompt.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MultimodalInputs {
    /// Input images in nondecreasing prompt-token position order, as
    /// `GenerationRequest::validate` requires.
    pub images: Vec<ImageInput>,
}

/// One token canvas a readout request denoises once over its prompt.
///
/// Canvas token `j` sits at the model position that follows the prompt's
/// logical positions by `j`. Every canvas token attends to the whole prompt
/// and to every token of its own row, and the pass writes no KV. The rows of
/// one request are independent computations over the same prompt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadoutRow {
    /// Canvas token ids.
    pub token_ids: Vec<u32>,
    /// Positions whose candidate log-probabilities the row reports, in
    /// increasing position order, which is also report order.
    pub slots: Vec<ReadoutSlot>,
}

/// One canvas position whose candidate log-probabilities a readout reports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadoutSlot {
    /// Index of the slot's token within its row.
    pub position: u32,
    /// Token ids whose natural-log probabilities, under the log-softmax over
    /// the full vocabulary of the model's logits at `position`, are reported
    /// in this order.
    pub candidates: Vec<u32>,
}

/// Block-diffusion sampling of a generating request.
///
/// Text is generated in blocks of `canvas_length` tokens. A block starts as
/// uniformly random tokens and is denoised for at most `max_steps` steps: a
/// step samples every position from the model's logits at a temperature that
/// falls linearly from `t_max` to `t_min`, accepts the lowest-entropy samples
/// within `entropy_bound` and renoises the others. A block stops early once
/// its argmax canvas has held for `stability_threshold` steps and its mean
/// entropy is below `confidence_threshold`. The block's argmax canvas is then
/// committed to the request's context as causal prompt, and the next block
/// follows it. Every draw follows the request's `SamplingParams::seed`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CanvasSampling {
    /// Tokens of one block.
    pub canvas_length: u32,
    /// Denoising steps a block runs at most.
    pub max_steps: u32,
    /// Entropy, in nats, the accepted samples of one step may carry beyond
    /// the largest one.
    pub entropy_bound: f32,
    /// Sampling temperature of a block's last step.
    pub t_min: f32,
    /// Sampling temperature of a block's first step.
    pub t_max: f32,
    /// Mean canvas entropy, in nats, below which a stable block stops.
    pub confidence_threshold: f32,
    /// Steps the argmax canvas must hold unchanged before a block stops;
    /// zero stops a block on confidence alone.
    pub stability_threshold: u32,
}

impl CanvasSampling {
    /// Returns the first parameter the scheduler cannot plan with, if any:
    /// an empty canvas or step limit, or a non-finite sampling value. The
    /// worker's sampler checks the sampling values' domain itself.
    fn invalid_parameter(&self) -> Option<&'static str> {
        if self.canvas_length == 0 {
            Some("canvas_length")
        } else if self.max_steps == 0 {
            Some("max_steps")
        } else if !self.entropy_bound.is_finite() {
            Some("entropy_bound")
        } else if !self.t_min.is_finite() {
            Some("t_min")
        } else if !self.t_max.is_finite() {
            Some("t_max")
        } else if !self.confidence_threshold.is_finite() {
            Some("confidence_threshold")
        } else {
            None
        }
    }
}

/// An encoded image and the model's requirements for adding it to context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageInput {
    /// Stable content hash used for encoder-cache identity; see
    /// [`encoder_cache_key`]. The engine derives encoder-cache keys from this
    /// value and the encoder's index and kind only, so omni preprocessing in
    /// the server folds `CachePolicy::isolation_key` into it to keep isolated
    /// requests from sharing entries, and folds in any request-dependent
    /// preprocessing parameter that changes the encoder product.
    pub hash: u64,
    /// Base64-encoded input image.
    pub b64: String,
    /// Exclusive prompt-token position at the end of this image's marker gap.
    /// Equal positions preserve input order, including multiple images at one marker.
    pub position: u32,
    /// Logical positions contributed after all encoders finish; independent of KV length.
    pub num_positions: u32,
    /// Required encoder stages in context write order, with their KV contributions.
    pub encoders: Vec<ImageEncoderInput>,
}

/// One encoder input and its physical KV contribution.
///
/// `num_kv_tokens` is exact when known. Otherwise `max_kv_tokens` bounds
/// the worker-selected length; absent limits use the loaded encoder capacity
/// (see [`ImageEncoderInput::kv_token_capacity`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageEncoderInput {
    /// Model encoder consuming this image. List order determines context write order.
    pub encoder: ImageIngestStep,
    /// Exact KV tokens this encoder writes, when known before execution.
    pub num_kv_tokens: Option<u32>,
    /// Upper bound on a worker-selected KV length. When both it and
    /// `num_kv_tokens` are present, `GenerationRequest::validate` requires
    /// `num_kv_tokens <= max_kv_tokens`.
    pub max_kv_tokens: Option<u32>,
}

/// One image ingest worker step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageIngestStep {
    /// Encodes image pixels into a latent representation.
    VaeEncode,
    /// Encodes image content into vision features.
    VitEncode,
}

/// Derives the cache identity of one ordered image encoder.
///
/// `step_index` is the encoder's index in `ImageInput::encoders`. The engine
/// derives the key when it looks up a cached product before encoding and
/// again when it stores the encoder's product, so both must use the same
/// index for a stored product to be found.
pub fn encoder_cache_key(image_hash: u64, step_index: usize, step: ImageIngestStep) -> u64 {
    // Apply FNV-1a to the complete domain tuple in a fixed byte order. Including
    // the stage index distinguishes repeated encoder kinds within one image input.
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in image_hash
        .to_le_bytes()
        .into_iter()
        .chain((step_index as u64).to_le_bytes())
        .chain([match step {
            ImageIngestStep::VaeEncode => 1,
            ImageIngestStep::VitEncode => 2,
        }])
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Product channel through which a materialized image reaches feedback encode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum FeedbackSource {
    /// Uses the device-resident image product directly.
    DeviceProduct,
    /// Uses the materialized image artifact.
    ArtifactProduct,
}

/// Und token used to continue after generated-image feedback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum FeedbackNextToken {
    /// Declares no automatic continuation token.
    None,
    /// Continues with the model's beginning-of-sequence token.
    Bos,
    /// Continues with the model's end-of-image token.
    EndOfImage,
    /// Continues with an explicit vocabulary token.
    Token {
        /// Vocabulary token identity.
        token_id: u32,
    },
}

/// Token conditions for switching text decoding to image generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ImageTrigger {
    /// Disables image-branch triggering.
    Disabled,
    /// Opens the branch after one exact token.
    Token {
        /// Triggering vocabulary token.
        token_id: u32,
    },
    /// Opens the branch after an exact generated suffix.
    Suffix {
        /// Triggering token sequence.
        token_ids: Vec<u32>,
    },
    /// Opens the branch when a closing round ends in a trigger suffix.
    RoundCloseThenSuffix {
        /// Tokens that close an understanding round.
        close_token_ids: Vec<u32>,
        /// Required suffix immediately before round closure.
        trigger_token_ids: Vec<u32>,
    },
}

impl ImageTrigger {
    /// Matches a branch-opening trigger during ordinary Und generation.
    pub fn matches_generated(&self, generated: &[u32]) -> bool {
        match self {
            Self::Token { token_id } => generated.last() == Some(token_id),
            Self::Suffix { token_ids } => !token_ids.is_empty() && generated.ends_with(token_ids),
            Self::Disabled | Self::RoundCloseThenSuffix { .. } => false,
        }
    }

    /// Matches a branch-opening suffix when the current Und round closes.
    pub fn matches_round_close(&self, round: &[u32], close_token_id: u32) -> bool {
        match self {
            Self::RoundCloseThenSuffix {
                close_token_ids,
                trigger_token_ids,
            } => {
                close_token_ids.contains(&close_token_id)
                    && !trigger_token_ids.is_empty()
                    && round.ends_with(trigger_token_ids)
            }
            _ => false,
        }
    }

    /// Returns the single token that directly opens a branch, when present.
    pub fn direct_token(&self) -> Option<u32> {
        match self {
            Self::Token { token_id } => Some(*token_id),
            Self::Disabled | Self::Suffix { .. } | Self::RoundCloseThenSuffix { .. } => None,
        }
    }

    /// Returns the token sequence that opens a branch during ordinary decode.
    pub fn generated_suffix(&self) -> Option<&[u32]> {
        match self {
            Self::Token { token_id } => Some(std::slice::from_ref(token_id)),
            Self::Suffix { token_ids } => Some(token_ids),
            Self::Disabled | Self::RoundCloseThenSuffix { .. } => None,
        }
    }

    /// Returns whether the trigger is evaluated when an understanding round closes.
    pub fn requires_round_close(&self) -> bool {
        matches!(self, Self::RoundCloseThenSuffix { .. })
    }

    /// Returns token identifiers that close a trigger-evaluated round.
    pub fn round_close_token_ids(&self) -> &[u32] {
        match self {
            Self::RoundCloseThenSuffix {
                close_token_ids, ..
            } => close_token_ids,
            _ => &[],
        }
    }
}

/// Model-specific conditions for starting image generation and encoding its result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageGenerationConfig {
    /// Tokens or suffixes that switch text decoding to image generation.
    pub trigger: ImageTrigger,
    /// Whether image-only requests generate internal text before their first image.
    /// When false, image computation starts after prompt and input-image encoding.
    pub requires_text_for_image: bool,
    /// Product channel through which a completed image reaches feedback
    /// encoding. With `None`, completed images do not reenter the context, and
    /// `GenerationRequest::validate` rejects requests whose text continues
    /// after an image.
    pub feedback_source: Option<FeedbackSource>,
    /// Encoder order and KV contributions of a completed image entering context.
    pub feedback_encoders: Vec<ImageEncoderInput>,
    /// Logical positions added after the final feedback encoder.
    pub num_feedback_positions: u32,
    /// Continuation input used when feedback does not sample a token.
    pub feedback_next_token: FeedbackNextToken,
    /// Whether the final feedback KV write samples the next text token.
    pub sample_feedback_continuation: bool,
}

impl ImageGenerationConfig {
    /// Returns whether the requested output can reach an image-generation branch.
    fn generates_images(&self, constraint: GenerationConstraint) -> bool {
        // The Gen branch opens through the trigger, or, for an image-only
        // request that needs no text first, directly after the prompt.
        constraint != GenerationConstraint::UndOnly
            && (!matches!(self.trigger, ImageTrigger::Disabled)
                || (constraint == GenerationConstraint::GenOnly && !self.requires_text_for_image))
    }

    /// Returns whether completed images reenter the context for continued text.
    fn feeds_back_images(&self, constraint: GenerationConstraint) -> bool {
        self.generates_images(constraint)
            && constraint == GenerationConstraint::Default
            && self.feedback_source.is_some()
    }

    /// Returns the runtime features required by the reachable generation graph.
    pub fn required_features(
        &self,
        constraint: GenerationConstraint,
        context_image_steps: impl IntoIterator<Item = ImageIngestStep>,
    ) -> GenerationFeatures {
        // Understanding execution is the common control path for every request.
        let mut needs = GenerationFeatures::UNDERSTANDING;

        // Context images require the encoder stages declared by their inputs.
        for step in context_image_steps {
            needs.insert(step.required_feature());
        }

        if self.generates_images(constraint) {
            needs.insert(GenerationFeatures::IMAGE_GENERATION);
        }

        // Feedback can make additional encoder stages reachable after image
        // materialization.
        if self.feeds_back_images(constraint) {
            for step in self.feedback_encoders.iter().map(|input| input.encoder) {
                needs.insert(step.required_feature());
            }
        }
        needs
    }
}

impl ImageIngestStep {
    /// Returns the runtime feature that executes this encoder stage.
    const fn required_feature(self) -> GenerationFeatures {
        match self {
            Self::VaeEncode => GenerationFeatures::LATENT_ENCODE,
            Self::VitEncode => GenerationFeatures::VISION_ENCODE,
        }
    }
}

bitflags::bitflags! {
    /// Generation branches present in a runtime or required by one request.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
    pub struct GenerationFeatures: u8 {
        /// Autoregressive understanding execution.
        const UNDERSTANDING = 1 << 0;
        /// Vision feature encoding.
        const VISION_ENCODE = 1 << 1;
        /// Variational-autoencoder latent encoding.
        const LATENT_ENCODE = 1 << 2;
        /// Diffusion denoising and image materialization.
        const IMAGE_GENERATION = 1 << 3;
        /// Denoising passes of token canvases that read a prompt's KV cache
        /// without writing it.
        const TOKEN_DENOISING = 1 << 4;
    }
}

impl GenerationFeatures {
    /// Returns the admission diagnostic name of one represented feature.
    ///
    /// Features are checked in the order understanding, latent encode, vision
    /// encode, image generation, token denoising; an empty set yields
    /// `runtime_generation_features`.
    pub fn name(self) -> &'static str {
        if self.contains(Self::UNDERSTANDING) {
            "runtime_und_execution"
        } else if self.contains(Self::LATENT_ENCODE) {
            "runtime_vae_encode"
        } else if self.contains(Self::VISION_ENCODE) {
            "runtime_vit_encode"
        } else if self.contains(Self::IMAGE_GENERATION) {
            "runtime_gen_denoise"
        } else if self.contains(Self::TOKEN_DENOISING) {
            "runtime_token_denoise"
        } else {
            "runtime_generation_features"
        }
    }
}

impl std::fmt::Display for GenerationFeatures {
    /// Writes the diagnostic name chosen by [`GenerationFeatures::name`].
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

/// Worker and scheduler limits needed to compile a bounded generation graph.
///
/// The server supplies the loaded model's requirements, and the engine's
/// `resolve_generation_limits` intersects them with the capacities the worker
/// reports. Zero in any numeric field other than `commit_marker_tokens`
/// declares no capacity for that resource;
/// [`GenerationRequest::validate_resources`] rejects requests that need it.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct GenerationLimits {
    /// Generation features implemented by the runtime.
    pub features: GenerationFeatures,
    /// Maximum latent grid units of one generated image, where one unit is one
    /// `latent_downsample` x `latent_downsample` pixel block.
    pub max_latent_units: u64,
    /// Pixel-to-latent spatial downsample factor.
    pub latent_downsample: u32,
    /// Maximum VAE grid tokens for a worker-selected length. It is also the
    /// divisor that turns `max_latent_feature_bytes` into bytes per latent
    /// unit in [`GenerationRequest::image_latent_bytes`].
    pub max_vae_grid_tokens: u32,
    /// Maximum vision grid tokens for a worker-selected length.
    pub max_vit_grid_tokens: u32,
    /// Maximum VAE feature product size in bytes.
    pub max_latent_feature_bytes: u64,
    /// Maximum vision feature product size in bytes.
    pub max_vision_feature_bytes: u64,
    /// Commit-marker tokens that the engine adds to an image's latent grid
    /// tokens to form each denoising call's query length.
    pub commit_marker_tokens: u32,
    /// Maximum classifier-free-guidance branches.
    pub max_cfg_branches: u32,
    /// Maximum concurrently pinned encoder-cache entries.
    pub encoder_cache_entries: u32,
}

impl GenerationLimits {
    /// Checks whether the runtime covers every feature required by a request.
    ///
    /// A failure carries the missing feature set for admission diagnostics.
    pub fn covers(&self, needs: GenerationFeatures) -> Result<(), GenerationFeatures> {
        let missing = needs.difference(self.features);
        if !missing.is_empty() {
            return Err(missing);
        }
        Ok(())
    }
}

impl GenerationRequest {
    /// Returns whether generated text belongs in the requested output modalities.
    pub fn emits_text(&self) -> bool {
        self.constraint != GenerationConstraint::GenOnly
    }

    /// Returns whether text decoding is needed, including internal image-control text.
    pub fn decodes_text(&self) -> bool {
        !self.starts_with_image()
    }

    /// Returns whether the requested modalities and model permit image output.
    pub fn generates_images(&self) -> bool {
        self.image_generation.generates_images(self.constraint)
    }

    /// Returns whether image-only generation starts immediately after the prompt.
    pub fn starts_with_image(&self) -> bool {
        self.constraint == GenerationConstraint::GenOnly
            && !self.image_generation.requires_text_for_image
    }

    /// Returns whether a generated image must be encoded into subsequent context.
    pub fn feeds_back_images(&self) -> bool {
        self.image_generation.feeds_back_images(self.constraint)
    }

    /// Returns whether text generation resumes after a completed image.
    pub fn continues_after_image(&self) -> bool {
        self.generates_images() && self.constraint == GenerationConstraint::Default
    }

    /// Returns whether a completed image finishes this request.
    pub fn finishes_after_image(&self) -> bool {
        self.generates_images() && self.constraint == GenerationConstraint::GenOnly
    }

    /// Maximum physical KV tokens required by the input, output budget, and image feedback.
    ///
    /// Sums the prompt, the `max_und_tokens` budget, each context image's
    /// encoder KV bounds, the feedback encoder bounds once per
    /// `ImageParams::max_images` when completed images feed back, and the
    /// negative prompt when the Gen branch is reachable. Saturation preserves
    /// conservative admission for host-sized token budgets.
    ///
    /// # Errors
    ///
    /// Returns `GenerationResourceError::UnboundedImageKv` when a counted
    /// encoder input resolves to a zero KV length.
    pub fn max_kv_tokens(
        &self,
        limits: &GenerationLimits,
    ) -> Result<usize, GenerationResourceError> {
        let input_images = self
            .multimodal_inputs
            .images
            .iter()
            .try_fold(0usize, |total, image| {
                Ok(total.saturating_add(encoder_kv_bound(&image.encoders, limits)?))
            })?;
        let feedback =
            if self.continues_after_image() && self.image_generation.feedback_source.is_some() {
                encoder_kv_bound(&self.image_generation.feedback_encoders, limits)?
                    .saturating_mul(self.image.max_images as usize)
            } else {
                0
            };
        Ok(self
            .prompt_token_ids
            .len()
            .saturating_add(self.max_und_tokens)
            .saturating_add(input_images)
            .saturating_add(feedback)
            .saturating_add(if self.generates_images() {
                self.negative_prompt_token_ids.len()
            } else {
                0
            }))
    }

    /// Encoder-cache entries that admission reserves for the ordered input encoders.
    /// Each ingest step retains its reservation even if another image has the same hash.
    ///
    /// Only context images count, not feedback encoders, and the count is zero
    /// when the cache policy neither reads nor writes.
    pub fn num_encoder_cache_entries(&self) -> usize {
        if self.cache.read || self.cache.write {
            self.multimodal_inputs
                .images
                .iter()
                .map(|image| image.encoders.len())
                .fold(0usize, usize::saturating_add)
        } else {
            0
        }
    }

    /// Validated latent grid size for one generated image, or zero for text-only work.
    ///
    /// # Errors
    ///
    /// Returns `MissingRuntimeBound` when `latent_downsample` or
    /// `max_latent_units` is zero, `ImageDimensionAlignment` when the image
    /// width or height is not a multiple of `latent_downsample`, and
    /// `LatentCapacity` when the grid exceeds `max_latent_units`.
    pub fn image_latent_units(
        &self,
        limits: &GenerationLimits,
    ) -> Result<u64, GenerationResourceError> {
        if !self.generates_images() {
            return Ok(0);
        }
        if limits.latent_downsample == 0 {
            return Err(GenerationResourceError::MissingRuntimeBound {
                resource: "latent_downsample",
            });
        }
        if !self.image.width.is_multiple_of(limits.latent_downsample)
            || !self.image.height.is_multiple_of(limits.latent_downsample)
        {
            return Err(GenerationResourceError::ImageDimensionAlignment {
                width: self.image.width,
                height: self.image.height,
                latent_downsample: limits.latent_downsample,
            });
        }
        if limits.max_latent_units == 0 {
            return Err(GenerationResourceError::MissingRuntimeBound {
                resource: "max_latent_units",
            });
        }
        let units = u64::from(self.image.width / limits.latent_downsample)
            .saturating_mul(u64::from(self.image.height / limits.latent_downsample));
        if units > limits.max_latent_units {
            return Err(GenerationResourceError::LatentCapacity {
                requested: units,
                available: limits.max_latent_units,
            });
        }
        Ok(units)
    }

    /// Allocation bytes for one image latent using the loaded model's byte density.
    ///
    /// The density is `max_latent_feature_bytes / max_vae_grid_tokens`,
    /// rounded up to whole bytes per latent unit.
    ///
    /// # Errors
    ///
    /// Returns every [`GenerationRequest::image_latent_units`] error, and
    /// `MissingRuntimeBound` naming `max_latent_feature_bytes` when either
    /// `max_latent_feature_bytes` or `max_vae_grid_tokens` is zero.
    pub fn image_latent_bytes(
        &self,
        limits: &GenerationLimits,
    ) -> Result<u64, GenerationResourceError> {
        if !self.generates_images() {
            return Ok(0);
        }
        let units = self.image_latent_units(limits)?;
        if limits.max_vae_grid_tokens == 0 || limits.max_latent_feature_bytes == 0 {
            return Err(GenerationResourceError::MissingRuntimeBound {
                resource: "max_latent_feature_bytes",
            });
        }
        let bytes_per_unit = limits
            .max_latent_feature_bytes
            .div_ceil(u64::from(limits.max_vae_grid_tokens));
        Ok(units.saturating_mul(bytes_per_unit))
    }

    /// Checks reachable encoder work, image geometry, and guidance against loaded capacity.
    /// Admission separately acquires KV, cache entries, and latent storage from their pools.
    ///
    /// # Errors
    ///
    /// Returns the first failure among, in check order: the
    /// [`GenerationRequest::max_kv_tokens`] error; `MissingRuntimeBound` for a
    /// reachable encoder kind whose feature-byte limit is zero;
    /// `MissingRuntimeBound` or `EncoderCacheCapacity` when reserved
    /// encoder-cache entries exceed `encoder_cache_entries`; the
    /// [`GenerationRequest::image_latent_bytes`] errors; and, when the Gen
    /// branch is reachable, `MissingRuntimeBound` or `CfgBranchCapacity` when
    /// the guidance branch count exceeds `max_cfg_branches`.
    pub fn validate_resources(
        &self,
        limits: &GenerationLimits,
    ) -> Result<(), GenerationResourceError> {
        self.max_kv_tokens(limits)?;

        // Each encoder kind reachable from context images or feedback needs
        // a nonzero feature-product size.
        let feedback = (self.continues_after_image()
            && self.image_generation.feedback_source.is_some())
        .then_some(self.image_generation.feedback_encoders.as_slice());
        for (step, bytes, resource) in [
            (
                ImageIngestStep::VaeEncode,
                limits.max_latent_feature_bytes,
                "max_latent_feature_bytes",
            ),
            (
                ImageIngestStep::VitEncode,
                limits.max_vision_feature_bytes,
                "max_vision_feature_bytes",
            ),
        ] {
            let used = self
                .multimodal_inputs
                .images
                .iter()
                .any(|image| image.encoders.iter().any(|input| input.encoder == step))
                || feedback.is_some_and(|inputs| inputs.iter().any(|input| input.encoder == step));
            if used && bytes == 0 {
                return Err(GenerationResourceError::MissingRuntimeBound { resource });
            }
        }

        let entries = self.num_encoder_cache_entries();
        if entries > 0 {
            if limits.encoder_cache_entries == 0 {
                return Err(GenerationResourceError::MissingRuntimeBound {
                    resource: "encoder_cache_entries",
                });
            }
            if entries > limits.encoder_cache_entries as usize {
                return Err(GenerationResourceError::EncoderCacheCapacity {
                    requested: entries,
                    available: limits.encoder_cache_entries,
                });
            }
        }

        self.image_latent_bytes(limits)?;

        if self.generates_images() {
            if limits.max_cfg_branches == 0 {
                return Err(GenerationResourceError::MissingRuntimeBound {
                    resource: "max_cfg_branches",
                });
            }
            let branches = u64::from(self.image.cfg_branch_count());
            if branches > u64::from(limits.max_cfg_branches) {
                return Err(GenerationResourceError::CfgBranchCapacity {
                    requested: branches,
                    available: limits.max_cfg_branches,
                });
            }
        }
        Ok(())
    }
}

/// Sums [`ImageEncoderInput::kv_token_capacity`] over one image's encoder
/// inputs, saturating instead of overflowing.
fn encoder_kv_bound(
    inputs: &[ImageEncoderInput],
    limits: &GenerationLimits,
) -> Result<usize, GenerationResourceError> {
    inputs.iter().try_fold(0usize, |total, input| {
        Ok(total.saturating_add(input.kv_token_capacity(limits)? as usize))
    })
}

impl ImageEncoderInput {
    /// Maximum KV contribution of this encoder, resolving dynamic lengths from the model.
    ///
    /// Uses `num_kv_tokens` when present, then `max_kv_tokens`, then the loaded
    /// grid-token capacity for the encoder kind. Fails with
    /// `GenerationResourceError::UnboundedImageKv` when the resolved length is
    /// zero.
    pub fn kv_token_capacity(
        &self,
        limits: &GenerationLimits,
    ) -> Result<u32, GenerationResourceError> {
        let capacity = match self.encoder {
            ImageIngestStep::VaeEncode => limits.max_vae_grid_tokens,
            ImageIngestStep::VitEncode => limits.max_vit_grid_tokens,
        };
        let tokens = self
            .num_kv_tokens
            .or(self.max_kv_tokens)
            .unwrap_or(capacity);
        if tokens == 0 {
            return Err(GenerationResourceError::UnboundedImageKv {
                call: self.encoder.as_str(),
            });
        }
        Ok(tokens)
    }
}

impl ImageIngestStep {
    /// Returns the stable call name used by resource diagnostics.
    fn as_str(self) -> &'static str {
        match self {
            Self::VaeEncode => "vae_encode",
            Self::VitEncode => "vit_encode",
        }
    }
}

/// Failures while deriving bounded generation resources.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GenerationResourceError {
    /// A worker-selected image KV length has no declared or loaded maximum.
    #[error("{call} has a worker-selected KV length but the runtime declares no bound")]
    UnboundedImageKv {
        /// Encoder call missing a bound.
        call: &'static str,
    },
    /// A resource the request needs has a zero runtime maximum.
    #[error("image generation requires the runtime to declare {resource}")]
    MissingRuntimeBound {
        /// Name of the missing resource maximum.
        resource: &'static str,
    },
    /// Image dimensions do not align to the latent grid.
    #[error(
        "image dimensions {width}x{height} must be divisible by the runtime latent downsample factor {latent_downsample}"
    )]
    ImageDimensionAlignment {
        /// Requested image width.
        width: u32,
        /// Requested image height.
        height: u32,
        /// Required latent downsample factor.
        latent_downsample: u32,
    },
    /// Requested latent grid exceeds runtime capacity.
    #[error("requested image latent units ({requested}) exceed runtime capacity ({available})")]
    LatentCapacity {
        /// Requested latent allocation units.
        requested: u64,
        /// Available latent allocation units.
        available: u64,
    },
    /// Requested guidance branches exceed runtime capacity.
    #[error("requested CFG branches ({requested}) exceed runtime capacity ({available})")]
    CfgBranchCapacity {
        /// Requested branch count.
        requested: u64,
        /// Available branch count.
        available: u32,
    },
    /// Context encoder-cache demand exceeds runtime capacity.
    #[error("requested encoder-cache entries ({requested}) exceed runtime capacity ({available})")]
    EncoderCacheCapacity {
        /// Requested cache entry count.
        requested: usize,
        /// Available cache entry count.
        available: u32,
    },
}

/// Scheduler-relevant prefix-cache behavior resolved during compilation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachePolicy {
    /// Whether admission may reuse cached prefixes or encoder outputs.
    pub read: bool,
    /// Whether completed context may populate caches.
    pub write: bool,
    /// Optional tenant or request-group cache namespace.
    pub isolation_key: Option<u64>,
}

impl Default for CachePolicy {
    /// Returns a shared-cache policy with reads and writes enabled.
    fn default() -> Self {
        Self {
            read: true,
            write: true,
            isolation_key: None,
        }
    }
}

/// Validated scheduler-facing generation request.
///
/// The value contains immutable request inputs. Capacity requirements are derived
/// from these inputs and loaded model limits; the engine owns actual reservations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationRequest {
    /// Engine request identity.
    pub request_id: RequestId,
    /// Final positive model prompt, including template and image-marker tokens.
    pub prompt_token_ids: Vec<u32>,
    /// Final negative conditioning tokens used by image generation.
    pub negative_prompt_token_ids: Vec<u32>,
    /// Media positions refer directly to `prompt_token_ids`.
    pub multimodal_inputs: MultimodalInputs,
    /// Requested understanding/image output constraint.
    pub constraint: GenerationConstraint,
    /// Text sampling parameters.
    pub sampling: SamplingParams,
    /// Image-generation parameters.
    pub image: ImageParams,
    /// Maximum understanding tokens generated by the request.
    pub max_und_tokens: usize,
    /// Text sequences that terminate generation.
    pub stop_strings: Vec<String>,
    /// Token identities that terminate generation.
    pub stop_token_ids: Vec<u32>,
    /// Scheduler priority.
    pub priority: i32,
    /// Prefix and encoder-cache policy.
    pub cache: CachePolicy,
    /// Whether a matched stopping token is included in public text output.
    pub include_stop_token: bool,
    /// Model-specific image trigger and feedback requirements.
    pub image_generation: ImageGenerationConfig,
    /// Canvas rows a readout request denoises once over its prompt, in
    /// report order. A readout generates nothing: it answers with the
    /// candidate log-probabilities of every row's slots, and an empty list
    /// makes the request a generating one.
    #[serde(default)]
    pub readout: Vec<ReadoutRow>,
    /// Block-diffusion sampling of a generating request, whose text is
    /// denoised in canvases rather than decoded token by token; `None`
    /// decodes autoregressively.
    #[serde(default)]
    pub canvas: Option<CanvasSampling>,
}

impl GenerationRequest {
    /// Returns whether the request reads canvas slots instead of generating.
    pub fn is_readout(&self) -> bool {
        !self.readout.is_empty()
    }

    /// Returns whether the request generates its text in denoised canvases.
    pub fn is_canvas_generation(&self) -> bool {
        self.canvas.is_some()
    }

    /// Returns the number of candidate log-probabilities a readout reports:
    /// one per candidate of every slot of every row.
    pub fn readout_candidates(&self) -> usize {
        self.readout
            .iter()
            .flat_map(|row| &row.slots)
            .map(|slot| slot.candidates.len())
            .sum()
    }

    /// Returns the runtime features the request's computation reaches.
    ///
    /// A readout or a canvas-generating request needs token denoising and
    /// the encoders of its context images; any other request needs what its
    /// generation graph reaches (`ImageGenerationConfig::required_features`).
    pub fn required_features(&self) -> GenerationFeatures {
        let context_steps = self
            .multimodal_inputs
            .images
            .iter()
            .flat_map(|image| image.encoders.iter().map(|input| input.encoder));
        if self.is_readout() || self.is_canvas_generation() {
            return context_steps.fold(GenerationFeatures::TOKEN_DENOISING, |needs, step| {
                needs | step.required_feature()
            });
        }
        self.image_generation
            .required_features(self.constraint, context_steps)
    }

    /// Validates image conditions, input positions, and sampling.
    ///
    /// These checks depend only on the request; checks against loaded
    /// capacity are in [`GenerationRequest::validate_resources`].
    pub fn validate(&self) -> Result<(), GenerationRequestError> {
        // Validate request-wide policy and parameter invariants before walking
        // the positioned multimodal inputs.
        if self.prompt_token_ids.is_empty() && self.multimodal_inputs.images.is_empty() {
            return Err(GenerationRequestError::EmptyContext);
        }
        if self.is_readout() {
            validate_readout(self)?;
        } else if self.max_und_tokens == 0 && !self.finishes_after_image() {
            return Err(GenerationRequestError::ZeroMaxUndTokens);
        }
        if let Some(canvas) = &self.canvas {
            validate_canvas(self, canvas)?;
        }

        self.sampling
            .validate()
            .map_err(GenerationRequestError::InvalidSampling)?;
        self.image
            .validate()
            .map_err(GenerationRequestError::InvalidImage)?;
        if self.decodes_text() && self.sampling.min_tokens > self.max_und_tokens {
            return Err(GenerationRequestError::MinTokensExceedsMaximum {
                min_tokens: self.sampling.min_tokens,
                max_und_tokens: self.max_und_tokens,
            });
        }
        if matches!(self.constraint, GenerationConstraint::GenOnly) && !self.generates_images() {
            return Err(GenerationRequestError::GenOnlyCannotProduceImage);
        }

        // Trigger patterns must be nonempty. Text that continues after an image
        // needs a feedback source and a continuation token, and configured
        // feedback needs valid encoder inputs and logical positions.
        match &self.image_generation.trigger {
            ImageTrigger::Suffix { token_ids } if token_ids.is_empty() => {
                return Err(GenerationRequestError::EmptyTriggerPattern);
            }
            ImageTrigger::RoundCloseThenSuffix {
                close_token_ids,
                trigger_token_ids,
            } if close_token_ids.is_empty() || trigger_token_ids.is_empty() => {
                return Err(GenerationRequestError::EmptyTriggerPattern);
            }
            ImageTrigger::Disabled
            | ImageTrigger::Token { .. }
            | ImageTrigger::Suffix { .. }
            | ImageTrigger::RoundCloseThenSuffix { .. } => {}
        }
        if self.continues_after_image() {
            if self.image_generation.feedback_source.is_none() {
                return Err(GenerationRequestError::MissingImageFeedback);
            }
            if self.image_generation.feedback_next_token == FeedbackNextToken::None {
                return Err(GenerationRequestError::MissingFeedbackToken);
            }
        }
        if self.image_generation.feedback_source.is_some() {
            validate_image_encoders(
                &self.image_generation.feedback_encoders,
                self.image_generation.num_feedback_positions,
            )?;
        }

        // Image positions belong to the final token vector. The input order is
        // also the order of encoder contributions when positions are equal.
        let context_tokens = self.prompt_token_ids.len();
        let mut previous_position = 0;
        for image in &self.multimodal_inputs.images {
            if image.b64.is_empty() {
                return Err(GenerationRequestError::EmptyImagePayload);
            }
            validate_image_encoders(&image.encoders, image.num_positions)?;
            if image.position as usize > context_tokens {
                return Err(GenerationRequestError::ImagePositionBeyondPrompt {
                    position: image.position,
                    prompt_tokens: context_tokens,
                });
            }
            if image.position < previous_position {
                return Err(GenerationRequestError::UnorderedImageInputs);
            }
            previous_position = image.position;
        }

        // Empty stop strings would match every output position and do not form
        // a meaningful termination boundary.
        if self.stop_strings.iter().any(String::is_empty) {
            return Err(GenerationRequestError::EmptyStopString);
        }
        Ok(())
    }
}

/// Validates a canvas-generating request: it generates text only, names the
/// seed its draws follow, and its sampling parameters are in range.
fn validate_canvas(
    request: &GenerationRequest,
    canvas: &CanvasSampling,
) -> Result<(), GenerationRequestError> {
    if request.is_readout() || request.generates_images() {
        return Err(GenerationRequestError::CanvasGeneratesTextOnly);
    }
    if request.sampling.seed.is_none() {
        return Err(GenerationRequestError::UnseededCanvas);
    }
    match canvas.invalid_parameter() {
        Some(parameter) => Err(GenerationRequestError::InvalidCanvasSampling { parameter }),
        None => Ok(()),
    }
}

/// Validates a readout's canvas rows: it generates no tokens or images, and
/// every row holds tokens and reads at least one slot inside itself, each
/// with at least one candidate.
fn validate_readout(request: &GenerationRequest) -> Result<(), GenerationRequestError> {
    if request.max_und_tokens != 0 || request.generates_images() {
        return Err(GenerationRequestError::GeneratingReadout);
    }
    for row in &request.readout {
        if row.token_ids.is_empty() || row.slots.is_empty() {
            return Err(GenerationRequestError::EmptyReadoutRow);
        }
        if row
            .slots
            .windows(2)
            .any(|pair| pair[0].position >= pair[1].position)
        {
            return Err(GenerationRequestError::UnorderedReadoutSlots);
        }
        for slot in &row.slots {
            if slot.position as usize >= row.token_ids.len() {
                return Err(GenerationRequestError::ReadoutSlotOutsideRow {
                    position: slot.position,
                    row_tokens: row.token_ids.len(),
                });
            }
            if slot.candidates.is_empty() {
                return Err(GenerationRequestError::EmptyReadoutSlot);
            }
        }
    }
    Ok(())
}

/// Validates an image's encoder inputs and logical position contribution.
fn validate_image_encoders(
    inputs: &[ImageEncoderInput],
    num_positions: u32,
) -> Result<(), GenerationRequestError> {
    if inputs.is_empty() {
        return Err(GenerationRequestError::MissingImageEncoders);
    }
    if num_positions == 0 {
        return Err(GenerationRequestError::ZeroImageLogicalPositions);
    }
    for input in inputs {
        if input.num_kv_tokens == Some(0) || input.max_kv_tokens == Some(0) {
            return Err(GenerationRequestError::ZeroImageKvBound);
        }
        if let (Some(tokens), Some(max_tokens)) = (input.num_kv_tokens, input.max_kv_tokens)
            && tokens > max_tokens
        {
            return Err(GenerationRequestError::ImageKvExceedsCapacity { tokens, max_tokens });
        }
    }
    Ok(())
}

/// Validation failures for a scheduler-facing generation request.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GenerationRequestError {
    /// The positive input contains neither tokens nor images.
    #[error("generation input must contain tokens or images")]
    EmptyContext,
    /// Understanding decode has a zero token budget.
    #[error("max_und_tokens must be positive")]
    ZeroMaxUndTokens,
    /// An image input, or configured image feedback, declares no encoder stages.
    #[error("image input has no encoders")]
    MissingImageEncoders,
    /// An input image carries no encoded payload.
    #[error("image payload must not be empty")]
    EmptyImagePayload,
    /// An image refers beyond the final positive prompt.
    #[error("image position {position} exceeds the {prompt_tokens}-token prompt")]
    ImagePositionBeyondPrompt {
        /// Exclusive token position of the image marker.
        position: u32,
        /// Length of the final token vector.
        prompt_tokens: usize,
    },
    /// Images would be consumed in a different order than their prompt positions.
    #[error("input images must follow prompt-token position order")]
    UnorderedImageInputs,
    /// An image input, or configured image feedback, contributes no logical
    /// model positions.
    #[error("image inputs and feedback must consume at least one logical position")]
    ZeroImageLogicalPositions,
    /// An explicit image KV length or capacity is zero.
    #[error("image KV lengths and capacities must be positive")]
    ZeroImageKvBound,
    /// An exact image contribution must fit its declared capacity.
    #[error("image KV length {tokens} exceeds capacity {max_tokens}")]
    ImageKvExceedsCapacity { tokens: u32, max_tokens: u32 },
    /// A suffix or round-close trigger contains no tokens.
    #[error("generation trigger patterns must not be empty")]
    EmptyTriggerPattern,
    /// Image-only generation cannot open an image branch.
    #[error("gen_only request cannot produce an image under the resolved policy")]
    GenOnlyCannotProduceImage,
    /// Image continuation has no feedback input.
    #[error("default generation continuation requires image feedback")]
    MissingImageFeedback,
    /// Image feedback has no understanding continuation token.
    #[error("default generation feedback requires a continuation token")]
    MissingFeedbackToken,
    /// A configured stop string is empty.
    #[error("stop strings must not be empty")]
    EmptyStopString,
    /// A readout request also asks to generate tokens or images.
    #[error("a readout request generates no tokens or images")]
    GeneratingReadout,
    /// A readout row has no tokens or reads no slot.
    #[error("every readout row must hold tokens and read at least one slot")]
    EmptyReadoutRow,
    /// A readout slot lies beyond its row.
    #[error("readout slot {position} lies outside its {row_tokens}-token row")]
    ReadoutSlotOutsideRow {
        /// Slot position within its row.
        position: u32,
        /// Length of the row.
        row_tokens: usize,
    },
    /// A readout slot names no candidate token.
    #[error("every readout slot must read at least one candidate")]
    EmptyReadoutSlot,
    /// A readout row's slot positions do not strictly increase.
    #[error("readout slots must be listed in increasing position order")]
    UnorderedReadoutSlots,
    /// A canvas-generating request also reads slots or generates images.
    #[error("a canvas-generating request generates text only")]
    CanvasGeneratesTextOnly,
    /// A canvas-generating request names no seed for its draws.
    #[error("a canvas-generating request requires a sampling seed")]
    UnseededCanvas,
    /// A block-diffusion sampling parameter is out of range.
    #[error("block-diffusion sampling parameter `{parameter}` is out of range")]
    InvalidCanvasSampling {
        /// Name of the parameter.
        parameter: &'static str,
    },
    /// Text sampling parameters are invalid.
    #[error("invalid sampling parameters: {0}")]
    InvalidSampling(#[source] SamplingParamsError),
    /// Image-generation parameters are invalid.
    #[error("invalid image parameters: {0}")]
    InvalidImage(#[source] ImageParamsError),
    /// Minimum-token floor exceeds the generation budget.
    #[error("min_tokens ({min_tokens}) exceeds max_und_tokens ({max_und_tokens})")]
    MinTokensExceedsMaximum {
        /// Required minimum generated tokens.
        min_tokens: usize,
        /// Maximum generated understanding tokens.
        max_und_tokens: usize,
    },
}

impl Default for ImageGenerationConfig {
    /// Disables image generation until the model configures a trigger or immediate start.
    fn default() -> Self {
        Self {
            trigger: ImageTrigger::Disabled,
            requires_text_for_image: true,
            feedback_source: None,
            feedback_next_token: FeedbackNextToken::None,
            num_feedback_positions: 0,
            feedback_encoders: Vec::new(),
            sample_feedback_continuation: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime_limits() -> GenerationLimits {
        GenerationLimits {
            features: GenerationFeatures::all(),
            max_latent_units: 4_096,
            latent_downsample: 16,
            max_vae_grid_tokens: 64,
            max_vit_grid_tokens: 64,
            max_latent_feature_bytes: 1 << 20,
            max_vision_feature_bytes: 1 << 20,
            commit_marker_tokens: 2,
            max_cfg_branches: 3,
            encoder_cache_entries: 4,
        }
    }

    /// A default-constraint request that reaches every branch: one context
    /// image with a bounded VAE input and an exact ViT input, a round-close
    /// trigger, and artifact feedback through both encoders. Its cache policy
    /// neither reads nor writes, so encoder-cache reservations are opt-in per
    /// test.
    fn complete_request() -> GenerationRequest {
        let policy = ImageGenerationConfig {
            trigger: ImageTrigger::RoundCloseThenSuffix {
                close_token_ids: vec![2, 3],
                trigger_token_ids: vec![40, 41],
            },
            feedback_source: Some(FeedbackSource::ArtifactProduct),
            feedback_next_token: FeedbackNextToken::Token { token_id: 12 },
            num_feedback_positions: 1,
            feedback_encoders: vec![
                ImageEncoderInput {
                    encoder: ImageIngestStep::VaeEncode,
                    num_kv_tokens: None,
                    max_kv_tokens: Some(64),
                },
                ImageEncoderInput {
                    encoder: ImageIngestStep::VitEncode,
                    num_kv_tokens: None,
                    max_kv_tokens: Some(64),
                },
            ],
            sample_feedback_continuation: false,
            ..ImageGenerationConfig::default()
        };
        let constraint = GenerationConstraint::Default;
        GenerationRequest {
            request_id: RequestId(7),
            prompt_token_ids: vec![1, 2, 3, 4],
            negative_prompt_token_ids: vec![9],
            multimodal_inputs: MultimodalInputs {
                images: vec![ImageInput {
                    hash: 17,
                    b64: "aW1hZ2U=".into(),
                    position: 2,
                    num_positions: 1,
                    encoders: vec![
                        ImageEncoderInput {
                            encoder: ImageIngestStep::VaeEncode,
                            num_kv_tokens: None,
                            max_kv_tokens: Some(64),
                        },
                        ImageEncoderInput {
                            encoder: ImageIngestStep::VitEncode,
                            num_kv_tokens: Some(32),
                            max_kv_tokens: None,
                        },
                    ],
                }],
            },
            constraint,
            sampling: SamplingParams::default(),
            image: ImageParams::default(),
            max_und_tokens: 16,
            include_stop_token: false,
            stop_strings: vec!["stop".into()],
            stop_token_ids: vec![2],
            priority: 3,
            cache: CachePolicy {
                read: false,
                write: false,
                isolation_key: Some(91),
            },
            image_generation: policy,
            readout: Vec::new(),
            canvas: None,
        }
    }

    #[test]
    fn generation_constraint_strings_are_canonical() {
        for (name, constraint) in [
            ("default", GenerationConstraint::Default),
            ("und_only", GenerationConstraint::UndOnly),
            ("gen_only", GenerationConstraint::GenOnly),
        ] {
            assert_eq!(constraint.as_str(), name);
            assert_eq!(name.parse::<GenerationConstraint>().unwrap(), constraint);
        }
        assert!("text".parse::<GenerationConstraint>().is_err());
    }

    #[test]
    fn encoder_cache_keys_are_stable_and_step_scoped() {
        let first = encoder_cache_key(17, 0, ImageIngestStep::VaeEncode);
        assert_eq!(first, encoder_cache_key(17, 0, ImageIngestStep::VaeEncode));
        assert_ne!(first, encoder_cache_key(18, 0, ImageIngestStep::VaeEncode));
        assert_ne!(first, encoder_cache_key(17, 1, ImageIngestStep::VaeEncode));
        assert_ne!(first, encoder_cache_key(17, 0, ImageIngestStep::VitEncode));
    }

    #[test]
    fn trigger_descriptors_match_only_their_declared_boundary() {
        let token = ImageTrigger::Token { token_id: 7 };
        assert!(token.matches_generated(&[1, 7]));
        assert!(!token.matches_generated(&[7, 1]));

        let suffix = ImageTrigger::Suffix {
            token_ids: vec![4, 5],
        };
        assert!(suffix.matches_generated(&[1, 4, 5]));
        assert!(!suffix.matches_round_close(&[1, 4, 5], 9));

        let round = ImageTrigger::RoundCloseThenSuffix {
            close_token_ids: vec![9, 10],
            trigger_token_ids: vec![4, 5],
        };
        assert!(!round.matches_generated(&[1, 4, 5]));
        assert!(round.matches_round_close(&[1, 4, 5], 9));
        assert!(!round.matches_round_close(&[1, 4, 5], 11));
    }

    #[test]
    fn canonical_generation_graph_roundtrips_and_validates() {
        let request = complete_request();
        request.validate().expect("valid canonical request");

        let encoded = serde_json::to_value(&request).expect("serialize request");
        let decoded: GenerationRequest =
            serde_json::from_value(encoded).expect("deserialize request");
        assert_eq!(decoded, request);
        decoded.validate().expect("round-tripped request is valid");
    }

    #[test]
    fn conservative_resources_cover_every_ingest_step_and_generated_feedback() {
        let mut request = complete_request();
        request.cache.read = true;
        request.cache.write = true;
        request.image.max_images = 2;
        request
            .validate_resources(&runtime_limits())
            .expect("supported image input");
        // Prompt, Und budget, the context image's VAE bound and exact ViT
        // length, two feedback images through both bounded encoders, and the
        // negative prompt.
        assert_eq!(
            request.max_kv_tokens(&runtime_limits()).unwrap(),
            4 + 16 + 64 + 32 + 2 * (64 + 64) + 1
        );
        // The default 512x512 image over a downsample of 16 is a 32x32 grid.
        assert_eq!(
            request.image_latent_units(&runtime_limits()).unwrap(),
            1_024
        );
        assert_eq!(request.num_encoder_cache_entries(), 2);
    }

    #[test]
    fn conservative_resources_use_each_exact_image_ingest_step() {
        // Exact encoder lengths are charged as declared even above the loaded
        // grid capacities, and an Und-only request charges no feedback or
        // negative prompt.
        let mut request = complete_request();
        request.prompt_token_ids = vec![1; 35];
        request.multimodal_inputs.images = vec![ImageInput {
            hash: 17,
            b64: "aW1hZ2U=".into(),
            position: 35,
            num_positions: 1,
            encoders: vec![
                ImageEncoderInput {
                    encoder: ImageIngestStep::VaeEncode,
                    num_kv_tokens: Some(1_026),
                    max_kv_tokens: None,
                },
                ImageEncoderInput {
                    encoder: ImageIngestStep::VitEncode,
                    num_kv_tokens: Some(1_371),
                    max_kv_tokens: None,
                },
            ],
        }];
        request.constraint = GenerationConstraint::UndOnly;
        request.image_generation.feedback_source = None;
        request.image_generation.feedback_encoders.clear();
        request.max_und_tokens = 256;

        assert_eq!(
            request.max_kv_tokens(&runtime_limits()).unwrap(),
            35 + 256 + 1_026 + 1_371
        );
    }

    #[test]
    fn request_capacity_rejects_unbounded_encoders_and_insufficient_model_limits() {
        let mut request = complete_request();
        let input = &mut request.multimodal_inputs.images[0].encoders[1];
        input.num_kv_tokens = None;
        input.max_kv_tokens = None;
        let mut limits = runtime_limits();
        limits.max_vit_grid_tokens = 0;
        assert_eq!(
            request.validate_resources(&limits),
            Err(GenerationResourceError::UnboundedImageKv { call: "vit_encode" })
        );

        // The remaining cases keep the unbounded ViT input, which resolves to
        // the loaded `max_vit_grid_tokens` under the default limits.
        let mut limits = runtime_limits();
        limits.max_latent_units = 1_023;
        assert!(matches!(
            request.validate_resources(&limits),
            Err(GenerationResourceError::LatentCapacity { .. })
        ));

        let mut limits = runtime_limits();
        limits.max_vision_feature_bytes = 0;
        assert_eq!(
            request.validate_resources(&limits),
            Err(GenerationResourceError::MissingRuntimeBound {
                resource: "max_vision_feature_bytes",
            })
        );

        let mut limits = runtime_limits();
        limits.max_latent_feature_bytes = 0;
        assert_eq!(
            request.validate_resources(&limits),
            Err(GenerationResourceError::MissingRuntimeBound {
                resource: "max_latent_feature_bytes",
            })
        );

        let mut limits = runtime_limits();
        limits.max_cfg_branches = 1;
        assert_eq!(
            request.validate_resources(&limits),
            Err(GenerationResourceError::CfgBranchCapacity {
                requested: 2,
                available: 1,
            })
        );

        request.cache.read = true;
        let mut limits = runtime_limits();
        limits.encoder_cache_entries = 1;
        assert_eq!(
            request.validate_resources(&limits),
            Err(GenerationResourceError::EncoderCacheCapacity {
                requested: 2,
                available: 1,
            })
        );
        // Disabling cache use removes its reservation; encoder computation remains.
        request.cache.read = false;
        request.validate_resources(&limits).unwrap();
    }

    #[test]
    fn request_capacity_requires_runtime_aligned_image_dimensions() {
        let request = complete_request();
        let mut limits = runtime_limits();
        limits.latent_downsample = 24;

        assert_eq!(
            request.validate_resources(&limits),
            Err(GenerationResourceError::ImageDimensionAlignment {
                width: request.image.width,
                height: request.image.height,
                latent_downsample: 24,
            })
        );
    }

    #[test]
    fn request_validation_rejects_invalid_image_boundaries() {
        let mut empty_ingest = complete_request();
        empty_ingest.multimodal_inputs.images[0].encoders.clear();
        assert_eq!(
            empty_ingest.validate(),
            Err(GenerationRequestError::MissingImageEncoders)
        );

        // Exact contributions and capacities are validated before admission.
        for (tokens, capacity, error) in [
            (Some(0), None, GenerationRequestError::ZeroImageKvBound),
            (None, Some(0), GenerationRequestError::ZeroImageKvBound),
            (
                Some(65),
                Some(64),
                GenerationRequestError::ImageKvExceedsCapacity {
                    tokens: 65,
                    max_tokens: 64,
                },
            ),
        ] {
            let mut request = complete_request();
            let input = &mut request.multimodal_inputs.images[0].encoders[0];
            input.num_kv_tokens = tokens;
            input.max_kv_tokens = capacity;
            assert_eq!(request.validate(), Err(error));
        }

        let mut exact_capacity = complete_request();
        exact_capacity.multimodal_inputs.images[0].encoders[0].num_kv_tokens = Some(64);
        assert_eq!(exact_capacity.validate(), Ok(()));

        let mut invalid_sampling = complete_request();
        invalid_sampling.sampling.top_p = 0.0;
        assert!(matches!(
            invalid_sampling.validate(),
            Err(GenerationRequestError::InvalidSampling(
                SamplingParamsError::TopP { .. }
            ))
        ));

        let mut invalid_image = complete_request();
        invalid_image.image.cfg_interval = (1.0, 0.0);
        assert!(matches!(
            invalid_image.validate(),
            Err(GenerationRequestError::InvalidImage(
                ImageParamsError::CfgIntervalOrder { .. }
            ))
        ));

        let mut invalid_floor = complete_request();
        invalid_floor.sampling.min_tokens = invalid_floor.max_und_tokens + 1;
        assert!(matches!(
            invalid_floor.validate(),
            Err(GenerationRequestError::MinTokensExceedsMaximum { .. })
        ));

        let mut misplaced_image = complete_request();
        misplaced_image.multimodal_inputs.images[0].position = 5;
        assert_eq!(
            misplaced_image.validate(),
            Err(GenerationRequestError::ImagePositionBeyondPrompt {
                position: 5,
                prompt_tokens: 4
            })
        );

        let mut unordered_images = complete_request();
        let mut preceding = unordered_images.multimodal_inputs.images[0].clone();
        preceding.position = 1;
        unordered_images.multimodal_inputs.images.push(preceding);
        assert_eq!(
            unordered_images.validate(),
            Err(GenerationRequestError::UnorderedImageInputs)
        );
    }

    /// A readout of one context image's prompt: two canvas rows, the first
    /// with two slots and the second with one.
    fn readout_request() -> GenerationRequest {
        let mut request = complete_request();
        request.constraint = GenerationConstraint::UndOnly;
        request.max_und_tokens = 0;
        request.image_generation = ImageGenerationConfig::default();
        request.multimodal_inputs.images[0].encoders.truncate(1);
        request.multimodal_inputs.images[0].encoders[0].encoder = ImageIngestStep::VitEncode;
        request.readout = vec![
            ReadoutRow {
                token_ids: vec![11, 4, 12, 4, 13],
                slots: vec![
                    ReadoutSlot {
                        position: 1,
                        candidates: vec![30, 31],
                    },
                    ReadoutSlot {
                        position: 3,
                        candidates: vec![40, 41, 42],
                    },
                ],
            },
            ReadoutRow {
                token_ids: vec![11, 30, 4],
                slots: vec![ReadoutSlot {
                    position: 2,
                    candidates: vec![50],
                }],
            },
        ];
        request
    }

    /// A readout generates nothing, counts every candidate it reports, and
    /// needs canvas denoising plus the encoders of its context images rather
    /// than text decoding.
    #[test]
    fn a_readout_reports_its_candidates_and_needs_canvas_denoising() {
        let request = readout_request();
        assert_eq!(request.validate(), Ok(()));
        assert!(request.is_readout());
        assert_eq!(request.readout_candidates(), 6);
        assert_eq!(
            request.required_features(),
            GenerationFeatures::TOKEN_DENOISING | GenerationFeatures::VISION_ENCODE
        );
        // Its KV holds the prompt and image only; canvases write none.
        assert_eq!(request.max_kv_tokens(&runtime_limits()).unwrap(), 4 + 64);
    }

    #[test]
    fn readout_validation_rejects_generation_and_misplaced_slots() {
        let mut generating = readout_request();
        generating.max_und_tokens = 4;
        assert_eq!(
            generating.validate(),
            Err(GenerationRequestError::GeneratingReadout)
        );

        let mut empty = readout_request();
        empty.readout[1].slots.clear();
        assert_eq!(
            empty.validate(),
            Err(GenerationRequestError::EmptyReadoutRow)
        );

        let mut outside = readout_request();
        outside.readout[1].slots[0].position = 3;
        assert_eq!(
            outside.validate(),
            Err(GenerationRequestError::ReadoutSlotOutsideRow {
                position: 3,
                row_tokens: 3
            })
        );

        let mut unread = readout_request();
        unread.readout[0].slots[0].candidates.clear();
        assert_eq!(
            unread.validate(),
            Err(GenerationRequestError::EmptyReadoutSlot)
        );

        let mut unordered = readout_request();
        unordered.readout[0].slots.swap(0, 1);
        assert_eq!(
            unordered.validate(),
            Err(GenerationRequestError::UnorderedReadoutSlots)
        );
    }
}
