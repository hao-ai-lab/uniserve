//! Generation policies, resource bounds, and scheduler-facing request values.
//!
//! These data-only descriptors define generation behavior before the scheduler
//! plans worker operations. Runtime ownership and model execution remain outside
//! this boundary.

use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::Modality;
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

/// Whether an Und token segment is user-visible or internal control/context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UndVisibility {
    /// Publishes tokens to the caller.
    #[default]
    Visible,
    /// Retains tokens as model-control context.
    Internal,
}

/// One context segment in the lowered request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ContextSegment {
    /// Ordered understanding tokens.
    UndTokens {
        /// Vocabulary token identities.
        token_ids: Vec<u32>,
        /// Caller visibility of the segment.
        #[serde(default)]
        visibility: UndVisibility,
    },
    /// Input image and its model-specific ingest recipe.
    Image {
        /// Encoded image payload and logical placement.
        image: ImageSegment,
        /// Encoder operations used to ingest the image.
        ingest: ImageIngestRecipe,
    },
}

/// Input image bytes plus placement in the rendered context stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageSegment {
    /// Stable content hash used for encoder-cache identity.
    pub hash: u64,
    /// Base64-encoded input image.
    pub b64: String,
    /// Logical position in the rendered context.
    pub placement: SegmentPlacement,
}

/// Logical placement of an image segment in the already-rendered Und stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum SegmentPlacement {
    /// The image encoder output fills the gap ending at this token index.
    AtToken {
        /// Exclusive token position at the end of the image gap.
        position: u32,
    },
    /// The image is appended after all Und tokens emitted by the context.
    Append,
}

/// Model-description recipe for turning an image segment into context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageIngestRecipe {
    /// Ordered encoder stages.
    pub steps: Vec<ImageIngestStep>,
    /// Logical model positions contributed by the image.
    pub logical_positions: u32,
    /// Physical KV effect corresponding to each encoder stage.
    pub step_kv_tokens: Vec<ImageKvEffect>,
    /// Model branch that consumes the encoded image.
    pub modality: Modality,
}

impl ImageIngestRecipe {
    /// Builds a single-step vision-encoder recipe.
    pub fn vit_only(logical_positions: u32, kv_tokens: ImageKvEffect) -> Self {
        Self {
            steps: vec![ImageIngestStep::VitEncode],
            logical_positions,
            step_kv_tokens: vec![kv_tokens],
            modality: Modality::Und,
        }
    }

    /// Builds a latent-encoder followed by vision-encoder recipe.
    pub fn vae_then_vit(
        logical_positions: u32,
        vae_kv_tokens: ImageKvEffect,
        vit_kv_tokens: ImageKvEffect,
    ) -> Self {
        Self {
            steps: vec![ImageIngestStep::VaeEncode, ImageIngestStep::VitEncode],
            logical_positions,
            step_kv_tokens: vec![vae_kv_tokens, vit_kv_tokens],
            modality: Modality::Und,
        }
    }

    /// Returns the declared KV effect for one ingest step.
    pub fn kv_effect(&self, step_index: usize) -> Option<ImageKvEffect> {
        self.step_kv_tokens.get(step_index).copied()
    }

    /// Returns stable per-step keys for reusable worker-side encoder outputs.
    pub fn encoder_cache_keys(&self, image_hash: u64) -> Vec<u64> {
        self.steps
            .iter()
            .copied()
            .enumerate()
            .map(|(step_index, step)| encoder_cache_key(image_hash, step_index, step))
            .collect()
    }
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

/// Derives the cache identity of one step in an image-ingest recipe.
pub fn encoder_cache_key(image_hash: u64, step_index: usize, step: ImageIngestStep) -> u64 {
    // Apply FNV-1a to the complete domain tuple in a fixed byte order. Including
    // the stage index distinguishes repeated encoder kinds within one recipe.
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

/// Physical KV effect of an image ingest operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ImageKvEffect {
    /// Uses the runtime-advertised bound for this encoder stage.
    WorkerDefined,
    /// Produces an exact number of physical KV tokens.
    Exact {
        /// Exact physical KV token count.
        tokens: u32,
    },
    /// Produces a worker-selected count up to a fixed maximum.
    Bounded {
        /// Maximum physical KV token count.
        max_tokens: u32,
    },
}

/// Generated-image feedback recipe supplied by the model description.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeneratedImageFeedbackRecipe {
    /// Product representation fed into the encoder.
    pub source: FeedbackSource,
    /// Understanding token used after feedback ingestion.
    pub next_und_token: FeedbackNextToken,
    /// Encoder stages used to ingest the generated image.
    pub ingest: ImageIngestRecipe,
    /// Whether feedback state samples its continuation token.
    pub sample_continuation: bool,
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

/// Model-description token-trigger matching lowered to scheduler-readable data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum TriggerPolicyDescriptor {
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

impl TriggerPolicyDescriptor {
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

/// Scheduler action for generated Und tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UndTokenAction {
    /// Publishes understanding tokens.
    Emit,
    /// Retains understanding tokens as internal context.
    KeepInternal,
    /// Rejects constraints that require understanding output.
    Reject,
}

/// Visibility rules per output constraint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisibilityPolicyDescriptor {
    /// Action for unconstrained generation.
    pub default: UndTokenAction,
    /// Action for understanding-only generation.
    pub und_only: UndTokenAction,
    /// Action for image-only generation.
    pub gen_only: UndTokenAction,
}

impl Default for VisibilityPolicyDescriptor {
    /// Returns visibility rules that emit understanding output for eligible requests.
    fn default() -> Self {
        Self {
            default: UndTokenAction::Emit,
            und_only: UndTokenAction::Emit,
            gen_only: UndTokenAction::KeepInternal,
        }
    }
}

/// Termination rules express whether branch completion finishes or continues.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TerminationPolicyDescriptor {
    /// Whether EOS ends the request.
    pub eos_finishes: bool,
    /// Whether configured stop conditions end the request.
    pub stop_finishes: bool,
    /// Whether the matched stop token is published.
    #[serde(default)]
    pub emit_stop_token: bool,
    /// Whether the generated-token bound ends the request.
    pub max_tokens_finishes: bool,
    /// Whether image commit completes an image-only request.
    pub gen_commit_finishes_gen_only: bool,
    /// Whether unconstrained generation resumes after image commit.
    pub gen_commit_continues_default: bool,
}

impl Default for TerminationPolicyDescriptor {
    /// Returns terminal defaults for text bounds and image-only completion.
    fn default() -> Self {
        Self {
            eos_finishes: true,
            stop_finishes: true,
            emit_stop_token: false,
            max_tokens_finishes: true,
            gen_commit_finishes_gen_only: true,
            gen_commit_continues_default: true,
        }
    }
}

/// How a Gen-only request reaches its first Gen branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenOnlyStartPolicyDescriptor {
    /// Internal Und decoding until the model trigger opens Gen.
    #[default]
    DiscoverTrigger,
    /// Enter Gen immediately after all context segments are prepared.
    Immediate,
}

/// Model-description generation policy consumed by the scheduler planner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationPolicyDescriptor {
    /// Model tokens that open an image-generation branch.
    pub trigger: TriggerPolicyDescriptor,
    /// Entry behavior for image-only requests.
    #[serde(default)]
    pub gen_only_start: GenOnlyStartPolicyDescriptor,
    /// Constraint-specific understanding-token visibility.
    pub visibility: VisibilityPolicyDescriptor,
    /// Request termination rules.
    pub termination: TerminationPolicyDescriptor,
    /// Generated-image feedback pipeline, when supported.
    pub feedback: Option<GeneratedImageFeedbackRecipe>,
}

/// Constraint-resolved behavior consumed by scheduler lifecycle planning.
///
/// The explicit decisions give admission and scheduling one shared
/// interpretation of the request constraint and model policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationBehaviorDescriptor {
    /// Whether the runtime executes understanding decode.
    pub und_decode: bool,
    /// Resolved handling for understanding tokens.
    pub und_tokens: UndTokenAction,
    /// Whether the request may produce images.
    pub gen_output: bool,
    /// Whether image generation starts immediately after context preparation.
    pub start_gen_after_context: bool,
    /// Whether committed images feed back into model context.
    pub generated_image_feedback: bool,
    /// Whether understanding decode resumes after image commit.
    pub continue_after_gen_commit: bool,
    /// Whether image commit completes the request.
    pub finish_after_gen_commit: bool,
}

impl GenerationBehaviorDescriptor {
    /// Resolves scheduler actions for a request constraint and model policy.
    pub fn resolve(constraint: GenerationConstraint, policy: &GenerationPolicyDescriptor) -> Self {
        // Resolve caller visibility independently from the mechanism that opens
        // the image-generation branch.
        let und_tokens = match constraint {
            GenerationConstraint::Default => policy.visibility.default,
            GenerationConstraint::UndOnly => policy.visibility.und_only,
            GenerationConstraint::GenOnly => policy.visibility.gen_only,
        };
        let start_gen_after_context = matches!(constraint, GenerationConstraint::GenOnly)
            && policy.gen_only_start == GenOnlyStartPolicyDescriptor::Immediate;
        let gen_output = !matches!(constraint, GenerationConstraint::UndOnly)
            && (!matches!(policy.trigger, TriggerPolicyDescriptor::Disabled)
                || start_gen_after_context);

        // Image commit is terminal for image-only requests and may return to
        // understanding decode for unconstrained requests.
        let finish_after_gen_commit = gen_output
            && matches!(constraint, GenerationConstraint::GenOnly)
            && policy.termination.gen_commit_finishes_gen_only;
        let continue_after_gen_commit = gen_output
            && matches!(constraint, GenerationConstraint::Default)
            && policy.termination.gen_commit_continues_default;

        Self {
            und_decode: !start_gen_after_context
                && (und_tokens != UndTokenAction::Reject || gen_output),
            und_tokens,
            gen_output,
            start_gen_after_context,
            generated_image_feedback: gen_output
                && continue_after_gen_commit
                && policy.feedback.is_some(),
            continue_after_gen_commit,
            finish_after_gen_commit,
        }
    }

    /// Returns whether understanding tokens are visible to the caller.
    pub fn emits_und(&self) -> bool {
        self.und_tokens == UndTokenAction::Emit
    }

    /// Returns the runtime features required by the reachable generation graph.
    pub fn required_features(
        &self,
        policy: &GenerationPolicyDescriptor,
        context_image_steps: impl IntoIterator<Item = ImageIngestStep>,
    ) -> GenerationFeatures {
        // Understanding execution is the common control path for every request.
        let mut needs = GenerationFeatures::UNDERSTANDING;

        // Context images require the encoder stages declared by their recipes.
        for step in context_image_steps {
            needs.insert(match step {
                ImageIngestStep::VaeEncode => GenerationFeatures::LATENT_ENCODE,
                ImageIngestStep::VitEncode => GenerationFeatures::VISION_ENCODE,
            });
        }
        if self.gen_output {
            needs.insert(GenerationFeatures::IMAGE_GENERATION);
        }

        // Feedback can make additional encoder stages reachable after image
        // materialization.
        if self.generated_image_feedback
            && let Some(feedback) = &policy.feedback
        {
            for step in feedback.ingest.steps.iter().copied() {
                needs.insert(match step {
                    ImageIngestStep::VaeEncode => GenerationFeatures::LATENT_ENCODE,
                    ImageIngestStep::VitEncode => GenerationFeatures::VISION_ENCODE,
                });
            }
        }
        needs
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
    }
}

impl GenerationFeatures {
    /// Returns the admission diagnostic name for the first represented feature.
    pub fn name(self) -> &'static str {
        if self.contains(Self::UNDERSTANDING) {
            "runtime_und_execution"
        } else if self.contains(Self::LATENT_ENCODE) {
            "runtime_vae_encode"
        } else if self.contains(Self::VISION_ENCODE) {
            "runtime_vit_encode"
        } else if self.contains(Self::IMAGE_GENERATION) {
            "runtime_gen_denoise"
        } else {
            "runtime_generation_features"
        }
    }
}

impl std::fmt::Display for GenerationFeatures {
    /// Writes the diagnostic name of the first represented feature.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

/// Conservative request-level resource declaration produced by compilation.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct GenerationResourceBounds {
    /// Understanding tokens present in the positive context.
    pub context_tokens: usize,
    /// Maximum physical KV tokens retained by the request.
    pub max_kv_tokens: usize,
    /// Maximum latent allocation units for one generated image.
    pub max_image_latent_units: u64,
    /// Maximum latent allocation bytes for one generated image.
    pub max_image_latent_bytes: u64,
    /// Maximum VAE feature product bytes.
    pub max_latent_feature_bytes: u64,
    /// Maximum vision feature product bytes.
    pub max_vision_feature_bytes: u64,
    /// Encoder cache entries the context may pin concurrently.
    pub encoder_cache_keys: Vec<u64>,
    /// Whether generated-image feedback makes the request non-replayable.
    pub generated_feedback_makes_non_replayable: bool,
}

/// Inputs used to derive conservative resources for one generation graph.
pub struct GenerationResources<'a> {
    /// Positive generation context.
    pub context: &'a [ContextSegment],
    /// Negative image-generation conditioning context.
    pub negative_context: &'a [ContextSegment],
    /// Constraint-resolved branch behavior.
    pub behavior: &'a GenerationBehaviorDescriptor,
    /// Model-specific generation policy.
    pub policy: &'a GenerationPolicyDescriptor,
    /// Image-generation parameters.
    pub image: &'a ImageParams,
    /// Maximum understanding tokens generated by the request.
    pub max_und_tokens: usize,
    /// Prefix and encoder-cache policy.
    pub cache: &'a GenerationCachePolicyDescriptor,
    /// Runtime-advertised capability limits.
    pub limits: &'a GenerationLimits,
}

/// Worker and scheduler limits needed to compile a bounded generation graph.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct GenerationLimits {
    /// Generation features implemented by the runtime.
    pub features: GenerationFeatures,
    /// Maximum latent allocation units for one request.
    pub max_latent_units: u64,
    /// Pixel-to-latent spatial downsample factor.
    pub latent_downsample: u32,
    /// Maximum VAE grid tokens for a worker-defined effect.
    pub max_vae_grid_tokens: u32,
    /// Maximum vision grid tokens for a worker-defined effect.
    pub max_vit_grid_tokens: u32,
    /// Maximum VAE feature product size in bytes.
    pub max_latent_feature_bytes: u64,
    /// Maximum vision feature product size in bytes.
    pub max_vision_feature_bytes: u64,
    /// KV tokens reserved for image commit markers.
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

impl GenerationResourceBounds {
    /// Derives conservative resource maxima for a bounded generation graph.
    pub fn conservative(inputs: GenerationResources<'_>) -> Result<Self, GenerationResourceError> {
        let GenerationResources {
            context,
            negative_context,
            behavior,
            policy,
            image,
            max_und_tokens,
            cache,
            limits,
        } = inputs;

        // Text and image KV contributions are accounted independently because
        // each image ingest step may declare a different physical token effect.
        let context_tokens = context
            .iter()
            .map(|segment| match segment {
                ContextSegment::UndTokens { token_ids, .. } => token_ids.len(),
                ContextSegment::Image { .. } => 0,
            })
            .sum::<usize>();
        let negative_tokens = negative_context
            .iter()
            .map(|segment| match segment {
                ContextSegment::UndTokens { token_ids, .. } => token_ids.len(),
                ContextSegment::Image { .. } => 0,
            })
            .sum::<usize>();
        let input_image_kv_tokens = context.iter().try_fold(0usize, |total, segment| {
            let ContextSegment::Image { ingest, .. } = segment else {
                return Ok(total);
            };
            Ok(total.saturating_add(ingest_kv_bound(ingest, limits)?))
        })?;

        // Generated-image feedback repeats its ingest contract once per maximum
        // output image and therefore contributes to worst-case KV capacity.
        let feedback_kv_per_image = if behavior.generated_image_feedback {
            match policy.feedback.as_ref() {
                Some(feedback) => ingest_kv_bound(&feedback.ingest, limits)?,
                None => return Err(GenerationResourceError::MissingFeedback),
            }
        } else {
            0
        };
        let generated_feedback_kv_tokens =
            feedback_kv_per_image.saturating_mul(image.max_images as usize);

        // Determine feature storage requirements from the exact ingest steps
        // reachable through context images or generated feedback.
        let feedback_ingest = behavior
            .generated_image_feedback
            .then(|| policy.feedback.as_ref().map(|feedback| &feedback.ingest))
            .flatten();
        let uses_ingest_step = |step| {
            context.iter().any(|segment| {
                matches!(
                    segment,
                    ContextSegment::Image { ingest, .. } if ingest.steps.contains(&step)
                )
            }) || feedback_ingest.is_some_and(|ingest| ingest.steps.contains(&step))
        };
        let uses_latent_features = uses_ingest_step(ImageIngestStep::VaeEncode);
        let uses_vision_features = uses_ingest_step(ImageIngestStep::VitEncode);
        if uses_latent_features && limits.max_latent_feature_bytes == 0 {
            return Err(GenerationResourceError::MissingRuntimeBound {
                resource: "max_latent_feature_bytes",
            });
        }
        if uses_vision_features && limits.max_vision_feature_bytes == 0 {
            return Err(GenerationResourceError::MissingRuntimeBound {
                resource: "max_vision_feature_bytes",
            });
        }

        // Cache capacity covers every distinct encoder step that this request
        // may pin concurrently.
        let encoder_cache_keys = if cache.read || cache.write {
            context
                .iter()
                .flat_map(|segment| match segment {
                    ContextSegment::Image { image, ingest } => {
                        ingest.encoder_cache_keys(image.hash)
                    }
                    ContextSegment::UndTokens { .. } => Vec::new(),
                })
                .collect()
        } else {
            Vec::new()
        };
        if !encoder_cache_keys.is_empty() {
            if limits.encoder_cache_entries == 0 {
                return Err(GenerationResourceError::MissingRuntimeBound {
                    resource: "encoder_cache_entries",
                });
            }
            if encoder_cache_keys.len() > limits.encoder_cache_entries as usize {
                return Err(GenerationResourceError::EncoderCacheCapacity {
                    requested: encoder_cache_keys.len(),
                    available: limits.encoder_cache_entries,
                });
            }
        }

        // Media generation validates grid alignment and runtime capacities
        // before deriving latent storage from the requested image dimensions.
        if behavior.gen_output && limits.latent_downsample == 0 {
            return Err(GenerationResourceError::MissingRuntimeBound {
                resource: "latent_downsample",
            });
        }
        if behavior.gen_output
            && (!image.width.is_multiple_of(limits.latent_downsample)
                || !image.height.is_multiple_of(limits.latent_downsample))
        {
            return Err(GenerationResourceError::ImageDimensionAlignment {
                width: image.width,
                height: image.height,
                latent_downsample: limits.latent_downsample,
            });
        }
        if behavior.gen_output && limits.max_latent_units == 0 {
            return Err(GenerationResourceError::MissingRuntimeBound {
                resource: "max_latent_units",
            });
        }
        if behavior.gen_output && limits.max_cfg_branches == 0 {
            return Err(GenerationResourceError::MissingRuntimeBound {
                resource: "max_cfg_branches",
            });
        }

        let latent_downsample = limits.latent_downsample.max(1);
        let requested_latent_units = u64::from(image.width / latent_downsample)
            .saturating_mul(u64::from(image.height / latent_downsample));
        if behavior.gen_output && requested_latent_units > limits.max_latent_units {
            return Err(GenerationResourceError::LatentCapacity {
                requested: requested_latent_units,
                available: limits.max_latent_units,
            });
        }

        // The worker's feature-byte maximum establishes a conservative byte
        // density for the requested latent grid.
        let image_latent_bytes = if behavior.gen_output {
            if limits.max_vae_grid_tokens == 0 || limits.max_latent_feature_bytes == 0 {
                return Err(GenerationResourceError::MissingRuntimeBound {
                    resource: "max_latent_feature_bytes",
                });
            }
            let bytes_per_unit = limits
                .max_latent_feature_bytes
                .div_ceil(u64::from(limits.max_vae_grid_tokens));
            requested_latent_units.saturating_mul(bytes_per_unit)
        } else {
            0
        };
        let requested_cfg_branches = u64::from(image.cfg_branch_count());
        if behavior.gen_output && requested_cfg_branches > u64::from(limits.max_cfg_branches) {
            return Err(GenerationResourceError::CfgBranchCapacity {
                requested: requested_cfg_branches,
                available: limits.max_cfg_branches,
            });
        }

        // Saturating sums keep the declaration conservative even when an input
        // approaches the host representation limit.
        Ok(Self {
            context_tokens,
            max_kv_tokens: context_tokens
                .saturating_add(max_und_tokens)
                .saturating_add(input_image_kv_tokens)
                .saturating_add(generated_feedback_kv_tokens)
                .saturating_add(behavior.gen_output.then_some(negative_tokens).unwrap_or(0)),
            max_image_latent_units: if behavior.gen_output {
                requested_latent_units
            } else {
                0
            },
            max_image_latent_bytes: image_latent_bytes,
            max_latent_feature_bytes: if uses_latent_features {
                limits.max_latent_feature_bytes
            } else {
                0
            },
            max_vision_feature_bytes: if uses_vision_features {
                limits.max_vision_feature_bytes
            } else {
                0
            },
            encoder_cache_keys,
            generated_feedback_makes_non_replayable: behavior.generated_image_feedback,
        })
    }

    /// Checks that this declaration covers every required resource maximum.
    pub fn validate_covers(&self, required: &Self) -> Result<(), GenerationResourceError> {
        // Compare scalar capacities through one table so every undersized field
        // produces the same structured diagnostic.
        for (resource, declared, required) in [
            (
                "max_kv_tokens",
                self.max_kv_tokens as u64,
                required.max_kv_tokens as u64,
            ),
            (
                "max_image_latent_units",
                self.max_image_latent_units,
                required.max_image_latent_units,
            ),
            (
                "max_image_latent_bytes",
                self.max_image_latent_bytes,
                required.max_image_latent_bytes,
            ),
            (
                "max_latent_feature_bytes",
                self.max_latent_feature_bytes,
                required.max_latent_feature_bytes,
            ),
            (
                "max_vision_feature_bytes",
                self.max_vision_feature_bytes,
                required.max_vision_feature_bytes,
            ),
        ] {
            if declared < required {
                return Err(GenerationResourceError::DeclaredBoundTooSmall {
                    resource,
                    declared,
                    required,
                });
            }
        }

        // Replayability is a capability declaration rather than a numeric
        // capacity and therefore requires a separate implication check.
        if required.generated_feedback_makes_non_replayable
            && !self.generated_feedback_makes_non_replayable
        {
            return Err(GenerationResourceError::MissingNonReplayableDeclaration);
        }
        Ok(())
    }
}

/// Computes the worst-case KV contribution of every configured image-ingest step.
fn ingest_kv_bound(
    ingest: &ImageIngestRecipe,
    limits: &GenerationLimits,
) -> Result<usize, GenerationResourceError> {
    // Stage/effect alignment is required before the two vectors can be folded
    // into one conservative bound.
    if ingest.steps.len() != ingest.step_kv_tokens.len() {
        return Err(GenerationResourceError::ImageIngestKvArity {
            steps: ingest.steps.len(),
            effects: ingest.step_kv_tokens.len(),
        });
    }
    ingest
        .steps
        .iter()
        .copied()
        .zip(ingest.step_kv_tokens.iter().copied())
        .try_fold(0usize, |total, (step, effect)| {
            let fallback = match step {
                ImageIngestStep::VaeEncode => limits.max_vae_grid_tokens,
                ImageIngestStep::VitEncode => limits.max_vit_grid_tokens,
            };
            let step_bound = kv_effect_bound(effect, fallback, step.as_str())?;
            Ok(total.saturating_add(step_bound))
        })
}

/// Resolves an exact, bounded, or runtime-defined KV contribution.
fn kv_effect_bound(
    effect: ImageKvEffect,
    fallback: u32,
    operation: &'static str,
) -> Result<usize, GenerationResourceError> {
    match effect {
        ImageKvEffect::Exact { tokens } => Ok(tokens as usize),
        ImageKvEffect::Bounded { max_tokens } => Ok(max_tokens as usize),
        ImageKvEffect::WorkerDefined if fallback > 0 => Ok(fallback as usize),
        ImageKvEffect::WorkerDefined => {
            Err(GenerationResourceError::UnboundedImageKv { operation })
        }
    }
}

impl ImageIngestStep {
    /// Returns the stable operation name used by resource diagnostics.
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
    /// Image ingest stages and KV effects have different lengths.
    #[error("image ingest declares {steps} steps but {effects} KV effects")]
    ImageIngestKvArity {
        /// Number of encoder stages.
        steps: usize,
        /// Number of declared KV effects.
        effects: usize,
    },
    /// A worker-defined image KV effect has no runtime maximum.
    #[error("{operation} has a worker-defined KV effect but the runtime declares no bound")]
    UnboundedImageKv {
        /// Encoder operation missing a bound.
        operation: &'static str,
    },
    /// Continuation requires an image-feedback recipe.
    #[error("generated image continuation requires a feedback resource recipe")]
    MissingFeedback,
    /// Image generation requires a runtime resource maximum that is zero.
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
    /// Resource-bound arithmetic overflowed.
    #[error("{resource} overflowed while computing the request resource bound")]
    ResourceOverflow {
        /// Resource whose bound overflowed.
        resource: &'static str,
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
    /// A caller-declared resource maximum is below the computed requirement.
    #[error("declared {resource} bound ({declared}) is below the required bound ({required})")]
    DeclaredBoundTooSmall {
        /// Name of the undersized resource.
        resource: &'static str,
        /// Caller-declared maximum.
        declared: u64,
        /// Computed required maximum.
        required: u64,
    },
    /// Image feedback omits its non-replayable declaration.
    #[error("generated image feedback must be declared non-replayable")]
    MissingNonReplayableDeclaration,
}

/// Scheduler-relevant prefix-cache behavior resolved during compilation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationCachePolicyDescriptor {
    /// Whether admission may reuse cached prefixes or encoder outputs.
    pub read: bool,
    /// Whether completed context may populate caches.
    pub write: bool,
    /// Optional tenant or request-group cache namespace.
    pub isolation_key: Option<u64>,
}

impl Default for GenerationCachePolicyDescriptor {
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
/// The value contains immutable request data and conservative resource bounds;
/// submission channels and mutable runtime state live in the engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationRequest {
    /// Engine request identity.
    pub request_id: RequestId,
    /// Ordered positive model context.
    pub context: Vec<ContextSegment>,
    /// Negative conditioning context used by image generation.
    pub negative_context: Vec<ContextSegment>,
    /// Requested understanding/image output constraint.
    pub constraint: GenerationConstraint,
    /// Constraint-resolved scheduler behavior.
    pub behavior: GenerationBehaviorDescriptor,
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
    pub cache: GenerationCachePolicyDescriptor,
    /// Model-specific trigger, visibility, feedback, and termination policy.
    pub policy: GenerationPolicyDescriptor,
    /// Conservative resource declaration checked during admission.
    pub resources: GenerationResourceBounds,
}

impl GenerationRequest {
    /// Collects understanding-token context in logical order.
    pub fn prompt_token_ids(&self) -> Vec<u32> {
        self.context
            .iter()
            .flat_map(|segment| match segment {
                ContextSegment::UndTokens { token_ids, .. } => token_ids.as_slice(),
                ContextSegment::Image { .. } => &[],
            })
            .copied()
            .collect()
    }

    /// Counts understanding tokens in the positive context.
    pub fn prompt_token_count(&self) -> usize {
        self.context
            .iter()
            .map(|segment| match segment {
                ContextSegment::UndTokens { token_ids, .. } => token_ids.len(),
                ContextSegment::Image { .. } => 0,
            })
            .sum()
    }

    /// Counts image segments in the positive context.
    pub fn context_image_count(&self) -> usize {
        self.context
            .iter()
            .filter(|segment| matches!(segment, ContextSegment::Image { .. }))
            .count()
    }

    /// Validates policy consistency, context layout, and declared bounds.
    pub fn validate(&self) -> Result<(), GenerationRequestError> {
        // Validate request-wide policy and parameter invariants before walking
        // the ordered context.
        if self.context.is_empty() {
            return Err(GenerationRequestError::EmptyContext);
        }
        if self.max_und_tokens == 0 && !self.behavior.finish_after_gen_commit {
            return Err(GenerationRequestError::ZeroMaxUndTokens);
        }
        if self.behavior != GenerationBehaviorDescriptor::resolve(self.constraint, &self.policy) {
            return Err(GenerationRequestError::BehaviorPolicyMismatch);
        }

        self.sampling
            .validate()
            .map_err(GenerationRequestError::InvalidSampling)?;
        self.image
            .validate()
            .map_err(GenerationRequestError::InvalidImage)?;
        if self.behavior.und_decode && self.sampling.min_tokens > self.max_und_tokens {
            return Err(GenerationRequestError::MinTokensExceedsMaximum {
                min_tokens: self.sampling.min_tokens,
                max_und_tokens: self.max_und_tokens,
            });
        }
        if matches!(self.constraint, GenerationConstraint::GenOnly) && !self.behavior.gen_output {
            return Err(GenerationRequestError::GenOnlyCannotProduceImage);
        }

        // Trigger and continuation policies must form a finite scheduler graph.
        if !self.policy.termination.max_tokens_finishes {
            return Err(GenerationRequestError::NonTerminalMaxTokensPolicy);
        }
        match &self.policy.trigger {
            TriggerPolicyDescriptor::Suffix { token_ids } if token_ids.is_empty() => {
                return Err(GenerationRequestError::EmptyTriggerPattern);
            }
            TriggerPolicyDescriptor::RoundCloseThenSuffix {
                close_token_ids,
                trigger_token_ids,
            } if close_token_ids.is_empty() || trigger_token_ids.is_empty() => {
                return Err(GenerationRequestError::EmptyTriggerPattern);
            }
            TriggerPolicyDescriptor::Disabled
            | TriggerPolicyDescriptor::Token { .. }
            | TriggerPolicyDescriptor::Suffix { .. }
            | TriggerPolicyDescriptor::RoundCloseThenSuffix { .. } => {}
        }
        if self.policy.trigger.requires_round_close() && !self.policy.termination.eos_finishes {
            return Err(GenerationRequestError::NonTerminalRoundClosePolicy);
        }
        if self.behavior.continue_after_gen_commit {
            let feedback = self
                .policy
                .feedback
                .as_ref()
                .ok_or(GenerationRequestError::MissingFeedbackRecipe)?;
            if feedback.next_und_token == FeedbackNextToken::None {
                return Err(GenerationRequestError::IncompleteFeedbackRecipe);
            }
        }
        if let Some(feedback) = &self.policy.feedback {
            validate_ingest_recipe(&feedback.ingest)?;
        }

        // Validate segment placement while deriving the context-dependent cache
        // identities and token count used by the resource declaration.
        let context_tokens = self.prompt_token_count();
        let mut seen_tokens = 0usize;
        let mut expected_encoder_cache_keys = Vec::new();
        for segment in &self.context {
            match segment {
                ContextSegment::UndTokens { token_ids, .. } => {
                    seen_tokens = seen_tokens.saturating_add(token_ids.len());
                }
                ContextSegment::Image { image, ingest } => {
                    if image.b64.is_empty() {
                        return Err(GenerationRequestError::EmptyImagePayload);
                    }
                    validate_ingest_recipe(ingest)?;
                    let expected_position = match image.placement {
                        SegmentPlacement::AtToken { position } => position as usize,
                        SegmentPlacement::Append => context_tokens,
                    };
                    if expected_position != seen_tokens {
                        return Err(GenerationRequestError::ImagePlacementMismatch {
                            expected: seen_tokens,
                            actual: expected_position,
                        });
                    }
                    if self.cache.read || self.cache.write {
                        expected_encoder_cache_keys.extend(ingest.encoder_cache_keys(image.hash));
                    }
                }
            }
        }

        // Negative conditioning is text-only because image ingest belongs to
        // the positive model context.
        if self
            .negative_context
            .iter()
            .any(|segment| matches!(segment, ContextSegment::Image { .. }))
        {
            return Err(GenerationRequestError::ImageInNegativeContext);
        }

        // Resource fields are supplied independently and must agree exactly
        // with the validated positive context.
        if self.resources.context_tokens != context_tokens {
            return Err(GenerationRequestError::ContextTokenBoundMismatch {
                expected: context_tokens,
                actual: self.resources.context_tokens,
            });
        }
        if self.resources.max_kv_tokens < context_tokens {
            return Err(GenerationRequestError::MaxKvBelowContext {
                context_tokens,
                max_kv_tokens: self.resources.max_kv_tokens,
            });
        }
        if self.resources.encoder_cache_keys != expected_encoder_cache_keys {
            return Err(GenerationRequestError::EncoderCacheKeysMismatch {
                expected: expected_encoder_cache_keys,
                actual: self.resources.encoder_cache_keys.clone(),
            });
        }

        // Empty stop strings would match every output position and do not form
        // a meaningful termination boundary.
        if self.stop_strings.iter().any(String::is_empty) {
            return Err(GenerationRequestError::EmptyStopString);
        }
        Ok(())
    }

    /// Recomputes required resources under `limits` and checks the declaration.
    pub fn validate_resources(
        &self,
        limits: &GenerationLimits,
    ) -> Result<(), GenerationResourceError> {
        let required = GenerationResourceBounds::conservative(GenerationResources {
            context: &self.context,
            negative_context: &self.negative_context,
            behavior: &self.behavior,
            policy: &self.policy,
            image: &self.image,
            max_und_tokens: self.max_und_tokens,
            cache: &self.cache,
            limits,
        })?;
        self.resources.validate_covers(&required)
    }
}

/// Validates an image-ingest recipe's stage alignment and positive bounds.
fn validate_ingest_recipe(recipe: &ImageIngestRecipe) -> Result<(), GenerationRequestError> {
    if recipe.steps.is_empty() {
        return Err(GenerationRequestError::EmptyImageIngestRecipe);
    }
    if recipe.steps.len() != recipe.step_kv_tokens.len() {
        return Err(GenerationRequestError::ImageIngestKvArity {
            steps: recipe.steps.len(),
            effects: recipe.step_kv_tokens.len(),
        });
    }
    if recipe.logical_positions == 0 {
        return Err(GenerationRequestError::ZeroImageLogicalPositions);
    }
    recipe
        .step_kv_tokens
        .iter()
        .copied()
        .try_for_each(validate_kv_effect)
}

/// Validates that an explicit image KV effect contributes a positive bound.
fn validate_kv_effect(effect: ImageKvEffect) -> Result<(), GenerationRequestError> {
    if matches!(
        effect,
        ImageKvEffect::Exact { tokens: 0 } | ImageKvEffect::Bounded { max_tokens: 0 }
    ) {
        return Err(GenerationRequestError::ZeroImageKvBound);
    }
    Ok(())
}

/// Validation failures for a scheduler-facing generation request.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GenerationRequestError {
    /// The positive context contains no segments.
    #[error("generation context must contain at least one segment")]
    EmptyContext,
    /// Image ingest stages and KV effects have different lengths.
    #[error("image ingest declares {steps} steps but {effects} KV effects")]
    ImageIngestKvArity {
        /// Number of encoder stages.
        steps: usize,
        /// Number of declared KV effects.
        effects: usize,
    },
    /// Understanding decode has a zero token budget.
    #[error("max_und_tokens must be positive")]
    ZeroMaxUndTokens,
    /// Resolved behavior disagrees with the request constraint and policy.
    #[error("resolved generation behavior does not match constraint and policy")]
    BehaviorPolicyMismatch,
    /// An image context segment declares no encoder stages.
    #[error("image context segment has an empty ingest recipe")]
    EmptyImageIngestRecipe,
    /// Negative conditioning contains an image segment.
    #[error("negative context may contain only Und token segments")]
    ImageInNegativeContext,
    /// An image context segment carries no encoded payload.
    #[error("image segment payload must not be empty")]
    EmptyImagePayload,
    /// An image segment does not follow the preceding understanding tokens.
    #[error(
        "image segment placement does not match ordered context: expected {expected}, got {actual}"
    )]
    ImagePlacementMismatch {
        /// Position implied by preceding context segments.
        expected: usize,
        /// Position declared by the image segment.
        actual: usize,
    },
    /// An image ingest recipe contributes no logical model positions.
    #[error("image ingest and feedback recipes must consume at least one logical position")]
    ZeroImageLogicalPositions,
    /// A bounded image KV effect has a zero maximum.
    #[error("bounded image KV effects must be positive")]
    ZeroImageKvBound,
    /// A suffix or round-close trigger contains no tokens.
    #[error("generation trigger patterns must not be empty")]
    EmptyTriggerPattern,
    /// Image-only generation cannot open an image branch.
    #[error("gen_only request cannot produce an image under the resolved policy")]
    GenOnlyCannotProduceImage,
    /// Image continuation has no feedback recipe.
    #[error("default generation continuation requires a feedback recipe")]
    MissingFeedbackRecipe,
    /// Image feedback has no understanding continuation token.
    #[error("default generation feedback requires a continuation token")]
    IncompleteFeedbackRecipe,
    /// The maximum-token policy does not terminate at the request bound.
    #[error("max-token policy must terminate at the declared request bound")]
    NonTerminalMaxTokensPolicy,
    /// A round-close trigger is paired with non-terminal round closure.
    #[error("round-close trigger policy requires terminal round closure")]
    NonTerminalRoundClosePolicy,
    /// Declared context-token count disagrees with the context.
    #[error("resource context-token bound mismatch: expected {expected}, got {actual}")]
    ContextTokenBoundMismatch {
        /// Token count derived from the context.
        expected: usize,
        /// Declared context-token count.
        actual: usize,
    },
    /// Declared KV bound cannot contain the positive context.
    #[error("max_kv_tokens ({max_kv_tokens}) is below context tokens ({context_tokens})")]
    MaxKvBelowContext {
        /// Tokens present in the positive context.
        context_tokens: usize,
        /// Declared maximum KV tokens.
        max_kv_tokens: usize,
    },
    /// Declared encoder-cache keys disagree with image ingest recipes.
    #[error(
        "encoder-cache key declaration does not match context: expected {expected:?}, got {actual:?}"
    )]
    EncoderCacheKeysMismatch {
        /// Keys derived from image context segments.
        expected: Vec<u64>,
        /// Keys declared in request resources.
        actual: Vec<u64>,
    },
    /// A configured stop string is empty.
    #[error("stop strings must not be empty")]
    EmptyStopString,
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

impl Default for GenerationPolicyDescriptor {
    /// Returns a text-only policy with visible understanding output.
    fn default() -> Self {
        Self {
            trigger: TriggerPolicyDescriptor::Disabled,
            gen_only_start: GenOnlyStartPolicyDescriptor::default(),
            visibility: VisibilityPolicyDescriptor::default(),
            termination: TerminationPolicyDescriptor::default(),
            feedback: None,
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

    fn complete_request() -> GenerationRequest {
        let policy = GenerationPolicyDescriptor {
            trigger: TriggerPolicyDescriptor::RoundCloseThenSuffix {
                close_token_ids: vec![2, 3],
                trigger_token_ids: vec![40, 41],
            },
            visibility: VisibilityPolicyDescriptor {
                default: UndTokenAction::Emit,
                und_only: UndTokenAction::Emit,
                gen_only: UndTokenAction::KeepInternal,
            },
            termination: TerminationPolicyDescriptor::default(),
            feedback: Some(GeneratedImageFeedbackRecipe {
                source: FeedbackSource::ArtifactProduct,
                next_und_token: FeedbackNextToken::Token { token_id: 12 },
                ingest: ImageIngestRecipe::vae_then_vit(
                    1,
                    ImageKvEffect::Bounded { max_tokens: 64 },
                    ImageKvEffect::Bounded { max_tokens: 64 },
                ),
                sample_continuation: false,
            }),
            ..GenerationPolicyDescriptor::default()
        };
        let constraint = GenerationConstraint::Default;
        let mut request = GenerationRequest {
            request_id: RequestId(7),
            context: vec![
                ContextSegment::UndTokens {
                    token_ids: vec![1, 2],
                    visibility: UndVisibility::Internal,
                },
                ContextSegment::Image {
                    image: ImageSegment {
                        hash: 17,
                        b64: "aW1hZ2U=".into(),
                        placement: SegmentPlacement::AtToken { position: 2 },
                    },
                    ingest: ImageIngestRecipe::vae_then_vit(
                        1,
                        ImageKvEffect::Bounded { max_tokens: 64 },
                        ImageKvEffect::Exact { tokens: 32 },
                    ),
                },
                ContextSegment::UndTokens {
                    token_ids: vec![3, 4],
                    visibility: UndVisibility::Visible,
                },
            ],
            negative_context: vec![ContextSegment::UndTokens {
                token_ids: vec![9],
                visibility: UndVisibility::Internal,
            }],
            behavior: GenerationBehaviorDescriptor::resolve(constraint, &policy),
            constraint,
            sampling: SamplingParams::default(),
            image: ImageParams::default(),
            max_und_tokens: 16,
            stop_strings: vec!["stop".into()],
            stop_token_ids: vec![2],
            priority: 3,
            cache: GenerationCachePolicyDescriptor {
                read: false,
                write: false,
                isolation_key: Some(91),
            },
            policy,
            resources: GenerationResourceBounds::default(),
        };
        request.resources = GenerationResourceBounds::conservative(GenerationResources {
            context: &request.context,
            negative_context: &request.negative_context,
            behavior: &request.behavior,
            policy: &request.policy,
            image: &request.image,
            max_und_tokens: request.max_und_tokens,
            cache: &request.cache,
            limits: &runtime_limits(),
        })
        .expect("bounded request fixture");
        request
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
        let recipe = ImageIngestRecipe::vae_then_vit(
            1,
            ImageKvEffect::WorkerDefined,
            ImageKvEffect::WorkerDefined,
        );
        let keys = recipe.encoder_cache_keys(17);
        assert_eq!(keys.len(), 2);
        assert_ne!(keys[0], keys[1]);
        assert_eq!(
            keys[1],
            encoder_cache_key(17, 1, ImageIngestStep::VitEncode)
        );
        assert_ne!(keys, recipe.encoder_cache_keys(18));
    }

    #[test]
    fn default_visibility_hides_gen_only_und_tokens() {
        let policy = VisibilityPolicyDescriptor::default();
        assert_eq!(policy.default, UndTokenAction::Emit);
        assert_eq!(policy.und_only, UndTokenAction::Emit);
        assert_eq!(policy.gen_only, UndTokenAction::KeepInternal);
    }

    #[test]
    fn behavior_is_resolved_before_scheduler_admission() {
        let policy = GenerationPolicyDescriptor {
            trigger: TriggerPolicyDescriptor::Token { token_id: 42 },
            feedback: Some(GeneratedImageFeedbackRecipe {
                source: FeedbackSource::DeviceProduct,
                next_und_token: FeedbackNextToken::EndOfImage,
                ingest: ImageIngestRecipe::vit_only(2, ImageKvEffect::WorkerDefined),
                sample_continuation: true,
            }),
            ..GenerationPolicyDescriptor::default()
        };
        let default = GenerationBehaviorDescriptor::resolve(GenerationConstraint::Default, &policy);
        assert!(default.gen_output);
        assert!(default.generated_image_feedback);
        assert!(default.continue_after_gen_commit);

        let und_only =
            GenerationBehaviorDescriptor::resolve(GenerationConstraint::UndOnly, &policy);
        assert!(!und_only.gen_output);
        assert!(und_only.emits_und());

        let gen_only =
            GenerationBehaviorDescriptor::resolve(GenerationConstraint::GenOnly, &policy);
        assert!(gen_only.gen_output);
        assert!(!gen_only.start_gen_after_context);
        assert!(gen_only.und_decode);
        assert!(!gen_only.emits_und());
        assert!(gen_only.finish_after_gen_commit);

        let immediate_policy = GenerationPolicyDescriptor {
            trigger: TriggerPolicyDescriptor::Disabled,
            gen_only_start: GenOnlyStartPolicyDescriptor::Immediate,
            ..GenerationPolicyDescriptor::default()
        };
        let immediate =
            GenerationBehaviorDescriptor::resolve(GenerationConstraint::GenOnly, &immediate_policy);
        assert!(immediate.gen_output);
        assert!(immediate.start_gen_after_context);
        assert!(!immediate.und_decode);
        assert!(immediate.finish_after_gen_commit);
        assert_eq!(
            immediate.required_features(&immediate_policy, []),
            GenerationFeatures::UNDERSTANDING | GenerationFeatures::IMAGE_GENERATION
        );
    }

    #[test]
    fn trigger_descriptors_match_only_their_declared_boundary() {
        let token = TriggerPolicyDescriptor::Token { token_id: 7 };
        assert!(token.matches_generated(&[1, 7]));
        assert!(!token.matches_generated(&[7, 1]));

        let suffix = TriggerPolicyDescriptor::Suffix {
            token_ids: vec![4, 5],
        };
        assert!(suffix.matches_generated(&[1, 4, 5]));
        assert!(!suffix.matches_round_close(&[1, 4, 5], 9));

        let round = TriggerPolicyDescriptor::RoundCloseThenSuffix {
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
        let bounds = GenerationResourceBounds::conservative(GenerationResources {
            context: &request.context,
            negative_context: &request.negative_context,
            behavior: &request.behavior,
            policy: &request.policy,
            image: &request.image,
            max_und_tokens: request.max_und_tokens,
            cache: &request.cache,
            limits: &runtime_limits(),
        })
        .expect("bounded resources");

        assert_eq!(bounds.context_tokens, 4);
        assert_eq!(bounds.max_kv_tokens, 4 + 16 + 64 + 32 + 2 * (64 + 64) + 1);
        assert_eq!(bounds.max_image_latent_units, 1_024);
        assert_eq!(bounds.encoder_cache_keys.len(), 2);
        assert!(bounds.generated_feedback_makes_non_replayable);
    }

    #[test]
    fn conservative_resources_use_each_exact_image_ingest_step() {
        let mut request = complete_request();
        request.context = vec![
            ContextSegment::UndTokens {
                token_ids: vec![1; 35],
                visibility: UndVisibility::Internal,
            },
            ContextSegment::Image {
                image: ImageSegment {
                    hash: 17,
                    b64: "aW1hZ2U=".into(),
                    placement: SegmentPlacement::AtToken { position: 35 },
                },
                ingest: ImageIngestRecipe::vae_then_vit(
                    1,
                    ImageKvEffect::Exact { tokens: 1_026 },
                    ImageKvEffect::Exact { tokens: 1_371 },
                ),
            },
        ];
        request.behavior.gen_output = false;
        request.behavior.generated_image_feedback = false;
        request.policy.feedback = None;
        request.max_und_tokens = 256;

        let bounds = GenerationResourceBounds::conservative(GenerationResources {
            context: &request.context,
            negative_context: &request.negative_context,
            behavior: &request.behavior,
            policy: &request.policy,
            image: &request.image,
            max_und_tokens: request.max_und_tokens,
            cache: &request.cache,
            limits: &runtime_limits(),
        })
        .expect("exact per-step image resources");

        assert_eq!(bounds.max_kv_tokens, 35 + 256 + 1_026 + 1_371);
        assert_eq!(bounds.max_kv_tokens.div_ceil(64), 42);
    }

    #[test]
    fn resource_compilation_rejects_unbounded_worker_kv_and_capacity_overflow() {
        let mut request = complete_request();
        let ContextSegment::Image { ingest, .. } = &mut request.context[1] else {
            panic!("image fixture");
        };
        ingest.step_kv_tokens[1] = ImageKvEffect::WorkerDefined;
        let mut limits = runtime_limits();
        limits.max_vit_grid_tokens = 0;
        assert_eq!(
            GenerationResourceBounds::conservative(GenerationResources {
                context: &request.context,
                negative_context: &request.negative_context,
                behavior: &request.behavior,
                policy: &request.policy,
                image: &request.image,
                max_und_tokens: request.max_und_tokens,
                cache: &request.cache,
                limits: &limits,
            }),
            Err(GenerationResourceError::UnboundedImageKv {
                operation: "vit_encode",
            })
        );

        let mut invalid_request = complete_request();
        let ContextSegment::Image { ingest, .. } = &mut invalid_request.context[1] else {
            panic!("image fixture");
        };
        ingest.step_kv_tokens.pop();
        assert_eq!(
            GenerationResourceBounds::conservative(GenerationResources {
                context: &invalid_request.context,
                negative_context: &invalid_request.negative_context,
                behavior: &invalid_request.behavior,
                policy: &invalid_request.policy,
                image: &invalid_request.image,
                max_und_tokens: invalid_request.max_und_tokens,
                cache: &invalid_request.cache,
                limits: &runtime_limits(),
            }),
            Err(GenerationResourceError::ImageIngestKvArity {
                steps: 2,
                effects: 1,
            })
        );

        let mut limits = runtime_limits();
        limits.max_latent_units = 1_023;
        assert!(matches!(
            request.validate_resources(&limits),
            Err(GenerationResourceError::LatentCapacity { .. })
        ));
    }

    #[test]
    fn resource_compilation_requires_runtime_aligned_image_dimensions() {
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
    fn declared_resources_must_cover_the_required_envelope() {
        let mut request = complete_request();
        request.resources.max_kv_tokens -= 1;
        assert!(matches!(
            request.validate_resources(&runtime_limits()),
            Err(GenerationResourceError::DeclaredBoundTooSmall {
                resource: "max_kv_tokens",
                ..
            })
        ));
    }

    #[test]
    fn request_validation_rejects_policy_drift_and_invalid_image_boundaries() {
        let mut behavior_drift = complete_request();
        behavior_drift.behavior.gen_output = false;
        assert_eq!(
            behavior_drift.validate(),
            Err(GenerationRequestError::BehaviorPolicyMismatch)
        );

        let mut empty_ingest = complete_request();
        let ContextSegment::Image { ingest, .. } = &mut empty_ingest.context[1] else {
            panic!("image fixture");
        };
        ingest.steps.clear();
        assert_eq!(
            empty_ingest.validate(),
            Err(GenerationRequestError::EmptyImageIngestRecipe)
        );

        let mut negative_image = complete_request();
        negative_image.negative_context = vec![negative_image.context[1].clone()];
        assert_eq!(
            negative_image.validate(),
            Err(GenerationRequestError::ImageInNegativeContext)
        );

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
        let ContextSegment::Image { image, .. } = &mut misplaced_image.context[1] else {
            panic!("image fixture");
        };
        image.placement = SegmentPlacement::AtToken { position: 1 };
        assert!(matches!(
            misplaced_image.validate(),
            Err(GenerationRequestError::ImagePlacementMismatch { .. })
        ));

        let mut invalid_resources = complete_request();
        invalid_resources.resources.encoder_cache_keys.push(1);
        assert!(matches!(
            invalid_resources.validate(),
            Err(GenerationRequestError::EncoderCacheKeysMismatch { .. })
        ));
    }
}
