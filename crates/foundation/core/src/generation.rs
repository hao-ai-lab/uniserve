//! Canonical generation request descriptors.
//!
//! These types describe the lowered generation paradigm before the scheduler
//! turns it into worker ops. They are data-only: no channels, worker handles,
//! scheduler state, or model-local logic belongs here.

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
    pub value: String,
}

/// Whether an Und token segment is user-visible or internal control/context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UndVisibility {
    #[default]
    Visible,
    Internal,
}

/// One context segment in the lowered request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ContextSegment {
    UndTokens {
        token_ids: Vec<u32>,
        #[serde(default)]
        visibility: UndVisibility,
    },
    Image {
        image: ImageSegment,
        ingest: ImageIngestRecipe,
    },
}

/// Input image bytes plus placement in the rendered context stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageSegment {
    pub hash: u64,
    pub b64: String,
    pub placement: SegmentPlacement,
}

/// Logical placement of an image segment in the already-rendered Und stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum SegmentPlacement {
    /// The image encoder output fills the gap ending at this token index.
    AtToken { position: u32 },
    /// The image is appended after all Und tokens emitted by the context.
    Append,
}

/// Model-description recipe for turning an image segment into context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageIngestRecipe {
    pub steps: Vec<ImageIngestStep>,
    pub logical_positions: u32,
    pub step_kv_tokens: Vec<ImageKvEffect>,
    pub modality: Modality,
}

impl ImageIngestRecipe {
    pub fn vit_only(logical_positions: u32, kv_tokens: ImageKvEffect) -> Self {
        Self {
            steps: vec![ImageIngestStep::VitEncode],
            logical_positions,
            step_kv_tokens: vec![kv_tokens],
            modality: Modality::Und,
        }
    }

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

    pub fn kv_effect(&self, step_index: usize) -> Option<ImageKvEffect> {
        self.step_kv_tokens.get(step_index).copied()
    }

    /// Stable per-step keys for reusable worker-side encoder outputs.
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
    VaeEncode,
    VitEncode,
}

/// Derive the cache identity of one step in an image-ingest recipe.
pub fn encoder_cache_key(image_hash: u64, step_index: usize, step: ImageIngestStep) -> u64 {
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
    WorkerDefined,
    Exact { tokens: u32 },
    Bounded { max_tokens: u32 },
}

/// Generated-image feedback recipe supplied by the model description.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeneratedImageFeedbackRecipe {
    pub source: FeedbackSource,
    pub next_und_token: FeedbackNextToken,
    pub ingest: ImageIngestRecipe,
    pub sample_continuation: bool,
}

/// Product channel through which a materialized image reaches feedback encode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum FeedbackSource {
    DeviceProduct,
    ArtifactProduct,
}

/// Und token used to continue after generated-image feedback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum FeedbackNextToken {
    None,
    Bos,
    EndOfImage,
    Token { token_id: u32 },
}

/// Model-description token-trigger matching lowered to scheduler-readable data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum TriggerPolicyDescriptor {
    Disabled,
    Token {
        token_id: u32,
    },
    Suffix {
        token_ids: Vec<u32>,
    },
    RoundCloseThenSuffix {
        close_token_ids: Vec<u32>,
        trigger_token_ids: Vec<u32>,
    },
}

impl TriggerPolicyDescriptor {
    /// Match a branch-opening trigger during ordinary Und generation.
    pub fn matches_generated(&self, generated: &[u32]) -> bool {
        match self {
            Self::Token { token_id } => generated.last() == Some(token_id),
            Self::Suffix { token_ids } => !token_ids.is_empty() && generated.ends_with(token_ids),
            Self::Disabled | Self::RoundCloseThenSuffix { .. } => false,
        }
    }

    /// Match a branch-opening suffix when the current Und round closes.
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

    /// Single token that directly opens a branch, when the policy has one.
    pub fn direct_token(&self) -> Option<u32> {
        match self {
            Self::Token { token_id } => Some(*token_id),
            Self::Disabled | Self::Suffix { .. } | Self::RoundCloseThenSuffix { .. } => None,
        }
    }

    /// Token sequence whose completion opens a branch during ordinary decode.
    pub fn generated_suffix(&self) -> Option<&[u32]> {
        match self {
            Self::Token { token_id } => Some(std::slice::from_ref(token_id)),
            Self::Suffix { token_ids } => Some(token_ids),
            Self::Disabled | Self::RoundCloseThenSuffix { .. } => None,
        }
    }

    pub fn requires_round_close(&self) -> bool {
        matches!(self, Self::RoundCloseThenSuffix { .. })
    }

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
    Emit,
    KeepInternal,
    Reject,
}

/// Visibility rules per output constraint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisibilityPolicyDescriptor {
    pub default: UndTokenAction,
    pub und_only: UndTokenAction,
    pub gen_only: UndTokenAction,
}

impl Default for VisibilityPolicyDescriptor {
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
    pub eos_finishes: bool,
    pub stop_finishes: bool,
    #[serde(default)]
    pub emit_stop_token: bool,
    pub max_tokens_finishes: bool,
    pub gen_commit_finishes_gen_only: bool,
    pub gen_commit_continues_default: bool,
}

impl Default for TerminationPolicyDescriptor {
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
    /// Decode internal Und tokens until the model trigger opens Gen.
    #[default]
    DiscoverTrigger,
    /// Enter Gen immediately after all context segments are prepared.
    Immediate,
}

/// Model-description generation policy consumed by the scheduler planner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationPolicyDescriptor {
    pub trigger: TriggerPolicyDescriptor,
    #[serde(default)]
    pub gen_only_start: GenOnlyStartPolicyDescriptor,
    pub visibility: VisibilityPolicyDescriptor,
    pub termination: TerminationPolicyDescriptor,
    pub feedback: Option<GeneratedImageFeedbackRecipe>,
}

/// Constraint-resolved behavior consumed by scheduler lifecycle planning.
///
/// Keeping these decisions explicit prevents scheduler code from interpreting
/// public constraint names. The profile/compiler resolves this descriptor from
/// the constraint and model policy before admission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationBehaviorDescriptor {
    pub und_decode: bool,
    pub und_tokens: UndTokenAction,
    pub gen_output: bool,
    pub start_gen_after_context: bool,
    pub generated_image_feedback: bool,
    pub continue_after_gen_commit: bool,
    pub finish_after_gen_commit: bool,
}

impl GenerationBehaviorDescriptor {
    pub fn resolve(constraint: GenerationConstraint, policy: &GenerationPolicyDescriptor) -> Self {
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

    pub fn emits_und(&self) -> bool {
        self.und_tokens == UndTokenAction::Emit
    }

    /// The generation branches this resolved behavior requires a runtime to
    /// execute, for admission-time capability gating.
    pub fn capability_needs(
        &self,
        policy: &GenerationPolicyDescriptor,
        context_image_steps: impl IntoIterator<Item = ImageIngestStep>,
    ) -> GenerationCapabilityNeeds {
        let mut needs = GenerationCapabilityNeeds {
            understanding: true,
            ..GenerationCapabilityNeeds::default()
        };
        for step in context_image_steps {
            needs.mark_encode(step);
        }
        if self.gen_output {
            needs.image_generation = true;
        }
        if self.generated_image_feedback
            && let Some(feedback) = &policy.feedback
        {
            for step in feedback.ingest.steps.iter().copied() {
                needs.mark_encode(step);
            }
        }
        needs
    }
}

/// The generation branches a request needs a runtime to execute. Understanding
/// (prompt ingestion and decode) is always required; the remaining branches are
/// set by the resolved behavior, its context image steps, and feedback recipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct GenerationCapabilityNeeds {
    pub understanding: bool,
    pub vision_encode: bool,
    pub latent_encode: bool,
    pub image_generation: bool,
}

impl GenerationCapabilityNeeds {
    fn mark_encode(&mut self, step: ImageIngestStep) {
        match step {
            ImageIngestStep::VaeEncode => self.latent_encode = true,
            ImageIngestStep::VitEncode => self.vision_encode = true,
        }
    }
}

/// Conservative request-level resource declaration produced by compilation.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct GenerationResourceBounds {
    pub context_tokens: usize,
    pub max_kv_tokens: usize,
    pub max_image_latent_units: u64,
    pub max_image_latent_bytes: u64,
    pub max_latent_feature_bytes: u64,
    pub max_vision_feature_bytes: u64,
    pub encoder_cache_keys: Vec<u64>,
    pub generated_feedback_makes_non_replayable: bool,
}

/// Inputs used to derive conservative resources for one generation graph.
pub struct GenerationResourceSpec<'a> {
    pub context: &'a [ContextSegment],
    pub negative_context: &'a [ContextSegment],
    pub behavior: &'a GenerationBehaviorDescriptor,
    pub policy: &'a GenerationPolicyDescriptor,
    pub image: &'a ImageParams,
    pub max_und_tokens: usize,
    pub cache: &'a GenerationCachePolicyDescriptor,
    pub capabilities: &'a GenerationRuntimeCapabilities,
}

/// Worker and scheduler limits needed to compile a bounded generation graph.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct GenerationRuntimeCapabilities {
    pub supports_understanding: bool,
    pub supports_vision_encode: bool,
    pub supports_latent_encode: bool,
    pub supports_image_generation: bool,
    pub max_latent_units: u64,
    pub latent_downsample: u32,
    pub max_vae_grid_tokens: u32,
    pub max_vit_grid_tokens: u32,
    pub max_latent_feature_bytes: u64,
    pub max_vision_feature_bytes: u64,
    pub commit_marker_tokens: u32,
    pub max_cfg_branches: u32,
    pub encoder_cache_entries: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GenerationCapabilityError {
    #[error("runtime_und_execution")]
    Understanding,
    #[error("runtime_vae_encode")]
    LatentEncode,
    #[error("runtime_vit_encode")]
    VisionEncode,
    #[error("runtime_gen_denoise")]
    ImageGeneration,
}

impl GenerationCapabilityError {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Understanding => "runtime_und_execution",
            Self::LatentEncode => "runtime_vae_encode",
            Self::VisionEncode => "runtime_vit_encode",
            Self::ImageGeneration => "runtime_gen_denoise",
        }
    }
}

impl GenerationRuntimeCapabilities {
    /// Whether this runtime covers every branch the request needs. On a gap,
    /// returns the admission-capability name of the first missing branch.
    pub fn covers(
        &self,
        needs: &GenerationCapabilityNeeds,
    ) -> Result<(), GenerationCapabilityError> {
        if needs.understanding && !self.supports_understanding {
            return Err(GenerationCapabilityError::Understanding);
        }
        if needs.latent_encode && !self.supports_latent_encode {
            return Err(GenerationCapabilityError::LatentEncode);
        }
        if needs.vision_encode && !self.supports_vision_encode {
            return Err(GenerationCapabilityError::VisionEncode);
        }
        if needs.image_generation && !self.supports_image_generation {
            return Err(GenerationCapabilityError::ImageGeneration);
        }
        Ok(())
    }
}

impl GenerationResourceBounds {
    pub fn conservative(spec: GenerationResourceSpec<'_>) -> Result<Self, GenerationResourceError> {
        let GenerationResourceSpec {
            context,
            negative_context,
            behavior,
            policy,
            image,
            max_und_tokens,
            cache,
            capabilities,
        } = spec;
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
            Ok(total.saturating_add(ingest_kv_bound(ingest, capabilities)?))
        })?;
        let feedback_kv_per_image = if behavior.generated_image_feedback {
            match policy.feedback.as_ref() {
                Some(feedback) => ingest_kv_bound(&feedback.ingest, capabilities)?,
                None => return Err(GenerationResourceError::MissingFeedback),
            }
        } else {
            0
        };
        let generated_feedback_kv_tokens =
            feedback_kv_per_image.saturating_mul(image.max_images as usize);
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
        if uses_latent_features && capabilities.max_latent_feature_bytes == 0 {
            return Err(GenerationResourceError::MissingRuntimeBound {
                resource: "max_latent_feature_bytes",
            });
        }
        if uses_vision_features && capabilities.max_vision_feature_bytes == 0 {
            return Err(GenerationResourceError::MissingRuntimeBound {
                resource: "max_vision_feature_bytes",
            });
        }
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
            if capabilities.encoder_cache_entries == 0 {
                return Err(GenerationResourceError::MissingRuntimeBound {
                    resource: "encoder_cache_entries",
                });
            }
            if encoder_cache_keys.len() > capabilities.encoder_cache_entries as usize {
                return Err(GenerationResourceError::EncoderCacheCapacity {
                    requested: encoder_cache_keys.len(),
                    available: capabilities.encoder_cache_entries,
                });
            }
        }
        if behavior.gen_output && capabilities.latent_downsample == 0 {
            return Err(GenerationResourceError::MissingRuntimeBound {
                resource: "latent_downsample",
            });
        }
        if behavior.gen_output
            && (!image.width.is_multiple_of(capabilities.latent_downsample)
                || !image.height.is_multiple_of(capabilities.latent_downsample))
        {
            return Err(GenerationResourceError::ImageDimensionAlignment {
                width: image.width,
                height: image.height,
                latent_downsample: capabilities.latent_downsample,
            });
        }
        if behavior.gen_output && capabilities.max_latent_units == 0 {
            return Err(GenerationResourceError::MissingRuntimeBound {
                resource: "max_latent_units",
            });
        }
        if behavior.gen_output && capabilities.max_cfg_branches == 0 {
            return Err(GenerationResourceError::MissingRuntimeBound {
                resource: "max_cfg_branches",
            });
        }
        let latent_downsample = capabilities.latent_downsample.max(1);
        let requested_latent_units = u64::from(image.width / latent_downsample)
            .saturating_mul(u64::from(image.height / latent_downsample));
        if behavior.gen_output && requested_latent_units > capabilities.max_latent_units {
            return Err(GenerationResourceError::LatentCapacity {
                requested: requested_latent_units,
                available: capabilities.max_latent_units,
            });
        }
        let image_latent_bytes = if behavior.gen_output {
            if capabilities.max_vae_grid_tokens == 0 || capabilities.max_latent_feature_bytes == 0 {
                return Err(GenerationResourceError::MissingRuntimeBound {
                    resource: "max_latent_feature_bytes",
                });
            }
            let bytes_per_unit = capabilities
                .max_latent_feature_bytes
                .div_ceil(u64::from(capabilities.max_vae_grid_tokens));
            requested_latent_units.saturating_mul(bytes_per_unit)
        } else {
            0
        };
        let requested_cfg_branches = u64::from(image.cfg_branch_count());
        if behavior.gen_output && requested_cfg_branches > u64::from(capabilities.max_cfg_branches)
        {
            return Err(GenerationResourceError::CfgBranchCapacity {
                requested: requested_cfg_branches,
                available: capabilities.max_cfg_branches,
            });
        }

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
                capabilities.max_latent_feature_bytes
            } else {
                0
            },
            max_vision_feature_bytes: if uses_vision_features {
                capabilities.max_vision_feature_bytes
            } else {
                0
            },
            encoder_cache_keys,
            generated_feedback_makes_non_replayable: behavior.generated_image_feedback,
        })
    }

    pub fn validate_covers(&self, required: &Self) -> Result<(), GenerationResourceError> {
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
        if required.generated_feedback_makes_non_replayable
            && !self.generated_feedback_makes_non_replayable
        {
            return Err(GenerationResourceError::MissingNonReplayableDeclaration);
        }
        Ok(())
    }
}

fn ingest_kv_bound(
    ingest: &ImageIngestRecipe,
    capabilities: &GenerationRuntimeCapabilities,
) -> Result<usize, GenerationResourceError> {
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
                ImageIngestStep::VaeEncode => capabilities.max_vae_grid_tokens,
                ImageIngestStep::VitEncode => capabilities.max_vit_grid_tokens,
            };
            let step_bound = kv_effect_bound(effect, fallback, step.as_str())?;
            Ok(total.saturating_add(step_bound))
        })
}

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
    fn as_str(self) -> &'static str {
        match self {
            Self::VaeEncode => "vae_encode",
            Self::VitEncode => "vit_encode",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GenerationResourceError {
    #[error("image ingest declares {steps} steps but {effects} KV effects")]
    ImageIngestKvArity { steps: usize, effects: usize },
    #[error("{operation} has a worker-defined KV effect but the runtime declares no bound")]
    UnboundedImageKv { operation: &'static str },
    #[error("generated image continuation requires a feedback resource recipe")]
    MissingFeedback,
    #[error("image generation requires the runtime to declare {resource}")]
    MissingRuntimeBound { resource: &'static str },
    #[error(
        "image dimensions {width}x{height} must be divisible by the runtime latent downsample factor {latent_downsample}"
    )]
    ImageDimensionAlignment {
        width: u32,
        height: u32,
        latent_downsample: u32,
    },
    #[error("{resource} overflowed while computing the request resource bound")]
    ResourceOverflow { resource: &'static str },
    #[error("requested image latent units ({requested}) exceed runtime capacity ({available})")]
    LatentCapacity { requested: u64, available: u64 },
    #[error("requested CFG branches ({requested}) exceed runtime capacity ({available})")]
    CfgBranchCapacity { requested: u64, available: u32 },
    #[error("requested encoder-cache entries ({requested}) exceed runtime capacity ({available})")]
    EncoderCacheCapacity { requested: usize, available: u32 },
    #[error("declared {resource} bound ({declared}) is below the required bound ({required})")]
    DeclaredBoundTooSmall {
        resource: &'static str,
        declared: u64,
        required: u64,
    },
    #[error("generated image feedback must be declared non-replayable")]
    MissingNonReplayableDeclaration,
}

/// Scheduler-relevant prefix-cache behavior resolved during compilation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationCachePolicyDescriptor {
    pub read: bool,
    pub write: bool,
    pub isolation_key: Option<u64>,
}

impl Default for GenerationCachePolicyDescriptor {
    fn default() -> Self {
        Self {
            read: true,
            write: true,
            isolation_key: None,
        }
    }
}

/// Canonical scheduler-facing generation request.
///
/// This is pure data. Event channels, streams, engine handles, worker state,
/// and scheduler cursors belong to submission and runtime layers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationRequest {
    pub request_id: RequestId,
    pub context: Vec<ContextSegment>,
    pub negative_context: Vec<ContextSegment>,
    pub constraint: GenerationConstraint,
    pub behavior: GenerationBehaviorDescriptor,
    pub sampling: SamplingParams,
    pub image: ImageParams,
    pub max_und_tokens: usize,
    pub stop_strings: Vec<String>,
    pub stop_token_ids: Vec<u32>,
    pub priority: i32,
    pub cache: GenerationCachePolicyDescriptor,
    pub policy: GenerationPolicyDescriptor,
    pub resources: GenerationResourceBounds,
}

impl GenerationRequest {
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

    pub fn prompt_token_count(&self) -> usize {
        self.context
            .iter()
            .map(|segment| match segment {
                ContextSegment::UndTokens { token_ids, .. } => token_ids.len(),
                ContextSegment::Image { .. } => 0,
            })
            .sum()
    }

    pub fn context_image_count(&self) -> usize {
        self.context
            .iter()
            .filter(|segment| matches!(segment, ContextSegment::Image { .. }))
            .count()
    }

    pub fn validate(&self) -> Result<(), GenerationRequestError> {
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
        if self
            .negative_context
            .iter()
            .any(|segment| matches!(segment, ContextSegment::Image { .. }))
        {
            return Err(GenerationRequestError::ImageInNegativeContext);
        }
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
        if self.stop_strings.iter().any(String::is_empty) {
            return Err(GenerationRequestError::EmptyStopString);
        }
        Ok(())
    }

    pub fn validate_resources(
        &self,
        capabilities: &GenerationRuntimeCapabilities,
    ) -> Result<(), GenerationResourceError> {
        let required = GenerationResourceBounds::conservative(GenerationResourceSpec {
            context: &self.context,
            negative_context: &self.negative_context,
            behavior: &self.behavior,
            policy: &self.policy,
            image: &self.image,
            max_und_tokens: self.max_und_tokens,
            cache: &self.cache,
            capabilities,
        })?;
        self.resources.validate_covers(&required)
    }
}

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

fn validate_kv_effect(effect: ImageKvEffect) -> Result<(), GenerationRequestError> {
    if matches!(
        effect,
        ImageKvEffect::Exact { tokens: 0 } | ImageKvEffect::Bounded { max_tokens: 0 }
    ) {
        return Err(GenerationRequestError::ZeroImageKvBound);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GenerationRequestError {
    #[error("generation context must contain at least one segment")]
    EmptyContext,
    #[error("image ingest declares {steps} steps but {effects} KV effects")]
    ImageIngestKvArity { steps: usize, effects: usize },
    #[error("max_und_tokens must be positive")]
    ZeroMaxUndTokens,
    #[error("resolved generation behavior does not match constraint and policy")]
    BehaviorPolicyMismatch,
    #[error("image context segment has an empty ingest recipe")]
    EmptyImageIngestRecipe,
    #[error("negative context may contain only Und token segments")]
    ImageInNegativeContext,
    #[error("image segment payload must not be empty")]
    EmptyImagePayload,
    #[error(
        "image segment placement does not match ordered context: expected {expected}, got {actual}"
    )]
    ImagePlacementMismatch { expected: usize, actual: usize },
    #[error("image ingest and feedback recipes must consume at least one logical position")]
    ZeroImageLogicalPositions,
    #[error("bounded image KV effects must be positive")]
    ZeroImageKvBound,
    #[error("generation trigger patterns must not be empty")]
    EmptyTriggerPattern,
    #[error("gen_only request cannot produce an image under the resolved policy")]
    GenOnlyCannotProduceImage,
    #[error("default generation continuation requires a feedback recipe")]
    MissingFeedbackRecipe,
    #[error("default generation feedback requires a continuation token")]
    IncompleteFeedbackRecipe,
    #[error("max-token policy must terminate at the declared request bound")]
    NonTerminalMaxTokensPolicy,
    #[error("round-close trigger policy requires terminal round closure")]
    NonTerminalRoundClosePolicy,
    #[error("resource context-token bound mismatch: expected {expected}, got {actual}")]
    ContextTokenBoundMismatch { expected: usize, actual: usize },
    #[error("max_kv_tokens ({max_kv_tokens}) is below context tokens ({context_tokens})")]
    MaxKvBelowContext {
        context_tokens: usize,
        max_kv_tokens: usize,
    },
    #[error(
        "encoder-cache key declaration does not match context: expected {expected:?}, got {actual:?}"
    )]
    EncoderCacheKeysMismatch {
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
    #[error("stop strings must not be empty")]
    EmptyStopString,
    #[error("invalid sampling parameters: {0}")]
    InvalidSampling(#[source] SamplingParamsError),
    #[error("invalid image parameters: {0}")]
    InvalidImage(#[source] ImageParamsError),
    #[error("min_tokens ({min_tokens}) exceeds max_und_tokens ({max_und_tokens})")]
    MinTokensExceedsMaximum {
        min_tokens: usize,
        max_und_tokens: usize,
    },
}

impl Default for GenerationPolicyDescriptor {
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

    fn runtime_capabilities() -> GenerationRuntimeCapabilities {
        GenerationRuntimeCapabilities {
            supports_understanding: true,
            supports_vision_encode: true,
            supports_latent_encode: true,
            supports_image_generation: true,
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
        request.resources = GenerationResourceBounds::conservative(GenerationResourceSpec {
            context: &request.context,
            negative_context: &request.negative_context,
            behavior: &request.behavior,
            policy: &request.policy,
            image: &request.image,
            max_und_tokens: request.max_und_tokens,
            cache: &request.cache,
            capabilities: &runtime_capabilities(),
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
            immediate.capability_needs(&immediate_policy, []),
            GenerationCapabilityNeeds {
                understanding: true,
                image_generation: true,
                ..GenerationCapabilityNeeds::default()
            }
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
        let bounds = GenerationResourceBounds::conservative(GenerationResourceSpec {
            context: &request.context,
            negative_context: &request.negative_context,
            behavior: &request.behavior,
            policy: &request.policy,
            image: &request.image,
            max_und_tokens: request.max_und_tokens,
            cache: &request.cache,
            capabilities: &runtime_capabilities(),
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

        let bounds = GenerationResourceBounds::conservative(GenerationResourceSpec {
            context: &request.context,
            negative_context: &request.negative_context,
            behavior: &request.behavior,
            policy: &request.policy,
            image: &request.image,
            max_und_tokens: request.max_und_tokens,
            cache: &request.cache,
            capabilities: &runtime_capabilities(),
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
        let mut capabilities = runtime_capabilities();
        capabilities.max_vit_grid_tokens = 0;
        assert_eq!(
            GenerationResourceBounds::conservative(GenerationResourceSpec {
                context: &request.context,
                negative_context: &request.negative_context,
                behavior: &request.behavior,
                policy: &request.policy,
                image: &request.image,
                max_und_tokens: request.max_und_tokens,
                cache: &request.cache,
                capabilities: &capabilities,
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
            GenerationResourceBounds::conservative(GenerationResourceSpec {
                context: &invalid_request.context,
                negative_context: &invalid_request.negative_context,
                behavior: &invalid_request.behavior,
                policy: &invalid_request.policy,
                image: &invalid_request.image,
                max_und_tokens: invalid_request.max_und_tokens,
                cache: &invalid_request.cache,
                capabilities: &runtime_capabilities(),
            }),
            Err(GenerationResourceError::ImageIngestKvArity {
                steps: 2,
                effects: 1,
            })
        );

        let mut capabilities = runtime_capabilities();
        capabilities.max_latent_units = 1_023;
        assert!(matches!(
            request.validate_resources(&capabilities),
            Err(GenerationResourceError::LatentCapacity { .. })
        ));
    }

    #[test]
    fn resource_compilation_requires_runtime_aligned_image_dimensions() {
        let request = complete_request();
        let mut capabilities = runtime_capabilities();
        capabilities.latent_downsample = 24;

        assert_eq!(
            request.validate_resources(&capabilities),
            Err(GenerationResourceError::ImageDimensionAlignment {
                width: request.image.width,
                height: request.image.height,
                latent_downsample: 24,
            })
        );
    }

    #[test]
    fn declared_resources_must_cover_the_capability_derived_envelope() {
        let mut request = complete_request();
        request.resources.max_kv_tokens -= 1;
        assert!(matches!(
            request.validate_resources(&runtime_capabilities()),
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
