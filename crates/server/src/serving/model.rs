//! Load-time model resolution and model-owned request tokenization.
//!
//! [`ResolvedModel`] binds tokenizer assets, generation policy, geometry, and
//! output processing. Its [`ResolvedModel::tokenize`] method lowers
//! [`GenerateReqInput`] into [`TokenizedGenerateReqInput`].

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;

use crate::config::EngineSettings;
use crate::profile::assets::{ResolvedModelFiles, resolve_model_file};
use crate::profile::omni::bagel::BagelProfile;
use crate::profile::omni::sensenova::SenseNovaProfile;
use crate::profile::tokenizer::{DynTokenizer, HuggingFaceTokenizer, TokenizerError};
use crate::profile::{
    CommonModelProfile, ModelDescription, ModelIdentity, ModelProfile, ProfileOverrides,
};
use thiserror::Error;
use uniserve_core::{
    ContextSegment as CoreContextSegment, GenerationBehaviorDescriptor,
    GenerationCachePolicyDescriptor, GenerationConstraint, GenerationFeatures, GenerationLimits,
    GenerationPolicyDescriptor, GenerationRequest, GenerationResourceBounds, ImageParams,
    RequestId, SamplingParams, UndVisibility,
};

use crate::serving::chat::{
    ChatRequest, ChatTemplateLoadOptions, HfChatRenderer, Qwen3ChatOutputProcessor,
};
use crate::serving::input::{
    GenerateReqInput, ModelEventIdentity, OutputDetail, OutputProcessorPolicy, PromptInput,
    TokenizedGenerateReqInput,
};
use crate::serving::text::{SamplingHints, TextDecodeOptions, resolve_max_tokens};
use crate::serving::{
    CacheAccounting, ResourceAccounting, Result, ServeError, cache_isolation_key,
};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
/// Public API endpoint supported by a resolved model.
pub enum ServedEndpoint {
    /// OpenAI-compatible chat-completions endpoint.
    ChatCompletions,
    /// OpenAI-compatible image-generations endpoint.
    ImageGenerations,
    /// OpenAI-compatible video-generations endpoint.
    VideoGenerations,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
/// Input or output modality supported by a resolved model.
pub enum ServedModality {
    /// Tokenized or decoded text.
    Text,
    /// Encoded or generated images.
    Image,
    /// Encoded or generated video.
    Video,
    /// Encoded or generated audio.
    Audio,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
/// Optional serving feature supported by a resolved model.
pub enum ServedFeature {
    /// Incremental response streaming.
    Streaming,
    /// Token and resource usage reporting.
    Usage,
    /// Prompt and generated-token log probabilities.
    Logprobs,
    /// Structured reasoning output.
    Reasoning,
    /// Structured function-tool calls.
    ToolCalling,
    /// Repeated transitions between generated modalities.
    RepeatedInterleave,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
/// Sampling control accepted by a resolved model.
pub enum ServedSamplingControl {
    /// Deterministic highest-probability sampling.
    Greedy,
    /// Temperature scaling.
    Temperature,
    /// Top-k candidate truncation.
    TopK,
    /// Nucleus candidate truncation.
    TopP,
    /// Minimum relative-probability truncation.
    MinP,
    /// Multiplicative repetition penalty.
    RepetitionPenalty,
    /// Frequency-based repetition penalty.
    FrequencyPenalty,
    /// Presence-based repetition penalty.
    PresencePenalty,
    /// Per-token additive logit adjustment.
    LogitBias,
    /// Explicit token allowlist.
    AllowedTokenIds,
    /// Text-sequence denylist.
    BadWords,
    /// Minimum generated-token count.
    MinTokens,
    /// Candidate log-probability reporting.
    Logprobs,
    /// Additional stop-token identifiers.
    StopTokenIds,
    /// End-of-sequence stopping policy.
    Eos,
    /// Decoded stop strings.
    StopStrings,
}

impl ServedSamplingControl {
    /// Sampling controls supported by token-generating model profiles.
    pub const ALL: [Self; 16] = [
        Self::Greedy,
        Self::Temperature,
        Self::TopK,
        Self::TopP,
        Self::MinP,
        Self::RepetitionPenalty,
        Self::FrequencyPenalty,
        Self::PresencePenalty,
        Self::LogitBias,
        Self::AllowedTokenIds,
        Self::BadWords,
        Self::MinTokens,
        Self::Logprobs,
        Self::StopTokenIds,
        Self::Eos,
        Self::StopStrings,
    ];
}

/// Exact public route declaration for one load-bound model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSupport {
    /// Public endpoints served by the model.
    pub endpoints: Vec<ServedEndpoint>,
    /// Input modalities accepted by the model.
    pub input_modalities: Vec<ServedModality>,
    /// Output modalities produced by the model.
    pub output_modalities: Vec<ServedModality>,
    /// Optional serving behaviors implemented by the model path.
    pub features: Vec<ServedFeature>,
    /// Sampling controls accepted by the model path.
    pub sampling_controls: Vec<ServedSamplingControl>,
}

/// The closed load-bound model owner.
pub enum ResolvedModel {
    /// Text-generation model with chat support.
    Text(Qwen3Desc),
    /// Multimodal understanding and image-generation model.
    Omni(OmniDesc),
    /// Media-generation model.
    Media(MiniMaxH3Desc),
}

/// Resolved multimodal model description.
pub enum OmniDesc {
    /// SenseNova multimodal model.
    SenseNova(SenseNovaDesc),
    /// Bagel multimodal model.
    Bagel(BagelDesc),
}

/// Typed assets awaiting validation against the running worker limits.
pub enum ResolvedAssets {
    /// Text-model assets.
    Text {
        /// Resolved common model profile.
        profile: CommonModelProfile,
        /// Tokenizer bound to the model vocabulary.
        tokenizer: DynTokenizer,
        /// Renderer bound to the model chat template.
        renderer: HfChatRenderer,
    },
    /// Multimodal-model assets.
    Omni {
        /// Resolved common model profile.
        profile: CommonModelProfile,
        /// Tokenizer bound to the model vocabulary.
        tokenizer: DynTokenizer,
        /// Renderer bound to the model chat template.
        renderer: HfChatRenderer,
        /// Profile-specific image preprocessing and generation policy.
        preprocessing: OmniPreprocessing,
    },
    /// Media-generation model assets.
    Media {
        /// Resolved common model profile.
        profile: CommonModelProfile,
        /// Tokenizer bound to the model vocabulary.
        tokenizer: DynTokenizer,
        /// Maximum generated video duration in seconds.
        max_video_seconds: f64,
        /// Number of scheduled predictions in the validated checkpoint contract.
        denoise_steps: u32,
        /// Whether the resolved contract admits one image reference.
        image_references: bool,
    },
}

/// Profile-specific multimodal preprocessing implementation.
pub enum OmniPreprocessing {
    /// SenseNova image preprocessing and generation policy.
    SenseNova(SenseNovaProfile),
    /// Bagel image preprocessing and generation policy.
    Bagel(BagelProfile),
}

#[derive(Debug, Error)]
/// Failure while resolving a configured model and its assets.
pub enum ModelResolutionError {
    /// Required numerical checkpoint metadata is missing or contradictory.
    #[error("invalid media checkpoint contract: {0}")]
    MediaContract(String),
    /// Model files or profile metadata cannot be resolved.
    #[error(transparent)]
    Assets(#[from] crate::profile::assets::Error),
    /// The model tokenizer cannot be loaded.
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    /// The model chat renderer cannot be initialized.
    #[error(transparent)]
    Chat(#[from] crate::serving::chat::Error),
    /// A resolved asset path cannot be prepared for worker access.
    #[error("model asset path `{path}` could not be prepared")]
    Io {
        /// Asset path that could not be prepared.
        path: PathBuf,
        /// Underlying filesystem error.
        #[source]
        source: std::io::Error,
    },
    /// The running worker lacks a feature required by the selected profile.
    #[error("configured model description `{description}` requires worker feature `{feature}`")]
    MissingFeature {
        /// Stable model-profile identifier.
        description: &'static str,
        /// Required runtime feature.
        feature: GenerationFeatures,
    },
}

impl ResolvedAssets {
    /// Resolves assets and validates the selected model against engine capabilities.
    pub(crate) async fn load(
        config: &crate::Config,
    ) -> std::result::Result<Self, ModelResolutionError> {
        let served_name = config
            .served_model_name
            .clone()
            .unwrap_or_else(|| config.model.clone());
        if config.model_description == ModelDescription::MiniMaxH3 {
            let tokenizer_path =
                resolve_model_file(&config.model, "tokenizer/tokenizer.json").await?;
            let tokenizer: DynTokenizer = Arc::new(HuggingFaceTokenizer::new(&tokenizer_path)?);
            let mut profile = ModelProfile::minimax_h3(&served_name);
            profile.common_mut().context_limits.max_model_tokens =
                Some(config.engine.max_model_len.unwrap_or(16_384));
            let ModelProfile::MiniMaxH3(profile) = profile else {
                unreachable!("MiniMax H3 construction returns its matching closed variant")
            };
            let denoise_steps = config
                .model_contract
                .as_ref()
                .and_then(|contract| contract.get("denoise_steps"))
                .and_then(serde_json::Value::as_u64)
                .and_then(|steps| u32::try_from(steps).ok())
                .filter(|steps| *steps > 0)
                .ok_or_else(|| ModelResolutionError::MediaContract(
                    "resolve the checkpoint with the installed worker before building the server".to_owned()
                ))?;
            return Ok(Self::Media {
                profile,
                tokenizer,
                max_video_seconds: config.engine.max_video_seconds,
                denoise_steps,
                image_references: config.model_contract.as_ref()
                    .and_then(|contract| contract.get("references"))
                    == Some(&serde_json::json!({"max": 1, "kinds": ["image"]})),
            });
        }

        let files = ResolvedModelFiles::new(&config.model).await?;
        let tokenizer: DynTokenizer = Arc::new(HuggingFaceTokenizer::new(&files.tokenizer_path)?);
        let configuration = ProfileOverrides {
            chat_template_override: config.chat_template.clone(),
            max_model_tokens: config.engine.max_model_len,
        };
        let mut profile = ModelProfile::resolve(
            config.model_description,
            &served_name,
            &files,
            &configuration,
            tokenizer.as_ref(),
        )?;
        let max_model_tokens = config
            .engine
            .max_model_len
            .or(profile.common().context_limits.max_model_tokens)
            .unwrap_or(EngineSettings::DEFAULT_MAX_MODEL_LEN);
        profile.common_mut().context_limits.max_model_tokens = Some(max_model_tokens);
        let renderer = HfChatRenderer::load(
            &files,
            ChatTemplateLoadOptions {
                chat_template_content_format: config.chat_template_content_format,
                chat_template: config.chat_template.clone(),
                default_chat_template_kwargs: config
                    .default_chat_template_kwargs
                    .clone()
                    .unwrap_or_default(),
            },
            None,
        )?;
        Ok(match profile {
            ModelProfile::Qwen3(profile) => Self::Text {
                profile,
                tokenizer,
                renderer,
            },
            ModelProfile::SenseNova(profile) => Self::Omni {
                profile: profile.common,
                tokenizer,
                renderer,
                preprocessing: OmniPreprocessing::SenseNova(profile.preprocessing),
            },
            ModelProfile::Bagel(profile) => Self::Omni {
                profile: profile.common,
                tokenizer,
                renderer,
                preprocessing: OmniPreprocessing::Bagel(profile.preprocessing),
            },
            ModelProfile::MiniMaxH3(_) => unreachable!("media assets return before file loading"),
        })
    }

    /// Builds a resolved model from local assets and an engine snapshot.
    pub fn from_files(
        description: ModelDescription,
        served_name: &str,
        files: &ResolvedModelFiles,
        configuration: &ProfileOverrides,
        tokenizer: DynTokenizer,
        renderer: HfChatRenderer,
    ) -> std::result::Result<Self, ModelResolutionError> {
        let profile = ModelProfile::resolve(
            description,
            served_name,
            files,
            configuration,
            tokenizer.as_ref(),
        )?;
        Ok(match profile {
            ModelProfile::Qwen3(profile) => Self::Text {
                profile,
                tokenizer,
                renderer,
            },
            ModelProfile::SenseNova(profile) => Self::Omni {
                profile: profile.common,
                tokenizer,
                renderer,
                preprocessing: OmniPreprocessing::SenseNova(profile.preprocessing),
            },
            ModelProfile::Bagel(profile) => Self::Omni {
                profile: profile.common,
                tokenizer,
                renderer,
                preprocessing: OmniPreprocessing::Bagel(profile.preprocessing),
            },
            ModelProfile::MiniMaxH3(profile) => Self::Media {
                profile,
                tokenizer,
                max_video_seconds: 15.0,
                denoise_steps: 4,
                image_references: false,
            },
        })
    }

    /// Returns the model's common capability profile.
    pub(crate) fn profile(&self) -> &CommonModelProfile {
        match self {
            Self::Text { profile, .. }
            | Self::Omni { profile, .. }
            | Self::Media { profile, .. } => profile,
        }
    }

    /// Returns the maximum combined context and generated token count.
    pub(crate) fn max_model_tokens(&self) -> u32 {
        self.profile()
            .context_limits
            .max_model_tokens
            .unwrap_or(EngineSettings::DEFAULT_MAX_MODEL_LEN)
    }

    /// Returns the request-state capacity advertised by the engine.
    pub(crate) fn request_slot_capacity(&self) -> usize {
        if matches!(self, Self::Media { .. }) {
            EngineSettings::MEDIA_IPC_SLOT_CAP
        } else {
            1 << 20
        }
    }

    /// Returns multimodal generation control tokens, when supported.
    pub(crate) fn generation_controls(&self) -> Option<&crate::profile::omni::GenerationControls> {
        match self {
            Self::Omni {
                preprocessing: OmniPreprocessing::SenseNova(value),
                ..
            } => Some(&value.controls),
            Self::Omni {
                preprocessing: OmniPreprocessing::Bagel(value),
                ..
            } => Some(&value.controls),
            Self::Text { .. } | Self::Media { .. } => None,
        }
    }

    /// Builds the engine runtime profile required by this model.
    pub(crate) fn runtime_profile(
        &self,
        model_dtype: uniserve_core::ModelDtype,
    ) -> uniserve_engine::RuntimeProfile {
        match self {
            Self::Text { .. } => uniserve_engine::RuntimeProfile::ar(model_dtype),
            Self::Media { .. } => uniserve_engine::RuntimeProfile::diffusion(model_dtype),
            Self::Omni {
                preprocessing: OmniPreprocessing::SenseNova(_),
                ..
            } => uniserve_engine::RuntimeProfile::umm(
                model_dtype,
                SenseNovaProfile::runtime_limits(model_dtype),
            ),
            Self::Omni {
                preprocessing: OmniPreprocessing::Bagel(_),
                ..
            } => uniserve_engine::RuntimeProfile::umm(
                model_dtype,
                BagelProfile::runtime_limits(model_dtype),
            ),
        }
    }
}

/// Text chat description: HF tokenization + chat template + fixed Qwen3 parser
/// policy.
pub struct Qwen3Desc {
    identity: ModelIdentity,
    tokenizer: DynTokenizer,
    renderer: HfChatRenderer,
    hints: SamplingHints,
    limits: GenerationLimits,
    sampling_controls: Vec<ServedSamplingControl>,
    logprobs_supported: bool,
    parse_reasoning: bool,
}

/// SenseNova omni description: image input, text output, image output, and
/// repeated interleave through description-owned framing/ingest/output filter.
pub struct SenseNovaDesc {
    identity: ModelIdentity,
    tokenizer: DynTokenizer,
    renderer: HfChatRenderer,
    preprocessing: SenseNovaProfile,
    limits: GenerationLimits,
    sampling_controls: Vec<ServedSamplingControl>,
    default_max_output_tokens: Option<u32>,
    max_model_tokens: u32,
}

/// BAGEL omni description: image input, text output, and image output.
pub struct BagelDesc {
    identity: ModelIdentity,
    tokenizer: DynTokenizer,
    renderer: HfChatRenderer,
    preprocessing: BagelProfile,
    limits: GenerationLimits,
    sampling_controls: Vec<ServedSamplingControl>,
    default_max_output_tokens: Option<u32>,
    max_model_tokens: u32,
}

/// Resolved MiniMax H3 video-generation description.
pub struct MiniMaxH3Desc {
    identity: ModelIdentity,
    tokenizer: DynTokenizer,
    max_prompt_tokens: u32,
    max_video_seconds: f64,
    denoise_steps: u32,
    image_references: bool,
}

impl ResolvedModel {
    /// Resolves the configured assets into a validated model description.
    ///
    /// The typed description selects the variant; required description-owned
    /// assets are checked before construction.
    pub fn resolve(
        assets: ResolvedAssets,
        limits: GenerationLimits,
        sampling_controls: Vec<ServedSamplingControl>,
        max_model_tokens: u32,
        parse_reasoning: bool,
    ) -> Result<Self> {
        // Resolution validates each asset family against the runtime features
        // required by its public serving contract.
        match assets {
            ResolvedAssets::Text {
                profile,
                tokenizer,
                renderer,
            } => {
                validate_runtime_features(
                    &profile.identity,
                    &limits,
                    GenerationFeatures::UNDERSTANDING,
                )?;
                let hints = sampling_hints(&profile, max_model_tokens);
                let logprobs_supported =
                    sampling_controls.contains(&ServedSamplingControl::Logprobs);
                Ok(Self::Text(Qwen3Desc {
                    identity: profile.identity,
                    tokenizer,
                    renderer,
                    hints,
                    limits,
                    sampling_controls,
                    logprobs_supported,
                    parse_reasoning,
                }))
            }
            ResolvedAssets::Omni {
                profile,
                tokenizer,
                renderer,
                preprocessing,
            } => {
                let (generation_policy, image_ingest) = match &preprocessing {
                    OmniPreprocessing::SenseNova(value) => {
                        (&value.generation_policy, &value.image_ingest)
                    }
                    OmniPreprocessing::Bagel(value) => {
                        (&value.generation_policy, &value.image_ingest)
                    }
                };
                validate_runtime_features(
                    &profile.identity,
                    &limits,
                    configured_omni_needs(generation_policy, image_ingest),
                )?;
                let default_max_output_tokens = profile
                    .context_limits
                    .max_output_tokens
                    .or(profile.generation_defaults.max_output_tokens);
                // The preprocessing profile selects the concrete multimodal
                // descriptor while sharing the validated identity and limits.
                Ok(Self::Omni(match preprocessing {
                    OmniPreprocessing::SenseNova(preprocessing) => {
                        OmniDesc::SenseNova(SenseNovaDesc {
                            identity: profile.identity,
                            tokenizer,
                            renderer,
                            preprocessing,
                            limits,
                            sampling_controls,
                            default_max_output_tokens,
                            max_model_tokens,
                        })
                    }
                    OmniPreprocessing::Bagel(preprocessing) => OmniDesc::Bagel(BagelDesc {
                        identity: profile.identity,
                        tokenizer,
                        renderer,
                        preprocessing,
                        limits,
                        sampling_controls,
                        default_max_output_tokens,
                        max_model_tokens,
                    }),
                }))
            }
            // Media assets carry all geometry limits needed by request lowering.
            ResolvedAssets::Media {
                profile,
                tokenizer,
                max_video_seconds,
                denoise_steps,
                image_references,
            } => Ok(Self::Media(MiniMaxH3Desc {
                identity: profile.identity,
                tokenizer,
                max_prompt_tokens: max_model_tokens,
                max_video_seconds,
                denoise_steps,
                image_references,
            })),
        }
    }

    /// Returns the model identity used by `/v1/models` and public events.
    pub fn served_identity(&self) -> &ModelIdentity {
        match self {
            Self::Text(d) => &d.identity,
            Self::Omni(OmniDesc::SenseNova(d)) => &d.identity,
            Self::Omni(OmniDesc::Bagel(d)) => &d.identity,
            Self::Media(d) => &d.identity,
        }
    }

    /// Returns the public served-model name.
    pub fn served_model_name(&self) -> &str {
        &self.served_identity().served_name
    }

    /// Public duration, geometry and prompt limits from the serving description.
    pub fn video_capabilities(&self) -> serde_json::Value {
        match self {
            Self::Media(description) => {
                let default_seconds = description.max_video_seconds.min(5.0);
                let mut suggested_seconds = vec![default_seconds];
                if description.max_video_seconds > default_seconds {
                    suggested_seconds.push(description.max_video_seconds);
                }
                serde_json::json!({
                    "tasks": ["t2va"],
                    "default_seconds": default_seconds,
                    "max_seconds": description.max_video_seconds,
                    "suggested_seconds": suggested_seconds,
                    "min_frames": 22, "fps": 24, "width": 1344, "height": 768,
                    "max_prompt_tokens": description.max_prompt_tokens,
                    "request_fields": ["model", "prompt", "seconds", "seed", "steps"],
                    "default_steps": description.denoise_steps + 1,
                    "supported_steps": [description.denoise_steps + 1],
                    "guidance_scale": 1.0,
                })
            }
            _ => serde_json::Value::Null,
        }
    }

    /// Capability comes from the resolved checkpoint, never its public alias.
    pub fn supports_image_references(&self) -> bool {
        matches!(self, Self::Media(description) if description.image_references)
    }

    /// Resolves and validates requested video dimensions and frame count.
    pub fn resolve_video_request_geometry(
        &self,
        request_id: &crate::serving::ServeRequestId,
        prompt: &str,
        seconds: f64,
        steps: Option<u32>,
    ) -> Result<(uniserve_core::MediaGeometry, Vec<u32>)> {
        let Self::Media(description) = self else {
            return Err(ServeError::UnsupportedFeature {
                request_id: request_id.clone(),
                feature: "video_generation",
            });
        };
        // Fixed checkpoint plans and their precomputed modulation products are
        // one numerical recipe. Never silently truncate or substitute its grid.
        let grid_points = description.denoise_steps + 1;
        if steps.is_some_and(|requested| requested != grid_points) {
            return Err(ServeError::Tokenize {
                request_id: request_id.clone(),
                source: crate::serving::TokenizeError::Invalid(format!(
                    "steps must equal the served checkpoint recipe ({grid_points} grid points)"
                )),
            });
        }
        // Tokenize and bound the prompt before deriving any media allocation.
        let prompt_token_ids = description
            .tokenizer
            .encode(prompt, false)
            .map_err(|source| ServeError::Tokenize {
                request_id: request_id.clone(),
                source: crate::serving::TokenizeError::Tokenizer(source),
            })?;
        if prompt_token_ids.is_empty() {
            return Err(ServeError::Tokenize {
                request_id: request_id.clone(),
                source: crate::serving::TokenizeError::Invalid(
                    "video prompt must contain at least one token".to_string(),
                ),
            });
        }
        if prompt_token_ids.len() > description.max_prompt_tokens as usize {
            return Err(ServeError::ContextLengthExceeded {
                request_id: request_id.clone(),
                prompt_tokens: prompt_token_ids.len(),
                max_tokens: description.max_prompt_tokens,
            });
        }
        // Duration is a public floating-point input and must be finite before
        // conversion to the fixed-width frame protocol.
        if !seconds.is_finite() || seconds <= 0.0 || seconds > description.max_video_seconds {
            return Err(ServeError::Tokenize {
                request_id: request_id.clone(),
                source: crate::serving::TokenizeError::Invalid(format!(
                    "video duration must be finite, positive, and at most {} seconds",
                    description.max_video_seconds
                )),
            });
        }
        let raw_frames = (seconds * 24.0).round();
        if raw_frames < 1.0 || raw_frames > f64::from(u32::MAX - 16) {
            return Err(ServeError::Tokenize {
                request_id: request_id.clone(),
                source: crate::serving::TokenizeError::Invalid(
                    "video duration cannot be represented by the configuration".to_string(),
                ),
            });
        }
        // H3 media geometry uses frame counts congruent to five modulo seventeen.
        let raw_frames = raw_frames as u32;
        let frame_count = raw_frames + (5 + 17 - raw_frames % 17) % 17;
        if frame_count < 22 {
            return Err(ServeError::Tokenize {
                request_id: request_id.clone(),
                source: crate::serving::TokenizeError::Invalid(
                    "video duration is shorter than the supported media geometry".to_string(),
                ),
            });
        }
        let prompt_tokens =
            u32::try_from(prompt_token_ids.len()).map_err(|_| ServeError::Tokenize {
                request_id: request_id.clone(),
                source: crate::serving::TokenizeError::Invalid(
                    "video prompt token count exceeds the protocol width".to_string(),
                ),
            })?;
        // Reconstruction windows are part of the model's temporal geometry.
        let video_units = (frame_count - 5) / 17;
        Ok((
            uniserve_core::MediaGeometry {
                frame_count,
                video_units,
                prompt_tokens,
                denoise_steps: description.denoise_steps,
            },
            prompt_token_ids,
        ))
    }

    /// Builds the model identity stamped onto accepted events.
    pub fn event_identity(&self) -> ModelEventIdentity {
        let identity = self.served_identity();
        ModelEventIdentity {
            served_name: identity.served_name.clone(),
            description: identity.description.id().to_string(),
        }
    }

    /// Returns whether the model supports image output.
    pub fn supports_image_output(&self) -> bool {
        matches!(self, Self::Omni(_))
    }

    /// Returns whether the model supports image input.
    pub fn supports_image_input(&self) -> bool {
        matches!(self, Self::Omni(_))
    }

    /// Returns the route limits exposed by model discovery and enforced by
    /// request admission.
    pub fn support(&self) -> ModelSupport {
        if matches!(self, Self::Media(_)) {
            return ModelSupport {
                endpoints: vec![ServedEndpoint::VideoGenerations],
                input_modalities: vec![ServedModality::Text],
                output_modalities: vec![ServedModality::Video, ServedModality::Audio],
                features: Vec::new(),
                sampling_controls: Vec::new(),
            };
        }
        let mut endpoints = vec![ServedEndpoint::ChatCompletions];
        let mut input_modalities = vec![ServedModality::Text];
        let mut output_modalities = vec![ServedModality::Text];
        let sampling_controls = match self {
            Self::Text(description) => description.sampling_controls.clone(),
            Self::Omni(OmniDesc::SenseNova(description)) => description.sampling_controls.clone(),
            Self::Omni(OmniDesc::Bagel(description)) => description.sampling_controls.clone(),
            Self::Media(_) => unreachable!("media limits returned above"),
        };
        let mut features = vec![ServedFeature::Streaming, ServedFeature::Usage];
        if sampling_controls.contains(&ServedSamplingControl::Logprobs) {
            features.push(ServedFeature::Logprobs);
        }
        match self {
            Self::Text(_) => {
                features.push(ServedFeature::Reasoning);
                features.push(ServedFeature::ToolCalling);
            }
            Self::Omni(OmniDesc::SenseNova(_)) => {
                endpoints.push(ServedEndpoint::ImageGenerations);
                input_modalities.push(ServedModality::Image);
                output_modalities.push(ServedModality::Image);
                features.push(ServedFeature::Reasoning);
                features.push(ServedFeature::RepeatedInterleave);
            }
            Self::Omni(OmniDesc::Bagel(_)) => {
                endpoints.push(ServedEndpoint::ImageGenerations);
                input_modalities.push(ServedModality::Image);
                output_modalities.push(ServedModality::Image);
            }
            Self::Media(_) => unreachable!("media limits returned above"),
        }
        ModelSupport {
            endpoints,
            input_modalities,
            output_modalities,
            features,
            sampling_controls,
        }
    }

    /// Validates request features against the resolved route.
    pub fn validate_request(&self, request: &GenerateReqInput) -> Result<()> {
        let reject = |feature: &'static str| ServeError::UnsupportedFeature {
            request_id: request.request_id.clone(),
            feature,
        };
        if matches!(self, Self::Media(_)) {
            return Err(reject("generation_endpoint"));
        }
        let has_input_image = request.has_input_image();
        if has_input_image && !self.supports_image_input() {
            return Err(reject("image_input"));
        }
        if request.modalities.includes_image() && !self.supports_image_output() {
            return Err(reject("image_output"));
        }
        let declared = self.support();
        if request.uses_tools() && !declared.features.contains(&ServedFeature::ToolCalling) {
            return Err(reject("tool_calling"));
        }
        if request.requests_reasoning() && !declared.features.contains(&ServedFeature::Reasoning) {
            return Err(reject("reasoning"));
        }
        let (limits, needs) = match self {
            Self::Text(d) => (&d.limits, GenerationFeatures::UNDERSTANDING),
            Self::Omni(OmniDesc::SenseNova(d)) => (
                &d.limits,
                omni_required_features(
                    &d.preprocessing.generation_policy,
                    &d.preprocessing.image_ingest,
                    request,
                ),
            ),
            Self::Omni(OmniDesc::Bagel(d)) => (
                &d.limits,
                omni_required_features(
                    &d.preprocessing.generation_policy,
                    &d.preprocessing.image_ingest,
                    request,
                ),
            ),
            Self::Media(_) => unreachable!("media generation was rejected above"),
        };
        if let Err(feature) = limits.covers(needs) {
            return Err(reject(feature.name()));
        }
        Ok(())
    }

    /// Tokenizes a generation request with the resolved model pipeline.
    pub fn tokenize(&self, request: GenerateReqInput) -> Result<TokenizedGenerateReqInput> {
        match self {
            Self::Text(d) => d.tokenize(request),
            Self::Omni(OmniDesc::SenseNova(d)) => d.tokenize(request),
            Self::Omni(OmniDesc::Bagel(d)) => d.tokenize(request),
            Self::Media(_) => Err(ServeError::UnsupportedFeature {
                request_id: request.request_id,
                feature: "generation_endpoint",
            }),
        }
    }
}

/// Returns the multimodal resources required by the active profile.
fn configured_omni_needs(
    policy: &GenerationPolicyDescriptor,
    image_ingest: &uniserve_core::ImageIngestRecipe,
) -> GenerationFeatures {
    GenerationBehaviorDescriptor::resolve(GenerationConstraint::Default, policy)
        .required_features(policy, image_ingest.steps.iter().copied())
}

/// Validates the runtime features.
fn validate_runtime_features(
    identity: &ModelIdentity,
    limits: &GenerationLimits,
    needs: GenerationFeatures,
) -> Result<()> {
    limits.covers(needs).map_err(|feature| {
        ServeError::ModelResolution(ModelResolutionError::MissingFeature {
            description: identity.description.id(),
            feature,
        })
    })
}

/// Returns the runtime features required for multimodal serving.
fn omni_required_features(
    policy: &GenerationPolicyDescriptor,
    image_ingest: &uniserve_core::ImageIngestRecipe,
    request: &GenerateReqInput,
) -> GenerationFeatures {
    let has_input_image = request.has_input_image();
    let constraint = crate::serving::omni::generation_constraint(request);
    let behavior = GenerationBehaviorDescriptor::resolve(constraint, policy);
    let context_steps = if has_input_image {
        image_ingest.steps.clone()
    } else {
        Vec::new()
    };
    behavior.required_features(policy, context_steps)
}

/// Returns the model-specific sampling hints.
fn sampling_hints(profile: &CommonModelProfile, max_model_tokens: u32) -> SamplingHints {
    let primary = profile.stop_tokens.primary_eos_token_id;
    let mut extra: BTreeSet<u32> = profile.stop_tokens.eos_token_ids.clone();
    if let Some(primary) = primary {
        extra.remove(&primary);
    }
    SamplingHints {
        primary_eos_token_id: primary,
        extra_eos_token_ids: extra,
        default_temperature: profile.generation_defaults.temperature,
        default_top_p: profile.generation_defaults.top_p,
        default_top_k: profile.generation_defaults.top_k,
        default_min_p: profile.generation_defaults.min_p,
        default_repetition_penalty: profile.generation_defaults.repetition_penalty,
        default_max_tokens: profile.generation_defaults.max_output_tokens,
        max_model_len: Some(max_model_tokens),
    }
}

impl Qwen3Desc {
    /// Tokenizes a Qwen3 request and attaches its request identity to any failure.
    fn tokenize(&self, request: GenerateReqInput) -> Result<TokenizedGenerateReqInput> {
        let request_id = request.request_id.clone();
        self.tokenize_inner(request)
            .map_err(|source| ServeError::Tokenize { request_id, source })
    }

    /// Renders input, resolves sampling and cache policy, and builds the canonical engine request.
    fn tokenize_inner(
        &self,
        request: GenerateReqInput,
    ) -> std::result::Result<TokenizedGenerateReqInput, crate::serving::TokenizeError> {
        let (prompt_token_ids, output_processor, skip_special_tokens) = match &request.prompt {
            PromptInput::Text(text) => {
                let ids = self.tokenizer.encode(text, false)?;
                (
                    ids,
                    OutputProcessorPolicy::None,
                    request.decode.skip_special_tokens,
                )
            }
            PromptInput::Chat {
                messages,
                tools,
                tool_choice,
                reasoning_effort,
            } => {
                let mut chat_request = ChatRequest {
                    messages: messages.clone(),
                    chat_options: crate::serving::chat::ChatOptions {
                        generation_prompt_mode:
                            crate::serving::chat::GenerationPromptMode::StartNewAssistant,
                        reasoning_effort: *reasoning_effort,
                    },
                    tools: tools.clone(),
                    tool_choice: *tool_choice,
                    decode_options: TextDecodeOptions {
                        skip_special_tokens: request.decode.skip_special_tokens,
                        include_stop_str_in_output: request.decode.include_stop_string_in_output,
                        stop_strings: (!request.stop.stop_strings.is_empty())
                            .then(|| request.stop.stop_strings.clone()),
                        min_tokens: request.sampling.min_tokens.unwrap_or(0),
                    },
                };
                chat_request.validate()?;
                // Build the processor once to apply parser-driven request
                // adjustments (e.g. disabling special-token skipping).
                let processor = Qwen3ChatOutputProcessor::new(
                    &mut chat_request,
                    std::sync::Arc::clone(&self.tokenizer),
                    self.parse_reasoning,
                )?;
                let rendered_text = self.renderer.render(&chat_request)?;
                let ids = self.tokenizer.encode(&rendered_text, false)?;
                let skip = chat_request.decode_options.skip_special_tokens;
                (ids, OutputProcessorPolicy::Qwen3(processor), skip)
            }
        };

        let prompt_len = prompt_token_ids.len() as u32;
        let lowered = self.lower_sampling(&request, prompt_len)?;

        let constraint = GenerationConstraint::UndOnly;
        let mut policy = GenerationPolicyDescriptor::default();
        policy.termination.emit_stop_token = request.decode.include_stop_string_in_output;
        let isolation_key = cache_isolation_key(
            request.cache.namespace.as_deref(),
            request.cache.salt.as_deref(),
        );
        let cache = GenerationCachePolicyDescriptor {
            read: !request.cache.bypass_read && !lowered.sampling.prompt_logprobs_requested(),
            write: !request.cache.no_store,
            isolation_key,
        };
        let prompt_logprobs_requested = lowered.sampling.prompt_logprobs_requested();
        let generated_logprobs_requested = lowered.sampling.generated_logprobs_requested();
        let max_und_tokens = lowered.max_tokens as usize;
        let resources = GenerationResourceBounds {
            context_tokens: prompt_len as usize,
            max_kv_tokens: (prompt_len as usize).saturating_add(max_und_tokens),
            ..GenerationResourceBounds::default()
        };
        let generation = GenerationRequest {
            request_id: RequestId(stable_hash(request.request_id.as_ref())),
            context: vec![CoreContextSegment::UndTokens {
                token_ids: prompt_token_ids.clone(),
                visibility: UndVisibility::Internal,
            }],
            negative_context: Vec::new(),
            constraint,
            behavior: GenerationBehaviorDescriptor::resolve(constraint, &policy),
            sampling: lowered.sampling,
            image: ImageParams::default(),
            max_und_tokens,
            stop_strings: request.stop.stop_strings.clone(),
            stop_token_ids: lowered.stop_token_ids,
            priority: request.scheduling.priority,
            cache: cache.clone(),
            policy,
            resources: resources.clone(),
        };
        generation.validate()?;

        let cache_accounting = CacheAccounting {
            read_enabled: cache.read,
            write_enabled: cache.write,
            encoder_pin_count: 0,
        };
        let resource_accounting = ResourceAccounting {
            expected_kv_tokens: resources.max_kv_tokens as u64,
            image_latent_units: 0,
            encoder_cache_pins: 0,
            replayable: true,
        };

        let decode = TextDecodeOptions {
            skip_special_tokens,
            include_stop_str_in_output: request.decode.include_stop_string_in_output,
            stop_strings: (!request.stop.stop_strings.is_empty())
                .then(|| request.stop.stop_strings.clone()),
            min_tokens: request.sampling.min_tokens.unwrap_or(0),
        };

        Ok(TokenizedGenerateReqInput {
            request_id: request.request_id,
            request: generation,
            tokenizer: std::sync::Arc::clone(&self.tokenizer),
            prompt_token_ids,
            decode,
            emit_token_ids: matches!(
                request.output,
                OutputDetail::Tokens | OutputDetail::Logprobs
            ),
            prompt_logprobs_requested,
            generated_logprobs_requested,
            skip_special_tokens,
            output_processor,
            identity: ModelEventIdentity {
                served_name: self.identity.served_name.clone(),
                description: self.identity.description.id().to_string(),
            },
            cache: cache_accounting,
            resources: resource_accounting,
        })
    }

    /// Resolves model defaults and request controls into validated engine sampling parameters.
    fn lower_sampling(
        &self,
        request: &GenerateReqInput,
        prompt_len: u32,
    ) -> std::result::Result<LoweredSampling, crate::serving::TokenizeError> {
        let sampling = &request.sampling;
        let stop = &request.stop;
        let hints = &self.hints;

        let temperature = sampling
            .temperature
            .or(hints.default_temperature)
            .unwrap_or(1.0);
        let top_p = sampling.top_p.or(hints.default_top_p).unwrap_or(1.0);
        let top_k = sampling.top_k.or(hints.default_top_k).unwrap_or(0);
        let min_p = sampling.min_p.or(hints.default_min_p).unwrap_or(0.0);
        let repetition_penalty = sampling
            .repetition_penalty
            .or(hints.default_repetition_penalty)
            .unwrap_or(1.0);
        let max_tokens = resolve_max_tokens(
            sampling.max_tokens,
            hints.default_max_tokens,
            hints.max_model_len,
            prompt_len,
        )?;
        let min_tokens = sampling.min_tokens.unwrap_or(0);
        let frequency_penalty = sampling.frequency_penalty.unwrap_or(0.0);
        let presence_penalty = sampling.presence_penalty.unwrap_or(0.0);

        let mut stop_token_ids = stop.stop_token_ids.clone();
        if !sampling.ignore_eos {
            for token_id in &hints.extra_eos_token_ids {
                if !stop_token_ids.contains(token_id) {
                    stop_token_ids.push(*token_id);
                }
            }
        }

        for (field, value) in [
            ("logprobs", stop.logprobs),
            ("prompt_logprobs", stop.prompt_logprobs),
        ] {
            if let Some(value) = value
                && value < -1
            {
                return Err(crate::serving::TokenizeError::InvalidLogprobCount { field, value });
            }
        }
        if min_tokens > max_tokens {
            return Err(crate::serving::TokenizeError::MinTokensExceedsMaximum {
                min_tokens,
                max_tokens,
            });
        }

        let bad_words_ids = tokenize_bad_words(&stop.bad_words, self.tokenizer.as_ref())?;

        let mut canonical_logit_bias: Vec<(u32, f32)> = stop
            .logit_bias
            .as_ref()
            .map(|biases| biases.iter().map(|(&token, &bias)| (token, bias)).collect())
            .unwrap_or_default();
        canonical_logit_bias.sort_by_key(|(token, _)| *token);

        let core = SamplingParams {
            temperature,
            top_k,
            top_p,
            ignore_eos: sampling.ignore_eos,
            seed: sampling.seed.map(|value| value as u64),
            min_p,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            logit_bias: canonical_logit_bias,
            min_tokens: min_tokens as usize,
            return_logprobs: stop.logprobs.is_some() || stop.logprob_token_ids.is_some(),
            n_logprobs: match stop.logprobs {
                Some(-1) => u32::MAX,
                Some(value) => value as u32,
                None => 0,
            },
            return_prompt_logprobs: stop.prompt_logprobs.is_some(),
            n_prompt_logprobs: match stop.prompt_logprobs {
                Some(-1) => u32::MAX,
                Some(value) => value as u32,
                None => 0,
            },
            logprob_token_ids: stop.logprob_token_ids.clone().unwrap_or_default(),
            bad_words_ids: bad_words_ids.unwrap_or_default(),
            allowed_token_ids: stop.allowed_token_ids.clone(),
            typical_p: 1.0,
            forced_token_ids: Vec::new(),
        };
        core.validate()?;

        // Logprob feature gate.
        if (stop.logprobs.is_some() || stop.prompt_logprobs.is_some()) && !self.logprobs_supported {
            return Err(crate::serving::TokenizeError::UnsupportedLogprobs);
        }

        Ok(LoweredSampling {
            sampling: core,
            max_tokens,
            stop_token_ids,
        })
    }
}

impl SenseNovaDesc {
    /// Tokenizes a request with the SenseNova multimodal preprocessing pipeline.
    fn tokenize(&self, request: GenerateReqInput) -> Result<TokenizedGenerateReqInput> {
        crate::serving::omni::tokenize_sensenova(
            &self.preprocessing,
            crate::serving::omni::RuntimeBinding {
                tokenizer: std::sync::Arc::clone(&self.tokenizer),
                renderer: &self.renderer,
                limits: &self.limits,
                default_max_output_tokens: self.default_max_output_tokens,
                max_model_tokens: self.max_model_tokens,
                identity: ModelEventIdentity {
                    served_name: self.identity.served_name.clone(),
                    description: self.identity.description.id().to_string(),
                },
            },
            request,
        )
    }
}

impl BagelDesc {
    /// Tokenizes a request with the Bagel multimodal preprocessing pipeline.
    fn tokenize(&self, request: GenerateReqInput) -> Result<TokenizedGenerateReqInput> {
        crate::serving::omni::tokenize_bagel(
            &self.preprocessing,
            crate::serving::omni::RuntimeBinding {
                tokenizer: std::sync::Arc::clone(&self.tokenizer),
                renderer: &self.renderer,
                limits: &self.limits,
                default_max_output_tokens: self.default_max_output_tokens,
                max_model_tokens: self.max_model_tokens,
                identity: ModelEventIdentity {
                    served_name: self.identity.served_name.clone(),
                    description: self.identity.description.id().to_string(),
                },
            },
            request,
        )
    }
}

struct LoweredSampling {
    sampling: SamplingParams,
    max_tokens: u32,
    stop_token_ids: Vec<u32>,
}

/// Converts bad-word strings into token-ID sequences, encoding each word both
/// with and without a leading space (prefix-space convention) and deduping.
fn tokenize_bad_words(
    bad_words: &[String],
    tokenizer: &crate::profile::tokenizer::HuggingFaceTokenizer,
) -> std::result::Result<Option<Vec<Vec<u32>>>, crate::profile::tokenizer::TokenizerError> {
    if bad_words.is_empty() {
        return Ok(None);
    }
    let mut all_token_ids = Vec::new();
    for bad_word in bad_words {
        let without_space = tokenizer.encode(bad_word, false)?;
        let with_space = tokenizer.encode(&format!(" {}", bad_word.trim_start()), false)?;
        let keep_with_space = !with_space.is_empty()
            && (without_space.is_empty()
                || (with_space[0] != without_space[0] && with_space.len() == without_space.len()));
        if !without_space.is_empty() {
            all_token_ids.push(without_space);
        }
        if keep_with_space {
            all_token_ids.push(with_space);
        }
    }
    Ok((!all_token_ids.is_empty()).then_some(all_token_ids))
}

/// Computes a stable hash for cache isolation.
fn stable_hash(value: &str) -> u64 {
    value
        .as_bytes()
        .iter()
        .fold(0xcbf29ce484222325, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        })
}
