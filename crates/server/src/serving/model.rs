//! Load-time model resolution and model-owned request tokenization.
//!
//! [`InputProcessor`] binds tokenizer assets, generation policy, geometry, and
//! output processing. [`InputProcessor::preprocess_text_request`] produces a [`GenerationRequest`]
//! and the [`ResponseOptions`] retained by the frontend.

use std::path::PathBuf;
use std::sync::Arc;

use crate::config::EngineSettings;
use crate::profile::assets::{ResolvedModelFiles, resolve_model_file};
use crate::profile::omni::bagel::BagelProfile;
use crate::profile::omni::sensenova::SenseNovaProfile;
use crate::profile::tokenizer::{DynTokenizer, HuggingFaceTokenizer, TokenizerError};
use crate::profile::{ModelConfig, ModelDescription, ModelParameters, SamplingDefaults};
use thiserror::Error;
use uniserve_core::{
    CachePolicy, GenerationConstraint, GenerationFeatures, GenerationLimits, GenerationRequest,
    ImageGenerationConfig, ImageParams, RequestId, SamplingParams,
};

use crate::serving::chat::{ChatTemplateLoadOptions, HfChatRenderer, Qwen3ChatOutputProcessor};
use crate::serving::input::{
    ModelEventIdentity, OutputDetail, OutputProcessorPolicy, PromptInput, ResponseOptions,
    TextPromptRequest,
};
use crate::serving::text::{TextDecodeOptions, resolve_max_tokens};
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

/// Owns the tokenizer and template resources used to produce final engine inputs.
/// Runtime capabilities are fixed at construction, before this value is shared.
pub struct InputProcessor {
    pub(super) config: ModelConfig,
    pub(super) tokenizer: DynTokenizer,
    pub(super) renderer: Option<HfChatRenderer>,
    pub(super) limits: GenerationLimits,
    sampling_controls: Vec<ServedSamplingControl>,
    parse_reasoning: bool,
}

#[derive(Debug, Error)]
/// Failure while resolving a configured model and its assets.
pub enum ModelResolutionError {
    /// Required numerical checkpoint metadata is missing or contradictory.
    #[error("invalid media checkpoint contract: {0}")]
    MediaContract(String),
    /// A token-generating model has no chat-template resource.
    #[error("text-generation model requires a chat template")]
    MissingTemplate,
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

impl ModelConfig {
    /// Loads model facts and the tokenizer/template resources needed by preprocessing.
    pub(crate) async fn load(
        config: &crate::Config,
    ) -> std::result::Result<(Self, DynTokenizer, Option<HfChatRenderer>), ModelResolutionError>
    {
        let served_name = config
            .served_model_name
            .clone()
            .unwrap_or_else(|| config.model.clone());
        if config.model_description == ModelDescription::MiniMaxH3 {
            let tokenizer_path =
                resolve_model_file(&config.model, "tokenizer/tokenizer.json").await?;
            let tokenizer: DynTokenizer = Arc::new(HuggingFaceTokenizer::new(&tokenizer_path)?);
            let num_inference_steps = config.model_contract.as_ref()
                .and_then(|contract| contract.get("denoise_steps"))
                .and_then(serde_json::Value::as_u64)
                .and_then(|steps| u32::try_from(steps).ok())
                .filter(|steps| *steps > 0)
                .ok_or_else(|| ModelResolutionError::MediaContract(
                    "resolve the checkpoint with the installed worker before building the server".to_owned()
                ))?;
            return Ok((
                Self {
                    served_name,
                    parameters: ModelParameters::MiniMaxH3 {
                        max_video_seconds: config.engine.max_video_seconds,
                        num_inference_steps,
                    },
                    sampling_defaults: SamplingDefaults::default(),
                    max_model_tokens: Some(config.engine.max_model_len.unwrap_or(16_384)),
                    primary_eos_token_id: None,
                    eos_token_ids: Default::default(),
                },
                tokenizer,
                None,
            ));
        }
        let files = ResolvedModelFiles::new(&config.model).await?;
        let tokenizer: DynTokenizer = Arc::new(HuggingFaceTokenizer::new(&files.tokenizer_path)?);
        let mut model = Self::from_files(
            config.model_description,
            &served_name,
            &files,
            config.engine.max_model_len,
            tokenizer.as_ref(),
        )?;
        model.max_model_tokens = Some(
            config
                .engine
                .max_model_len
                .or(model.max_model_tokens)
                .unwrap_or(EngineSettings::DEFAULT_MAX_MODEL_LEN),
        );
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
        Ok((model, tokenizer, Some(renderer)))
    }

    /// Model's effective startup context ceiling.
    pub(crate) fn max_model_tokens(&self) -> u32 {
        self.max_model_tokens
            .unwrap_or(EngineSettings::DEFAULT_MAX_MODEL_LEN)
    }

    /// IPC payload capacity required by this model's request descriptors.
    pub(crate) fn request_slot_capacity(&self) -> usize {
        if matches!(self.parameters, ModelParameters::MiniMaxH3 { .. }) {
            EngineSettings::MEDIA_IPC_SLOT_CAP
        } else {
            1 << 20
        }
    }

    /// Vocabulary markers used by multimodal scheduling and prompt construction.
    pub(crate) fn generation_controls(&self) -> Option<&crate::profile::omni::GenerationControls> {
        match &self.parameters {
            ModelParameters::SenseNova(value) => Some(&value.controls),
            ModelParameters::Bagel(value) => Some(&value.controls),
            ModelParameters::Qwen3 | ModelParameters::MiniMaxH3 { .. } => None,
        }
    }

    /// Execution family selected by the loaded model settings.
    pub(crate) fn runtime_family(&self) -> uniserve_core::RuntimeFamily {
        match self.parameters {
            ModelParameters::Qwen3 => uniserve_core::RuntimeFamily::Ar,
            ModelParameters::SenseNova(_) | ModelParameters::Bagel(_) => {
                uniserve_core::RuntimeFamily::Umm
            }
            ModelParameters::MiniMaxH3 { .. } => uniserve_core::RuntimeFamily::Diffusion,
        }
    }

    /// Model requirements used at startup before intersecting loaded worker capacity.
    pub(crate) fn generation_limits(
        &self,
        model_dtype: uniserve_core::ModelDtype,
    ) -> uniserve_core::GenerationLimits {
        match &self.parameters {
            ModelParameters::Qwen3 => uniserve_core::GenerationLimits {
                features: uniserve_core::GenerationFeatures::UNDERSTANDING,
                latent_downsample: 1,
                max_cfg_branches: 1,
                ..Default::default()
            },
            ModelParameters::MiniMaxH3 { .. } => uniserve_core::GenerationLimits {
                latent_downsample: 1,
                max_cfg_branches: 1,
                ..Default::default()
            },
            ModelParameters::SenseNova(_) => SenseNovaProfile::runtime_limits(model_dtype),
            ModelParameters::Bagel(_) => BagelProfile::runtime_limits(model_dtype),
        }
    }
}

impl InputProcessor {
    /// Binds model resources to verified worker capabilities without rebuilding model data.
    pub fn new(
        mut config: ModelConfig,
        tokenizer: DynTokenizer,
        renderer: Option<HfChatRenderer>,
        limits: GenerationLimits,
        sampling_controls: Vec<ServedSamplingControl>,
        max_model_tokens: u32,
        parse_reasoning: bool,
    ) -> Result<Self> {
        let needs = match &config.parameters {
            ModelParameters::Qwen3 => GenerationFeatures::UNDERSTANDING,
            ModelParameters::SenseNova(profile) => {
                configured_omni_needs(&profile.image_generation, &profile.image_encoders)
            }
            ModelParameters::Bagel(profile) => {
                configured_omni_needs(&profile.image_generation, &profile.image_encoders)
            }
            ModelParameters::MiniMaxH3 { .. } => GenerationFeatures::empty(),
        };
        validate_runtime_features(&config, &limits, needs)?;
        if !matches!(config.parameters, ModelParameters::MiniMaxH3 { .. }) && renderer.is_none() {
            return Err(ServeError::ModelResolution(
                ModelResolutionError::MissingTemplate,
            ));
        }
        // Bind the actual worker ceiling once before sharing immutable model facts.
        config.max_model_tokens = Some(max_model_tokens);
        Ok(Self {
            config,
            tokenizer,
            renderer,
            limits,
            sampling_controls,
            parse_reasoning,
        })
    }

    /// Immutable facts of the loaded model.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Public served-model name.
    pub fn served_model_name(&self) -> &str {
        &self.config.served_name
    }

    /// Public duration, geometry and prompt limits from the serving description.
    pub fn video_capabilities(&self) -> serde_json::Value {
        match &self.config.parameters {
            ModelParameters::MiniMaxH3 {
                max_video_seconds, ..
            } => {
                let default_seconds = max_video_seconds.min(5.0);
                let mut suggested_seconds = vec![default_seconds];
                if *max_video_seconds > default_seconds {
                    suggested_seconds.push(*max_video_seconds);
                }
                serde_json::json!({
                    "tasks": ["t2va"],
                    "default_seconds": default_seconds,
                    "max_seconds": *max_video_seconds,
                    "suggested_seconds": suggested_seconds,
                    "min_frames": 22, "fps": 24, "width": 1344, "height": 768,
                    "max_prompt_tokens": self.config.max_model_tokens(),
                    "request_fields": ["model", "prompt", "seconds", "seed"],
                })
            }
            _ => serde_json::Value::Null,
        }
    }

    /// Validates the video API request, tokenizes its prompt, and prepares the
    /// checkpoint frame/chunk counts for direct engine submission.
    pub fn preprocess_video_request(
        &self,
        request_id: &crate::serving::ServeRequestId,
        request: crate::openai::VideoGenerationRequest,
    ) -> std::result::Result<uniserve_core::DiffusionRequest, crate::openai::ApiError> {
        let crate::openai::VideoGenerationRequest {
            model,
            prompt,
            seed,
            seconds,
        } = request;
        crate::openai::utils::check_model_served(&model, self.served_model_name())?;
        if prompt.trim().is_empty() {
            return Err(crate::openai::ApiError::invalid_request(
                "prompt must not be empty".to_string(),
                Some("prompt"),
            ));
        }
        let ModelParameters::MiniMaxH3 {
            max_video_seconds,
            num_inference_steps,
        } = &self.config.parameters
        else {
            return Err(crate::openai::serve_error_to_api(
                ServeError::UnsupportedFeature {
                    request_id: request_id.clone(),
                    feature: "video_generation",
                },
            ));
        };
        // Tokenize and bound the prompt before deriving any media allocation.
        let prompt_token_ids = self
            .tokenizer
            .encode(&prompt, false)
            .map_err(|source| ServeError::Tokenize {
                request_id: request_id.clone(),
                source: crate::serving::TokenizeError::Tokenizer(source),
            })
            .map_err(crate::openai::serve_error_to_api)?;
        if prompt_token_ids.is_empty() {
            return Err(crate::openai::serve_error_to_api(ServeError::Tokenize {
                request_id: request_id.clone(),
                source: crate::serving::TokenizeError::Invalid(
                    "video prompt must contain at least one token".to_string(),
                ),
            }));
        }
        if prompt_token_ids.len() > self.config.max_model_tokens() as usize {
            return Err(crate::openai::serve_error_to_api(
                ServeError::ContextLengthExceeded {
                    request_id: request_id.clone(),
                    prompt_tokens: prompt_token_ids.len(),
                    max_tokens: self.config.max_model_tokens(),
                },
            ));
        }
        // Duration is a public floating-point input and must be finite before
        // conversion to the fixed-width frame protocol.
        if !seconds.is_finite() || seconds <= 0.0 || seconds > *max_video_seconds {
            return Err(crate::openai::serve_error_to_api(ServeError::Tokenize {
                request_id: request_id.clone(),
                source: crate::serving::TokenizeError::Invalid(format!(
                    "video duration must be finite, positive, and at most {} seconds",
                    *max_video_seconds
                )),
            }));
        }
        let raw_frames = (seconds * 24.0).round();
        if raw_frames < 1.0 || raw_frames > f64::from(u32::MAX - 16) {
            return Err(crate::openai::serve_error_to_api(ServeError::Tokenize {
                request_id: request_id.clone(),
                source: crate::serving::TokenizeError::Invalid(
                    "video duration cannot be represented by the configuration".to_string(),
                ),
            }));
        }
        let frame_count = align_num_frames(raw_frames as u32);
        if frame_count < 22 {
            return Err(crate::openai::serve_error_to_api(ServeError::Tokenize {
                request_id: request_id.clone(),
                source: crate::serving::TokenizeError::Invalid(
                    "video duration is shorter than the supported media geometry".to_string(),
                ),
            }));
        }
        // Each H3 VAE chunk consumes a temporal latent window and emits its
        // non-overlapping frame interval; the model owns overlap reconstruction.
        let num_decode_chunks = (frame_count - 5) / 17;
        Ok(uniserve_core::DiffusionRequest {
            request_id: uniserve_core::RequestId(0),
            prompt_token_ids,
            priority: 0,
            sampling: uniserve_core::DiffusionSamplingParams {
                num_frames: frame_count,
                num_decode_chunks,
                num_inference_steps: *num_inference_steps,
                seed,
            },
        })
    }

    /// Builds the model identity stamped onto accepted events.
    pub fn event_identity(&self) -> ModelEventIdentity {
        ModelEventIdentity {
            served_name: self.config.served_name.clone(),
            description: self.config.description().id().to_string(),
        }
    }

    /// Returns whether the model supports image output.
    pub fn supports_image_output(&self) -> bool {
        matches!(
            self.config.parameters,
            ModelParameters::SenseNova(_) | ModelParameters::Bagel(_)
        )
    }

    /// Returns whether the model supports image input.
    pub fn supports_image_input(&self) -> bool {
        matches!(
            self.config.parameters,
            ModelParameters::SenseNova(_) | ModelParameters::Bagel(_)
        )
    }

    /// Returns the route limits exposed by model discovery and enforced by
    /// request admission.
    pub fn support(&self) -> ModelSupport {
        if matches!(self.config.parameters, ModelParameters::MiniMaxH3 { .. }) {
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
        let sampling_controls = self.sampling_controls.clone();
        let mut features = vec![ServedFeature::Streaming, ServedFeature::Usage];
        if sampling_controls.contains(&ServedSamplingControl::Logprobs) {
            features.push(ServedFeature::Logprobs);
        }
        match &self.config.parameters {
            ModelParameters::Qwen3 => {
                features.push(ServedFeature::Reasoning);
                features.push(ServedFeature::ToolCalling);
            }
            ModelParameters::SenseNova(_) => {
                endpoints.push(ServedEndpoint::ImageGenerations);
                input_modalities.push(ServedModality::Image);
                output_modalities.push(ServedModality::Image);
                features.push(ServedFeature::Reasoning);
                features.push(ServedFeature::RepeatedInterleave);
            }
            ModelParameters::Bagel(_) => {
                endpoints.push(ServedEndpoint::ImageGenerations);
                input_modalities.push(ServedModality::Image);
                output_modalities.push(ServedModality::Image);
            }
            ModelParameters::MiniMaxH3 { .. } => unreachable!("media limits returned above"),
        }
        ModelSupport {
            endpoints,
            input_modalities,
            output_modalities,
            features,
            sampling_controls,
        }
    }

    /// Validates the prompt and requested modalities against the loaded model.
    fn validate_generation_features(
        &self,
        request_id: &crate::serving::ServeRequestId,
        prompt: &PromptInput,
        has_input_image: bool,
        modalities: crate::serving::ModalitySelection,
    ) -> Result<()> {
        let reject = |feature| ServeError::UnsupportedFeature {
            request_id: request_id.clone(),
            feature,
        };
        if matches!(self.config.parameters, ModelParameters::MiniMaxH3 { .. }) {
            return Err(reject("generation_endpoint"));
        }
        if has_input_image && !self.supports_image_input() {
            return Err(reject("image_input"));
        }
        if modalities.includes_image() && !self.supports_image_output() {
            return Err(reject("image_output"));
        }
        let declared = self.support();
        if let PromptInput::Chat(chat) = prompt {
            let uses_tools = !chat.tools.is_empty()
                || chat.messages.iter().any(|message| match message {
                    crate::serving::chat::ChatMessage::Developer { tools, .. } => {
                        tools.as_ref().is_some_and(|tools| !tools.is_empty())
                    }
                    crate::serving::chat::ChatMessage::Assistant { content } => {
                        content.has_tool_calls()
                    }
                    crate::serving::chat::ChatMessage::ToolResponse { .. } => true,
                    _ => false,
                });
            if uses_tools && !declared.features.contains(&ServedFeature::ToolCalling) {
                return Err(reject("tool_calling"));
            }
            if chat.chat_options.reasoning_effort.is_some()
                && !declared.features.contains(&ServedFeature::Reasoning)
            {
                return Err(reject("reasoning"));
            }
        }
        let needs = match &self.config.parameters {
            ModelParameters::Qwen3 => GenerationFeatures::UNDERSTANDING,
            ModelParameters::SenseNova(profile) => omni_required_features(
                &profile.image_generation,
                &profile.image_encoders,
                has_input_image,
                modalities,
            ),
            ModelParameters::Bagel(profile) => omni_required_features(
                &profile.image_generation,
                &profile.image_encoders,
                has_input_image,
                modalities,
            ),
            ModelParameters::MiniMaxH3 { .. } => unreachable!("video endpoint checked above"),
        };
        self.limits
            .covers(needs)
            .map_err(|feature| reject(feature.name()))
    }

    /// Preprocesses a programmatic text prompt without a chat-template conversion.
    pub fn preprocess_text_request(
        &self,
        request: TextPromptRequest,
    ) -> Result<(GenerationRequest, ResponseOptions)> {
        let TextPromptRequest {
            request_id,
            prompt,
            images,
            modalities,
            sampling,
            stop,
            negative_text,
            image_gen,
            cache_namespace,
            cache_salt,
            bypass_cache_read,
            no_cache_store,
            priority,
            output,
            decode,
        } = request;
        let cache = CachePolicy {
            read: !bypass_cache_read,
            write: !no_cache_store,
            isolation_key: cache_isolation_key(cache_namespace.as_deref(), cache_salt.as_deref()),
        };
        self.preprocess_generation(
            request_id,
            PromptInput::Text(prompt),
            images,
            modalities,
            sampling,
            stop,
            negative_text,
            image_gen,
            cache,
            priority,
            output,
            decode,
        )
    }

    /// Tokenizes model input and resolves its final engine and output requirements.
    pub(super) fn preprocess_generation(
        &self,
        request_id: crate::serving::ServeRequestId,
        prompt: PromptInput,
        images: Vec<crate::serving::ImageInput>,
        modalities: crate::serving::ModalitySelection,
        sampling: crate::serving::SamplingConfig,
        stop: crate::serving::StopConfig,
        negative_text: Option<String>,
        image_gen: Option<crate::serving::ImageGenControls>,
        cache: CachePolicy,
        priority: i32,
        output: OutputDetail,
        decode: crate::serving::DecodeControls,
    ) -> Result<(GenerationRequest, ResponseOptions)> {
        let has_input_image = !images.is_empty()
            || matches!(&prompt, PromptInput::Chat(chat) if chat.has_multimodal());
        self.validate_generation_features(&request_id, &prompt, has_input_image, modalities)?;
        let constraint = crate::serving::omni::generation_constraint(modalities);
        let mut generation = GenerationRequest {
            request_id: RequestId(stable_hash(request_id.as_ref())),
            prompt_token_ids: Vec::new(),
            negative_prompt_token_ids: Vec::new(),
            multimodal_inputs: Default::default(),
            constraint,
            sampling: SamplingParams {
                seed: image_gen
                    .as_ref()
                    .and_then(|image| image.seed)
                    .or_else(|| sampling.seed.and_then(|seed| seed.try_into().ok())),
                ..SamplingParams::default()
            },
            image: ImageParams::default(),
            max_und_tokens: 0,
            include_stop_token: false,
            stop_strings: stop.stop_strings.clone(),
            stop_token_ids: stop.stop_token_ids.clone(),
            priority,
            cache,
            image_generation: ImageGenerationConfig::default(),
        };
        let mut decode = TextDecodeOptions {
            skip_special_tokens: decode.skip_special_tokens,
            include_stop_str_in_output: decode.include_stop_string_in_output,
            stop_strings: (!stop.stop_strings.is_empty()).then(|| stop.stop_strings.clone()),
            min_tokens: sampling.min_tokens.unwrap_or(0),
        };

        // Each model fills its actual computation inputs in the same request.
        // No partially prepared request is exposed to the submission boundary.
        let (output_processor, max_kv_tokens, image_latent_units) =
            (|| -> std::result::Result<_, crate::serving::TokenizeError> {
                let output_processor = match &self.config.parameters {
                    ModelParameters::Qwen3 => self.preprocess_qwen3_input(
                        prompt,
                        &sampling,
                        &stop,
                        &mut generation,
                        &mut decode,
                    )?,
                    ModelParameters::SenseNova(profile) => {
                        crate::serving::omni::preprocess_sensenova(
                            profile,
                            self,
                            &request_id,
                            prompt,
                            images,
                            negative_text,
                            image_gen,
                            &mut generation,
                        )?
                    }
                    ModelParameters::Bagel(profile) => crate::serving::omni::preprocess_bagel(
                        profile,
                        self,
                        &request_id,
                        prompt,
                        images,
                        negative_text,
                        image_gen,
                        &mut generation,
                    )?,
                    ModelParameters::MiniMaxH3 { .. } => {
                        unreachable!("generation features checked above")
                    }
                };
                if matches!(self.config.parameters, ModelParameters::Qwen3) {
                    generation.validate()?;
                } else {
                    crate::serving::omni::prepare_generation_resources(
                        self,
                        &sampling,
                        &stop,
                        &mut generation,
                    )?;
                }
                Ok((
                    output_processor,
                    generation.max_kv_tokens(&self.limits)?,
                    generation.image_latent_units(&self.limits)?,
                ))
            })()
            .map_err(|source| ServeError::Tokenize {
                request_id: request_id.clone(),
                source,
            })?;

        let response = ResponseOptions {
            request_id,
            tokenizer: Arc::clone(&self.tokenizer),
            // Detokenization needs the prompt after ownership moves to the engine.
            prompt_token_ids: generation.prompt_token_ids.clone(),
            decode,
            emit_token_ids: matches!(output, OutputDetail::Tokens | OutputDetail::Logprobs),
            prompt_logprobs_requested: generation.sampling.prompt_logprobs_requested(),
            generated_logprobs_requested: generation.sampling.generated_logprobs_requested(),
            output_processor,
            identity: self.event_identity(),
            cache: CacheAccounting {
                read_enabled: generation.cache.read,
                write_enabled: generation.cache.write,
                encoder_pin_count: generation.num_encoder_cache_entries(),
            },
            resources: ResourceAccounting {
                expected_kv_tokens: max_kv_tokens as u64,
                image_latent_units: image_latent_units,
                encoder_cache_pins: generation.num_encoder_cache_entries(),
                replayable: !generation.feeds_back_images(),
            },
        };
        Ok((generation, response))
    }
}

/// Fast H3 reconstructs frame counts congruent to five modulo seventeen.
/// The caller checks that adding at most sixteen frames cannot overflow.
fn align_num_frames(num_frames: u32) -> u32 {
    num_frames + (22 - num_frames % 17) % 17
}

/// Returns the multimodal resources required by the active profile.
fn configured_omni_needs(
    policy: &ImageGenerationConfig,
    image_encoders: &[uniserve_core::ImageEncoderInput],
) -> GenerationFeatures {
    policy.required_features(
        GenerationConstraint::Default,
        image_encoders.iter().map(|input| input.encoder),
    )
}

/// Validates the runtime features.
fn validate_runtime_features(
    config: &ModelConfig,
    limits: &GenerationLimits,
    needs: GenerationFeatures,
) -> Result<()> {
    limits.covers(needs).map_err(|feature| {
        ServeError::ModelResolution(ModelResolutionError::MissingFeature {
            description: config.description().id(),
            feature,
        })
    })
}

/// Returns the runtime features required for multimodal serving.
fn omni_required_features(
    policy: &ImageGenerationConfig,
    image_encoders: &[uniserve_core::ImageEncoderInput],
    has_input_image: bool,
    modalities: crate::serving::ModalitySelection,
) -> GenerationFeatures {
    let constraint = crate::serving::omni::generation_constraint(modalities);

    let context_steps = if has_input_image {
        image_encoders
            .iter()
            .map(|input| input.encoder)
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    policy.required_features(constraint, context_steps)
}

impl InputProcessor {
    /// Renders a Qwen prompt and fills its token, sampling, and KV requirements.
    fn preprocess_qwen3_input(
        &self,
        prompt: PromptInput,
        controls: &crate::serving::SamplingConfig,
        stop: &crate::serving::StopConfig,
        generation: &mut GenerationRequest,
        decode: &mut TextDecodeOptions,
    ) -> std::result::Result<OutputProcessorPolicy, crate::serving::TokenizeError> {
        let (prompt_token_ids, output_processor, skip_special_tokens) = match prompt {
            PromptInput::Text(text) => {
                let ids = self.tokenizer.encode(&text, false)?;
                (ids, OutputProcessorPolicy::None, decode.skip_special_tokens)
            }
            PromptInput::Chat(mut chat_request) => {
                chat_request.decode_options = decode.clone();
                chat_request.validate()?;
                // Build the processor once to apply parser-driven request
                // adjustments (e.g. disabling special-token skipping).
                let processor = Qwen3ChatOutputProcessor::new(
                    &mut chat_request,
                    std::sync::Arc::clone(&self.tokenizer),
                    self.parse_reasoning,
                )?;
                let rendered_text = self
                    .renderer
                    .as_ref()
                    .expect("Qwen3 requires a chat renderer")
                    .render(&chat_request)?;
                let ids = self.tokenizer.encode(&rendered_text, false)?;
                let skip = chat_request.decode_options.skip_special_tokens;
                (ids, OutputProcessorPolicy::Qwen3(processor), skip)
            }
        };

        let prompt_len = prompt_token_ids.len() as u32;
        let (sampling, max_tokens, stop_token_ids) =
            self.resolve_sampling(controls, stop, prompt_len)?;

        generation.prompt_token_ids = prompt_token_ids;
        generation.sampling = sampling;
        generation.stop_token_ids = stop_token_ids;
        generation.max_und_tokens = max_tokens as usize;
        generation.include_stop_token = decode.include_stop_str_in_output;
        // Prompt logprobs require computation for every prompt token.
        generation.cache.read &= !generation.sampling.prompt_logprobs_requested();
        decode.skip_special_tokens = skip_special_tokens;
        Ok(output_processor)
    }

    /// Resolves model defaults and request controls into validated engine sampling parameters.
    fn resolve_sampling(
        &self,
        sampling: &crate::serving::SamplingConfig,
        stop: &crate::serving::StopConfig,
        prompt_len: u32,
    ) -> std::result::Result<(SamplingParams, u32, Vec<u32>), crate::serving::TokenizeError> {
        let defaults = &self.config.sampling_defaults;

        let temperature = sampling.temperature.or(defaults.temperature).unwrap_or(1.0);
        let top_p = sampling.top_p.or(defaults.top_p).unwrap_or(1.0);
        let top_k = sampling.top_k.or(defaults.top_k).unwrap_or(0);
        let min_p = sampling.min_p.or(defaults.min_p).unwrap_or(0.0);
        let repetition_penalty = sampling
            .repetition_penalty
            .or(defaults.repetition_penalty)
            .unwrap_or(1.0);
        let max_tokens = resolve_max_tokens(
            sampling.max_tokens,
            defaults.max_output_tokens,
            Some(self.config.max_model_tokens()),
            prompt_len,
        )?;
        let min_tokens = sampling.min_tokens.unwrap_or(0);
        let frequency_penalty = sampling.frequency_penalty.unwrap_or(0.0);
        let presence_penalty = sampling.presence_penalty.unwrap_or(0.0);

        let mut stop_token_ids = stop.stop_token_ids.clone();
        if !sampling.ignore_eos {
            for token_id in self
                .config
                .eos_token_ids
                .iter()
                .filter(|token| Some(**token) != self.config.primary_eos_token_id)
            {
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
        if (stop.logprobs.is_some() || stop.prompt_logprobs.is_some())
            && !self
                .sampling_controls
                .contains(&ServedSamplingControl::Logprobs)
        {
            return Err(crate::serving::TokenizeError::UnsupportedLogprobs);
        }

        Ok((core, max_tokens, stop_token_ids))
    }
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

/// Computes a deterministic identifier for a preprocessed request.
fn stable_hash(value: &str) -> u64 {
    value
        .as_bytes()
        .iter()
        .fold(0xcbf29ce484222325, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        })
}
