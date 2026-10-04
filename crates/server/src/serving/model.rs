//! Load-time model resolution and model-owned request tokenization.
//!
//! `ModelConfig::load` resolves checkpoint facts, the tokenizer, and the chat renderer at
//! startup: a diffusers pipeline index selects a video model (which has no renderer), and
//! otherwise the root `config.json` selects a token-generating model. [`InputProcessor::new`]
//! then binds the capabilities the connected worker advertises ([`WorkerCapabilities`]) and
//! the result is shared immutably by the serving runtime.
//!
//! [`InputProcessor`] binds tokenizer assets, generation limits, sampling policy, and
//! output processing. [`InputProcessor::preprocess_text_request`] produces a [`GenerationRequest`]
//! and the [`ResponseOptions`] retained by the frontend. The chat and image entry points in the
//! sibling `preprocessing` module route through the same `InputProcessor::preprocess_generation`,
//! which delegates SenseNova and Bagel inputs to the sibling `omni` module; `omni` reads the
//! `pub(super)` fields.
//! [`InputProcessor::preprocess_video_request`] produces a `DiffusionRequest` for MiniMax H3.

use std::path::PathBuf;
use std::sync::Arc;

use crate::config::EngineSettings;
use crate::profile::assets::{ResolvedModelFiles, resolve_model_file, resolve_pipeline_index};
use crate::profile::omni::bagel::BagelProfile;
use crate::profile::omni::sensenova::SenseNovaProfile;
use crate::profile::tokenizer::{DynTokenizer, HuggingFaceTokenizer, TokenizerError};
use crate::profile::{ModelConfig, ModelDescription, ModelParameters};
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
    ///
    /// `EngineClient::served_sampling_controls` advertises this full set when the worker
    /// supports token sampling and no controls otherwise.
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

/// Exact public route declaration for one load-bound model, built by
/// [`InputProcessor::support`].
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
///
/// Preprocessing methods are synchronous and CPU-bound; `ServingRuntime` runs its request
/// preprocessing on the blocking pool.
pub struct InputProcessor {
    pub(super) config: ModelConfig,
    pub(super) tokenizer: DynTokenizer,
    // `None` only for a video model; `InputProcessor::new` rejects a missing renderer for
    // every other model.
    pub(super) renderer: Option<HfChatRenderer>,
    // Worker-advertised generation features and resource limits.
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
    ///
    /// Returns the renderer as `None` for a diffusers pipeline. The returned
    /// `max_model_tokens` is always set: the configured `max_model_len`, else, for a root
    /// configuration, its `max_position_embeddings` or `EngineSettings::DEFAULT_MAX_MODEL_LEN`,
    /// and for a pipeline the model's default prompt limit.
    pub(crate) async fn load(
        config: &crate::Config,
    ) -> std::result::Result<(Self, DynTokenizer, Option<HfChatRenderer>), ModelResolutionError>
    {
        let served_name = config
            .served_model_name
            .clone()
            .unwrap_or_else(|| config.model.clone());
        // A diffusers pipeline declares its class and component folders in a
        // root index rather than a root `config.json`, so the index selects the
        // profile and locates the tokenizer component before any other asset
        // is resolved.
        if let Some(index) = resolve_pipeline_index(&config.model).await? {
            let description =
                ModelDescription::from_pipeline_class(&index.class_name).ok_or_else(|| {
                    crate::profile::assets::Error::UnsupportedPipeline {
                        class_name: index.class_name.clone(),
                    }
                })?;
            let tokenizer_path = resolve_model_file(
                &config.model,
                &index.component_file("tokenizer", "tokenizer.json")?,
            )
            .await?;
            let tokenizer: DynTokenizer = Arc::new(HuggingFaceTokenizer::new(&tokenizer_path)?);
            let model = Self::from_pipeline(
                &served_name,
                description,
                config.engine.max_video_seconds,
                config.engine.max_model_len,
            )?;
            return Ok((model, tokenizer, None));
        }

        let files = ResolvedModelFiles::new(&config.model).await?;
        let tokenizer: DynTokenizer = Arc::new(HuggingFaceTokenizer::new(&files.tokenizer_path)?);
        let mut model = Self::from_files(
            &served_name,
            &config.model,
            &files,
            config.engine.max_model_len,
            tokenizer.as_ref(),
        )
        .await?;
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
    ///
    /// `InputProcessor::new` replaces the stored value with `WorkerCapabilities::max_model_tokens`,
    /// so after binding this returns the served limit.
    pub(crate) fn max_model_tokens(&self) -> u32 {
        self.max_model_tokens
            .unwrap_or(EngineSettings::DEFAULT_MAX_MODEL_LEN)
    }

    /// Worker IPC slot capacity this model's messages require in each direction.
    ///
    /// A host product stays in a shared-storage segment on its host; across
    /// hosts its bytes ride the rank channel, in the producing rank's result
    /// and in the consuming rank's batch, so the capacity admits the largest
    /// such product a rank publishes at once. A native FastH3 decode unit
    /// holds 22 RGB frames at 1344x768 (68,124,672 bytes) plus its protocol
    /// envelope. The rings grow to what a message needs, so the capacity costs
    /// nothing until a message uses it.
    pub(crate) fn channel_payload_capacity(&self) -> usize {
        if matches!(self.parameters, ModelParameters::MiniMaxH3 { .. }) {
            72 << 20
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

/// Capabilities the loaded worker advertises during its startup handshake.
///
/// These are facts about the running worker rather than the checkpoint, so they
/// arrive after the engine connects and are bound onto the model once.
pub struct WorkerCapabilities {
    /// Runtime generation features and resource limits.
    pub limits: GenerationLimits,
    /// Sampling controls the engine exposes to served requests.
    pub sampling_controls: Vec<ServedSamplingControl>,
    /// Effective context ceiling after intersecting model and worker limits.
    pub max_model_tokens: u32,
    /// Fixed media prediction count, zero for a worker that serves no video.
    pub denoise_steps: u32,
}

impl InputProcessor {
    /// Binds model resources to verified worker capabilities without rebuilding model data.
    ///
    /// Replaces the model's context ceiling with `worker.max_model_tokens` and, for MiniMax H3,
    /// its denoise-step count with `worker.denoise_steps`; `denoise_steps` is ignored for other
    /// models.
    ///
    /// # Errors
    ///
    /// Returns `ServeError::ModelResolution` with:
    /// - `MissingFeature` when the worker limits do not cover the features the configured
    ///   model needs;
    /// - `MissingTemplate` when a token-generating model has no chat renderer;
    /// - `MediaContract` when the model is MiniMax H3 and the worker advertises zero denoise
    ///   steps.
    pub fn new(
        mut config: ModelConfig,
        tokenizer: DynTokenizer,
        renderer: Option<HfChatRenderer>,
        worker: WorkerCapabilities,
        parse_reasoning: bool,
    ) -> Result<Self> {
        let WorkerCapabilities {
            limits,
            sampling_controls,
            max_model_tokens,
            denoise_steps,
        } = worker;

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
        // Bind the actual worker ceilings once before sharing immutable model facts.
        config.max_model_tokens = Some(max_model_tokens);
        if let ModelParameters::MiniMaxH3 {
            max_video_seconds,
            num_inference_steps,
        } = &mut config.parameters
        {
            validate_video_capacity(*max_video_seconds).map_err(|message| {
                ServeError::ModelResolution(ModelResolutionError::MediaContract(message))
            })?;
            // The denoise-step count belongs to the loaded numerical plan, so
            // the worker handshake is its only authority.
            if denoise_steps == 0 {
                return Err(ServeError::ModelResolution(
                    ModelResolutionError::MediaContract(
                        "worker advertised no denoise steps for a video checkpoint".to_owned(),
                    ),
                ));
            }
            *num_inference_steps = denoise_steps;
        }

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
    ///
    /// Returns `Value::Null` for a model that serves no video; the Dynamo worker uses that to
    /// refuse a non-MiniMax H3 checkpoint. `min_seconds` and `model_max_seconds` are the API's
    /// explicit-duration range; `max_seconds` is the deployment's configured capacity, which is
    /// a deployment limit whenever it is below `model_max_seconds`. The frame rate and default
    /// duration (the lesser of 5 seconds and the capacity) match `video_sampling`.
    pub fn video_capabilities(&self) -> serde_json::Value {
        match &self.config.parameters {
            ModelParameters::MiniMaxH3 {
                max_video_seconds, ..
            } => {
                let default_seconds = default_video_seconds(*max_video_seconds);
                let mut suggested_seconds = vec![default_seconds];
                if *max_video_seconds > default_seconds {
                    suggested_seconds.push(*max_video_seconds);
                }
                serde_json::json!({
                    "tasks": ["t2va"],
                    "default_seconds": default_seconds,
                    "min_seconds": MIN_VIDEO_SECONDS,
                    "max_seconds": *max_video_seconds,
                    "model_max_seconds": MAX_VIDEO_SECONDS,
                    "suggested_seconds": suggested_seconds,
                    "fps": VIDEO_FPS, "width": 1344, "height": 768,
                    "max_prompt_tokens": self.config.max_model_tokens(),
                    "request_fields": ["model", "prompt", "seconds", "seed"],
                })
            }
            _ => serde_json::Value::Null,
        }
    }

    /// Validates the video API request, tokenizes its prompt, and prepares the
    /// checkpoint frame and media unit counts for direct engine submission.
    ///
    /// The returned request carries a placeholder `RequestId(0)`; the caller must replace it
    /// with the identifier reserved by `EngineClient::register_request` before submission.
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
        let (_, sampling) = self.video_sampling(request_id, seconds, seed)?;
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
        Ok(uniserve_core::DiffusionRequest {
            request_id: uniserve_core::RequestId(0),
            task: uniserve_core::VideoTask::T2va,
            text_tags: vec![1; prompt_token_ids.len()],
            prompt_token_ids,
            conditions: Vec::new(),
            media: Vec::new(),
            priority: 0,
            sampling,
        })
    }

    /// Resolves the advertised duration default and the model's frame alignment.
    ///
    /// Returns the requested duration in seconds (the default when omitted) and the diffusion
    /// sampling parameters, whose frame count is the aligned output length `video_frame_count`
    /// derives. The asynchronous video route calls this directly to learn both before
    /// submission.
    ///
    /// # Errors
    ///
    /// Returns an API error when the model serves no video, or when the duration is not a
    /// finite number of seconds within `[MIN_VIDEO_SECONDS, max_video_seconds]`.
    pub fn video_sampling(
        &self,
        request_id: &crate::serving::ServeRequestId,
        seconds: Option<f64>,
        seed: u64,
    ) -> std::result::Result<(f64, uniserve_core::DiffusionSamplingParams), crate::openai::ApiError>
    {
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
        let seconds = seconds.unwrap_or_else(|| default_video_seconds(*max_video_seconds));
        let frame_count = video_frame_count(seconds, *max_video_seconds).map_err(|message| {
            crate::openai::serve_error_to_api(ServeError::Tokenize {
                request_id: request_id.clone(),
                source: crate::serving::TokenizeError::Invalid(message),
            })
        })?;
        // Each H3 video media unit consumes a temporal latent window and emits its
        // non-overlapping frame interval; the model owns overlap reconstruction.
        let video_units = (frame_count - 5) / 17;
        Ok((
            seconds,
            uniserve_core::DiffusionSamplingParams {
                num_frames: frame_count,
                video_units,
                num_inference_steps: *num_inference_steps,
                seed,
                // FastH3's text-only API serves its trained 16:9 canvas.
                width: 1344,
                height: 768,
            },
        ))
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

    /// Declares the endpoints, modalities, features, and sampling controls this model serves.
    ///
    /// Within the server, `validate_generation_features` consults its `features` to refuse
    /// tool-calling and reasoning requests when the model does not declare those features.
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
    ///
    /// Every refusal is `ServeError::UnsupportedFeature`. The final check asks the worker
    /// limits to cover the features this request reaches; for SenseNova and Bagel these
    /// depend on the selected modalities and on whether the request carries an input image.
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
            // Tool definitions, prior assistant tool calls, and tool responses all require a
            // model that parses tool calls.
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
    ///
    /// The returned request carries a placeholder engine identifier (see
    /// `InputProcessor::preprocess_generation`).
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
    ///
    /// The returned `GenerationRequest::request_id` is a placeholder derived from the external
    /// identifier. `EngineClient::submit_generation` accepts only the identifier reserved by
    /// `EngineClient::register_request`, so the caller must replace it before submission.
    ///
    /// # Errors
    ///
    /// Returns `ServeError::UnsupportedFeature` from feature validation and
    /// `ServeError::Tokenize` for every failure during tokenization, validation, and resource
    /// sizing.
    #[allow(clippy::too_many_arguments)]
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
                // An image-generation seed takes precedence over the text sampling seed.
                seed: image_gen
                    .as_ref()
                    .and_then(|image| image.seed)
                    .or_else(|| sampling.seed.map(|seed| seed as u64)),
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
        // The closure gathers every `TokenizeError` so it maps once to `ServeError::Tokenize`.
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
                image_latent_units,
                encoder_cache_pins: generation.num_encoder_cache_entries(),
                replayable: !generation.feeds_back_images(),
            },
        };
        Ok((generation, response))
    }
}

/// Shortest explicit video duration, in seconds, the MiniMax H3 API admits.
pub const MIN_VIDEO_SECONDS: f64 = 4.0;

/// Longest explicit video duration, in seconds, the MiniMax H3 API admits. A deployment may
/// configure a lower `max_video_seconds` capacity, never a higher one.
pub const MAX_VIDEO_SECONDS: f64 = 15.0;

/// Frame rate, in frames per second, of every MiniMax H3 video.
pub const VIDEO_FPS: u32 = 24;

/// Returns the duration, in seconds, of a video request that omits `seconds`.
///
/// The default is 5 seconds, capped at the deployment's `max_video_seconds` so an omitted
/// duration is always within bounds (`validate_video_capacity` keeps the capacity at or above
/// `MIN_VIDEO_SECONDS`). `InputProcessor::video_sampling` resolves omitted durations with it
/// and `InputProcessor::video_capabilities` advertises it; any other video entry point into
/// the same engine must resolve omitted durations with it too.
pub fn default_video_seconds(max_video_seconds: f64) -> f64 {
    max_video_seconds.min(5.0)
}

/// Checks a deployment's video duration capacity against the API range.
///
/// # Errors
///
/// Returns a message when `max_video_seconds` is not finite or lies outside
/// `[MIN_VIDEO_SECONDS, MAX_VIDEO_SECONDS]`: a smaller capacity admits no request, and a
/// larger one exceeds what the API accepts.
pub fn validate_video_capacity(max_video_seconds: f64) -> std::result::Result<(), String> {
    if max_video_seconds.is_finite()
        && (MIN_VIDEO_SECONDS..=MAX_VIDEO_SECONDS).contains(&max_video_seconds)
    {
        Ok(())
    } else {
        Err(format!(
            "max_video_seconds must lie in [{MIN_VIDEO_SECONDS}, {MAX_VIDEO_SECONDS}], got \
             {max_video_seconds}"
        ))
    }
}

/// Converts an explicit MiniMax H3 duration into its output frame count.
///
/// The requested frame count is `seconds * 24` rounded half to even, as the reference
/// implementations round with Python's `round`, and the output extends it upward to the next
/// complete native temporal window, a count of the form `17n + 5`. The output may therefore
/// last longer than requested: 4 seconds yields 107 frames and 15 seconds 362.
///
/// # Errors
///
/// Returns a message when `seconds` is not finite or lies outside
/// `[MIN_VIDEO_SECONDS, max_video_seconds]`.
pub fn video_frame_count(seconds: f64, max_video_seconds: f64) -> std::result::Result<u32, String> {
    if !seconds.is_finite() || !(MIN_VIDEO_SECONDS..=max_video_seconds).contains(&seconds) {
        return Err(format!(
            "video duration must be a finite number of seconds in [{MIN_VIDEO_SECONDS}, \
             {max_video_seconds}]"
        ));
    }
    // The range bounds the product to [96, 360], so the conversion is exact.
    let requested = (seconds * f64::from(VIDEO_FPS)).round_ties_even() as u32;
    Ok(requested + (22 - requested % 17) % 17)
}

/// Returns the multimodal resources required by the active profile.
///
/// Used at load time: evaluates `GenerationConstraint::Default` (both understanding and
/// generation branches enabled) with every configured image encoder. Per-request checks use
/// `omni_required_features`.
fn configured_omni_needs(
    policy: &ImageGenerationConfig,
    image_encoders: &[uniserve_core::ImageEncoderInput],
) -> GenerationFeatures {
    policy.required_features(
        GenerationConstraint::Default,
        image_encoders.iter().map(|input| input.encoder),
    )
}

/// Refuses to bind a model whose required features the worker limits do not cover,
/// reporting the uncovered features as `ModelResolutionError::MissingFeature`.
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
///
/// The profile's context image encoders count only when the request carries an input image;
/// `ImageGenerationConfig::required_features` adds feedback encoders independently of it.
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
                let processor = Qwen3ChatOutputProcessor::new(
                    &mut chat_request,
                    std::sync::Arc::clone(&self.tokenizer),
                    self.parse_reasoning,
                )?;
                let rendered_text = self
                    .renderer
                    .as_ref()
                    .ok_or(crate::serving::chat::Error::MissingChatTemplate)?
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
        // Prompt logprobs require computation for every prompt token, which a prefix-cache
        // read would skip.
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

        let max_tokens = resolve_max_tokens(
            sampling.max_tokens,
            defaults.max_output_tokens,
            Some(self.config.max_model_tokens()),
            prompt_len,
        )?;
        let min_tokens = sampling.min_tokens.unwrap_or(0);

        // Unless the request ignores EOS, the checkpoint's EOS ids other than the primary one
        // become explicit stop tokens.
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

        if min_tokens > max_tokens {
            return Err(crate::serving::TokenizeError::MinTokensExceedsMaximum {
                min_tokens,
                max_tokens,
            });
        }

        let mut core = SamplingParams {
            temperature: defaults.temperature.unwrap_or(1.0),
            top_k: defaults.top_k.unwrap_or(0),
            top_p: defaults.top_p.unwrap_or(1.0),
            min_p: defaults.min_p.unwrap_or(0.0),
            repetition_penalty: defaults.repetition_penalty.unwrap_or(1.0),
            ..SamplingParams::default()
        };
        super::sampling::apply_sampling(&self.tokenizer, sampling, stop, &mut core)?;

        // Logprob requests need `Logprobs` among the served sampling controls, which
        // `EngineClient::served_sampling_controls` includes only when the worker supports
        // token sampling.
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

/// Computes a deterministic identifier for a preprocessed request (64-bit FNV-1a).
///
/// The value is a placeholder: `ServingRuntime` replaces it with the engine-reserved
/// identifier before submission.
fn stable_hash(value: &str) -> u64 {
    value
        .as_bytes()
        .iter()
        .fold(0xcbf29ce484222325, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        })
}
