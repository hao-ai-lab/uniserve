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
//! [`InputProcessor::preprocess_video_request`] produces a `DiffusionRequest` for MiniMax H3
//! through the deployment's `VideoService`.

use std::path::PathBuf;
use std::sync::Arc;

use crate::config::EngineSettings;
use crate::profile::assets::{PipelineCheckpoint, ResolvedModelFiles};
use crate::profile::omni::bagel::BagelProfile;
use crate::profile::omni::sensenova::SenseNovaProfile;
use crate::profile::tokenizer::{DynTokenizer, HuggingFaceTokenizer, TokenizerError};
use crate::profile::{ModelConfig, ModelDescription, ModelParameters};
use thiserror::Error;
use uniserve_core::{
    CachePolicy, GenerationConstraint, GenerationFeatures, GenerationLimits, GenerationRequest,
    ImageGenerationConfig, ImageParams, RequestId, SamplingParams,
};

use crate::serving::chat::template::renderer::hf::MultimodalRenderInfo;
use crate::serving::chat::{
    ChatOutputProcessor, ChatTemplateLoadOptions, HfChatRenderer, Qwen3ChatOutputProcessor,
};
use crate::serving::input::{
    ModelEventIdentity, OutputDetail, OutputProcessorPolicy, PromptInput, ResponseOptions,
    TextPromptRequest,
};
use crate::serving::text::{TextDecodeOptions, resolve_max_tokens};
use crate::serving::video::plan::VisionConfig;
use crate::serving::video::{PreparedVideo, VideoService};
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
    /// TypeSafe System One decision-readout endpoint (`POST /v1/systemone`).
    SystemOne,
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
    pub(super) parse_reasoning: bool,
    // What the deployment's video denoiser serves; set exactly for MiniMax H3.
    video: Option<VideoService>,
    // Whether a Qwen3-family chat template instructs the tool-call format
    // `Qwen3XmlToolParser` reads; false for every other family.
    qwen3_tool_calls: bool,
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

/// The model facts and preprocessing resources startup resolves.
pub(crate) struct LoadedModel {
    /// Immutable model facts.
    pub config: ModelConfig,
    /// The prompt tokenizer.
    pub tokenizer: DynTokenizer,
    /// The chat renderer; `None` for a diffusers pipeline.
    pub renderer: Option<HfChatRenderer>,
    /// The vision processor geometry of a video pipeline's conditioner.
    pub vision: Option<VisionConfig>,
}

/// Loads a diffusers pipeline's `tokenizer` component as the pipeline's own
/// `transformers` tokenizer loads it: `tokenizer.json` with the special
/// tokens its `tokenizer_config.json`, when published, declares.
///
/// The conditioner reads token ids, so a declared token missing here would
/// split into text tokens the reference never presents (MiniMax-H3 declares
/// its `<d>` dialogue marker only in the configuration).
pub(crate) async fn pipeline_tokenizer(
    pipeline: &PipelineCheckpoint,
) -> std::result::Result<HuggingFaceTokenizer, ModelResolutionError> {
    let path = pipeline
        .component_file("tokenizer", "tokenizer.json")
        .await?;
    let tokenizer = match pipeline
        .component_file("tokenizer", "tokenizer_config.json")
        .await
    {
        Ok(config) => HuggingFaceTokenizer::with_config(&path, &config)?,
        Err(crate::profile::assets::Error::MissingFile { .. }) => HuggingFaceTokenizer::new(&path)?,
        Err(error) => return Err(error.into()),
    };
    Ok(tokenizer)
}

impl ModelConfig {
    /// Loads model facts and the tokenizer/template resources needed by preprocessing.
    ///
    /// A diffusers pipeline has no renderer and carries its conditioner's
    /// vision processor geometry instead. The returned `max_model_tokens` is
    /// always set: the configured `max_model_len`, else, for a root
    /// configuration, its `max_position_embeddings` or
    /// `EngineSettings::DEFAULT_MAX_MODEL_LEN`, and for a pipeline the model's
    /// default prompt limit.
    pub(crate) async fn load(
        config: &crate::Config,
    ) -> std::result::Result<LoadedModel, ModelResolutionError> {
        let served_name = config
            .served_model_name
            .clone()
            .unwrap_or_else(|| config.model.clone());
        // A diffusers pipeline declares its class and component folders in a
        // root index rather than a root `config.json`, so the index selects the
        // profile and locates the tokenizer component before any other asset
        // is resolved. A component folder the checkpoint omits is read from
        // the base revision it pins, as its workers read it.
        let mut indexed_description = None;
        if let Some(pipeline) = PipelineCheckpoint::resolve(&config.model).await? {
            let description = ModelDescription::from_pipeline_class(pipeline.class_name())
                .ok_or_else(|| crate::profile::assets::Error::UnsupportedPipeline {
                    class_name: pipeline.class_name().to_owned(),
                })?;
            if description.is_pipeline() {
                let tokenizer: DynTokenizer = Arc::new(pipeline_tokenizer(&pipeline).await?);
                // The conditioner's Qwen3-VL processor fixes how condition media
                // is patched; requests are planned against it.
                let mut processor = Vec::with_capacity(2);
                for name in ["preprocessor_config.json", "video_preprocessor_config.json"] {
                    processor.push(pipeline.component_file("processor", name).await?);
                }
                let vision = VisionConfig::read(&processor[0], &processor[1])
                    .map_err(|error| ModelResolutionError::MediaContract(format!("{error:#}")))?;
                let model = Self::from_pipeline(
                    &served_name,
                    description,
                    config.engine.max_video_seconds,
                    config.engine.max_model_len,
                )?;
                return Ok(LoadedModel {
                    config: model,
                    tokenizer,
                    renderer: None,
                    vision: Some(vision),
                });
            }
            // Root-configured families can publish an index too. Both
            // descriptors must identify the same numerical architecture.
            indexed_description = Some(description);
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
        if let Some(indexed) = indexed_description
            && indexed != model.description()
        {
            return Err(crate::profile::assets::Error::invalid(format!(
                "the checkpoint index names `{}` but config.json describes `{}`",
                indexed.id(),
                model.description().id()
            ))
            .into());
        }
        model.max_model_tokens = Some(
            config
                .engine
                .max_model_len
                .or(model.max_model_tokens)
                .unwrap_or(EngineSettings::DEFAULT_MAX_MODEL_LEN),
        );
        // DiffusionGemma's template renders each image part itself, as one
        // image token that preprocessing expands; the omni models replace
        // image parts before rendering.
        let multimodal = match &model.parameters {
            ModelParameters::DiffusionGemma(profile) => tokenizer
                .id_to_token(profile.tokens.image)
                .map(|placeholder_token| MultimodalRenderInfo { placeholder_token }),
            _ => None,
        };
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
            multimodal,
        )?;
        Ok(LoadedModel {
            config: model,
            tokenizer,
            renderer: Some(renderer),
            vision: None,
        })
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
    /// such product a rank publishes at once. A MiniMax H3 decode unit holds
    /// 22 RGB frames of its canvas. The canvas rule caps a canvas at
    /// 768x1344 pixels before rounding each side to 32 pixels, which adds at
    /// most 4% (68,124,672 bytes for 22 frames at the cap), so 72 MiB holds
    /// the unit and its protocol envelope. The rings grow to what a message
    /// needs, so the capacity costs nothing until a message uses it.
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
            ModelParameters::Qwen3
            | ModelParameters::MiniMaxH3 { .. }
            | ModelParameters::DiffusionGemma(_) => None,
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
            ModelParameters::DiffusionGemma(_) => uniserve_core::RuntimeFamily::BlockDiffusion,
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
            // A readout prefills text and encoded images and denoises its
            // canvases once over them. An image writes one KV entry per soft
            // token; the worker's encoder entries bound the feature bytes and
            // the number of cached encodings, so the model states no bound of
            // its own for either.
            ModelParameters::DiffusionGemma(profile) => uniserve_core::GenerationLimits {
                features: uniserve_core::GenerationFeatures::TOKEN_DENOISING
                    | uniserve_core::GenerationFeatures::VISION_ENCODE,
                latent_downsample: 1,
                max_vit_grid_tokens: profile.images.max_soft_tokens,
                max_vision_feature_bytes: u64::MAX,
                max_cfg_branches: 1,
                encoder_cache_entries: u32::MAX,
                ..Default::default()
            },
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
}

impl InputProcessor {
    /// Binds model resources to verified worker capabilities without rebuilding model data.
    ///
    /// Replaces the model's context ceiling with `worker.max_model_tokens`. `video` is the
    /// video service a MiniMax H3 deployment's denoiser handshake built, and must be absent
    /// for every other model.
    ///
    /// # Errors
    ///
    /// Returns `ServeError::ModelResolution` with:
    /// - `MissingFeature` when the worker limits do not cover the features the configured
    ///   model needs;
    /// - `MissingTemplate` when a token-generating model has no chat renderer;
    /// - `MediaContract` when the model is MiniMax H3 and its duration capacity is out of
    ///   range or it has no video service, or when another model has one.
    pub fn new(
        mut config: ModelConfig,
        tokenizer: DynTokenizer,
        renderer: Option<HfChatRenderer>,
        worker: WorkerCapabilities,
        video: Option<VideoService>,
        parse_reasoning: bool,
    ) -> Result<Self> {
        let WorkerCapabilities {
            limits,
            sampling_controls,
            max_model_tokens,
        } = worker;

        let needs = match &config.parameters {
            ModelParameters::Qwen3 => GenerationFeatures::UNDERSTANDING,
            // The System One readout denoises canvases over its prompt.
            ModelParameters::DiffusionGemma(_) => GenerationFeatures::TOKEN_DENOISING,
            ModelParameters::SenseNova(profile) => {
                configured_omni_needs(&profile.image_generation, &profile.image_encoders)
            }
            ModelParameters::Bagel(profile) => {
                configured_omni_needs(&profile.image_generation, &profile.image_encoders)
            }
            ModelParameters::MiniMaxH3 { .. } => GenerationFeatures::empty(),
        };
        validate_runtime_features(&config, &limits, needs)?;

        let contract = |message: &str| {
            ServeError::ModelResolution(ModelResolutionError::MediaContract(message.to_owned()))
        };
        if let ModelParameters::MiniMaxH3 { max_video_seconds } = &config.parameters {
            validate_video_capacity(*max_video_seconds).map_err(|message| contract(&message))?;
            if video.is_none() {
                return Err(contract(
                    "a video checkpoint is served without its video denoiser",
                ));
            }
        } else {
            if video.is_some() {
                return Err(contract("only a video checkpoint has a video denoiser"));
            }
            if renderer.is_none() {
                return Err(ServeError::ModelResolution(
                    ModelResolutionError::MissingTemplate,
                ));
            }
        }
        // A Qwen3-family checkpoint serves tool calls only when its template
        // asks for the format the Qwen3 parser reads; otherwise requests with
        // tools are refused rather than returned with unparsed calls.
        let qwen3_tool_calls = match (&config.parameters, &renderer) {
            (ModelParameters::Qwen3, Some(renderer)) => {
                crate::serving::chat::output::template_instructs_json_tool_calls(renderer)
            }
            _ => false,
        };
        // Bind the actual worker ceilings once before sharing immutable model facts.
        config.max_model_tokens = Some(max_model_tokens);

        Ok(Self {
            config,
            tokenizer,
            renderer,
            limits,
            sampling_controls,
            parse_reasoning,
            video,
            qwen3_tool_calls,
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

    /// The video service of a MiniMax H3 deployment; `None` for other models.
    pub fn video_service(&self) -> Option<&VideoService> {
        self.video.as_ref()
    }

    /// The served video contract, as `GET /v1/capabilities` reports it.
    ///
    /// Returns `Value::Null` for a model that serves no video; the Dynamo worker uses that to
    /// refuse a non-MiniMax H3 checkpoint. See `VideoService::capabilities`.
    pub fn video_capabilities(&self) -> serde_json::Value {
        self.video
            .as_ref()
            .map_or(serde_json::Value::Null, |video| {
                video.capabilities(self.config.max_model_tokens())
            })
    }

    /// Validates a video request, fetches, probes and plans its media, presents its prompt,
    /// and sizes it for direct engine submission.
    ///
    /// The returned request carries a placeholder `RequestId(0)`; the caller must replace it
    /// with the identifier reserved by `EngineClient::register_request` before submission.
    ///
    /// # Errors
    ///
    /// Returns an API error when the model serves no video, the request names another model,
    /// or `VideoService::prepare` refuses it.
    pub async fn preprocess_video_request(
        &self,
        request_id: &crate::serving::ServeRequestId,
        request: &crate::openai::VideoGenerationRequest,
    ) -> std::result::Result<PreparedVideo, crate::openai::ApiError> {
        let Some(video) = &self.video else {
            return Err(crate::openai::serve_error_to_api(
                ServeError::UnsupportedFeature {
                    request_id: request_id.clone(),
                    feature: "video_generation",
                },
            ));
        };
        crate::openai::utils::check_model_served(&request.model, self.served_model_name())?;
        video
            .prepare(request_id, request, self.config.max_model_tokens())
            .await
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
            ModelParameters::SenseNova(_)
                | ModelParameters::Bagel(_)
                | ModelParameters::DiffusionGemma(_)
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
        if matches!(self.config.parameters, ModelParameters::DiffusionGemma(_)) {
            // Block diffusion commits whole denoised canvases, so of the token
            // sampling controls only the stopping ones apply; temperature,
            // truncation, penalties, and logprobs have no meaning there.
            let sampling_controls = self
                .sampling_controls
                .iter()
                .copied()
                .filter(|control| {
                    matches!(
                        control,
                        ServedSamplingControl::Eos | ServedSamplingControl::StopStrings
                    )
                })
                .collect();
            return ModelSupport {
                endpoints: vec![ServedEndpoint::SystemOne, ServedEndpoint::ChatCompletions],
                input_modalities: vec![ServedModality::Text, ServedModality::Image],
                output_modalities: vec![ServedModality::Text],
                features: vec![
                    ServedFeature::Streaming,
                    ServedFeature::Usage,
                    ServedFeature::Reasoning,
                    ServedFeature::ToolCalling,
                ],
                sampling_controls,
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
                if self.qwen3_tool_calls {
                    features.push(ServedFeature::ToolCalling);
                }
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
            ModelParameters::MiniMaxH3 { .. } | ModelParameters::DiffusionGemma(_) => {
                unreachable!("media and block-diffusion support returned above")
            }
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
    /// limits to cover the features this request reaches; for SenseNova and Bagel these depend
    /// on the selected modalities, and for them and DiffusionGemma on whether the request
    /// carries an input image.
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
            // Block diffusion denoises canvases over a prompt whose images the
            // vision encoder writes into it.
            ModelParameters::DiffusionGemma(_) => {
                GenerationFeatures::TOKEN_DENOISING
                    | if has_input_image {
                        GenerationFeatures::VISION_ENCODE
                    } else {
                        GenerationFeatures::empty()
                    }
            }
            ModelParameters::MiniMaxH3 { .. } => {
                unreachable!("video generation refused above")
            }
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
    /// `images` are the prompt's input images: the top-level images of a text prompt, or the
    /// resolved `image_url` parts of a chat prompt in `chat_image_urls` order.
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
        if matches!(self.config.parameters, ModelParameters::DiffusionGemma(_)) {
            crate::serving::diffusion_gemma::refuse_token_sampling_controls(
                &request_id,
                &sampling,
                &stop,
            )?;
        }
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
            readout: Vec::new(),
            canvas: None,
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
                    ModelParameters::DiffusionGemma(profile) => self
                        .preprocess_diffusion_gemma_input(
                            profile,
                            prompt,
                            &images,
                            &sampling,
                            &mut generation,
                            &mut decode,
                        )?,
                    ModelParameters::MiniMaxH3 { .. } => {
                        unreachable!("generation features checked above")
                    }
                };
                if matches!(
                    self.config.parameters,
                    ModelParameters::Qwen3 | ModelParameters::DiffusionGemma(_)
                ) {
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
                (
                    ids,
                    OutputProcessorPolicy::Chat(ChatOutputProcessor::Qwen3(processor)),
                    skip,
                )
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::fs;

    use serde_json::json;
    use tempfile::TempDir;
    use tokenizers::models::bpe::{BPE, Vocab};
    use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};
    use uniserve_core::{CanvasSampling, GenerationFeatures, GenerationLimits};

    use super::LoadedModel;
    use crate::Config;
    use crate::profile::tokenizer::HuggingFaceTokenizer;
    use crate::profile::{ModelConfig, ModelDescription, ModelParameters};
    use crate::serving::{
        InputProcessor, ServeRequestId, ServedEndpoint, ServedFeature, ServedModality,
        ServedSamplingControl, WorkerCapabilities,
    };

    /// Writes a checkpoint laid out as DiffusionGemma publishes it: a root
    /// transformers `config.json` whose `model_type` is `root_model_type`,
    /// beside a `DiffusionGemmaPipeline` diffusers index. The tokenizer maps
    /// each ASCII character to one token and adds the Gemma control tokens,
    /// including the thought-channel delimiters its reasoning parser reads.
    fn indexed_checkpoint(root_model_type: &str) -> TempDir {
        let directory = tempfile::tempdir().unwrap();
        let mut vocab = Vocab::from_iter([("<unk>".to_owned(), 0_u32)]);
        for code in 1_u32..=127 {
            vocab.insert(char::from_u32(code).unwrap().to_string(), code);
        }
        let model = BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .unk_token("<unk>".to_owned())
            .build()
            .unwrap();
        let mut builder = TokenizerBuilder::new(model);
        builder.add_special_tokens(
            &[
                "<pad>",
                "<eos>",
                "<bos>",
                "<mask>",
                "<turn|>",
                "<|image>",
                "<|image|>",
                "<image|>",
                "<|channel>",
                "<channel|>",
            ]
            .map(|token| AddedToken::from(token, true)),
        );
        let tokenizer_path = directory.path().join("tokenizer.json");
        builder.save(&tokenizer_path, false).unwrap();
        let tokenizer = HuggingFaceTokenizer::new(&tokenizer_path).unwrap();
        let id = |token: &str| tokenizer.token_to_id(token).unwrap();

        let write = |name: &str, value: serde_json::Value| {
            fs::write(directory.path().join(name), value.to_string()).unwrap();
        };
        write(
            "tokenizer_config.json",
            json!({"bos_token": "<bos>", "eos_token": "<eos>",
                   "chat_template": "{{ bos_token }}{{ messages[0]['content'] }}"}),
        );
        write(
            "config.json",
            json!({
                "architectures": ["DiffusionGemmaForBlockDiffusion"],
                "model_type": root_model_type,
                "canvas_length": 256,
                "image_token_id": id("<|image|>"),
                "boi_token_id": id("<|image>"),
                "eoi_token_id": id("<image|>"),
                "vision_soft_tokens_per_image": 280,
                "vision_config": {"patch_size": 16, "pooling_kernel_size": 3},
                "text_config": {"model_type": "diffusion_gemma_text", "max_position_embeddings": 262144},
            }),
        );
        write(
            "generation_config.json",
            json!({
                "eos_token_id": [1, 106, 50],
                "max_new_tokens": 256,
                "max_denoising_steps": 48,
                "sampler_config": {"_cls_name": "EntropyBoundSamplerConfig", "entropy_bound": 0.1},
                "t_min": 0.4,
                "t_max": 0.8,
                "confidence_threshold": 0.005,
                "stability_threshold": 1,
            }),
        );
        write(
            "model_index.json",
            json!({
                "_class_name": "DiffusionGemmaPipeline",
                "model": ["transformers", "DiffusionGemmaForBlockDiffusion"],
                "scheduler": ["diffusers", "BlockRefinementScheduler"],
            }),
        );
        directory
    }

    fn config(directory: &TempDir) -> Config {
        Config {
            model: directory.path().to_str().unwrap().to_owned(),
            served_model_name: Some("diffusion-gemma".to_owned()),
            ..Config::default()
        }
    }

    /// The diffusers index beside the root configuration does not make the
    /// checkpoint a pipeline: the root configuration resolves the model, its
    /// text limit from `text_config`, and its block-diffusion settings.
    #[tokio::test]
    async fn an_indexed_diffusion_gemma_checkpoint_resolves_from_its_root_config() {
        let directory = indexed_checkpoint("diffusion_gemma");

        let LoadedModel {
            config: model,
            tokenizer,
            renderer,
            ..
        } = ModelConfig::load(&config(&directory)).await.unwrap();

        assert_eq!(model.description(), ModelDescription::DiffusionGemma);
        assert_eq!(model.max_model_tokens, Some(262_144));
        assert!(model.eos_token_ids.is_superset(&[1, 50, 106].into()));
        assert!(renderer.is_some());
        let ModelParameters::DiffusionGemma(profile) = &model.parameters else {
            unreachable!("the description is DiffusionGemma");
        };
        assert_eq!(profile.canvas_length, 256);
        assert_eq!(
            profile.tokens.mask,
            tokenizer.token_to_id("<mask>").unwrap()
        );
        assert_eq!(
            profile.tokens.turn_end,
            tokenizer.token_to_id("<turn|>").unwrap()
        );
        assert_eq!(profile.denoising.max_denoising_steps, 48);
        assert!(!profile.quantized);
        assert_eq!(
            model.runtime_family(),
            uniserve_core::RuntimeFamily::BlockDiffusion
        );
    }

    #[tokio::test]
    async fn a_checkpoint_index_must_name_the_family_of_its_root_config() {
        let directory = indexed_checkpoint("qwen3");

        let error = ModelConfig::load(&config(&directory)).await.err().unwrap();

        let message = error.to_string();
        assert!(
            message.contains("`diffusion_gemma`") && message.contains("`qwen3`"),
            "{message}"
        );
    }

    /// Binds a DiffusionGemma checkpoint to a block-diffusion worker whose
    /// context holds `max_model_tokens` tokens.
    async fn diffusion_gemma_processor(
        directory: &TempDir,
        max_model_tokens: u32,
    ) -> InputProcessor {
        let LoadedModel {
            config: model,
            tokenizer,
            renderer,
            ..
        } = ModelConfig::load(&config(directory)).await.unwrap();
        InputProcessor::new(
            model,
            tokenizer,
            renderer,
            WorkerCapabilities {
                limits: GenerationLimits {
                    features: GenerationFeatures::TOKEN_DENOISING
                        | GenerationFeatures::VISION_ENCODE,
                    latent_downsample: 1,
                    max_cfg_branches: 1,
                    max_vit_grid_tokens: 280,
                    max_vision_feature_bytes: u64::MAX,
                    encoder_cache_entries: u32::MAX,
                    ..Default::default()
                },
                sampling_controls: ServedSamplingControl::ALL.to_vec(),
                max_model_tokens,
            },
            None,
            true,
        )
        .unwrap()
    }

    fn chat_request(fields: serde_json::Value) -> crate::openai::ChatCompletionRequest {
        let mut request = json!({
            "model": "diffusion-gemma",
            "messages": [{"role": "user", "content": "Hello"}],
        });
        request
            .as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        serde_json::from_value(request).unwrap()
    }

    /// DiffusionGemma declares the System One readout and chat completions,
    /// with only the stopping controls of token sampling.
    #[tokio::test]
    async fn diffusion_gemma_declares_its_endpoints() {
        let directory = indexed_checkpoint("diffusion_gemma");
        let processor = diffusion_gemma_processor(&directory, 65_536).await;

        let support = processor.support();
        assert_eq!(
            support.endpoints,
            [ServedEndpoint::SystemOne, ServedEndpoint::ChatCompletions]
        );
        assert_eq!(
            support.input_modalities,
            [ServedModality::Text, ServedModality::Image]
        );
        assert_eq!(support.output_modalities, [ServedModality::Text]);
        assert_eq!(
            support.features,
            [
                ServedFeature::Streaming,
                ServedFeature::Usage,
                ServedFeature::Reasoning,
                ServedFeature::ToolCalling
            ]
        );
        assert_eq!(
            support.sampling_controls,
            [
                ServedSamplingControl::Eos,
                ServedSamplingControl::StopStrings
            ]
        );
    }

    /// A chat request generates its reply in canvases under the checkpoint's
    /// block-diffusion sampling, seeded by the request or, without a seed,
    /// by a fresh random one, and truncated at `max_completion_tokens`.
    #[tokio::test]
    async fn a_diffusion_gemma_chat_reply_is_generated_in_canvases() {
        let directory = indexed_checkpoint("diffusion_gemma");
        let processor = diffusion_gemma_processor(&directory, 65_536).await;

        let (generation, _) = processor
            .preprocess_chat_request(
                ServeRequestId::new("chat"),
                chat_request(json!({
                    "seed": 5,
                    "max_completion_tokens": 300,
                    "stop": ["END"],
                })),
                Vec::new(),
            )
            .unwrap();
        assert_eq!(
            generation.canvas,
            Some(CanvasSampling {
                canvas_length: 256,
                max_steps: 48,
                entropy_bound: 0.1,
                t_min: 0.4,
                t_max: 0.8,
                confidence_threshold: 0.005,
                stability_threshold: 1,
            })
        );
        assert_eq!(generation.sampling.seed, Some(5));
        assert_eq!(generation.max_und_tokens, 300);
        assert_eq!(generation.stop_strings, ["END"]);
        let tokenizer =
            HuggingFaceTokenizer::new(&directory.path().join("tokenizer.json")).unwrap();
        assert_eq!(
            generation.prompt_token_ids,
            tokenizer.encode("<bos>Hello", false).unwrap()
        );
        assert!(generation.multimodal_inputs.images.is_empty());

        let (unseeded, _) = processor
            .preprocess_chat_request(
                ServeRequestId::new("unseeded"),
                chat_request(json!({})),
                Vec::new(),
            )
            .unwrap();
        assert!(unseeded.sampling.seed.is_some());
        // The checkpoint's `max_new_tokens` is the default length.
        assert_eq!(unseeded.max_und_tokens, 256);
    }

    /// A reply fills only whole canvases of the context the prompt leaves.
    #[tokio::test]
    async fn a_diffusion_gemma_reply_fits_whole_canvases_of_the_context() {
        let directory = indexed_checkpoint("diffusion_gemma");
        // `<bos>Hello` is six tokens, which leave 594 positions: two canvases.
        let processor = diffusion_gemma_processor(&directory, 600).await;
        let (generation, _) = processor
            .preprocess_chat_request(
                ServeRequestId::new("long"),
                chat_request(json!({"max_completion_tokens": 1000})),
                Vec::new(),
            )
            .unwrap();
        assert_eq!(generation.max_und_tokens, 512);

        let processor = diffusion_gemma_processor(&directory, 200).await;
        let error = processor
            .preprocess_chat_request(
                ServeRequestId::new("full"),
                chat_request(json!({})),
                Vec::new(),
            )
            .err()
            .unwrap();
        assert_eq!(error.status_code().as_u16(), 400);
        let message = error.to_error_response().error.message;
        assert!(message.contains("256-token canvas"), "{message}");
    }

    /// An image part renders as the image token and expands into the image's
    /// soft-token run, where the engine writes the image's vision features.
    #[tokio::test]
    async fn a_diffusion_gemma_chat_image_enters_at_its_part() {
        let directory = indexed_checkpoint("diffusion_gemma");
        let processor = diffusion_gemma_processor(&directory, 65_536).await;
        let mut png = Vec::new();
        image::RgbImage::new(64, 48)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let image = crate::serving::ImageInput::from_bytes(png).unwrap();

        let request = chat_request(json!({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                {"type": "text", "text": "What is this?"},
            ]}],
        }));
        let (generation, _) = processor
            .preprocess_chat_request(ServeRequestId::new("image"), request, vec![image])
            .unwrap();

        // The image's soft-token run leaves the prompt between `<|image>` and
        // `<image|>`; the image enters there with one position and one KV
        // entry per soft token, as many as the Gemma-4 processor sizes.
        let tokenizer =
            HuggingFaceTokenizer::new(&directory.path().join("tokenizer.json")).unwrap();
        let image_start = tokenizer.token_to_id("<|image>").unwrap();
        let [input] = generation.multimodal_inputs.images.as_slice() else {
            panic!("{:?}", generation.multimodal_inputs.images);
        };
        let ModelParameters::DiffusionGemma(profile) = &processor.config().parameters else {
            unreachable!("the checkpoint is DiffusionGemma");
        };
        let soft_tokens = profile.images.soft_tokens(64, 48).unwrap();
        assert_eq!(input.num_positions, soft_tokens);
        assert_eq!(input.encoders[0].num_kv_tokens, Some(soft_tokens));
        let position = input.position as usize;
        assert_eq!(generation.prompt_token_ids[position - 1], image_start);
        assert_eq!(
            generation.prompt_token_ids[position],
            tokenizer.token_to_id("<image|>").unwrap()
        );
        assert!(
            !generation
                .prompt_token_ids
                .contains(&tokenizer.token_to_id("<|image|>").unwrap())
        );
    }

    /// Token-sampling controls and grammar constraints have no meaning for a
    /// denoised canvas; each is refused with a 400 that names it. Grammar
    /// constraints (`response_format` and guided decoding) have no field in
    /// the chat schema, so the request body itself is refused, naming the
    /// field.
    #[tokio::test]
    async fn diffusion_gemma_refuses_token_sampling_controls() {
        for field in ["response_format", "guided_json", "structured_outputs"] {
            let error = serde_json::from_value::<crate::openai::ChatCompletionRequest>(json!({
                "model": "diffusion-gemma",
                "messages": [{"role": "user", "content": "Hello"}],
                field: {"type": "json_object"},
            }))
            .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains(&format!("unknown field `{field}`")),
                "{error}"
            );
        }

        let directory = indexed_checkpoint("diffusion_gemma");
        let processor = diffusion_gemma_processor(&directory, 65_536).await;
        for (control, value) in [
            ("temperature", json!(0.7)),
            ("top_p", json!(0.9)),
            ("top_k", json!(5)),
            ("min_p", json!(0.1)),
            ("frequency_penalty", json!(0.5)),
            ("presence_penalty", json!(0.5)),
            ("repetition_penalty", json!(1.1)),
            ("logit_bias", json!({"5": 1.0})),
            ("allowed_token_ids", json!([5])),
            ("bad_words", json!(["x"])),
            ("logprobs", json!(true)),
            ("prompt_logprobs", json!(1)),
            ("min_tokens", json!(3)),
            ("ignore_eos", json!(true)),
        ] {
            let error = processor
                .preprocess_chat_request(
                    ServeRequestId::new("refused"),
                    chat_request(json!({ control: value })),
                    Vec::new(),
                )
                .err()
                .unwrap();
            assert_eq!(error.status_code().as_u16(), 400, "{control}");
            let body = error.to_error_response().error;
            assert_eq!(body.param.as_deref(), Some(control));
            assert!(body.message.contains(control), "{}", body.message);
        }
    }
}
