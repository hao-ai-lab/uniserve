// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental in-process Dynamo worker for UniServe FastH3.
//!
//! This reference integration embeds UniServe's Rust scheduler and engine in
//! the Dynamo worker process. The preferred production boundary remains the
//! native HTTP sidecar, which does not couple either project's dependencies.
//!
//! The Dynamo frontend forwards each `/v1/videos` request to
//! `RawEngine::generate`, nesting unknown client fields under `extra_args`.
//! The worker serves FastH3 text-to-video-and-audio at the canvases its
//! `--video-resolutions` and `--video-aspect-ratios` prepare, and refuses at
//! startup a deployment that serves any other task or canvas, such as a
//! MiniMax-H3 base checkpoint: `prepare_request` maps the Dynamo request onto
//! a `t2va` request of UniServe's video API and refuses any control FastH3
//! does not implement rather than ignoring it. The request then enters
//! `ServingRuntime::generate_video`, the lifecycle the HTTP `/v1/videos`
//! route uses, and a successful request yields one terminal response object
//! carrying the single MP4 artifact.
//! UniServe's own HTTP listener is never started.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use base64::Engine as _;
use clap::Parser;
use dynamo_backend_common::{
    AsyncEngineContext, BackendError, CommonArgs, DisaggregationMode, DynamoError, EngineConfig,
    ErrorType, GenerateContext, ModelInput, RawEngine, WorkerConfig as DynamoWorkerConfig,
};
use futures::StreamExt as _;
use futures::stream::BoxStream;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tokio::sync::OnceCell;
use tokio_util::sync::CancellationToken;
use uniserve_core::{MediaKind, VideoTask};
use uniserve_engine::{
    DEFAULT_LONG_PREFILL_THRESHOLD, DEFAULT_MAX_BATCH, DEFAULT_MAX_NUM_BATCHED_TOKENS,
    SchedulingPolicy, WorkerConfig, WorkerProcessArgs,
};
use uniserve_server::{
    AppState, Config, EngineSettings, HttpListenerMode, ModelDescription,
    openai::{VideoGenerationRequest, VideoTarget, serve_error_to_api},
    profile::omni::resolution::ResolutionName,
    profile::video::{VideoRasters, VideoResolution},
    serving::{
        FinishStatus, RequestOutput, ServeRequestId, VIDEO_FPS, validate_video_capacity,
        video_frame_count,
    },
};

// The served canvases come from `--video-resolutions` and
// `--video-aspect-ratios`, as for `uniserve serve`; the loaded checkpoint
// determines the step count.
const H3_AUDIO_SAMPLE_RATE: i32 = 32_000;
/// The duration of a Dynamo request that omits `seconds`, at most the
/// deployment's capacity.
const DEFAULT_SECONDS: f64 = 5.0;

#[derive(Clone, Parser)]
#[command(
    name = "uniserve-dynamo-worker",
    about = "Experimental in-process Dynamo worker for UniServe FastH3."
)]
struct Args {
    #[command(flatten)]
    common: CommonArgs,

    /// Complete local FastH3 VSA checkpoint root.
    #[arg(long)]
    model_path: String,

    #[arg(long, default_value = "FastH3")]
    served_model_name: String,

    #[arg(long, default_value = "python3")]
    worker_python: PathBuf,

    /// JSON deployment configuration: the Worker instances, their node/device
    /// ranks, and the FastH3 components placed on them, for example
    /// `configs/fast_h3/ulysses4.json`.
    #[arg(long, value_name = "FILE", value_parser = read_workers)]
    workers: Box<[WorkerConfig]>,

    /// This process's host identity. The engine owns exactly the placed ranks
    /// whose node names it.
    #[arg(long, default_value = "localhost")]
    host_identity: String,

    // These options mirror the `uniserve serve` flags of the same names and
    // feed the engine configuration `build_state` assembles.
    #[arg(long, default_value_t = 16_384, value_parser = clap::value_parser!(u32).range(1..))]
    max_model_len: u32,

    #[arg(long, default_value_t = 15.0)]
    max_video_seconds: f64,

    #[arg(long, value_delimiter = ',', default_value = "768p")]
    video_resolutions: Vec<VideoResolution>,

    #[arg(long, value_delimiter = ',', default_value = "16:9,9:16")]
    video_aspect_ratios: Vec<ResolutionName>,

    #[arg(long, default_value_t = 2)]
    max_running_requests: usize,

    #[arg(long, default_value = r#"{}"#, value_parser = parse_json_object)]
    quantization_config: Value,

    #[arg(long, default_value = "auto", value_parser = ["off", "auto", "full"])]
    graph_policy: String,
}

/// Reads and validates a deployment configuration file.
fn read_workers(path: &str) -> Result<Box<[WorkerConfig]>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("could not read deployment configuration {path}: {error}"))?;
    let workers: Vec<WorkerConfig> = serde_json::from_str(&text)
        .map_err(|error| format!("invalid deployment configuration {path}: {error}"))?;
    WorkerConfig::validate_all(&workers).map_err(|error| error.to_string())?;
    Ok(workers.into_boxed_slice())
}

/// Parses a command-line value that must be a JSON object.
fn parse_json_object(raw: &str) -> Result<Value, String> {
    let value: Value = serde_json::from_str(raw).map_err(|error| error.to_string())?;
    value
        .is_object()
        .then_some(value)
        .ok_or_else(|| "expected a JSON object".to_string())
}

/// Dynamo `RawEngine` serving FastH3 video generation from an in-process
/// UniServe engine.
pub struct DynamoFastH3Engine {
    args: Args,
    /// The validated product of `--video-resolutions` and
    /// `--video-aspect-ratios`.
    rasters: VideoRasters,
    /// The UniServe serving state, set once by `start`. Every other trait
    /// method reads it and treats an unset state as not started.
    state: OnceCell<Arc<AppState>>,
    /// Cancelled by `cleanup`; every in-flight `generate` stream then aborts
    /// its UniServe request and ends without a response.
    cancel: CancellationToken,
}

impl DynamoFastH3Engine {
    /// Parses the process command line into an unstarted engine and the
    /// Dynamo worker registration that advertises it as a `videos` endpoint.
    ///
    /// Clap exits the process on a malformed command line, including an
    /// unreadable or invalid `--workers` deployment file. Settings this
    /// worker cannot honor (disaggregation, encoder routing, RL routes, a
    /// non-finite or non-positive video length, a zero request limit, an
    /// empty, repeated or untrained video raster selection) return
    /// an invalid-argument error. No rank starts until `RawEngine::start`.
    pub fn from_args() -> Result<(Self, DynamoWorkerConfig), DynamoError> {
        Self::try_from_args(<Args as Parser>::parse())
    }

    fn try_from_args(args: Args) -> Result<(Self, DynamoWorkerConfig), DynamoError> {
        if args.common.disaggregation_mode != DisaggregationMode::Aggregated {
            return Err(invalid_argument(
                "UniServe FastH3 supports only aggregated Dynamo workers",
            ));
        }
        if args.common.route_to_encoder {
            return Err(invalid_argument("route-to-encoder is not supported"));
        }
        if args.common.enable_rl {
            return Err(invalid_argument("RL engine routes are not supported"));
        }
        validate_video_capacity(args.max_video_seconds).map_err(invalid_argument)?;
        if args.max_running_requests == 0 {
            return Err(invalid_argument("max-running-requests must be positive"));
        }
        let rasters = VideoRasters::new(&args.video_resolutions, &args.video_aspect_ratios)
            .map_err(invalid_argument)?;

        let config = DynamoWorkerConfig {
            namespace: args.common.namespace.clone(),
            component: args.common.component.clone(),
            endpoint: args.common.endpoint.clone(),
            endpoint_types: "videos".to_string(),
            model_name: args.served_model_name.clone(),
            served_model_name: Some(args.served_model_name.clone()),
            model_input: ModelInput::Text,
            custom_jinja_template: None,
            tool_call_parser: None,
            reasoning_parser: None,
            exclude_tools_when_tool_choice_none: true,
            enable_local_indexer: false,
            enable_kv_routing: false,
            disaggregation_mode: DisaggregationMode::Aggregated,
            route_to_encoder: false,
            enable_rl: false,
            ..Default::default()
        };
        Ok((
            Self {
                args,
                rasters,
                state: OnceCell::new(),
                cancel: CancellationToken::new(),
            },
            config,
        ))
    }

    /// Starts UniServe's engine and worker ranks for the configured FastH3
    /// checkpoint and returns the serving state.
    ///
    /// Fails with an invalid argument when the checkpoint's pipeline index
    /// cannot be resolved or does not name MiniMax H3, when the configuration
    /// does not validate, or when the started model reports no video
    /// capabilities. It fails with an engine error when UniServe cannot
    /// start. A refusal after startup shuts the engine down before returning.
    async fn build_state(&self) -> Result<Arc<AppState>, DynamoError> {
        // A MiniMax H3 checkpoint names its pipeline class in a root index in
        // place of config.json; refuse anything else before a rank starts.
        let pipeline =
            uniserve_server::profile::assets::resolve_pipeline_index(&self.args.model_path)
                .await
                .map_err(|error| invalid_argument(error.to_string()))?;
        if pipeline.and_then(|index| ModelDescription::from_pipeline_class(&index.class_name))
            != Some(ModelDescription::MiniMaxH3)
        {
            return Err(invalid_argument(
                "model-path is not a FastH3 video-generation checkpoint",
            ));
        }
        let worker_process = WorkerProcessArgs {
            python: self.args.worker_python.clone(),
            model: self.args.model_path.clone(),
            host: self.args.host_identity.clone(),
            quantization_config: self.args.quantization_config.clone(),
            graph_policy: self.args.graph_policy.clone(),
            ..WorkerProcessArgs::default()
        };

        // Only `uniserve_server::http` binds `listener_mode`, and this worker
        // never serves UniServe's HTTP API; a loopback ephemeral address
        // satisfies `Config::validate`.
        let config = Config {
            engine: EngineSettings {
                max_batch: DEFAULT_MAX_BATCH,
                max_num_batched_tokens: DEFAULT_MAX_NUM_BATCHED_TOKENS,
                max_num_seqs: self.args.max_running_requests,
                long_prefill_threshold: DEFAULT_LONG_PREFILL_THRESHOLD,
                mixed_prefill_tokens: 0,
                prefix_cache: true,
                scheduler_policy: SchedulingPolicy::Fcfs,
                max_model_len: Some(self.args.max_model_len),
                max_video_seconds: self.args.max_video_seconds,
                video_resolutions: self.args.video_resolutions.clone(),
                video_aspect_ratios: self.args.video_aspect_ratios.clone(),
                // The Dynamo request form carries no conditions, so the
                // worker provisions no condition rows.
                max_condition_rows: 0,
                workers: self.args.workers.to_vec(),
                transfer: Default::default(),
                data_parallel_size: 1,
                expert_parallel: None,
                worker_process,
            },
            model: self.args.model_path.clone(),
            served_model_name: Some(self.args.served_model_name.clone()),
            listener_mode: HttpListenerMode::BindTcp {
                host: "127.0.0.1".to_string(),
                port: 0,
            },
            log_stats: true,
            ..Config::default()
        };
        config
            .validate()
            .map_err(|error| invalid_argument(error.to_string()))?;
        let state = uniserve_server::build_state(&config)
            .await
            .map_err(|error| engine_error(format!("failed to start UniServe: {error:#}")))?;

        // Use the same checkpoint-derived schedule as the HTTP serving path.
        // A request may confirm that schedule, but cannot override it.
        let capabilities = state.runtime().model().video_capabilities();
        if capabilities.is_null() {
            let _ = state.engine().shutdown().await;
            return Err(invalid_argument("checkpoint is not MiniMax H3"));
        }
        if let Err(message) = check_fasth3(&capabilities, &self.rasters) {
            let _ = state.engine().shutdown().await;
            return Err(invalid_argument(message));
        }
        Ok(state)
    }
}

#[async_trait]
impl RawEngine for DynamoFastH3Engine {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        // The check refuses a second start before it launches ranks; `set`
        // refuses the second of two starts that raced past it.
        if self.state.initialized() {
            return Err(engine_error("UniServe engine already started"));
        }
        let state = self.build_state().await?;
        self.state
            .set(state)
            .map_err(|_| engine_error("UniServe engine already started"))?;
        Ok(EngineConfig {
            model: self.args.served_model_name.clone(),
            served_model_name: Some(self.args.served_model_name.clone()),
            runtime_data: runtime_data(self.args.max_video_seconds, &self.rasters),
            llm: None,
            ..Default::default()
        })
    }

    async fn generate(
        &self,
        request: Value,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<Value, DynamoError>>, DynamoError> {
        let state = Arc::clone(
            self.state
                .get()
                .ok_or_else(|| engine_error("generate called before start"))?,
        );
        let prepared = prepare_request(
            request,
            &self.args.served_model_name,
            self.args.max_video_seconds,
            &self.rasters,
            state
                .runtime()
                .model()
                .video_service()
                .map_or(0, |video| video.num_inference_steps()),
        )?;
        // The Dynamo context id is the UniServe request id, so `abort` can name
        // the same request from its own context.
        let request_id = ServeRequestId::new(ctx.id().to_string());

        // The serving runtime owns the request lifecycle the HTTP video route
        // uses: identity registration, prompt preprocessing, submission and
        // terminal accounting.
        let (_, mut events) = state
            .runtime()
            .generate_video(request_id.clone(), prepared.request)
            .await
            .map_err(api_error)?;
        let cancel = self.cancel.clone();
        let served_model_name = self.args.served_model_name.clone();
        let response_format = prepared.response_format;

        // Unless stopped, the stream yields exactly one item: the response
        // object, or the error that ended the request. A Dynamo stop or
        // `cleanup` instead aborts the UniServe request and ends the stream
        // without an item; abort failures are ignored.
        Ok(Box::pin(async_stream::stream! {
            let started_at = Instant::now();
            let mut artifact = None;
            loop {
                // `biased` polls the stop and shutdown signals first, so they
                // take precedence over an event that is ready at the same time.
                let event = tokio::select! {
                    biased;
                    _ = ctx.stopped() => {
                        let _ = state.runtime().abort(request_id.clone()).await;
                        return;
                    }
                    _ = cancel.cancelled() => {
                        let _ = state.runtime().abort(request_id.clone()).await;
                        return;
                    }
                    event = events.next() => event,
                };
                match event {
                    Some(Ok(RequestOutput::Artifact(value))) => {
                        if artifact.replace(value).is_some() {
                            yield Err(engine_error("UniServe produced multiple media artifacts"));
                            return;
                        }
                    }
                    Some(Ok(RequestOutput::Finished { reason: FinishStatus::Stop { .. }, .. })) => break,
                    Some(Ok(RequestOutput::Finished { reason, .. })) => {
                        yield Err(engine_error(format!("video generation ended without an artifact: {reason:?}")));
                        return;
                    }
                    Some(Ok(RequestOutput::Rejected { kind, message, .. })) => {
                        yield Err(api_error(uniserve_server::openai::ApiError::rejected(kind, message)));
                        return;
                    }
                    Some(Ok(RequestOutput::Failed { message, .. })) => {
                        yield Err(engine_error(message));
                        return;
                    }
                    // Lifecycle, usage and progress events have no place in
                    // the single non-streaming response.
                    Some(Ok(
                        RequestOutput::Accepted { .. }
                        | RequestOutput::Usage { .. }
                        | RequestOutput::Scheduled { .. }
                        | RequestOutput::MediaProgress { .. },
                    )) => {}
                    Some(Ok(_)) => {
                        yield Err(engine_error("UniServe emitted an incompatible event"));
                        return;
                    }
                    Some(Err(error)) => {
                        yield Err(api_error(serve_error_to_api(error)));
                        return;
                    }
                    None => {
                        yield Err(engine_error("UniServe generation stream stopped"));
                        return;
                    }
                }
            }
            let Some(artifact) = artifact else {
                yield Err(engine_error("UniServe produced no media artifact"));
                return;
            };
            if artifact.media_kind != MediaKind::Video || artifact.content_type != "video/mp4" {
                yield Err(engine_error(format!(
                    "UniServe produced {:?} with content type {:?}, expected video/mp4",
                    artifact.media_kind, artifact.content_type
                )));
                return;
            }
            yield Ok(video_response(
                &request_id,
                &served_model_name,
                response_format,
                artifact.media.as_bytes(),
                started_at.elapsed(),
            ));
        }))
    }

    async fn abort(&self, ctx: Arc<dyn AsyncEngineContext>) {
        if let Some(state) = self.state.get() {
            let _ = state
                .runtime()
                .abort(ServeRequestId::new(ctx.id().to_string()))
                .await;
        }
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        self.cancel.cancel();
        if let Some(state) = self.state.get() {
            state
                .engine()
                .shutdown()
                .await
                .map_err(|error| engine_error(format!("UniServe shutdown failed: {error}")))?;
        }
        Ok(())
    }
}

/// The request body Dynamo's frontend forwards: the fields of its
/// `NvCreateVideoRequest` as dispatched to a worker.
///
/// Unknown fields are refused. The frontend moves unknown top-level client
/// fields under `extra_args["media_passthrough"]`; `prepare_request` reads
/// `resolution` and `aspect_ratio` from there and refuses any other extra
/// argument.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DynamoVideoRequest {
    model: String,
    prompt: String,
    #[serde(default)]
    input_reference: Option<String>,
    #[serde(default)]
    seconds: Option<i32>,
    #[serde(default)]
    size: Option<String>,
    #[serde(default)]
    user: Option<String>,
    #[serde(default)]
    response_format: Option<ResponseFormat>,
    #[serde(default)]
    output_format: Option<String>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    nvext: Option<VideoNvExt>,
    #[serde(default)]
    extra_args: Option<Map<String, Value>>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ResponseFormat {
    #[default]
    Url,
    B64Json,
}

/// Dynamo's `nvext` video extensions. `prepare_request` accepts only the
/// fields FastH3 can honor: `seed` at any non-negative value, and `fps`,
/// `num_frames` and `num_inference_steps` only at the values FastH3 runs with.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct VideoNvExt {
    #[serde(default)]
    annotations: Option<Vec<String>>,
    #[serde(default)]
    fps: Option<i32>,
    #[serde(default)]
    num_frames: Option<i32>,
    #[serde(default)]
    negative_prompt: Option<String>,
    #[serde(default)]
    num_inference_steps: Option<i32>,
    #[serde(default)]
    guidance_scale: Option<f32>,
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default)]
    boundary_ratio: Option<f32>,
    #[serde(default)]
    guidance_scale_2: Option<f32>,
}

#[derive(Debug)]
struct PreparedRequest {
    request: VideoGenerationRequest,
    response_format: ResponseFormat,
}

/// Maps a Dynamo video request onto a `t2va` `VideoGenerationRequest` whose
/// target is the selected raster and the request's duration.
///
/// The raster is selected by `size` (`WIDTHxHEIGHT`, one of the served
/// rasters), by the passthrough `resolution` and `aspect_ratio` fields, or by
/// both when they name the same raster; omitted, it is the deployment's
/// default. Its resolution class names the target's short edge.
/// Omitted fields default to seed 0, a `url` response, and 5 seconds, capped
/// at `max_video_seconds`. `nvext.num_frames` is only checked against the
/// aligned frame count of `seconds`; UniServe derives the frame count from
/// the target duration again when it prepares the request.
/// `nvext.num_inference_steps` counts denoiser evaluations, `denoise_steps`.
/// Every refusal is an invalid-argument error.
fn prepare_request(
    value: Value,
    served_model_name: &str,
    max_video_seconds: f64,
    rasters: &VideoRasters,
    denoise_steps: u32,
) -> Result<PreparedRequest, DynamoError> {
    let request: DynamoVideoRequest = serde_json::from_value(value)
        .map_err(|error| invalid_argument(format!("invalid video request: {error}")))?;
    if request.model != served_model_name {
        return Err(invalid_argument(format!(
            "model {:?} is not served; expected {:?}",
            request.model, served_model_name
        )));
    }
    if request.prompt.trim().is_empty() {
        return Err(invalid_argument("prompt must not be empty"));
    }
    reject_present("input_reference", request.input_reference.as_ref())?;
    reject_present("user", request.user.as_ref())?;
    if request.stream == Some(true) {
        return Err(invalid_argument("stream=true is not supported"));
    }
    if request
        .output_format
        .as_deref()
        .is_some_and(|format| !format.eq_ignore_ascii_case("mp4"))
    {
        return Err(invalid_argument("output_format must be mp4"));
    }
    let (mut resolution, mut aspect_ratio) = passthrough_raster(request.extra_args)?;
    if let Some(size) = request.size.as_deref() {
        let raster = size
            .split_once('x')
            .and_then(|(width, height)| Some((width.parse().ok()?, height.parse().ok()?)))
            .and_then(|(width, height)| rasters.find_size(width, height))
            .ok_or_else(|| {
                invalid_argument(format!(
                    "size must be one of {}",
                    served_sizes(rasters).join(", ")
                ))
            })?;
        if resolution.is_some_and(|value| value != raster.resolution)
            || aspect_ratio.is_some_and(|value| value != raster.aspect_ratio)
        {
            return Err(invalid_argument(format!(
                "size {size} conflicts with resolution or aspect_ratio"
            )));
        }
        resolution = Some(raster.resolution);
        aspect_ratio = Some(raster.aspect_ratio);
    }
    // Refuse an unprovisioned raster here; the video service refuses it
    // again when it plans the target.
    let raster = rasters
        .select(resolution, aspect_ratio)
        .map_err(invalid_argument)?;

    // `seconds` is an integer in Dynamo's API; the server's duration contract
    // bounds it and derives the frame count `nvext.num_frames` must match.
    let seconds = request
        .seconds
        .map_or_else(|| DEFAULT_SECONDS.min(max_video_seconds), f64::from);
    let frames = video_frame_count(seconds, max_video_seconds).map_err(invalid_argument)?;
    let nvext = request.nvext.unwrap_or_default();
    reject_present("nvext.annotations", nvext.annotations.as_ref())?;
    reject_present("nvext.negative_prompt", nvext.negative_prompt.as_ref())?;
    reject_present("nvext.guidance_scale", nvext.guidance_scale.as_ref())?;
    reject_present("nvext.boundary_ratio", nvext.boundary_ratio.as_ref())?;
    reject_present("nvext.guidance_scale_2", nvext.guidance_scale_2.as_ref())?;
    if nvext.fps.is_some_and(|fps| fps != VIDEO_FPS as i32) {
        return Err(invalid_argument(format!("nvext.fps must be {VIDEO_FPS}")));
    }
    if nvext.num_frames.is_some_and(|value| value != frames as i32) {
        return Err(invalid_argument(format!(
            "nvext.num_frames must be {frames} for {seconds} seconds"
        )));
    }
    if nvext
        .num_inference_steps
        .is_some_and(|steps| i64::from(steps) != i64::from(denoise_steps))
    {
        return Err(invalid_argument(format!(
            "nvext.num_inference_steps must be {denoise_steps}"
        )));
    }
    // Dynamo's seed is signed and UniServe's unsigned.
    let seed = match nvext.seed {
        Some(seed) if seed < 0 => return Err(invalid_argument("nvext.seed must not be negative")),
        Some(seed) => seed as u64,
        None => 0,
    };
    Ok(PreparedRequest {
        request: VideoGenerationRequest {
            model: request.model,
            prompt: request.prompt,
            task: VideoTask::T2va,
            conditions: Vec::new(),
            target: VideoTarget {
                short_edge: raster.resolution.short_edge(),
                aspect_ratio: raster.aspect_ratio.as_str().to_owned(),
                duration_seconds: Some(seconds),
            },
            seed,
            num_inference_steps: None,
            flow_shift: None,
            audio_flow_shift: None,
            num_outputs_per_prompt: None,
            n: None,
            quality: None,
            seconds: None,
            size: None,
            width: None,
            height: None,
        },
        response_format: request.response_format.unwrap_or_default(),
    })
}

/// Checks that a deployment's video capabilities are FastH3's: `t2va` alone
/// on exactly the canvases of `rasters`, the requests `prepare_request`
/// builds.
fn check_fasth3(capabilities: &Value, rasters: &VideoRasters) -> Result<(), String> {
    let tasks = &capabilities["tasks"];
    let canvases = &capabilities["canvas"]["canvases"];
    let expected: Vec<Value> = rasters
        .rasters()
        .map(|raster| json!({"width": raster.width, "height": raster.height}))
        .collect();
    let served = canvases.as_array().is_some_and(|served| {
        served.len() == expected.len() && expected.iter().all(|canvas| served.contains(canvas))
    });
    if *tasks == json!(["t2va"]) && served {
        return Ok(());
    }
    Err(format!(
        "the Dynamo worker serves FastH3 text-to-video at {} only; this deployment serves \
         tasks {tasks} on canvases {canvases}",
        served_sizes(rasters).join(", ")
    ))
}

fn reject_present<T>(field: &str, value: Option<&T>) -> Result<(), DynamoError> {
    match value {
        Some(_) => Err(invalid_argument(format!(
            "{field} is not supported by UniServe FastH3"
        ))),
        None => Ok(()),
    }
}

/// Builds the terminal response body in the shape of Dynamo's
/// `NvVideosResponse`.
///
/// The MP4 is embedded in the response: `Url` returns it as a `data:` URL and
/// `B64Json` as bare base64. `fps` and `audio_sample_rate` report the fixed
/// FastH3 contract, not values read from the artifact.
fn video_response(
    id: &str,
    model: &str,
    response_format: ResponseFormat,
    media: &[u8],
    elapsed: Duration,
) -> Value {
    let encoded = base64::engine::general_purpose::STANDARD.encode(media);
    let (url, b64_json) = match response_format {
        ResponseFormat::Url => (Some(format!("data:video/mp4;base64,{encoded}")), None),
        ResponseFormat::B64Json => (None, Some(encoded)),
    };
    json!({
        "id": id,
        "object": "video",
        "model": model,
        "status": "completed",
        "progress": 100,
        "created": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64,
        "data": [{
            "output_format": "mp4",
            "url": url,
            "b64_json": b64_json,
            "fps": VIDEO_FPS,
            "audio_sample_rate": H3_AUDIO_SAMPLE_RATE,
        }],
        "inference_time_s": elapsed.as_secs_f64(),
    })
}

/// Reads `resolution` and `aspect_ratio` from the frontend's
/// `extra_args["media_passthrough"]`, refusing every other extra argument.
fn passthrough_raster(
    extra_args: Option<Map<String, Value>>,
) -> Result<(Option<VideoResolution>, Option<ResolutionName>), DynamoError> {
    let mut extra_args = extra_args.unwrap_or_default();
    let passthrough = match extra_args.remove("media_passthrough") {
        None => Map::new(),
        Some(Value::Object(passthrough)) => passthrough,
        Some(_) => return Err(invalid_argument("extra video fields are not supported")),
    };
    if !extra_args.is_empty()
        || passthrough
            .keys()
            .any(|key| key != "resolution" && key != "aspect_ratio")
    {
        return Err(invalid_argument("extra video fields are not supported"));
    }
    let field = |name: &str| -> Result<Option<String>, DynamoError> {
        match passthrough.get(name) {
            None => Ok(None),
            Some(Value::String(value)) => Ok(Some(value.clone())),
            Some(_) => Err(invalid_argument(format!("{name} must be a string"))),
        }
    };
    let resolution = field("resolution")?
        .map(|value| value.parse::<VideoResolution>())
        .transpose()
        .map_err(invalid_argument)?;
    let aspect_ratio = field("aspect_ratio")?
        .map(|value| value.parse::<ResolutionName>())
        .transpose()
        .map_err(|error| invalid_argument(error.to_string()))?;
    Ok((resolution, aspect_ratio))
}

/// Served rasters as `WIDTHxHEIGHT`, the default first.
fn served_sizes(rasters: &VideoRasters) -> Vec<String> {
    rasters
        .rasters()
        .map(|raster| format!("{}x{}", raster.width, raster.height))
        .collect()
}

/// Registration metadata Dynamo copies into the model's runtime config.
fn runtime_data(max_video_seconds: f64, rasters: &VideoRasters) -> HashMap<String, Value> {
    let default = rasters.select(None, None).ok();
    BTreeMap::from([
        ("backend".to_string(), json!("uniserve-inprocess")),
        ("task".to_string(), json!("t2va")),
        ("fps".to_string(), json!(VIDEO_FPS)),
        (
            "width".to_string(),
            json!(default.map(|raster| raster.width)),
        ),
        (
            "height".to_string(),
            json!(default.map(|raster| raster.height)),
        ),
        ("sizes".to_string(), json!(served_sizes(rasters))),
        ("resolutions".to_string(), json!(rasters.resolutions())),
        ("aspect_ratios".to_string(), json!(rasters.aspect_ratios())),
        ("max_video_seconds".to_string(), json!(max_video_seconds)),
    ])
    .into_iter()
    .collect()
}

/// Maps a UniServe API error to Dynamo's error type: a 4xx status becomes an
/// invalid argument and any other status an unknown backend error.
fn api_error(error: uniserve_server::openai::ApiError) -> DynamoError {
    let kind = if error.status_code().is_client_error() {
        BackendError::InvalidArgument
    } else {
        BackendError::Unknown
    };
    dynamo_error(kind, error.to_error_response().error.message)
}

fn invalid_argument(message: impl Into<String>) -> DynamoError {
    dynamo_error(BackendError::InvalidArgument, message)
}

fn engine_error(message: impl Into<String>) -> DynamoError {
    dynamo_error(BackendError::Unknown, message)
}

fn dynamo_error(kind: BackendError, message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(kind))
        .message(message)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> Value {
        json!({
            "model": "FastH3",
            "prompt": "A stream in a forest",
            "seconds": 5,
            "response_format": "b64_json",
            "nvext": {"fps": 24, "num_frames": 124, "num_inference_steps": 4, "seed": 1000}
        })
    }

    fn capabilities(tasks: Value, canvases: Value) -> Value {
        json!({"tasks": tasks, "canvas": {"canvases": canvases}})
    }

    #[test]
    fn only_fasth3_deployments_are_served() {
        let rasters = VideoRasters::default();
        let served = json!([
            {"width": 768, "height": 1344},
            {"width": 1344, "height": 768},
        ]);
        assert!(check_fasth3(&capabilities(json!(["t2va"]), served.clone()), &rasters).is_ok());
        // Another canvas set than the configured one is refused.
        let landscape = json!([{"width": 1344, "height": 768}]);
        assert!(check_fasth3(&capabilities(json!(["t2va"]), landscape), &rasters).is_err());
        // A base checkpoint serves more tasks.
        assert!(
            check_fasth3(
                &capabilities(json!(["t2va", "fl2va"]), served.clone()),
                &rasters
            )
            .is_err()
        );
        assert!(check_fasth3(&capabilities(json!(["ref2va"]), served), &rasters).is_err());
    }

    #[test]
    fn request_maps_to_uniserve_contract() {
        let prepared = match prepare_request(request(), "FastH3", 15.0, &VideoRasters::default(), 4)
        {
            Ok(prepared) => prepared,
            Err(error) => panic!("supported request was rejected: {error}"),
        };
        assert_eq!(prepared.request.task, VideoTask::T2va);
        // The deployment's default raster: 768p 16:9.
        assert_eq!(prepared.request.target.short_edge, 768);
        assert_eq!(prepared.request.target.aspect_ratio, "16:9");
        assert_eq!(prepared.request.target.duration_seconds, Some(5.0));
        assert_eq!(prepared.request.seed, 1000);
        assert!(matches!(prepared.response_format, ResponseFormat::B64Json));
    }

    #[test]
    fn size_or_passthrough_fields_select_the_target() {
        let rasters = VideoRasters::new(
            &[VideoResolution::P768, VideoResolution::P480],
            &[ResolutionName::Landscape16x9, ResolutionName::Landscape21x9],
        )
        .unwrap_or_else(|error| panic!("trained rasters were rejected: {error}"));
        let prepare = |request: Value| prepare_request(request, "FastH3", 15.0, &rasters, 4);
        for (size, short_edge, aspect_ratio) in
            [("1344x768", 768, "16:9"), ("992x416", 480, "21:9")]
        {
            let by_size = match prepare(json!({"model": "FastH3", "prompt": "x", "size": size})) {
                Ok(prepared) => prepared.request,
                Err(error) => panic!("size {size} was rejected: {error}"),
            };
            assert_eq!(by_size.target.short_edge, short_edge);
            assert_eq!(by_size.target.aspect_ratio, aspect_ratio);
            let resolution = format!("{short_edge}p");
            let passthrough = json!({"resolution": resolution, "aspect_ratio": aspect_ratio});
            let by_name = match prepare(json!({
                "model": "FastH3",
                "prompt": "x",
                "extra_args": {"media_passthrough": passthrough.clone()},
            })) {
                Ok(prepared) => prepared.request,
                Err(error) => panic!("{passthrough} was rejected: {error}"),
            };
            assert_eq!(by_name, by_size);
            assert!(
                prepare(json!({
                    "model": "FastH3",
                    "prompt": "x",
                    "size": size,
                    "extra_args": {"media_passthrough": passthrough},
                }))
                .is_ok()
            );
        }
        for request in [
            json!({"model": "FastH3", "prompt": "x", "size": "768x1344"}),
            json!({"model": "FastH3", "prompt": "x", "size": "1344x768",
                "extra_args": {"media_passthrough": {"resolution": "480p"}}}),
            json!({"model": "FastH3", "prompt": "x",
                "extra_args": {"media_passthrough": {"aspect_ratio": "1:1"}}}),
            json!({"model": "FastH3", "prompt": "x",
                "extra_args": {"media_passthrough": {"resolution": 480}}}),
        ] {
            assert!(prepare(request).is_err());
        }
        let runtime = runtime_data(15.0, &rasters);
        assert_eq!(runtime["width"], 1344);
        assert_eq!(runtime["height"], 768);
        assert_eq!(
            runtime["sizes"],
            json!(["1344x768", "1536x672", "832x480", "992x416"])
        );
    }

    /// An omitted duration is 5 seconds, or the deployment's maximum when
    /// that is shorter.
    #[test]
    fn an_omitted_duration_defaults_to_five_seconds() {
        for (max_video_seconds, expected) in [(15.0, 5.0), (4.5, 4.5)] {
            let request = json!({"model": "FastH3", "prompt": "A stream in a forest"});
            let prepared = match prepare_request(
                request,
                "FastH3",
                max_video_seconds,
                &VideoRasters::default(),
                8,
            ) {
                Ok(prepared) => prepared,
                Err(error) => panic!("max {max_video_seconds}s refused the default: {error}"),
            };
            assert_eq!(prepared.request.target.duration_seconds, Some(expected));
        }
    }

    // Each request carries one field or value FastH3 does not support. With
    // the default 5 seconds, `num_frames` must be 124, so 120 is refused.
    #[test]
    fn unsupported_control_is_rejected() {
        for request in [
            json!({"model":"FastH3", "prompt":"x", "input_reference":"image"}),
            json!({"model":"FastH3", "prompt":"x", "stream":true}),
            json!({"model":"FastH3", "prompt":"x", "size":"832x480"}),
            json!({"model":"FastH3", "prompt":"x", "nvext":{"seed":-1}}),
            json!({"model":"FastH3", "prompt":"x", "nvext":{"num_frames":120}}),
            json!({"model":"FastH3", "prompt":"x", "seconds":3}),
            json!({"model":"FastH3", "prompt":"x", "seconds":16}),
            json!({"model":"FastH3", "prompt":"x", "extra_args":{"media_passthrough":{"foo":1}}}),
        ] {
            assert!(prepare_request(request, "FastH3", 15.0, &VideoRasters::default(), 8).is_err());
        }
    }

    /// Explicit steps confirm the loaded schedule; they never change it.
    #[test]
    fn inference_steps_must_match_the_checkpoint() {
        for checkpoint_steps in [4, 8] {
            for requested_steps in [4, 8, 0, -1] {
                let mut value = request();
                value["nvext"]["num_inference_steps"] = json!(requested_steps);
                assert_eq!(
                    prepare_request(
                        value,
                        "FastH3",
                        15.0,
                        &VideoRasters::default(),
                        checkpoint_steps
                    )
                    .is_ok(),
                    i64::from(requested_steps) == i64::from(checkpoint_steps),
                );
            }
        }
    }

    #[test]
    fn response_formats_embed_the_mp4() {
        let response = video_response(
            "id",
            "FastH3",
            ResponseFormat::B64Json,
            b"mp4",
            Duration::from_millis(1_500),
        );
        assert_eq!(response["data"][0]["b64_json"], "bXA0");
        assert!(response["data"][0]["url"].is_null());
        assert_eq!(response["inference_time_s"], 1.5);

        let response = video_response("id", "FastH3", ResponseFormat::Url, b"mp4", Duration::ZERO);
        assert_eq!(response["data"][0]["url"], "data:video/mp4;base64,bXA0");
    }
}
