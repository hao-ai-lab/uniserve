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
//! `--video-resolutions` and `--video-aspect-ratios` prepare, as the
//! deployment declares them, and refuses at startup a deployment that serves
//! any other task, such as a MiniMax-H3 base checkpoint: `prepare_request`
//! maps the Dynamo request onto a `t2va` request of UniServe's video API and
//! refuses any control FastH3 does not implement rather than ignoring it.
//! The request then enters
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
    serving::{FinishStatus, MAX_VIDEO_SECONDS, RequestOutput, ServeRequestId, video_frame_count},
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
    /// `configs/minimax_h3/fasth3-ulysses4.json`.
    #[arg(long, value_name = "FILE", value_parser = read_workers)]
    workers: Box<[WorkerConfig]>,

    /// This process's host identity. The engine owns exactly the placed ranks
    /// whose node names it.
    #[arg(long, default_value = "localhost")]
    host_identity: String,

    // These options mirror the `uniserve serve` flags of the same names and
    // feed the engine configuration `build_state` assembles. The default
    // prompt length is the one `uniserve serve` takes for a MiniMax-H3
    // pipeline; the text encoder itself accepts more.
    #[arg(long, default_value_t = 16_384, value_parser = clap::value_parser!(u32).range(1..))]
    max_model_len: u32,

    #[arg(long, default_value_t = MAX_VIDEO_SECONDS)]
    max_video_seconds: f64,

    #[arg(long, default_value = "768p")]
    video_resolutions: String,

    #[arg(long, default_value = "16:9,9:16")]
    video_aspect_ratios: String,

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
    /// The videos the started deployment generates, set once by `start` with
    /// `state`.
    outputs: OnceCell<Outputs>,
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
    /// non-finite or non-positive video length, a zero request limit) return
    /// an invalid-argument error. No rank starts until `RawEngine::start`,
    /// which refuses a video raster selection the model does not generate.
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
        if args.max_running_requests == 0 {
            return Err(invalid_argument("max-running-requests must be positive"));
        }

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
                outputs: OnceCell::new(),
                state: OnceCell::new(),
                cancel: CancellationToken::new(),
            },
            config,
        ))
    }

    /// Starts UniServe's engine and worker ranks for the configured FastH3
    /// checkpoint and returns the serving state with the videos it generates.
    ///
    /// Fails with an invalid argument when the checkpoint's pipeline index
    /// cannot be resolved or does not name MiniMax H3, when the configuration
    /// does not validate, or when the started model reports no video
    /// capabilities or serves a task other than `t2va`. It fails with an
    /// engine error when UniServe cannot start, including a raster selection
    /// its workers refuse. A refusal after startup shuts the engine down
    /// before returning.
    async fn build_state(&self) -> Result<(Arc<AppState>, Outputs), DynamoError> {
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
            video_resolutions: Some(self.args.video_resolutions.clone()),
            video_aspect_ratios: Some(self.args.video_aspect_ratios.clone()),
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
                max_num_waiting: None,
                long_prefill_threshold: DEFAULT_LONG_PREFILL_THRESHOLD,
                prefix_cache: true,
                scheduler_policy: SchedulingPolicy::Fcfs,
                max_model_len: Some(self.args.max_model_len),
                max_video_seconds: self.args.max_video_seconds,
                // The Dynamo request form carries no conditions, so the
                // worker provisions no condition rows.
                max_condition_rows: Some(0),
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
        let outputs = match check_fasth3(&capabilities) {
            Ok(outputs) => outputs,
            Err(message) => {
                let _ = state.engine().shutdown().await;
                return Err(invalid_argument(message));
            }
        };
        Ok((state, outputs))
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
        let (state, outputs) = self.build_state().await?;
        let runtime_data = runtime_data(self.args.max_video_seconds, &outputs);
        self.state
            .set(state)
            .map_err(|_| engine_error("UniServe engine already started"))?;
        self.outputs
            .set(outputs)
            .map_err(|_| engine_error("UniServe engine already started"))?;
        Ok(EngineConfig {
            model: self.args.served_model_name.clone(),
            served_model_name: Some(self.args.served_model_name.clone()),
            runtime_data,
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
        let outputs = self
            .outputs
            .get()
            .ok_or_else(|| engine_error("generate called before start"))?;
        let prepared = prepare_request(
            request,
            &self.args.served_model_name,
            self.args.max_video_seconds,
            outputs,
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
        let frame_rate = outputs.frame_rate;

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
                frame_rate,
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
/// fields FastH3 can honor: `seed` at any non-negative value, and `fps` and
/// `num_frames` only at the values FastH3 runs with. The checkpoint fixes the
/// denoising schedule, so `num_inference_steps` is refused like the other
/// controls FastH3 does not take.
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
/// Every refusal is an invalid-argument error.
fn prepare_request(
    value: Value,
    served_model_name: &str,
    max_video_seconds: f64,
    outputs: &Outputs,
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
    let (mut short_edge, mut aspect_ratio) = passthrough_raster(request.extra_args)?;
    if let Some(size) = request.size.as_deref() {
        let raster = size
            .split_once('x')
            .and_then(|(width, height)| Some((width.parse().ok()?, height.parse().ok()?)))
            .and_then(|(width, height)| outputs.find_size(width, height))
            .ok_or_else(|| {
                invalid_argument(format!(
                    "size must be one of {}",
                    outputs.sizes().join(", ")
                ))
            })?;
        if short_edge.is_some_and(|value| value != raster.short_edge)
            || aspect_ratio
                .as_deref()
                .is_some_and(|value| value != raster.aspect_ratio)
        {
            return Err(invalid_argument(format!(
                "size {size} conflicts with resolution or aspect_ratio"
            )));
        }
        short_edge = Some(raster.short_edge);
        aspect_ratio = Some(raster.aspect_ratio.clone());
    }
    let raster = outputs
        .select(short_edge, aspect_ratio.as_deref())
        .map_err(invalid_argument)?;

    // `seconds` is an integer in Dynamo's API; the server's duration contract
    // bounds it and derives the frame count `nvext.num_frames` must match.
    let seconds = request
        .seconds
        .map_or_else(|| DEFAULT_SECONDS.min(max_video_seconds), f64::from);
    let frames = video_frame_count(
        seconds,
        max_video_seconds,
        outputs.frame_rate,
        &outputs.frames,
    )
    .map_err(invalid_argument)?;
    let nvext = request.nvext.unwrap_or_default();
    reject_present("nvext.annotations", nvext.annotations.as_ref())?;
    reject_present("nvext.negative_prompt", nvext.negative_prompt.as_ref())?;
    reject_present(
        "nvext.num_inference_steps",
        nvext.num_inference_steps.as_ref(),
    )?;
    reject_present("nvext.guidance_scale", nvext.guidance_scale.as_ref())?;
    reject_present("nvext.boundary_ratio", nvext.boundary_ratio.as_ref())?;
    reject_present("nvext.guidance_scale_2", nvext.guidance_scale_2.as_ref())?;
    if nvext
        .fps
        .is_some_and(|fps| fps != outputs.frame_rate as i32)
    {
        return Err(invalid_argument(format!(
            "nvext.fps must be {}",
            outputs.frame_rate
        )));
    }
    if nvext.num_frames.is_some_and(|value| value != frames as i32) {
        return Err(invalid_argument(format!(
            "nvext.num_frames must be {frames} for {seconds} seconds"
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
                short_edge: raster.short_edge,
                aspect_ratio: raster.aspect_ratio.clone(),
                duration_seconds: Some(seconds),
            },
            seed,
        },
        response_format: request.response_format.unwrap_or_default(),
    })
}

/// Checks that a deployment's video capabilities are FastH3's, `t2va` alone,
/// the requests `prepare_request` builds, and returns the rasters it serves.
fn check_fasth3(capabilities: &Value) -> Result<Outputs, String> {
    let tasks = &capabilities["tasks"];
    if *tasks != json!(["t2va"]) {
        return Err(format!(
            "the Dynamo worker serves FastH3 text-to-video only; this deployment serves \
             tasks {tasks}"
        ));
    }
    Outputs::from_capabilities(capabilities)
}

/// A raster the deployment serves and the request fields that select it.
#[derive(Debug, Clone, PartialEq)]
struct Raster {
    /// The resolution class's short edge in pixels, `target.short_edge`.
    short_edge: u32,
    /// The display aspect ratio as `W:H`, `target.aspect_ratio`.
    aspect_ratio: String,
    width: u32,
    height: u32,
}

/// The videos a deployment generates, as its capabilities report them: the
/// rasters, every prepared resolution crossed with every prepared aspect
/// ratio in the order its operator selected them, and their frame rate. The
/// first raster is the default of a request that names neither a resolution
/// nor an aspect ratio.
#[derive(Debug, Clone, PartialEq)]
struct Outputs {
    rasters: Vec<Raster>,
    /// Frames per second of every generated video.
    frame_rate: u32,
    /// How the deployment's VAE chunks a video's frames.
    frames: uniserve_engine::FrameChunks,
}

impl Outputs {
    /// Reads the served rasters from `canvas.sizes` of the capabilities:
    /// short edge, then aspect ratio, to the raster; and the frame rate.
    fn from_capabilities(capabilities: &Value) -> Result<Self, String> {
        let sizes = capabilities["canvas"]["sizes"].as_object();
        let mut rasters = Vec::new();
        for (short_edge, ratios) in sizes.into_iter().flatten() {
            for (aspect_ratio, size) in ratios.as_object().into_iter().flatten() {
                let raster = (|| {
                    Some(Raster {
                        short_edge: short_edge.parse().ok()?,
                        aspect_ratio: aspect_ratio.clone(),
                        width: u32::try_from(size["width"].as_u64()?).ok()?,
                        height: u32::try_from(size["height"].as_u64()?).ok()?,
                    })
                })();
                rasters.push(raster.ok_or_else(|| {
                    format!(
                        "the deployment reports an unreadable canvas {short_edge} {aspect_ratio}"
                    )
                })?);
            }
        }
        if rasters.is_empty() {
            return Err("the deployment serves no canvas".to_owned());
        }
        let frame_rate = capabilities["fps"]
            .as_u64()
            .and_then(|fps| u32::try_from(fps).ok())
            .ok_or("the deployment reports no frame rate")?;
        let frames = serde_json::from_value(capabilities["frames"].clone())
            .map_err(|error| format!("the deployment reports no frame rule: {error}"))?;
        Ok(Self {
            rasters,
            frame_rate,
            frames,
        })
    }

    /// Selects a served raster; an omitted field takes the default's.
    ///
    /// # Errors
    ///
    /// Returns a message naming the served values when the deployment does
    /// not serve the requested resolution, aspect ratio or their pair.
    fn select(
        &self,
        short_edge: Option<u32>,
        aspect_ratio: Option<&str>,
    ) -> Result<&Raster, String> {
        let default = &self.rasters[0];
        let short_edge = short_edge.unwrap_or(default.short_edge);
        let aspect_ratio = aspect_ratio.unwrap_or(&default.aspect_ratio);
        if !self
            .rasters
            .iter()
            .any(|raster| raster.short_edge == short_edge)
        {
            return Err(format!(
                "resolution must be one of {}",
                self.resolutions().join(", ")
            ));
        }
        if !self
            .rasters
            .iter()
            .any(|raster| raster.aspect_ratio == aspect_ratio)
        {
            return Err(format!(
                "aspect_ratio must be one of {}",
                self.aspect_ratios().join(", ")
            ));
        }
        self.rasters
            .iter()
            .find(|raster| raster.short_edge == short_edge && raster.aspect_ratio == aspect_ratio)
            .ok_or_else(|| format!("no canvas is served for {short_edge}p {aspect_ratio}"))
    }

    /// Returns the served raster with exactly these pixel dimensions.
    fn find_size(&self, width: u32, height: u32) -> Option<&Raster> {
        self.rasters
            .iter()
            .find(|raster| raster.width == width && raster.height == height)
    }

    /// Served rasters as `WIDTHxHEIGHT`, the default first.
    fn sizes(&self) -> Vec<String> {
        self.rasters
            .iter()
            .map(|raster| format!("{}x{}", raster.width, raster.height))
            .collect()
    }

    /// Served resolution classes, as `768p`, the default first.
    fn resolutions(&self) -> Vec<String> {
        let mut classes: Vec<String> = Vec::new();
        for raster in &self.rasters {
            let class = format!("{}p", raster.short_edge);
            if !classes.contains(&class) {
                classes.push(class);
            }
        }
        classes
    }

    /// Served aspect ratios, the default first.
    fn aspect_ratios(&self) -> Vec<String> {
        let mut ratios: Vec<String> = Vec::new();
        for raster in &self.rasters {
            if !ratios.contains(&raster.aspect_ratio) {
                ratios.push(raster.aspect_ratio.clone());
            }
        }
        ratios
    }
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
/// `B64Json` as bare base64. `fps` is the deployment's frame rate and
/// `audio_sample_rate` the fixed FastH3 contract, not values read from the
/// artifact.
fn video_response(
    id: &str,
    model: &str,
    response_format: ResponseFormat,
    media: &[u8],
    frame_rate: u32,
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
            "fps": frame_rate,
            "audio_sample_rate": H3_AUDIO_SAMPLE_RATE,
        }],
        "inference_time_s": elapsed.as_secs_f64(),
    })
}

/// Reads `resolution` (a class such as `768p`, returned as its short edge) and
/// `aspect_ratio` from the frontend's `extra_args["media_passthrough"]`,
/// refusing every other extra argument.
fn passthrough_raster(
    extra_args: Option<Map<String, Value>>,
) -> Result<(Option<u32>, Option<String>), DynamoError> {
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
    let short_edge = field("resolution")?
        .map(|value| {
            value
                .strip_suffix('p')
                .and_then(|edge| edge.parse::<u32>().ok())
                .ok_or_else(|| {
                    invalid_argument(format!(
                        "resolution must be a class such as 768p, got {value:?}"
                    ))
                })
        })
        .transpose()?;
    Ok((short_edge, field("aspect_ratio")?))
}

/// Registration metadata Dynamo copies into the model's runtime config.
fn runtime_data(max_video_seconds: f64, outputs: &Outputs) -> HashMap<String, Value> {
    let default = &outputs.rasters[0];
    BTreeMap::from([
        ("backend".to_string(), json!("uniserve-inprocess")),
        ("task".to_string(), json!("t2va")),
        ("fps".to_string(), json!(outputs.frame_rate)),
        ("width".to_string(), json!(default.width)),
        ("height".to_string(), json!(default.height)),
        ("sizes".to_string(), json!(outputs.sizes())),
        ("resolutions".to_string(), json!(outputs.resolutions())),
        ("aspect_ratios".to_string(), json!(outputs.aspect_ratios())),
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
            "nvext": {"fps": 24, "num_frames": 124, "seed": 1000}
        })
    }

    fn capabilities(tasks: Value, sizes: Value) -> Value {
        json!({
            "tasks": tasks,
            "fps": 24,
            "frames": {
                "min_frames": 22,
                "chunk_frames": 17,
                "chunk_latent_frames": 5,
                "tail_latent_frames": 2,
            },
            "canvas": {"sizes": sizes},
        })
    }

    /// The videos of a deployment preparing 768p 16:9 and 9:16 at 24 fps.
    fn default_outputs() -> Outputs {
        let sizes = json!({"768": {
            "16:9": {"width": 1344, "height": 768},
            "9:16": {"width": 768, "height": 1344},
        }});
        check_fasth3(&capabilities(json!(["t2va"]), sizes)).unwrap()
    }

    #[test]
    fn only_fasth3_deployments_are_served() {
        let sizes = json!({"768": {"16:9": {"width": 1344, "height": 768}}});
        let outputs = check_fasth3(&capabilities(json!(["t2va"]), sizes.clone())).unwrap();
        assert_eq!(outputs.sizes(), ["1344x768"]);
        // A base checkpoint serves more tasks.
        assert!(check_fasth3(&capabilities(json!(["t2va", "fl2va"]), sizes.clone())).is_err());
        assert!(check_fasth3(&capabilities(json!(["ref2va"]), sizes)).is_err());
        // A deployment that serves no canvas has nothing to generate.
        assert!(check_fasth3(&capabilities(json!(["t2va"]), json!({}))).is_err());
    }

    #[test]
    fn request_maps_to_uniserve_contract() {
        let prepared = match prepare_request(request(), "FastH3", 15.0, &default_outputs()) {
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
        let sizes = json!({
            "768": {
                "16:9": {"width": 1344, "height": 768},
                "21:9": {"width": 1536, "height": 672},
            },
            "480": {
                "16:9": {"width": 832, "height": 480},
                "21:9": {"width": 992, "height": 416},
            },
        });
        let outputs = check_fasth3(&capabilities(json!(["t2va"]), sizes)).unwrap();
        let prepare = |request: Value| prepare_request(request, "FastH3", 15.0, &outputs);
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
        let runtime = runtime_data(15.0, &outputs);
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
            let prepared =
                match prepare_request(request, "FastH3", max_video_seconds, &default_outputs()) {
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
            json!({"model":"FastH3", "prompt":"x", "nvext":{"num_inference_steps":4}}),
            json!({"model":"FastH3", "prompt":"x", "seconds":3}),
            json!({"model":"FastH3", "prompt":"x", "seconds":16}),
            json!({"model":"FastH3", "prompt":"x", "extra_args":{"media_passthrough":{"foo":1}}}),
        ] {
            assert!(prepare_request(request, "FastH3", 15.0, &default_outputs()).is_err());
        }
    }

    #[test]
    fn response_formats_embed_the_mp4() {
        let response = video_response(
            "id",
            "FastH3",
            ResponseFormat::B64Json,
            b"mp4",
            24,
            Duration::from_millis(1_500),
        );
        assert_eq!(response["data"][0]["b64_json"], "bXA0");
        assert!(response["data"][0]["url"].is_null());
        assert_eq!(response["inference_time_s"], 1.5);

        let response = video_response(
            "id",
            "FastH3",
            ResponseFormat::Url,
            b"mp4",
            24,
            Duration::ZERO,
        );
        assert_eq!(response["data"][0]["url"], "data:video/mp4;base64,bXA0");
    }
}
