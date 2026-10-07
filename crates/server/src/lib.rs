//! UniServe HTTP frontend, model profiles, request lowering, and output assembly.
//!
//! The crate resolves one configuration into shared application
//! state and exposes OpenAI-compatible routes backed by the in-process engine.
//!
//! [`build_state`] is the startup path: it resolves the model's assets into a
//! `profile::ModelConfig`, a tokenizer, and a chat renderer, derives the
//! engine and worker-process settings from the model and the configuration,
//! starts the engine (`engine_client`), and binds the model's input
//! processing to the capabilities the engine reports (`serving`). The HTTP
//! layer (`http`) serves requests from the resulting [`AppState`].

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod config;
/// Server-facing engine submission and health client.
pub mod engine_client;
/// HTTP listener, middleware, and route construction.
pub mod http;
/// OpenAI-compatible schemas and conversion helpers.
pub mod openai;
/// Model assets and behavior profiles.
pub mod profile;
/// Model-aware request admission and output streaming.
pub mod serving;
mod state;
mod video_jobs;

use std::sync::Arc;

use crate::engine_client::EngineClient;
use crate::profile::ModelConfig;
pub use crate::profile::ModelDescription;
use crate::serving::LoadedModel;
pub use crate::serving::chat::ChatTemplateContentFormatOption;
use crate::serving::{InputProcessor, ServingRuntime};
use anyhow::{Context as _, Result};
pub use config::{Config, EngineSettings, HttpListenerMode, VideoMediaSettings};
use tracing::info;
pub use uniserve_engine::SchedulingPolicy;
use uniserve_engine::{EngineConfig, SpecialTokenIds, WorkerProcessArgs};

pub use crate::http::{ApiError, build_router, serve};
pub use crate::state::AppState;

/// Resolves the control tokens the engine uses for the loaded model.
///
/// `bos` and `end_of_image` come from the profile's `GenerationControls` and
/// are `0` for profiles without them (`Qwen3` and `MiniMaxH3`). The EOS list
/// is the model's complete EOS set with the primary EOS placed first (and
/// added when the set lacks it). The order matters to the engine scheduler,
/// which substitutes `eos[0]` for the token of a completion that commits
/// none. The primary EOS is the nonzero `GenerationControls::eos` when
/// present, otherwise `ModelConfig::primary_eos_token_id`; when neither
/// exists the list keeps the set's ascending order.
fn special_token_ids(model: &ModelConfig) -> SpecialTokenIds {
    let controls = model.generation_controls();
    let bos = controls.map_or(0, |value| value.bos);
    let end_of_image = controls.map_or(0, |value| value.end_of_image);
    let primary_eos = controls
        .map(|value| value.eos)
        .filter(|value| *value != 0)
        .or(model.primary_eos_token_id);
    let mut eos = model.eos_token_ids.iter().copied().collect::<Vec<_>>();
    if let Some(primary_eos) = primary_eos {
        eos.retain(|value| *value != primary_eos);
        eos.insert(0, primary_eos);
    }
    SpecialTokenIds {
        bos,
        eos,
        end_of_image,
    }
}

/// Builds the shared application state for one resolved model and one engine client.
///
/// Starts the engine and its worker processes as a side effect, so the
/// returned state owns a running engine.
///
/// # Errors
///
/// Fails when the model assets cannot be resolved, when the per-run call
/// bound (`max_batch`) or `max_num_batched_tokens`
/// does not fit the worker's `u32` fields, when the engine fails to start, or
/// when the model description cannot be bound to the capabilities the engine
/// reports.
pub async fn build_state(config: &Config) -> Result<Arc<AppState>> {
    let LoadedModel {
        config: model_config,
        tokenizer,
        renderer,
        vision,
    } = ModelConfig::load(config)
        .await
        .with_context(|| format!("failed to resolve model assets for `{}`", config.model))?;
    let effective_max_model_len = model_config.max_model_tokens();
    let channel_payload_capacity = model_config.channel_payload_capacity();
    let control_tokens = special_token_ids(&model_config);
    let generation_limits =
        model_config.generation_limits(config.engine.worker_process.model_dtype);

    info!(
        workers = ?config.engine.workers,
        block_size = ?config.engine.worker_process.block_size,
        queue_depth = config.engine.worker_process.queue_depth,
        "starting UniServe Rust engine"
    );

    // Admission limits running requests, not the worker's execution capacity.
    // Media workers retain two internal state slots even when admission permits
    // only one request. Keep their configured batch capacity independent.
    let max_batch_calls = u32::try_from(config.engine.max_batch.max(1))
        .context("max_batch exceeds the worker field width")?;
    let max_batch_tokens = u32::try_from(config.engine.max_num_batched_tokens)
        .context("max_num_batched_tokens exceeds the worker field width")?;

    // The model profile states the IPC payload its products need, and both
    // directions of the worker channel must carry it: the request slot uses
    // exactly that capacity, and the response slot uses the configured
    // `resp_slot_cap` only when it is larger.
    let worker_process = WorkerProcessArgs {
        model: config.model.clone(),
        base_model: config.base_model.clone(),
        req_slot_cap: channel_payload_capacity,
        resp_slot_cap: config
            .engine
            .worker_process
            .resp_slot_cap
            .max(channel_payload_capacity),
        max_batch_calls,
        max_batch_tokens,
        max_model_len: effective_max_model_len,
        max_video_seconds: config.engine.max_video_seconds,
        max_condition_rows: config.engine.max_condition_rows,
        // Video workers provision only the frame counts the API admits.
        min_video_seconds: Some(crate::serving::MIN_VIDEO_SECONDS),
        // Video workers prepare exactly the canvases of the configured
        // resolutions and aspect ratios, and report them back as the only
        // canvases requests may target.
        video_frame_sizes: match &model_config.parameters {
            crate::profile::ModelParameters::MiniMaxH3 { .. } => Some(
                crate::profile::video::VideoRasters::new(
                    &config.engine.video_resolutions,
                    &config.engine.video_aspect_ratios,
                )
                .map_err(|message| anyhow::anyhow!("invalid video deployment: {message}"))?
                .worker_frame_sizes(),
            ),
            _ => None,
        },
        ..config.engine.worker_process.clone()
    };
    let engine_config = EngineConfig {
        runtime_family: model_config.runtime_family(),
        generation_limits,
        max_batch: config.engine.max_batch,
        max_num_batched_tokens: config.engine.max_num_batched_tokens,
        max_num_seqs: config.engine.max_num_seqs,
        long_prefill_threshold: config.engine.long_prefill_threshold,
        mixed_prefill_tokens: config.engine.mixed_prefill_tokens,
        prefix_cache: config.engine.prefix_cache,
        scheduler_policy: config.engine.scheduler_policy,
        max_model_len: effective_max_model_len,
        workers: config.engine.workers.clone(),
        transfer: config.engine.transfer.clone(),
        data_parallel_size: config.engine.data_parallel_size,
        expert_parallel: config.engine.expert_parallel,
        worker_process,
        bos: control_tokens.bos,
        eos: control_tokens.eos,
        end_of_image: control_tokens.end_of_image,
    };
    let client =
        EngineClient::connect(engine_config).context("failed to start the UniServe engine")?;

    let engine = Arc::new(client);
    // The served context limit never exceeds the length the engine reports,
    // which is the `max_model_len` passed in `engine_config`.
    let route_max_model_len = effective_max_model_len.min(engine.max_model_len());
    // A video checkpoint serves what its placed denoiser declares in the
    // worker handshake, planned against the conditioner's vision processor.
    let video = match &model_config.parameters {
        crate::profile::ModelParameters::MiniMaxH3 { max_video_seconds } => {
            let denoiser = engine
                .video_denoiser()
                .context("the video worker reported no video denoiser")?;
            let vision = vision.context("the video checkpoint has no vision processor")?;
            Some(
                crate::serving::video::VideoService::new(
                    denoiser,
                    vision,
                    *max_video_seconds,
                    config.engine.max_condition_rows,
                    &config.video_media,
                    Arc::clone(&tokenizer),
                )
                .context("failed to bind the video denoiser")?,
            )
        }
        _ => None,
    };
    let model = InputProcessor::new(
        model_config,
        tokenizer,
        renderer,
        crate::serving::WorkerCapabilities {
            limits: engine.generation_limits(),
            sampling_controls: engine.served_sampling_controls(),
            max_model_tokens: route_max_model_len,
        },
        video,
        config.reasoning_parsing,
    )
    .context("failed to bind the configured model description")?;
    let runtime = ServingRuntime::new(model, Arc::clone(&engine), config.log_stats);

    Ok(Arc::new(
        AppState::new(runtime)
            .with_log_requests(config.enable_log_requests)
            .with_request_id_headers(config.enable_request_id_headers)
            .with_api_key(config.api_key.clone())
            .with_request_timeout(config.request_timeout)
            .with_max_concurrent_requests(config.max_concurrent_requests)
            .with_video_body_limit(config.video_media.body_limit()),
    ))
}
