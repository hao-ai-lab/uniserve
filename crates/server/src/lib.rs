//! Deployment state composition for the configured UniServe serving surface.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod config;
pub mod engine_client;
pub mod http;
pub mod openai;
pub mod profile;
mod scheduler_stats;
pub mod serving;
mod state;

use std::sync::Arc;

use crate::engine_client::EngineClient;
pub use crate::profile::ModelDescription;
pub use crate::serving::chat::ChatTemplateContentFormatOption;
use crate::serving::{ResolvedAssets, ResolvedModel, ServingRuntime};
use anyhow::{Context as _, Result};
pub use config::{Config, EngineBackendKind, EngineSettings, HttpListenerMode};
use tracing::info;
pub use uniserve_engine::SchedulingPolicy;
use uniserve_engine::{EngineCoreConfig, SimEngine, SimExecutor, WorkerProcessArgs};

pub use crate::http::{ApiError, build_router, serve};
pub use crate::state::AppState;

#[derive(Debug, Clone, PartialEq, Eq)]
struct RuntimeControlTokens {
    bos: u32,
    eos: Vec<u32>,
    start_of_image: u32,
    end_of_image: u32,
}

fn runtime_control_tokens(
    assets: &ResolvedAssets,
    backend: EngineBackendKind,
) -> RuntimeControlTokens {
    let controls = assets.generation_controls();
    let bos = controls.map_or(0, |value| value.bos);
    let start_of_image = controls.map_or(0, |value| value.start_of_image);
    let end_of_image = controls.map_or(0, |value| value.end_of_image);
    let primary_eos = controls
        .map(|value| value.eos)
        .filter(|value| *value != 0)
        .or(assets.profile().stop_tokens.primary_eos_token_id);
    let mut eos = assets
        .profile()
        .stop_tokens
        .eos_token_ids
        .iter()
        .copied()
        .collect::<Vec<_>>();
    if let Some(primary_eos) = primary_eos {
        eos.retain(|value| *value != primary_eos);
        eos.insert(0, primary_eos);
    }
    if backend == EngineBackendKind::Sim && eos.is_empty() {
        eos.push(151645);
    }
    RuntimeControlTokens {
        bos,
        eos,
        start_of_image,
        end_of_image,
    }
}

/// Build the shared application state for one resolved model and one engine client.
pub async fn build_state(config: &Config) -> Result<Arc<AppState>> {
    let assets = ResolvedAssets::load(config)
        .await
        .with_context(|| format!("failed to resolve model assets for `{}`", config.model))?;
    let effective_max_model_len = assets.max_model_tokens();
    let request_slot_capacity = assets.request_slot_capacity();
    let control_tokens = runtime_control_tokens(&assets, config.engine.backend);
    let runtime_profile = assets.runtime_profile(config.engine.worker_process.model_dtype.clone());

    let eos = control_tokens.eos.clone();
    info!(
        backend = ?config.engine.backend,
        device = %config.engine.worker_process.device,
        block_size = config.engine.worker_process.block_size,
        pipeline_depth = config.engine.worker_process.pipeline_depth,
        "starting UniServe Rust engine"
    );
    let max_batch_operations = u32::try_from(
        config
            .engine
            .max_batch
            .max(1)
            .min(config.engine.max_num_seqs.max(1)),
    )
    .context("max_batch exceeds the worker field width")?;
    let max_batch_tokens = u32::try_from(config.engine.max_num_batched_tokens)
        .context("max_num_batched_tokens exceeds the worker field width")?;
    let worker_process = WorkerProcessArgs {
        model: config.model.clone(),
        req_slot_cap: request_slot_capacity,
        max_batch_operations,
        max_batch_tokens,
        max_model_len: effective_max_model_len,
        max_video_seconds: config.engine.max_video_seconds,
        ..config.engine.worker_process.clone()
    };
    let engine_config = EngineCoreConfig {
        runtime_family: match &assets {
            ResolvedAssets::Text { .. } => uniserve_core::RuntimeFamily::Ar,
            ResolvedAssets::Omni { .. } => uniserve_core::RuntimeFamily::Umm,
            ResolvedAssets::Media { .. } => uniserve_core::RuntimeFamily::Diffusion,
        },
        runtime_profile,
        max_batch: config.engine.max_batch,
        max_num_batched_tokens: config.engine.max_num_batched_tokens,
        max_num_seqs: config.engine.max_num_seqs,
        long_prefill_threshold: config.engine.long_prefill_threshold,
        mixed_prefill_tokens: config.engine.mixed_prefill_tokens,
        scheduler_policy: config.engine.scheduler_policy,
        max_model_len: effective_max_model_len,
        workers: config.engine.workers.clone(),
        transfer: config.engine.transfer.clone(),
        worker_process,
        bos: control_tokens.bos,
        eos,
        end_of_image: control_tokens.end_of_image,
    };
    let client = if config.engine.backend == EngineBackendKind::Sim {
        let mut sim = SimEngine::new();
        let special_tokens = [
            control_tokens.bos,
            control_tokens.start_of_image,
            control_tokens.end_of_image,
        ]
        .into_iter()
        .filter(|token| *token != 0)
        .collect::<Vec<_>>();
        sim.configure_control_tokens(
            engine_config.eos.first().copied().unwrap_or(151645),
            &special_tokens,
        );
        let executor = SimExecutor::new(sim);
        let command_waker = executor.command_waker();
        EngineClient::connect_with_executor_and_waker(
            engine_config,
            Box::new(executor),
            command_waker,
        )
    } else {
        EngineClient::connect(engine_config)
    }
    .context("failed to start the UniServe engine")?;

    let engine = Arc::new(client);
    let snapshot = engine.snapshot();
    let route_max_model_len = effective_max_model_len.min(snapshot.max_model_len);
    let model = ResolvedModel::resolve(
        assets,
        snapshot.generation_limits,
        snapshot.sampling_controls,
        route_max_model_len,
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
            .with_max_concurrent_requests(config.max_concurrent_requests),
    ))
}
