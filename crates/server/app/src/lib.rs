//! Deployment state composition for the configured UniServe serving surface.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod config;
pub mod http;
mod runtime_client;
mod scheduler_stats;
mod state;

use std::sync::Arc;

use anyhow::{Context as _, Result};
pub use config::{Config, EngineBackendKind, EngineConnection, EngineSettings, HttpListenerMode};
use tracing::info;
use uniserve_engine_gateway::EngineGateway;
use uniserve_engine_gateway::transport::protocol::generation::GenerationControlTokens;
use uniserve_engine_gateway::transport::{EngineCoreClient, TransportMode, ZmqClientConfig};
pub use uniserve_engine_runtime::SchedulingPolicy;
use uniserve_engine_runtime::{EngineBackend, EngineCoreConfig};
pub use uniserve_model_profile::ModelDescription;
use uniserve_model_profile::assets::ResolvedModelFiles;
use uniserve_model_profile::tokenizer::{DynTokenizer, HuggingFaceTokenizer};
use uniserve_model_profile::{ModelProfile, ProfileDeploymentConfig};
pub use uniserve_serving::chat::ChatTemplateContentFormatOption;
use uniserve_serving::chat::{ChatTemplateLoadOptions, HfChatRenderer};
use uniserve_serving::{ResolvedModel, ServingRuntime};
use uniserve_sim::{SimEngine, SimExecutor};

pub use crate::http::{ApiError, build_router, serve};
use crate::runtime_client::RuntimeEngineClient;
pub use crate::state::AppState;

#[derive(Debug, Clone, PartialEq, Eq)]
struct RuntimeControlTokens {
    bos: u32,
    eos: Vec<u32>,
    start_of_image: u32,
    end_of_image: u32,
}

fn runtime_control_tokens(
    profile: &ModelProfile,
    backend: EngineBackendKind,
) -> RuntimeControlTokens {
    let controls = match profile {
        ModelProfile::Qwen3(_) => None,
        ModelProfile::SenseNova(profile) => Some(&profile.preprocessing.controls),
        ModelProfile::Bagel(profile) => Some(&profile.preprocessing.controls),
    };
    let bos = controls.map_or(0, |value| value.bos);
    let start_of_image = controls.map_or(0, |value| value.start_of_image);
    let end_of_image = controls.map_or(0, |value| value.end_of_image);
    let primary_eos = controls
        .map(|value| value.eos)
        .filter(|value| *value != 0)
        .or(profile.common().stop_tokens.primary_eos_token_id);
    let mut eos = profile
        .common()
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

fn load_tokenizer(files: &ResolvedModelFiles) -> Result<DynTokenizer> {
    Ok(Arc::new(
        HuggingFaceTokenizer::new(&files.tokenizer_path).with_context(|| {
            format!(
                "failed to load tokenizer from {}",
                files.tokenizer_path.display()
            )
        })?,
    ))
}

async fn resolve_model_assets(
    config: &Config,
) -> Result<(ModelProfile, DynTokenizer, HfChatRenderer, u32)> {
    let files = ResolvedModelFiles::new(&config.model)
        .await
        .with_context(|| format!("failed to resolve model files for `{}`", config.model))?;
    let tokenizer = load_tokenizer(&files)?;
    let deployment = ProfileDeploymentConfig {
        chat_template_override: config.chat_template.clone(),
        max_model_tokens: config.engine.max_model_len,
    };
    let mut profile = ModelProfile::resolve(
        config.model_description,
        &config.model,
        &files,
        &deployment,
        tokenizer.as_ref(),
    )
    .with_context(|| format!("failed to resolve model profile for `{}`", config.model))?;
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
    )
    .context("failed to load the configured Hugging Face chat template")?;
    Ok((profile, tokenizer, renderer, max_model_tokens))
}

/// Build the shared application state for one resolved model and one engine
/// gateway.
pub async fn build_state(config: &Config) -> Result<Arc<AppState>> {
    let (mut profile, tokenizer, renderer, effective_max_model_len) =
        resolve_model_assets(config).await?;
    let control_tokens = runtime_control_tokens(&profile, config.engine.backend);

    let (backend, eos) = match config.engine.backend {
        EngineBackendKind::Sim => (EngineBackend::Sim, control_tokens.eos.clone()),
        EngineBackendKind::Worker => (EngineBackend::Worker, control_tokens.eos.clone()),
    };
    let client = match &config.engine.connection {
        EngineConnection::InProcess => {
            info!(
                ?backend,
                device = %config.engine.device,
                block_size = config.engine.block_size,
                pipeline_depth = config.engine.pipeline_depth,
                "starting UniServe Rust engine (in-process)"
            );
            let engine_config = EngineCoreConfig {
                model: config.model.clone(),
                device: config.engine.device.clone(),
                attention_backend: config.engine.attention_backend.clone(),
                backend,
                block_size: config.engine.block_size,
                pipeline_depth: config.engine.pipeline_depth,
                max_batch: config.engine.max_batch,
                max_num_batched_tokens: config.engine.max_num_batched_tokens,
                max_num_seqs: config.engine.max_num_seqs,
                long_prefill_threshold: config.engine.long_prefill_threshold,
                mixed_prefill_tokens: config.engine.mixed_prefill_tokens,
                scheduler_policy: config.engine.scheduler_policy,
                max_model_len: effective_max_model_len,
                kv_token_capacity: config.engine.kv_token_capacity,
                worker_python: config.engine.worker_python.clone(),
                worker_ranks: config.engine.worker_ranks,
                workers: config.engine.workers.clone(),
                transfer: config.engine.transfer.clone(),
                worker_launch: config.engine.worker_launch.clone(),
                req_slot_cap: 1 << 20,
                resp_slot_cap: config.engine.resp_slot_cap,
                bos: control_tokens.bos,
                eos,
                end_of_image: control_tokens.end_of_image,
            };
            let runtime_client = if backend == EngineBackend::Sim {
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
                RuntimeEngineClient::connect_with_executor(
                    engine_config,
                    Box::new(SimExecutor::new(Box::new(sim))),
                )
            } else {
                RuntimeEngineClient::connect(engine_config)
            }
            .context("failed to start the UniServe engine")?;
            EngineCoreClient::from_in_process(runtime_client)
        }
        connection => {
            let controls = GenerationControlTokens {
                bos: control_tokens.bos,
                eos,
                end_of_image: control_tokens.end_of_image,
            };
            let transport_mode = match connection.clone() {
                EngineConnection::Handshake {
                    handshake_address,
                    advertised_host,
                    engine_count,
                    ready_timeout,
                } => {
                    info!(
                        %handshake_address,
                        engine_count,
                        "connecting to engine cores (handshake-owner mode)"
                    );
                    TransportMode::HandshakeOwner {
                        handshake_address,
                        advertised_host,
                        engine_count,
                        ready_timeout,
                        local_input_address: None,
                        local_output_address: None,
                    }
                }
                EngineConnection::Bootstrapped {
                    input_address,
                    output_address,
                    engine_count,
                    ready_timeout,
                } => {
                    info!(
                        %input_address,
                        %output_address,
                        engine_count,
                        "connecting to engine cores (bootstrapped mode)"
                    );
                    TransportMode::Bootstrapped {
                        input_address,
                        output_address,
                        engine_count,
                        ready_timeout,
                    }
                }
                EngineConnection::InProcess => unreachable!("handled above"),
            };
            EngineCoreClient::connect_zmq(ZmqClientConfig {
                transport_mode,
                model_name: config.model.clone(),
                client_index: 0,
                generation_controls: Some(controls),
            })
            .await
            .context("failed to connect to the UniServe engine cores")?
        }
    };

    let gateway = EngineGateway::new(client).with_log_stats(!config.disable_log_stats);
    let engine_status = gateway.status();
    let snapshot = gateway.snapshot();
    let route_max_model_len = effective_max_model_len.min(snapshot.max_model_len);
    profile.common_mut().context_limits.max_model_tokens = Some(route_max_model_len);
    let model = ResolvedModel::resolve(
        profile,
        tokenizer,
        renderer,
        snapshot.generation_capabilities,
        route_max_model_len,
    )
    .context("failed to bind the configured model description")?;
    let public_model_name = config
        .served_model_name
        .clone()
        .unwrap_or_else(|| model.served_model_name().to_string());
    let runtime = ServingRuntime::new(model, gateway);

    Ok(Arc::new(
        AppState::new(public_model_name, runtime, engine_status)
            .with_log_requests(config.enable_log_requests)
            .with_request_id_headers(config.enable_request_id_headers)
            .with_api_key(config.api_key.clone())
            .with_request_timeout(config.request_timeout)
            .with_max_concurrent_requests(config.max_concurrent_requests),
    ))
}
