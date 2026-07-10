//! Deployment state composition for UniServe serving.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod config;
pub mod grpc;
pub mod http;
mod lora;
mod runtime_client;
mod scheduler_stats;
mod server_info;
mod state;

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context as _, Result};
pub use config::{
    Config, EngineBackendKind, EngineConnection, EngineSettings, HttpListenerMode, TokenizerMode,
};
use tracing::info;
use uniserve_engine_gateway::EngineGateway;
use uniserve_engine_gateway::transport::protocol::generation::GenerationControlTokens;
use uniserve_engine_gateway::transport::{EngineCoreClient, TransportMode, ZmqClientConfig};
pub use uniserve_engine_runtime::SchedulingPolicy;
use uniserve_engine_runtime::{EngineBackend, EngineCoreConfig};
use uniserve_model_profile::ModelProfile;
use uniserve_serving::ServingRuntime;
pub use uniserve_serving::chat::{
    ChatTemplateContentFormatOption, ParserSelection, RendererSelection,
};
use uniserve_serving::chat::{LoadModelBackendsOptions, load_model_backends};
use uniserve_sim::{SimEngine, SimExecutor};

pub use crate::http::{ApiError, build_router, serve, serve_with_router_extension};
pub use crate::lora::{LoadLoraError, LoraModelResolution, UnloadLoraError};
use crate::runtime_client::RuntimeEngineClient;
pub use crate::server_info::{ServerInfoConfigFormat, ServerInfoSnapshot};
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
    let dialect = profile.generation_dialect.as_ref();
    let bos = dialect.map_or(0, |value| value.controls.bos);
    let start_of_image = dialect.map_or(0, |value| value.controls.start_of_image);
    let end_of_image = dialect.map_or(0, |value| value.controls.end_of_image);
    let primary_eos = dialect
        .map(|value| value.controls.eos)
        .filter(|value| *value != 0)
        .or(profile.stop_tokens.primary_eos_token_id);
    let mut eos = profile
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

/// Build the shared application state for one configured model and one engine
/// client.
pub async fn build_state(config: &Config) -> Result<Arc<AppState>> {
    // Load both backends from the same model metadata so they stay in sync.
    let loaded = load_model_backends(
        &config.model,
        LoadModelBackendsOptions {
            renderer: config.renderer,
            language_model_only: config.language_model_only,
            chat_template: config.chat_template.clone(),
            chat_template_content_format: config.chat_template_content_format,
            default_chat_template_kwargs: config
                .default_chat_template_kwargs
                .clone()
                .unwrap_or_default(),
        },
    )
    .await
    .context("failed to create chat/text backends")?;
    let mut profile = loaded.profile;
    let text_backend = loaded.text_backend;
    let chat_backend = loaded.chat_backend;

    // Resolve the effective context length: an explicit `--max-model-len`
    // override wins; otherwise derive the model's real context length
    // (`max_position_embeddings`) from the loaded backend so we don't silently
    // truncate long-context models to the 8192 default.
    let model_max_model_len = text_backend
        .sampling_hints()
        .ok()
        .and_then(|hints| hints.max_model_len);
    let effective_max_model_len = config
        .engine
        .max_model_len
        .or(model_max_model_len)
        .unwrap_or(EngineSettings::DEFAULT_MAX_MODEL_LEN);

    // Resolve scheduler control tokens once from the profile. Text-only models
    // contribute their repository EOS set without requiring an image dialect.
    profile.context_limits.max_model_tokens = Some(effective_max_model_len);
    profile.parsers.tools = config.tool_call_parser.to_string();
    profile.parsers.reasoning = config.uniserve_reasoning_parser.to_string();
    let control_tokens = runtime_control_tokens(&profile, config.engine.backend);

    // UniServe owns the engine + scheduler in Rust; Python (or the sim) only
    // runs the model forward pass. The engine runs either on a thread inside
    // this process (the zero-hop default) or as one or more headless
    // `uniserve engine` processes behind the wire protocol.
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
                // Text generation terminates model EOS via scheduler control
                // tokens; explicit request stop tokens stay per request. The
                // sim backend gets its fabricated EOS so it terminates too.
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
            // Socket modes ship tokenizer-resolved control tokens to engines
            // during startup; sim engines retain their configured EOS.
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
    let engine_control = gateway.app_control();
    let runtime = ServingRuntime::new(profile, gateway, text_backend, chat_backend)
        .with_max_model_len(effective_max_model_len)
        .with_tool_call_parser(config.tool_call_parser.clone())
        .with_reasoning_parser(config.uniserve_reasoning_parser.clone());

    // If no served names are specified, expose a local checkpoint's directory
    // name as the public ID while still accepting the full path.
    let served_model_names = if config.served_model_name.is_empty() {
        default_served_model_names(&config.model)
    } else {
        config.served_model_name.clone()
    };

    Ok(Arc::new(
        AppState::new(served_model_names, runtime, engine_control)
            .with_log_requests(config.enable_log_requests)
            .with_request_id_headers(config.enable_request_id_headers)
            .with_api_key(config.api_key.clone())
            .with_admin_api_key(config.admin_api_key.clone())
            .with_request_timeout(config.request_timeout)
            .with_max_concurrent_requests(config.max_concurrent_requests)
            .with_server_dev_mode(config.server_dev_mode)
            .with_runtime_lora_updating(config.enable_lora)
            .with_runtime_lora_allowed_path_prefixes(config.lora_allowed_path_prefixes.clone())
            .with_server_info(ServerInfoSnapshot::from_config(config)),
    ))
}

fn default_served_model_names(model: &str) -> Vec<String> {
    let mut names = Vec::new();
    let path = Path::new(model);
    if (path.is_absolute() || path.exists())
        && let Some(name) = path.file_name().and_then(|value| value.to_str())
        && !name.is_empty()
        && name != model
    {
        names.push(name.to_string());
    }
    names.push(model.to_string());
    names
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use uniserve_model_profile::ModelProfile;

    use super::{default_served_model_names, runtime_control_tokens};
    use crate::EngineBackendKind;

    #[test]
    fn local_model_path_defaults_to_basename_alias_then_full_path() {
        assert_eq!(
            default_served_model_names("/home/models/Qwen3-0.6B-Base"),
            vec![
                "Qwen3-0.6B-Base".to_string(),
                "/home/models/Qwen3-0.6B-Base".to_string(),
            ],
        );
    }

    #[test]
    fn remote_model_id_stays_exact() {
        assert_eq!(
            default_served_model_names("Qwen/Qwen3-0.6B-Base"),
            vec!["Qwen/Qwen3-0.6B-Base".to_string()],
        );
    }

    #[test]
    fn text_only_profile_resolves_engine_controls_without_generation_dialect() {
        let mut profile = ModelProfile::text_only("qwen");
        profile.stop_tokens.primary_eos_token_id = Some(2);
        profile.stop_tokens.eos_token_ids = BTreeSet::from([2, 3]);

        let controls = runtime_control_tokens(&profile, EngineBackendKind::Worker);

        assert_eq!(controls.bos, 0);
        assert_eq!(controls.eos, vec![2, 3]);
        assert_eq!(controls.start_of_image, 0);
        assert_eq!(controls.end_of_image, 0);
    }
}
