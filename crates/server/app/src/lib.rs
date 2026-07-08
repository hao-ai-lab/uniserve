//! Deployment state composition for UniServe serving.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod config;
mod lora;
mod runtime_client;
mod scheduler_stats;
mod server_info;
mod state;

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context as _, Result};
pub use config::{Config, EngineBackendKind, EngineConnection, EngineSettings, HttpListenerMode};
use tracing::info;
use uniserve_chat::{ChatLlm, LoadModelBackendsOptions, load_model_backends};
pub use uniserve_chat::{ChatTemplateContentFormatOption, ParserSelection, RendererSelection};
use uniserve_engine_client::protocol::native::NativeControlTokens;
use uniserve_engine_client::{EngineCoreClient, TransportMode, ZmqClientConfig};
pub use uniserve_engine_runtime::SchedulingPolicy;
use uniserve_engine_runtime::{EngineBackend, EngineCoreConfig};
use uniserve_llm::Llm;
use uniserve_sim::{SimEngine, SimExecutor};
use uniserve_text::TextLlm;

pub use crate::lora::{LoadLoraError, LoraModelResolution, UnloadLoraError};
use crate::runtime_client::RuntimeEngineClient;
pub use crate::server_info::{ServerInfoConfigFormat, ServerInfoSnapshot};
pub use crate::state::AppState;
use uniserve_native_api::resolve_native_profile_for_model;

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

    // Resolve the model's image/interleave control tokens once from the
    // tokenizer; they drive both the scheduler's image FSM and native-surface
    // prompt ingest.
    let native_profile =
        resolve_native_profile_for_model(&config.model, &*text_backend.tokenizer());
    let native_controls = &native_profile.controls;

    // UniServe owns the engine + scheduler in Rust; Python (or the sim) only
    // runs the model forward pass. The engine runs either on a thread inside
    // this process (the zero-hop default) or as one or more headless
    // `uniserve engine` processes behind the wire protocol.
    let (backend, eos) = match config.engine.backend {
        EngineBackendKind::Sim => (EngineBackend::Sim, vec![151645]),
        EngineBackendKind::Worker => (
            EngineBackend::Worker,
            if native_controls.eos != 0 {
                vec![native_controls.eos]
            } else {
                Vec::new()
            },
        ),
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
                bos: native_controls.bos,
                eos,
                start_of_image: native_controls.start_of_image,
                end_of_image: native_controls.end_of_image,
                image_start_ids: native_controls.image_start_ids.clone(),
            };
            let runtime_client = if backend == EngineBackend::Sim {
                RuntimeEngineClient::connect_with_executor(
                    engine_config,
                    Box::new(SimExecutor::new(Box::new(SimEngine::new()))),
                )
            } else {
                RuntimeEngineClient::connect(engine_config)
            }
            .context("failed to start the UniServe engine")?;
            EngineCoreClient::from_in_process(runtime_client)
        }
        connection => {
            // Socket modes: ship the tokenizer-resolved control tokens to the
            // engines over the handshake INIT extension; sim engines keep
            // their fabricated EOS (see EngineCoreConfig::apply_native_controls).
            let controls = NativeControlTokens {
                bos: native_controls.bos,
                eos,
                start_of_image: native_controls.start_of_image,
                end_of_image: native_controls.end_of_image,
                image_start_ids: native_controls.image_start_ids.clone(),
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
                native_controls: Some(controls),
            })
            .await
            .context("failed to connect to the UniServe engine cores")?
        }
    };

    let llm = Llm::new(client).with_log_stats(!config.disable_log_stats);
    let text = TextLlm::new(llm, text_backend);

    let chat = ChatLlm::new(text, chat_backend)
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
        AppState::new(served_model_names, chat)
            .with_log_requests(config.enable_log_requests)
            .with_request_id_headers(config.enable_request_id_headers)
            .with_native_profile(native_profile)
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
    use super::default_served_model_names;

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
}
