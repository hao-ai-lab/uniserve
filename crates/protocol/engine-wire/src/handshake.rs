use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{ModelDtype, OpaqueValue};
use uniserve_core::GenerationRuntimeCapabilities;

/// Decoded engine startup-handshake payload sent on the handshake socket.
///
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReadyMessage {
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub local: Option<bool>,
    #[serde(default)]
    pub headless: Option<bool>,
    #[serde(default)]
    pub parallel_config_hash: Option<String>,
}

/// Post-initialization configuration sent from each engine on the input socket
/// registration message, after the handshake completes.
///
/// Contains values that may differ from the original config (e.g.
/// `max_model_len` after KV cache auto-fitting, `num_gpu_blocks` after
/// profiling).
///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCoreReadyResponse {
    /// Engine-reported maximum model context length (auto-fitted after
    /// KV cache profiling and may differ from the original config value).
    pub max_model_len: u64,
    /// Number of GPU blocks available for KV cache on this engine.
    pub num_gpu_blocks: u64,
    /// DP coordinator stats publish address, if applicable.
    pub dp_stats_address: Option<String>,
    /// Effective model dtype after the reference resolves `--dtype`.
    pub dtype: ModelDtype,
    /// the reference version reported by the engine process.
    pub uniserve_version: String,
    /// Post-load worker limits used by serving-time capability and resource
    /// validation.
    pub generation_capabilities: GenerationRuntimeCapabilities,
}

/// Frontend-owned ZMQ addresses that are sent to the engine during startup
/// handshake initialization.
///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeAddresses {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub coordinator_input: Option<String>,
    pub coordinator_output: Option<String>,
    pub frontend_stats_publish_address: Option<String>,
}

/// Startup handshake payload sent from the frontend to initialize an engine
/// after receiving `HELLO`.
///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeInitMessage {
    pub addresses: HandshakeAddresses,
    pub parallel_config: BTreeMap<String, OpaqueValue>,
    /// Model control-token ids resolved by the frontend tokenizer for scheduler
    /// lifecycle decisions.
    pub generation_controls: Option<crate::generation::GenerationControlTokens>,
    /// Shared filesystem namespace required by media-capable engines.
    #[serde(default)]
    pub media_spool: Option<String>,
}
