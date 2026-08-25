//! Socket-mode engine client: ZMQ ROUTER/PULL transport to one or more
//! headless `uniserve engine` processes. The startup handshake carries resolved
//! generation control tokens and carries canonical generation events unchanged.

use std::sync::Arc;
use std::time::Duration;

use futures::future::join_all;
use tokio::sync::mpsc;
use tokio_util::task::AbortOnDropHandle;
use tracing::{debug, info, trace};

use uniserve_core::GenerationRuntimeCapabilities;
use uniserve_engine_wire::generation::GenerationControlTokens;

use crate::client::{StreamControl, StreamControlRequest};
use crate::error::{Error, Result};
use crate::generation::{GenerationEventStream, GenerationSubmission};
use crate::protocol::handshake::EngineCoreReadyResponse;
use crate::protocol::{EngineRequest, ModelDtype};
use crate::zmq::imp::{ClientInner, run_output_dispatcher_loop, run_stream_control_loop};

pub(crate) mod imp;
pub(crate) mod state;
pub(crate) mod transport;

pub use transport::{ConnectedEngine, EngineId};

/// How the frontend acquires its request/event transport with headless
/// engine processes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransportMode {
    /// The frontend owns the startup handshake and allocates or binds the
    /// transport addresses itself before replying to engine `HELLO` messages.
    HandshakeOwner {
        /// Shared handshake endpoint that engines dial during startup.
        handshake_address: String,
        /// Host/IP that engines should use to connect back to the frontend
        /// transport sockets.
        advertised_host: String,
        /// Total number of engines expected to join this transport.
        engine_count: usize,
        /// Maximum time to wait for each startup phase to complete. Must cover
        /// model load: the engine answers READY only after its worker loaded.
        ready_timeout: Duration,
        /// Optional explicit bind address for the input ROUTER socket.
        local_input_address: Option<String>,
        /// Optional explicit bind address for the output PULL socket.
        local_output_address: Option<String>,
    },

    /// An external supervisor has already chosen the transport addresses, and
    /// the frontend only needs to bind them and wait for engine registration
    /// frames.
    Bootstrapped {
        /// Input ROUTER socket address that engines will connect to for
        /// requests.
        input_address: String,
        /// Output PULL socket address that engines will connect to for
        /// responses.
        output_address: String,
        /// Total number of engines expected to register on this transport.
        engine_count: usize,
        /// Maximum time to wait for all expected engines to register.
        ready_timeout: Duration,
    },
}

/// Configuration for connecting the frontend to already-running (or
/// concurrently starting) engine processes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZmqClientConfig {
    /// Frontend-to-engine transport setup.
    pub transport_mode: TransportMode,
    /// Model name used for frontend-side metrics labels.
    pub model_name: String,
    /// Frontend client index stamped onto every request.
    pub client_index: u32,
    /// Control-token ids resolved from the tokenizer, shipped to each engine
    /// in the handshake INIT.
    pub generation_controls: Option<GenerationControlTokens>,
}

impl ZmqClientConfig {
    /// Create a config with the given handshake address, expecting a single
    /// engine, and default values for all other fields.
    pub fn new_single(handshake_address: impl Into<String>) -> Self {
        Self {
            transport_mode: TransportMode::HandshakeOwner {
                handshake_address: handshake_address.into(),
                advertised_host: "127.0.0.1".to_string(),
                engine_count: 1,
                ready_timeout: Duration::from_secs(1800),
                local_input_address: None,
                local_output_address: None,
            },
            model_name: String::new(),
            client_index: 0,
            generation_controls: None,
        }
    }

    pub fn with_model_name(mut self, model_name: impl Into<String>) -> Self {
        self.model_name = model_name.into();
        self
    }

    pub fn with_generation_controls(mut self, controls: Option<GenerationControlTokens>) -> Self {
        self.generation_controls = controls;
        self
    }
}

/// The ZMQ-based engine client talking to headless engine processes.
pub struct ZmqEngineCoreClient {
    config: ZmqClientConfig,
    input_address: String,
    output_address: String,
    engines: Vec<ConnectedEngine>,
    inner: Arc<ClientInner>,
    control_tx: mpsc::UnboundedSender<StreamControlRequest>,

    // Background tasks
    output_task: AbortOnDropHandle<()>,
    dispatcher_task: AbortOnDropHandle<()>,
    control_task: AbortOnDropHandle<()>,
}

impl ZmqEngineCoreClient {
    /// Connect to engine processes using the configured transport mode.
    ///
    /// In handshake-owned mode this method drives the full engine startup
    /// handshake. In bootstrapped mode it binds the provided frontend sockets
    /// and waits for the expected engine registration frames.
    pub async fn connect(config: ZmqClientConfig) -> Result<Self> {
        let connected = match &config.transport_mode {
            TransportMode::HandshakeOwner {
                handshake_address,
                advertised_host,
                engine_count,
                ready_timeout,
                local_input_address,
                local_output_address,
            } => {
                transport::connect_handshake(
                    handshake_address,
                    *engine_count,
                    advertised_host,
                    local_input_address.as_deref(),
                    local_output_address.as_deref(),
                    config.generation_controls.clone(),
                    *ready_timeout,
                )
                .await?
            }

            TransportMode::Bootstrapped {
                input_address,
                output_address,
                engine_count,
                ready_timeout,
            } => {
                transport::connect_bootstrapped(
                    input_address,
                    output_address,
                    *engine_count,
                    *ready_timeout,
                )
                .await?
            }
        };

        let (output_tx, output_rx) = mpsc::channel(64);
        let (control_tx, control_rx) = mpsc::unbounded_channel();
        let engines = connected.engines;
        let inner = Arc::new(ClientInner::new(
            connected.input_send,
            config.model_name.clone(),
            &engines,
        ));
        let output_task = AbortOnDropHandle::new(tokio::spawn(transport::run_output_loop(
            connected.output_socket,
            output_tx,
        )));
        let dispatcher_task = AbortOnDropHandle::new(tokio::spawn(run_output_dispatcher_loop(
            Arc::clone(&inner),
            output_rx,
        )));
        let control_task = AbortOnDropHandle::new(tokio::spawn(run_stream_control_loop(
            Arc::clone(&inner),
            control_rx,
        )));

        Ok(Self {
            config,
            input_address: connected.input_address,
            output_address: connected.output_address,
            engines,
            inner,
            control_tx,
            output_task,
            dispatcher_task,
            control_task,
        })
    }

    /// Return the address of the input socket that the client uses to send
    /// requests to the engines.
    pub fn input_address(&self) -> &str {
        &self.input_address
    }

    /// Return the address of the output socket that the client listens on for
    /// engine responses.
    pub fn output_address(&self) -> &str {
        &self.output_address
    }

    pub fn engine_count(&self) -> usize {
        self.engines.len()
    }

    /// Return the engine routing identities of all connected engines.
    pub fn engine_identities(&self) -> Vec<&[u8]> {
        self.engines
            .iter()
            .map(|engine| &*engine.engine_id)
            .collect()
    }

    /// Return the ready responses received from all engines on the input
    /// socket.
    pub fn ready_responses(&self) -> Vec<&EngineCoreReadyResponse> {
        self.engines
            .iter()
            .map(|engine| &engine.ready_response)
            .collect()
    }

    /// Return the engine-reported effective model dtype.
    pub fn model_dtype(&self) -> ModelDtype {
        let Some(engine) = self.engines.first() else {
            debug_assert!(false, "engine core client requires at least one engine");
            return ModelDtype::Float16;
        };
        engine.ready_response.dtype
    }

    /// Return the engine-reported version string.
    pub fn engine_version(&self) -> &str {
        let Some(engine) = self.engines.first() else {
            debug_assert!(false, "engine core client requires at least one engine");
            return "";
        };
        engine.ready_response.uniserve_version.as_str()
    }

    /// Return the total number of GPU blocks summed across all connected
    /// engines.
    pub fn total_num_gpu_blocks(&self) -> u64 {
        debug_assert!(
            !self.engines.is_empty(),
            "engine core client requires at least one engine"
        );
        self.engines
            .iter()
            .map(|engine| engine.ready_response.num_gpu_blocks)
            .sum()
    }

    /// Return the minimum engine-reported `max_model_len` across all engines.
    pub fn max_model_len(&self) -> u32 {
        debug_assert!(
            !self.engines.is_empty(),
            "engine core client requires at least one engine"
        );
        self.engines
            .iter()
            .map(|engine| engine.ready_response.max_model_len as u32)
            .min()
            .unwrap_or(0)
    }

    pub fn generation_capabilities(&self) -> GenerationRuntimeCapabilities {
        let Some(first) = self.engines.first() else {
            return GenerationRuntimeCapabilities::default();
        };
        let mut combined = first.ready_response.generation_capabilities.clone();
        for engine in &self.engines[1..] {
            let caps = &engine.ready_response.generation_capabilities;
            combined.supports_understanding &= caps.supports_understanding;
            combined.supports_vision_encode &= caps.supports_vision_encode;
            combined.supports_latent_encode &= caps.supports_latent_encode;
            combined.supports_image_generation &= caps.supports_image_generation;
            combined.max_latent_units = combined.max_latent_units.min(caps.max_latent_units);
            combined.latent_downsample = combined.latent_downsample.min(caps.latent_downsample);
            combined.max_vae_grid_tokens =
                combined.max_vae_grid_tokens.max(caps.max_vae_grid_tokens);
            combined.max_vit_grid_tokens =
                combined.max_vit_grid_tokens.max(caps.max_vit_grid_tokens);
            combined.max_latent_feature_bytes = combined
                .max_latent_feature_bytes
                .min(caps.max_latent_feature_bytes);
            combined.max_vision_feature_bytes = combined
                .max_vision_feature_bytes
                .min(caps.max_vision_feature_bytes);
            combined.commit_marker_tokens =
                combined.commit_marker_tokens.max(caps.commit_marker_tokens);
            combined.max_cfg_branches = combined.max_cfg_branches.min(caps.max_cfg_branches);
            combined.encoder_cache_entries = combined
                .encoder_cache_entries
                .min(caps.encoder_cache_entries);
        }
        combined
    }

    pub fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    /// Return whether the client still considers the engines healthy.
    pub fn is_healthy(&self) -> bool {
        self.inner.is_healthy()
    }

    /// Return the first persistent health error observed by the client, if any.
    pub fn health_error(&self) -> Option<Arc<Error>> {
        self.inner.health_error()
    }
}

// Client API implementation.
impl ZmqEngineCoreClient {
    pub async fn submit_generation(
        &self,
        submission: GenerationSubmission,
    ) -> Result<GenerationEventStream> {
        let acknowledge_on_receive = submission.request.stop_strings.is_empty();
        let request = submission.into_envelope(self.config.client_index);
        request.validate()?;
        trace!(
            request_id = %request.external_request_id,
            client_index = request.client_index,
            "submitting generation request"
        );

        let request_id = request.external_request_id.clone();
        let data_parallel_rank = request.data_parallel_rank;
        let (engine_id, rx) = self
            .inner
            .register_request(request_id.clone(), data_parallel_rank)?;

        debug!(request_id, ?engine_id, "registered request to engine");

        if let Err(error) = self
            .inner
            .send_to_engine(&engine_id, EngineRequest::Submit(Box::new(request)))
            .await
        {
            self.inner.rollback_request(&request_id);
            return Err(error);
        }

        let cancel_request_id = request_id.clone();
        let acknowledge_request_id = request_id;
        let cancel_tx = self.control_tx.clone();
        let acknowledge_tx = self.control_tx.clone();
        Ok(GenerationEventStream::with_control_policy(
            rx,
            move |cause, output_token_count| {
                let _ = cancel_tx.send(StreamControlRequest {
                    request_id: cancel_request_id,
                    control: StreamControl::Cancel {
                        cause,
                        output_token_count,
                    },
                });
            },
            move |output_token_count| {
                let _ = acknowledge_tx.send(StreamControlRequest {
                    request_id: acknowledge_request_id.clone(),
                    control: StreamControl::Acknowledge { output_token_count },
                });
            },
            acknowledge_on_receive,
        ))
    }

    /// Abort currently in-flight requests by request ID.
    pub async fn abort(&self, ids: &[String]) -> Result<()> {
        let abortable = self.inner.abortable_request_ids(ids)?;

        trace!(request_ids = ?ids, abortable_request_ids = ?abortable, "sending abort request ids");

        if abortable.is_empty() {
            return Ok(());
        }

        for (engine_id, request_ids) in abortable {
            self.inner
                .do_abort_requests(&engine_id, &request_ids)
                .await?;
        }
        Ok(())
    }

    /// Cancel currently in-flight requests by request ID.
    pub async fn cancel(&self, ids: &[String]) -> Result<()> {
        let cancellable = self.inner.abortable_request_ids(ids)?;

        trace!(request_ids = ?ids, cancellable_request_ids = ?cancellable, "sending cancel request ids");

        for (engine_id, request_ids) in cancellable {
            self.inner
                .do_cancel_requests(&engine_id, &request_ids)
                .await?;
        }
        Ok(())
    }

    /// Shut down local client tasks and close transport state.
    pub async fn shutdown(self) -> Result<()> {
        let Self {
            inner,
            control_tx,
            output_task,
            dispatcher_task,
            control_task,
            ..
        } = self;

        info!("shutting down engine client");
        inner.shutdown();
        drop(control_tx);

        // Abort all client tasks first, then await them, in dependency order.
        let tasks = vec![control_task, dispatcher_task, output_task];
        tasks.iter().for_each(|t| t.abort());
        join_all(tasks).await;

        info!("engine client shut down");
        Ok(())
    }
}
