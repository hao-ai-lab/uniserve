//! Socket-mode engine client: ZMQ ROUTER/PULL transport to one or more
//! headless `uniserve engine` processes. UniServe extends the handshake INIT
//! with `native_controls`, and
//! [`ZmqEngineCoreClient::generate_native`] adapts the wire stream back into
//! typed text+image [`GenEvent`]s.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use futures::future::{join_all, try_join_all};
use tokio::sync::mpsc;
use tokio_util::task::AbortOnDropHandle;
use tracing::{debug, info, trace};

use uniserve_engine_wire::native::NativeControlTokens;

use crate::client::{AbortRequest, EngineCoreOutputStream};
use crate::error::{Error, Result};
use crate::native::{
    NativeEventStream, NativeGenerateRequest, native_request_to_wire,
    native_stream_from_wire_stream,
};
use crate::protocol::handshake::EngineCoreReadyResponse;
use crate::protocol::lora::LoraRequest;
use crate::protocol::utility::EngineCoreUtilityRequest;
use crate::protocol::{EngineCoreControlRequest, EngineCoreRequest, ModelDtype};
use crate::zmq::imp::{ClientInner, run_abort_loop, run_output_dispatcher_loop};

pub(crate) mod imp;
pub(crate) mod state;
pub(crate) mod transport;

pub use transport::{ConnectedEngine, EngineId};

/// How the frontend acquires its request/response transport with headless
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
    /// in the handshake INIT (UniServe extension).
    pub native_controls: Option<NativeControlTokens>,
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
            native_controls: None,
        }
    }

    pub fn with_model_name(mut self, model_name: impl Into<String>) -> Self {
        self.model_name = model_name.into();
        self
    }

    pub fn with_native_controls(mut self, controls: Option<NativeControlTokens>) -> Self {
        self.native_controls = controls;
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
    abort_tx: mpsc::UnboundedSender<AbortRequest>,
    native_seq: AtomicU64,

    // Background tasks
    output_task: AbortOnDropHandle<()>,
    dispatcher_task: AbortOnDropHandle<()>,
    abort_task: AbortOnDropHandle<()>,
}

impl ZmqEngineCoreClient {
    /// Connect to engine processes using the configured transport mode.

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
                    config.native_controls.clone(),
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
        let (abort_tx, abort_rx) = mpsc::unbounded_channel();
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
        let abort_task =
            AbortOnDropHandle::new(tokio::spawn(run_abort_loop(Arc::clone(&inner), abort_rx)));

        Ok(Self {
            config,
            input_address: connected.input_address,
            output_address: connected.output_address,
            engines,
            inner,
            abort_tx,
            native_seq: AtomicU64::new(1),
            output_task,
            dispatcher_task,
            abort_task,
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
    /// Add a new request to an engine and return a per-request raw output
    /// stream.
    pub async fn call(&self, mut req: EngineCoreRequest) -> Result<EngineCoreOutputStream> {
        req.client_index = self.config.client_index;
        req.validate()?;
        trace!(
            request_id = %req.request_id,
            client_index = req.client_index,
            "sending add request"
        );

        let request_id = req.request_id.clone();
        let data_parallel_rank = req.data_parallel_rank;
        let (engine_id, rx) = self
            .inner
            .register_request(request_id.clone(), data_parallel_rank)?;

        debug!(
            request_id = req.request_id,
            ?engine_id,
            "registered request to engine"
        );

        if let Err(error) = self
            .inner
            .send_to_engine(&engine_id, EngineCoreControlRequest::Add(Box::new(req)))
            .await
        {
            // Failed to send the request to the engine, roll back the registration.
            self.inner.rollback_request(&request_id);
            return Err(error);
        }

        Ok(EngineCoreOutputStream::new(
            request_id,
            self.abort_tx.clone(),
            rx,
        ))
    }

    /// Submit a native image/interleave request over the wire and adapt the
    /// per-request output stream back into typed text+image [`GenEvent`]s.
    pub async fn generate_native(&self, req: NativeGenerateRequest) -> Result<NativeEventStream> {
        let seq = self.native_seq.fetch_add(1, Ordering::Relaxed);
        let request_id = format!("native-{}-{}", std::process::id(), seq);
        let wire = native_request_to_wire(req, request_id);
        let stream = self.call(wire).await?;
        Ok(native_stream_from_wire_stream(stream))
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

    /// Call a typed utility method on all connected engines, returning one
    /// decoded result per connected engine if all calls succeed or an error
    /// if any call fails.
    pub async fn call_utility<T, A>(&self, method: &str, args: A) -> Result<Vec<T>>
    where
        T: serde::de::DeserializeOwned,
        A: serde::Serialize + std::fmt::Debug,
    {
        trace!(
            method,
            client_index = self.config.client_index,
            engine_count = self.engines.len(),
            "sending utility request"
        );

        // Phase 1: allocate one call id per engine and build the per-engine
        // request payloads up-front. Any failure here must roll back the call
        // ids already allocated so they do not leak until shutdown.
        let mut pending_calls = Vec::with_capacity(self.engines.len());
        let mut prepared_sends = Vec::with_capacity(self.engines.len());
        for engine in &self.engines {
            let (call_id, rx) = match self.inner.allocate_and_register_utility_call() {
                Ok(pair) => pair,
                Err(err) => {
                    self.inner
                        .unregister_utility_calls(pending_calls.iter().map(|(id, _)| *id));
                    return Err(err);
                }
            };
            let request = match EngineCoreUtilityRequest::new(
                self.config.client_index,
                call_id,
                method,
                &args,
            ) {
                Ok(request) => request,
                Err(err) => {
                    self.inner.unregister_utility_calls(
                        pending_calls
                            .iter()
                            .map(|(id, _)| *id)
                            .chain(std::iter::once(call_id)),
                    );
                    return Err(err.into());
                }
            };
            pending_calls.push((call_id, rx));
            prepared_sends.push((&engine.engine_id, request));
        }

        // Phase 2: dispatch every utility request concurrently; fail fast on
        // the first transport error and roll back.
        let send_futures = prepared_sends.iter().map(|(engine_id, request)| {
            self.inner.send_to_engine(
                engine_id,
                EngineCoreControlRequest::Utility(Box::new(request.clone())),
            )
        });
        if let Err(err) = try_join_all(send_futures).await {
            self.inner
                .unregister_utility_calls(pending_calls.iter().map(|(id, _)| *id));
            return Err(err);
        }

        // Phase 3: wait for all engines to respond and preserve the per-engine
        // result list.
        let futures = pending_calls.into_iter().map(|(call_id, rx)| async move {
            let output = rx.await.map_err(|_| Error::UtilityCallClosed {
                method: method.to_string(),
                call_id,
            })??;
            output.into_typed_result(method).map_err(Error::from)
        });
        try_join_all(futures).await
    }

    /// Execute `collective_rpc` on all engines and flatten all engine results
    /// into one list.
    pub async fn collective_rpc<A, K>(
        &self,
        method: &str,
        timeout: Option<f64>,
        args: A,
        kwargs: K,
    ) -> Result<Vec<rmpv::Value>>
    where
        A: serde::Serialize + std::fmt::Debug,
        K: serde::Serialize + std::fmt::Debug,
    {
        let results = self
            .call_utility::<rmpv::Value, _>("collective_rpc", (method, timeout, args, kwargs))
            .await?;

        Ok(results
            .into_iter()
            .flat_map(|result| match result {
                // Each engine's result is itself the worker-level result list.
                rmpv::Value::Array(results) => results,
                other => vec![other],
            })
            .collect())
    }

    /// Return whether the engine is currently sleeping at any level.
    pub async fn is_sleeping(&self) -> Result<bool> {
        let results: Vec<bool> = self.call_utility("is_sleeping", ()).await?;
        let first = *results
            .first()
            .ok_or_else(|| Error::InconsistentUtilityResults {
                method: "is_sleeping".to_string(),
                values: "[]".to_string(),
            })?;
        if results.iter().all(|&v| v == first) {
            Ok(first)
        } else {
            Err(Error::InconsistentUtilityResults {
                method: "is_sleeping".to_string(),
                values: format!("{results:?}"),
            })
        }
    }

    /// Reset the multi-modal cache.
    pub async fn reset_mm_cache(&self) -> Result<()> {
        self.call_utility::<(), _>("reset_mm_cache", ()).await?;
        Ok(())
    }

    /// Reset the encoder cache.
    pub async fn reset_encoder_cache(&self) -> Result<()> {
        self.call_utility::<(), _>("reset_encoder_cache", ())
            .await?;
        Ok(())
    }

    /// Reset the prefix cache. Returns `true` only when every engine confirms
    /// the reset (AND aggregation).
    pub async fn reset_prefix_cache(&self) -> Result<bool> {
        let results: Vec<bool> = self.call_utility("reset_prefix_cache", ()).await?;
        if results.is_empty() {
            return Err(Error::InconsistentUtilityResults {
                method: "reset_prefix_cache".to_string(),
                values: "[]".to_string(),
            });
        }
        Ok(results.into_iter().all(|ok| ok))
    }

    /// Load or refresh one LoRA adapter on every connected engine.
    pub async fn add_lora(&self, lora_request: &LoraRequest) -> Result<bool> {
        Ok(self
            .call_utility::<bool, _>("add_lora", (lora_request,))
            .await?
            .into_iter()
            .all(|loaded| loaded))
    }

    /// Remove one LoRA adapter from every connected engine.
    pub async fn remove_lora(&self, lora_id: u64) -> Result<bool> {
        Ok(self
            .call_utility::<bool, _>("remove_lora", (lora_id,))
            .await?
            .into_iter()
            .all(|removed| removed))
    }

    /// Put the engines to sleep.
    pub async fn sleep(&self, level: u32, mode: &str) -> Result<()> {
        self.call_utility::<(), _>("sleep", (level, mode)).await?;
        Ok(())
    }

    /// Wake the engines from sleep.
    pub async fn wake_up(&self, tags: Option<Vec<String>>) -> Result<()> {
        self.call_utility::<(), _>("wake_up", (tags,)).await?;
        Ok(())
    }

    /// Shut down local client tasks and close transport state.
    pub async fn shutdown(self) -> Result<()> {
        let Self {
            inner,
            abort_tx,
            output_task,
            dispatcher_task,
            abort_task,
            ..
        } = self;

        info!("shutting down engine client");
        inner.shutdown();
        drop(abort_tx);

        // Abort all client tasks first, then await them, in dependency order.
        let tasks = vec![abort_task, dispatcher_task, output_task];
        tasks.iter().for_each(|t| t.abort());
        join_all(tasks).await;

        info!("engine client shut down");
        Ok(())
    }
}
