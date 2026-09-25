//! In-process engine construction, submission, and lifecycle management.
//!
//! A request passes through the client in two steps. `register_request`
//! reserves an engine `RequestId` for an external identifier before
//! preprocessing, so duplicates are refused early and control commands can
//! reach a request that is still being prepared. `submit_generation` or
//! `submit_media` then claims that registration and hands the request to the
//! engine; the returned `EventRx` releases the claim when its stream ends or
//! it is dropped.

use std::sync::Arc;

use super::error::{Error, Result};
use uniserve_core::{GenerationLimits, ModelDtype, Request, RequestId, RuntimeFamily};
use uniserve_engine::{EngineCore, EngineHandle, EventRx, Executor};

/// In-process engine client owned by the server layer.
///
/// Construction spawns a Tokio task, so every constructor must be called
/// within a Tokio runtime.
pub struct EngineClient {
    core: Arc<EngineCore>,
    pub(crate) requests: Arc<super::requests::RequestRegistry>,
}

impl EngineClient {
    /// Returns the shared engine handle.
    fn handle(&self) -> EngineHandle {
        self.core.handle()
    }

    /// Starts an engine from its worker-backed core configuration.
    pub fn connect(config: uniserve_engine::EngineConfig) -> Result<Self> {
        let core = EngineCore::new(config).map_err(|e| Error::ClientClosed {
            message: format!("failed to start the UniServe engine: {e:?}"),
        })?;
        Self::from_core(core)
    }

    /// Starts an engine over an explicitly supplied executor.
    pub fn connect_with_executor(
        config: uniserve_engine::EngineConfig,
        executor: Box<dyn Executor>,
    ) -> Result<Self> {
        let core =
            EngineCore::with_executor(config, executor).map_err(|e| Error::ClientClosed {
                message: format!("failed to start the UniServe engine: {e:?}"),
            })?;
        Self::from_core(core)
    }

    /// Starts an engine over an executor and explicit command waker.
    pub fn connect_with_executor_and_waker(
        config: uniserve_engine::EngineConfig,
        executor: Box<dyn Executor>,
        command_waker: uniserve_core::CommandWaker,
    ) -> Result<Self> {
        let core =
            EngineCore::with_executor_and_waker(config, executor, command_waker).map_err(|e| {
                Error::ClientClosed {
                    message: format!("failed to start the UniServe engine: {e:?}"),
                }
            })?;
        Self::from_core(core)
    }

    /// Wraps an initialized engine core with request tracking and periodic statistics export.
    ///
    /// The export task publishes one scheduler-stats snapshot per second as
    /// engine `0`, matching `engine_count`. It holds only a weak reference to
    /// the core, so it does not keep the engine alive and exits on the first
    /// tick after the client is dropped.
    fn from_core(core: EngineCore) -> Result<Self> {
        let core = Arc::new(core);

        let requests = Arc::new(super::requests::RequestRegistry::default());
        {
            let stats = Arc::clone(core.stats());
            let block_size = core.info().kv_block_size();
            let model_name = core.model_name().to_string();
            let guard = Arc::downgrade(&core);
            tokio::spawn(async move {
                // The reporter turns cumulative scheduler counters into
                // per-snapshot increments, so one instance lives across ticks.
                let mut reporter = uniserve_engine::SchedulerStatsReporter::default();
                let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                loop {
                    interval.tick().await;
                    if guard.upgrade().is_none() {
                        return;
                    }
                    let snapshot = reporter.snapshot(&stats, block_size);
                    crate::engine_client::metrics::record_scheduler_stats(
                        &uniserve_observability::METRICS.scheduler,
                        &model_name,
                        0,
                        &snapshot,
                    );
                }
            });
        }

        Ok(Self { core, requests })
    }
}

impl EngineClient {
    /// Returns the configured model name.
    pub fn model_name(&self) -> &str {
        self.core.model_name()
    }

    /// Returns the number of logical engine instances.
    pub fn engine_count(&self) -> usize {
        1
    }

    /// Returns the maximum supported model context length.
    pub fn max_model_len(&self) -> u32 {
        self.core.max_model_len()
    }

    /// Returns worker-advertised generation limits.
    pub fn generation_limits(&self) -> GenerationLimits {
        self.core.generation_limits()
    }

    /// Returns the fixed media prediction count advertised by the loaded model.
    pub fn denoise_steps(&self) -> u32 {
        self.core.info().denoise_steps()
    }

    /// Returns whether the worker supports token sampling calls.
    pub fn supports_token_sampling(&self) -> bool {
        self.core.supports_token_sampling()
    }

    /// Returns the worker's effective model dtype.
    pub fn model_dtype(&self) -> ModelDtype {
        self.core.model_dtype()
    }

    /// Returns aggregate paged-KV capacity across worker pools.
    pub fn total_num_gpu_blocks(&self) -> u64 {
        self.core.info().kv_num_blocks() as u64
    }

    /// Returns whether the engine can accept requests.
    pub fn is_healthy(&self) -> bool {
        !self.core.is_dead()
    }

    /// Reserves a unique request ID before preprocessing. The caller must submit
    /// the returned ID or complete its decoded-response lifecycle on failure.
    ///
    /// Returns `Error::DuplicateRequestId` when a live request already holds
    /// `external_request_id`. With `output_identity`, the registration also
    /// carries per-request lifecycle statistics.
    pub fn register_request(
        &self,
        external_request_id: String,
        output_identity: Option<crate::serving::ModelEventIdentity>,
    ) -> Result<RequestId> {
        let rid = self.core.next_request_id();
        if !self
            .requests
            .register(external_request_id.clone().into(), rid, output_identity)
        {
            return Err(Error::DuplicateRequestId {
                request_id: external_request_id,
            });
        }
        Ok(rid)
    }

    /// Submits the owned tokenized request under its previously reserved identity.
    ///
    /// # Errors
    ///
    /// - `Error::UnknownRequestId` or `Error::DuplicateRequestId` when the
    ///   registration cannot be claimed (see `RequestRegistry::claim_submission`).
    /// - `Error::ClientClosed` when the runtime family is `Diffusion`.
    /// - `Error::Submit` when the engine refuses the request.
    ///
    /// Every error after a successful claim releases the claim again through
    /// `RequestRegistry::release_engine`, which keeps a registration that
    /// carries lifecycle statistics for the caller to complete and removes one
    /// that does not.
    pub async fn submit_generation(
        &self,
        external_request_id: String,
        request: uniserve_core::GenerationRequest,
    ) -> Result<EventRx> {
        let rid = request.request_id;
        self.requests.claim_submission(&external_request_id, rid)?;
        let request = match self.core.runtime_family() {
            RuntimeFamily::Ar => Request::Ar(request),
            RuntimeFamily::Umm => Request::Umm(request),
            RuntimeFamily::Diffusion => {
                self.requests.release_engine(&external_request_id, rid);
                return Err(Error::ClientClosed {
                    message: "text generation is unavailable for a diffusion runtime".to_string(),
                });
            }
        };
        let mut scheduler_rx = self.core.submit(request).map_err(|error| {
            self.requests.release_engine(&external_request_id, rid);
            Error::from(error)
        })?;
        // Once the event stream ends or the receiver is dropped, the registry
        // forgets the engine side of the record, so a later control command no
        // longer reaches the engine.
        let requests = Arc::clone(&self.requests);
        scheduler_rx.set_on_finish(move || {
            requests.release_engine(&external_request_id, rid);
        });
        Ok(scheduler_rx)
    }

    /// Submits one terminal media request and returns its event receiver.
    ///
    /// Fails like `submit_generation`, except that `Error::ClientClosed` is
    /// returned when the runtime family is not `Diffusion`.
    pub async fn submit_media(
        &self,
        external_request_id: String,
        request: uniserve_core::DiffusionRequest,
    ) -> Result<EventRx> {
        let rid = request.request_id;
        self.requests.claim_submission(&external_request_id, rid)?;
        if self.core.runtime_family() != RuntimeFamily::Diffusion {
            self.requests.release_engine(&external_request_id, rid);
            return Err(Error::ClientClosed {
                message: "diffusion generation is unavailable for this runtime".to_string(),
            });
        }
        let mut scheduler_rx = self
            .core
            .submit(Request::Diffusion(request))
            .map_err(|error| {
                self.requests.release_engine(&external_request_id, rid);
                Error::from(error)
            })?;
        let requests = Arc::clone(&self.requests);
        scheduler_rx.set_on_finish(move || {
            requests.release_engine(&external_request_id, rid);
        });
        Ok(scheduler_rx)
    }

    /// Aborts each supplied external request identifier.
    ///
    /// Through `RequestRegistry::mark_control`, a registered request that
    /// carries lifecycle statistics moves to `Aborting`, which overrides a
    /// pending cancellation; one without statistics whose registration is not
    /// claimed is removed, so its later submission fails with
    /// `Error::UnknownRequestId`.
    /// The abort reaches the engine (`EngineHandle::abort`) only while a
    /// submission holds the registration's claim, from `claim_submission`
    /// until `release_engine`. Unknown identifiers are ignored, and the call
    /// always returns `Ok`.
    pub async fn abort<I, S>(&self, ids: I) -> Result<()>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let handle = self.handle();
        for id in ids {
            if let Some(rid) = self
                .requests
                .mark_control(id.as_ref(), crate::serving::RequestLifecycleState::Aborting)
            {
                handle.abort(rid);
            }
        }
        Ok(())
    }

    /// Cancels each supplied request at its consumed output prefix.
    ///
    /// Registry handling matches `abort`, except that a request with lifecycle
    /// statistics moves to `Cancelling` and keeps a pending `Aborting`. The
    /// cancellation reaches the engine (`EngineHandle::cancel`) under the same
    /// claim condition. Unknown identifiers are ignored, and the call always
    /// returns `Ok`.
    pub async fn cancel<I, S>(&self, ids: I) -> Result<()>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let handle = self.handle();
        for id in ids {
            if let Some(rid) = self.requests.mark_control(
                id.as_ref(),
                crate::serving::RequestLifecycleState::Cancelling,
            ) {
                handle.cancel(rid);
            }
        }
        Ok(())
    }

    /// Requests graceful shutdown and waits until the engine's scheduler
    /// thread has exited.
    ///
    /// The scheduler finishes outstanding requests and closes its executor,
    /// which on a multi-rank deployment waits for every rank to drain and
    /// exit. The join therefore runs on Tokio's blocking pool, leaving the
    /// calling runtime thread free for other tasks. Repeated calls are
    /// harmless.
    ///
    /// # Errors
    ///
    /// Returns `Error::ClientClosed` when the blocking shutdown task does not
    /// complete, which happens only if it panics or its runtime is shutting
    /// down.
    pub async fn shutdown(&self) -> Result<()> {
        let core = Arc::clone(&self.core);
        tokio::task::spawn_blocking(move || core.shutdown())
            .await
            .map_err(|error| Error::ClientClosed {
                message: format!("engine shutdown did not complete: {error}"),
            })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use crate::engine_client::EngineClient;
    use uniserve_core::{
        EngineCoreOutput, FeedbackNextToken, FeedbackSource, GenerationConstraint,
        GenerationRequest, ImageEncoderInput, ImageGenerationConfig, ImageIngestStep, ImageParams,
        ImageTrigger, SamplingParams,
    };
    use uniserve_engine::{EngineConfig, Executor};

    #[tokio::test]
    async fn sim_engine_generates_tokens_through_engine_client() {
        let client = EngineClient::connect_with_executor(
            EngineConfig::sim("sim-model"),
            Box::new(uniserve_engine::SimExecutor::new(
                uniserve_engine::SimEngine::new(),
            )),
        )
        .expect("connect in-process sim engine");

        let constraint = GenerationConstraint::UndOnly;
        let policy = ImageGenerationConfig::default();
        let generation = GenerationRequest {
            request_id: client
                .register_request("req-text".to_string(), None)
                .expect("reserve request"),
            prompt_token_ids: vec![1, 2, 3, 4],
            multimodal_inputs: Default::default(),
            negative_prompt_token_ids: Vec::new(),
            constraint,
            sampling: SamplingParams::default(),
            image: ImageParams::default(),
            max_und_tokens: 64,
            include_stop_token: false,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache: Default::default(),
            image_generation: policy,
        };
        let mut stream = client
            .submit_generation("req-text".to_string(), generation)
            .await
            .expect("submit request");

        let mut tokens = 0usize;
        let mut finish = None;
        while let Some(event) = stream.next().await {
            match event {
                EngineCoreOutput::TextToken { .. } => tokens += 1,
                EngineCoreOutput::Finished { reason, .. } => {
                    finish = Some(reason);
                    break;
                }
                EngineCoreOutput::Rejected { message, .. } => panic!("request rejected: {message}"),
                EngineCoreOutput::Error { message }
                | EngineCoreOutput::ArtifactUnavailable { message } => {
                    panic!("engine error: {message}")
                }
                _ => {}
            }
        }

        assert!(
            tokens >= 1,
            "expected at least one generated token, got {tokens}"
        );
        assert_eq!(
            finish,
            Some(uniserve_core::FinishReason::Eos),
            "expected the request to terminate on the synthetic EOS"
        );

        client.shutdown().await.expect("shutdown engine");
    }

    #[tokio::test]
    async fn sim_engine_generates_image_through_engine_client() {
        let client = EngineClient::connect_with_executor(
            EngineConfig::sim("sim-model"),
            Box::new(uniserve_engine::SimExecutor::new(
                uniserve_engine::SimEngine::new(),
            )),
        )
        .expect("connect in-process sim engine");

        let constraint = GenerationConstraint::GenOnly;
        let policy = ImageGenerationConfig {
            trigger: ImageTrigger::Token { token_id: 1000 },
            requires_text_for_image: false,
            feedback_source: Some(FeedbackSource::DeviceProduct),
            feedback_next_token: FeedbackNextToken::EndOfImage,
            num_feedback_positions: 2,
            feedback_encoders: vec![ImageEncoderInput {
                encoder: ImageIngestStep::VitEncode,
                num_kv_tokens: None,
                max_kv_tokens: None,
            }],
            sample_feedback_continuation: true,
        };
        let request = GenerationRequest {
            request_id: client
                .register_request("req-image".to_string(), None)
                .expect("reserve request"),
            prompt_token_ids: vec![1, 2, 3],
            multimodal_inputs: Default::default(),
            negative_prompt_token_ids: Vec::new(),
            constraint,
            sampling: SamplingParams::default(),
            image: ImageParams {
                steps: 4,
                ..ImageParams::default()
            },
            max_und_tokens: 0,
            include_stop_token: false,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache: Default::default(),
            image_generation: policy,
        };
        request
            .validate_resources(&client.generation_limits())
            .expect("request resources must fit the in-process runtime");

        let mut stream = client
            .submit_generation("req-image".to_string(), request)
            .await
            .expect("submit generation request");

        let mut begins = 0usize;
        let mut steps = 0usize;
        let mut dones = 0usize;
        let mut finished = false;
        while let Some(ev) = stream.next().await {
            match ev {
                EngineCoreOutput::ImageBegin { .. } => begins += 1,
                EngineCoreOutput::ImageStep { .. } => steps += 1,
                EngineCoreOutput::ImageDone { .. } => dones += 1,
                EngineCoreOutput::Finished { images, .. } => {
                    assert_eq!(images, 1, "expected exactly one image");
                    finished = true;
                    break;
                }
                EngineCoreOutput::Rejected { message, .. } => panic!("request rejected: {message}"),
                EngineCoreOutput::Error { message }
                | EngineCoreOutput::ArtifactUnavailable { message } => {
                    panic!("engine error: {message}")
                }
                _ => {}
            }
        }

        assert_eq!(begins, 1, "expected one ImageBegin");
        assert!(
            (1..=4).contains(&steps),
            "expected per-step diffusion progress, got {steps}"
        );
        assert_eq!(dones, 1, "expected one ImageDone");
        assert!(finished, "expected a terminal Finished event");

        client.shutdown().await.expect("shutdown engine");
    }

    /// Simulated executor whose teardown lasts until another task releases
    /// it, as closing worker processes that drain and exit takes time.
    struct GatedCloseExecutor {
        inner: uniserve_engine::SimExecutor,
        release: std::sync::mpsc::Receiver<()>,
        released: Arc<AtomicBool>,
    }

    impl Executor for GatedCloseExecutor {
        fn info(&self) -> &uniserve_engine::ExecutorInfo {
            self.inner.info()
        }

        fn is_ready(&self, worker: &uniserve_engine::WorkerId) -> bool {
            self.inner.is_ready(worker)
        }

        fn has_capacity(&self, worker: &uniserve_engine::WorkerId) -> bool {
            self.inner.has_capacity(worker)
        }

        fn command_has_capacity(&self, command: &uniserve_worker_ipc::BatchCommand) -> bool {
            self.inner.command_has_capacity(command)
        }

        fn submit(
            &mut self,
            batch: uniserve_engine::ExecutionBatch,
        ) -> Result<(), uniserve_engine::ExecutorSubmitError> {
            self.inner.submit(batch)
        }

        fn poll(
            &mut self,
            timeout: Duration,
        ) -> Result<Option<uniserve_engine::BatchResult>, uniserve_engine::ExecutorError> {
            self.inner.poll(timeout)
        }

        fn close(&mut self) -> Result<(), uniserve_engine::ExecutorError> {
            // The wait is bounded so that a blocked runtime fails the test
            // instead of hanging it.
            let released = self.release.recv_timeout(Duration::from_secs(5)).is_ok();
            self.released.store(released, Ordering::SeqCst);
            self.inner.close()
        }
    }

    /// Shutdown waits for executor teardown without holding the runtime
    /// thread: on a single-threaded runtime, the task that ends the teardown
    /// runs while `shutdown` is pending.
    #[tokio::test(flavor = "current_thread")]
    async fn shutdown_leaves_the_runtime_free_during_executor_teardown() {
        let (release_tx, release) = std::sync::mpsc::channel();
        let released = Arc::new(AtomicBool::new(false));
        let client = EngineClient::connect_with_executor(
            EngineConfig::sim("sim-model"),
            Box::new(GatedCloseExecutor {
                inner: uniserve_engine::SimExecutor::new(uniserve_engine::SimEngine::new()),
                release,
                released: Arc::clone(&released),
            }),
        )
        .expect("connect in-process sim engine");

        let releaser = tokio::spawn(async move {
            let _ = release_tx.send(());
        });
        client.shutdown().await.expect("shutdown engine");
        releaser.await.expect("release task");

        assert!(
            released.load(Ordering::SeqCst),
            "executor teardown held the runtime thread"
        );
    }
}
