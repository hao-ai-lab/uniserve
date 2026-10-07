//! In-process engine construction, submission, and lifecycle management.
//!
//! A request passes through the client in two steps. `register_request`
//! reserves an engine `RequestId` for an external identifier before
//! preprocessing, so duplicates are refused early and control commands can
//! reach a request that is still being prepared. `submit_generation` or
//! `submit_media` then claims that registration and hands the request to one
//! engine core; the returned `EventRx` releases the claim when its stream ends
//! or it is dropped.
//!
//! A data-parallel deployment runs one engine core per replica
//! (`EngineCore::replicas`), each with its own scheduler, KV cache and
//! workers. The client routes each submitted request to the live core with
//! the fewest requests in flight, scanning from a start that advances after
//! every pick so ties rotate across the replicas. This is SGLang's
//! `total_requests` data-parallel balancing (the argmin of running plus
//! waiting requests, `data_parallel_controller.py:778-782`) and vLLM's
//! data-parallel client score with equal waiting and running weights and a
//! rotating start (`v1/engine/core_client.py:1398-1426`). Both read load
//! snapshots published by the replicas; the cores here share the client's
//! process, so the counts are exact at every pick.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use super::error::{Error, Result};
use uniserve_core::{GenerationLimits, ModelDtype, Request, RequestId, RuntimeFamily};
use uniserve_engine::{EngineCore, EventRx, Executor, SubmitError};

/// In-process engine client owned by the server layer.
///
/// Construction spawns one Tokio task per engine core, so every constructor
/// must be called within a Tokio runtime.
pub struct EngineClient {
    /// One core per data-parallel replica; the replicas serve the same model
    /// with the same limits.
    cores: Box<[Arc<EngineCore>]>,
    /// Routing state shared with every submitted request's completion hook.
    routing: Arc<Routing>,
    pub(crate) requests: Arc<super::requests::RequestRegistry>,
}

/// Which core holds each submitted request, and how many each holds.
#[derive(Default)]
struct Routing {
    /// Requests submitted to each core whose event stream has not ended.
    in_flight: Box<[AtomicUsize]>,
    /// The core a routing scan starts from.
    start: AtomicUsize,
    /// The core of every submitted request whose event stream has not ended,
    /// so control commands reach the core that holds the request.
    cores: Mutex<HashMap<RequestId, usize>>,
}

impl Routing {
    fn new(cores: usize) -> Self {
        Self {
            in_flight: (0..cores).map(|_| AtomicUsize::new(0)).collect(),
            ..Self::default()
        }
    }

    /// Picks the live core with the fewest requests in flight.
    ///
    /// The scan starts one core later after every pick, so equal loads rotate
    /// across the replicas. Returns `None` when every core is dead.
    fn pick(&self, cores: &[Arc<EngineCore>]) -> Option<usize> {
        let count = cores.len();
        let start = self.start.fetch_add(1, Ordering::Relaxed) % count;
        (0..count)
            .map(|offset| (start + offset) % count)
            .filter(|&index| !cores[index].is_dead())
            .min_by_key(|&index| self.in_flight[index].load(Ordering::Relaxed))
    }

    /// Records `request` on `core` until `release` runs for it.
    fn hold(&self, request: RequestId, core: usize) {
        self.in_flight[core].fetch_add(1, Ordering::Relaxed);
        self.lock().insert(request, core);
    }

    /// Forgets `request`, once its event stream has ended or its submission
    /// failed.
    fn release(&self, request: RequestId) {
        if let Some(core) = self.lock().remove(&request) {
            self.in_flight[core].fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// The core that holds `request`, while its event stream lasts.
    fn core(&self, request: RequestId) -> Option<usize> {
        self.lock().get(&request).copied()
    }

    /// Locks the request map and recovers it after poisoning.
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<RequestId, usize>> {
        self.cores
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl EngineClient {
    /// Starts the engine cores of a worker-backed configuration: one core, or
    /// one per data-parallel replica (`EngineConfig::data_parallel_size`).
    pub fn connect(config: uniserve_engine::EngineConfig) -> Result<Self> {
        let cores = EngineCore::replicas(config).map_err(|e| Error::ClientClosed {
            message: format!("failed to start the UniServe engine: {e:?}"),
        })?;
        Self::from_cores(cores)
    }

    /// Starts an engine over an explicitly supplied executor.
    pub fn connect_with_executor(
        config: uniserve_engine::EngineConfig,
        executor: Box<dyn Executor>,
    ) -> Result<Self> {
        Self::connect_with_executors(config, vec![executor])
    }

    /// Starts one engine core per supplied executor, each serving one
    /// data-parallel replica.
    pub fn connect_with_executors(
        config: uniserve_engine::EngineConfig,
        executors: Vec<Box<dyn Executor>>,
    ) -> Result<Self> {
        let cores = executors
            .into_iter()
            .map(|executor| {
                EngineCore::with_executor(config.clone(), executor).map_err(|e| {
                    Error::ClientClosed {
                        message: format!("failed to start the UniServe engine: {e:?}"),
                    }
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Self::from_cores(cores)
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
        Self::from_cores(vec![core])
    }

    /// Wraps initialized engine cores with request tracking and periodic
    /// statistics export.
    ///
    /// The replicas must serve the same model under the same limits, since a
    /// request is validated once against the client's limits and may land on
    /// any replica; a mismatch fails. One export task per core publishes a
    /// scheduler-stats snapshot every second under that core's engine index,
    /// matching `engine_count`. Each task holds only a weak reference to its
    /// core, so it does not keep the engine alive and exits on the first tick
    /// after the client is dropped. The request registry records completed
    /// requests under the same model name as engine 0.
    fn from_cores(cores: Vec<EngineCore>) -> Result<Self> {
        let cores: Box<[Arc<EngineCore>]> = cores.into_iter().map(Arc::new).collect();
        let first = cores.first().ok_or_else(|| Error::ClientClosed {
            message: "an engine client needs at least one engine core".to_string(),
        })?;
        for core in &cores[1..] {
            if core.model_name() != first.model_name()
                || core.max_model_len() != first.max_model_len()
                || core.generation_limits() != first.generation_limits()
                || core.model_dtype() != first.model_dtype()
                || core.runtime_family() != first.runtime_family()
                || core.info().denoise_steps() != first.info().denoise_steps()
                || core.supports_token_sampling() != first.supports_token_sampling()
            {
                return Err(Error::ClientClosed {
                    message: "data-parallel replicas serve different models or limits".to_string(),
                });
            }
        }

        let requests = Arc::new(super::requests::RequestRegistry::new(
            first.model_name().to_string(),
        ));
        for (engine, core) in (0u32..).zip(cores.iter()) {
            let stats = Arc::clone(core.stats());
            let model_name = core.model_name().to_string();
            let guard = Arc::downgrade(core);
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
                    let snapshot = reporter.snapshot(&stats);
                    crate::engine_client::metrics::record_scheduler_stats(
                        &uniserve_observability::METRICS.scheduler,
                        &model_name,
                        engine,
                        &snapshot,
                    );
                }
            });
        }

        Ok(Self {
            routing: Arc::new(Routing::new(cores.len())),
            cores,
            requests,
        })
    }

    /// The first core, whose model identity and limits every replica shares.
    fn first(&self) -> &EngineCore {
        &self.cores[0]
    }
}

impl EngineClient {
    /// Returns the configured model name.
    pub fn model_name(&self) -> &str {
        self.first().model_name()
    }

    /// Returns the number of engine cores, one per data-parallel replica.
    pub fn engine_count(&self) -> usize {
        self.cores.len()
    }

    /// Returns the maximum supported model context length.
    pub fn max_model_len(&self) -> u32 {
        self.first().max_model_len()
    }

    /// Returns worker-advertised generation limits.
    pub fn generation_limits(&self) -> GenerationLimits {
        self.first().generation_limits()
    }

    /// Returns what the deployment's video denoiser serves, as its worker
    /// reported at startup; `None` for a model without one.
    pub fn video_denoiser(&self) -> Option<uniserve_engine::VideoDenoiserInfo> {
        self.first().info().video_denoiser.clone()
    }

    /// Returns whether the worker supports token sampling calls.
    pub fn supports_token_sampling(&self) -> bool {
        self.first().supports_token_sampling()
    }

    /// Returns the worker's effective model dtype.
    pub fn model_dtype(&self) -> ModelDtype {
        self.first().model_dtype()
    }

    /// Returns the allocatable paged-KV units the schedulers plan against,
    /// summed over the replicas.
    pub fn total_kv_units(&self) -> u64 {
        self.cores
            .iter()
            .map(|core| u64::from(core.info().kv_usable_units()))
            .sum()
    }

    /// Returns whether the engine can accept requests: whether any replica's
    /// core is alive. Requests route only to live replicas.
    pub fn is_healthy(&self) -> bool {
        self.cores.iter().any(|core| !core.is_dead())
    }

    /// Reserves a unique request ID before preprocessing. The caller must submit
    /// the returned ID or complete its decoded-response lifecycle on failure.
    ///
    /// Returns `Error::DuplicateRequestId` when a live request already holds
    /// `external_request_id`. With `output_identity`, the registration also
    /// carries per-request lifecycle statistics. Identifiers come from the
    /// first core's allocator alone, so they are unique across the replicas.
    pub fn register_request(
        &self,
        external_request_id: String,
        output_identity: Option<crate::serving::ModelEventIdentity>,
    ) -> Result<RequestId> {
        let rid = self.first().next_request_id();
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
        let request = match self.first().runtime_family() {
            RuntimeFamily::Ar => Request::Ar(request),
            RuntimeFamily::Umm => Request::Umm(request),
            RuntimeFamily::BlockDiffusion => Request::BlockDiffusion(request),
            RuntimeFamily::Diffusion => {
                self.requests.claim_submission(&external_request_id, rid)?;
                self.requests.release_engine(&external_request_id, rid);
                return Err(Error::ClientClosed {
                    message: "text generation is unavailable for a diffusion runtime".to_string(),
                });
            }
        };
        self.submit(external_request_id, rid, request)
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
        if self.first().runtime_family() != RuntimeFamily::Diffusion {
            self.requests.claim_submission(&external_request_id, rid)?;
            self.requests.release_engine(&external_request_id, rid);
            return Err(Error::ClientClosed {
                message: "diffusion generation is unavailable for this runtime".to_string(),
            });
        }
        self.submit(external_request_id, rid, Request::Diffusion(request))
    }

    /// Routes one claimed request to a replica and attaches its release.
    ///
    /// The request is recorded on its core before the claim, so a control
    /// command accepted once the claim holds finds the core. Once the event
    /// stream ends or the receiver is dropped, the registry forgets the engine
    /// side of the record, so a later control command no longer reaches the
    /// engine, and the routing forgets the request.
    fn submit(
        &self,
        external_request_id: String,
        rid: RequestId,
        request: Request,
    ) -> Result<EventRx> {
        let core = self
            .routing
            .pick(&self.cores)
            .ok_or(Error::Submit(SubmitError::Dead))?;
        self.routing.hold(rid, core);
        if let Err(error) = self.requests.claim_submission(&external_request_id, rid) {
            self.routing.release(rid);
            return Err(error);
        }
        let mut scheduler_rx = self.cores[core].submit(request).map_err(|error| {
            self.requests.release_engine(&external_request_id, rid);
            self.routing.release(rid);
            Error::from(error)
        })?;
        let requests = Arc::clone(&self.requests);
        let routing = Arc::clone(&self.routing);
        scheduler_rx.set_on_finish(move || {
            requests.release_engine(&external_request_id, rid);
            routing.release(rid);
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
    /// The abort reaches the engine core holding the request
    /// (`EngineHandle::abort`) only while a submission holds the
    /// registration's claim, from `claim_submission` until `release_engine`.
    /// Unknown identifiers are ignored, and the call always returns `Ok`.
    pub async fn abort<I, S>(&self, ids: I) -> Result<()>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for id in ids {
            if let Some(rid) = self
                .requests
                .mark_control(id.as_ref(), crate::serving::RequestLifecycleState::Aborting)
                && let Some(core) = self.routing.core(rid)
            {
                self.cores[core].handle().abort(rid);
            }
        }
        Ok(())
    }

    /// Cancels each supplied request at its consumed output prefix.
    ///
    /// Registry handling matches `abort`, except that a request with lifecycle
    /// statistics moves to `Cancelling` and keeps a pending `Aborting`. The
    /// cancellation reaches the engine core holding the request
    /// (`EngineHandle::cancel`) under the same claim condition. Unknown
    /// identifiers are ignored, and the call always returns `Ok`.
    pub async fn cancel<I, S>(&self, ids: I) -> Result<()>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for id in ids {
            if let Some(rid) = self.requests.mark_control(
                id.as_ref(),
                crate::serving::RequestLifecycleState::Cancelling,
            ) && let Some(core) = self.routing.core(rid)
            {
                self.cores[core].handle().cancel(rid);
            }
        }
        Ok(())
    }

    /// Requests graceful shutdown of every engine core and waits until their
    /// scheduler threads have exited.
    ///
    /// Each scheduler finishes outstanding requests and closes its executor,
    /// which on a multi-rank deployment waits for every rank to drain and
    /// exit. The cores shut down concurrently on Tokio's blocking pool,
    /// leaving the calling runtime thread free for other tasks. Repeated
    /// calls are harmless.
    ///
    /// # Errors
    ///
    /// Returns `Error::ClientClosed` when a blocking shutdown task does not
    /// complete, which happens only if it panics or its runtime is shutting
    /// down.
    pub async fn shutdown(&self) -> Result<()> {
        let tasks = self
            .cores
            .iter()
            .map(|core| {
                let core = Arc::clone(core);
                tokio::task::spawn_blocking(move || core.shutdown())
            })
            .collect::<Vec<_>>();
        for task in tasks {
            task.await.map_err(|error| Error::ClientClosed {
                message: format!("engine shutdown did not complete: {error}"),
            })?;
        }
        Ok(())
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
            readout: Vec::new(),
            canvas: None,
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

    /// A text request of `rid` that generates until the simulator's EOS.
    fn text_request(rid: uniserve_core::RequestId) -> GenerationRequest {
        GenerationRequest {
            request_id: rid,
            prompt_token_ids: vec![1, 2, 3, 4],
            multimodal_inputs: Default::default(),
            negative_prompt_token_ids: Vec::new(),
            constraint: GenerationConstraint::UndOnly,
            sampling: SamplingParams::default(),
            image: ImageParams::default(),
            max_und_tokens: 64,
            include_stop_token: false,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache: Default::default(),
            image_generation: ImageGenerationConfig::default(),
            readout: Vec::new(),
            canvas: None,
        }
    }

    /// Two requests in flight together land on different data-parallel
    /// replicas, each served entirely by its own replica.
    #[tokio::test]
    async fn concurrent_requests_spread_over_data_parallel_replicas() {
        let mut executors: Vec<Box<dyn Executor>> = Vec::new();
        let mut observers = Vec::new();
        for _ in 0..2 {
            let mut executor = uniserve_engine::SimExecutor::new(uniserve_engine::SimEngine::new());
            observers.push(executor.observe());
            executors.push(Box::new(executor));
        }
        let client =
            EngineClient::connect_with_executors(EngineConfig::sim("sim-model"), executors)
                .expect("connect two sim replicas");
        assert_eq!(client.engine_count(), 2);

        // Both requests are submitted before either stream is read, so the
        // first is still in flight when the second is routed.
        let mut streams = Vec::new();
        let mut ids = Vec::new();
        for name in ["first", "second"] {
            let rid = client
                .register_request(name.to_string(), None)
                .expect("reserve request");
            ids.push(rid);
            streams.push(
                client
                    .submit_generation(name.to_string(), text_request(rid))
                    .await
                    .expect("submit request"),
            );
        }
        for stream in &mut streams {
            while let Some(event) = stream.next().await {
                if let EngineCoreOutput::Finished { reason, .. } = event {
                    assert_eq!(reason, uniserve_core::FinishReason::Eos);
                    break;
                }
            }
        }
        client.shutdown().await.expect("shutdown engines");

        let served = observers
            .iter()
            .map(|observer| {
                observer
                    .try_iter()
                    .filter_map(|event| match event {
                        uniserve_engine::BatchEvent::Submitted(batch) => Some(batch),
                        uniserve_engine::BatchEvent::Resolved { .. } => None,
                    })
                    .flat_map(|batch| batch.requests)
                    .map(|(call, _)| call.request_key.request_id)
                    .collect::<std::collections::BTreeSet<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(served.iter().map(|ids| ids.len()).sum::<usize>(), 2);
        assert!(served.iter().all(|ids| ids.len() == 1));
        assert!(
            ids.iter()
                .all(|id| served.iter().any(|ids| ids.contains(id)))
        );
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
            readout: Vec::new(),
            canvas: None,
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
