//! In-process engine construction, submission, and lifecycle management.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tracing::warn;

use super::error::{Error, Result};
use super::media::MediaSubmission;
use uniserve_core::{GenerationLimits, ModelDtype, Request, RequestId, RuntimeFamily};
use uniserve_engine::{EngineCore, EngineHandle, EventRx, Executor};

use crate::serving::TokenizedGenerateReqInput;

/// In-process engine client owned by the server layer.
pub struct EngineClient {
    core: Arc<EngineCore>,
    active: SharedActiveRequests,
    _stats_guard: Arc<()>,
}

type ActiveRequests = HashMap<String, RequestId>;
type SharedActiveRequests = Arc<Mutex<ActiveRequests>>;

/// Locks the active-request registry.
fn lock_active(active: &Mutex<ActiveRequests>) -> std::sync::MutexGuard<'_, ActiveRequests> {
    match active.lock() {
        Ok(guard) => guard,
        Err(error) => {
            warn!("in-process active request map lock poisoned");
            error.into_inner()
        }
    }
}

/// Removes the active request.
fn remove_active_request(active: &Mutex<ActiveRequests>, request_id: &str, rid: RequestId) {
    let mut active = lock_active(active);
    if active.get(request_id) == Some(&rid) {
        active.remove(request_id);
    }
}

impl EngineClient {
    /// Returns the shared engine handle.
    fn handle(&self) -> EngineHandle {
        self.core.handle()
    }

    /// Starts an engine from its worker-backed core configuration.
    pub fn connect(config: uniserve_engine::EngineCoreConfig) -> Result<Self> {
        let core = EngineCore::new(config).map_err(|e| Error::ClientClosed {
            message: format!("failed to start the UniServe engine: {e:?}"),
        })?;
        Self::from_core(core)
    }

    /// Starts an engine over an explicitly supplied executor.
    pub fn connect_with_executor(
        config: uniserve_engine::EngineCoreConfig,
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
        config: uniserve_engine::EngineCoreConfig,
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
    fn from_core(core: EngineCore) -> Result<Self> {
        let core = Arc::new(core);

        let active: SharedActiveRequests = Arc::new(Mutex::new(HashMap::new()));
        let stats_guard = Arc::new(());
        {
            let stats = Arc::clone(core.stats());
            let block_size = core.info().kv_block_size();
            let model_name = core.model_name().to_string();
            let guard = Arc::downgrade(&stats_guard);
            tokio::spawn(async move {
                let mut reporter = crate::scheduler_stats::SchedStatsReporter::default();
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

        Ok(Self {
            core,
            active,
            _stats_guard: stats_guard,
        })
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

    /// Returns whether the worker supports token sampling operations.
    pub fn supports_token_sampling(&self) -> bool {
        self.core.supports_token_sampling()
    }

    /// Returns the worker's effective model dtype.
    pub fn model_dtype(&self) -> ModelDtype {
        self.core.model_dtype()
    }

    /// Returns the worker's UniServe protocol version string.
    pub fn uniserve_version(&self) -> &str {
        "uniserve"
    }

    /// Returns aggregate paged-KV capacity across worker pools.
    pub fn total_num_gpu_blocks(&self) -> u64 {
        self.core.info().kv_num_blocks() as u64
    }

    /// Returns whether the engine can accept requests.
    pub fn is_healthy(&self) -> bool {
        !self.core.is_dead()
    }

    /// Returns the terminal health error, if the engine failed.
    pub fn health_error(&self) -> Option<Arc<Error>> {
        None
    }

    /// Submits one tokenized generation request and returns its event receiver.
    pub async fn submit_generation(&self, input: &TokenizedGenerateReqInput) -> Result<EventRx> {
        self.submit_generation_request(input.request_id.to_string(), input.request.clone())
            .await
    }

    /// Maps an external request identity to an engine identity and submits the canonical request.
    async fn submit_generation_request(
        &self,
        external_request_id: String,
        mut request: uniserve_core::GenerationRequest,
    ) -> Result<EventRx> {
        let rid = self.core.next_request_id();
        request.request_id = rid;
        {
            let mut active = lock_active(&self.active);
            if active.contains_key(&external_request_id) {
                return Err(Error::DuplicateRequestId {
                    request_id: external_request_id,
                });
            }
            active.insert(external_request_id.clone(), rid);
        }
        let request = match self.core.runtime_family() {
            RuntimeFamily::Ar => Request::Ar(request),
            RuntimeFamily::Umm => Request::Umm(request),
            RuntimeFamily::Diffusion => {
                remove_active_request(&self.active, &external_request_id, rid);
                return Err(Error::ClientClosed {
                    message: "text generation is unavailable for a diffusion runtime".to_string(),
                });
            }
        };
        let mut scheduler_rx = self.core.submit(request).map_err(|error| {
            remove_active_request(&self.active, &external_request_id, rid);
            Error::from(error)
        })?;
        let active = Arc::clone(&self.active);
        let active_id = external_request_id.clone();
        scheduler_rx.set_on_finish(move || {
            remove_active_request(&active, &active_id, rid);
        });
        Ok(scheduler_rx)
    }

    /// Submits one terminal media request and returns its event receiver.
    pub async fn submit_media(&self, submission: MediaSubmission) -> Result<EventRx> {
        let MediaSubmission {
            external_request_id,
            prompt_token_ids,
            seed,
            priority,
            geometry,
            ..
        } = submission;
        let rid = self.core.next_request_id();
        {
            let mut active = lock_active(&self.active);
            if active.contains_key(&external_request_id) {
                return Err(Error::DuplicateRequestId {
                    request_id: external_request_id,
                });
            }
            active.insert(external_request_id.clone(), rid);
        }
        if self.core.runtime_family() != RuntimeFamily::Diffusion {
            remove_active_request(&self.active, &external_request_id, rid);
            return Err(Error::ClientClosed {
                message: "diffusion generation is unavailable for this runtime".to_string(),
            });
        }
        let request = uniserve_core::DiffusionRequest {
            request_id: rid,
            prompt_token_ids,
            seed,
            priority,
            geometry,
        };
        let mut scheduler_rx = self
            .core
            .submit(Request::Diffusion(request))
            .map_err(|error| {
                remove_active_request(&self.active, &external_request_id, rid);
                Error::from(error)
            })?;
        let active = Arc::clone(&self.active);
        let active_id = external_request_id.clone();
        scheduler_rx.set_on_finish(move || {
            remove_active_request(&active, &active_id, rid);
        });
        Ok(scheduler_rx)
    }

    /// Aborts each supplied external request identifier.
    pub async fn abort<I, S>(&self, ids: I) -> Result<()>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let handle = self.handle();
        let active = lock_active(&self.active);
        for id in ids {
            if let Some(rid) = active.get(id.as_ref()).copied() {
                handle.abort(rid);
            }
        }
        Ok(())
    }

    /// Cancels each supplied request at its consumed output prefix.
    pub async fn cancel<I, S>(&self, ids: I) -> Result<()>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let handle = self.handle();
        let active = lock_active(&self.active);
        for id in ids {
            if let Some(rid) = active.get(id.as_ref()).copied() {
                handle.cancel(rid);
            }
        }
        Ok(())
    }

    /// Requests graceful shutdown and joins the engine owner thread.
    pub async fn shutdown(&self) -> Result<()> {
        self.core.shutdown();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::engine_client::EngineClient;
    use uniserve_core::{
        ContextSegment, Event, FeedbackNextToken, FeedbackSource, GeneratedImageFeedbackRecipe,
        GenerationBehaviorDescriptor, GenerationConstraint, GenerationPolicyDescriptor,
        GenerationRequest, GenerationResourceBounds, ImageIngestRecipe, ImageKvEffect, ImageParams,
        RequestId, SamplingParams, TriggerPolicyDescriptor, UndVisibility,
    };
    use uniserve_engine::EngineCoreConfig;

    #[tokio::test]
    async fn sim_engine_generates_tokens_through_engine_client() {
        let client = EngineClient::connect_with_executor(
            EngineCoreConfig::sim("sim-model"),
            Box::new(uniserve_engine::SimExecutor::new(
                uniserve_engine::SimEngine::new(),
            )),
        )
        .expect("connect in-process sim engine");

        let constraint = GenerationConstraint::UndOnly;
        let policy = GenerationPolicyDescriptor::default();
        let generation = GenerationRequest {
            request_id: RequestId(0),
            context: vec![ContextSegment::UndTokens {
                token_ids: vec![1, 2, 3, 4],
                visibility: UndVisibility::Internal,
            }],
            negative_context: Vec::new(),
            constraint,
            behavior: GenerationBehaviorDescriptor::resolve(constraint, &policy),
            sampling: SamplingParams::default(),
            image: ImageParams::default(),
            max_und_tokens: 64,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache: Default::default(),
            policy,
            resources: GenerationResourceBounds {
                context_tokens: 4,
                max_kv_tokens: 68,
                ..Default::default()
            },
        };
        let mut stream = client
            .submit_generation_request("req-text".to_string(), generation)
            .await
            .expect("submit request");

        let mut tokens = 0usize;
        let mut finish = None;
        while let Some(event) = stream.next().await {
            match event {
                Event::TextToken { .. } => tokens += 1,
                Event::Finished { reason, .. } => {
                    finish = Some(reason);
                    break;
                }
                Event::Rejected { message } => panic!("request rejected: {message}"),
                Event::Error { message } => panic!("engine error: {message}"),
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
            EngineCoreConfig::sim("sim-model"),
            Box::new(uniserve_engine::SimExecutor::new(
                uniserve_engine::SimEngine::new(),
            )),
        )
        .expect("connect in-process sim engine");

        let constraint = GenerationConstraint::GenOnly;
        let policy = GenerationPolicyDescriptor {
            trigger: TriggerPolicyDescriptor::Token { token_id: 1000 },
            gen_only_start: uniserve_core::GenOnlyStartPolicyDescriptor::Immediate,
            feedback: Some(GeneratedImageFeedbackRecipe {
                source: FeedbackSource::DeviceProduct,
                next_und_token: FeedbackNextToken::EndOfImage,
                ingest: ImageIngestRecipe::vit_only(2, ImageKvEffect::WorkerDefined),
                sample_continuation: true,
            }),
            ..GenerationPolicyDescriptor::default()
        };
        let mut request = GenerationRequest {
            request_id: RequestId(0),
            context: vec![ContextSegment::UndTokens {
                token_ids: vec![1, 2, 3],
                visibility: UndVisibility::Internal,
            }],
            negative_context: Vec::new(),
            constraint,
            behavior: GenerationBehaviorDescriptor::resolve(constraint, &policy),
            sampling: SamplingParams::default(),
            image: ImageParams {
                steps: 4,
                ..ImageParams::default()
            },
            max_und_tokens: 0,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache: Default::default(),
            policy,
            resources: GenerationResourceBounds {
                context_tokens: 3,
                max_kv_tokens: 3,
                ..GenerationResourceBounds::default()
            },
        };
        request.resources =
            GenerationResourceBounds::conservative(uniserve_core::GenerationResources {
                context: &request.context,
                negative_context: &request.negative_context,
                behavior: &request.behavior,
                policy: &request.policy,
                image: &request.image,
                max_und_tokens: request.max_und_tokens,
                cache: &request.cache,
                limits: &client.generation_limits(),
            })
            .expect("request resources must fit the in-process runtime");

        let mut stream = client
            .submit_generation_request("req-image".to_string(), request)
            .await
            .expect("submit generation request");

        let mut begins = 0usize;
        let mut steps = 0usize;
        let mut dones = 0usize;
        let mut finished = false;
        while let Some(ev) = stream.next().await {
            match ev {
                Event::ImageBegin { .. } => begins += 1,
                Event::ImageStep { .. } => steps += 1,
                Event::ImageDone { .. } => dones += 1,
                Event::Finished { images, .. } => {
                    assert_eq!(images, 1, "expected exactly one image");
                    finished = true;
                    break;
                }
                Event::Rejected { message } => panic!("request rejected: {message}"),
                Event::Error { message } => panic!("engine error: {message}"),
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
}
