//! In-process engine construction, submission, and lifecycle management.

use std::sync::Arc;

use super::error::{Error, Result};
use uniserve_core::{GenerationLimits, ModelDtype, Request, RequestId, RuntimeFamily};
use uniserve_engine::{EngineCore, EngineHandle, EventRx, Executor};

/// In-process engine client owned by the server layer.
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
    fn from_core(core: EngineCore) -> Result<Self> {
        let core = Arc::new(core);

        let requests = Arc::new(super::requests::RequestRegistry::default());
        {
            let stats = Arc::clone(core.stats());
            let block_size = core.info().kv_block_size();
            let model_name = core.model_name().to_string();
            let guard = Arc::downgrade(&core);
            tokio::spawn(async move {
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

    /// Fixed media prediction count advertised by the loaded model.
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
        let requests = Arc::clone(&self.requests);
        scheduler_rx.set_on_finish(move || {
            requests.release_engine(&external_request_id, rid);
        });
        Ok(scheduler_rx)
    }

    /// Submits one terminal media request and returns its event receiver.
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
        EngineCoreOutput, FeedbackNextToken, FeedbackSource, GenerationConstraint,
        GenerationRequest, ImageEncoderInput, ImageGenerationConfig, ImageIngestStep, ImageParams,
        ImageTrigger, SamplingParams,
    };
    use uniserve_engine::EngineConfig;

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
            ..ImageGenerationConfig::default()
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
}
