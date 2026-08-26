use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::sync::mpsc;
use tracing::warn;

use super::client::StreamCancelCause;
use super::error::{Error, Result};
use super::generation::{GenerationEventStream, GenerationSubmission};
use super::media::{MediaEventStream, MediaSubmission};
use uniserve_core::GenerationEvent;
use uniserve_core::{GenerationRuntimeCapabilities, ModelDtype, RequestId};
use uniserve_engine::EngineCore;
use uniserve_engine::EngineHandle;
use uniserve_engine::executor::Executor;

/// In-process engine client owned by the server layer.
pub struct EngineClient {
    core: Arc<EngineCore>,
    active: SharedActiveRequests,
    _stats_guard: Arc<()>,
}

type ActiveRequests = HashMap<String, RequestId>;
type SharedActiveRequests = Arc<Mutex<ActiveRequests>>;

fn lock_active(active: &Mutex<ActiveRequests>) -> std::sync::MutexGuard<'_, ActiveRequests> {
    match active.lock() {
        Ok(guard) => guard,
        Err(error) => {
            warn!("in-process active request map lock poisoned");
            error.into_inner()
        }
    }
}

fn remove_active_request(active: &Mutex<ActiveRequests>, request_id: &str, rid: RequestId) {
    let mut active = lock_active(active);
    if active.get(request_id) == Some(&rid) {
        active.remove(request_id);
    }
}

impl EngineClient {
    fn handle(&self) -> EngineHandle {
        self.core.handle()
    }

    pub fn connect(config: uniserve_engine::EngineCoreConfig) -> Result<Self> {
        let core = EngineCore::new(config).map_err(|e| Error::ClientClosed {
            message: format!("failed to start the UniServe engine: {e:?}"),
        })?;
        Self::from_core(core)
    }

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

    fn from_core(core: EngineCore) -> Result<Self> {
        let core = Arc::new(core);

        let active: SharedActiveRequests = Arc::new(Mutex::new(HashMap::new()));
        let stats_guard = Arc::new(());
        {
            let stats = Arc::clone(core.stats());
            let block_size = core.caps().block_size;
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
                    let wire = reporter.snapshot(&stats, block_size);
                    crate::engine_client::metrics::record_scheduler_stats(
                        &uniserve_observability::METRICS.scheduler,
                        &model_name,
                        0,
                        &wire,
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
    pub fn model_name(&self) -> &str {
        self.core.model_name()
    }

    pub fn engine_count(&self) -> usize {
        1
    }

    pub fn max_model_len(&self) -> u32 {
        self.core.max_model_len()
    }

    pub fn generation_capabilities(&self) -> GenerationRuntimeCapabilities {
        self.core.generation_capabilities()
    }

    pub fn model_dtype(&self) -> ModelDtype {
        self.core.model_dtype()
    }

    pub fn uniserve_version(&self) -> &str {
        "uniserve"
    }

    pub fn total_num_gpu_blocks(&self) -> u64 {
        self.core.caps().num_blocks as u64
    }

    pub fn is_healthy(&self) -> bool {
        !self.core.is_dead()
    }

    pub fn health_error(&self) -> Option<Arc<Error>> {
        None
    }

    pub async fn submit_generation(
        &self,
        submission: GenerationSubmission,
    ) -> Result<GenerationEventStream> {
        let GenerationSubmission {
            external_request_id,
            mut request,
            ..
        } = submission;
        let rid = self.core.next_request_id();
        let decoder_ack_required = !request.stop_strings.is_empty();
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
        let mut scheduler_rx = self.core.submit(request).map_err(|e| {
            remove_active_request(&self.active, &external_request_id, rid);
            Error::ClientClosed {
                message: e.to_string(),
            }
        })?;
        scheduler_rx.delegate_cancellation();
        let (event_tx, event_rx) =
            mpsc::channel::<GenerationEvent>(uniserve_engine::EVENT_BUFFER_CAPACITY);
        let active = Arc::clone(&self.active);
        let active_id = external_request_id.clone();
        tokio::spawn(async move {
            let mut reached_terminal = false;
            while let Some(event) = scheduler_rx.recv().await {
                reached_terminal = matches!(
                    event,
                    GenerationEvent::Finished { .. }
                        | GenerationEvent::Rejected { .. }
                        | GenerationEvent::Error { .. }
                );
                if event_tx.send(event).await.is_err() {
                    remove_active_request(&active, &active_id, rid);
                    return;
                }
                if reached_terminal {
                    break;
                }
            }
            if !reached_terminal {
                let _ = event_tx
                    .send(GenerationEvent::Error {
                        message: "generation event stream closed before a terminal event"
                            .to_string(),
                    })
                    .await;
            }
            remove_active_request(&active, &active_id, rid);
        });
        let handle = self.handle();
        let acknowledge_handle = handle.clone();
        Ok(GenerationEventStream::with_control_policy(
            event_rx,
            move |cause, output_token_count| match cause {
                StreamCancelCause::StopStringMatched => {
                    handle.stop_at(rid, output_token_count);
                }
                StreamCancelCause::DroppedStream => {
                    handle.cancel_at(rid, output_token_count);
                }
            },
            move |output_token_count| {
                acknowledge_handle.acknowledge_at(rid, output_token_count);
            },
            !decoder_ack_required,
        ))
    }

    pub async fn submit_media(&self, submission: MediaSubmission) -> Result<MediaEventStream> {
        let MediaSubmission {
            external_request_id,
            prompt,
            seed,
            priority,
            output_path,
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
        let request = uniserve_core::MediaRequest {
            request_id: rid,
            prompt,
            seed,
            priority,
            output_path,
        };
        let mut scheduler_rx = self
            .core
            .handle()
            .submit_media(request)
            .map_err(|message| {
                remove_active_request(&self.active, &external_request_id, rid);
                Error::ClientClosed { message }
            })?;
        scheduler_rx.delegate_cancellation();
        let (event_tx, event_rx) = mpsc::channel(1);
        let active = Arc::clone(&self.active);
        let active_id = external_request_id.clone();
        tokio::spawn(async move {
            let event = scheduler_rx
                .recv()
                .await
                .unwrap_or(uniserve_core::MediaEvent::Aborted);
            let _ = event_tx.send(event).await;
            remove_active_request(&active, &active_id, rid);
        });
        let handle = self.handle();
        Ok(MediaEventStream::with_cancel(event_rx, move || {
            handle.cancel(rid);
        }))
    }

    pub async fn abort(&self, ids: &[String]) -> Result<()> {
        let handle = self.handle();
        let active = lock_active(&self.active);
        for id in ids {
            if let Some(rid) = active.get(id).copied() {
                handle.abort(rid);
            }
        }
        Ok(())
    }

    pub async fn cancel(&self, ids: &[String]) -> Result<()> {
        let handle = self.handle();
        let active = lock_active(&self.active);
        for id in ids {
            if let Some(rid) = active.get(id).copied() {
                handle.cancel(rid);
            }
        }
        Ok(())
    }

    pub async fn shutdown(self) -> Result<()> {
        self.core.shutdown();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::engine_client::{EngineClient, GenerationSubmission};
    use uniserve_core::{
        ContextSegment, FeedbackNextToken, FeedbackSource, GeneratedImageFeedbackRecipe,
        GenerationBehaviorDescriptor, GenerationConstraint, GenerationEvent,
        GenerationPolicyDescriptor, GenerationRequest, GenerationResourceBounds, ImageIngestRecipe,
        ImageKvEffect, ImageParams, RequestId, SamplingParams, TriggerPolicyDescriptor,
        UndVisibility,
    };
    use uniserve_engine::EngineCoreConfig;

    #[tokio::test]
    async fn sim_engine_generates_tokens_through_engine_client() {
        let client = EngineClient::connect_with_executor(
            EngineCoreConfig::sim("sim-model"),
            Box::new(uniserve_engine::sim::SimExecutor::new(Box::new(
                uniserve_engine::sim::SimEngine::new(),
            ))),
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
            .submit_generation(GenerationSubmission::new("req-text", generation))
            .await
            .expect("submit request");

        let mut tokens = 0usize;
        let mut finish = None;
        while let Some(event) = stream.next().await {
            match event {
                GenerationEvent::TextToken { .. } => tokens += 1,
                GenerationEvent::Finished { reason, .. } => {
                    finish = Some(reason);
                    break;
                }
                GenerationEvent::Rejected { message } => panic!("request rejected: {message}"),
                GenerationEvent::Error { message } => panic!("engine error: {message}"),
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
            Box::new(uniserve_engine::sim::SimExecutor::new(Box::new(
                uniserve_engine::sim::SimEngine::new(),
            ))),
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
        request.resources = GenerationResourceBounds::conservative(
            &request.context,
            &request.negative_context,
            &request.behavior,
            &request.policy,
            &request.image,
            request.max_und_tokens,
            &request.cache,
            &client.generation_capabilities(),
        )
        .expect("request resources must fit the in-process runtime");

        let mut stream = client
            .submit_generation(GenerationSubmission::new("req-image", request))
            .await
            .expect("submit generation request");

        let mut begins = 0usize;
        let mut steps = 0usize;
        let mut dones = 0usize;
        let mut finished = false;
        while let Some(ev) = stream.next().await {
            match ev {
                GenerationEvent::ImageBegin { .. } => begins += 1,
                GenerationEvent::ImageStep { .. } => steps += 1,
                GenerationEvent::ImageDone { .. } => dones += 1,
                GenerationEvent::Finished { images, .. } => {
                    assert_eq!(images, 1, "expected exactly one image");
                    finished = true;
                    break;
                }
                GenerationEvent::Rejected { message } => panic!("request rejected: {message}"),
                GenerationEvent::Error { message } => panic!("engine error: {message}"),
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
