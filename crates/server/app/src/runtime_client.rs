use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::sync::mpsc;
use tracing::{debug, warn};

use uniserve_core::{GenerationRuntimeCapabilities, ModelDtype, RequestId};
use uniserve_engine_api::{EngineHandle, GenEvent};
use uniserve_engine_gateway::transport::protocol::EngineCoreRequest;
use uniserve_engine_gateway::transport::protocol::lora::LoraRequest;
use uniserve_engine_gateway::transport::{
    EngineCoreOutputStream, EngineCoreStreamOutput, Error, GenerationEventStream,
    GenerationSubmission, InProcessEngineClient, Result, StreamCancelRequest,
};
use uniserve_engine_runtime::EngineCore;
use uniserve_engine_wire::translate::{AdapterParams, run_event_adapter, to_generation_request};
use uniserve_executor::Executor;

/// Current unix timestamp in fractional seconds, matching the wire timestamps
/// the metrics layer expects.
///
/// routes through the single shared epoch helper so it matches the
/// frontend/scheduler wall-clock timestamps.
fn now_secs() -> f64 {
    uniserve_core::now_unix_secs()
}

/// Runtime-backed in-process engine client owned by the server layer.
pub(crate) struct RuntimeEngineClient {
    core: Arc<EngineCore>,
    active: SharedActiveRequests,
    cancel_tx: mpsc::UnboundedSender<StreamCancelRequest>,
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

impl RuntimeEngineClient {
    pub(crate) fn connect(config: uniserve_engine_runtime::EngineCoreConfig) -> Result<Self> {
        let core = EngineCore::new(config).map_err(|e| Error::ClientClosed {
            message: format!("failed to start the UniServe engine: {e:?}"),
        })?;
        Self::from_core(core)
    }

    pub(crate) fn connect_with_executor(
        config: uniserve_engine_runtime::EngineCoreConfig,
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
        let (cancel_tx, mut cancel_rx) = mpsc::unbounded_channel::<StreamCancelRequest>();

        {
            let handle = core.handle();
            let active = Arc::clone(&active);
            tokio::spawn(async move {
                while let Some(req) = cancel_rx.recv().await {
                    let rid = lock_active(&active).get(&req.request_id).copied();
                    if let Some(rid) = rid {
                        debug!(request_id = req.request_id, ?req.cause, "cancelling request");
                        handle.cancel(rid);
                    }
                }
            });
        }

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
                    uniserve_engine_gateway::transport::metrics::record_scheduler_stats(
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
            cancel_tx,
            _stats_guard: stats_guard,
        })
    }

    fn handle(&self) -> EngineHandle {
        self.core.handle()
    }
}

impl InProcessEngineClient for RuntimeEngineClient {
    fn model_name(&self) -> &str {
        self.core.model_name()
    }

    fn engine_count(&self) -> usize {
        1
    }

    fn max_model_len(&self) -> u32 {
        self.core.max_model_len()
    }

    fn generation_capabilities(&self) -> GenerationRuntimeCapabilities {
        self.core.generation_capabilities()
    }

    fn model_dtype(&self) -> ModelDtype {
        self.core.model_dtype()
    }

    fn uniserve_version(&self) -> &str {
        "uniserve"
    }

    fn total_num_gpu_blocks(&self) -> u64 {
        self.core.caps().num_blocks as u64
    }

    fn is_healthy(&self) -> bool {
        !self.core.is_dead()
    }

    fn call(&self, req: EngineCoreRequest) -> Result<EngineCoreOutputStream> {
        let request_id = req.request_id.clone();
        let rid = self.core.next_request_id();
        lock_active(&self.active).insert(request_id.clone(), rid);

        let params = AdapterParams {
            request_id: request_id.clone(),
            want_logprobs: req.generation.sampling.n_logprobs > 0,
        };

        let generate = to_generation_request(&req, rid).map_err(|error| {
            lock_active(&self.active).remove(&request_id);
            Error::ClientClosed {
                message: error.to_string(),
            }
        })?;
        let event_rx = self.core.submit(generate).map_err(|e| {
            lock_active(&self.active).remove(&request_id);
            Error::ClientClosed {
                message: e.to_string(),
            }
        })?;

        let (out_tx, out_rx) = mpsc::channel::<Result<EngineCoreStreamOutput>>(
            EngineCoreOutputStream::BUFFER_CAPACITY,
        );
        let active = Arc::clone(&self.active);
        tokio::spawn(async move {
            let request_id = params.request_id.clone();
            run_event_adapter(params, event_rx, |output| {
                let out_tx = out_tx.clone();
                async move {
                    out_tx
                        .send(Ok(EngineCoreStreamOutput {
                            engine_index: 0,
                            timestamp: now_secs(),
                            output,
                        }))
                        .await
                        .is_ok()
                }
            })
            .await;
            lock_active(&active).remove(&request_id);
        });

        Ok(EngineCoreOutputStream::new(
            request_id,
            self.cancel_tx.clone(),
            out_rx,
        ))
    }

    fn submit_generation(&self, submission: GenerationSubmission) -> Result<GenerationEventStream> {
        let GenerationSubmission {
            external_request_id,
            mut request,
            ..
        } = submission;
        let rid = self.core.next_request_id();
        request.request_id = rid;
        lock_active(&self.active).insert(external_request_id.clone(), rid);
        let mut scheduler_rx = self.core.submit(request).map_err(|e| {
            lock_active(&self.active).remove(&external_request_id);
            Error::ClientClosed {
                message: e.to_string(),
            }
        })?;
        let (event_tx, event_rx) = mpsc::channel::<GenEvent>(
            uniserve_engine_gateway::generation::GENERATION_EVENT_BUFFER_CAPACITY,
        );
        let active = Arc::clone(&self.active);
        let active_id = external_request_id.clone();
        tokio::spawn(async move {
            while let Some(event) = scheduler_rx.recv().await {
                let terminal = matches!(
                    event,
                    GenEvent::Finished { .. } | GenEvent::Rejected { .. } | GenEvent::Error { .. }
                );
                if event_tx.send(event).await.is_err() || terminal {
                    break;
                }
            }
            lock_active(&active).remove(&active_id);
        });
        let handle = self.handle();
        let active = Arc::clone(&self.active);
        let active_id = external_request_id;
        Ok(GenerationEventStream::with_cancel(event_rx, move || {
            handle.cancel(rid);
            lock_active(&active).remove(&active_id);
        }))
    }

    fn abort(&self, ids: &[String]) -> Result<()> {
        let handle = self.handle();
        let active = lock_active(&self.active);
        for id in ids {
            if let Some(rid) = active.get(id).copied() {
                handle.abort(rid);
            }
        }
        Ok(())
    }

    fn cancel(&self, ids: &[String]) -> Result<()> {
        let handle = self.handle();
        let active = lock_active(&self.active);
        for id in ids {
            if let Some(rid) = active.get(id).copied() {
                handle.cancel(rid);
            }
        }
        Ok(())
    }

    fn reset_prefix_cache(
        &self,
        reset_running_requests: bool,
        reset_connector: bool,
    ) -> Result<bool> {
        self.core
            .reset_prefix_cache(reset_running_requests, reset_connector)
            .map_err(|error| uniserve_engine_gateway::Error::UnsupportedControl {
                control: error.to_string(),
            })
    }

    fn reset_mm_cache(&self) -> Result<()> {
        self.reset_encoder_cache()
    }

    fn reset_encoder_cache(&self) -> Result<()> {
        self.core.reset_encoder_cache();
        Ok(())
    }

    fn is_sleeping(&self) -> Result<bool> {
        Ok(self.core.is_sleeping())
    }

    fn sleep(&self, _level: u32, _mode: &str) -> Result<()> {
        self.core.sleep();
        Ok(())
    }

    fn wake_up(&self, _tags: Option<Vec<String>>) -> Result<()> {
        self.core.wake_up();
        Ok(())
    }

    fn add_lora(&self, lora_request: &LoraRequest) -> Result<bool> {
        Ok(self.core.add_lora(
            lora_request.lora_int_id as u32,
            lora_request.lora_path.clone(),
        ))
    }

    fn remove_lora(&self, lora_id: u64) -> Result<bool> {
        Ok(self.core.remove_lora(lora_id as u32))
    }

    fn collective_rpc(&self, method: &str) -> Result<Vec<rmpv::Value>> {
        let acks = self
            .core
            .collective_rpc(method)
            .map_err(|message| Error::ClientClosed { message })?;
        Ok(acks
            .into_iter()
            .map(|(rank, ok, _msg)| {
                rmpv::Value::Map(vec![
                    (rmpv::Value::from("rank"), rmpv::Value::from(rank)),
                    (rmpv::Value::from("ok"), rmpv::Value::Boolean(ok)),
                ])
            })
            .collect())
    }

    fn shutdown(self: Box<Self>) -> Result<()> {
        self.core.shutdown();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt as _;
    use uniserve_core::{
        CommitRecipe, ContextSegment, FeedbackNextToken, FeedbackWriteback,
        GeneratedImageFeedbackRecipe, GenerationBehaviorDescriptor, GenerationConstraint,
        GenerationPolicyDescriptor, GenerationRequest, GenerationResourceBounds, ImageParams,
        RequestId, SamplingParams, TriggerPolicyDescriptor, UndVisibility,
    };
    use uniserve_engine_gateway::transport::protocol::{EngineCoreFinishReason, EngineCoreRequest};
    use uniserve_engine_gateway::transport::{EngineCoreClient, GenEvent, GenerationSubmission};
    use uniserve_engine_runtime::EngineCoreConfig;

    use super::RuntimeEngineClient;

    #[tokio::test]
    async fn sim_engine_generates_tokens_through_in_process_adapter() {
        let client = EngineCoreClient::from_in_process(
            RuntimeEngineClient::connect_with_executor(
                EngineCoreConfig::sim("sim-model"),
                Box::new(uniserve_sim::SimExecutor::new(Box::new(
                    uniserve_sim::SimEngine::new(),
                ))),
            )
            .expect("connect in-process sim engine"),
        );

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
            lora_id: None,
            grammar: None,
            cache: Default::default(),
            policy,
            resources: GenerationResourceBounds {
                context_tokens: 4,
                max_kv_tokens: 68,
                ..Default::default()
            },
        };
        let request = EngineCoreRequest::new("req-1".to_string(), generation);

        let mut stream = client.call(request).await.expect("submit request");

        let mut tokens = 0usize;
        let mut finish = None;
        while let Some(item) = stream.next().await {
            let output = item.expect("stream item").output;
            tokens += output.new_token_ids.len();
            if let Some(reason) = output.finish_reason {
                finish = Some(reason);
                break;
            }
        }

        assert!(
            tokens >= 1,
            "expected at least one generated token, got {tokens}"
        );
        assert_eq!(
            finish,
            Some(EngineCoreFinishReason::Stop),
            "expected the request to terminate on the synthetic EOS"
        );

        client.shutdown().await.expect("shutdown engine");
    }

    #[tokio::test]
    async fn sim_engine_generates_image_through_native_adapter() {
        let client = EngineCoreClient::from_in_process(
            RuntimeEngineClient::connect_with_executor(
                EngineCoreConfig::sim("sim-model"),
                Box::new(uniserve_sim::SimExecutor::new(Box::new(
                    uniserve_sim::SimEngine::new(),
                ))),
            )
            .expect("connect in-process sim engine"),
        );

        let constraint = GenerationConstraint::GenOnly;
        let policy = GenerationPolicyDescriptor {
            trigger: TriggerPolicyDescriptor::Token { token_id: 1000 },
            gen_only_start: uniserve_core::GenOnlyStartPolicyDescriptor::Immediate,
            feedback: Some(GeneratedImageFeedbackRecipe {
                commit: CommitRecipe::CommitGenThenWriteback,
                writeback: FeedbackWriteback::DirectKv,
                next_und_token: FeedbackNextToken::EndOfImage,
                logical_positions: 2,
                physical_kv_tokens: uniserve_core::ImageKvEffect::WorkerDefined,
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
            lora_id: None,
            grammar: None,
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
            .expect("submit native request");

        let mut begins = 0usize;
        let mut steps = 0usize;
        let mut dones = 0usize;
        let mut finished = false;
        while let Some(ev) = stream.next().await {
            match ev {
                GenEvent::ImageBegin { .. } => begins += 1,
                GenEvent::ImageStep { .. } => steps += 1,
                GenEvent::ImageDone { .. } => dones += 1,
                GenEvent::Finished { images, .. } => {
                    assert_eq!(images, 1, "expected exactly one image");
                    finished = true;
                    break;
                }
                GenEvent::Rejected { message } => panic!("request rejected: {message}"),
                GenEvent::Error { message } => panic!("engine error: {message}"),
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
