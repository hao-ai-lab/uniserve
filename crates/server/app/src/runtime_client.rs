use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::sync::mpsc;
use tracing::{debug, warn};

use uniserve_core::RequestId;
use uniserve_engine_api::{EngineHandle, GenEvent, GenerateRequest as UGenerateRequest};
use uniserve_engine_client::protocol::lora::LoraRequest;
use uniserve_engine_client::protocol::{EngineCoreRequest, ModelDtype};
use uniserve_engine_client::{
    AbortRequest, EngineCoreOutputStream, EngineCoreStreamOutput, Error, InProcessEngineClient,
    NativeEventStream, NativeGenerateRequest, Result,
};
use uniserve_engine_runtime::EngineCore;
use uniserve_engine_wire::translate::{AdapterParams, run_event_adapter, to_generate_request};
use uniserve_executor::Executor;

/// Current unix timestamp in fractional seconds, matching the wire timestamps
/// the metrics layer expects.

/// routes through the single shared epoch helper so it matches the
/// frontend/scheduler wall-clock timestamps.
fn now_secs() -> f64 {
    uniserve_core::now_unix_secs()
}

/// Runtime-backed in-process engine client owned by the server layer.
pub(crate) struct RuntimeEngineClient {
    core: Arc<EngineCore>,
    active: SharedActiveRequests,
    abort_tx: mpsc::UnboundedSender<AbortRequest>,
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
        let (abort_tx, mut abort_rx) = mpsc::unbounded_channel::<AbortRequest>();

        {
            let handle = core.handle();
            let active = Arc::clone(&active);
            tokio::spawn(async move {
                while let Some(req) = abort_rx.recv().await {
                    let rid = lock_active(&active).get(&req.request_id).copied();
                    if let Some(rid) = rid {
                        debug!(request_id = req.request_id, ?req.cause, "aborting request");
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
                    uniserve_engine_client::metrics::record_scheduler_stats(
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
            abort_tx,
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

    fn model_dtype(&self) -> ModelDtype {
 // shared kv_dtype-alias parser (see `ModelDtype::from_kv_str`).
        ModelDtype::from_kv_str(self.core.caps().kv_dtype.as_str())
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
            want_logprobs: req
                .sampling_params
                .as_ref()
                .and_then(|s| s.logprobs)
                .unwrap_or(0)
                > 0,
            native: req.native.is_some(),
        };

        let (event_tx, event_rx) = mpsc::unbounded_channel::<GenEvent>();
        let generate = to_generate_request(&req, rid, event_tx);
        if let Err(e) = self.core.submit(generate) {
            lock_active(&self.active).remove(&request_id);
            return Err(Error::ClientClosed {
                message: e.to_string(),
            });
        }

        let (out_tx, out_rx) = mpsc::unbounded_channel::<Result<EngineCoreStreamOutput>>();
        let active = Arc::clone(&self.active);
        tokio::spawn(async move {
            let request_id = params.request_id.clone();
            run_event_adapter(params, event_rx, |output| {
                out_tx
                    .send(Ok(EngineCoreStreamOutput {
                        engine_index: 0,
                        timestamp: now_secs(),
                        output,
                    }))
                    .is_ok()
            })
            .await;
            lock_active(&active).remove(&request_id);
        });

        Ok(EngineCoreOutputStream::new(
            request_id,
            self.abort_tx.clone(),
            out_rx,
        ))
    }

    fn generate_native(&self, req: NativeGenerateRequest) -> Result<NativeEventStream> {
        let rid = self.core.next_request_id();
        let (event_tx, event_rx) = mpsc::unbounded_channel::<GenEvent>();
        let mut generate = UGenerateRequest::new(
            rid,
            req.prompt_ids,
            req.sampling,
            req.image,
            req.mode,
            req.max_tokens,
            event_tx,
        );
        generate.neg_prompt_ids = req.neg_prompt_ids;
        generate.mm_items = req.mm_items;
        generate.stop_token_ids = req.stop_token_ids;
        self.core
            .submit(generate)
            .map_err(|e| Error::ClientClosed {
                message: e.to_string(),
            })?;
        let handle = self.handle();
        Ok(NativeEventStream::with_cancel(event_rx, move || {
            handle.cancel(rid);
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

    fn reset_prefix_cache(
        &self,
        _reset_running_requests: bool,
        _reset_connector: bool,
    ) -> Result<bool> {
        Ok(self.core.reset_prefix_cache())
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
    use uniserve_engine_client::protocol::{
        EngineCoreFinishReason, EngineCoreRequest, EngineCoreSamplingParams,
    };
    use uniserve_engine_client::{
        EngineCoreClient, EngineSamplingParams, GenEvent, GenMode, ImageParams,
        NativeGenerateRequest,
    };
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

        let request = EngineCoreRequest {
            request_id: "req-1".to_string(),
            prompt_token_ids: Some(vec![1, 2, 3, 4]),
            sampling_params: Some(EngineCoreSamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                max_tokens: 64,
                ..EngineCoreSamplingParams::for_test()
            }),
            ..Default::default()
        };

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

        let request = NativeGenerateRequest {
            prompt_ids: vec![1, 2, 3],
            neg_prompt_ids: vec![],
            sampling: EngineSamplingParams::default(),
            image: ImageParams {
                steps: 4,
                ..ImageParams::default()
            },
            mode: GenMode::Image,
            max_tokens: 0,
            mm_items: vec![],
            stop_token_ids: vec![],
        };

        let mut stream = client
            .generate_native(request)
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
