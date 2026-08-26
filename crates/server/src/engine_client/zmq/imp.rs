//! Shared request routing, event dispatch, exact-prefix control, and health state.

use std::collections::BTreeMap;
use std::sync::Arc;

use arc_swap::ArcSwapOption;
use futures::future::join_all;
use parking_lot::Mutex;
use thiserror_ext::AsReport as _;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};
use zeromq::RouterSendHalf;

use crate::engine_client::client::state::EventReceiver;
use crate::engine_client::client::{StreamCancelCause, StreamControl, StreamControlRequest};
use crate::engine_client::error::{client_closed, dispatcher_closed};
use crate::engine_client::zmq::state::RequestRegistry;
use crate::engine_client::zmq::transport::{self, ConnectedEngine, EngineId};
use crate::engine_client::{Error, Result};
use uniserve_core::codec::stats::SchedulerStats;
use uniserve_core::codec::{
    AcknowledgeAt, CancelAt, EngineRequest, GenerationEventBatch, RoutedGenerationEvent, StopAt,
};

pub(crate) struct ClientInner {
    input_send: RouterSendHalf,
    model_name: String,
    request_reg: Mutex<RequestRegistry>,
    health_error: ArcSwapOption<Error>,
}

impl ClientInner {
    /// Create a new instance with the given input send half after the startup
    /// handshake completes.
    pub(crate) fn new(
        input_send: RouterSendHalf,
        model_name: String,
        engines: &[ConnectedEngine],
    ) -> Self {
        Self {
            input_send,
            model_name,
            request_reg: Mutex::new(RequestRegistry::new(engines)),
            health_error: ArcSwapOption::empty(),
        }
    }

    /// Get the model name associated with this client used for metrics
    /// labeling.
    pub(crate) fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Register a newly added request. Return the selected engine id and the
    /// per-request output channel bound to its `request_id`.
    ///
    /// When `data_parallel_rank` is provided, the request is routed to that
    /// specific engine rank, bypassing load balancing.
    pub(crate) fn register_request(
        &self,
        request_id: String,
        data_parallel_rank: Option<u32>,
    ) -> Result<(EngineId, EventReceiver)> {
        let mut registry = self.request_reg.lock();
        if registry.is_closed() {
            return Err(self.closed_error());
        }
        registry.register(request_id, data_parallel_rank)
    }

    /// Undo a request registration when submission fails.
    pub(crate) fn rollback_request(&self, request_id: &str) {
        let _ = self.request_reg.lock().remove(request_id);
    }

    /// Filter the given request IDs to the subset that are still tracked as
    /// active and can be aborted, grouped by the engine that originally
    /// accepted them.
    pub(crate) fn abortable_request_ids(
        &self,
        request_ids: &[String],
    ) -> Result<BTreeMap<EngineId, Vec<String>>> {
        let registry = self.request_reg.lock();
        if registry.is_closed() {
            return Err(self.closed_error());
        }
        Ok(registry.abortable_request_ids(request_ids))
    }

    /// Obtain stream senders for a whole engine event batch with one registry
    /// lock acquisition.
    pub(crate) fn take_senders_for_events<'a>(
        &self,
        events: impl IntoIterator<Item = &'a RoutedGenerationEvent>,
    ) -> Vec<Option<mpsc::Sender<uniserve_core::GenEvent>>> {
        self.request_reg.lock().senders_for_events(events)
    }

    /// Apply one scheduler stats update for the given engine to the local
    /// routing state. Returns `false` if the engine is unknown to the
    /// client.
    pub(crate) fn apply_scheduler_stats(&self, engine_index: u32, stats: &SchedulerStats) -> bool {
        self.request_reg
            .lock()
            .apply_scheduler_stats(engine_index, stats)
    }

    /// Close every active event stream with the first persistent health error.
    pub(crate) fn close_registries(&self, error: Arc<Error>) {
        let persistent_error = self.record_health_error(error);
        let request_senders = self.request_reg.lock().close();

        for sender in request_senders {
            let _ = sender.try_send(uniserve_core::GenEvent::Error {
                message: persistent_error.to_string(),
            });
        }
    }

    /// Return the first persistent health error observed by the client, if any.
    pub(crate) fn health_error(&self) -> Option<Arc<Error>> {
        self.health_error.load_full()
    }

    /// Return whether the client still considers the engine healthy.
    pub(crate) fn is_healthy(&self) -> bool {
        self.health_error.load().is_none()
    }

    /// Send one control-path message to the engine. The request type tag is
    /// derived from the variant, so the type frame and payload frame cannot
    /// disagree. Generation requests are registered through `register_request`
    /// to ensure the request stream is tracked.
    pub(crate) async fn send_to_engine(
        &self,
        engine_id: &EngineId,
        request: EngineRequest,
    ) -> Result<()> {
        let (type_frame, payload) = request.encode_frames()?;
        let mut input_send = self.input_send.clone();
        transport::send_message(&mut input_send, engine_id, type_frame, payload).await?;
        Ok(())
    }

    /// Handle an abort request by sending the abort message to the engine.
    pub(crate) async fn do_abort_requests(
        &self,
        engine_id: &EngineId,
        request_ids: &[String],
    ) -> Result<()> {
        self.send_to_engine(engine_id, EngineRequest::Abort(request_ids.to_vec()))
            .await
    }

    pub(crate) async fn do_cancel_requests(
        &self,
        engine_id: &EngineId,
        request_ids: &[String],
    ) -> Result<()> {
        self.send_to_engine(engine_id, EngineRequest::Cancel(request_ids.to_vec()))
            .await
    }

    pub(crate) async fn do_cancel_at_requests(
        &self,
        engine_id: &EngineId,
        requests: &[CancelAt],
    ) -> Result<()> {
        self.send_to_engine(engine_id, EngineRequest::CancelAt(requests.to_vec()))
            .await
    }

    pub(crate) async fn do_acknowledge_at_requests(
        &self,
        engine_id: &EngineId,
        requests: &[AcknowledgeAt],
    ) -> Result<()> {
        self.send_to_engine(engine_id, EngineRequest::AcknowledgeAt(requests.to_vec()))
            .await
    }

    pub(crate) async fn do_stop_at_requests(
        &self,
        engine_id: &EngineId,
        requests: &[StopAt],
    ) -> Result<()> {
        self.send_to_engine(engine_id, EngineRequest::StopAt(requests.to_vec()))
            .await
    }

    /// Shut down by closing all active request streams with a sticky client closed error.
    pub(crate) fn shutdown(&self) {
        self.close_registries(Arc::new(client_closed!("engine client shut down")));
    }

    /// Return the engine that owns a detached stream. The registry retains the request until its
    /// terminal event so its external identity cannot be reused while cancellation is in flight.
    pub(crate) fn detached_stream_cancel_target(&self, request_id: &str) -> Option<EngineId> {
        let registry = self.request_reg.lock();
        if registry.is_closed() {
            return None;
        }
        registry.engine_for_request(request_id)
    }

    pub(crate) fn stream_control_target(&self, request_id: &str) -> Option<EngineId> {
        let registry = self.request_reg.lock();
        if registry.is_closed() {
            return None;
        }
        registry.engine_for_request(request_id)
    }

    /// Publish the first persistent health error and return the sticky error
    /// recorded for this client. Later failures do not overwrite the first
    /// one so `/health` and post-close callers observe a stable cause.
    fn record_health_error(&self, error: Arc<Error>) -> Arc<Error> {
        if let Some(existing) = self.health_error.load_full() {
            return existing;
        }
        self.health_error.rcu(|current| {
            current
                .as_ref()
                .map(Arc::clone)
                .unwrap_or_else(|| Arc::clone(&error))
        });
        self.health_error.load_full().unwrap_or(error)
    }

    /// Assert there is a recorded health error and return a `Shared` variant
    /// wrapping it for error returns when the client is already closed.
    fn closed_error(&self) -> Error {
        self.health_error
            .load_full()
            .map(Error::Shared)
            .unwrap_or_else(|| client_closed!("engine client registry is closed"))
    }
}

/// Background loop that cancels requests whose output streams stop being consumed.
pub(crate) async fn run_stream_control_loop(
    inner: Arc<ClientInner>,
    mut control_rx: mpsc::UnboundedReceiver<StreamControlRequest>,
) {
    // Coalesce bursts of stream controls per engine.
    const MAX_DRAIN: usize = 1024;
    let mut batch: Vec<StreamControlRequest> = Vec::new();

    while control_rx.recv_many(&mut batch, MAX_DRAIN).await > 0 {
        let mut cutoffs_by_engine: BTreeMap<EngineId, Vec<CancelAt>> = BTreeMap::new();
        let mut stops_by_engine: BTreeMap<EngineId, Vec<StopAt>> = BTreeMap::new();
        let mut acknowledgements_by_engine: BTreeMap<EngineId, Vec<AcknowledgeAt>> =
            BTreeMap::new();

        for StreamControlRequest {
            request_id,
            control,
        } in batch.drain(..)
        {
            match control {
                StreamControl::Cancel {
                    cause,
                    output_token_count,
                } => {
                    let Some(engine_id) = inner.detached_stream_cancel_target(&request_id) else {
                        debug!(request_id, "skip stream cancellation for inactive request");
                        continue;
                    };
                    match cause {
                        StreamCancelCause::DroppedStream => {
                            info!(request_id, "cancelling request due to dropped stream");
                            cutoffs_by_engine
                                .entry(engine_id)
                                .or_default()
                                .push(CancelAt {
                                    external_request_id: request_id,
                                    output_token_count: output_token_count as u64,
                                });
                        }
                        StreamCancelCause::StopStringMatched => {
                            debug!(
                                request_id,
                                "cancelling request after frontend stop-string match"
                            );
                            stops_by_engine.entry(engine_id).or_default().push(StopAt {
                                external_request_id: request_id,
                                output_token_count: output_token_count as u64,
                            });
                        }
                    }
                }
                StreamControl::Acknowledge { output_token_count } => {
                    let Some(engine_id) = inner.stream_control_target(&request_id) else {
                        debug!(
                            request_id,
                            "skip semantic acknowledgement for inactive request"
                        );
                        continue;
                    };
                    acknowledgements_by_engine
                        .entry(engine_id)
                        .or_default()
                        .push(AcknowledgeAt {
                            external_request_id: request_id,
                            output_token_count: output_token_count as u64,
                        });
                }
            }
        }

        for (engine_id, requests) in cutoffs_by_engine {
            if let Err(error) = inner.do_cancel_at_requests(&engine_id, &requests).await {
                warn!(
                    ?engine_id,
                    ?requests,
                    error = %error.as_report(),
                    "failed to cancel requests at their consumed output prefix"
                );
            }
        }
        for (engine_id, requests) in stops_by_engine {
            if let Err(error) = inner.do_stop_at_requests(&engine_id, &requests).await {
                warn!(
                    ?engine_id,
                    ?requests,
                    error = %error.as_report(),
                    "failed to complete requests at matched stop prefixes"
                );
            }
        }
        for (engine_id, requests) in acknowledgements_by_engine {
            if let Err(error) = inner
                .do_acknowledge_at_requests(&engine_id, &requests)
                .await
            {
                warn!(
                    ?engine_id,
                    ?requests,
                    error = %error.as_report(),
                    "failed to acknowledge decoded token prefixes"
                );
            }
        }
    }
}

/// Background loop that listens for engine events and dispatches them to
/// the corresponding request streams based on their `request_id`.
pub(crate) async fn run_output_dispatcher_loop(
    inner: Arc<ClientInner>,
    mut output_rx: mpsc::Receiver<Result<GenerationEventBatch>>,
) {
    let result: Result<()> = async {
        loop {
            let batch = match output_rx.recv().await {
                Some(batch) => batch,
                None => Err(dispatcher_closed!("engine event dispatcher channel closed")),
            }?;

            let senders = inner.take_senders_for_events(&batch.events);
            let mut deliveries = BTreeMap::<
                String,
                Vec<(
                    mpsc::Sender<uniserve_core::GenEvent>,
                    uniserve_core::GenEvent,
                )>,
            >::new();
            for (routed, sender) in batch.events.into_iter().zip(senders) {
                let request_id = routed.external_request_id;
                let Some(sender) = sender else {
                    debug!(request_id, "dropping event for inactive request");
                    continue;
                };
                deliveries
                    .entry(request_id)
                    .or_default()
                    .push((sender, routed.event));
            }
            join_all(
                deliveries
                    .into_iter()
                    .map(|(request_id, events)| async move {
                        for (sender, event) in events {
                            if sender.send(event).await.is_err() {
                                debug!(request_id, "generation event receiver dropped");
                                break;
                            }
                        }
                    }),
            )
            .await;

            if let Some(scheduler_stats) = batch.scheduler_stats.as_ref() {
                if !inner.apply_scheduler_stats(batch.engine_index, scheduler_stats) {
                    debug!(
                        engine_index = batch.engine_index,
                        "dropping scheduler stats for unknown engine"
                    );
                }
                crate::engine_client::metrics::record_scheduler_stats(
                    &uniserve_observability::METRICS.scheduler,
                    inner.model_name(),
                    batch.engine_index,
                    scheduler_stats,
                );
            }
        }
    }
    .await;
    let Err(error) = result else { return };

    warn!(error = %error.as_report(), "output dispatcher exiting with error");
    inner.close_registries(Arc::new(error));
}
