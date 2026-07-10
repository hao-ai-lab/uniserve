//! Northbound engine contract: submit a [`GenerationSubmission`] and receive a stream
//! of [`GenEvent`]s over a per-request channel. Supports text and image events.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
use uniserve_core::RequestId;

pub use uniserve_core::{
    GenerationConstraint as Constraint, GenerationRequest, GrammarSpec, ImageParams as ImgParams,
    SamplingParams as SampParams,
};

/// Finish reasons using the `STOP`/`LENGTH`/`ABORT`/`ERROR` split.
/// `Stop` is a stop-string or stop-token hit (distinct from model `Eos`);
/// `Aborted` is a server-side abort (distinct from client `Cancelled`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    Eos,
    MaxTokens,
    /// A stop string or explicit stop-token id matched.
    Stop,
    ImageDone,
    /// Client-side cancel (receiver dropped or explicit cancel).
    Cancelled,
    /// Server-side abort (admin / lifecycle).
    Aborted,
    /// A repetition guard terminated generation.
    Repetition,
    Error,
}

/// One ranked vocabulary candidate at a generated or prompt token position.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprob {
    pub token_id: u32,
    pub logprob: f32,
    pub rank: u32,
}

/// Ranked candidates for one scored token position.
#[derive(Debug, Clone, PartialEq)]
pub struct PositionLogprobs {
    pub entries: Vec<TokenLogprob>,
}

/// Typed text and image event stream emitted to callers.
#[derive(Debug, Clone)]
pub enum GenEvent {
    Scheduled {
        queued_at: f64,
        scheduled_at: f64,
    },
    TextToken {
        id: u32,
        logprob: Option<f32>,
    },
    /// Ranked candidates for the just-emitted token, including the sampled token.
    TokenLogprobs {
        id: u32,
        candidates: Vec<TokenLogprob>,
    },
    /// Prompt positions scored by one prefill chunk, in prompt order.
    PromptLogprobs {
        positions: Vec<PositionLogprobs>,
    },
    ImageBegin {
        image_id: u32,
        height: u32,
        width: u32,
        steps: u16,
    },
    ImageStep {
        image_id: u32,
        step: u16,
    },
    ImageCommit {
        image_id: u32,
    },
    ImageDone {
        image_id: u32,
        height: u32,
        width: u32,
        bytes: u64,
        sha256: String,
        pixels_png_b64: String,
    },
    Finished {
        reason: FinishReason,
        /// The matched stop string or stop token, when `reason == Stop`.
        stop_reason: Option<String>,
        prompt_tokens: usize,
        completion_tokens: usize,
        images: usize,
        kv_transfer_params: Option<serde_json::Value>,
    },
    Rejected {
        message: String,
    },
    Error {
        message: String,
    },
}

/// Engine-to-caller channel; dropping the receiver cancels the request.
pub type EventTx = tokio::sync::mpsc::UnboundedSender<GenEvent>;
pub type EventRx = tokio::sync::mpsc::UnboundedReceiver<GenEvent>;

pub fn event_channel() -> (EventTx, EventRx) {
    tokio::sync::mpsc::unbounded_channel()
}

/// Submission plumbing kept separate from the pure generation request value.
pub struct GenerationSubmission {
    pub request: GenerationRequest,
    pub event_tx: EventTx,
}

impl GenerationSubmission {
    pub fn new(request: GenerationRequest, event_tx: EventTx) -> Self {
        Self { request, event_tx }
    }
}

/// Collective RPC reply channel payload.
pub type CollectiveRpcReply =
    std::sync::mpsc::Sender<Result<Vec<(u32, bool, Option<String>)>, String>>;

/// Reply channel for one acknowledged prefix-cache reset transaction.
pub type PrefixCacheResetReply = std::sync::mpsc::Sender<Result<bool, String>>;

/// Command sent from a frontend handler to the scheduler thread.
pub enum Command {
    Submit(Box<GenerationSubmission>),
    /// Client-side cancel → `FinishReason::Cancelled`.
    Cancel(RequestId),
    /// Server-side abort → `FinishReason::Aborted`.
    Abort(RequestId),
    /// Clear the prefix cache after applying the requested running-request policy.
    ResetPrefixCache {
        reset_running_requests: bool,
        reply: PrefixCacheResetReply,
    },
    /// Clear the encoder cache (`/reset_encoder_cache`, `/reset_mm_cache`).
    ResetEncoderCache,
    /// Pause/resume admission (the `/sleep` and `/wake_up` endpoints).
    SetSleeping(bool),
    /// Load a LoRA adapter (the `/v1/load_lora_adapter` endpoint).
    LoadLora {
        id: u32,
        path: String,
    },
    /// Unload a LoRA adapter (the `/v1/unload_lora_adapter` endpoint).
    UnloadLora {
        id: u32,
    },
    /// Execute one control method on every worker rank and reply with
    /// `(rank, ok, message)` acks — the collective_rpc surface.
    CollectiveRpc {
        method: String,
        reply: CollectiveRpcReply,
    },
    Shutdown,
}

/// Cloneable front door over the scheduler. `submit` enqueues to the scheduler;
/// dropping the last handle tears the engine down.
///
/// Every send fires the [`CommandWaker`] right after enqueuing, so when the
/// scheduler is parked on an event-driven executor it wakes immediately to
/// observe the command instead of waiting out the park's safety-net timeout.
/// With the no-op waker (the polling path / sim), this is free.
#[derive(Clone)]
pub struct EngineHandle {
    tx: crossbeam_channel::Sender<Command>,
    waker: uniserve_core::CommandWaker,
}

impl EngineHandle {
    /// Construct a handle with the no-op waker (the polling / sim path, which
    /// observes commands through its own timed wait).
    pub fn new(tx: crossbeam_channel::Sender<Command>) -> Self {
        Self::with_waker(tx, uniserve_core::CommandWaker::noop())
    }

    /// Construct a handle that fires `waker` after every enqueue, used when the
    /// engine drives an event-driven executor that parks between steps.
    pub fn with_waker(
        tx: crossbeam_channel::Sender<Command>,
        waker: uniserve_core::CommandWaker,
    ) -> Self {
        Self { tx, waker }
    }

    /// Enqueue a command and wake any parked scheduler. Centralizes the
    /// send-then-wake order so no caller can forget the wake.
    fn send(&self, cmd: Command) -> Result<(), crossbeam_channel::SendError<Command>> {
        let r = self.tx.send(cmd);
        // Wake only on a successful enqueue: if the channel is closed there is
        // no scheduler to wake, and the error is propagated to the caller.
        if r.is_ok() {
            self.waker.wake();
        }
        r
    }

    pub fn submit(&self, request: GenerationRequest) -> Result<EventRx, String> {
        let (event_tx, event_rx) = event_channel();
        self.send(Command::Submit(Box::new(GenerationSubmission::new(
            request, event_tx,
        ))))
        .map_err(|e| e.to_string())?;
        Ok(event_rx)
    }
    pub fn cancel(&self, id: RequestId) {
        let _ = self.send(Command::Cancel(id));
    }
    /// Server-side abort, distinct from a client cancel.
    pub fn abort(&self, id: RequestId) {
        let _ = self.send(Command::Abort(id));
    }
    /// Clear the prefix cache and wait for the scheduler to acknowledge the transaction.
    pub fn reset_prefix_cache(&self, reset_running_requests: bool) -> Result<bool, String> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        self.send(Command::ResetPrefixCache {
            reset_running_requests,
            reply: reply_tx,
        })
        .map_err(|error| error.to_string())?;
        reply_rx
            .recv_timeout(std::time::Duration::from_secs(60))
            .map_err(|error| format!("prefix-cache reset reply channel: {error}"))?
    }
    /// Clear the encoder cache.
    pub fn reset_encoder_cache(&self) {
        let _ = self.send(Command::ResetEncoderCache);
    }
    /// Pause/resume admission.
    pub fn set_sleeping(&self, sleeping: bool) {
        let _ = self.send(Command::SetSleeping(sleeping));
    }
    /// Load/unload a LoRA adapter through the control plane.
    pub fn load_lora(&self, id: u32, path: String) {
        let _ = self.send(Command::LoadLora { id, path });
    }
    pub fn unload_lora(&self, id: u32) {
        let _ = self.send(Command::UnloadLora { id });
    }
    /// Execute one control method on every worker rank, awaiting per-rank acks
    /// (blocks the caller; the scheduler executes it inline between steps).
    pub fn collective_rpc(
        &self,
        method: impl Into<String>,
    ) -> Result<Vec<(u32, bool, Option<String>)>, String> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        self.send(Command::CollectiveRpc {
            method: method.into(),
            reply: reply_tx,
        })
        .map_err(|e| e.to_string())?;
        reply_rx
            .recv_timeout(std::time::Duration::from_secs(60))
            .map_err(|e| format!("collective_rpc reply channel: {e}"))?
    }
    pub fn shutdown(&self) {
        let _ = self.send(Command::Shutdown);
    }
}

#[cfg(test)]
mod tests {
    use uniserve_core::{
        ContextSegment, GenerationBehaviorDescriptor, GenerationConstraint,
        GenerationPolicyDescriptor, GenerationResourceBounds, ImageParams, RequestId,
        SamplingParams, UndVisibility,
    };

    use super::*;

    fn test_request(request_id: u64) -> GenerationRequest {
        let constraint = GenerationConstraint::UndOnly;
        let policy = GenerationPolicyDescriptor::default();
        GenerationRequest {
            request_id: RequestId(request_id),
            context: vec![ContextSegment::UndTokens {
                token_ids: vec![1, 2, 3],
                visibility: UndVisibility::Internal,
            }],
            negative_context: Vec::new(),
            constraint,
            behavior: GenerationBehaviorDescriptor::resolve(constraint, &policy),
            sampling: SamplingParams::default(),
            image: ImageParams::default(),
            max_und_tokens: 32,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            lora_id: None,
            grammar: None,
            cache: Default::default(),
            policy,
            resources: GenerationResourceBounds {
                context_tokens: 3,
                max_kv_tokens: 35,
                ..GenerationResourceBounds::default()
            },
        }
    }

    /// The canonical request is pure value data with explicit context, policy,
    /// behavior, and resource declarations.
    #[test]
    fn generation_request_is_canonical_pure_data() {
        let request = test_request(7);

        assert_eq!(request.request_id, RequestId(7));
        assert_eq!(request.prompt_token_count(), 3);
        assert_eq!(request.max_und_tokens, 32);
        assert_eq!(request.constraint, GenerationConstraint::UndOnly);
        assert!(request.negative_context.is_empty());
        assert!(request.stop_strings.is_empty());
        assert!(request.stop_token_ids.is_empty());
        assert_eq!(request.priority, 0);
        assert_eq!(request.lora_id, None);
        assert_eq!(request.context_image_count(), 0);
        assert_eq!(request.grammar, None);
        assert!(request.cache.read);
        assert!(request.cache.write);
        assert!(request.validate().is_ok());
    }

    /// `submit` enqueues a `Command::Submit` carrying the request, and the
    /// scheduler side receives exactly that request id.
    #[test]
    fn submit_enqueues_submit_command_with_request() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = EngineHandle::new(tx);

        let _events = handle.submit(test_request(11)).unwrap();

        match rx.recv().unwrap() {
            Command::Submit(submission) => {
                assert_eq!(submission.request.request_id, RequestId(11));
            }
            _ => panic!("expected Submit command"),
        }
    }

    /// `cancel` and `abort` send distinct commands (client cancel vs server
    /// abort) carrying the target request id.
    #[test]
    fn cancel_and_abort_send_distinct_commands() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = EngineHandle::new(tx);

        handle.cancel(RequestId(1));
        handle.abort(RequestId(2));

        match rx.recv().unwrap() {
            Command::Cancel(id) => assert_eq!(id, RequestId(1)),
            _ => panic!("expected Cancel command"),
        }
        match rx.recv().unwrap() {
            Command::Abort(id) => assert_eq!(id, RequestId(2)),
            _ => panic!("expected Abort command"),
        }
    }

    /// LoRA load/unload map onto their respective commands with id and path
    /// preserved.
    #[test]
    fn load_and_unload_lora_send_lora_commands() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = EngineHandle::new(tx);

        handle.load_lora(3, "/adapters/x".to_string());
        handle.unload_lora(3);

        match rx.recv().unwrap() {
            Command::LoadLora { id, path } => {
                assert_eq!(id, 3);
                assert_eq!(path, "/adapters/x");
            }
            _ => panic!("expected LoadLora command"),
        }
        match rx.recv().unwrap() {
            Command::UnloadLora { id } => assert_eq!(id, 3),
            _ => panic!("expected UnloadLora command"),
        }
    }

    /// After the scheduler side of the channel is dropped, `submit` reports an
    /// error rather than panicking or silently dropping the request.
    #[test]
    fn submit_after_receiver_dropped_returns_error() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = EngineHandle::new(tx);
        drop(rx);

        let result = handle.submit(test_request(99));
        assert!(result.is_err());
    }

    /// The waker fires exactly once per successful enqueue, and not at all when
    /// the channel is closed (no scheduler to wake).
    #[test]
    fn waker_fires_on_successful_enqueue_only() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let wakes = Arc::new(AtomicUsize::new(0));
        let wakes_clone = Arc::clone(&wakes);
        let waker = uniserve_core::CommandWaker::new(move || {
            wakes_clone.fetch_add(1, Ordering::SeqCst);
        });

        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = EngineHandle::with_waker(tx, waker);

        handle.reset_encoder_cache();
        handle.set_sleeping(true);
        assert_eq!(wakes.load(Ordering::SeqCst), 2);

        // Closing the channel means there is no scheduler to wake.
        drop(rx);
        handle.reset_encoder_cache();
        assert_eq!(wakes.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn prefix_cache_reset_round_trips_policy_and_result() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = EngineHandle::new(tx);
        let scheduler = std::thread::spawn(move || match rx.recv().unwrap() {
            Command::ResetPrefixCache {
                reset_running_requests,
                reply,
            } => {
                assert!(reset_running_requests);
                reply.send(Ok(true)).unwrap();
            }
            _ => panic!("expected ResetPrefixCache command"),
        });

        assert!(handle.reset_prefix_cache(true).unwrap());
        scheduler.join().unwrap();
    }

    /// `collective_rpc` delivers the method to the scheduler side and returns
    /// the per-rank acks the scheduler replies with.
    #[test]
    fn collective_rpc_round_trips_method_and_reply() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = EngineHandle::new(tx);

        // Stand in for the scheduler: receive the command, reply with acks.
        let scheduler = std::thread::spawn(move || match rx.recv().unwrap() {
            Command::CollectiveRpc { method, reply } => {
                assert_eq!(method, "warmup");
                reply
                    .send(Ok(vec![
                        (0, true, None),
                        (1, false, Some("oops".to_string())),
                    ]))
                    .unwrap();
            }
            _ => panic!("expected CollectiveRpc command"),
        });

        let acks = handle.collective_rpc("warmup").unwrap();
        scheduler.join().unwrap();

        assert_eq!(acks.len(), 2);
        assert_eq!(acks[0], (0, true, None));
        assert_eq!(acks[1], (1, false, Some("oops".to_string())));
    }

    /// Cloned handles share the same underlying channel: a command sent through
    /// the clone is observed by the original's receiver.
    #[test]
    fn cloned_handle_shares_channel() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = EngineHandle::new(tx);
        let clone = handle.clone();

        clone.shutdown();

        assert!(matches!(rx.recv().unwrap(), Command::Shutdown));
    }

    /// Finish reasons are distinct values, so a client-side cancel never
    /// compares equal to a server-side abort.
    #[test]
    fn finish_reason_variants_are_distinct() {
        assert_ne!(FinishReason::Cancelled, FinishReason::Aborted);
        assert_ne!(FinishReason::Eos, FinishReason::Stop);
        assert_eq!(FinishReason::MaxTokens, FinishReason::MaxTokens);
    }

    /// A `GenEvent::Finished` carries the finish reason and terminal token
    /// counts as its payload (the type is not `PartialEq`, so match on it).
    #[test]
    fn gen_event_finished_carries_reason_and_counts() {
        let event = GenEvent::Finished {
            reason: FinishReason::Stop,
            stop_reason: Some("</s>".to_string()),
            prompt_tokens: 4,
            completion_tokens: 9,
            images: 0,
            kv_transfer_params: None,
        };

        match event {
            GenEvent::Finished {
                reason,
                stop_reason,
                prompt_tokens,
                completion_tokens,
                images,
                kv_transfer_params,
            } => {
                assert_eq!(reason, FinishReason::Stop);
                assert_eq!(stop_reason, Some("</s>".to_string()));
                assert_eq!(prompt_tokens, 4);
                assert_eq!(completion_tokens, 9);
                assert_eq!(images, 0);
                assert_eq!(kv_transfer_params, None);
            }
            other => panic!("expected Finished, got {other:?}"),
        }
    }
}
