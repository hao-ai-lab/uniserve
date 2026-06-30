//! Northbound engine contract: submit a [`GenerateRequest`] and receive a stream
//! of [`GenEvent`]s over a per-request channel. Supports text and image events.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
use uniserve_core::{GenMode, ImageParams, RequestId, SamplingParams};

pub use uniserve_core::{GenMode as Mode, ImageParams as ImgParams, SamplingParams as SampParams};

/// A staged multimodal input item: an image (or audio/video) referenced by content
/// hash that occupies `num_tokens` positions in the AR sequence once its encoder
/// embeddings are spliced in. The bytes go in over HTTP and to the worker as
/// input; the embedding never returns to the host — only the content `hash` and
/// a worker-side `encoder_handle` are retained on the host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MmItem {
    /// Content hash (encoder-cache key).
    pub hash: u64,
    /// Start position of this item's span within the flattened `prompt_ids`.
    pub position: u32,
    /// Number of AR positions the encoder output occupies.
    pub num_tokens: u32,
    /// Input-image bytes (base64), shipped to the worker encode ops. Empty for
    /// items already resolved to an encoder handle on the host.
    pub b64: String,
}

/// One part of an interleaved prompt, as the chat layer produces it.
#[derive(Debug, Clone)]
pub enum PromptPart {
    Text(String),
    Image { hash: u64, num_tokens: u32 },
}

/// Interleaved prompt: text plus staged input-image references.
#[derive(Debug, Clone, Default)]
pub struct Prompt {
    pub text: String,
    pub parts: Vec<PromptPart>,
}

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
    Error,
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
    /// Top-k logprob alternatives for the just-emitted token, when requested.
    TokenLogprobs {
        id: u32,
        top: Vec<(u32, f32)>,
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

/// A structured-output constraint attached to a request. Compiled engine-side
/// into a per-step token mask via the `allowed_tokens` descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrammarSpec {
    /// The output must be exactly one of these token-id sequences (the
    /// guided-choice form; the frontend tokenized the choice strings).
    Choice(Vec<Vec<u32>>),
}

/// A submitted request. The prompt arrives already tokenized by the server's
/// ingest stage; the scheduler and worker speak token ids only.
pub struct GenerateRequest {
    pub request_id: RequestId,
    pub prompt_ids: Vec<u32>,
    pub neg_prompt_ids: Vec<u32>, // CFG text-unconditional prompt (may be empty)
    pub sampling: SamplingParams,
    pub image: ImageParams,
    pub mode: GenMode,
    pub max_tokens: usize,
    /// Stop strings (matched on the detokenized suffix) and explicit stop token
    /// ids — the request terminates with `FinishReason::Stop` on a hit.
    pub stop_strings: Vec<String>,
    pub stop_token_ids: Vec<u32>,
    /// Priority for the `Priority` scheduling policy (lower = sooner).
    pub priority: i32,
    /// Optional LoRA adapter id applied to this request's ops.
    pub lora_id: Option<u32>,
    /// Staged multimodal input items (encoded before prefill).
    pub mm_items: Vec<MmItem>,
    /// Structured-output constraint (compiled engine-side).
    pub grammar: Option<GrammarSpec>,
    /// When `true`, the scheduler will not read this request's prompt prefix
    /// from the prefix cache (newly computed blocks may still populate it).
    /// Surfaced by the gRPC `bypass_prefix_cache` flag and the wire
    /// `skip_reading_prefix_cache` sampling field.
    pub skip_reading_prefix_cache: bool,
    pub event_tx: EventTx,
}

impl GenerateRequest {
    /// A minimal request for tests / internal construction (no stops, default priority).
    pub fn new(
        request_id: RequestId,
        prompt_ids: Vec<u32>,
        sampling: SamplingParams,
        image: ImageParams,
        mode: GenMode,
        max_tokens: usize,
        event_tx: EventTx,
    ) -> Self {
        Self {
            request_id,
            prompt_ids,
            neg_prompt_ids: Vec::new(),
            sampling,
            image,
            mode,
            max_tokens,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            lora_id: None,
            mm_items: Vec::new(),
            grammar: None,
            skip_reading_prefix_cache: false,
            event_tx,
        }
    }
}

/// Collective RPC reply channel payload.
pub type CollectiveRpcReply =
    std::sync::mpsc::Sender<Result<Vec<(u32, bool, Option<String>)>, String>>;

/// Command sent from a frontend handler to the scheduler thread.
pub enum Command {
    Submit(Box<GenerateRequest>),
    /// Client-side cancel → `FinishReason::Cancelled`.
    Cancel(RequestId),
    /// Server-side abort → `FinishReason::Aborted`.
    Abort(RequestId),
    /// Clear the prefix cache (the `/reset_prefix_cache` endpoint).
    ResetPrefixCache,
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

    pub fn submit(&self, req: GenerateRequest) -> Result<(), String> {
        self.send(Command::Submit(Box::new(req)))
            .map_err(|e| e.to_string())
    }
    pub fn cancel(&self, id: RequestId) {
        let _ = self.send(Command::Cancel(id));
    }
    /// Server-side abort, distinct from a client cancel.
    pub fn abort(&self, id: RequestId) {
        let _ = self.send(Command::Abort(id));
    }
    /// Clear the prefix cache.
    pub fn reset_prefix_cache(&self) {
        let _ = self.send(Command::ResetPrefixCache);
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
    use uniserve_core::{GenMode, ImageParams, RequestId, SamplingParams};

    use super::*;

    fn test_request(request_id: u64) -> GenerateRequest {
        let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
        GenerateRequest::new(
            RequestId(request_id),
            vec![1, 2, 3],
            SamplingParams::default(),
            ImageParams::default(),
            GenMode::Text,
            32,
            event_tx,
        )
    }

 /// The convenience constructor fills the request-shaping defaults: no stops,
 /// zero priority, no LoRA, and empty multimodal/grammar fields.
    #[test]
    fn generate_request_new_applies_minimal_defaults() {
        let request = test_request(7);

        assert_eq!(request.request_id, RequestId(7));
        assert_eq!(request.prompt_ids, vec![1, 2, 3]);
        assert_eq!(request.max_tokens, 32);
        assert_eq!(request.mode, GenMode::Text);
        assert!(request.neg_prompt_ids.is_empty());
        assert!(request.stop_strings.is_empty());
        assert!(request.stop_token_ids.is_empty());
        assert_eq!(request.priority, 0);
        assert_eq!(request.lora_id, None);
        assert!(request.mm_items.is_empty());
        assert_eq!(request.grammar, None);
        assert!(!request.skip_reading_prefix_cache);
    }

 /// `submit` enqueues a `Command::Submit` carrying the request, and the
 /// scheduler side receives exactly that request id.
    #[test]
    fn submit_enqueues_submit_command_with_request() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = EngineHandle::new(tx);

        handle.submit(test_request(11)).unwrap();

        match rx.recv().unwrap() {
            Command::Submit(request) => assert_eq!(request.request_id, RequestId(11)),
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

        handle.reset_prefix_cache();
        handle.set_sleeping(true);
        assert_eq!(wakes.load(Ordering::SeqCst), 2);

 // Closing the channel means there is no scheduler to wake.
        drop(rx);
        handle.reset_encoder_cache();
        assert_eq!(wakes.load(Ordering::SeqCst), 2);
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
                    .send(Ok(vec![(0, true, None), (1, false, Some("oops".to_string()))]))
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
        };

        match event {
            GenEvent::Finished {
                reason,
                stop_reason,
                prompt_tokens,
                completion_tokens,
                images,
            } => {
                assert_eq!(reason, FinishReason::Stop);
                assert_eq!(stop_reason, Some("</s>".to_string()));
                assert_eq!(prompt_tokens, 4);
                assert_eq!(completion_tokens, 9);
                assert_eq!(images, 0);
            }
            other => panic!("expected Finished, got {other:?}"),
        }
    }
}
