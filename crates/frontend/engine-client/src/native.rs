use futures::StreamExt as _;
use tokio::sync::mpsc;

use crate::client::EngineCoreOutputStream;
use crate::protocol::native::{NativeRequestExt, WireMmItem};
use crate::protocol::{EngineCoreRequest, EngineCoreSamplingParams};

pub use uniserve_core::{GenMode, ImageParams, SamplingParams as EngineSamplingParams};
pub use uniserve_engine_api::{FinishReason as NativeFinishReason, GenEvent, MmItem};

/// Reverse adaptation of one wire output into the typed native [`GenEvent`]s a
/// stream consumer expects. Re-exported from [`uniserve_engine_wire`] so the
/// client and the headless engine share a single definition.
pub use uniserve_engine_wire::translate::wire_output_to_gen_events;

/// Inputs for a native, image/interleave-capable generate request.

/// This bypasses the text-only engine request DTO so callers can express pure
/// text-to-image, interleaved text+image, and image-understanding flows.
pub struct NativeGenerateRequest {
    pub prompt_ids: Vec<u32>,
    pub neg_prompt_ids: Vec<u32>,
    pub sampling: EngineSamplingParams,
    pub image: ImageParams,
    pub mode: GenMode,
    pub max_tokens: usize,
    pub mm_items: Vec<MmItem>,
    pub stop_token_ids: Vec<u32>,
}

pub(crate) fn native_request_to_wire(
    req: NativeGenerateRequest,
    request_id: String,
) -> EngineCoreRequest {
    let sampling = to_wire_sampling(&req.sampling, req.max_tokens, &req.stop_token_ids);
    EngineCoreRequest {
        request_id,
        prompt_token_ids: Some(req.prompt_ids),
        sampling_params: Some(sampling),
        arrival_time: now_secs(),
        native: Some(NativeRequestExt {
            mode: req.mode,
            image: req.image,
            neg_prompt_ids: req.neg_prompt_ids,
            mm_items: req
                .mm_items
                .into_iter()
                .map(|m| WireMmItem {
                    hash: m.hash,
                    position: m.position,
                    num_tokens: m.num_tokens,
                    b64: m.b64,
                })
                .collect(),
        }),
        ..Default::default()
    }
}

pub(crate) fn native_stream_from_wire_stream(
    mut stream: EngineCoreOutputStream,
) -> NativeEventStream {
    let (tx, rx) = mpsc::unbounded_channel::<GenEvent>();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = tx.closed() => return,
                item = stream.next() => match item {
                    Some(Ok(out)) => {
                        for ev in wire_output_to_gen_events(&out.output) {
                            if tx.send(ev).is_err() {
                                return;
                            }
                        }
                    }
                    Some(Err(error)) => {
                        let _ = tx.send(GenEvent::Error { message: error.to_string() });
                        return;
                    }
                    None => return,
                },
            }
        }
    });
    NativeEventStream::new(rx)
}

fn to_wire_sampling(
    s: &uniserve_core::SamplingParams,
    max_tokens: usize,
    stop_token_ids: &[u32],
) -> EngineCoreSamplingParams {
    EngineCoreSamplingParams {
        temperature: s.temperature,
        top_p: s.top_p,
        top_k: s.top_k,
        seed: s.seed.map(|x| x as i64),
        max_tokens: max_tokens as u32,
        min_tokens: s.min_tokens as u32,
        ignore_eos: s.ignore_eos,
        logprobs: (s.n_logprobs > 0).then_some(s.n_logprobs as i32),
        prompt_logprobs: None,
        min_p: s.min_p,
        frequency_penalty: s.frequency_penalty,
        presence_penalty: s.presence_penalty,
        repetition_penalty: s.repetition_penalty,
        stop_token_ids: stop_token_ids.to_vec(),
        eos_token_id: None,
        all_stop_token_ids: stop_token_ids.iter().copied().collect(),
        logit_bias: (!s.logit_bias.is_empty()).then(|| s.logit_bias.iter().copied().collect()),
        allowed_token_ids: s.allowed_token_ids.clone(),
        bad_words_token_ids: (!s.bad_words_ids.is_empty()).then(|| s.bad_words_ids.clone()),
        choice_token_ids: None,
        structured_outputs: None,
        logprob_token_ids: None,
        skip_reading_prefix_cache: None,
        extra_args: None,
    }
}

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or_default()
}

/// A stream of typed text+image [`GenEvent`]s for one native request.
pub struct NativeEventStream {
    rx: mpsc::UnboundedReceiver<GenEvent>,
    cancel: Option<Box<dyn FnOnce() + Send + 'static>>,
    finished: bool,
}

impl NativeEventStream {
    pub fn new(rx: mpsc::UnboundedReceiver<GenEvent>) -> Self {
        Self {
            rx,
            cancel: None,
            finished: false,
        }
    }

    pub fn with_cancel(
        rx: mpsc::UnboundedReceiver<GenEvent>,
        cancel: impl FnOnce() + Send + 'static,
    ) -> Self {
        Self {
            rx,
            cancel: Some(Box::new(cancel)),
            finished: false,
        }
    }

    /// Await the next event, or `None` once the stream is exhausted.
    pub async fn next(&mut self) -> Option<GenEvent> {
        let ev = self.rx.recv().await?;
        if matches!(
            ev,
            GenEvent::Finished { .. } | GenEvent::Rejected { .. } | GenEvent::Error { .. }
        ) {
            self.finished = true;
            self.cancel = None;
        }
        Some(ev)
    }
}

impl Drop for NativeEventStream {
    fn drop(&mut self) {
        if !self.finished
            && let Some(cancel) = self.cancel.take()
        {
            cancel();
        }
    }
}
