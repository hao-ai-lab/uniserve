use tokio::sync::mpsc;

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

