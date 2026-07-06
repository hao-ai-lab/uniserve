use serde_json::{Value, json};
use uniserve_engine_client::{GenEvent, NativeFinishReason};
use uniserve_text::tokenizer::DynTokenizer;

pub fn is_terminal(event: &GenEvent) -> bool {
    matches!(
        event,
        GenEvent::Finished { .. } | GenEvent::Rejected { .. } | GenEvent::Error { .. }
    )
}

pub struct Detok {
    tok: DynTokenizer,
    ids: Vec<u32>,
    prev: String,
}

impl Detok {
    pub fn new(tok: DynTokenizer) -> Self {
        Self {
            tok,
            ids: Vec::new(),
            prev: String::new(),
        }
    }

    fn push(&mut self, id: u32) -> String {
        self.ids.push(id);
        let full = self.tok.decode(&self.ids, true).unwrap_or_default();
        let delta = full
            .strip_prefix(&self.prev)
            .map(str::to_string)
            .unwrap_or_else(|| full.clone());
        self.prev = full;
        delta
    }
}

pub fn event_json(event: &GenEvent, detok: &mut Detok) -> Value {
    match event {
        GenEvent::Scheduled {
            queued_at,
            scheduled_at,
        } => json!({"type":"scheduled","queued_at":queued_at,"scheduled_at":scheduled_at}),
        GenEvent::TextToken { id, .. } => {
            json!({"type":"text","id":id,"text":detok.push(*id)})
        }
        GenEvent::TokenLogprobs { id, top } => json!({"type":"logprobs","id":id,"top":top}),
        GenEvent::ImageBegin {
            image_id,
            height,
            width,
            steps,
        } => {
            json!({"type":"image_begin","image_id":image_id,"height":height,"width":width,"steps":steps})
        }
        GenEvent::ImageStep { image_id, step } => {
            json!({"type":"image_step","image_id":image_id,"step":step})
        }
        GenEvent::ImageDone {
            image_id,
            height,
            width,
            bytes,
            sha256,
            pixels_png_b64,
        } => json!({
            "type":"image_done",
            "image_id":image_id,
            "height":height,
            "width":width,
            "bytes":bytes,
            "sha256":sha256,
            "pixels_png_b64":pixels_png_b64
        }),
        GenEvent::Finished {
            prompt_tokens,
            completion_tokens,
            images,
            stop_reason,
            ..
        } => json!({"type":"finished","reason":finish_reason(event),
                   "prompt_tokens":prompt_tokens,"completion_tokens":completion_tokens,"images":images,
                   "stop_reason": stop_reason}),
        GenEvent::Rejected { message } => json!({"type":"rejected","message":message}),
        GenEvent::Error { message } => json!({"type":"error","message":message}),
    }
}

fn finish_reason(event: &GenEvent) -> &'static str {
    match event {
        GenEvent::Finished { reason, .. } => match reason {
            NativeFinishReason::Eos => "eos",
            NativeFinishReason::MaxTokens => "max_tokens",
            NativeFinishReason::Stop => "stop",
            NativeFinishReason::ImageDone => "image_done",
            NativeFinishReason::Cancelled => "cancelled",
            NativeFinishReason::Aborted => "aborted",
            NativeFinishReason::Error => "error",
        },
        _ => "stop",
    }
}
