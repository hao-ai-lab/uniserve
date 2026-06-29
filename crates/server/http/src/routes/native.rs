//! UniServe-native image / interleaved-generation surface.

use std::convert::Infallible;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::stream;
use serde_json::{Value, json};

use uniserve_native_api::events::{Detok, event_json, is_terminal};
use uniserve_native_api::{NativeGenerateBody, NativeImageBody, NativeRequestBuilder};
use uniserve_server_app::AppState;

use crate::error::ApiError;

pub(crate) async fn generate(
    State(st): State<Arc<AppState>>,
    Json(body): Json<NativeGenerateBody>,
) -> Response {
    let tokenizer = st.chat().text().tokenizer();
    let request =
        match NativeRequestBuilder::new(Arc::clone(&tokenizer), st.native_profile()).build(&body) {
            Ok(request) => request,
            Err(error) => {
                return ApiError::invalid_request(error.message().to_string(), None).into_response();
            }
        };

    let native_stream = match st
        .chat()
        .uniserve_engine_client()
        .generate_native(request)
        .await
    {
        Ok(stream) => stream,
        Err(error) => {
            return ApiError::server_error(error.to_string()).into_response();
        }
    };

    let sse = stream::unfold(
        (
            native_stream,
            Detok::new(tokenizer),
            NativeReasoningTagFilter::default(),
            false,
        ),
        |(mut native_stream, mut detok, mut reasoning_filter, done)| async move {
            if done {
                return None;
            }
            let ev = native_stream.next().await?;
            let terminal = is_terminal(&ev);
            let mut payload = event_json(&ev, &mut detok);
            if payload.get("type").and_then(Value::as_str) == Some("text")
                && let Some(text) = payload.get("text").and_then(Value::as_str)
            {
                payload["text"] = Value::String(reasoning_filter.push(text));
            }
            let event = Event::default().data(payload.to_string());
            Some((
                Ok::<Event, Infallible>(event),
                (native_stream, detok, reasoning_filter, terminal),
            ))
        },
    );

    Sse::new(sse)
        .keep_alive(KeepAlive::default())
        .into_response()
}

#[derive(Debug, Default)]
struct NativeReasoningTagFilter {
    pending: String,
    hidden: bool,
}

impl NativeReasoningTagFilter {
    fn push(&mut self, text: &str) -> String {
        self.pending.push_str(text);
        let mut visible = String::new();

        loop {
            if self.hidden {
                if let Some(end) = self.pending.find("</think>") {
                    self.pending.drain(..end + "</think>".len());
                    self.hidden = false;
                    continue;
                }

                let keep = suffix_prefix_len(&self.pending, "</think>");
                let drop_to = self.pending.len().saturating_sub(keep);
                self.pending.drain(..drop_to);
                break;
            } else if let Some((start, marker)) = first_marker(&self.pending) {
                visible.push_str(&self.pending[..start]);
                self.pending.drain(..start + marker.len());
                if marker == "<think>" {
                    self.hidden = true;
                }
                continue;
            }

            let keep = suffix_prefix_len_any(&self.pending, VISIBLE_MARKERS);
            let emit_to = self.pending.len().saturating_sub(keep);
            visible.push_str(&self.pending[..emit_to]);
            self.pending.drain(..emit_to);
            break;
        }

        visible
    }
}

const VISIBLE_MARKERS: &[&str] = &["<think>", "</think>", "<answer>", "</answer>"];

fn first_marker(text: &str) -> Option<(usize, &'static str)> {
    VISIBLE_MARKERS
        .iter()
        .copied()
        .filter_map(|marker| text.find(marker).map(|idx| (idx, marker)))
        .min_by_key(|(idx, _)| *idx)
}

fn suffix_prefix_len_any(text: &str, markers: &[&str]) -> usize {
    markers
        .iter()
        .map(|marker| suffix_prefix_len(text, marker))
        .max()
        .unwrap_or(0)
}

fn suffix_prefix_len(text: &str, marker: &str) -> usize {
    let max = text.len().min(marker.len().saturating_sub(1));
    for len in (1..=max).rev() {
        if text.is_char_boundary(text.len() - len)
            && marker.is_char_boundary(len)
            && text[text.len() - len..].eq(&marker[..len])
        {
            return len;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::NativeReasoningTagFilter;

    #[test]
    fn native_reasoning_tag_filter_strips_split_think_delimiters() {
        let mut filter = NativeReasoningTagFilter::default();
        let mut out = String::new();

        for chunk in ["Here ", "<thi", "nk>secret", "</thi", "nk> guide"] {
            out.push_str(&filter.push(chunk));
        }

        assert_eq!(out, "Here  guide");
    }

    #[test]
    fn native_reasoning_tag_filter_preserves_text_around_multiple_blocks() {
        let mut filter = NativeReasoningTagFilter::default();
        let chunks = [
            "Sonoma <think>plan</think>",
            " Sequoia",
            " <think>more</think>Tahoe",
            " and Golden Gate.",
        ];
        let out = chunks
            .into_iter()
            .map(|chunk| filter.push(chunk))
            .collect::<String>();

        assert_eq!(out, "Sonoma  Sequoia Tahoe and Golden Gate.");
    }

    #[test]
    fn native_reasoning_tag_filter_hides_unclosed_reasoning_until_close() {
        let mut filter = NativeReasoningTagFilter::default();

        assert_eq!(filter.push("<think>draft plan"), "");
        assert_eq!(filter.push(" and more</thi"), "");
        assert_eq!(filter.push("nk>Final answer."), "Final answer.");
    }

    #[test]
    fn native_reasoning_tag_filter_strips_answer_wrappers() {
        let mut filter = NativeReasoningTagFilter::default();
        let chunks = ["<ans", "wer>Sonoma guide", "</ans", "wer>"];
        let out = chunks
            .into_iter()
            .map(|chunk| filter.push(chunk))
            .collect::<String>();

        assert_eq!(out, "Sonoma guide");
    }
}

/// OpenAI-style pure text-to-image generation.
pub(crate) async fn images_generations(
    State(st): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Response {
    let prompt = body
        .get("prompt")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .to_string();
    let parsed_size = body
        .get("size")
        .and_then(|value| value.as_str())
        .and_then(|size| {
            size.split_once('x')
                .and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?)))
        });
    let image = NativeImageBody {
        width: parsed_size.map(|(w, _)| w),
        height: parsed_size.map(|(_, h)| h),
        steps: body
            .get("steps")
            .and_then(|value| value.as_u64())
            .map(|value| value as u16),
        seed: body.get("seed").and_then(|value| value.as_u64()),
        negative_prompt: body
            .get("negative_prompt")
            .and_then(|value| value.as_str())
            .map(str::to_owned),
        ..Default::default()
    };
    let native = NativeGenerateBody {
        prompt,
        mode: Some("image".into()),
        image: Some(image),
        ..Default::default()
    };
    let tokenizer = st.chat().text().tokenizer();
    let request = match NativeRequestBuilder::new(tokenizer, st.native_profile()).build(&native) {
        Ok(request) => request,
        Err(error) => {
            return ApiError::invalid_request(error.message().to_string(), None).into_response();
        }
    };

    let mut native_stream = match st
        .chat()
        .uniserve_engine_client()
        .generate_native(request)
        .await
    {
        Ok(stream) => stream,
        Err(error) => {
            return ApiError::server_error(error.to_string()).into_response();
        }
    };

    let mut data = Vec::new();
    while let Some(ev) = native_stream.next().await {
        match ev {
            uniserve_engine_client::GenEvent::ImageDone {
                height,
                width,
                bytes,
                sha256,
                pixels_png_b64,
                ..
            } => data.push(json!({
                "b64_json": pixels_png_b64,
                "height": height,
                "width": width,
                "bytes": bytes,
                "sha256": sha256
            })),
            ref event if is_terminal(event) => break,
            _ => {}
        }
    }

    Json(json!({"created": 0, "data": data})).into_response()
}
