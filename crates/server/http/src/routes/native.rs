//! UniServe-native generation surface.

use std::convert::Infallible;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::{StreamExt as _, stream};
use serde_json::{Value, json};

use uniserve_engine_client::GenerationConstraint;
use uniserve_native_api::events::Detok;
use uniserve_native_api::{
    NativeDelimitedText, NativeGenerateBody, NativeImageBody, NativeOutputFilter,
    NativeRequestBuilder,
};
use uniserve_reasoning_parser::DelimitedReasoningParser;
use uniserve_server_app::AppState;
use uniserve_serving::{FinishStatus, ServeError, ServeEvent};
use uniserve_text::tokenizer::DynTokenizer;

use crate::error::ApiError;

pub(crate) async fn generate(
    State(st): State<Arc<AppState>>,
    Json(body): Json<NativeGenerateBody>,
) -> Response {
    let tokenizer = st.chat().text().tokenizer();
    let request = match NativeRequestBuilder::new(Arc::clone(&tokenizer), st.native_profile())
        .build(&body)
    {
        Ok(request) => request,
        Err(error) => {
            return ApiError::invalid_request(error.message().to_string(), None).into_response();
        }
    };

    let prompt_ids = request.prompt_ids.clone();
    let text_filter = match NativeTextOutputFilter::new(
        st.native_profile().output_filter.clone(),
        Arc::clone(&tokenizer),
        &prompt_ids,
    ) {
        Ok(filter) => filter,
        Err(error) => {
            return ApiError::server_error(error.to_string()).into_response();
        }
    };

    let serve_stream = match st
        .runtime()
        .serve_native("native-generate".to_string(), request)
        .await
    {
        Ok(stream) => stream,
        Err(error) => {
            return serve_error_to_api(error).into_response();
        }
    };

    let sse = stream::unfold(
        (
            serve_stream,
            Detok::new(tokenizer),
            text_filter,
            NativeTerminalUsage::default(),
            false,
            false,
        ),
        |(mut serve_stream, mut detok, mut text_filter, mut usage, done, flush_after_terminal)| async move {
            if done {
                return None;
            }
            if flush_after_terminal {
                let event = Event::default().comment("terminal");
                return Some((
                    Ok::<Event, Infallible>(event),
                    (serve_stream, detok, text_filter, usage, true, false),
                ));
            }
            loop {
                let ev = serve_stream.next().await?;
                match ev {
                    Ok(ev) => {
                        let terminal = serve_event_is_terminal(&ev);
                        let Some(payload) =
                            native_event_json(ev, &mut detok, &mut text_filter, &mut usage)
                        else {
                            continue;
                        };
                        let event = Event::default().data(payload.to_string());
                        return Some((
                            Ok::<Event, Infallible>(event),
                            (serve_stream, detok, text_filter, usage, false, terminal),
                        ));
                    }
                    Err(error) => {
                        let payload = json!({"type":"error","message":serve_error_message(error)});
                        let event = Event::default().data(payload.to_string());
                        return Some((
                            Ok::<Event, Infallible>(event),
                            (serve_stream, detok, text_filter, usage, false, true),
                        ));
                    }
                }
            }
        },
    );

    Sse::new(sse)
        .keep_alive(KeepAlive::default())
        .into_response()
}

#[derive(Debug, Clone, Copy, Default)]
struct NativeTerminalUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    images: u32,
}

fn native_event_json(
    event: ServeEvent,
    detok: &mut Detok,
    text_filter: &mut NativeTextOutputFilter,
    usage: &mut NativeTerminalUsage,
) -> Option<Value> {
    match event {
        ServeEvent::Scheduled {
            queued_at,
            scheduled_at,
            ..
        } => Some(json!({
            "type":"scheduled",
            "queued_at":queued_at.unwrap_or_default(),
            "scheduled_at":scheduled_at.unwrap_or_default()
        })),
        ServeEvent::TextDelta {
            text, token_ids, ..
        } => {
            let id = token_ids.first().copied().unwrap_or_default();
            let mut delta = text;
            for token_id in token_ids {
                delta.push_str(&detok.push(token_id));
            }
            Some(json!({"type":"text","id":id,"text":text_filter.push(&delta)}))
        }
        ServeEvent::TokenLogprobs { token_id, top, .. } => {
            Some(json!({"type":"logprobs","id":token_id,"top":top}))
        }
        ServeEvent::ImageBegin {
            image_id,
            height,
            width,
            steps,
            ..
        } => Some(json!({
            "type":"image_begin",
            "image_id":parse_image_id(&image_id),
            "height":height.unwrap_or_default(),
            "width":width.unwrap_or_default(),
            "steps":steps.unwrap_or_default()
        })),
        ServeEvent::ImageStep { image_id, step, .. } => {
            Some(json!({"type":"image_step","image_id":parse_image_id(&image_id),"step":step}))
        }
        ServeEvent::ImageDone {
            image_id,
            height,
            width,
            bytes,
            sha256,
            pixels_png_b64,
            ..
        } => Some(json!({
            "type":"image_done",
            "image_id":parse_image_id(&image_id),
            "height":height.unwrap_or_default(),
            "width":width.unwrap_or_default(),
            "bytes":bytes.unwrap_or_default(),
            "sha256":sha256.unwrap_or_default(),
            "pixels_png_b64":pixels_png_b64.unwrap_or_default()
        })),
        ServeEvent::Usage {
            prompt_tokens,
            visible_output_tokens,
            image_count,
            ..
        } => {
            usage.prompt_tokens = prompt_tokens;
            usage.completion_tokens = visible_output_tokens;
            usage.images = image_count;
            None
        }
        ServeEvent::Finished {
            reason,
            finish_detail,
            ..
        } => Some(json!({
            "type":"finished",
            "reason":finish_detail.unwrap_or_else(|| native_finish_status_as_str(&reason).to_string()),
            "prompt_tokens":usage.prompt_tokens,
            "completion_tokens":usage.completion_tokens,
            "images":usage.images,
            "stop_reason":finish_status_stop_reason(&reason)
        })),
        ServeEvent::Rejected { message, .. } => Some(json!({"type":"rejected","message":message})),
        ServeEvent::Failed { message, .. } => Some(json!({"type":"error","message":message})),
        _ => None,
    }
}

fn serve_event_is_terminal(event: &ServeEvent) -> bool {
    matches!(
        event,
        ServeEvent::Finished { .. } | ServeEvent::Rejected { .. } | ServeEvent::Failed { .. }
    )
}

fn parse_image_id(image_id: &str) -> u32 {
    image_id.parse().unwrap_or_default()
}

fn native_finish_status_as_str(status: &FinishStatus) -> &'static str {
    match status {
        FinishStatus::Stop { .. } => "stop",
        FinishStatus::Length => "max_tokens",
        FinishStatus::Abort => "aborted",
        FinishStatus::Error => "error",
        FinishStatus::Repetition => "stop",
    }
}

fn finish_status_stop_reason(status: &FinishStatus) -> Option<Value> {
    match status {
        FinishStatus::Stop { stop_reason } => stop_reason.clone(),
        FinishStatus::Length
        | FinishStatus::Abort
        | FinishStatus::Error
        | FinishStatus::Repetition => None,
    }
}

fn serve_error_to_api(error: ServeError) -> ApiError {
    ApiError::server_error(serve_error_message(error))
}

fn serve_error_message(error: ServeError) -> String {
    match error {
        ServeError::UnsupportedRuntimeExtension { key, .. } => {
            format!("unsupported runtime extension `{key}`")
        }
        ServeError::UnsupportedOutputCount { requested, .. } => {
            format!("only one output is supported, got {requested}")
        }
        ServeError::Text(error) => format!("text runtime error: {error}"),
        ServeError::Chat(error) => format!("chat runtime error: {error}"),
        ServeError::Engine(message) => format!("engine runtime error: {message}"),
    }
}

pub(crate) struct NativeTextOutputFilter {
    reasoning: Option<DelimitedReasoningParser>,
    visible_wrappers: NativeVisibleWrapperFilter,
}

impl NativeTextOutputFilter {
    pub(crate) fn new(
        spec: NativeOutputFilter,
        tokenizer: DynTokenizer,
        prompt_token_ids: &[u32],
    ) -> uniserve_reasoning_parser::Result<Self> {
        let reasoning = if let Some(reasoning) = spec.reasoning.clone() {
            let mut parser =
                DelimitedReasoningParser::new(tokenizer, reasoning.start, reasoning.end, false)?;
            parser.initialize(prompt_token_ids);
            Some(parser)
        } else {
            None
        };
        Ok(Self {
            reasoning,
            visible_wrappers: NativeVisibleWrapperFilter::new(spec.visible_wrappers),
        })
    }

    pub(crate) fn push(&mut self, text: &str) -> String {
        let content = if let Some(reasoning) = self.reasoning.as_mut() {
            reasoning.push(text).content.unwrap_or_default()
        } else {
            text.to_string()
        };
        self.visible_wrappers.push(&content)
    }
}

struct NativeVisibleWrapperFilter {
    wrappers: Vec<NativeDelimitedText>,
    pending: String,
}

impl NativeVisibleWrapperFilter {
    fn new(wrappers: Vec<NativeDelimitedText>) -> Self {
        Self {
            wrappers,
            pending: String::new(),
        }
    }

    fn push(&mut self, text: &str) -> String {
        if self.wrappers.is_empty() {
            return text.to_string();
        }
        self.pending.push_str(text);
        let mut visible = String::new();

        loop {
            if let Some((start, marker_len)) = self.first_marker() {
                visible.push_str(&self.pending[..start]);
                self.pending.drain(..start + marker_len);
                continue;
            }

            let keep = trailing_marker_prefix_len_any(&self.pending, &self.markers());
            let emit_to = self.pending.len().saturating_sub(keep);
            visible.push_str(&self.pending[..emit_to]);
            self.pending.drain(..emit_to);
            break;
        }

        visible
    }

    fn markers(&self) -> Vec<&str> {
        let mut markers = Vec::new();
        for wrapper in &self.wrappers {
            markers.push(wrapper.start.as_str());
            markers.push(wrapper.end.as_str());
        }
        markers
    }

    fn first_marker(&self) -> Option<(usize, usize)> {
        let mut matches = Vec::new();
        for wrapper in &self.wrappers {
            matches.push(wrapper.start.as_str());
            matches.push(wrapper.end.as_str());
        }
        matches
            .into_iter()
            .filter_map(|marker| self.pending.find(marker).map(|idx| (idx, marker.len())))
            .min_by_key(|(idx, _)| *idx)
    }
}

fn trailing_marker_prefix_len_any(text: &str, markers: &[&str]) -> usize {
    markers
        .iter()
        .map(|marker| trailing_marker_prefix_len(text, marker))
        .max()
        .unwrap_or(0)
}

fn trailing_marker_prefix_len(text: &str, marker: &str) -> usize {
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
    use std::sync::Arc;

    use uniserve_native_api::{NativeDelimitedText, NativeOutputFilter};
    use uniserve_text::tokenizer::{DynTokenizer, Tokenizer};

    use super::NativeTextOutputFilter;

    #[derive(Debug)]
    struct FilterTokenizer;

    impl Tokenizer for FilterTokenizer {
        fn encode(
            &self,
            text: &str,
            _add_special_tokens: bool,
        ) -> uniserve_text::tokenizer::Result<Vec<u32>> {
            Ok(text.bytes().map(u32::from).collect())
        }

        fn decode(
            &self,
            token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> uniserve_text::tokenizer::Result<String> {
            Ok(
                String::from_utf8_lossy(&token_ids.iter().map(|id| *id as u8).collect::<Vec<_>>())
                    .into_owned(),
            )
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            match token {
                "<think>" => Some(1),
                "</think>" => Some(2),
                _ => None,
            }
        }
    }

    fn tokenizer() -> DynTokenizer {
        Arc::new(FilterTokenizer)
    }

    fn filter() -> NativeTextOutputFilter {
        NativeTextOutputFilter::new(
            NativeOutputFilter {
                reasoning: Some(NativeDelimitedText {
                    start: "<think>".into(),
                    end: "</think>".into(),
                }),
                visible_wrappers: vec![NativeDelimitedText {
                    start: "<answer>".into(),
                    end: "</answer>".into(),
                }],
            },
            tokenizer(),
            &[],
        )
        .unwrap()
    }

    #[test]
    fn native_reasoning_tag_filter_strips_split_think_delimiters() {
        let mut filter = filter();
        let mut out = String::new();

        for chunk in ["Here ", "<thi", "nk>secret", "</thi", "nk> guide"] {
            out.push_str(&filter.push(chunk));
        }

        assert_eq!(out, "Here  guide");
    }

    #[test]
    fn native_reasoning_tag_filter_preserves_text_around_multiple_blocks() {
        let mut filter = filter();
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
        let mut filter = filter();

        assert_eq!(filter.push("<think>draft plan"), "");
        assert_eq!(filter.push(" and more</thi"), "");
        assert_eq!(filter.push("nk>Final answer."), "Final answer.");
    }

    #[test]
    fn native_reasoning_tag_filter_strips_answer_wrappers() {
        let mut filter = filter();
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
        constraint: Some(GenerationConstraint::GenOnly),
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

    let mut serve_stream = match st
        .runtime()
        .serve_native("images-generations".to_string(), request)
        .await
    {
        Ok(stream) => stream,
        Err(error) => {
            return serve_error_to_api(error).into_response();
        }
    };

    let mut data = Vec::new();
    while let Some(ev) = serve_stream.next().await {
        match ev {
            Ok(ServeEvent::ImageDone {
                height,
                width,
                bytes,
                sha256,
                pixels_png_b64,
                ..
            }) => data.push(json!({
                "b64_json": pixels_png_b64.unwrap_or_default(),
                "height": height.unwrap_or_default(),
                "width": width.unwrap_or_default(),
                "bytes": bytes.unwrap_or_default(),
                "sha256": sha256.unwrap_or_default()
            })),
            Ok(event) if serve_event_is_terminal(&event) => break,
            Err(error) => return serve_error_to_api(error).into_response(),
            _ => {}
        }
    }

    Json(json!({"created": 0, "data": data})).into_response()
}
