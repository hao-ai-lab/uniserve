//! Wire DTO ↔ engine type translation, shared by the in-process client adapter
//! and the headless engine process.
//!
//! Northbound, a per-request `GenEvent` stream is adapted into
//! [`EngineCoreOutput`]s: text tokens (with the sampled/top-k logprob fusion),
//! `Scheduled` timestamps as `EngineCoreEvent`s, and — for native
//! image/interleave requests — typed image events and finish statistics on the
//! UniServe `native` extension. Southbound, an [`EngineCoreRequest`] (plus its
//! optional `native` extension) becomes an `uniserve_engine_api::GenerateRequest`.

use crate::logprobs::{Logprobs, MaybeWireLogprobs, PositionLogprobs, TokenLogprob};
use crate::native::{NativeFinishExt, NativeOutputExt, WireImageEvent};
use crate::{
    EngineCoreEvent, EngineCoreEventType, EngineCoreFinishReason, EngineCoreOutput,
    EngineCoreRequest, EngineCoreSamplingParams, StopReason,
};
use tokio::sync::mpsc;
use uniserve_core::{GenMode, ImageParams, RequestId, SamplingParams as USampling};
use uniserve_engine_api::{EventTx, FinishReason, GenEvent, GenerateRequest, MmItem};

/// Translate the wire sampling DTO into UniServe sampling params.

/// Disposition of the wire fields this engine does **not** carry into
/// `uniserve_core::SamplingParams`:
/// - `prompt_logprobs`: accepted as advisory (the OpenAI `echo` path and the
/// chat/completions surfaces deliberately tolerate it), but the worker emits
/// only per-token sampled/top-k logprobs, so prompt-position logprobs are not
/// produced. Plumbing them end-to-end is a worker (batched prompt-logprobs)
/// change tracked in the deferred-rewrites doc, not a bounded translation fix.
/// - `logprob_token_ids` and the parallel `WireLogprobs` ndarray form: a dead
/// representation modeled after the upstream Python engine that no component
/// here populates; reserved for the same future worker support.
/// - `structured_outputs`: only the guided-`choice` form is enforced (compiled
/// into a `GrammarSpec::Choice` in `to_generate_request`); `json`/`regex`/
/// `grammar`/`json_object`/`structural_tag` are accepted as advisory.
/// - `skip_reading_prefix_cache`: honored — plumbed via `to_generate_request`.
pub fn to_uniserve_sampling(sp: Option<&EngineCoreSamplingParams>) -> USampling {
    let Some(sp) = sp else {
        return USampling::default();
    };
    USampling {
        temperature: sp.temperature,
        top_k: sp.top_k,
        top_p: sp.top_p,
        ignore_eos: sp.ignore_eos,
        seed: sp.seed.map(|s| s as u64),
        min_p: sp.min_p,
        repetition_penalty: sp.repetition_penalty,
        frequency_penalty: sp.frequency_penalty,
        presence_penalty: sp.presence_penalty,
        logit_bias: sp
            .logit_bias
            .as_ref()
            .map(|m| m.iter().map(|(k, v)| (*k, *v)).collect())
            .unwrap_or_default(),
        min_tokens: sp.min_tokens as usize,
        // `logprobs` is `None` (disabled), a positive count, or `-1` (the full
        // vocabulary). `n_logprobs` is a `u32` count with no "all" sentinel, and
        // the worker clamps it with `min(n_logprobs, vocab_size)`, so map the
        // full-vocab request to `u32::MAX` (clamped to vocab downstream) rather
        // than silently collapsing it to `0` (== logprobs disabled).
        n_logprobs: match sp.logprobs {
            Some(n) if n < 0 => u32::MAX,
            Some(n) => n as u32,
            None => 0,
        },
        bad_words_ids: sp.bad_words_token_ids.clone().unwrap_or_default(),
        allowed_token_ids: sp.allowed_token_ids.clone(),
    }
}

/// The per-request explicit stop-token set. `all_stop_token_ids` is retained as
/// a frontend/worker sampling DTO field, but scheduler termination should not
/// treat it as explicit stops; model EOS is handled by the scheduler's EOS path
/// and gated by `ignore_eos`.
pub fn stop_token_ids(sp: Option<&EngineCoreSamplingParams>) -> Vec<u32> {
    sp.map(|s| s.stop_token_ids.clone()).unwrap_or_default()
}

/// Translate one wire request (text or native image/interleave) into an
/// `uniserve_engine_api::GenerateRequest` bound to the scheduler id `rid`.
pub fn to_generate_request(
    req: &EngineCoreRequest,
    rid: RequestId,
    event_tx: EventTx,
) -> GenerateRequest {
    let sampling = to_uniserve_sampling(req.sampling_params.as_ref());
    let max_tokens = req
        .sampling_params
        .as_ref()
        .map(|s| s.max_tokens as usize)
        .unwrap_or(0);
    let stop = stop_token_ids(req.sampling_params.as_ref());
    let prompt_ids = req.prompt_token_ids.clone().unwrap_or_default();

    let (mode, image, neg_prompt_ids, mm_items) = match &req.native {
        Some(ext) => (
            ext.mode,
            ext.image.clone(),
            ext.neg_prompt_ids.clone(),
            ext.mm_items
                .iter()
                .map(|m| MmItem {
                    hash: m.hash,
                    position: m.position,
                    num_tokens: m.num_tokens,
                    b64: m.b64.clone(),
                })
                .collect(),
        ),
        None => (
            GenMode::Text,
            ImageParams::default(),
            Vec::new(),
            Vec::new(),
        ),
    };

    let mut generate =
        GenerateRequest::new(rid, prompt_ids, sampling, image, mode, max_tokens, event_tx);
    generate.stop_token_ids = stop;
    generate.neg_prompt_ids = neg_prompt_ids;
    generate.mm_items = mm_items;
    generate.priority = req.priority;
    // `lora_int_id` is u64 on the reference-shaped wire struct, but the
    // registry guarantees it fits u32 at allocation (LoraManager id-space guard),
    // so this narrowing is lossless.
    generate.lora_id = req.lora_request.as_ref().map(|l| l.lora_int_id as u32);
    // Structured outputs: the frontend tokenized the guided choices; the engine
    // compiles them into a per-step token-mask grammar.
    generate.grammar = req
        .sampling_params
        .as_ref()
        .and_then(|s| s.choice_token_ids.clone())
        .filter(|c| !c.is_empty())
        .map(uniserve_engine_api::GrammarSpec::Choice);
    // honor the per-request prefix-cache read opt-out (
    // field the engine silently dropped) so `bypass_prefix_cache` is real.
    generate.skip_reading_prefix_cache = req
        .sampling_params
        .as_ref()
        .and_then(|s| s.skip_reading_prefix_cache)
        .unwrap_or(false);
    generate
}

/// Map a UniServe finish reason onto the wire finish reason and stop reason.
pub fn map_finish(
    reason: &FinishReason,
    stop_reason: Option<String>,
) -> (EngineCoreFinishReason, Option<StopReason>) {
    match reason {
        FinishReason::Eos => (EngineCoreFinishReason::Stop, None),
        FinishReason::Stop => (
            EngineCoreFinishReason::Stop,
            stop_reason.map(StopReason::Text),
        ),
        FinishReason::MaxTokens => (EngineCoreFinishReason::Length, None),
        FinishReason::Cancelled | FinishReason::Aborted => (EngineCoreFinishReason::Abort, None),
        FinishReason::ImageDone => (EngineCoreFinishReason::Stop, None),
        FinishReason::Error => (EngineCoreFinishReason::Error, None),
    }
}

/// The native finish-reason string carried on the wire extension.
pub fn native_reason_str(reason: &FinishReason) -> &'static str {
    match reason {
        FinishReason::Eos => "eos",
        FinishReason::MaxTokens => "max_tokens",
        FinishReason::Stop => "stop",
        FinishReason::ImageDone => "image_done",
        FinishReason::Cancelled => "cancelled",
        FinishReason::Aborted => "aborted",
        FinishReason::Error => "error",
    }
}

/// Parse the wire reason string back into the native finish reason.
pub fn parse_native_reason(reason: &str) -> FinishReason {
    match reason {
        "eos" => FinishReason::Eos,
        "max_tokens" => FinishReason::MaxTokens,
        "stop" => FinishReason::Stop,
        "image_done" => FinishReason::ImageDone,
        "cancelled" => FinishReason::Cancelled,
        "aborted" => FinishReason::Aborted,
        _ => FinishReason::Error,
    }
}

/// Build a one-position logprobs payload for a single generated token: the
/// sampled token first, followed by any returned top-k alternatives.
pub fn build_logprobs(
    token_id: u32,
    sampled: Option<f32>,
    top: Option<Vec<(u32, f32)>>,
) -> Option<MaybeWireLogprobs> {
    let sampled = sampled?;
    // The sampler returns the sampled token's logprob but not its true vocab
    // rank (it does not count how many masked logits outrank it), so emit rank
    // `0` ("rank unknown") rather than fabricating `1`. A fabricated `1` would
    // both lie about the sampled token's position and collide with the first
    // top-k alternative's rank, which is the genuine 1-based candidate rank.
    let mut entries = vec![TokenLogprob {
        token_id,
        logprob: sampled,
        rank: 0,
    }];
    if let Some(top) = top {
        for (rank, (tid, lp)) in top.into_iter().enumerate() {
            entries.push(TokenLogprob {
                token_id: tid,
                logprob: lp,
                rank: (rank + 1) as u32,
            });
        }
    }
    Some(MaybeWireLogprobs::Direct(Logprobs {
        positions: vec![PositionLogprobs { entries }],
    }))
}

/// Adapter parameters for one request's event stream.
#[derive(Debug, Clone)]
pub struct AdapterParams {
    pub request_id: String,
    /// Emit per-token logprobs (`sampling_params.logprobs > 0`).
    pub want_logprobs: bool,
    /// The request is a native image/interleave request: image events and
    /// finish statistics ride the wire `native` extension, and `Scheduled`
    /// timestamps are surfaced as `EngineCoreEvent`s.
    pub native: bool,
}

/// Drive one request's `GenEvent` stream into wire outputs.

/// `emit` returns `false` when the consumer is gone, which stops adaptation.
/// Returns when the stream reaches a terminal event or the consumer drops.
pub async fn run_event_adapter(
    params: AdapterParams,
    mut events: mpsc::UnboundedReceiver<GenEvent>,
    mut emit: impl FnMut(EngineCoreOutput) -> bool,
) {
    let AdapterParams {
        request_id,
        want_logprobs,
        native,
    } = params;

    // A `TextToken` is held until its `TokenLogprobs` arrives so the sampled
    // token and its top-k alternatives land in one wire output. This coalescing
    // is ONLY needed when logprobs were requested: the scheduler emits
    // `TokenLogprobs` iff `want_logprobs`, and it does so in the same step right
    // after the `TextToken`. When logprobs are NOT requested, holding the token
    // would delay it until the *next* token (a full decode step, ~25ms),
    // inflating streaming TTFT; so in that case we emit immediately.
    let mut pending: Option<(u32, Option<f32>)> = None;

    let token_output = |id: u32, lp: Option<f32>, top: Option<Vec<(u32, f32)>>| EngineCoreOutput {
        request_id: request_id.clone(),
        new_token_ids: vec![id],
        new_logprobs: if want_logprobs {
            build_logprobs(id, lp, top)
        } else {
            None
        },
        ..Default::default()
    };
    macro_rules! flush_pending {
        () => {
            if let Some((pid, plp)) = pending.take()
                && !emit(token_output(pid, plp, None))
            {
                break;
            }
        };
    }

    while let Some(ev) = events.recv().await {
        match ev {
            GenEvent::TextToken { id, logprob } => {
                flush_pending!();
                if want_logprobs {
                    // Coalesce with the TokenLogprobs emitted in this same step.
                    pending = Some((id, logprob));
                } else if !emit(token_output(id, logprob, None)) {
                    break;
                }
            }
            GenEvent::TokenLogprobs { id, top } => {
                let (pid, plp) = pending.take().unwrap_or((id, None));
                if !emit(token_output(pid, plp, Some(top))) {
                    break;
                }
            }
            GenEvent::Scheduled {
                queued_at,
                scheduled_at,
            } if native => {
                flush_pending!();
                let output = EngineCoreOutput {
                    request_id: request_id.clone(),
                    events: Some(vec![
                        EngineCoreEvent {
                            r#type: EngineCoreEventType::Queued,
                            timestamp: queued_at,
                        },
                        EngineCoreEvent {
                            r#type: EngineCoreEventType::Scheduled,
                            timestamp: scheduled_at,
                        },
                    ]),
                    ..Default::default()
                };
                if !emit(output) {
                    break;
                }
            }
            GenEvent::ImageBegin {
                image_id,
                height,
                width,
                steps,
            } if native => {
                // Flush the held text token first: image events must not
                // overtake the text that preceded them in interleaved streams.
                flush_pending!();
                let output = native_image_output(
                    &request_id,
                    WireImageEvent::Begin {
                        image_id,
                        height,
                        width,
                        steps,
                    },
                );
                if !emit(output) {
                    break;
                }
            }
            GenEvent::ImageStep { image_id, step } if native => {
                flush_pending!();
                let output =
                    native_image_output(&request_id, WireImageEvent::Step { image_id, step });
                if !emit(output) {
                    break;
                }
            }
            GenEvent::ImageDone {
                image_id,
                height,
                width,
                bytes,
                sha256,
                pixels_png_b64,
            } if native => {
                flush_pending!();
                let output = native_image_output(
                    &request_id,
                    WireImageEvent::Done {
                        image_id,
                        height,
                        width,
                        bytes,
                        sha256,
                        png_b64: pixels_png_b64,
                    },
                );
                if !emit(output) {
                    break;
                }
            }
            GenEvent::Finished {
                reason,
                stop_reason,
                prompt_tokens,
                completion_tokens,
                images,
            } => {
                flush_pending!();
                let (finish_reason, stop) = map_finish(&reason, stop_reason);
                let _ = emit(EngineCoreOutput {
                    request_id: request_id.clone(),
                    finish_reason: Some(finish_reason),
                    stop_reason: stop,
                    native: native.then(|| NativeOutputExt {
                        image: None,
                        finish: Some(NativeFinishExt {
                            reason: native_reason_str(&reason).to_string(),
                            prompt_tokens: prompt_tokens as u64,
                            completion_tokens: completion_tokens as u64,
                            images: images as u64,
                            message: None,
                        }),
                    }),
                    ..Default::default()
                });
                break;
            }
            GenEvent::Rejected { ref message } | GenEvent::Error { ref message } => {
                let rejected = matches!(ev, GenEvent::Rejected { .. });
                tracing::warn!(request_id, message, "request failed in the engine");
                let _ = emit(EngineCoreOutput {
                    request_id: request_id.clone(),
                    finish_reason: Some(EngineCoreFinishReason::Error),
                    native: native.then(|| NativeOutputExt {
                        image: None,
                        finish: Some(NativeFinishExt {
                            reason: if rejected { "rejected" } else { "error" }.to_string(),
                            message: Some(message.clone()),
                            ..Default::default()
                        }),
                    }),
                    ..Default::default()
                });
                break;
            }
            // Non-native requests have no wire shape for these; drop them, as
            // the in-process text path always has.
            GenEvent::Scheduled { .. }
            | GenEvent::ImageBegin { .. }
            | GenEvent::ImageStep { .. }
            | GenEvent::ImageDone { .. } => {}
        }
    }
}

fn native_image_output(request_id: &str, image: WireImageEvent) -> EngineCoreOutput {
    EngineCoreOutput {
        request_id: request_id.to_string(),
        native: Some(NativeOutputExt {
            image: Some(image),
            finish: None,
        }),
        ..Default::default()
    }
}

/// Reverse adaptation (client side): one wire output back into the typed
/// `GenEvent`s a native stream consumer expects. Text token logprobs are
/// reconstructed from the sampled-first logprobs encoding of
/// [`build_logprobs`].
pub fn wire_output_to_gen_events(output: &EngineCoreOutput) -> Vec<GenEvent> {
    let mut events = Vec::new();

    if let Some(evs) = &output.events {
        let queued = evs
            .iter()
            .find(|e| e.r#type == EngineCoreEventType::Queued)
            .map(|e| e.timestamp);
        if let Some(scheduled) = evs
            .iter()
            .find(|e| e.r#type == EngineCoreEventType::Scheduled)
        {
            events.push(GenEvent::Scheduled {
                queued_at: queued.unwrap_or(scheduled.timestamp),
                scheduled_at: scheduled.timestamp,
            });
        }
    }

    let positions: Vec<&PositionLogprobs> = match &output.new_logprobs {
        Some(MaybeWireLogprobs::Direct(lp)) => lp.positions.iter().collect(),
        _ => Vec::new(),
    };
    for (i, &id) in output.new_token_ids.iter().enumerate() {
        let pos = positions.get(i);
        let logprob = pos.and_then(|p| p.entries.first()).map(|e| e.logprob);
        events.push(GenEvent::TextToken { id, logprob });
        if let Some(p) = pos
            && p.entries.len() > 1
        {
            let top: Vec<(u32, f32)> = p.entries[1..]
                .iter()
                .map(|e| (e.token_id, e.logprob))
                .collect();
            events.push(GenEvent::TokenLogprobs { id, top });
        }
    }

    if let Some(ext) = &output.native
        && let Some(image) = &ext.image
    {
        events.push(match image.clone() {
            WireImageEvent::Begin {
                image_id,
                height,
                width,
                steps,
            } => GenEvent::ImageBegin {
                image_id,
                height,
                width,
                steps,
            },
            WireImageEvent::Step { image_id, step } => GenEvent::ImageStep { image_id, step },
            WireImageEvent::Done {
                image_id,
                height,
                width,
                bytes,
                sha256,
                png_b64,
            } => GenEvent::ImageDone {
                image_id,
                height,
                width,
                bytes,
                sha256,
                pixels_png_b64: png_b64,
            },
        });
    }

    if let Some(finish_reason) = output.finish_reason {
        let native_finish = output.native.as_ref().and_then(|n| n.finish.as_ref());
        match native_finish {
            Some(f) if f.reason == "rejected" => events.push(GenEvent::Rejected {
                message: f.message.clone().unwrap_or_default(),
            }),
            Some(f) if f.reason == "error" => events.push(GenEvent::Error {
                message: f.message.clone().unwrap_or_default(),
            }),
            Some(f) => events.push(GenEvent::Finished {
                reason: parse_native_reason(&f.reason),
                stop_reason: match &output.stop_reason {
                    Some(StopReason::Text(s)) => Some(s.clone()),
                    Some(StopReason::TokenId(id)) => Some(format!("token:{id}")),
                    None => None,
                },
                prompt_tokens: f.prompt_tokens as usize,
                completion_tokens: f.completion_tokens as usize,
                images: f.images as usize,
            }),
            None => {
                // A native stream should always carry the finish extension;
                // fall back to a coarse mapping if it is missing.
                let reason = match finish_reason {
                    EngineCoreFinishReason::Stop => FinishReason::Eos,
                    EngineCoreFinishReason::Length => FinishReason::MaxTokens,
                    EngineCoreFinishReason::Abort => FinishReason::Aborted,
                    EngineCoreFinishReason::Error | EngineCoreFinishReason::Repetition => {
                        FinishReason::Error
                    }
                };
                events.push(GenEvent::Finished {
                    reason,
                    stop_reason: None,
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    images: 0,
                });
            }
        }
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::NativeRequestExt;

    #[test]
    fn build_logprobs_sampled_entry_has_no_fabricated_rank() {
        // The sampled token's true vocab rank is unknown to the sampler, so it
        // is emitted as `0` ("unknown") and must not collide with the first
        // top-k alternative's genuine 1-based rank.
        let lp = build_logprobs(7, Some(-0.5), Some(vec![(7, -0.5), (9, -1.2)]))
            .expect("sampled logprob present");
        let positions = match lp {
            MaybeWireLogprobs::Direct(l) => l.positions,
            other => panic!("expected Direct logprobs, got {other:?}"),
        };
        assert_eq!(positions.len(), 1);
        let entries = &positions[0].entries;
        assert_eq!(entries.len(), 3);
        // Sampled token first, rank unknown.
        assert_eq!((entries[0].token_id, entries[0].rank), (7, 0));
        // Alternatives carry 1-based candidate ranks.
        assert_eq!(entries[1].rank, 1);
        assert_eq!(entries[2].rank, 2);
    }

    #[test]
    fn full_vocab_logprobs_request_maps_to_max_not_zero() {
        let mut sp = EngineCoreSamplingParams::for_test();
        sp.logprobs = Some(-1);
        let u = to_uniserve_sampling(Some(&sp));
        // `-1` (full vocabulary) must not collapse to `0` (disabled); it maps to
        // the max count, which the worker clamps to the vocab size.
        assert_eq!(u.n_logprobs, u32::MAX);

        sp.logprobs = Some(5);
        assert_eq!(to_uniserve_sampling(Some(&sp)).n_logprobs, 5);

        sp.logprobs = None;
        assert_eq!(to_uniserve_sampling(Some(&sp)).n_logprobs, 0);
    }

    #[test]
    fn text_request_translates_with_stop_set() {
        let mut sp = EngineCoreSamplingParams::for_test();
        sp.max_tokens = 16;
        sp.stop_token_ids = vec![42];
        sp.all_stop_token_ids = [151645u32, 151643].into_iter().collect();
        let req = EngineCoreRequest {
            request_id: "r1".into(),
            prompt_token_ids: Some(vec![1, 2, 3]),
            sampling_params: Some(sp),
            ..Default::default()
        };
        let (tx, _rx) = mpsc::unbounded_channel();
        let g = to_generate_request(&req, RequestId(7), tx);
        assert_eq!(g.request_id, RequestId(7));
        assert_eq!(g.prompt_ids, vec![1, 2, 3]);
        assert_eq!(g.max_tokens, 16);
        assert_eq!(g.mode, GenMode::Text);
        assert_eq!(g.stop_token_ids, vec![42]);
    }

    #[test]
    fn skip_reading_prefix_cache_is_plumbed_to_generate_request() {
        // the wire opt-out is honored at this boundary.
        let mut sp = EngineCoreSamplingParams::for_test();
        sp.skip_reading_prefix_cache = Some(true);
        let req = EngineCoreRequest {
            request_id: "r1".into(),
            prompt_token_ids: Some(vec![1, 2, 3]),
            sampling_params: Some(sp),
            ..Default::default()
        };
        let (tx, _rx) = mpsc::unbounded_channel();
        assert!(to_generate_request(&req, RequestId(7), tx).skip_reading_prefix_cache);

        // Absent / false defaults to reading the cache (the common case).
        let mut sp = EngineCoreSamplingParams::for_test();
        sp.skip_reading_prefix_cache = None;
        let req = EngineCoreRequest {
            request_id: "r1".into(),
            prompt_token_ids: Some(vec![1, 2, 3]),
            sampling_params: Some(sp),
            ..Default::default()
        };
        let (tx, _rx) = mpsc::unbounded_channel();
        assert!(!to_generate_request(&req, RequestId(7), tx).skip_reading_prefix_cache);
    }

    #[test]
    fn text_request_translates_ignore_eos_for_lookahead() {
        let mut sp = EngineCoreSamplingParams::for_test();
        sp.temperature = 0.0;
        sp.ignore_eos = true;
        sp.stop_token_ids = Vec::new();
        sp.all_stop_token_ids = [151645u32, 151643].into_iter().collect();
        let req = EngineCoreRequest {
            request_id: "r1".into(),
            prompt_token_ids: Some(vec![1, 2, 3]),
            sampling_params: Some(sp),
            ..Default::default()
        };
        let (tx, _rx) = mpsc::unbounded_channel();
        let g = to_generate_request(&req, RequestId(7), tx);

        assert!(g.sampling.ignore_eos);
        assert_eq!(g.stop_token_ids, Vec::<u32>::new());
    }

    #[test]
    fn native_request_translates_mode_and_items() {
        let req = EngineCoreRequest {
            request_id: "r2".into(),
            prompt_token_ids: Some(vec![5]),
            sampling_params: Some(EngineCoreSamplingParams::for_test()),
            native: Some(NativeRequestExt {
                mode: GenMode::AutoInterleave,
                image: ImageParams {
                    steps: 7,
                    ..Default::default()
                },
                neg_prompt_ids: vec![9],
                mm_items: vec![],
            }),
            ..Default::default()
        };
        let (tx, _rx) = mpsc::unbounded_channel();
        let g = to_generate_request(&req, RequestId(1), tx);
        assert_eq!(g.mode, GenMode::AutoInterleave);
        assert_eq!(g.image.steps, 7);
        assert_eq!(g.neg_prompt_ids, vec![9]);
    }

    /// GenEvents adapted onto the wire and back arrive intact (the socket-mode
    /// native stream roundtrip).
    #[tokio::test]
    async fn native_event_wire_roundtrip() {
        let (tx, rx) = mpsc::unbounded_channel();
        let events = vec![
            GenEvent::TextToken {
                id: 11,
                logprob: None,
            },
            GenEvent::ImageBegin {
                image_id: 1,
                height: 64,
                width: 64,
                steps: 2,
            },
            GenEvent::ImageStep {
                image_id: 1,
                step: 1,
            },
            GenEvent::ImageDone {
                image_id: 1,
                height: 2,
                width: 3,
                bytes: 3,
                sha256: "b5d4045c3f466fa91fe2cc6abe79232a1a57cdf104f7a26e716e0a1e2789df78".into(),
                pixels_png_b64: "QUJD".into(),
            },
            GenEvent::Finished {
                reason: FinishReason::Eos,
                stop_reason: None,
                prompt_tokens: 3,
                completion_tokens: 1,
                images: 1,
            },
        ];
        for ev in events {
            tx.send(ev).unwrap();
        }
        drop(tx);

        let mut outputs = Vec::new();
        run_event_adapter(
            AdapterParams {
                request_id: "r".into(),
                want_logprobs: false,
                native: true,
            },
            rx,
            |o| {
                outputs.push(o);
                true
            },
        )
        .await;

        let roundtripped: Vec<GenEvent> =
            outputs.iter().flat_map(wire_output_to_gen_events).collect();
        let kinds: Vec<&str> = roundtripped
            .iter()
            .map(|e| match e {
                GenEvent::TextToken { .. } => "text",
                GenEvent::ImageBegin { .. } => "begin",
                GenEvent::ImageStep { .. } => "step",
                GenEvent::ImageDone { .. } => "done",
                GenEvent::Finished { .. } => "finished",
                _ => "other",
            })
            .collect();
        assert_eq!(kinds, vec!["text", "begin", "step", "done", "finished"]);
        match &roundtripped[4] {
            GenEvent::Finished {
                reason,
                prompt_tokens,
                completion_tokens,
                images,
                ..
            } => {
                assert_eq!(*reason, FinishReason::Eos);
                assert_eq!((*prompt_tokens, *completion_tokens, *images), (3, 1, 1));
            }
            other => panic!("expected Finished, got {other:?}"),
        }
    }
}
