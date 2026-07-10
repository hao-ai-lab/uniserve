//! Wire DTO ↔ engine type translation, shared by the in-process client adapter
//! and the headless engine process.
//!
//! Northbound, a per-request `GenEvent` stream becomes engine output DTOs.
//! Southbound, each engine request contains one canonical generation request.

use std::future::Future;

use crate::generation::{GenerationFinish, GenerationOutput, WireImageEvent};
use crate::logprobs::{Logprobs, MaybeWireLogprobs, PositionLogprobs, TokenLogprob};
use crate::{
    EngineCoreEvent, EngineCoreEventType, EngineCoreFinishReason, EngineCoreOutput,
    EngineCoreRequest, EngineCoreSamplingParams, StopReason,
};
use tokio::sync::mpsc;
use uniserve_core::{GenerationRequest, RequestId, SamplingParams as USampling};
use uniserve_engine_api::{
    FinishReason, GenEvent, PositionLogprobs as SemanticPositionLogprobs,
    TokenLogprob as SemanticTokenLogprob,
};

/// Normalize frontend sampling values into the canonical scheduler shape.
/// Grammar and cache policy are lowered into their dedicated request fields.
pub fn to_uniserve_sampling(sp: Option<&EngineCoreSamplingParams>) -> USampling {
    let Some(sp) = sp else {
        return USampling::default();
    };
    let mut logit_bias = sp
        .logit_bias
        .as_ref()
        .map(|biases| biases.iter().map(|(&token, &bias)| (token, bias)).collect())
        .unwrap_or_else(Vec::new);
    logit_bias.sort_by_key(|(token, _)| *token);
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
        logit_bias,
        min_tokens: sp.min_tokens as usize,
        return_logprobs: sp.logprobs.is_some() || sp.logprob_token_ids.is_some(),
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
        return_prompt_logprobs: sp.prompt_logprobs.is_some(),
        n_prompt_logprobs: match sp.prompt_logprobs {
            Some(n) if n < 0 => u32::MAX,
            Some(n) => n as u32,
            None => 0,
        },
        logprob_token_ids: sp.logprob_token_ids.clone().unwrap_or_default(),
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

/// Extract the canonical generation request from its transport envelope.
pub fn to_generation_request(
    req: &EngineCoreRequest,
    rid: RequestId,
) -> crate::Result<GenerationRequest> {
    let mut request = req.generation.clone();
    request.request_id = rid;
    request
        .validate()
        .map_err(|error| crate::Error::InvalidGenerationRequest {
            message: error.to_string(),
        })?;
    Ok(request)
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
        FinishReason::Cancelled => (EngineCoreFinishReason::Cancelled, None),
        FinishReason::Aborted => (EngineCoreFinishReason::Aborted, None),
        FinishReason::Repetition => (EngineCoreFinishReason::Repetition, None),
        FinishReason::ImageDone => (EngineCoreFinishReason::Stop, None),
        FinishReason::Error => (EngineCoreFinishReason::Error, None),
    }
}

/// Stable finish-reason string carried on the wire.
pub fn generation_reason_str(reason: &FinishReason) -> &'static str {
    match reason {
        FinishReason::Eos => "eos",
        FinishReason::MaxTokens => "max_tokens",
        FinishReason::Stop => "stop",
        FinishReason::ImageDone => "image_done",
        FinishReason::Cancelled => "cancelled",
        FinishReason::Aborted => "aborted",
        FinishReason::Repetition => "repetition",
        FinishReason::Error => "error",
    }
}

/// Parse the wire reason string back into the canonical finish reason.
pub fn parse_generation_reason(reason: &str) -> FinishReason {
    match reason {
        "eos" => FinishReason::Eos,
        "max_tokens" => FinishReason::MaxTokens,
        "stop" => FinishReason::Stop,
        "image_done" => FinishReason::ImageDone,
        "cancelled" => FinishReason::Cancelled,
        "aborted" => FinishReason::Aborted,
        "repetition" => FinishReason::Repetition,
        _ => FinishReason::Error,
    }
}

/// Build a one-position logprobs payload for a single generated token: the
/// sampled token first, followed by any returned top-k alternatives.
pub fn build_logprobs(
    token_id: u32,
    sampled: Option<f32>,
    candidates: Option<Vec<SemanticTokenLogprob>>,
) -> Option<MaybeWireLogprobs> {
    let sampled = sampled?;
    let mut entries: Vec<TokenLogprob> = candidates
        .unwrap_or_default()
        .into_iter()
        .map(|entry| TokenLogprob {
            token_id: entry.token_id,
            logprob: entry.logprob,
            rank: entry.rank,
        })
        .collect();
    if !entries.iter().any(|entry| entry.token_id == token_id) {
        entries.insert(
            0,
            TokenLogprob {
                token_id,
                logprob: sampled,
                rank: 0,
            },
        );
    }
    Some(MaybeWireLogprobs::Direct(Logprobs {
        positions: vec![PositionLogprobs { entries }],
    }))
}

fn prompt_logprobs_to_wire(positions: Vec<SemanticPositionLogprobs>) -> MaybeWireLogprobs {
    MaybeWireLogprobs::Direct(Logprobs {
        positions: positions
            .into_iter()
            .map(|position| PositionLogprobs {
                entries: position
                    .entries
                    .into_iter()
                    .map(|entry| TokenLogprob {
                        token_id: entry.token_id,
                        logprob: entry.logprob,
                        rank: entry.rank,
                    })
                    .collect(),
            })
            .collect(),
    })
}

/// Adapter parameters for one request's event stream.
#[derive(Debug, Clone)]
pub struct AdapterParams {
    pub request_id: String,
    /// Emit per-token logprobs (`sampling_params.logprobs > 0`).
    pub want_logprobs: bool,
}

/// Drive one request's `GenEvent` stream into wire outputs.
///
/// `emit` returns `false` when the consumer is gone, which stops adaptation.
/// Returns when the stream reaches a terminal event or the consumer drops.
pub async fn run_event_adapter<Emit, EmitFuture>(
    params: AdapterParams,
    mut events: mpsc::UnboundedReceiver<GenEvent>,
    mut emit: Emit,
) where
    Emit: FnMut(EngineCoreOutput) -> EmitFuture,
    EmitFuture: Future<Output = bool>,
{
    let AdapterParams {
        request_id,
        want_logprobs,
    } = params;

    // A `TextToken` is held until its `TokenLogprobs` arrives so the sampled
    // token and its top-k alternatives land in one wire output. This coalescing
    // is ONLY needed when logprobs were requested: the scheduler emits
    // `TokenLogprobs` iff `want_logprobs`, and it does so in the same step right
    // after the `TextToken`. When logprobs are NOT requested, holding the token
    // would delay it until the *next* token (a full decode step, ~25ms),
    // inflating streaming TTFT; so in that case we emit immediately.
    let mut pending: Option<(u32, Option<f32>)> = None;

    let token_output = |id: u32, lp: Option<f32>, candidates: Option<Vec<SemanticTokenLogprob>>| {
        EngineCoreOutput {
            request_id: request_id.clone(),
            new_token_ids: vec![id],
            new_logprobs: if want_logprobs {
                build_logprobs(id, lp, candidates)
            } else {
                None
            },
            ..Default::default()
        }
    };
    macro_rules! flush_pending {
        () => {
            if let Some((pid, plp)) = pending.take()
                && !emit(token_output(pid, plp, None)).await
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
                } else if !emit(token_output(id, logprob, None)).await {
                    break;
                }
            }
            GenEvent::TokenLogprobs { id, candidates } => {
                let (pid, plp) = pending.take().unwrap_or((id, None));
                if !emit(token_output(pid, plp, Some(candidates))).await {
                    break;
                }
            }
            GenEvent::PromptLogprobs { positions } => {
                flush_pending!();
                if !emit(EngineCoreOutput {
                    request_id: request_id.clone(),
                    new_prompt_logprobs_tensors: Some(prompt_logprobs_to_wire(positions)),
                    ..Default::default()
                })
                .await
                {
                    break;
                }
            }
            GenEvent::Scheduled {
                queued_at,
                scheduled_at,
            } => {
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
                if !emit(output).await {
                    break;
                }
            }
            GenEvent::ImageBegin {
                image_id,
                height,
                width,
                steps,
            } => {
                // Flush the held text token first: image events must not
                // overtake the text that preceded them in a mixed-output stream.
                flush_pending!();
                let output = generation_image_output(
                    &request_id,
                    WireImageEvent::Begin {
                        image_id,
                        height,
                        width,
                        steps,
                    },
                );
                if !emit(output).await {
                    break;
                }
            }
            GenEvent::ImageStep { image_id, step } => {
                flush_pending!();
                let output =
                    generation_image_output(&request_id, WireImageEvent::Step { image_id, step });
                if !emit(output).await {
                    break;
                }
            }
            GenEvent::ImageCommit { image_id } => {
                flush_pending!();
                let output =
                    generation_image_output(&request_id, WireImageEvent::Commit { image_id });
                if !emit(output).await {
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
            } => {
                flush_pending!();
                let output = generation_image_output(
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
                if !emit(output).await {
                    break;
                }
            }
            GenEvent::Finished {
                reason,
                stop_reason,
                prompt_tokens,
                completion_tokens,
                images,
                kv_transfer_params,
            } => {
                flush_pending!();
                let (finish_reason, stop) = map_finish(&reason, stop_reason);
                let _ = emit(EngineCoreOutput {
                    request_id: request_id.clone(),
                    finish_reason: Some(finish_reason),
                    stop_reason: stop,
                    kv_transfer_params,
                    generation: Some(GenerationOutput {
                        image: None,
                        finish: Some(GenerationFinish {
                            reason: generation_reason_str(&reason).to_string(),
                            prompt_tokens: prompt_tokens as u64,
                            completion_tokens: completion_tokens as u64,
                            images: images as u64,
                            message: None,
                        }),
                    }),
                    ..Default::default()
                })
                .await;
                break;
            }
            GenEvent::Rejected { ref message } | GenEvent::Error { ref message } => {
                let rejected = matches!(ev, GenEvent::Rejected { .. });
                tracing::warn!(request_id, message, "request failed in the engine");
                let _ = emit(EngineCoreOutput {
                    request_id: request_id.clone(),
                    finish_reason: Some(EngineCoreFinishReason::Error),
                    generation: Some(GenerationOutput {
                        image: None,
                        finish: Some(GenerationFinish {
                            reason: if rejected { "rejected" } else { "error" }.to_string(),
                            message: Some(message.clone()),
                            ..Default::default()
                        }),
                    }),
                    ..Default::default()
                })
                .await;
                break;
            }
        }
    }
}

fn generation_image_output(request_id: &str, image: WireImageEvent) -> EngineCoreOutput {
    EngineCoreOutput {
        request_id: request_id.to_string(),
        generation: Some(GenerationOutput {
            image: Some(image),
            finish: None,
        }),
        ..Default::default()
    }
}

/// Reverse adaptation (client side): one wire output back into the typed
/// canonical `GenEvent`s. Text token logprobs are
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
        if let Some(position) = pos {
            events.push(GenEvent::TokenLogprobs {
                id,
                candidates: position
                    .entries
                    .iter()
                    .map(|entry| SemanticTokenLogprob {
                        token_id: entry.token_id,
                        logprob: entry.logprob,
                        rank: entry.rank,
                    })
                    .collect(),
            });
        }
    }

    if let Some(MaybeWireLogprobs::Direct(prompt)) = &output.new_prompt_logprobs_tensors {
        events.push(GenEvent::PromptLogprobs {
            positions: prompt
                .positions
                .iter()
                .map(|position| SemanticPositionLogprobs {
                    entries: position
                        .entries
                        .iter()
                        .map(|entry| SemanticTokenLogprob {
                            token_id: entry.token_id,
                            logprob: entry.logprob,
                            rank: entry.rank,
                        })
                        .collect(),
                })
                .collect(),
        });
    }

    if let Some(ext) = &output.generation
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
            WireImageEvent::Commit { image_id } => GenEvent::ImageCommit { image_id },
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
        let generation_finish = output.generation.as_ref().and_then(|n| n.finish.as_ref());
        match generation_finish {
            Some(f) if f.reason == "rejected" => events.push(GenEvent::Rejected {
                message: f.message.clone().unwrap_or_default(),
            }),
            Some(f) if f.reason == "error" => events.push(GenEvent::Error {
                message: f.message.clone().unwrap_or_default(),
            }),
            Some(f) => events.push(GenEvent::Finished {
                reason: parse_generation_reason(&f.reason),
                stop_reason: match &output.stop_reason {
                    Some(StopReason::Text(s)) => Some(s.clone()),
                    Some(StopReason::TokenId(id)) => Some(format!("token:{id}")),
                    None => None,
                },
                prompt_tokens: f.prompt_tokens as usize,
                completion_tokens: f.completion_tokens as usize,
                images: f.images as usize,
                kv_transfer_params: output.kv_transfer_params.clone(),
            }),
            None => events.push(GenEvent::Error {
                message: format!(
                    "terminal engine output {finish_reason:?} omitted generation finish statistics"
                ),
            }),
        }
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniserve_core::{
        CommitRecipe, ContextSegment, FeedbackNextToken, FeedbackWriteback,
        GeneratedImageFeedbackRecipe, GenerationBehaviorDescriptor, GenerationConstraint,
        GenerationPolicyDescriptor, GenerationResourceBounds, ImageKvEffect, ImageParams,
        UndVisibility,
    };

    fn canonical_generation_request() -> GenerationRequest {
        let constraint = GenerationConstraint::Default;
        let policy = GenerationPolicyDescriptor {
            trigger: uniserve_core::TriggerPolicyDescriptor::Token { token_id: 42 },
            feedback: Some(GeneratedImageFeedbackRecipe {
                commit: CommitRecipe::CommitGenThenWriteback,
                writeback: FeedbackWriteback::DirectKv,
                next_und_token: FeedbackNextToken::EndOfImage,
                logical_positions: 2,
                physical_kv_tokens: ImageKvEffect::Bounded { max_tokens: 64 },
            }),
            ..GenerationPolicyDescriptor::default()
        };
        GenerationRequest {
            request_id: RequestId(99),
            context: vec![ContextSegment::UndTokens {
                token_ids: vec![5],
                visibility: UndVisibility::Internal,
            }],
            negative_context: vec![ContextSegment::UndTokens {
                token_ids: vec![9],
                visibility: UndVisibility::Internal,
            }],
            constraint,
            behavior: GenerationBehaviorDescriptor::resolve(constraint, &policy),
            sampling: USampling::default(),
            image: ImageParams {
                steps: 7,
                ..ImageParams::default()
            },
            max_und_tokens: 16,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            lora_id: None,
            grammar: None,
            cache: Default::default(),
            policy,
            resources: GenerationResourceBounds {
                context_tokens: 1,
                max_kv_tokens: 128,
                generated_feedback_makes_non_replayable: true,
                ..GenerationResourceBounds::default()
            },
        }
    }

    #[test]
    fn build_logprobs_preserves_measured_candidate_ranks() {
        let lp = build_logprobs(
            7,
            Some(-0.5),
            Some(vec![
                SemanticTokenLogprob {
                    token_id: 7,
                    logprob: -0.5,
                    rank: 3,
                },
                SemanticTokenLogprob {
                    token_id: 9,
                    logprob: -1.2,
                    rank: 5,
                },
            ]),
        )
        .expect("sampled logprob present");
        let positions = match lp {
            MaybeWireLogprobs::Direct(l) => l.positions,
            other => panic!("expected Direct logprobs, got {other:?}"),
        };
        assert_eq!(positions.len(), 1);
        let entries = &positions[0].entries;
        assert_eq!(entries.len(), 2);
        assert_eq!((entries[0].token_id, entries[0].rank), (7, 3));
        assert_eq!((entries[1].token_id, entries[1].rank), (9, 5));
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
    fn canonical_request_translates_constraint_and_items() {
        let req = EngineCoreRequest::new("r2".into(), canonical_generation_request());
        let g = to_generation_request(&req, RequestId(1)).expect("canonical request");
        assert_eq!(g.constraint, GenerationConstraint::Default);
        assert_eq!(g.request_id, RequestId(1));
        assert_eq!(g.image.steps, 7);
        assert_eq!(
            g.negative_context,
            vec![ContextSegment::UndTokens {
                token_ids: vec![9],
                visibility: UndVisibility::Internal,
            }]
        );
    }

    /// GenEvents adapted onto the wire and back arrive intact.
    #[tokio::test]
    async fn generation_event_wire_roundtrip() {
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
            GenEvent::ImageCommit { image_id: 1 },
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
                kv_transfer_params: Some(serde_json::json!({"connector": "x"})),
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
            },
            rx,
            |o| {
                outputs.push(o);
                std::future::ready(true)
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
                GenEvent::ImageCommit { .. } => "commit",
                GenEvent::ImageDone { .. } => "done",
                GenEvent::Finished { .. } => "finished",
                _ => "other",
            })
            .collect();
        assert_eq!(
            kinds,
            vec!["text", "begin", "step", "commit", "done", "finished"]
        );
        match &roundtripped[5] {
            GenEvent::Finished {
                reason,
                prompt_tokens,
                completion_tokens,
                images,
                kv_transfer_params,
                ..
            } => {
                assert_eq!(*reason, FinishReason::Eos);
                assert_eq!((*prompt_tokens, *completion_tokens, *images), (3, 1, 1));
                assert_eq!(
                    kv_transfer_params.as_ref(),
                    Some(&serde_json::json!({"connector": "x"}))
                );
            }
            other => panic!("expected Finished, got {other:?}"),
        }
    }
}
