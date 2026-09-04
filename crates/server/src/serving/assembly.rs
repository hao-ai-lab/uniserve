//! Assembly of engine events into public serving events.
//!
//! The assembler applies model-selected output processing, tracks usage, and
//! emits exactly one terminal result for each accepted request.

use super::*;

/// Runtime output sink built from the model-supplied [`OutputProcessorPolicy`].
enum OutputSink {
    /// Raw visible text.
    Raw,
    /// SenseNova reasoning and visible-answer filtering.
    SenseNova(SenseNovaOutputProcessor),
}

/// Constructs the model-selected semantic output processor for one request.
fn build_output_sink(
    request_id: &ServeRequestId,
    policy: OutputProcessorPolicy,
    tokenizer: &crate::serving::text::tokenizer::DynTokenizer,
    prompt_token_ids: &[u32],
) -> Result<OutputSink> {
    match policy {
        OutputProcessorPolicy::None | OutputProcessorPolicy::Bagel => Ok(OutputSink::Raw),
        OutputProcessorPolicy::Qwen3(_) => unreachable!("Qwen3 uses the pull chat pipeline"),
        OutputProcessorPolicy::SenseNova(output_filter) => {
            let processor = SenseNovaOutputProcessor::new(
                output_filter,
                std::sync::Arc::clone(tokenizer),
                prompt_token_ids,
            )
            .map_err(|error| ServeError::OutputProcessing {
                request_id: request_id.clone(),
                source: OutputProcessingError::Reasoning(error),
            })?;
            Ok(OutputSink::SenseNova(processor))
        }
    }
}

struct ChatDone {
    prompt_token_count: usize,
    output_token_count: usize,
    visible_output_token_count: usize,
    internal_token_count: usize,
    finish_reason: FinishReason,
}

enum MappedChatEvent {
    Ignore,
    Event(ServeEvent),
    Done(ChatDone),
}

struct EmitContext<'a> {
    request_id: &'a ServeRequestId,
    event: &'a EventContext,
    started: &'a Instant,
}

struct TerminalAccounting {
    queue_us: Option<u64>,
    first_visible_output_us: Option<u64>,
    image_count: u32,
    image_steps: u32,
}

/// Maps one structured chat event into the transport-neutral serving event model.
fn map_chat_event(event: ChatEvent) -> MappedChatEvent {
    match event {
        // Start metadata is consumed by the assembly loop before mapping.
        ChatEvent::Start { .. } => MappedChatEvent::Ignore,
        // Assistant content blocks preserve their processor-assigned indices.
        ChatEvent::BlockStart { index, kind } => {
            MappedChatEvent::Event(ServeEvent::OutputBlockStart {
                candidate_id: CandidateId::PRIMARY,
                index,
                kind,
            })
        }
        ChatEvent::BlockDelta { kind, delta, .. } => match kind {
            AssistantBlockKind::Text => MappedChatEvent::Event(ServeEvent::TextDelta {
                candidate_id: CandidateId::PRIMARY,
                text: delta,
                token_ids: Vec::new(),
                logprobs: None,
            }),
            AssistantBlockKind::Reasoning => MappedChatEvent::Event(ServeEvent::ReasoningDelta {
                candidate_id: CandidateId::PRIMARY,
                text: delta,
            }),
            AssistantBlockKind::ToolCall => MappedChatEvent::Ignore,
        },
        // Logprobs travel independently from decoded text so either surface can
        // be suppressed without altering token accounting.
        ChatEvent::LogprobsDelta {
            token_ids,
            logprobs,
        } => MappedChatEvent::Event(ServeEvent::TextDelta {
            candidate_id: CandidateId::PRIMARY,
            text: String::new(),
            token_ids,
            logprobs,
        }),
        ChatEvent::BlockEnd { index, block } => {
            MappedChatEvent::Event(ServeEvent::OutputBlockEnd {
                candidate_id: CandidateId::PRIMARY,
                index,
                block,
            })
        }
        // Tool calls use explicit start, argument, and terminal events so
        // streaming consumers can assemble structured calls incrementally.
        ChatEvent::ToolCallStart { index, id, name } => {
            MappedChatEvent::Event(ServeEvent::ToolCallStart {
                candidate_id: CandidateId::PRIMARY,
                index,
                id,
                name,
            })
        }
        ChatEvent::ToolCallArgumentsDelta { index, delta } => {
            MappedChatEvent::Event(ServeEvent::ToolCallArgumentsDelta {
                candidate_id: CandidateId::PRIMARY,
                index,
                delta,
            })
        }
        ChatEvent::ToolCallEnd { index, call } => MappedChatEvent::Event(ServeEvent::ToolCallEnd {
            candidate_id: CandidateId::PRIMARY,
            index,
            id: call.id,
            name: call.name,
            arguments: call.arguments,
        }),
        // Terminal processor accounting becomes the common serving terminal.
        ChatEvent::Done {
            prompt_token_count,
            output_token_count,
            visible_output_token_count,
            internal_token_count,
            finish_reason,
            ..
        } => MappedChatEvent::Done(ChatDone {
            prompt_token_count,
            output_token_count,
            visible_output_token_count,
            internal_token_count,
            finish_reason,
        }),
    }
}

/// Applies text filtering and emits visible, reasoning, and terminal updates in order.
async fn emit_text_update(
    context: &EmitContext<'_>,
    sink: &mut OutputSink,
    text: String,
    token_ids: Vec<u32>,
    logprobs: Option<DecodedLogprobs>,
    finished: Option<crate::serving::text::Finished>,
    first_visible_output_us: &mut Option<u64>,
    y: &mut TryYielder<ServeEvent, ServeError>,
) -> Result<Option<ChatDone>> {
    let delta: SenseNovaTextDelta = match sink {
        OutputSink::SenseNova(processor) => processor.push(&text),
        OutputSink::Raw => SenseNovaTextDelta {
            visible: text,
            reasoning: String::new(),
        },
    };

    if !delta.reasoning.is_empty() {
        first_visible_output_us.get_or_insert_with(|| context.started.elapsed().as_micros() as u64);
        y.yield_ok(ServeEvent::ReasoningDelta {
            candidate_id: CandidateId::PRIMARY,
            text: delta.reasoning,
        })
        .await;
    }
    if !delta.visible.is_empty() {
        first_visible_output_us.get_or_insert_with(|| context.started.elapsed().as_micros() as u64);
    }
    if !delta.visible.is_empty()
        || !token_ids.is_empty()
        || logprobs.is_some()
        || finished.is_some()
    {
        y.yield_ok(ServeEvent::TextDelta {
            candidate_id: CandidateId::PRIMARY,
            text: delta.visible,
            token_ids,
            logprobs,
        })
        .await;
    }
    Ok(finished.map(|finished| ChatDone {
        prompt_token_count: finished.prompt_token_count,
        output_token_count: finished.output_token_count,
        visible_output_token_count: finished
            .output_token_count
            .saturating_sub(finished.internal_token_count),
        internal_token_count: finished.internal_token_count,
        finish_reason: finished.finish_reason,
    }))
}

/// Emits final usage followed by exactly one terminal serving event.
async fn emit_terminal(
    context: &EmitContext<'_>,
    accounting: TerminalAccounting,
    finish_detail: Option<String>,
    done: ChatDone,
    y: &mut TryYielder<ServeEvent, ServeError>,
) {
    debug_assert_eq!(
        done.output_token_count,
        done.visible_output_token_count
            .saturating_add(done.internal_token_count)
    );
    y.yield_ok(ServeEvent::Usage {
        prompt_tokens: done.prompt_token_count.min(u32::MAX as usize) as u32,
        visible_output_tokens: done.visible_output_token_count.min(u32::MAX as usize) as u32,
        internal_tokens: done.internal_token_count.min(u32::MAX as usize) as u32,
        image_count: accounting.image_count,
        image_steps: accounting.image_steps,
        cache: context.event.cache.clone(),
        resources: context.event.resources.clone(),
        timings: RuntimeTimings {
            compile_us: context.event.compile_duration_us,
            queue_us: accounting.queue_us,
            first_visible_output_us: accounting.first_visible_output_us,
            total_us: context.started.elapsed().as_micros() as u64,
        },
    })
    .await;
    match done.finish_reason.reason() {
        uniserve_core::FinishReason::Cancelled => {
            y.yield_ok(ServeEvent::Cancelled {
                request_id: context.request_id.clone(),
            })
            .await;
        }
        uniserve_core::FinishReason::Aborted => {
            y.yield_ok(ServeEvent::Aborted {
                request_id: context.request_id.clone(),
            })
            .await;
        }
        uniserve_core::FinishReason::Error => {
            y.yield_ok(ServeEvent::Failed {
                request_id: context.request_id.clone(),
                message: finish_detail.unwrap_or_else(|| "engine execution failed".to_string()),
            })
            .await;
        }
        _ => {
            y.yield_ok(ServeEvent::Finished {
                candidate_id: CandidateId::PRIMARY,
                reason: FinishStatus::from(&done.finish_reason),
                finish_detail,
            })
            .await;
        }
    }
}

#[try_stream]
/// Applies chat output processing and assembles public serving events.
pub(super) async fn assemble_chat_event_stream(
    assembly: StreamInput,
    processor: Qwen3ChatOutputProcessor,
    mut y: TryYielder<ServeEvent, ServeError>,
) -> Result<()> {
    let StreamInput {
        request_id,
        event_context,
        prompt_token_ids,
        tokenizer,
        prompt_logprobs_requested,
        generated_logprobs_requested,
        emit_token_ids,
        decode_options,
        stream,
    } = assembly;

    // Decode raw engine events first, then apply the model-selected structured
    // chat processor before exposing any public event.
    let started = Instant::now();
    let emit_context = EmitContext {
        request_id: &request_id,
        event: &event_context,
        started: &started,
    };
    let decoded = crate::serving::text::output::decoded_text_event_stream(
        request_id.to_string(),
        tokenizer,
        prompt_token_ids,
        prompt_logprobs_requested,
        generated_logprobs_requested,
        stream,
        decode_options,
        true,
    );
    let output = processor
        .process(decoded)
        .map_err(|error| ServeError::OutputProcessing {
            request_id: request_id.clone(),
            source: OutputProcessingError::Chat(error),
        })?;
    futures::pin_mut!(output);

    // A start event establishes prompt metadata and must precede every output
    // block, token delta, tool call, and terminal result.
    let mut accepted = false;
    let mut queue_us = None;
    let mut first_visible_output_us = None;
    while let Some(next) = output.next().await {
        let event = next.map_err(|error| ServeError::OutputProcessing {
            request_id: request_id.clone(),
            source: OutputProcessingError::Chat(error),
        })?;
        let mut event = match event {
            ChatEvent::Start {
                prompt_token_ids,
                prompt_logprobs,
                queued_at,
                scheduled_at,
            } => {
                if accepted {
                    return Err(ServeError::OutputProcessing {
                        request_id: request_id.clone(),
                        source: OutputProcessingError::Malformed(
                            "chat output processor emitted duplicate start metadata".to_string(),
                        ),
                    });
                }
                y.yield_ok(ServeEvent::Accepted {
                    request_id: request_id.clone(),
                    served_name: event_context.served_name.clone(),
                    description: event_context.description.clone(),
                    compile_duration_us: event_context.compile_duration_us,
                    prompt_token_count: prompt_token_ids.len(),
                    prompt_token_ids: prompt_token_ids.to_vec(),
                    prompt_logprobs,
                })
                .await;
                if let (Some(queued_at), Some(scheduled_at)) = (queued_at, scheduled_at) {
                    queue_us = Some(((scheduled_at - queued_at).max(0.0) * 1_000_000.0) as u64);
                    y.yield_ok(ServeEvent::Scheduled {
                        request_id: request_id.clone(),
                        queued_at: Some(queued_at),
                        scheduled_at: Some(scheduled_at),
                        cache: event_context.cache.clone(),
                        resources: event_context.resources.clone(),
                    })
                    .await;
                    event_context
                        .metrics
                        .scheduled
                        .fetch_add(1, Ordering::Relaxed);
                }
                accepted = true;
                continue;
            }
            event => event,
        };

        if !accepted {
            return Err(ServeError::OutputProcessing {
                request_id: request_id.clone(),
                source: OutputProcessingError::Malformed(
                    "chat output arrived before start metadata".to_string(),
                ),
            });
        }

        // Token identifiers are removed at the final public boundary while the
        // processor retains their internal accounting.
        if let ChatEvent::LogprobsDelta { token_ids, .. } = &mut event
            && !emit_token_ids
        {
            token_ids.clear();
        }

        match map_chat_event(event) {
            MappedChatEvent::Ignore => {}
            MappedChatEvent::Event(event) => {
                let visible = matches!(&event, ServeEvent::TextDelta { text, .. } if !text.is_empty())
                    || matches!(&event, ServeEvent::ReasoningDelta { text, .. } if !text.is_empty())
                    || matches!(&event, ServeEvent::ToolCallStart { .. });
                if visible {
                    first_visible_output_us
                        .get_or_insert_with(|| started.elapsed().as_micros() as u64);
                }
                y.yield_ok(event).await;
            }
            MappedChatEvent::Done(done) => {
                // The first processor terminal owns final metrics and is the
                // only successful exit from the assembled stream.
                let finish_detail =
                    generation_finish_detail(done.finish_reason.reason()).to_string();
                emit_terminal(
                    &emit_context,
                    TerminalAccounting {
                        queue_us,
                        first_visible_output_us,
                        image_count: 0,
                        image_steps: 0,
                    },
                    Some(finish_detail),
                    done,
                    &mut y,
                )
                .await;
                return Ok(());
            }
        }
    }

    Err(ServeError::OutputProcessing {
        request_id,
        source: OutputProcessingError::Malformed(
            "chat output processor closed before terminal output".to_string(),
        ),
    })
}

struct RawAssemblerState {
    first_visible_output_us: Option<u64>,
    queue_us: Option<u64>,
    emitted_output_tokens: u32,
    image_count: u32,
    image_steps: u32,
    prompt_positions: Vec<uniserve_core::PositionLogprobs>,
    accepted: bool,
    pending_scheduled: Option<(f64, f64)>,
    pending_token: Option<u32>,
    pending_image_events: Vec<ServeEvent>,
    sink: OutputSink,
}

struct RawTokenEmitContext<'a, 'tokenizer> {
    emit: &'a EmitContext<'a>,
    prompt_token_ids: &'a [u32],
    emit_token_ids: bool,
    decode_options: &'a mut TextDecodeOptions,
    decoder: &'a mut crate::profile::tokenizer::IncrementalDecoder<'tokenizer>,
    stream: &'a mut EventRx,
    y: &'a mut TryYielder<ServeEvent, ServeError>,
}

impl RawAssemblerState {
    /// Emits acceptance metadata and any scheduling event that arrived before it.
    async fn emit_accepted(
        &mut self,
        request_id: &ServeRequestId,
        event_context: &EventContext,
        prompt_token_ids: &[u32],
        prompt_logprobs: Option<DecodedPromptLogprobs>,
        y: &mut TryYielder<ServeEvent, ServeError>,
    ) {
        y.yield_ok(ServeEvent::Accepted {
            request_id: request_id.clone(),
            served_name: event_context.served_name.clone(),
            description: event_context.description.clone(),
            compile_duration_us: event_context.compile_duration_us,
            prompt_token_count: prompt_token_ids.len(),
            prompt_token_ids: prompt_token_ids.to_vec(),
            prompt_logprobs,
        })
        .await;
        self.accepted = true;
        if let Some((queued_at, scheduled_at)) = self.pending_scheduled.take() {
            y.yield_ok(ServeEvent::Scheduled {
                request_id: request_id.clone(),
                queued_at: Some(queued_at),
                scheduled_at: Some(scheduled_at),
                cache: event_context.cache.clone(),
                resources: event_context.resources.clone(),
            })
            .await;
        }
    }

    /// Flushes the pending images.
    async fn flush_pending_images(&mut self, y: &mut TryYielder<ServeEvent, ServeError>) {
        for event in self.pending_image_events.drain(..) {
            y.yield_ok(event).await;
        }
    }

    /// Ensures the output ready.
    fn ensure_output_ready(
        &self,
        request_id: &ServeRequestId,
        event_kind: &'static str,
    ) -> Result<()> {
        if self.accepted && self.pending_token.is_none() {
            Ok(())
        } else {
            Err(malformed_output(
                request_id.clone(),
                format!("{event_kind} arrived before output metadata was complete"),
            ))
        }
    }

    /// Decodes one committed token while enforcing stop strings and output ordering.
    async fn consume_token(
        &mut self,
        id: u32,
        logprobs: Option<DecodedLogprobs>,
        context: &mut RawTokenEmitContext<'_, '_>,
    ) -> Result<bool> {
        // Image events generated before this text token remain ordered ahead of
        // the token at the public serving boundary.
        self.flush_pending_images(context.y).await;
        self.emitted_output_tokens = self.emitted_output_tokens.saturating_add(1);

        let new_bytes =
            context
                .decoder
                .push_token(id)
                .map_err(|error| ServeError::OutputProcessing {
                    request_id: context.emit.request_id.clone(),
                    source: OutputProcessingError::Tokenizer(error),
                })?;

        // Stop matching begins only after the minimum output length and checks
        // the newly decoded suffix, including cross-token boundaries.
        let matched_stop = if self.emitted_output_tokens > context.decode_options.min_tokens {
            context
                .decode_options
                .stop_strings
                .as_ref()
                .and_then(|stops| {
                    crate::serving::text::output::matches_stop_string(
                        stops,
                        context.decoder.output(),
                        new_bytes,
                    )
                })
        } else {
            None
        };

        // A matched stop flushes held-back decoder bytes at the exact requested
        // inclusion boundary; ordinary tokens emit only the next safe chunk.
        let (text, stop_string) = if let Some((index, offset)) = matched_stop {
            let stop_string = context
                .decode_options
                .stop_strings
                .as_mut()
                .ok_or_else(|| {
                    malformed_output(
                        context.emit.request_id.clone(),
                        "stop-string match lost its configured stop set",
                    )
                })?
                .swap_remove(index);
            let truncate_to = if context.decode_options.include_stop_str_in_output {
                offset + stop_string.len()
            } else {
                offset
            };
            let (last_chunk, _) = context.decoder.flush(Some(truncate_to)).map_err(|error| {
                ServeError::OutputProcessing {
                    request_id: context.emit.request_id.clone(),
                    source: OutputProcessingError::Tokenizer(error),
                }
            })?;
            (last_chunk.unwrap_or_default(), Some(stop_string))
        } else {
            (context.decoder.next_chunk().unwrap_or_default(), None)
        };

        let finished = stop_string
            .as_ref()
            .map(|stop_string| crate::serving::text::Finished {
                prompt_token_count: context.prompt_token_ids.len(),
                output_token_count: self.emitted_output_tokens as usize,
                internal_token_count: 0,
                finish_reason: FinishReason::with_stop_reason(
                    uniserve_core::FinishReason::Stop,
                    Some(StopReason::Text(stop_string.clone())),
                ),
            });

        // Acknowledgement advances the engine's safe public prefix. A matched
        // stop instead closes generation at the consumed token boundary.
        if stop_string.is_none() {
            context.stream.acknowledge_consumed_prefix();
        } else {
            context
                .stream
                .cancel_at_consumed_prefix(StreamCancelCause::StopStringMatched);
        }

        let emitted_ids = if context.emit_token_ids {
            vec![id]
        } else {
            Vec::new()
        };

        let done = emit_text_update(
            context.emit,
            &mut self.sink,
            text,
            emitted_ids,
            logprobs,
            finished,
            &mut self.first_visible_output_us,
            context.y,
        )
        .await?;

        if stop_string.is_none() {
            return Ok(false);
        }

        // Stop matching must synchronously produce the terminal accounting event.
        let done = done.ok_or_else(|| {
            malformed_output(
                context.emit.request_id.clone(),
                "output processor omitted the stop-string terminal event",
            )
        })?;
        emit_terminal(
            context.emit,
            TerminalAccounting {
                queue_us: self.queue_us,
                first_visible_output_us: self.first_visible_output_us,
                image_count: self.image_count,
                image_steps: self.image_steps,
            },
            Some("stop".to_string()),
            done,
            context.y,
        )
        .await;
        Ok(true)
    }
}

#[try_stream]
/// Assembles model output according to the request's output policy.
///
/// Engine protocol events are validated and correlated before becoming public;
/// malformed ordering terminates the stream with an output-processing error.
pub(super) async fn assemble_event_stream(
    assembly: StreamInput,
    output_processor: OutputProcessorPolicy,
    mut y: TryYielder<ServeEvent, ServeError>,
) -> Result<()> {
    let StreamInput {
        request_id,
        event_context,
        prompt_token_ids,
        tokenizer,
        prompt_logprobs_requested,
        generated_logprobs_requested,
        emit_token_ids,
        mut decode_options,
        mut stream,
    } = assembly;

    let started = Instant::now();
    let emit_context = EmitContext {
        request_id: &request_id,
        event: &event_context,
        started: &started,
    };

    let expected_prompt_positions = prompt_token_ids.len().saturating_sub(1);
    let sink = build_output_sink(&request_id, output_processor, &tokenizer, &prompt_token_ids)?;

    let mut state = RawAssemblerState {
        first_visible_output_us: None,
        queue_us: None,
        emitted_output_tokens: 0,
        image_count: 0,
        image_steps: 0,
        prompt_positions: Vec::new(),
        accepted: false,
        pending_scheduled: None,
        pending_token: None,
        pending_image_events: Vec::new(),
        sink,
    };

    let mut decoder = tokenizer.create_decode_stream(
        &prompt_token_ids,
        decode_options.skip_special_tokens,
        stop_string_holdback_bytes(&decode_options),
    );

    // Acceptance waits for prompt scores when the public response promises them.
    if !prompt_logprobs_requested || expected_prompt_positions == 0 {
        let prompt_logprobs = if prompt_logprobs_requested {
            let first_token_id = prompt_token_ids.first().copied().ok_or_else(|| {
                malformed_output(
                    request_id.clone(),
                    "prompt logprobs require a non-empty tokenized prompt",
                )
            })?;
            let first_token = tokenizer
                .decode(&[first_token_id], event_context.skip_special_tokens)
                .map_err(|error| ServeError::OutputProcessing {
                    request_id: request_id.clone(),
                    source: OutputProcessingError::Tokenizer(error),
                })?;
            Some(DecodedPromptLogprobs {
                first_token_id,
                first_token,
                scored_positions: Vec::new(),
            })
        } else {
            None
        };
        state
            .emit_accepted(
                &request_id,
                &event_context,
                &prompt_token_ids,
                prompt_logprobs,
                &mut y,
            )
            .await;
    }

    while let Some(event) = stream.next().await {
        match event {
            Event::Scheduled {
                queued_at,
                scheduled_at,
            } => {
                state.queue_us = Some(((scheduled_at - queued_at).max(0.0) * 1_000_000.0) as u64);
                if state.accepted {
                    y.yield_ok(ServeEvent::Scheduled {
                        request_id: request_id.clone(),
                        queued_at: Some(queued_at),
                        scheduled_at: Some(scheduled_at),
                        cache: event_context.cache.clone(),
                        resources: event_context.resources.clone(),
                    })
                    .await;
                } else {
                    state.pending_scheduled = Some((queued_at, scheduled_at));
                }

                event_context
                    .metrics
                    .scheduled
                    .fetch_add(1, Ordering::Relaxed);
            }
            Event::PromptLogprobs { positions } => {
                state.prompt_positions.extend(positions);
                if state.prompt_positions.len() > expected_prompt_positions {
                    return Err(malformed_output(
                        request_id.clone(),
                        "engine returned more prompt logprob positions than requested",
                    ));
                }

                if !state.accepted && state.prompt_positions.len() == expected_prompt_positions {
                    let positions = std::mem::take(&mut state.prompt_positions);
                    let decoded = crate::serving::text::output::decode_prompt_logprobs(
                        &request_id,
                        tokenizer.as_ref(),
                        &prompt_token_ids,
                        &positions,
                        event_context.skip_special_tokens,
                    )
                    .map_err(|error| ServeError::OutputProcessing {
                        request_id: request_id.clone(),
                        source: OutputProcessingError::Text(error),
                    })?;
                    state
                        .emit_accepted(
                            &request_id,
                            &event_context,
                            &prompt_token_ids,
                            Some(decoded),
                            &mut y,
                        )
                        .await;
                }
            }
            Event::TextToken { id, .. } => {
                if !state.accepted {
                    return Err(malformed_output(
                        request_id.clone(),
                        "engine began generation before prompt logprobs were complete",
                    ));
                }
                if state.pending_token.is_some() {
                    return Err(malformed_output(
                        request_id.clone(),
                        "engine emitted a new token before resolving prior logprobs",
                    ));
                }

                // Generated token scores arrive as a separate event and must be
                // paired before the token enters incremental decoding.
                if generated_logprobs_requested {
                    state.pending_token = Some(id);
                } else {
                    let mut context = RawTokenEmitContext {
                        emit: &emit_context,
                        prompt_token_ids: &prompt_token_ids,
                        emit_token_ids,
                        decode_options: &mut decode_options,
                        decoder: &mut decoder,
                        stream: &mut stream,
                        y: &mut y,
                    };
                    if state.consume_token(id, None, &mut context).await? {
                        return Ok(());
                    }
                }
            }
            Event::TokenLogprobs { id, candidates } => {
                let pending = state.pending_token.take().ok_or_else(|| {
                    malformed_output(
                        request_id.clone(),
                        "engine returned token logprobs without a pending token",
                    )
                })?;
                if pending != id || candidates.first().is_none_or(|entry| entry.token_id != id) {
                    return Err(malformed_output(
                        request_id.clone(),
                        "token logprobs do not match the emitted token",
                    ));
                }
                let logprobs = crate::serving::text::output::decode_logprobs(
                    tokenizer.as_ref(),
                    &[uniserve_core::PositionLogprobs {
                        entries: candidates,
                    }],
                    event_context.skip_special_tokens,
                )
                .map_err(|error| ServeError::OutputProcessing {
                    request_id: request_id.clone(),
                    source: OutputProcessingError::Text(error),
                })?;

                let mut context = RawTokenEmitContext {
                    emit: &emit_context,
                    prompt_token_ids: &prompt_token_ids,
                    emit_token_ids,
                    decode_options: &mut decode_options,
                    decoder: &mut decoder,
                    stream: &mut stream,
                    y: &mut y,
                };
                if state
                    .consume_token(id, Some(logprobs), &mut context)
                    .await?
                {
                    return Ok(());
                }
            }
            Event::ImageBegin {
                image_id,
                height,
                width,
                steps,
            } => {
                state.ensure_output_ready(&request_id, "image-begin event")?;
                state
                    .first_visible_output_us
                    .get_or_insert_with(|| started.elapsed().as_micros() as u64);
                y.yield_ok(ServeEvent::ImageBegin {
                    candidate_id: CandidateId::PRIMARY,
                    image_id: image_id.to_string(),
                    width: Some(width),
                    height: Some(height),
                    steps: Some(steps as u32),
                    elapsed_us: started.elapsed().as_micros() as u64,
                })
                .await;
            }
            Event::ImageStep { image_id, step } => {
                state.ensure_output_ready(&request_id, "image-step event")?;
                state.image_steps = state.image_steps.saturating_add(1);
                y.yield_ok(ServeEvent::ImageStep {
                    candidate_id: CandidateId::PRIMARY,
                    image_id: image_id.to_string(),
                    step: step as u32,
                    elapsed_us: started.elapsed().as_micros() as u64,
                })
                .await;
            }
            Event::ImageCommit { image_id } => {
                state.ensure_output_ready(&request_id, "image-commit event")?;

                // Commit and payload publication remain adjacent even when text
                // decoding interleaves with the engine's image lifecycle.
                state.pending_image_events.push(ServeEvent::ImageCommit {
                    candidate_id: CandidateId::PRIMARY,
                    image_id: image_id.to_string(),
                    elapsed_us: started.elapsed().as_micros() as u64,
                });
            }
            Event::ImageDone {
                image_id,
                height,
                width,
                bytes,
                sha256,
                pixels_png_b64,
            } => {
                state.ensure_output_ready(&request_id, "image-done event")?;
                state.image_count = state.image_count.saturating_add(1);

                state.pending_image_events.push(ServeEvent::ImageDone {
                    candidate_id: CandidateId::PRIMARY,
                    image_id: image_id.to_string(),
                    width: Some(width),
                    height: Some(height),
                    bytes: Some(bytes),
                    sha256: Some(sha256),
                    pixels_png_b64: Some(pixels_png_b64),
                    elapsed_us: started.elapsed().as_micros() as u64,
                });
            }
            Event::Finished {
                reason,
                stop_reason,
                prompt_tokens,
                completion_tokens,
                images,
            } => {
                state.ensure_output_ready(&request_id, "terminal event")?;
                state.flush_pending_images(&mut y).await;

                // Flush decoder holdback before constructing terminal usage.
                let (last_chunk, _) =
                    decoder
                        .flush(None)
                        .map_err(|error| ServeError::OutputProcessing {
                            request_id: request_id.clone(),
                            source: OutputProcessingError::Tokenizer(error),
                        })?;
                let finish_detail = generation_finish_detail(&reason).to_string();
                let finished = crate::serving::text::Finished {
                    prompt_token_count: prompt_tokens,
                    output_token_count: completion_tokens,
                    internal_token_count: completion_tokens
                        .saturating_sub(state.emitted_output_tokens as usize),
                    finish_reason: generation_text_finish_reason(reason, stop_reason),
                };

                let done = emit_text_update(
                    &emit_context,
                    &mut state.sink,
                    last_chunk.unwrap_or_default(),
                    Vec::new(),
                    None,
                    Some(finished),
                    &mut state.first_visible_output_us,
                    &mut y,
                )
                .await?
                .ok_or_else(|| {
                    malformed_output(
                        request_id.clone(),
                        "output processor omitted the engine terminal event",
                    )
                })?;

                state.image_count = state.image_count.max(images.min(u32::MAX as usize) as u32);
                emit_terminal(
                    &emit_context,
                    TerminalAccounting {
                        queue_us: state.queue_us,
                        first_visible_output_us: state.first_visible_output_us,
                        image_count: state.image_count,
                        image_steps: state.image_steps,
                    },
                    Some(finish_detail),
                    done,
                    &mut y,
                )
                .await;
                return Ok(());
            }
            Event::Rejected { message } => {
                state.flush_pending_images(&mut y).await;
                y.yield_ok(ServeEvent::Rejected {
                    request_id: request_id.clone(),
                    message,
                })
                .await;
                return Ok(());
            }
            Event::Error { message } => {
                state.flush_pending_images(&mut y).await;
                y.yield_ok(ServeEvent::Failed {
                    request_id: request_id.clone(),
                    message,
                })
                .await;
                return Ok(());
            }
            Event::Artifact(_) => {
                return Err(malformed_output(
                    request_id,
                    "generation request received a media lifecycle event",
                ));
            }
        }
    }

    // A clean transport close without a protocol terminal is still malformed.
    state.flush_pending_images(&mut y).await;
    Err(malformed_output(
        request_id,
        "engine stream closed before a terminal event",
    ))
}

/// Returns the text finish reason from a generation event.
fn generation_text_finish_reason(
    reason: uniserve_core::FinishReason,
    stop_reason: Option<uniserve_core::StopReason>,
) -> FinishReason {
    if reason == uniserve_core::FinishReason::Stop {
        return FinishReason::with_stop_reason(
            reason,
            stop_reason.map(|reason| match reason {
                uniserve_core::StopReason::Token(id) => StopReason::TokenId(id),
                uniserve_core::StopReason::String(value) => StopReason::Text(value),
            }),
        );
    }
    FinishReason::new(reason)
}

/// Returns structured finish details from a generation event.
fn generation_finish_detail(reason: &uniserve_core::FinishReason) -> &'static str {
    match reason {
        uniserve_core::FinishReason::Completed => "completed",
        uniserve_core::FinishReason::Eos => "eos",
        uniserve_core::FinishReason::MaxTokens => "max_tokens",
        uniserve_core::FinishReason::Stop => "stop",
        uniserve_core::FinishReason::ImageDone => "image_done",
        uniserve_core::FinishReason::Cancelled => "cancelled",
        uniserve_core::FinishReason::Aborted => "aborted",
        uniserve_core::FinishReason::Repetition => "repetition",
        uniserve_core::FinishReason::Error => "error",
    }
}
