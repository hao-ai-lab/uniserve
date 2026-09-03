use super::*;

impl EngineLoop {
    pub(super) fn enqueue(&mut self, req: GenerationRequest, event_tx: EventTx) {
        if self.memory.cache.is_none() {
            self.trace_record(json!({
                "event": "request_rejected",
                "at_s": now(),
                "request_id": req.request_id.0,
                "reason": "missing_kv_resources",
            }));
            let _ = event_tx.send(Event::Rejected {
                message: "generation request requires worker KV resources".into(),
            });
            return;
        }
        let context = match RuntimeContext::lower(&req) {
            Ok(context) => context,
            Err(error) => {
                self.trace_record(json!({
                    "event": "request_rejected",
                    "at_s": now(),
                    "request_id": req.request_id.0,
                    "reason": "invalid_request",
                    "detail": format!("{error:?}"),
                }));
                let _ = event_tx.send(Event::Rejected {
                    message: format!("invalid generation request: {error:?}"),
                });
                return;
            }
        };
        if let Some(feature) = self.missing_required_feature(&req) {
            self.trace_record(json!({
                "event": "request_rejected",
                "at_s": now(),
                "request_id": req.request_id.0,
                "reason": "missing_worker_feature",
                "detail": format!("{feature}"),
            }));
            let _ = event_tx.send(Event::Rejected {
                message: format!(
                    "generation request requires worker feature `{feature}`, but the worker does not support it"
                ),
            });
            return;
        }
        if let Err(error) = req.validate_resources(&self.profile.generation_limits) {
            self.trace_record(json!({
                "event": "request_rejected",
                "at_s": now(),
                "request_id": req.request_id.0,
                "reason": "invalid_resource_declaration",
                "detail": error.to_string(),
            }));
            let _ = event_tx.send(Event::Rejected {
                message: format!("invalid generation resource declaration: {error}"),
            });
            return;
        }
        // Waiting-queue backpressure sheds load instead of letting the waiting
        // queue grow without bound under overload —
        // an unbounded burst would otherwise OOM the process and take down every
        // in-flight request. Reject the new submit with a typed event.
        let waiting = self.pending_request_count() + self.scheduler.output.retained_len();
        if waiting >= self.scheduler.config.max_num_waiting {
            self.trace_record(json!({
                "event": "request_rejected",
                "at_s": now(),
                "request_id": req.request_id.0,
                "reason": "queue_full",
                "waiting": waiting,
                "max_num_waiting": self.scheduler.config.max_num_waiting,
                "behavior": {
                    "und_decode": req.behavior.und_decode,
                    "und_tokens": format!("{:?}", req.behavior.und_tokens),
                    "gen_output": req.behavior.gen_output,
                    "generated_image_feedback": req.behavior.generated_image_feedback,
                },
                "prompt_tokens": context.prompt_ids.len(),
            }));
            let _ = event_tx.send(Event::Rejected {
                message: "scheduler waiting queue is full".into(),
            });
            return;
        }
        let worst = req
            .resources
            .max_kv_tokens
            .div_ceil(self.info.kv_block_size() as usize);
        // Multimodal requests reserve their configured bounded KV envelope at
        // admission so excess concurrency queues instead of exhausting KV.
        let reserve_worstcase = !context.images.is_empty() || req.behavior.gen_output;
        // A request with staged images usually encodes them before prefill.
        // Context-image requests prefill the text before each image position,
        // then encode the image into that marker gap.
        let phase0 = Phase::Prefill;
        let finish_token_ids = finish_token_ids(&req, &self.ctrl.eos);
        let st = ReqState {
            finish_token_ids,
            allocations: None,
            flow_prefix: None,
            epoch: self.next_epoch,
            version: 0,
            resolved_producer_op_id: 0,
            committed_version: 0,
            committed_producer_op_id: 0,
            control_seq: 0,
            public_event_limit: 0,
            token_cutoffs: BTreeMap::new(),
            pending_commits: VecDeque::new(),
            cancel_cutoff: None,
            latest_device_version: None,
            speculative_chain_invalidated: false,
            cursor: GenerationCursor::new(phase0, worst, reserve_worstcase),
            context,
            output: RequestOutput::new(event_tx),
            queued_at: now(),
            terminal_intent: super::TerminalIntent::None,
            req,
        };
        self.next_epoch = self.next_epoch.saturating_add(1);
        self.trace_request_queued(&st, "pending");
        let request_id = st.req.request_id;
        self.scheduler
            .enqueue(request_id, st.req.priority, st.queued_at);
        self.waiting.insert(request_id, st);
    }

    pub(super) fn enqueue_media(&mut self, submission: PendingMedia) {
        let request = &submission.request;
        if let Err(message) = request.validate() {
            let _ = submission.event_tx.send(Event::Rejected {
                message: message.to_string(),
            });
            return;
        }
        let required = [
            OpKind::DiffusionPrepare,
            OpKind::DiffusionStep,
            OpKind::DiffusionDecode,
        ];
        if required
            .iter()
            .any(|variant| !self.info.supported_ops.contains(variant))
        {
            let _ = submission.event_tx.send(Event::Rejected {
                message: "worker does not support the fixed media flow".to_string(),
            });
            return;
        }
        if self.info.request_slots < 2
            || self.info.latent_pages < 3
            || self.info.latent_page_units == 0
        {
            let _ = submission.event_tx.send(Event::Rejected {
                message: "worker does not provide two resident media state slots".to_string(),
            });
            return;
        }
        if self.scheduler.waiting_len()
            + self.scheduler.waiting_media_len()
            + self.scheduler.output.retained_len()
            >= self.scheduler.config.max_num_waiting
        {
            let _ = submission.event_tx.send(Event::Rejected {
                message: "scheduler waiting queue is full".to_string(),
            });
            return;
        }
        let request_id = submission.request.request_id;
        self.waiting_media.insert(request_id, submission);
        self.scheduler.enqueue_media(request_id);
    }

    pub(super) fn admit_media(&mut self) {
        while self.running_request_count() < self.scheduler.config.max_num_seqs {
            let Some(id) = self.scheduler.pop_media() else {
                break;
            };
            let submission = self
                .waiting_media
                .remove(&id)
                .expect("scheduler media order names runtime state");
            let epoch = self.next_epoch;
            let request_key = RequestKey::new(self.authority_id, id, epoch);
            let Ok(request_slot) = self.memory.alloc(request_key, MemoryLayout::RequestSlot) else {
                self.waiting_media.insert(id, submission);
                self.scheduler.push_media_front(id);
                break;
            };
            let Ok(latent) = self.memory.alloc(
                request_key,
                MemoryLayout::Latent {
                    units: u64::from(self.info.latent_page_units),
                },
            ) else {
                self.memory.free(request_slot);
                self.waiting_media.insert(id, submission);
                self.scheduler.push_media_front(id);
                break;
            };
            let allocations = MediaAllocations {
                request_slot,
                latent,
            };
            let request_pool_idx = allocations.request_slot();
            self.next_epoch = self.next_epoch.saturating_add(1);
            let admission = NewRequest::new_media(
                request_key,
                request_pool_idx,
                DiffusionRequestParams {
                    prompt_token_ids: submission.request.prompt_token_ids.clone(),
                    seed: submission.request.seed,
                    geometry: MediaGeometry {
                        frame_count: submission.request.geometry.frame_count,
                        decode_units: submission.request.geometry.decode_units,
                        prompt_tokens: submission.request.geometry.prompt_tokens,
                        denoise_steps: submission.request.geometry.denoise_steps,
                    },
                },
            )
            .expect("validated media admission");
            let root = Checkpoint::admission_root(OpId(0));
            self.running_media.insert(
                id,
                MediaFlowState {
                    request: submission.request,
                    event_tx: submission.event_tx,
                    allocations,
                    admission,
                    admission_state: DiffusionRequestParamsState::Unsubmitted,
                    committed: MediaCursor::default(),
                    projected: MediaCursor::default(),
                    fixed_parent: root.clone(),
                    projected_parent: root,
                    terminal_intent: MediaTerminalIntent::None,
                    artifact: None,
                },
            );
            self.scheduler.running_order.push(id);
        }
    }

    pub(super) fn num_vae(&self, ip: &uniserve_core::ImageParams) -> u64 {
        let dl = u64::from(self.profile.generation_limits.latent_downsample).max(1);
        (ip.height as u64 / dl) * (ip.width as u64 / dl)
    }

    pub(super) fn cap_max_vae_grid_tokens(&self) -> usize {
        if self.profile.generation_limits.max_vae_grid_tokens > 0 {
            self.profile.generation_limits.max_vae_grid_tokens as usize
        } else {
            self.info.latent_capacity_units().min(usize::MAX as u64) as usize
        }
    }

    pub(super) fn missing_required_feature(
        &self,
        request: &GenerationRequest,
    ) -> Option<uniserve_core::GenerationFeatures> {
        let context_steps = request.context.iter().flat_map(|segment| match segment {
            uniserve_core::ContextSegment::Image { ingest, .. } => ingest.steps.clone(),
            uniserve_core::ContextSegment::UndTokens { .. } => Vec::new(),
        });
        let needs = request
            .behavior
            .required_features(&request.policy, context_steps);
        self.profile.generation_limits.covers(needs).err()
    }

    pub(super) fn worker_tracks_image_latent(&self) -> bool {
        self.info.latent_page_units > 0 && self.info.latent_pages > 1
    }

    pub(super) fn worker_image_latent_used(&self) -> u64 {
        (self.memory.latent_pages.used_pages() as u64)
            .saturating_mul(u64::from(self.info.latent_page_units))
    }

    pub(super) fn worker_image_latent_units_for(&self, st: &ReqState) -> u64 {
        let downsample = (self.profile.generation_limits.latent_downsample as u64).max(1);
        let (height, width) = (st.req.image.height, st.req.image.width);
        ceil_div_u64((height as u64).max(1), downsample)
            * ceil_div_u64((width as u64).max(1), downsample)
    }

    pub(super) fn ensure_flow_prefix(&mut self, id: RequestId) -> bool {
        let (prefix_tokens, needs_alternative) = self
            .running
            .get(&id)
            .map(|state| {
                (
                    state.context.negative_prompt_ids.len(),
                    cfg_branch_count(&state.req.image) > 1,
                )
            })
            .unwrap_or_default();
        if !needs_alternative
            || self
                .running
                .get(&id)
                .is_some_and(|state| state.flow_prefix.is_some())
        {
            return true;
        }
        let Some(state) = self.running.get(&id) else {
            return false;
        };
        let request_key = RequestKey::new(self.authority_id, id, state.epoch);
        let Ok(request_slot) = self.memory.alloc(request_key, MemoryLayout::RequestSlot) else {
            return false;
        };
        let groups = self.memory.cache().block_pool.num_groups() as u32;
        let Ok(kv) = self.memory.alloc(
            request_key,
            MemoryLayout::Kv {
                tokens: prefix_tokens as u32,
                groups,
            },
        ) else {
            self.memory.free(request_slot);
            return false;
        };
        let new_pages = kv
            .kv_tables()
            .expect("flow prefix KV allocation")
            .iter()
            .map(|table| (table.group_id() as u32, table.page_ids()))
            .collect();
        let allocations = RequestAllocations {
            request_slot,
            kv,
            latent: None,
            buffers: HashMap::new(),
        };
        let Some(state) = self.running.get_mut(&id) else {
            allocations.free(&mut self.memory);
            return false;
        };
        state.flow_prefix = Some(FlowPrefixState {
            allocations,
            new_pages,
            diffusion_finalized: false,
        });
        true
    }

    pub(super) fn free_flow_prefix(&mut self, id: RequestId) {
        let prefix = self
            .running
            .get_mut(&id)
            .and_then(|state| state.flow_prefix.take());
        if let Some(prefix) = prefix {
            prefix.allocations.free(&mut self.memory);
        }
    }

    /// One loop iteration of the schedule-ahead loop. Returns true if any work
    /// was submitted or any result resolved.
    pub(super) fn reap_cancellations(&mut self) {
        // A cancelled request closes only after every submitted descendant has
        // resolved. Host KV ownership then remains pinned through the close
        // acknowledgement and ordered worker request retirement.
        let cancelled: Vec<(RequestId, TerminalIntent)> = self
            .running
            .iter()
            .filter(|(id, state)| {
                state.terminal_intent.is_terminal() && !self.inflight.contains(**id)
            })
            .map(|(id, state)| (*id, state.terminal_intent))
            .collect();
        for (id, intent) in cancelled {
            let reason = match intent {
                TerminalIntent::StopMatched => FinishReason::Stop,
                TerminalIntent::Abort => FinishReason::Aborted,
                TerminalIntent::Cancel => FinishReason::Cancelled,
                TerminalIntent::None => continue,
            };
            self.finish(id, reason);
        }
        let media = self
            .scheduler
            .running_order
            .iter()
            .filter_map(|id| {
                self.media_state(*id)
                    .filter(|state| {
                        state.terminal_intent.is_terminal()
                            && !self.inflight.contains(state.request.request_id)
                    })
                    .map(|state| (state.request.request_id, state.terminal_intent.clone()))
            })
            .collect::<Vec<_>>();
        for (id, intent) in media {
            let (event, reason) = match intent {
                MediaTerminalIntent::Failure(message) => {
                    (DiffusionTerminal::Failed(message), CloseReason::Error)
                }
                MediaTerminalIntent::Finish(reason) => {
                    (DiffusionTerminal::Finished(reason), CloseReason::Cancelled)
                }
                MediaTerminalIntent::None => continue,
            };
            self.finish_media(id, event, reason, None);
        }
    }

    /// Admit: consume the waiting-queue head while budgets allow (vLLM's
    /// posture — head-of-line, `max_num_seqs`-capped). Reserving requests
    /// allocate their full worst-case KV here, which is what makes them
    /// resident for its complete lifetime.
    pub(super) fn admit(&mut self) {
        let bs = self.info.kv_block_size() as usize;
        loop {
            if self.running_request_count() >= self.scheduler.config.max_num_seqs
                || self.memory.request_slots.is_empty()
            {
                break;
            }
            let Some(head_id) = self.scheduler.peek() else {
                break;
            };
            let head = self
                .waiting
                .get(&head_id)
                .expect("scheduler waiting order names runtime state");
            if head.cursor.resources.reserve_worstcase {
                let need = head.cursor.resources.worstcase_blocks;
                let encoder_entries = head.req.resources.encoder_cache_keys.len();
                let encoder_ok = self
                    .memory
                    .reserved_encoder_entries
                    .saturating_add(encoder_entries)
                    <= self.memory.encoder_cache.budget();
                if need > self.memory.usable_blocks() {
                    let id = self.scheduler.pop().unwrap();
                    let st = self.waiting.remove(&id).unwrap();
                    self.trace_record(json!({
                        "event": "request_rejected",
                        "at_s": now(),
                        "request_id": st.req.request_id.0,
                        "reason": "too_large",
                        "needed_blocks": need,
                        "usable_blocks": self.memory.usable_blocks(),
                        "generation": &st.req.behavior,
                        "prompt_tokens": st.context.prompt_ids.len(),
                    }));
                    let _ = st.output.event_tx.send(Event::Rejected {
                        message: "request exceeds total KV capacity".into(),
                    });
                    continue;
                }
                if self.memory.free_blocks() >= need && encoder_ok {
                    let id = self.scheduler.pop().unwrap();
                    let st = self.waiting.remove(&id).unwrap();
                    let id = st.req.request_id;
                    self.admit_running(st);
                    // Physically allocate the worst case now: nothing can take
                    // these blocks, so this request can never fail mid-flight.
                    self.ensure_request_capacity(id, need * bs);
                    self.memory.reserved_blocks += need;
                    continue;
                }
            } else {
                let n = head.context.prompt_ids.len();
                let text_usable_blocks = (0..self.memory.cache().block_pool.num_groups())
                    .map(|group| self.memory.cache().block_pool.group_capacity(group))
                    .min()
                    .unwrap_or_default();
                let prefix_hit = prefix_hit(
                    &self.memory.cache().coordinator,
                    head,
                    &self.memory.cache().block_pool,
                );
                let cached_prefix_blocks = prefix_hit.cached_blocks;
                let cached_prefix_blocks = cached_prefix_blocks.min(n.div_ceil(bs));
                let cached_prefix_tokens = cached_prefix_blocks.saturating_mul(bs);
                let uncached_remaining = n.saturating_sub(cached_prefix_tokens);
                let first_uncached_chunk = uncached_remaining
                    .min(self.scheduler.config.long_prefill_threshold)
                    .min(self.scheduler.config.max_num_batched_tokens)
                    .max(1);
                let first_chunk_blocks = cached_prefix_tokens
                    .saturating_add(first_uncached_chunk)
                    .div_ceil(bs)
                    .saturating_sub(cached_prefix_blocks);
                if n > text_usable_blocks * bs {
                    let id = self.scheduler.pop().unwrap();
                    let st = self.waiting.remove(&id).unwrap();
                    self.trace_record(json!({
                        "event": "request_rejected",
                        "at_s": now(),
                        "request_id": st.req.request_id.0,
                        "reason": "too_large",
                        "needed_blocks": first_chunk_blocks,
                        "usable_blocks": text_usable_blocks,
                        "generation": &st.req.behavior,
                        "prompt_tokens": st.context.prompt_ids.len(),
                    }));
                    let _ = st.output.event_tx.send(Event::Rejected {
                        message: "request exceeds total KV capacity".into(),
                    });
                    continue;
                }
                let capacity_available = prefix_hit.cached_free_blocks.len()
                    == self.memory.cache().block_pool.num_groups()
                    && prefix_hit.cached_free_blocks.iter().enumerate().all(
                        |(group, cached_free)| {
                            self.memory
                                .cache()
                                .block_pool
                                .free_blocks_in_group(group)
                                .saturating_sub(*cached_free)
                                >= first_chunk_blocks
                        },
                    );
                if capacity_available {
                    let id = self.scheduler.pop().unwrap();
                    let st = self.waiting.remove(&id).unwrap();
                    self.admit_running(st);
                    continue;
                }
            }

            // Exact checkpoints retain their physical KV identity. Until the
            // configured route provides relocatable checkpoint storage, a
            // resident request remains non-preemptible and admission queues.
            break;
        }
    }

    pub(super) fn admit_running(&mut self, mut st: ReqState) {
        let id = st.req.request_id;
        let request_key = RequestKey::new(self.authority_id, id, st.epoch);
        let request_slot = self
            .memory
            .alloc(request_key, MemoryLayout::RequestSlot)
            .expect("admission checked request-slot capacity");
        let kv = self
            .memory
            .alloc(
                request_key,
                MemoryLayout::Kv {
                    tokens: 0,
                    groups: self.memory.cache().block_pool.num_groups() as u32,
                },
            )
            .expect("empty KV allocation is valid");
        st.allocations = Some(RequestAllocations {
            request_slot,
            kv,
            latent: None,
            buffers: HashMap::new(),
        });
        let q = st.queued_at;
        let scheduled_at = now();
        let queue_wait_us = ((scheduled_at - q).max(0.0) * 1_000_000.0) as u64;
        self.stats
            .timing
            .queue_wait_count
            .fetch_add(1, Ordering::Relaxed);
        self.stats
            .timing
            .queue_wait_us_total
            .fetch_add(queue_wait_us, Ordering::Relaxed);
        self.stats
            .timing
            .queue_wait_us_max
            .fetch_max(queue_wait_us, Ordering::Relaxed);
        let generation = st.req.behavior.clone();
        let phase = st.cursor.phase;
        let prompt_tokens = st.context.prompt_ids.len();
        let max_tokens = st.req.max_und_tokens;
        let priority = st.req.priority;
        let reserve_worstcase = st.cursor.resources.reserve_worstcase;
        let worstcase_blocks = st.cursor.resources.worstcase_blocks;
        let encoder_entries = st.req.resources.encoder_cache_keys.len();
        if st.output.enqueue(Event::Scheduled {
            queued_at: q,
            scheduled_at,
        }) {
            st.terminal_intent = TerminalIntent::Cancel;
        }
        self.running.insert(id, st);
        self.scheduler.running_order.push(id);
        self.memory.reserved_encoder_entries = self
            .memory
            .reserved_encoder_entries
            .saturating_add(encoder_entries);
        self.trace_record(json!({
            "event": "request_admitted",
            "at_s": scheduled_at,
            "request_id": id.0,
            "queued_at_s": q,
            "queue_wait_s": scheduled_at - q,
            "generation": generation,
            "phase": phase,
            "prompt_tokens": prompt_tokens,
            "max_tokens": max_tokens,
            "priority": priority,
            "reserve_worstcase": reserve_worstcase,
            "worstcase_blocks": worstcase_blocks,
            "running": self.running.len(),
            "pending": self.scheduler.waiting_len(),
            "free_blocks": self.memory.free_blocks(),
            "reserved_blocks": self.memory.reserved_blocks,
            "reserved_encoder_entries": self.memory.reserved_encoder_entries,
        }));
        let (runtime, memory) = (&mut self.runtime, &self.memory);
        if let Some(st) = runtime.state_mut().running.get_mut(&id) {
            let kv = memory.cache();
            acquire_cached_prefix(&kv.coordinator, st, &kv.block_pool, &self.stats);
        }
    }
}

fn prefix_hit(
    coordinator: &KvCacheCoordinator,
    state: &ReqState,
    pool: &BlockPool,
) -> crate::kv::PrefixHit {
    coordinator.probe_prefix(
        pool,
        state.effective_prompt(),
        state.req.cache.read,
        state.has_context_images(),
        state.req.cache.isolation_key,
    )
}

fn acquire_cached_prefix(
    coordinator: &KvCacheCoordinator,
    state: &mut ReqState,
    pool: &BlockPool,
    stats: &SchedStats,
) {
    let prompt = state.effective_prompt().to_vec();
    let has_context_images = state.has_context_images();
    let cache_read = state.req.cache.read;
    let isolation_key = state.req.cache.isolation_key;
    let hit = coordinator
        .acquire_prefix(
            pool,
            state.block_tables_mut(),
            &prompt,
            cache_read,
            has_context_images,
            isolation_key,
        )
        .expect("request cache groups match the KV coordinator");
    let block_size = pool.block_size();
    let full_blocks = prompt.len() / block_size;
    let query_blocks = if prompt.len().is_multiple_of(block_size) {
        full_blocks.saturating_sub(1)
    } else {
        full_blocks
    };
    stats
        .prefix
        .queries
        .fetch_add(query_blocks as u64, Ordering::Relaxed);
    stats
        .prefix
        .hits
        .fetch_add(hit.cached_blocks as u64, Ordering::Relaxed);
    stats
        .prefix
        .hit_tokens
        .fetch_add((hit.cached_blocks * block_size) as u64, Ordering::Relaxed);
    state.cursor.replay.block_hashes = hit.block_hashes;
    state.cursor.replay.prefix_cached_blocks = hit.cached_blocks;
    state.cursor.ingest.prompt_cursor = (hit.cached_blocks * block_size) as u32;
    state.cursor.und.logical_pos = state.cursor.ingest.prompt_cursor;
    state.cursor.und.physical_kv_len = state.cursor.ingest.prompt_cursor;
}
