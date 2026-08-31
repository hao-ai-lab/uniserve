use super::*;

impl Scheduler {
    pub(super) fn enqueue(&mut self, req: GenerationRequest, event_tx: EventTx) {
        if self.kv_budget.cache.is_none() {
            let _ = event_tx.send(GenerationEvent::Rejected {
                message: "generation request requires worker KV resources".into(),
            });
            return;
        }
        let context = match SchedulerContext::lower(&req) {
            Ok(context) => context,
            Err(error) => {
                let _ = event_tx.send(GenerationEvent::Rejected {
                    message: format!("invalid generation request: {error:?}"),
                });
                return;
            }
        };
        if let Some(feature) = self.missing_required_feature(&req) {
            let _ = event_tx.send(GenerationEvent::Rejected {
                message: format!(
                    "generation request requires worker feature `{feature}`, but the worker does not support it"
                ),
            });
            return;
        }
        if let Err(error) = req.validate_resources(&self.info.generation_limits()) {
            let _ = event_tx.send(GenerationEvent::Rejected {
                message: format!("invalid generation resource declaration: {error}"),
            });
            return;
        }
        // Waiting-queue backpressure sheds load instead of letting the waiting
        // queue grow without bound under overload —
        // an unbounded burst would otherwise OOM the process and take down every
        // in-flight request. Reject the new submit with a typed event.
        let waiting = self.pending_request_count() + self.output.retained_len();
        if waiting >= self.config.max_num_waiting {
            self.trace_record(json!({
                "event": "request_rejected",
                "at_s": now(),
                "request_id": req.request_id.0,
                "reason": "queue_full",
                "waiting": waiting,
                "max_num_waiting": self.config.max_num_waiting,
                "behavior": {
                    "und_decode": req.behavior.und_decode,
                    "und_tokens": format!("{:?}", req.behavior.und_tokens),
                    "gen_output": req.behavior.gen_output,
                    "generated_image_feedback": req.behavior.generated_image_feedback,
                },
                "prompt_tokens": context.prompt_ids.len(),
            }));
            let _ = event_tx.send(GenerationEvent::Rejected {
                message: "scheduler waiting queue is full".into(),
            });
            return;
        }
        let worst = req
            .resources
            .max_kv_tokens
            .div_ceil(self.info.block_size as usize);
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
            block_tables: (0..self.kv_budget.cache().block_pool.num_groups())
                .map(|group| BlockTable::new(group, self.info.block_size as usize))
                .collect(),
            flow_prefix: None,
            request_pool_idx: 0,
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
            cursor: GenerationCursor::new(phase0, worst, reserve_worstcase),
            context,
            output: RequestOutput::new(event_tx),
            queued_at: now(),
            terminal_intent: super::TerminalIntent::None,
            req,
        };
        self.next_epoch = self.next_epoch.saturating_add(1);
        self.trace_request_queued(&st, "pending");
        self.pending.add_request(st);
    }

    pub(super) fn enqueue_media(&mut self, submission: PendingMedia) {
        let request = &submission.request;
        if let Err(message) = request.validate() {
            let _ = submission.event_tx.send(MediaEvent::Rejected {
                message: message.to_string(),
            });
            return;
        }
        let required = [
            ForwardMode::GenTransition,
            ForwardMode::GenFlow,
            ForwardMode::GenDecode,
            ForwardMode::Materialize,
        ];
        if required
            .iter()
            .any(|variant| !self.info.supported_work.contains(variant))
        {
            let _ = submission.event_tx.send(MediaEvent::Rejected {
                message: "worker does not support the fixed media flow".to_string(),
            });
            return;
        }
        if self.info.max_request_pool_size < 2
            || self.info.num_latent_pages < 3
            || self.info.latent_page_units == 0
        {
            let _ = submission.event_tx.send(MediaEvent::Rejected {
                message: "worker does not provide two resident media state slots".to_string(),
            });
            return;
        }
        if self.pending.len() + self.pending_media.len() + self.output.retained_len()
            >= self.config.max_num_waiting
        {
            let _ = submission.event_tx.send(MediaEvent::Rejected {
                message: "scheduler waiting queue is full".to_string(),
            });
            return;
        }
        self.pending_media.push_back(submission);
    }

    pub(super) fn admit_media(&mut self) {
        while !self.kv_budget.request_slots.is_empty()
            && self.running_request_count() < self.config.max_num_seqs
        {
            let Some(submission) = self.pending_media.pop_front() else {
                break;
            };
            let id = submission.request.request_id;
            let Some(request_pool_idx) = self.kv_budget.request_slots.acquire() else {
                self.pending_media.push_front(submission);
                break;
            };
            if !self
                .kv_budget
                .latent_pages
                .reserve(id, u64::from(self.info.latent_page_units))
            {
                let _ = self.kv_budget.request_slots.release(request_pool_idx);
                self.pending_media.push_front(submission);
                break;
            }
            let epoch = self.next_epoch;
            self.next_epoch = self.next_epoch.saturating_add(1);
            let request_key = RequestKey::new(self.authority_id, id, epoch);
            let admission = NewRequest::new_media(
                request_key,
                request_pool_idx,
                MediaAdmission {
                    prompt_token_ids: submission.request.prompt_token_ids.clone(),
                    seed: submission.request.seed,
                    profile: MediaProfileId::MinimaxH3T2va,
                    output_path: submission.request.output_path.clone(),
                    plan: MediaPlan {
                        frame_count: submission.request.plan.frame_count,
                        video_decode_units: submission.request.plan.video_decode_units,
                        audio_latent_frames: submission.request.plan.audio_latent_frames,
                        prompt_tokens: submission.request.plan.prompt_tokens,
                        denoise_steps: submission.request.plan.denoise_steps,
                    },
                },
            )
            .expect("validated media admission");
            let root = VersionRef::admission_root(request_key, OpId(0));
            self.running_media.insert(
                id,
                MediaFlowState {
                    request: submission.request,
                    event_tx: submission.event_tx,
                    request_pool_idx,
                    admission,
                    admission_state: MediaAdmissionState::Unsubmitted,
                    committed: MediaCursor::default(),
                    projected: MediaCursor::default(),
                    fixed_parent: root.clone(),
                    projected_parent: root,
                    terminal_intent: MediaTerminalIntent::None,
                },
            );
            self.order.push(id);
        }
    }

    pub(super) fn num_vae(&self, ip: &uniserve_core::ImageParams) -> u64 {
        let dl = u64::from(self.info.latent_downsample).max(1);
        (ip.height as u64 / dl) * (ip.width as u64 / dl)
    }

    pub(super) fn cap_max_vae_grid_tokens(&self) -> usize {
        if self.info.max_vae_grid_tokens > 0 {
            self.info.max_vae_grid_tokens as usize
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
        self.info.generation_limits().covers(needs).err()
    }

    pub(super) fn worker_tracks_image_latent(&self) -> bool {
        self.info.latent_page_units > 0
            && self.info.num_latent_pages > 1
            && self
                .info
                .resource_classes
                .contains(&ResourceClass::ImageLatent)
    }

    pub(super) fn worker_image_latent_used(&self) -> u64 {
        (self.kv_budget.latent_pages.used_pages() as u64)
            .saturating_mul(u64::from(self.info.latent_page_units))
    }

    pub(super) fn worker_image_latent_units_for(&self, st: &ReqState) -> u64 {
        let downsample = (self.info.latent_downsample as u64).max(1);
        let (height, width) = (st.req.image.height, st.req.image.width);
        ceil_div_u64((height as u64).max(1), downsample)
            * ceil_div_u64((width as u64).max(1), downsample)
    }

    pub(super) fn can_schedule_denoise(&self, id: RequestId) -> bool {
        let requested_units = self
            .running
            .get(&id)
            .map(|st| self.worker_image_latent_units_for(st).max(1))
            .unwrap_or(0);
        let worker_capacity_ok = !self.worker_tracks_image_latent()
            || self.kv_budget.latent_pages.can_reserve(id, requested_units);
        worker_capacity_ok
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
        let Some(request_pool_idx) = self.kv_budget.request_slots.acquire() else {
            return false;
        };
        let mut block_tables = (0..self.kv_budget.cache().block_pool.num_groups())
            .map(|group| BlockTable::new(group, self.info.block_size as usize))
            .collect::<Vec<_>>();
        let Some(new_pages) = self.kv_budget.cache().coordinator.ensure_capacity(
            &self.kv_budget.cache().block_pool,
            &mut block_tables,
            prefix_tokens,
        ) else {
            let _ = self.kv_budget.request_slots.release(request_pool_idx);
            return false;
        };
        let Some(state) = self.running.get_mut(&id) else {
            let _ = self.kv_budget.request_slots.release(request_pool_idx);
            return false;
        };
        state.flow_prefix = Some(FlowPrefixState {
            request_pool_idx,
            block_tables,
            new_pages,
            materialized: false,
        });
        true
    }

    pub(super) fn release_flow_prefix(&mut self, id: RequestId) {
        let prefix = self
            .running
            .get_mut(&id)
            .and_then(|state| state.flow_prefix.take());
        if let Some(prefix) = prefix
            && let Err(error) = self
                .kv_budget
                .request_slots
                .release(prefix.request_pool_idx)
        {
            tracing::error!(
                request_id = id.0,
                request_pool_idx = prefix.request_pool_idx,
                error,
                "failed to release scheduler flow-prefix slot"
            );
            self.fatal = true;
        }
    }

    /// One loop iteration of the schedule-ahead loop. Returns true if any work
    /// was submitted or any result resolved.
    pub(super) fn reap_cancellations(&mut self) {
        // A cancelled request closes only after every submitted descendant has
        // resolved. Host KV ownership then remains pinned through the close
        // acknowledgement and ordered worker session retirement.
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
            .order
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
                    (MediaEvent::Failed { message }, CloseReason::Error)
                }
                MediaTerminalIntent::Cancel => (MediaEvent::Aborted, CloseReason::Cancelled),
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
        let bs = self.info.block_size as usize;
        loop {
            if self.running_request_count() >= self.config.max_num_seqs
                || self.kv_budget.request_slots.is_empty()
            {
                break;
            }
            let Some(head) = self.pending.peek_request() else {
                break;
            };
            if head.cursor.resources.reserve_worstcase {
                let need = head.cursor.resources.worstcase_blocks;
                let encoder_entries = head.req.resources.encoder_cache_keys.len();
                let encoder_ok = self
                    .kv_budget
                    .reserved_encoder_entries
                    .saturating_add(encoder_entries)
                    <= self.kv_budget.encoder_cache.budget();
                if need > self.kv_budget.usable_blocks() {
                    let st = self.pending.pop_request().unwrap();
                    self.trace_record(json!({
                        "event": "request_rejected",
                        "at_s": now(),
                        "request_id": st.req.request_id.0,
                        "reason": "too_large",
                        "needed_blocks": need,
                        "usable_blocks": self.kv_budget.usable_blocks(),
                        "generation": &st.req.behavior,
                        "prompt_tokens": st.context.prompt_ids.len(),
                    }));
                    let _ = st.output.event_tx.send(GenerationEvent::Rejected {
                        message: "request exceeds total KV capacity".into(),
                    });
                    continue;
                }
                if self.kv_budget.free_blocks() >= need && encoder_ok {
                    let st = self.pending.pop_request().unwrap();
                    let id = st.req.request_id;
                    self.admit_running(st);
                    // Physically allocate the worst case now: nothing can take
                    // these blocks, so this request can never fail mid-flight.
                    self.ensure_request_capacity(id, need * bs);
                    self.kv_budget.reserved_blocks += need;
                    continue;
                }
            } else {
                let n = head.context.prompt_ids.len();
                let text_usable_blocks = (0..self.kv_budget.cache().block_pool.num_groups())
                    .map(|group| self.kv_budget.cache().block_pool.group_capacity(group))
                    .min()
                    .unwrap_or_default();
                let prefix_hit = prefix_hit(
                    &self.kv_budget.cache().coordinator,
                    head,
                    &self.kv_budget.cache().block_pool,
                );
                let cached_prefix_blocks = prefix_hit.cached_blocks;
                let cached_prefix_blocks = cached_prefix_blocks.min(n.div_ceil(bs));
                let cached_prefix_tokens = cached_prefix_blocks.saturating_mul(bs);
                let uncached_remaining = n.saturating_sub(cached_prefix_tokens);
                let first_uncached_chunk = uncached_remaining
                    .min(self.config.long_prefill_threshold)
                    .min(self.config.max_num_batched_tokens)
                    .max(1);
                let first_chunk_blocks = cached_prefix_tokens
                    .saturating_add(first_uncached_chunk)
                    .div_ceil(bs)
                    .saturating_sub(cached_prefix_blocks);
                if n > text_usable_blocks * bs {
                    let st = self.pending.pop_request().unwrap();
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
                    let _ = st.output.event_tx.send(GenerationEvent::Rejected {
                        message: "request exceeds total KV capacity".into(),
                    });
                    continue;
                }
                let capacity_available = prefix_hit.cached_free_blocks.len()
                    == self.kv_budget.cache().block_pool.num_groups()
                    && prefix_hit.cached_free_blocks.iter().enumerate().all(
                        |(group, cached_free)| {
                            self.kv_budget
                                .cache()
                                .block_pool
                                .free_blocks_in_group(group)
                                .saturating_sub(*cached_free)
                                >= first_chunk_blocks
                        },
                    );
                if capacity_available {
                    let st = self.pending.pop_request().unwrap();
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
        st.request_pool_idx = self
            .kv_budget
            .request_slots
            .acquire()
            .expect("admission checked request-slot capacity");
        let id = st.req.request_id;
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
        if st.output.enqueue(GenerationEvent::Scheduled {
            queued_at: q,
            scheduled_at,
        }) {
            st.terminal_intent = TerminalIntent::Cancel;
        }
        self.running.insert(id, st);
        self.order.push(id);
        self.kv_budget.reserved_encoder_entries = self
            .kv_budget
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
            "pending": self.pending.len(),
            "free_blocks": self.kv_budget.free_blocks(),
            "reserved_blocks": self.kv_budget.reserved_blocks,
            "reserved_encoder_entries": self.kv_budget.reserved_encoder_entries,
        }));
        if let Some(st) = self.running.get_mut(&id) {
            let kv = self.kv_budget.cache();
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
    let hit = coordinator
        .acquire_prefix(
            pool,
            &mut state.block_tables,
            &prompt,
            state.req.cache.read,
            has_context_images,
            state.req.cache.isolation_key,
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
