use super::*;

impl Scheduler {
    pub(super) fn enqueue(&mut self, req: GenerationRequest, event_tx: EventTx) {
        if self.kv.is_none() {
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
        if let Some(capability) = self.missing_required_capability(&req) {
            let _ = event_tx.send(GenerationEvent::Rejected {
                message: format!(
                    "generation request requires worker capability `{capability}`, but the worker does not support it"
                ),
            });
            return;
        }
        if let Err(error) = req.validate_resources(&self.caps.generation_runtime_capabilities()) {
            let _ = event_tx.send(GenerationEvent::Rejected {
                message: format!("invalid generation resource declaration: {error}"),
            });
            return;
        }
        // Admission backpressure sheds load instead of letting the waiting
        // queue grow without bound under overload —
        // an unbounded burst would otherwise OOM the process and take down every
        // in-flight request. Reject the new submit with a typed event.
        let waiting = self.pending_request_count() + self.completed_outputs.len();
        if waiting >= self.config.max_num_waiting {
            self.record_decision(
                req.request_id,
                crate::scheduler::policy::PolicyReason::RejectedTooLarge,
                0,
            );
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
            .div_ceil(self.caps.block_size as usize);
        // Multimodal requests reserve their configured bounded KV envelope at
        // admission so excess concurrency queues instead of exhausting KV.
        let reserve_worstcase = !context.images.is_empty() || req.behavior.gen_output;
        // A request with staged images usually encodes them before prefill.
        // Context-image requests prefill the text before each image position,
        // then encode the image into that marker gap.
        let phase0 = Phase::Prefill;
        // A lifecycle trace keyed by the canonical request key.
        let trace = crate::scheduler::trace::RequestTrace::new(
            RequestKey::new(self.authority_id, req.request_id, self.next_epoch),
            uniserve_core::TraceId(req.request_id.0),
        );
        let st = ReqState {
            block_tables: (0..self.kv_state().block_pool.num_groups())
                .map(|group| BlockTable::new(group, self.caps.block_size as usize))
                .collect(),
            flow_prefix: None,
            request_pool_idx: 0,
            epoch: self.next_epoch,
            version: 0,
            admission_digest: None,
            resolved_semantic: String::new(),
            resolved_producer_op_id: 0,
            committed_version: 0,
            committed_semantic: String::new(),
            committed_producer_op_id: 0,
            control_seq: 0,
            public_event_seq: 0,
            public_event_limit: 0,
            public_token_seq: 0,
            semantic_token_seq: 0,
            output_journal: VecDeque::new(),
            token_cutoffs: BTreeMap::new(),
            pending_commits: VecDeque::new(),
            cancel_cutoff: None,
            latest_device_version: None,
            cursor: GenerationCursor::new(phase0, worst, reserve_worstcase),
            context,
            event_tx,
            queued_at: now(),
            cancelled: false,
            aborted: false,
            stop_matched: false,
            cpu_pending: None,
            cpu_masks: None,
            cpu_generation: 0,
            trace,
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
            .any(|variant| !self.caps.supported_work.contains(variant))
        {
            let _ = submission.event_tx.send(MediaEvent::Rejected {
                message: "worker does not support the fixed media flow".to_string(),
            });
            return;
        }
        if self.caps.max_request_pool_size < 2
            || self.caps.num_latent_pages < 3
            || self.caps.latent_page_units == 0
        {
            let _ = submission.event_tx.send(MediaEvent::Rejected {
                message: "worker does not provide two resident media state slots".to_string(),
            });
            return;
        }
        if self.pending.len() + self.pending_media.len() + self.completed_outputs.len()
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
        while !self.request_slots.is_empty()
            && self.running_request_count() < self.config.max_num_seqs
        {
            let Some(submission) = self.pending_media.pop_front() else {
                break;
            };
            let id = submission.request.request_id;
            let Some(request_pool_idx) = self.request_slots.acquire() else {
                self.pending_media.push_front(submission);
                break;
            };
            if !self
                .latent_pages
                .reserve(id, u64::from(self.caps.latent_page_units))
            {
                let _ = self.request_slots.release(request_pool_idx);
                self.pending_media.push_front(submission);
                break;
            }
            let epoch = self.next_epoch;
            self.next_epoch = self.next_epoch.saturating_add(1);
            let request_key = RequestKey::new(self.authority_id, id, epoch);
            let admission = Admission::new_media(
                request_key,
                request_pool_idx,
                MediaAdmission {
                    prompt: submission.request.prompt.clone(),
                    seed: submission.request.seed,
                    profile: MediaProfileId::MinimaxH3T2va,
                    output_path: submission.request.output_path.clone(),
                },
            )
            .expect("validated media admission");
            let root = VersionRef::admission_root(request_key, OpId(0), admission.digest.clone());
            self.running.insert_media(MediaFlowState {
                request: submission.request,
                event_tx: submission.event_tx,
                request_pool_idx,
                admission,
                admission_sent: false,
                committed: MediaCursor::default(),
                projected: MediaCursor::default(),
                fixed_parent: root.clone(),
                projected_parent: root,
                cancelled: false,
                failure: None,
            });
            self.order.push(id);
        }
    }

    pub(super) fn num_vae(&self, ip: &uniserve_core::ImageParams) -> u64 {
        let dl = u64::from(self.caps.latent_downsample).max(1);
        (ip.height as u64 / dl) * (ip.width as u64 / dl)
    }

    pub(super) fn cap_max_vae_grid_tokens(&self) -> usize {
        if self.caps.max_vae_grid_tokens > 0 {
            self.caps.max_vae_grid_tokens as usize
        } else {
            self.caps.latent_capacity_units().min(usize::MAX as u64) as usize
        }
    }

    pub(super) fn missing_required_capability(
        &self,
        request: &GenerationRequest,
    ) -> Option<&'static str> {
        let context_steps = request.context.iter().flat_map(|segment| match segment {
            uniserve_core::ContextSegment::Image { ingest, .. } => ingest.steps.clone(),
            uniserve_core::ContextSegment::UndTokens { .. } => Vec::new(),
        });
        let needs = request
            .behavior
            .capability_needs(&request.policy, context_steps);
        self.caps
            .generation_runtime_capabilities()
            .covers(&needs)
            .err()
    }

    pub(super) fn worker_tracks_image_latent(&self) -> bool {
        self.caps.latent_page_units > 0
            && self.caps.num_latent_pages > 1
            && self
                .caps
                .resource_classes
                .contains(&ResourceClass::ImageLatent)
    }

    pub(super) fn worker_image_latent_used(&self) -> u64 {
        (self.latent_pages.used_pages() as u64)
            .saturating_mul(u64::from(self.caps.latent_page_units))
    }

    pub(super) fn worker_image_latent_units_for(&self, st: &ReqState) -> u64 {
        let downsample = (self.caps.latent_downsample as u64).max(1);
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
            || self.latent_pages.can_reserve(id, requested_units);
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
        let Some(request_pool_idx) = self.request_slots.acquire() else {
            return false;
        };
        let mut block_tables = (0..self.kv_state().block_pool.num_groups())
            .map(|group| BlockTable::new(group, self.caps.block_size as usize))
            .collect::<Vec<_>>();
        let Some(new_pages) = self.kv_state().coordinator.ensure_capacity(
            &self.kv_state().block_pool,
            &mut block_tables,
            prefix_tokens,
        ) else {
            let _ = self.request_slots.release(request_pool_idx);
            return false;
        };
        let Some(state) = self.running.get_mut(&id) else {
            let _ = self.request_slots.release(request_pool_idx);
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
            && let Err(error) = self.request_slots.release(prefix.request_pool_idx)
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

    pub(super) fn reserve_transition_resources(
        &mut self,
        transition: &mut PlannedTransition,
    ) -> bool {
        let id = transition.request_id;
        if transition.operation_variant == ForwardMode::GenFlow && !self.ensure_flow_prefix(id) {
            return false;
        }
        let resources = &transition.resources;
        if resources.latent_units > 0
            && self.worker_tracks_image_latent()
            && !self.latent_pages.can_reserve(id, resources.latent_units)
        {
            return false;
        }
        let uses_transfer = transition.bounds.max_transfer_bytes > 0;
        if uses_transfer && self.inflight_transfers >= self.transfer_capacity {
            return false;
        }
        if resources.latent_units > 0
            && self.worker_tracks_image_latent()
            && !self.latent_pages.reserve(id, resources.latent_units)
        {
            return false;
        }
        if uses_transfer {
            self.inflight_transfers += 1;
        }
        transition.reserved_us = uniserve_core::now_monotonic_us();
        true
    }

    pub(super) fn release_transition_resources(&mut self, id: RequestId, apply: &SchedulerApply) {
        for class in &apply.release_on_apply {
            match class {
                uniserve_worker_ipc::ResourceClass::ImageLatent => {
                    self.latent_pages.release(id);
                }
                _ => {}
            }
        }
    }

    /// KV tokens a decode op must have room for: the sampled token plus any
    /// speculative draft the worker verifies alongside it.
    pub(super) fn decode_capacity_target(&self, pos: usize, spec_len: usize) -> usize {
        pos.saturating_add(1).saturating_add(spec_len)
    }

    /// One loop iteration of the schedule-ahead loop. Returns true if any work
    /// was submitted or any result resolved.
    pub(super) fn reap_cancellations(&mut self) {
        // A cancelled request closes only after every submitted descendant has
        // resolved. Host KV ownership then remains pinned through the close
        // acknowledgement and ordered worker session retirement.
        let cancelled: Vec<(RequestId, bool, bool)> = self
            .running
            .iter()
            .filter(|(id, s)| s.cancelled && !self.has_inflight(**id))
            .map(|(k, s)| (*k, s.aborted, s.stop_matched))
            .collect();
        for (id, aborted, stop_matched) in cancelled {
            let reason = if stop_matched {
                FinishReason::Stop
            } else if aborted {
                FinishReason::Aborted
            } else {
                FinishReason::Cancelled
            };
            self.finish(id, reason);
        }
        let media = self
            .order
            .iter()
            .filter_map(|id| {
                self.media_state(*id)
                    .filter(|state| state.cancelled && !self.has_inflight(state.request.request_id))
                    .map(|state| state.request.request_id)
            })
            .collect::<Vec<_>>();
        for id in media {
            let event = self
                .media_state(id)
                .and_then(|state| state.failure.clone())
                .map_or(MediaEvent::Aborted, |message| MediaEvent::Failed {
                    message,
                });
            let reason = if matches!(event, MediaEvent::Failed { .. }) {
                CloseReason::Error
            } else {
                CloseReason::Cancelled
            };
            self.finish_media(id, event, reason, None);
        }
    }

    /// Admission: consume the waiting-queue head while budgets allow (vLLM's
    /// posture — head-of-line, `max_num_seqs`-capped). Reserving requests
    /// allocate their full worst-case KV here, which is what makes them
    /// resident for its complete lifetime.
    pub(super) fn admit(&mut self) {
        let bs = self.caps.block_size as usize;
        loop {
            if self.running_request_count() >= self.config.max_num_seqs
                || self.request_slots.is_empty()
            {
                break;
            }
            let Some(head) = self.pending.peek_request() else {
                break;
            };
            if head.resources.reserve_worstcase {
                let need = head.resources.worstcase_blocks;
                let encoder_entries = head.req.resources.encoder_cache_keys.len();
                let encoder_ok = self
                    .reserved_encoder_entries
                    .saturating_add(encoder_entries)
                    <= self.enc_cache.budget();
                if need > self.usable_kv_blocks() {
                    let st = self.pending.pop_request().unwrap();
                    self.record_decision(
                        st.req.request_id,
                        crate::scheduler::policy::PolicyReason::RejectedTooLarge,
                        need,
                    );
                    self.trace_record(json!({
                        "event": "request_rejected",
                        "at_s": now(),
                        "request_id": st.req.request_id.0,
                        "reason": "too_large",
                        "needed_blocks": need,
                        "usable_blocks": self.usable_kv_blocks(),
                        "generation": behavior_str(&st.req),
                        "prompt_tokens": st.context.prompt_ids.len(),
                    }));
                    let _ = st.event_tx.send(GenerationEvent::Rejected {
                        message: "request exceeds total KV capacity".into(),
                    });
                    continue;
                }
                if self.free_kv_blocks() >= need && encoder_ok {
                    let st = self.pending.pop_request().unwrap();
                    let id = st.req.request_id;
                    self.admit_running(st);
                    // Physically allocate the worst case now: nothing can take
                    // these blocks, so this request can never fail mid-flight.
                    self.ensure_request_capacity(id, need * bs);
                    self.reserved_blocks += need;
                    self.record_decision(
                        id,
                        crate::scheduler::policy::PolicyReason::Admitted,
                        need,
                    );
                    continue;
                }
            } else {
                let n = head.context.prompt_ids.len();
                let text_usable_blocks = (0..self.kv_state().block_pool.num_groups())
                    .map(|group| self.kv_state().block_pool.group_capacity(group))
                    .min()
                    .unwrap_or_default();
                let prefix_admission = crate::scheduler::prefix_cache::cached_blocks_for_admission(
                    &self.kv_state().coordinator,
                    head,
                    &self.kv_state().block_pool,
                );
                let cached_prefix_blocks = prefix_admission.cached_blocks;
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
                    self.record_decision(
                        st.req.request_id,
                        crate::scheduler::policy::PolicyReason::RejectedTooLarge,
                        first_chunk_blocks,
                    );
                    self.trace_record(json!({
                        "event": "request_rejected",
                        "at_s": now(),
                        "request_id": st.req.request_id.0,
                        "reason": "too_large",
                        "needed_blocks": first_chunk_blocks,
                        "usable_blocks": text_usable_blocks,
                        "generation": behavior_str(&st.req),
                        "prompt_tokens": st.context.prompt_ids.len(),
                    }));
                    let _ = st.event_tx.send(GenerationEvent::Rejected {
                        message: "request exceeds total KV capacity".into(),
                    });
                    continue;
                }
                let capacity_available = prefix_admission.cached_free_blocks.len()
                    == self.kv_state().block_pool.num_groups()
                    && prefix_admission.cached_free_blocks.iter().enumerate().all(
                        |(group, cached_free)| {
                            self.kv_state()
                                .block_pool
                                .free_blocks_in_group(group)
                                .saturating_sub(*cached_free)
                                >= first_chunk_blocks
                        },
                    );
                if capacity_available {
                    let st = self.pending.pop_request().unwrap();
                    let id = st.req.request_id;
                    self.admit_running(st);
                    self.record_decision(
                        id,
                        crate::scheduler::policy::PolicyReason::Admitted,
                        first_chunk_blocks,
                    );
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
        let generation = behavior_str(&st.req);
        let phase = phase_str(st.lifecycle.phase);
        let prompt_tokens = st.context.prompt_ids.len();
        let max_tokens = st.req.max_und_tokens;
        let priority = st.req.priority;
        let reserve_worstcase = st.resources.reserve_worstcase;
        let worstcase_blocks = st.resources.worstcase_blocks;
        let encoder_entries = st.req.resources.encoder_cache_keys.len();
        self.emit_st(
            &mut st,
            GenerationEvent::Scheduled {
                queued_at: q,
                scheduled_at,
            },
        );
        st.trace.mark_admitted(uniserve_core::now_monotonic_us());
        self.running.insert(id, st);
        self.order.push(id);
        self.reserved_encoder_entries = self
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
            "free_blocks": self.free_kv_blocks(),
            "reserved_blocks": self.reserved_blocks,
            "reserved_encoder_entries": self.reserved_encoder_entries,
        }));
        if let Some(st) = self.running.get_mut(&id) {
            let kv = self
                .kv
                .as_ref()
                .expect("generation admission requires worker KV resources");
            crate::scheduler::prefix_cache::lookup(
                &kv.coordinator,
                st,
                &kv.block_pool,
                &self.stats,
            );
        }
    }
}
