use super::*;

impl Scheduler {
    pub fn step(&mut self) -> bool {
        let progressed = self.step_nonblocking();
        if progressed || self.executor.in_flight() == 0 {
            return progressed;
        }
        match self.executor.next_result() {
            Ok(result) => {
                self.apply_result(result);
                self.refill_executor();
                self.stats
                    .general
                    .in_flight
                    .store(self.executor.in_flight(), Ordering::Relaxed);
                self.publish_cache_stats();
            }
            Err(error) => {
                self.on_executor_error(error);
                self.stats
                    .general
                    .in_flight
                    .store(self.executor.in_flight(), Ordering::Relaxed);
                self.publish_cache_stats();
            }
        }
        true
    }

    /// Nonblocking schedule-ahead tick used by the owner-thread reactor. It
    /// drains ready results, reaps cancellations, and fills available executor
    /// slots, but leaves any blocking result wait to `run`.
    pub(super) fn step_nonblocking(&mut self) -> bool {
        let _span = tracing::trace_span!("scheduler.step").entered();
        let mut progressed = self.flush_output_journals();
        // 1. Resolve one completed batch. Refilling immediately after one
        // completion preserves an occupied execution slot when multiple
        // responses become ready together at pipeline depth greater than one.
        // The owner loop returns here without parking while progress is being
        // made, so subsequent ready completions are handled on successive turns.
        progressed |= self.poll_one_result();

        progressed |= self.refill_executor();

        self.stats
            .general
            .in_flight
            .store(self.executor.in_flight(), Ordering::Relaxed);
        self.publish_cache_stats();
        progressed
    }

    pub(super) fn refill_executor(&mut self) -> bool {
        let mut progressed = false;

        // 2. reap cancellations before assembling.
        self.reap_cancellations();

        // 3. Admit request/resource residency even
        // while every execution slot is occupied. This lets the next batch see
        // the complete resident cohort instead of admitting only when a slot
        // happens to open.
        self.admit();
        self.admit_media();

        // 4. submit as many batches as pipeline capacity allows.
        while self.executor.can_submit() {
            self.admit();
            self.admit_media();
            if self.prefer_media && self.submit_media_batch() {
                self.prefer_media = false;
                progressed = true;
                continue;
            }
            let (new_reqs, ops) = self.assemble();
            if ops.is_empty() {
                if self.submit_media_batch() {
                    self.prefer_media = false;
                    progressed = true;
                    continue;
                }
                break;
            }
            // A prompt batch carries its own state-mutating controls plus a
            // foreign terminal close only when no earlier control for that
            // session remains queued. Other foreign controls stay pending so
            // the prompt session set remains disjoint from in-flight decode
            // work without reordering a request's control stream.
            let prompt_batch = ops
                .iter()
                .all(|op| batch_kind(op.operation_variant) == BatchKind::Prefill);
            let controls: Vec<Control> = if prompt_batch {
                let sessions: HashSet<RequestId> = ops.iter().map(|op| op.request_id).collect();
                let mut blocked_foreign = HashSet::new();
                let mut own = VecDeque::new();
                let mut foreign = VecDeque::new();
                for control in self.pending_controls.drain(..) {
                    let request_id = control.request_key().session_id;
                    if sessions.contains(&request_id)
                        || (matches!(control, Control::Close { .. })
                            && !blocked_foreign.contains(&request_id))
                    {
                        own.push_back(control);
                    } else {
                        blocked_foreign.insert(request_id);
                        foreign.push_back(control);
                    }
                }
                self.pending_controls = foreign;
                own.into()
            } else {
                self.pending_controls.drain(..).collect()
            };
            if !self.submit_batch(new_reqs, ops, controls) {
                break;
            }
            self.prefer_media = true;
            progressed = true;
        }
        while self.executor.can_submit() && !self.pending_controls.is_empty() {
            let controls = self.pending_controls.drain(..).collect();
            if !self.submit_batch(Vec::new(), Vec::new(), controls) {
                break;
            }
            progressed = true;
        }

        progressed
    }

    pub(super) fn media_completion_product(
        &mut self,
        request_key: RequestKey,
        op_id: OpId,
    ) -> ProductRef {
        let generation = u32::try_from(self.next_product_generation.max(1))
            .expect("media product generation exhausted");
        self.next_product_generation = u64::from(generation).saturating_add(1);
        ProductRef {
            request_key,
            producer_op_id: op_id,
            output_index: 0,
            generation,
            kind: ProductKind::Completion,
            storage_class: StorageClass::RequestRelay,
            dtype: DType::U32,
            shape_bound: ShapeBound {
                dims: vec![DimBound::Static(1)],
            },
            point_range: PointRange {
                base_point: 1,
                max_points: 1,
            },
        }
    }

    pub(super) fn submit_media_batch(&mut self) -> bool {
        let max_unresolved = usize::try_from(self.info.max_unresolved_window.max(1))
            .unwrap_or(usize::MAX)
            .min(self.executor.pipeline_depth().max(1));
        let mut candidates = self
            .order
            .iter()
            .enumerate()
            .filter_map(|(index, id)| {
                self.media_state(*id).and_then(|state| {
                    let inflight = self.inflight.len(*id);
                    (!state.terminal_intent.is_terminal()
                        && state.admission_state != MediaAdmissionState::InFlight
                        && inflight < max_unresolved
                        && next_media_quantum(
                            state.projected,
                            state
                                .admission
                                .media
                                .as_ref()
                                .expect("media admission")
                                .geometry,
                        )
                        .is_some())
                    .then_some((inflight, index, *id))
                })
            })
            .collect::<Vec<_>>();
        candidates.sort_unstable_by_key(|(inflight, index, _)| (*inflight > 0, *index));
        candidates.truncate(self.config.max_batch);
        if candidates.is_empty() {
            return false;
        }

        let step = self.inflight.next_step();
        let submit_at = Instant::now();
        let collective_seq = self.next_collective_seq.max(1);
        self.next_collective_seq = collective_seq.saturating_add(1);
        let mut admissions = Vec::new();
        let mut operations = Vec::with_capacity(candidates.len());
        let mut latent_placements = Vec::new();
        let mut reconstruction_placements = Vec::new();
        let (media_closes, remaining_controls): (VecDeque<_>, VecDeque<_>) = self
            .pending_controls
            .drain(..)
            .partition(|control| matches!(control, Control::Close { .. }));
        self.pending_controls = remaining_controls;
        let mut controls = media_closes.into_iter().collect::<Vec<_>>();

        for (_, _, id) in candidates {
            let (request_key, parent, predicate, quantum, cursor_after) = {
                let state = self.media_state(id).expect("media candidate exists");
                let quantum = next_media_quantum(
                    state.projected,
                    state
                        .admission
                        .media
                        .as_ref()
                        .expect("media admission")
                        .geometry,
                )
                .expect("media candidate is runnable");
                let predicate = self
                    .inflight
                    .operations
                    .get(&id)
                    .and_then(|inflight| inflight.back())
                    .and_then(|inflight| inflight.operation.outputs.first().cloned());
                (
                    state.admission.request_key,
                    state.projected_parent.clone(),
                    predicate,
                    quantum,
                    advance_media_cursor(state.projected, quantum),
                )
            };
            let op_id = OpId(self.next_op_id.max(1));
            self.next_op_id = self.next_op_id.saturating_add(1);
            let work = media_work(quantum);
            let outputs = if matches!(quantum, MediaQuantum::Materialize) {
                Vec::new()
            } else {
                vec![self.media_completion_product(request_key, op_id)]
            };
            let operation = Operation {
                request_key,
                op_id,
                parent,
                work,
                route: RouteId(0),
                domain: uniserve_worker_ipc::Domain::Flow,
                advances_state: false,
                bounds: Bounds {
                    max_points: 1,
                    ..Bounds::default()
                },
                inputs: Vec::new(),
                outputs,
                predicate,
                rng: None,
                control_seq: 0,
            }
            .sealed();
            if matches!(
                quantum,
                MediaQuantum::Prepare | MediaQuantum::Denoise { .. }
            ) {
                let (start_step, step_count) = match quantum {
                    MediaQuantum::Denoise { step } => (step, 1),
                    _ => (0, 0),
                };
                latent_placements.push(LatentPlacement {
                    request_key,
                    op_id,
                    page_table: self.kv_budget.latent_pages.pages_for(id).to_vec(),
                    latent_units: self.info.latent_page_units,
                    height: 768,
                    width: 1344,
                    start_step,
                    step_count,
                });
            }
            match quantum {
                MediaQuantum::ReconstructVideo {
                    start_unit,
                    unit_count,
                } => reconstruction_placements.push(ReconstructionPlacement {
                    request_key,
                    op_id,
                    kind: ReconstructionKind::Video,
                    start_unit,
                    unit_count,
                }),
                MediaQuantum::ReconstructAudio => {
                    reconstruction_placements.push(ReconstructionPlacement {
                        request_key,
                        op_id,
                        kind: ReconstructionKind::Audio,
                        start_unit: 0,
                        unit_count: 1,
                    })
                }
                _ => {}
            }
            if operation.parent.producer_op_id.0 > 0 {
                controls.push(Control::Release {
                    request_key,
                    op_id: operation.parent.producer_op_id,
                });
            }
            let projected_parent = VersionRef {
                request_key,
                producer_op_id: op_id,
                point: Point::Device {
                    point_index: 1,
                    selected_point: None,
                },
            };
            let state = self.media_state_mut(id).expect("media candidate exists");
            if state.admission_state == MediaAdmissionState::Unsubmitted {
                admissions.push(state.admission.clone());
                state.admission_state = MediaAdmissionState::InFlight;
            }
            state.projected = cursor_after;
            state.projected_parent = projected_parent;
            let operation_for_batch = operation.clone();
            self.register_media_inflight(operation, cursor_after, submit_at);
            operations.push(operation_for_batch);
        }

        let partition = BatchPartition {
            partition_id: 1,
            submission_group: 1,
            collective_seq,
            domain: uniserve_worker_ipc::Domain::Flow,
            route: RouteId(0),
            attention: AttentionRegime::None,
            shape_class: 0,
            operations,
            block_tables: Vec::new(),
            new_cache_pages: Vec::new(),
            forward_rows: Vec::new(),
            latent_placements,
            reconstruction_placements,
        };
        self.inflight.batch_started.insert(step, submit_at);
        self.inflight.batch_partitions.insert(
            step,
            HashMap::from([(
                1,
                SubmittedPartitionAccounting {
                    domain: uniserve_worker_ipc::Domain::Flow,
                    mixed: false,
                    submission_group: 1,
                    operation_count: partition.operations.len(),
                },
            )]),
        );
        self.inflight
            .batch_group_worker_exec_us
            .insert(step, HashMap::new());
        let batch = Batch::new(step, admissions, vec![partition]).with_controls(controls.clone());
        if let Err(error) = self.executor.submit(batch) {
            self.inflight.batch_started.remove(&step);
            self.inflight.batch_partitions.remove(&step);
            self.inflight.batch_group_worker_exec_us.remove(&step);
            if error.downcast_ref::<WorkerLossError>().is_some() {
                self.on_executor_error(error);
            } else {
                self.fatal = true;
                self.fail_all_running(&error.to_string());
            }
            return false;
        }
        if !controls.is_empty() {
            self.inflight.control_batches.insert(step, controls);
        }
        true
    }

    /// Surface cache observability (events drained into counters).
    pub(super) fn publish_cache_stats(&mut self) {
        self.stats
            .general
            .running
            .store(self.running_request_count(), Ordering::Relaxed);
        self.stats
            .general
            .pending
            .store(self.pending_request_count(), Ordering::Relaxed);
        self.stats
            .kv_cache
            .free_blocks
            .store(self.kv_budget.free_blocks(), Ordering::Relaxed);
        self.stats
            .general
            .in_flight
            .store(self.executor.in_flight(), Ordering::Relaxed);
        // drain the manager's event ring so it doesn't grow unbounded; the
        // counters below already aggregate it, but draining keeps memory bounded.
        if let Some(kv) = self.kv_budget.cache.as_ref() {
            let _ = kv.block_pool.drain_events();
            self.stats
                .kv_cache
                .blocks_evicted
                .store(kv.block_pool.stats().evictions, Ordering::Relaxed);
            self.stats
                .kv_cache
                .blocks_stored
                .store(kv.block_pool.stats().blocks_stored, Ordering::Relaxed);
            self.stats
                .kv_cache
                .cached_blocks
                .store(kv.block_pool.cached_blocks(), Ordering::Relaxed);
        }
        self.stats.encoder.cache_queries.store(
            self.kv_budget.encoder_cache.stats.queries,
            Ordering::Relaxed,
        );
        self.stats
            .encoder
            .cache_hits
            .store(self.kv_budget.encoder_cache.stats.hits, Ordering::Relaxed);
        self.stats
            .encoder
            .cached
            .store(self.kv_budget.encoder_cache.len(), Ordering::Relaxed);
    }

    /// Resolve at most one ready result so assembly can refill the freed slot
    /// before another completion is consumed.
    pub(super) fn poll_one_result(&mut self) -> bool {
        let _span = tracing::trace_span!("scheduler.poll_one_result").entered();
        match self.executor.poll() {
            Ok(Some(result)) => {
                self.apply_result(result);
                true
            }
            Ok(None) => false,
            Err(error) => {
                self.on_executor_error(error);
                true
            }
        }
    }

    pub(super) fn flow_prefix_is_schedulable(&self, id: RequestId) -> bool {
        let prefix_is_pending = self
            .running
            .get(&id)
            .and_then(|state| state.flow_prefix.as_ref())
            .is_some_and(|prefix| !prefix.materialized);
        !prefix_is_pending
            || !self
                .inflight
                .operations
                .get(&id)
                .into_iter()
                .flatten()
                .any(|op| op.operation.work == ForwardMode::MediaDenoise)
    }

    pub(super) fn projected_cursor(&self, id: RequestId) -> Option<GenerationCursor> {
        let st = self.running.get(&id)?;
        let inflight = self
            .inflight
            .operations
            .get(&id)
            .into_iter()
            .flatten()
            .map(|inflight| (&inflight.operation, inflight.generation_apply()));
        let mut projected = st.cursor.project(inflight)?;
        if projected.phase == Phase::DenoiseGen
            && projected.image_gen.steps_done >= st.req.image.steps
        {
            projected.phase = Phase::CommitGen;
        }
        if projected.phase == Phase::Prefill
            && projected.ingest.prompt_cursor >= st.context.prompt_ids.len() as u32
            && projected.ingest.mm_cursor >= st.context.images.len()
        {
            projected.phase = Phase::DecodeUnd;
        }
        Some(projected)
    }

    pub(super) fn fixed_version(&self, id: RequestId) -> Option<VersionRef> {
        let state = self.running.get(&id)?;
        state.cursor.resources.worker_registered.then_some(())?;
        Some(VersionRef {
            request_key: RequestKey::new(self.authority_id, id, state.epoch),
            producer_op_id: OpId(state.resolved_producer_op_id),
            point: Point::Fixed {
                point_index: state.version as u32,
            },
        })
    }

    pub(super) fn projected_parent(&self, id: RequestId) -> Option<VersionRef> {
        let operation = self
            .inflight
            .operations
            .get(&id)?
            .iter()
            .rev()
            .find(|inflight| inflight.operation.advances_state)
            .map(|inflight| &inflight.operation);
        let Some(operation) = operation else {
            return self.fixed_version(id);
        };
        let selected_point = (operation.bounds.max_points > 1)
            .then(|| {
                operation
                    .outputs
                    .iter()
                    .find(|output| output.kind == ProductKind::SelectedPoint)
                    .cloned()
            })
            .flatten();
        if operation.bounds.max_points > 1 && selected_point.is_none() {
            return None;
        }
        Some(VersionRef {
            request_key: operation.request_key,
            producer_op_id: operation.op_id,
            point: Point::Device {
                point_index: u32::from(selected_point.is_none()),
                selected_point,
            },
        })
    }

    pub(super) fn public_limit_for(&self, id: RequestId, apply: &SchedulerApply) -> u64 {
        self.running.get(&id).map_or(0, |state| {
            state.public_event_limit.max(
                state
                    .output
                    .event_seq
                    .saturating_add(apply.output_event_bound as u64),
            )
        })
    }

    pub(super) fn queue_commit(
        &mut self,
        id: RequestId,
        expected_parent: VersionRef,
        selected: VersionRef,
        public_event_limit: u64,
    ) {
        let Some(state) = self.running.get_mut(&id) else {
            return;
        };
        state.control_seq = state.control_seq.saturating_add(1);
        state.committed_version = match &selected.point {
            Point::Fixed { point_index, .. } => u64::from(*point_index),
            Point::Device { .. } => state.committed_version,
        };
        state.committed_producer_op_id = selected.producer_op_id.0;
        state.public_event_limit = public_event_limit;
        self.pending_controls.push_back(Control::Commit {
            request_key: selected.request_key,
            control_seq: state.control_seq,
            expected_parent,
            selected,
            public_event_limit,
            disposition: Disposition::Publish,
        });
    }

    pub(super) fn acknowledge_controls(&mut self, controls: &[Control]) {
        for control in controls {
            if let Control::Close { request_key, .. } = control {
                let id = request_key.session_id;
                if let Some(retiring) = self.retiring_media.get(&id) {
                    if retiring.request_key != *request_key {
                        tracing::error!(
                            request_id = id.0,
                            "media close acknowledgement mismatched its session"
                        );
                        self.fatal = true;
                        continue;
                    }
                    let request_pool_idx = retiring.request_pool_idx;
                    match self.drop_worker_session(id) {
                        Ok(_) => {
                            self.retiring_media.remove(&id);
                            self.kv_budget.latent_pages.release(id);
                            if let Err(error) =
                                self.kv_budget.request_slots.release(request_pool_idx)
                            {
                                tracing::error!(
                                    request_id = id.0,
                                    error,
                                    "failed to release media request slot"
                                );
                                self.fatal = true;
                            }
                        }
                        Err(error) if error.downcast_ref::<WorkerLossError>().is_some() => {
                            self.on_executor_error(error);
                            return;
                        }
                        Err(error) => {
                            tracing::error!(request_id = id.0, %error, "media worker session retirement failed");
                            self.fatal = true;
                        }
                    }
                    continue;
                }
                let Some(retiring) = self.retiring_sessions.get(&id) else {
                    tracing::error!(
                        request_id = id.0,
                        epoch = request_key.epoch,
                        "close acknowledgement does not match a retiring session"
                    );
                    self.fatal = true;
                    continue;
                };
                if retiring.request_key != *request_key {
                    tracing::error!(
                        request_id = id.0,
                        epoch = request_key.epoch,
                        "close acknowledgement does not match a retiring session"
                    );
                    self.fatal = true;
                    continue;
                }
                let request_pool_idx = retiring.request_pool_idx;
                let flow_pool_idx = retiring
                    .flow_prefix
                    .as_ref()
                    .map(|prefix| prefix.request_pool_idx);
                match self.drop_worker_session(id) {
                    Ok(_) => {
                        self.retiring_sessions.remove(&id);
                        self.kv_budget.latent_pages.release(id);
                        if let Some(index) = flow_pool_idx
                            && let Err(error) = self.kv_budget.request_slots.release(index)
                        {
                            tracing::error!(
                                request_id = id.0,
                                request_pool_idx = index,
                                error,
                                "failed to release scheduler flow-prefix slot"
                            );
                            self.fatal = true;
                        }
                        if let Err(error) = self.kv_budget.request_slots.release(request_pool_idx) {
                            tracing::error!(
                                request_id = id.0,
                                request_pool_idx,
                                error,
                                "failed to release scheduler request slot"
                            );
                            self.fatal = true;
                        }
                    }
                    Err(error) if error.downcast_ref::<WorkerLossError>().is_some() => {
                        self.on_executor_error(error);
                        return;
                    }
                    Err(error) => {
                        tracing::error!(
                            request_id = id.0,
                            %error,
                            "worker session retirement did not complete"
                        );
                        self.fatal = true;
                    }
                }
            }
        }
    }

    fn drop_worker_session(&mut self, id: RequestId) -> anyhow::Result<()> {
        let acknowledgements = self
            .executor
            .control_wait(ControlOp::DropSession(id), None)?;
        anyhow::ensure!(
            !acknowledgements.is_empty(),
            "worker session retirement returned no acknowledgements"
        );
        for acknowledgement in acknowledgements {
            if let Err(error) = acknowledgement.result {
                anyhow::bail!(
                    "worker rank {} rejected session retirement: {error}",
                    acknowledgement.rank
                );
            }
        }
        Ok(())
    }

    /// Whether a request may keep an additional operation in flight at the
    /// predecessor's not-yet-observed selected point. Eligibility requires an
    /// exact projected cursor and a reachable predicate product for the target
    /// work leaf.
    pub(super) fn can_queue_successor(&self, id: RequestId, target: ForwardMode) -> bool {
        let Some(state) = self.running.get(&id) else {
            return false;
        };
        let Some(queue) = self
            .inflight
            .operations
            .get(&id)
            .filter(|queue| !queue.is_empty())
        else {
            return false;
        };
        if queue.len() >= self.executor.pipeline_depth().max(1)
            || queue.len() >= self.info.max_unresolved_window as usize
            || state.terminal_intent.is_terminal()
            || self.inflight.finishes.contains_key(&id)
            || !Self::device_token_relay_eligible(state)
        {
            return false;
        }
        let Some(predecessor) = queue.back() else {
            return false;
        };
        if target.requires_fixed_parent() {
            return false;
        }
        if !self
            .executor
            .device_products_reachable(predecessor.operation.work, target)
        {
            return false;
        }
        if matches!(
            state.req.policy.trigger,
            uniserve_core::TriggerPolicyDescriptor::RoundCloseThenSuffix { .. }
        ) {
            return false;
        }
        if target == ForwardMode::TokenDecode {
            let feedback_continuation = matches!(
                predecessor.generation_apply().intent,
                TransitionIntent::FeedbackState {
                    is_final_step: true,
                    ..
                }
            ) && state
                .req
                .policy
                .feedback
                .as_ref()
                .is_some_and(|feedback| feedback.sample_continuation)
                && predecessor
                    .operation
                    .outputs
                    .iter()
                    .any(|output| output.kind == ProductKind::Token);
            if feedback_continuation {
                return self
                    .projected_inflight_variant(id)
                    .is_some_and(|variant| variant == ForwardMode::TokenDecode)
                    && state.cursor.und.tokens_emitted.saturating_add(1)
                        < state.req.max_und_tokens;
            }
            if !matches!(state.cursor.phase, Phase::Prefill | Phase::DecodeUnd)
                || (state.cursor.phase == Phase::Prefill && state.starts_gen_after_context())
                || queue.iter().any(|op| {
                    !matches!(
                        op.operation.work,
                        ForwardMode::TokenExtend | ForwardMode::TokenDecode
                    )
                })
            {
                return false;
            }
            let Some(projected) = self.projected_cursor(id) else {
                return false;
            };
            return projected.ingest.prompt_cursor as usize >= state.effective_prompt().len()
                && state.cursor.ingest.mm_cursor >= state.context.images.len()
                && state.cursor.und.tokens_emitted.saturating_add(queue.len())
                    < state.req.max_und_tokens;
        }
        predecessor
            .operation
            .outputs
            .iter()
            .any(|output| output.kind == ProductKind::Completion)
            && self
                .projected_inflight_variant(id)
                .is_some_and(|variant| variant == target)
    }

    pub(super) fn projected_inflight_variant(&self, id: RequestId) -> Option<ForwardMode> {
        if !self.inflight.contains(id) {
            return None;
        }
        let state = self.running.get(&id)?;
        let cursor = self.projected_cursor(id)?;
        if cursor.ingest.prompt_cursor < state.context.prompt_ids.len() as u32
            || cursor.ingest.mm_cursor < state.context.images.len()
        {
            return None;
        }
        let last_intent = self
            .inflight
            .operations
            .get(&id)
            .and_then(|ops| ops.back())
            .map(|inflight| &inflight.generation_apply().intent);
        let chainable = match last_intent {
            Some(TransitionIntent::CommitGen { .. }) => {
                state.req.behavior.generated_image_feedback
                    && state.req.policy.feedback.as_ref().is_some_and(|feedback| {
                        feedback.source == uniserve_core::FeedbackSource::DeviceProduct
                    })
            }
            Some(TransitionIntent::FeedbackState {
                is_final_step: true,
                ..
            }) => state
                .req
                .policy
                .feedback
                .as_ref()
                .is_some_and(|feedback| feedback.sample_continuation),
            _ => true,
        };
        if !chainable {
            return None;
        }
        Some(match cursor.phase {
            Phase::Prefill | Phase::DecodeUnd => ForwardMode::TokenDecode,
            Phase::CloseKv | Phase::FeedbackState => ForwardMode::TokenExtend,
            Phase::PublishKv => ForwardMode::TransferKvPublish,
            Phase::PrepareGen => ForwardMode::MediaPrepare,
            Phase::DenoiseGen => ForwardMode::MediaDenoise,
            Phase::CommitGen => ForwardMode::Materialize,
            Phase::FeedbackEncode => {
                let feedback = state.req.policy.feedback.as_ref()?;
                match feedback.ingest.steps.get(cursor.feedback.ingest_step)? {
                    ImageIngestStep::VaeEncode => ForwardMode::EncodeLatent,
                    ImageIngestStep::VitEncode => ForwardMode::EncodeVision,
                }
            }
            Phase::Encode | Phase::IngestState => return None,
        })
    }

    pub(super) fn device_token_relay_eligible(state: &ReqState) -> bool {
        // A successor may consume the parent's device-selected point before host
        // observation whenever its own sampling state is device-representable
        // from registered coordinates and device products alone. Greedy and
        // stochastic selection (temperature, top-k, top-p, min-p, typical),
        // penalties folded from a device-resident committed count base plus
        // per-operation deltas, requested logprobs, the minimum-token floor and
        // force-finish flag (staged at the successor's exact projected point),
        // static allowed-token, logit-bias, and single-token bad-word masks,
        // positional forced tokens, and device finish predicates (EOS and
        // stop-token ids) all qualify. Stop strings also qualify: the request
        // samples device-continuously and registers bounded provisional
        // descendants, and a matched stop retracts every descendant beyond the
        // exact accepted prefix through the ordered close cutoff. Only
        // multi-token bad-word automata keep the request host-paced, because
        // their next mask depends on the not-yet-observed suffix.
        let sampling = &state.req.sampling;
        (!state.req.behavior.gen_output || state.req.policy.trigger.direct_token().is_some())
            && sampling.bad_words_ids.iter().all(|word| word.len() == 1)
    }

    /// Whether the latest host-resolved token may remain the exact device input
    /// to the next decode. CPU stop and EOS decisions are complete before this
    /// check; a continuing request therefore names the same sampled token.
    pub(super) fn can_reuse_resolved_token_product(&self, id: RequestId) -> bool {
        self.running.get(&id).is_some_and(|state| {
            state.cursor.resources.worker_registered
                && state.cursor.und.tokens_emitted > 0
                && state
                    .latest_device_version
                    .as_ref()
                    .is_some_and(|resident| {
                        self.executor
                            .device_products_reachable(resident.producer, ForwardMode::TokenDecode)
                    })
                && state.is_replayable_text()
                && state.cursor.phase == Phase::DecodeUnd
                && !state.cursor.ingest.round_closing
                && (!state.req.behavior.gen_output
                    || state.req.policy.trigger.direct_token().is_some())
        })
    }

    pub(super) fn can_schedule_next(&self, id: RequestId) -> bool {
        !self.inflight.finishes.contains_key(&id)
            && self.output_window_ready(id)
            && self
                .running
                .get(&id)
                .is_some_and(|state| self.pending_commit_horizon_open(state))
            && (!self.inflight.contains(id)
                || self
                    .projected_inflight_variant(id)
                    .is_some_and(|target| self.can_queue_successor(id, target)))
    }

    /// Whether the request may register another operation without exceeding its
    /// bounded provisional horizon. A request with no held commits is always
    /// open; a stop-string request that has decoded prefixes awaiting the
    /// frontend decision may run ahead by at most the unresolved-window depth,
    /// after which it waits for an acknowledgement so the horizon stays finite.
    pub(super) fn pending_commit_horizon_open(&self, state: &ReqState) -> bool {
        let horizon = self.info.max_unresolved_window.max(1) as usize;
        state.pending_commits.len() < horizon
    }

    pub(super) fn output_window_ready(&self, id: RequestId) -> bool {
        let Some(state) = self.running.get(&id) else {
            return false;
        };
        if state.output.is_closed() {
            return false;
        }
        let available = state.output.available_capacity();
        let reserved = self
            .inflight
            .operations
            .get(&id)
            .into_iter()
            .flatten()
            .map(|operation| operation.generation_apply().output_event_bound)
            .sum::<usize>();
        available
            .saturating_sub(reserved)
            .saturating_sub(OUTPUT_TERMINAL_RESERVE)
            >= self.next_output_bound(id)
    }

    pub(super) fn next_output_bound(&self, id: RequestId) -> usize {
        match self.peek_next_operation_variant(id) {
            Some(ForwardMode::TokenExtend | ForwardMode::TokenDecode) => 4,
            Some(ForwardMode::MediaDenoise) => {
                usize::from(self.denoise_step_burst).saturating_add(2)
            }
            Some(ForwardMode::Materialize) => 3,
            Some(_) | None => 2,
        }
    }

    pub(super) fn register_inflight(
        &mut self,
        operation: Operation,
        apply: SchedulerApply,
        started: Instant,
        queue_us: u64,
    ) {
        self.register_inflight_apply(
            operation,
            InflightApply::Generation(apply),
            started,
            queue_us,
        );
    }

    pub(super) fn register_media_inflight(
        &mut self,
        operation: Operation,
        cursor_after: MediaCursor,
        started: Instant,
    ) {
        self.register_inflight_apply(operation, InflightApply::Media(cursor_after), started, 0);
    }

    pub(super) fn register_inflight_apply(
        &mut self,
        operation: Operation,
        apply: InflightApply,
        started: Instant,
        queue_us: u64,
    ) {
        let request_id = operation.request_key.session_id;
        let domain = self.stats.domains.get(operation.domain);
        let active = domain.active_credits.fetch_add(1, Ordering::Relaxed) + 1;
        domain.peak_credits.fetch_max(active, Ordering::Relaxed);
        domain.launched_operations.fetch_add(1, Ordering::Relaxed);
        domain.queue_us.fetch_add(queue_us, Ordering::Relaxed);
        self.inflight
            .operations
            .entry(request_id)
            .or_default()
            .push_back(InflightOp {
                operation,
                apply,
                started,
            });
    }

    pub(super) fn record_domain_backpressure(&self, domain: uniserve_worker_ipc::Domain) {
        self.stats
            .domains
            .get(domain)
            .backpressure_events
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn record_domain_partition(
        &self,
        accounting: SubmittedPartitionAccounting,
        timing: TimingCounters,
    ) {
        let stats = self.stats.domains.get(accounting.domain);
        stats.completed_partitions.fetch_add(1, Ordering::Relaxed);
        stats
            .launch_us
            .fetch_add(timing.queued_us, Ordering::Relaxed);
        stats
            .device_us
            .fetch_add(timing.device_us, Ordering::Relaxed);
        stats.completion_us.fetch_add(
            timing.copy_us.saturating_add(timing.host_us),
            Ordering::Relaxed,
        );
        if accounting.mixed {
            stats.co_resident_partitions.fetch_add(1, Ordering::Relaxed);
            stats
                .co_resident_us
                .fetch_add(timing.device_us, Ordering::Relaxed);
        }
    }

    pub(super) fn record_domain_completion(
        &self,
        domain: uniserve_worker_ipc::Domain,
        status: OpStatus,
    ) {
        let stats = self.stats.domains.get(domain);
        stats.completed_operations.fetch_add(1, Ordering::Relaxed);
        match status {
            OpStatus::Ok => {}
            OpStatus::Predicated => {
                stats.predicated_operations.fetch_add(1, Ordering::Relaxed);
            }
            OpStatus::Error => {
                stats.error_operations.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub(super) fn reclaim_domain_credit(&self, domain: uniserve_worker_ipc::Domain, failed: bool) {
        let stats = self.stats.domains.get(domain);
        let active = stats.active_credits.load(Ordering::Relaxed);
        if active == 0 {
            tracing::error!(?domain, "domain operation credit accounting underflowed");
            return;
        }
        stats.active_credits.fetch_sub(1, Ordering::Relaxed);
        stats.reclaimed_credits.fetch_add(1, Ordering::Relaxed);
        if failed {
            stats.error_operations.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(super) fn fail_inflight_domain_credits(&self) {
        for inflight in self.inflight.operations.values().flatten() {
            self.reclaim_domain_credit(inflight.operation.domain, true);
        }
    }

    pub(super) fn stage_completion(
        &mut self,
        record: ModelOutput,
        products: Arc<[ProductPayload]>,
    ) {
        let id = record.request_key.session_id;
        let op_id = record.op_id.0;
        let known = self.inflight.operations.get(&id).is_some_and(|queue| {
            queue.iter().any(|inflight| {
                inflight.operation.request_key == record.request_key
                    && inflight.operation.op_id.0 == op_id
            })
        });
        let duplicate = self
            .inflight
            .completions
            .get(&id)
            .is_some_and(|pending| pending.contains_key(&op_id));
        if !known || duplicate {
            self.trace_record(json!({
                "event": "unknown_result_op_id",
                "at_s": now(),
                "request_id": id.0,
                "op_id": op_id,
            }));
            if let Some(state) = self.media_state_mut(id) {
                state.terminal_intent = MediaTerminalIntent::Failure(
                    "worker returned an unknown media operation".to_string(),
                );
            } else if self.running.contains_key(&id) {
                self.finish(id, FinishReason::Error);
            }
            return;
        }
        let arrival_seq = self.inflight.next_arrival();
        self.inflight.completions.entry(id).or_default().insert(
            op_id,
            PendingCompletion {
                record,
                products,
                arrival_seq,
            },
        );
    }

    pub(super) fn apply_media_completion(
        &mut self,
        operation: Operation,
        cursor_after: MediaCursor,
        record: ModelOutput,
    ) {
        let id = record.request_key.session_id;
        let Some(state) = self.media_state(id) else {
            return;
        };
        let already_failed = matches!(state.terminal_intent, MediaTerminalIntent::Failure(_));
        let media_output = record.media_output.clone();
        let media_output_valid =
            (operation.work == ForwardMode::Materialize) == media_output.is_some();
        let valid = record.status == OpStatus::Ok
            && record.request_key == operation.request_key
            && record.op_id == operation.op_id
            && (!operation.advances_state || record.selected_point > 0)
            && media_output_valid;
        if valid
            && let Some(state) = self.media_state_mut(id)
            && state.admission_state == MediaAdmissionState::InFlight
        {
            state.admission_state = MediaAdmissionState::Registered;
        }
        if !already_failed {
            if !valid {
                if let Some(state) = self.media_state_mut(id) {
                    state.terminal_intent =
                        MediaTerminalIntent::Failure("media worker operation failed".to_string());
                }
            } else if let Some(state) = self.media_state_mut(id) {
                state.committed = cursor_after;
                if let Some(output) = media_output {
                    state.artifact = Some(MediaArtifact {
                        handle: output.handle,
                        bytes: output.bytes,
                    });
                }
                if operation.advances_state {
                    state.fixed_parent = VersionRef {
                        request_key: record.request_key,
                        producer_op_id: record.op_id,
                        point: Point::Fixed {
                            point_index: record.selected_point,
                        },
                    };
                }
            }
        }

        let terminal = self.media_state(id).and_then(|state| {
            if self.inflight.contains(id) {
                return None;
            }
            if let MediaTerminalIntent::Failure(message) = &state.terminal_intent {
                Some((
                    MediaEvent::Failed {
                        message: message.clone(),
                    },
                    CloseReason::Error,
                ))
            } else if matches!(state.terminal_intent, MediaTerminalIntent::Cancel)
                || state.event_tx.is_closed()
            {
                Some((MediaEvent::Aborted, CloseReason::Cancelled))
            } else if state.committed.materialized {
                let event = state.artifact.clone().map_or_else(
                    || MediaEvent::Failed {
                        message: "media output was not materialized".to_string(),
                    },
                    |artifact| MediaEvent::Completed { artifact },
                );
                let reason = if matches!(event, MediaEvent::Completed { .. }) {
                    CloseReason::Completed
                } else {
                    CloseReason::Error
                };
                Some((event, reason))
            } else {
                None
            }
        });
        if let Some((event, reason)) = terminal {
            self.finish_media(id, event, reason, None);
        }
    }

    pub(super) fn finish_media(
        &mut self,
        id: RequestId,
        event: MediaEvent,
        reason: CloseReason,
        cutoff: Option<VersionRef>,
    ) {
        let Some(state) = self.take_media_state(id) else {
            return;
        };
        self.order.retain(|candidate| *candidate != id);
        if !matches!(event, MediaEvent::Completed { .. })
            && let Some(artifact) = state.artifact.as_ref()
        {
            if let Ok(name) = std::ffi::CString::new(format!("/{}", artifact.handle)) {
                // SAFETY: the worker-provided handle was validated before it entered media state.
                let result = unsafe { libc::shm_unlink(name.as_ptr()) };
                if result != 0
                    && std::io::Error::last_os_error().raw_os_error() != Some(libc::ENOENT)
                {
                    tracing::warn!(
                        request_id = id.0,
                        handle = artifact.handle,
                        "failed to discard unclaimed media shared memory"
                    );
                }
            }
        }
        let _ = state.event_tx.send(event);
        match state.admission_state {
            MediaAdmissionState::Unsubmitted => {
                self.kv_budget.latent_pages.release(id);
                let _ = self.kv_budget.request_slots.release(state.request_pool_idx);
                return;
            }
            MediaAdmissionState::InFlight => {
                if let Err(error) = self.drop_worker_session(id) {
                    if error.downcast_ref::<WorkerLossError>().is_some() {
                        self.on_executor_error(error);
                        return;
                    }
                    tracing::error!(
                        request_id = id.0,
                        %error,
                        "failed to retire an unregistered media admission"
                    );
                    self.fatal = true;
                    return;
                }
                self.kv_budget.latent_pages.release(id);
                if let Err(error) = self.kv_budget.request_slots.release(state.request_pool_idx) {
                    tracing::error!(
                        request_id = id.0,
                        request_pool_idx = state.request_pool_idx,
                        error,
                        "failed to release media request slot"
                    );
                    self.fatal = true;
                }
                return;
            }
            MediaAdmissionState::Registered => {}
        }
        let request_key = state.admission.request_key;
        self.pending_controls.push_back(Control::Close {
            request_key,
            control_seq: 1,
            cutoff: cutoff.unwrap_or(state.fixed_parent),
            reason,
        });
        self.retiring_media.insert(
            id,
            RetiringMedia {
                request_key,
                request_pool_idx: state.request_pool_idx,
            },
        );
    }

    /// Resolve the front in-flight op for `id` by the worker's echoed `op_id`.
    pub(super) fn pop_inflight(
        &mut self,
        request_key: RequestKey,
        op_id: u64,
    ) -> Option<(Operation, InflightApply, Instant)> {
        let inflight = self.inflight.pop(request_key, op_id)?;
        self.reclaim_domain_credit(inflight.operation.domain, false);
        Some((inflight.operation, inflight.apply, inflight.started))
    }

    pub(super) fn release_products(&mut self, products: Vec<ProductRef>) {
        if products.is_empty() {
            return;
        }
        let mut handles = products
            .into_iter()
            .map(|product| u64::from(product.generation))
            .collect::<Vec<_>>();
        handles.sort_unstable();
        handles.dedup();
        self.gated_control(ControlOp::ReleaseProducts(handles));
    }

    /// Failure policy after an executor/worker error.
    /// A typed non-fatal [`WorkerExecError`] fails the in-flight requests but
    /// keeps the engine alive to serve subsequent requests; anything else (a
    /// fatal worker error, ring/transport death) latches the engine fatal.
    ///
    /// The typed taxonomy's `code` and `retryable` drive
    /// real policy rather than being log-only. The worker's `fatal` bit is the
    /// baseline, but the host *escalates* host-bug classes to fatal even when
    /// the worker marked them non-fatal (a `SCHEDULER_BUG`/`INVARIANT_VIOLATION`
    /// means the control plane can no longer be trusted), and it picks the log
    /// severity by class so a benign `InputError` does not spam warnings.
    pub(super) fn on_executor_error(&mut self, e: anyhow::Error) {
        if e.downcast_ref::<WorkerLossError>().is_some() {
            if self.executor.resets_all_state_after_worker_loss() {
                tracing::warn!("worker state was reset; terminating affected live sessions: {e}");
                self.fail_all_after_worker_loss(&e.to_string());
            } else {
                tracing::error!("worker state was lost without a complete executor reset: {e}");
                self.fatal = true;
                self.fail_all_running(&e.to_string());
            }
            return;
        }
        let exec = e.downcast_ref::<WorkerExecError>();
        let worker_fatal = exec.map(|w| w.fatal).unwrap_or(true);
        let code = exec.and_then(|w| w.code.as_deref());
        let retryable = exec.map(|w| w.retryable).unwrap_or(false);
        // Host-bug classes are latched fatal regardless of the worker's bit:
        // continuing to schedule against a violated invariant is unsafe.
        let host_escalates_fatal =
            matches!(code, Some("SchedulerBug") | Some("InvariantViolation"));
        let fatal = worker_fatal || host_escalates_fatal;
        if fatal {
            tracing::error!(
                ?code,
                retryable,
                "fatal executor error (engine will stop): {e}"
            );
            self.fatal = true;
        } else if matches!(code, Some("InputError")) {
            // A malformed request is the client's fault, not a worker problem:
            // fail just that request at info level instead of warn-spam.
            tracing::info!(?code, "request rejected by worker (failing in-flight): {e}");
        } else {
            // Non-fatal worker errors keep the worker up. `retryable` is
            // surfaced so a recoverable class (OOM/transient) is visible to
            // operators; no automatic requeue is attempted here.
            tracing::warn!(
                ?code,
                retryable,
                "non-fatal worker error (failing in-flight, worker stays up): {e}"
            );
        }
        self.fail_all_inflight(&format!("{e}"));
    }

    pub(super) fn apply_result(&mut self, report: CompletionReport) {
        let result_step_id = report.step_id;
        let mut returned_accounting = HashMap::with_capacity(report.partitions.len());
        let mut invalid_partition = false;
        let batch_complete =
            if let Some(pending) = self.inflight.batch_partitions.get_mut(&result_step_id) {
                for partition in &report.partitions {
                    if let Some(accounting) = pending.remove(&partition.partition_id) {
                        if accounting.operation_count != partition.completions.len() {
                            tracing::error!(
                                step_id = result_step_id,
                                partition_id = partition.partition_id,
                                expected_operations = accounting.operation_count,
                                actual_completions = partition.completions.len(),
                                "executor returned an incomplete partition"
                            );
                            invalid_partition = true;
                        }
                        returned_accounting.insert(partition.partition_id, accounting);
                    } else {
                        tracing::error!(
                            step_id = result_step_id,
                            partition_id = partition.partition_id,
                            "executor returned a duplicate or unknown partition"
                        );
                        invalid_partition = true;
                    }
                }
                pending.is_empty()
            } else {
                tracing::error!(
                    step_id = result_step_id,
                    "executor returned a result for an unknown batch"
                );
                invalid_partition = true;
                false
            };
        if invalid_partition {
            self.fatal = true;
            self.fail_all_running("executor returned an invalid completion partition");
            return;
        }
        let forward_stats = report
            .partitions
            .iter()
            .filter_map(|partition| partition.forward_stats.clone())
            .collect::<Vec<_>>();
        let completion_count = report
            .partitions
            .iter()
            .map(|partition| partition.completions.len())
            .sum();
        let trace_enabled = self.trace_enabled();
        let mut domain_partition_trace = trace_enabled.then(Vec::new);
        for partition in &report.partitions {
            let accounting = returned_accounting[&partition.partition_id];
            let timing = partition.completions.iter().fold(
                TimingCounters::default(),
                |mut aggregate, record| {
                    aggregate.queued_us = aggregate.queued_us.max(record.timing_counters.queued_us);
                    aggregate.device_us = aggregate.device_us.max(record.timing_counters.device_us);
                    aggregate.copy_us = aggregate.copy_us.max(record.timing_counters.copy_us);
                    aggregate.host_us = aggregate.host_us.max(record.timing_counters.host_us);
                    aggregate
                },
            );
            self.record_domain_partition(accounting, timing);
            for record in &partition.completions {
                self.record_domain_completion(accounting.domain, record.status);
            }
            if let Some(worker_exec_us) = partition.worker_exec_us {
                let groups = self
                    .inflight
                    .batch_group_worker_exec_us
                    .get_mut(&result_step_id)
                    .expect("known batch has worker timing state");
                groups
                    .entry(accounting.submission_group)
                    .and_modify(|current| *current = (*current).max(worker_exec_us))
                    .or_insert(worker_exec_us);
            }
            if let Some(trace) = domain_partition_trace.as_mut() {
                trace.push(json!({
                    "partition_id": partition.partition_id,
                    "submission_group": accounting.submission_group,
                    "domain": accounting.domain,
                    "operations": accounting.operation_count,
                    "execution": if accounting.mixed { "tensorized_mixed" } else { "domain_homogeneous" },
                    "queue_us": timing.queued_us,
                    "device_us": timing.device_us,
                    "completion_us": timing.copy_us.saturating_add(timing.host_us),
                    "co_resident_us": if accounting.mixed {
                        timing.device_us
                    } else {
                        0
                    },
                }));
            }
        }
        if batch_complete {
            self.inflight.batch_partitions.remove(&result_step_id);
            if let Some(controls) = self.inflight.control_batches.remove(&result_step_id) {
                self.acknowledge_controls(&controls);
            }
        }
        let batch_roundtrip_us = self
            .inflight
            .batch_started
            .get(&result_step_id)
            .map(|start| start.elapsed().as_micros() as u64)
            .unwrap_or(0);
        if batch_complete {
            self.inflight.batch_started.remove(&result_step_id);
            self.inflight.prefill_steps.remove(&result_step_id);
        }
        let worker_us = if batch_complete {
            self.inflight
                .batch_group_worker_exec_us
                .remove(&result_step_id)
                .into_iter()
                .flat_map(|groups| groups.into_values())
                .fold(0, u64::saturating_add)
        } else {
            0
        };
        if batch_complete {
            self.stats
                .timing
                .last_worker_exec_us
                .store(worker_us, Ordering::Relaxed);
        }
        if batch_complete {
            self.stats
                .timing
                .worker_exec_us_total
                .fetch_add(worker_us, Ordering::Relaxed);
            self.stats
                .timing
                .batch_roundtrip_us_total
                .fetch_add(batch_roundtrip_us, Ordering::Relaxed);
            self.stats
                .timing
                .batch_timing_count
                .fetch_add(1, Ordering::Relaxed);
        }
        let forward_stats_trace = trace_enabled.then(|| {
            forward_stats
                .iter()
                .map(worker_forward_stats_trace)
                .collect::<Vec<_>>()
        });
        for stats in &forward_stats {
            self.record_worker_forward_stats(Some(stats));
        }
        for partition in report.partitions {
            let products = Arc::<[ProductPayload]>::from(partition.products);
            for record in partition.completions {
                self.stage_completion(record, Arc::clone(&products));
            }
        }
        let mut resolved_ops = trace_enabled.then(|| Vec::with_capacity(completion_count));
        let mut progress_ops = trace_enabled.then(|| Vec::with_capacity(completion_count));
        loop {
            let completions = self.inflight.take_ready();
            if completions.is_empty() {
                break;
            }
            let mut to_resolve = Vec::with_capacity(completions.len());
            for completion in completions {
                let PendingCompletion {
                    record,
                    products,
                    arrival_seq,
                } = completion;
                let id = record.request_key.session_id;
                let op_id = record.op_id.0;
                let Some((operation, apply, started)) =
                    self.pop_inflight(record.request_key, op_id)
                else {
                    self.trace_record(json!({
                        "event": "unknown_result_op_id",
                        "at_s": now(),
                        "request_id": id.0,
                        "op_id": op_id,
                    }));
                    if let Some(state) = self.media_state_mut(id) {
                        state.terminal_intent = MediaTerminalIntent::Failure(
                            "worker returned an out-of-order media operation".to_string(),
                        );
                    } else if self.running.contains_key(&id) {
                        self.finish(id, FinishReason::Error);
                    }
                    continue;
                };
                let operation_variant = operation.work;
                let roundtrip_us = started.elapsed().as_micros() as u64;
                let apply = match apply {
                    InflightApply::Media(cursor_after) => {
                        self.apply_media_completion(operation, cursor_after, record);
                        continue;
                    }
                    InflightApply::Generation(apply) => apply,
                };
                let view = SequenceView::from_report(&record, products.as_ref());
                let sampled_token_ids_len = view.committed_tokens.len();
                let sampled_token_ids_last = view.committed_tokens.last().copied();
                if let Some(resolved_ops) = resolved_ops.as_mut() {
                    let image_hw = view
                        .image_png
                        .as_deref()
                        .and_then(|png| validate_png_artifact(png, None))
                        .map(|metadata| (metadata.height, metadata.width));
                    resolved_ops.push(json!({
                        "request_id": id.0,
                        "op_id": op_id,
                        "operation_type": operation_variant.as_str(),
                        "domain": operation.domain,
                        "transition_delta": &apply.intent,
                        "transition_replayability": apply.replayability_after_apply,
                        "roundtrip_us": roundtrip_us,
                        "worker_queue_us": record.timing_counters.queued_us,
                        "device_us": record.timing_counters.device_us,
                        "completion_copy_us": record.timing_counters.copy_us,
                        "completion_ready_to_observed_us": record.timing_counters.host_us,
                        "sampled_token": sampled_token_ids_last.is_some(),
                        "sampled_token_ids_len": sampled_token_ids_len,
                        "sampled_token_ids_last": sampled_token_ids_last,
                        "flow_done": view.flow_done,
                        "steps_completed": record.logical_lengths.latent_len,
                        "image_done": view.image_png.is_some(),
                        "image_hw": image_hw,
                        "kv_tokens": view.kv_visible_len,
                        "product_handle": view.encode_generation,
                    }));
                }
                let predicated_parent_point = (record.status == OpStatus::Predicated).then(|| {
                    self.running
                        .get(&id)
                        .map_or(0, |state| state.version.min(u64::from(u32::MAX)) as u32)
                });
                if let Err(error) = apply.validate_result(
                    &operation,
                    &record,
                    products.as_ref(),
                    predicated_parent_point,
                ) {
                    self.trace_record(json!({
                        "event": "transition_validation_failed",
                        "at_s": now(),
                        "request_id": id.0,
                        "op_id": op_id,
                        "operation_type": operation_variant.as_str(),
                        "error": error.detail(),
                    }));
                    if self.running.contains_key(&id) {
                        self.finish_after_inflight(id, FinishReason::Error, None);
                    }
                    continue;
                }
                if record.status == OpStatus::Ok && !operation.advances_state {
                    let completion_has_device_consumer =
                        self.inflight.operations.get(&id).is_some_and(|queue| {
                            queue.iter().any(|inflight| {
                                inflight
                                    .operation
                                    .predicate
                                    .as_ref()
                                    .is_some_and(|predicate| {
                                        predicate.producer_op_id == operation.op_id
                                    })
                            })
                        });
                    if !completion_has_device_consumer {
                        let completed_predicates = operation
                            .outputs
                            .iter()
                            .filter(|output| output.kind == ProductKind::Completion)
                            .cloned()
                            .collect::<Vec<_>>();
                        self.release_products(completed_predicates);
                    }
                }
                let semantic_blocked = self
                    .running
                    .get(&id)
                    .is_some_and(|state| state.terminal_intent.is_terminal())
                    || self.inflight.finishes.contains_key(&id);
                if semantic_blocked {
                    self.kv_budget.release_transition(id, &apply);
                    self.finish_pending_if_idle(id);
                    continue;
                }
                let expected_parent = self.fixed_version(id);
                let prefix_versions =
                    token_prefix_versions(Some(&operation), &record, expected_parent.as_ref());
                let cursor_result = if record.status == OpStatus::Ok {
                    self.running.get_mut(&id).map(|state| {
                        state
                            .cursor
                            .apply(&operation, &apply, Some((&record, products.as_ref())))
                    })
                } else {
                    None
                };
                if let Some(Err(error)) = cursor_result {
                    self.trace_record(json!({
                        "event": "cursor_transition_failed",
                        "at_s": now(),
                        "request_id": id.0,
                        "op_id": op_id,
                        "operation_type": operation_variant.as_str(),
                        "error": error.to_string(),
                    }));
                    if self.running.contains_key(&id) {
                        self.finish_after_inflight(id, FinishReason::Error, None);
                    }
                    continue;
                }
                // A state-advancing completion resolves a new point. Ordered commit
                // control emission below decides when that point becomes semantic.
                let advanced = record.status == OpStatus::Ok
                    && record.selected_point > 0
                    && operation.advances_state;
                let latest_device_version = if advanced {
                    let token = operation
                        .outputs
                        .iter()
                        .find(|output| {
                            output.kind == ProductKind::Token
                                && output.storage_class
                                    == uniserve_worker_ipc::StorageClass::RequestRelay
                        })
                        .cloned();
                    token.map(|token| ResidentDeviceVersion {
                        producer: operation.work,
                        token,
                        version: VersionRef {
                            request_key: operation.request_key,
                            producer_op_id: operation.op_id,
                            point: Point::Device {
                                point_index: record.selected_point,
                                selected_point: None,
                            },
                        },
                    })
                } else {
                    None
                };
                let retain_device_version = !self.inflight.contains(id);
                if let Some(state) = self.running.get_mut(&id)
                    && advanced
                {
                    state.version = u64::from(record.selected_point);
                    state.resolved_producer_op_id = record.op_id.0;
                    state.latest_device_version = if retain_device_version {
                        latest_device_version
                    } else {
                        None
                    };
                }
                let selected_fixed = self.fixed_version(id);
                // A stop-string token op defers its semantic commit until the
                // frontend decoder rules on its exact prefix; every other
                // advancing op commits immediately in completion order. The
                // deferred commit is carried to the resolve pass below, where the
                // op's public token count is known and it joins the ordered
                // pending-commit queue.
                let mut deferred_commit: Option<(VersionRef, VersionRef, u64)> = None;
                if advanced
                    && let (Some(expected_parent), Some(selected)) =
                        (expected_parent, selected_fixed.clone())
                {
                    let public_event_limit = self.public_limit_for(id, &apply);
                    let decoder_decision_required = matches!(
                        operation_variant,
                        ForwardMode::TokenExtend
                            | ForwardMode::TokenDecode
                            | ForwardMode::TokenVerify
                    ) && self
                        .running
                        .get(&id)
                        .is_some_and(|state| !state.req.stop_strings.is_empty());
                    if decoder_decision_required {
                        deferred_commit = Some((expected_parent, selected, public_event_limit));
                    } else {
                        self.queue_commit(id, expected_parent, selected, public_event_limit);
                    }
                }
                let release_flow_prefix = operation_variant == ForwardMode::MediaDenoise
                    && record.status == OpStatus::Ok
                    && match &apply.intent {
                        TransitionIntent::DenoiseGen {
                            start_step,
                            step_count,
                            ..
                        } => self.running.get(&id).is_some_and(|state| {
                            start_step.saturating_add(*step_count) >= state.req.image.steps
                        }),
                        _ => false,
                    };
                if operation_variant == ForwardMode::MediaDenoise
                    && record.status == OpStatus::Ok
                    && let Some(prefix) = self
                        .running
                        .get_mut(&id)
                        .and_then(|state| state.flow_prefix.as_mut())
                {
                    prefix.materialized = true;
                }
                self.kv_budget.release_transition(id, &apply);
                if release_flow_prefix {
                    self.release_flow_prefix(id);
                }
                if matches!(
                    operation_variant,
                    ForwardMode::MediaDenoise | ForwardMode::Materialize
                ) {
                    let consumed_latents = operation
                        .inputs
                        .iter()
                        .filter(|product| product.kind == ProductKind::Latent)
                        .cloned()
                        .collect::<Vec<_>>();
                    if !consumed_latents.is_empty() {
                        self.release_products(consumed_latents);
                    }
                }
                let priority = completion_priority(operation_variant);
                let public_tokens_before = self
                    .running
                    .get(&id)
                    .map_or(0, |state| state.output.tokens_sent);
                if record.status == OpStatus::Predicated {
                    if let Some(state) = self.running.get_mut(&id) {
                        state.cursor.resources.blocks_sent = state
                            .cursor
                            .resources
                            .blocks_sent
                            .saturating_sub(apply.new_blocks);
                    }
                    let unused_products = operation.outputs.to_vec();
                    self.release_products(unused_products);
                    self.finish_pending_if_idle(id);
                } else {
                    to_resolve.push((
                        priority,
                        arrival_seq,
                        id,
                        operation,
                        apply,
                        view,
                        selected_fixed,
                        public_tokens_before,
                        prefix_versions,
                        deferred_commit,
                    ));
                }
            }
            to_resolve.sort_by_key(|(priority, seq_index, ..)| (*priority, *seq_index));
            for (
                _priority,
                _seq_index,
                id,
                operation,
                apply,
                view,
                selected_fixed,
                public_tokens_before,
                prefix_versions,
                deferred_commit,
            ) in to_resolve
            {
                let token_operation = matches!(
                    operation.work,
                    ForwardMode::TokenExtend | ForwardMode::TokenDecode | ForwardMode::TokenVerify
                );
                if self.running.contains_key(&id) && !self.inflight.finishes.contains_key(&id) {
                    self.resolve(id, operation, apply, view, prefix_versions.clone());
                }
                if token_operation {
                    let mut immediate_commit: Option<(VersionRef, VersionRef, u64)> = None;
                    if let Some(state) = self.running.get_mut(&id) {
                        let emitted_public = state.output.tokens_sent > public_tokens_before;
                        if emitted_public {
                            let emitted = state.output.tokens_sent - public_tokens_before;
                            for (offset, selected) in
                                prefix_versions.into_iter().take(emitted).enumerate()
                            {
                                state
                                    .token_cutoffs
                                    .insert(public_tokens_before + offset + 1, selected);
                            }
                            if emitted > 0
                                && !state.token_cutoffs.contains_key(&state.output.tokens_sent)
                                && let Some(selected) = selected_fixed
                            {
                                state
                                    .token_cutoffs
                                    .insert(state.output.tokens_sent, selected);
                            }
                        }
                        if let Some((expected_parent, selected, public_event_limit)) =
                            deferred_commit
                        {
                            // Chain the deferred commit onto the tip of the
                            // pending queue so the worker applies commits in
                            // exact parent order once the decoder acknowledges
                            // each prefix. An op that emits no public token
                            // joins the token count of the commit ahead of it,
                            // or commits immediately when the queue is empty.
                            let chained_parent = state
                                .pending_commits
                                .back()
                                .map(|pending| pending.selected.clone())
                                .unwrap_or(expected_parent);
                            let token_count = if emitted_public {
                                Some(state.output.tokens_sent)
                            } else {
                                state
                                    .pending_commits
                                    .back()
                                    .and_then(|pending| pending.token_count)
                            };
                            if token_count.is_none() && state.pending_commits.is_empty() {
                                immediate_commit =
                                    Some((chained_parent, selected, public_event_limit));
                            } else {
                                state.pending_commits.push_back(PendingSemanticCommit {
                                    token_count,
                                    expected_parent: chained_parent,
                                    selected,
                                    public_event_limit,
                                });
                            }
                        }
                    }
                    if let Some((expected_parent, selected, public_event_limit)) = immediate_commit
                    {
                        self.queue_commit(id, expected_parent, selected, public_event_limit);
                    }
                }
                self.finish_pending_if_idle(id);
                if let (Some(st), Some(progress_ops)) =
                    (self.running.get(&id), progress_ops.as_mut())
                {
                    progress_ops.push(json!({
                        "request_id": id.0,
                        "phase": st.cursor.phase,
                        "generated_tokens": st.cursor.und.tokens_emitted,
                        "images_done": st.cursor.image_gen.images_done,
                        "image_id": st.cursor.image_gen.image_id,
                        "steps_done": st.cursor.image_gen.steps_done,
                        "pos": st.cursor.und.logical_pos,
                        "kvlen": st.cursor.und.physical_kv_len,
                        "next_token": st.cursor.und.next_token,
                        "text_since_image": st.cursor.und.text_since_image,
                        "gen_branch_pending": st.cursor.image_gen.branch_pending,
                        "context_round_closing": st.cursor.ingest.round_closing,
                    }));
                }
            }
        }
        if let (Some(resolved_ops), Some(progress_ops)) = (resolved_ops, progress_ops) {
            self.trace_record(json!({
                "event": "batch_resolved",
                "at_s": now(),
                "step_id": result_step_id,
                "worker_exec_us": worker_us,
                "host_roundtrip_us": batch_roundtrip_us,
                "batch_complete": batch_complete,
                "domains": domain_partition_trace,
                "forward_stats": forward_stats_trace,
                "batch_size": resolved_ops.len(),
                "ops": resolved_ops,
                "progress": progress_ops,
                "running": self.running.len(),
                "pending": self.pending.len(),
                "in_flight": self.executor.in_flight(),
            }));
        }
    }

    pub(super) fn record_worker_forward_stats(&self, stats: Option<&WorkerForwardStats>) {
        let Some(stats) = stats else {
            return;
        };
        add_worker_forward_map(&self.stats.worker.forward_mode_counts, &stats.mode_counts);
        add_worker_forward_map(&self.stats.worker.forward_mode_tokens, &stats.mode_tokens);
        add_worker_forward_map(&self.stats.worker.forward_mode_us, &stats.mode_us);
        add_worker_forward_map(&self.stats.worker.forward_component_us, &stats.component_us);
        // Fold worker forward stats maps into scheduler stats.
        add_worker_forward_map(
            &self.stats.worker.attention_backend_counts,
            &stats.attention_backend_counts,
        );
        add_worker_forward_map(
            &self.stats.worker.cuda_graph_runtime_mode_counts,
            &stats.cuda_graph_runtime_mode_counts,
        );
        self.stats
            .worker
            .attention_launches
            .fetch_add(stats.attention_launches, Ordering::Relaxed);
        self.stats
            .worker
            .attention_us
            .fetch_add(stats.attention_us, Ordering::Relaxed);
        self.stats
            .worker
            .cuda_graph_captures
            .fetch_add(stats.cuda_graph_captures, Ordering::Relaxed);
        self.stats
            .worker
            .cuda_graph_replays
            .fetch_add(stats.cuda_graph_replays, Ordering::Relaxed);
        self.stats
            .worker
            .cuda_graph_misses
            .fetch_add(stats.cuda_graph_misses, Ordering::Relaxed);
        self.stats
            .worker
            .cuda_graph_fallbacks
            .fetch_add(stats.cuda_graph_fallbacks, Ordering::Relaxed);
        self.stats
            .worker
            .cuda_graph_unpadded_tokens
            .fetch_add(stats.cuda_graph_unpadded_tokens, Ordering::Relaxed);
        self.stats
            .worker
            .cuda_graph_padded_tokens
            .fetch_add(stats.cuda_graph_padded_tokens, Ordering::Relaxed);
        self.stats
            .worker
            .text_decode_token_relay_hits
            .fetch_add(stats.text_decode_token_relay_hits, Ordering::Relaxed);
        self.stats
            .worker
            .text_decode_token_relay_misses
            .fetch_add(stats.text_decode_token_relay_misses, Ordering::Relaxed);
        self.stats
            .worker
            .text_decode_position_relay_hits
            .fetch_add(stats.text_decode_position_relay_hits, Ordering::Relaxed);
        self.stats
            .worker
            .text_decode_position_relay_misses
            .fetch_add(stats.text_decode_position_relay_misses, Ordering::Relaxed);
        self.stats
            .worker
            .flashinfer_decode_plan_calls
            .fetch_add(stats.flashinfer_decode_plan_calls, Ordering::Relaxed);
        self.stats
            .worker
            .flashinfer_decode_plan_reuses
            .fetch_add(stats.flashinfer_decode_plan_reuses, Ordering::Relaxed);
        self.stats
            .worker
            .flashinfer_decode_plan_rows
            .fetch_add(stats.flashinfer_decode_plan_rows, Ordering::Relaxed);
        self.stats
            .worker
            .flashinfer_decode_plan_indices
            .fetch_add(stats.flashinfer_decode_plan_indices, Ordering::Relaxed);
        self.stats
            .worker
            .flashinfer_decode_graph_plan_calls
            .fetch_add(stats.flashinfer_decode_graph_plan_calls, Ordering::Relaxed);
        self.stats
            .worker
            .flashinfer_decode_graph_plan_reuses
            .fetch_add(stats.flashinfer_decode_graph_plan_reuses, Ordering::Relaxed);
        self.stats
            .worker
            .spec_verify_rows
            .fetch_add(stats.spec_verify_rows, Ordering::Relaxed);
        self.stats
            .worker
            .spec_verify_draft_tokens
            .fetch_add(stats.spec_verify_draft_tokens, Ordering::Relaxed);
        self.stats
            .worker
            .spec_verify_accepted_tokens
            .fetch_add(stats.spec_verify_accepted_tokens, Ordering::Relaxed);
        self.stats
            .worker
            .spec_verify_rejected_tokens
            .fetch_add(stats.spec_verify_rejected_tokens, Ordering::Relaxed);
        self.stats
            .worker
            .spec_verify_committed_tokens
            .fetch_add(stats.spec_verify_committed_tokens, Ordering::Relaxed);
        if !stats.spec_verify_path_counts.is_empty() {
            let mut path_counts = self
                .stats
                .worker
                .spec_verify_path_counts
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for (path, count) in &stats.spec_verify_path_counts {
                *path_counts.entry(path.clone()).or_default() += *count;
            }
        }
    }

    pub(super) fn fail_all_inflight(&mut self, msg: &str) {
        self.fail_inflight_domain_credits();
        // the submitted batches whose results will now never return are
        // failed here, so drop their pending submit-timestamps too — otherwise
        // `batch_started` accumulates orphaned entries for every failed batch.
        let (ids, controls) = self.inflight.clear_failed();
        self.acknowledge_controls(&controls);
        for id in ids {
            if self.media_state(id).is_some() {
                self.finish_media(
                    id,
                    MediaEvent::Failed {
                        message: msg.to_string(),
                    },
                    CloseReason::Error,
                    None,
                );
            } else if self.running.contains_key(&id) {
                self.emit(
                    id,
                    GenerationEvent::Error {
                        message: msg.to_string(),
                    },
                );
                self.finish(id, FinishReason::Error);
            }
        }
    }

    fn fail_all_after_worker_loss(&mut self, message: &str) {
        self.fail_inflight_domain_credits();
        let _ = self.inflight.clear_failed();
        self.pending_controls.clear();

        for state in self.running_media.values_mut() {
            state.admission_state = MediaAdmissionState::Unsubmitted;
        }
        let media = self.media_ids();
        for id in media {
            self.finish_media(
                id,
                MediaEvent::Failed {
                    message: message.to_string(),
                },
                CloseReason::Error,
                None,
            );
        }

        for state in self.running.values_mut() {
            state.cursor.resources.worker_registered = false;
        }
        let ids = self.running.keys().copied().collect::<Vec<_>>();
        for id in ids {
            self.emit(
                id,
                GenerationEvent::Error {
                    message: message.to_string(),
                },
            );
            self.finish(id, FinishReason::Error);
        }

        let retiring_media = std::mem::take(&mut self.retiring_media);
        for (id, retiring) in retiring_media {
            self.kv_budget.latent_pages.release(id);
            if let Err(error) = self
                .kv_budget
                .request_slots
                .release(retiring.request_pool_idx)
            {
                tracing::error!(
                    request_id = id.0,
                    request_pool_idx = retiring.request_pool_idx,
                    error,
                    "failed to release media request slot after worker loss"
                );
                self.fatal = true;
            }
        }

        let retiring_sessions = std::mem::take(&mut self.retiring_sessions);
        for (id, retiring) in retiring_sessions {
            self.kv_budget.latent_pages.release(id);
            if let Some(prefix) = retiring.flow_prefix
                && let Err(error) = self
                    .kv_budget
                    .request_slots
                    .release(prefix.request_pool_idx)
            {
                tracing::error!(
                    request_id = id.0,
                    request_pool_idx = prefix.request_pool_idx,
                    error,
                    "failed to release flow-prefix request slot after worker loss"
                );
                self.fatal = true;
            }
            if let Err(error) = self
                .kv_budget
                .request_slots
                .release(retiring.request_pool_idx)
            {
                tracing::error!(
                    request_id = id.0,
                    request_pool_idx = retiring.request_pool_idx,
                    error,
                    "failed to release request slot after worker loss"
                );
                self.fatal = true;
            }
        }
        self.pending_controls.clear();
        self.kv_budget.reset_after_worker_loss(&self.info);
    }

    pub(super) fn fail_all_running(&mut self, message: &str) {
        self.fail_inflight_domain_credits();
        let _ = self.inflight.clear_failed();
        while let Some(submission) = self.pending_media.pop_front() {
            let _ = submission.event_tx.send(MediaEvent::Failed {
                message: message.to_string(),
            });
        }
        let media = self.media_ids();
        for id in media {
            self.finish_media(
                id,
                MediaEvent::Failed {
                    message: message.to_string(),
                },
                CloseReason::Error,
                None,
            );
        }
        let ids = self.running.keys().copied().collect::<Vec<_>>();
        for id in ids {
            self.emit(
                id,
                GenerationEvent::Error {
                    message: message.to_string(),
                },
            );
            self.finish(id, FinishReason::Error);
        }
    }
}
