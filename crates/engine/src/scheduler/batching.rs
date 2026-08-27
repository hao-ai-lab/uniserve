use super::*;

impl Scheduler {
    /// Assemble the per-step batch: walk the priority order, ask each request for at most one op, clip prefill chunks to the remaining token budget, and pair first-dispatch requests with their typed admission record.
    pub(super) fn assemble(&mut self) -> (Vec<Admission>, Vec<PlannedTransition>) {
        let ids = self.assembly_order();
        let lane = self.select_assembly_lane(&ids);
        let (new_reqs, ops) = self.assemble_pass(&ids, lane);
        if ops.is_empty() && lane == Some(AssemblyLane::Prefill) {
            // Prefill runs first *if possible* (SGLang's order). When no
            // prefill op could actually be built (e.g. blocked on KV memory
            // it cannot displace), fall through to the decode lane instead of
            // idling — otherwise a starved waiting prompt would stall ready
            // decodes forever.
            return self.assemble_pass(&ids, Some(AssemblyLane::Decode));
        }
        (new_reqs, ops)
    }

    pub(super) fn assemble_pass(
        &mut self,
        ids: &[RequestId],
        lane: Option<AssemblyLane>,
    ) -> (Vec<Admission>, Vec<PlannedTransition>) {
        let mut admissions: Vec<Admission> = Vec::new();
        let mut ops: Vec<PlannedTransition> = Vec::new();
        let mut selected: HashSet<RequestId> = HashSet::new();
        // vLLM's per-step token budget with the clip rule: the budget, not the
        // chunk threshold, is the binding constraint.
        let mut budget: usize = self.config.max_num_batched_tokens;
        // Text prefill tokens may ride along inside a decode batch (mixed
        // extend+decode forward): the prompt work then shares the decode
        // step's weight sweep instead of paying a full sweep of its own.
        // Mixed prefill rows are appended after the decode rows so the batch
        // keeps an extend row last (graph token-bucket padding extends the
        // last row).
        let mut mixed_left: usize = if lane == Some(AssemblyLane::Decode) {
            self.config.mixed_prefill_tokens
        } else {
            0
        };
        let mut mixed_ops: Vec<PlannedTransition> = Vec::new();
        let denoise_occupies_decode_pipeline =
            lane == Some(AssemblyLane::Decode) && self.any_denoise_inflight();
        for id in ids.iter().copied() {
            if ops.len() + mixed_ops.len() >= self.config.max_batch {
                break;
            }
            if budget == 0 {
                break;
            }
            if !self.can_schedule_next(id) {
                continue;
            }
            let cancelled = self
                .running
                .get(&id)
                .map(|state| state.terminal_intent.is_terminal())
                .unwrap_or(true);
            if cancelled {
                continue;
            }
            let next_type = self.peek_next_operation_variant(id);
            // When a decode pass admits text prefill rows, the worker receives a
            // single mixed forward. There is no `supports_mixed_op_kinds` gate;
            // co-batched rows shift each other's numerics only through inherent
            // batched-kernel FP non-invariance, not structural corruption.
            let mut mixed_prefill = false;
            if let (Some(target), Some(operation_variant)) = (lane, next_type)
                && {
                    let candidate_lane = assembly_lane(operation_variant);
                    matches!(candidate_lane, AssemblyLane::Prefill | AssemblyLane::Decode)
                        && candidate_lane != target
                }
            {
                mixed_prefill = target == AssemblyLane::Decode
                    && operation_variant == ForwardMode::TokenExtend
                    && mixed_left > 0
                    && self.running.get(&id).is_some_and(|st| {
                        st.is_replayable_text() && !st.req.sampling.prompt_logprobs_requested()
                    });
                if !mixed_prefill {
                    continue;
                }
            }
            if next_type == Some(ForwardMode::GenFlow)
                && (denoise_occupies_decode_pipeline
                    || !self.flow_prefix_is_schedulable(id)
                    || !self.can_schedule_denoise(id))
            {
                continue;
            }
            // Diagnostic: a denoise step and text rows in one forward take the
            // packed route, which costs more than running each on its own. When
            // the flag is set a flow op only opens an empty batch, and the loop
            // below closes the batch as soon as one is placed.
            if self.flow_exclusive_batch
                && next_type == Some(ForwardMode::GenFlow)
                && !(ops.is_empty() && mixed_ops.is_empty())
            {
                continue;
            }
            // Build one operation when its exact resident resources fit.
            let op_budget = if mixed_prefill {
                budget.min(mixed_left)
            } else {
                budget
            };
            if let Some(mut op) = self.next_transition(id, op_budget) {
                if mixed_prefill {
                    mixed_left = mixed_left.saturating_sub(planned_op_token_cost(&op));
                }
                budget = budget.saturating_sub(planned_op_token_cost(&op));
                if !self.reserve_transition_resources(&mut op) {
                    self.record_domain_backpressure(op.domain);
                    self.return_unsent_blocks(id, &op);
                    tracing::debug!(
                        request_id = id.0,
                        "operation registration is paused by physical resource pressure"
                    );
                    continue;
                }
                let finish_token_ids = self
                    .running
                    .get(&id)
                    .map(|state| canonical_continuation_stop_token_ids(&state.req, &self.ctrl.eos))
                    .unwrap_or_default();
                if let Some(st) = self.running.get_mut(&id)
                    && !st.cursor.resources.worker_registered
                {
                    st.cursor.resources.worker_registered = true;
                    let request_key = RequestKey::new(self.authority_id, id, st.epoch);
                    let admission = Admission::new(
                        request_key,
                        st.request_pool_idx,
                        Some(UndAdmission {
                            sampling: st.req.sampling.clone(),
                            negative_token_ids: st.context.negative_prompt_ids.clone(),
                            finish_token_ids,
                            initial_position: st.cursor.ingest.prompt_cursor,
                        }),
                        st.req.behavior.gen_output.then(|| GenAdmission {
                            image: st.req.image.clone(),
                        }),
                    )
                    .expect("validated request produces a valid admission");
                    // The admission-root version is produced by the synthetic
                    // admission, identified by operation-id 0 — the sentinel the
                    // worker seeds as its initial committed version — so the first
                    // operation's fixed parent matches the worker's committed state.
                    st.resolved_semantic = admission.digest.clone();
                    st.resolved_producer_op_id = 0;
                    st.committed_semantic = admission.digest.clone();
                    st.committed_producer_op_id = 0;
                    st.latest_device_version = None;
                    st.admission_digest = Some(admission.digest.clone());
                    st.token_cutoffs.clear();
                    st.token_cutoffs.insert(
                        st.public_token_seq,
                        VersionRef {
                            request_key,
                            producer_op_id: OpId(0),
                            point: Point::Fixed {
                                point_index: 0,
                                semantic_digest: admission.digest.clone(),
                            },
                        },
                    );
                    admissions.push(admission);
                }
                selected.insert(id);
                let placed_flow = op.operation_variant == ForwardMode::GenFlow;
                if mixed_prefill {
                    mixed_ops.push(op);
                } else {
                    ops.push(op);
                }
                if self.flow_exclusive_batch && placed_flow {
                    budget = 0;
                }
            }
        }
        ops.extend(mixed_ops);
        (admissions, ops)
    }

    pub(super) fn select_assembly_lane(&self, ids: &[RequestId]) -> Option<AssemblyLane> {
        let mut projected_decode_ready = false;
        let mut committed_decode_ready = false;
        let mut prefill_ready = false;
        for id in ids.iter().copied() {
            if self
                .running
                .get(&id)
                .map(|state| state.terminal_intent.is_terminal())
                .unwrap_or(true)
            {
                continue;
            }
            let Some(operation_type) = self.peek_next_operation_variant(id) else {
                continue;
            };
            match assembly_lane(operation_type) {
                AssemblyLane::Decode => {
                    if self.can_schedule_next(id) {
                        if self.has_inflight(id) {
                            projected_decode_ready = true;
                        } else {
                            committed_decode_ready = true;
                        }
                    }
                }
                AssemblyLane::Prefill => {
                    if self.can_schedule_next(id) {
                        prefill_ready = true;
                    }
                }
                AssemblyLane::Other => {}
            }
        }
        if prefill_ready && self.prefill_steps.len() < PREFILL_WINDOW_CREDITS {
            Some(AssemblyLane::Prefill)
        } else if committed_decode_ready || projected_decode_ready {
            Some(AssemblyLane::Decode)
        } else if prefill_ready {
            Some(AssemblyLane::Prefill)
        } else {
            None
        }
    }

    pub(super) fn assembly_order(&self) -> Vec<RequestId> {
        let mut ids: Vec<(usize, RequestId)> = self
            .order
            .iter()
            .enumerate()
            .filter_map(|(index, id)| self.running.get(id).is_some().then_some((index, *id)))
            .collect();
        ids.sort_by_key(|(idx, id)| (self.assembly_priority(*id), *idx));
        ids.into_iter().map(|(_, id)| id).collect()
    }

    pub(super) fn assembly_priority(&self, id: RequestId) -> u8 {
        match self.peek_next_operation_variant(id) {
            Some(
                ForwardMode::EncodeVision | ForwardMode::EncodeLatent | ForwardMode::TokenExtend,
            ) => 0,
            Some(
                ForwardMode::TokenDecode
                | ForwardMode::TokenVerify
                | ForwardMode::Materialize
                | ForwardMode::TransferKvInstall,
            ) => 1,
            Some(ForwardMode::GenFlow | ForwardMode::GenDecode) => 2,
            Some(
                ForwardMode::Draft
                | ForwardMode::GenTransition
                | ForwardMode::TransferProduct
                | ForwardMode::TransferKvPublish,
            )
            | None => 3,
        }
    }

    pub(super) fn peek_next_operation_variant(&self, id: RequestId) -> Option<ForwardMode> {
        let st = self.running.get(&id)?;
        if st.cursor.image_gen.branch_pending {
            return None;
        }
        if let Some(variant) = self.projected_inflight_variant(id) {
            return Some(variant);
        }
        if st.has_context_images()
            && st.cursor.lifecycle.phase == Phase::Prefill
            && self.projected_cursor(id).is_some_and(|cursor| {
                st.context
                    .images
                    .get(st.cursor.ingest.mm_cursor)
                    .is_some_and(|item| item.position as usize == cursor.prompt_cursor as usize)
            })
        {
            return st.pending_image_step().map(|step| match step {
                ImageIngestStep::VaeEncode => ForwardMode::EncodeLatent,
                ImageIngestStep::VitEncode => ForwardMode::EncodeVision,
            });
        }
        Some(match st.cursor.lifecycle.phase {
            Phase::Encode => match st.pending_image_step()? {
                ImageIngestStep::VaeEncode => ForwardMode::EncodeLatent,
                ImageIngestStep::VitEncode => ForwardMode::EncodeVision,
            },
            Phase::IngestState => ForwardMode::TokenExtend,
            Phase::Prefill => ForwardMode::TokenExtend,
            Phase::DecodeUnd => ForwardMode::TokenDecode,
            Phase::CloseKv => ForwardMode::TokenExtend,
            Phase::PublishKv => ForwardMode::TransferKvPublish,
            Phase::TransitionGen => ForwardMode::GenTransition,
            Phase::DenoiseGen if st.cursor.image_gen.steps_done >= st.req.image.steps => {
                ForwardMode::Materialize
            }
            Phase::DenoiseGen => ForwardMode::GenFlow,
            Phase::CommitGen => ForwardMode::Materialize,
            Phase::FeedbackEncode => {
                let feedback = st.req.policy.feedback.as_ref()?;
                match feedback.ingest.steps.get(st.cursor.feedback.ingest_step)? {
                    ImageIngestStep::VaeEncode => ForwardMode::EncodeLatent,
                    ImageIngestStep::VitEncode => ForwardMode::EncodeVision,
                }
            }
            Phase::FeedbackState => ForwardMode::TokenExtend,
        })
    }

    pub(super) fn submit_batch(
        &mut self,
        admissions: Vec<Admission>,
        transitions: Vec<PlannedTransition>,
        mut controls: Vec<Control>,
    ) -> bool {
        let _span =
            tracing::trace_span!("scheduler.submit_batch", ops = transitions.len()).entered();
        self.step_id += 1;
        let step = self.step_id;
        let submit_at = Instant::now();
        let mut wire_ops = Vec::with_capacity(transitions.len());
        let mut input_products = Vec::new();
        let mut trace_ops = self
            .trace_enabled()
            .then(|| Vec::with_capacity(transitions.len()));
        let admitted_request_keys = admissions
            .iter()
            .map(|admission| admission.request_key)
            .collect::<HashSet<_>>();
        let mut block_tables = HashMap::with_capacity(transitions.len());
        let mut new_cache_pages = HashMap::with_capacity(transitions.len());
        let mut forward_rows = HashMap::with_capacity(transitions.len());
        let mut latent_placements = HashMap::with_capacity(transitions.len());
        for mut transition in transitions {
            let oid = self.next_op_id;
            self.next_op_id += 1;
            let request_id = transition.request_id;
            let Some((epoch, latest_device_version)) =
                self.running.get(&request_id).and_then(|state| {
                    state
                        .admission_digest
                        .as_ref()
                        .map(|_| (state.epoch, state.latest_device_version.clone()))
                })
            else {
                tracing::error!(
                    request_id = request_id.0,
                    "planned operation lost its session"
                );
                self.fatal = true;
                return false;
            };
            if self.has_inflight(request_id)
                && !self.can_queue_successor(request_id, transition.operation_variant)
            {
                tracing::error!(
                    request_id = request_id.0,
                    "scheduler attempted to issue an unsafe projected successor"
                );
                self.fatal = true;
                return false;
            }
            let request_key = RequestKey::new(self.authority_id, request_id, epoch);
            // A device successor roots on the exact selected-point product of
            // either its in-flight predecessor or the latest resolved operation.
            // The first operation and CPU-gated transitions use the fixed
            // semantically committed parent.
            let projected_successor = self.has_inflight(request_id);
            let reusable_device_version =
                if !projected_successor && self.can_reuse_resolved_token_product(request_id) {
                    latest_device_version
                } else {
                    None
                };
            let (parent, predicate) = if projected_successor {
                let Some(parent) = self.projected_parent(request_id) else {
                    tracing::error!(
                        request_id = request_id.0,
                        "projected successor has no semantic parent"
                    );
                    self.fatal = true;
                    return false;
                };
                let Some(predecessor) = self
                    .inflight_ops
                    .get(&request_id)
                    .and_then(|queue| queue.back())
                    .map(|op| &op.operation)
                else {
                    tracing::error!(
                        request_id = request_id.0,
                        "projected successor lost its predecessor operation"
                    );
                    self.fatal = true;
                    return false;
                };
                let predicate_kind = if transition.operation_variant == ForwardMode::TokenDecode {
                    ProductKind::Token
                } else {
                    ProductKind::Completion
                };
                (
                    parent,
                    predecessor
                        .outputs
                        .iter()
                        .find(|output| output.kind == predicate_kind)
                        .cloned(),
                )
            } else if let Some(parent) = reusable_device_version {
                (parent.version, Some(parent.token))
            } else {
                let Some(parent) = self.fixed_version(request_id) else {
                    self.fatal = true;
                    return false;
                };
                (parent, None)
            };
            transition.predicate = predicate;
            transition.control_seq = self
                .running
                .get(&request_id)
                .map_or(0, |state| state.control_seq);
            let kv_lengths = transition_kv_lengths(&transition.delta);
            let new_page_count = transition.new_blocks.len();
            let mut operation_block_tables = Vec::new();
            let mut operation_new_cache_pages = Vec::new();
            let mut operation_forward_rows = Vec::new();
            if let Some(lengths) = kv_lengths {
                let table_changed =
                    admitted_request_keys.contains(&request_key) || new_page_count > 0;
                for group_id in 0..self.kv_state().block_pool.num_groups() {
                    let page_ids = self
                        .running
                        .get(&request_id)
                        .and_then(|state| state.block_tables.get(group_id))
                        .map(BlockTable::page_ids)
                        .unwrap_or_default();
                    let request_pool_idx = self
                        .running
                        .get(&request_id)
                        .map(|state| state.request_pool_idx)
                        .expect("registered request has a live slot");
                    if table_changed {
                        operation_block_tables.push(WireBlockTable {
                            request_pool_idx,
                            group_id: group_id as u32,
                            allocated_tokens: u32::try_from(
                                page_ids.len().saturating_mul(self.caps.block_size as usize),
                            )
                            .unwrap_or(u32::MAX),
                            page_ids: page_ids.clone(),
                        });
                    }
                    let fresh_pages = if admitted_request_keys.contains(&request_key) {
                        let retained_pages = if group_id == 0 {
                            self.running
                                .get(&request_id)
                                .map(|state| {
                                    (state.cursor.ingest.prompt_cursor as usize)
                                        .div_ceil(self.caps.block_size as usize)
                                })
                                .unwrap_or_default()
                                .min(page_ids.len())
                        } else {
                            0
                        };
                        page_ids[retained_pages..].to_vec()
                    } else if new_page_count > 0 {
                        page_ids[page_ids.len().saturating_sub(new_page_count)..].to_vec()
                    } else {
                        Vec::new()
                    };
                    if !fresh_pages.is_empty() {
                        operation_new_cache_pages.push(CachePageAllocation {
                            request_pool_idx,
                            group_id: group_id as u32,
                            page_ids: fresh_pages,
                        });
                    }
                }
                if lengths.input > 0 {
                    operation_forward_rows.push(RowGeometry {
                        operation_index: 0,
                        request_pool_index: self
                            .running
                            .get(&request_id)
                            .map(|state| state.request_pool_idx)
                            .expect("registered request has a live slot"),
                        seq_len: lengths.visible,
                        query_len: lengths.input,
                    });
                }
            }
            let output_event_bound = transition_output_bound(&transition);
            let planned_us = transition.planned_us;
            let reserved_us = transition.reserved_us;
            let registered = transition.register(
                request_key,
                OpId(oid),
                parent,
                &mut self.next_product_generation,
                output_event_bound,
            );
            let (operation, apply, payloads) = match registered {
                Ok(registered) => registered,
                Err(error) => {
                    tracing::error!(
                        request_id = request_id.0,
                        ?error,
                        "scheduler authority exhausted product identity space"
                    );
                    self.fatal = true;
                    self.fail_all_running("scheduler authority exhausted product identity space");
                    return false;
                }
            };
            let operation_identity = (operation.request_key, operation.op_id);
            if operation.work == ForwardMode::GenFlow {
                let conditioning_tokens = match &apply.delta {
                    TransitionDelta::DenoiseGen {
                        physical_kv_len, ..
                    } => *physical_kv_len,
                    _ => unreachable!("generation flow has a non-flow scheduler delta"),
                };
                let query_len = u32::try_from(
                    self.running
                        .get(&request_id)
                        .map(|state| self.num_vae(&state.req.image))
                        .unwrap_or_default()
                        .saturating_add(u64::from(self.caps.commit_marker_tokens)),
                )
                .unwrap_or(u32::MAX);
                let Some(state) = self.running.get_mut(&request_id) else {
                    self.fatal = true;
                    return false;
                };
                let main_slot = state.request_pool_idx;
                let mut alternative = None;
                if let Some(prefix) = state.flow_prefix.as_mut() {
                    let allocated_tokens = prefix
                        .block_tables
                        .first()
                        .map(|table| table.capacity_tokens())
                        .unwrap_or_default();
                    if !prefix.materialized || !prefix.new_pages.is_empty() {
                        for table in &prefix.block_tables {
                            operation_block_tables.push(WireBlockTable {
                                request_pool_idx: prefix.request_pool_idx,
                                group_id: table.group_id() as u32,
                                page_ids: table.page_ids(),
                                allocated_tokens: u32::try_from(allocated_tokens)
                                    .unwrap_or(u32::MAX),
                            });
                        }
                    }
                    for (group_id, pages) in std::mem::take(&mut prefix.new_pages) {
                        if !pages.is_empty() {
                            operation_new_cache_pages.push(CachePageAllocation {
                                request_pool_idx: prefix.request_pool_idx,
                                group_id,
                                page_ids: pages,
                            });
                        }
                    }
                    let prefix_len = state.context.negative_prompt_ids.len() as u32;
                    if prefix_len > 0 && !prefix.materialized {
                        operation_forward_rows.push(RowGeometry {
                            operation_index: 0,
                            request_pool_index: prefix.request_pool_idx,
                            seq_len: 0,
                            query_len: prefix_len,
                        });
                    }
                    alternative = Some((prefix.request_pool_idx, prefix_len));
                }
                for branch in 0..usize::from(cfg_branch_count(&state.req.image).max(1)) {
                    let (request_pool_index, seq_len) = if branch == 0 {
                        (main_slot, conditioning_tokens)
                    } else {
                        alternative.unwrap_or((main_slot, conditioning_tokens))
                    };
                    operation_forward_rows.push(RowGeometry {
                        operation_index: 0,
                        request_pool_index,
                        seq_len,
                        query_len,
                    });
                }
            }
            if !operation_block_tables.is_empty() {
                block_tables.insert(operation_identity, operation_block_tables);
            }
            if !operation_new_cache_pages.is_empty() {
                new_cache_pages.insert(operation_identity, operation_new_cache_pages);
            }
            if !operation_forward_rows.is_empty() {
                forward_rows.insert(operation_identity, operation_forward_rows);
            }
            if matches!(
                operation.work,
                ForwardMode::GenTransition | ForwardMode::GenFlow
            ) || operation
                .inputs
                .iter()
                .any(|reference| reference.kind == uniserve_worker_ipc::ProductKind::Latent)
            {
                let Some(state) = self.running.get(&request_id) else {
                    self.fatal = true;
                    return false;
                };
                let page_table = self.latent_pages.pages_for(request_id).to_vec();
                let latent_units = self.worker_image_latent_units_for(state).max(1);
                let (start_step, step_count) = match &apply.delta {
                    TransitionDelta::DenoiseGen {
                        start_step,
                        step_count,
                        ..
                    } => (u32::from(*start_step), u32::from(*step_count)),
                    TransitionDelta::TransitionGen { .. } => (0, 0),
                    TransitionDelta::CommitGen { step, .. } => (u32::from(*step), 0),
                    _ => {
                        tracing::error!(
                            request_id = request_id.0,
                            operation = operation.work.as_wire_str(),
                            "latent operation has no declared schedule placement"
                        );
                        self.fatal = true;
                        return false;
                    }
                };
                latent_placements.insert(
                    (operation.request_key, operation.op_id),
                    LatentPlacement {
                        request_key: operation.request_key,
                        op_id: operation.op_id,
                        page_table,
                        latent_units: latent_units.min(u64::from(u32::MAX)) as u32,
                        height: state.req.image.height,
                        width: state.req.image.width,
                        start_step,
                        step_count,
                    },
                );
            }
            let operation_variant = operation.work.as_wire_str();
            if let Some(trace_ops) = trace_ops.as_mut() {
                let phase = self
                    .running
                    .get(&request_id)
                    .map(|state| phase_str(state.cursor.lifecycle.phase));
                trace_ops.push(json!({
                    "request_id": request_id.0,
                    "op_id": operation.op_id.0,
                    "operation_type": operation_variant,
                    "phase": phase,
                    "operation": operation_trace(&operation, &apply),
                    "transition": apply.delta.as_str(),
                    "resources": {
                        "release_on_apply": apply.release_on_apply,
                        "replayability_after_apply": apply.replayability_after_apply.as_str(),
                    },
                    "visibility": {
                        "und_tokens": format!("{:?}", apply.visibility.und_tokens),
                        "generated_image": apply.visibility.generated_image,
                    },
                }));
            }
            input_products.extend(payloads);
            let op_key = crate::scheduler::trace::OperationKey::from(&operation);
            let submitted_us = uniserve_core::now_monotonic_us();
            self.register_inflight(
                operation.clone(),
                apply,
                submit_at,
                submitted_us.saturating_sub(planned_us),
            );
            wire_ops.push(operation);
            if let Some(st) = self.running.get_mut(&request_id) {
                st.latest_device_version = None;
                // Backfill the two pre-registration phases from the builder now
                // that the operation carries its canonical identity.
                st.trace.stamp(
                    op_key,
                    Some(operation_variant),
                    crate::scheduler::trace::LifecyclePhase::Planned,
                    planned_us,
                );
                if reserved_us != 0 {
                    st.trace.stamp(
                        op_key,
                        Some(operation_variant),
                        crate::scheduler::trace::LifecyclePhase::LogicalResourcesReserved,
                        reserved_us,
                    );
                }
                st.trace.stamp(
                    op_key,
                    Some(operation_variant),
                    crate::scheduler::trace::LifecyclePhase::WorkerRegistrationComplete,
                    submitted_us,
                );
                st.trace.stamp(
                    op_key,
                    Some(operation_variant),
                    crate::scheduler::trace::LifecyclePhase::Submitted,
                    submitted_us,
                );
            }
        }
        self.peak_ops_in_batch = self.peak_ops_in_batch.max(wire_ops.len());
        self.stats
            .general
            .peak_ops
            .fetch_max(wire_ops.len(), Ordering::Relaxed);
        self.stats.general.steps.fetch_add(1, Ordering::Relaxed);
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
            .store(self.free_kv_blocks(), Ordering::Relaxed);
        let mixed = wire_ops.first().is_some_and(|first| {
            wire_ops
                .iter()
                .any(|operation| operation.work != first.work)
        });
        self.batch_started.insert(step, submit_at);
        if wire_ops
            .iter()
            .any(|operation| assembly_lane(operation.work) == AssemblyLane::Prefill)
        {
            self.prefill_steps.insert(step);
        }
        if let Some(trace_ops) = trace_ops {
            let operation_types: Vec<&'static str> = wire_ops
                .iter()
                .map(|operation| operation.work.as_wire_str())
                .collect();
            let req_ids: Vec<u64> = wire_ops
                .iter()
                .map(|operation| operation.request_key.session_id.0)
                .collect();
            let admitted_session_ids: Vec<u64> = admissions
                .iter()
                .map(|admission| admission.request_key.session_id.0)
                .collect();
            self.trace_record(json!({
                "event": "batch_submitted",
                "at_s": now(),
                "step_id": step,
                "batch_size": wire_ops.len(),
                "mixed": mixed,
                "operation_types": operation_types,
                "request_ids": req_ids,
                "admitted_session_ids": admitted_session_ids,
                "ops": trace_ops,
                "scheduler": {
                    "policy": policy_str(self.config.policy),
                    "max_batch": self.config.max_batch,
                    "max_num_batched_tokens": self.config.max_num_batched_tokens,
                },
                "running": self.running.len(),
                "pending": self.pending.len(),
                "in_flight_before_submit": self.executor.in_flight(),
                "free_blocks": self.free_kv_blocks(),
                "reserved_blocks": self.reserved_blocks,
                "worker_image_latent_active": self.worker_image_latent_used(),
                "worker_image_latent_capacity": self.caps.latent_capacity_units(),
            }));
        }
        if mixed {
            let operation_types: Vec<&'static str> = wire_ops
                .iter()
                .map(|operation| operation.work.as_wire_str())
                .collect();
            let req_ids: Vec<u64> = wire_ops
                .iter()
                .map(|operation| operation.request_key.session_id.0)
                .collect();
            tracing::debug!(
                step_id = self.step_id,
                ?operation_types,
                ?req_ids,
                "submitting mixed forward batch"
            );
        }
        let releases = wire_ops
            .iter()
            .filter(|operation| operation.parent.producer_op_id.0 > 0)
            .map(|operation| Control::Release {
                request_key: operation.request_key,
                op_id: operation.parent.producer_op_id,
            })
            .collect::<Vec<_>>();
        controls.extend(releases);
        let partitions = self.partition_batch(
            wire_ops,
            &block_tables,
            &new_cache_pages,
            &forward_rows,
            &latent_placements,
        );
        let partition_accounting = partitions
            .iter()
            .map(|partition| {
                (
                    partition.partition_id,
                    SubmittedPartitionAccounting {
                        domain: partition.domain,
                        execution: partition.execution,
                        submission_group: partition.submission_group,
                        operation_count: partition.operations.len(),
                    },
                )
            })
            .collect::<HashMap<_, _>>();
        self.batch_partitions.insert(step, partition_accounting);
        self.batch_group_worker_exec_us.insert(step, HashMap::new());
        let batch = Batch::new(step, admissions, partitions)
            .with_controls(controls.clone())
            .with_input_products(input_products);
        if let Err(e) = self.executor.submit(batch) {
            self.batch_started.remove(&step);
            self.prefill_steps.remove(&step);
            self.batch_partitions.remove(&step);
            self.batch_group_worker_exec_us.remove(&step);
            self.trace_record(json!({
                "event": "batch_submit_failed",
                "at_s": now(),
                "step_id": step,
                "error": format!("{e}"),
            }));
            tracing::error!("executor submit failed: {e}");
            self.fatal = true;
            self.fail_all_inflight(&format!("{e}"));
            false
        } else {
            if !controls.is_empty() {
                self.control_batches.insert(step, controls);
            }
            true
        }
    }

    pub(super) fn partition_batch(
        &mut self,
        operations: Vec<Operation>,
        block_tables: &HashMap<(RequestKey, OpId), Vec<WireBlockTable>>,
        new_cache_pages: &HashMap<(RequestKey, OpId), Vec<CachePageAllocation>>,
        forward_rows: &HashMap<(RequestKey, OpId), Vec<RowGeometry>>,
        latent_placements: &HashMap<(RequestKey, OpId), LatentPlacement>,
    ) -> Vec<BatchPartition> {
        let mut routes: RouteDomainOperations = Vec::new();
        for operation in operations {
            let route = operation.route;
            let domain = operation.domain;
            let groups = if let Some((_, groups)) =
                routes.iter_mut().find(|(candidate, _)| *candidate == route)
            {
                groups
            } else {
                routes.push((route, Vec::new()));
                &mut routes.last_mut().expect("route was inserted").1
            };
            if let Some((_, members)) = groups
                .iter_mut()
                .find(|(candidate, _)| *candidate == domain)
            {
                members.push(operation);
            } else {
                groups.push((domain, vec![operation]));
            }
        }
        let mut partitions = Vec::new();
        let mut next_partition_id = 1u32;
        let mut next_submission_group = 1u32;
        for (route, groups) in routes {
            let mixed_capable = !self.caps.mixed_buckets.is_empty();
            let mut mixed_candidates = Vec::new();
            let mut homogeneous = Vec::new();
            for (domain, operations) in groups {
                if !mixed_capable {
                    homogeneous.push((domain, operations));
                    continue;
                }
                let (candidates, independent): (Vec<_>, Vec<_>) = operations
                    .into_iter()
                    .partition(|operation| tensorized_mixed_runner_work(operation.work));
                if !candidates.is_empty() {
                    mixed_candidates.push((domain, candidates));
                }
                if !independent.is_empty() {
                    homogeneous.push((domain, independent));
                }
            }
            while let Some(mixed_group) = extract_mixed_group(
                &mut mixed_candidates,
                &self.caps.mixed_buckets,
                forward_rows,
                latent_placements,
            ) {
                let collective_seq = {
                    let value = self.next_collective_seq.max(1);
                    self.next_collective_seq = value.saturating_add(1);
                    value
                };
                let attention = partition_attention(
                    &mixed_group
                        .iter()
                        .flat_map(|(_, operations)| operations.iter().cloned())
                        .collect::<Vec<_>>(),
                );
                for (domain, operations) in mixed_group {
                    let partition_block_tables = self.block_tables(&operations, block_tables);
                    let partition_new_cache_pages =
                        self.new_cache_pages(&operations, new_cache_pages);
                    let partition_forward_rows = self.forward_rows(&operations, forward_rows);
                    let latent_placements = self.latent_placements(&operations, latent_placements);
                    partitions.push(BatchPartition {
                        partition_id: next_partition_id,
                        submission_group: next_submission_group,
                        collective_seq,
                        domain,
                        route,
                        execution: ExecutionCapability::TensorizedMixed,
                        attention,
                        shape_class: 0,
                        operations,
                        block_tables: partition_block_tables,
                        new_cache_pages: partition_new_cache_pages,
                        forward_rows: partition_forward_rows,
                        latent_placements,
                        decode_placements: Vec::new(),
                    });
                    next_partition_id = next_partition_id.saturating_add(1);
                }
                next_submission_group = next_submission_group.saturating_add(1);
            }
            homogeneous.extend(
                mixed_candidates
                    .into_iter()
                    .filter(|(_, operations)| !operations.is_empty()),
            );
            for (domain, operations) in homogeneous {
                let collective_seq = {
                    let value = self.next_collective_seq.max(1);
                    self.next_collective_seq = value.saturating_add(1);
                    value
                };
                let partition_block_tables = self.block_tables(&operations, block_tables);
                let partition_new_cache_pages = self.new_cache_pages(&operations, new_cache_pages);
                let partition_forward_rows = self.forward_rows(&operations, forward_rows);
                let latent_placements = self.latent_placements(&operations, latent_placements);
                partitions.push(BatchPartition {
                    partition_id: next_partition_id,
                    submission_group: next_submission_group,
                    collective_seq,
                    domain,
                    route,
                    execution: ExecutionCapability::DomainHomogeneous,
                    attention: partition_attention(&operations),
                    shape_class: 0,
                    operations,
                    block_tables: partition_block_tables,
                    new_cache_pages: partition_new_cache_pages,
                    forward_rows: partition_forward_rows,
                    latent_placements,
                    decode_placements: Vec::new(),
                });
                next_partition_id = next_partition_id.saturating_add(1);
                next_submission_group = next_submission_group.saturating_add(1);
            }
        }
        partitions
    }

    pub(super) fn block_tables(
        &self,
        operations: &[Operation],
        tables: &HashMap<(RequestKey, OpId), Vec<WireBlockTable>>,
    ) -> Vec<WireBlockTable> {
        operations
            .iter()
            .flat_map(|operation| {
                tables
                    .get(&(operation.request_key, operation.op_id))
                    .into_iter()
                    .flatten()
                    .cloned()
            })
            .collect()
    }

    pub(super) fn new_cache_pages(
        &self,
        operations: &[Operation],
        allocations: &HashMap<(RequestKey, OpId), Vec<CachePageAllocation>>,
    ) -> Vec<CachePageAllocation> {
        operations
            .iter()
            .flat_map(|operation| {
                allocations
                    .get(&(operation.request_key, operation.op_id))
                    .into_iter()
                    .flatten()
                    .cloned()
            })
            .collect()
    }

    pub(super) fn forward_rows(
        &self,
        operations: &[Operation],
        rows: &HashMap<(RequestKey, OpId), Vec<RowGeometry>>,
    ) -> Vec<RowGeometry> {
        operations
            .iter()
            .enumerate()
            .flat_map(|(operation_index, operation)| {
                rows.get(&(operation.request_key, operation.op_id))
                    .into_iter()
                    .flatten()
                    .copied()
                    .map(move |mut row| {
                        row.operation_index = operation_index as u32;
                        row
                    })
            })
            .collect()
    }

    pub(super) fn latent_placements(
        &self,
        operations: &[Operation],
        placements: &HashMap<(RequestKey, OpId), LatentPlacement>,
    ) -> Vec<LatentPlacement> {
        operations
            .iter()
            .filter_map(|operation| {
                placements
                    .get(&(operation.request_key, operation.op_id))
                    .cloned()
            })
            .collect()
    }

    pub(super) fn generated_trigger_matches(st: &ReqState) -> bool {
        st.req
            .policy
            .trigger
            .matches_generated(&st.cursor.replay.generated_ids)
    }

    pub(super) fn direct_trigger_matches(st: &ReqState, token_id: u32) -> bool {
        st.req.policy.trigger.direct_token() == Some(token_id)
    }

    pub(super) fn feedback_next_token(&self, id: RequestId) -> Option<u32> {
        let next = self
            .running
            .get(&id)?
            .req
            .policy
            .feedback
            .as_ref()?
            .next_und_token;
        match next {
            uniserve_core::FeedbackNextToken::None => None,
            uniserve_core::FeedbackNextToken::Bos => Some(self.ctrl.bos),
            uniserve_core::FeedbackNextToken::EndOfImage => Some(self.ctrl.end_of_image),
            uniserve_core::FeedbackNextToken::Token { token_id } => Some(token_id),
        }
    }

    pub(super) fn prefilled_gen_trigger(&self, id: RequestId) -> bool {
        let Some(st) = self.running.get(&id) else {
            return false;
        };
        st.can_open_gen_branch()
            && st
                .req
                .policy
                .trigger
                .matches_generated(st.effective_prompt())
    }

    pub(super) fn ensure_request_capacity(&mut self, id: RequestId, total_tokens: usize) -> bool {
        let kv = self
            .kv
            .as_ref()
            .expect("generation scheduling requires worker KV resources");
        let Some(state) = self.running.get_mut(&id) else {
            return false;
        };
        kv.coordinator
            .ensure_capacity(&kv.block_pool, &mut state.block_tables, total_tokens)
            .is_some()
    }

    pub(super) fn activate_request_tables(&self, id: RequestId) {
        let kv = self.kv_state();
        if let Some(state) = self.running.get(&id) {
            for table in &state.block_tables {
                table.activate(&kv.block_pool);
            }
        }
    }

    /// The block-id delta since the last op for this request (the stateful-diff
    /// contract): everything `blocks_for` holds beyond what already crossed.
    pub(super) fn take_new_blocks(&mut self, id: RequestId) -> Vec<BlockId> {
        let Some(st) = self.running.get_mut(&id) else {
            return Vec::new();
        };
        let all = st.block_tables[0].page_ids();
        let sent = st.cursor.resources.blocks_sent.min(all.len());
        let new = all[sent..].to_vec();
        st.cursor.resources.blocks_sent = all.len();
        new
    }

    pub(super) fn return_unsent_blocks(&mut self, id: RequestId, transition: &PlannedTransition) {
        if let Some(state) = self.running.get_mut(&id) {
            state.cursor.resources.blocks_sent = state
                .cursor
                .resources
                .blocks_sent
                .saturating_sub(transition.new_blocks.len());
        }
    }

    /// Data-plane causality gate for the request's next operation. A staged
    /// executor retains a cross-pool consumer until every exact input product
    /// has a published transfer descriptor. Direct executors are immediately
    /// ready because producer and consumer share worker-resident ownership.
    pub(super) fn stage_ready(&self, id: RequestId) -> bool {
        self.executor.stage_ready(id)
    }

    pub(super) fn plan_intent(
        &mut self,
        id: RequestId,
        cursor: CursorProjection,
        intent: TransitionIntent,
    ) -> Option<PlannedTransition> {
        let planned = {
            let request = &self.running.get(&id)?.req;
            self.planner.plan(request, cursor, intent)
        };
        match planned {
            Ok(transition) => Some(transition),
            Err(error) => {
                tracing::error!(request_id = id.0, ?error, "generation planning failed");
                self.finish(id, FinishReason::Error);
                None
            }
        }
    }

    pub(super) fn next_transition(
        &mut self,
        id: RequestId,
        budget: usize,
    ) -> Option<PlannedTransition> {
        if !self.stage_ready(id) {
            return None;
        }
        let projection = self.projected_cursor(id)?;
        let context_pending = self.running.get(&id).is_some_and(|st| {
            projection.prompt_cursor < st.context.prompt_ids.len() as u32
                || st.cursor.ingest.mm_cursor < st.context.images.len()
        });
        if context_pending {
            return self.next_context_ingest_transition(id, budget, projection);
        }
        if self.running.get(&id)?.cursor.image_gen.branch_pending
            && !self.promote_gen_branch_reservation(id)
        {
            return None;
        }
        let projected_branch = self.projected_branch(id)?;
        let phase = projected_branch.phase;
        match phase {
            Phase::Encode => None,
            Phase::Prefill => {
                let st = self.running.get(&id)?;
                let prompt = st.effective_prompt().to_vec();
                let n = prompt.len();
                let projection = self.projected_cursor(id)?;
                let cursor = projection.prompt_cursor as usize;
                let (segment_index, segment_end) =
                    st.context.token_segment_at(cursor).unwrap_or((0, n));
                // Chunked prefill with the clip rule: the chunk is bounded by
                // the remaining step budget and the long-prefill threshold.
                let chunk_cap = (n - cursor)
                    .min(self.config.long_prefill_threshold)
                    .min(budget.max(1));
                let end = (cursor + chunk_cap.max(1))
                    .min(n)
                    .min(segment_end.max(cursor + 1));
                if !self.ensure_request_capacity(id, end) {
                    return None;
                }
                let chunk: Vec<u32> = prompt[cursor..end].to_vec();
                let sampling_state = self.sampling_state(id, 0);
                let new_blocks = self.take_new_blocks(id);
                self.plan_intent(
                    id,
                    projection,
                    TransitionIntent::IngestText {
                        segment_index,
                        prompt_start: cursor as u32,
                        token_ids: chunk,
                        new_blocks,
                        sampling_state,
                    },
                )
            }
            Phase::DecodeUnd => {
                let projected_successor = self.has_inflight(id);
                // A successor registered before its predecessors resolve is
                // `inflight_len` unresolved points ahead of the committed cursor;
                // its minimum-token floor and force-finish flag are staged at
                // that exact projected point.
                let projected = if projected_successor {
                    self.inflight_len(id)
                } else {
                    0
                };
                let sampling_state = self.sampling_state(id, projected);
                let st = self.running.get(&id)?;
                // The prior committed token this decode continues from. It is
                // attached as a host input only when no exact selected-point
                // product is eligible for device continuation.
                let input_token = st.cursor.und.next_token;
                let projection = self.projected_cursor(id)?;
                let pos = projection.logical_pos;
                let relay_input = projected_successor || self.can_reuse_resolved_token_product(id);
                let capacity_target = self.decode_capacity_target(pos as usize, 0);
                if !self.ensure_request_capacity(id, capacity_target) {
                    return None;
                }
                let new_blocks = self.take_new_blocks(id);
                self.plan_intent(
                    id,
                    projection,
                    TransitionIntent::DecodeUnd {
                        position: pos,
                        new_blocks,
                        spec_token_ids: None,
                        sampling_state,
                        input_token,
                        relay_input,
                    },
                )
            }
            Phase::CloseKv => {
                let token = self.running.get(&id)?.cursor.und.next_token;
                let capacity_target =
                    self.decode_capacity_target(projection.physical_kv_len as usize, 0);
                if !self.ensure_request_capacity(id, capacity_target) {
                    return None;
                }
                let image_id = projected_branch.image_id;
                let position = projection.logical_pos;
                let physical_position = projection.physical_kv_len;
                let new_blocks = self.take_new_blocks(id);
                self.plan_intent(
                    id,
                    projection,
                    TransitionIntent::CloseKv {
                        image_id,
                        position,
                        physical_position,
                        token,
                        relay_input: false,
                        new_blocks,
                    },
                )
            }
            Phase::PublishKv => self.plan_intent(
                id,
                projection,
                TransitionIntent::PublishKv {
                    image_id: projected_branch.image_id,
                },
            ),
            Phase::TransitionGen => {
                let st = self.running.get(&id)?;
                let conditioning = projected_branch.conditioning.clone()?;
                let image_id = projected_branch.image_id;
                let latent_units = self.worker_image_latent_units_for(st).max(1);
                self.plan_intent(
                    id,
                    projection,
                    TransitionIntent::TransitionGen {
                        image_id,
                        latent_units,
                        conditioning,
                    },
                )
            }
            Phase::DenoiseGen => {
                // The flow runs exactly `image.steps` denoise quanta; once they
                // are all done, advance to the commit phase and plan its
                // transition rather than a spurious zero-length step. Termination
                // is host-driven off the committed step count, not a worker
                // completion flag.
                let st = self.running.get(&id)?;
                let timestep = projected_branch.steps_done;
                let remaining = st.req.image.steps.saturating_sub(timestep);
                if remaining == 0 {
                    return None;
                }
                let denoise_step_count = self.denoise_step_burst.max(1).min(remaining);
                let cfg = cfg_params(&st.req.image, cfg_branch_count(&st.req.image));
                let latent_units = self.worker_image_latent_units_for(st).max(1);
                let image_id = projected_branch.image_id;
                let conditioning = projected_branch.conditioning.clone()?;
                let latent = projected_branch.latent.clone()?;
                let projection = self.projected_cursor(id)?;
                self.plan_intent(
                    id,
                    projection,
                    TransitionIntent::DenoiseGen {
                        image_id,
                        start_step: timestep,
                        step_count: denoise_step_count,
                        cfg,
                        latent_units,
                        conditioning,
                        latent,
                    },
                )
            }
            Phase::CommitGen => {
                let image_id = projected_branch.image_id;
                let latent = projected_branch.latent.clone()?;
                let projection = self.projected_cursor(id)?;
                self.plan_intent(
                    id,
                    projection,
                    TransitionIntent::CommitGen { image_id, latent },
                )
            }
            Phase::FeedbackEncode => {
                let st = self.running.get(&id)?;
                let feedback = st.req.policy.feedback.as_ref()?;
                let step_index = projected_branch.feedback_step;
                let step = feedback.ingest.steps.get(step_index).copied()?;
                let image_id = projected_branch.image_id;
                let source = (feedback.source == uniserve_core::FeedbackSource::DeviceProduct)
                    .then(|| projected_branch.feedback_source.clone())
                    .flatten();
                let image_b64 = if source.is_none() {
                    st.cursor.feedback.image_b64.clone().unwrap_or_default()
                } else {
                    String::new()
                };
                let projection = self.projected_cursor(id)?;
                self.plan_intent(
                    id,
                    projection,
                    TransitionIntent::EncodeFeedback {
                        image_id,
                        step_index,
                        step,
                        source,
                        image_b64,
                    },
                )
            }
            Phase::FeedbackState => {
                let st = self.running.get(&id)?;
                let feedback = st.req.policy.feedback.as_ref()?;
                let step_index = projected_branch.feedback_step;
                let is_final_step = step_index + 1 == feedback.ingest.steps.len();
                let image_id = projected_branch.image_id;
                let logical_positions = feedback.ingest.logical_positions;
                let physical_kv_tokens = feedback.ingest.kv_effect(step_index)?;
                let feature = projected_branch.feedback_feature.clone();
                let sample_continuation = is_final_step && feedback.sample_continuation;
                let Some(feature) = feature else {
                    self.finish(id, FinishReason::Error);
                    return None;
                };
                let projection = self.projected_cursor(id)?;
                let new_blocks = self.take_new_blocks(id);
                let sampling_state = self.sampling_state(id, 0);
                self.plan_intent(
                    id,
                    projection,
                    TransitionIntent::FeedbackState {
                        image_id,
                        step_index,
                        is_final_step,
                        position: projection.logical_pos,
                        logical_positions,
                        physical_kv_tokens,
                        feature,
                        sample_continuation,
                        new_blocks,
                        sampling_state,
                    },
                )
            }
            Phase::IngestState => self.next_context_ingest_transition(id, budget, projection),
        }
    }

    pub(super) fn next_context_ingest_transition(
        &mut self,
        id: RequestId,
        budget: usize,
        projection: CursorProjection,
    ) -> Option<PlannedTransition> {
        let cursor = projection.prompt_cursor as usize;
        let image = self
            .running
            .get(&id)?
            .context
            .images
            .get(self.running.get(&id)?.cursor.ingest.mm_cursor)
            .cloned();
        if let Some(image) = image.as_ref()
            && image.position as usize == cursor
        {
            let state = self.running.get(&id)?;
            let step_index = state.cursor.ingest.pending_image_step;
            let step = image.ingest.steps.get(step_index).copied()?;
            let is_final_step = step_index + 1 == image.ingest.steps.len();
            if state.cursor.lifecycle.phase == Phase::IngestState {
                let feature = state.cursor.ingest.encoded_product.clone()?;
                let physical_kv_tokens = image.ingest.kv_effect(step_index)?;
                let physical_bound = match physical_kv_tokens {
                    uniserve_core::ImageKvEffect::Exact { tokens } => tokens,
                    uniserve_core::ImageKvEffect::Bounded { max_tokens } => max_tokens,
                    uniserve_core::ImageKvEffect::WorkerDefined => match step {
                        uniserve_core::ImageIngestStep::VaeEncode => {
                            self.cap_max_vae_grid_tokens().min(u32::MAX as usize) as u32
                        }
                        uniserve_core::ImageIngestStep::VitEncode => self.caps.max_vit_grid_tokens,
                    },
                };
                if !self.ensure_request_capacity(
                    id,
                    projection.physical_kv_len.saturating_add(physical_bound) as usize,
                ) {
                    return None;
                }
                let new_blocks = self.take_new_blocks(id);
                return self.plan_intent(
                    id,
                    projection,
                    TransitionIntent::IngestImageState {
                        segment_index: image.segment_index,
                        step_index,
                        is_final_step,
                        position: projection.logical_pos,
                        logical_positions: image.ingest.logical_positions,
                        physical_kv_tokens,
                        feature,
                        new_blocks,
                    },
                );
            }
            let cache_read = state.req.cache.read;
            let cache_write = state.req.cache.write;
            let cache_key = encoder_cache_key(image.hash, step_index, step);
            let cached = if cache_read {
                self.enc_cache.lookup_product(cache_key)
            } else {
                None
            };
            if let Some(cached_product) = cached {
                let product = self.enc_cache.acquire(cache_key)?;
                if product != cached_product {
                    return None;
                }
                let Some(state) = self.running.get_mut(&id) else {
                    let _ = self.enc_cache.release(cache_key, &product);
                    return None;
                };
                state
                    .cursor
                    .ingest
                    .acquired_encoder_pins
                    .push(EncoderCachePin {
                        key: cache_key,
                        product: product.clone(),
                    });
                state.cursor.ingest.encoded_product = Some(product);
                state.cursor.lifecycle.phase = Phase::IngestState;
                return self.next_context_ingest_transition(id, budget, projection);
            }
            let persistent_cache_key = cache_write.then_some(cache_key);
            return self.plan_intent(
                id,
                projection,
                TransitionIntent::EncodeImage {
                    segment_index: image.segment_index,
                    step_index,
                    step,
                    encoder_cache_key: persistent_cache_key,
                    image_b64: image.b64.clone(),
                    source_product: None,
                },
            );
        }

        let (prompt, segment_index, segment_end, next_image) = {
            let st = self.running.get(&id)?;
            let prompt = st.context.prompt_ids.clone();
            let (segment_index, segment_end) = st
                .context
                .token_segment_at(cursor)
                .unwrap_or((0, prompt.len()));
            let next_image = image
                .as_ref()
                .map_or(prompt.len(), |image| image.position as usize);
            (prompt, segment_index, segment_end, next_image)
        };
        if cursor >= prompt.len() {
            return None;
        }
        let end = cursor
            .saturating_add(budget.max(1).min(self.config.long_prefill_threshold))
            .min(prompt.len())
            .min(segment_end)
            .min(next_image.max(cursor + 1));
        if !self.ensure_request_capacity(id, projection.physical_kv_len as usize + end - cursor) {
            return None;
        }
        let sampling_state = self.sampling_state(id, 0);
        let new_blocks = self.take_new_blocks(id);
        self.plan_intent(
            id,
            projection,
            TransitionIntent::IngestText {
                segment_index,
                prompt_start: cursor as u32,
                token_ids: prompt[cursor..end].to_vec(),
                new_blocks,
                sampling_state,
            },
        )
    }

    /// Decide branch-vs-finish once a declared round-close token is committed.
    pub(super) fn close_context_round(&mut self, id: RequestId, close_token: u32) {
        let (triggered, can_open_gen_branch, n_gen, max_tokens, eos_finishes) = {
            let Some(st) = self.running.get(&id) else {
                return;
            };
            (
                st.req
                    .policy
                    .trigger
                    .matches_round_close(&st.cursor.und.round_tokens, close_token),
                st.can_open_gen_branch(),
                st.cursor.und.tokens_emitted,
                st.req.max_und_tokens,
                st.req.policy.termination.eos_finishes,
            )
        };
        if triggered && can_open_gen_branch && n_gen < max_tokens {
            return self.begin_image(id);
        }
        if n_gen < max_tokens && !eos_finishes {
            if let Some(st) = self.running.get_mut(&id) {
                st.cursor.und.round_tokens.clear();
                st.cursor.und.next_token = close_token;
                st.cursor.lifecycle.phase = Phase::DecodeUnd;
            }
            return;
        }
        self.finish(
            id,
            if n_gen >= max_tokens {
                FinishReason::MaxTokens
            } else {
                FinishReason::Eos
            },
        )
    }

    /// Compute the operation's allowed/suppress masks from the host-side
    /// logits-processor pipeline (minimum-token floor, bad-words, allowed
    /// tokens). `n_generated` is the count of generated tokens the sampled point
    /// follows: for a successor registered before its predecessors are observed
    /// it is the committed count plus the unresolved window depth, so the
    /// minimum-token floor is evaluated at the successor's own exact point
    /// without reading any device token. `projected_images_done` includes final
    /// feedback states already registered ahead of the operation, so the image
    /// cap masks its branch token at the same point on every pipeline depth.
    pub(super) fn token_masks(
        &mut self,
        id: RequestId,
        n_generated: usize,
        projected_images_done: usize,
    ) -> (Option<Vec<u32>>, Option<Vec<u32>>) {
        let st = match self.running.get(&id) {
            Some(s) => s,
            None => return (None, None),
        };
        if self.cpu_continuation_required(st) {
            return st
                .cpu_masks
                .as_ref()
                .map(|masks| (masks.allowed.clone(), masks.suppress.clone()))
                .unwrap_or((Some(Vec::new()), None));
        }
        let ctx = crate::scheduler::logits::ProcCtx {
            n_generated,
            eos: &self.ctrl.eos,
            generated: &st.cursor.replay.generated_ids,
            sampling: &st.req.sampling,
        };
        let masks = crate::scheduler::logits::run_pipeline(&self.logits_pipeline, &ctx);
        let allowed = masks.allowed;
        let mut suppress = masks.suppress;
        // Image-budget enforcement: once a request has produced max_images, a
        // direct branch-opening token is
        // suppressed, so an eager or positively-biased model must return to
        // text/EOS instead of emitting an un-actionable trigger into its
        // own context forever. Suppression beats logit bias (bias skips
        // -inf'd logits on the worker).
        if st.req.behavior.gen_output
            && projected_images_done >= st.req.image.max_images as usize
            && let Some(trigger) = st.req.policy.trigger.direct_token()
        {
            suppress.get_or_insert_with(Vec::new).push(trigger);
        }
        (allowed, suppress)
    }

    /// The branch-local static sampling state staged for one operation.
    ///
    /// `projected` is the number of unresolved predecessors the operation is
    /// registered behind: zero for a host-paced operation, the in-flight window
    /// depth for a successor registered before its predecessors are observed.
    /// The minimum-token floor and the force-finish flag are evaluated at the
    /// operation's own exact point (`tokens_emitted + projected`). Penalty
    /// counts are never staged here — they are a device-resident committed base
    /// plus per-operation deltas the worker folds on commit, so no host token
    /// history participates in a successor's penalty input.
    pub(super) fn sampling_state(&mut self, id: RequestId, projected: usize) -> SamplingState {
        let n_generated = self.running.get(&id).map_or(0, |state| {
            state.cursor.und.tokens_emitted.saturating_add(projected)
        });
        let projected_images_done = self.running.get(&id).map_or(0, |state| {
            state.cursor.image_gen.images_done.saturating_add(
                self.inflight_ops
                    .get(&id)
                    .into_iter()
                    .flatten()
                    .filter(|inflight| {
                        matches!(
                            inflight.generation_apply().delta,
                            TransitionDelta::FeedbackState {
                                is_final_step: true,
                                ..
                            }
                        )
                    })
                    .count(),
            )
        });
        let (allowed_token_ids, suppressed_token_ids) =
            self.token_masks(id, n_generated, projected_images_done);
        let Some(state) = self.running.get(&id) else {
            return SamplingState::default();
        };
        let finish_token_ids = canonical_continuation_stop_token_ids(&state.req, &self.ctrl.eos);
        let transition_token_ids = (state.req.behavior.gen_output
            && projected_images_done < state.req.image.max_images as usize)
            .then(|| state.req.policy.trigger.direct_token())
            .flatten()
            .into_iter()
            .collect();
        SamplingState {
            allowed_token_ids,
            suppressed_token_ids: suppressed_token_ids.unwrap_or_default(),
            finish_token_ids,
            transition_token_ids,
            force_finish: n_generated.saturating_add(1) >= state.req.max_und_tokens,
        }
    }

    pub(super) fn invalidate_cpu_masks(&mut self, id: RequestId) {
        let required = self
            .running
            .get(&id)
            .is_some_and(|state| self.cpu_continuation_required(state));
        if required && let Some(state) = self.running.get_mut(&id) {
            state.cpu_masks = None;
        }
    }
}
