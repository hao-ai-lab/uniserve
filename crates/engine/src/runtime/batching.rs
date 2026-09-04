//! Batch selection, operation planning, and physical placement assembly.
//!
//! Each pass selects a compatible execution lane, applies sequence and token
//! budgets, and emits at most one planned operation per eligible request.

use super::*;

/// Computes the target decode capacity for a scheduling round.
fn decode_capacity_target(pos: usize, spec_len: usize) -> usize {
    pos.saturating_add(1).saturating_add(spec_len)
}

impl EngineLoop {
    /// Assembles one scheduling-step batch in priority order.
    ///
    /// Each request contributes at most one operation, prefill chunks consume only
    /// the remaining token budget, and first dispatch carries typed admission.
    pub(super) fn assemble(&mut self) -> (Vec<NewRequest>, Vec<NextOp>) {
        let ids = self.assembly_order();
        let lane = self.select_batch_kind(&ids);
        let (new_reqs, ops) = self.assemble_pass(&ids, lane);
        if ops.is_empty() && lane == Some(BatchKind::Prefill) {
            // A blocked prefill lane must not prevent already-ready decode work
            // from using the execution slot.
            return self.assemble_pass(&ids, Some(BatchKind::Decode));
        }
        (new_reqs, ops)
    }

    /// Reserves persistent buffers, latent pages, and transfer capacity for one transition.
    fn reserve_transition_resources(&mut self, transition: &mut NextOp) -> bool {
        let id = transition.request_id;
        if transition.operation_variant == RunKind::DiffusionStep && !self.ensure_flow_prefix(id) {
            return false;
        }
        let resources = &transition.resources;
        let uses_transfer = transition.bounds.max_transfer_bytes > 0;
        if uses_transfer && self.inflight.inflight_transfers >= self.inflight.transfer_capacity {
            return false;
        }
        let request_key = self
            .running
            .get(&id)
            .map(|state| RequestKey::new(self.authority_id, id, state.epoch));
        let Some(request_key) = request_key else {
            return false;
        };
        let mut buffer_allocations = Vec::new();
        for bytes in transition
            .outputs
            .iter()
            .filter(|output| output.uses_persistent_buffer())
            .map(ProductRef::max_bytes)
        {
            let allocation = match self.memory.alloc(
                request_key,
                MemoryLayout::Buffer {
                    bytes,
                    alignment: 256,
                },
            ) {
                Ok(allocation) => allocation,
                Err(_) => {
                    for allocation in buffer_allocations {
                        self.memory.free(allocation);
                    }
                    if let Some(product) = self.memory.encoder_cache.evict_one() {
                        self.free_products(vec![product]);
                    }
                    return false;
                }
            };
            buffer_allocations.push(allocation);
        }
        if resources.latent_units > 0 && self.worker_tracks_image_latent() {
            let (runtime, memory) = (&mut self.runtime, &mut self.memory);
            let authority_id = runtime.state().authority_id;
            let Some(state) = runtime.state_mut().running.get_mut(&id) else {
                return false;
            };
            let request_key = RequestKey::new(authority_id, id, state.epoch);
            let allocations = state.allocations_mut();
            if let Some(allocation) = allocations.latent.as_mut() {
                if memory
                    .grow(
                        allocation,
                        MemoryLayout::Latent {
                            units: resources.latent_units,
                        },
                    )
                    .is_err()
                {
                    for allocation in buffer_allocations {
                        memory.free(allocation);
                    }
                    return false;
                }
            } else {
                let Ok(allocation) = memory.alloc(
                    request_key,
                    MemoryLayout::Latent {
                        units: resources.latent_units,
                    },
                ) else {
                    for allocation in buffer_allocations {
                        memory.free(allocation);
                    }
                    return false;
                };
                allocations.latent = Some(allocation);
            }
        }
        transition.buffer_allocations = buffer_allocations;
        if uses_transfer {
            self.inflight.inflight_transfers += 1;
        }
        transition.reserved_us = uniserve_core::now_monotonic_us();
        true
    }

    /// Selects compatible request transitions within one lane's token and sequence budgets.
    pub(super) fn assemble_pass(
        &mut self,
        ids: &[RequestId],
        lane: Option<BatchKind>,
    ) -> (Vec<NewRequest>, Vec<NextOp>) {
        let mut admissions: Vec<NewRequest> = Vec::new();
        let mut ops: Vec<NextOp> = Vec::new();
        let mut selected: HashSet<RequestId> = HashSet::new();
        // The per-step token budget is the binding limit; individual prefill
        // chunks are clipped to its remaining capacity.
        let mut budget: usize = self.scheduler.config.max_num_batched_tokens;
        // Text prefill tokens may ride along inside a decode batch (mixed
        // extend+decode forward): the prompt work then shares the decode
        // step's weight sweep instead of paying a full sweep of its own.
        // Mixed prefill rows are appended after the decode rows so the batch
        // keeps an extend row last (graph token-bucket padding extends the
        // last row).
        let mut mixed_left: usize = if lane == Some(BatchKind::Decode) {
            self.scheduler.config.mixed_prefill_tokens
        } else {
            0
        };
        let mut mixed_ops: Vec<NextOp> = Vec::new();
        let denoise_occupies_decode_pipeline =
            lane == Some(BatchKind::Decode) && self.inflight.any_denoise();
        for id in ids.iter().copied() {
            if ops.len() + mixed_ops.len() >= self.scheduler.config.max_batch {
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
                    let candidate_lane = batch_kind(operation_variant);
                    matches!(candidate_lane, BatchKind::Prefill | BatchKind::Decode)
                        && candidate_lane != target
                }
            {
                mixed_prefill = target == BatchKind::Decode
                    && operation_variant == RunKind::ArExtend
                    && mixed_left > 0
                    && self.running.get(&id).is_some_and(|st| {
                        st.is_replayable_text() && !st.req.sampling.prompt_logprobs_requested()
                    });
                if !mixed_prefill {
                    continue;
                }
            }
            if next_type == Some(RunKind::DiffusionStep)
                && (denoise_occupies_decode_pipeline || !self.flow_prefix_is_schedulable(id))
            {
                continue;
            }
            // Diagnostic: a denoise step and text rows in one forward take the
            // packed route, which costs more than running each on its own. When
            // the flag is set a flow op only opens an empty batch, and the loop
            // below closes the batch as soon as one is placed.
            if self.flow_exclusive_batch
                && next_type == Some(RunKind::DiffusionStep)
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
                    self.record_domain_backpressure(op.kind.domain());
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
                    .map(|state| state.finish_token_ids.clone())
                    .unwrap_or_default();
                let authority_id = self.runtime.state().authority_id;
                if let Some(st) = self.runtime.state_mut().running.get_mut(&id)
                    && !st.cursor.resources.worker_registered
                {
                    st.cursor.resources.worker_registered = true;
                    let request_key = RequestKey::new(authority_id, id, st.epoch);
                    let admission = NewRequest::new(
                        request_key,
                        st.request_pool_idx(),
                        Some(ArRequestParams {
                            sampling: st.req.sampling.clone(),
                            negative_token_ids: st.context.negative_prompt_ids.clone(),
                            finish_token_ids,
                            initial_position: st.cursor.ingest.prompt_cursor,
                        }),
                        st.req.behavior.gen_output.then(|| UmmRequestParams {
                            image: st.req.image.clone(),
                        }),
                    )
                    .expect("validated request produces a valid admission");
                    // The admission-root version is produced by the synthetic
                    // admission, identified by operation-id 0 — the sentinel the
                    // worker seeds as its initial committed version — so the first
                    // operation's fixed parent matches the worker's committed state.
                    st.resolved_producer_op_id = 0;
                    st.committed_producer_op_id = 0;
                    st.latest_device_version = None;
                    st.token_cutoffs.clear();
                    st.token_cutoffs.insert(
                        st.output.tokens_sent,
                        Checkpoint {
                            op_id: OpId(0),
                            point: CheckpointPoint::Fixed(0),
                        },
                    );
                    admissions.push(admission);
                }
                selected.insert(id);
                let placed_denoise = op.operation_variant == RunKind::DiffusionStep;
                if mixed_prefill {
                    mixed_ops.push(op);
                } else {
                    ops.push(op);
                }
                if self.flow_exclusive_batch && placed_denoise {
                    budget = 0;
                }
            }
        }
        ops.extend(mixed_ops);
        (admissions, ops)
    }

    /// Chooses the highest-priority execution lane that has schedulable work.
    pub(super) fn select_batch_kind(&self, ids: &[RequestId]) -> Option<BatchKind> {
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
            match batch_kind(operation_type) {
                BatchKind::Decode => {
                    if self.can_schedule_next(id) {
                        if self.inflight.contains(id) {
                            projected_decode_ready = true;
                        } else {
                            committed_decode_ready = true;
                        }
                    }
                }
                BatchKind::Prefill => {
                    if self.can_schedule_next(id) {
                        prefill_ready = true;
                    }
                }
                BatchKind::Media => {}
            }
        }
        if prefill_ready && self.inflight.prefill_steps.len() < PREFILL_WINDOW_CREDITS {
            Some(BatchKind::Prefill)
        } else if committed_decode_ready || projected_decode_ready {
            Some(BatchKind::Decode)
        } else if prefill_ready {
            Some(BatchKind::Prefill)
        } else {
            None
        }
    }

    /// Returns the stable order key for batch assembly.
    pub(super) fn assembly_order(&self) -> Vec<RequestId> {
        let mut ids: Vec<(usize, RequestId)> = self
            .scheduler
            .running_order
            .iter()
            .enumerate()
            .filter_map(|(index, id)| self.running.get(id).is_some().then_some((index, *id)))
            .collect();
        ids.sort_by_key(|(idx, id)| (self.assembly_priority(*id), *idx));
        ids.into_iter().map(|(_, id)| id).collect()
    }

    /// Returns the scheduling priority used during batch assembly.
    pub(super) fn assembly_priority(&self, id: RequestId) -> u8 {
        match self.peek_next_operation_variant(id) {
            Some(RunKind::EncoderVision | RunKind::EncoderLatent | RunKind::ArExtend) => 0,
            Some(
                RunKind::ArDecode
                | RunKind::ArVerify
                | RunKind::DiffusionFinalize
                | RunKind::TransferKvInstall,
            ) => 1,
            Some(RunKind::DiffusionStep | RunKind::DiffusionDecode) => 2,
            Some(
                RunKind::DiffusionPrepare | RunKind::TransferProduct | RunKind::TransferKvPublish,
            )
            | None => 3,
        }
    }

    /// Determines the next operation kind without mutating request or resource state.
    pub(super) fn peek_next_operation_variant(&self, id: RequestId) -> Option<RunKind> {
        let st = self.running.get(&id)?;
        if st.cursor.image_gen.branch_pending {
            return None;
        }
        if let Some(variant) = self.projected_inflight_variant(id) {
            return Some(variant);
        }
        if st.has_context_images()
            && st.cursor.phase == Phase::Prefill
            && self.projected_cursor(id).is_some_and(|cursor| {
                st.context
                    .images
                    .get(st.cursor.ingest.mm_cursor)
                    .is_some_and(|item| {
                        item.position as usize == cursor.ingest.prompt_cursor as usize
                    })
            })
        {
            return st.pending_image_step().map(|step| match step {
                ImageIngestStep::VaeEncode => RunKind::EncoderLatent,
                ImageIngestStep::VitEncode => RunKind::EncoderVision,
            });
        }
        Some(match st.cursor.phase {
            Phase::Encode => match st.pending_image_step()? {
                ImageIngestStep::VaeEncode => RunKind::EncoderLatent,
                ImageIngestStep::VitEncode => RunKind::EncoderVision,
            },
            Phase::IngestState => RunKind::ArExtend,
            Phase::Prefill => RunKind::ArExtend,
            Phase::DecodeUnd => RunKind::ArDecode,
            Phase::CloseKv => RunKind::ArExtend,
            Phase::PublishKv => RunKind::TransferKvPublish,
            Phase::PrepareGen => RunKind::DiffusionPrepare,
            Phase::DenoiseGen if st.cursor.image_gen.steps_done >= st.req.image.steps => {
                RunKind::DiffusionFinalize
            }
            Phase::DenoiseGen => RunKind::DiffusionStep,
            Phase::CommitGen => RunKind::DiffusionFinalize,
            Phase::FeedbackEncode => {
                let feedback = st.req.policy.feedback.as_ref()?;
                match feedback.ingest.steps.get(st.cursor.feedback.ingest_step)? {
                    ImageIngestStep::VaeEncode => RunKind::EncoderLatent,
                    ImageIngestStep::VitEncode => RunKind::EncoderVision,
                }
            }
            Phase::FeedbackState => RunKind::ArExtend,
        })
    }

    /// Lowers planned transitions into one physical batch and transfers ownership to the executor.
    pub(super) fn submit_batch(
        &mut self,
        admissions: Vec<NewRequest>,
        transitions: Vec<NextOp>,
        commands: Vec<BatchCommand>,
    ) -> bool {
        let _span =
            tracing::trace_span!("scheduler.submit_batch", ops = transitions.len()).entered();
        let batch_id = self.inflight.next_batch_id();
        let submit_at = Instant::now();

        // Placement maps are keyed by the registered operation identity and are
        // joined with logical operations only after every transition is lowered.
        let mut operations = Vec::with_capacity(transitions.len());
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
        let mut buffer_placements = HashMap::with_capacity(transitions.len());

        for mut transition in transitions {
            let oid = self.next_op_id;
            self.next_op_id += 1;
            let request_id = transition.request_id;

            // Planning may queue successors speculatively, but registration still
            // requires the request epoch and exact resident predecessor.
            let Some((epoch, latest_device_version)) =
                self.running.get(&request_id).and_then(|state| {
                    state
                        .cursor
                        .resources
                        .worker_registered
                        .then(|| (state.epoch, state.latest_device_version.clone()))
                })
            else {
                tracing::error!(
                    request_id = request_id.0,
                    "planned operation lost its request"
                );
                self.fatal = true;
                return false;
            };
            if self.inflight.contains(request_id)
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
            let projected_successor = self.inflight.contains(request_id);
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
                    .inflight
                    .operations
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
                let predicate_kind = if transition.operation_variant == RunKind::ArDecode {
                    ProductKind::Token
                } else {
                    ProductKind::Completion
                };
                (
                    parent,
                    predecessor
                        .outputs()
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

            // KV descriptors include complete tables only on admission or growth;
            // fresh-page lists identify storage the worker must initialize now.
            let kv_lengths = transition_kv_lengths(&transition.intent);
            let new_page_count = transition.new_blocks.len();
            let mut operation_block_tables = Vec::new();
            let mut operation_new_cache_pages = Vec::new();
            let mut operation_forward_rows = Vec::new();
            if let Some(lengths) = kv_lengths {
                let table_changed =
                    admitted_request_keys.contains(&request_key) || new_page_count > 0;
                for group_id in 0..self.memory.cache().block_pool.num_groups() {
                    let page_ids = self
                        .running
                        .get(&request_id)
                        .and_then(|state| state.block_tables().get(group_id))
                        .map(BlockTable::page_ids)
                        .unwrap_or_default();
                    let request_pool_idx = self
                        .running
                        .get(&request_id)
                        .map(|state| state.request_pool_idx())
                        .expect("registered request has a live slot");
                    if table_changed {
                        operation_block_tables.push(IpcBlockTable {
                            request_pool_idx,
                            group_id: group_id as u32,
                            allocated_tokens: u32::try_from(
                                page_ids
                                    .len()
                                    .saturating_mul(self.info.kv_block_size() as usize),
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
                                        .div_ceil(self.info.kv_block_size() as usize)
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
                            .map(|state| state.request_pool_idx())
                            .expect("registered request has a live slot"),
                        seq_len: lengths.visible,
                        query_len: lengths.input,
                    });
                }
            }

            let output_event_bound = transition_output_bound(&transition);
            let planned_us = transition.planned_us;
            let reserved_buffers = std::mem::take(&mut transition.buffer_allocations);
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
                    for allocation in reserved_buffers {
                        self.memory.free(allocation);
                    }
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

            // Persistent product buffers transfer ownership from the transition
            // reservation into request state using the minted product identities.
            let operation_identity = (operation.request_key, operation.op_id);
            let persistent_outputs = operation
                .outputs()
                .iter()
                .filter(|output| output.uses_persistent_buffer())
                .map(ProductRef::buffer_id)
                .collect::<Vec<_>>();
            if persistent_outputs.len() != reserved_buffers.len() {
                for allocation in reserved_buffers {
                    self.memory.free(allocation);
                }
                tracing::error!(
                    request_id = request_id.0,
                    "registered operation changed its persistent buffer set"
                );
                self.fatal = true;
                self.fail_all_running("registered operation changed its persistent buffer set");
                return false;
            }
            let mut operation_buffers = Vec::with_capacity(reserved_buffers.len());
            let Some(state) = self.running.get_mut(&request_id) else {
                for allocation in reserved_buffers {
                    self.memory.free(allocation);
                }
                self.fatal = true;
                return false;
            };
            for (buffer, allocation) in persistent_outputs.into_iter().zip(reserved_buffers) {
                let (offset, bytes) = match allocation.placement() {
                    Placement::Buffer { offset, bytes } => (*offset, *bytes),
                    _ => unreachable!("persistent output allocation has a non-buffer placement"),
                };
                let replaced = state.allocations_mut().buffers.insert(buffer, allocation);
                debug_assert!(replaced.is_none(), "buffer identity was reused");
                operation_buffers.push(BufferPlacement {
                    buffer,
                    offset,
                    bytes,
                });
            }
            if !operation_buffers.is_empty() {
                buffer_placements.insert(operation_identity, operation_buffers);
            }

            // Diffusion rows include the positive branch and any negative CFG
            // branch, each bound to its own request-state row.
            if operation.kind == RunKind::DiffusionStep {
                let conditioning_tokens = match &apply.intent {
                    TransitionIntent::DenoiseGen {
                        physical_kv_len, ..
                    } => *physical_kv_len,
                    _ => unreachable!("media denoise has a non-flow scheduler delta"),
                };
                let query_len = u32::try_from(
                    self.running
                        .get(&request_id)
                        .map(|state| self.num_vae(&state.req.image))
                        .unwrap_or_default()
                        .saturating_add(u64::from(
                            self.profile.generation_limits.commit_marker_tokens,
                        )),
                )
                .unwrap_or(u32::MAX);
                let Some(state) = self.running.get_mut(&request_id) else {
                    self.fatal = true;
                    return false;
                };
                let main_slot = state.request_pool_idx();
                let mut alternative = None;
                if let Some(prefix) = state.flow_prefix.as_mut() {
                    let allocated_tokens = prefix
                        .block_tables()
                        .first()
                        .map(|table| table.capacity_tokens())
                        .unwrap_or_default();
                    if !prefix.diffusion_finalized || !prefix.new_pages.is_empty() {
                        for table in prefix.block_tables() {
                            operation_block_tables.push(IpcBlockTable {
                                request_pool_idx: prefix.request_pool_idx(),
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
                                request_pool_idx: prefix.request_pool_idx(),
                                group_id,
                                page_ids: pages,
                            });
                        }
                    }
                    let prefix_len = state.context.negative_prompt_ids.len() as u32;
                    if prefix_len > 0 && !prefix.diffusion_finalized {
                        operation_forward_rows.push(RowGeometry {
                            operation_index: 0,
                            request_pool_index: prefix.request_pool_idx(),
                            seq_len: 0,
                            query_len: prefix_len,
                        });
                    }
                    alternative = Some((prefix.request_pool_idx(), prefix_len));
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

            // Latent placements describe the request-owned page table and the
            // exact denoising interval executed by this operation.
            if matches!(
                operation.kind,
                RunKind::DiffusionPrepare | RunKind::DiffusionStep
            ) || operation
                .inputs()
                .iter()
                .any(|reference| reference.kind == uniserve_worker_ipc::ProductKind::Latent)
            {
                let Some(state) = self.running.get(&request_id) else {
                    self.fatal = true;
                    return false;
                };
                let page_table = state
                    .allocations()
                    .latent
                    .as_ref()
                    .and_then(|allocation| match allocation.placement() {
                        Placement::Latent { pages, .. } => Some(pages.clone()),
                        _ => None,
                    })
                    .unwrap_or_default();
                let latent_units = self.worker_image_latent_units_for(state).max(1);
                let (start_step, step_count) = match &apply.intent {
                    TransitionIntent::DenoiseGen {
                        start_step,
                        step_count,
                        ..
                    } => (u32::from(*start_step), u32::from(*step_count)),
                    TransitionIntent::PrepareGen { .. } => (0, 0),
                    TransitionIntent::CommitGen { step, .. } => (u32::from(*step), 0),
                    _ => {
                        tracing::error!(
                            request_id = request_id.0,
                            operation = operation.kind.as_str(),
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

            let operation_variant = operation.kind.as_str();
            if let Some(trace_ops) = trace_ops.as_mut() {
                let phase = self
                    .running
                    .get(&request_id)
                    .map(|state| state.cursor.phase);
                trace_ops.push(json!({
                    "request_id": request_id.0,
                    "op_id": operation.op_id.0,
                    "operation_type": operation_variant,
                    "phase": phase,
                    "operation": operation_trace(&operation, &apply),
                    "transition": &apply.intent,
                    "resources": {
                        "free_latent_on_apply": apply.free_latent_on_apply,
                        "replayability_after_apply": apply.replayability_after_apply,
                    },
                    "visibility": {
                        "und_tokens": format!("{:?}", apply.visibility.und_tokens),
                        "generated_image": apply.visibility.generated_image,
                    },
                }));
            }

            input_products.extend(payloads);
            let submitted_us = uniserve_core::now_monotonic_us();
            self.register_inflight(
                operation.clone(),
                apply,
                submit_at,
                submitted_us.saturating_sub(planned_us),
            );
            operations.push(operation);
            if let Some(st) = self.running.get_mut(&request_id) {
                st.latest_device_version = None;
            }
        }

        // Snapshot scheduler gauges at the batch boundary before ownership moves
        // into the executor.
        self.peak_ops_in_batch = self.peak_ops_in_batch.max(operations.len());
        self.stats
            .general
            .peak_ops
            .fetch_max(operations.len(), Ordering::Relaxed);
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
            .store(self.memory.free_blocks(), Ordering::Relaxed);
        let mixed = operations.first().is_some_and(|first| {
            operations
                .iter()
                .any(|operation| operation.kind != first.kind)
        });
        self.inflight.batch_started.insert(batch_id, submit_at);
        if operations
            .iter()
            .any(|operation| batch_kind(operation.kind) == BatchKind::Prefill)
        {
            self.inflight.prefill_steps.insert(batch_id);
        }

        if let Some(trace_ops) = trace_ops {
            let operation_types: Vec<&'static str> = operations
                .iter()
                .map(|operation| operation.kind.as_str())
                .collect();
            let req_ids: Vec<u64> = operations
                .iter()
                .map(|operation| operation.request_key.request_id.0)
                .collect();
            let admitted_request_ids: Vec<u64> = admissions
                .iter()
                .map(|admission| admission.request_key.request_id.0)
                .collect();
            self.trace_record(json!({
                "event": "batch_submitted",
                "at_s": now(),
                "batch_id": batch_id,
                "batch_size": operations.len(),
                "mixed": mixed,
                "operation_types": operation_types,
                "request_ids": req_ids,
                "admitted_request_ids": admitted_request_ids,
                "commands": commands,
                "ops": trace_ops,
                "scheduler": {
                    "policy": self.scheduler.config.policy,
                    "max_batch": self.scheduler.config.max_batch,
                    "max_num_batched_tokens": self.scheduler.config.max_num_batched_tokens,
                },
                "running": self.running.len(),
                "pending": self.scheduler.waiting_len(),
                "in_flight_before_submit": self.inflight.batch_started.len(),
                "free_blocks": self.memory.free_blocks(),
                "reserved_blocks": self.memory.reserved_blocks,
                "worker_image_latent_active": self.worker_image_latent_used(),
                "worker_image_latent_capacity": self.info.latent_capacity_units(),
            }));
        }
        if mixed {
            let operation_types: Vec<&'static str> = operations
                .iter()
                .map(|operation| operation.kind.as_str())
                .collect();
            let req_ids: Vec<u64> = operations
                .iter()
                .map(|operation| operation.request_key.request_id.0)
                .collect();
            tracing::debug!(
                batch_id,
                ?operation_types,
                ?req_ids,
                "submitting mixed forward batch"
            );
        }

        // Join each registered operation with the placement records accumulated
        // under its identity, then record the exact expected completion set.
        let logical_ops = operations
            .into_iter()
            .map(|operation| {
                let identity = (operation.request_key, operation.op_id);
                LogicalOp::new(
                    operation,
                    OpPlacement {
                        block_tables: block_tables.remove(&identity).unwrap_or_default(),
                        new_cache_pages: new_cache_pages.remove(&identity).unwrap_or_default(),
                        forward_rows: forward_rows.remove(&identity).unwrap_or_default(),
                        latent: latent_placements.remove(&identity),
                        decode: None,
                        buffers: buffer_placements.remove(&identity).unwrap_or_default(),
                    },
                )
            })
            .collect::<Vec<_>>();
        self.inflight.batch_operations.insert(
            batch_id,
            logical_ops
                .iter()
                .map(|op| (op.request_key(), op.id()))
                .collect(),
        );
        self.inflight
            .batch_group_worker_exec_us
            .insert(batch_id, HashMap::new());

        let mut batch_commands = admissions
            .into_iter()
            .map(|request| BatchCommand::Start { request })
            .collect::<Vec<_>>();
        batch_commands.extend(commands.clone());
        let batch = Batch::new(batch_id, logical_ops, batch_commands, input_products);
        if !commands.is_empty() {
            self.inflight.command_batches.insert(batch_id, commands);
        }

        // Backpressure retains the fully lowered batch for a later retry; a
        // terminal failure removes every in-flight index created above.
        match self.executor.submit(batch) {
            Ok(()) => true,
            Err(ExecutorSubmitError::WouldBlock(batch)) => {
                self.pending_submission = Some(batch);
                false
            }
            Err(ExecutorSubmitError::Failed(error)) => {
                self.inflight.batch_started.remove(&batch_id);
                self.inflight.prefill_steps.remove(&batch_id);
                self.inflight.batch_operations.remove(&batch_id);
                self.inflight.batch_group_worker_exec_us.remove(&batch_id);
                self.inflight.command_batches.remove(&batch_id);
                self.trace_record(json!({
                    "event": "batch_submit_failed",
                    "at_s": now(),
                    "batch_id": batch_id,
                    "error": format!("{error}"),
                }));
                tracing::error!("executor submit failed: {error}");
                if error.downcast_ref::<WorkerLossError>().is_some() {
                    self.on_executor_error(error);
                } else {
                    self.fatal = true;
                    self.fail_all_inflight(&format!("{error}"));
                }
                false
            }
        }
    }

    /// Returns whether generated output satisfies the trigger.
    pub(super) fn generated_trigger_matches(st: &ReqState) -> bool {
        st.req
            .policy
            .trigger
            .matches_generated(&st.cursor.replay.generated_ids)
    }

    /// Returns whether direct input satisfies the trigger.
    pub(super) fn direct_trigger_matches(st: &ReqState, token_id: u32) -> bool {
        st.req.policy.trigger.direct_token() == Some(token_id)
    }

    /// Returns the next token fed back into generation.
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

    /// Builds the generation trigger after prefill.
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

    /// Ensures the request capacity.
    pub(super) fn ensure_request_capacity(&mut self, id: RequestId, total_tokens: usize) -> bool {
        let (runtime, memory) = (&mut self.runtime, &mut self.memory);
        let Some(state) = runtime.state_mut().running.get_mut(&id) else {
            return false;
        };
        let groups = state.block_tables().len() as u32;
        memory
            .grow(
                &mut state.allocations_mut().kv,
                MemoryLayout::Kv {
                    tokens: total_tokens.min(u32::MAX as usize) as u32,
                    groups,
                },
            )
            .is_ok()
    }

    /// Activates the KV tables reserved for a request.
    pub(super) fn activate_request_tables(&self, id: RequestId) {
        let kv = self.memory.cache();
        if let Some(state) = self.running.get(&id) {
            for table in state.block_tables() {
                table.activate(&kv.block_pool);
            }
        }
    }

    /// Takes the block-ID delta accumulated since the request's previous operation.
    ///
    /// The delta contains every `blocks_for` entry beyond the last transmitted boundary.
    pub(super) fn take_new_blocks(&mut self, id: RequestId) -> Vec<BlockId> {
        let Some(st) = self.running.get_mut(&id) else {
            return Vec::new();
        };
        let all = st.block_tables()[0].page_ids();
        let sent = st.cursor.resources.blocks_sent.min(all.len());
        let new = all[sent..].to_vec();
        st.cursor.resources.blocks_sent = all.len();
        new
    }

    /// Returns unsent KV blocks to the free pool.
    pub(super) fn return_unsent_blocks(&mut self, id: RequestId, transition: &NextOp) {
        if let Some(state) = self.running.get_mut(&id) {
            state.cursor.resources.blocks_sent = state
                .cursor
                .resources
                .blocks_sent
                .saturating_sub(transition.new_blocks.len());
        }
    }

    /// Converts one state-machine intent into a bounded operation or fails the request.
    pub(super) fn plan_intent(
        &mut self,
        id: RequestId,
        cursor: &GenerationCursor,
        intent: TransitionIntent,
    ) -> Option<NextOp> {
        let planned = {
            let request = &self.running.get(&id)?.req;
            let kv_bytes_per_token = self
                .info
                .kv_cache
                .as_ref()
                .map_or(0, |cache| cache.bytes_per_token);
            plan(
                self.latent_dtype,
                kv_bytes_per_token,
                request,
                cursor,
                intent,
            )
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

    /// Plans the next schedulable transition from projected request state.
    pub(super) fn next_transition(&mut self, id: RequestId, budget: usize) -> Option<NextOp> {
        let projection = self.projected_cursor(id)?;
        let context_pending = self.running.get(&id).is_some_and(|st| {
            projection.ingest.prompt_cursor < st.context.prompt_ids.len() as u32
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
        let projected_branch = self.projected_cursor(id)?;
        let phase = projected_branch.phase;
        match phase {
            Phase::Encode => None,
            Phase::Prefill => {
                let st = self.running.get(&id)?;
                let prompt = st.effective_prompt().to_vec();
                let n = prompt.len();
                let projection = self.projected_cursor(id)?;
                let cursor = projection.ingest.prompt_cursor as usize;
                let (segment_index, segment_end) =
                    st.context.token_segment_at(cursor).unwrap_or((0, n));
                // Chunked prefill with the clip rule: the chunk is bounded by
                // the remaining step budget and the long-prefill threshold.
                let chunk_cap = (n - cursor)
                    .min(self.scheduler.config.long_prefill_threshold)
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
                    &projection,
                    TransitionIntent::IngestText {
                        segment_index,
                        start: cursor as u32,
                        end: end as u32,
                        logical_start: projection.und.logical_pos,
                        physical_start: projection.und.physical_kv_len,
                        token_ids: chunk,
                        new_blocks,
                        sampling_state,
                    },
                )
            }
            Phase::DecodeUnd => {
                let projected_successor = self.inflight.contains(id);
                // A successor registered before its predecessors resolve is
                // `inflight_len` unresolved points ahead of the committed cursor;
                // its minimum-token floor and force-finish flag are staged at
                // that exact projected point.
                let projected = if projected_successor {
                    self.inflight.len(id)
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
                let pos = projection.und.logical_pos;
                let relay_input = projected_successor || self.can_reuse_resolved_token_product(id);
                let capacity_target = decode_capacity_target(pos as usize, 0);
                if !self.ensure_request_capacity(id, capacity_target) {
                    return None;
                }
                let new_blocks = self.take_new_blocks(id);
                self.plan_intent(
                    id,
                    &projection,
                    TransitionIntent::DecodeUnd {
                        logical_position: pos,
                        physical_position: projection.und.physical_kv_len,
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
                    decode_capacity_target(projection.und.physical_kv_len as usize, 0);
                if !self.ensure_request_capacity(id, capacity_target) {
                    return None;
                }
                let image_id = projected_branch.image_gen.image_id;
                let position = projection.und.logical_pos;
                let physical_position = projection.und.physical_kv_len;
                let new_blocks = self.take_new_blocks(id);
                self.plan_intent(
                    id,
                    &projection,
                    TransitionIntent::CloseKv {
                        image_id,
                        logical_position: position,
                        physical_position,
                        token,
                        relay_input: false,
                        new_blocks,
                    },
                )
            }
            Phase::PublishKv => self.plan_intent(
                id,
                &projection,
                TransitionIntent::PublishKv {
                    image_id: projected_branch.image_gen.image_id,
                    physical_kv_len: projection.und.physical_kv_len,
                },
            ),
            Phase::PrepareGen => {
                let st = self.running.get(&id)?;
                let conditioning = projected_branch.image_gen.conditioning.clone()?;
                let image_id = projected_branch.image_gen.image_id;
                let latent_units = self.worker_image_latent_units_for(st).max(1);
                self.plan_intent(
                    id,
                    &projection,
                    TransitionIntent::PrepareGen {
                        image_id,
                        physical_kv_len: projection.und.physical_kv_len,
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
                let timestep = projected_branch.image_gen.steps_done;
                let remaining = st.req.image.steps.saturating_sub(timestep);
                if remaining == 0 {
                    return None;
                }
                let denoise_step_count = self.denoise_step_burst.max(1).min(remaining);
                let cfg = cfg_params(&st.req.image, cfg_branch_count(&st.req.image));
                let latent_units = self.worker_image_latent_units_for(st).max(1);
                let image_id = projected_branch.image_gen.image_id;
                let conditioning = projected_branch.image_gen.conditioning.clone()?;
                let latent = projected_branch.image_gen.latent.clone()?;
                let projection = self.projected_cursor(id)?;
                self.plan_intent(
                    id,
                    &projection,
                    TransitionIntent::DenoiseGen {
                        image_id,
                        physical_kv_len: projection.und.physical_kv_len,
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
                let image_id = projected_branch.image_gen.image_id;
                let latent = projected_branch.image_gen.latent.clone()?;
                let step = self.running.get(&id)?.req.image.steps;
                let projection = self.projected_cursor(id)?;
                self.plan_intent(
                    id,
                    &projection,
                    TransitionIntent::CommitGen {
                        image_id,
                        step,
                        latent,
                    },
                )
            }
            Phase::FeedbackEncode => {
                let st = self.running.get(&id)?;
                let feedback = st.req.policy.feedback.as_ref()?;
                let step_index = projected_branch.feedback.ingest_step;
                let step = feedback.ingest.steps.get(step_index).copied()?;
                let image_id = projected_branch.image_gen.image_id;
                let source = (feedback.source == uniserve_core::FeedbackSource::DeviceProduct)
                    .then(|| projected_branch.feedback.source_product.clone())
                    .flatten();
                let image_b64 = if source.is_none() {
                    st.cursor.feedback.image_b64.clone().unwrap_or_default()
                } else {
                    String::new()
                };
                let projection = self.projected_cursor(id)?;
                self.plan_intent(
                    id,
                    &projection,
                    TransitionIntent::EncodeFeedbackStep {
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
                let step_index = projected_branch.feedback.ingest_step;
                let is_final_step = step_index + 1 == feedback.ingest.steps.len();
                let image_id = projected_branch.image_gen.image_id;
                let logical_positions = feedback.ingest.logical_positions;
                let physical_kv_tokens = feedback.ingest.kv_effect(step_index)?;
                let feature = projected_branch.feedback.encoded_product.clone();
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
                    &projection,
                    TransitionIntent::FeedbackState {
                        image_id,
                        step_index,
                        is_final_step,
                        position: projection.und.logical_pos,
                        physical_start: projection.und.physical_kv_len,
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

    /// Plans the next ordered text or image context-ingest transition within `budget`.
    pub(super) fn next_context_ingest_transition(
        &mut self,
        id: RequestId,
        budget: usize,
        projection: GenerationCursor,
    ) -> Option<NextOp> {
        let cursor = projection.ingest.prompt_cursor as usize;

        // An image anchored at the current text cursor takes precedence over
        // further text ingestion so logical multimodal order is preserved.
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

            // Once an encoded feature exists, apply its declared KV effect at
            // the projected cursor and reserve the required cache pages.
            if state.cursor.phase == Phase::IngestState {
                let feature = state.cursor.ingest.encoded_product.clone()?;
                let physical_kv_tokens = image.ingest.kv_effect(step_index)?;
                let physical_bound = match physical_kv_tokens {
                    uniserve_core::ImageKvEffect::Exact { tokens } => tokens,
                    uniserve_core::ImageKvEffect::Bounded { max_tokens } => max_tokens,
                    uniserve_core::ImageKvEffect::WorkerDefined => match step {
                        uniserve_core::ImageIngestStep::VaeEncode => {
                            self.cap_max_vae_grid_tokens().min(u32::MAX as usize) as u32
                        }
                        uniserve_core::ImageIngestStep::VitEncode => {
                            self.profile.generation_limits.max_vit_grid_tokens
                        }
                    },
                };

                if !self.ensure_request_capacity(
                    id,
                    projection
                        .und
                        .physical_kv_len
                        .saturating_add(physical_bound) as usize,
                ) {
                    return None;
                }

                let new_blocks = self.take_new_blocks(id);
                return self.plan_intent(
                    id,
                    &projection,
                    TransitionIntent::IngestImageState {
                        segment_index: image.segment_index,
                        step_index,
                        is_final_step,
                        position: projection.und.logical_pos,
                        physical_start: projection.und.physical_kv_len,
                        logical_positions: image.ingest.logical_positions,
                        physical_kv_tokens,
                        feature,
                        new_blocks,
                    },
                );
            }

            // Encoder-cache hits are pinned before entering ingest state. If
            // request ownership disappears during acquisition, release the pin.
            let cache_read = state.req.cache.read;
            let cache_write = state.req.cache.write;
            let cache_key = encoder_cache_key(image.hash, step_index, step);
            let cached = if cache_read {
                self.memory.encoder_cache.lookup_product(cache_key)
            } else {
                None
            };
            if let Some(cached_product) = cached {
                let product = self.memory.encoder_cache.acquire(cache_key)?;
                if product != cached_product {
                    return None;
                }
                let Some(state) = self.running.get_mut(&id) else {
                    let _ = self.memory.encoder_cache.release(cache_key, &product);
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
                state.cursor.phase = Phase::IngestState;
                return self.next_context_ingest_transition(id, budget, projection);
            }

            // A cache miss schedules the encoder step; writable requests attach
            // a persistent key so completion can populate the cache.
            let persistent_cache_key = cache_write.then_some(cache_key);
            return self.plan_intent(
                id,
                &projection,
                TransitionIntent::EncodeImageStep {
                    segment_index: image.segment_index,
                    step_index,
                    step,
                    encoder_cache_key: persistent_cache_key,
                    image_b64: image.b64.clone(),
                    source_product: None,
                },
            );
        }

        // Select a text chunk bounded by the request budget, scheduler chunk
        // limit, semantic segment boundary, and next image position.
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
            .saturating_add(
                budget
                    .max(1)
                    .min(self.scheduler.config.long_prefill_threshold),
            )
            .min(prompt.len())
            .min(segment_end)
            .min(next_image.max(cursor + 1));

        if !self.ensure_request_capacity(id, projection.und.physical_kv_len as usize + end - cursor)
        {
            return None;
        }

        let sampling_state = self.sampling_state(id, 0);
        let new_blocks = self.take_new_blocks(id);
        self.plan_intent(
            id,
            &projection,
            TransitionIntent::IngestText {
                segment_index,
                start: cursor as u32,
                end: end as u32,
                logical_start: projection.und.logical_pos,
                physical_start: projection.und.physical_kv_len,
                token_ids: prompt[cursor..end].to_vec(),
                new_blocks,
                sampling_state,
            },
        )
    }

    /// Chooses whether a committed round-close token opens a branch or finishes the request.
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
                st.cursor.phase = Phase::DecodeUnd;
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

    /// Computes the operation's allowed/suppress masks from the host-side
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
        let ctx = crate::runtime::logits::ProcCtx {
            n_generated,
            eos: &self.ctrl.eos,
            generated: &st.cursor.replay.generated_ids,
            sampling: &st.req.sampling,
        };
        let masks = crate::runtime::logits::run_pipeline(&self.logits_pipeline, &ctx);
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

    /// Builds the branch-local static sampling state for one operation.
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
                self.inflight
                    .operations
                    .get(&id)
                    .into_iter()
                    .flatten()
                    .filter(|inflight| {
                        matches!(
                            inflight.generation_apply().intent,
                            TransitionIntent::FeedbackState {
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
        let finish_token_ids = state.finish_token_ids.clone();
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
}
