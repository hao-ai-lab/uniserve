//! Batch selection, call planning, and physical params assembly.
//!
//! Each pass selects a compatible execution lane, applies sequence and token
//! budgets, and emits at most one planned call per eligible request.

use super::*;
use uniserve_worker_ipc::{ForwardMode, MediaCall, TransferMode};

/// Computes the target decode capacity for a scheduling round.
fn decode_capacity_target(pos: usize, spec_len: usize) -> usize {
    pos.saturating_add(1).saturating_add(spec_len)
}

impl Scheduler {
    /// Assembles one scheduling-step batch in priority order.
    ///
    /// Each request contributes at most one call, prefill chunks consume only
    /// the remaining token budget, and first dispatch carries typed admission.
    pub(super) fn assemble(&mut self) -> Vec<ExecutionBatch> {
        let ids = self.assembly_order();
        let lane = self.select_batch_kind(&ids);
        let batches = self.assemble_batch(&ids, lane);
        if batches.is_empty() && !self.fatal && lane == Some(BatchKind::Prefill) {
            // A blocked prefill lane must not prevent already-ready decode work
            // from using the execution slot.
            return self.assemble_batch(&ids, Some(BatchKind::Decode));
        }
        batches
    }

    /// Charges actual token work, including each denoising step and CFG branch.
    fn computation_token_cost(&self, call: &Call) -> usize {
        match call.code {
            CallKind::Media(MediaCall::Denoising) => {
                let Some(state) = self.running.get(&call.request_key.request_id) else {
                    return 0;
                };
                usize::try_from(self.worker_image_latent_units_for(state).max(1))
                    .unwrap_or(usize::MAX)
                    .saturating_mul(usize::from(state.req.image.cfg_branch_count().max(1)))
                    .saturating_mul(call.bounds.max_tokens.max(1) as usize)
            }
            CallKind::Forward(ForwardMode::Verify) => 1 + call.input_token_ids.len(),
            _ => call.bounds.max_tokens as usize,
        }
    }

    /// Reserves persistent buffers, latent pages, and transfer capacity for one computation.
    fn reserve_generation_resources(&mut self, call: &Call) -> Option<Vec<Allocation>> {
        let id = call.request_key.request_id;
        if call.code == CallKind::Media(MediaCall::Denoising) && !self.ensure_flow_prefix(id) {
            return None;
        }
        let uses_transfer = call.bounds.max_transfer_bytes > 0;
        if uses_transfer && self.num_pending_transfers >= self.transfer_capacity {
            return None;
        }
        let request_key = self
            .running
            .get(&id)
            .map(|state| RequestKey::new(self.engine_id, id, state.request_epoch));
        let Some(request_key) = request_key else {
            return None;
        };
        if self.worker_target(request_key, call.code).is_none() {
            return None;
        }
        let mut buffer_allocations = Vec::new();
        for bytes in call.buffer_outputs().map(TensorRef::max_bytes) {
            let allocation = match self.buffer_pool.allocate(request_key, bytes, 256) {
                Ok(allocation) => allocation,
                Err(_) => {
                    for allocation in buffer_allocations {
                        self.free_allocation(allocation);
                    }
                    if let Some(product) = self.encoder_cache.evict_one() {
                        self.free_buffers([product.buffer_id()]);
                    }
                    return None;
                }
            };
            buffer_allocations.push(allocation);
        }
        if matches!(
            call.code,
            CallKind::Media(MediaCall::LatentPreparation) | CallKind::Media(MediaCall::Denoising)
        ) && self.worker_tracks_image_latent()
        {
            let latent_units = self
                .running
                .get(&id)
                .map(|state| self.worker_image_latent_units_for(state).max(1))?;
            let Some(state) = self.running.get_mut(&id) else {
                return None;
            };
            let request_key = RequestKey::new(self.engine_id, id, state.request_epoch);
            let allocations = state.allocations_mut();
            let result = if let Some(allocation) = allocations.latent.as_mut() {
                self.latent_pool.grow(allocation, latent_units)
            } else {
                self.latent_pool
                    .allocate(request_key, latent_units)
                    .map(|allocation| {
                        allocations.latent = Some(allocation);
                    })
            };
            if result.is_err() {
                for allocation in buffer_allocations {
                    self.buffer_pool.free(allocation);
                }
                return None;
            }
        }
        if uses_transfer {
            self.num_pending_transfers += 1;
        }
        Some(buffer_allocations)
    }

    /// Selects compatible call kinds within one lane's token and sequence budgets.
    pub(super) fn assemble_batch(
        &mut self,
        ids: &[RequestId],
        lane: Option<BatchKind>,
    ) -> Vec<ExecutionBatch> {
        // One batch per computation: a batch is one numerical call on one
        // component, so a rank receives it as a single homogeneous group. The
        // component a call binds to follows from its computation, so the two
        // together name one group. Computations keep the order in which their
        // first call was selected.
        let mut code_order: Vec<CallKind> = Vec::new();
        let mut code_batches: HashMap<CallKind, ExecutionBatch> = HashMap::new();
        let submit_at = Instant::now();
        // The per-step token budget is the binding limit; individual prefill
        // chunks are clipped to its remaining capacity.
        let mut budget: usize = self.config.max_num_batched_tokens;
        // A decode pass may also co-schedule text prefill tokens, bounded by
        // this budget, so a prompt does not wait for a pass of its own. Each
        // computation still travels as its own batch and its own numerical
        // call.
        let mut mixed_left: usize = if lane == Some(BatchKind::Decode) {
            self.config.mixed_prefill_tokens
        } else {
            0
        };
        let mut selected = 0usize;
        for id in ids.iter().copied() {
            if selected >= self.config.max_batch {
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
            let next_type = self.peek_next_call_variant(id);
            // A decode pass may co-schedule text prefill rows. They are
            // dispatched as their own batch, so the two call kinds remain
            // separate numerical calls.
            let mut mixed_prefill = false;
            if let (Some(target), Some(call_variant)) = (lane, next_type)
                && {
                    let candidate_lane = batch_kind(call_variant);
                    matches!(candidate_lane, BatchKind::Prefill | BatchKind::Decode)
                        && candidate_lane != target
                }
            {
                mixed_prefill = target == BatchKind::Decode
                    && call_variant == CallKind::Forward(ForwardMode::Prefill)
                    && mixed_left > 0
                    && self.running.get(&id).is_some_and(|st| {
                        st.is_replayable_text() && !st.req.sampling.prompt_logprobs_requested()
                    });
                if !mixed_prefill {
                    continue;
                }
            }
            // The denoiser is an exclusive lane: the oldest request holds it
            // for its whole step sequence, and a younger request's denoising
            // dispatches only once that sequence has completed. Denoising
            // steps of different requests are therefore never in flight
            // together, in any lane.
            if next_type == Some(CallKind::Media(MediaCall::Denoising))
                && (self.denoiser_lane_held_by_other(id) || !self.flow_prefix_is_schedulable(id))
            {
                continue;
            }
            // Build one call when its exact resident resources fit.
            let call_budget = if mixed_prefill {
                budget.min(mixed_left)
            } else {
                budget
            };
            if let Some(mut call) = self.next_generation_computation(id, call_budget) {
                let planned_us = uniserve_core::now_monotonic_us();
                if mixed_prefill {
                    mixed_left = mixed_left.saturating_sub(self.computation_token_cost(&call));
                }
                budget = budget.saturating_sub(self.computation_token_cost(&call));
                let Some(reserved_buffers) = self.reserve_generation_resources(&call) else {
                    self.record_domain_backpressure(call.code);
                    if let Some(state) = self.running.get_mut(&id) {
                        state.num_kv_blocks_sent = state
                            .num_kv_blocks_sent
                            .saturating_sub(call.bounds.max_kv_pages as usize);
                    }
                    tracing::debug!(
                        request_id = id.0,
                        "call registration is paused by physical resource pressure"
                    );
                    continue;
                };
                let code = call.code;
                if !code_batches.contains_key(&code) {
                    let batch_id = self.next_batch_id();
                    code_batches.insert(
                        code,
                        ExecutionBatch::new(batch_id, Vec::new(), Vec::new(), Vec::new()),
                    );
                    code_order.push(code);
                }
                let request_index = u32::try_from(code_batches[&code].requests.len())
                    .expect("selected request count fits the IPC index");
                call.call_id = CallId::new(code_batches[&code].id, request_index);
                call.coordinates = self
                    .projected_coordinates(id)
                    .expect("scheduled request retains its running state");
                let finish_token_ids = self
                    .running
                    .get(&id)
                    .map(|state| state.finish_token_ids.clone())
                    .unwrap_or_default();
                let engine_id = self.engine_id;
                let mut admitted = false;
                if let Some(st) = self.running.get_mut(&id)
                    && !st.worker_registered
                {
                    st.worker_registered = true;
                    let request_key = RequestKey::new(engine_id, id, st.request_epoch);
                    let admission = NewRequest::new(
                        request_key,
                        st.request_pool_idx(),
                        Some(ArRequestParams {
                            sampling: st.req.sampling.clone(),
                            negative_token_ids: st.req.negative_prompt_token_ids.clone(),
                            finish_token_ids,
                            initial_position: st.num_computed_prompt_tokens,
                        }),
                        st.req.generates_images().then(|| UmmRequestParams {
                            image: st.req.image.clone(),
                        }),
                    )
                    .expect("validated request produces a valid admission");
                    // Admission is the first request-state dependency.
                    st.last_state_call_id = CallId::default();
                    st.latest_token = None;
                    code_batches
                        .get_mut(&code)
                        .expect("computation batch exists")
                        .commands
                        .push(BatchCommand::Start { request: admission });
                    admitted = true;
                }
                let mut batch = code_batches
                    .remove(&code)
                    .expect("computation batch exists");
                let prepared = self.prepare_generation_call(
                    call,
                    reserved_buffers,
                    admitted,
                    planned_us,
                    &mut batch,
                );
                code_batches.insert(code, batch);
                if prepared.is_none() {
                    // Preparation marks the scheduler fatal before it fails.
                    return Vec::new();
                }
                selected += 1;
            }
        }
        code_order.retain(|code| {
            code_batches
                .get(code)
                .is_some_and(|batch| !batch.requests.is_empty())
        });
        if code_order.is_empty() {
            return Vec::new();
        }

        // Prompt commands remain disjoint from earlier state writers. The
        // round's retirements travel with its last batch so no earlier call
        // loses the state it still reads. They do not travel in a batch of
        // their own: a rank's queue depth bounds submissions, not round trips,
        // so a command-only batch costs the round a queue slot and halves how
        // many rounds a depth-two rank can hold in flight.
        let prompt_only = code_order
            .iter()
            .all(|code| batch_kind(*code) == BatchKind::Prefill);
        let commands = if prompt_only {
            let requests = code_order
                .iter()
                .flat_map(|code| code_batches[code].requests.iter())
                .map(|(call, _)| call.request_key.request_id)
                .collect::<HashSet<_>>();
            self.take_commands(|command| {
                requests.contains(&command.request_key().request_id)
                    || matches!(command, BatchCommand::Finish { .. })
            })
        } else {
            self.take_commands(|_| true)
        };
        let last = *code_order
            .last()
            .expect("selected calls name a computation");
        let mut batches = Vec::with_capacity(code_order.len());
        for code in code_order {
            let mut batch = code_batches
                .remove(&code)
                .expect("computation batch exists");
            if code == last {
                batch.commands.extend(commands.iter().cloned());
            }
            batches.push(self.finish_generation_batch(batch, submit_at));
        }
        batches
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
            let Some(call_type) = self.peek_next_call_variant(id) else {
                continue;
            };
            match batch_kind(call_type) {
                BatchKind::Decode => {
                    if self.can_schedule_next(id) {
                        if self.has_pending_calls(id) {
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
        if prefill_ready
            && self
                .pending_batches
                .values()
                .filter(|batch| batch.prefill)
                .count()
                < PREFILL_WINDOW_CREDITS
        {
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
        match self.peek_next_call_variant(id) {
            Some(
                CallKind::Media(MediaCall::TextEncoding)
                | CallKind::Media(MediaCall::VisionEncoding)
                | CallKind::Media(MediaCall::LatentEncoding)
                | CallKind::Forward(ForwardMode::Prefill),
            ) => 0,
            Some(
                CallKind::Forward(ForwardMode::Decode)
                | CallKind::Forward(ForwardMode::Verify)
                | CallKind::Media(MediaCall::ImageDecoding)
                | CallKind::Transfer(TransferMode::KvInstall),
            ) => 1,
            Some(CallKind::Media(
                MediaCall::Denoising
                | MediaCall::VideoDecoding
                | MediaCall::AudioDecoding
                | MediaCall::VideoEncoding
                | MediaCall::AudioEncoding
                | MediaCall::Muxing,
            )) => 2,
            Some(
                CallKind::Media(MediaCall::LatentPreparation)
                | CallKind::Transfer(TransferMode::Tensor)
                | CallKind::Transfer(TransferMode::KvPublish),
            )
            | None => 3,
        }
    }

    /// Determines the next call kind without mutating request or resource state.
    pub(super) fn peek_next_call_variant(&self, id: RequestId) -> Option<CallKind> {
        let st = self.running.get(&id)?;
        if st.image_reservation_pending {
            return None;
        }
        if let Some(variant) = self.pending_successor_code(id) {
            return Some(variant);
        }
        if st.has_context_images()
            && st.phase == Phase::Prefill
            && st
                .req
                .multimodal_inputs
                .images
                .get(st.num_ingested_images)
                .is_some_and(|item| Some(item.position) == self.num_scheduled_prompt_tokens(id))
        {
            return st.pending_image_step().map(|step| match step {
                ImageIngestStep::VaeEncode => CallKind::Media(MediaCall::LatentEncoding),
                ImageIngestStep::VitEncode => CallKind::Media(MediaCall::VisionEncoding),
            });
        }
        Some(match st.phase {
            Phase::Encode => match st.pending_image_step()? {
                ImageIngestStep::VaeEncode => CallKind::Media(MediaCall::LatentEncoding),
                ImageIngestStep::VitEncode => CallKind::Media(MediaCall::VisionEncoding),
            },
            Phase::IngestState => CallKind::Forward(ForwardMode::Prefill),
            Phase::Prefill => CallKind::Forward(ForwardMode::Prefill),
            Phase::DecodeUnd => CallKind::Forward(ForwardMode::Decode),
            Phase::CloseKv => CallKind::Forward(ForwardMode::Prefill),
            Phase::PublishKv => CallKind::Transfer(TransferMode::KvPublish),
            Phase::PrepareGen => CallKind::Media(MediaCall::LatentPreparation),
            Phase::DenoiseGen if st.num_completed_denoise_steps >= st.req.image.steps => {
                CallKind::Media(MediaCall::ImageDecoding)
            }
            Phase::DenoiseGen => CallKind::Media(MediaCall::Denoising),
            Phase::CommitGen => CallKind::Media(MediaCall::ImageDecoding),
            Phase::FeedbackEncode => {
                let feedback = &st.req.image_generation;
                feedback.feedback_source.as_ref()?;
                match &feedback
                    .feedback_encoders
                    .get(st.feedback_encoder_index)?
                    .encoder
                {
                    ImageIngestStep::VaeEncode => CallKind::Media(MediaCall::LatentEncoding),
                    ImageIngestStep::VitEncode => CallKind::Media(MediaCall::VisionEncoding),
                }
            }
            Phase::FeedbackState => CallKind::Forward(ForwardMode::Prefill),
        })
    }

    /// Registers one selected computation with its allocated physical inputs and outputs.
    fn prepare_generation_call(
        &mut self,
        mut call: Call,
        reserved_buffers: Vec<Allocation>,
        admitted: bool,
        planned_us: u64,
        batch: &mut ExecutionBatch,
    ) -> Option<()> {
        let request_id = call.request_key.request_id;

        // Planning may queue successors speculatively, but registration still
        // requires the request epoch and exact resident predecessor.
        let Some((request_epoch, latest_token)) = self.running.get(&request_id).and_then(|state| {
            state
                .worker_registered
                .then(|| (state.request_epoch, state.latest_token.clone()))
        }) else {
            tracing::error!(request_id = request_id.0, "planned call lost its request");
            self.fatal = true;
            return None;
        };
        if self.has_pending_calls(request_id) && !self.can_queue_successor(request_id, call.code) {
            tracing::error!(
                request_id = request_id.0,
                "scheduler attempted to issue an unsafe projected successor"
            );
            self.fatal = true;
            return None;
        }

        let request_key = RequestKey::new(self.engine_id, request_id, request_epoch);
        // A device successor consumes the token or completion tensor of
        // either its in-flight predecessor or the latest resolved call.
        // The first call and host-observed transitions use the latest
        // accepted call identity.
        let projected_successor = self.has_pending_calls(request_id);
        let reusable_device_token =
            if !projected_successor && self.can_reuse_resolved_token_product(request_id) {
                latest_token
            } else {
                None
            };
        let predicate = if projected_successor {
            if self.execution_predecessor(request_id).is_none() {
                tracing::error!(
                    request_id = request_id.0,
                    "projected successor has no execution predecessor"
                );
                self.fatal = true;
                return None;
            }
            let Some(predecessor) = self
                .pending_calls
                .get(&request_id)
                .and_then(|queue| queue.back())
                .map(|call| &call.call)
            else {
                tracing::error!(
                    request_id = request_id.0,
                    "projected successor lost its predecessor call"
                );
                self.fatal = true;
                return None;
            };
            if call.code == CallKind::Forward(ForwardMode::Decode) {
                predecessor.token_output.clone()
            } else {
                predecessor.completion_output.clone()
            }
        } else if let Some(parent) = reusable_device_token {
            Some(parent)
        } else {
            if self.state_predecessor(request_id).is_none() {
                self.fatal = true;
                return None;
            }
            None
        };

        call.predicate = predicate;
        // KV descriptors include complete tables only on admission or growth;
        // fresh-page lists identify storage the worker must initialize now.
        // Freeze physical inputs before registering this computation as pending.
        // Accepted progress may stop on cancellation while these inputs still drain.
        let (_, visible) = self.scheduled_token_lengths(request_id)?;
        let kv_lengths = match call.code {
            CallKind::Forward(ForwardMode::Prefill) => Some(KvLengths {
                visible,
                input: if is_prompt_extend(&call) {
                    call.input_token_ids.len().min(u32::MAX as usize) as u32
                } else if consumes_image_features(&call) {
                    call.bounds.max_tokens
                } else {
                    1
                },
            }),
            CallKind::Forward(ForwardMode::Decode) | CallKind::Forward(ForwardMode::Verify) => {
                Some(KvLengths { visible, input: 1 })
            }
            CallKind::Transfer(TransferMode::KvPublish)
            | CallKind::Media(MediaCall::LatentPreparation)
            | CallKind::Media(MediaCall::Denoising) => Some(KvLengths { visible, input: 0 }),
            _ => None,
        };
        let start_step = match call.code {
            CallKind::Media(MediaCall::LatentPreparation) => Some(0),
            CallKind::Media(MediaCall::Denoising) => {
                self.num_scheduled_denoise_steps(request_id).map(u32::from)
            }
            CallKind::Media(MediaCall::ImageDecoding) => self
                .running
                .get(&request_id)
                .map(|state| u32::from(state.req.image.steps)),
            _ => None,
        };
        let num_image_kv_tokens = if consumes_image_features(&call) {
            let state = self.running.get(&request_id)?;
            let encoder = if is_feedback_computation(&call) {
                state
                    .req
                    .image_generation
                    .feedback_encoders
                    .get(self.scheduled_feedback(request_id)?.0)
            } else {
                state
                    .req
                    .multimodal_inputs
                    .images
                    .get(state.num_ingested_images)
                    .and_then(|image| image.encoders.get(state.image_encoder_index))
            };
            encoder.and_then(|encoder| encoder.num_kv_tokens)
        } else {
            None
        };
        let new_page_count = call.bounds.max_kv_pages as usize;
        let mut call_block_tables = Vec::new();
        let mut call_new_cache_pages = Vec::new();
        let mut call_forward = ForwardBatch::default();
        if let Some(lengths) = kv_lengths {
            let table_changed = admitted || new_page_count > 0;
            for group_id in 0..self.cache().block_pool.num_groups() {
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
                    call_block_tables.push(IpcBlockTable {
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
                let fresh_pages = if admitted {
                    let retained_pages = if group_id == 0 {
                        self.running
                            .get(&request_id)
                            .map(|state| {
                                (state.num_computed_prompt_tokens as usize)
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
                    call_new_cache_pages.push(CachePageAllocation {
                        request_pool_idx,
                        group_id: group_id as u32,
                        page_ids: fresh_pages,
                    });
                }
            }
            if lengths.input > 0 {
                call_forward.push(
                    0,
                    self.running
                        .get(&request_id)
                        .map(|state| state.request_pool_idx())
                        .expect("registered request has a live slot"),
                    lengths.visible + lengths.input,
                    lengths.input,
                    true,
                );
            }
        }

        if let Err(error) =
            generation::register_call(&mut call, request_key, &mut self.next_product_generation)
        {
            for allocation in reserved_buffers {
                self.free_allocation(allocation);
            }
            tracing::error!(
                request_id = request_id.0,
                ?error,
                "scheduler authority exhausted product identity space"
            );
            self.fatal = true;
            self.fail_all_running("scheduler authority exhausted product identity space");
            return None;
        }

        // Persistent product buffers transfer ownership from the local
        // reservation into request state using the minted product identities.
        let persistent_outputs = call
            .buffer_outputs()
            .map(TensorRef::buffer_id)
            .collect::<Vec<_>>();
        if persistent_outputs.len() != reserved_buffers.len() {
            for allocation in reserved_buffers {
                self.free_allocation(allocation);
            }
            tracing::error!(
                request_id = request_id.0,
                "registered call changed its persistent buffer set"
            );
            self.fatal = true;
            self.fail_all_running("registered call changed its persistent buffer set");
            return None;
        }
        let mut call_buffers = Vec::with_capacity(reserved_buffers.len());
        let Some(state) = self.running.get_mut(&request_id) else {
            for allocation in reserved_buffers {
                self.free_allocation(allocation);
            }
            self.fatal = true;
            return None;
        };
        for (buffer, allocation) in persistent_outputs.into_iter().zip(reserved_buffers) {
            let (offset, bytes) = match &allocation {
                Allocation::Buffer { offset, bytes, .. } => (*offset, *bytes),
                _ => unreachable!("persistent output allocation has a non-buffer params"),
            };
            let replaced = state.allocations_mut().buffers.insert(buffer, allocation);
            debug_assert!(replaced.is_none(), "buffer identity was reused");
            call_buffers.push(BufferAllocation {
                buffer,
                offset,
                bytes,
            });
        }

        // Diffusion rows include the positive branch and any negative CFG
        // branch, each bound to its own request-state row.
        if call.code == CallKind::Media(MediaCall::Denoising) {
            let conditioning_tokens = kv_lengths
                .expect("denoising declares its conditioning KV range")
                .visible;
            let query_len = u32::try_from(
                self.running
                    .get(&request_id)
                    .map(|state| self.num_vae(&state.req.image))
                    .unwrap_or_default()
                    .saturating_add(u64::from(self.generation_limits.commit_marker_tokens)),
            )
            .unwrap_or(u32::MAX);
            let Some(state) = self.running.get_mut(&request_id) else {
                self.fatal = true;
                return None;
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
                        call_block_tables.push(IpcBlockTable {
                            request_pool_idx: prefix.request_pool_idx(),
                            group_id: table.group_id() as u32,
                            page_ids: table.page_ids(),
                            allocated_tokens: u32::try_from(allocated_tokens).unwrap_or(u32::MAX),
                        });
                    }
                }
                for (group_id, pages) in std::mem::take(&mut prefix.new_pages) {
                    if !pages.is_empty() {
                        call_new_cache_pages.push(CachePageAllocation {
                            request_pool_idx: prefix.request_pool_idx(),
                            group_id,
                            page_ids: pages,
                        });
                    }
                }
                let prefix_len = state.req.negative_prompt_token_ids.len() as u32;
                if prefix_len > 0 && !prefix.diffusion_finalized {
                    call_forward.push(0, prefix.request_pool_idx(), prefix_len, prefix_len, true);
                }
                alternative = Some((prefix.request_pool_idx(), prefix_len));
            }
            for branch in 0..usize::from(cfg_branch_count(&state.req.image).max(1)) {
                let (request_pool_index, seq_len) = if branch == 0 {
                    (main_slot, conditioning_tokens)
                } else {
                    alternative.unwrap_or((main_slot, conditioning_tokens))
                };
                call_forward.push(0, request_pool_index, seq_len + query_len, query_len, false);
            }
        }

        let mut latent = None;

        // Latent allocations describe the request-owned page table and the
        // exact denoising interval executed by this call.
        if matches!(
            call.code,
            CallKind::Media(MediaCall::LatentPreparation) | CallKind::Media(MediaCall::Denoising)
        ) || call.latent_input.is_some()
        {
            let Some(state) = self.running.get(&request_id) else {
                self.fatal = true;
                return None;
            };
            let page_table = state
                .allocations()
                .latent
                .as_ref()
                .and_then(|allocation| match allocation {
                    Allocation::Latent { pages, .. } => Some(pages.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            let latent_units = self.worker_image_latent_units_for(state).max(1);
            let Some(start_step) = start_step else {
                tracing::error!(
                    request_id = request_id.0,
                    call = call.code.as_str(),
                    "latent call has no submitted step"
                );
                self.fatal = true;
                return None;
            };
            let step_count = if call.code == CallKind::Media(MediaCall::Denoising) {
                call.bounds.max_tokens
            } else {
                0
            };
            latent = Some(LatentParams {
                request_key: call.request_key,
                call_id: call.call_id,
                page_table,
                latent_units: latent_units.min(u64::from(u32::MAX)) as u32,
                height: state.req.image.height,
                width: state.req.image.width,
                start_step,
                step_count,
            });
        }

        let (worker, entry) = self.select_worker(&call);
        call.component = entry;

        let submitted_us = uniserve_core::now_monotonic_us();
        self.register_inflight(
            call.clone(),
            InflightInput::Generation {
                image_kv: if consumes_image_features(&call) {
                    let lengths = kv_lengths.expect("image extension declares its KV input range");
                    Some((lengths.visible, num_image_kv_tokens))
                } else {
                    None
                },
                start_step: latent.as_ref().map(|input| input.start_step),
            },
            submitted_us.saturating_sub(planned_us),
        );
        batch.requests.push((
            call,
            RequestPlacement {
                worker,
                request_pool_idx: None,
                block_tables: call_block_tables,
                new_cache_pages: call_new_cache_pages,
                forward: call_forward,
                latent,
                decode: None,
                buffers: call_buffers,
            },
        ));
        if let Some(st) = self.running.get_mut(&request_id) {
            st.latest_token = None;
        }
        Some(())
    }

    /// Finalizes batch statistics and pending-result ownership before submission.
    pub(super) fn finish_generation_batch(
        &mut self,
        mut batch: ExecutionBatch,
        submit_at: Instant,
    ) -> ExecutionBatch {
        if batch.id == 0 {
            batch.id = self.next_batch_id();
        }

        // Snapshot scheduler gauges at the batch boundary before ownership moves
        // into the executor.
        self.peak_calls_in_batch = self.peak_calls_in_batch.max(batch.requests.len());
        self.stats
            .general
            .peak_calls
            .fetch_max(batch.requests.len(), Ordering::Relaxed);
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
            .store(self.free_blocks(), Ordering::Relaxed);
        self.register_pending_batch(&batch, submit_at);
        batch
    }

    /// Returns whether generated output satisfies the trigger.
    pub(super) fn generated_trigger_matches(st: &RequestState) -> bool {
        st.req
            .image_generation
            .trigger
            .matches_generated(&st.generated_token_ids)
    }

    /// Returns whether direct input satisfies the trigger.
    pub(super) fn direct_trigger_matches(st: &RequestState, token_id: u32) -> bool {
        st.req.image_generation.trigger.direct_token() == Some(token_id)
    }

    /// Returns the next token fed back into generation.
    pub(super) fn feedback_next_token(&self, id: RequestId) -> Option<u32> {
        let next = self
            .running
            .get(&id)?
            .req
            .image_generation
            .feedback_next_token;
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
                .image_generation
                .trigger
                .matches_generated(&st.req.prompt_token_ids)
    }

    /// Ensures the request capacity.
    pub(super) fn ensure_request_capacity(&mut self, id: RequestId, total_tokens: usize) -> bool {
        let Some(state) = self.running.get_mut(&id) else {
            return false;
        };
        let Some(cache) = self.cache.as_ref() else {
            return false;
        };
        cache
            .grow(
                &mut state.allocations_mut().kv,
                total_tokens.min(u32::MAX as usize) as u32,
            )
            .is_ok()
    }

    /// Activates the KV tables reserved for a request.
    pub(super) fn activate_request_tables(&self, id: RequestId) {
        let kv = self.cache();
        if let Some(state) = self.running.get(&id) {
            for table in state.block_tables() {
                table.activate(&kv.block_pool);
            }
        }
    }

    /// Builds a computation and records the KV pages owned by its next dispatch.
    pub(super) fn plan_computation(
        &mut self,
        id: RequestId,
        build: impl FnOnce(&Self, &GenerationRequest) -> Result<Call, generation::PlanningError>,
    ) -> Option<Call> {
        let planned = build(self, &self.running.get(&id)?.req);
        match planned {
            Ok(mut call) => {
                if matches!(
                    call.code,
                    CallKind::Forward(ForwardMode::Prefill)
                        | CallKind::Forward(ForwardMode::Decode)
                        | CallKind::Forward(ForwardMode::Verify)
                ) {
                    // The KV owner keeps page identities. The scheduled record
                    // only needs the number of fresh pages for dispatch and drain.
                    let state = self.running.get_mut(&id)?;
                    let allocated = state.block_tables()[0].len();
                    call.bounds.max_kv_pages = allocated
                        .saturating_sub(state.num_kv_blocks_sent)
                        .min(u32::MAX as usize)
                        as u32;
                    state.num_kv_blocks_sent = allocated;
                }
                Some(call)
            }
            Err(error) => {
                tracing::error!(request_id = id.0, ?error, "generation planning failed");
                self.finish(id, FinishReason::Error);
                None
            }
        }
    }

    /// Selects the next computation using accepted state and pending input ranges.
    pub(super) fn next_generation_computation(
        &mut self,
        id: RequestId,
        budget: usize,
    ) -> Option<Call> {
        let (logical_position, kv_visible_len) = self.scheduled_token_lengths(id)?;
        let context_pending = self.running.get(&id).is_some_and(|st| {
            self.num_scheduled_prompt_tokens(id)
                .is_some_and(|count| count < st.req.prompt_token_ids.len() as u32)
                || st.num_ingested_images < st.req.multimodal_inputs.images.len()
        });
        if context_pending {
            return self.next_context_computation(id, budget);
        }
        if self.running.get(&id)?.image_reservation_pending
            && !self.promote_gen_branch_reservation(id)
        {
            return None;
        }
        let phase = self.next_generation_phase(id)?;
        let image_id = self.running.get(&id)?.image_id;
        match phase {
            Phase::Encode => None,
            Phase::Prefill => {
                let st = self.running.get(&id)?;
                let n = st.req.prompt_token_ids.len();
                let cursor = self.num_scheduled_prompt_tokens(id)? as usize;
                // Chunked prefill with the clip rule: the chunk is bounded by
                // the remaining step budget and the long-prefill threshold.
                let chunk_cap = (n - cursor)
                    .min(self.config.long_prefill_threshold)
                    .min(budget.max(1));
                let end = (cursor + chunk_cap.max(1)).min(n);
                if !self.ensure_request_capacity(id, end) {
                    return None;
                }
                let sampling_state = self.build_token_masks(id, 0);

                self.plan_computation(id, |_scheduler, request| {
                    generation::plan_prompt(request, cursor as u32, end as u32, sampling_state)
                })
            }
            Phase::DecodeUnd => {
                let projected_successor = self.has_pending_calls(id);
                // Stage minimum-token and force-finish constraints at the
                // successor's scheduled token position. Device predicates
                // prevent an inactive descendant from accepting that position.
                let projected = if projected_successor {
                    self.num_pending_calls(id)
                } else {
                    0
                };
                let sampling_state = self.build_token_masks(id, projected);
                let st = self.running.get(&id)?;
                // The prior committed token this decode continues from. It is
                // attached as a host input only when no exact selected-point
                // product is eligible for device continuation.
                let input_token = st.next_token;
                let pos = logical_position;
                let relay_input = projected_successor || self.can_reuse_resolved_token_product(id);
                let capacity_target = decode_capacity_target(pos as usize, 0);
                if !self.ensure_request_capacity(id, capacity_target) {
                    return None;
                }

                self.plan_computation(id, |_scheduler, request| {
                    generation::plan_decode(
                        request,
                        pos,
                        None,
                        input_token,
                        relay_input,
                        sampling_state,
                    )
                })
            }
            Phase::CloseKv => {
                let token = self.running.get(&id)?.next_token;
                let capacity_target = decode_capacity_target(kv_visible_len as usize, 0);
                if !self.ensure_request_capacity(id, capacity_target) {
                    return None;
                }

                self.plan_computation(id, |_scheduler, request| {
                    generation::plan_close_kv(request, token, false)
                })
            }
            Phase::PublishKv => self.plan_computation(id, |scheduler, request| {
                generation::plan_kv_publish(
                    scheduler
                        .info
                        .kv_cache
                        .as_ref()
                        .map_or(0, |cache| cache.publication_bytes_per_token()),
                    request,
                    kv_visible_len,
                )
            }),
            Phase::PrepareGen => {
                let conditioning = self.kv_conditioning(id)?;
                self.plan_computation(id, |scheduler, request| {
                    generation::plan_diffusion_prepare(
                        &scheduler.generation_limits,
                        scheduler.latent_dtype,
                        request,
                        image_id,
                        conditioning,
                    )
                })
            }
            Phase::DenoiseGen => {
                // Each scheduled interval consumes only the remaining model steps.
                // Accepted completion remains separately tracked on the request.
                let st = self.running.get(&id)?;
                let timestep = self.num_scheduled_denoise_steps(id)?;
                let remaining = st.req.image.steps.saturating_sub(timestep);
                if remaining == 0 {
                    return None;
                }
                let denoise_step_count = self.denoise_step_burst.max(1).min(remaining);
                let conditioning = self.kv_conditioning(id)?;
                let latent = self
                    .pending_output(id, |call| call.latent_output.as_ref())
                    .or(self.running.get(&id)?.image_latent.as_ref())
                    .cloned()?;
                self.plan_computation(id, |scheduler, request| {
                    generation::plan_diffusion_step(
                        &scheduler.generation_limits,
                        scheduler.latent_dtype,
                        request,
                        denoise_step_count,
                        conditioning,
                        latent,
                    )
                })
            }
            Phase::CommitGen => {
                let latent = self
                    .pending_output(id, |call| call.latent_output.as_ref())
                    .or(self.running.get(&id)?.image_latent.as_ref())
                    .cloned()?;
                self.plan_computation(id, |_scheduler, request| {
                    generation::plan_diffusion_finalize(request, latent)
                })
            }
            Phase::FeedbackEncode => {
                let st = self.running.get(&id)?;
                let feedback = &st.req.image_generation;
                feedback.feedback_source.as_ref()?;
                let step_index = self.scheduled_feedback(id)?.0;
                let step = feedback.feedback_encoders.get(step_index)?.encoder;
                let source = (feedback.feedback_source
                    == Some(uniserve_core::FeedbackSource::DeviceProduct))
                .then(|| {
                    self.pending_output(id, |call| call.image_output.as_ref())
                        .or(st.feedback_source.as_ref())
                        .cloned()
                })
                .flatten();
                let image_b64 = if source.is_none() {
                    st.feedback_image_b64.clone().unwrap_or_default()
                } else {
                    String::new()
                };
                self.plan_computation(id, |scheduler, request| {
                    generation::plan_encode(
                        &scheduler.generation_limits,
                        request,
                        step,
                        image_b64,
                        source,
                        true,
                    )
                })
            }
            Phase::FeedbackState => {
                let st = self.running.get(&id)?;
                let feedback = &st.req.image_generation;
                feedback.feedback_source.as_ref()?;
                let step_index = self.scheduled_feedback(id)?.0;
                let is_final_step = step_index + 1 == feedback.feedback_encoders.len();
                let logical_positions = feedback.num_feedback_positions;
                let encoder_input = *feedback.feedback_encoders.get(step_index)?;
                let feature = self
                    .pending_output(id, |call| call.encoder_output.as_ref())
                    .or(st.feedback_features.as_ref())
                    .cloned();
                let sample_continuation = is_final_step && feedback.sample_feedback_continuation;
                let Some(feature) = feature else {
                    self.finish(id, FinishReason::Error);
                    return None;
                };

                let sampling_state = self.build_token_masks(id, 0);
                self.plan_computation(id, |scheduler, request| {
                    generation::plan_image_extend(
                        &scheduler.generation_limits,
                        request,
                        encoder_input,
                        feature,
                        true,
                        sample_continuation,
                        if sample_continuation {
                            sampling_state
                        } else {
                            None
                        },
                        Some(u64::from(
                            logical_position.saturating_add(logical_positions.max(1)),
                        )),
                    )
                })
            }
            Phase::IngestState => self.next_context_computation(id, budget),
        }
    }

    /// Plans the next ordered text or image context-ingest transition within `budget`.
    pub(super) fn next_context_computation(
        &mut self,
        id: RequestId,
        budget: usize,
    ) -> Option<Call> {
        let cursor = self.num_scheduled_prompt_tokens(id)? as usize;
        let (_, kv_visible_len) = self.scheduled_token_lengths(id)?;

        // An image anchored at the current text cursor takes precedence over
        // further text ingestion so logical multimodal order is preserved.
        let image = self
            .running
            .get(&id)?
            .req
            .multimodal_inputs
            .images
            .get(self.running.get(&id)?.num_ingested_images)
            .cloned();
        if let Some(image) = image.as_ref()
            && image.position as usize == cursor
        {
            let state = self.running.get(&id)?;
            let step_index = state.image_encoder_index;
            let step = image.encoders.get(step_index)?.encoder;

            // Once an encoded feature exists, apply its declared KV effect at
            // its scheduled positions and reserve the required cache pages.
            if state.phase == Phase::IngestState {
                let feature = state.input_image_features.clone()?;
                let encoder_input = *image.encoders.get(step_index)?;
                let physical_bound = encoder_input
                    .kv_token_capacity(&self.generation_limits)
                    .expect("input encoder capacity was validated at admission");

                if !self.ensure_request_capacity(
                    id,
                    kv_visible_len.saturating_add(physical_bound) as usize,
                ) {
                    return None;
                }

                return self.plan_computation(id, |scheduler, request| {
                    generation::plan_image_extend(
                        &scheduler.generation_limits,
                        request,
                        encoder_input,
                        feature,
                        false,
                        false,
                        None,
                        None,
                    )
                });
            }

            // Encoder-cache hits are pinned before entering ingest state. If
            // request ownership disappears during acquisition, release the pin.
            let cache_read = state.req.cache.read;
            let cache_key = encoder_cache_key(image.hash, step_index, step);
            let cached = if cache_read {
                self.encoder_cache.lookup_product(cache_key)
            } else {
                None
            };
            if let Some(cached_product) = cached {
                let product = self.encoder_cache.acquire(cache_key)?;
                if product != cached_product {
                    return None;
                }
                let Some(state) = self.running.get_mut(&id) else {
                    let _ = self.encoder_cache.release(cache_key, &product);
                    return None;
                };
                state.encoder_cache_pins.push(EncoderCachePin {
                    key: cache_key,
                    product: product.clone(),
                });
                state.input_image_features = Some(product);
                state.phase = Phase::IngestState;
                return self.next_context_computation(id, budget);
            }

            // A cache miss schedules the encoder. Completion derives the same
            // key from the immutable image input when cache writes are enabled.
            return self.plan_computation(id, |scheduler, request| {
                generation::plan_encode(
                    &scheduler.generation_limits,
                    request,
                    step,
                    image.b64.clone(),
                    None,
                    false,
                )
            });
        }

        // Select a text chunk bounded by the request budget, scheduler chunk
        // limit, and next image position.
        let (prompt_len, next_image) = {
            let st = self.running.get(&id)?;
            let prompt_len = st.req.prompt_token_ids.len();
            let next_image = image
                .as_ref()
                .map_or(prompt_len, |image| image.position as usize);
            (prompt_len, next_image)
        };
        if cursor >= prompt_len {
            return None;
        }

        let end = cursor
            .saturating_add(budget.max(1).min(self.config.long_prefill_threshold))
            .min(prompt_len)
            .min(next_image.max(cursor + 1));

        if !self.ensure_request_capacity(id, kv_visible_len as usize + end - cursor) {
            return None;
        }

        let sampling_state = self.build_token_masks(id, 0);

        self.plan_computation(id, |_scheduler, request| {
            generation::plan_prompt(request, cursor as u32, end as u32, sampling_state)
        })
    }

    /// Chooses whether a committed round-close token opens a branch or finishes the request.
    pub(super) fn close_context_round(&mut self, id: RequestId, close_token: u32) {
        let (triggered, can_open_gen_branch, n_gen, max_tokens) = {
            let Some(st) = self.running.get(&id) else {
                return;
            };
            (
                st.req
                    .image_generation
                    .trigger
                    .matches_round_close(&st.round_token_ids, close_token),
                st.can_open_gen_branch(),
                st.num_generated_tokens,
                st.req.max_und_tokens,
            )
        };
        if triggered && can_open_gen_branch && n_gen < max_tokens {
            return self.begin_image(id);
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

    /// Builds only constraints that vary with the scheduled sampling position.
    /// Admission supplies the static whitelist and finish tokens. `projected`
    /// counts unresolved predecessors, so minimum-token and image limits apply
    /// at the successor's position without observing device results. Penalties
    /// remain owned by the worker's accepted-token state.
    pub(super) fn build_token_masks(
        &self,
        id: RequestId,
        projected: usize,
    ) -> Option<SamplingState> {
        let n_generated = self.running.get(&id).map_or(0, |state| {
            state.num_generated_tokens.saturating_add(projected)
        });
        let projected_images_done = self.running.get(&id).map_or(0, |state| {
            state
                .num_generated_images
                .saturating_add(self.scheduled_feedback(id).map_or(0, |(_, images)| images))
        });
        let state = self.running.get(&id)?;
        let sampling = &state.req.sampling;
        let generated = &state.generated_token_ids;
        let mut suppress = Vec::new();
        if n_generated < sampling.min_tokens {
            suppress.extend_from_slice(&self.ctrl.eos);
        }
        for word in &sampling.bad_words_ids {
            let Some((&last, prefix)) = word.split_last() else {
                continue;
            };
            if generated.ends_with(prefix) {
                suppress.push(last);
            }
        }
        // Image-budget enforcement: once a request has produced max_images, a
        // direct branch-opening token is
        // suppressed, so an eager or positively-biased model must return to
        // text/EOS instead of emitting an un-actionable trigger into its
        // own context forever. Suppression beats logit bias (bias skips
        // -inf'd logits on the worker).
        if state.req.generates_images()
            && projected_images_done >= state.req.image.max_images as usize
            && let Some(trigger) = state.req.image_generation.trigger.direct_token()
        {
            suppress.push(trigger);
        }
        suppress.sort_unstable();
        suppress.dedup();

        let transition_token_ids = (state.req.generates_images()
            && projected_images_done < state.req.image.max_images as usize)
            .then(|| state.req.image_generation.trigger.direct_token())
            .flatten()
            .into_iter()
            .collect();
        let sampling = SamplingState {
            allowed_token_ids: None,
            suppressed_token_ids: suppress,
            finish_token_ids: Vec::new(),
            transition_token_ids,
            force_finish: n_generated.saturating_add(1) >= state.req.max_und_tokens,
        };
        (sampling != SamplingState::default()).then_some(sampling)
    }
}
