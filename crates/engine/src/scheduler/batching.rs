//! Batch selection, call planning, and physical params assembly.
//!
//! Each pass selects a compatible execution lane, applies sequence and token
//! budgets, and emits at most one planned call per eligible request.
//!
//! This is the generation path: token requests in `running`, including the
//! image-generation and feedback phases of unified multimodal requests.
//! `Scheduler::schedule_batches` alternates it with the video media path.
//! A pass plans each request's next call from its accepted state plus the
//! calls still in flight (`scheduled_token_lengths`,
//! `num_scheduled_denoise_steps`), so a successor can be planned before its
//! predecessor resolves. Each planned call then reserves its storage, is
//! registered as in flight, and joins the pass's batch for its `CallKind`:
//! one batch per computation, each a single numerical call on one component.
//!
//! The lane (`BatchKind`) only separates prefill from decode. `BatchKind::Media`
//! calls are eligible in every pass, and a pass with no lane applies no filter.

use super::*;
use uniserve_worker_ipc::{ForwardMode, MediaCall, TransferMode};

/// Returns the KV capacity, in tokens, that must be allocated before a decode
/// that writes position `pos` and `spec_len` further positions.
fn decode_capacity_target(pos: usize, spec_len: usize) -> usize {
    pos.saturating_add(1).saturating_add(spec_len)
}

impl Scheduler {
    /// Assembles one scheduling-step batch in priority order.
    ///
    /// Each request contributes at most one call, prefill chunks consume only
    /// the remaining token budget, and first dispatch carries typed admission.
    ///
    /// Returns the pass's batches, empty when nothing can be scheduled or call
    /// preparation failed; most preparation failures latch engine-fatal.
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
    ///
    /// A denoising call costs its latent units times its guidance branches
    /// times its step count, which it declares as `bounds.max_tokens`; it
    /// costs nothing once its request has left `running`. A verify call costs
    /// its input tokens plus one, and every other call its `max_tokens` bound.
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
    ///
    /// Returns the call's output buffer spans, in `Call::buffer_outputs`
    /// order, or `None` when the call cannot run now: the flow prefix of a
    /// denoising call cannot be allocated, transfer capacity is exhausted,
    /// the request is not running, `Placement::worker_target` finds no worker,
    /// a buffer or latent-page allocation fails, or (after latching
    /// engine-fatal) the request holds no allocations.
    ///
    /// Buffer spans reserved before a failure are returned to the pool, but a
    /// failure is not free of side effects: a flow prefix already allocated
    /// stays with the request, and a failed buffer allocation evicts one
    /// unpinned encoder-cache product, if any, and queues a `Free` for its
    /// buffer. Only a successful reservation counts against the transfer
    /// capacity.
    fn reserve_generation_resources(&mut self, call: &Call) -> Option<Vec<BufferSpan>> {
        let id = call.request_key.request_id;
        if call.code == CallKind::Media(MediaCall::Denoising) && !self.ensure_flow_prefix(id) {
            return None;
        }
        let uses_transfer = call.bounds.max_transfer_bytes > 0;
        if uses_transfer && self.inflight.num_pending_transfers >= self.transfer_capacity {
            return None;
        }
        let request_key = self
            .running
            .get(&id)
            .map(|state| RequestKey::new(self.engine_id, id, state.request_epoch))?;
        self.placement
            .worker_target(self.executor.as_ref(), &self.info, request_key, call.code)?;

        let mut buffer_allocations = Vec::new();
        for bytes in call.buffer_outputs().map(TensorRef::max_bytes) {
            let allocation = match self.storage.buffer_pool.allocate(request_key, bytes, 256) {
                Ok(allocation) => allocation,
                Err(_) => {
                    for allocation in buffer_allocations {
                        self.storage.buffer_pool.free(allocation);
                    }
                    // The evicted product's span returns to the pool only
                    // when workers acknowledge the queued `Free` command.
                    if let Some(product) = self.storage.encoder_cache.evict_one() {
                        self.free_buffers([product.buffer_id()]);
                    }
                    return None;
                }
            };
            buffer_allocations.push(allocation);
        }

        // Latent pages are reserved only when the worker advertises a paged
        // latent pool (`worker_tracks_image_latent`). They grow in place
        // across the trajectory's calls; a failed grow leaves the existing
        // pages allocated.
        if matches!(
            call.code,
            CallKind::Media(MediaCall::LatentPreparation) | CallKind::Media(MediaCall::Denoising)
        ) && self.worker_tracks_image_latent()
        {
            let latent_units = self
                .running
                .get(&id)
                .map(|state| self.worker_image_latent_units_for(state).max(1))?;
            let state = self.running.get_mut(&id)?;
            let Some(allocations) = state.allocations_mut() else {
                for allocation in buffer_allocations {
                    self.storage.buffer_pool.free(allocation);
                }
                self.invariant_broken("a request planning latent storage is admitted");
                return None;
            };
            let result = if let Some(allocation) = allocations.latent.as_mut() {
                self.storage.latent_pool.grow(allocation, latent_units)
            } else {
                self.storage
                    .latent_pool
                    .allocate(latent_units)
                    .map(|allocation| {
                        allocations.latent = Some(allocation);
                    })
            };
            if result.is_err() {
                for allocation in buffer_allocations {
                    self.storage.buffer_pool.free(allocation);
                }
                return None;
            }
        }

        if uses_transfer {
            self.inflight.num_pending_transfers += 1;
        }
        Some(buffer_allocations)
    }

    /// Selects compatible call kinds within one lane's token and sequence budgets.
    ///
    /// Walks `ids` in assembly order until `max_batch` calls are selected or
    /// the token budget is spent. Every selected call is registered in flight
    /// before this returns. A failed call preparation returns no batches; most
    /// such failures latch engine-fatal first.
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
            // separate numerical calls. Only replayable text requests without
            // prompt logprobs qualify. The lane check rejects only a prefill
            // call in a decode pass or the reverse; `BatchKind::Media` calls
            // pass.
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
            // The denoiser is an exclusive lane: a request holds it while one
            // of its denoising calls is in flight, so denoising steps of
            // different requests are never in flight together, in any lane.
            // Between two steps of one request, another request may take it.
            if next_type == Some(CallKind::Media(MediaCall::Denoising))
                && (self.inflight.denoiser_lane_held_by_other(id)
                    || !self.flow_prefix_is_schedulable(id))
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

                // Only a call that obtains its storage is charged to the
                // pass. A call that cannot run now leaves the budget to later
                // requests; in particular, requests waiting for the flow-prefix
                // row must not spend the pass before the request that holds
                // the row, which is the one whose steps release it.
                let Some(reserved_buffers) = self.reserve_generation_resources(&call) else {
                    self.record_domain_backpressure(call.code);
                    // Planning a forward advanced `num_kv_blocks_sent` past its
                    // fresh pages; roll it back so the next plan declares them.
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
                let cost = self.computation_token_cost(&call);
                if mixed_prefill {
                    mixed_left = mixed_left.saturating_sub(cost);
                }
                budget = budget.saturating_sub(cost);

                // Identify the call by its batch and row, and stamp the
                // coordinates projected through the request's in-flight calls.
                let code = call.code;
                let batch = code_batches.entry(code).or_insert_with(|| {
                    code_order.push(code);
                    ExecutionBatch::new(
                        self.inflight.next_batch_id(),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                    )
                });
                let Ok(request_index) = u32::try_from(batch.requests.len()) else {
                    self.invariant_broken("a batch's request count fits the IPC index");
                    return Vec::new();
                };
                call.call_id = CallId::new(batch.id, request_index);
                let Some(coordinates) = self.projected_coordinates(id) else {
                    self.invariant_broken("a scheduled request retains its running state");
                    return Vec::new();
                };
                call.coordinates = coordinates;

                // A request's first call carries its `Start` admission in the
                // same batch.
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
                    let request_key = RequestKey::new(engine_id, id, st.request_epoch);
                    // Admission reserved the request row, and submission
                    // validated the parameters this admission carries.
                    let admission = st
                        .request_pool_idx()
                        .zip(u32::try_from(st.req.multimodal_inputs.images.len()).ok())
                        .map(|(request_pool_idx, input_images)| {
                            NewRequest::new(
                                request_key,
                                request_pool_idx,
                                Some(ArRequestParams {
                                    sampling: st.req.sampling.clone(),
                                    negative_token_ids: st.req.negative_prompt_token_ids.clone(),
                                    finish_token_ids,
                                    initial_position: st.num_computed_prompt_tokens,
                                }),
                                st.req.generates_images().then(|| st.req.image.clone()),
                                input_images,
                            )
                        });
                    let Some(Ok(admission)) = admission else {
                        self.invariant_broken("a running request forms a valid worker admission");
                        return Vec::new();
                    };
                    st.worker_registered = true;
                    // Admission is the first request-state dependency.
                    st.last_state_call_id = CallId::default();
                    st.latest_token = None;
                    batch.commands.push(BatchCommand::Start {
                        request: Box::new(admission),
                    });
                    admitted = true;
                }
                let prepared = self.prepare_generation_call(
                    call,
                    reserved_buffers,
                    admitted,
                    planned_us,
                    batch,
                );
                if prepared.is_none() {
                    // Most preparation failures latch engine-fatal first.
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
        let Some(&last) = code_order.last() else {
            return Vec::new();
        };

        // A prefill-only pass takes only the queued commands of its own
        // requests plus `Finish` commands; other requests' `Free` commands
        // wait for a later batch. The round's retirements travel with its last
        // batch so no earlier call loses the state it still reads. They do not
        // travel in a batch of their own: a rank's queue depth bounds
        // submissions, not round trips, so a command-only batch costs the
        // round a queue slot and halves how many rounds a depth-two rank can
        // hold in flight.
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
        let mut batches = Vec::with_capacity(code_order.len());
        for code in code_order {
            let Some(mut batch) = code_batches.remove(&code) else {
                self.invariant_broken("every selected call kind has its batch");
                return Vec::new();
            };
            if code == last {
                batch.commands.extend(commands.iter().cloned());
            }
            batches.push(self.finish_generation_batch(batch, submit_at));
        }
        batches
    }

    /// Chooses the highest-priority execution lane that has schedulable work.
    ///
    /// Prefill wins while fewer than `PREFILL_WINDOW_CREDITS` batches carrying
    /// `BatchKind::Prefill` calls await their results; past that, ready decode
    /// work takes the pass, and prefill runs only when no decode is ready.
    /// `BatchKind::Media` readiness never selects a lane: with only such work
    /// ready this returns `None`, and `assemble_batch` then applies no lane
    /// filter.
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
                        if self.inflight.has_pending_calls(id) {
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
                .inflight
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

    /// Returns running requests in batch-assembly order: by
    /// `assembly_priority`, then by admission order within a priority.
    pub(super) fn assembly_order(&self) -> Vec<RequestId> {
        let mut ids: Vec<(usize, RequestId)> = self
            .running_order
            .iter()
            .enumerate()
            .filter_map(|(index, id)| self.running.contains_key(id).then_some((index, *id)))
            .collect();
        ids.sort_by_key(|(idx, id)| (self.assembly_priority(*id), *idx));
        ids.into_iter().map(|(_, id)| id).collect()
    }

    /// Returns the scheduling priority used during batch assembly; lower
    /// values are assembled first.
    ///
    /// Context ingestion (encoders and prefill) comes first; then token decode
    /// and verification, image decoding, and KV installation; then denoising
    /// and the video and audio media calls; and last latent preparation,
    /// tensor transfers, KV publication, and requests with no next call.
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
    ///
    /// A call chained onto the request's in-flight calls takes precedence.
    /// Otherwise an input image anchored at the scheduled prompt cursor is
    /// encoded before more text, and the request's phase decides. Returns
    /// `None` when the request is not running, waits on its image-branch
    /// reservation, or has no call to issue.
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
            Phase::DenoiseGen if st.denoising.is_complete() => {
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
    ///
    /// Sets the call's device predicate, derives its KV, forward-row, and
    /// latent inputs from the lengths its in-flight predecessors project,
    /// mints product identities, moves `reserved_buffers` into the request's
    /// allocations, selects the executing worker, registers the call in
    /// flight, and appends it with its `RequestPlacement` to `batch`.
    /// `admitted` marks a call whose batch carries the request's `Start`, so
    /// its complete KV tables are sent.
    ///
    /// Returns `None` when a scheduler invariant fails, including an unsafe
    /// projected successor; most such paths latch engine-fatal first.
    fn prepare_generation_call(
        &mut self,
        mut call: Call,
        reserved_buffers: Vec<BufferSpan>,
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
        if self.inflight.has_pending_calls(request_id)
            && !self.can_queue_successor(request_id, call.code)
        {
            tracing::error!(
                request_id = request_id.0,
                "scheduler attempted to issue an unsafe projected successor"
            );
            self.fatal = true;
            return None;
        }

        let request_key = RequestKey::new(self.engine_id, request_id, request_epoch);
        // A device successor consumes the token or completion tensor of
        // either its in-flight predecessor or the latest resolved call as its
        // predicate: a decode follows the predecessor's sampled token, any
        // other successor its completion product. The first call and
        // host-observed transitions use the latest accepted call identity and
        // carry no predicate.
        let projected_successor = self.inflight.has_pending_calls(request_id);
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
                .inflight
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

        // Freeze physical inputs before registering this computation as pending.
        // Accepted progress may stop on cancellation while these inputs still drain.
        // `visible` is the KV length after the in-flight predecessors; `input`
        // counts the tokens this call appends. Publication, latent preparation,
        // and denoising only read the request's KV.
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
        // Latent preparation opens the trajectory at step zero, a denoising
        // call starts after the steps already scheduled, and image decoding
        // sits at the end of the schedule.
        let start_step = match call.code {
            CallKind::Media(MediaCall::LatentPreparation) => Some(0),
            CallKind::Media(MediaCall::Denoising) => self.num_scheduled_denoise_steps(request_id),
            CallKind::Media(MediaCall::ImageDecoding) => self
                .running
                .get(&request_id)
                .map(|state| u32::from(state.req.image.steps)),
            _ => None,
        };
        // An image-feature extension declares the KV tokens of the encoder
        // that produced its feature: the scheduled feedback encoder, or the
        // current input image's encoder.
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

        // `max_kv_pages` counts the pages the first group's table gained since
        // the previous dispatch (`plan_computation`).
        let new_page_count = call.bounds.max_kv_pages as usize;
        let mut call_block_tables = Vec::new();
        let mut call_new_cache_pages = Vec::new();
        let mut call_forward = ForwardBatch::default();
        if let Some(lengths) = kv_lengths {
            // Registration above found the request running, and a running
            // token request always has the text KV cache.
            let Some((state, tables, request_pool_idx, cache)) =
                self.running.get(&request_id).and_then(|state| {
                    Some((
                        state,
                        state.block_tables()?,
                        state.request_pool_idx()?,
                        self.storage.cache()?,
                    ))
                })
            else {
                self.invariant_broken("a registered token request is admitted with a KV cache");
                return None;
            };
            // KV descriptors include complete tables only on admission or
            // growth; fresh-page lists identify storage the worker must
            // initialize now. On admission every page is fresh except the
            // group-0 pages already holding the computed prompt prefix
            // (`num_computed_prompt_tokens`); on growth each group's last
            // `new_page_count` pages are.
            let table_changed = admitted || new_page_count > 0;
            for group_id in 0..cache.block_pool.num_groups() {
                let page_ids = tables
                    .get(group_id)
                    .map(BlockTable::page_ids)
                    .unwrap_or_default();
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
                        (state.num_computed_prompt_tokens as usize)
                            .div_ceil(self.info.kv_block_size() as usize)
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
            // One forward row per call that appends KV: it attends to the
            // visible prefix plus its input and persists the input in KV.
            if lengths.input > 0 {
                call_forward.push(
                    0,
                    request_pool_idx,
                    lengths.visible + lengths.input,
                    lengths.input,
                    true,
                );
            }
        }

        // Assign the call's final product identities from the scheduler-wide
        // generation counter; exhausting it fails every running request.
        if let Err(error) =
            generation::register_call(&mut call, request_key, &mut self.next_product_generation)
        {
            for allocation in reserved_buffers {
                self.storage.buffer_pool.free(allocation);
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
                self.storage.buffer_pool.free(allocation);
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
        let Some(allocations) = self
            .running
            .get_mut(&request_id)
            .and_then(|state| state.allocations_mut())
        else {
            for allocation in reserved_buffers {
                self.storage.buffer_pool.free(allocation);
            }
            self.invariant_broken("a registered call's request is admitted");
            return None;
        };
        for (buffer, allocation) in persistent_outputs.into_iter().zip(reserved_buffers) {
            let (offset, bytes) = (allocation.offset, allocation.bytes);
            let replaced = allocations.buffers.insert(buffer, allocation);
            debug_assert!(replaced.is_none(), "buffer identity was reused");
            call_buffers.push(BufferAllocation {
                buffer,
                offset,
                bytes,
            });
        }

        // A denoising call adds one query row per guidance branch; its query
        // is the latent grid plus the commit marker tokens and writes no KV.
        // Branch 0 attends to the request's own conditioning KV. Every other
        // branch attends to the flow prefix (the negative-prompt KV in its own
        // request row) when the request holds one, and to the request's own
        // KV otherwise. Until a denoising call completes successfully and
        // finalizes the prefix, the call also prefills a non-empty negative
        // prompt into it; the prefix's tables are resent until then and
        // whenever the prefix has gained pages.
        if call.code == CallKind::Media(MediaCall::Denoising) {
            let Some(KvLengths {
                visible: conditioning_tokens,
                ..
            }) = kv_lengths
            else {
                self.invariant_broken("denoising declares its conditioning KV range");
                return None;
            };
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
            let Some(main_slot) = state.request_pool_idx() else {
                self.invariant_broken("a denoising request is admitted");
                return None;
            };
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
            let Some(allocations) = state.allocations() else {
                self.invariant_broken("a request with latent work is admitted");
                return None;
            };
            let placement = LatentPlacement {
                page_table: allocations
                    .latent
                    .as_ref()
                    .map(|allocation| allocation.pages.clone())
                    .unwrap_or_default(),
                latent_units: self
                    .worker_image_latent_units_for(state)
                    .max(1)
                    .min(u64::from(u32::MAX)) as u32,
                height: state.req.image.height,
                width: state.req.image.width,
            };
            let Some(start_step) = start_step else {
                tracing::error!(
                    request_id = request_id.0,
                    call = call.code.as_str(),
                    "latent call has no submitted step"
                );
                self.fatal = true;
                return None;
            };
            // A denoising call declares its planned step count as its token
            // bound; the interval repeats it for the worker.
            let step_count = if call.code == CallKind::Media(MediaCall::Denoising) {
                call.bounds.max_tokens
            } else {
                0
            };
            latent = Some(Denoising::params(
                call.request_key,
                call.call_id,
                &placement,
                start_step,
                step_count,
            ));
        }

        // Worker selection records the request's residency on that worker.
        let Some((worker, entry)) =
            self.placement
                .select_worker(self.executor.as_ref(), &self.info, &call)
        else {
            self.invariant_broken("a planned call retains an executable component");
            return None;
        };
        call.component = entry;

        let image_kv = if consumes_image_features(&call) {
            let Some(lengths) = kv_lengths else {
                self.invariant_broken("an image extension declares its KV input range");
                return None;
            };
            Some((lengths.visible, num_image_kv_tokens))
        } else {
            None
        };
        let submitted_us = uniserve_core::now_monotonic_us();
        self.register_inflight(
            call.clone(),
            InflightInput::Generation {
                image_kv,
                latent: latent.clone(),
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
                readers: Default::default(),
            },
        ));
        // The resolved token product serves only the first call after its
        // resolution; later calls chain from the in-flight one.
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
        // Command-only batches from `refill_executor` arrive with id zero.
        if batch.id == 0 {
            batch.id = self.inflight.next_batch_id();
        }

        // Snapshot scheduler gauges at the batch boundary before ownership moves
        // into the executor.
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
            .store(self.storage.free_blocks(), Ordering::Relaxed);
        self.inflight.register_pending_batch(&batch, submit_at);
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

    /// Returns whether the prompt itself matches the image-generation trigger
    /// and the request can open an image branch; `false` for a request that
    /// is not running.
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

    /// Grows the request's KV tables to hold `total_tokens` tokens.
    ///
    /// Returns `false` when the request is not running, the cache cannot grow,
    /// or (after latching engine-fatal) a running request lacks its
    /// allocations or the KV cache.
    pub(super) fn ensure_request_capacity(&mut self, id: RequestId, total_tokens: usize) -> bool {
        let Some(state) = self.running.get_mut(&id) else {
            return false;
        };
        let (Some(allocations), Some(cache)) =
            (state.allocations_mut(), self.storage.cache.as_ref())
        else {
            self.invariant_broken("a running token request is admitted with a KV cache");
            return false;
        };
        cache
            .grow(
                &mut allocations.kv,
                total_tokens.min(u32::MAX as usize) as u32,
            )
            .is_ok()
    }

    /// Activates the KV tables reserved for a request.
    ///
    /// A request holds tables only once admitted, and only token requests,
    /// which always have the KV cache, hold them.
    pub(super) fn activate_request_tables(&self, id: RequestId) {
        let tables = self.running.get(&id).and_then(RequestState::block_tables);
        if let (Some(tables), Some(kv)) = (tables, self.storage.cache()) {
            for table in tables {
                table.activate(&kv.block_pool);
            }
        }
    }

    /// Builds a computation and records the KV pages owned by its next dispatch.
    ///
    /// For a forward call, sets `bounds.max_kv_pages` to the pages the first
    /// group's table gained since the last dispatch and advances
    /// `num_kv_blocks_sent`; a caller that drops the call must roll that back.
    ///
    /// Returns `None` when the request is not running, when planning fails
    /// (the request is then finished with `Error`), or, after latching
    /// engine-fatal, when a forward's request holds no KV tables.
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
                    let Some(allocated) = state
                        .block_tables()
                        .and_then(|tables| tables.first())
                        .map(BlockTable::len)
                    else {
                        self.invariant_broken("a request planning a forward holds its KV tables");
                        return None;
                    };
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
    ///
    /// Prompt text and input images still to ingest go through
    /// `next_context_computation`; otherwise the request's next phase decides.
    /// Returns `None` when nothing can be planned now, including when the KV
    /// capacity a call needs or the image-branch reservation is unavailable.
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
                let sampling_state = self.build_token_masks(id, 0, 0);

                self.plan_computation(id, |_scheduler, request| {
                    generation::plan_prompt(request, cursor as u32, end as u32, sampling_state)
                })
            }
            Phase::DecodeUnd => {
                let projected_successor = self.inflight.has_pending_calls(id);
                // Stage minimum-token and force-finish constraints at the
                // successor's scheduled token position. Device predicates
                // prevent an inactive descendant from accepting that position.
                let projected = if projected_successor {
                    self.inflight.num_pending_calls(id)
                } else {
                    0
                };
                let sampling_state = self.build_token_masks(id, projected, 0);
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
                let scheduled = self.num_scheduled_denoise_steps(id)?;
                let (_, count) = st
                    .denoising
                    .next(scheduled, u32::from(self.denoise_step_burst))?;
                let denoise_step_count = u16::try_from(count).ok()?;
                let conditioning = self.kv_conditioning(id)?;
                let latent = self
                    .pending_output(id, |call| call.latent_output.as_ref())
                    .or(self.running.get(&id)?.denoising.latent())
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
                    .or(self.running.get(&id)?.denoising.latent())
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

                let sampling_state = self.build_token_masks(id, 0, usize::from(is_final_step));
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
                // Submission validated every input encoder's capacity.
                let Ok(physical_bound) = encoder_input.kv_token_capacity(&self.generation_limits)
                else {
                    self.invariant_broken("a queued input encoder has a valid KV capacity");
                    return None;
                };

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
                self.storage.encoder_cache.lookup_product(cache_key)
            } else {
                None
            };
            if let Some(cached_product) = cached {
                let product = self.storage.encoder_cache.acquire(cache_key)?;
                if product != cached_product {
                    return None;
                }
                let Some(state) = self.running.get_mut(&id) else {
                    let _ = self.storage.encoder_cache.release(cache_key, &product);
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

        let sampling_state = self.build_token_masks(id, 0, 0);

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
    ///
    /// `completing_images` counts image completions the planned call itself
    /// finishes, which the final feedback step does. Returns `None` when the
    /// request is not running or nothing needs constraining.
    pub(super) fn build_token_masks(
        &self,
        id: RequestId,
        projected: usize,
        completing_images: usize,
    ) -> Option<SamplingState> {
        let n_generated = self.running.get(&id).map_or(0, |state| {
            state.num_generated_tokens.saturating_add(projected)
        });
        let projected_images_done = self.running.get(&id).map_or(0, |state| {
            state
                .num_generated_images
                .saturating_add(self.scheduled_feedback(id).map_or(0, |(_, images)| images))
                .saturating_add(completing_images)
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
        // Disallow unavailable transitions before sampling. The sampled token,
        // its scores, and any device-resident successor input must agree.
        if (!state.req.generates_images()
            || projected_images_done >= state.req.image.max_images as usize)
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
