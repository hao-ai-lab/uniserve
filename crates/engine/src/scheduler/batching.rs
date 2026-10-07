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
//!
//! A pass's prefill calls, and likewise its decode calls, travel as one
//! numerical call, which workers evaluate with graphs captured at startup; a
//! prefill or decode batch holds at most the calls those graphs hold, as
//! the workers report it (`Scheduler::call_limit`). A context prefill with
//! vision blocks stages several forward rows; a pass's prefill rows, like its
//! canvas rows, stay within the rows a worker stages per call (`PassRows`).

use super::*;
use uniserve_worker_ipc::{ForwardMode, MediaCall, TransferMode, VisionInput};

/// Forward rows a pass may still add to its numerical calls, within the rows
/// a worker stages per call: canvas rows across its token-denoising calls,
/// and context rows across its prefill calls.
#[derive(Debug, Clone, Copy)]
pub(super) struct PassRows {
    pub(super) canvas: usize,
    pub(super) context: usize,
}

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

    /// Returns the most calls one batch of `code` may hold.
    ///
    /// Workers report the rows their captured prefill and decode graphs hold
    /// (`WorkerInfo::max_prefill_calls`, `WorkerInfo::max_decode_calls`); zero
    /// means those calls run eagerly and only the pass's call bound applies,
    /// as it does to every other computation.
    pub(super) fn call_limit(&self, code: CallKind) -> usize {
        let reported = match code {
            CallKind::Forward(ForwardMode::Prefill) => self.info.max_prefill_calls,
            CallKind::Forward(ForwardMode::Decode) => self.info.max_decode_calls,
            _ => 0,
        };
        match reported {
            0 => self.config.max_batch,
            limit => self.config.max_batch.min(limit as usize),
        }
    }

    /// Selects compatible call kinds within one lane's token and sequence budgets.
    ///
    /// Walks `ids` in assembly order until `max_batch` calls are selected or
    /// the token budget is spent. A prefill or decode batch stops growing
    /// at its `call_limit`; requests whose next call is of that computation
    /// then wait for a later pass. Every selected call is registered in flight
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
        // Context calls expand into one forward row per text/image segment.
        // Both staging and the reported prefill graph capacity bound those
        // numerical rows, independently of the number of requests batched.
        let row_staging = (self.info.max_batch_calls.min(self.info.request_slots) as usize).max(1);
        let mut rows_left = PassRows {
            canvas: row_staging,
            context: row_staging.min(self.call_limit(CallKind::Forward(ForwardMode::Prefill))),
        };
        let limits = [
            CallKind::Forward(ForwardMode::Prefill),
            CallKind::Forward(ForwardMode::Decode),
        ]
        .map(|code| (code, self.call_limit(code)));
        let full = |batches: &HashMap<CallKind, ExecutionBatch>, code: CallKind| {
            limits.iter().any(|(bounded, limit)| {
                *bounded == code
                    && batches
                        .get(&code)
                        .is_some_and(|batch| batch.requests.len() >= *limit)
            })
        };
        // Host-lane occupancy, built when the pass meets its first encoder
        // call carrying inline image bytes and charged with each one it
        // selects.
        let mut image_lanes: Option<super::execution::LaneLedger> = None;
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
            if next_type.is_some_and(|code| full(&code_batches, code)) {
                continue;
            }
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
            if let Some(mut call) = self.next_generation_computation(id, call_budget, rows_left) {
                // A planned call joins its computation's batch only while it
                // has room; otherwise it waits for a later pass, as a call
                // without resources does below. A context prefill plans
                // within the pass's context rows.
                let prefill = call.code == CallKind::Forward(ForwardMode::Prefill);
                if full(&code_batches, call.code)
                    || (prefill && generation::prefill_rows(&call) > rows_left.context)
                {
                    if let Some(state) = self.running.get_mut(&id) {
                        state.num_kv_units_sent = state
                            .num_kv_units_sent
                            .saturating_sub(u64::from(call.bounds.max_kv_units));
                    }
                    continue;
                }
                // An encoder call carrying inline image bytes holds a host
                // task on each rank of its component while the worker
                // prepares the image, so it waits here, as a media call
                // does, until those host lanes have room.
                let image_lane =
                    match call.code {
                        CallKind::Media(media_call) if call.input_image.is_some() => {
                            let Some(request_key) = self.running.get(&id).map(|state| {
                                RequestKey::new(self.engine_id, id, state.request_epoch)
                            }) else {
                                continue;
                            };
                            let demand = self.image_lane_demand(request_key, media_call);
                            let ledger = image_lanes.get_or_insert_with(|| self.lane_occupancy());
                            if !ledger.admits(&demand, id, self) {
                                self.record_domain_backpressure(call.code);
                                if let Some(state) = self.running.get_mut(&id) {
                                    state.num_kv_units_sent = state
                                        .num_kv_units_sent
                                        .saturating_sub(u64::from(call.bounds.max_kv_units));
                                }
                                continue;
                            }
                            Some(demand)
                        }
                        _ => None,
                    };
                let planned_us = uniserve_core::now_monotonic_us();

                // Only a call that obtains its storage is charged to the
                // pass. A call that cannot run now leaves the budget to later
                // requests; in particular, requests waiting for the flow-prefix
                // row must not spend the pass before the request that holds
                // the row, which is the one whose steps release it.
                let Some(reserved_buffers) = self.reserve_generation_resources(&call) else {
                    self.record_domain_backpressure(call.code);
                    // Planning a forward advanced `num_kv_units_sent` past its
                    // fresh units; roll it back so the next plan declares them.
                    if let Some(state) = self.running.get_mut(&id) {
                        state.num_kv_units_sent = state
                            .num_kv_units_sent
                            .saturating_sub(u64::from(call.bounds.max_kv_units));
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
                if call.readout.is_some() {
                    let rows = self.running.get(&id).and_then(|state| {
                        let start = self.scheduled_readout_rows(id)?;
                        state.readout_rows_covering(start, call.input_token_ids.len())
                    });
                    rows_left.canvas = rows_left.canvas.saturating_sub(rows.unwrap_or(0));
                } else if call.canvas.is_some() {
                    rows_left.canvas = rows_left.canvas.saturating_sub(1);
                } else if prefill {
                    rows_left.context = rows_left
                        .context
                        .saturating_sub(generation::prefill_rows(&call));
                }

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
                                    canvas: st.req.canvas,
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
                if let (Some(demand), Some(ledger)) = (&image_lane, image_lanes.as_mut()) {
                    ledger.occupy(demand, id);
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
    /// A ready readout pass takes the decode lane first: one pass answers
    /// the readout and releases its KV. Otherwise prefill wins while fewer
    /// than `PREFILL_WINDOW_CREDITS` batches carrying `BatchKind::Prefill`
    /// calls await their results; past that window, ready decode work takes
    /// the pass, and prefill runs only when no decode is ready.
    ///
    /// A generating block's canvas steps are decode work under this window,
    /// not ahead of it: a block's next step is ready again as soon as the
    /// step before it is submitted, so a prompt ordered behind every ready
    /// step would wait until the running blocks stop instead of joining
    /// their canvas batches.
    ///
    /// Prefill still shares a decode-lane pass within the mixed-prefill token
    /// budget. `BatchKind::Media` readiness never selects a lane: with only
    /// such work ready this returns `None`, and `assemble_batch` then applies
    /// no lane filter.
    pub(super) fn select_batch_kind(&self, ids: &[RequestId]) -> Option<BatchKind> {
        let mut readout_ready = false;
        let mut decode_ready = false;
        let mut prefill_ready = false;
        for id in ids.iter().copied() {
            let Some(state) = self.running.get(&id) else {
                continue;
            };
            if state.terminal_intent.is_terminal() {
                continue;
            }
            let Some(call_type) = self.peek_next_call_variant(id) else {
                continue;
            };
            let lane = batch_kind(call_type);
            if lane == BatchKind::Media || !self.can_schedule_next(id) {
                continue;
            }
            match lane {
                BatchKind::Decode if state.req.is_readout() => readout_ready = true,
                BatchKind::Decode => decode_ready = true,
                BatchKind::Prefill => prefill_ready = true,
                BatchKind::Media => {}
            }
        }
        let prefill_window_open = self
            .inflight
            .pending_batches
            .values()
            .filter(|batch| batch.prefill)
            .count()
            < PREFILL_WINDOW_CREDITS;
        if readout_ready {
            Some(BatchKind::Decode)
        } else if prefill_ready && prefill_window_open {
            Some(BatchKind::Prefill)
        } else if decode_ready {
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
    /// Context ingestion (encoders and prefill) comes first; then token decode,
    /// verification and canvas denoising, image decoding, and KV installation;
    /// then denoising and the video and audio media calls; and last latent
    /// preparation, tensor transfers, KV publication, and requests with no
    /// next call.
    pub(super) fn assembly_priority(&self, id: RequestId) -> u8 {
        match self.peek_next_call_variant(id) {
            Some(
                CallKind::Media(MediaCall::TextEncoding)
                | CallKind::Media(MediaCall::VisionEncoding)
                | CallKind::Media(MediaCall::LatentEncoding)
                | CallKind::Forward(ForwardMode::Prefill),
            ) => 0,
            Some(
                CallKind::Forward(
                    ForwardMode::Decode | ForwardMode::Verify | ForwardMode::TokenDenoising,
                )
                | CallKind::Media(MediaCall::ImageDecoding)
                | CallKind::Transfer(TransferMode::KvInstall),
            ) => 1,
            Some(CallKind::Media(
                MediaCall::MediaReading
                | MediaCall::Denoising
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
    /// Otherwise, while the context is ingested, an input-image block the
    /// next context prefill needs is encoded first
    /// (`Scheduler::next_context_encode`); a block whose product the encoder
    /// cache holds still reports its encoder here. Otherwise the request's
    /// phase decides. Returns `None` when the request is not running, waits
    /// on its image-branch reservation, or has no call to issue.
    pub(super) fn peek_next_call_variant(&self, id: RequestId) -> Option<CallKind> {
        let st = self.running.get(&id)?;
        if st.image_reservation_pending {
            return None;
        }
        if let Some(variant) = self.pending_successor_code(id) {
            return Some(variant);
        }
        if st.phase == Phase::Prefill
            && let Some((_, step)) = self.next_context_encode(id)
        {
            return Some(match step {
                ImageIngestStep::VaeEncode => CallKind::Media(MediaCall::LatentEncoding),
                ImageIngestStep::VitEncode => CallKind::Media(MediaCall::VisionEncoding),
            });
        }
        Some(match st.phase {
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
            Phase::Readout | Phase::Canvas => CallKind::Forward(ForwardMode::TokenDenoising),
            Phase::CommitCanvas => CallKind::Forward(ForwardMode::Prefill),
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
        // other successor its completion product unless it follows
        // unconditionally (`follows_unconditionally`). The first call and
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
            } else if self.follows_unconditionally(request_id, call.code) {
                None
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
            // A context prefill's bound counts its prompt tokens and its
            // vision blocks' exact KV lengths.
            CallKind::Forward(ForwardMode::Prefill) => Some(KvLengths {
                visible,
                input: if is_prompt_extend(&call) || consumes_image_features(&call) {
                    call.bounds.max_tokens
                } else {
                    1
                },
            }),
            CallKind::Forward(ForwardMode::Decode) | CallKind::Forward(ForwardMode::Verify) => {
                Some(KvLengths { visible, input: 1 })
            }
            CallKind::Forward(ForwardMode::TokenDenoising)
            | CallKind::Transfer(TransferMode::KvPublish)
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

        // A canvas pass covers the readout rows after those already scheduled;
        // each row's canvas becomes one read-only forward row below.
        let canvas_lengths = if call.readout.is_some() {
            let rows = self.scheduled_readout_rows(request_id).and_then(|start| {
                let state = self.running.get(&request_id)?;
                let count = state.readout_rows_covering(start, call.input_token_ids.len())?;
                Some(
                    state.req.readout[start..start + count]
                        .iter()
                        .map(|row| row.token_ids.len() as u32)
                        .collect::<Vec<_>>(),
                )
            });
            let Some(rows) = rows else {
                self.invariant_broken("a canvas pass covers whole scheduled readout rows");
                return None;
            };
            rows
        } else if call.canvas.is_some() {
            // A generation step denoises the request's resident canvas of
            // `max_tokens` tokens.
            vec![call.bounds.max_tokens]
        } else {
            Vec::new()
        };

        let context_segments = if is_prompt_extend(&call) && !call.vision_inputs.is_empty() {
            let segments = self.context_segments(request_id, &call);
            let Some(segments) = segments.filter(|segments| {
                kv_lengths
                    .as_ref()
                    .is_some_and(|lengths| segments.iter().sum::<u32>() == lengths.input)
            }) else {
                self.invariant_broken(
                    "a context prefill's vision blocks are the request's next vision blocks",
                );
                return None;
            };
            segments
        } else {
            Vec::new()
        };

        let mut call_block_tables = Vec::new();
        let mut call_new_cache_units = Vec::new();
        let mut call_forward = ForwardBatch::default();
        if let Some(lengths) = kv_lengths {
            // Registration above found the request running, and a running
            // token request always has its KV tables.
            let Some(state) = self.running.get_mut(&request_id) else {
                self.invariant_broken("a registered token request is admitted with a KV cache");
                return None;
            };
            let fresh_count = u64::from(call.bounds.max_kv_units);
            let declared = state.num_kv_units_sent;
            let (Some(request_pool_idx), Some(allocation)) =
                (state.request_pool_idx(), state.kv_mut())
            else {
                self.invariant_broken("a registered token request is admitted with a KV cache");
                return None;
            };

            // A table is declared when its held page interval changed since
            // its last declaration (growth, or window retirement in a
            // sliding-window group) or it holds fresh units, which the worker
            // must reset before use. A forward's fresh units are those
            // allocated since the declared watermark (`plan_computation`).
            // Admission (or re-admission) declares every table again, with
            // every allocated page fresh; pages acquired from the prefix
            // cache already hold computed KV and are never fresh.
            if admitted {
                allocation.clear_sent();
            }
            let fresh_since = if admitted {
                Some(0)
            } else {
                (fresh_count > 0).then(|| declared.saturating_sub(fresh_count))
            };
            for table in &mut allocation.tables {
                let fresh = fresh_since
                    .map(|serial| table.fresh_units(serial))
                    .unwrap_or_default();
                if table.changed_since_sent() || !fresh.is_empty() {
                    call_block_tables.push(IpcBlockTable {
                        request_pool_idx,
                        group_id: table.group_id() as u32,
                        start_page: u32::try_from(table.start_page()).unwrap_or(u32::MAX),
                        unit_ids: table.unit_ids(),
                        allocated_tokens: u32::try_from(table.capacity_tokens())
                            .unwrap_or(u32::MAX),
                    });
                    table.mark_sent();
                }
                if !fresh.is_empty() {
                    call_new_cache_units.push(CacheUnitAllocation {
                        request_pool_idx,
                        group_id: table.group_id() as u32,
                        unit_ids: fresh,
                    });
                }
            }
            // A context prefill with vision blocks appends one forward row per
            // segment, in context order: the prompt tokens between blocks, and
            // each block, which the worker evaluates as one attention block.
            // Every other call that appends KV has one row. Each row attends
            // to the visible prefix and the rows before it, and persists its
            // tokens in KV.
            if !context_segments.is_empty() {
                let mut end = lengths.visible;
                for &segment in &context_segments {
                    end += segment;
                    call_forward.push(0, request_pool_idx, end, segment, true);
                }
            } else if lengths.input > 0 {
                call_forward.push(
                    0,
                    request_pool_idx,
                    lengths.visible + lengths.input,
                    lengths.input,
                    true,
                );
            }
            // One read-only row per canvas: it attends to the visible prefix
            // and its own canvas, and persists nothing.
            for canvas in &canvas_lengths {
                call_forward.push(
                    0,
                    request_pool_idx,
                    lengths.visible + canvas,
                    *canvas,
                    false,
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
                if !prefix.diffusion_finalized || !prefix.new_units.is_empty() {
                    for table in prefix.block_tables() {
                        call_block_tables.push(IpcBlockTable {
                            request_pool_idx: prefix.request_pool_idx(),
                            group_id: table.group_id() as u32,
                            start_page: u32::try_from(table.start_page()).unwrap_or(u32::MAX),
                            unit_ids: table.unit_ids(),
                            allocated_tokens: u32::try_from(table.capacity_tokens())
                                .unwrap_or(u32::MAX),
                        });
                    }
                }
                for (group_id, units) in std::mem::take(&mut prefix.new_units) {
                    if !units.is_empty() {
                        call_new_cache_units.push(CacheUnitAllocation {
                            request_pool_idx: prefix.request_pool_idx(),
                            group_id,
                            unit_ids: units,
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

        // An image extension writes its block's KV, and a context prefill
        // with vision blocks exactly its bound: its prompt tokens and blocks.
        let image_kv = if consumes_image_features(&call) || !call.vision_inputs.is_empty() {
            let Some(lengths) = kv_lengths else {
                self.invariant_broken("an image extension declares its KV input range");
                return None;
            };
            Some((
                lengths.visible,
                if consumes_image_features(&call) {
                    num_image_kv_tokens
                } else {
                    Some(lengths.input)
                },
            ))
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
                new_cache_units: call_new_cache_units,
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
            .free_units
            .store(self.storage.free_units(), Ordering::Relaxed);
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

    /// Grows the request's KV tables for a call that reads from KV token
    /// `read_start` and holds `total_tokens` tokens.
    ///
    /// A sliding-window group needs only the pages the read's window and the
    /// call's tokens intersect; every group's units come from one pool.
    /// Returns `false` when the request is not running, the cache cannot
    /// grow, or (after latching engine-fatal) a running request lacks its
    /// allocations or the KV cache.
    pub(super) fn ensure_request_capacity(
        &mut self,
        id: RequestId,
        read_start: usize,
        total_tokens: usize,
    ) -> bool {
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
                read_start.min(u32::MAX as usize) as u32,
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

    /// Builds a computation and records the KV units owned by its next dispatch.
    ///
    /// For a forward call, sets `bounds.max_kv_units` to the units every
    /// group's table gained since the last dispatch and advances
    /// `num_kv_units_sent`; a caller that drops the call must roll that back.
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
                    // The KV owner keeps unit identities. The scheduled record
                    // only needs the number of fresh units for dispatch and drain.
                    let state = self.running.get_mut(&id)?;
                    let Some(allocated) = state
                        .allocations()
                        .map(|allocations| allocations.kv.allocated_units())
                    else {
                        self.invariant_broken("a request planning a forward holds its KV tables");
                        return None;
                    };
                    call.bounds.max_kv_units = allocated
                        .saturating_sub(state.num_kv_units_sent)
                        .min(u64::from(u32::MAX))
                        as u32;
                    state.num_kv_units_sent = allocated;
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
    /// A canvas pass takes the next readout rows that fit both `budget`
    /// tokens and the pass's canvas `rows`, and a context prefill the
    /// segments that fit its context rows. Returns `None` when nothing can be
    /// planned now, including when the KV capacity a call needs, the
    /// image-branch reservation, or room for one whole canvas row is
    /// unavailable.
    pub(super) fn next_generation_computation(
        &mut self,
        id: RequestId,
        budget: usize,
        rows: PassRows,
    ) -> Option<Call> {
        let canvas_rows = rows.canvas;
        let (logical_position, kv_visible_len) = self.scheduled_token_lengths(id)?;
        let context_pending = self.running.get(&id).is_some_and(|st| {
            self.num_scheduled_prompt_tokens(id)
                .is_some_and(|count| count < st.req.prompt_token_ids.len() as u32)
                || self
                    .num_scheduled_images(id)
                    .is_some_and(|count| count < st.req.multimodal_inputs.images.len())
        });
        if context_pending {
            return self.next_context_computation(id, budget, rows.context);
        }
        if self.running.get(&id)?.image_reservation_pending
            && !self.promote_gen_branch_reservation(id)
        {
            return None;
        }
        let phase = self.next_generation_phase(id)?;
        let image_id = self.running.get(&id)?.image_id;
        match phase {
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
                if !self.ensure_request_capacity(id, kv_visible_len as usize, end) {
                    return None;
                }
                let sampling_state = self.build_token_masks(id, 0, 0);

                self.plan_computation(id, |_scheduler, request| {
                    generation::plan_prompt(
                        request,
                        cursor as u32,
                        end as u32,
                        Vec::new(),
                        0,
                        sampling_state,
                    )
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
                if !self.ensure_request_capacity(id, kv_visible_len as usize, capacity_target) {
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
                if !self.ensure_request_capacity(id, kv_visible_len as usize, capacity_target) {
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
                        .map_or(0, |cache| cache.publication_bytes(kv_visible_len)),
                    request,
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
            Phase::IngestState => self.next_context_computation(id, budget, rows.context),
            Phase::Readout => {
                // Whole rows join the pass in report order while they fit its
                // remaining tokens and canvas rows.
                let start = self.scheduled_readout_rows(id)?;
                let st = self.running.get(&id)?;
                let mut end = start;
                let mut tokens = 0usize;
                for row in st.req.readout.get(start..)? {
                    if end - start >= canvas_rows || tokens + row.token_ids.len() > budget {
                        break;
                    }
                    tokens += row.token_ids.len();
                    end += 1;
                }
                if end == start {
                    return None;
                }
                self.plan_computation(id, |_scheduler, request| {
                    generation::plan_readout(request, start..end)
                })
            }
            Phase::Canvas => {
                // One step of the request's block is one read-only canvas row
                // of the pass. The next step may be queued behind the steps
                // still in flight, predicated on the last one's completion:
                // the worker runs it as a no-op once an earlier step stopped
                // the block, so the block's steps run back to back without
                // waiting for each result. A step past the block's step limit
                // is never queued, since the last step always stops it.
                //
                // A block's first step may likewise be queued behind the
                // prefill that completes its context: the prompt's last
                // chunk, or the previous block's commit, which makes the
                // planned step the next block's step zero.
                let st = self.running.get(&id)?;
                let canvas = st.req.canvas.as_ref()?;
                let canvas_length = canvas.canvas_length as usize;
                let pending = self.inflight.num_pending_calls(id);
                if (pending > 0
                    && !self
                        .can_queue_successor(id, CallKind::Forward(ForwardMode::TokenDenoising)))
                    || canvas_rows == 0
                    || canvas_length > budget
                {
                    return None;
                }
                // The steps in flight that the planned step follows are those
                // behind the latest queued prefill: steps queued before a
                // commit are the stopped block's no-ops.
                let queued = self.inflight.pending_calls.get(&id);
                let steps_in_flight = queued.map_or(0, |calls| {
                    calls
                        .iter()
                        .rev()
                        .take_while(|inflight| !is_prompt_extend(&inflight.call))
                        .filter(|inflight| inflight.call.canvas.is_some())
                        .count()
                });
                let commit_in_flight = st.phase == Phase::CommitCanvas
                    && queued.is_some_and(|calls| {
                        calls
                            .iter()
                            .any(|inflight| is_prompt_extend(&inflight.call))
                    });
                let (block, first_step) = if commit_in_flight {
                    (st.canvas_block.saturating_add(1), 0)
                } else {
                    (st.canvas_block, st.canvas_step)
                };
                let step =
                    first_step.saturating_add(u32::try_from(steps_in_flight).unwrap_or(u32::MAX));
                if step >= canvas.max_steps {
                    return None;
                }
                self.plan_computation(id, |_scheduler, request| {
                    generation::plan_canvas_step(request, block, step)
                })
            }
            Phase::CommitCanvas => {
                // The stopped block's tokens extend the context as one causal
                // prefill row. The commit is planned once the host accepts
                // the step that stopped the block, queued without a predicate
                // behind the block's steps still in flight, which are no-ops
                // (`follows_unconditionally`), so the worker launches it
                // while they run.
                let st = self.running.get(&id)?;
                let tokens = st.canvas_commit.clone();
                if (self.inflight.has_pending_calls(id)
                    && !self.can_queue_successor(id, CallKind::Forward(ForwardMode::Prefill)))
                    || tokens.len() > budget
                {
                    return None;
                }
                let end = (kv_visible_len as usize).saturating_add(tokens.len());
                if !self.ensure_request_capacity(id, kv_visible_len as usize, end) {
                    return None;
                }
                self.plan_computation(id, |_scheduler, request| {
                    generation::plan_canvas_commit(request, &tokens)
                })
            }
        }
    }

    /// Plans the next context-ingest call within `budget` tokens.
    ///
    /// The context is the prompt with the blocks of its input images at their
    /// positions: each encoder of an image writes one block, in encoder order.
    /// A vision block is written by a context prefill together with the prompt
    /// tokens around it, as one attention block of that call; a latent block
    /// is written by an image extension of its own, which ends the preceding
    /// context prefill. Every block's encoder product exists before the call
    /// that writes it: the next block the context needs is encoded first
    /// (`next_context_encode`), or pinned from the encoder cache without a
    /// call.
    ///
    /// A context prefill holds prompt tokens and whole vision blocks, within
    /// the pass's `budget` and the long-prefill threshold; a block that would
    /// not fit waits for the next chunk, which it starts, and a chunk that
    /// starts with a block may exceed the threshold within the budget. Its
    /// forward rows (`generation::prefill_rows`) stay within `context_rows`.
    /// Returns `None` when nothing can be planned now, including when the
    /// next block does not fit this pass or the request's KV capacity is
    /// unavailable.
    pub(super) fn next_context_computation(
        &mut self,
        id: RequestId,
        budget: usize,
        context_rows: usize,
    ) -> Option<Call> {
        let cursor = self.num_scheduled_prompt_tokens(id)? as usize;
        let (_, kv_visible_len) = self.scheduled_token_lengths(id)?;

        // A latent block's encoder product awaits its own extension.
        if self.running.get(&id)?.phase == Phase::IngestState {
            let state = self.running.get(&id)?;
            let feature = state.input_image_features.clone()?;
            let (_, encoder_input) = state.input_block(state.block_cursor())?;
            let encoder_input = *encoder_input;
            // Submission validated every input encoder's capacity.
            let Ok(physical_bound) = encoder_input.kv_token_capacity(&self.generation_limits)
            else {
                self.invariant_broken("a queued input encoder has a valid KV capacity");
                return None;
            };
            if !self.ensure_request_capacity(
                id,
                kv_visible_len as usize,
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

        // Encode the next block the context needs. A product the encoder
        // cache holds is pinned for the request instead, without a call.
        while let Some((block, step)) = self.next_context_encode(id) {
            let state = self.running.get(&id)?;
            let (image, _) = state.input_block(block)?;
            let cache_key = encoder_cache_key(image.hash, block.encoder, step);
            let image_b64 = image.b64.clone();
            let cached = if state.req.cache.read {
                self.storage.encoder_cache.lookup_product(cache_key)
            } else {
                None
            };
            let Some(cached_product) = cached else {
                // A cache miss schedules the encoder. Nothing else of the
                // request is planned while it is in flight, so its completion
                // finds the same block here.
                return self.plan_computation(id, |scheduler, request| {
                    generation::plan_encode(
                        &scheduler.generation_limits,
                        request,
                        step,
                        image_b64,
                        None,
                        false,
                    )
                });
            };
            let product = self.storage.encoder_cache.acquire(cache_key)?;
            if product != cached_product {
                return None;
            }
            // If request ownership disappeared during acquisition, release
            // the pin.
            let Some(state) = self.running.get_mut(&id) else {
                let _ = self.storage.encoder_cache.release(cache_key, &product);
                return None;
            };
            state.encoder_cache_pins.push(EncoderCachePin {
                key: cache_key,
                product: product.clone(),
            });
            if step == ImageIngestStep::VaeEncode {
                state.input_image_features = Some(product);
                state.phase = Phase::IngestState;
                return self.next_context_computation(id, budget, context_rows);
            }
            state.context_features.push(product);
        }

        // Select the next chunk: prompt tokens up to the budget, and each
        // encoded vision block they reach whole. A latent block or a block
        // not yet encoded ends the chunk.
        let limit = budget.max(1).min(self.config.long_prefill_threshold);
        let (end, blocks, block_tokens) = {
            let state = self.running.get(&id)?;
            let prompt_len = state.req.prompt_token_ids.len();
            let mut end = cursor;
            let mut used = 0usize;
            let mut rows = 0usize;
            let mut blocks = Vec::new();
            let mut block_tokens = 0u32;
            let mut block = self.scheduled_context(id)?.blocks;
            // In-flight context prefills hold the leading features.
            let in_flight = state.blocks_between(state.block_cursor(), block);
            loop {
                let next = state.input_block(block);
                let text_stop = next.map_or(prompt_len, |(image, _)| image.position as usize);
                // A run of prompt tokens is one forward row, as is a block.
                let room = if rows < context_rows {
                    limit.saturating_sub(used)
                } else {
                    0
                };
                let taken = text_stop.saturating_sub(end).min(room);
                if taken > 0 {
                    rows += 1;
                }
                end += taken;
                used += taken;
                let Some((_, input)) = next else {
                    break;
                };
                let Some(feature) = state.context_features.get(in_flight + blocks.len()) else {
                    break;
                };
                if end < text_stop
                    || input.encoder != ImageIngestStep::VitEncode
                    || rows >= context_rows
                {
                    break;
                }
                // Submission validated every vision block's exact KV length.
                let Some(tokens) = input.num_kv_tokens else {
                    self.invariant_broken("an input vision block declares its exact KV length");
                    return None;
                };
                // A block joins only whole: one that would overflow the chunk
                // starts the next one, where it may exceed the threshold
                // within the pass's budget.
                let fits =
                    used + tokens as usize <= limit || (used == 0 && tokens as usize <= budget);
                if !fits {
                    break;
                }
                blocks.push(VisionInput {
                    offset: u32::try_from(end - cursor).unwrap_or(u32::MAX),
                    feature: feature.clone(),
                });
                rows += 1;
                used += tokens as usize;
                block_tokens = block_tokens.saturating_add(tokens);
                block = state.next_block(block)?;
            }
            (end, blocks, block_tokens)
        };
        if end == cursor && blocks.is_empty() {
            return None;
        }

        let written = end - cursor + block_tokens as usize;
        if !self.ensure_request_capacity(
            id,
            kv_visible_len as usize,
            kv_visible_len as usize + written,
        ) {
            return None;
        }

        let sampling_state = self.build_token_masks(id, 0, 0);
        self.plan_computation(id, |_scheduler, request| {
            generation::plan_prompt(
                request,
                cursor as u32,
                end as u32,
                blocks,
                block_tokens,
                sampling_state,
            )
        })
    }

    /// Returns the next input-image block to encode before the context can
    /// advance, with its encoder, or `None` when the next context call needs
    /// no encode.
    ///
    /// Walks the blocks after the accepted block cursor, whose leading
    /// products `context_features` holds: the latent block at the scheduled
    /// prompt cursor is encoded for its own extension; vision blocks are
    /// encoded in order ahead of the context prefill that writes them, while
    /// their image lies within the long-prefill threshold of the scheduled
    /// prompt cursor. The walk stops at a latent block past that cursor,
    /// which the chunk before it does not reach. A request plans nothing
    /// while its encode is in flight, so the encode's completion identifies
    /// its block by the same walk.
    pub(super) fn next_context_encode(
        &self,
        id: RequestId,
    ) -> Option<(BlockCursor, ImageIngestStep)> {
        let state = self.running.get(&id)?;
        if state.phase == Phase::IngestState {
            return None;
        }
        let cursor = self.num_scheduled_prompt_tokens(id)? as usize;
        let reach = cursor.saturating_add(self.config.long_prefill_threshold);
        let mut block = state.block_cursor();
        let mut encoded = 0;
        while let Some((image, input)) = state.input_block(block) {
            let position = image.position as usize;
            if input.encoder == ImageIngestStep::VaeEncode {
                return (position == cursor).then_some((block, input.encoder));
            }
            if position > cursor && position >= reach {
                return None;
            }
            if encoded == state.context_features.len() {
                return Some((block, input.encoder));
            }
            encoded += 1;
            block = state.next_block(block)?;
        }
        None
    }

    /// Returns the forward-row lengths of a context prefill `call` with vision
    /// blocks, in context order: the prompt tokens before, between and after
    /// its blocks, and each block's exact KV length. The blocks are the
    /// request's next ones after its scheduled block cursor.
    fn context_segments(&self, id: RequestId, call: &Call) -> Option<Vec<u32>> {
        let state = self.running.get(&id)?;
        let mut block = self.scheduled_context(id)?.blocks;
        let mut segments = Vec::with_capacity(2 * call.vision_inputs.len() + 1);
        let mut text_start = 0u32;
        for input in &call.vision_inputs {
            if input.offset < text_start {
                return None;
            }
            if input.offset > text_start {
                segments.push(input.offset - text_start);
                text_start = input.offset;
            }
            let (_, encoder) = state.input_block(block)?;
            segments.push(encoder.num_kv_tokens?);
            block = state.next_block(block)?;
        }
        let text = u32::try_from(call.input_token_ids.len()).ok()?;
        if text > text_start {
            segments.push(text - text_start);
        }
        Some(segments)
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
