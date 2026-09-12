//! Batch submission, completion application, and allocation reclamation.
//!
//! The loop keeps executor progress non-blocking until work is outstanding, then
//! parks with a bounded liveness deadline. Completion application validates
//! request and operation identities before mutating runtime state.

use super::*;

impl EngineLoop {
    /// Enumerates loaded entries that can execute the requested operation.
    pub(super) fn worker_candidates(
        &self,
        kind: OpCode,
    ) -> impl Iterator<Item = (&crate::WorkerId, &str, &WorkerInfo)> {
        let entry = match kind {
            OpCode::EncoderText => "text_encoder",
            OpCode::DiffusionPrepare => "denoiser",
            OpCode::DiffusionStep => "denoiser",
            OpCode::MediaAppend => "output",
            OpCode::DiffusionFinalize => "denoiser",
            OpCode::DiffusionDecode => "video_decoder",
            OpCode::EncoderVision => "vision_encoder",
            OpCode::EncoderLatent => "latent_encoder",
            _ => "model",
        };
        self.entry_candidates(kind, entry)
    }

    pub(super) fn entry_candidates<'a>(
        &'a self,
        kind: OpCode,
        entry: &'a str,
    ) -> impl Iterator<Item = (&'a crate::WorkerId, &'a str, &'a WorkerInfo)> {
        self.executor
            .info()
            .workers
            .iter()
            .filter_map(move |(id, info)| {
                if !info.supported_ops.contains(&kind) {
                    return None;
                }
                let bound_entry = if info.components.iter().any(|binding| binding.name == entry) {
                    entry
                } else if info.components.is_empty()
                    || info
                        .components
                        .iter()
                        .any(|binding| binding.name == "model")
                {
                    "model"
                } else {
                    return None;
                };
                Some((id, bound_entry, info))
            })
    }

    /// Uses the static entry owner when it has destination capacity.
    pub(super) fn worker_target(
        &self,
        _request: RequestKey,
        kind: OpCode,
    ) -> Option<(&crate::WorkerId, &str)> {
        let (id, bound_entry, _) = self.worker_candidates(kind).next()?;
        self.executor.has_capacity(id).then_some((id, bound_entry))
    }

    /// Binds planned work to its configured entry and records request residency.
    pub(super) fn select_worker(&mut self, operation: &Operation) -> (crate::WorkerId, String) {
        let (id, bound_entry) = if operation.entry == "model" {
            self.worker_target(operation.request_key, operation.kind())
        } else {
            self.entry_candidates(operation.kind(), &operation.entry)
                .find(|(id, _, _)| self.executor.has_capacity(id))
                .map(|(id, entry, _)| (id, entry))
        }
        .expect("planned operation retains an executable entry");
        let target = (id.clone(), bound_entry.to_owned());
        self.worker_affinity
            .insert((operation.request_key, target.1.clone()), target.0.clone());
        target
    }

    /// Includes lifecycle commands in destination ordering, even when a batch has no compute.
    fn batch_workers(&self, batch: &Batch) -> HashSet<crate::WorkerId> {
        let mut targets = batch
            .ops
            .iter()
            .map(|op| op.target.0.clone())
            .collect::<HashSet<_>>();
        for command in &batch.commands {
            targets.extend(
                self.worker_affinity
                    .iter()
                    .filter_map(|((request, _), worker)| {
                        (*request == command.request_key()).then_some(worker.clone())
                    }),
            );
        }
        targets
    }

    /// Retains destination order when earlier work is waiting for local capacity.
    pub(super) fn submit_bound_batch(&mut self, batch: Batch) -> Result<(), ExecutorSubmitError> {
        let targets = self.batch_workers(&batch);
        if self
            .pending_submissions
            .iter()
            .any(|pending| !self.batch_workers(pending).is_disjoint(&targets))
        {
            return Err(ExecutorSubmitError::WouldBlock(batch));
        }
        self.executor.submit(batch)
    }

    /// Releases the request latent.
    fn free_request_latent(&mut self, id: RequestId) {
        let allocation = self
            .running
            .get_mut(&id)
            .and_then(|state| state.allocations_mut().latent.take());
        if let Some(allocation) = allocation {
            self.memory.free(allocation);
        }
    }
    /// Advances scheduler and executor work, blocking only for an outstanding result.
    ///
    /// Returns whether the loop made progress or handled an executor outcome.
    pub fn step(&mut self) -> bool {
        let progressed = self.step_nonblocking();
        if progressed || self.inflight.batch_started.is_empty() {
            return progressed;
        }
        match self.executor.poll(Duration::from_secs(300)) {
            Ok(Some(result)) => {
                self.apply_result(result);
                self.refill_executor();
                self.publish_cache_stats();
            }
            Ok(None) => return false,
            Err(error) => {
                self.on_executor_error(error);
                self.publish_cache_stats();
            }
        }
        true
    }

    /// Advances one nonblocking schedule-ahead tick for the owner-thread reactor. It
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
            .store(self.inflight.batch_started.len(), Ordering::Relaxed);
        self.publish_cache_stats();
        progressed
    }

    /// Drains ready work, admits requests, and fills every available executor slot.
    pub(super) fn refill_executor(&mut self) -> bool {
        let mut progressed = false;
        self.worker_affinity.retain(|(request, _), _| {
            let id = request.request_id;
            self.running.contains_key(&id)
                || self.running_media.contains_key(&id)
                || self.retiring_requests.contains_key(&id)
        });

        let pending_count = self.pending_submissions.len();
        let mut blocked_workers = HashSet::new();
        for _ in 0..pending_count {
            let batch = self
                .pending_submissions
                .pop_front()
                .expect("pending count is fixed");
            if !self.batch_workers(&batch).is_disjoint(&blocked_workers) {
                self.pending_submissions.push_back(batch);
                continue;
            }
            match self.executor.submit(batch) {
                Ok(()) => progressed = true,
                Err(ExecutorSubmitError::WouldBlock(batch)) => {
                    blocked_workers.extend(self.batch_workers(&batch));
                    self.pending_submissions.push_back(batch);
                }
                Err(ExecutorSubmitError::Failed(error)) => {
                    self.on_executor_error(error);
                    return true;
                }
            }
        }

        // 2. reap cancellations before assembling.
        self.reap_cancellations();

        // 3. Admit request/resource residency even
        // while every execution slot is occupied. This lets the next batch see
        // the complete resident cohort instead of admitting only when a slot
        // happens to open.
        self.admit();
        self.admit_media();

        // 4. submit as many batches as pipeline capacity allows.
        loop {
            self.admit();
            self.admit_media();
            if self.inflight.batch_started.len() >= self.info.queue_depth.max(1) as usize {
                break;
            }
            if self.scheduler.prefer_media && self.submit_media_batch() {
                self.scheduler.prefer_media = false;
                progressed = true;
                continue;
            }
            let (new_reqs, ops) = self.assemble();
            if ops.is_empty() {
                if self.submit_media_batch() {
                    self.scheduler.prefer_media = false;
                    progressed = true;
                    continue;
                }
                break;
            }
            // A prompt batch carries its own state-mutating commands plus a
            // foreign terminal finish only when no earlier command for that
            // request remains queued. Other foreign commands stay pending so
            // the prompt request set remains disjoint from in-flight decode
            // work without reordering a request's command stream.
            let prompt_batch = ops
                .iter()
                .all(|op| batch_kind(op.operation_variant) == BatchKind::Prefill);
            let commands: Vec<BatchCommand> = if prompt_batch {
                let requests: HashSet<RequestId> = ops.iter().map(|op| op.request_id).collect();
                self.take_commands(|command| {
                    requests.contains(&command.request_key().request_id)
                        || matches!(
                            command,
                            BatchCommand::Finish { .. } | BatchCommand::Retire { .. }
                        )
                })
            } else {
                self.take_commands(|_| true)
            };
            if !self.submit_batch(new_reqs, ops, commands) {
                break;
            }
            self.scheduler.prefer_media = true;
            progressed = true;
        }
        while self.pending_submissions.is_empty()
            && !self.pending_commands.is_empty()
            && self.inflight.batch_started.len() < self.info.queue_depth.max(1) as usize
        {
            let commands = self.take_commands(|_| true);
            if commands.is_empty() {
                break;
            }
            if !self.submit_batch(Vec::new(), Vec::new(), commands) {
                break;
            }
            progressed = true;
        }

        progressed
    }

    /// Allocates a request-scoped product reference for a terminal media result.
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
            dtype: DType::U8,
            shape_bound: ShapeBound::default(),
            point_range: PointRange {
                base_point: 1,
                max_points: 1,
            },
        }
    }

    /// Select a ready branch with capacity at its actual static computation entry.
    fn next_media_quantum(&self, state: &MediaFlowState) -> Option<MediaQuantum> {
        use uniserve_worker_ipc::MediaStageRole;

        let plan = self.info.media_plan.as_ref()?;
        let cursor = state.projected;
        let geometry = state.request.geometry;
        let mut ready = Vec::new();
        let produced = |product: &ProductRef| {
            !self
                .inflight
                .operations
                .get(&state.request.request_id)
                .is_some_and(|ops| {
                    ops.iter()
                        .any(|op| op.operation.op_id == product.producer_op_id)
                })
        };
        for stage in &plan.stages {
            if stage.dependencies.iter().any(|dependency| {
                plan.stage(dependency).is_none_or(|dependency| {
                    !cursor.completed(dependency, geometry.denoise_steps, geometry.video_units)
                })
            }) {
                continue;
            }
            match stage.role() {
                Some(MediaStageRole::Encode) if !cursor.encoded => ready.push(MediaQuantum::Encode),
                Some(MediaStageRole::Prepare) if !cursor.prepared => {
                    ready.push(MediaQuantum::Prepare)
                }
                Some(MediaStageRole::Denoise) if cursor.denoise_step < stage.count => {
                    ready.push(MediaQuantum::Denoise {
                        step: cursor.denoise_step,
                    });
                }
                Some(MediaStageRole::VideoDecode)
                    if cursor.video_decoded < geometry.video_units =>
                {
                    let (_, entry, info) = self
                        .entry_candidates(stage.operation, &stage.entry)
                        .next()?;
                    let component = info
                        .components
                        .iter()
                        .find(|component| component.name == entry)?;
                    let width = u32::try_from(component.config.ranks.len()).ok()?;
                    ready.push(MediaQuantum::Decode {
                        track: MediaTrack::Video,
                        cursor: cursor.video_decoded,
                        max_units: width.min(geometry.video_units - cursor.video_decoded),
                    });
                }
                Some(MediaStageRole::AudioDecode) if !cursor.audio_decoded => {
                    ready.push(MediaQuantum::Decode {
                        track: MediaTrack::Audio,
                        cursor: 0,
                        max_units: 1,
                    })
                }
                Some(MediaStageRole::VideoAppend) => {
                    if let Some((count, product)) = state.video_segments.get(&cursor.video_written)
                        && produced(product)
                    {
                        ready.push(MediaQuantum::Append {
                            track: MediaTrack::Video,
                            cursor: cursor.video_written,
                            max_units: *count,
                        });
                    }
                }
                Some(MediaStageRole::AudioAppend)
                    if !cursor.audio_written && state.audio.as_ref().is_some_and(produced) =>
                {
                    ready.push(MediaQuantum::Append {
                        track: MediaTrack::Audio,
                        cursor: 0,
                        max_units: 1,
                    });
                }
                Some(MediaStageRole::Finalize)
                    if !cursor.finalized
                        && stage.dependencies.iter().all(|dependency| {
                            plan.stage(dependency).is_some_and(|dependency| {
                                state.committed.completed(
                                    dependency,
                                    geometry.denoise_steps,
                                    geometry.video_units,
                                )
                            })
                        }) =>
                {
                    ready.push(MediaQuantum::Finalize);
                }
                _ => {}
            }
        }
        // Consume completed decoder products before starting more decode work.
        ready.sort_by_key(|quantum| !matches!(quantum, MediaQuantum::Append { .. }));
        ready.into_iter().find(|quantum| {
            quantum.target(plan).is_some_and(|(kind, entry)| {
                self.entry_candidates(kind, entry)
                    .any(|(worker, _, _)| self.executor.has_capacity(worker))
            })
        })
    }

    /// Select eligible media requests and submit one bounded batch. Independent
    /// audio and video branches carry Tensor edges, without a state predecessor.
    pub(super) fn submit_media_batch(&mut self) -> bool {
        let max_unresolved =
            usize::try_from(self.info.max_unresolved_ops.max(1)).unwrap_or(usize::MAX);
        let mut candidates = self
            .scheduler
            .running_order
            .iter()
            .enumerate()
            .filter_map(|(index, id)| {
                let state = self.media_state(*id)?;
                let inflight = self.inflight.len(*id);
                if state.terminal_intent.is_terminal()
                    || state.admission_state == DiffusionRequestParamsState::InFlight
                    || inflight >= max_unresolved
                {
                    return None;
                }
                self.next_media_quantum(state)
                    .map(|quantum| (inflight, index, *id, quantum))
            })
            .collect::<Vec<_>>();
        candidates.sort_unstable_by_key(|(inflight, index, ..)| (*inflight > 0, *index));
        candidates.truncate(self.scheduler.config.max_batch);
        if candidates.is_empty() {
            return false;
        }

        let batch_id = self.inflight.next_batch_id();
        let submit_at = Instant::now();
        let candidate_requests = candidates
            .iter()
            .map(|(_, _, id, _)| *id)
            .collect::<HashSet<_>>();
        let commands = self.take_commands(|command| {
            candidate_requests.contains(&command.request_key().request_id)
                || matches!(
                    command,
                    BatchCommand::Finish { .. } | BatchCommand::Retire { .. }
                )
        });
        let mut admissions = Vec::new();
        let mut logical_ops = Vec::with_capacity(candidates.len());
        let mut inline = Vec::new();
        for (_, _, id, quantum) in candidates {
            let op_id = if quantum == MediaQuantum::Encode {
                self.media_state(id)
                    .expect("media state exists")
                    .conditioning
                    .producer_op_id
            } else {
                let op = OpId(self.next_op_id.max(1));
                self.next_op_id = op.0.saturating_add(1);
                op
            };
            let plan = self
                .info
                .media_plan
                .as_ref()
                .expect("admitted media retains its model execution plan");
            let (work, entry) = quantum
                .target(plan)
                .expect("media quantum belongs to its validated model plan");
            let entry = entry.to_owned();
            let state = self.media_state(id).expect("media candidate exists");
            let request_key = state.admission.request_key;
            let stateful = work.advances_state();
            let last_step = matches!(quantum, MediaQuantum::Denoise { step } if step + 1 == state.request.geometry.denoise_steps);
            let predicate = if stateful {
                self.inflight
                    .operations
                    .get(&id)
                    .and_then(|ops| {
                        ops.iter()
                            .find(|op| op.operation.op_id == state.projected_parent.op_id)
                    })
                    .and_then(|op| {
                        op.operation
                            .outputs()
                            .iter()
                            .find(|output| output.kind == ProductKind::Completion)
                    })
                    .cloned()
            } else {
                None
            };
            let input_from = plan
                .stage_by_role(quantum.stage_role())
                .and_then(|stage| stage.input_from.as_deref());
            let input_role = input_from
                .and_then(|stage| plan.stage(stage))
                .and_then(uniserve_worker_ipc::MediaPlanStage::role);
            let inputs = match (input_role, quantum) {
                (Some(uniserve_worker_ipc::MediaStageRole::Encode), _) => {
                    vec![state.conditioning.clone()]
                }
                (
                    Some(uniserve_worker_ipc::MediaStageRole::Denoise),
                    MediaQuantum::Decode { track, .. },
                ) => {
                    vec![state.latents[usize::from(track == MediaTrack::Audio)].clone()]
                }
                (
                    Some(uniserve_worker_ipc::MediaStageRole::VideoDecode),
                    MediaQuantum::Append {
                        track: MediaTrack::Video,
                        cursor,
                        ..
                    },
                ) => vec![state.video_segments[&cursor].1.clone()],
                (
                    Some(uniserve_worker_ipc::MediaStageRole::AudioDecode),
                    MediaQuantum::Append { .. },
                ) => {
                    vec![state.audio.as_ref().expect("audio is ready").clone()]
                }
                (None, MediaQuantum::Encode) => state
                    .admission
                    .diffusion
                    .as_ref()
                    .expect("media request parameters")
                    .references
                    .iter()
                    .filter_map(|reference| reference.pixels.clone())
                    .collect(),
                (None, _) => Vec::new(),
                _ => unreachable!("validated media plan input disagrees with its stage role"),
            };
            if quantum == MediaQuantum::Encode {
                if let Some(image) = &state.request.image_reference {
                    inline.push(ProductPayload {
                        product: inputs[0].clone(),
                        value: uniserve_worker_ipc::InlineValue::Bytes(image.pixels.clone()),
                    });
                }
            }
            let mut buffers = Vec::new();
            let mut outputs = Vec::new();
            if quantum == MediaQuantum::Encode
                || last_step
                || matches!(quantum, MediaQuantum::Decode { .. })
            {
                let count = if last_step { 2 } else { 1 };
                for index in 0..count {
                    let reserved = &state.allocations.tensors[&(entry.to_owned(), index)];
                    let mut shape_bound = reserved.shape_bound.clone();
                    let start = if let MediaQuantum::Decode {
                        track: MediaTrack::Video,
                        cursor,
                        max_units,
                    } = quantum
                    {
                        shape_bound.dims[0] = DimBound::Static(max_units);
                        cursor
                    } else {
                        0
                    };
                    let product = ProductRef {
                        request_key,
                        producer_op_id: op_id,
                        output_index: index as u16,
                        generation: 1,
                        kind: ProductKind::Tensor,
                        storage_class: StorageClass::DeviceTensor,
                        dtype: reserved.dtype,
                        shape_bound,
                        point_range: PointRange::default(),
                    };
                    buffers.push(reserved.bind(&product, start));
                    outputs.push(product);
                }
            }
            let latent = if stateful {
                let (start_step, step_count) = if let MediaQuantum::Denoise { step } = quantum {
                    (step, 1)
                } else {
                    (0, 0)
                };
                Some(LatentParams {
                    request_key,
                    op_id,
                    page_table: Vec::new(),
                    latent_units: 0,
                    height: 768,
                    width: 1344,
                    start_step,
                    step_count,
                })
            } else {
                None
            };
            let parent = stateful.then(|| state.projected_parent.clone());
            if outputs.is_empty() && stateful {
                let completion = self.media_completion_product(request_key, op_id);
                outputs.push(completion);
            }
            let operation = Operation {
                request_key,
                op_id,
                parent,
                entry: entry.to_owned(),
                payload: OpPayload::new(
                    work,
                    Bounds {
                        max_points: u32::from(stateful),
                        ..Bounds::default()
                    },
                    inputs,
                    outputs,
                    predicate,
                    None,
                    0,
                ),
            }
            .sealed();
            let decode = match quantum {
                MediaQuantum::Decode {
                    track,
                    cursor,
                    max_units,
                }
                | MediaQuantum::Append {
                    track,
                    cursor,
                    max_units,
                } => Some(DecodeRange {
                    request_key,
                    op_id,
                    track,
                    cursor,
                    max_units,
                }),
                _ => None,
            };
            let state = self.media_state_mut(id).expect("media candidate exists");
            if state.admission_state == DiffusionRequestParamsState::Unsubmitted {
                admissions.push(state.admission.clone());
                state.admission_state = DiffusionRequestParamsState::InFlight;
            }
            state.projected = advance_media_cursor(state.projected, quantum);
            if stateful {
                state.projected_parent = Checkpoint {
                    op_id,
                    point: CheckpointPoint::DeviceSelected,
                };
            }
            if last_step {
                state.latents = operation.outputs().to_vec();
            }
            match quantum {
                MediaQuantum::Decode {
                    track: MediaTrack::Video,
                    cursor,
                    max_units,
                } => {
                    state
                        .video_segments
                        .insert(cursor, (max_units, operation.outputs()[0].clone()));
                }
                MediaQuantum::Decode {
                    track: MediaTrack::Audio,
                    ..
                } => state.audio = Some(operation.outputs()[0].clone()),
                _ => {}
            }
            let target = self.select_worker(&operation);
            self.register_media_inflight(operation.clone(), quantum, submit_at);
            logical_ops.push(LogicalOp {
                block_tables: Vec::new(),
                new_cache_pages: Vec::new(),
                forward_rows: Vec::new(),
                latent,
                decode,
                buffers,
                ..LogicalOp::new(operation, target)
            });
        }
        self.inflight.batch_started.insert(batch_id, submit_at);
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
        let batch = Batch::new(batch_id, logical_ops, batch_commands, inline);
        if !commands.is_empty() {
            self.inflight.command_batches.insert(batch_id, commands);
        }
        match self.submit_bound_batch(batch) {
            Ok(()) => true,
            Err(ExecutorSubmitError::WouldBlock(batch)) => {
                self.pending_submissions.push_back(batch);
                true
            }
            Err(ExecutorSubmitError::Failed(error)) => {
                tracing::error!(batch_id, "media submission failed: {error:#}");
                self.inflight.batch_started.remove(&batch_id);
                self.inflight.batch_operations.remove(&batch_id);
                self.inflight.batch_group_worker_exec_us.remove(&batch_id);
                self.inflight.command_batches.remove(&batch_id);
                if error.downcast_ref::<WorkerFailure>().is_some() {
                    self.on_executor_error(error);
                } else {
                    self.fatal = true;
                    self.fail_all_running(&error.to_string());
                }
                false
            }
        }
    }

    /// Publishes cache state and drained cache events to scheduler counters.
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
            .store(self.memory.free_blocks(), Ordering::Relaxed);
        self.stats
            .general
            .in_flight
            .store(self.inflight.batch_started.len(), Ordering::Relaxed);
        // drain the manager's event ring so it doesn't grow unbounded; the
        // counters below already aggregate it, but draining keeps memory bounded.
        if let Some(kv) = self.memory.cache.as_ref() {
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
        self.stats
            .encoder
            .cache_queries
            .store(self.memory.encoder_cache.stats.queries, Ordering::Relaxed);
        self.stats
            .encoder
            .cache_hits
            .store(self.memory.encoder_cache.stats.hits, Ordering::Relaxed);
        self.stats
            .encoder
            .cached
            .store(self.memory.encoder_cache.len(), Ordering::Relaxed);
    }

    /// Resolves at most one ready result so assembly can refill the freed slot
    /// before another completion is consumed.
    pub(super) fn poll_one_result(&mut self) -> bool {
        let _span = tracing::trace_span!("scheduler.poll_one_result").entered();
        match self.executor.poll(Duration::ZERO) {
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

    /// Returns whether a flow prefix can be scheduled now.
    pub(super) fn flow_prefix_is_schedulable(&self, id: RequestId) -> bool {
        let prefix_is_pending = self
            .running
            .get(&id)
            .and_then(|state| state.flow_prefix.as_ref())
            .is_some_and(|prefix| !prefix.diffusion_finalized);
        !prefix_is_pending
            || !self
                .inflight
                .operations
                .get(&id)
                .into_iter()
                .flatten()
                .any(|op| op.operation.kind() == OpCode::DiffusionStep)
    }

    /// Projects committed request state through every queued state-advancing operation.
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

    /// Returns a fixed cache-version provider.
    pub(super) fn fixed_version(&self, id: RequestId) -> Option<Checkpoint> {
        let state = self.running.get(&id)?;
        state.cursor.resources.worker_registered.then_some(())?;
        Some(Checkpoint {
            op_id: OpId(state.resolved_producer_op_id),
            point: CheckpointPoint::Fixed(state.version as u32),
        })
    }

    /// Returns the checkpoint that the next state-advancing operation must consume.
    pub(super) fn projected_parent(&self, id: RequestId) -> Option<Checkpoint> {
        let operation = self
            .inflight
            .operations
            .get(&id)?
            .iter()
            .rev()
            .find(|inflight| inflight.operation.advances_state())
            .map(|inflight| &inflight.operation);
        let Some(operation) = operation else {
            return self.fixed_version(id);
        };
        let selected_point = (operation.bounds().max_points > 1)
            .then(|| {
                operation
                    .outputs()
                    .iter()
                    .find(|output| output.kind == ProductKind::SelectedPoint)
                    .cloned()
            })
            .flatten();
        if operation.bounds().max_points > 1 && selected_point.is_none() {
            return None;
        }
        Some(Checkpoint {
            op_id: operation.op_id,
            point: CheckpointPoint::DeviceSelected,
        })
    }

    /// Returns the public output limit for an operation.
    pub(super) fn public_limit_for(&self, id: RequestId, apply: &RuntimeApply) -> u64 {
        self.running.get(&id).map_or(0, |state| {
            state.public_event_limit.max(
                state
                    .output
                    .event_seq
                    .saturating_add(apply.output_event_bound as u64),
            )
        })
    }

    /// Queues a semantic checkpoint commit behind the frontend's visible-output acknowledgement.
    pub(super) fn queue_commit(
        &mut self,
        id: RequestId,
        expected_parent: Checkpoint,
        selected: Checkpoint,
        public_event_limit: u64,
    ) {
        let Some(state) = self.running.get_mut(&id) else {
            return;
        };
        state.control_seq = state.control_seq.saturating_add(1);
        state.committed_version = match &selected.point {
            CheckpointPoint::Fixed(point) => u64::from(*point),
            CheckpointPoint::DeviceSelected => state.committed_version,
        };
        state.committed_producer_op_id = selected.op_id.0;
        state.public_event_limit = public_event_limit;
        let control_seq = state.control_seq;
        let epoch = state.epoch;
        let _ = state;
        let request_key = RequestKey::new(self.authority_id, id, epoch);
        self.pending_commands.push_back(BatchCommand::Commit {
            request_key,
            control_seq,
            expected_parent,
            selected,
            public_event_limit,
            disposition: Disposition::Publish,
        });
    }

    /// Applies completed worker commands to scheduler-owned allocation state.
    pub(super) fn acknowledge_commands(&mut self, commands: &[BatchCommand]) {
        for command in commands {
            if let BatchCommand::Free { buffer } = command {
                // Buffer ownership may sit with an already queued free, a live
                // request, or a request awaiting its close acknowledgement.
                let id = buffer.owner.request_id;
                let allocation = self
                    .pending_buffer_frees
                    .remove(buffer)
                    .or_else(|| {
                        self.running
                            .get_mut(&id)
                            .filter(|state| state.epoch == buffer.owner.epoch)
                            .and_then(|state| state.allocations_mut().take_buffer(*buffer))
                    })
                    .or_else(|| {
                        self.retiring_requests
                            .get_mut(&id)
                            .filter(|state| state.request_key == buffer.owner)
                            .and_then(|state| state.buffers.remove(buffer))
                    });
                if let Some(allocation) = allocation {
                    self.memory.free(allocation);
                }
            } else if let BatchCommand::Finish { request_key, .. }
            | BatchCommand::Retire { request_key, .. } = command
            {
                let id = request_key.request_id;
                let Some(retiring) = self.retiring_requests.get(&id) else {
                    tracing::error!(
                        request_id = id.0,
                        epoch = request_key.epoch,
                        "close acknowledgement does not match a retiring request"
                    );
                    self.fatal = true;
                    continue;
                };
                if retiring.request_key != *request_key {
                    tracing::error!(
                        request_id = id.0,
                        epoch = request_key.epoch,
                        "close acknowledgement does not match a retiring request"
                    );
                    self.fatal = true;
                    continue;
                }

                let retiring = self
                    .retiring_requests
                    .remove(&id)
                    .expect("retiring request exists");
                for allocation in retiring.buffers.into_values().chain(retiring.allocations) {
                    self.memory.free(allocation);
                }
            }
        }
    }

    /// Returns whether a request may keep an additional operation in flight at the
    /// predecessor's not-yet-observed selected point. Eligibility requires an
    /// exact projected cursor and a reachable predicate product for the target
    /// work leaf.
    pub(super) fn can_queue_successor(&self, id: RequestId, target: OpCode) -> bool {
        // Successor projection requires both live runtime state and an unresolved
        // predecessor from which to derive the device-selected checkpoint.
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

        // Bound speculation by worker capacity and exclude lineages whose
        // terminal or relay state makes the projection unsafe.
        if queue.len() >= self.info.max_unresolved_ops as usize
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
        if matches!(
            state.req.policy.trigger,
            uniserve_core::TriggerPolicyDescriptor::RoundCloseThenSuffix { .. }
        ) {
            return false;
        }

        // Decode successors support both feedback continuation and ordinary
        // prompt/decode pipelining, with different cursor evidence for each.
        if target == OpCode::ArDecode {
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
                    .outputs()
                    .iter()
                    .any(|output| output.kind == ProductKind::Token);
            if feedback_continuation {
                return self
                    .projected_inflight_variant(id)
                    .is_some_and(|variant| variant == OpCode::ArDecode)
                    && state.cursor.und.tokens_emitted.saturating_add(1)
                        < state.req.max_und_tokens;
            }

            if !matches!(state.cursor.phase, Phase::Prefill | Phase::DecodeUnd)
                || (state.cursor.phase == Phase::Prefill && state.starts_gen_after_context())
                || queue
                    .iter()
                    .any(|op| !matches!(op.operation.kind(), OpCode::ArExtend | OpCode::ArDecode))
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

        // Other successor kinds require a completion predicate and an exact
        // projected physical variant match.
        predecessor
            .operation
            .outputs()
            .iter()
            .any(|output| output.kind == ProductKind::Completion)
            && self
                .projected_inflight_variant(id)
                .is_some_and(|variant| variant == target)
    }

    /// Infers the next pipelined operation kind from projected in-flight request state.
    pub(super) fn projected_inflight_variant(&self, id: RequestId) -> Option<OpCode> {
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
            Phase::Prefill | Phase::DecodeUnd => OpCode::ArDecode,
            Phase::CloseKv | Phase::FeedbackState => OpCode::ArExtend,
            Phase::PublishKv => OpCode::TransferKvPublish,
            Phase::PrepareGen => OpCode::DiffusionPrepare,
            Phase::DenoiseGen => OpCode::DiffusionStep,
            Phase::CommitGen => OpCode::DiffusionFinalize,
            Phase::FeedbackEncode => {
                let feedback = state.req.policy.feedback.as_ref()?;
                match feedback.ingest.steps.get(cursor.feedback.ingest_step)? {
                    ImageIngestStep::VaeEncode => OpCode::EncoderLatent,
                    ImageIngestStep::VitEncode => OpCode::EncoderVision,
                }
            }
            Phase::Encode | Phase::IngestState => return None,
        })
    }

    /// Returns whether a successor can consume device-selected tokens before host observation.
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

    /// Returns whether the latest host-resolved token may remain the exact device input
    /// to the next decode. CPU stop and EOS decisions are complete before this
    /// check; a continuing request therefore names the same sampled token.
    pub(super) fn can_reuse_resolved_token_product(&self, id: RequestId) -> bool {
        self.running.get(&id).is_some_and(|state| {
            state.cursor.resources.worker_registered
                && state.cursor.und.tokens_emitted > 0
                && state.latest_device_version.as_ref().is_some()
                && state.is_replayable_text()
                && state.cursor.phase == Phase::DecodeUnd
                && !state.cursor.ingest.round_closing
                && (!state.req.behavior.gen_output
                    || state.req.policy.trigger.direct_token().is_some())
        })
    }

    /// Returns whether the next operation can be scheduled.
    pub(super) fn can_schedule_next(&self, id: RequestId) -> bool {
        !self.inflight.finishes.contains_key(&id)
            && self.output_window_ready(id)
            && self.running.get(&id).is_some_and(|state| {
                !state.speculative_chain_invalidated
                    && self.pending_commit_horizon_open(state)
                    && self.peek_next_operation_variant(id).is_none_or(|kind| {
                        self.worker_target(
                            RequestKey::new(self.authority_id, id, state.epoch),
                            kind,
                        )
                        .is_some()
                    })
            })
            && (!self.inflight.contains(id)
                || self
                    .projected_inflight_variant(id)
                    .is_some_and(|target| self.can_queue_successor(id, target)))
    }

    /// Returns whether the request may register another operation without exceeding its
    /// bounded provisional horizon. A request with no held commits is always
    /// open; a stop-string request that has decoded prefixes awaiting the
    /// frontend decision may run ahead by at most the unresolved-window depth,
    /// after which it waits for an acknowledgement so the horizon stays finite.
    pub(super) fn pending_commit_horizon_open(&self, state: &ReqState) -> bool {
        let horizon = self.info.max_unresolved_ops.max(1) as usize;
        state.pending_commits.len() < horizon
    }

    /// Returns whether output capacity can cover every unresolved token-producing operation.
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

    /// Returns the output-size bound for the next operation.
    pub(super) fn next_output_bound(&self, id: RequestId) -> usize {
        match self.peek_next_operation_variant(id) {
            Some(OpCode::ArExtend | OpCode::ArDecode) => 4,
            Some(OpCode::DiffusionStep) => usize::from(self.denoise_step_burst).saturating_add(2),
            Some(OpCode::DiffusionFinalize) => 3,
            Some(_) | None => 2,
        }
    }

    /// Registers a submitted generation operation and its cursor transition.
    pub(super) fn register_inflight(
        &mut self,
        operation: Operation,
        apply: RuntimeApply,
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

    /// Registers a submitted media operation and its projected media cursor.
    pub(super) fn register_media_inflight(
        &mut self,
        operation: Operation,
        quantum: MediaQuantum,
        started: Instant,
    ) {
        self.register_inflight_apply(operation, InflightApply::Media(quantum), started, 0);
    }

    /// Registers a submitted operation and charges its domain and transfer credits.
    pub(super) fn register_inflight_apply(
        &mut self,
        operation: Operation,
        apply: InflightApply,
        started: Instant,
        queue_us: u64,
    ) {
        let request_id = operation.request_key.request_id;
        let domain = self.stats.domains.get(operation.domain());
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

    /// Records the domain backpressure.
    pub(super) fn record_domain_backpressure(&self, domain: uniserve_worker_ipc::Domain) {
        self.stats
            .domains
            .get(domain)
            .backpressure_events
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Accumulates timing and completion counters for one returned domain operation.
    pub(super) fn record_domain_run(
        &self,
        accounting: SubmittedRunAccounting,
        timing: TimingCounters,
    ) {
        let stats = self.stats.domains.get(accounting.domain);
        stats.completed_runs.fetch_add(1, Ordering::Relaxed);
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
            stats.co_resident_runs.fetch_add(1, Ordering::Relaxed);
            stats
                .co_resident_us
                .fetch_add(timing.device_us, Ordering::Relaxed);
        }
    }

    /// Records the domain completion.
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

    /// Reclaims the domain credit.
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

    /// Fails the inflight domain credits.
    pub(super) fn fail_inflight_domain_credits(&self) {
        for inflight in self.inflight.operations.values().flatten() {
            self.reclaim_domain_credit(inflight.operation.domain(), true);
        }
    }

    /// Stages one validated completion until earlier operations for the request are applied.
    pub(super) fn stage_completion(
        &mut self,
        record: ModelOutput,
        products: Arc<[ProductPayload]>,
    ) {
        let id = record.request_key.request_id;
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
                state.terminal_intent = TerminalIntent::Failure(
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

    /// Validates and applies one ordered media completion to the request cursor.
    pub(super) fn apply_media_completion(
        &mut self,
        operation: Operation,
        quantum: MediaQuantum,
        record: ModelOutput,
    ) {
        let id = record.request_key.request_id;
        let Some(state) = self.media_state(id) else {
            return;
        };
        let mut consumed_products = Vec::new();
        let already_failed = matches!(state.terminal_intent, TerminalIntent::Failure(_));
        let media_output = record.media_output().cloned();
        let media_output_valid =
            (operation.kind() == OpCode::DiffusionFinalize) == media_output.is_some();
        let valid = record.status == OpStatus::Ok
            && record.request_key == operation.request_key
            && record.op_id == operation.op_id
            && (!operation.advances_state() || record.selected_point > 0)
            && media_output_valid;
        if valid
            && let Some(state) = self.media_state_mut(id)
            && state.admission_state == DiffusionRequestParamsState::InFlight
        {
            state.admission_state = DiffusionRequestParamsState::Registered;
        }
        if !already_failed {
            if !valid {
                if let Some(state) = self.media_state_mut(id) {
                    state.terminal_intent =
                        TerminalIntent::Failure("media worker operation failed".to_string());
                }
            } else if let Some(state) = self.media_state_mut(id) {
                state.committed = advance_media_cursor(state.committed, quantum);
                match quantum {
                    MediaQuantum::Append {
                        track: MediaTrack::Video,
                        cursor,
                        ..
                    } => {
                        if let Some((_count, product)) = state.video_segments.remove(&cursor) {
                            consumed_products.push(product);
                        }
                    }
                    MediaQuantum::Append {
                        track: MediaTrack::Audio,
                        ..
                    } => {
                        if let Some(product) = state.audio.take() {
                            consumed_products.push(product);
                        }
                    }
                    _ => {}
                }
                let phase = match quantum {
                    MediaQuantum::Encode => "preparing",
                    MediaQuantum::Prepare | MediaQuantum::Denoise { .. } => {
                        if state.committed.denoise_step == state.request.geometry.denoise_steps {
                            "decoding"
                        } else {
                            "denoising"
                        }
                    }
                    MediaQuantum::Decode { .. } | MediaQuantum::Append { .. } => "decoding",
                    MediaQuantum::Finalize => "finalizing",
                };
                let _ = state.event_tx.send(Event::MediaProgress {
                    phase: phase.to_owned(),
                    completed_steps: state.committed.denoise_step,
                });
                if let Some(output) = media_output {
                    state.artifact = Some(ArtifactEvent {
                        media_kind: MediaKind::Video,
                        content_type: "video/mp4".to_string(),
                        bytes: output.bytes,
                        artifact: match output.handle {
                            uniserve_worker_ipc::ArtifactHandle::PosixShm { name } => {
                                ArtifactHandle::PosixShm { name }
                            }
                        },
                    });
                }
                if operation.advances_state() {
                    state.fixed_parent = Checkpoint {
                        op_id: record.op_id,
                        point: CheckpointPoint::Fixed(record.selected_point),
                    };
                }
            }
        }

        if !consumed_products.is_empty() {
            self.free_products(consumed_products);
        }

        let terminal = self.media_state(id).and_then(|state| {
            if self.inflight.contains(id) {
                return None;
            }
            if let TerminalIntent::Failure(message) = &state.terminal_intent {
                Some((
                    DiffusionTerminal::Failed(message.clone()),
                    CloseReason::Error,
                ))
            } else if let TerminalIntent::Finish(reason) = &state.terminal_intent {
                Some((
                    DiffusionTerminal::Finished(reason.clone()),
                    CloseReason::Cancelled,
                ))
            } else if state.event_tx.is_closed() {
                Some((
                    DiffusionTerminal::Finished(FinishReason::Cancelled),
                    CloseReason::Cancelled,
                ))
            } else if state.committed.finalized {
                let event = state.artifact.clone().map_or_else(
                    || DiffusionTerminal::Failed("media output was not finalized".to_string()),
                    DiffusionTerminal::Completed,
                );
                let reason = if matches!(event, DiffusionTerminal::Completed(_)) {
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

    /// Emits terminal media output and releases all request-owned resources.
    pub(super) fn finish_media(
        &mut self,
        id: RequestId,
        event: DiffusionTerminal,
        reason: CloseReason,
        cutoff: Option<Checkpoint>,
    ) {
        let Some(state) = self.take_media_state(id) else {
            return;
        };
        self.scheduler
            .running_order
            .retain(|candidate| *candidate != id);
        if !matches!(event, DiffusionTerminal::Completed(_))
            && let Some(artifact) = state.artifact.as_ref()
        {
            let artifact_name = artifact.artifact.posix_shm_name();
            if let Ok(name) = std::ffi::CString::new(format!("/{artifact_name}")) {
                // SAFETY: the worker-provided handle was validated before it entered media state.
                let result = unsafe { libc::shm_unlink(name.as_ptr()) };
                if result != 0
                    && std::io::Error::last_os_error().raw_os_error() != Some(libc::ENOENT)
                {
                    tracing::warn!(
                        request_id = id.0,
                        handle = artifact_name,
                        "failed to discard unclaimed media shared memory"
                    );
                }
            }
        }
        match event {
            DiffusionTerminal::Completed(artifact) => {
                let _ = state.event_tx.send(Event::Artifact(artifact));
                let _ = state.event_tx.send(Event::Finished {
                    reason: FinishReason::Completed,
                    stop_reason: None,
                    prompt_tokens: state.request.prompt_token_ids.len(),
                    completion_tokens: 0,
                    images: 0,
                });
            }
            DiffusionTerminal::Failed(message) => {
                let _ = state.event_tx.send(Event::Error { message });
            }
            DiffusionTerminal::Finished(reason) => {
                let _ = state.event_tx.send(Event::Finished {
                    reason,
                    stop_reason: None,
                    prompt_tokens: state.request.prompt_token_ids.len(),
                    completion_tokens: 0,
                    images: 0,
                });
            }
        }
        match state.admission_state {
            DiffusionRequestParamsState::Unsubmitted => {
                state.allocations.free(&mut self.memory);
                return;
            }
            _ => {}
        }
        let request_key = state.admission.request_key;
        self.pending_commands.push_back(
            if reason == CloseReason::Error
                || state.admission_state == DiffusionRequestParamsState::Unsubmitted
            {
                BatchCommand::Retire {
                    request_key,
                    retained_buffers: Vec::new(),
                }
            } else {
                BatchCommand::Finish {
                    request_key,
                    control_seq: 1,
                    cutoff: cutoff.unwrap_or(state.fixed_parent),
                    reason,
                    retained_buffers: Vec::new(),
                }
            },
        );
        self.retiring_requests.insert(
            id,
            RetiringRequest {
                request_key,
                allocations: state.allocations.into_allocations().collect(),
                buffers: HashMap::new(),
            },
        );
    }

    /// Resolves the front in-flight op for `id` by the worker's echoed `op_id`.
    pub(super) fn pop_inflight(
        &mut self,
        request_key: RequestKey,
        op_id: u64,
    ) -> Option<(Operation, InflightApply, Instant)> {
        let inflight = self.inflight.pop(request_key, op_id)?;
        self.reclaim_domain_credit(inflight.operation.domain(), false);
        Some((inflight.operation, inflight.apply, inflight.started))
    }

    /// Releases product allocations owned by completed operations.
    pub(super) fn free_products(&mut self, products: Vec<ProductRef>) {
        let mut released = HashSet::new();
        for product in products {
            let buffer = product.buffer_id();
            if !released.insert(buffer) {
                continue;
            }
            if let Some(allocation) = self.memory.take_encoder_buffer(buffer) {
                if let Some(previous) = self.pending_buffer_frees.insert(buffer, allocation) {
                    tracing::error!(?buffer, "buffer free identity was already pending");
                    self.memory.free(previous);
                    self.fatal = true;
                }
            }
            self.pending_commands
                .push_back(BatchCommand::Free { buffer });
        }
    }

    /// Preserve unrelated requests after a reconciled Worker failure. Unclassified
    /// executor errors and control-plane invariant violations remain fatal.
    pub(super) fn on_executor_error(&mut self, error: anyhow::Error) {
        if let Some(failure) = error.downcast_ref::<WorkerFailure>() {
            let execution = failure.execution.as_ref();
            let code = execution.and_then(|value| value.code.as_deref());
            let retryable = execution.is_some_and(|value| value.retryable);
            if matches!(code, Some("SchedulerBug" | "InvariantViolation")) {
                self.fatal = true;
                tracing::error!(worker = %failure.worker_id, ?code, "control-plane invariant failed: {error}");
            } else if code == Some("InputError") {
                tracing::info!(worker = %failure.worker_id, ?code, "Worker rejected request: {error}");
            } else {
                tracing::warn!(worker = %failure.worker_id, ?code, retryable, "Worker failed affected work: {error}");
            }
            self.fail_after_worker_failure(failure);
            return;
        }
        self.fatal = true;
        tracing::error!("fatal executor error: {error}");
        self.fail_all_inflight(&error.to_string());
    }

    /// Reconciles one physical batch result with logical operations, state, and ownership.
    pub(super) fn apply_result(&mut self, report: BatchResult) {
        let result_batch_id = report.batch_id;
        let mut returned_domains: HashMap<uniserve_worker_ipc::Domain, (usize, TimingCounters)> =
            HashMap::new();
        let mut invalid_result = false;

        // A completion can mutate state only while its logical batch remains owned
        // by the in-flight window.
        if !self
            .inflight
            .batch_operations
            .contains_key(&result_batch_id)
        {
            tracing::error!(
                batch_id = result_batch_id,
                "executor returned a result for an unknown logical batch"
            );
            self.fatal = true;
            self.fail_all_running("executor returned a result for an unknown logical batch");
            return;
        }

        // Consume each expected operation identity exactly once while folding the
        // rank-local timing records into their scheduler domains.
        for result in &report.results {
            let record = &result.output;
            let identity = (record.request_key, record.op_id);
            let removed = self
                .inflight
                .batch_operations
                .get_mut(&result_batch_id)
                .is_some_and(|pending| pending.remove(&identity));
            if !removed {
                tracing::error!(
                    batch_id = result_batch_id,
                    request_id = record.request_key.request_id.0,
                    op_id = record.op_id.0,
                    "executor returned a duplicate or unknown operation result"
                );
                invalid_result = true;
                continue;
            }
            let Some(domain) = self
                .inflight
                .operations
                .get(&record.request_key.request_id)
                .and_then(|operations| {
                    operations
                        .iter()
                        .find(|inflight| inflight.operation.op_id == record.op_id)
                })
                .map(|inflight| inflight.operation.domain())
            else {
                invalid_result = true;
                continue;
            };
            let entry = returned_domains
                .entry(domain)
                .or_insert((0, TimingCounters::default()));
            entry.0 += 1;
            entry.1.queued_us = entry.1.queued_us.max(record.timing_counters.queued_us);
            entry.1.device_us = entry.1.device_us.max(record.timing_counters.device_us);
            entry.1.copy_us = entry.1.copy_us.max(record.timing_counters.copy_us);
            entry.1.host_us = entry.1.host_us.max(record.timing_counters.host_us);
        }

        let operations_complete = self
            .inflight
            .batch_operations
            .get(&result_batch_id)
            .is_some_and(HashSet::is_empty);
        let batch_complete = report.done;
        if batch_complete && !operations_complete {
            invalid_result = true;
        }
        if invalid_result {
            self.fatal = true;
            self.fail_all_running("executor returned an invalid operation result");
            return;
        }

        // Publish domain and batch accounting before individual request state is
        // advanced, so every accepted completion contributes exactly once.
        let forward_stats = report.forward_stats;
        let completion_count = report.results.len();
        let trace_enabled = self.trace_enabled();
        let mut domain_run_trace = trace_enabled.then(Vec::new);
        for (index, (domain, (operation_count, timing))) in returned_domains.into_iter().enumerate()
        {
            let accounting = SubmittedRunAccounting {
                domain,
                mixed: false,
                submission_group: index as u32 + 1,
                operation_count,
            };
            self.record_domain_run(accounting, timing);
            for result in &report.results {
                let record = &result.output;
                let record_domain = self
                    .inflight
                    .operations
                    .get(&record.request_key.request_id)
                    .and_then(|operations| {
                        operations
                            .iter()
                            .find(|inflight| inflight.operation.op_id == record.op_id)
                    })
                    .map(|inflight| inflight.operation.domain());
                if record_domain == Some(domain) {
                    self.record_domain_completion(domain, record.status);
                }
            }
            if let Some(trace) = domain_run_trace.as_mut() {
                trace.push(json!({
                    "result_group": accounting.submission_group,
                    "domain": accounting.domain,
                    "operations": accounting.operation_count,
                    "execution": "domain_homogeneous",
                    "queue_us": timing.queued_us,
                    "device_us": timing.device_us,
                    "completion_us": timing.copy_us.saturating_add(timing.host_us),
                    "co_resident_us": 0,
                }));
            }
        }

        if !report.worker_exec_us.is_empty() {
            let groups = self
                .inflight
                .batch_group_worker_exec_us
                .get_mut(&result_batch_id)
                .expect("known batch has worker timing state");
            for worker_exec_us in report.worker_exec_us {
                let group = groups.len() as u32 + 1;
                groups.insert(group, worker_exec_us);
            }
        }

        if batch_complete {
            self.inflight.batch_operations.remove(&result_batch_id);
            if let Some(commands) = self.inflight.command_batches.remove(&result_batch_id) {
                for (index, command) in commands.into_iter().enumerate() {
                    let failed = report.command_results.iter().any(|result| {
                        result.command_index == index as u32
                            && result.outcome == crate::executor::CommandOutcome::Failed
                    });
                    if failed {
                        match command {
                            BatchCommand::Finish {
                                request_key,
                                retained_buffers,
                                ..
                            } => {
                                self.pending_commands.push_back(BatchCommand::Retire {
                                    request_key,
                                    retained_buffers,
                                });
                            }
                            BatchCommand::Free { .. } | BatchCommand::Retire { .. } => {
                                self.pending_commands.push_back(command);
                            }
                            BatchCommand::Start { .. } | BatchCommand::Commit { .. } => {}
                        }
                    } else {
                        self.acknowledge_commands(std::slice::from_ref(&command));
                    }
                }
            }
        }

        let batch_roundtrip_us = self
            .inflight
            .batch_started
            .get(&result_batch_id)
            .map(|start| start.elapsed().as_micros() as u64)
            .unwrap_or(0);

        if batch_complete {
            self.inflight.batch_started.remove(&result_batch_id);
            self.inflight.prefill_steps.remove(&result_batch_id);
        }
        let worker_us = if batch_complete {
            self.inflight
                .batch_group_worker_exec_us
                .remove(&result_batch_id)
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

        // Staging decouples executor arrival order from request-local dependency
        // order. Only a ready prefix is removed by `take_ready` below.
        for result in report.results {
            self.stage_completion(result.output, Arc::from(result.products));
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
                let id = record.request_key.request_id;
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
                        state.terminal_intent = TerminalIntent::Failure(
                            "worker returned an out-of-order media operation".to_string(),
                        );
                    } else if self.running.contains_key(&id) {
                        self.finish(id, FinishReason::Error);
                    }
                    continue;
                };

                let operation_variant = operation.kind();
                let roundtrip_us = started.elapsed().as_micros() as u64;

                // Media operations update their independent cursor immediately;
                // generation operations continue through semantic validation.
                let apply = match apply {
                    InflightApply::Media(quantum) => {
                        self.apply_media_completion(operation, quantum, record);
                        continue;
                    }
                    InflightApply::Generation(apply) => apply,
                };

                if self
                    .inflight
                    .finishes
                    .get(&id)
                    .is_some_and(|finish| finish.reason == FinishReason::Error)
                {
                    self.free_products(operation.outputs().to_vec());
                    self.finish_pending_if_idle(id);
                    continue;
                }

                let discard_invalidated_descendant = self
                    .running
                    .get(&id)
                    .is_some_and(|state| state.speculative_chain_invalidated);
                if discard_invalidated_descendant {
                    let has_unresolved_descendants = self.inflight.contains(id);
                    if let Some(state) = self.running.get_mut(&id) {
                        state.cursor.resources.blocks_sent = state
                            .cursor
                            .resources
                            .blocks_sent
                            .saturating_sub(apply.new_blocks);
                        state.speculative_chain_invalidated = has_unresolved_descendants;
                    }
                    self.free_products(operation.outputs().to_vec());
                    self.finish_pending_if_idle(id);
                    continue;
                }

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
                        "domain": operation.domain(),
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
                        "steps_completed": record.logical_lengths().latent_len,
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

                // Completion products used only as device predicates can be released
                // once no in-flight descendant refers to them.
                if record.status == OpStatus::Ok && !operation.advances_state() {
                    let completion_has_device_consumer =
                        self.inflight.operations.get(&id).is_some_and(|queue| {
                            queue.iter().any(|inflight| {
                                inflight
                                    .operation
                                    .predicate()
                                    .as_ref()
                                    .is_some_and(|predicate| {
                                        predicate.producer_op_id == operation.op_id
                                    })
                            })
                        });
                    if !completion_has_device_consumer {
                        let completed_predicates = operation
                            .outputs()
                            .iter()
                            .filter(|output| output.kind == ProductKind::Completion)
                            .cloned()
                            .collect::<Vec<_>>();
                        self.free_products(completed_predicates);
                    }
                }

                let semantic_blocked = self
                    .running
                    .get(&id)
                    .is_some_and(|state| state.terminal_intent.is_terminal())
                    || self.inflight.finishes.contains_key(&id);
                if semantic_blocked {
                    if apply.free_latent_on_apply {
                        self.free_request_latent(id);
                    }
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
                    && operation.advances_state();
                let latest_device_version = if advanced {
                    let token = operation
                        .outputs()
                        .iter()
                        .find(|output| {
                            output.kind == ProductKind::Token
                                && output.storage_class
                                    == uniserve_worker_ipc::StorageClass::RequestRelay
                        })
                        .cloned();
                    token.map(|token| ResidentDeviceVersion {
                        token,
                        version: Checkpoint {
                            op_id: operation.op_id,
                            point: CheckpointPoint::DeviceSelected,
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
                let mut deferred_commit: Option<(Checkpoint, Checkpoint, u64)> = None;
                if advanced
                    && let (Some(expected_parent), Some(selected)) =
                        (expected_parent, selected_fixed.clone())
                {
                    let public_event_limit = self.public_limit_for(id, &apply);
                    let decoder_decision_required = matches!(
                        operation_variant,
                        OpCode::ArExtend | OpCode::ArDecode | OpCode::ArVerify
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

                // Reclaim resources whose lifetime ends at this transition before
                // making its public output eligible for resolution.
                let free_flow_prefix = operation_variant == OpCode::DiffusionStep
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
                if operation_variant == OpCode::DiffusionStep
                    && record.status == OpStatus::Ok
                    && let Some(prefix) = self
                        .running
                        .get_mut(&id)
                        .and_then(|state| state.flow_prefix.as_mut())
                {
                    prefix.diffusion_finalized = true;
                }
                if apply.free_latent_on_apply {
                    self.free_request_latent(id);
                }
                if free_flow_prefix {
                    self.free_flow_prefix(id);
                }
                if matches!(
                    operation_variant,
                    OpCode::DiffusionStep | OpCode::DiffusionFinalize
                ) {
                    let consumed_latents = operation
                        .inputs()
                        .iter()
                        .filter(|product| product.kind == ProductKind::Latent)
                        .cloned()
                        .collect::<Vec<_>>();
                    if !consumed_latents.is_empty() {
                        self.free_products(consumed_latents);
                    }
                }

                let priority = completion_priority(operation_variant);
                let public_tokens_before = self
                    .running
                    .get(&id)
                    .map_or(0, |state| state.output.tokens_sent);
                if record.status == OpStatus::Predicated {
                    let has_unresolved_descendants = self.inflight.contains(id);
                    if let Some(state) = self.running.get_mut(&id) {
                        state.cursor.resources.blocks_sent = state
                            .cursor
                            .resources
                            .blocks_sent
                            .saturating_sub(apply.new_blocks);
                        state.speculative_chain_invalidated = has_unresolved_descendants;
                    }
                    let unused_products = operation.outputs().to_vec();
                    self.free_products(unused_products);
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

            // Resolve semantic effects in lifecycle priority and stable arrival
            // order, independent of the executor's physical completion order.
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
                    operation.kind(),
                    OpCode::ArExtend | OpCode::ArDecode | OpCode::ArVerify
                );
                if self.running.contains_key(&id) && !self.inflight.finishes.contains_key(&id) {
                    self.resolve(id, operation, apply, view, prefix_versions.clone());
                }
                if token_operation {
                    let mut immediate_commit: Option<(Checkpoint, Checkpoint, u64)> = None;
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
                "batch_id": result_batch_id,
                "worker_exec_us": worker_us,
                "host_roundtrip_us": batch_roundtrip_us,
                "batch_complete": batch_complete,
                "domains": domain_run_trace,
                "forward_stats": forward_stats_trace,
                "batch_size": resolved_ops.len(),
                "ops": resolved_ops,
                "progress": progress_ops,
                "running": self.running.len(),
                "pending": self.scheduler.waiting_len(),
                "in_flight": self.inflight.batch_started.len(),
            }));
        }
    }

    /// Merges optional worker forward-pass counters into scheduler statistics.
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

    /// Fails every submitted operation while preserving requests that can be rescheduled.
    pub(super) fn fail_all_inflight(&mut self, msg: &str) {
        self.pending_submissions.clear();
        self.fail_inflight_domain_credits();
        // Submitted batches cannot return after this boundary, so their timing
        // and command ownership must be retired together.
        let (ids, commands) = self.inflight.clear_failed();
        self.acknowledge_commands(&commands);
        for id in ids {
            if self.media_state(id).is_some() {
                self.finish_media(
                    id,
                    DiffusionTerminal::Failed(msg.to_string()),
                    CloseReason::Error,
                    None,
                );
            } else if self.running.contains_key(&id) {
                self.emit(
                    id,
                    Event::Error {
                        message: msg.to_string(),
                    },
                );
                self.finish(id, FinishReason::Error);
            }
        }
    }

    /// Reconciles failed work without resetting allocations owned by live consumers.
    fn fail_after_worker_failure(&mut self, loss: &WorkerFailure) {
        if let Some(cache) = &self.memory.cache {
            for endpoint in &loss.endpoints {
                cache.block_pool.invalidate_source(endpoint);
            }
        }
        let mut requests = loss.requests.iter().copied().collect::<HashSet<_>>();
        let mut products = loss.products.iter().cloned().collect::<HashSet<_>>();
        loop {
            let before = (requests.len(), products.len());
            for batch in &self.pending_submissions {
                for op in &batch.ops {
                    if (!loss.endpoints.is_empty() && op.target.0 == loss.worker_id)
                        || op
                            .payload
                            .inputs
                            .iter()
                            .chain(op.payload.predicate.iter())
                            .any(|product| products.contains(product))
                    {
                        requests.insert(op.request);
                    }
                    if requests.contains(&op.request) {
                        products.extend(op.payload.outputs.iter().cloned());
                    }
                }
            }
            if before == (requests.len(), products.len()) {
                break;
            }
        }
        let reclaimable = self.memory.encoder_cache.invalidate_products(&products);
        self.free_products(reclaimable);
        let mut retired = loss.retired.clone();
        let mut pending = VecDeque::new();
        for mut batch in std::mem::take(&mut self.pending_submissions) {
            retired.extend(
                batch
                    .retire_requests(&requests)
                    .into_iter()
                    .map(|op| (batch.id, op.request, op.id)),
            );
            if let Some(commands) = self.inflight.command_batches.get_mut(&batch.id) {
                *commands = batch
                    .commands
                    .iter()
                    .filter(|command| !matches!(command, BatchCommand::Start { .. }))
                    .cloned()
                    .collect();
            }
            pending.push_back(batch);
        }
        for (batch_id, request, op) in retired {
            let Some(inflight) = self.inflight.retire_operation(batch_id, request, op) else {
                self.fatal = true;
                tracing::error!(
                    batch_id,
                    ?request,
                    ?op,
                    "Worker failure named an unknown pending operation"
                );
                return;
            };
            self.reclaim_domain_credit(inflight.operation.domain(), true);
        }
        for batch in pending {
            if batch.ops.is_empty() && batch.commands.is_empty() {
                self.inflight.batch_operations.remove(&batch.id);
                self.inflight.batch_started.remove(&batch.id);
                self.inflight.prefill_steps.remove(&batch.id);
                self.inflight.batch_group_worker_exec_us.remove(&batch.id);
                self.inflight.command_batches.remove(&batch.id);
            } else {
                self.pending_submissions.push_back(batch);
            }
        }
        self.pending_commands.retain_mut(|command| {
            if !requests.contains(&command.request_key()) {
                return true;
            }
            match command {
                BatchCommand::Start { .. } | BatchCommand::Commit { .. } => false,
                BatchCommand::Finish {
                    request_key,
                    retained_buffers,
                    ..
                } => {
                    *command = BatchCommand::Retire {
                        request_key: *request_key,
                        retained_buffers: std::mem::take(retained_buffers),
                    };
                    true
                }
                BatchCommand::Free { .. } | BatchCommand::Retire { .. } => true,
            }
        });
        let authority_id = self.authority_id;
        for request in requests {
            let id = request.request_id;
            let error_emitted = self
                .inflight
                .finishes
                .get(&id)
                .is_some_and(|finish| finish.reason == FinishReason::Error);
            if let Some(state) = self.media_state_mut(id) {
                if state.admission.request_key == request {
                    state.terminal_intent = TerminalIntent::Failure(loss.message.clone());
                }
            } else if let Some(state) = self.running.get_mut(&id) {
                if request != RequestKey::new(authority_id, id, state.epoch) {
                    continue;
                }
                state.pending_commits.clear();
                if !error_emitted {
                    self.emit(
                        id,
                        Event::Error {
                            message: loss.message.clone(),
                        },
                    );
                }
                self.inflight.finishes.insert(
                    id,
                    PendingFinish {
                        reason: FinishReason::Error,
                        stop_reason: None,
                    },
                );
            }
            // A successful later completion may already be waiting behind the now
            // abandoned operation. It owns real completion evidence and can drain.
            loop {
                let Some(op_id) = self
                    .inflight
                    .operations
                    .get(&id)
                    .and_then(|queue| queue.front())
                    .filter(|inflight| inflight.operation.request_key == request)
                    .map(|inflight| inflight.operation.op_id.0)
                else {
                    break;
                };
                let Some(completion) = self
                    .inflight
                    .completions
                    .get_mut(&id)
                    .and_then(|values| values.remove(&op_id))
                else {
                    break;
                };
                let Some((operation, _, _)) = self.pop_inflight(request, op_id) else {
                    unreachable!("ready completion owns its operation");
                };
                drop(completion);
                self.free_products(operation.outputs().to_vec());
            }
            if self
                .inflight
                .completions
                .get(&id)
                .is_some_and(BTreeMap::is_empty)
            {
                self.inflight.completions.remove(&id);
            }
            if self
                .media_state(id)
                .is_some_and(|state| state.admission.request_key == request)
            {
                if !self.inflight.contains(id) {
                    self.finish_media(
                        id,
                        DiffusionTerminal::Failed(loss.message.clone()),
                        CloseReason::Error,
                        None,
                    );
                }
            } else if self.running.contains_key(&id) {
                self.finish_after_inflight(id, FinishReason::Error, None);
            }
        }
    }

    /// Fails every queued and running request and releases scheduler-owned resources.
    pub(super) fn fail_all_running(&mut self, message: &str) {
        self.pending_submissions.clear();
        self.fail_inflight_domain_credits();
        let _ = self.inflight.clear_failed();
        while let Some(id) = self.scheduler.pop_media() {
            let submission = self
                .waiting_media
                .remove(&id)
                .expect("scheduler media order names runtime state");
            let _ = submission.event_tx.send(Event::Error {
                message: message.to_string(),
            });
        }
        let media = self.media_ids();
        for id in media {
            self.finish_media(
                id,
                DiffusionTerminal::Failed(message.to_string()),
                CloseReason::Error,
                None,
            );
        }
        let ids = self.running.keys().copied().collect::<Vec<_>>();
        for id in ids {
            self.emit(
                id,
                Event::Error {
                    message: message.to_string(),
                },
            );
            self.finish(id, FinishReason::Error);
        }
    }
}
