//! Batch submission, completion application, and allocation reclamation.
//!
//! The loop keeps executor progress non-blocking until work is outstanding, then
//! parks with a bounded liveness deadline. Completion application validates
//! request and operation identities before mutating runtime state.

use super::*;
use uniserve_worker_ipc::{ForwardMode, PipelineStage, TransferMode};

impl Scheduler {
    /// Enumerates loaded entries that can execute the requested operation.
    pub(super) fn worker_candidates(
        &self,
        kind: Computation,
    ) -> impl Iterator<Item = (&crate::WorkerId, &str, &WorkerInfo)> {
        let entry = match kind {
            Computation::Pipeline(PipelineStage::TextEncoding) => "text_encoder",
            Computation::Pipeline(PipelineStage::LatentPreparation) => "denoiser",
            Computation::Pipeline(PipelineStage::Denoising) => "denoiser",
            Computation::Pipeline(
                PipelineStage::VideoEncoding | PipelineStage::AudioEncoding | PipelineStage::Muxing,
            ) => "output",
            Computation::Pipeline(PipelineStage::ImageDecoding) => "denoiser",
            Computation::Pipeline(PipelineStage::VideoDecoding) => "video_decoder",
            Computation::Pipeline(PipelineStage::AudioDecoding) => "audio_decoder",
            Computation::Pipeline(PipelineStage::VisionEncoding) => "vision_encoder",
            Computation::Pipeline(PipelineStage::LatentEncoding) => "latent_encoder",
            _ => "model",
        };
        self.entry_candidates(kind, entry)
    }

    pub(super) fn entry_candidates<'a>(
        &'a self,
        kind: Computation,
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
        kind: Computation,
    ) -> Option<(&crate::WorkerId, &str)> {
        let (id, bound_entry, _) = self.worker_candidates(kind).next()?;
        self.executor.has_capacity(id).then_some((id, bound_entry))
    }

    /// Binds planned work to its configured entry and records request residency.
    pub(super) fn select_worker(
        &mut self,
        operation: &ScheduledRequest,
    ) -> (crate::WorkerId, String) {
        let (id, bound_entry) = if operation.entry == "model" {
            self.worker_target(operation.request_key, operation.code)
        } else {
            self.entry_candidates(operation.code, &operation.entry)
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
    fn batch_workers(&self, batch: &ExecutionBatch) -> HashSet<crate::WorkerId> {
        let mut targets = batch
            .requests
            .iter()
            .map(|(_, placement)| placement.worker.clone())
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

    /// Dispatches queued work while preserving order at every shared destination.
    /// The caller retains blocked destinations for the whole scheduling tick, so
    /// assembling more work cannot repeatedly retry an unavailable worker.
    fn dispatch_submissions(&mut self, blocked_workers: &mut HashSet<crate::WorkerId>) -> bool {
        let mut progressed = false;
        let pending_count = self.pending_submissions.len();
        for _ in 0..pending_count {
            let batch = self
                .pending_submissions
                .pop_front()
                .expect("pending count is fixed");
            let targets = self.batch_workers(&batch);
            if !targets.is_disjoint(blocked_workers) {
                // A batch waiting on one destination also orders later work at
                // its other destinations; independent destinations can proceed.
                blocked_workers.extend(targets);
                self.pending_submissions.push_back(batch);
                continue;
            }
            let batch_id = batch.id;
            match self.executor.submit(batch) {
                Ok(()) => progressed = true,
                Err(ExecutorSubmitError::WouldBlock(batch)) => {
                    blocked_workers.extend(targets);
                    self.pending_submissions.push_back(batch);
                }
                Err(ExecutorSubmitError::Failed(error)) => {
                    self.trace_record(json!({
                        "event": "batch_submit_failed",
                        "at_s": now(),
                        "batch_id": batch_id,
                        "error": format!("{error}"),
                    }));
                    // Failure reconciliation owns the pending identities until
                    // every affected computation has been retired.
                    self.on_executor_error(error);
                    return true;
                }
            }
        }
        progressed
    }

    /// Registers the computations and command receipts owned by one scheduled batch.
    pub(super) fn register_pending_batch(&mut self, batch: &ExecutionBatch, started: Instant) {
        self.pending_batches.insert(
            batch.id,
            PendingBatch {
                started,
                operations: batch
                    .requests
                    .iter()
                    .map(|(operation, _)| (operation.request_key, operation.op_id))
                    .collect(),
                commands: batch
                    .commands
                    .iter()
                    .filter(|command| !matches!(command, BatchCommand::Start { .. }))
                    .cloned()
                    .collect(),
                worker_exec_us: 0,
                prefill: batch
                    .requests
                    .iter()
                    .any(|(operation, _)| batch_kind(operation.code) == BatchKind::Prefill),
            },
        );
    }

    /// Releases the request latent.
    fn free_request_latent(&mut self, id: RequestId) {
        let allocation = self
            .running
            .get_mut(&id)
            .and_then(|state| state.allocations_mut().latent.take());
        if let Some(allocation) = allocation {
            self.free_allocation(allocation);
        }
    }
    /// Advances scheduler and executor work, blocking only for an outstanding result.
    ///
    /// Returns whether the loop made progress or handled an executor outcome.
    pub fn step(&mut self) -> bool {
        let progressed = self.step_nonblocking();
        if progressed || self.pending_batches.is_empty() {
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
        // responses become ready together at queue depth greater than one.
        // The owner loop returns here without parking while progress is being
        // made, so subsequent ready completions are handled on successive turns.
        progressed |= self.poll_one_result();

        progressed |= self.refill_executor();

        self.stats
            .general
            .in_flight
            .store(self.pending_batches.len(), Ordering::Relaxed);
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

        let mut blocked_workers = HashSet::new();
        progressed |= self.dispatch_submissions(&mut blocked_workers);
        if self.fatal {
            return progressed;
        }

        // 2. reap cancellations before assembling.
        self.reap_cancellations();

        // 3. Admit request/resource residency even
        // while every execution slot is occupied. This lets the next batch see
        // the complete resident cohort instead of admitting only when a slot
        // happens to open.
        self.admit();
        self.admit_media();

        // 4. submit as many batches as the rank queues admit.
        loop {
            self.admit();
            self.admit_media();
            if self.pending_batches.len() >= self.info.queue_depth.max(1) as usize {
                break;
            }
            let Some(batch) = self.schedule_batch() else {
                break;
            };
            self.pending_submissions.push_back(batch);
            progressed = true;
            self.dispatch_submissions(&mut blocked_workers);
            if self.fatal {
                return progressed;
            }
        }
        while self.pending_submissions.is_empty()
            && !self.pending_commands.is_empty()
            && self.pending_batches.len() < self.info.queue_depth.max(1) as usize
        {
            let commands = self.take_commands(|_| true);
            if commands.is_empty() {
                break;
            }
            let batch = self.finish_generation_batch(
                ExecutionBatch::new(0, Vec::new(), commands, Vec::new()),
                Instant::now(),
            );
            self.pending_submissions.push_back(batch);
            progressed = true;
            self.dispatch_submissions(&mut blocked_workers);
            if self.fatal {
                return progressed;
            }
        }

        progressed
    }

    /// Selects one batch under the shared queue budget and existing family fairness.
    fn schedule_batch(&mut self) -> Option<ExecutionBatch> {
        if self.prefer_media
            && let Some(batch) = self.prepare_media_batch()
        {
            self.prefer_media = false;
            return Some(batch);
        }
        let Some(batch) = self.assemble() else {
            if self.fatal {
                return None;
            }
            let batch = self.prepare_media_batch()?;
            self.prefer_media = false;
            return Some(batch);
        };
        self.prefer_media = true;
        Some(batch)
    }

    /// Allocates a request-scoped product reference for a terminal media result.
    pub(super) fn media_completion_product(
        &mut self,
        request_key: RequestKey,
        op_id: ComputationId,
    ) -> TensorRef {
        let generation = u32::try_from(self.next_product_generation.max(1))
            .expect("media product generation exhausted");
        self.next_product_generation = u64::from(generation).saturating_add(1);
        TensorRef {
            request_key,
            producer_op_id: op_id,
            output_index: 0,
            generation,
            dtype: DType::U8,
            shape_bound: ShapeBound::default(),
        }
    }
}

/// The lanes one media call occupies while it is in flight.
///
/// A component lane measured in media units carries `units`; every other
/// component lane is exclusive and admits one request at a time. `host` counts
/// the tasks the call places on the muxer rank's bounded host executor.
struct LaneDemand {
    component: Option<String>,
    units: u32,
    host: u32,
}

/// Lane occupancy across the media calls in flight.
#[derive(Default)]
struct LaneLedger {
    exclusive: HashMap<String, RequestId>,
    units: HashMap<String, u32>,
    host: u32,
}

impl LaneLedger {
    /// Returns whether one request's call fits the lanes it occupies.
    fn admits(&self, demand: &LaneDemand, request: RequestId, scheduler: &Scheduler) -> bool {
        if self.host + demand.host > scheduler.info.host_lane_capacity.max(1) {
            return false;
        }
        let Some(component) = demand.component.as_deref() else {
            return true;
        };
        if demand.units > 0 {
            let capacity = scheduler.device_lane_units(component).unwrap_or(1);
            self.units.get(component).copied().unwrap_or(0) + demand.units <= capacity
        } else {
            self.exclusive
                .get(component)
                .is_none_or(|holder| *holder == request)
        }
    }

    /// Marks the lanes one request's call occupies until it completes.
    fn occupy(&mut self, demand: &LaneDemand, request: RequestId) {
        self.host += demand.host;
        let Some(component) = demand.component.as_deref() else {
            return;
        };
        if demand.units > 0 {
            *self.units.entry(component.to_owned()).or_default() += demand.units;
        } else {
            self.exclusive.insert(component.to_owned(), request);
        }
    }
}

impl Scheduler {
    /// Lists every call of one request whose inputs are produced.
    ///
    /// A request is a set of calls with data dependencies, so a tick offers all
    /// of them and the lane ledger decides which ones fit. At most one call of
    /// each kind is ready at once: the cursors this returns against advance as
    /// calls are scheduled, so a unit group only becomes ready once its
    /// predecessor is submitted.
    fn ready_calls(&self, state: &MediaFlowState) -> Vec<PipelineStage> {
        let sampling = state.request.sampling;
        let produced = |product: &TensorRef| {
            !self
                .pending_operations
                .get(&state.request.request_id)
                .is_some_and(|ops| {
                    ops.iter()
                        .any(|op| op.operation.op_id == product.producer_op_id)
                })
        };
        if !state.text_encoding_scheduled {
            return vec![PipelineStage::TextEncoding];
        }
        if !state.latent_preparation_scheduled {
            return vec![PipelineStage::LatentPreparation];
        }
        if state.num_scheduled_steps < sampling.num_inference_steps {
            return vec![PipelineStage::Denoising];
        }

        // Consume completed decoder outputs first. Audio and video retain
        // independent readiness and capacity when the other branch is busy.
        let mut ready = Vec::new();
        if let Some((_, product)) = state.video_segments.get(&state.num_scheduled_video_chunks)
            && produced(product)
        {
            ready.push(PipelineStage::VideoEncoding);
        }
        if !state.audio_encoding_scheduled && state.audio.as_ref().is_some_and(produced) {
            ready.push(PipelineStage::AudioEncoding);
        }
        if state.num_scheduled_decode_chunks < sampling.num_decode_chunks {
            ready.push(PipelineStage::VideoDecoding);
        }
        if !state.audio_decoding_scheduled {
            ready.push(PipelineStage::AudioDecoding);
        }
        // Mux waits for completed writes, never just their scheduled counts.
        if !state.muxing_scheduled
            && state.audio_encoded
            && state.num_encoded_video_chunks == sampling.num_decode_chunks
        {
            ready.push(PipelineStage::Muxing);
        }
        ready
    }

    /// Returns the component whose lanes one media call occupies.
    fn media_component(&self, stage: PipelineStage) -> Option<String> {
        self.info.pipeline_components.get(&stage).cloned()
    }

    /// Returns a component's device lane capacity in media units.
    ///
    /// A component that distributes its work measures its lane in the units its
    /// ranks reconstruct together; every other component's lane is exclusive
    /// and admits one request at a time.
    fn device_lane_units(&self, component: &str) -> Option<u32> {
        self.info
            .components
            .iter()
            .find(|binding| binding.name == component)
            .and_then(|binding| {
                binding.config.distribution.as_ref().map(|_| {
                    (binding.config.ranks.len() * binding.config.units_per_rank.max(1)) as u32
                })
            })
    }

    /// Returns the lanes one media call occupies and how much of each.
    fn lane_demand(&self, stage: PipelineStage, units: u32) -> LaneDemand {
        let component = self.media_component(stage);
        let device_units = match stage {
            PipelineStage::VideoDecoding | PipelineStage::AudioDecoding => component
                .as_deref()
                .and_then(|name| self.device_lane_units(name))
                .map_or(0, |_| units.max(1)),
            _ => 0,
        };
        let host = match stage {
            PipelineStage::VideoEncoding | PipelineStage::AudioEncoding | PipelineStage::Muxing => {
                1
            }
            _ => 0,
        };
        LaneDemand {
            component,
            units: device_units,
            host,
        }
    }

    /// Accumulates the lanes the media calls in flight occupy.
    fn lane_occupancy(&self) -> LaneLedger {
        let mut ledger = LaneLedger::default();
        for (id, queue) in &self.pending_operations {
            for op in queue {
                let Computation::Pipeline(stage) = op.operation.code else {
                    continue;
                };
                let units = match &op.input {
                    InflightInput::Media {
                        decode: Some(range),
                        ..
                    } => range.max_units,
                    _ => 1,
                };
                ledger.occupy(&self.lane_demand(stage, units), *id);
            }
        }
        ledger
    }

    /// Returns whether the rank group that owns one call can accept a batch.
    fn stage_has_queue_capacity(&self, stage: PipelineStage) -> bool {
        self.info
            .pipeline_components
            .get(&stage)
            .is_some_and(|entry| {
                self.entry_candidates(Computation::Pipeline(stage), entry)
                    .any(|(worker, _, _)| self.executor.has_capacity(worker))
            })
    }

    /// Returns the media units one scheduled call of this kind would occupy.
    fn call_units(&self, state: &MediaFlowState, stage: PipelineStage) -> u32 {
        match stage {
            PipelineStage::VideoDecoding => self
                .media_component(stage)
                .as_deref()
                .and_then(|name| self.device_lane_units(name))
                .unwrap_or(1)
                .min(state.request.sampling.num_decode_chunks - state.num_scheduled_decode_chunks),
            _ => 1,
        }
    }

    /// Select eligible media requests and prepare their bounded computation inputs. Independent
    /// audio and video branches carry Tensor edges, without a state predecessor.
    pub(super) fn prepare_media_batch(&mut self) -> Option<ExecutionBatch> {
        // Every resident request offers the complete set of calls whose inputs
        // are produced, in arrival order, and each one is dispatched when the
        // lanes it occupies have free capacity. There is no per-kind rule and
        // no cap on how many calls of one request are in flight.
        let mut ledger = self.lane_occupancy();
        let mut candidates = Vec::new();
        for id in self.running_order.clone() {
            let Some(state) = self.media_state(id) else {
                continue;
            };
            if state.terminal_intent.is_terminal()
                || state.admission_state == WorkerRegistration::InFlight
            {
                continue;
            }
            for stage in self.ready_calls(state) {
                let demand = self.lane_demand(stage, self.call_units(state, stage));
                if !ledger.admits(&demand, id, self) || !self.stage_has_queue_capacity(stage) {
                    continue;
                }
                ledger.occupy(&demand, id);
                candidates.push((id, stage));
            }
        }
        if candidates.is_empty() {
            return None;
        }

        let batch_id = self.next_batch_id();
        let submit_at = Instant::now();
        let candidate_requests = candidates.iter().map(|(id, _)| *id).collect::<HashSet<_>>();
        let commands = self.take_commands(|command| {
            candidate_requests.contains(&command.request_key().request_id)
                || matches!(command, BatchCommand::Finish { .. })
        });
        let mut admissions = Vec::new();
        let mut logical_ops = Vec::with_capacity(candidates.len());
        for (request_index, (id, stage)) in candidates.into_iter().enumerate() {
            let op_id = ComputationId::new(
                batch_id,
                u32::try_from(request_index).expect("selected request count fits the IPC index"),
            );
            let work = Computation::Pipeline(stage);
            let entry = self.info.pipeline_components[&stage].clone();
            let state = self.media_state(id).expect("media candidate exists");
            let request_key = state.admission.request_key;
            let stateful = work.advances_state();
            let step = state.num_scheduled_steps;
            let last_step = stage == PipelineStage::Denoising
                && step + 1 == state.request.sampling.num_inference_steps;
            // Freeze the actual decode/write interval before advancing scheduled
            // counters. Completion consumes this same range from the submission.
            let decode = match stage {
                PipelineStage::VideoDecoding => {
                    let (_, bound, info) = self
                        .entry_candidates(work, &entry)
                        .next()
                        .expect("scheduled decoder has a configured owner");
                    let component = info
                        .components
                        .iter()
                        .find(|component| component.name == bound)
                        .expect("scheduled decoder has a loaded component");
                    let width = u32::try_from(component.config.ranks.len())
                        .expect("loaded component rank count fits the decode range");
                    Some(DecodeRange {
                        request_key,
                        op_id,
                        cursor: state.num_scheduled_decode_chunks,
                        max_units: width.min(
                            state.request.sampling.num_decode_chunks
                                - state.num_scheduled_decode_chunks,
                        ),
                    })
                }
                PipelineStage::VideoEncoding => Some(DecodeRange {
                    request_key,
                    op_id,
                    cursor: state.num_scheduled_video_chunks,
                    max_units: state.video_segments[&state.num_scheduled_video_chunks].0,
                }),
                PipelineStage::AudioDecoding | PipelineStage::AudioEncoding => Some(DecodeRange {
                    request_key,
                    op_id,
                    cursor: 0,
                    max_units: 1,
                }),
                _ => None,
            };
            let predicate = if stateful {
                self.pending_operations
                    .get(&id)
                    .and_then(|ops| {
                        ops.iter()
                            .find(|op| op.operation.op_id == state.predecessor)
                    })
                    .and_then(|op| op.operation.completion_output.as_ref())
                    .cloned()
            } else {
                None
            };
            let inputs = match stage {
                PipelineStage::LatentPreparation => vec![
                    state
                        .conditioning
                        .as_ref()
                        .expect("text encoding has declared its conditioning output")
                        .clone(),
                ],
                PipelineStage::VideoDecoding => vec![state.latents[0].clone()],
                PipelineStage::AudioDecoding => vec![state.latents[1].clone()],
                PipelineStage::VideoEncoding => {
                    let range = decode.as_ref().expect("video write has an input range");
                    vec![state.video_segments[&range.cursor].1.clone()]
                }
                PipelineStage::AudioEncoding => {
                    vec![state.audio.as_ref().expect("audio is ready").clone()]
                }
                _ => Vec::new(),
            };
            let mut buffers = Vec::new();
            let mut outputs = Vec::new();
            if stage == PipelineStage::TextEncoding
                || last_step
                || matches!(
                    stage,
                    PipelineStage::VideoDecoding | PipelineStage::AudioDecoding
                )
            {
                let count = if last_step { 2 } else { 1 };
                for index in 0..count {
                    let reserved = &state.allocations.tensors[&(entry.to_owned(), index)];
                    let mut shape_bound = reserved.shape_bound.clone();
                    let start = if stage == PipelineStage::VideoDecoding {
                        let range = decode.as_ref().expect("video decode has an input range");
                        shape_bound.dims[0] = DimBound::Static(range.max_units);
                        range.cursor
                    } else {
                        0
                    };
                    let product = TensorRef {
                        request_key,
                        producer_op_id: op_id,
                        output_index: index as u16,
                        generation: 1,
                        dtype: reserved.dtype,
                        shape_bound,
                    };
                    buffers.push(reserved.bind(&product, start));
                    outputs.push(product);
                }
            }
            let latent = if stateful {
                let (start_step, step_count) = if stage == PipelineStage::Denoising {
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
            let predecessor = stateful.then(|| state.predecessor);
            let completion_output = (outputs.is_empty() && stateful)
                .then(|| self.media_completion_product(request_key, op_id));
            let mut operation = ScheduledRequest {
                token_input: None,

                token_output: None,
                vision_input: None,
                latent_feature_input: None,
                encoder_output: None,
                latent_input: None,
                latent_output: None,
                image_input: None,
                image_output: None,
                transition_output: None,

                input_image: None,
                kv_input: None,
                kv_output: None,
                input_token_ids: Vec::new(),
                sampling_state: None,
                request_key,
                op_id,
                predecessor,
                entry: entry.to_owned(),
                code: work,
                bounds: Bounds::default(),
                inputs: inputs,
                outputs: outputs,
                completion_output,
                predicate: predicate,
                rng: None,
            };
            let state = self.media_state_mut(id).expect("media candidate exists");
            if state.admission_state == WorkerRegistration::Unsubmitted {
                admissions.push(state.admission.clone());
                state.admission_state = WorkerRegistration::InFlight;
            }
            match stage {
                PipelineStage::TextEncoding => {
                    state.conditioning = operation.outputs.first().cloned();
                    state.text_encoding_scheduled = true;
                }
                PipelineStage::LatentPreparation => state.latent_preparation_scheduled = true,
                PipelineStage::Denoising => state.num_scheduled_steps = step + 1,
                PipelineStage::VideoDecoding => {
                    let range = decode.as_ref().expect("video decode has an input range");
                    state.num_scheduled_decode_chunks += range.max_units;
                    state.video_segments.insert(
                        range.cursor,
                        (range.max_units, operation.outputs[0].clone()),
                    );
                }
                PipelineStage::AudioDecoding => {
                    state.audio_decoding_scheduled = true;
                    state.audio = Some(operation.outputs[0].clone());
                }
                PipelineStage::VideoEncoding => {
                    state.num_scheduled_video_chunks += decode
                        .as_ref()
                        .expect("video write has an input range")
                        .max_units;
                }
                PipelineStage::AudioEncoding => state.audio_encoding_scheduled = true,
                PipelineStage::Muxing => state.muxing_scheduled = true,
                PipelineStage::VisionEncoding
                | PipelineStage::LatentEncoding
                | PipelineStage::ImageDecoding => {
                    unreachable!("video scheduling selects only its fixed stages")
                }
            }
            if stateful {
                state.predecessor = op_id;
            }
            if last_step {
                state.latents = operation.outputs.clone();
            }
            let (worker, entry) = self.select_worker(&operation);
            operation.entry = entry;
            let placement = RequestPlacement {
                worker,
                block_tables: Vec::new(),
                new_cache_pages: Vec::new(),
                forward: uniserve_worker_ipc::ForwardBatch::default(),
                latent,
                decode,
                buffers,
            };
            self.register_inflight(
                operation.clone(),
                InflightInput::Media {
                    latent: placement.latent.clone(),
                    decode: placement.decode.clone(),
                },
                submit_at,
                0,
            );
            logical_ops.push((operation, placement));
        }
        let mut batch_commands = admissions
            .into_iter()
            .map(|request| BatchCommand::Start { request })
            .collect::<Vec<_>>();
        batch_commands.extend(commands);
        let batch = ExecutionBatch::new(batch_id, logical_ops, batch_commands, Vec::new());
        self.register_pending_batch(&batch, submit_at);
        Some(batch)
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
            .store(self.free_blocks(), Ordering::Relaxed);
        self.stats
            .general
            .in_flight
            .store(self.pending_batches.len(), Ordering::Relaxed);
        // drain the manager's event ring so it doesn't grow unbounded; the
        // counters below already aggregate it, but draining keeps memory bounded.
        if let Some(kv) = self.cache.as_ref() {
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
            .store(self.encoder_cache.stats.queries, Ordering::Relaxed);
        self.stats
            .encoder
            .cache_hits
            .store(self.encoder_cache.stats.hits, Ordering::Relaxed);
        self.stats
            .encoder
            .cached
            .store(self.encoder_cache.len(), Ordering::Relaxed);
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
                .pending_operations
                .get(&id)
                .into_iter()
                .flatten()
                .any(|op| op.operation.code == Computation::Pipeline(PipelineStage::Denoising))
    }

    /// Counts prompt tokens accepted or present in pending forward inputs.
    pub(super) fn num_scheduled_prompt_tokens(&self, id: RequestId) -> Option<u32> {
        let state = self.running.get(&id)?;
        Some(
            self.pending_operations
                .get(&id)
                .into_iter()
                .flatten()
                .filter(|pending| is_prompt_extend(&pending.operation))
                .fold(state.num_computed_prompt_tokens, |count, pending| {
                    count.saturating_add(
                        pending
                            .operation
                            .input_token_ids
                            .len()
                            .min(u32::MAX as usize) as u32,
                    )
                }),
        )
    }

    /// Computes only token extents needed for capacity and successor inputs.
    /// Pending work contributes its submitted maximum; accepted lengths remain
    /// request state and are never advanced by this scheduling calculation.
    pub(super) fn scheduled_token_lengths(&self, id: RequestId) -> Option<(u32, u32)> {
        let state = self.running.get(&id)?;
        let (mut logical, mut physical) = (state.logical_position, state.kv_visible_len);
        let mut image_index = state.num_ingested_images;
        let mut encoder_index = state.image_encoder_index;
        let mut feedback_index = state.feedback_encoder_index;
        for pending in self.pending_operations.get(&id).into_iter().flatten() {
            let operation = &pending.operation;
            match operation.code {
                Computation::Forward(ForwardMode::Prefill) if is_prompt_extend(operation) => {
                    let count = operation.input_token_ids.len().min(u32::MAX as usize) as u32;
                    logical = logical.saturating_add(count);
                    physical = physical.saturating_add(count);
                }
                Computation::Forward(ForwardMode::Prefill)
                    if consumes_image_features(operation) =>
                {
                    physical = physical.saturating_add(operation.bounds.max_tokens);
                    if is_feedback_computation(operation) {
                        feedback_index += 1;
                        if feedback_index == state.req.image_generation.feedback_encoders.len() {
                            logical = logical
                                .saturating_add(state.req.image_generation.num_feedback_positions);
                            feedback_index = 0;
                        }
                    } else {
                        let image = state.req.multimodal_inputs.images.get(image_index)?;
                        encoder_index += 1;
                        if encoder_index == image.encoders.len() {
                            logical = logical.saturating_add(image.num_positions);
                            encoder_index = 0;
                            image_index += 1;
                        }
                    }
                }
                Computation::Forward(ForwardMode::Prefill) => physical = physical.saturating_add(1),
                Computation::Forward(ForwardMode::Decode)
                | Computation::Forward(ForwardMode::Verify) => {
                    logical = logical.saturating_add(1);
                    physical = physical.saturating_add(1);
                }
                Computation::Pipeline(PipelineStage::ImageDecoding) => feedback_index = 0,
                _ => {}
            }
        }
        Some((logical, physical))
    }

    /// Last denoising step covered by submitted intervals, without accepting them.
    pub(super) fn num_scheduled_denoise_steps(&self, id: RequestId) -> Option<u16> {
        let state = self.running.get(&id)?;
        Some(self.pending_operations.get(&id).into_iter().flatten().fold(
            state.num_completed_denoise_steps,
            |steps, pending| match pending.operation.code {
                Computation::Pipeline(PipelineStage::LatentPreparation) => 0,
                Computation::Pipeline(PipelineStage::Denoising) => steps.saturating_add(
                    pending.operation.bounds.max_tokens.min(u32::from(u16::MAX)) as u16,
                ),
                _ => steps,
            },
        ))
    }

    /// Returns the next feedback encoder index and number of pending image completions.
    /// Only actual feature writes advance the ordered encoder inputs.
    pub(super) fn scheduled_feedback(&self, id: RequestId) -> Option<(usize, usize)> {
        let state = self.running.get(&id)?;
        let mut index = state.feedback_encoder_index;
        let mut images = 0;
        for pending in self.pending_operations.get(&id).into_iter().flatten() {
            let operation = &pending.operation;
            if operation.code == Computation::Pipeline(PipelineStage::ImageDecoding) {
                index = 0;
            } else if consumes_image_features(operation) && is_feedback_computation(operation) {
                index += 1;
                if index == state.req.image_generation.feedback_encoders.len() {
                    index = 0;
                    images += 1;
                }
            }
        }
        Some((index, images))
    }

    /// Cache conditioning is the latest admitted publication or the accepted request base.
    pub(super) fn kv_conditioning(&self, id: RequestId) -> Option<uniserve_worker_ipc::BufferId> {
        self.pending_operations
            .get(&id)
            .and_then(|pending| {
                pending
                    .iter()
                    .rev()
                    .find_map(|item| item.operation.kv_output)
            })
            .or_else(|| {
                self.running
                    .get(&id)
                    .and_then(|request| request.image_conditioning)
            })
    }

    /// A successor refers to the producer's registered output generation directly.
    pub(super) fn pending_output(
        &self,
        id: RequestId,
        select: impl for<'a> Fn(&'a ScheduledRequest) -> Option<&'a TensorRef>,
    ) -> Option<&TensorRef> {
        self.pending_operations
            .get(&id)?
            .iter()
            .rev()
            .find_map(|pending| select(&pending.operation))
    }

    /// Select the next computation from the last pending input, or from accepted
    /// request progress when no computation remains. No request state is replayed.
    pub(super) fn next_generation_phase(&self, id: RequestId) -> Option<Phase> {
        let state = self.running.get(&id)?;
        let pending = self
            .pending_operations
            .get(&id)
            .and_then(|queue| queue.back());
        let phase = match pending.map(|pending| &pending.operation) {
            Some(operation) => match operation.code {
                Computation::Forward(ForwardMode::Prefill) if is_prompt_extend(operation) => {
                    Phase::Prefill
                }
                Computation::Forward(ForwardMode::Decode)
                | Computation::Forward(ForwardMode::Verify) => Phase::Prefill,
                Computation::Forward(ForwardMode::Prefill)
                    if consumes_image_features(operation) =>
                {
                    if is_feedback_computation(operation) {
                        if self.scheduled_feedback(id)?.0 == 0 {
                            Phase::DecodeUnd
                        } else {
                            Phase::FeedbackEncode
                        }
                    } else {
                        let image = state
                            .req
                            .multimodal_inputs
                            .images
                            .get(state.num_ingested_images)?;
                        if state.image_encoder_index + 1 == image.encoders.len() {
                            Phase::Prefill
                        } else {
                            Phase::Encode
                        }
                    }
                }
                Computation::Forward(ForwardMode::Prefill) => Phase::PublishKv,
                Computation::Transfer(TransferMode::KvPublish) => Phase::PrepareGen,
                Computation::Pipeline(PipelineStage::LatentPreparation)
                | Computation::Pipeline(PipelineStage::Denoising) => Phase::DenoiseGen,
                Computation::Pipeline(PipelineStage::ImageDecoding) => Phase::FeedbackEncode,
                Computation::Pipeline(PipelineStage::VisionEncoding)
                | Computation::Pipeline(PipelineStage::LatentEncoding)
                    if is_feedback_computation(operation) =>
                {
                    Phase::FeedbackState
                }
                _ => state.phase,
            },
            None => state.phase,
        };
        Some(match phase {
            Phase::Prefill
                if self.num_scheduled_prompt_tokens(id)?
                    >= state.req.prompt_token_ids.len() as u32
                    && state.num_ingested_images >= state.req.multimodal_inputs.images.len() =>
            {
                Phase::DecodeUnd
            }
            Phase::DenoiseGen if self.num_scheduled_denoise_steps(id)? >= state.req.image.steps => {
                Phase::CommitGen
            }
            phase => phase,
        })
    }

    /// Returns the latest accepted state operation identity.
    pub(super) fn state_predecessor(&self, id: RequestId) -> Option<ComputationId> {
        let state = self.running.get(&id)?;
        state.worker_registered.then_some(())?;
        Some(state.last_state_op_id)
    }

    /// Returns the operation that precedes the next state advancement.
    pub(super) fn execution_predecessor(&self, id: RequestId) -> Option<ComputationId> {
        let operation = self
            .pending_operations
            .get(&id)?
            .iter()
            .rev()
            .find(|inflight| inflight.operation.advances_state())
            .map(|inflight| &inflight.operation);
        let Some(operation) = operation else {
            return self.state_predecessor(id);
        };
        // Accepted lengths are published in device request state; the successor
        // needs the producing computation identity while its host report is pending.
        Some(operation.op_id)
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
                            .filter(|state| state.request_epoch == buffer.owner.request_epoch)
                            .and_then(|state| state.allocations_mut().take_buffer(*buffer))
                    })
                    .or_else(|| {
                        self.retiring_requests
                            .get_mut(&id)
                            .filter(|state| state.request_key == buffer.owner)
                            .and_then(|state| state.buffers.remove(buffer))
                    });
                if let Some(allocation) = allocation {
                    self.free_allocation(allocation);
                }
            } else if let BatchCommand::Finish { request_key, .. } = command {
                let id = request_key.request_id;
                let Some(retiring) = self.retiring_requests.get(&id) else {
                    tracing::error!(
                        request_id = id.0,
                        request_epoch = request_key.request_epoch,
                        "close acknowledgement does not match a retiring request"
                    );
                    self.fatal = true;
                    continue;
                };
                if retiring.request_key != *request_key {
                    tracing::error!(
                        request_id = id.0,
                        request_epoch = request_key.request_epoch,
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
                    self.free_allocation(allocation);
                }
            }
        }
    }

    /// Whether a bounded successor can execute from its predecessor's device
    /// results. The submitted input must establish its required lengths and predicate.
    pub(super) fn can_queue_successor(&self, id: RequestId, target: Computation) -> bool {
        // Successor projection requires both live runtime state and an unresolved
        // predecessor whose device products carry the accepted execution result.
        let Some(state) = self.running.get(&id) else {
            return false;
        };
        let Some(queue) = self
            .pending_operations
            .get(&id)
            .filter(|queue| !queue.is_empty())
        else {
            return false;
        };

        // Bound speculation by worker capacity and exclude lineages whose
        // terminal or relay state makes the projection unsafe.
        if queue.len() >= self.info.max_unresolved_ops as usize
            || state.terminal_intent.is_terminal()
            || self.pending_finishes.contains_key(&id)
            || !Self::device_token_relay_eligible(state)
        {
            return false;
        }
        let Some(predecessor) = queue.back() else {
            return false;
        };

        if target == Computation::Transfer(TransferMode::KvPublish) {
            return false;
        }
        if matches!(
            state.req.image_generation.trigger,
            uniserve_core::ImageTrigger::RoundCloseThenSuffix { .. }
        ) {
            return false;
        }

        // Decode successors support both feedback continuation and ordinary
        // prompt/decode pipelining, using each predecessor's actual inputs.
        if target == Computation::Forward(ForwardMode::Decode) {
            let feedback_continuation = consumes_image_features(&predecessor.operation)
                && is_feedback_computation(&predecessor.operation)
                && self
                    .scheduled_feedback(id)
                    .is_some_and(|(index, _)| index == 0)
                && state.req.image_generation.sample_feedback_continuation
                && predecessor.operation.token_output.is_some();
            if feedback_continuation {
                return self
                    .pending_successor_code(id)
                    .is_some_and(|variant| variant == Computation::Forward(ForwardMode::Decode))
                    && state.num_generated_tokens.saturating_add(1) < state.req.max_und_tokens;
            }

            if !matches!(state.phase, Phase::Prefill | Phase::DecodeUnd)
                || (state.phase == Phase::Prefill && state.starts_gen_after_context())
                || queue.iter().any(|op| {
                    !matches!(
                        op.operation.code,
                        Computation::Forward(ForwardMode::Prefill)
                            | Computation::Forward(ForwardMode::Decode)
                    )
                })
            {
                return false;
            }
            return self
                .num_scheduled_prompt_tokens(id)
                .is_some_and(|count| count as usize >= state.req.prompt_token_ids.len())
                && state.num_ingested_images >= state.req.multimodal_inputs.images.len()
                && state.num_generated_tokens.saturating_add(queue.len())
                    < state.req.max_und_tokens;
        }

        // Other successor kinds require a completion predicate and an exact
        // projected physical variant match.
        predecessor.operation.completion_output.is_some()
            && self
                .pending_successor_code(id)
                .is_some_and(|variant| variant == target)
    }

    /// Determine the next pipelined computation from its submitted predecessor.
    pub(super) fn pending_successor_code(&self, id: RequestId) -> Option<Computation> {
        if !self.has_pending_operations(id) {
            return None;
        }
        let state = self.running.get(&id)?;
        if self.num_scheduled_prompt_tokens(id)? < state.req.prompt_token_ids.len() as u32
            || state.num_ingested_images < state.req.multimodal_inputs.images.len()
        {
            return None;
        }
        let last = &self.pending_operations.get(&id)?.back()?.operation;
        let chainable = if last.code == Computation::Pipeline(PipelineStage::ImageDecoding) {
            state.req.feeds_back_images()
                && state.req.image_generation.feedback_source
                    == Some(uniserve_core::FeedbackSource::DeviceProduct)
        } else if consumes_image_features(last)
            && is_feedback_computation(last)
            && self.scheduled_feedback(id)?.0 == 0
        {
            state.req.image_generation.sample_feedback_continuation
        } else {
            true
        };
        if !chainable {
            return None;
        }
        Some(match self.next_generation_phase(id)? {
            Phase::Prefill | Phase::DecodeUnd => Computation::Forward(ForwardMode::Decode),
            Phase::CloseKv | Phase::FeedbackState => Computation::Forward(ForwardMode::Prefill),
            Phase::PublishKv => Computation::Transfer(TransferMode::KvPublish),
            Phase::PrepareGen => Computation::Pipeline(PipelineStage::LatentPreparation),
            Phase::DenoiseGen => Computation::Pipeline(PipelineStage::Denoising),
            Phase::CommitGen => Computation::Pipeline(PipelineStage::ImageDecoding),
            Phase::FeedbackEncode => {
                let feedback = &state.req.image_generation;
                feedback.feedback_source.as_ref()?;
                match feedback
                    .feedback_encoders
                    .get(self.scheduled_feedback(id)?.0)?
                    .encoder
                {
                    ImageIngestStep::VaeEncode => {
                        Computation::Pipeline(PipelineStage::LatentEncoding)
                    }
                    ImageIngestStep::VitEncode => {
                        Computation::Pipeline(PipelineStage::VisionEncoding)
                    }
                }
            }
            Phase::Encode | Phase::IngestState => return None,
        })
    }

    /// Returns whether a successor can consume device-selected tokens before host observation.
    pub(super) fn device_token_relay_eligible(state: &RequestState) -> bool {
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
        // descendants; a matched stop suppresses output beyond the accepted
        // prefix and closes the physical request after its readers drain. Only
        // multi-token bad-word automata keep the request host-paced, because
        // their next mask depends on the not-yet-observed suffix.
        let sampling = &state.req.sampling;
        (!state.req.generates_images()
            || state.req.image_generation.trigger.direct_token().is_some())
            && sampling.bad_words_ids.iter().all(|word| word.len() == 1)
    }

    /// Returns whether the latest host-resolved token may remain the exact device input
    /// to the next decode. CPU stop and EOS decisions are complete before this
    /// check; a continuing request therefore names the same sampled token.
    pub(super) fn can_reuse_resolved_token_product(&self, id: RequestId) -> bool {
        self.running.get(&id).is_some_and(|state| {
            state.worker_registered
                && state.num_generated_tokens > 0
                && state.latest_token.as_ref().is_some()
                && state.is_replayable_text()
                && state.phase == Phase::DecodeUnd
                && !state.round_closing
                && (!state.req.generates_images()
                    || state.req.image_generation.trigger.direct_token().is_some())
        })
    }

    /// Returns whether the next operation can be scheduled.
    pub(super) fn can_schedule_next(&self, id: RequestId) -> bool {
        !self.pending_finishes.contains_key(&id)
            && self.output_window_ready(id)
            && self.running.get(&id).is_some_and(|state| {
                !state.speculative_chain_invalidated
                    && state.output.decoder_boundaries.len()
                        < self.info.max_unresolved_ops.max(1) as usize
                    && self.peek_next_operation_variant(id).is_none_or(|kind| {
                        self.worker_target(
                            RequestKey::new(self.engine_id, id, state.request_epoch),
                            kind,
                        )
                        .is_some()
                    })
            })
            && (!self.has_pending_operations(id)
                || self
                    .pending_successor_code(id)
                    .is_some_and(|target| self.can_queue_successor(id, target)))
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
            .pending_operations
            .get(&id)
            .into_iter()
            .flatten()
            .map(|pending| {
                operation_output_bound(pending.operation.code, &pending.operation.bounds)
            })
            .sum::<usize>();
        available
            .saturating_sub(reserved)
            .saturating_sub(OUTPUT_TERMINAL_RESERVE)
            >= self.next_output_bound(id)
    }

    /// Returns the output-size bound for the next operation.
    pub(super) fn next_output_bound(&self, id: RequestId) -> usize {
        match self.peek_next_operation_variant(id) {
            Some(
                Computation::Forward(ForwardMode::Prefill)
                | Computation::Forward(ForwardMode::Decode),
            ) => 4,
            Some(Computation::Pipeline(PipelineStage::Denoising)) => {
                usize::from(self.denoise_step_burst).saturating_add(2)
            }
            Some(Computation::Pipeline(PipelineStage::ImageDecoding)) => 3,
            Some(_) | None => 2,
        }
    }

    /// Retains submitted inputs and charges their domain and transfer credits.
    pub(super) fn register_inflight(
        &mut self,
        operation: ScheduledRequest,
        input: InflightInput,
        started: Instant,
        queue_us: u64,
    ) {
        let request_id = operation.request_key.request_id;
        let domain = self.stats.domains.get(operation.code);
        let active = domain.active_credits.fetch_add(1, Ordering::Relaxed) + 1;
        domain.peak_credits.fetch_max(active, Ordering::Relaxed);
        domain.launched_operations.fetch_add(1, Ordering::Relaxed);
        domain.queue_us.fetch_add(queue_us, Ordering::Relaxed);
        self.pending_operations
            .entry(request_id)
            .or_default()
            .push_back(InflightOp {
                operation,
                input,
                started,
            });
    }

    /// Records the domain backpressure.
    pub(super) fn record_domain_backpressure(&self, computation: uniserve_worker_ipc::Computation) {
        self.stats
            .domains
            .get(computation)
            .backpressure_events
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Accumulates timing once per public metric group in a returned batch.
    fn record_domain_run(stats: &super::stats::DomainStats, timing: TimingCounters) {
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
    }

    /// Records the domain completion.
    pub(super) fn record_domain_completion(
        &self,
        computation: uniserve_worker_ipc::Computation,
        status: OpStatus,
    ) {
        let stats = self.stats.domains.get(computation);
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
    pub(super) fn reclaim_domain_credit(
        &self,
        computation: uniserve_worker_ipc::Computation,
        failed: bool,
    ) {
        let stats = self.stats.domains.get(computation);
        let active = stats.active_credits.load(Ordering::Relaxed);
        if active == 0 {
            tracing::error!(
                ?computation,
                "domain operation credit accounting underflowed"
            );
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
        for inflight in self.pending_operations.values().flatten() {
            self.reclaim_domain_credit(inflight.operation.code, true);
        }
    }

    /// Stages one validated completion until earlier operations for the request are applied.
    pub(super) fn stage_completion(
        &mut self,
        mut record: uniserve_worker_ipc::RequestOutput,
        media: Result<Option<Arc<SharedMedia>>, String>,
    ) {
        let id = record.request_key.request_id;
        let op_id = record.op_id;
        let known = self.pending_operations.get(&id).is_some_and(|queue| {
            queue.iter().any(|inflight| {
                inflight.operation.request_key == record.request_key
                    && inflight.operation.op_id == op_id
            })
        });
        let duplicate = self
            .pending_completions
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
        let media = match media {
            Ok(media) => media,
            Err(message) => {
                self.trace_record(json!({
                    "event": "media_mapping_failed",
                    "request_id": record.request_key.request_id.0,
                    "op_id": record.op_id,
                    "detail": &message,
                }));
                let event = EngineCoreOutput::ArtifactUnavailable { message };
                if let Some(state) = self.media_state(id) {
                    let _ = state.event_tx.send(event);
                } else {
                    self.emit(id, event);
                }
                record.status = OpStatus::Error;
                record.error_code = Some(uniserve_worker_ipc::ErrorCode::ResourceExhausted);
                None
            }
        };
        let arrival_seq = self.next_arrival();
        self.pending_completions.entry(id).or_default().insert(
            op_id,
            PendingCompletion {
                record,
                media,
                arrival_seq,
            },
        );
    }

    /// Validate a stage result and advance only its completed request fields.
    pub(super) fn process_diffusion_result(
        &mut self,
        operation: ScheduledRequest,
        latent: Option<LatentParams>,
        decode: Option<DecodeRange>,
        record: uniserve_worker_ipc::RequestOutput,
        media: Option<Arc<SharedMedia>>,
    ) {
        let id = record.request_key.request_id;
        let Some(state) = self.media_state(id) else {
            return;
        };
        let mut consumed_products = Vec::new();
        let already_failed = matches!(state.terminal_intent, TerminalIntent::Failure(_));
        let media_output_valid =
            (operation.code == Computation::Pipeline(PipelineStage::Muxing)) == media.is_some();
        let step_valid = operation.code != Computation::Pipeline(PipelineStage::Denoising)
            || latent.as_ref().is_some_and(|interval| {
                record.num_completed_steps == interval.start_step + interval.step_count
            });
        let valid = record.status == OpStatus::Ok
            && record.request_key == operation.request_key
            && record.op_id == operation.op_id
            && media_output_valid
            && step_valid;
        if valid
            && let Some(state) = self.media_state_mut(id)
            && state.admission_state == WorkerRegistration::InFlight
        {
            state.admission_state = WorkerRegistration::Registered;
        }
        if !already_failed {
            if !valid {
                if let Some(state) = self.media_state_mut(id) {
                    state.terminal_intent =
                        TerminalIntent::Failure("media worker operation failed".to_string());
                }
            } else if let Some(state) = self.media_state_mut(id) {
                match operation.code {
                    Computation::Pipeline(PipelineStage::Denoising) => {
                        state.num_completed_steps = record.num_completed_steps;
                    }
                    Computation::Pipeline(PipelineStage::VideoEncoding) => {
                        let range = decode
                            .as_ref()
                            .expect("submitted media write includes its input range");
                        state.num_encoded_video_chunks += range.max_units;
                        if let Some((_, product)) = state.video_segments.remove(&range.cursor) {
                            consumed_products.push(product);
                        }
                    }
                    Computation::Pipeline(PipelineStage::AudioEncoding) => {
                        state.audio_encoded = true;
                        if let Some(product) = state.audio.take() {
                            consumed_products.push(product);
                        }
                    }
                    Computation::Pipeline(PipelineStage::Muxing) => state.muxed = true,
                    _ => {}
                }
                let phase = match operation.code {
                    Computation::Pipeline(PipelineStage::TextEncoding) => "preparing",
                    Computation::Pipeline(PipelineStage::LatentPreparation)
                    | Computation::Pipeline(PipelineStage::Denoising) => {
                        if state.num_completed_steps == state.request.sampling.num_inference_steps {
                            "decoding"
                        } else {
                            "denoising"
                        }
                    }
                    Computation::Pipeline(
                        PipelineStage::VideoDecoding
                        | PipelineStage::AudioDecoding
                        | PipelineStage::VideoEncoding
                        | PipelineStage::AudioEncoding,
                    ) => "decoding",
                    Computation::Pipeline(PipelineStage::Muxing) => "finalizing",
                    _ => unreachable!("media request completed a non-media computation"),
                };
                let _ = state.event_tx.send(EngineCoreOutput::MediaProgress {
                    phase: phase.to_owned(),
                    completed_steps: state.num_completed_steps,
                });
                if let Some(media) = media {
                    state.artifact = Some(ArtifactEvent {
                        media_kind: MediaKind::Video,
                        content_type: "video/mp4".to_string(),
                        media,
                    });
                }
            }
        }

        if !consumed_products.is_empty() {
            self.free_buffers(
                consumed_products
                    .into_iter()
                    .map(|product| product.buffer_id()),
            );
        }

        let terminal = self.media_state(id).and_then(|state| {
            if self.has_pending_operations(id) {
                return None;
            }
            if let TerminalIntent::Failure(message) = &state.terminal_intent {
                Some(DiffusionTerminal::Failed(message.clone()))
            } else if let TerminalIntent::Finish(reason) = &state.terminal_intent {
                Some(DiffusionTerminal::Finished(reason.clone()))
            } else if state.event_tx.is_closed() {
                Some(DiffusionTerminal::Finished(FinishReason::Cancelled))
            } else if state.muxed {
                let event = state.artifact.clone().map_or_else(
                    || DiffusionTerminal::Failed("media output was not finalized".to_string()),
                    DiffusionTerminal::Completed,
                );
                Some(event)
            } else {
                None
            }
        });
        if let Some(event) = terminal {
            self.finish_media(id, event);
        }
    }

    /// Emits terminal media output and releases all request-owned resources.
    pub(super) fn finish_media(&mut self, id: RequestId, event: DiffusionTerminal) {
        let Some(state) = self.take_media_state(id) else {
            return;
        };
        self.running_order.retain(|candidate| *candidate != id);
        match event {
            DiffusionTerminal::Completed(artifact) => {
                let _ = state.event_tx.send(EngineCoreOutput::Artifact(artifact));
                let _ = state.event_tx.send(EngineCoreOutput::Finished {
                    reason: FinishReason::Completed,
                    stop_reason: None,
                    prompt_tokens: state.request.prompt_token_ids.len(),
                    completion_tokens: 0,
                    images: 0,
                });
            }
            DiffusionTerminal::Failed(message) => {
                let _ = state.event_tx.send(EngineCoreOutput::Error { message });
            }
            DiffusionTerminal::Finished(reason) => {
                let _ = state.event_tx.send(EngineCoreOutput::Finished {
                    reason,
                    stop_reason: None,
                    prompt_tokens: state.request.prompt_token_ids.len(),
                    completion_tokens: 0,
                    images: 0,
                });
            }
        }
        match state.admission_state {
            WorkerRegistration::Unsubmitted => {
                state.allocations.free(self);
                return;
            }
            _ => {}
        }
        let request_key = state.admission.request_key;
        self.pending_commands.push_back(BatchCommand::Finish {
            request_key,
            retained_buffers: Vec::new(),
        });
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
        op_id: ComputationId,
    ) -> Option<(ScheduledRequest, InflightInput, Instant)> {
        let inflight = self.pop_pending_operation(request_key, op_id)?;
        self.reclaim_domain_credit(inflight.operation.code, false);
        Some((inflight.operation, inflight.input, inflight.started))
    }

    /// Revoke completed buffer identities while retaining allocations until release acknowledgement.
    pub(super) fn free_buffers(
        &mut self,
        buffers: impl IntoIterator<Item = uniserve_worker_ipc::BufferId>,
    ) {
        let mut released = HashSet::new();
        for buffer in buffers {
            if !released.insert(buffer) {
                continue;
            }
            if let Some(allocation) = self.take_encoder_buffer(buffer) {
                if let Some(previous) = self.pending_buffer_frees.insert(buffer, allocation) {
                    tracing::error!(?buffer, "buffer free identity was already pending");
                    self.free_allocation(previous);
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
            if matches!(code, Some("SchedulerBug" | "InvariantViolation")) {
                self.fatal = true;
                tracing::error!(worker = %failure.worker_id, ?code, "control-plane invariant failed: {error}");
            } else if code == Some("InputError") {
                tracing::info!(worker = %failure.worker_id, ?code, "Worker rejected request: {error}");
            } else {
                tracing::warn!(worker = %failure.worker_id, ?code, "Worker failed affected work: {error}");
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
        // Aggregate timings by the stable public metric groups. Multiple concrete
        // stages in one group count as one returned run, as do multiple requests.
        let mut returned_groups: [Option<(usize, TimingCounters)>; 3] = [None; 3];
        let mut invalid_result = false;

        // A completion can mutate state only while its logical batch remains owned
        // by the in-flight window.
        if !self.pending_batches.contains_key(&result_batch_id) {
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
                .pending_batches
                .get_mut(&result_batch_id)
                .is_some_and(|pending| pending.operations.remove(&identity));
            if !removed {
                tracing::error!(
                    batch_id = result_batch_id,
                    request_id = record.request_key.request_id.0,
                    op_id = ?record.op_id,
                    "executor returned a duplicate or unknown operation result"
                );
                invalid_result = true;
                continue;
            }
            let Some(computation) = self
                .pending_operations
                .get(&record.request_key.request_id)
                .and_then(|operations| {
                    operations
                        .iter()
                        .find(|inflight| inflight.operation.op_id == record.op_id)
                })
                .map(|inflight| inflight.operation.code)
            else {
                invalid_result = true;
                continue;
            };
            let entry = returned_groups[super::stats::ExecutionDomainStats::index(computation)]
                .get_or_insert((0, TimingCounters::default()));
            entry.0 += 1;
            entry.1.queued_us = entry.1.queued_us.max(record.timing_counters.queued_us);
            entry.1.device_us = entry.1.device_us.max(record.timing_counters.device_us);
            entry.1.copy_us = entry.1.copy_us.max(record.timing_counters.copy_us);
            entry.1.host_us = entry.1.host_us.max(record.timing_counters.host_us);
        }

        let operations_complete = self
            .pending_batches
            .get(&result_batch_id)
            .is_some_and(|batch| batch.operations.is_empty());
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
        for result in &report.results {
            let record = &result.output;
            if let Some(computation) = self
                .pending_operations
                .get(&record.request_key.request_id)
                .and_then(|operations| {
                    operations
                        .iter()
                        .find(|inflight| inflight.operation.op_id == record.op_id)
                })
                .map(|inflight| inflight.operation.code)
            {
                self.record_domain_completion(computation, record.status);
            }
        }
        for (index, returned) in returned_groups.into_iter().enumerate() {
            let Some((operation_count, timing)) = returned else {
                continue;
            };
            let (label, stats) = self.stats.domains.groups()[index];
            Self::record_domain_run(stats, timing);
            if let Some(trace) = domain_run_trace.as_mut() {
                trace.push(json!({
                    "result_group": index as u32 + 1,
                    "domain": label,
                    "operations": operation_count,
                    "execution": "domain_homogeneous",
                    "queue_us": timing.queued_us,
                    "device_us": timing.device_us,
                    "completion_us": timing.copy_us.saturating_add(timing.host_us),
                    "co_resident_us": 0,
                }));
            }
        }

        let pending = self
            .pending_batches
            .get_mut(&result_batch_id)
            .expect("validated batch remains owned until reconciliation");
        pending.worker_exec_us = report
            .worker_exec_us
            .into_iter()
            .fold(pending.worker_exec_us, u64::saturating_add);
        let batch_roundtrip_us = pending.started.elapsed().as_micros() as u64;
        let worker_us = if batch_complete {
            let pending = self
                .pending_batches
                .remove(&result_batch_id)
                .expect("completed batch is owned");
            for (index, command) in pending.commands.into_iter().enumerate() {
                let failed = report.command_results.iter().any(|result| {
                    result.command_index == index as u32
                        && result.outcome == crate::executor::CommandOutcome::Failed
                });
                if failed {
                    match command {
                        BatchCommand::Finish { .. } | BatchCommand::Free { .. } => {
                            self.pending_commands.push_back(command);
                        }
                        BatchCommand::Start { .. } => {}
                    }
                } else {
                    self.acknowledge_commands(std::slice::from_ref(&command));
                }
            }
            pending.worker_exec_us
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
            self.stage_completion(result.output, result.media);
        }

        let mut resolved_ops = trace_enabled.then(|| Vec::with_capacity(completion_count));
        let mut progress_ops = trace_enabled.then(|| Vec::with_capacity(completion_count));

        loop {
            let completions = self.take_ready_completions();
            if completions.is_empty() {
                break;
            }

            let mut to_resolve = Vec::with_capacity(completions.len());
            for completion in completions {
                let PendingCompletion {
                    record,
                    media,
                    arrival_seq,
                } = completion;
                let id = record.request_key.request_id;
                let op_id = record.op_id;
                let Some((operation, input, started)) =
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

                let operation_variant = operation.code;
                let roundtrip_us = started.elapsed().as_micros() as u64;

                // Media operations update their independent stage progress immediately;
                // generation operations continue through semantic validation.
                let (image_kv, start_step) = match input {
                    InflightInput::Media { latent, decode } => {
                        self.process_diffusion_result(operation, latent, decode, record, media);
                        continue;
                    }
                    InflightInput::Generation {
                        image_kv,
                        start_step,
                    } => (image_kv, start_step),
                };

                if self
                    .pending_finishes
                    .get(&id)
                    .is_some_and(|finish| finish.reason == FinishReason::Error)
                {
                    self.free_buffers(operation.output_buffers());
                    self.finish_pending_if_idle(id);
                    continue;
                }

                let discard_invalidated_descendant = self
                    .running
                    .get(&id)
                    .is_some_and(|state| state.speculative_chain_invalidated);
                if discard_invalidated_descendant {
                    let has_unresolved_descendants = self.has_pending_operations(id);
                    if let Some(state) = self.running.get_mut(&id) {
                        state.num_kv_blocks_sent = state
                            .num_kv_blocks_sent
                            .saturating_sub(operation.bounds.max_kv_pages as usize);
                        state.speculative_chain_invalidated = has_unresolved_descendants;
                    }
                    self.free_buffers(operation.output_buffers());
                    self.finish_pending_if_idle(id);
                    continue;
                }

                let sampled_token_ids_len = record.committed_tokens.len();
                let sampled_token_ids_last = record.committed_tokens.last().copied();
                if let Some(resolved_ops) = resolved_ops.as_mut() {
                    let image = media
                        .as_ref()
                        .and_then(|value| std::str::from_utf8(value.as_bytes()).ok());
                    let image_hw = image
                        .and_then(|png| validate_png_artifact(png, None))
                        .map(|metadata| (metadata.height, metadata.width));
                    resolved_ops.push(json!({
                        "request_id": id.0,
                        "op_id": op_id,
                        "operation_type": operation_variant.as_str(),
                        "computation": operation.code,
                        "operation": operation_trace(&operation),
                        "roundtrip_us": roundtrip_us,
                        "worker_queue_us": record.timing_counters.queued_us,
                        "device_us": record.timing_counters.device_us,
                        "completion_copy_us": record.timing_counters.copy_us,
                        "completion_ready_to_observed_us": record.timing_counters.host_us,
                        "sampled_token": sampled_token_ids_last.is_some(),
                        "sampled_token_ids_len": sampled_token_ids_len,
                        "sampled_token_ids_last": sampled_token_ids_last,
                        "flow_done": record.finish_flags.eos || record.finish_flags.length || record.finish_flags.stop,
                        "steps_completed": record.num_completed_steps,
                        "image_done": image.is_some(),
                        "image_hw": image_hw,
                        "kv_tokens": record.kv_visible_len,
                        "product_handle": record.product_generations.first().copied(),
                    }));
                }

                let Some(state) = self
                    .running
                    .get(&id)
                    .filter(|state| state.request_epoch == operation.request_key.request_epoch)
                else {
                    // Finish already owns retirement for a closed epoch, including
                    // outputs still being produced when cancellation arrived.
                    continue;
                };
                if let Err(error) = generation::validate_generation_result(
                    &operation,
                    image_kv,
                    start_step,
                    state,
                    &record,
                    media.as_deref(),
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
                        self.pending_operations.get(&id).is_some_and(|queue| {
                            queue.iter().any(|inflight| {
                                inflight.operation.predicate.as_ref().as_ref().is_some_and(
                                    |predicate| predicate.producer_op_id == operation.op_id,
                                )
                            })
                        });
                    if !completion_has_device_consumer {
                        let completed_predicates = operation
                            .completion_output
                            .iter()
                            .chain(operation.transition_output.iter())
                            .cloned()
                            .collect::<Vec<_>>();
                        self.free_buffers(
                            completed_predicates
                                .into_iter()
                                .map(|product| product.buffer_id()),
                        );
                    }
                }

                let semantic_blocked = self
                    .running
                    .get(&id)
                    .is_some_and(|state| state.terminal_intent.is_terminal())
                    || self.pending_finishes.contains_key(&id);
                if semantic_blocked {
                    if operation.code == Computation::Pipeline(PipelineStage::ImageDecoding) {
                        self.free_request_latent(id);
                    }
                    self.finish_pending_if_idle(id);
                    continue;
                }

                let progress_result = if record.status == OpStatus::Ok {
                    self.running
                        .get_mut(&id)
                        .map(|state| state.process_generation_result(&operation, &record))
                } else {
                    None
                };
                if let Some(Err(error)) = progress_result {
                    self.trace_record(json!({
                        "event": "generation_result_invalid",
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

                // Apply accepted progress directly; the output decoder separately
                // decides the user-visible prefix and stop-string boundary.
                let advanced = record.status == OpStatus::Ok && operation.advances_state();
                let latest_token = if advanced {
                    let token = operation.token_output.clone();
                    token
                } else {
                    None
                };
                let retain_device_token = !self.has_pending_operations(id);
                if let Some(state) = self.running.get_mut(&id)
                    && advanced
                {
                    state.last_state_op_id = record.op_id;
                    state.latest_token = if retain_device_token {
                        latest_token
                    } else {
                        None
                    };
                }

                // Reclaim resources whose lifetime ends at this transition before
                // making its public output eligible for resolution.
                let free_flow_prefix = operation_variant
                    == Computation::Pipeline(PipelineStage::Denoising)
                    && record.status == OpStatus::Ok
                    && self.running.get(&id).is_some_and(|state| {
                        record.num_completed_steps >= u32::from(state.req.image.steps)
                    });
                if operation_variant == Computation::Pipeline(PipelineStage::Denoising)
                    && record.status == OpStatus::Ok
                    && let Some(prefix) = self
                        .running
                        .get_mut(&id)
                        .and_then(|state| state.flow_prefix.as_mut())
                {
                    prefix.diffusion_finalized = true;
                }
                if operation.code == Computation::Pipeline(PipelineStage::ImageDecoding) {
                    self.free_request_latent(id);
                }
                if free_flow_prefix {
                    self.free_flow_prefix(id);
                }
                if matches!(
                    operation_variant,
                    Computation::Pipeline(PipelineStage::Denoising)
                        | Computation::Pipeline(PipelineStage::ImageDecoding)
                ) {
                    let consumed_latents =
                        operation.latent_input.iter().cloned().collect::<Vec<_>>();
                    if !consumed_latents.is_empty() {
                        self.free_buffers(
                            consumed_latents
                                .into_iter()
                                .map(|product| product.buffer_id()),
                        );
                    }
                }

                let priority = completion_priority(operation_variant);
                let public_tokens_before = self
                    .running
                    .get(&id)
                    .map_or(0, |state| state.output.tokens_sent);
                if record.status == OpStatus::Predicated {
                    let has_unresolved_descendants = self.has_pending_operations(id);
                    if let Some(state) = self.running.get_mut(&id) {
                        state.num_kv_blocks_sent = state
                            .num_kv_blocks_sent
                            .saturating_sub(operation.bounds.max_kv_pages as usize);
                        state.speculative_chain_invalidated = has_unresolved_descendants;
                    }
                    self.free_buffers(operation.output_buffers());
                    self.finish_pending_if_idle(id);
                } else {
                    to_resolve.push((
                        priority,
                        arrival_seq,
                        id,
                        operation,
                        record,
                        media,
                        public_tokens_before,
                    ));
                }
            }

            // Resolve semantic effects in lifecycle priority and stable arrival
            // order, independent of the executor's physical completion order.
            to_resolve.sort_by_key(|(priority, seq_index, ..)| (*priority, *seq_index));
            for (_priority, _seq_index, id, operation, record, media, public_tokens_before) in
                to_resolve
            {
                let token_operation = matches!(
                    operation.code,
                    Computation::Forward(ForwardMode::Prefill)
                        | Computation::Forward(ForwardMode::Decode)
                        | Computation::Forward(ForwardMode::Verify)
                );
                // Reserve output space before resolving a terminal token: finishing
                // must wait for the decoder's stop-string decision on that batch.
                let awaiting_decoder = token_operation
                    && self
                        .running
                        .get(&id)
                        .is_some_and(|state| !state.req.stop_strings.is_empty());
                if awaiting_decoder {
                    self.running
                        .get_mut(&id)
                        .unwrap()
                        .output
                        .decoder_boundaries
                        .push_back(usize::MAX);
                }
                if self.running.contains_key(&id) && !self.pending_finishes.contains_key(&id) {
                    self.resolve(id, operation, record, media.as_deref());
                }
                if awaiting_decoder {
                    if let Some(state) = self.running.get_mut(&id) {
                        if state.output.tokens_sent > public_tokens_before {
                            *state.output.decoder_boundaries.back_mut().unwrap() =
                                state.output.tokens_sent;
                        } else {
                            state.output.decoder_boundaries.pop_back();
                        }
                    }
                }
                self.finish_pending_if_idle(id);
                if let (Some(st), Some(progress_ops)) =
                    (self.running.get(&id), progress_ops.as_mut())
                {
                    progress_ops.push(json!({
                        "request_id": id.0,
                        "phase": st.phase,
                        "generated_tokens": st.num_generated_tokens,
                        "images_done": st.num_generated_images,
                        "image_id": st.image_id,
                        "steps_done": st.num_completed_denoise_steps,
                        "pos": st.logical_position,
                        "kvlen": st.kv_visible_len,
                        "next_token": st.next_token,
                        "text_since_image": st.text_tokens_since_image,
                        "gen_branch_pending": st.image_reservation_pending,
                        "context_round_closing": st.round_closing,
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
                "pending": self.waiting_order.len(),
                "in_flight": self.pending_batches.len(),
            }));
        }
    }

    /// Merges optional worker forward-pass counters into scheduler statistics.
    pub(super) fn record_worker_forward_stats(&self, stats: Option<&ForwardStats>) {
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
        let (ids, commands) = self.clear_failed_operations();
        self.acknowledge_commands(&commands);
        for id in ids {
            if self.media_state(id).is_some() {
                self.finish_media(id, DiffusionTerminal::Failed(msg.to_string()));
            } else if self.running.contains_key(&id) {
                self.emit(
                    id,
                    EngineCoreOutput::Error {
                        message: msg.to_string(),
                    },
                );
                self.finish(id, FinishReason::Error);
            }
        }
    }

    /// Reconciles failed work without resetting allocations owned by live consumers.
    fn fail_after_worker_failure(&mut self, loss: &WorkerFailure) {
        if let Some(cache) = &self.cache {
            for endpoint in &loss.endpoints {
                cache.block_pool.invalidate_source(endpoint);
            }
        }
        let mut requests = loss.requests.iter().copied().collect::<HashSet<_>>();
        let mut buffers = loss.buffers.iter().cloned().collect::<HashSet<_>>();
        loop {
            let before = (requests.len(), buffers.len());
            for batch in &self.pending_submissions {
                for (op, placement) in &batch.requests {
                    if (!loss.endpoints.is_empty() && placement.worker == loss.worker_id)
                        || op.input_buffers().any(|buffer| buffers.contains(&buffer))
                    {
                        requests.insert(op.request_key);
                    }
                    if requests.contains(&op.request_key) {
                        buffers.extend(op.output_buffers());
                    }
                }
            }
            if before == (requests.len(), buffers.len()) {
                break;
            }
        }
        let reclaimable = self.encoder_cache.invalidate_buffers(&buffers);
        self.free_buffers(reclaimable.into_iter().map(|product| product.buffer_id()));
        let mut retired = loss.retired.clone();
        let mut pending = VecDeque::new();
        for mut batch in std::mem::take(&mut self.pending_submissions) {
            retired.extend(
                batch
                    .retire_requests(&requests)
                    .into_iter()
                    .map(|(op, _)| (batch.id, op.request_key, op.op_id)),
            );
            if let Some(pending) = self.pending_batches.get_mut(&batch.id) {
                pending.commands = batch
                    .commands
                    .iter()
                    .filter(|command| !matches!(command, BatchCommand::Start { .. }))
                    .cloned()
                    .collect();
            }
            pending.push_back(batch);
        }
        for (batch_id, request, op) in retired {
            let Some(inflight) = self.retire_operation(batch_id, request, op) else {
                self.fatal = true;
                tracing::error!(
                    batch_id,
                    ?request,
                    ?op,
                    "Worker failure named an unknown pending operation"
                );
                return;
            };
            self.reclaim_domain_credit(inflight.operation.code, true);
        }
        for batch in pending {
            if batch.requests.is_empty() && batch.commands.is_empty() {
                self.pending_batches.remove(&batch.id);
            } else {
                self.pending_submissions.push_back(batch);
            }
        }
        self.pending_commands.retain_mut(|command| {
            if !requests.contains(&command.request_key()) {
                return true;
            }
            !matches!(command, BatchCommand::Start { .. })
        });
        let engine_id = self.engine_id;
        for request in requests {
            let id = request.request_id;
            let error_emitted = self
                .pending_finishes
                .get(&id)
                .is_some_and(|finish| finish.reason == FinishReason::Error);
            if let Some(state) = self.media_state_mut(id) {
                if state.admission.request_key == request {
                    state.terminal_intent = TerminalIntent::Failure(loss.message.clone());
                }
            } else if let Some(state) = self.running.get_mut(&id) {
                if request != RequestKey::new(engine_id, id, state.request_epoch) {
                    continue;
                }
                state.output.decoder_boundaries.clear();
                if !error_emitted {
                    self.emit(
                        id,
                        EngineCoreOutput::Error {
                            message: loss.message.clone(),
                        },
                    );
                }
                self.pending_finishes.insert(
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
                    .pending_operations
                    .get(&id)
                    .and_then(|queue| queue.front())
                    .filter(|inflight| inflight.operation.request_key == request)
                    .map(|inflight| inflight.operation.op_id)
                else {
                    break;
                };
                let Some(completion) = self
                    .pending_completions
                    .get_mut(&id)
                    .and_then(|values| values.remove(&op_id))
                else {
                    break;
                };
                let Some((operation, _, _)) = self.pop_inflight(request, op_id) else {
                    unreachable!("ready completion owns its operation");
                };
                drop(completion);
                self.free_buffers(operation.output_buffers());
            }
            if self
                .pending_completions
                .get(&id)
                .is_some_and(BTreeMap::is_empty)
            {
                self.pending_completions.remove(&id);
            }
            if self
                .media_state(id)
                .is_some_and(|state| state.admission.request_key == request)
            {
                if !self.has_pending_operations(id) {
                    self.finish_media(id, DiffusionTerminal::Failed(loss.message.clone()));
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
        let _ = self.clear_failed_operations();
        while let Some(id) = self.waiting_media_order.pop_front() {
            let submission = self
                .waiting_media
                .remove(&id)
                .expect("scheduler media order names runtime state");
            let _ = submission.event_tx.send(EngineCoreOutput::Error {
                message: message.to_string(),
            });
        }
        let media = self.media_ids();
        for id in media {
            self.finish_media(id, DiffusionTerminal::Failed(message.to_string()));
        }
        let ids = self.running.keys().copied().collect::<Vec<_>>();
        for id in ids {
            self.emit(
                id,
                EngineCoreOutput::Error {
                    message: message.to_string(),
                },
            );
            self.finish(id, FinishReason::Error);
        }
    }
}
