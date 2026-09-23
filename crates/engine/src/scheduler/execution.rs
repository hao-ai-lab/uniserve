//! Batch submission, completion application, and allocation reclamation.
//!
//! The loop keeps executor progress non-blocking until work is outstanding, then
//! parks with a bounded liveness deadline. Completion application validates
//! request and call identities before mutating runtime state.

use super::*;
use uniserve_worker_ipc::{CallCoordinates, ForwardMode, MediaCall, TransferMode};

impl Scheduler {
    /// Dispatches queued work while preserving order at every shared destination.
    /// The caller retains blocked destinations for the whole scheduling tick, so
    /// assembling more work cannot repeatedly retry an unavailable worker.
    fn dispatch_submissions(&mut self, blocked_workers: &mut HashSet<crate::WorkerId>) -> bool {
        let mut progressed = false;
        let pending_count = self.inflight.pending_submissions.len();
        for _ in 0..pending_count {
            let batch = self
                .inflight
                .pending_submissions
                .pop_front()
                .expect("pending count is fixed");
            let targets = self.placement.batch_workers(&batch);
            if !targets.is_disjoint(blocked_workers) {
                // A batch waiting on one destination also orders later work at
                // its other destinations; independent destinations can proceed.
                blocked_workers.extend(targets);
                self.inflight.pending_submissions.push_back(batch);
                continue;
            }
            match self.executor.submit(batch) {
                Ok(()) => progressed = true,
                Err(ExecutorSubmitError::WouldBlock(batch)) => {
                    blocked_workers.extend(targets);
                    self.inflight.pending_submissions.push_back(batch);
                }
                Err(ExecutorSubmitError::Failed(error)) => {
                    // Failure reconciliation owns the pending identities until
                    // every affected computation has been retired.
                    self.on_executor_error(error);
                    return true;
                }
            }
        }
        progressed
    }

    /// Returns whether any rank queue can still accept a batch.
    ///
    /// `queue_depth` bounds the batches in flight on one rank, not across the
    /// instance: a batch carries one computation to one component, so a media
    /// pass occupies a slot on each component it touches rather than a single
    /// shared slot. Assembling stops once every rank queue is full.
    fn a_rank_queue_admits(&self) -> bool {
        self.executor
            .info()
            .workers
            .iter()
            .any(|(worker, _)| self.executor.has_capacity(worker))
    }

    /// Releases the request latent.
    fn free_request_latent(&mut self, id: RequestId) {
        let allocation = self
            .running
            .get_mut(&id)
            .and_then(|state| state.allocations_mut().latent.take());
        if let Some(allocation) = allocation {
            self.storage.latent_pool.free(allocation);
        }
    }
    /// Advances scheduler and executor work, blocking only for an outstanding result.
    ///
    /// Returns whether the loop made progress or handled an executor outcome.
    pub fn step(&mut self, commands: &Receiver<Command>) -> bool {
        if self.drain_commands(commands) {
            self.abort_all_requests();
            let _ = self.executor.close();
            return false;
        }
        let progressed = self.step_nonblocking();
        if progressed || self.inflight.pending_batches.is_empty() {
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
            .store(self.inflight.pending_batches.len(), Ordering::Relaxed);
        self.publish_cache_stats();
        progressed
    }

    /// Drains ready work, admits requests, and fills every available executor slot.
    pub(super) fn refill_executor(&mut self) -> bool {
        let mut progressed = false;
        self.placement.affinity.retain(|(request, _), _| {
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
            if !self.a_rank_queue_admits() {
                break;
            }
            let batches = self.schedule_batches();
            if batches.is_empty() {
                break;
            }
            self.inflight.pending_submissions.extend(batches);
            progressed = true;
            self.dispatch_submissions(&mut blocked_workers);
            if self.fatal {
                return progressed;
            }
        }
        while self.inflight.pending_submissions.is_empty()
            && !self.inflight.pending_commands.is_empty()
            && self.a_rank_queue_admits()
        {
            let commands = self.take_commands(|_| true);
            if commands.is_empty() {
                break;
            }
            let batch = self.finish_generation_batch(
                ExecutionBatch::new(0, Vec::new(), commands, Vec::new()),
                Instant::now(),
            );
            self.inflight.pending_submissions.push_back(batch);
            progressed = true;
            self.dispatch_submissions(&mut blocked_workers);
            if self.fatal {
                return progressed;
            }
        }

        progressed
    }

    /// Selects the batches of one pass under the shared queue budget and
    /// existing family fairness. A pass yields one batch per `(kind,
    /// component)` it selected, each a single numerical call.
    fn schedule_batches(&mut self) -> Vec<ExecutionBatch> {
        if self.prefer_media {
            let batches = self.prepare_media_batches();
            if !batches.is_empty() {
                self.prefer_media = false;
                return batches;
            }
        }
        let batches = self.assemble();
        if batches.is_empty() {
            if self.fatal {
                return Vec::new();
            }
            let batches = self.prepare_media_batches();
            if !batches.is_empty() {
                self.prefer_media = false;
            }
            return batches;
        }
        self.prefer_media = true;
        batches
    }

    /// Allocates a request-scoped product reference for a terminal media result.
    pub(super) fn media_completion_product(
        &mut self,
        request_key: RequestKey,
        call_id: CallId,
    ) -> TensorRef {
        let generation = u32::try_from(self.next_product_generation.max(1))
            .expect("media product generation exhausted");
        self.next_product_generation = u64::from(generation).saturating_add(1);
        TensorRef {
            request_key,
            producer_call_id: call_id,
            output_index: 0,
            generation,
            dtype: DType::U8,
            shape_bound: ShapeBound::default(),
        }
    }
}

/// The calls that read a video call's products, or `None` for a call
/// outside the video call graph.
///
/// This is the same graph `ready_calls` walks and the input binding reads:
/// text encoding feeds latent preparation, which opens the denoising ladder;
/// each step feeds the next and the last feeds both decoders; a decoder's
/// media units feed their encoder, whose encoded unit rows feed the muxer,
/// which also assembles the encoded audio. The producing rank is told which
/// ranks read its products from this, because it cannot know on its own.
pub(crate) fn consuming_calls(media_call: MediaCall) -> Option<&'static [MediaCall]> {
    Some(match media_call {
        MediaCall::TextEncoding => &[MediaCall::LatentPreparation],
        MediaCall::LatentPreparation => &[MediaCall::Denoising],
        MediaCall::Denoising => &[
            MediaCall::Denoising,
            MediaCall::VideoDecoding,
            MediaCall::AudioDecoding,
        ],
        MediaCall::VideoDecoding => &[MediaCall::VideoEncoding],
        MediaCall::VideoEncoding => &[MediaCall::Muxing],
        MediaCall::AudioDecoding => &[MediaCall::AudioEncoding],
        MediaCall::AudioEncoding | MediaCall::Muxing => &[],
        MediaCall::VisionEncoding | MediaCall::LatentEncoding | MediaCall::ImageDecoding => {
            return None;
        }
    })
}

/// The lanes one media call occupies while it is in flight.
///
/// A component lane measured in media units carries `units`; every other
/// component lane is exclusive and admits one request at a time. `host_ranks`
/// names the host lanes the call places a task on, one slot on each.
struct LaneDemand {
    component: Option<(crate::WorkerId, String)>,
    units: u32,
    /// Host lanes this call occupies, each named by its worker and rank with
    /// the number of tasks the call places on it: one per media unit the
    /// rank encodes, one for any other host call.
    host_ranks: Vec<(crate::WorkerId, u32, u32)>,
}

/// Lane occupancy across the media calls in flight.
#[derive(Default)]
struct LaneLedger {
    exclusive: HashMap<(crate::WorkerId, String), RequestId>,
    units: HashMap<(crate::WorkerId, String), u32>,
    /// A host rank owns one host lane, so occupancy is counted per rank of
    /// each worker.
    host: HashMap<(crate::WorkerId, u32), u32>,
}

impl LaneLedger {
    /// Returns whether one request's call fits the lanes it occupies.
    fn admits(&self, demand: &LaneDemand, request: RequestId, scheduler: &Scheduler) -> bool {
        if demand.host_ranks.iter().any(|(worker, rank, tasks)| {
            let capacity = scheduler.host_lane_capacity(worker);
            let key = (worker.clone(), *rank);
            self.host.get(&key).copied().unwrap_or(0) + tasks > capacity
        }) {
            return false;
        }
        let Some((worker, component)) = demand.component.as_ref() else {
            return true;
        };
        if demand.units > 0 {
            let capacity = scheduler.device_lane_units(worker, component).unwrap_or(1);
            self.units
                .get(&(worker.clone(), component.clone()))
                .copied()
                .unwrap_or(0)
                + demand.units
                <= capacity
        } else {
            self.exclusive
                .get(&(worker.clone(), component.clone()))
                .is_none_or(|holder| *holder == request)
        }
    }

    /// Marks the lanes one request's call occupies until it completes.
    fn occupy(&mut self, demand: &LaneDemand, request: RequestId) {
        for (worker, rank, tasks) in &demand.host_ranks {
            *self.host.entry((worker.clone(), *rank)).or_default() += tasks;
        }
        let Some((worker, component)) = demand.component.as_ref() else {
            return;
        };
        let key = (worker.clone(), component.clone());
        if demand.units > 0 {
            *self.units.entry(key).or_default() += demand.units;
        } else {
            self.exclusive.insert(key, request);
        }
    }
}

impl Scheduler {
    /// The cursors of the completed encode rounds the muxer can take next:
    /// the contiguous run that starts where the units it holds end.
    fn ready_encode_rounds(state: &MediaFlowState) -> Vec<u32> {
        let mut rounds = Vec::new();
        let mut cursor = state.handed_units;
        while let Some(units) = state.encoded_ready.get(&cursor) {
            rounds.push(cursor);
            cursor += units;
        }
        rounds
    }

    /// Lists every call of one request whose inputs are produced.
    ///
    /// A request is a set of calls with data dependencies, so a tick offers all
    /// of them and the lane ledger decides which ones fit. At most one call of
    /// each kind is ready at once: the cursors this returns against advance as
    /// calls are scheduled, so a unit group only becomes ready once its
    /// predecessor is submitted.
    fn ready_calls(&self, state: &MediaFlowState) -> Vec<MediaCall> {
        let sampling = state.request.sampling;
        let produced = |product: &TensorRef| {
            !self
                .inflight
                .pending_calls
                .get(&state.request.request_id)
                .is_some_and(|calls| {
                    calls
                        .iter()
                        .any(|call| call.call.call_id == product.producer_call_id)
                })
        };
        if !state.text_encoding_scheduled {
            return vec![MediaCall::TextEncoding];
        }
        if !state.latent_preparation_scheduled {
            return vec![MediaCall::LatentPreparation];
        }
        if self.scheduled_steps(state.request.request_id, &state.denoising)
            < state.denoising.steps()
        {
            return vec![MediaCall::Denoising];
        }

        // Consume completed decoder outputs first. Audio and video retain
        // independent readiness and capacity when the other branch is busy.
        let mut ready = Vec::new();
        if let Some((_, product)) = state.decoded_units.get(&state.scheduled_encode_units)
            && produced(product)
        {
            ready.push(MediaCall::VideoEncoding);
        }
        if !state.audio_encoding_scheduled && state.audio.as_ref().is_some_and(produced) {
            ready.push(MediaCall::AudioEncoding);
        }
        if state.scheduled_decode_units < sampling.video_units {
            ready.push(MediaCall::VideoDecoding);
        }
        if !state.audio_decoding_scheduled {
            ready.push(MediaCall::AudioDecoding);
        }
        // The muxer consumes media units as they arrive: a muxing call takes
        // the encode rounds that have completed since the last one, in
        // order, and a final call carrying no units assembles the artifact
        // once every unit and the audio track are in. It waits for completed
        // writes, never just their scheduled counts.
        if !state.muxing_in_flight && !state.muxed {
            let next_round_ready = state.encoded_ready.contains_key(&state.handed_units);
            let everything_in = state.audio_encoded
                && state.handed_units == sampling.video_units
                && !state.final_muxing_scheduled;
            if next_round_ready || everything_in {
                ready.push(MediaCall::Muxing);
            }
        }
        ready
    }

    /// Returns the steps of a request's trajectory covered once its submitted
    /// calls complete, from their latent intervals.
    pub(super) fn scheduled_steps(&self, id: RequestId, progress: &Denoising) -> u32 {
        progress.scheduled(
            self.inflight
                .pending_calls
                .get(&id)
                .into_iter()
                .flatten()
                .filter_map(|pending| match pending.call.code {
                    CallKind::Media(media_call) => Some((media_call, pending.input.latent())),
                    _ => None,
                }),
        )
    }

    /// Returns the raster of the video the denoiser's samples decode to, as the
    /// video decoder declares its media unit output: units, frames, height,
    /// width and channels.
    fn video_raster(&self) -> (u32, u32) {
        let component = self.media_component(MediaCall::VideoDecoding);
        let dims = self
            .executor
            .info()
            .workers
            .iter()
            .flat_map(|(_, info)| &info.components)
            .find(|binding| Some(&binding.name) == component.as_ref())
            .and_then(|binding| binding.outputs.first())
            .map(|output| output.shape_bound.dims.clone())
            .unwrap_or_default();
        let fixed = |index: usize| match dims.get(index) {
            Some(DimBound::Static(value)) => *value,
            _ => 0,
        };
        (fixed(2), fixed(3))
    }

    /// Returns where a video request's samples live on its denoiser worker.
    ///
    /// A standalone denoiser's latent pool gives every request slot the same
    /// run of consecutive pages after its sentinel page, so the worker's
    /// advertised pool and slot count name the pages of `slot`. The raster is
    /// the video the samples decode to.
    fn sample_placement(&self, worker: &crate::WorkerId, slot: u32) -> LatentPlacement {
        let info = self
            .executor
            .info()
            .workers
            .iter()
            .find(|(id, _)| id == worker)
            .map(|(_, info)| info)
            .expect("a placed call's worker is configured");
        let pages = info.latent_pages.saturating_sub(1) / info.request_slots.max(1);
        let first = 1 + slot.saturating_sub(1) * pages;
        let (height, width) = self.video_raster();
        LatentPlacement {
            page_table: (first..first + pages).collect(),
            latent_units: pages * info.latent_page_units,
            height,
            width,
        }
    }

    /// Returns the component whose lanes one media call occupies.
    fn media_component(&self, media_call: MediaCall) -> Option<String> {
        self.info.media_components.get(&media_call).cloned()
    }

    /// Returns how many media units one call of this kind covers at once.
    ///
    /// A distributed component reconstructs one media unit per rank per round,
    /// so its rank count is the width of a round.
    fn component_width(&self, request: RequestKey, work: CallKind, component: &str) -> u32 {
        let owner = self
            .placement
            .affinity
            .get(&(request, component.to_owned()));
        let (_, bound, info) = self
            .placement
            .component_candidates(self.executor.as_ref(), work, component)
            .find(|(worker, _, _)| owner.is_none_or(|owner| *worker == owner))
            .expect("scheduled decoder has a configured owner");
        let component = info
            .components
            .iter()
            .find(|component| component.name == bound)
            .expect("scheduled decoder has a loaded component");
        let units_per_rank = if component.config.distribution.is_some() {
            component.config.units_per_rank.max(1)
        } else {
            1
        };
        u32::try_from(component.config.ranks.len().saturating_mul(units_per_rank))
            .expect("loaded component rank count fits the decode range")
    }

    /// Returns a component's device lane capacity in media units.
    ///
    /// A component that distributes its work measures its lane in the units its
    /// ranks reconstruct together; every other component's lane is exclusive
    /// and admits one request at a time.
    fn device_lane_units(&self, worker: &crate::WorkerId, component: &str) -> Option<u32> {
        self.executor
            .info()
            .workers
            .iter()
            .find(|(id, _)| id == worker)
            .and_then(|(_, info)| {
                info.components
                    .iter()
                    .find(|binding| binding.name == component)
            })
            .and_then(|binding| {
                binding.config.distribution.as_ref().map(|_| {
                    (binding.config.ranks.len() * binding.config.units_per_rank.max(1)) as u32
                })
            })
    }

    /// Returns the lanes one media call occupies and how much of each.
    fn lane_demand(&self, request: RequestKey, media_call: MediaCall, units: u32) -> LaneDemand {
        let component = self.media_component(media_call);
        let placed_component = component.as_ref().and_then(|component| {
            self.placement
                .affinity
                .get(&(request, component.clone()))
                .map(|worker| (worker.clone(), component.clone()))
        });
        let device_units = match media_call {
            // A device lane is occupied by a decode round. Encoding a media
            // unit is admitted on its rank's host lane, so it overlaps the
            // decode round that follows it rather than excluding it, and the
            // post-processing it begins with rides on the same admission.
            MediaCall::VideoDecoding | MediaCall::AudioDecoding => placed_component
                .as_ref()
                .and_then(|(worker, name)| self.device_lane_units(worker, name))
                .map_or(0, |_| units.max(1)),
            _ => 0,
        };
        let host_ranks = match media_call {
            MediaCall::VideoEncoding | MediaCall::AudioEncoding | MediaCall::Muxing => {
                self.host_lane_ranks(request, component.as_deref(), units)
            }
            _ => Vec::new(),
        };
        LaneDemand {
            component: placed_component,
            units: device_units,
            host_ranks,
        }
    }

    /// Returns the host lanes one call of a component occupies, each named by
    /// the worker holding the component and the rank within it, with the
    /// tasks the call places there.
    ///
    /// A distributed component's round runs on as many of its ranks as it has
    /// media units, each rank taking one task per unit it encodes; any other
    /// component runs one task on each of its ranks.
    fn host_lane_ranks(
        &self,
        request: RequestKey,
        component: Option<&str>,
        units: u32,
    ) -> Vec<(crate::WorkerId, u32, u32)> {
        let Some(component) = component else {
            return Vec::new();
        };
        let owner = self
            .placement
            .affinity
            .get(&(request, component.to_owned()));
        let Some((worker, info)) = self.executor.info().workers.iter().find(|(id, info)| {
            owner.is_none_or(|owner| id == owner)
                && info
                    .components
                    .iter()
                    .any(|binding| binding.name == component)
        }) else {
            return Vec::new();
        };
        let Some(binding) = info
            .components
            .iter()
            .find(|binding| binding.name == component)
        else {
            return Vec::new();
        };
        let ranks = &binding.config.ranks;
        let per_rank = binding.config.units_per_rank.max(1) as u32;
        let (width, tasks): (usize, Box<dyn Fn(usize) -> u32>) = match binding.config.distribution {
            Some(_) => (
                (units.max(1) as usize)
                    .div_ceil(per_rank as usize)
                    .min(ranks.len()),
                Box::new(move |position: usize| {
                    let start = position as u32 * per_rank;
                    units.max(1).saturating_sub(start).min(per_rank)
                }),
            ),
            None => (ranks.len(), Box::new(|_| 1)),
        };
        ranks[..width]
            .iter()
            .enumerate()
            .map(|(position, rank)| {
                (
                    worker.clone(),
                    u32::try_from(*rank).expect("a rank index fits the lane ledger"),
                    tasks(position),
                )
            })
            .collect()
    }

    /// Returns the host lane capacity of one worker's ranks.
    fn host_lane_capacity(&self, worker: &crate::WorkerId) -> u32 {
        self.executor
            .info()
            .workers
            .iter()
            .find(|(id, _)| id == worker)
            .map_or(1, |(_, info)| info.host_lane_capacity.max(1))
    }

    /// Accumulates the lanes the media calls in flight occupy.
    fn lane_occupancy(&self) -> LaneLedger {
        let mut ledger = LaneLedger::default();
        for (id, queue) in &self.inflight.pending_calls {
            for inflight in queue {
                let CallKind::Media(media_call) = inflight.call.code else {
                    continue;
                };
                let units = match &inflight.input {
                    InflightInput::Media {
                        decode: Some(range),
                        ..
                    } => range.max_units,
                    _ => 1,
                };
                ledger.occupy(
                    &self.lane_demand(inflight.call.request_key, media_call, units),
                    *id,
                );
            }
        }
        ledger
    }

    /// Returns whether the rank group that owns one call can accept a batch.
    fn call_has_queue_capacity(&self, request: RequestKey, media_call: MediaCall) -> bool {
        self.info
            .media_components
            .get(&media_call)
            .is_some_and(|component| {
                self.placement
                    .component_target(
                        self.executor.as_ref(),
                        request,
                        CallKind::Media(media_call),
                        component,
                    )
                    .is_some()
            })
    }

    /// Returns the media units one scheduled call of this kind would occupy.
    ///
    /// A decode round covers one media unit on each rank of its component, so
    /// it occupies that component's whole lane unless the track has fewer
    /// units left than the component has ranks.
    fn call_units(&self, state: &MediaFlowState, media_call: MediaCall) -> u32 {
        let width = || {
            self.media_component(media_call)
                .as_deref()
                .and_then(|name| {
                    let worker = self
                        .placement
                        .affinity
                        .get(&(state.admission.request_key, name.to_owned()))?;
                    self.device_lane_units(worker, name)
                })
                .unwrap_or(1)
        };
        match media_call {
            MediaCall::VideoDecoding => {
                width().min(state.request.sampling.video_units - state.scheduled_decode_units)
            }
            MediaCall::AudioDecoding => width(),
            // An encode round covers the media units the decode round produced.
            MediaCall::VideoEncoding => state
                .decoded_units
                .get(&state.scheduled_encode_units)
                .map_or(1, |(units, _)| *units),
            _ => 1,
        }
    }

    /// Select eligible media requests and prepare their bounded computation inputs. Independent
    /// audio and video branches carry Tensor edges, without a state predecessor.
    pub(super) fn prepare_media_batches(&mut self) -> Vec<ExecutionBatch> {
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
            for media_call in self.ready_calls(state) {
                let demand = self.lane_demand(
                    state.admission.request_key,
                    media_call,
                    self.call_units(state, media_call),
                );
                if !ledger.admits(&demand, id, self)
                    || !self.call_has_queue_capacity(state.admission.request_key, media_call)
                {
                    continue;
                }
                ledger.occupy(&demand, id);
                candidates.push((id, media_call));
            }
        }
        if candidates.is_empty() {
            return Vec::new();
        }

        let submit_at = Instant::now();
        let candidate_requests = candidates.iter().map(|(id, _)| *id).collect::<HashSet<_>>();
        let commands = self.take_commands(|command| {
            candidate_requests.contains(&command.request_key().request_id)
                || matches!(command, BatchCommand::Finish { .. })
        });
        // One batch per media call: a batch is one numerical call on one component,
        // so a rank receives it as a single homogeneous group. Calls keep the
        // order in which their first call was selected.
        let mut call_order = Vec::new();
        let mut call_batches: HashMap<MediaCall, (u64, Vec<(Call, RequestPlacement)>)> =
            HashMap::new();
        for (_, media_call) in &candidates {
            if !call_batches.contains_key(media_call) {
                call_batches.insert(*media_call, (self.inflight.next_batch_id(), Vec::new()));
                call_order.push(*media_call);
            }
        }
        let mut admissions = Vec::new();
        for (id, media_call) in candidates.into_iter() {
            let (batch_id, request_index) = {
                let (batch_id, calls) = &call_batches[&media_call];
                (*batch_id, calls.len())
            };
            let call_id = CallId::new(
                batch_id,
                u32::try_from(request_index).expect("selected request count fits the IPC index"),
            );
            let work = CallKind::Media(media_call);
            let component = self.info.media_components[&media_call].clone();
            let state = self.media_state(id).expect("media candidate exists");
            let request_key = state.admission.request_key;
            let stateful = work.advances_state();
            // A video request advances its ladder one step per call.
            let step = self.scheduled_steps(id, &state.denoising);
            let last_step = media_call == MediaCall::Denoising && state.denoising.ends(step, 1);
            // Freeze the actual decode/write interval before advancing scheduled
            // counters. Completion consumes this same range from the submission.
            let decode = match media_call {
                MediaCall::VideoDecoding => Some(DecodeRange {
                    request_key,
                    call_id,
                    cursor: state.scheduled_decode_units,
                    max_units: self
                        .component_width(request_key, work, &component)
                        .min(state.request.sampling.video_units - state.scheduled_decode_units),
                }),
                MediaCall::VideoEncoding => Some(DecodeRange {
                    request_key,
                    call_id,
                    cursor: state.scheduled_encode_units,
                    max_units: state.decoded_units[&state.scheduled_encode_units].0,
                }),
                MediaCall::AudioDecoding => Some(DecodeRange {
                    request_key,
                    call_id,
                    cursor: 0,
                    max_units: self.component_width(request_key, work, &component),
                }),
                MediaCall::AudioEncoding => Some(DecodeRange {
                    request_key,
                    call_id,
                    cursor: 0,
                    max_units: 1,
                }),
                _ => None,
            };
            let predicate = if stateful {
                self.inflight
                    .pending_calls
                    .get(&id)
                    .and_then(|calls| {
                        calls
                            .iter()
                            .find(|call| call.call.call_id == state.predecessor)
                    })
                    .and_then(|call| call.call.completion_output.as_ref())
                    .cloned()
            } else {
                None
            };
            let inputs = match media_call {
                MediaCall::LatentPreparation => vec![
                    state
                        .conditioning
                        .as_ref()
                        .expect("text encoding has declared its conditioning output")
                        .clone(),
                ],
                MediaCall::VideoDecoding => vec![state.latents[0].clone()],
                MediaCall::AudioDecoding => vec![state.latents[1].clone()],
                MediaCall::VideoEncoding => {
                    let range = decode.as_ref().expect("video write has an input range");
                    vec![state.decoded_units[&range.cursor].1.clone()]
                }
                MediaCall::AudioEncoding => {
                    vec![state.audio.as_ref().expect("audio is ready").clone()]
                }
                // The muxer takes the completed encode rounds that follow the
                // last it was handed, in media unit order; the final call
                // carries none.
                MediaCall::Muxing => Self::ready_encode_rounds(state)
                    .into_iter()
                    .map(|cursor| state.encoded_units[&cursor].1.clone())
                    .collect(),
                _ => Vec::new(),
            };
            let mut outputs = Vec::new();
            let mut output_starts = Vec::new();
            if media_call == MediaCall::TextEncoding
                || last_step
                || matches!(
                    media_call,
                    MediaCall::VideoDecoding | MediaCall::AudioDecoding | MediaCall::VideoEncoding
                )
            {
                // A component declares its products in the order its methods
                // do: the denoiser's last step reserves both latents, every
                // other call its component's first product.
                let declared: &[u32] = if last_step { &[0, 1] } else { &[0] };
                for &index in declared {
                    let reserved = &state.allocations.tensors[&(component.to_owned(), index)];
                    let mut shape_bound = reserved.shape_bound.clone();
                    // A media unit round writes its own slice of the track's
                    // reservation, whether the slice holds decoded media units
                    // or the rows encoded from them.
                    let start = if matches!(
                        media_call,
                        MediaCall::VideoDecoding | MediaCall::VideoEncoding
                    ) {
                        let range = decode.as_ref().expect("a media round has a unit range");
                        shape_bound.dims[0] = DimBound::Static(range.max_units);
                        range.cursor
                    } else {
                        0
                    };
                    let product = TensorRef {
                        request_key,
                        producer_call_id: call_id,
                        output_index: index as u16,
                        generation: 1,
                        dtype: reserved.dtype,
                        shape_bound,
                    };
                    outputs.push(product);
                    output_starts.push(start);
                }
            }
            // The step interval of a call that advances the trajectory; its
            // pages follow from the worker the call is placed on.
            let interval = stateful.then_some(if media_call == MediaCall::Denoising {
                (step, 1)
            } else {
                (0, 0)
            });
            let completion_output = (outputs.is_empty() && stateful)
                .then(|| self.media_completion_product(request_key, call_id));
            let mut call = Call {
                consumer_slots: Vec::new(),
                token_input: None,
                coordinates: CallCoordinates {
                    logical_position: 0,
                    kv_visible_len: 0,
                    kv_computed_len: 0,
                    flow_step: step,
                },

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
                call_id,
                component: component.to_owned(),
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
            for (product, start) in call.outputs.iter().zip(&output_starts) {
                let replaced = state.buffer_bindings.insert(
                    product.buffer_id(),
                    (component.clone(), u32::from(product.output_index), *start),
                );
                debug_assert!(replaced.is_none(), "media buffer identity was reused");
            }
            match media_call {
                MediaCall::TextEncoding => {
                    state.conditioning = call.outputs.first().cloned();
                    state.text_encoding_scheduled = true;
                }
                MediaCall::LatentPreparation => state.latent_preparation_scheduled = true,
                // Submitted steps are derived from the call's latent interval.
                MediaCall::Denoising => {}
                MediaCall::VideoDecoding => {
                    let range = decode.as_ref().expect("video decode has an input range");
                    state.scheduled_decode_units += range.max_units;
                    state
                        .decoded_units
                        .insert(range.cursor, (range.max_units, call.outputs[0].clone()));
                }
                MediaCall::AudioDecoding => {
                    state.audio_decoding_scheduled = true;
                    state.audio = Some(call.outputs[0].clone());
                }
                MediaCall::VideoEncoding => {
                    let range = decode.as_ref().expect("video write has an input range");
                    state.scheduled_encode_units += range.max_units;
                    state
                        .encoded_units
                        .insert(range.cursor, (range.max_units, call.outputs[0].clone()));
                }
                MediaCall::AudioEncoding => state.audio_encoding_scheduled = true,
                MediaCall::Muxing => {
                    let rounds = Self::ready_encode_rounds(state);
                    if rounds.is_empty() {
                        state.final_muxing_scheduled = true;
                    }
                    for cursor in rounds {
                        let units = state
                            .encoded_ready
                            .remove(&cursor)
                            .expect("a ready encode round was counted");
                        let (_, product) = state
                            .encoded_units
                            .remove(&cursor)
                            .expect("a ready encode round has its product");
                        state.handed_units += units;
                        state.muxing_inputs.push(product);
                    }
                    state.muxing_in_flight = true;
                }
                MediaCall::VisionEncoding
                | MediaCall::LatentEncoding
                | MediaCall::ImageDecoding => {
                    unreachable!("video scheduling selects only its fixed calls")
                }
            }
            if stateful {
                state.predecessor = call_id;
            }
            if last_step {
                state.latents = call.outputs.clone();
            }
            let (worker, selected_component) =
                self.placement
                    .select_worker(self.executor.as_ref(), &self.info, &call);
            call.component = selected_component;
            let state = self.media_state(id).expect("selected media request exists");
            let mut bound = HashSet::new();
            let buffers = call
                .buffer_inputs()
                .chain(call.buffer_outputs())
                .filter(|product| bound.insert(product.buffer_id()))
                .map(|product| {
                    let (component, index, start) = &state.buffer_bindings[&product.buffer_id()];
                    state.allocations.tensors[&(component.clone(), *index)]
                        .bind(product, *start, &worker)
                })
                .collect();
            let request_pool_idx = self
                .media_state(id)
                .expect("selected media request exists")
                .allocations
                .request_slot(&worker);
            let latent = interval.map(|(start_step, step_count)| {
                Denoising::params(
                    request_key,
                    call_id,
                    &self.sample_placement(&worker, request_pool_idx),
                    start_step,
                    step_count,
                )
            });
            let placement = RequestPlacement {
                worker,
                request_pool_idx: Some(request_pool_idx),
                block_tables: Vec::new(),
                new_cache_pages: Vec::new(),
                forward: uniserve_worker_ipc::ForwardBatch::default(),
                latent,
                decode,
                buffers,
            };
            self.register_inflight(
                call.clone(),
                InflightInput::Media {
                    latent: placement.latent.clone(),
                    decode: placement.decode.clone(),
                },
                0,
            );
            call_batches
                .get_mut(&media_call)
                .expect("media_call batch exists")
                .1
                .push((call, placement));
        }

        // A request's admission travels with the first call that uses it, and
        // the round's retirements travel with the last batch so no earlier call
        // loses the state it still reads.
        let mut starts: HashMap<MediaCall, Vec<BatchCommand>> = HashMap::new();
        for request in admissions {
            let owner = call_order
                .iter()
                .copied()
                .find(|media_call| {
                    call_batches[media_call]
                        .1
                        .iter()
                        .any(|(call, _)| call.request_key == request.request_key)
                })
                .expect("admitted request has a selected call");
            starts
                .entry(owner)
                .or_default()
                .push(BatchCommand::Start { request });
        }
        let mut batches = Vec::with_capacity(call_order.len() + 1);
        for media_call in call_order {
            let (batch_id, calls) = call_batches
                .remove(&media_call)
                .expect("media_call batch exists");
            let batch_commands = starts.remove(&media_call).unwrap_or_default();
            let batch = ExecutionBatch::new(batch_id, calls, batch_commands, Vec::new());
            self.inflight.register_pending_batch(&batch, submit_at);
            batches.push(batch);
        }
        // Retirement is its own batch, submitted after the calls of this pass.
        // Its result is the retirement acknowledgement, so no call's result
        // waits for storage the round no longer needs.
        if !commands.is_empty() {
            let batch = ExecutionBatch::new(
                self.inflight.next_batch_id(),
                Vec::new(),
                commands,
                Vec::new(),
            );
            self.inflight.register_pending_batch(&batch, submit_at);
            batches.push(batch);
        }
        batches
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
            .store(self.storage.free_blocks(), Ordering::Relaxed);
        self.stats
            .general
            .in_flight
            .store(self.inflight.pending_batches.len(), Ordering::Relaxed);
        // drain the manager's event ring so it doesn't grow unbounded; the
        // counters below already aggregate it, but draining bounds queued reports.
        if let Some(kv) = self.storage.cache.as_ref() {
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
            .store(self.storage.encoder_cache.stats.queries, Ordering::Relaxed);
        self.stats
            .encoder
            .cache_hits
            .store(self.storage.encoder_cache.stats.hits, Ordering::Relaxed);
        self.stats
            .encoder
            .cached
            .store(self.storage.encoder_cache.len(), Ordering::Relaxed);
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
                .pending_calls
                .get(&id)
                .into_iter()
                .flatten()
                .any(|call| call.call.code == CallKind::Media(MediaCall::Denoising))
    }

    /// Counts prompt tokens accepted or present in pending forward inputs.
    pub(super) fn num_scheduled_prompt_tokens(&self, id: RequestId) -> Option<u32> {
        let state = self.running.get(&id)?;
        Some(
            self.inflight
                .pending_calls
                .get(&id)
                .into_iter()
                .flatten()
                .filter(|pending| is_prompt_extend(&pending.call))
                .fold(state.num_computed_prompt_tokens, |count, pending| {
                    count.saturating_add(
                        pending.call.input_token_ids.len().min(u32::MAX as usize) as u32
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
        for pending in self.inflight.pending_calls.get(&id).into_iter().flatten() {
            let call = &pending.call;
            match call.code {
                CallKind::Forward(ForwardMode::Prefill) if is_prompt_extend(call) => {
                    let count = call.input_token_ids.len().min(u32::MAX as usize) as u32;
                    logical = logical.saturating_add(count);
                    physical = physical.saturating_add(count);
                }
                CallKind::Forward(ForwardMode::Prefill) if consumes_image_features(call) => {
                    physical = physical.saturating_add(call.bounds.max_tokens);
                    if is_feedback_computation(call) {
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
                CallKind::Forward(ForwardMode::Prefill) => physical = physical.saturating_add(1),
                CallKind::Forward(ForwardMode::Decode) | CallKind::Forward(ForwardMode::Verify) => {
                    logical = logical.saturating_add(1);
                    physical = physical.saturating_add(1);
                }
                CallKind::Media(MediaCall::ImageDecoding) => feedback_index = 0,
                _ => {}
            }
        }
        Some((logical, physical))
    }

    /// The coordinates the next call of this request executes at.
    ///
    /// Each value is the request's own accepted state projected through the
    /// calls already submitted, so a rank receives them instead of chaining
    /// them from its predecessors. A submitted forward collapses the computed
    /// extent onto the visible one: it initializes exactly the tokens it makes
    /// visible. Only a verifier leaves the two apart, and its acceptance
    /// resolves before any successor is scheduled.
    pub(super) fn projected_coordinates(&self, id: RequestId) -> Option<CallCoordinates> {
        let state = self.running.get(&id)?;
        let (logical_position, kv_visible_len) = self.scheduled_token_lengths(id)?;
        let forward_submitted = self
            .inflight
            .pending_calls
            .get(&id)
            .into_iter()
            .flatten()
            .any(|pending| matches!(pending.call.code, CallKind::Forward(_)));
        Some(CallCoordinates {
            logical_position,
            kv_visible_len,
            kv_computed_len: if forward_submitted {
                kv_visible_len
            } else {
                state.kv_computed_len.max(kv_visible_len)
            },
            flow_step: self.num_scheduled_denoise_steps(id)?,
        })
    }

    /// Last denoising step covered by submitted intervals, without accepting them.
    pub(super) fn num_scheduled_denoise_steps(&self, id: RequestId) -> Option<u32> {
        let state = self.running.get(&id)?;
        Some(self.scheduled_steps(id, &state.denoising))
    }

    /// Returns the next feedback encoder index and number of pending image completions.
    /// Only actual feature writes advance the ordered encoder inputs.
    pub(super) fn scheduled_feedback(&self, id: RequestId) -> Option<(usize, usize)> {
        let state = self.running.get(&id)?;
        let mut index = state.feedback_encoder_index;
        let mut images = 0;
        for pending in self.inflight.pending_calls.get(&id).into_iter().flatten() {
            let call = &pending.call;
            if call.code == CallKind::Media(MediaCall::ImageDecoding) {
                index = 0;
            } else if consumes_image_features(call) && is_feedback_computation(call) {
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
        self.inflight
            .pending_calls
            .get(&id)
            .and_then(|pending| pending.iter().rev().find_map(|item| item.call.kv_output))
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
        select: impl for<'a> Fn(&'a Call) -> Option<&'a TensorRef>,
    ) -> Option<&TensorRef> {
        self.inflight
            .pending_calls
            .get(&id)?
            .iter()
            .rev()
            .find_map(|pending| select(&pending.call))
    }

    /// Select the next computation from the last pending input, or from accepted
    /// request progress when no computation remains. No request state is replayed.
    pub(super) fn next_generation_phase(&self, id: RequestId) -> Option<Phase> {
        let state = self.running.get(&id)?;
        let pending = self
            .inflight
            .pending_calls
            .get(&id)
            .and_then(|queue| queue.back());
        let phase = match pending.map(|pending| &pending.call) {
            Some(call) => match call.code {
                CallKind::Forward(ForwardMode::Prefill) if is_prompt_extend(call) => Phase::Prefill,
                CallKind::Forward(ForwardMode::Decode) | CallKind::Forward(ForwardMode::Verify) => {
                    Phase::Prefill
                }
                CallKind::Forward(ForwardMode::Prefill) if consumes_image_features(call) => {
                    if is_feedback_computation(call) {
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
                CallKind::Forward(ForwardMode::Prefill) => Phase::PublishKv,
                CallKind::Transfer(TransferMode::KvPublish) => Phase::PrepareGen,
                CallKind::Media(MediaCall::LatentPreparation)
                | CallKind::Media(MediaCall::Denoising) => Phase::DenoiseGen,
                CallKind::Media(MediaCall::ImageDecoding) => Phase::FeedbackEncode,
                CallKind::Media(MediaCall::VisionEncoding)
                | CallKind::Media(MediaCall::LatentEncoding)
                    if is_feedback_computation(call) =>
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
            Phase::DenoiseGen
                if self.num_scheduled_denoise_steps(id)? >= state.denoising.steps() =>
            {
                Phase::CommitGen
            }
            phase => phase,
        })
    }

    /// Returns the latest accepted state call identity.
    pub(super) fn state_predecessor(&self, id: RequestId) -> Option<CallId> {
        let state = self.running.get(&id)?;
        state.worker_registered.then_some(())?;
        Some(state.last_state_call_id)
    }

    /// Returns the call that precedes the next state advancement.
    pub(super) fn execution_predecessor(&self, id: RequestId) -> Option<CallId> {
        let call = self
            .inflight
            .pending_calls
            .get(&id)?
            .iter()
            .rev()
            .find(|inflight| inflight.call.advances_state())
            .map(|inflight| &inflight.call);
        let Some(call) = call else {
            return self.state_predecessor(id);
        };
        // Accepted lengths are published in device request state; the successor
        // needs the producing computation identity while its host report is pending.
        Some(call.call_id)
    }

    /// Applies completed worker commands to scheduler-owned allocation state.
    pub(super) fn acknowledge_commands(&mut self, commands: &[BatchCommand]) {
        for command in commands {
            if let BatchCommand::Free { buffer } = command {
                // Buffer ownership may sit with an already queued free, a live
                // request, or a request awaiting its close acknowledgement.
                let id = buffer.owner.request_id;
                let allocation = self
                    .storage
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
                    self.storage.buffer_pool.free(allocation);
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
                for buffer in retiring.buffers.into_values() {
                    self.storage.buffer_pool.free(buffer);
                }
                for allocation in retiring.allocations {
                    allocation.free(&mut self.storage);
                }
                if let Some(media) = retiring.media_allocations {
                    media.free(&mut self.storage);
                }
            }
        }
    }

    /// Whether a bounded successor can execute from its predecessor's device
    /// results. The submitted input must establish its required lengths and predicate.
    pub(super) fn can_queue_successor(&self, id: RequestId, target: CallKind) -> bool {
        // Successor projection requires both live runtime state and an unresolved
        // predecessor whose device products carry the accepted execution result.
        let Some(state) = self.running.get(&id) else {
            return false;
        };
        let Some(queue) = self
            .inflight
            .pending_calls
            .get(&id)
            .filter(|queue| !queue.is_empty())
        else {
            return false;
        };

        // Bound speculation by worker capacity and exclude requests whose
        // terminal or relay state makes the projection unsafe.
        if queue.len() >= self.info.max_unresolved_calls as usize
            || state.terminal_intent.is_terminal()
            || self.inflight.pending_finishes.contains_key(&id)
            || !Self::device_token_relay_eligible(state)
        {
            return false;
        }
        let Some(predecessor) = queue.back() else {
            return false;
        };

        if target == CallKind::Transfer(TransferMode::KvPublish) {
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
        if target == CallKind::Forward(ForwardMode::Decode) {
            let feedback_continuation = consumes_image_features(&predecessor.call)
                && is_feedback_computation(&predecessor.call)
                && self
                    .scheduled_feedback(id)
                    .is_some_and(|(index, _)| index == 0)
                && state.req.image_generation.sample_feedback_continuation
                && predecessor.call.token_output.is_some();
            if feedback_continuation {
                return self
                    .pending_successor_code(id)
                    .is_some_and(|variant| variant == CallKind::Forward(ForwardMode::Decode))
                    && state.num_generated_tokens.saturating_add(1) < state.req.max_und_tokens;
            }

            if !matches!(state.phase, Phase::Prefill | Phase::DecodeUnd)
                || (state.phase == Phase::Prefill && state.starts_gen_after_context())
                || queue.iter().any(|call| {
                    !matches!(
                        call.call.code,
                        CallKind::Forward(ForwardMode::Prefill)
                            | CallKind::Forward(ForwardMode::Decode)
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
        predecessor.call.completion_output.is_some()
            && self
                .pending_successor_code(id)
                .is_some_and(|variant| variant == target)
    }

    /// Determine the next pipelined computation from its submitted predecessor.
    pub(super) fn pending_successor_code(&self, id: RequestId) -> Option<CallKind> {
        if !self.inflight.has_pending_calls(id) {
            return None;
        }
        let state = self.running.get(&id)?;
        if self.num_scheduled_prompt_tokens(id)? < state.req.prompt_token_ids.len() as u32
            || state.num_ingested_images < state.req.multimodal_inputs.images.len()
        {
            return None;
        }
        let last = &self.inflight.pending_calls.get(&id)?.back()?.call;
        let chainable = if last.code == CallKind::Media(MediaCall::ImageDecoding) {
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
            Phase::Prefill | Phase::DecodeUnd => CallKind::Forward(ForwardMode::Decode),
            Phase::CloseKv | Phase::FeedbackState => CallKind::Forward(ForwardMode::Prefill),
            Phase::PublishKv => CallKind::Transfer(TransferMode::KvPublish),
            Phase::PrepareGen => CallKind::Media(MediaCall::LatentPreparation),
            Phase::DenoiseGen => CallKind::Media(MediaCall::Denoising),
            Phase::CommitGen => CallKind::Media(MediaCall::ImageDecoding),
            Phase::FeedbackEncode => {
                let feedback = &state.req.image_generation;
                feedback.feedback_source.as_ref()?;
                match feedback
                    .feedback_encoders
                    .get(self.scheduled_feedback(id)?.0)?
                    .encoder
                {
                    ImageIngestStep::VaeEncode => CallKind::Media(MediaCall::LatentEncoding),
                    ImageIngestStep::VitEncode => CallKind::Media(MediaCall::VisionEncoding),
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
        // per-call deltas, requested logprobs, the minimum-token floor and
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

    /// Returns whether the next call can be scheduled.
    pub(super) fn can_schedule_next(&self, id: RequestId) -> bool {
        !self.inflight.pending_finishes.contains_key(&id)
            && self.output_window_ready(id)
            && self.running.get(&id).is_some_and(|state| {
                !state.speculative_chain_invalidated
                    && state.output.decoder_boundaries.len()
                        < self.info.max_unresolved_calls.max(1) as usize
                    && self.peek_next_call_variant(id).is_none_or(|kind| {
                        self.placement
                            .worker_target(
                                self.executor.as_ref(),
                                &self.info,
                                RequestKey::new(self.engine_id, id, state.request_epoch),
                                kind,
                            )
                            .is_some()
                    })
            })
            && (!self.inflight.has_pending_calls(id)
                || self
                    .pending_successor_code(id)
                    .is_some_and(|target| self.can_queue_successor(id, target)))
    }

    /// Returns whether output capacity can cover every unresolved token-producing call.
    pub(super) fn output_window_ready(&self, id: RequestId) -> bool {
        let Some(state) = self.running.get(&id) else {
            return false;
        };
        if state.output.events.is_closed() {
            return false;
        }
        let available = state.output.events.available_capacity();
        let reserved = self
            .inflight
            .pending_calls
            .get(&id)
            .into_iter()
            .flatten()
            .map(|pending| call_output_bound(pending.call.code, &pending.call.bounds))
            .sum::<usize>();
        available
            .saturating_sub(reserved)
            .saturating_sub(OUTPUT_TERMINAL_RESERVE)
            >= self.next_output_bound(id)
    }

    /// Returns the output-size bound for the next call.
    pub(super) fn next_output_bound(&self, id: RequestId) -> usize {
        match self.peek_next_call_variant(id) {
            Some(
                CallKind::Forward(ForwardMode::Prefill) | CallKind::Forward(ForwardMode::Decode),
            ) => 4,
            Some(CallKind::Media(MediaCall::Denoising)) => {
                usize::from(self.denoise_step_burst).saturating_add(2)
            }
            Some(CallKind::Media(MediaCall::ImageDecoding)) => 3,
            Some(_) | None => 2,
        }
    }

    /// Retains submitted inputs and charges their domain and transfer credits.
    pub(super) fn register_inflight(&mut self, call: Call, input: InflightInput, queue_us: u64) {
        let request_id = call.request_key.request_id;
        let domain = self.stats.domains.get(call.code);
        let active = domain.active_credits.fetch_add(1, Ordering::Relaxed) + 1;
        domain.peak_credits.fetch_max(active, Ordering::Relaxed);
        domain.launched_calls.fetch_add(1, Ordering::Relaxed);
        domain.queue_us.fetch_add(queue_us, Ordering::Relaxed);
        self.inflight
            .pending_calls
            .entry(request_id)
            .or_default()
            .push_back(InflightCall { call, input });
    }

    /// Records the domain backpressure.
    pub(super) fn record_domain_backpressure(&self, computation: uniserve_worker_ipc::CallKind) {
        self.stats
            .domains
            .get(computation)
            .backpressure_events
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Accumulates timing once per public metric group in a returned batch.
    fn record_domain_batch(stats: &super::stats::DomainStats, timing: TimingCounters) {
        stats.completed_batches.fetch_add(1, Ordering::Relaxed);
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
        computation: uniserve_worker_ipc::CallKind,
        status: CallStatus,
    ) {
        let stats = self.stats.domains.get(computation);
        stats.completed_calls.fetch_add(1, Ordering::Relaxed);
        match status {
            CallStatus::Ok => {}
            CallStatus::Predicated => {
                stats.predicated_calls.fetch_add(1, Ordering::Relaxed);
            }
            CallStatus::Error => {
                stats.error_calls.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Reclaims the domain credit.
    pub(super) fn reclaim_domain_credit(
        &self,
        computation: uniserve_worker_ipc::CallKind,
        failed: bool,
    ) {
        let stats = self.stats.domains.get(computation);
        let active = stats.active_credits.load(Ordering::Relaxed);
        if active == 0 {
            tracing::error!(?computation, "domain call credit accounting underflowed");
            return;
        }
        stats.active_credits.fetch_sub(1, Ordering::Relaxed);
        stats.reclaimed_credits.fetch_add(1, Ordering::Relaxed);
        if failed {
            stats.error_calls.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Fails the inflight domain credits.
    pub(super) fn fail_inflight_domain_credits(&self) {
        for inflight in self.inflight.pending_calls.values().flatten() {
            self.reclaim_domain_credit(inflight.call.code, true);
        }
    }

    /// Holds one validated completion until earlier calls for the request are applied.
    pub(super) fn stage_completion(
        &mut self,
        mut record: uniserve_worker_ipc::RequestOutput,
        media: Result<Option<Arc<SharedMedia>>, String>,
    ) {
        let id = record.request_key.request_id;
        let call_id = record.call_id;
        let known = self.inflight.pending_calls.get(&id).is_some_and(|queue| {
            queue.iter().any(|inflight| {
                inflight.call.request_key == record.request_key && inflight.call.call_id == call_id
            })
        });
        let duplicate = self
            .inflight
            .pending_completions
            .get(&id)
            .is_some_and(|pending| pending.contains_key(&call_id));
        if !known || duplicate {
            if let Some(state) = self.media_state_mut(id) {
                state.terminal_intent =
                    TerminalIntent::Failure("worker returned an unknown media call".to_string());
            } else if self.running.contains_key(&id) {
                self.finish(id, FinishReason::Error);
            }
            return;
        }
        let media = match media {
            Ok(media) => media,
            Err(message) => {
                let event = EngineCoreOutput::ArtifactUnavailable { message };
                if let Some(state) = self.media_state_mut(id) {
                    let _ = state.output.enqueue(event);
                } else {
                    self.emit(id, event);
                }
                record.status = CallStatus::Error;
                record.error_code = Some(uniserve_worker_ipc::ErrorCode::ResourceExhausted);
                None
            }
        };
        let arrival_seq = self.inflight.next_arrival();
        self.inflight
            .pending_completions
            .entry(id)
            .or_default()
            .insert(
                call_id,
                PendingCompletion {
                    record,
                    media,
                    arrival_seq,
                },
            );
    }

    /// Validate a media call result and advance only its completed request fields.
    pub(super) fn process_diffusion_result(
        &mut self,
        call: Call,
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
        // Only the final muxing call, the one that carries no media units,
        // returns the artifact; every other call returns none.
        let media_output_valid = if call.code == CallKind::Media(MediaCall::Muxing) {
            media.is_some() == call.inputs.is_empty()
        } else {
            media.is_none()
        };
        let step_valid = call.code != CallKind::Media(MediaCall::Denoising)
            || latent
                .as_ref()
                .is_some_and(|interval| Denoising::completes(interval, record.num_completed_steps));
        let valid = record.status == CallStatus::Ok
            && record.request_key == call.request_key
            && record.call_id == call.call_id
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
                        TerminalIntent::Failure("media worker call failed".to_string());
                }
            } else if let Some(state) = self.media_state_mut(id) {
                match call.code {
                    CallKind::Media(MediaCall::LatentPreparation) => state.denoising.open(None),
                    CallKind::Media(MediaCall::Denoising) => {
                        let interval = latent.as_ref().expect("a valid step has its interval");
                        let accepted =
                            state
                                .denoising
                                .accept(interval, record.num_completed_steps, None);
                        debug_assert!(accepted, "a valid step completes its interval");
                    }
                    CallKind::Media(MediaCall::VideoEncoding) => {
                        let range = decode
                            .as_ref()
                            .expect("submitted media write includes its input range");
                        state.encoded_video_units += range.max_units;
                        state.encoded_ready.insert(range.cursor, range.max_units);
                        if let Some((_, product)) = state.decoded_units.remove(&range.cursor) {
                            consumed_products.push(product);
                        }
                    }
                    CallKind::Media(MediaCall::AudioEncoding) => {
                        state.audio_encoded = true;
                        if let Some(product) = state.audio.take() {
                            consumed_products.push(product);
                        }
                    }
                    CallKind::Media(MediaCall::Muxing) => {
                        // The muxer has taken these units into the container,
                        // so the encoded products it consumed retire with the
                        // call; the artifact arrives with the final call.
                        state.muxing_in_flight = false;
                        consumed_products.extend(state.muxing_inputs.drain(..));
                        if media.is_some() {
                            state.muxed = true;
                        }
                    }
                    _ => {}
                }
                let phase = match call.code {
                    CallKind::Media(MediaCall::TextEncoding) => "preparing",
                    CallKind::Media(MediaCall::LatentPreparation)
                    | CallKind::Media(MediaCall::Denoising) => {
                        if state.denoising.is_complete() {
                            "decoding"
                        } else {
                            "denoising"
                        }
                    }
                    CallKind::Media(
                        MediaCall::VideoDecoding
                        | MediaCall::AudioDecoding
                        | MediaCall::VideoEncoding
                        | MediaCall::AudioEncoding,
                    ) => "decoding",
                    CallKind::Media(MediaCall::Muxing) => "finalizing",
                    _ => unreachable!("media request completed a non-media computation"),
                };
                let _ = state.output.enqueue(EngineCoreOutput::MediaProgress {
                    phase: phase.to_owned(),
                    completed_steps: state.denoising.completed(),
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
            if self.inflight.has_pending_calls(id) {
                return None;
            }
            if let TerminalIntent::Failure(message) = &state.terminal_intent {
                Some(DiffusionTerminal::Failed(message.clone()))
            } else if let TerminalIntent::Finish(reason) = &state.terminal_intent {
                Some(DiffusionTerminal::Finished(reason.clone()))
            } else if state.output.is_closed() {
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
        let Some(mut state) = self.take_media_state(id) else {
            return;
        };
        self.running_order.retain(|candidate| *candidate != id);
        match event {
            DiffusionTerminal::Completed(artifact) => {
                let _ = state.output.enqueue(EngineCoreOutput::Artifact(artifact));
                let _ = state.output.enqueue(EngineCoreOutput::Finished {
                    reason: FinishReason::Completed,
                    stop_reason: None,
                    prompt_tokens: state.request.prompt_token_ids.len(),
                    completion_tokens: 0,
                    images: 0,
                });
            }
            DiffusionTerminal::Failed(message) => {
                let _ = state.output.enqueue(EngineCoreOutput::Error { message });
            }
            DiffusionTerminal::Finished(reason) => {
                let _ = state.output.enqueue(EngineCoreOutput::Finished {
                    reason,
                    stop_reason: None,
                    prompt_tokens: state.request.prompt_token_ids.len(),
                    completion_tokens: 0,
                    images: 0,
                });
            }
        }
        self.output.retire(id, state.output);
        match state.admission_state {
            WorkerRegistration::Unsubmitted => {
                state.allocations.free(&mut self.storage);
                return;
            }
            _ => {}
        }
        let request_key = state.admission.request_key;
        self.inflight
            .pending_commands
            .push_back(BatchCommand::Finish {
                request_key,
                retained_buffers: Vec::new(),
            });
        self.retiring_requests.insert(
            id,
            RetiringRequest {
                request_key,
                allocations: Vec::new(),
                media_allocations: Some(state.allocations),
                buffers: HashMap::new(),
            },
        );
    }

    /// Resolves the front in-flight call for `id` by the worker's echoed `call_id`.
    pub(super) fn pop_inflight(
        &mut self,
        request_key: RequestKey,
        call_id: CallId,
    ) -> Option<(Call, InflightInput)> {
        let inflight = self.inflight.pop_pending_call(request_key, call_id)?;
        self.reclaim_domain_credit(inflight.call.code, false);
        Some((inflight.call, inflight.input))
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
            if let Some(allocation) = self.storage.take_encoder_buffer(buffer) {
                if let Some(previous) = self.storage.pending_buffer_frees.insert(buffer, allocation)
                {
                    tracing::error!(?buffer, "buffer free identity was already pending");
                    self.storage.buffer_pool.free(previous);
                    self.fatal = true;
                }
            }
            self.inflight
                .pending_commands
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

    /// Reconciles one physical batch result with logical calls, state, and ownership.
    pub(super) fn apply_result(&mut self, report: BatchResult) {
        let result_batch_id = report.batch_id;
        // Aggregate timings by the stable public metric groups. Multiple concrete
        // calls in one group count as one returned run, as do multiple requests.
        let mut returned_groups: [Option<TimingCounters>; 3] = [None; 3];
        let mut invalid_result = false;

        // A completion can mutate state only while its logical batch remains owned
        // by the in-flight window.
        if !self.inflight.pending_batches.contains_key(&result_batch_id) {
            tracing::error!(
                batch_id = result_batch_id,
                "executor returned a result for an unknown logical batch"
            );
            self.fatal = true;
            self.fail_all_running("executor returned a result for an unknown logical batch");
            return;
        }

        // Consume each expected call identity exactly once while folding the
        // rank-local timing records into their scheduler domains.
        for result in &report.results {
            let record = &result.output;
            let identity = (record.request_key, record.call_id);
            let removed = self
                .inflight
                .pending_batches
                .get_mut(&result_batch_id)
                .is_some_and(|pending| pending.calls.remove(&identity));
            if !removed {
                tracing::error!(
                    batch_id = result_batch_id,
                    request_id = record.request_key.request_id.0,
                    call_id = ?record.call_id,
                    "executor returned a duplicate or unknown call result"
                );
                invalid_result = true;
                continue;
            }
            let Some(computation) = self
                .inflight
                .pending_calls
                .get(&record.request_key.request_id)
                .and_then(|calls| {
                    calls
                        .iter()
                        .find(|inflight| inflight.call.call_id == record.call_id)
                })
                .map(|inflight| inflight.call.code)
            else {
                invalid_result = true;
                continue;
            };
            let component = returned_groups[super::stats::ExecutionDomainStats::index(computation)]
                .get_or_insert(TimingCounters::default());
            component.queued_us = component.queued_us.max(record.timing_counters.queued_us);
            component.device_us = component.device_us.max(record.timing_counters.device_us);
            component.copy_us = component.copy_us.max(record.timing_counters.copy_us);
            component.host_us = component.host_us.max(record.timing_counters.host_us);
        }

        let calls_complete = self
            .inflight
            .pending_batches
            .get(&result_batch_id)
            .is_some_and(|batch| batch.calls.is_empty());
        let batch_complete = report.done;
        if batch_complete && !calls_complete {
            invalid_result = true;
        }
        if invalid_result {
            self.fatal = true;
            self.fail_all_running("executor returned an invalid call result");
            return;
        }

        // Publish domain and batch accounting before individual request state is
        // advanced, so every accepted completion contributes exactly once.
        let forward_stats = report.forward_stats;
        for result in &report.results {
            let record = &result.output;
            if let Some(computation) = self
                .inflight
                .pending_calls
                .get(&record.request_key.request_id)
                .and_then(|calls| {
                    calls
                        .iter()
                        .find(|inflight| inflight.call.call_id == record.call_id)
                })
                .map(|inflight| inflight.call.code)
            {
                self.record_domain_completion(computation, record.status);
            }
        }
        for (index, returned) in returned_groups.into_iter().enumerate() {
            let Some(timing) = returned else {
                continue;
            };
            let (_, stats) = self.stats.domains.groups()[index];
            Self::record_domain_batch(stats, timing);
        }

        let pending = self
            .inflight
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
                .inflight
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
                            self.inflight.pending_commands.push_back(command);
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

        for stats in &forward_stats {
            self.record_worker_forward_stats(Some(stats));
        }

        // Staging decouples executor arrival order from request-local dependency
        // order. Only a ready prefix is removed by `take_ready` below.
        for result in report.results {
            self.stage_completion(result.output, result.media);
        }

        loop {
            let completions = self.inflight.take_ready_completions();
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
                let call_id = record.call_id;
                let Some((call, input)) = self.pop_inflight(record.request_key, call_id) else {
                    if let Some(state) = self.media_state_mut(id) {
                        state.terminal_intent = TerminalIntent::Failure(
                            "worker returned an out-of-order media call".to_string(),
                        );
                    } else if self.running.contains_key(&id) {
                        self.finish(id, FinishReason::Error);
                    }
                    continue;
                };

                let call_variant = call.code;

                // Media calls update their independent call progress immediately;
                // generation calls continue through semantic validation.
                let (image_kv, latent) = match input {
                    InflightInput::Media { latent, decode } => {
                        self.process_diffusion_result(call, latent, decode, record, media);
                        continue;
                    }
                    InflightInput::Generation { image_kv, latent } => (image_kv, latent),
                };

                if self
                    .inflight
                    .pending_finishes
                    .get(&id)
                    .is_some_and(|finish| finish.reason == FinishReason::Error)
                {
                    self.free_buffers(call.output_buffers());
                    self.finish_pending_if_idle(id);
                    continue;
                }

                let discard_invalidated_descendant = self
                    .running
                    .get(&id)
                    .is_some_and(|state| state.speculative_chain_invalidated);
                if discard_invalidated_descendant {
                    let has_unresolved_descendants = self.inflight.has_pending_calls(id);
                    if let Some(state) = self.running.get_mut(&id) {
                        state.num_kv_blocks_sent = state
                            .num_kv_blocks_sent
                            .saturating_sub(call.bounds.max_kv_pages as usize);
                        state.speculative_chain_invalidated = has_unresolved_descendants;
                    }
                    self.free_buffers(call.output_buffers());
                    self.finish_pending_if_idle(id);
                    continue;
                }

                let Some(state) = self
                    .running
                    .get(&id)
                    .filter(|state| state.request_epoch == call.request_key.request_epoch)
                else {
                    // Finish already owns retirement for a closed epoch, including
                    // outputs still being produced when cancellation arrived.
                    continue;
                };
                if generation::validate_generation_result(
                    &call,
                    image_kv,
                    latent.as_ref(),
                    state,
                    &record,
                    media.as_deref(),
                )
                .is_err()
                {
                    if self.running.contains_key(&id) {
                        self.finish_after_inflight(id, FinishReason::Error, None);
                    }
                    continue;
                }

                // Completion products used only as device predicates can be released
                // once no in-flight descendant refers to them.
                if record.status == CallStatus::Ok && !call.advances_state() {
                    let completion_has_device_consumer =
                        self.inflight.pending_calls.get(&id).is_some_and(|queue| {
                            queue.iter().any(|inflight| {
                                inflight
                                    .call
                                    .predicate
                                    .as_ref()
                                    .as_ref()
                                    .is_some_and(|predicate| {
                                        predicate.producer_call_id == call.call_id
                                    })
                            })
                        });
                    if !completion_has_device_consumer {
                        let completed_predicates = call
                            .completion_output
                            .iter()
                            .chain(call.transition_output.iter())
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
                    || self.inflight.pending_finishes.contains_key(&id);
                if semantic_blocked {
                    if call.code == CallKind::Media(MediaCall::ImageDecoding) {
                        self.free_request_latent(id);
                    }
                    self.finish_pending_if_idle(id);
                    continue;
                }

                let progress_result = if record.status == CallStatus::Ok {
                    self.running.get_mut(&id).map(|state| {
                        state.process_generation_result(&call, latent.as_ref(), &record)
                    })
                } else {
                    None
                };
                if let Some(Err(_)) = progress_result {
                    if self.running.contains_key(&id) {
                        self.finish_after_inflight(id, FinishReason::Error, None);
                    }
                    continue;
                }

                // Apply accepted progress directly; the output decoder separately
                // decides the user-visible prefix and stop-string boundary.
                let advanced = record.status == CallStatus::Ok && call.advances_state();
                let latest_token = if advanced {
                    let token = call.token_output.clone();
                    token
                } else {
                    None
                };
                let retain_device_token = !self.inflight.has_pending_calls(id);
                if let Some(state) = self.running.get_mut(&id)
                    && advanced
                {
                    state.last_state_call_id = record.call_id;
                    state.latest_token = if retain_device_token {
                        latest_token
                    } else {
                        None
                    };
                }

                // Reclaim resources whose lifetime ends at this transition before
                // making its public output eligible for resolution.
                let free_flow_prefix = call_variant == CallKind::Media(MediaCall::Denoising)
                    && record.status == CallStatus::Ok
                    && self
                        .running
                        .get(&id)
                        .is_some_and(|state| record.num_completed_steps >= state.denoising.steps());
                if call_variant == CallKind::Media(MediaCall::Denoising)
                    && record.status == CallStatus::Ok
                    && let Some(prefix) = self
                        .running
                        .get_mut(&id)
                        .and_then(|state| state.flow_prefix.as_mut())
                {
                    prefix.diffusion_finalized = true;
                }
                if call.code == CallKind::Media(MediaCall::ImageDecoding) {
                    self.free_request_latent(id);
                }
                if free_flow_prefix {
                    self.free_flow_prefix(id);
                }
                if matches!(
                    call_variant,
                    CallKind::Media(MediaCall::Denoising)
                        | CallKind::Media(MediaCall::ImageDecoding)
                ) {
                    let consumed_latents = call.latent_input.iter().cloned().collect::<Vec<_>>();
                    if !consumed_latents.is_empty() {
                        self.free_buffers(
                            consumed_latents
                                .into_iter()
                                .map(|product| product.buffer_id()),
                        );
                    }
                }

                let priority = completion_priority(call_variant);
                let public_tokens_before = self
                    .running
                    .get(&id)
                    .map_or(0, |state| state.output.tokens_sent);
                if record.status == CallStatus::Predicated {
                    let has_unresolved_descendants = self.inflight.has_pending_calls(id);
                    if let Some(state) = self.running.get_mut(&id) {
                        state.num_kv_blocks_sent = state
                            .num_kv_blocks_sent
                            .saturating_sub(call.bounds.max_kv_pages as usize);
                        state.speculative_chain_invalidated = has_unresolved_descendants;
                    }
                    self.free_buffers(call.output_buffers());
                    self.finish_pending_if_idle(id);
                } else {
                    to_resolve.push((
                        priority,
                        arrival_seq,
                        id,
                        call,
                        record,
                        media,
                        public_tokens_before,
                    ));
                }
            }

            // Resolve semantic effects in lifecycle priority and stable arrival
            // order, independent of the executor's physical completion order.
            to_resolve.sort_by_key(|(priority, seq_index, ..)| (*priority, *seq_index));
            for (_priority, _seq_index, id, call, record, media, public_tokens_before) in to_resolve
            {
                let token_call = matches!(
                    call.code,
                    CallKind::Forward(ForwardMode::Prefill)
                        | CallKind::Forward(ForwardMode::Decode)
                        | CallKind::Forward(ForwardMode::Verify)
                );
                // Reserve output space before resolving a terminal token: finishing
                // must wait for the decoder's stop-string decision on that batch.
                let awaiting_decoder = token_call
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
                if self.running.contains_key(&id)
                    && !self.inflight.pending_finishes.contains_key(&id)
                {
                    self.resolve(id, call, record, media.as_deref());
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
            }
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

    /// Fails every submitted call while preserving requests that can be rescheduled.
    pub(super) fn fail_all_inflight(&mut self, msg: &str) {
        self.inflight.pending_submissions.clear();
        self.fail_inflight_domain_credits();
        // Submitted batches cannot return after this boundary, so their timing
        // and command ownership must be retired together.
        let (ids, commands) = self.inflight.clear_failed_calls();
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
        if let Some(cache) = &self.storage.cache {
            for endpoint in &loss.endpoints {
                cache.block_pool.invalidate_source(endpoint);
            }
        }
        let mut requests = loss.requests.iter().copied().collect::<HashSet<_>>();
        let mut buffers = loss.buffers.iter().cloned().collect::<HashSet<_>>();
        loop {
            let before = (requests.len(), buffers.len());
            for batch in &self.inflight.pending_submissions {
                for (call, placement) in &batch.requests {
                    if (!loss.endpoints.is_empty() && placement.worker == loss.worker_id)
                        || call.input_buffers().any(|buffer| buffers.contains(&buffer))
                    {
                        requests.insert(call.request_key);
                    }
                    if requests.contains(&call.request_key) {
                        buffers.extend(call.output_buffers());
                    }
                }
            }
            if before == (requests.len(), buffers.len()) {
                break;
            }
        }
        let reclaimable = self.storage.encoder_cache.invalidate_buffers(&buffers);
        self.free_buffers(reclaimable.into_iter().map(|product| product.buffer_id()));
        let mut retired = loss.retired.clone();
        let mut pending = VecDeque::new();
        for mut batch in std::mem::take(&mut self.inflight.pending_submissions) {
            retired.extend(
                batch
                    .retire_requests(&requests)
                    .into_iter()
                    .map(|(call, _)| (batch.id, call.request_key, call.call_id)),
            );
            if let Some(pending) = self.inflight.pending_batches.get_mut(&batch.id) {
                pending.commands = batch
                    .commands
                    .iter()
                    .filter(|command| !matches!(command, BatchCommand::Start { .. }))
                    .cloned()
                    .collect();
            }
            pending.push_back(batch);
        }
        for (batch_id, request, call_id) in retired {
            let Some(inflight) = self.inflight.retire_call(batch_id, request, call_id) else {
                self.fatal = true;
                tracing::error!(
                    batch_id,
                    ?request,
                    ?call_id,
                    "Worker failure named an unknown pending call"
                );
                return;
            };
            self.reclaim_domain_credit(inflight.call.code, true);
        }
        for batch in pending {
            if batch.requests.is_empty() && batch.commands.is_empty() {
                self.inflight.pending_batches.remove(&batch.id);
            } else {
                self.inflight.pending_submissions.push_back(batch);
            }
        }
        self.inflight.pending_commands.retain_mut(|command| {
            if !requests.contains(&command.request_key()) {
                return true;
            }
            !matches!(command, BatchCommand::Start { .. })
        });
        let engine_id = self.engine_id;
        for request in requests {
            let id = request.request_id;
            let error_emitted = self
                .inflight
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
                self.inflight.pending_finishes.insert(
                    id,
                    PendingFinish {
                        reason: FinishReason::Error,
                        stop_reason: None,
                    },
                );
            }
            // A successful later completion may already be waiting behind the now
            // abandoned call. It owns real completion evidence and can drain.
            loop {
                let Some(call_id) = self
                    .inflight
                    .pending_calls
                    .get(&id)
                    .and_then(|queue| queue.front())
                    .filter(|inflight| inflight.call.request_key == request)
                    .map(|inflight| inflight.call.call_id)
                else {
                    break;
                };
                let Some(completion) = self
                    .inflight
                    .pending_completions
                    .get_mut(&id)
                    .and_then(|values| values.remove(&call_id))
                else {
                    break;
                };
                let Some((call, _)) = self.pop_inflight(request, call_id) else {
                    unreachable!("ready completion owns its call");
                };
                drop(completion);
                self.free_buffers(call.output_buffers());
            }
            if self
                .inflight
                .pending_completions
                .get(&id)
                .is_some_and(BTreeMap::is_empty)
            {
                self.inflight.pending_completions.remove(&id);
            }
            if self
                .media_state(id)
                .is_some_and(|state| state.admission.request_key == request)
            {
                if !self.inflight.has_pending_calls(id) {
                    self.finish_media(id, DiffusionTerminal::Failed(loss.message.clone()));
                }
            } else if self.running.contains_key(&id) {
                self.finish_after_inflight(id, FinishReason::Error, None);
            }
        }
    }

    /// Fails every queued and running request and releases scheduler-owned resources.
    pub(super) fn fail_all_running(&mut self, message: &str) {
        self.inflight.pending_submissions.clear();
        self.fail_inflight_domain_credits();
        let _ = self.inflight.clear_failed_calls();
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
