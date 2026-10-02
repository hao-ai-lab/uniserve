//! Request admission, resource reservation, and waiting-queue insertion.
//!
//! Submission (`enqueue`, `enqueue_media`) validates a request and checks it
//! against the loaded worker's capabilities and the waiting-queue bound. It
//! either queues the request or rejects it synchronously with
//! `EngineCoreOutput::Rejected` on the request's event channel. Admission
//! (`admit`, `admit_media`), which the scheduling pass runs before and
//! between batch assembly, moves requests from the head of the token and
//! media waiting queues into the running set while `max_num_seqs` and
//! storage allow. A head that does not fit yet blocks the requests behind it
//! in its queue, and running requests are never preempted to make room.
//!
//! A token request reserves a request row and one KV table per group; its
//! prefill worker is recorded in `Placement::affinity`. A media request
//! reserves a request row and every declared result buffer on each worker of
//! its route.

use super::*;
use uniserve_worker_ipc::ForwardMode;

impl Scheduler {
    /// Validates and queues one token-generation request or rejects it synchronously.
    ///
    /// Rejects with `RejectionKind::Invalid` when the runtime has no KV cache,
    /// the request fails validation, the worker lacks a required generation
    /// feature, or `validate_resources` or `max_kv_tokens` fails against
    /// `generation_limits`; rejects with `RejectionKind::Overloaded` when the
    /// waiting bound is reached. A rejection is sent on `event_tx` and its
    /// send result is ignored.
    pub(super) fn enqueue(&mut self, req: GenerationRequest, event_tx: EventTx) {
        if self.storage.cache.is_none() {
            let _ = event_tx.send(EngineCoreOutput::Rejected {
                kind: RejectionKind::Invalid,
                message: "generation request requires worker KV resources".into(),
            });
            return;
        }
        if let Err(error) = req.validate() {
            let _ = event_tx.send(EngineCoreOutput::Rejected {
                kind: RejectionKind::Invalid,
                message: format!("invalid generation request: {error:?}"),
            });
            return;
        }
        if let Some(feature) = self.missing_required_feature(&req) {
            let _ = event_tx.send(EngineCoreOutput::Rejected {
                kind: RejectionKind::Invalid,
                message: format!(
                    "generation request requires worker feature `{feature}`, but the worker does not support it"
                ),
            });
            return;
        }
        // Resource validation includes the KV bound, so a request that passes
        // it has a worst-case KV token count.
        let max_kv_tokens = match req
            .validate_resources(&self.generation_limits)
            .and_then(|()| req.max_kv_tokens(&self.generation_limits))
        {
            Ok(tokens) => tokens,
            Err(error) => {
                let _ = event_tx.send(EngineCoreOutput::Rejected {
                    kind: RejectionKind::Invalid,
                    message: format!("invalid generation resource requirements: {error}"),
                });
                return;
            }
        };
        // Backpressure: under overload the new submission is rejected rather
        // than letting queued requests grow without bound and exhaust process
        // memory. The bound covers both waiting queues and retired requests
        // whose events are still undelivered; `enqueue_media` applies the
        // same count.
        let waiting = self.pending_request_count() + self.output.retained_len();
        if waiting >= self.config.max_num_waiting {
            let _ = event_tx.send(EngineCoreOutput::Rejected {
                kind: RejectionKind::Overloaded,
                message: "scheduler waiting queue is full".into(),
            });
            return;
        }
        // Worst-case KV in blocks; `admit` reserves it only for requests with
        // `reserve_worstcase` set.
        let worst = max_kv_tokens.div_ceil(self.info.kv_block_size() as usize);
        // Multimodal requests reserve their configured bounded KV envelope at
        // admission so excess concurrency queues instead of exhausting KV.
        let reserve_worstcase = !req.multimodal_inputs.images.is_empty() || req.generates_images();
        let finish_token_ids = finish_token_ids(&req, &self.ctrl.eos);
        let st = RequestState {
            finish_token_ids,
            allocations: None,
            flow_prefix: None,
            request_epoch: self.next_request_epoch,
            last_state_call_id: CallId::default(),
            latest_token: None,
            speculative_chain_invalidated: false,
            // Every request starts in prefill. A request with context images
            // prefills the text up to each image position, then encodes that
            // image into the marker gap before continuing.
            phase: Phase::Prefill,
            num_computed_prompt_tokens: 0,
            num_ingested_images: 0,
            image_encoder_index: 0,
            input_image_features: None,
            round_closing: false,
            logical_position: 0,
            kv_visible_len: 0,
            kv_computed_len: 0,
            next_token: 0,
            num_generated_tokens: 0,
            image_id: 0,
            num_generated_images: 0,
            image_reservation_pending: false,
            denoising: Denoising::new(u32::from(req.image.steps)),
            image_conditioning: None,
            feedback_encoder_index: 0,
            feedback_source: None,
            feedback_features: None,
            worker_registered: false,
            num_kv_blocks_sent: 0,
            reserve_worstcase,
            max_reserved_kv_blocks: worst,
            prefix_block_hashes: Vec::new(),
            prefix_cached: false,
            generated_token_ids: Vec::new(),
            round_token_ids: Vec::new(),
            text_tokens_since_image: 0,
            feedback_image_b64: None,
            replayable: true,
            encoder_cache_pins: Vec::new(),
            transient_encoder_products: Vec::new(),
            output: RequestOutput::new(event_tx),
            queued_at: now(),
            terminal_intent: super::TerminalIntent::None,
            req,
        };
        self.next_request_epoch = self.next_request_epoch.saturating_add(1);

        // Admission examines only the queue head, so the insertion position
        // fixes the admission order. Under `Priority` a request goes after
        // every queued request with a lower priority value, or an equal one
        // and an earlier or equal arrival, so ties stay FIFO.
        let position = match self.config.policy {
            SchedulingPolicy::Fcfs => self.waiting.len(),
            SchedulingPolicy::Priority => self.waiting.partition_point(|queued| {
                (queued.req.priority, queued.queued_at) <= (st.req.priority, st.queued_at)
            }),
        };
        self.waiting.insert(position, st);
    }

    /// Resolves the result tensors a media request reserves at admission.
    ///
    /// Reads the output declarations of the component serving text encoding,
    /// denoising, video decoding, video encoding, and audio decoding, as bound
    /// on each call's first placement candidate. The text encoder's
    /// `DimBound::Device` dimensions become `num_prompt_tokens`, and the
    /// leading dimension of the video decoder and encoder outputs becomes
    /// `sampling.video_units`. Returns one `(component, output index, dtype,
    /// shape bound)` entry per declared output.
    ///
    /// Returns `None` when one of those calls has no reported component or no
    /// candidate, the candidate declares no binding for the component it
    /// serves, a component declares an unexpected number of outputs, a text
    /// encoder output has no device dimension, a video output's leading
    /// dimension is missing or not a device bound, or the prompt length or
    /// unit count is zero or exceeds the declared maximum. `enqueue_media`
    /// rejects such a request, and `admit_media` calls this again to size the
    /// reservation.
    fn media_outputs(
        &self,
        sampling: uniserve_core::DiffusionSamplingParams,
        num_prompt_tokens: u32,
    ) -> Option<Vec<(String, u32, DType, ShapeBound)>> {
        use uniserve_worker_ipc::MediaCall;

        let mut outputs = Vec::new();
        for (role, output_count) in [
            (MediaCall::TextEncoding, 1),
            (MediaCall::Denoising, 2),
            // The video decoder declares its decoded media units and the
            // video codec the rows encoded from them; both are indexed by
            // media unit.
            (MediaCall::VideoDecoding, 1),
            (MediaCall::VideoEncoding, 1),
            (MediaCall::AudioDecoding, 1),
        ] {
            let name = self.info.media_components.get(&role)?;
            let (_, bound, info) = self
                .placement
                .component_candidates(self.executor.as_ref(), CallKind::Media(role), name)
                .next()?;
            let component = info
                .components
                .iter()
                .find(|component| component.name == bound)?;
            if component.outputs.len() != output_count {
                return None;
            }
            for (index, output) in component.outputs.iter().enumerate() {
                let mut shape = output.shape_bound.clone();
                if role == MediaCall::TextEncoding {
                    let mut selected = false;
                    for dim in &mut shape.dims {
                        if let DimBound::Device { max } = *dim {
                            if num_prompt_tokens == 0 || num_prompt_tokens > max {
                                return None;
                            }
                            *dim = DimBound::Static(num_prompt_tokens);
                            selected = true;
                        }
                    }
                    if !selected {
                        return None;
                    }
                } else if matches!(role, MediaCall::VideoDecoding | MediaCall::VideoEncoding) {
                    let Some(DimBound::Device { max }) = shape.dims.first().copied() else {
                        return None;
                    };
                    if sampling.video_units == 0 || sampling.video_units > max {
                        return None;
                    }
                    shape.dims[0] = DimBound::Static(sampling.video_units);
                }
                outputs.push((name.clone(), index as u32, output.dtype, shape));
            }
        }
        Some(outputs)
    }

    /// Chooses one physical owner for every component of a media request.
    ///
    /// The denoiser is selected first because it defines the expensive replica
    /// residency. Components co-located with that worker follow it; shared
    /// components such as a TP text encoder and the host codec worker retain
    /// their own independently bounded request rows. Otherwise the candidate
    /// with the fewest distinct requests placed on it in `Placement::affinity`
    /// wins, then the one with the most free request rows.
    ///
    /// Returns a map from component name to worker, or `None` when some
    /// component has no ready candidate that is already on the route or has
    /// a free request row. When the executor reports admissible video
    /// codecs for the video decoder's chosen worker, the video codec's
    /// candidates are further limited to those.
    fn media_routes(&self) -> Option<HashMap<String, crate::WorkerId>> {
        use uniserve_worker_ipc::MediaCall;

        let mut required = self
            .info
            .media_components
            .iter()
            .map(|(call, component)| (*call, component.clone()))
            .collect::<Vec<_>>();
        // Routing order: the denoiser first, then the calls that feed it, then
        // decoders before encoders, since the video codec's admissible
        // workers depend on the route chosen for the video decoder.
        required.sort_by_key(|(call, _)| match call {
            MediaCall::Denoising => 0,
            MediaCall::TextEncoding | MediaCall::LatentPreparation => 1,
            MediaCall::VideoDecoding | MediaCall::AudioDecoding => 2,
            MediaCall::VideoEncoding | MediaCall::AudioEncoding | MediaCall::Muxing => 3,
            _ => 4,
        });

        let mut routes = HashMap::new();
        let mut selected_workers = HashSet::new();
        for (call, component) in required {
            // A component serving several media calls is routed once.
            if routes.contains_key(&component) {
                continue;
            }
            // A request's media units are encoded where they were decoded, so
            // its encoder is one of the replicas admissible for its decoder.
            let admissible = (call == MediaCall::VideoEncoding)
                .then(|| {
                    let decoder = self.info.media_components.get(&MediaCall::VideoDecoding)?;
                    let worker = routes.get(decoder)?;
                    self.executor.info().video_codecs.get(worker)
                })
                .flatten();
            let candidates = self
                .placement
                .component_candidates(self.executor.as_ref(), CallKind::Media(call), &component)
                .filter(|(worker, _, _)| {
                    admissible.is_none_or(|encoders| encoders.contains(*worker))
                        && self.executor.is_ready(worker)
                        && (selected_workers.contains(*worker)
                            || self
                                .storage
                                .media_storage
                                .get(*worker)
                                .is_some_and(|storage| storage.requests.available() > 0))
                })
                .collect::<Vec<_>>();

            // A worker already on the route is preferred, so co-located
            // components follow the denoiser. It needs no further free row
            // because `reserve_media` takes one request row per distinct
            // worker.
            let resident = candidates
                .iter()
                .copied()
                .find(|(worker, _, _)| selected_workers.contains(*worker));
            let chosen = resident.or_else(|| {
                candidates.into_iter().min_by_key(|(worker, _, _)| {
                    let active = self
                        .placement
                        .affinity
                        .iter()
                        .filter_map(|((request, _), owner)| (owner == *worker).then_some(*request))
                        .collect::<HashSet<_>>()
                        .len();
                    let available = self.storage.media_storage[*worker].requests.available();
                    (active, usize::MAX - available)
                })
            })?;
            selected_workers.insert(chosen.0.clone());
            routes.insert(component.clone(), chosen.0.clone());
        }
        Some(routes)
    }

    /// Validates and queues one terminal media-generation request.
    ///
    /// Rejects with `RejectionKind::Invalid` when the request fails
    /// validation, the worker reports no muxing component, the request's
    /// inference-step count differs from the loaded model's, `media_outputs`
    /// cannot bound its results, or the worker has fewer than two request
    /// slots; rejects with `RejectionKind::Overloaded` when the waiting bound
    /// is reached. A rejection's send result is ignored.
    pub(super) fn enqueue_media(&mut self, submission: PendingMedia) {
        let request = &submission.request;
        if let Err(message) = request.validate() {
            let _ = submission.event_tx.send(EngineCoreOutput::Rejected {
                kind: RejectionKind::Invalid,
                message: message.to_string(),
            });
            return;
        }
        // A worker reports the component serving each media call it
        // implements, so a deployment that assembles an artifact is the one
        // that can serve a video request; reporting some call is not enough.
        if !self
            .info
            .media_components
            .contains_key(&uniserve_worker_ipc::MediaCall::Muxing)
        {
            let _ = submission.event_tx.send(EngineCoreOutput::Rejected {
                kind: RejectionKind::Invalid,
                message: "worker does not provide the video media components".to_string(),
            });
            return;
        }
        if request.sampling.num_inference_steps != self.info.num_inference_steps {
            let _ = submission.event_tx.send(EngineCoreOutput::Rejected {
                kind: RejectionKind::Invalid,
                message: "request prediction count disagrees with the loaded model".to_string(),
            });
            return;
        }
        if self
            .media_outputs(request.sampling, request.prompt_token_ids.len() as u32)
            .is_none()
        {
            let _ = submission.event_tx.send(EngineCoreOutput::Rejected {
                kind: RejectionKind::Invalid,
                message: "loaded media components cannot represent the requested output bounds"
                    .into(),
            });
            return;
        }
        if self.info.request_slots < 2 {
            let _ = submission.event_tx.send(EngineCoreOutput::Rejected {
                kind: RejectionKind::Invalid,
                message: "worker does not provide two resident media state slots".to_string(),
            });
            return;
        }
        if self.waiting.len() + self.waiting_media.len() + self.output.retained_len()
            >= self.config.max_num_waiting
        {
            let _ = submission.event_tx.send(EngineCoreOutput::Rejected {
                kind: RejectionKind::Overloaded,
                message: "scheduler waiting queue is full".to_string(),
            });
            return;
        }
        self.waiting_media.push_back(submission);
    }

    /// Reserves a media request's row and output buffers on every routed worker.
    ///
    /// Every declared output is reserved in full on every routed worker, not
    /// only on the worker whose component produces it; spans are sized from
    /// the shape bound's maximum element count and 256-byte aligned.
    ///
    /// Returns `Ok(None)`, with nothing reserved, when a request row or buffer
    /// span cannot be allocated on some worker now; `admit_media` then leaves
    /// the request at the head of the queue. Returns `Err` when a routed
    /// worker has no media storage, after releasing what was already reserved
    /// on known workers.
    fn reserve_media(
        &mut self,
        route_workers: &HashSet<crate::WorkerId>,
        request_key: RequestKey,
        outputs: Vec<(String, u32, DType, ShapeBound)>,
    ) -> Result<Option<MediaAllocations>, UnknownMediaWorker> {
        let mut reserved = MediaAllocations {
            request_slots: HashMap::new(),
            tensors: HashMap::new(),
        };
        for worker in route_workers {
            let Some(storage) = self.storage.media_storage.get_mut(worker) else {
                reserved.free(&mut self.storage)?;
                return Err(UnknownMediaWorker(worker.clone()));
            };
            let Ok(slot) = storage.requests.allocate() else {
                reserved.free(&mut self.storage)?;
                return Ok(None);
            };
            reserved.request_slots.insert(worker.clone(), slot);
        }
        for (component, index, dtype, shape_bound) in outputs {
            let bytes = shape_bound
                .max_elements()
                .saturating_mul(dtype.element_bytes());
            let mut tensor = MediaTensorAllocation {
                allocations: HashMap::new(),
                dtype,
                shape_bound,
            };
            for worker in route_workers {
                let Some(storage) = self.storage.media_storage.get_mut(worker) else {
                    reserved.tensors.insert((component, index), tensor);
                    reserved.free(&mut self.storage)?;
                    return Err(UnknownMediaWorker(worker.clone()));
                };
                let Ok(buffer) = storage.buffers.allocate(request_key, bytes, 256) else {
                    reserved.tensors.insert((component, index), tensor);
                    reserved.free(&mut self.storage)?;
                    return Ok(None);
                };
                tensor.allocations.insert(worker.clone(), buffer);
            }
            reserved.tensors.insert((component, index), tensor);
        }
        Ok(Some(reserved))
    }

    /// Admits queued media requests while request and product storage remain available.
    ///
    /// Media and token requests share the `max_num_seqs` running bound.
    /// Admission stops at the first request that cannot be routed or
    /// reserved, leaving it at the queue head. A broken scheduler invariant
    /// also requeues the request, latches engine-fatal, and stops.
    pub(super) fn admit_media(&mut self) {
        while self.running_request_count() < self.config.max_num_seqs {
            let Some(submission) = self.waiting_media.pop_front() else {
                break;
            };
            let id = submission.request.request_id;
            let request_epoch = self.next_request_epoch;
            let request_key = RequestKey::new(self.engine_id, id, request_epoch);
            let sampling = submission.request.sampling;
            let Some(routes) = self.media_routes() else {
                self.waiting_media.push_front(submission);
                break;
            };
            // Submission validated these bounds against the same loaded
            // components, so a request reaching admission has them.
            let Some(outputs) =
                self.media_outputs(sampling, submission.request.prompt_token_ids.len() as u32)
            else {
                self.waiting_media.push_front(submission);
                self.invariant_broken("a queued media request has valid output bounds");
                break;
            };
            let route_workers = routes.values().cloned().collect::<HashSet<_>>();
            let allocations = match self.reserve_media(&route_workers, request_key, outputs) {
                Ok(Some(allocations)) => allocations,
                Ok(None) => {
                    self.waiting_media.push_front(submission);
                    break;
                }
                Err(unknown) => {
                    self.waiting_media.push_front(submission);
                    self.invariant_broken(&unknown.to_string());
                    break;
                }
            };

            // `allocations` holds the request's row on every routed worker;
            // the admission carries the denoiser's row, and a call placed on
            // another worker uses that worker's row.
            let primary_component = self.info.media_components[&MediaCall::Denoising].as_str();
            let request_pool_idx = allocations.request_slot(&routes[primary_component]);
            // Submission validated the prompt and sampling this admission carries.
            let admission = match NewRequest::new_media(
                request_key,
                request_pool_idx,
                submission.request.prompt_token_ids.clone(),
                submission.request.sampling,
            ) {
                Ok(admission) => admission,
                Err(error) => {
                    let released = allocations.free(&mut self.storage);
                    self.waiting_media.push_front(submission);
                    self.invariant_broken(&format!(
                        "a queued media request forms a valid admission: {error}"
                    ));
                    if let Err(unknown) = released {
                        self.invariant_broken(&unknown.to_string());
                    }
                    break;
                }
            };

            // The admission commits here; nothing below requeues the request.
            for (component, worker) in &routes {
                self.placement
                    .affinity
                    .insert((request_key, component.clone()), worker.clone());
            }
            self.next_request_epoch = self.next_request_epoch.saturating_add(1);
            let root = CallId::new(0, 0);
            let steps = submission.request.sampling.num_inference_steps;
            let mut state = MediaFlowState {
                request: submission.request,
                output: output::EventJournal::new(submission.event_tx),
                allocations,
                buffer_bindings: HashMap::new(),
                conditioning: None,
                latents: Vec::new(),
                decoded_units: BTreeMap::new(),
                encoded_units: BTreeMap::new(),
                encoded_ready: BTreeMap::new(),
                handed_units: 0,
                muxing_inputs: Vec::new(),
                audio: None,
                admission,
                admission_state: WorkerRegistration::Unsubmitted,
                text_encoding_scheduled: false,
                latent_preparation_scheduled: false,
                denoising: Denoising::new(steps),
                scheduled_decode_units: 0,
                scheduled_encode_units: 0,
                encoded_video_units: 0,
                audio_decoding_scheduled: false,
                audio_encoding_scheduled: false,
                audio_encoded: false,
                muxing_in_flight: false,
                final_muxing_scheduled: false,
                muxed: false,
                predecessor: root,
                terminal_intent: TerminalIntent::None,
                artifact: None,
            };

            // The interval from receipt to admission is the queue wait. The
            // engine and rank own the preparation that follows, which the
            // request's public phases report from here on.
            let scheduled_at = now();
            self.record_queue_wait(submission.queued_at, scheduled_at);
            if !state.output.enqueue(EngineCoreOutput::Scheduled {
                queued_at: submission.queued_at,
                scheduled_at,
            }) {
                state.terminal_intent.finish(FinishReason::Cancelled);
            }
            self.running_media.insert(id, state);
            self.running_order.push(id);
        }
    }

    /// Returns the number of latent positions of an image of `ip`'s size.
    ///
    /// Computes `(height / d) * (width / d)` with floor division, where `d` is
    /// `latent_downsample` (at least 1). Denoising call planning adds
    /// `commit_marker_tokens` to it for the call's query length.
    pub(super) fn num_vae(&self, ip: &uniserve_core::ImageParams) -> u64 {
        let dl = u64::from(self.generation_limits.latent_downsample).max(1);
        (ip.height as u64 / dl) * (ip.width as u64 / dl)
    }

    /// Returns a required runtime feature that the worker lacks.
    pub(super) fn missing_required_feature(
        &self,
        request: &GenerationRequest,
    ) -> Option<uniserve_core::GenerationFeatures> {
        let context_steps = request
            .multimodal_inputs
            .images
            .iter()
            .flat_map(|image| image.encoders.iter().map(|input| input.encoder));
        let needs = request
            .image_generation
            .required_features(request.constraint, context_steps);
        self.generation_limits.covers(needs).err()
    }

    /// Returns whether the worker tracks image-latent capacity.
    pub(super) fn worker_tracks_image_latent(&self) -> bool {
        self.info.latent_page_units > 0 && self.info.latent_pages > 1
    }

    /// Computes image-latent capacity required by a request.
    ///
    /// The result counts latent positions: with `d = latent_downsample` (at
    /// least 1), it is `ceil(height / d) * ceil(width / d)`, each side
    /// counting at least one position. Unlike `num_vae`, partial positions
    /// round up.
    pub(super) fn worker_image_latent_units_for(&self, st: &RequestState) -> u64 {
        let downsample = (self.generation_limits.latent_downsample as u64).max(1);
        let (height, width) = (st.req.image.height, st.req.image.width);
        ceil_div_u64((height as u64).max(1), downsample)
            * ceil_div_u64((width as u64).max(1), downsample)
    }

    /// Materializes the negative-prompt KV prefix required by multi-branch guidance.
    ///
    /// Allocates the prefix its own request row and KV tables sized to the
    /// negative prompt, and stores them in `RequestState::flow_prefix`.
    /// Returns `true` when the prefix is in place, the request uses a single
    /// guidance branch, or the request is not running. Returns `false`,
    /// holding nothing new, when no request row or KV capacity is free, or
    /// (after latching engine-fatal) the KV cache is missing;
    /// `reserve_generation_resources` then declines the denoising call.
    pub(super) fn ensure_flow_prefix(&mut self, id: RequestId) -> bool {
        let (prefix_tokens, needs_alternative) = self
            .running
            .get(&id)
            .map(|state| {
                (
                    state.req.negative_prompt_token_ids.len(),
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
        if !self.running.contains_key(&id) {
            return false;
        }
        let Ok(request_slot) = self.storage.request_pool.allocate() else {
            return false;
        };
        let Some(cache) = self.storage.cache() else {
            self.storage.request_pool.free(request_slot);
            self.invariant_broken("a running token request has a KV cache");
            return false;
        };
        let Ok(kv) = cache.allocate(prefix_tokens as u32) else {
            self.storage.request_pool.free(request_slot);
            return false;
        };
        let new_pages = kv
            .tables
            .iter()
            .map(|table| (table.group_id() as u32, table.page_ids()))
            .collect();
        let allocations = RequestAllocations {
            request_slot,
            kv,
            latent: None,
            buffers: HashMap::new(),
        };
        let Some(state) = self.running.get_mut(&id) else {
            allocations.free(&mut self.storage);
            return false;
        };
        state.flow_prefix = Some(FlowPrefixState {
            allocations,
            new_pages,
            diffusion_finalized: false,
        });
        true
    }

    /// Releases the request's flow-prefix allocations.
    pub(super) fn free_flow_prefix(&mut self, id: RequestId) {
        let prefix = self
            .running
            .get_mut(&id)
            .and_then(|state| state.flow_prefix.take());
        if let Some(prefix) = prefix {
            prefix.allocations.free(&mut self.storage);
        }
    }

    /// Reaps terminal requests after all submitted descendants resolve.
    pub(super) fn reap_cancellations(&mut self) {
        // A cancelled request closes only after every submitted descendant has
        // resolved. Host KV ownership then remains pinned through the close
        // acknowledgement and ordered worker request retirement.
        let cancelled: Vec<(RequestId, TerminalIntent)> = self
            .running
            .iter()
            .filter(|(id, state)| {
                state.terminal_intent.is_terminal() && !self.inflight.has_pending_calls(**id)
            })
            .map(|(id, state)| (*id, state.terminal_intent.clone()))
            .collect();
        for (id, intent) in cancelled {
            let reason = match intent {
                TerminalIntent::Finish(reason) => reason,
                TerminalIntent::Failure(_) => FinishReason::Error,
                TerminalIntent::None => continue,
            };
            self.finish(id, reason);
        }
        let media = self
            .running_order
            .iter()
            .filter_map(|id| {
                self.media_state(*id)
                    .filter(|state| {
                        state.terminal_intent.is_terminal()
                            && !self.inflight.has_pending_calls(state.request.request_id)
                    })
                    .map(|state| (state.request.request_id, state.terminal_intent.clone()))
            })
            .collect::<Vec<_>>();
        for (id, intent) in media {
            let event = match intent {
                TerminalIntent::Failure(message) => DiffusionTerminal::Failed(message),
                TerminalIntent::Finish(reason) => DiffusionTerminal::Finished(reason),
                TerminalIntent::None => continue,
            };
            self.finish_media(id, event);
        }
    }

    /// Admits requests from the queue head while sequence and storage budgets allow.
    ///
    /// Requests configured for worst-case reservation acquire their full KV capacity here,
    /// keeping that capacity resident for the request lifetime.
    ///
    /// A worst-case request is admitted when free KV blocks cover
    /// `max_reserved_kv_blocks` and its encoder-cache entries fit the budget;
    /// any other request needs only the blocks of its first prefill chunk
    /// beyond its cached prefix. A head that can never fit the total KV
    /// capacity is rejected with `RejectionKind::Invalid`. Admission stops
    /// when the queue is empty, `max_num_seqs` is reached, no request row is
    /// free, the head does not fit yet or finds no ready prefill worker, or a
    /// scheduler invariant breaks.
    pub(super) fn admit(&mut self) {
        let bs = self.info.kv_block_size() as usize;
        loop {
            if self.running_request_count() >= self.config.max_num_seqs
                || self.storage.request_pool.is_empty()
            {
                break;
            }
            let Some(head) = self.waiting.front() else {
                break;
            };

            if head.reserve_worstcase {
                let need = head.max_reserved_kv_blocks;
                let encoder_entries = head.req.num_encoder_cache_entries();
                let encoder_ok = self
                    .storage
                    .reserved_encoder_entries
                    .saturating_add(encoder_entries)
                    <= self.storage.encoder_cache.budget();
                if need > self.storage.usable_blocks() {
                    let Some(st) = self.waiting.pop_front() else {
                        break;
                    };
                    let _ = st.output.events.event_tx.send(EngineCoreOutput::Rejected {
                        kind: RejectionKind::Invalid,
                        message: "request exceeds total KV capacity".into(),
                    });
                    continue;
                }
                if self.storage.free_blocks() >= need && encoder_ok {
                    let Some((target, _)) = self.prefill_target(head) else {
                        break;
                    };
                    let Some(st) = self.waiting.pop_front() else {
                        break;
                    };
                    let reservation = match self.reserve_admission(&target.0) {
                        Ok(reservation) => reservation,
                        Err(invariant) => {
                            self.waiting.push_front(st);
                            self.invariant_broken(invariant);
                            break;
                        }
                    };
                    let id = st.req.request_id;
                    self.admit_running(st, target, reservation);
                    // Grow the tables to the whole worst case now, so later
                    // admissions cannot take these blocks. The result is not
                    // checked here; when an image branch opens,
                    // `promote_gen_branch_reservation` finishes the request
                    // with an error if its first table holds fewer than
                    // `need` blocks.
                    self.ensure_request_capacity(id, need * bs);
                    self.storage.reserved_blocks += need;
                    continue;
                }
            } else {
                let n = head.req.prompt_token_ids.len();
                // Submission rejects token requests when the worker has no KV
                // cache, so a queued one always finds it.
                let Some(cache) = self.storage.cache() else {
                    self.invariant_broken("a queued token request has a KV cache");
                    break;
                };
                let text_usable_blocks = (0..cache.block_pool.num_groups())
                    .map(|group| cache.block_pool.group_capacity(group))
                    .min()
                    .unwrap_or_default();
                let Some((target, prefix_hit)) = self.prefill_target(head) else {
                    break;
                };

                // Blocks the first prefill chunk needs beyond the cached
                // prefix. The chunk is bounded by `long_prefill_threshold` and
                // `max_num_batched_tokens` and holds at least one token.
                let cached_prefix_blocks = prefix_hit.cached_blocks;
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
                // The prompt alone must fit the smallest group's capacity.
                if n > text_usable_blocks * bs {
                    let Some(st) = self.waiting.pop_front() else {
                        break;
                    };
                    let _ = st.output.events.event_tx.send(EngineCoreOutput::Rejected {
                        kind: RejectionKind::Invalid,
                        message: "request exceeds total KV capacity".into(),
                    });
                    continue;
                }

                // A hit page that is currently unreferenced counts as free
                // until acquisition takes it, so it is subtracted from each
                // group's free pages.
                let capacity_available = prefix_hit.cached_free_blocks.len()
                    == cache.block_pool.num_groups()
                    && prefix_hit.cached_free_blocks.iter().enumerate().all(
                        |(group, cached_free)| {
                            cache
                                .block_pool
                                .free_blocks_in_group(group)
                                .saturating_sub(*cached_free)
                                >= first_chunk_blocks
                        },
                    );
                if capacity_available {
                    let Some(st) = self.waiting.pop_front() else {
                        break;
                    };
                    let reservation = match self.reserve_admission(&target.0) {
                        Ok(reservation) => reservation,
                        Err(invariant) => {
                            self.waiting.push_front(st);
                            self.invariant_broken(invariant);
                            break;
                        }
                    };
                    self.admit_running(st, target, reservation);
                    continue;
                }
            }

            // The head does not fit yet. Resident requests keep their physical
            // KV pages and are never preempted or relocated to make room, and
            // later requests do not bypass the head, so admission waits for
            // capacity to free.
            break;
        }
    }

    /// Probes the prefix cache on the configured prefill owner.
    ///
    /// Picks the first ready worker that serves prefill and returns it with
    /// the component name it binds, plus the prefix hit on that worker's
    /// endpoint (prefix-cache residency is keyed by endpoint). The hit is an
    /// estimate; `acquire_cached_prefix` takes the actual prefix. Returns
    /// `None` when the runtime has no KV cache or no prefill worker is ready.
    fn prefill_target(
        &self,
        state: &RequestState,
    ) -> Option<((crate::WorkerId, String), crate::kv::PrefixHit)> {
        let cache = self.storage.cache()?;
        self.placement
            .worker_candidates(
                self.executor.as_ref(),
                &self.info,
                CallKind::Forward(ForwardMode::Prefill),
            )
            .filter(|(worker, _, _)| self.executor.is_ready(worker))
            .map(|(worker, component, info)| {
                let hit = cache.coordinator.probe_prefix(
                    &cache.block_pool,
                    &state.req.prompt_token_ids,
                    state.req.cache.read,
                    state.has_context_images(),
                    state.req.cache.isolation_key,
                    &info.endpoint,
                );
                ((worker.clone(), component.to_owned()), hit)
            })
            .next()
    }

    /// Accumulates one request's interval from queue entry to admission.
    ///
    /// Timestamps are in seconds; the statistics are in microseconds, and a
    /// negative interval counts as zero.
    fn record_queue_wait(&self, queued_at: f64, scheduled_at: f64) {
        let queue_wait_us = ((scheduled_at - queued_at).max(0.0) * 1_000_000.0) as u64;
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
    }

    /// Reserves what admission to `target` needs before the request commits.
    ///
    /// Admission checked request-row capacity and chose `target` among loaded
    /// workers, and the text KV cache exists for every queued token request, so
    /// an error names the scheduler invariant that no longer holds. Nothing
    /// stays reserved on error.
    fn reserve_admission(
        &mut self,
        target: &crate::WorkerId,
    ) -> Result<AdmissionReservation, &'static str> {
        let source = self
            .executor
            .info()
            .workers
            .iter()
            .find(|(worker, _)| worker == target)
            .map(|(_, worker)| worker.endpoint.clone())
            .ok_or("the selected prefill worker remains loaded")?;
        let cache = self
            .storage
            .cache
            .as_ref()
            .ok_or("a queued token request has a KV cache")?;
        let request_slot = self
            .storage
            .request_pool
            .allocate()
            .map_err(|_| "admission checked request-row capacity")?;
        // Admission owns the request row and an initially empty table for every
        // KV group before the request enters the runnable set.
        let Ok(kv) = cache.allocate(0) else {
            self.storage.request_pool.free(request_slot);
            return Err("an empty KV allocation always fits");
        };
        Ok(AdmissionReservation {
            source,
            request_slot,
            kv,
        })
    }

    /// Moves one validated request into the runnable set with its reservation.
    ///
    /// Records `target`'s worker as the owner of its component for the
    /// request, emits the public `Scheduled` event (a closed receiver marks
    /// the request cancelled), adds its encoder-cache entries to the reserved
    /// count, and acquires its cached prefix, which sets the computed-prompt
    /// cursors to the reused tokens. A failed acquisition latches
    /// engine-fatal.
    pub(super) fn admit_running(
        &mut self,
        mut st: RequestState,
        target: (crate::WorkerId, String),
        reservation: AdmissionReservation,
    ) {
        let id = st.req.request_id;
        let request_key = RequestKey::new(self.engine_id, id, st.request_epoch);
        let AdmissionReservation {
            source,
            request_slot,
            kv,
        } = reservation;
        self.placement
            .affinity
            .insert((request_key, target.1), target.0);
        st.allocations = Some(RequestAllocations {
            request_slot,
            kv,
            latent: None,
            buffers: HashMap::new(),
        });

        // Queue latency is finalized at the same timestamp exposed through the
        // public scheduling event.
        let q = st.queued_at;
        let scheduled_at = now();
        self.record_queue_wait(q, scheduled_at);

        let encoder_entries = st.req.num_encoder_cache_entries();
        if !st.output.events.enqueue(EngineCoreOutput::Scheduled {
            queued_at: q,
            scheduled_at,
        }) {
            st.terminal_intent = TerminalIntent::Finish(FinishReason::Cancelled);
        }

        // Running order, resource reservations, and runtime ownership advance
        // together so the next scheduling pass observes one coherent admission.
        self.running.insert(id, st);
        self.running_order.push(id);
        self.storage.reserved_encoder_entries = self
            .storage
            .reserved_encoder_entries
            .saturating_add(encoder_entries);

        // Prefix-cache acquisition pins every reused block to this request's
        // newly installed block tables.
        let acquired = match (self.running.get_mut(&id), self.storage.cache.as_ref()) {
            (Some(st), Some(kv)) => {
                acquire_cached_prefix(&kv.coordinator, st, &kv.block_pool, &self.stats, &source)
            }
            _ => Err("an admitted request runs with a KV cache"),
        };
        if let Err(invariant) = acquired {
            self.invariant_broken(invariant);
        }
    }
}

/// Resources a token request holds from the moment admission commits it.
pub(super) struct AdmissionReservation {
    /// The prefill worker's endpoint, which keys its prefix-cache residency.
    source: uniserve_worker_ipc::WorkerEndpoint,
    request_slot: RequestSlot,
    kv: KvAllocation,
}

/// Acquires a complete cross-group prefix hit and records cache accounting on the request.
///
/// Fails, acquiring nothing, when the request holds no block tables or its
/// tables do not match the pool's KV groups.
fn acquire_cached_prefix(
    coordinator: &KvCacheCoordinator,
    state: &mut RequestState,
    pool: &BlockPool,
    stats: &SchedulerStats,
    source: &uniserve_worker_ipc::WorkerEndpoint,
) -> Result<(), &'static str> {
    let prompt = state.req.prompt_token_ids.clone();
    let has_context_images = state.has_context_images();
    let cache_read = state.req.cache.read;
    let isolation_key = state.req.cache.isolation_key;
    let tables = state
        .block_tables_mut()
        .ok_or("an admitted request holds its block tables")?;
    let hit = coordinator
        .acquire_prefix(
            pool,
            tables,
            &prompt,
            cache_read,
            has_context_images,
            isolation_key,
            source,
        )
        .ok_or("an admitted request's block tables cover every KV group")?;

    // Queries count the blocks eligible for lookup under the same rule as
    // `prefix_lookup_limit` in `kv`: a block-aligned prompt's last block is
    // never reused, so prefill computes at least one token.
    let block_size = pool.block_size();
    let full_blocks = prompt.len() / block_size;
    let query_blocks = if prompt.len().is_multiple_of(block_size) {
        full_blocks.saturating_sub(1)
    } else {
        full_blocks
    };
    stats
        .prefix
        .queries
        .fetch_add(query_blocks as u64, Ordering::Relaxed);
    stats
        .prefix
        .hits
        .fetch_add(hit.cached_blocks as u64, Ordering::Relaxed);
    stats
        .prefix
        .hit_tokens
        .fetch_add((hit.cached_blocks * block_size) as u64, Ordering::Relaxed);

    // The reused tokens are already in KV, so every prompt and KV cursor
    // starts past them.
    state.prefix_block_hashes = hit.block_hashes;
    state.num_computed_prompt_tokens = (hit.cached_blocks * block_size) as u32;
    state.logical_position = state.num_computed_prompt_tokens;
    state.kv_visible_len = state.num_computed_prompt_tokens;
    state.kv_computed_len = state.num_computed_prompt_tokens;
    Ok(())
}
