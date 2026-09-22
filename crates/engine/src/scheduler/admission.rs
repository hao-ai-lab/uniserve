//! Request admission, resource reservation, and waiting-queue insertion.

use super::*;
use uniserve_worker_ipc::ForwardMode;

impl Scheduler {
    /// Validates and queues one token-generation request or rejects it synchronously.
    pub(super) fn enqueue(&mut self, req: GenerationRequest, event_tx: EventTx) {
        if self.storage.cache.is_none() {
            let _ = event_tx.send(EngineCoreOutput::Rejected {
                message: "generation request requires worker KV resources".into(),
            });
            return;
        }
        if let Err(error) = req.validate() {
            let _ = event_tx.send(EngineCoreOutput::Rejected {
                message: format!("invalid generation request: {error:?}"),
            });
            return;
        }
        if let Some(feature) = self.missing_required_feature(&req) {
            let _ = event_tx.send(EngineCoreOutput::Rejected {
                message: format!(
                    "generation request requires worker feature `{feature}`, but the worker does not support it"
                ),
            });
            return;
        }
        if let Err(error) = req.validate_resources(&self.generation_limits) {
            let _ = event_tx.send(EngineCoreOutput::Rejected {
                message: format!("invalid generation resource requirements: {error}"),
            });
            return;
        }
        // Waiting-queue backpressure sheds load instead of letting the waiting
        // queue grow without bound under overload —
        // an unbounded burst would otherwise OOM the process and take down every
        // in-flight request. Reject the new submit with a typed event.
        let waiting = self.pending_request_count() + self.output.retained_len();
        if waiting >= self.config.max_num_waiting {
            let _ = event_tx.send(EngineCoreOutput::Rejected {
                message: "scheduler waiting queue is full".into(),
            });
            return;
        }
        let worst = req
            .max_kv_tokens(&self.generation_limits)
            .expect("request capacity was validated before queueing")
            .div_ceil(self.info.kv_block_size() as usize);
        // Multimodal requests reserve their configured bounded KV envelope at
        // admission so excess concurrency queues instead of exhausting KV.
        let reserve_worstcase = !req.multimodal_inputs.images.is_empty() || req.generates_images();
        // A request with staged images usually encodes them before prefill.
        // Context-image requests prefill the text before each image position,
        // then encode the image into that marker gap.
        let finish_token_ids = finish_token_ids(&req, &self.ctrl.eos);
        let st = RequestState {
            finish_token_ids,
            allocations: None,
            flow_prefix: None,
            request_epoch: self.next_request_epoch,
            last_state_call_id: CallId::default(),
            latest_token: None,
            speculative_chain_invalidated: false,
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
            num_completed_denoise_steps: 0,
            image_conditioning: None,
            image_latent: None,
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
        let request_id = st.req.request_id;
        let position = match self.config.policy {
            SchedulingPolicy::Fcfs => self.waiting_order.len(),
            SchedulingPolicy::Priority => self.waiting_order.partition_point(|id| {
                let queued = &self.waiting[id];
                (queued.req.priority, queued.queued_at) <= (st.req.priority, st.queued_at)
            }),
        };
        self.waiting_order.insert(position, request_id);
        self.waiting.insert(request_id, st);
    }

    /// Resolve storage from loaded numerical result contracts before admission.
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
            // video encoder the rows encoded from them; both are indexed by
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
    /// their own independently bounded request rows.
    fn media_routes(&self) -> Option<HashMap<String, crate::WorkerId>> {
        use uniserve_worker_ipc::MediaCall;

        let mut required = self
            .info
            .media_components
            .iter()
            .map(|(call, component)| (*call, component.clone()))
            .collect::<Vec<_>>();
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
            if routes.contains_key(&component) {
                continue;
            }
            let candidates = self
                .placement
                .component_candidates(self.executor.as_ref(), CallKind::Media(call), &component)
                .filter(|(worker, _, _)| {
                    self.executor.is_ready(worker)
                        && (selected_workers.contains(*worker)
                            || self
                                .storage
                                .media_storage
                                .get(*worker)
                                .is_some_and(|storage| storage.requests.available() > 0))
                })
                .collect::<Vec<_>>();
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
    pub(super) fn enqueue_media(&mut self, submission: PendingMedia) {
        let request = &submission.request;
        if let Err(message) = request.validate() {
            let _ = submission.event_tx.send(EngineCoreOutput::Rejected {
                message: message.to_string(),
            });
            return;
        }
        // A worker reports the component serving each media call it implements, so a
        // deployment that assembles an artifact is the one that can serve a
        // video request; reporting some call is not enough.
        if !self
            .info
            .media_components
            .contains_key(&uniserve_worker_ipc::MediaCall::Muxing)
        {
            let _ = submission.event_tx.send(EngineCoreOutput::Rejected {
                message: "worker does not provide the video media components".to_string(),
            });
            return;
        }
        if request.sampling.num_inference_steps != self.info.num_inference_steps {
            let _ = submission.event_tx.send(EngineCoreOutput::Rejected {
                message: "request prediction count disagrees with the loaded model".to_string(),
            });
            return;
        }
        if self
            .media_outputs(request.sampling, request.prompt_token_ids.len() as u32)
            .is_none()
        {
            let _ = submission.event_tx.send(EngineCoreOutput::Rejected {
                message: "loaded media components cannot represent the requested output bounds"
                    .into(),
            });
            return;
        }
        if self.info.request_slots < 2 {
            let _ = submission.event_tx.send(EngineCoreOutput::Rejected {
                message: "worker does not provide two resident media state slots".to_string(),
            });
            return;
        }
        if self.waiting_order.len() + self.waiting_media_order.len() + self.output.retained_len()
            >= self.config.max_num_waiting
        {
            let _ = submission.event_tx.send(EngineCoreOutput::Rejected {
                message: "scheduler waiting queue is full".to_string(),
            });
            return;
        }
        let request_id = submission.request.request_id;
        self.waiting_media.insert(request_id, submission);
        self.waiting_media_order.push_back(request_id);
    }

    /// Admits queued media requests while request and product storage remain available.
    pub(super) fn admit_media(&mut self) {
        while self.running_request_count() < self.config.max_num_seqs {
            let Some(id) = self.waiting_media_order.pop_front() else {
                break;
            };
            let submission = self
                .waiting_media
                .remove(&id)
                .expect("scheduler media order names runtime state");
            let request_epoch = self.next_request_epoch;
            let request_key = RequestKey::new(self.engine_id, id, request_epoch);
            let sampling = submission.request.sampling;
            let Some(routes) = self.media_routes() else {
                self.waiting_media.insert(id, submission);
                self.waiting_media_order.push_front(id);
                break;
            };
            let route_workers = routes.values().cloned().collect::<HashSet<_>>();
            let mut request_slots = HashMap::new();
            let mut reserved = true;
            for worker in &route_workers {
                let allocation = self
                    .storage
                    .media_storage
                    .get_mut(worker)
                    .expect("media route names a loaded worker")
                    .requests
                    .allocate();
                match allocation {
                    Ok(allocation) => {
                        request_slots.insert(worker.clone(), allocation);
                    }
                    Err(_) => {
                        reserved = false;
                        break;
                    }
                }
            }
            if !reserved {
                for (worker, allocation) in request_slots {
                    self.storage.free_media_request(&worker, allocation);
                }
                self.waiting_media.insert(id, submission);
                self.waiting_media_order.push_front(id);
                break;
            }
            let outputs = self
                .media_outputs(sampling, submission.request.prompt_token_ids.len() as u32)
                .expect("queued media has valid output bounds");
            let mut tensors = HashMap::new();
            reserved = true;
            for (component, index, dtype, shape_bound) in outputs {
                let bytes = shape_bound
                    .max_elements()
                    .saturating_mul(dtype.element_bytes());
                let mut allocations = HashMap::new();
                for worker in &route_workers {
                    let allocation = self
                        .storage
                        .media_storage
                        .get_mut(worker)
                        .expect("media route names a loaded worker")
                        .buffers
                        .allocate(request_key, bytes, 256);
                    let Ok(allocation) = allocation else {
                        for (allocated_worker, allocation) in std::mem::take(&mut allocations) {
                            self.storage
                                .free_media_buffer(&allocated_worker, allocation);
                        }
                        reserved = false;
                        break;
                    };
                    allocations.insert(worker.clone(), allocation);
                }
                if !reserved {
                    break;
                }
                tensors.insert(
                    (component, index),
                    MediaTensorAllocation {
                        allocations,
                        dtype,
                        shape_bound,
                    },
                );
            }
            if !reserved {
                for tensor in tensors.into_values() {
                    for (worker, allocation) in tensor.allocations {
                        self.storage.free_media_buffer(&worker, allocation);
                    }
                }
                for (worker, allocation) in request_slots {
                    self.storage.free_media_request(&worker, allocation);
                }
                self.waiting_media.insert(id, submission);
                self.waiting_media_order.push_front(id);
                break;
            }
            let allocations = MediaAllocations {
                request_slots,
                tensors,
            };
            let primary_component = self.info.media_components[&MediaCall::Denoising].as_str();
            let request_pool_idx = allocations.request_slot(&routes[primary_component]);
            for (component, worker) in &routes {
                self.placement
                    .affinity
                    .insert((request_key, component.clone()), worker.clone());
            }
            self.next_request_epoch = self.next_request_epoch.saturating_add(1);
            let admission = NewRequest::new_media(
                request_key,
                request_pool_idx,
                submission.request.prompt_token_ids.clone(),
                submission.request.sampling,
            )
            .expect("validated media admission");
            let root = CallId::new(0, 0);
            // The interval from receipt to admission is the queue wait. The
            // engine and rank own the preparation that follows.
            self.running_media.insert(
                id,
                MediaFlowState {
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
                    num_scheduled_steps: 0,
                    num_completed_steps: 0,
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
                },
            );
            self.running_order.push(id);
        }
    }

    /// Returns the number of configured VAE workers.
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
    pub(super) fn worker_image_latent_units_for(&self, st: &RequestState) -> u64 {
        let downsample = (self.generation_limits.latent_downsample as u64).max(1);
        let (height, width) = (st.req.image.height, st.req.image.width);
        ceil_div_u64((height as u64).max(1), downsample)
            * ceil_div_u64((width as u64).max(1), downsample)
    }

    /// Materializes the negative-prompt KV prefix required by multi-branch guidance.
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
        let Ok(kv) = self.storage.cache().allocate(prefix_tokens as u32) else {
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
    pub(super) fn admit(&mut self) {
        let bs = self.info.kv_block_size() as usize;
        loop {
            if self.running_request_count() >= self.config.max_num_seqs
                || self.storage.request_pool.is_empty()
            {
                break;
            }
            let Some(head_id) = self.waiting_order.front().copied() else {
                break;
            };
            let head = self
                .waiting
                .get(&head_id)
                .expect("scheduler waiting order names runtime state");
            if head.reserve_worstcase {
                let need = head.max_reserved_kv_blocks;
                let encoder_entries = head.req.num_encoder_cache_entries();
                let encoder_ok = self
                    .storage
                    .reserved_encoder_entries
                    .saturating_add(encoder_entries)
                    <= self.storage.encoder_cache.budget();
                if need > self.storage.usable_blocks() {
                    let id = self.waiting_order.pop_front().unwrap();
                    let st = self.waiting.remove(&id).unwrap();
                    let _ = st.output.events.event_tx.send(EngineCoreOutput::Rejected {
                        message: "request exceeds total KV capacity".into(),
                    });
                    continue;
                }
                if self.storage.free_blocks() >= need && encoder_ok {
                    let Some((target, _)) = self.prefill_target(head) else {
                        break;
                    };
                    let id = self.waiting_order.pop_front().unwrap();
                    let st = self.waiting.remove(&id).unwrap();
                    let id = st.req.request_id;
                    self.admit_running(st, target);
                    // Physically allocate the worst case now: nothing can take
                    // these blocks, so this request can never fail mid-flight.
                    self.ensure_request_capacity(id, need * bs);
                    self.storage.reserved_blocks += need;
                    continue;
                }
            } else {
                let n = head.req.prompt_token_ids.len();
                let text_usable_blocks = (0..self.storage.cache().block_pool.num_groups())
                    .map(|group| self.storage.cache().block_pool.group_capacity(group))
                    .min()
                    .unwrap_or_default();
                let Some((target, prefix_hit)) = self.prefill_target(head) else {
                    break;
                };
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
                if n > text_usable_blocks * bs {
                    let id = self.waiting_order.pop_front().unwrap();
                    let st = self.waiting.remove(&id).unwrap();
                    let _ = st.output.events.event_tx.send(EngineCoreOutput::Rejected {
                        message: "request exceeds total KV capacity".into(),
                    });
                    continue;
                }
                let capacity_available = prefix_hit.cached_free_blocks.len()
                    == self.storage.cache().block_pool.num_groups()
                    && prefix_hit.cached_free_blocks.iter().enumerate().all(
                        |(group, cached_free)| {
                            self.storage
                                .cache()
                                .block_pool
                                .free_blocks_in_group(group)
                                .saturating_sub(*cached_free)
                                >= first_chunk_blocks
                        },
                    );
                if capacity_available {
                    let id = self.waiting_order.pop_front().unwrap();
                    let st = self.waiting.remove(&id).unwrap();
                    self.admit_running(st, target);
                    continue;
                }
            }

            // Exact checkpoints retain their physical KV identity. Until the
            // configured route provides relocatable checkpoint storage, a
            // resident request remains non-preemptible and admission queues.
            break;
        }
    }

    /// Probes the prefix cache on the configured prefill owner.
    fn prefill_target(
        &self,
        state: &RequestState,
    ) -> Option<((crate::WorkerId, String), crate::kv::PrefixHit)> {
        let cache = self.storage.cache();
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

    /// Reserves request resources and moves one validated request into the runnable set.
    pub(super) fn admit_running(
        &mut self,
        mut st: RequestState,
        target: (crate::WorkerId, String),
    ) {
        let id = st.req.request_id;
        let request_key = RequestKey::new(self.engine_id, id, st.request_epoch);
        let source = self
            .executor
            .info()
            .workers
            .iter()
            .find(|(worker, _)| *worker == target.0)
            .expect("selected prefill Worker remains loaded")
            .1
            .endpoint
            .clone();
        self.placement
            .affinity
            .insert((request_key, target.1), target.0);

        // Admission owns the request row and an initially empty table for every
        // KV group before the request enters the runnable set.
        let request_slot = self
            .storage
            .request_pool
            .allocate()
            .expect("admission checked request-slot capacity");
        let kv = self
            .storage
            .cache()
            .allocate(0)
            .expect("empty KV allocation is valid");
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
        let queue_wait_us = ((scheduled_at - q).max(0.0) * 1_000_000.0) as u64;
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
        if let Some(st) = self.running.get_mut(&id) {
            let kv = self
                .storage
                .cache
                .as_ref()
                .expect("generation has a KV cache");
            acquire_cached_prefix(&kv.coordinator, st, &kv.block_pool, &self.stats, &source);
        }
    }
}

/// Acquires a complete cross-group prefix hit and records cache accounting on the request.
fn acquire_cached_prefix(
    coordinator: &KvCacheCoordinator,
    state: &mut RequestState,
    pool: &BlockPool,
    stats: &SchedulerStats,
    source: &uniserve_worker_ipc::WorkerEndpoint,
) {
    let prompt = state.req.prompt_token_ids.clone();
    let has_context_images = state.has_context_images();
    let cache_read = state.req.cache.read;
    let isolation_key = state.req.cache.isolation_key;
    let hit = coordinator
        .acquire_prefix(
            pool,
            state.block_tables_mut(),
            &prompt,
            cache_read,
            has_context_images,
            isolation_key,
            source,
        )
        .expect("request cache groups match the KV coordinator");
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
    state.prefix_block_hashes = hit.block_hashes;
    state.num_computed_prompt_tokens = (hit.cached_blocks * block_size) as u32;
    state.logical_position = state.num_computed_prompt_tokens;
    state.kv_visible_len = state.num_computed_prompt_tokens;
    state.kv_computed_len = state.num_computed_prompt_tokens;
}
