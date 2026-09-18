//! Request admission, resource reservation, and waiting-queue insertion.

use super::*;
use uniserve_worker_ipc::ForwardMode;

impl Scheduler {
    /// Validates and queues one token-generation request or rejects it synchronously.
    pub(super) fn enqueue(&mut self, req: GenerationRequest, event_tx: EventTx) {
        if self.cache.is_none() {
            self.trace_record(json!({
                "event": "request_rejected",
                "at_s": now(),
                "request_id": req.request_id.0,
                "reason": "missing_kv_resources",
            }));
            let _ = event_tx.send(EngineCoreOutput::Rejected {
                message: "generation request requires worker KV resources".into(),
            });
            return;
        }
        if let Err(error) = req.validate() {
            self.trace_record(json!({
                "event": "request_rejected",
                "at_s": now(),
                "request_id": req.request_id.0,
                "reason": "invalid_request",
                "detail": format!("{error:?}"),
            }));
            let _ = event_tx.send(EngineCoreOutput::Rejected {
                message: format!("invalid generation request: {error:?}"),
            });
            return;
        }
        if let Some(feature) = self.missing_required_feature(&req) {
            self.trace_record(json!({
                "event": "request_rejected",
                "at_s": now(),
                "request_id": req.request_id.0,
                "reason": "missing_worker_feature",
                "detail": format!("{feature}"),
            }));
            let _ = event_tx.send(EngineCoreOutput::Rejected {
                message: format!(
                    "generation request requires worker feature `{feature}`, but the worker does not support it"
                ),
            });
            return;
        }
        if let Err(error) = req.validate_resources(&self.generation_limits) {
            self.trace_record(json!({
                "event": "request_rejected",
                "at_s": now(),
                "request_id": req.request_id.0,
                "reason": "model_capacity_exceeded",
                "detail": error.to_string(),
            }));
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
            self.trace_record(json!({
                "event": "request_rejected",
                "at_s": now(),
                "request_id": req.request_id.0,
                "reason": "queue_full",
                "waiting": waiting,
                "max_num_waiting": self.config.max_num_waiting,
                "behavior": {
                    "und_decode": req.decodes_text(),
                    "und_tokens": format!("{:?}", req.emits_text()),
                    "gen_output": req.generates_images(),
                    "generated_image_feedback": req.feeds_back_images(),
                },
                "prompt_tokens": req.prompt_token_ids.len(),
            }));
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
        self.trace_request_queued(&st, "pending");
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
        use uniserve_worker_ipc::PipelineStage;

        let mut outputs = Vec::new();
        for (role, output_count) in [
            (PipelineStage::TextEncoding, 1),
            (PipelineStage::Denoising, 2),
            // The video decoder's entry declares the decoded windows and the
            // media units encoded from them; both are indexed by media unit.
            (PipelineStage::VideoDecoding, 2),
            (PipelineStage::AudioDecoding, 1),
        ] {
            let entry = self.info.pipeline_components.get(&role)?;
            let (_, bound, info) = self
                .entry_candidates(CallKind::Pipeline(role), entry)
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
                if role == PipelineStage::TextEncoding {
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
                } else if role == PipelineStage::VideoDecoding {
                    let Some(DimBound::Device { max }) = shape.dims.first().copied() else {
                        return None;
                    };
                    if sampling.num_decode_chunks == 0 || sampling.num_decode_chunks > max {
                        return None;
                    }
                    shape.dims[0] = DimBound::Static(sampling.num_decode_chunks);
                }
                outputs.push((entry.clone(), index as u32, output.dtype, shape));
            }
        }
        Some(outputs)
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
        if self.info.pipeline_components.is_empty() {
            let _ = submission.event_tx.send(EngineCoreOutput::Rejected {
                message: "worker does not provide video pipeline components".to_string(),
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
                message: "loaded media entries cannot represent the requested output bounds".into(),
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
            if self.info.pipeline_components.iter().any(|(stage, entry)| {
                !self
                    .entry_candidates(CallKind::Pipeline(*stage), entry)
                    .any(|(worker, _, _)| self.executor.is_ready(worker))
            }) {
                self.waiting_media.insert(id, submission);
                self.waiting_media_order.push_front(id);
                break;
            }
            let Ok(request_slot) = self.request_pool.allocate(request_key) else {
                self.waiting_media.insert(id, submission);
                self.waiting_media_order.push_front(id);
                break;
            };
            let outputs = self
                .media_outputs(sampling, submission.request.prompt_token_ids.len() as u32)
                .expect("queued media has valid output bounds");
            let mut tensors = HashMap::new();
            let mut reserved = true;
            for (entry, index, dtype, shape_bound) in outputs {
                let bytes = shape_bound
                    .max_elements()
                    .saturating_mul(dtype.element_bytes());
                let Ok(allocation) = self.buffer_pool.allocate(request_key, bytes, 256) else {
                    reserved = false;
                    break;
                };
                tensors.insert(
                    (entry, index),
                    MediaTensorAllocation {
                        allocation,
                        dtype,
                        shape_bound,
                    },
                );
            }
            if !reserved {
                for tensor in tensors.into_values() {
                    self.free_allocation(tensor.allocation);
                }
                self.free_allocation(request_slot);
                self.waiting_media.insert(id, submission);
                self.waiting_media_order.push_front(id);
                break;
            }
            let allocations = MediaAllocations {
                request_slot,
                tensors,
            };
            let request_pool_idx = allocations.request_slot();
            self.next_request_epoch = self.next_request_epoch.saturating_add(1);
            let admission = NewRequest::new_media(
                request_key,
                request_pool_idx,
                submission.request.prompt_token_ids.clone(),
                submission.request.sampling,
            )
            .expect("validated media admission");
            let root = CallId::new(0, 0);
            self.running_media.insert(
                id,
                MediaFlowState {
                    request: submission.request,
                    event_tx: submission.event_tx,
                    allocations,
                    conditioning: None,
                    latents: Vec::new(),
                    video_segments: BTreeMap::new(),
                    encoded_segments: BTreeMap::new(),
                    audio: None,
                    admission,
                    admission_state: WorkerRegistration::Unsubmitted,
                    text_encoding_scheduled: false,
                    latent_preparation_scheduled: false,
                    num_scheduled_steps: 0,
                    num_completed_steps: 0,
                    num_scheduled_decode_chunks: 0,
                    num_scheduled_video_chunks: 0,
                    num_encoded_video_chunks: 0,
                    audio_decoding_scheduled: false,
                    audio_encoding_scheduled: false,
                    audio_encoded: false,
                    muxing_scheduled: false,
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

    /// Returns the worker used image-latent capacity.
    pub(super) fn worker_image_latent_used(&self) -> u64 {
        (self.latent_pool.used_pages() as u64)
            .saturating_mul(u64::from(self.info.latent_page_units))
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
        let Some(state) = self.running.get(&id) else {
            return false;
        };
        let request_key = RequestKey::new(self.engine_id, id, state.request_epoch);
        let Ok(request_slot) = self.request_pool.allocate(request_key) else {
            return false;
        };
        let Ok(kv) = self.cache().allocate(request_key, prefix_tokens as u32) else {
            self.free_allocation(request_slot);
            return false;
        };
        let new_pages = kv
            .kv_tables()
            .expect("flow prefix KV allocation")
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
            allocations.free(self);
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
            prefix.allocations.free(self);
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
                state.terminal_intent.is_terminal() && !self.has_pending_calls(**id)
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
                            && !self.has_pending_calls(state.request.request_id)
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

    /// Admits requests from the queue head while sequence and memory budgets allow.
    ///
    /// Requests configured for worst-case reservation acquire their full KV capacity here,
    /// keeping that capacity resident for the request lifetime.
    pub(super) fn admit(&mut self) {
        let bs = self.info.kv_block_size() as usize;
        loop {
            if self.running_request_count() >= self.config.max_num_seqs
                || self.request_pool.is_empty()
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
                    .reserved_encoder_entries
                    .saturating_add(encoder_entries)
                    <= self.encoder_cache.budget();
                if need > self.usable_blocks() {
                    let id = self.waiting_order.pop_front().unwrap();
                    let st = self.waiting.remove(&id).unwrap();
                    self.trace_record(json!({
                        "event": "request_rejected",
                        "at_s": now(),
                        "request_id": st.req.request_id.0,
                        "reason": "too_large",
                        "needed_blocks": need,
                        "usable_blocks": self.usable_blocks(),
                        "generation": Self::generation_trace(&st.req),
                        "prompt_tokens": st.req.prompt_token_ids.len(),
                    }));
                    let _ = st.output.event_tx.send(EngineCoreOutput::Rejected {
                        message: "request exceeds total KV capacity".into(),
                    });
                    continue;
                }
                if self.free_blocks() >= need && encoder_ok {
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
                    self.reserved_blocks += need;
                    continue;
                }
            } else {
                let n = head.req.prompt_token_ids.len();
                let text_usable_blocks = (0..self.cache().block_pool.num_groups())
                    .map(|group| self.cache().block_pool.group_capacity(group))
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
                    self.trace_record(json!({
                        "event": "request_rejected",
                        "at_s": now(),
                        "request_id": st.req.request_id.0,
                        "reason": "too_large",
                        "needed_blocks": first_chunk_blocks,
                        "usable_blocks": text_usable_blocks,
                        "generation": Self::generation_trace(&st.req),
                        "prompt_tokens": st.req.prompt_token_ids.len(),
                    }));
                    let _ = st.output.event_tx.send(EngineCoreOutput::Rejected {
                        message: "request exceeds total KV capacity".into(),
                    });
                    continue;
                }
                let capacity_available = prefix_hit.cached_free_blocks.len()
                    == self.cache().block_pool.num_groups()
                    && prefix_hit.cached_free_blocks.iter().enumerate().all(
                        |(group, cached_free)| {
                            self.cache()
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
        let cache = self.cache();
        self.worker_candidates(CallKind::Forward(ForwardMode::Prefill))
            .filter(|(worker, _, _)| self.executor.is_ready(worker))
            .map(|(worker, entry, info)| {
                let hit = cache.coordinator.probe_prefix(
                    &cache.block_pool,
                    &state.req.prompt_token_ids,
                    state.req.cache.read,
                    state.has_context_images(),
                    state.req.cache.isolation_key,
                    &info.endpoint,
                );
                ((worker.clone(), entry.to_owned()), hit)
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
        self.worker_affinity
            .insert((request_key, target.1), target.0);

        // Admission owns the request row and an initially empty table for every
        // KV group before the request enters the runnable set.
        let request_slot = self
            .request_pool
            .allocate(request_key)
            .expect("admission checked request-slot capacity");
        let kv = self
            .cache()
            .allocate(request_key, 0)
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

        // Snapshot trace fields before moving the request into runtime storage.
        let generation = Self::generation_trace(&st.req);
        let phase = st.phase;
        let prompt_tokens = st.req.prompt_token_ids.len();
        let max_tokens = st.req.max_und_tokens;
        let priority = st.req.priority;
        let reserve_worstcase = st.reserve_worstcase;
        let worstcase_blocks = st.max_reserved_kv_blocks;
        let encoder_entries = st.req.num_encoder_cache_entries();
        if st.output.enqueue(EngineCoreOutput::Scheduled {
            queued_at: q,
            scheduled_at,
        }) {
            st.terminal_intent = TerminalIntent::Finish(FinishReason::Cancelled);
        }

        // Running order, resource reservations, and runtime ownership advance
        // together so the next scheduling pass observes one coherent admission.
        self.running.insert(id, st);
        self.running_order.push(id);
        self.reserved_encoder_entries = self
            .reserved_encoder_entries
            .saturating_add(encoder_entries);

        self.trace_record(json!({
            "event": "request_admitted",
            "at_s": scheduled_at,
            "request_id": id.0,
            "queued_at_s": q,
            "queue_wait_s": scheduled_at - q,
            "generation": generation,
            "phase": phase,
            "prompt_tokens": prompt_tokens,
            "max_tokens": max_tokens,
            "priority": priority,
            "reserve_worstcase": reserve_worstcase,
            "worstcase_blocks": worstcase_blocks,
            "running": self.running.len(),
            "pending": self.waiting_order.len(),
            "free_blocks": self.free_blocks(),
            "reserved_blocks": self.reserved_blocks,
            "reserved_encoder_entries": self.reserved_encoder_entries,
        }));

        // Prefix-cache acquisition pins every reused block to this request's
        // newly installed block tables.
        if let Some(st) = self.running.get_mut(&id) {
            let kv = self.cache.as_ref().expect("generation has a KV cache");
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
