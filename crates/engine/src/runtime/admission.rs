//! Request admission, resource reservation, and waiting-queue insertion.

use super::*;

impl EngineLoop {
    /// Validates and queues one token-generation request or rejects it synchronously.
    pub(super) fn enqueue(&mut self, req: GenerationRequest, event_tx: EventTx) {
        if self.memory.cache.is_none() {
            self.trace_record(json!({
                "event": "request_rejected",
                "at_s": now(),
                "request_id": req.request_id.0,
                "reason": "missing_kv_resources",
            }));
            let _ = event_tx.send(Event::Rejected {
                message: "generation request requires worker KV resources".into(),
            });
            return;
        }
        let context = match RuntimeContext::lower(&req) {
            Ok(context) => context,
            Err(error) => {
                self.trace_record(json!({
                    "event": "request_rejected",
                    "at_s": now(),
                    "request_id": req.request_id.0,
                    "reason": "invalid_request",
                    "detail": format!("{error:?}"),
                }));
                let _ = event_tx.send(Event::Rejected {
                    message: format!("invalid generation request: {error:?}"),
                });
                return;
            }
        };
        if let Some(feature) = self.missing_required_feature(&req) {
            self.trace_record(json!({
                "event": "request_rejected",
                "at_s": now(),
                "request_id": req.request_id.0,
                "reason": "missing_worker_feature",
                "detail": format!("{feature}"),
            }));
            let _ = event_tx.send(Event::Rejected {
                message: format!(
                    "generation request requires worker feature `{feature}`, but the worker does not support it"
                ),
            });
            return;
        }
        if let Err(error) = req.validate_resources(&self.profile.generation_limits) {
            self.trace_record(json!({
                "event": "request_rejected",
                "at_s": now(),
                "request_id": req.request_id.0,
                "reason": "invalid_resource_declaration",
                "detail": error.to_string(),
            }));
            let _ = event_tx.send(Event::Rejected {
                message: format!("invalid generation resource declaration: {error}"),
            });
            return;
        }
        // Waiting-queue backpressure sheds load instead of letting the waiting
        // queue grow without bound under overload —
        // an unbounded burst would otherwise OOM the process and take down every
        // in-flight request. Reject the new submit with a typed event.
        let waiting = self.pending_request_count() + self.scheduler.output.retained_len();
        if waiting >= self.scheduler.config.max_num_waiting {
            self.trace_record(json!({
                "event": "request_rejected",
                "at_s": now(),
                "request_id": req.request_id.0,
                "reason": "queue_full",
                "waiting": waiting,
                "max_num_waiting": self.scheduler.config.max_num_waiting,
                "behavior": {
                    "und_decode": req.behavior.und_decode,
                    "und_tokens": format!("{:?}", req.behavior.und_tokens),
                    "gen_output": req.behavior.gen_output,
                    "generated_image_feedback": req.behavior.generated_image_feedback,
                },
                "prompt_tokens": context.prompt_ids.len(),
            }));
            let _ = event_tx.send(Event::Rejected {
                message: "scheduler waiting queue is full".into(),
            });
            return;
        }
        let worst = req
            .resources
            .max_kv_tokens
            .div_ceil(self.info.kv_block_size() as usize);
        // Multimodal requests reserve their configured bounded KV envelope at
        // admission so excess concurrency queues instead of exhausting KV.
        let reserve_worstcase = !context.images.is_empty() || req.behavior.gen_output;
        // A request with staged images usually encodes them before prefill.
        // Context-image requests prefill the text before each image position,
        // then encode the image into that marker gap.
        let phase0 = Phase::Prefill;
        let finish_token_ids = finish_token_ids(&req, &self.ctrl.eos);
        let st = ReqState {
            finish_token_ids,
            allocations: None,
            flow_prefix: None,
            epoch: self.next_epoch,
            version: 0,
            resolved_producer_op_id: 0,
            committed_version: 0,
            committed_producer_op_id: 0,
            control_seq: 0,
            public_event_limit: 0,
            token_cutoffs: BTreeMap::new(),
            pending_commits: VecDeque::new(),
            cancel_cutoff: None,
            latest_device_version: None,
            speculative_chain_invalidated: false,
            cursor: GenerationCursor::new(phase0, worst, reserve_worstcase),
            context,
            output: RequestOutput::new(event_tx),
            queued_at: now(),
            terminal_intent: super::TerminalIntent::None,
            req,
        };
        self.next_epoch = self.next_epoch.saturating_add(1);
        self.trace_request_queued(&st, "pending");
        let request_id = st.req.request_id;
        self.scheduler
            .enqueue(request_id, st.req.priority, st.queued_at);
        self.waiting.insert(request_id, st);
    }

    /// Resolve storage from loaded numerical result contracts before admission.
    fn media_tensor_specs(
        &self,
        geometry: uniserve_core::MediaGeometry,
    ) -> Option<Vec<(String, u32, DType, ShapeBound)>> {
        use uniserve_worker_ipc::MediaStageRole;

        let plan = self.info.media_plan.as_ref()?;
        let mut specs = Vec::new();
        for (role, outputs) in [
            (MediaStageRole::Encode, 1),
            (MediaStageRole::Denoise, 2),
            (MediaStageRole::VideoDecode, 1),
            (MediaStageRole::AudioDecode, 1),
        ] {
            let stage = plan.stage_by_role(role)?;
            let (_, bound, info) = self
                .entry_candidates(stage.operation, &stage.entry)
                .next()?;
            let component = info
                .components
                .iter()
                .find(|component| component.name == bound)?;
            if component.outputs.len() != outputs {
                return None;
            }
            for (index, output) in component.outputs.iter().enumerate() {
                let mut shape = output.shape_bound.clone();
                if role == MediaStageRole::Encode {
                    let mut selected = false;
                    for dim in &mut shape.dims {
                        if let DimBound::Device { max } = *dim {
                            if geometry.prompt_tokens == 0 || geometry.prompt_tokens > max {
                                return None;
                            }
                            *dim = DimBound::Static(geometry.prompt_tokens);
                            selected = true;
                        }
                    }
                    if !selected {
                        return None;
                    }
                } else if role == MediaStageRole::VideoDecode {
                    let Some(DimBound::Device { max }) = shape.dims.first().copied() else {
                        return None;
                    };
                    if geometry.video_units == 0 || geometry.video_units > max {
                        return None;
                    }
                    shape.dims[0] = DimBound::Static(geometry.video_units);
                }
                specs.push((stage.entry.clone(), index as u32, output.dtype, shape));
            }
        }
        Some(specs)
    }

    /// Validates and queues one terminal media-generation request.
    pub(super) fn enqueue_media(&mut self, submission: PendingMedia) {
        let request = &submission.request;
        if let Err(message) = request.validate() {
            let _ = submission.event_tx.send(Event::Rejected {
                message: message.to_string(),
            });
            return;
        }
        let Some(plan) = self.info.media_plan.as_ref() else {
            let _ = submission.event_tx.send(Event::Rejected {
                message: "worker does not declare a terminal media plan".to_string(),
            });
            return;
        };
        if plan
            .stages
            .iter()
            .any(|stage| !self.info.supported_ops.contains(&stage.operation))
        {
            let _ = submission.event_tx.send(Event::Rejected {
                message: "worker does not support its declared media plan".to_string(),
            });
            return;
        }
        if request.geometry.denoise_steps != plan.denoise_steps() {
            let _ = submission.event_tx.send(Event::Rejected {
                message: "request prediction count disagrees with the loaded media plan"
                    .to_string(),
            });
            return;
        }
        if self.media_tensor_specs(request.geometry).is_none() {
            let _ = submission.event_tx.send(Event::Rejected {
                message: "loaded media entries cannot represent the requested tensor geometry"
                    .into(),
            });
            return;
        }
        if self.info.request_slots < 2 {
            let _ = submission.event_tx.send(Event::Rejected {
                message: "worker does not provide two resident media state slots".to_string(),
            });
            return;
        }
        if self.scheduler.waiting_len()
            + self.scheduler.waiting_media_len()
            + self.scheduler.output.retained_len()
            >= self.scheduler.config.max_num_waiting
        {
            let _ = submission.event_tx.send(Event::Rejected {
                message: "scheduler waiting queue is full".to_string(),
            });
            return;
        }
        let request_id = submission.request.request_id;
        self.waiting_media.insert(request_id, submission);
        self.scheduler.enqueue_media(request_id);
    }

    /// Admits queued media requests while request and product storage remain available.
    pub(super) fn admit_media(&mut self) {
        while self.running_request_count() < self.scheduler.config.max_num_seqs {
            let Some(id) = self.scheduler.pop_media() else {
                break;
            };
            let submission = self
                .waiting_media
                .remove(&id)
                .expect("scheduler media order names runtime state");
            let epoch = self.next_epoch;
            let request_key = RequestKey::new(self.authority_id, id, epoch);
            let geometry = submission.request.geometry;
            let plan = self
                .info
                .media_plan
                .as_ref()
                .expect("queued media retains a validated execution plan");
            if plan.stages.iter().any(|stage| {
                !self
                    .entry_candidates(stage.operation, &stage.entry)
                    .any(|(worker, _, _)| self.executor.is_ready(worker))
            }) {
                self.waiting_media.insert(id, submission);
                self.scheduler.push_media_front(id);
                break;
            }
            let Ok(request_slot) = self.memory.alloc(request_key, MemoryLayout::RequestSlot) else {
                self.waiting_media.insert(id, submission);
                self.scheduler.push_media_front(id);
                break;
            };
            let specs = self
                .media_tensor_specs(geometry)
                .expect("queued media has valid result geometry");
            let mut tensors = HashMap::new();
            let mut reserved = true;
            for (entry, index, dtype, shape_bound) in specs {
                let bytes = shape_bound
                    .max_elements()
                    .saturating_mul(dtype.element_bytes());
                let Ok(allocation) = self.memory.alloc(
                    request_key,
                    MemoryLayout::Buffer {
                        bytes,
                        alignment: 256,
                    },
                ) else {
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
                    self.memory.free(tensor.allocation);
                }
                self.memory.free(request_slot);
                self.waiting_media.insert(id, submission);
                self.scheduler.push_media_front(id);
                break;
            }
            let encoder_op = OpId(self.next_op_id.max(1));
            let encode_entry = &plan
                .stage_by_role(uniserve_worker_ipc::MediaStageRole::Encode)
                .expect("validated media plan has an encode stage")
                .entry;
            let encoded = &tensors[&(encode_entry.clone(), 0)];
            let conditioning = ProductRef {
                request_key,
                producer_op_id: encoder_op,
                output_index: 0,
                generation: 1,
                kind: ProductKind::Tensor,
                storage_class: StorageClass::DeviceTensor,
                dtype: encoded.dtype,
                shape_bound: encoded.shape_bound.clone(),
                point_range: PointRange::default(),
            };
            self.next_op_id = encoder_op.0.saturating_add(1);
            let allocations = MediaAllocations {
                request_slot,
                tensors,
            };
            let request_pool_idx = allocations.request_slot();
            self.next_epoch = self.next_epoch.saturating_add(1);
            let admission = NewRequest::new_media(
                request_key,
                request_pool_idx,
                DiffusionRequestParams {
                    references: submission
                        .request
                        .image_reference
                        .as_ref()
                        .map(|image| {
                            vec![uniserve_worker_ipc::DecodedReference {
                                kind: "image".into(),
                                task: "first_frame".into(),
                                role: "first_frame".into(),
                                include_audio: false,
                                pixels: Some(ProductRef {
                                    request_key,
                                    // The reserved host-input index cannot alias the
                                    // encoder's output conditioning tensor.
                                    producer_op_id: encoder_op,
                                    output_index: u16::MAX,
                                    generation: 1,
                                    kind: ProductKind::Tensor,
                                    storage_class: StorageClass::HostStaging,
                                    dtype: DType::U8,
                                    shape_bound: ShapeBound {
                                        dims: [1, image.height, image.width, 3]
                                            .into_iter()
                                            .map(DimBound::Static)
                                            .collect(),
                                    },
                                    point_range: PointRange::default(),
                                }),
                                audio: None,
                                fps_num: 0,
                                fps_den: 1,
                            }]
                        })
                        .unwrap_or_default(),
                    prompt_token_ids: submission.request.prompt_token_ids.clone(),
                    seed: submission.request.seed,
                    geometry: MediaGeometry {
                        frame_count: submission.request.geometry.frame_count,
                        video_units: submission.request.geometry.video_units,
                        prompt_tokens: submission.request.geometry.prompt_tokens,
                        denoise_steps: submission.request.geometry.denoise_steps,
                    },
                },
            )
            .expect("validated media admission");
            let root = Checkpoint::admission_root(OpId(0));
            self.running_media.insert(
                id,
                MediaFlowState {
                    request: submission.request,
                    event_tx: submission.event_tx,
                    allocations,
                    conditioning,
                    latents: Vec::new(),
                    video_segments: BTreeMap::new(),
                    audio: None,
                    admission,
                    admission_state: DiffusionRequestParamsState::Unsubmitted,
                    committed: MediaCursor::default(),
                    projected: MediaCursor::default(),
                    fixed_parent: root.clone(),
                    projected_parent: root,
                    terminal_intent: TerminalIntent::None,
                    artifact: None,
                },
            );
            self.scheduler.running_order.push(id);
        }
    }

    /// Returns the number of configured VAE workers.
    pub(super) fn num_vae(&self, ip: &uniserve_core::ImageParams) -> u64 {
        let dl = u64::from(self.profile.generation_limits.latent_downsample).max(1);
        (ip.height as u64 / dl) * (ip.width as u64 / dl)
    }

    /// Caps image grid tokens to the configured VAE limit.
    pub(super) fn cap_max_vae_grid_tokens(&self) -> usize {
        if self.profile.generation_limits.max_vae_grid_tokens > 0 {
            self.profile.generation_limits.max_vae_grid_tokens as usize
        } else {
            self.info.latent_capacity_units().min(usize::MAX as u64) as usize
        }
    }

    /// Returns a required runtime feature that the worker lacks.
    pub(super) fn missing_required_feature(
        &self,
        request: &GenerationRequest,
    ) -> Option<uniserve_core::GenerationFeatures> {
        let context_steps = request.context.iter().flat_map(|segment| match segment {
            uniserve_core::ContextSegment::Image { ingest, .. } => ingest.steps.clone(),
            uniserve_core::ContextSegment::UndTokens { .. } => Vec::new(),
        });
        let needs = request
            .behavior
            .required_features(&request.policy, context_steps);
        self.profile.generation_limits.covers(needs).err()
    }

    /// Returns whether the worker tracks image-latent capacity.
    pub(super) fn worker_tracks_image_latent(&self) -> bool {
        self.info.latent_page_units > 0 && self.info.latent_pages > 1
    }

    /// Returns the worker used image-latent capacity.
    pub(super) fn worker_image_latent_used(&self) -> u64 {
        (self.memory.latent_pages.used_pages() as u64)
            .saturating_mul(u64::from(self.info.latent_page_units))
    }

    /// Computes image-latent capacity required by a request.
    pub(super) fn worker_image_latent_units_for(&self, st: &ReqState) -> u64 {
        let downsample = (self.profile.generation_limits.latent_downsample as u64).max(1);
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
                    state.context.negative_prompt_ids.len(),
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
        let request_key = RequestKey::new(self.authority_id, id, state.epoch);
        let Ok(request_slot) = self.memory.alloc(request_key, MemoryLayout::RequestSlot) else {
            return false;
        };
        let groups = self.memory.cache().block_pool.num_groups() as u32;
        let Ok(kv) = self.memory.alloc(
            request_key,
            MemoryLayout::Kv {
                tokens: prefix_tokens as u32,
                groups,
            },
        ) else {
            self.memory.free(request_slot);
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
            allocations.free(&mut self.memory);
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
            prefix.allocations.free(&mut self.memory);
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
                state.terminal_intent.is_terminal() && !self.inflight.contains(**id)
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
            .scheduler
            .running_order
            .iter()
            .filter_map(|id| {
                self.media_state(*id)
                    .filter(|state| {
                        state.terminal_intent.is_terminal()
                            && !self.inflight.contains(state.request.request_id)
                    })
                    .map(|state| (state.request.request_id, state.terminal_intent.clone()))
            })
            .collect::<Vec<_>>();
        for (id, intent) in media {
            let (event, reason) = match intent {
                TerminalIntent::Failure(message) => {
                    (DiffusionTerminal::Failed(message), CloseReason::Error)
                }
                TerminalIntent::Finish(reason) => {
                    (DiffusionTerminal::Finished(reason), CloseReason::Cancelled)
                }
                TerminalIntent::None => continue,
            };
            self.finish_media(id, event, reason, None);
        }
    }

    /// Admits requests from the queue head while sequence and memory budgets allow.
    ///
    /// Requests configured for worst-case reservation acquire their full KV capacity here,
    /// keeping that capacity resident for the request lifetime.
    pub(super) fn admit(&mut self) {
        let bs = self.info.kv_block_size() as usize;
        loop {
            if self.running_request_count() >= self.scheduler.config.max_num_seqs
                || self.memory.request_slots.is_empty()
            {
                break;
            }
            let Some(head_id) = self.scheduler.peek() else {
                break;
            };
            let head = self
                .waiting
                .get(&head_id)
                .expect("scheduler waiting order names runtime state");
            if head.cursor.resources.reserve_worstcase {
                let need = head.cursor.resources.worstcase_blocks;
                let encoder_entries = head.req.resources.encoder_cache_keys.len();
                let encoder_ok = self
                    .memory
                    .reserved_encoder_entries
                    .saturating_add(encoder_entries)
                    <= self.memory.encoder_cache.budget();
                if need > self.memory.usable_blocks() {
                    let id = self.scheduler.pop().unwrap();
                    let st = self.waiting.remove(&id).unwrap();
                    self.trace_record(json!({
                        "event": "request_rejected",
                        "at_s": now(),
                        "request_id": st.req.request_id.0,
                        "reason": "too_large",
                        "needed_blocks": need,
                        "usable_blocks": self.memory.usable_blocks(),
                        "generation": &st.req.behavior,
                        "prompt_tokens": st.context.prompt_ids.len(),
                    }));
                    let _ = st.output.event_tx.send(Event::Rejected {
                        message: "request exceeds total KV capacity".into(),
                    });
                    continue;
                }
                if self.memory.free_blocks() >= need && encoder_ok {
                    let Some((target, _)) = self.prefill_target(head) else {
                        break;
                    };
                    let id = self.scheduler.pop().unwrap();
                    let st = self.waiting.remove(&id).unwrap();
                    let id = st.req.request_id;
                    self.admit_running(st, target);
                    // Physically allocate the worst case now: nothing can take
                    // these blocks, so this request can never fail mid-flight.
                    self.ensure_request_capacity(id, need * bs);
                    self.memory.reserved_blocks += need;
                    continue;
                }
            } else {
                let n = head.context.prompt_ids.len();
                let text_usable_blocks = (0..self.memory.cache().block_pool.num_groups())
                    .map(|group| self.memory.cache().block_pool.group_capacity(group))
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
                    .min(self.scheduler.config.long_prefill_threshold)
                    .min(self.scheduler.config.max_num_batched_tokens)
                    .max(1);
                let first_chunk_blocks = cached_prefix_tokens
                    .saturating_add(first_uncached_chunk)
                    .div_ceil(bs)
                    .saturating_sub(cached_prefix_blocks);
                if n > text_usable_blocks * bs {
                    let id = self.scheduler.pop().unwrap();
                    let st = self.waiting.remove(&id).unwrap();
                    self.trace_record(json!({
                        "event": "request_rejected",
                        "at_s": now(),
                        "request_id": st.req.request_id.0,
                        "reason": "too_large",
                        "needed_blocks": first_chunk_blocks,
                        "usable_blocks": text_usable_blocks,
                        "generation": &st.req.behavior,
                        "prompt_tokens": st.context.prompt_ids.len(),
                    }));
                    let _ = st.output.event_tx.send(Event::Rejected {
                        message: "request exceeds total KV capacity".into(),
                    });
                    continue;
                }
                let capacity_available = prefix_hit.cached_free_blocks.len()
                    == self.memory.cache().block_pool.num_groups()
                    && prefix_hit.cached_free_blocks.iter().enumerate().all(
                        |(group, cached_free)| {
                            self.memory
                                .cache()
                                .block_pool
                                .free_blocks_in_group(group)
                                .saturating_sub(*cached_free)
                                >= first_chunk_blocks
                        },
                    );
                if capacity_available {
                    let id = self.scheduler.pop().unwrap();
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
        state: &ReqState,
    ) -> Option<((crate::WorkerId, String), crate::kv::PrefixHit)> {
        let cache = self.memory.cache();
        self.worker_candidates(OpCode::ArExtend)
            .filter(|(worker, _, _)| self.executor.is_ready(worker))
            .map(|(worker, entry, info)| {
                let hit = cache.coordinator.probe_prefix(
                    &cache.block_pool,
                    state.effective_prompt(),
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
    pub(super) fn admit_running(&mut self, mut st: ReqState, target: (crate::WorkerId, String)) {
        let id = st.req.request_id;
        let request_key = RequestKey::new(self.authority_id, id, st.epoch);
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
            .memory
            .alloc(request_key, MemoryLayout::RequestSlot)
            .expect("admission checked request-slot capacity");
        let kv = self
            .memory
            .alloc(
                request_key,
                MemoryLayout::Kv {
                    tokens: 0,
                    groups: self.memory.cache().block_pool.num_groups() as u32,
                },
            )
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
        let generation = st.req.behavior.clone();
        let phase = st.cursor.phase;
        let prompt_tokens = st.context.prompt_ids.len();
        let max_tokens = st.req.max_und_tokens;
        let priority = st.req.priority;
        let reserve_worstcase = st.cursor.resources.reserve_worstcase;
        let worstcase_blocks = st.cursor.resources.worstcase_blocks;
        let encoder_entries = st.req.resources.encoder_cache_keys.len();
        if st.output.enqueue(Event::Scheduled {
            queued_at: q,
            scheduled_at,
        }) {
            st.terminal_intent = TerminalIntent::Finish(FinishReason::Cancelled);
        }

        // Running order, resource reservations, and runtime ownership advance
        // together so the next scheduling pass observes one coherent admission.
        self.running.insert(id, st);
        self.scheduler.running_order.push(id);
        self.memory.reserved_encoder_entries = self
            .memory
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
            "pending": self.scheduler.waiting_len(),
            "free_blocks": self.memory.free_blocks(),
            "reserved_blocks": self.memory.reserved_blocks,
            "reserved_encoder_entries": self.memory.reserved_encoder_entries,
        }));

        // Prefix-cache acquisition pins every reused block to this request's
        // newly installed block tables.
        let memory = &self.memory;
        if let Some(st) = self.running.get_mut(&id) {
            let kv = memory.cache();
            acquire_cached_prefix(&kv.coordinator, st, &kv.block_pool, &self.stats, &source);
        }
    }
}

/// Acquires a complete cross-group prefix hit and records cache accounting on the request.
fn acquire_cached_prefix(
    coordinator: &KvCacheCoordinator,
    state: &mut ReqState,
    pool: &BlockPool,
    stats: &SchedStats,
    source: &uniserve_worker_ipc::WorkerEndpoint,
) {
    let prompt = state.effective_prompt().to_vec();
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
    state.cursor.replay.block_hashes = hit.block_hashes;
    state.cursor.replay.prefix_cached_blocks = hit.cached_blocks;
    state.cursor.ingest.prompt_cursor = (hit.cached_blocks * block_size) as u32;
    state.cursor.und.logical_pos = state.cursor.ingest.prompt_cursor;
    state.cursor.und.physical_kv_len = state.cursor.ingest.prompt_cursor;
}
